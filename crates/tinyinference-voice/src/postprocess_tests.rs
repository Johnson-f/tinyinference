use axum::{Json, Router, routing::post};
use serde_json::{Value, json};

use super::*;

const TIMEOUT: Duration = Duration::from_secs(5);

async fn spawn_mock(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://127.0.0.1:{}", addr.port())
}

fn chat_response(content: &str) -> Value {
    json!({
        "id": "chatcmpl-test",
        "choices": [{
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    })
}

/// A runtime pointed at `base` and a service pre-seeded to `state`.
fn runtime_and_service(base: &str, state: &str) -> (RuntimeConfig, LocalAiService) {
    let mut runtime = RuntimeConfig::default();
    runtime.local_ai.runtime_enabled = true;
    runtime.local_ai.base_url = Some(base.to_string());
    let service = LocalAiService::new(&runtime);
    service.replace_status_state(state);
    (runtime, service)
}

#[tokio::test]
async fn empty_and_whitespace_text_is_returned_unchanged() {
    let (runtime, service) = runtime_and_service("http://127.0.0.1:1", "ready");
    assert_eq!(
        cleanup_transcription(&service, &runtime, "", None, TIMEOUT).await,
        ""
    );
    assert_eq!(
        cleanup_transcription(&service, &runtime, "   ", None, TIMEOUT).await,
        "   "
    );
}

#[tokio::test]
async fn not_ready_service_returns_raw_text_without_calling_the_llm() {
    // The unreachable endpoint would surface as an error fallback too, so
    // count requests on a live mock to prove the call was skipped.
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = hits.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { Json(chat_response("cleaned")) }
        }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "not_ready");

    let result =
        cleanup_transcription(&service, &runtime, "um hello uh world", None, TIMEOUT).await;

    assert_eq!(result, "um hello uh world");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn ready_service_returns_the_trimmed_cleaned_text() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { Json(chat_response("  Hello, world.  ")) }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "ready");

    let result = cleanup_transcription(&service, &runtime, "um hello world", None, TIMEOUT).await;

    assert_eq!(result, "Hello, world.");
}

#[tokio::test]
async fn degraded_service_still_runs_cleanup() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { Json(chat_response("Cleaned.")) }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "degraded");

    let result = cleanup_transcription(&service, &runtime, "uh cleaned", None, TIMEOUT).await;

    assert_eq!(result, "Cleaned.");
}

#[tokio::test]
async fn empty_llm_response_falls_back_to_raw_text() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { Json(chat_response("   ")) }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "ready");

    let result = cleanup_transcription(&service, &runtime, "keep me", None, TIMEOUT).await;

    assert_eq!(result, "keep me");
}

#[tokio::test]
async fn llm_error_response_falls_back_to_raw_text() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "ready");

    let result = cleanup_transcription(&service, &runtime, "raw text", None, TIMEOUT).await;

    assert_eq!(result, "raw text");
}

#[tokio::test(start_paused = true)]
async fn slow_llm_times_out_and_falls_back_to_raw_text() {
    let started = std::sync::Arc::new(tokio::sync::Notify::new());
    let handler_started = started.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let handler_started = handler_started.clone();
            async move {
                handler_started.notify_one();
                std::future::pending::<Json<Value>>().await
            }
        }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "ready");
    let cleanup = tokio::spawn(async move {
        cleanup_transcription(
            &service,
            &runtime,
            "raw text",
            None,
            Duration::from_millis(100),
        )
        .await
    });
    started.notified().await;
    tokio::time::advance(Duration::from_millis(101)).await;
    let result = cleanup.await.unwrap();

    assert_eq!(result, "raw text");
}

/// Captures the user prompt the model receives, echoing it back.
async fn echo_prompt_result(context: Option<&str>) -> String {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|Json(body): Json<Value>| async move {
            let user = body["messages"]
                .as_array()
                .and_then(|m| m.iter().find(|m| m["role"] == "user"))
                .and_then(|m| m["content"].as_str())
                .unwrap_or_default()
                .to_string();
            Json(chat_response(&user))
        }),
    );
    let base = spawn_mock(app).await;
    let (runtime, service) = runtime_and_service(&base, "ready");
    cleanup_transcription(&service, &runtime, "raw text", context, TIMEOUT).await
}

#[tokio::test]
async fn conversation_context_is_prepended_to_the_prompt() {
    let prompt = echo_prompt_result(Some("previous turn: check the oven")).await;
    assert!(prompt.starts_with("Conversation context:\n"), "{prompt}");
    assert!(prompt.contains("previous turn: check the oven"));
    assert!(prompt.contains("Transcribed text to clean up:\nraw text"));
}

#[tokio::test]
async fn whitespace_only_context_is_not_forwarded() {
    assert_eq!(echo_prompt_result(Some("   ")).await, "raw text");
    assert_eq!(echo_prompt_result(None).await, "raw text");
}
