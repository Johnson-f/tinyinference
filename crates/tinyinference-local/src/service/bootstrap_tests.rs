use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::{Json, Router, routing::get};
use serde_json::json;

use super::*;

#[test]
fn autosummary_debounce_blocks_repeated_calls_inside_window() {
    let mut config = Config::default();
    config.local_ai.autosummary_debounce_ms = 60_000;
    let service = LocalAiService::new(&config);

    assert!(service.should_run_memory_autosummary(&config));
    assert!(!service.should_run_memory_autosummary(&config));
}

/// Mock Ollama that answers `/api/tags` and counts every request that would
/// mutate the runtime (`/api/pull`, `/api/create`, `/api/delete`) or any
/// unknown path.
async fn spawn_counting_ollama(forbidden_hits: Arc<AtomicUsize>) -> String {
    let fallback_hits = Arc::clone(&forbidden_hits);
    let app = Router::new()
        .route(
            "/api/tags",
            get(|| async { Json(json!({ "models": [{ "name": "gemma3:1b-it-qat" }] })) }),
        )
        .fallback(move || {
            let hits = Arc::clone(&fallback_hits);
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                axum::http::StatusCode::NOT_FOUND
            }
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://127.0.0.1:{}", addr.port())
}

fn opted_in(base_url: &str) -> Config {
    let mut config = Config::default();
    config.local_ai.runtime_enabled = true;
    config.local_ai.opt_in_confirmed = true;
    config.local_ai.provider = "ollama".to_string();
    config.local_ai.base_url = Some(base_url.to_string());
    config
}

/// A local TCP port with nothing listening on it.
fn unbound_base_url() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn bootstrap_without_opt_in_is_disabled_and_probes_nothing() {
    let forbidden = Arc::new(AtomicUsize::new(0));
    let base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let mut config = opted_in(&base);
    config.local_ai.opt_in_confirmed = false;
    let service = LocalAiService::new(&config);

    service.bootstrap(&config).await;

    assert_eq!(service.status().state, "disabled");
    assert_eq!(forbidden.load(Ordering::SeqCst), 0);
}

/// Regression: bootstrap is a read-only probe. Against a healthy endpoint it
/// reports `ready` and never asks the runtime to pull a model.
#[tokio::test]
async fn bootstrap_against_reachable_endpoint_is_ready_without_pulling() {
    let forbidden = Arc::new(AtomicUsize::new(0));
    let base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let mut config = opted_in(&base);
    // Models the endpoint does NOT serve: the old bootstrap would have pulled
    // them. The probe must not.
    config.local_ai.chat_model_id = "gemma3:4b-it-qat".to_string();
    config.local_ai.embedding_model_id = "bge-m3".to_string();
    config.local_ai.vision_model_id = "moondream:1.8b-v2-q4_K_S".to_string();
    let service = LocalAiService::new(&config);

    service.bootstrap(&config).await;

    let status = service.status();
    assert_eq!(status.state, "ready");
    assert_eq!(status.warning, None);
    assert_eq!(status.error_category, None);
    assert_eq!(
        forbidden.load(Ordering::SeqCst),
        0,
        "bootstrap must only GET /api/tags — no /api/pull or other mutating call"
    );
}

/// Regression: with nothing listening, bootstrap reports `unreachable` and
/// does not try to start a runtime (it returns promptly and the endpoint is
/// still not listening afterwards).
#[tokio::test]
async fn bootstrap_with_nothing_listening_is_unreachable_and_spawns_nothing() {
    let base = unbound_base_url();
    let config = opted_in(&base);
    let service = LocalAiService::new(&config);

    let started = std::time::Instant::now();
    service.bootstrap(&config).await;
    let elapsed = started.elapsed();

    let status = service.status();
    assert_eq!(status.state, "unreachable");
    assert_eq!(status.error_category.as_deref(), Some("server"));
    assert!(
        status
            .warning
            .as_deref()
            .is_some_and(|w| w.contains("not reachable")),
        "warning must tell the user to start their runtime: {:?}",
        status.warning
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "probe must not wait on a spawned runtime (took {elapsed:?})"
    );
    let port = base.rsplit(':').next().unwrap();
    assert!(
        std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_err(),
        "nothing may have been started on the configured endpoint"
    );
}

/// An unreachable endpoint is probed again on the next call, so a runtime the
/// user starts later is picked up without a restart.
#[tokio::test]
async fn unreachable_endpoint_is_reprobed_on_next_bootstrap() {
    let config = opted_in(&unbound_base_url());
    let service = LocalAiService::new(&config);
    service.bootstrap(&config).await;
    assert_eq!(service.status().state, "unreachable");

    let forbidden = Arc::new(AtomicUsize::new(0));
    let base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let config = opted_in(&base);
    service.bootstrap(&config).await;

    assert_eq!(service.status().state, "ready");
    assert_eq!(forbidden.load(Ordering::SeqCst), 0);
}

/// OpenAI-compatible runtimes (LM Studio) are probed with `GET /v1/models`.
#[tokio::test]
async fn bootstrap_probes_openai_compatible_endpoint_with_v1_models() {
    let hits = Arc::new(AtomicUsize::new(0));
    let models_hits = Arc::clone(&hits);
    let app = Router::new().route(
        "/v1/models",
        get(move || {
            let hits = Arc::clone(&models_hits);
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                Json(json!({ "data": [{ "id": "local-model" }] }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let mut config = opted_in(&format!("http://127.0.0.1:{}/v1", addr.port()));
    config.local_ai.provider = "lm_studio".to_string();
    let service = LocalAiService::new(&config);

    service.bootstrap(&config).await;

    assert_eq!(service.status().state, "ready");
    assert_eq!(service.status().provider, "lm_studio");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[test]
fn vision_mode_follows_configured_vision_model() {
    let mut config = Config::default();
    let service = LocalAiService::new(&config);
    assert_eq!(service.status().vision_mode, "disabled");
    assert_eq!(service.status().vision_state, "disabled");

    config.local_ai.vision_model_id = "moondream:1.8b-v2-q4_K_S".to_string();
    let service = LocalAiService::new(&config);
    assert_eq!(service.status().vision_mode, "ondemand");
    assert_eq!(service.status().vision_state, "idle");
}
