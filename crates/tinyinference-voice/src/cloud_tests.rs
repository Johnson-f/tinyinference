use super::*;
use axum::{
    Router,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use std::sync::{Arc, Mutex};

async fn test_server(app: Router) -> reqwest::Url {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{address}/v1/audio/transcriptions")
        .parse()
        .unwrap()
}

#[test]
fn response_sanitization_removes_exact_and_prefixed_secrets() {
    let detail = sanitize_response_detail(
        "authorization=opaque-token and sk-provider-secret",
        "opaque-token",
    );
    assert!(!detail.contains("opaque-token"));
    assert!(!detail.contains("sk-provider-secret"));
    assert!(detail.contains("[REDACTED]"));
}

#[test]
fn rejects_invalid_or_empty_audio_before_making_a_request() {
    let options = CloudTranscribeOptions::default();
    let client = reqwest::Client::new();
    let url: reqwest::Url = "http://127.0.0.1:1/transcribe".parse().unwrap();

    let empty = tokio::runtime::Runtime::new().unwrap().block_on(transcribe(
        &client,
        url.clone(),
        "token",
        " ",
        &options,
    ));
    assert_eq!(empty.unwrap_err(), "audio_base64 is required");

    let invalid = tokio::runtime::Runtime::new().unwrap().block_on(transcribe(
        &client,
        url,
        "token",
        "not base64!",
        &options,
    ));
    assert!(invalid.unwrap_err().starts_with("invalid base64 audio:"));
}

#[tokio::test]
async fn uploads_multipart_audio_with_defaults_and_returns_trimmed_text() {
    let authorization = Arc::new(Mutex::new(None));
    let captured = Arc::clone(&authorization);
    let app = Router::new().route(
        "/v1/audio/transcriptions",
        post(move |headers: HeaderMap| {
            let captured = Arc::clone(&captured);
            async move {
                *captured.lock().unwrap() = headers
                    .get(AUTHORIZATION)
                    .and_then(|value| value.to_str().ok())
                    .map(ToOwned::to_owned);
                (StatusCode::OK, r#"{"text":"  hello world  "}"#)
            }
        }),
    );
    let url = test_server(app).await;

    let result = transcribe(
        &reqwest::Client::new(),
        url,
        "secret-token",
        "AQID",
        &CloudTranscribeOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        result,
        CloudTranscribeResult {
            text: "hello world".to_string()
        }
    );
    assert_eq!(
        authorization.lock().unwrap().as_deref(),
        Some("Bearer secret-token")
    );
}

#[tokio::test]
async fn failure_responses_redact_bearer_tokens() {
    let app = Router::new().route(
        "/v1/audio/transcriptions",
        post(|| async { (StatusCode::BAD_GATEWAY, "token=secret-token sk-hidden") }),
    );
    let error = transcribe(
        &reqwest::Client::new(),
        test_server(app).await,
        "secret-token",
        "AQID",
        &CloudTranscribeOptions::default(),
    )
    .await
    .unwrap_err();

    assert!(error.contains("502 Bad Gateway"));
    assert!(!error.contains("secret-token"));
    assert!(!error.contains("sk-hidden"));
}

#[test]
fn nonempty_trims_and_discards_blank_values() {
    assert_eq!(nonempty(Some(" model ")), Some("model"));
    assert_eq!(nonempty(Some("\t\n")), None);
    assert_eq!(nonempty(None), None);
}
