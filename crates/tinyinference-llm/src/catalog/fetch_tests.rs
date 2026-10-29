use super::*;
use axum::{
    Router,
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    routing::any,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[tokio::test]
async fn codex_oauth_discovery_request_fetches_models_with_protocol_headers() {
    use crate::providers::{
        ProviderKind, ProviderSpec,
        openai::codex::OpenAiCodexRouting,
        registry::{ProviderBuildOptions, ProviderRegistry},
    };

    let seen = Arc::new(Mutex::new(None));
    let seen_request = Arc::clone(&seen);
    let app = Router::new().route(
        "/models",
        any(move |request: Request<Body>| {
            let seen_request = Arc::clone(&seen_request);
            async move {
                *seen_request.lock().expect("request lock") =
                    Some((request.uri().to_string(), request.headers().clone()));
                r#"{"models":[{"slug":"gpt-test"}]}"#
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve fixture") });

    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok("oauth-test-token".to_string()));
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    let route = OpenAiCodexRouting {
        endpoint: format!("http://{address}"),
        using_oauth: true,
        account_id: Some("account-123".into()),
    };
    let query = [("client_version".to_string(), "0.130.0".to_string())];
    let request = registry
        .discovery_request(
            &spec,
            &ProviderBuildOptions {
                codex: Some(&route),
                extra_query_params: &query,
                user_agent: Some("codex_cli_rs/0.0.0 (Fixture 1.0)"),
                ..Default::default()
            },
        )
        .expect("build discovery request")
        .expect("Codex listing is discoverable");
    let models = list_models(&reqwest::Client::new(), request)
        .await
        .expect("Codex model listing");
    assert_eq!(models[0].id, "gpt-test");
    let (uri, headers) = seen
        .lock()
        .expect("request lock")
        .clone()
        .expect("request seen");
    assert_eq!(uri, "/models?client_version=0.130.0");
    assert_eq!(
        headers
            .get_all(reqwest::header::AUTHORIZATION)
            .iter()
            .count(),
        1,
        "discovery sends exactly one Authorization header"
    );
    assert_eq!(headers["authorization"], "Bearer oauth-test-token");
    assert_eq!(headers["chatgpt-account-id"], "account-123");
    assert_eq!(headers["originator"], "codex_cli_rs");
    assert_eq!(headers["openai-beta"], "responses=experimental");
    assert_eq!(headers["user-agent"], "codex_cli_rs/0.0.0 (Fixture 1.0)");
    task.abort();
}

async fn serve(
    status: StatusCode,
    body: impl Into<String>,
) -> (String, tokio::task::JoinHandle<()>) {
    let body = Arc::new(body.into());
    let app = Router::new().route(
        "/{*path}",
        any(move || {
            let body = Arc::clone(&body);
            async move { (status, (*body).clone()) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve fixture");
    });
    (format!("http://{address}"), task)
}

#[tokio::test]
async fn list_models_uses_discovery_listing_url_and_headers() {
    let seen = Arc::new(Mutex::new(None));
    let seen_request = Arc::clone(&seen);
    let app = Router::new().route(
        "/models",
        any(move |request: Request<Body>| {
            let seen_request = Arc::clone(&seen_request);
            async move {
                *seen_request.lock().expect("request lock") = Some((
                    request.uri().to_string(),
                    request.headers().get("authorization").cloned(),
                ));
                r#"{"data":[{"id":"model-a"}]}"#
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve fixture") });
    let client = reqwest::Client::new();
    let request =
        crate::model::discover::DiscoveryRequest::new(format!("http://{address}/v1"), "model-a")
            .with_listing_url(format!(
                "http://{address}/models?catalog=codex&client_version=1"
            ))
            .with_header("Authorization", "Bearer fixture-token");

    let models = list_models(&client, request).await.expect("model listing");
    assert_eq!(models[0].id, "model-a");
    let (uri, auth) = seen
        .lock()
        .expect("request lock")
        .clone()
        .expect("request seen");
    assert_eq!(uri, "/models?catalog=codex&client_version=1");
    assert_eq!(
        auth.and_then(|value| value.to_str().ok().map(str::to_owned))
            .as_deref(),
        Some("Bearer fixture-token")
    );
    task.abort();
}

#[tokio::test]
async fn list_models_preserves_404_hint_and_rejects_success_error_payload() {
    let (base, task) = serve(StatusCode::NOT_FOUND, "missing").await;
    let client = reqwest::Client::new();
    let error = list_models(
        &client,
        crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
    )
    .await
    .expect_err("404 should fail");
    assert!(error.to_string().contains("provider returned 404:"));
    assert!(
        error
            .to_string()
            .contains("does not expose a `/models` endpoint")
    );
    task.abort();

    let (base, task) = serve(StatusCode::UNAUTHORIZED, "invalid provider token").await;
    let error = list_models(
        &client,
        crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
    )
    .await
    .expect_err("401 should fail");
    assert!(error.to_string().contains("provider returned 401:"));
    assert!(error.to_string().contains("invalid provider token"));
    task.abort();

    let (base, task) = serve(StatusCode::OK, r#"{"error":{"message":"bad route"}}"#).await;
    let error = list_models(
        &client,
        crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
    )
    .await
    .expect_err("error payload should fail");
    assert!(
        error
            .to_string()
            .contains("provider returned error payload: bad route")
    );
    task.abort();
}

#[tokio::test]
async fn list_models_error_exposes_only_the_actual_http_status() {
    let client = reqwest::Client::new();

    for (status, expected) in [
        (StatusCode::UNAUTHORIZED, 401),
        (StatusCode::FORBIDDEN, 403),
        (StatusCode::NOT_FOUND, 404),
    ] {
        let (base, task) = serve(status, "response mentions 401").await;
        let error = list_models(
            &client,
            crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
        )
        .await
        .expect_err("non-success HTTP status should fail");

        assert_eq!(error.http_status(), Some(expected));
        task.abort();
    }

    let (base, task) = serve(StatusCode::OK, r#"{"error":{"message":"401 rejected"}}"#).await;
    let error = list_models(
        &client,
        crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
    )
    .await
    .expect_err("error payload should fail");
    assert_eq!(error.http_status(), None);
    task.abort();

    let (base, task) = serve(StatusCode::OK, "not JSON").await;
    let error = list_models(
        &client,
        crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
    )
    .await
    .expect_err("malformed JSON should fail");
    assert_eq!(error.http_status(), None);
    task.abort();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind closed fixture");
    let address = listener.local_addr().expect("closed fixture address");
    drop(listener);
    let error = list_models(
        &client,
        crate::model::discover::DiscoveryRequest::new(format!("http://{address}/v1"), "model"),
    )
    .await
    .expect_err("closed local port should fail");
    assert_eq!(error.http_status(), None);
}

#[tokio::test]
async fn malformed_success_body_has_a_bounded_sanitized_diagnostic() {
    let token = "sk-synthetic-secret-12345678901234567890";
    let body = format!("{token} {}", "界".repeat(500));
    let (base, task) = serve(StatusCode::OK, body).await;
    let error = list_models(
        &reqwest::Client::new(),
        crate::model::discover::DiscoveryRequest::new(format!("{base}/v1"), "model"),
    )
    .await
    .expect_err("malformed JSON should fail");
    let message = error.to_string();
    assert!(message.contains("failed to parse JSON"));
    assert!(message.contains("[REDACTED]"), "{message}");
    assert!(!message.contains(token));
    assert!(message.chars().count() < 500, "diagnostic was not bounded");
    task.abort();
}

#[tokio::test]
async fn openrouter_validation_sends_bearer_and_rejects_error_payload() {
    let seen = Arc::new(Mutex::new(None));
    let seen_headers = Arc::clone(&seen);
    let app = Router::new().route(
        "/api/v1/key",
        any(move |headers: HeaderMap| {
            let seen_headers = Arc::clone(&seen_headers);
            async move {
                *seen_headers.lock().expect("header lock") = headers.get("authorization").cloned();
                r#"{"error":{"message":"invalid key"}}"#
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve fixture") });
    let error = validate_openrouter_key(
        &reqwest::Client::new(),
        &format!("http://{address}/api/v1"),
        "fixture-key",
    )
    .await
    .expect_err("error payload should fail");
    assert!(error.to_string().contains("invalid key"));
    assert_eq!(
        seen.lock()
            .expect("header lock")
            .as_ref()
            .and_then(|value| value.to_str().ok()),
        Some("Bearer fixture-key")
    );
    task.abort();
}

#[tokio::test]
async fn openrouter_validation_error_exposes_actual_http_status() {
    let (base, task) = serve(StatusCode::FORBIDDEN, "response mentions 401").await;
    let error = validate_openrouter_key(&reqwest::Client::new(), &base, "fixture-key")
        .await
        .expect_err("non-success HTTP status should fail");

    assert_eq!(error.http_status(), Some(403));
    assert_eq!(
        error.to_string(),
        "OpenRouter key validation returned 403: response mentions 401"
    );
    task.abort();
}

#[tokio::test]
async fn openrouter_validation_rejects_blank_key_without_sending_request() {
    let requests = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&requests);
    let app = Router::new().route(
        "/{*path}",
        any(move || {
            let seen = Arc::clone(&seen);
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                "{}"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve fixture") });
    for key in ["", " \t\n"] {
        let error = validate_openrouter_key(
            &reqwest::Client::new(),
            &format!("http://{address}/api/v1"),
            key,
        )
        .await
        .expect_err("blank key should be rejected");
        assert!(error.to_string().contains("API key is required"));
    }
    assert_eq!(requests.load(Ordering::SeqCst), 0);
    task.abort();
}
