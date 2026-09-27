//! Client transport, retry, measurement, and secret-handling tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, SystemTime},
};

use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::Mutex,
};

use super::*;
use crate::{Choice, Question};

fn request() -> EvaluationRequest {
    EvaluationRequest::jev(
        json!({"message": "review this"}),
        BTreeMap::from([(
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("Who should answer?"),
                criteria: BTreeMap::from([("alice".to_owned(), None), ("bob".to_owned(), None)]),
            }),
        )]),
    )
}

fn success() -> String {
    json!({
        "model": "jev-latest",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "bob",
                "probabilities": {"alice": 0.2, "bob": 0.8},
                "confidence": 0.6
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 2}
    })
    .to_string()
}

fn response(status: u16, body: &str, extra_headers: &str) -> String {
    let reason = if status == 200 { "OK" } else { "Error" };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n{body}",
        body.len()
    )
}

async fn server(responses: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    tokio::spawn(async move {
        for reply in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0_u8; 16_384];
            let count = stream.read(&mut buffer).await.unwrap();
            recorded
                .lock()
                .await
                .push(String::from_utf8_lossy(&buffer[..count]).into_owned());
            stream.write_all(reply.as_bytes()).await.unwrap();
        }
    });
    (format!("http://{address}"), requests)
}

async fn slow_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
    });
    format!("http://{address}")
}

fn config(base_url: String) -> ClientConfig {
    let mut config = ClientConfig::new("secret-test-key");
    config.base_url = base_url;
    config.timeout = Duration::from_secs(1);
    config.retry = RetryPolicy {
        max_retries: 0,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(5),
    };
    config
}

fn unavailable_base_url() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
}

#[tokio::test]
async fn sends_the_documented_endpoint_and_bearer_header() {
    let (base_url, requests) = server(vec![response(
        200,
        &success(),
        "x-request-id: request-7\r\n",
    )])
    .await;
    let result = Client::new(config(base_url))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap();
    assert_eq!(result.attempts, 1);
    assert_eq!(result.request_id.as_deref(), Some("request-7"));
    assert_eq!(result.response.usage.input_tokens, Some(12));
    let sent = requests.lock().await.join("");
    assert!(sent.starts_with("POST /v1/systemone HTTP/1.1"));
    assert!(
        sent.to_ascii_lowercase()
            .contains("authorization: bearer secret-test-key")
    );
    assert!(sent.contains("\"model\":\"jev-latest\""));
}

#[tokio::test]
async fn an_exact_endpoint_override_is_not_extended_with_a_provider_path() {
    let (base_url, requests) = server(vec![response(
        200,
        &success().replace("jev-latest", "typesafe/jev-1.13-20260917"),
        "",
    )])
    .await;
    let endpoint = format!("{base_url}/api/alpha/decisions");
    let mut config = ClientConfig::openrouter("secret-test-key").with_endpoint_url(endpoint);
    config.timeout = Duration::from_secs(1);
    config.retry.max_retries = 0;

    Client::new(config)
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap();

    assert!(
        requests
            .lock()
            .await
            .join("")
            .starts_with("POST /api/alpha/decisions HTTP/1.1")
    );
}

#[tokio::test]
async fn openrouter_uses_system_one_and_accepts_a_resolved_jev_model() {
    let (base_url, requests) = server(vec![response(
        200,
        &success().replace("jev-latest", "typesafe/jev-1.13-20260917"),
        "",
    )])
    .await;
    let mut config = ClientConfig::openrouter("secret-test-key");
    config.base_url = base_url;
    config.timeout = Duration::from_secs(1);
    config.retry.max_retries = 0;
    let result = Client::new(config)
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap();
    assert_eq!(result.response.model, "typesafe/jev-1.13-20260917");
    let sent = requests.lock().await.join("");
    assert!(sent.starts_with("POST /v1/systemone HTTP/1.1"));
}

#[tokio::test]
async fn tinyhumans_proxy_uses_the_direct_system_one_path() {
    let (base_url, requests) = server(vec![response(
        200,
        &success().replace("jev-latest", "typesafe/jev-1.13-20260917"),
        "",
    )])
    .await;
    let mut config = ClientConfig::tinyhumans_openrouter("secret-test-key");
    config.base_url = base_url;
    config.timeout = Duration::from_secs(1);
    config.retry.max_retries = 0;
    Client::new(config)
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap();
    assert!(
        requests
            .lock()
            .await
            .join("")
            .starts_with("POST /agent-integrations/openrouter/systemone HTTP/1.1")
    );
}

#[test]
fn sdk_name_is_sanitized_and_sent_only_to_the_exact_tinyhumans_proxy() {
    let config = ClientConfig::tinyhumans_openrouter("key")
        .with_sdk_name(" OpenCompany\r\nInjected: yes / test ");
    let client = Client::new(config.clone()).unwrap();
    let outgoing = client.evaluation_request(&request()).build().unwrap();
    assert_eq!(
        outgoing.headers().get("x-sdk-name").unwrap(),
        "opencompanyinjectedyestest"
    );

    let direct = Client::new(ClientConfig::openrouter("key").with_sdk_name("openhuman")).unwrap();
    assert!(
        direct
            .evaluation_request(&request())
            .build()
            .unwrap()
            .headers()
            .get("x-sdk-name")
            .is_none()
    );

    for endpoint in [
        "https://api.tinyhumans.ai.evil.example/agent-integrations/openrouter/systemone",
        "https://api.tinyhumans.ai:444/agent-integrations/openrouter/systemone",
        "https://example.com/agent-integrations/openrouter/systemone",
    ] {
        let other = Client::new(config.clone().with_endpoint_url(endpoint)).unwrap();
        assert!(
            other
                .evaluation_request(&request())
                .build()
                .unwrap()
                .headers()
                .get("x-sdk-name")
                .is_none(),
            "must not attribute {endpoint}"
        );
    }
    assert!(
        Client::new(config.with_endpoint_url(
            "https://api.tinyhumans.ai/agent-integrations/openrouter/systemone?redirect=1"
        ))
        .is_err()
    );
}

#[tokio::test]
async fn retries_rate_limits_and_reports_attempts() {
    let (base_url, requests) =
        server(vec![response(429, "{}", ""), response(200, &success(), "")]).await;
    let mut config = config(base_url);
    config.retry.max_retries = 1;
    let result = Client::new(config)
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap();
    assert_eq!(result.attempts, 2);
    assert_eq!(requests.lock().await.len(), 2);
}

#[tokio::test]
async fn authentication_is_terminal() {
    let (base_url, requests) = server(vec![response(401, "{}", "")]).await;
    let error = Client::new(config(base_url))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(error.error, Error::Authentication));
    assert_eq!(error.attempts, 1);
    assert_eq!(requests.lock().await.len(), 1);
}

#[tokio::test]
async fn malformed_success_body_is_a_decode_failure() {
    let (base_url, _) = server(vec![response(200, "not-json", "")]).await;
    let error = Client::new(config(base_url))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(error.error, Error::Decode { .. }));
}

#[tokio::test]
async fn debug_output_redacts_the_api_key() {
    let config = config("http://127.0.0.1:1".to_owned());
    let rendered = format!("{config:?}");
    assert!(rendered.contains("[REDACTED]"));
    assert!(!rendered.contains("secret-test-key"));
    let client = Client::new(config).unwrap();
    let rendered = format!("{client:?}");
    assert!(rendered.contains("[REDACTED]"));
    assert!(!rendered.contains("secret-test-key"));
}

#[test]
fn rejects_invalid_configuration_before_transport() {
    let mut empty = ClientConfig::new("");
    empty.base_url = "not a URL".to_owned();
    assert!(matches!(
        Client::new(empty),
        Err(Error::InvalidConfig { .. })
    ));
    let mut invalid_url = ClientConfig::new("key");
    invalid_url.base_url = "not a URL".into();
    assert!(matches!(
        Client::new(invalid_url),
        Err(Error::InvalidConfig { .. })
    ));
}

#[test]
fn validates_every_configuration_bound_and_redacted_key_replacement() {
    let replaced = ClientConfig::new("old").with_api_key("new-secret");
    let rendered = format!("{replaced:?}");
    assert!(!rendered.contains("new-secret"));
    let endpoint =
        ClientConfig::new("key").with_endpoint_url("https://example.com/custom/decisions");
    assert_eq!(
        endpoint.endpoint_url(),
        Some("https://example.com/custom/decisions")
    );
    for endpoint_url in [
        "not a URL",
        "file:///tmp/decisions",
        "http://example.com/decisions",
        "http://localhost:8080/decisions",
        "https://user:password@example.com/decisions",
        "https://example.com/decisions?tenant=x",
        "https://example.com/decisions#fragment",
    ] {
        let config = ClientConfig::new("key").with_endpoint_url(endpoint_url);
        assert!(matches!(
            Client::new(config),
            Err(Error::InvalidConfig { .. })
        ));
    }

    let mut scheme = ClientConfig::new("key");
    scheme.base_url = "file:///tmp/socket".into();
    assert!(matches!(
        Client::new(scheme),
        Err(Error::InvalidConfig { .. })
    ));

    for base_url in ["http://example.com", "http://localhost:8080"] {
        let mut cleartext = ClientConfig::new("key");
        cleartext.base_url = base_url.into();
        assert!(matches!(
            Client::new(cleartext),
            Err(Error::InvalidConfig { .. })
        ));
    }
    let mut secure = ClientConfig::new("key");
    secure.base_url = "https://example.com".into();
    assert!(Client::new(secure).is_ok());
    let openrouter = ClientConfig::openrouter("key");
    assert_eq!(openrouter.base_url, "https://openrouter.ai/api");
    assert_eq!(openrouter.provider, Provider::OpenRouter);
    let tinyhumans = ClientConfig::tinyhumans_openrouter("key");
    assert_eq!(tinyhumans.base_url, "https://api.tinyhumans.ai");
    assert_eq!(
        tinyhumans.system_one_path,
        "agent-integrations/openrouter/systemone"
    );
    assert_eq!(tinyhumans.provider, Provider::OpenRouter);
    let mut ipv6_loopback = ClientConfig::new("key");
    ipv6_loopback.base_url = "http://[::1]:8080".into();
    assert!(Client::new(ipv6_loopback).is_ok());
    let mut userinfo = ClientConfig::new("key");
    userinfo.base_url = "https://user:password@example.com".into();
    assert!(matches!(
        Client::new(userinfo),
        Err(Error::InvalidConfig { .. })
    ));
    for base_url in [
        "https://example.com?tenant=x",
        "https://example.com#fragment",
    ] {
        let mut component = ClientConfig::new("key");
        component.base_url = base_url.into();
        assert!(matches!(
            Client::new(component),
            Err(Error::InvalidConfig { .. })
        ));
    }

    let mut timeout = ClientConfig::new("key");
    timeout.timeout = Duration::ZERO;
    assert!(matches!(
        Client::new(timeout),
        Err(Error::InvalidConfig { .. })
    ));

    let mut retry = ClientConfig::new("key");
    retry.retry.initial_backoff = Duration::ZERO;
    assert!(matches!(
        Client::new(retry),
        Err(Error::InvalidConfig { .. })
    ));
    let mut unbounded = ClientConfig::new("key");
    unbounded.retry.max_retries = u32::MAX;
    assert!(matches!(
        Client::new(unbounded),
        Err(Error::InvalidConfig { .. })
    ));
}

#[test]
fn retry_delay_is_exponential_and_bounded() {
    let policy = RetryPolicy {
        max_retries: 5,
        initial_backoff: Duration::from_millis(10),
        max_backoff: Duration::from_millis(25),
    };
    assert_eq!(policy.delay(1), Duration::from_millis(10));
    assert_eq!(policy.delay(2), Duration::from_millis(20));
    assert_eq!(policy.delay(30), Duration::from_millis(25));
}

#[test]
fn status_classification_covers_terminal_and_retryable_classes() {
    assert!(matches!(
        classify_status(StatusCode::BAD_REQUEST, None),
        Failure::Terminal(Error::Unprocessable)
    ));
    assert!(matches!(
        classify_status(StatusCode::NOT_FOUND, None),
        Failure::Terminal(Error::HttpStatus { status: 404 })
    ));
    assert!(matches!(
        classify_status(StatusCode::INTERNAL_SERVER_ERROR, None),
        Failure::Retryable {
            error: Error::HttpStatus { status: 500 },
            ..
        }
    ));
    assert!(matches!(
        classify_status(StatusCode::REQUEST_TIMEOUT, None),
        Failure::Retryable {
            error: Error::Timeout,
            ..
        }
    ));
    assert!(matches!(
        classify_status(StatusCode::from_u16(529).unwrap(), None),
        Failure::Retryable {
            error: Error::Overloaded,
            ..
        }
    ));
    assert_eq!(
        parse_retry_after(Some(&reqwest::header::HeaderValue::from_static("3"))),
        Some(Duration::from_secs(3))
    );
    assert_eq!(
        parse_retry_after(Some(&reqwest::header::HeaderValue::from_static("date"))),
        None
    );
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let future = now + Duration::from_secs(30);
    let date = httpdate::fmt_http_date(future);
    let header = reqwest::header::HeaderValue::from_str(&date).unwrap();
    assert_eq!(
        parse_retry_after_at(Some(&header), now),
        Some(Duration::from_secs(30))
    );
}

#[tokio::test]
async fn timeout_is_retryable_but_respects_the_attempt_bound() {
    let mut config = config(slow_server().await);
    config.timeout = Duration::from_millis(5);
    let error = Client::new(config)
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(error.error, Error::Timeout));
    assert_eq!(error.attempts, 1);
    assert!(error.latency >= Duration::from_millis(5));
}

#[tokio::test]
async fn exhausted_rate_limit_returns_the_classified_error() {
    let (base_url, _) = server(vec![response(429, "{}", "")]).await;
    let error = Client::new(config(base_url))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(error.error, Error::RateLimited));
    assert_eq!(error.attempts, 1);
}

#[tokio::test]
async fn local_validation_and_response_validation_report_failure_metadata() {
    let client = Client::new(config("http://127.0.0.1:1".into())).unwrap();
    let invalid = EvaluationRequest::jev("state", BTreeMap::new());
    let failure = client.evaluate(&invalid).await.unwrap_err();
    assert!(matches!(failure.error, Error::InvalidRequest { .. }));
    assert_eq!(failure.attempts, 0);

    let body = json!({
        "model": "jev-latest",
        "answers": {},
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let (base_url, _) = server(vec![response(200, &body, "")]).await;
    let failure = Client::new(config(base_url))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(failure.error, Error::InvalidResponse { .. }));
    assert_eq!(failure.attempts, 1);
}

#[tokio::test]
async fn connection_failure_is_classified_as_transport() {
    let failure = Client::new(config(unavailable_base_url()))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(failure.error, Error::Transport { .. }));
    assert_eq!(failure.attempts, 1);
}

#[tokio::test]
async fn redirect_is_not_followed() {
    let (base_url, requests) = server(vec![response(
        307,
        "{}",
        "Location: http://example.com/downgrade\r\n",
    )])
    .await;
    let failure = Client::new(config(base_url))
        .unwrap()
        .evaluate(&request())
        .await
        .unwrap_err();
    assert!(matches!(failure.error, Error::HttpStatus { status: 307 }));
    assert_eq!(requests.lock().await.len(), 1);
}
