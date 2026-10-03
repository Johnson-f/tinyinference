use super::*;

use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{collections::VecDeque, sync::Mutex};

const ONE_RESULT: &str = r#"{"data":[{"index":0,"relevance_score":0.75}]}"#;

fn request() -> RerankRequest {
    RerankRequest::new("query", vec!["document".into()])
}

struct ScriptedTransport {
    responses: Mutex<VecDeque<Result<reqwest::Response>>>,
    requests: Mutex<Vec<reqwest::Request>>,
}

#[async_trait]
impl HttpTransport for ScriptedTransport {
    async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response> {
        self.requests.lock().unwrap().push(request);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected provider call")
    }
}

fn response(status: u16, body: &str) -> Result<reqwest::Response> {
    Ok(http::Response::builder()
        .status(status)
        .body(body.to_owned())
        .unwrap()
        .into())
}

fn scripted(responses: Vec<Result<reqwest::Response>>) -> (VoyageReranker, Arc<ScriptedTransport>) {
    let transport = Arc::new(ScriptedTransport {
        responses: Mutex::new(responses.into()),
        requests: Mutex::new(vec![]),
    });
    let mut model = VoyageReranker::with_options(
        "test-key",
        VoyageRerankConfig {
            base_url: "http://localhost/v1".into(),
            ..VoyageRerankConfig::default()
        },
    )
    .unwrap();
    model.transport = transport.clone();
    (model, transport)
}

#[tokio::test]
async fn voyage_preserves_request_text_and_maps_ranked_positions_with_usage() {
    let (model, transport) = scripted(vec![response(
        200,
        r#"{
        "data":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}],
        "usage":{"total_tokens":17},"extra":"ignored"
    }"#,
    )]);
    let mut request = RerankRequest::new(" Which? ", vec!["line\n\"one\"".into(), "café".into()]);
    request.top_k = Some(9);
    let result = model.rerank(request).await.unwrap();
    assert_eq!(
        result.results,
        vec![
            super::super::RerankResult {
                index: 1,
                relevance_score: 0.9
            },
            super::super::RerankResult {
                index: 0,
                relevance_score: 0.2
            },
        ]
    );
    assert_eq!(
        result.usage,
        Some(super::super::RerankUsage { input_tokens: 17 })
    );
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method(), reqwest::Method::POST);
    assert_eq!(requests[0].url().as_str(), "http://localhost/v1/rerank");
    assert_eq!(
        requests[0].headers()[reqwest::header::AUTHORIZATION],
        "Bearer test-key"
    );
    let body: serde_json::Value =
        serde_json::from_slice(requests[0].body().unwrap().as_bytes().unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "query":" Which? ", "documents":["line\n\"one\"", "café"], "model":"rerank-3",
            "top_k":2, "return_documents":false, "truncation":false
        })
    );
}

#[tokio::test]
async fn duplicate_result_positions_are_not_returned_as_success() {
    let (model, _) = scripted(vec![response(
        200,
        r#"{
        "data":[{"index":0,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]
    }"#,
    )]);
    let result = model
        .rerank(RerankRequest::new("query", vec!["a".into(), "b".into()]))
        .await;
    assert!(matches!(
        result,
        Err(Error::Rerank(RerankError::InvalidResponse { .. }))
    ));
}

#[tokio::test]
async fn invalid_results_fail_without_returning_partial_rankings() {
    for body in [
        "not json",
        "{}",
        r#"{"data":[]}"#,
        r#"{"data":[{"index":1,"relevance_score":0.5}]}"#,
        r#"{"data":[{"index":-1,"relevance_score":0.5}]}"#,
        r#"{"data":[{"index":0.5,"relevance_score":0.5}]}"#,
        r#"{"data":[{"index":"0","relevance_score":0.5}]}"#,
        r#"{"data":[{"relevance_score":0.5}]}"#,
        r#"{"data":[{"index":0}]}"#,
        r#"{"data":[{"index":0,"relevance_score":"high"}]}"#,
        r#"{"data":[{"index":0,"relevance_score":1e400}]}"#,
        r#"{"data":[{"index":0,"relevance_score":0.5},{"index":0,"relevance_score":0.3}]}"#,
    ] {
        let (model, transport) = scripted(vec![response(200, body)]);
        assert!(
            matches!(
                model.rerank(request()).await,
                Err(Error::Rerank(RerankError::InvalidResponse { .. }))
            ),
            "{body}"
        );
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn results_sort_by_score_and_original_index_without_deduplicating_text() {
    let (model, _) = scripted(vec![response(
        200,
        r#"{"data":[
        {"index":2,"relevance_score":-2.0},
        {"index":1,"relevance_score":7.0},
        {"index":0,"relevance_score":7.0}
    ]}"#,
    )]);
    let result = model
        .rerank(RerankRequest::new("query", vec!["same".into(); 3]))
        .await
        .unwrap();
    assert_eq!(
        result
            .results
            .iter()
            .map(|hit| hit.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(result.results[2].relevance_score, -2.0);
}

#[tokio::test]
async fn malformed_usage_does_not_destroy_valid_rankings() {
    for usage in [
        json!(null),
        json!("bad"),
        json!({}),
        json!({"total_tokens":-1}),
        json!({"total_tokens":1.5}),
        json!({"total_tokens":1e30}),
        json!({"total_tokens":"7"}),
    ] {
        let body = json!({"data":[{"index":0,"relevance_score":0.75}], "usage":usage});
        let (model, _) = scripted(vec![response(200, &body.to_string())]);
        let result = model.rerank(request()).await.unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.usage, None);
    }
    let (model, _) = scripted(vec![response(
        200,
        r#"{
        "data":[{"index":0,"relevance_score":0.75}],"usage":{"total_tokens":0}
    }"#,
    )]);
    assert_eq!(
        model.rerank(request()).await.unwrap().usage,
        Some(RerankUsage { input_tokens: 0 })
    );
}

#[tokio::test]
async fn invalid_inputs_fail_before_transport_and_zero_results_are_a_noop() {
    for invalid in [
        RerankRequest::new(" \n", vec!["document".into()]),
        RerankRequest::new("query", vec!["document".into(), " \t".into()]),
        RerankRequest::new("query", vec!["document".into(); 1001]),
    ] {
        let (model, transport) = scripted(vec![]);
        assert!(matches!(
            model.rerank(invalid).await,
            Err(Error::Validation(_))
        ));
        assert!(transport.requests.lock().unwrap().is_empty());
    }
    let (model, transport) = scripted(vec![]);
    let mut empty = RerankRequest::new("", vec!["".into()]);
    empty.top_k = Some(0);
    assert!(model.rerank(empty).await.unwrap().results.is_empty());
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[test]
fn rejects_invalid_configuration_without_exposing_credentials() {
    for endpoint in [
        "",
        "file:///tmp/provider",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/v1#secret",
        "http://example.com/v1",
    ] {
        let error = VoyageReranker::with_options(
            "secret",
            VoyageRerankConfig {
                base_url: endpoint.into(),
                ..VoyageRerankConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(!format!("{error:?}").contains("secret"));
    }
    for config in [
        VoyageRerankConfig {
            model: " ".into(),
            ..Default::default()
        },
        VoyageRerankConfig {
            timeout: Duration::ZERO,
            ..Default::default()
        },
        VoyageRerankConfig {
            max_retries: 4,
            ..Default::default()
        },
        VoyageRerankConfig {
            max_request_bytes: 0,
            ..Default::default()
        },
        VoyageRerankConfig {
            max_response_bytes: 0,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            VoyageReranker::with_options("secret", config),
            Err(Error::Validation(_))
        ));
    }
    for key in ["", " \t", "secret\nheader"] {
        assert!(matches!(
            VoyageReranker::new(key),
            Err(Error::Validation(_))
        ));
    }
}

#[tokio::test]
async fn request_limit_counts_json_escaping_and_truncation_is_explicit() {
    let (mut model, transport) = scripted(vec![]);
    model.config.max_request_bytes = 200;
    let too_large = RerankRequest::new("query", vec!["\u{0001}".repeat(40)]);
    assert!(matches!(
        model.rerank(too_large).await,
        Err(Error::Validation(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());

    let (model, transport) = scripted(vec![response(200, ONE_RESULT)]);
    let mut truncate = request();
    truncate.truncate = true;
    model.rerank(truncate).await.unwrap();
    let requests = transport.requests.lock().unwrap();
    let body: Value =
        serde_json::from_slice(requests[0].body().unwrap().as_bytes().unwrap()).unwrap();
    assert_eq!(body["truncation"], true);
}

#[tokio::test]
async fn response_limit_is_enforced_on_chunks_with_absent_or_false_length() {
    for length in [None, Some("1"), Some("100")] {
        let stream = futures::stream::iter(vec![Ok::<_, std::io::Error>("123"), Ok("456")]);
        let mut builder = http::Response::builder();
        if let Some(length) = length {
            builder = builder.header("content-length", length);
        }
        let incoming = builder
            .body(reqwest::Body::wrap_stream(stream))
            .unwrap()
            .into();
        let (mut model, transport) = scripted(vec![Ok(incoming)]);
        model.config.max_response_bytes = 5;
        assert!(matches!(
            model.rerank(request()).await,
            Err(Error::Rerank(RerankError::ResponseTooLarge { limit: 5 }))
        ));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn rate_limit_response_retries_then_returns_success() {
    let throttled = http::Response::builder()
        .status(429)
        .header("retry-after", "2")
        .body("secret provider body".to_owned())
        .unwrap()
        .into();
    let (model, transport) = scripted(vec![Ok(throttled), response(200, ONE_RESULT)]);
    let start = tokio::time::Instant::now();
    assert_eq!(model.rerank(request()).await.unwrap().results[0].index, 0);
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
    assert_eq!(start.elapsed(), Duration::from_secs(2));
}

#[tokio::test]
async fn precancellation_wins_even_for_empty_or_invalid_input() {
    for mut input in [
        RerankRequest::new("", vec![]),
        RerankRequest::new("", vec!["".into()]),
    ] {
        let (model, transport) = scripted(vec![]);
        input.cancellation.cancel();
        input.top_k = Some(0);
        assert!(matches!(model.rerank(input).await, Err(Error::Cancelled)));
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test(start_paused = true)]
async fn deadline_covers_a_response_body_that_never_finishes() {
    let pending = futures::stream::pending::<std::result::Result<String, std::io::Error>>();
    let incoming = http::Response::new(reqwest::Body::wrap_stream(pending)).into();
    let (mut model, transport) = scripted(vec![Ok(incoming)]);
    model.config.timeout = Duration::from_secs(3);
    let result = tokio::time::timeout(Duration::from_secs(10), model.rerank(request()))
        .await
        .expect("the operation must enforce its own deadline");
    assert!(matches!(result, Err(Error::Rerank(RerankError::Timeout))));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn only_documented_transient_statuses_retry_with_a_bounded_attempt_count() {
    for status in [429, 500, 502, 503, 504] {
        let (model, transport) = scripted(vec![response(status, ""), response(200, ONE_RESULT)]);
        assert!(model.rerank(request()).await.is_ok(), "status {status}");
        assert_eq!(transport.requests.lock().unwrap().len(), 2);
    }
    let (model, transport) = scripted((0..4).map(|_| response(503, "")).collect());
    assert!(matches!(
        model.rerank(request()).await,
        Err(Error::Rerank(RerankError::HttpStatus { status: 503 }))
    ));
    assert_eq!(transport.requests.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn permanent_failures_and_disabled_retries_make_one_attempt() {
    for status in [301, 400, 401, 403, 404, 422, 501] {
        let (model, transport) = scripted(vec![response(status, "secret-query test-key")]);
        let error = model.rerank(request()).await.unwrap_err();
        match status {
            401 | 403 => assert!(
                matches!(error, Error::Rerank(RerankError::Authentication { status: actual }) if actual == status)
            ),
            _ => assert!(
                matches!(error, Error::Rerank(RerankError::HttpStatus { status: actual }) if actual == status)
            ),
        }
        assert!(!format!("{error:?} {error}").contains("secret-query"));
        assert!(!format!("{error:?} {error}").contains("test-key"));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
    let throttled = http::Response::builder()
        .status(429)
        .header("retry-after", "9")
        .body(String::new())
        .unwrap()
        .into();
    let (mut model, transport) = scripted(vec![Ok(throttled)]);
    model.config.max_retries = 0;
    assert!(matches!(
        model.rerank(request()).await,
        Err(Error::Rerank(RerankError::RateLimited {
            retry_after_ms: Some(9000)
        }))
    ));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn transport_and_body_read_failures_are_not_replayed() {
    let failing_body = futures::stream::iter(vec![Err::<String, _>(std::io::Error::other(
        "test-key secret-query",
    ))]);
    for incoming in [
        Err(RerankError::Transport.into()),
        Ok(http::Response::new(reqwest::Body::wrap_stream(failing_body)).into()),
    ] {
        let (model, transport) = scripted(vec![incoming]);
        let error = model.rerank(request()).await.unwrap_err();
        assert!(matches!(error, Error::Rerank(RerankError::Transport)));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
        assert!(!format!("{error:?} {error}").contains("test-key"));
        assert!(!format!("{error:?} {error}").contains("secret-query"));
    }
}

struct DropSignal(Arc<AtomicBool>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct StalledTransport {
    dropped: Arc<AtomicBool>,
}

#[async_trait]
impl HttpTransport for StalledTransport {
    async fn send(&self, _request: reqwest::Request) -> Result<reqwest::Response> {
        let _signal = DropSignal(self.dropped.clone());
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancellation_drops_in_flight_transport_and_body_read() {
    for waiting_for_body in [false, true] {
        let dropped = Arc::new(AtomicBool::new(false));
        let (mut model, _) = scripted(vec![]);
        if waiting_for_body {
            let signal = DropSignal(dropped.clone());
            let body = futures::stream::once(async move {
                let _signal = signal;
                std::future::pending::<std::result::Result<String, std::io::Error>>().await
            });
            model = scripted(vec![Ok(http::Response::new(reqwest::Body::wrap_stream(
                body,
            ))
            .into())])
            .0;
        } else {
            model.transport = Arc::new(StalledTransport {
                dropped: dropped.clone(),
            });
        }
        let input = request();
        let cancel = input.cancellation.clone();
        let operation = model.rerank(input);
        tokio::pin!(operation);
        assert!(futures::poll!(&mut operation).is_pending());
        assert!(!dropped.load(Ordering::SeqCst));
        cancel.cancel();
        assert!(matches!(operation.await, Err(Error::Cancelled)));
        assert!(dropped.load(Ordering::SeqCst));
    }
}

#[tokio::test(start_paused = true)]
async fn cancellation_interrupts_pacing_and_backoff_without_later_attempts() {
    for waiting_for_pacing in [false, true] {
        let (mut model, transport) = scripted(vec![response(429, "")]);
        if waiting_for_pacing {
            model.config.base_url = "https://rerank-cancel-pacing.invalid/v1".into();
            crate::rate_limit::acquire(&model.config.base_url).await;
        }
        let input = request();
        let cancel = input.cancellation.clone();
        let operation = model.rerank(input);
        tokio::pin!(operation);
        assert!(futures::poll!(&mut operation).is_pending());
        cancel.cancel();
        assert!(matches!(operation.await, Err(Error::Cancelled)));
        tokio::time::advance(Duration::from_secs(60)).await;
        assert_eq!(
            transport.requests.lock().unwrap().len(),
            usize::from(!waiting_for_pacing)
        );
    }
}

#[tokio::test(start_paused = true)]
async fn total_deadline_covers_pacing_and_all_retries() {
    let (mut model, transport) = scripted(vec![]);
    model.config.base_url = "https://rerank-deadline-pacing.invalid/v1".into();
    model.config.timeout = Duration::from_millis(100);
    crate::rate_limit::acquire(&model.config.base_url).await;
    assert!(matches!(
        model.rerank(request()).await,
        Err(Error::Rerank(RerankError::Timeout))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());

    let (mut model, transport) = scripted(vec![response(503, ""), response(503, "")]);
    model.config.timeout = Duration::from_secs(2);
    let start = tokio::time::Instant::now();
    assert!(matches!(
        model.rerank(request()).await,
        Err(Error::Rerank(RerankError::Timeout))
    ));
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
    assert_eq!(start.elapsed(), Duration::from_secs(2));
}

struct CancellingTransport {
    cancellation: RerankCancellation,
}

#[async_trait]
impl HttpTransport for CancellingTransport {
    async fn send(&self, _request: reqwest::Request) -> Result<reqwest::Response> {
        self.cancellation.cancel();
        Err(RerankError::Transport.into())
    }
}

#[tokio::test(start_paused = true)]
async fn observed_cancellation_wins_over_failure_and_simultaneous_deadline() {
    let (mut model, _) = scripted(vec![]);
    let input = request();
    model.transport = Arc::new(CancellingTransport {
        cancellation: input.cancellation.clone(),
    });
    assert!(matches!(model.rerank(input).await, Err(Error::Cancelled)));

    model.transport = Arc::new(StalledTransport {
        dropped: Arc::new(AtomicBool::new(false)),
    });
    model.config.timeout = Duration::from_secs(1);
    let input = request();
    let cancel = input.cancellation.clone();
    let operation = model.rerank(input);
    tokio::pin!(operation);
    assert!(futures::poll!(&mut operation).is_pending());
    tokio::time::advance(Duration::from_secs(1)).await;
    cancel.cancel();
    assert!(matches!(operation.await, Err(Error::Cancelled)));
}

#[test]
fn debug_omits_document_content_and_credentials() {
    let model = VoyageReranker::with_options(
        "secret-key",
        VoyageRerankConfig {
            model: "secret-key".into(),
            base_url: "https://example.com/secret-key".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!format!("{model:?}").contains("secret-key"));
    let request = RerankRequest::new("private-query", vec!["private-document".into()]);
    let debug = format!("{request:?}");
    assert!(!debug.contains("private-query"));
    assert!(!debug.contains("private-document"));
}
