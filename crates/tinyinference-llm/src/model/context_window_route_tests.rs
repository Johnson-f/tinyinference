use super::*;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, Uri},
    routing::get,
};
use serde_json::{Value, json};
use tinyinference_providers::route::LocalProviderKind;

type RecordedRequests = Arc<Mutex<Vec<(String, Value, bool)>>>;

#[tokio::test]
async fn route_resolution_preserves_override_and_host_managed_mapping() {
    let resolver = ContextWindowResolver::new();
    let route = ContextWindowRouteInput {
        config_override: Some(200_000),
        managed_default: Some(ManagedDefaultWindow {
            model: "host/managed-default".to_owned(),
            window: 1_000_000,
        }),
        endpoint: Some("http://127.0.0.1:1/v1".to_owned()),
        ..Default::default()
    };
    let resolved = resolver
        .resolve_for_route(
            "managed",
            "host/managed-default",
            None,
            route,
            &ModelLimitsCache::default(),
            || panic!("an override must not construct a network client"),
        )
        .await;

    assert_eq!(resolved.window, Some(200_000));
    assert_eq!(resolved.source, WindowSource::ConfigOverride);
}

#[tokio::test]
async fn ollama_route_builds_native_show_request_and_uses_host_client_factory() {
    let requests: RecordedRequests = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/api/show", axum::routing::post(show_model))
        .with_state(requests.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let cache = ModelLimitsCache::default();
    let resolver = ContextWindowResolver::new();
    let route = ContextWindowRouteInput {
        endpoint: Some(format!("http://{address}")),
        ..Default::default()
    };
    let resolved = resolver
        .resolve_for_route(
            "ollama",
            "qwen3:14b@0.2",
            Some(LocalProviderKind::Ollama),
            route,
            &cache,
            || {
                reqwest::Client::builder()
                    .default_headers(
                        [(
                            reqwest::header::HeaderName::from_static("x-route-client"),
                            reqwest::header::HeaderValue::from_static("host-configured"),
                        )]
                        .into_iter()
                        .collect(),
                    )
                    .build()
                    .unwrap()
            },
        )
        .await;
    server.abort();

    assert_eq!(resolved.window, Some(40_960));
    assert_eq!(resolved.source, WindowSource::ProviderReported);
    let requests = requests.lock().unwrap();
    assert!(
        requests
            .iter()
            .any(|(path, _, configured)| path == "/v1/models" && *configured)
    );
    assert!(requests.iter().any(|(path, body, configured)| {
        path == "/api/show" && body == &json!({"model":"qwen3:14b"}) && *configured
    }));
}

#[tokio::test]
async fn ollama_route_accepts_a_host_fetcher_without_network_or_client_construction() {
    let fetcher = RecordingOllamaFetcher::default();
    let resolved = ContextWindowResolver::new()
        .resolve_for_route_with_fetcher(
            "ollama",
            "qwen3:14b@0.2",
            Some(LocalProviderKind::Ollama),
            ContextWindowRouteInput {
                endpoint: Some("http://ollama.test".to_owned()),
                ..Default::default()
            },
            &ModelLimitsCache::default(),
            &fetcher,
        )
        .await;

    assert_eq!(resolved.window, Some(40_960));
    assert_eq!(resolved.source, WindowSource::ProviderReported);
    let requests = fetcher.requests.lock().unwrap();
    assert!(
        requests
            .iter()
            .any(|(method, url, _)| { method == "GET" && url == "http://ollama.test/v1/models" })
    );
    assert!(requests.iter().any(|(method, url, body)| {
        method == "POST"
            && url == "http://ollama.test/api/show"
            && body.as_ref() == Some(&json!({"model":"qwen3:14b"}))
    }));
}

#[tokio::test]
async fn route_resolution_keeps_learned_limits_and_endpoint_identity_isolated() {
    let calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/{route}/v1/models", get(route_models))
        .with_state(calls.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let cache = ModelLimitsCache::default();
    let resolver = ContextWindowResolver::new();
    let first_endpoint = format!("http://{address}/first/v1");
    let first = resolver
        .resolve_for_route(
            "provider",
            "model",
            None,
            ContextWindowRouteInput {
                endpoint: Some(first_endpoint.clone()),
                ..Default::default()
            },
            &cache,
            reqwest::Client::new,
        )
        .await;
    assert_eq!(first.window, Some(128_000));
    assert_eq!(first.max_output_tokens, Some(4_096));

    super::super::discover::record_overflow_error_in(
        &cache,
        &first_endpoint,
        "model",
        "maximum context length is 64000 tokens",
    );
    let learned = resolver
        .resolve_for_route(
            "provider",
            "model",
            None,
            ContextWindowRouteInput {
                endpoint: Some(first_endpoint),
                ..Default::default()
            },
            &cache,
            reqwest::Client::new,
        )
        .await;
    assert_eq!(learned.window, Some(64_000));
    assert_eq!(learned.source, WindowSource::LearnedFromOverflow);
    assert_eq!(learned.max_output_tokens, Some(4_096));

    let other_endpoint = format!("http://{address}/other/v1");
    let other = resolver
        .resolve_for_route(
            "provider",
            "model",
            None,
            ContextWindowRouteInput {
                endpoint: Some(other_endpoint),
                ..Default::default()
            },
            &cache,
            reqwest::Client::new,
        )
        .await;
    server.abort();

    assert_eq!(other.window, Some(96_000));
    assert_eq!(other.source, WindowSource::ProviderReported);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

async fn list_models(State(requests): State<RecordedRequests>, headers: HeaderMap) -> Json<Value> {
    let configured = headers
        .get("x-route-client")
        .and_then(|value| value.to_str().ok())
        == Some("host-configured");
    requests
        .lock()
        .unwrap()
        .push(("/v1/models".to_owned(), Value::Null, configured));
    Json(json!({"object":"list","data":[{"id":"qwen3:14b"}]}))
}

async fn show_model(
    State(requests): State<RecordedRequests>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    let configured = headers
        .get("x-route-client")
        .and_then(|value| value.to_str().ok())
        == Some("host-configured");
    requests
        .lock()
        .unwrap()
        .push(("/api/show".to_owned(), body, configured));
    Json(json!({
        "model_info": {"qwen3.context_length":40960},
        "capabilities":["completion"]
    }))
}

#[derive(Default)]
struct RecordingOllamaFetcher {
    requests: Mutex<Vec<(String, String, Option<Value>)>>,
}

#[async_trait::async_trait]
impl ModelListingFetcher for RecordingOllamaFetcher {
    async fn get_json(&self, url: &str, _headers: &[(String, String)]) -> crate::Result<Value> {
        self.requests
            .lock()
            .unwrap()
            .push(("GET".to_owned(), url.to_owned(), None));
        Ok(json!({"data":[{"id":"qwen3:14b"}]}))
    }

    async fn post_json(
        &self,
        url: &str,
        _headers: &[(String, String)],
        body: &Value,
    ) -> crate::Result<Value> {
        self.requests
            .lock()
            .unwrap()
            .push(("POST".to_owned(), url.to_owned(), Some(body.clone())));
        Ok(json!({
            "model_info":{"qwen3.context_length":40960},
            "capabilities":["completion"]
        }))
    }
}

async fn route_models(State(calls): State<Arc<AtomicUsize>>, uri: Uri) -> Json<Value> {
    calls.fetch_add(1, Ordering::SeqCst);
    let window = if uri.path().starts_with("/first/") {
        128_000
    } else {
        96_000
    };
    Json(json!({"data":[{"id":"model","context_length":window,"max_output_tokens":4096}]}))
}

#[tokio::test]
async fn route_static_precedence_uses_host_mapping_workload_catalog_and_local_profile() {
    let resolver = ContextWindowResolver::new();
    let cache = ModelLimitsCache::default();
    let managed = resolver
        .resolve_for_route(
            "managed",
            "host/managed-default",
            None,
            ContextWindowRouteInput {
                managed_default: Some(ManagedDefaultWindow {
                    model: "host/managed-default".to_owned(),
                    window: 1_000_000,
                }),
                catalog_window: Some(80_000),
                ..Default::default()
            },
            &cache,
            || panic!("a route without an endpoint must not build a client"),
        )
        .await;
    assert_eq!(managed.window, Some(1_000_000));
    assert_eq!(managed.source, WindowSource::StaticGuess);

    let workload = resolver
        .resolve_for_route(
            "external",
            "hint:agentic",
            None,
            ContextWindowRouteInput {
                catalog_window: Some(70_000),
                ..Default::default()
            },
            &cache,
            || panic!("a route without an endpoint must not build a client"),
        )
        .await;
    assert_eq!(workload.window, Some(200_000));

    let catalog = resolver
        .resolve_for_route(
            "external",
            "unlisted-model",
            None,
            ContextWindowRouteInput {
                catalog_window: Some(70_000),
                ..Default::default()
            },
            &cache,
            || panic!("a route without an endpoint must not build a client"),
        )
        .await;
    assert_eq!(catalog.window, Some(70_000));

    let local = resolver
        .resolve_for_route(
            "lmstudio",
            "unknown-local-model",
            Some(LocalProviderKind::LmStudio),
            ContextWindowRouteInput {
                local_window: Some(32_000),
                ..Default::default()
            },
            &cache,
            || panic!("a route without an endpoint must not build a client"),
        )
        .await;
    assert_eq!(local.window, Some(32_000));
    assert_eq!(local.source, WindowSource::LocalProfile);
}

#[test]
fn workload_context_hints_cover_legacy_aliases_without_a_managed_model_id() {
    assert_eq!(
        super::super::workload::context_window_for_model(" hint:reasoning "),
        Some(1_000_000)
    );
    assert_eq!(
        super::super::workload::context_window_for_model("agentic-v1"),
        Some(200_000)
    );
    assert_eq!(
        super::super::workload::context_window_for_model("burst-v1"),
        Some(128_000)
    );
    assert_eq!(
        super::super::workload::context_window_for_model("gemma3:1b"),
        Some(8_192)
    );
    assert_eq!(
        super::super::workload::context_window_for_model("chat"),
        Some(1_000_000)
    );
    assert_eq!(
        super::super::workload::context_window_for_model("host/managed-default"),
        None
    );
}
