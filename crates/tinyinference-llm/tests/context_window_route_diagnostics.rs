//! Public-route diagnostics behavior with a scoped tracing subscriber.

use std::sync::{Arc, Mutex};

use axum::{Json, Router, routing::get};
use serde_json::{Value, json};
use tinyinference_llm::model::{
    ContextWindowResolver, ContextWindowRouteInput, WindowSource, discover::ModelLimitsCache,
};
use tinyinference_providers::route::LocalProviderKind;
use tracing::{Event, Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, prelude::*};

#[tokio::test]
async fn route_diagnostics_warn_once_skip_local_fallback_and_redact_discovery_endpoint() {
    let capture = CapturedEvents::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);
    let resolver = ContextWindowResolver::new();
    let cache = ModelLimitsCache::default();

    for _ in 0..2 {
        resolver
            .resolve_for_route(
                "external",
                "route-test-static-warning-once",
                None,
                ContextWindowRouteInput {
                    catalog_window: Some(64_000),
                    ..Default::default()
                },
                &cache,
                || panic!("a route without an endpoint must not build a client"),
            )
            .await;
    }

    let local = resolver
        .resolve_for_route(
            "lmstudio",
            "route-test-unknown-local",
            Some(LocalProviderKind::LmStudio),
            ContextWindowRouteInput::default(),
            &cache,
            || panic!("a route without an endpoint must not build a client"),
        )
        .await;
    assert_eq!(local.source, WindowSource::LocalProfile);

    let app = Router::new().route("/v1/models", get(redaction_models));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let endpoint = format!("http://operator:route-secret@{address}/v1");
    let provider = resolver
        .resolve_for_route(
            "provider",
            "route-test-provider-window",
            None,
            ContextWindowRouteInput {
                endpoint: Some(endpoint),
                ..Default::default()
            },
            &cache,
            reqwest::Client::new,
        )
        .await;
    server.abort();
    assert_eq!(provider.source, WindowSource::ProviderReported);

    let events = capture.0.lock().unwrap();
    let warnings: Vec<_> = events
        .iter()
        .filter(|(level, message)| {
            level == "WARN"
                && message.contains("context window is a static guess")
                && message.contains("route-test-static-warning-once")
        })
        .collect();
    let repeated_guess_debug = events
        .iter()
        .filter(|(level, message)| {
            level == "DEBUG"
                && message.contains("context window is a static guess")
                && message.contains("route-test-static-warning-once")
        })
        .count();
    assert_eq!(warnings.len(), 1);
    assert!(repeated_guess_debug >= 1);
    let warning_position = events
        .iter()
        .position(|(level, message)| {
            level == "WARN"
                && message.contains("context window is a static guess")
                && message.contains("route-test-static-warning-once")
        })
        .unwrap();
    assert!(
        events
            .iter()
            .enumerate()
            .any(|(position, (level, message))| {
                position > warning_position
                    && level == "DEBUG"
                    && message.contains("context window is a static guess")
                    && message.contains("route-test-static-warning-once")
            })
    );
    assert!(!events.iter().any(|(level, message)| {
        level == "WARN"
            && message.contains("context window is a static guess")
            && (message.contains("route-test-unknown-local")
                || message.contains("route-test-provider-window"))
    }));
    assert!(
        !events.iter().any(|(_, message)| {
            message.contains("route-secret")
                && (message.contains("route-test-provider-window") || message.contains("endpoint="))
        }),
        "captured events: {events:#?}"
    );
    assert!(
        events.iter().any(|(_, message)| {
            message.contains("endpoint=")
                && message.contains(&format!("http://127.0.0.1:{}/v1", address.port()))
        }),
        "captured events: {events:#?}"
    );
}

async fn redaction_models() -> Json<Value> {
    Json(json!({
        "data": [{
            "id": "route-test-provider-window",
            "context_length": 128000,
            "max_output_tokens": 8192
        }]
    }))
}

#[derive(Clone, Default)]
struct CapturedEvents(Arc<Mutex<Vec<(String, String)>>>);

impl<S: Subscriber> Layer<S> for CapturedEvents {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        let mut fields = CapturedFields::default();
        event.record(&mut fields);
        self.0.lock().unwrap().push((
            event.metadata().level().to_string(),
            fields
                .0
                .into_iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(" "),
        ));
    }
}

#[derive(Default)]
struct CapturedFields(Vec<(String, String)>);

impl Visit for CapturedFields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_owned(), value.to_owned()));
    }
}
