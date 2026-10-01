use super::*;

#[tokio::test]
async fn list_models_degrades_on_200_with_non_json_body() {
    // TAURI-RUST-560: a 2xx response whose body is not an Ollama tags JSON
    // object (a different local server/proxy, a captive portal, an HTML page
    // bound to the configured Ollama port) must degrade gracefully — return
    // `Err` so the diagnostics caller surfaces `tags_error` and an empty model
    // list — rather than emit an `error!`-level event that floods Sentry on
    // every diagnostics poll. The parse-failure log is now demoted to `warn!`
    // (a breadcrumb) to match the A3T non-success treatment.
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/api/tags",
        // 200 OK, but the body is an HTML page, not Ollama tags JSON.
        get(|| async {
            (
                axum::http::StatusCode::OK,
                "<!doctype html><html><head><title>Sign in</title></head>\
                 <body>Captive portal</body></html>",
            )
        }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = Config::default();
    let service = LocalAiService::new(&config);
    let err = service.list_models_at(&base).await.unwrap_err();
    assert!(
        err.contains("parse failed"),
        "200 non-JSON body must yield a graceful parse-failed Err, got: {err}"
    );
    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn lm_studio_list_models_returns_loaded_models() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/models",
        get(|| async {
            Json(json!({
                "object": "list",
                "data": [
                    { "id": "local-model", "object": "model", "owned_by": "lm-studio" },
                    { "id": "second-model", "object": "model", "owned_by": "lm-studio" }
                ]
            }))
        }),
    );
    let base = spawn_mock(app).await;
    let config = lm_studio_config(&base);
    let service = LocalAiService::new(&config);

    let models = service
        .list_lm_studio_models(&config)
        .await
        .expect("lm studio models");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name, "local-model");
    assert!(
        service
            .has_lm_studio_model(&config, "local-model")
            .await
            .expect("has model")
    );
}

#[tokio::test]
async fn lm_studio_diagnostics_reports_loaded_chat_model() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/models",
        get(|| async {
            Json(json!({
                "data": [
                    { "id": "local-model", "object": "model", "owned_by": "lm-studio" }
                ]
            }))
        }),
    );
    let base = spawn_mock(app).await;
    let config = lm_studio_config(&base);
    let service = LocalAiService::new(&config);

    let diag = service.diagnostics(&config).await.expect("diagnostics");

    assert_eq!(diag["provider"].as_str(), Some("lm_studio"));
    assert_eq!(diag["lm_studio_running"], true);
    assert_eq!(diag["expected"]["chat_found"], true);
    assert_eq!(diag["ok"], true);
}

/// Regression for GH #5053: a custom OpenAI-compatible BYOK endpoint on
/// localhost (e.g. LM Studio at `http://localhost:1234/v1`) whose `provider`
/// tag still defaults to `ollama` must be probed with `/v1/models`, NOT the
/// Ollama-native `/api/tags`. The mock serves ONLY `/v1/models` and no
/// `/api/tags`, so before the fix diagnostics took the Ollama branch,
/// hit an unrouted `/v1/api/tags`, and reported the model absent; after the
/// fix the `/v1` endpoint type routes discovery to `/v1/models` and the model
/// is found.
#[tokio::test]
async fn diagnostics_openai_compatible_v1_endpoint_uses_v1_models_not_api_tags() {
    let _guard = crate::service::inference_test_guard();

    // OpenAI-compatible server: exposes `/v1/models` and deliberately no
    // `/api/tags` — an Ollama probe here would 404 (silently empty discovery).
    let app = Router::new().route(
        "/v1/models",
        get(|| async {
            Json(json!({
                "data": [
                    { "id": "local-model", "object": "model", "owned_by": "lm-studio" }
                ]
            }))
        }),
    );
    let base = spawn_mock(app).await;

    // The #5053 config: a `/v1` OpenAI-compatible endpoint whose provider tag is
    // the defaulted `ollama` (not `lm_studio`).
    let mut config = lm_studio_config(&base);
    config.local_ai.provider = "ollama".to_string();

    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");

    // `lm_studio_running` is emitted only by the OpenAI-compatible (`/v1/models`)
    // diagnostics path — the Ollama branch reports `ollama_running` and leaves
    // this key null. Its presence proves discovery was routed by endpoint type,
    // not sent to `/api/tags`.
    assert_eq!(diag["lm_studio_running"], true);
    let installed = diag["installed_models"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        installed
            .iter()
            .any(|m| m["name"].as_str() == Some("local-model")),
        "OpenAI-compatible /v1 endpoint must discover models via /v1/models, got: {:?}",
        installed
    );
}

#[tokio::test]
async fn lm_studio_diagnostics_flags_missing_chat_model() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/models",
        get(|| async {
            Json(json!({
                "data": [
                    { "id": "other-model", "object": "model", "owned_by": "lm-studio" }
                ]
            }))
        }),
    );
    let base = spawn_mock(app).await;
    let config = lm_studio_config(&base);
    let service = LocalAiService::new(&config);

    let diag = service.diagnostics(&config).await.expect("diagnostics");

    assert_eq!(diag["provider"].as_str(), Some("lm_studio"));
    assert_eq!(diag["expected"]["chat_found"], false);
    assert_eq!(diag["ok"], false);
    assert!(
        diag["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue.as_str().unwrap_or("").contains("local-model"))
    );
}

#[tokio::test]
async fn lm_studio_diagnostics_surfaces_reachable_model_list_errors() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route("/v1/models", get(|| async { "not json" }));
    let base = spawn_mock(app).await;
    let config = lm_studio_config(&base);
    let service = LocalAiService::new(&config);

    let diag = service.diagnostics(&config).await.expect("diagnostics");

    assert_eq!(diag["provider"].as_str(), Some("lm_studio"));
    assert_eq!(diag["lm_studio_running"], true);
    assert_eq!(diag["ok"], false);
    assert!(diag["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            .as_str()
            .unwrap_or("")
            .contains("Failed to list LM Studio models")
    }));
    assert!(
        !diag["repair_actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["action"].as_str() == Some("load_lm_studio_model"))
    );
}

// ---- owned-PID lifecycle ------------------------------------------------
//
// These tests pin the contract that `kill_ollama_server` only touches
// daemons openhuman spawned itself, and that the kill path actually
// reaches the child process (the previous `taskkill /F /IM ollama.exe` /
// `pkill -f` would terminate any Ollama on the host, including ones the
// user started outside openhuman — the issue #1622 friendly-fire bug).

// ── ollama_binary_present short-circuit tests ─────────────────────────────

// The custom-path branch of `ollama_binary_present` is covered by
// `assets_status_sets_ollama_available_false_when_binary_missing` above, which
// already calls `service.ollama_binary_present(&config)` and asserts that
// downstream `assets_status` reports `ollama_available = false` whenever the
// helper returns false. A dedicated nonexistent-custom-path test that scrubs
// PATH globally was attempted but caused parallel-test interference (PATH=""
// poisoned the local_ai_test_guard mutex for sibling tests that legitimately
// rely on PATH). The behavior is covered; an isolated branch test would
// require per-process isolation that the existing harness doesn't support.

#[tokio::test]
async fn diagnostics_gates_models_by_context_window() {
    let _guard = crate::service::inference_test_guard();

    // /api/tags lists two models; /api/show reports their context windows:
    // one at the 8192 floor (accepted) and one well below (rejected).
    let app = Router::new()
        .route(
            "/api/tags",
            get(|| async {
                Json(json!({
                    "models": [
                        {"name": "bge-m3:latest", "modified_at": "", "size": 1u64, "digest": "d"},
                        {"name": "tiny-embed:latest", "modified_at": "", "size": 2u64, "digest": "d"}
                    ]
                }))
            }),
        )
        .route(
            "/api/show",
            axum::routing::post(|Json(body): Json<serde_json::Value>| async move {
                let model = body["model"].as_str().unwrap_or_default().to_string();
                let ctx = if model.starts_with("bge-m3") { 8192 } else { 2048 };
                Json(json!({
                    "model_info": {
                        "general.architecture": "bert",
                        "bert.context_length": ctx,
                    },
                    "capabilities": ["embedding"],
                }))
            }),
        );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = Config::default();
    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");

    assert_eq!(diag["ollama_running"], true);
    assert_eq!(diag["context_requirement"]["min_context_tokens"], 8192);

    let models = diag["installed_models"]
        .as_array()
        .expect("installed_models");
    let by_name = |needle: &str| {
        models
            .iter()
            .find(|m| m["name"].as_str().unwrap_or("").starts_with(needle))
            .unwrap_or_else(|| panic!("model {needle} missing"))
            .clone()
    };

    let accepted = by_name("bge-m3");
    assert_eq!(accepted["context_length"], 8192);
    assert_eq!(accepted["eligibility"]["status"], "ok");

    let rejected = by_name("tiny-embed");
    assert_eq!(rejected["context_length"], 2048);
    assert_eq!(rejected["eligibility"]["status"], "below_minimum");
    assert_eq!(rejected["eligibility"]["required"], 8192);

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}
