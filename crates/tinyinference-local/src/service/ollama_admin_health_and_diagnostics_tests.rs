use super::*;

#[tokio::test]
async fn test_ollama_connection_returns_reachable_with_model_count() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/api/tags",
        get(|| async {
            Json(json!({
                "models": [
                    {"name": "llama3:latest", "modified_at": "", "size": 1u64, "digest": "d"},
                    {"name": "mistral:7b", "modified_at": "", "size": 2u64, "digest": "d"}
                ]
            }))
        }),
    );
    let base = spawn_mock(app).await;

    let result = super::super::test_ollama_connection(&base).await.unwrap();
    assert_eq!(result["reachable"], true);
    assert_eq!(result["models_count"], 2);
    assert!(result["error"].is_null());
}

#[tokio::test]
async fn test_ollama_connection_returns_unreachable_on_server_error() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/api/tags",
        get(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
    );
    let base = spawn_mock(app).await;

    let result = super::super::test_ollama_connection(&base).await.unwrap();
    assert_eq!(result["reachable"], false);
    assert!(!result["error"].as_str().unwrap_or("").is_empty());
}

#[tokio::test]
async fn test_ollama_connection_returns_unreachable_on_connect_failure() {
    let _guard = crate::service::inference_test_guard();

    let result = super::super::test_ollama_connection("http://127.0.0.1:1")
        .await
        .unwrap();
    assert_eq!(result["reachable"], false);
    assert!(!result["error"].as_str().unwrap_or("").is_empty());
}

#[tokio::test]
async fn test_ollama_connection_rejects_invalid_url() {
    let _guard = crate::service::inference_test_guard();

    let err = super::super::test_ollama_connection("not-a-url")
        .await
        .unwrap_err();
    assert!(
        !err.is_empty(),
        "expected validation error, got empty string"
    );
}

#[tokio::test]
async fn diagnostics_reports_server_unreachable_when_url_unbound() {
    let _guard = crate::service::inference_test_guard();

    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", "http://127.0.0.1:1");
    }
    let config = Config::default();
    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");
    assert_eq!(diag["ollama_running"], false);
    assert!(
        diag["ollama_base_url"].as_str().is_some(),
        "diagnostics must include ollama_base_url"
    );
    let issues = diag["issues"].as_array().cloned().unwrap_or_default();
    assert!(
        !issues.is_empty(),
        "unreachable server must surface an issue"
    );
    assert!(
        issues
            .iter()
            .any(|v| v.as_str().unwrap_or("").contains("not running"))
    );
    let repair_actions = diag["repair_actions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        repair_actions.is_empty(),
        "OpenHuman should not suggest app-managed repair actions anymore"
    );
    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn diagnostics_with_running_server_but_missing_models_flags_issues() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route("/api/tags", get(|| async { Json(json!({ "models": [] })) }));
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = Config::default();
    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");
    assert_eq!(diag["ollama_running"], true);
    assert_eq!(
        diag["ollama_base_url"].as_str(),
        Some(base.as_str()),
        "diagnostics must echo back the base url being checked"
    );
    // No models are installed → expected chat model issue surfaces, telling
    // the user to pull it themselves (the runtime never pulls).
    let issues = diag["issues"].as_array().cloned().unwrap_or_default();
    assert!(!issues.is_empty());
    assert!(
        issues
            .iter()
            .filter_map(|issue| issue.as_str())
            .any(|issue| issue.contains("is not installed; run `ollama pull ")),
        "missing-model issue should carry an `ollama pull` hint, got: {issues:?}"
    );
    let repair_actions = diag["repair_actions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        repair_actions.is_empty(),
        "missing models should no longer surface app-managed pull actions"
    );
    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn diagnostics_ok_when_expected_models_are_present() {
    let _guard = crate::service::inference_test_guard();

    let config = Config::default();
    let chat = crate::models::effective_chat_model_id(&config);
    let embedding = crate::models::effective_embedding_model_id(&config);
    let chat_tag = format!("{}:latest", chat);
    let embed_tag = format!("{}:latest", embedding);
    let app = Router::new().route(
        "/api/tags",
        get(move || {
            let chat_tag = chat_tag.clone();
            let embed_tag = embed_tag.clone();
            async move {
                Json(json!({
                    "models": [
                        { "name": chat_tag, "modified_at": "", "size": 1u64, "digest": "d" },
                        { "name": embed_tag, "modified_at": "", "size": 2u64, "digest": "e" },
                    ]
                }))
            }
        }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");
    assert_eq!(diag["ollama_running"], true);
    assert_eq!(diag["expected"]["chat_found"], true);
    assert_eq!(diag["expected"]["embedding_found"], true);
    assert!(diag["ollama_base_url"].as_str().is_some());
    // All required models present → no issues and no repair actions.
    let issues = diag["issues"].as_array().cloned().unwrap_or_default();
    assert!(
        issues.is_empty(),
        "all models present should produce no issues, got: {:?}",
        issues
    );
    let repair_actions = diag["repair_actions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        repair_actions.is_empty(),
        "no issues should produce no repair actions"
    );
    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn diagnostics_reports_broken_runner_even_when_models_are_present() {
    let _guard = crate::service::inference_test_guard();

    let config = Config::default();
    let chat = crate::models::effective_chat_model_id(&config);
    let embedding = crate::models::effective_embedding_model_id(&config);
    let chat_tag = format!("{}:latest", chat);
    let embed_tag = format!("{}:latest", embedding);
    let app = Router::new()
        .route(
            "/api/tags",
            get(move || {
                let chat_tag = chat_tag.clone();
                let embed_tag = embed_tag.clone();
                async move {
                    Json(json!({
                        "models": [
                            { "name": chat_tag, "modified_at": "", "size": 1u64, "digest": "d" },
                            { "name": embed_tag, "modified_at": "", "size": 2u64, "digest": "e" },
                        ]
                    }))
                }
            }),
        )
        .route(
            "/api/show",
            axum::routing::post(|Json(body): Json<serde_json::Value>| async move {
                let model = body["name"]
                    .as_str()
                    .or_else(|| body["model"].as_str())
                    .unwrap_or_default();
                if model == "___nonexistent_probe___" {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "fork/exec /broken/ollama: no such file or directory".to_string(),
                    );
                }
                (
                    axum::http::StatusCode::OK,
                    json!({
                        "model_info": {
                            "general.architecture": "bert",
                            "bert.context_length": 8192,
                        },
                        "capabilities": ["embedding"],
                    })
                    .to_string(),
                )
            }),
        );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }

    assert_eq!(diag["ollama_running"], true);
    assert_eq!(diag["ok"], false);
    let issues = diag["issues"].as_array().cloned().unwrap_or_default();
    assert!(
        issues.iter().any(|issue| issue
            .as_str()
            .unwrap_or_default()
            .contains("cannot execute models")),
        "diagnostics should report the broken Ollama runner, got: {:?}",
        issues
    );
}

#[tokio::test]
async fn diagnostics_repair_actions_field_always_present() {
    // Verifies that the "repair_actions" key is always present in the diagnostics
    // JSON, regardless of the server state, so the UI can always iterate over it.
    let _guard = crate::service::inference_test_guard();

    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", "http://127.0.0.1:1");
    }
    let config = Config::default();
    let service = LocalAiService::new(&config);
    let diag = service.diagnostics(&config).await.expect("diagnostics");

    assert!(
        diag["repair_actions"].is_array(),
        "repair_actions must always be a JSON array"
    );

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn list_models_returns_parsed_payload() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/api/tags",
        get(|| async {
            Json(json!({
                "models": [
                    { "name": "a:latest", "modified_at": "t", "size": 1u64, "digest": "d1" },
                    { "name": "b:v2", "modified_at": "t", "size": 2u64, "digest": "d2" }
                ]
            }))
        }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = Config::default();
    let service = LocalAiService::new(&config);
    let models = service.list_models_at(&base).await.expect("list_models");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name, "a:latest");
    assert_eq!(models[1].name, "b:v2");
    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn list_models_errors_on_non_success() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/api/tags",
        get(|| async { (axum::http::StatusCode::SERVICE_UNAVAILABLE, "down") }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = Config::default();
    let service = LocalAiService::new(&config);
    let err = service.list_models_at(&base).await.unwrap_err();
    assert!(err.contains("503") || err.contains("tags failed"));
    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}
