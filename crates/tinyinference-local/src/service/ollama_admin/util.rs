use crate::ollama::{OllamaTagsResponse, validate_ollama_url};

pub(in crate::service) fn models_error_means_unreachable(error: &str) -> bool {
    error.starts_with("lm studio models request failed:")
}

/// Test connectivity to a user-supplied Ollama URL.
///
/// Validates the URL via [`validate_ollama_url`], then issues a GET to
/// `{normalized_url}/api/tags` with a 3-second timeout.
/// Returns a JSON object with `reachable`, optional `error`, and
/// `models_count` when reachable.
pub async fn test_ollama_connection(url: &str) -> Result<serde_json::Value, String> {
    let normalized = validate_ollama_url(url)?;
    log::debug!("[local_ai] test_ollama_connection: testing url={normalized}");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    match client.get(format!("{normalized}/api/tags")).send().await {
        Ok(resp) if resp.status().is_success() => {
            let models_count = resp
                .json::<OllamaTagsResponse>()
                .await
                .map(|t| t.models.len())
                .unwrap_or(0);
            log::debug!(
                "[local_ai] test_ollama_connection: reachable url={normalized} models={models_count}"
            );
            Ok(serde_json::json!({
                "reachable": true,
                "error": null,
                "models_count": models_count,
            }))
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("server responded with status {status}: {}", body.trim());
            log::debug!(
                "[local_ai] test_ollama_connection: unreachable url={normalized} err={err}"
            );
            Ok(serde_json::json!({
                "reachable": false,
                "error": err,
                "models_count": null,
            }))
        }
        Err(e) => {
            let err = e.to_string();
            log::debug!(
                "[local_ai] test_ollama_connection: connection failed url={normalized} err={err}"
            );
            Ok(serde_json::json!({
                "reachable": false,
                "error": err,
                "models_count": null,
            }))
        }
    }
}
