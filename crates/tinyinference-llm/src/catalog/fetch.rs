//! HTTP helpers for model catalogs and OpenRouter key validation.

use super::{ModelInfo, parse_models_response};
use crate::model::discover::DiscoveryRequest;
use std::fmt;

/// A catalog request or response failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListModelsError(String);

impl ListModelsError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ListModelsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ListModelsError {}

/// Fetches and parses the model catalog described by a discovery request.
///
/// Host callers provide the `reqwest::Client`, so TLS roots and proxy policy
/// remain host-owned. A custom `listing_url` and all discovery headers are
/// preserved, including Codex query parameters and OAuth headers.
///
/// # Errors
/// Returns an error for request failures, non-success HTTP statuses, malformed
/// JSON, provider error envelopes, or an invalid catalog response shape.
pub async fn list_models(
    client: &reqwest::Client,
    request: DiscoveryRequest,
) -> Result<Vec<ModelInfo>, ListModelsError> {
    let url = request
        .listing_url
        .clone()
        .unwrap_or_else(|| format!("{}/models", request.endpoint.trim_end_matches('/')));
    let mut builder = client.get(&url);
    for (name, value) in request.headers {
        builder = builder.header(&name, &value);
    }
    builder = builder.timeout(request.timeout);
    let response = builder.send().await.map_err(|error| {
        ListModelsError::new(format!(
            "[providers][list_models] HTTP request failed: {}",
            error.without_url()
        ))
    })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let snippet = snippet(&body);
        if status.as_u16() == 404 {
            return Err(ListModelsError::new(format!(
                "provider returned 404: {snippet} — the configured base URL does not expose a `/models` endpoint; check the provider's base URL (it usually ends in `/v1`)"
            )));
        }
        return Err(ListModelsError::new(format!(
            "provider returned {}: {snippet}",
            status.as_u16()
        )));
    }

    let raw_body = response.text().await.map_err(|error| {
        ListModelsError::new(format!(
            "[providers][list_models] failed to read response body: {}",
            error.without_url()
        ))
    })?;
    let body: serde_json::Value = serde_json::from_str(&raw_body).map_err(|error| {
        ListModelsError::new(format!(
            "[providers][list_models] failed to parse JSON: {error} (body: {})",
            snippet(&raw_body)
        ))
    })?;
    if let Some(error) = body.get("error") {
        let message = error
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                error
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| error.to_string());
        return Err(ListModelsError::new(format!(
            "provider returned error payload: {}",
            tinyinference_core::sanitize::sanitize_api_error(&message)
        )));
    }
    parse_models_response(&body).map_err(|error| ListModelsError::new(error.to_string()))
}

/// Validates an OpenRouter API key using its authenticated `/key` endpoint.
///
/// # Errors
/// Returns an error when the key is empty, the request fails, the endpoint
/// rejects the key, or a successful response contains an error payload.
pub async fn validate_openrouter_key(
    client: &reqwest::Client,
    base: &str,
    api_key: &str,
) -> Result<(), ListModelsError> {
    if api_key.trim().is_empty() {
        return Err(ListModelsError::new(
            "OpenRouter API key is required before enabling the provider",
        ));
    }
    let url = format!("{}/key", base.trim_end_matches('/'));
    let response = client
        .get(url)
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(|error| {
            ListModelsError::new(format!(
                "[providers][list_models] OpenRouter key validation failed: {}",
                error.without_url()
            ))
        })?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(ListModelsError::new(format!(
            "OpenRouter key validation returned {}: {}",
            status.as_u16(),
            snippet(&body)
        )));
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body)
        && let Some(error) = value.get("error")
    {
        let message = error
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                error
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| error.to_string());
        return Err(ListModelsError::new(format!(
            "OpenRouter key validation returned error payload: {}",
            tinyinference_core::sanitize::sanitize_api_error(&message)
        )));
    }
    Ok(())
}

pub(super) fn snippet(body: &str) -> String {
    let sanitized = tinyinference_core::sanitize::sanitize_api_error(body);
    let truncated = sanitized
        .chars()
        .take(300)
        .collect::<String>()
        .trim_end()
        .to_string();
    if sanitized.chars().count() > 300 {
        format!("{truncated}...")
    } else {
        truncated
    }
}
