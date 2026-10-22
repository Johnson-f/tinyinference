//! TinyHumans managed model listing and context-limit discovery.

use crate::catalog::{ListModelsError, ModelInfo, append_query_param, list_models as fetch_models};
use crate::model::discover::DiscoveryRequest;

/// Builds the managed gateway's authenticated context-discovery request.
#[must_use]
pub fn discovery_request(
    base: &str,
    model: &str,
    auth: Option<&str>,
    extra_headers: &[(String, String)],
) -> DiscoveryRequest {
    let base = endpoint(base);
    let listing = append_query_param(&format!("{base}/models"), "catalog", "openrouter");
    let mut request = DiscoveryRequest::new(base, model)
        .with_listing_url(listing)
        .with_single_model_probe(false);
    if let Some(token) = auth.map(str::trim).filter(|token| !token.is_empty()) {
        request = request.with_header("Authorization", format!("Bearer {token}"));
    }
    for (name, value) in extra_headers {
        request
            .headers
            .retain(|(old_name, _)| !old_name.eq_ignore_ascii_case(name));
        request = request.with_header(name.clone(), value.clone());
    }
    request
}

/// Fetches the model listing through TinyInference's host-client-aware API.
///
/// # Errors
/// Returns listing transport, HTTP, provider-envelope, or decoding failures.
pub async fn list_models(
    client: &reqwest::Client,
    request: DiscoveryRequest,
) -> std::result::Result<Vec<ModelInfo>, ListModelsError> {
    fetch_models(client, request).await
}

pub(super) fn endpoint(base: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    if base.ends_with("/openai/v1") {
        base.to_owned()
    } else {
        format!("{base}/openai/v1")
    }
}
