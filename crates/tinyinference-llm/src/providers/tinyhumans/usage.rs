//! Projection of managed usage fields into normalized usage metadata.

use crate::model::ModelResponse;

#[derive(Debug, Default, serde::Deserialize)]
struct Envelope {
    #[serde(default)]
    usage: Option<EnvelopeUsage>,
    #[serde(default)]
    billing: Option<EnvelopeBilling>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct EnvelopeUsage {
    #[serde(default)]
    cached_input_tokens: Option<u64>,
    #[serde(default)]
    context_window: Option<u64>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct EnvelopeBilling {
    #[serde(default)]
    charged_amount_usd: Option<f64>,
}

/// Projects `openhuman.{billing,usage}` into standard usage metadata.
///
/// The original envelope remains in `raw`; `openhuman_usage_meta` is merged
/// into it for cost consumers, and cached input usage fills an empty normalized
/// `cache_read_tokens` value.
#[must_use]
pub fn project_tinyhumans_envelope(mut response: ModelResponse) -> ModelResponse {
    let envelope: Envelope = response
        .raw
        .as_ref()
        .and_then(|raw| raw.get("openhuman"))
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default();
    let charged = envelope
        .billing
        .and_then(|billing| billing.charged_amount_usd);
    let context_window = envelope
        .usage
        .as_ref()
        .and_then(|usage| usage.context_window)
        .unwrap_or_default();
    if let (Some(usage), Some(cached)) = (
        response.usage.as_mut(),
        envelope.usage.and_then(|usage| usage.cached_input_tokens),
    ) && usage.cache_read_tokens == 0
    {
        usage.cache_read_tokens = cached;
    }
    response.raw = crate::usage::merge_usage_meta(response.raw, charged, context_window);
    response
}

#[cfg(test)]
#[path = "usage_tests.rs"]
mod tests;
