//! TinyHumans-compatible reasoning and thread request fields.

use serde_json::{Value, json};

use crate::model::{ModelRequest, ReasoningConfig, ReasoningEffort};

/// Request metadata key used by callers to disable reasoning for one request.
pub const REASONING_OFF_METADATA_KEY: &str = "openhuman_reasoning_off";

/// Applies the managed gateway's `reasoning` object to a request.
///
/// An explicit provider option wins over metadata; the metadata disable hint
/// wins over provider-neutral reasoning. The neutral field is consumed after
/// conversion so the OpenAI adapter does not emit a second reasoning field.
#[must_use]
pub fn apply_reasoning(mut request: ModelRequest) -> ModelRequest {
    if request.provider_options.get("reasoning").is_some() {
        request.reasoning = None;
        return request;
    }
    let off = request
        .metadata
        .get(REASONING_OFF_METADATA_KEY)
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let neutral = request.reasoning.take();
    let reasoning = if off {
        Some(json!({ "enabled": false }))
    } else {
        neutral.as_ref().and_then(reasoning_object)
    };
    let Some(reasoning) = reasoning else {
        return request;
    };
    let mut options = request.provider_options.clone();
    if !options.is_object() {
        options = Value::Object(serde_json::Map::new());
    }
    if let Some(map) = options.as_object_mut() {
        map.insert("reasoning".into(), reasoning);
    }
    request.with_provider_options(options)
}

fn reasoning_object(config: &ReasoningConfig) -> Option<Value> {
    if config.effort == Some(ReasoningEffort::None) {
        return Some(json!({ "enabled": false }));
    }
    if let Some(budget) = config.budget_tokens {
        return Some(json!({ "max_tokens": budget }));
    }
    config
        .effort
        .map(|effort| json!({ "effort": effort.as_str() }))
}

/// Adds the explicitly configured thread id to provider options.
#[must_use]
pub fn with_thread_id(mut request: ModelRequest, thread_id: Option<&str>) -> ModelRequest {
    let Some(thread_id) = thread_id.map(str::trim).filter(|value| !value.is_empty()) else {
        return request;
    };
    let mut options = request.provider_options.clone();
    if !options.is_object() {
        options = Value::Object(serde_json::Map::new());
    }
    if let Some(map) = options.as_object_mut() {
        map.insert("thread_id".into(), Value::String(thread_id.to_owned()));
    }
    request = request.with_provider_options(options);
    request
}
