//! Serializable DTOs for local AI status and RPC responses.

#![allow(
    missing_docs,
    reason = "wire DTO field names are self-describing and stable"
)]

use serde::{Deserialize, Serialize};

use crate::models::{self as model_ids, LocalModelConfig};
use crate::provider::provider_from_name;

/// Observable state of the configured local inference endpoint.
///
/// `state` is one of:
/// - `"disabled"`: local inference is turned off in the host config.
/// - `"idle"`: enabled but the endpoint has not been probed yet.
/// - `"ready"`: the endpoint answered its model listing promptly.
/// - `"degraded"`: the endpoint is alive but slow, or answered the model
///   listing with an error; inference may still work.
/// - `"unreachable"`: nothing answered at the configured endpoint. The user
///   has to start their runtime; this crate never starts one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalAiStatus {
    pub state: String,
    pub model_id: String,
    pub chat_model_id: String,
    pub vision_model_id: String,
    pub embedding_model_id: String,
    pub stt_model_id: String,
    pub tts_voice_id: String,
    pub vision_state: String,
    pub vision_mode: String,
    pub embedding_state: String,
    pub stt_state: String,
    pub tts_state: String,
    pub provider: String,
    pub warning: Option<String>,
    /// Extended error text (e.g. the endpoint probe failure) for UI display.
    pub error_detail: Option<String>,
    /// Category of failure: `"server"` when the configured endpoint could not
    /// be reached or answered unexpectedly, or `None`.
    pub error_category: Option<String>,
    pub model_path: Option<String>,
    pub active_backend: String,
    pub backend_reason: Option<String>,
    pub last_latency_ms: Option<u64>,
    pub prompt_toks_per_sec: Option<f32>,
    pub gen_toks_per_sec: Option<f32>,
}

impl LocalAiStatus {
    /// Creates a disabled status snapshot from host model settings.
    pub fn disabled(config: &impl LocalModelConfig, vision_mode: &str) -> Self {
        let provider = provider_from_name(config.local_provider_name());
        Self {
            state: "disabled".to_string(),
            model_id: model_ids::effective_chat_model_id(config),
            chat_model_id: model_ids::effective_chat_model_id(config),
            vision_model_id: model_ids::effective_vision_model_id(config),
            embedding_model_id: model_ids::effective_embedding_model_id(config),
            stt_model_id: model_ids::effective_stt_model_id(config),
            tts_voice_id: model_ids::effective_tts_voice_id(config),
            vision_state: "disabled".to_string(),
            vision_mode: vision_mode.to_ascii_lowercase(),
            embedding_state: "disabled".to_string(),
            stt_state: "disabled".to_string(),
            tts_state: "disabled".to_string(),
            provider: provider.as_str().to_string(),
            warning: None,
            error_detail: None,
            error_category: None,
            model_path: None,
            active_backend: provider.as_str().to_string(),
            backend_reason: None,
            last_latency_ms: None,
            prompt_toks_per_sec: None,
            gen_toks_per_sec: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalAiEmbeddingResult {
    pub model_id: String,
    pub dimensions: usize,
    pub vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalAiSpeechResult {
    pub text: String,
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalAiTtsResult {
    pub output_path: String,
    pub voice_id: String,
}

#[cfg(test)]
#[path = "status_test.rs"]
#[allow(clippy::field_reassign_with_default)]
mod tests;
