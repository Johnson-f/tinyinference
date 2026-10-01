//! Local inference service over a user-run endpoint (Ollama, LM Studio, or
//! any OpenAI-compatible server).
//!
//! The service probes the configured endpoint and runs inference against it.
//! It never downloads models, installs runtimes, or spawns/stops a runtime
//! process: the user installs and runs their runtime and pulls their models.

#![allow(
    dead_code,
    missing_docs,
    clippy::await_holding_lock,
    clippy::field_reassign_with_default,
    reason = "runtime internals and wire-shaped settings are exercised across feature-specific hosts"
)]

mod bootstrap;
mod lm_studio;
mod model_rpc;
mod ollama_admin;
mod public_infer;
pub use ollama_admin::test_ollama_connection;
pub mod paths;
mod vision_embed;

use crate::status::LocalAiStatus;
use parking_lot::Mutex;
use std::path::PathBuf;

#[cfg(test)]
static INFERENCE_TEST_MUTEX: once_cell::sync::Lazy<std::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(()));

#[cfg(test)]
pub(crate) fn inference_test_guard() -> std::sync::MutexGuard<'static, ()> {
    INFERENCE_TEST_MUTEX
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

/// Whether a local vision model is configured.
///
/// Vision is on-demand against the user's endpoint: it is enabled exactly when
/// the host names a vision model, and is never preloaded or pulled.
pub(crate) fn vision_configured(settings: &LocalRuntimeSettings) -> bool {
    !settings.vision_model_id.trim().is_empty()
}

/// Wire label for the vision mode: `"ondemand"` when a vision model is
/// configured, `"disabled"` otherwise.
pub(crate) fn vision_mode_label(settings: &LocalRuntimeSettings) -> &'static str {
    if vision_configured(settings) {
        "ondemand"
    } else {
        "disabled"
    }
}

/// Host-selected local runtime settings consumed by the runtime service.
///
/// These describe an endpoint the user already runs. There are no download
/// URLs, preload flags, tiers, or binary paths: the user installs the runtime
/// and pulls the models.
#[derive(Clone, Default)]
pub struct LocalRuntimeSettings {
    /// Use local inference. Authoritative only together with
    /// `opt_in_confirmed` (see `LocalAiService::bootstrap`).
    pub runtime_enabled: bool,
    pub provider: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_id: String,
    pub chat_model_id: String,
    pub vision_model_id: String,
    pub embedding_model_id: String,
    pub stt_model_id: String,
    pub tts_voice_id: String,
    pub autosummary_debounce_ms: u64,
    pub opt_in_confirmed: bool,
    pub num_ctx: Option<u32>,
}

impl std::fmt::Debug for LocalRuntimeSettings {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalRuntimeSettings")
            .field("runtime_enabled", &self.runtime_enabled)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("model_id", &self.model_id)
            .field("chat_model_id", &self.chat_model_id)
            .field("vision_model_id", &self.vision_model_id)
            .field("embedding_model_id", &self.embedding_model_id)
            .field("stt_model_id", &self.stt_model_id)
            .field("tts_voice_id", &self.tts_voice_id)
            .field("num_ctx", &self.num_ctx)
            .finish_non_exhaustive()
    }
}

/// Complete host snapshot needed by local inference execution.
#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    pub local_ai: LocalRuntimeSettings,
    pub workspace_dir: PathBuf,
    pub config_path: PathBuf,
    pub shared_root_dir: PathBuf,
    pub default_temperature: f64,
}

impl crate::models::LocalModelConfig for RuntimeConfig {
    fn local_provider_name(&self) -> &str {
        &self.local_ai.provider
    }
    fn local_chat_model_id(&self) -> &str {
        &self.local_ai.chat_model_id
    }
    fn local_legacy_model_id(&self) -> &str {
        &self.local_ai.model_id
    }
    fn local_vision_model_id(&self) -> &str {
        &self.local_ai.vision_model_id
    }
    fn local_embedding_model_id(&self) -> &str {
        &self.local_ai.embedding_model_id
    }
    fn local_stt_model_id(&self) -> &str {
        &self.local_ai.stt_model_id
    }
    fn local_tts_voice_id(&self) -> &str {
        &self.local_ai.tts_voice_id
    }
}

pub struct LocalAiService {
    pub(crate) status: Mutex<LocalAiStatus>,
    pub(crate) bootstrap_lock: tokio::sync::Mutex<()>,
    pub(crate) last_memory_summary_at: Mutex<Option<std::time::Instant>>,
    pub(crate) http: reqwest::Client,
}

impl std::fmt::Debug for LocalAiService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalAiService")
            .field("status", &self.status.lock())
            .finish_non_exhaustive()
    }
}

impl LocalAiService {
    /// Replaces the observable runtime state and returns its previous value.
    ///
    /// Hosts may use this when an external capability changes readiness.
    pub fn replace_status_state(&self, state: impl Into<String>) -> String {
        std::mem::replace(&mut self.status.lock().state, state.into())
    }

    /// Marks hosted speech recognition ready after a host-owned STT call.
    pub fn mark_stt_ready(&self) {
        self.status.lock().stt_state = "ready".to_string();
    }

    /// Marks local speech synthesis ready after a host-owned TTS call.
    pub fn mark_tts_ready(&self) {
        self.status.lock().tts_state = "ready".to_string();
    }
}
