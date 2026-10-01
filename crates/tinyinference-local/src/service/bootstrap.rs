//! Service construction, status snapshots, and the endpoint probe.
//!
//! `bootstrap` only *probes* the user's configured endpoint. It never spawns a
//! runtime process and never asks the runtime to pull a model.

use crate::lm_studio::lm_studio_base_url;
use crate::models as model_ids;
use crate::ollama::ollama_base_url_from_override;
use crate::provider::{
    LocalAiProvider, ModelDiscoveryApi, model_discovery_api, provider_from_name,
};
use crate::service::RuntimeConfig as Config;
use crate::status::LocalAiStatus;

use super::LocalAiService;
use super::ollama_admin::OllamaHealthStatus;

/// Result of probing the configured endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EndpointProbe {
    /// The endpoint answered its model listing promptly.
    Ready,
    /// The endpoint is alive but slow, or answered with an error.
    Degraded(String),
    /// Nothing answered at the endpoint.
    Unreachable(String),
}

impl LocalAiService {
    pub fn new(config: &Config) -> Self {
        let model_id = model_ids::effective_chat_model_id(config);
        let provider = provider_from_name(&config.local_ai.provider);
        Self {
            status: parking_lot::Mutex::new(LocalAiStatus {
                state: "idle".to_string(),
                model_id: model_id.clone(),
                chat_model_id: model_id,
                vision_model_id: model_ids::effective_vision_model_id(config),
                embedding_model_id: model_ids::effective_embedding_model_id(config),
                stt_model_id: model_ids::effective_stt_model_id(config),
                tts_voice_id: model_ids::effective_tts_voice_id(config),
                vision_state: initial_vision_state(config),
                vision_mode: vision_mode_str(config),
                embedding_state: "idle".to_string(),
                stt_state: "idle".to_string(),
                tts_state: "idle".to_string(),
                provider: provider.as_str().to_string(),
                warning: None,
                error_detail: None,
                error_category: None,
                model_path: Some(model_path_for_config(config)),
                active_backend: provider.as_str().to_string(),
                backend_reason: None,
                last_latency_ms: None,
                prompt_toks_per_sec: None,
                gen_toks_per_sec: None,
            }),
            bootstrap_lock: tokio::sync::Mutex::new(()),
            last_memory_summary_at: parking_lot::Mutex::new(None),
            http: reqwest::Client::builder()
                // Local models can take >30s on cold start and first-token generation.
                // Keep the total timeout generous so inline autocomplete and local
                // chat stay reliable.
                .timeout(std::time::Duration::from_secs(120))
                // ...but bound the *connect* phase tightly. When the user's
                // runtime isn't running, the default connect timeout (long on
                // Windows loopback) would stall every probe. 500ms is well under
                // any realistic loopback connect latency; if the server is up,
                // reqwest's per-request `.timeout()` still bounds the rest of
                // the exchange.
                .connect_timeout(std::time::Duration::from_millis(500))
                .build()
                .unwrap_or_else(|e| {
                    log::warn!("[local_ai] reqwest client build failed, falling back to default client: {e}");
                    reqwest::Client::new()
                }),
        }
    }

    pub fn status(&self) -> LocalAiStatus {
        self.status.lock().clone()
    }

    pub fn reset_to_idle(&self, config: &Config) {
        let model_id = model_ids::effective_chat_model_id(config);
        let provider = provider_from_name(&config.local_ai.provider);
        let mut status = self.status.lock();
        status.state = "idle".to_string();
        status.model_id = model_id.clone();
        status.chat_model_id = model_id;
        status.vision_model_id = model_ids::effective_vision_model_id(config);
        status.embedding_model_id = model_ids::effective_embedding_model_id(config);
        status.stt_model_id = model_ids::effective_stt_model_id(config);
        status.tts_voice_id = model_ids::effective_tts_voice_id(config);
        status.vision_state = initial_vision_state(config);
        status.vision_mode = vision_mode_str(config);
        status.embedding_state = "idle".to_string();
        status.stt_state = "idle".to_string();
        status.tts_state = "idle".to_string();
        status.provider = provider.as_str().to_string();
        status.warning = None;
        status.error_detail = None;
        status.error_category = None;
        status.model_path = Some(model_path_for_config(config));
        status.active_backend = provider.as_str().to_string();
        status.backend_reason = None;
        status.last_latency_ms = None;
        status.prompt_toks_per_sec = None;
        status.gen_toks_per_sec = None;
    }

    pub fn mark_degraded(&self, warning: String) {
        log::warn!("[local_ai] mark_degraded: {warning}");
        let mut status = self.status.lock();
        status.state = "degraded".to_string();
        status.warning = Some(warning);
    }

    /// Force the status field to `"disabled"` so the UI flips immediately
    /// after the user turns local inference off, without waiting for the next
    /// status poll.
    pub fn mark_disabled(&self, config: &Config) {
        log::info!("[local_ai] mark_disabled: status forced to disabled by gate toggle");
        *self.status.lock() = LocalAiStatus::disabled(config, &vision_mode_str(config));
    }

    /// Probe the configured endpoint and record whether it is usable.
    ///
    /// Sets `state` to `"disabled"`, `"ready"`, `"degraded"`, or
    /// `"unreachable"`. This is a cheap read-only probe (Ollama `GET
    /// /api/tags`, or `GET /v1/models` for OpenAI-compatible runtimes): it
    /// never spawns a runtime process and never pulls a model. A `ready` or
    /// `degraded` state is kept until [`Self::reset_to_idle`]; an
    /// `unreachable` endpoint is probed again on the next call, so a runtime
    /// the user starts later is picked up.
    pub async fn bootstrap(&self, config: &Config) {
        let _guard = self.bootstrap_lock.lock().await;

        if !local_inference_enabled(config) {
            tracing::debug!(
                runtime_enabled = config.local_ai.runtime_enabled,
                opt_in_confirmed = config.local_ai.opt_in_confirmed,
                "[local_ai] bootstrap: local inference not opted in; status disabled"
            );
            *self.status.lock() = LocalAiStatus::disabled(config, &vision_mode_str(config));
            return;
        }

        if matches!(self.status.lock().state.as_str(), "ready" | "degraded") {
            return;
        }

        let provider = provider_from_name(&config.local_ai.provider);
        {
            let model_id = model_ids::effective_chat_model_id(config);
            let mut status = self.status.lock();
            status.model_id = model_id.clone();
            status.chat_model_id = model_id;
            status.vision_model_id = model_ids::effective_vision_model_id(config);
            status.embedding_model_id = model_ids::effective_embedding_model_id(config);
            status.stt_model_id = model_ids::effective_stt_model_id(config);
            status.tts_voice_id = model_ids::effective_tts_voice_id(config);
            status.provider = provider.as_str().to_string();
            status.vision_mode = vision_mode_str(config);
            status.active_backend = provider.as_str().to_string();
            status.backend_reason = Some(format!(
                "Inference delegated to {} runtime",
                provider.display_name()
            ));
            status.model_path = Some(model_path_for_config(config));
        }

        let (endpoint, probe) = self.probe_endpoint(config).await;
        let safe_endpoint = tinyinference_core::sanitize::redact_url(&endpoint);
        tracing::debug!(
            provider = provider.as_str(),
            endpoint = %safe_endpoint,
            ?probe,
            "[local_ai] bootstrap: endpoint probe finished"
        );

        let mut status = self.status.lock();
        match probe {
            EndpointProbe::Ready => {
                status.state = "ready".to_string();
                status.vision_state = initial_vision_state(config);
                status.embedding_state = "idle".to_string();
                status.warning = None;
                status.error_detail = None;
                status.error_category = None;
            }
            EndpointProbe::Degraded(detail) => {
                status.state = "degraded".to_string();
                status.warning = Some(format!(
                    "Local {} runtime at {safe_endpoint} is responding slowly or with errors",
                    provider.display_name()
                ));
                status.error_detail = Some(detail);
                status.error_category = Some("server".to_string());
            }
            EndpointProbe::Unreachable(detail) => {
                status.state = "unreachable".to_string();
                status.warning = Some(format!(
                    "Local {} runtime is not reachable at {safe_endpoint}. Start it and load \
                     your models yourself; OpenHuman does not install or launch it.",
                    provider.display_name()
                ));
                status.error_detail = Some(detail);
                status.error_category = Some("server".to_string());
            }
        }
    }

    /// Probe the configured endpoint without side effects. Returns the probed
    /// base URL and the verdict.
    async fn probe_endpoint(&self, config: &Config) -> (String, EndpointProbe) {
        let provider = provider_from_name(&config.local_ai.provider);
        let ollama_base = ollama_base_url_from_override(config.local_ai.base_url.as_deref());
        let openai_shaped = provider == LocalAiProvider::LmStudio
            || model_discovery_api(&config.local_ai.provider, &ollama_base)
                == ModelDiscoveryApi::OpenAiModels;

        if openai_shaped {
            let base = lm_studio_base_url(config.local_ai.base_url.as_deref());
            let verdict = match self.list_lm_studio_models(config).await {
                Ok(_) => EndpointProbe::Ready,
                Err(err) if super::ollama_admin::models_error_means_unreachable(&err) => {
                    EndpointProbe::Unreachable(err)
                }
                Err(err) => EndpointProbe::Degraded(err),
            };
            return (base, verdict);
        }

        let verdict = match self.ollama_health_status_at(&ollama_base).await {
            OllamaHealthStatus::Running => EndpointProbe::Ready,
            OllamaHealthStatus::Degraded => {
                EndpointProbe::Degraded("health probe answered only on the slow retry".to_string())
            }
            OllamaHealthStatus::Stopped => {
                EndpointProbe::Unreachable(format!("no healthy answer from {ollama_base}/api/tags"))
            }
        };
        (ollama_base, verdict)
    }

    pub fn should_run_memory_autosummary(&self, config: &Config) -> bool {
        let mut guard = self.last_memory_summary_at.lock();
        let now = std::time::Instant::now();
        match *guard {
            Some(last)
                if now.duration_since(last).as_millis()
                    < u128::from(config.local_ai.autosummary_debounce_ms) =>
            {
                false
            }
            _ => {
                *guard = Some(now);
                true
            }
        }
    }
}

/// Local inference is opt-in. `opt_in_confirmed` is authoritative: an
/// explicit opt-in enables it even when a stale on-disk `runtime_enabled` is
/// false, and without it local inference stays disabled.
pub(crate) fn local_inference_enabled(config: &Config) -> bool {
    config.local_ai.opt_in_confirmed
}

fn initial_vision_state(config: &Config) -> String {
    if super::vision_configured(&config.local_ai) {
        "idle".to_string()
    } else {
        "disabled".to_string()
    }
}

fn vision_mode_str(config: &Config) -> String {
    super::vision_mode_label(&config.local_ai).to_string()
}

fn model_path_for_config(config: &Config) -> String {
    let model_id = model_ids::effective_chat_model_id(config);
    match provider_from_name(&config.local_ai.provider) {
        LocalAiProvider::Ollama => format!("ollama://{model_id}"),
        LocalAiProvider::LmStudio => format!("lmstudio://{model_id}"),
    }
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
