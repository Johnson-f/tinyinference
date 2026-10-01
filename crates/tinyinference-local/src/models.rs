//! Local model and voice identifier resolution.
//!
//! Models are pulled and served by the user's own runtime, so a configured
//! model ID is passed through unchanged. Defaults apply only when no ID is
//! configured (and, for Ollama, to a few legacy pre-Ollama chat values that
//! are not Ollama tags at all). User-managed runtimes (LM Studio, MLX, OMLX,
//! OpenAI-compatible) get no Ollama-branded default.

use super::profile::{LocalProviderKind, kind_from_provider_string};

fn is_user_managed_runtime(provider: &str) -> bool {
    kind_from_provider_string(provider).is_some_and(|kind| kind != LocalProviderKind::Ollama)
}

/// Host configuration fields needed to resolve local inference model IDs.
///
/// Applications implement this trait on their configuration root so model
/// selection remains in TinyInference without coupling the crate to a host's
/// configuration schema.
pub trait LocalModelConfig {
    /// Configured local runtime provider name.
    fn local_provider_name(&self) -> &str;
    /// Preferred chat model ID.
    fn local_chat_model_id(&self) -> &str;
    /// Legacy fallback chat model ID.
    fn local_legacy_model_id(&self) -> &str;
    /// Configured vision model ID.
    fn local_vision_model_id(&self) -> &str;
    /// Configured embedding model ID.
    fn local_embedding_model_id(&self) -> &str;
    /// Configured speech-to-text model ID.
    fn local_stt_model_id(&self) -> &str;
    /// Configured text-to-speech voice ID.
    fn local_tts_voice_id(&self) -> &str;
}

const VISION_MODEL_SUGGESTIONS: &[&str] =
    &["moondream:1.8b-v2-q4_K_S", "llava:7b", "gemma3:4b-it-qat"];

/// Default Ollama chat model for managed local inference.
pub const DEFAULT_OLLAMA_MODEL: &str = "gemma3:1b-it-qat";

/// The pinned Moondream build that the `moondream` / `moondream:1.8b`
/// shorthands resolve to, and the low-RAM tier's bundled vision model.
///
/// Must name a genuinely vision-capable model: it is what an alias rewrite
/// lands on, and what the "for example …" suggestions point users at.
///
/// Moondream is the smallest vision model pullable with no extra setup
/// (~1.7 GB across model + projector layers), which keeps it affordable on the
/// low-RAM tiers where vision is most likely to be enabled on demand.
///
/// There is deliberately no `DEFAULT_OLLAMA_VISION_MODEL` any more (#5146 P1).
/// It existed only as the substitute for a chat-only `vision_model_id`, and
/// that substitution is exactly the bug: the user's explicit choice was
/// overridden, this model was auto-pulled behind their back, and the request
/// then failed with `ollama vision returned empty content`. A misconfigured
/// vision model is now an actionable error, so there is nothing left to
/// default *to*.
pub const DEFAULT_LOW_VISION_MODEL: &str = "moondream:1.8b-v2-q4_K_S";
/// Default Ollama embedding model for managed local inference.
pub const DEFAULT_OLLAMA_EMBED_MODEL: &str = "bge-m3";

/// Guarantee a vision request never reaches a chat-only model: `Ok(id)` when
/// `resolved` accepts image input, `Err(actionable message)` when it does not.
///
/// Whether a vision model is configured at all is decided upstream (vision is
/// disabled when `vision_model_id` is empty). What is left for this function
/// is the capability question alone.
///
/// # Why this errors instead of substituting (#5146 P1)
///
/// It used to swap in the default vision model and return that, which produced
/// the worst failure in the whole vision path: a user who set a chat-only
/// `vision_model_id` got a *different* model silently selected, that model
/// auto-pulled (~1.7 GB with no visible progress), and then — since the
/// substitute answers many prompt phrasings with an empty string — the cryptic
/// `ollama vision returned empty content`. Three surprises deep, none of them
/// naming the actual mistake.
///
/// Substituting is the wrong shape regardless of which default is chosen: the
/// user made an explicit choice and it was silently overridden, the same class
/// of bug as a silent provider switch (#5146 §2.1). Say what is wrong and let
/// them fix it. The message deliberately mirrors the tinyagents Ollama
/// embeddings adapter, which names the offending model and a concrete next step
/// in one line.
///
/// An earlier incarnation of this guard was an allowlist
/// (`MVP_ALLOWED_VISION_MODELS = &[""]`) that matched only the empty string and
/// so rewrote *every* configured vision model to `""`, including capable ones —
/// which is how a nameless model pull once came about. Both that bug and its
/// replacement failed the same way: they answered "which model?" with
/// something the user never asked for.
fn enforce_vision_capability(resolved: &str) -> crate::Result<String> {
    if tinyinference_llm::model::model_id_supports_vision(resolved) {
        return Ok(resolved.to_string());
    }
    tracing::warn!(
        resolved,
        "[local_ai] configured vision model is chat-only; refusing to substitute"
    );
    let suggestions = VISION_MODEL_SUGGESTIONS.join("`, `");
    Err(crate::Error::VisionModelUnsupported(format!(
        "the selected vision model `{resolved}` is not vision-capable — it cannot accept image \
         input. Set `local_ai.vision_model_id` to a vision-capable model (for example \
         `{suggestions}`) and pull it with `ollama pull <model>`, or route the vision workload \
         to a cloud provider with `vision_provider`."
    )))
}

/// Resolve the effective local chat model: the configured ID unchanged, or the
/// Ollama default when none is configured.
pub fn effective_chat_model_id(config: &impl LocalModelConfig) -> String {
    if is_user_managed_runtime(config.local_provider_name()) {
        let model_id = raw_chat_model_id(config);
        tracing::debug!(
            provider = config.local_provider_name(),
            has_model = !model_id.is_empty(),
            "[local_ai] effective_chat_model_id: using provider-managed model id"
        );
        return model_id;
    }

    let raw = if !config.local_chat_model_id().trim().is_empty() {
        config.local_chat_model_id().trim()
    } else {
        config.local_legacy_model_id().trim()
    };
    if raw.is_empty() {
        return DEFAULT_OLLAMA_MODEL.to_string();
    }
    let lower = raw.to_ascii_lowercase();
    if lower.ends_with(".gguf")
        || lower.contains("huggingface.co/")
        || lower == "qwen3-1.7b"
        || lower == "qwen2.5-1.5b-instruct"
    {
        return DEFAULT_OLLAMA_MODEL.to_string();
    }
    raw.to_string()
}

fn raw_chat_model_id(config: &impl LocalModelConfig) -> String {
    // For LM Studio the user must set `local_ai.chat_model_id` explicitly —
    // there is no sensible Ollama-branded default to fall back to. Return an
    // empty string so callers (diagnostics, status) surface the missing-model
    // warning rather than silently requesting "gemma3:1b-it-qat" from LM Studio.
    let raw = if !config.local_chat_model_id().trim().is_empty() {
        config.local_chat_model_id().trim()
    } else {
        config.local_legacy_model_id().trim()
    };
    if raw.is_empty() {
        tracing::debug!(
            provider = "lm_studio",
            "[local_ai] raw_chat_model_id: no LM Studio chat model configured"
        );
    }
    raw.to_string()
}

/// Apply the alias rewrite that maps a family name onto the pinned tag we
/// actually ship (`moondream` -> `moondream:1.8b-v2-q4_K_S`).
///
/// This is *not* a substitution: it resolves to the same model the user asked
/// for, so it never needs to be reported to them.
fn apply_vision_alias(raw: &str) -> &str {
    let lower = raw.to_ascii_lowercase();
    if lower == "moondream:1.8b" || lower == "moondream" {
        DEFAULT_LOW_VISION_MODEL
    } else {
        raw
    }
}

/// Resolve the vision model for status / reporting surfaces.
///
/// An empty return means "there is no **usable** vision model" — either none is
/// configured (a legitimate state; the low tiers ship no vision model) or the
/// configured one cannot accept images. A non-empty return is always a
/// vision-capable id.
///
/// Since #5146 P1 this no longer substitutes a default for a chat-only
/// configured model: a substituted id is a model the user never chose.
///
/// Call [`resolve_vision_model_id`] instead when about to issue an actual
/// vision request — it distinguishes "not configured" from "not vision-capable"
/// and returns an actionable message for each.
///
/// The capability predicate is consulted directly here rather than by calling
/// `enforce_vision_capability` and discarding its `Err`: that helper emits a
/// `tracing::warn!` and formats the full suggestion message, and this resolver
/// feeds polled status/diagnostics surfaces. Routing through it would log a
/// warning and burn a `format!` on *every poll* for anyone with a misconfigured
/// `vision_model_id`. The warning belongs at request time, where it is
/// actionable; `effective_and_resolved_vision_ids_agree_on_usability` keeps the
/// two paths pinned to the same verdict.
pub fn effective_vision_model_id(config: &impl LocalModelConfig) -> String {
    let raw = config.local_vision_model_id().trim();
    if raw.is_empty() {
        return String::new();
    }
    let resolved = apply_vision_alias(raw);
    if tinyinference_llm::model::model_id_supports_vision(resolved) {
        resolved.to_string()
    } else {
        String::new()
    }
}

/// Resolve the vision model for a real vision request.
///
/// Never returns an empty id, and never silently swaps the user's choice. The
/// two failure modes get distinct, actionable messages (#5146 §Part 1, P1):
///
/// - **nothing configured** — say what to set and which models to pull;
/// - **configured but chat-only** — name the offending model, because "pull
///   `moondream:…`" is a non-sequitur to someone who configured `gemma3:1b`.
pub fn resolve_vision_model_id(config: &impl LocalModelConfig) -> crate::Result<String> {
    let raw = config.local_vision_model_id().trim();
    if raw.is_empty() {
        let suggestions = VISION_MODEL_SUGGESTIONS.join("`, `");
        tracing::warn!("[local_ai] vision request with no vision model configured");
        return Err(crate::Error::VisionModelNotConfigured(format!(
            "no local vision model is configured. Set `local_ai.vision_model_id` to a \
             vision-capable model (for example `{suggestions}`) and pull it with \
             `ollama pull <model>`, or route the vision workload to a cloud provider \
             with `vision_provider`."
        )));
    }
    enforce_vision_capability(apply_vision_alias(raw))
}

/// Resolve the effective local embedding model for the selected runtime.
pub fn effective_embedding_model_id(config: &impl LocalModelConfig) -> String {
    let raw = config.local_embedding_model_id().trim();

    // User-managed runtimes get no Ollama-branded default (#3920): an empty
    // id stays empty so diagnostics surface the missing model.
    if is_user_managed_runtime(config.local_provider_name()) {
        if raw.is_empty() {
            tracing::debug!(
                provider = config.local_provider_name(),
                "[local_ai] effective_embedding_model_id: no LM Studio embedding model configured"
            );
            return String::new();
        }
        tracing::debug!(
            provider = config.local_provider_name(),
            "[local_ai] effective_embedding_model_id: using provider-managed embedding id"
        );
        return raw.to_string();
    }

    if raw.is_empty() {
        return DEFAULT_OLLAMA_EMBED_MODEL.to_string();
    }
    raw.to_string()
}

/// Resolve the configured speech-to-text model or its default.
pub fn effective_stt_model_id(config: &impl LocalModelConfig) -> String {
    let raw = config.local_stt_model_id().trim();
    if raw.is_empty() {
        "ggml-base-q5_1.bin".to_string()
    } else {
        raw.to_string()
    }
}

/// Resolve the configured text-to-speech voice or its default.
pub fn effective_tts_voice_id(config: &impl LocalModelConfig) -> String {
    let raw = config.local_tts_voice_id().trim();
    if raw.is_empty() {
        "en_US-lessac-medium".to_string()
    } else {
        raw.to_string()
    }
}

#[cfg(test)]
#[path = "models_test.rs"]
mod tests;
