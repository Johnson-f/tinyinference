//! Ordered provider failure demotion decisions shared with host reporters.

use super::http::ProviderHttpFailureFacts;

/// The user or provider state represented by a demoted HTTP failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemotionKind {
    /// The request exceeded a local or provider budget.
    BudgetExhausted,
    /// A local inference server has no model loaded.
    NoModelLoaded,
    /// A custom OpenAI-compatible upstream rejected the request.
    CustomUpstreamBadRequest,
    /// A provider rejected access under its client policy.
    AccessPolicyDenied,
    /// The provider rejected model or request configuration.
    ConfigurationRejected,
    /// The request exceeded the model's context window.
    ContextWindowExceeded,
    /// The provider account exhausted its plan quota.
    QuotaExhausted,
    /// The provider rejected an API key.
    ApiKeyRejected,
    /// An OAuth session with the provider expired.
    OAuthSessionExpired,
    /// The provider account has insufficient credits.
    InsufficientCredits,
    /// Ollama Cloud returned its known internal-error envelope.
    OllamaCloudInternalError,
    /// The provider rejected content under a moderation policy.
    ModerationRejected,
}

/// Chooses the first applicable demotion according to the shared host ladder.
///
/// Facts intentionally overlap. The order here is observable to callers and
/// preserves the host reporter's precedence. A managed provider is excluded
/// from provider-owned configuration, API-key, and OAuth decisions, except
/// that an unknown-model rejection remains user configuration state.
pub fn demote_decision(
    facts: &ProviderHttpFailureFacts,
    provider_is_managed: bool,
) -> Option<DemotionKind> {
    if facts.budget_exhausted {
        Some(DemotionKind::BudgetExhausted)
    } else if facts.no_model_loaded {
        Some(DemotionKind::NoModelLoaded)
    } else if facts.custom_upstream_bad_request {
        Some(DemotionKind::CustomUpstreamBadRequest)
    } else if facts.access_policy_denied {
        Some(DemotionKind::AccessPolicyDenied)
    } else if facts.configuration_rejected && (facts.unknown_model || !provider_is_managed) {
        Some(DemotionKind::ConfigurationRejected)
    } else if facts.context_window_exceeded {
        Some(DemotionKind::ContextWindowExceeded)
    } else if facts.quota_exhausted {
        Some(DemotionKind::QuotaExhausted)
    } else if facts.api_key_rejected && !provider_is_managed {
        Some(DemotionKind::ApiKeyRejected)
    } else if facts.oauth_session_expired && !provider_is_managed {
        Some(DemotionKind::OAuthSessionExpired)
    } else if facts.insufficient_credits {
        Some(DemotionKind::InsufficientCredits)
    } else if facts.ollama_cloud_internal_error {
        Some(DemotionKind::OllamaCloudInternalError)
    } else if facts.moderation_rejected {
        Some(DemotionKind::ModerationRejected)
    } else {
        None
    }
}

#[cfg(test)]
#[path = "demotion_tests.rs"]
mod tests;
