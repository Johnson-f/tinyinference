//! Provider-neutral user-facing copy for common inference failures.

/// A provider failure class with default user-facing copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailureCopyClass {
    /// A provider's chat template cannot accept the request shape.
    ChatTemplateRejected,
    /// The provider rate-limited the request and retry may succeed.
    RateLimited,
    /// The provider rate-limited the request because of an account limit.
    RateLimitedBilling,
    /// The provider request timed out.
    Timeout,
    /// Provider authentication was rejected.
    Authentication,
    /// The request payload exceeded the provider's size limit.
    PayloadTooLarge,
    /// The request exceeded the selected model's context window.
    ContextOverflow,
    /// The provider rejected a model or request setting.
    ModelConfigRejected,
    /// The selected model is unavailable from the provider.
    ModelUnavailable,
    /// The selected model does not support a requested capability.
    CapabilityUnsupported,
    /// The provider is temporarily unavailable.
    ProviderUnavailable,
    /// The provider rejected the request.
    RequestRejected,
    /// The connection to the provider failed.
    Network,
    /// An otherwise unclassified inference error occurred.
    Inference,
    /// The provider returned no response content.
    EmptyResponse,
    /// A class defined by a host or downstream library.
    HostDefined(&'static str),
}

/// Wire labels and user-facing copy for one failure class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailureCopy {
    /// Stable `chat_error` event type.
    pub error_type: &'static str,
    /// Source label associated with the failure.
    pub source: &'static str,
    /// Whether retrying the same request may succeed.
    pub retryable: bool,
    /// English fallback shown when the consumer has no translated value.
    pub copy: &'static str,
    /// Stable frontend translation key.
    pub key: &'static str,
}

/// Host-provided copy that can override or extend the neutral defaults.
pub trait FailureCopyProvider {
    /// Returns host-specific copy for `class`, when the host has an override.
    fn copy(&self, class: FailureCopyClass) -> Option<FailureCopy>;
}

/// Looks up host-provided copy first and falls back to the neutral table.
pub fn failure_copy_with(
    provider: &dyn FailureCopyProvider,
    class: FailureCopyClass,
) -> Option<FailureCopy> {
    provider.copy(class).or_else(|| default_failure_copy(class))
}

/// Looks up the provider-neutral default copy for `class`.
pub fn default_failure_copy(class: FailureCopyClass) -> Option<FailureCopy> {
    use FailureCopyClass::*;
    let (error_type, source, retryable, key, copy) = match class {
        ChatTemplateRejected => (
            "chat_template_rejected",
            "provider",
            true,
            "chat_error.chat_template_rejected",
            "The model's chat template rejected the request. Try a different model or request format.",
        ),
        RateLimited => (
            "rate_limited",
            "provider",
            true,
            "chat_error.rate_limited",
            "Your AI provider is rate-limiting requests. This is a transient upstream limit, not a thread-level block — you can retry in this thread.",
        ),
        RateLimitedBilling => (
            "rate_limited",
            "provider",
            false,
            "chat_error.rate_limited_billing",
            "The provider rejected requests because of a billing or plan limit. Check the provider account or choose another model.",
        ),
        Timeout => (
            "timeout",
            "transport",
            true,
            "chat_error.timeout",
            "The request timed out. Please check your connection and try again.",
        ),
        Authentication => (
            "auth_error",
            "config",
            false,
            "chat_error.auth_error",
            "There's an authentication issue with the AI provider. Please check the credentials configured for this provider.",
        ),
        PayloadTooLarge => (
            "payload_too_large",
            "config",
            false,
            "chat_error.payload_too_large",
            "Your message or attachment is too large for this model. Shorten it or remove the attachment — or start a new thread.",
        ),
        ContextOverflow => (
            "context_overflow",
            "config",
            false,
            "chat_error.context_overflow",
            "The conversation is too long. Please start a new chat.",
        ),
        ModelConfigRejected => (
            "model_unavailable",
            "config",
            false,
            "chat_error.model_config_rejected",
            "The provider rejected the model or one of its settings. Check the model configuration.",
        ),
        ModelUnavailable => (
            "model_unavailable",
            "config",
            false,
            "chat_error.model_unavailable",
            "The selected model isn't available on your provider. Check your model settings.",
        ),
        CapabilityUnsupported => (
            "capability_unsupported",
            "config",
            false,
            "chat_error.capability_unsupported",
            "This model can't process the requested input type. Remove that input or switch to a capable model.",
        ),
        ProviderUnavailable => (
            "provider_error",
            "provider",
            true,
            "chat_error.provider_unavailable",
            "The AI provider is temporarily unavailable. Please try again later.",
        ),
        RequestRejected => (
            "provider_request_rejected",
            "provider",
            false,
            "chat_error.request_rejected",
            "The AI provider rejected the request — this is usually a model or parameter incompatibility. Check your model configuration.",
        ),
        Network => (
            "network",
            "transport",
            true,
            "chat_error.network",
            "The connection to the AI service dropped mid-response — usually a sleep/wake or network change. Please try again.",
        ),
        Inference => (
            "inference",
            "provider",
            true,
            "chat_error.inference",
            "Something went wrong. Please try again.",
        ),
        EmptyResponse => (
            "empty_response",
            "agent_loop",
            true,
            "chat_error.empty_response",
            "The model returned an empty response. Please retry. If it keeps happening, try a different model or check its setup.",
        ),
        HostDefined(_) => return None,
    };
    Some(FailureCopy {
        error_type,
        source,
        retryable,
        copy,
        key,
    })
}

#[cfg(test)]
#[path = "copy_tests.rs"]
mod tests;
