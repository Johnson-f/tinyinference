use super::*;

struct HostCopy;

impl FailureCopyProvider for HostCopy {
    fn copy(&self, class: FailureCopyClass) -> Option<FailureCopy> {
        match class {
            FailureCopyClass::RateLimited => Some(FailureCopy {
                error_type: "host_rate_limit",
                source: "host_provider",
                retryable: false,
                copy: "Host-specific rate-limit guidance.",
                key: "chat_error.host_rate_limited",
            }),
            FailureCopyClass::HostDefined("host_account_limit") => Some(FailureCopy {
                error_type: "account_limit",
                source: "host_billing",
                retryable: false,
                copy: "Host-defined account limit.",
                key: "chat_error.host_account_limit",
            }),
            _ => None,
        }
    }
}

#[test]
fn lookup_uses_host_override_and_falls_back_to_neutral_copy() {
    let host = HostCopy;
    let override_copy = failure_copy_with(&host, FailureCopyClass::RateLimited).unwrap();
    assert_eq!(override_copy.key, "chat_error.host_rate_limited");
    assert_eq!(override_copy.error_type, "host_rate_limit");
    assert!(!override_copy.retryable);

    let fallback = failure_copy_with(&host, FailureCopyClass::Timeout).unwrap();
    assert_eq!(
        fallback,
        default_failure_copy(FailureCopyClass::Timeout).unwrap()
    );
    assert_eq!(fallback.key, "chat_error.timeout");

    let extension =
        failure_copy_with(&host, FailureCopyClass::HostDefined("host_account_limit")).unwrap();
    assert_eq!(extension.error_type, "account_limit");
    assert_eq!(extension.key, "chat_error.host_account_limit");
    assert_eq!(
        failure_copy_with(&host, FailureCopyClass::HostDefined("unknown")),
        None
    );
}

#[test]
fn neutral_table_keeps_frontend_metadata_stable() {
    let expected = [
        (
            FailureCopyClass::ChatTemplateRejected,
            "chat_template_rejected",
            "provider",
            true,
            "chat_error.chat_template_rejected",
        ),
        (
            FailureCopyClass::RateLimited,
            "rate_limited",
            "provider",
            true,
            "chat_error.rate_limited",
        ),
        (
            FailureCopyClass::RateLimitedBilling,
            "rate_limited",
            "provider",
            false,
            "chat_error.rate_limited_billing",
        ),
        (
            FailureCopyClass::Timeout,
            "timeout",
            "transport",
            true,
            "chat_error.timeout",
        ),
        (
            FailureCopyClass::Authentication,
            "auth_error",
            "config",
            false,
            "chat_error.auth_error",
        ),
        (
            FailureCopyClass::PayloadTooLarge,
            "payload_too_large",
            "config",
            false,
            "chat_error.payload_too_large",
        ),
        (
            FailureCopyClass::ContextOverflow,
            "context_overflow",
            "config",
            false,
            "chat_error.context_overflow",
        ),
        (
            FailureCopyClass::ModelConfigRejected,
            "model_unavailable",
            "config",
            false,
            "chat_error.model_config_rejected",
        ),
        (
            FailureCopyClass::ModelUnavailable,
            "model_unavailable",
            "config",
            false,
            "chat_error.model_unavailable",
        ),
        (
            FailureCopyClass::CapabilityUnsupported,
            "capability_unsupported",
            "config",
            false,
            "chat_error.capability_unsupported",
        ),
        (
            FailureCopyClass::ProviderUnavailable,
            "provider_error",
            "provider",
            true,
            "chat_error.provider_unavailable",
        ),
        (
            FailureCopyClass::RequestRejected,
            "provider_request_rejected",
            "provider",
            false,
            "chat_error.request_rejected",
        ),
        (
            FailureCopyClass::Network,
            "network",
            "transport",
            true,
            "chat_error.network",
        ),
        (
            FailureCopyClass::Inference,
            "inference",
            "provider",
            true,
            "chat_error.inference",
        ),
        (
            FailureCopyClass::EmptyResponse,
            "empty_response",
            "agent_loop",
            true,
            "chat_error.empty_response",
        ),
    ];
    for (class, error_type, source, retryable, key) in expected {
        let row = default_failure_copy(class).unwrap();
        assert_eq!(
            (row.error_type, row.source, row.retryable, row.key),
            (error_type, source, retryable, key)
        );
        assert!(!row.copy.is_empty());
    }
    assert_eq!(
        default_failure_copy(FailureCopyClass::HostDefined("unknown")),
        None
    );
}
