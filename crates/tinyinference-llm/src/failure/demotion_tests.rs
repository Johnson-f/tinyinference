use super::*;

#[test]
fn demotion_precedence_matches_the_host_ladder_for_overlapping_facts() {
    let cases = [
        (
            ProviderHttpFailureFacts {
                budget_exhausted: true,
                no_model_loaded: true,
                ..Default::default()
            },
            DemotionKind::BudgetExhausted,
        ),
        (
            ProviderHttpFailureFacts {
                no_model_loaded: true,
                custom_upstream_bad_request: true,
                ..Default::default()
            },
            DemotionKind::NoModelLoaded,
        ),
        (
            ProviderHttpFailureFacts {
                custom_upstream_bad_request: true,
                access_policy_denied: true,
                ..Default::default()
            },
            DemotionKind::CustomUpstreamBadRequest,
        ),
        (
            ProviderHttpFailureFacts {
                access_policy_denied: true,
                configuration_rejected: true,
                ..Default::default()
            },
            DemotionKind::AccessPolicyDenied,
        ),
        (
            ProviderHttpFailureFacts {
                configuration_rejected: true,
                context_window_exceeded: true,
                ..Default::default()
            },
            DemotionKind::ConfigurationRejected,
        ),
        (
            ProviderHttpFailureFacts {
                context_window_exceeded: true,
                quota_exhausted: true,
                ..Default::default()
            },
            DemotionKind::ContextWindowExceeded,
        ),
        (
            ProviderHttpFailureFacts {
                quota_exhausted: true,
                api_key_rejected: true,
                ..Default::default()
            },
            DemotionKind::QuotaExhausted,
        ),
        (
            ProviderHttpFailureFacts {
                api_key_rejected: true,
                oauth_session_expired: true,
                ..Default::default()
            },
            DemotionKind::ApiKeyRejected,
        ),
        (
            ProviderHttpFailureFacts {
                oauth_session_expired: true,
                insufficient_credits: true,
                ..Default::default()
            },
            DemotionKind::OAuthSessionExpired,
        ),
        (
            ProviderHttpFailureFacts {
                insufficient_credits: true,
                ollama_cloud_internal_error: true,
                ..Default::default()
            },
            DemotionKind::InsufficientCredits,
        ),
        (
            ProviderHttpFailureFacts {
                ollama_cloud_internal_error: true,
                moderation_rejected: true,
                ..Default::default()
            },
            DemotionKind::OllamaCloudInternalError,
        ),
    ];

    for (facts, expected) in cases {
        assert_eq!(demote_decision(&facts, false), Some(expected));
    }
}

#[test]
fn managed_provider_keeps_only_unknown_model_configuration_demotions() {
    let rejected = ProviderHttpFailureFacts {
        configuration_rejected: true,
        ..Default::default()
    };
    assert_eq!(demote_decision(&rejected, true), None);

    let unknown_model = ProviderHttpFailureFacts {
        unknown_model: true,
        ..rejected
    };
    assert_eq!(
        demote_decision(&unknown_model, true),
        Some(DemotionKind::ConfigurationRejected)
    );
}

#[test]
fn managed_provider_auth_facts_are_not_classified_as_provider_credentials() {
    let facts = ProviderHttpFailureFacts {
        api_key_rejected: true,
        ..Default::default()
    };
    assert_eq!(demote_decision(&facts, true), None);

    let facts = ProviderHttpFailureFacts {
        oauth_session_expired: true,
        ..Default::default()
    };
    assert_eq!(demote_decision(&facts, true), None);
}

#[test]
fn unrelated_failures_are_not_demoted() {
    assert_eq!(
        demote_decision(&ProviderHttpFailureFacts::default(), false),
        None
    );
}

#[test]
fn moderation_rejections_are_demoted_for_managed_and_configured_providers() {
    let facts = ProviderHttpFailureFacts {
        moderation_rejected: true,
        ..Default::default()
    };
    for provider_is_managed in [false, true] {
        assert_eq!(
            demote_decision(&facts, provider_is_managed),
            Some(DemotionKind::ModerationRejected)
        );
    }
}
