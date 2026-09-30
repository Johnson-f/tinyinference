use super::*;
use crate::model::ProviderError;

#[test]
fn structured_status_ignores_unanchored_numbers() {
    assert_eq!(
        structured_http_status("API error (403 Forbidden): nope"),
        Some(403)
    );
    assert_eq!(structured_http_status("HTTP 404 Not Found"), Some(404));
    assert_eq!(structured_http_status("status: 401"), Some(401));
    assert_eq!(structured_http_status("408 Request Timeout"), Some(408));
    assert_eq!(structured_http_status("upstream took 450ms"), None);
    assert_eq!(structured_http_status("gpt-4-0409 returned nothing"), None);
}

#[test]
fn provider_failure_taxonomy_is_complete() {
    assert_eq!(
        classify_provider_failure(Some(401), None, "invalid api key"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "too many requests"),
        ProviderFailureClass::RateLimited
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "insufficient_balance"),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert_eq!(
        classify_provider_failure(None, None, "503 Service Unavailable"),
        ProviderFailureClass::UpstreamUnhealthy
    );
    assert_eq!(
        classify_provider_failure(None, None, "api key not set"),
        ProviderFailureClass::NonRetryable
    );
}

#[test]
fn structured_status_takes_precedence_over_message_heuristics() {
    assert_eq!(
        classify_provider_failure(Some(400), None, "proxy said bad gateway"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(None, Some("invalid_request"), "502 bad gateway"),
        ProviderFailureClass::NonRetryable
    );
}

#[test]
fn structured_provider_error_uses_the_same_classifier() {
    let error = ProviderError {
        status: Some(429),
        code: Some("insufficient_quota".into()),
        message: "quota exhausted".into(),
        ..ProviderError::default()
    };
    assert_eq!(
        classify_provider_error(&error),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert!(!provider_error_is_retryable(&error));
}

#[test]
fn normalized_provider_retryability_is_authoritative() {
    let error = ProviderError {
        message: "malformed streaming payload".into(),
        retryable: false,
        ..ProviderError::default()
    };
    assert_eq!(
        classify_provider_error(&error),
        ProviderFailureClass::NonRetryable
    );
    assert!(!provider_error_is_retryable(&error));
}

#[test]
fn retry_after_accepts_integer_and_fractional_seconds() {
    assert_eq!(parse_retry_after_ms("Retry-After: 5"), Some(5_000));
    assert_eq!(
        parse_retry_after_ms("retry_after: 2.5 seconds"),
        Some(2_500)
    );
    assert_eq!(parse_retry_after_ms("Retry-After 7"), Some(7_000));
    assert_eq!(parse_retry_after_ms("no retry hint"), None);
    assert_eq!(
        parse_retry_after_ms("Retry-After: 1000000000000000000000000000000"),
        None
    );
    assert_eq!(parse_retry_after_ms("Retry-After: inf"), None);
}

#[test]
fn business_limit_text_is_recognised_without_a_status() {
    for message in [
        "your plan does not include this model",
        "insufficient_balance",
        "out of credits",
        "error code 1311 returned",
        "1113",
    ] {
        assert!(contains_business_limit(message), "{message}");
    }
    for message in [
        "too many requests",
        "rate limit exceeded",
        "code 13110 upstream",
        "code 21113",
    ] {
        assert!(!contains_business_limit(message), "{message}");
    }
}

// ── Body phrase matchers ─────────────────────────────────────────────────────

/// Verbatim TAURI-RUST-C9A provider body: the Kiro IDE proxy wraps its own
/// 402 monthly-quota refusal inside a 500 envelope.
const C9A_BODY: &str = "kiro API error (500 Internal Server Error): \
    {\"error\":{\"message\":\"HTTP 402 from Kiro IDE: {\\\"message\\\":\\\"You have \
    reached the limit.\\\",\\\"reason\\\":\\\"MONTHLY_REQUEST_COUNT\\\"}\",\
    \"type\":\"server_error\"}}";

/// Verbatim TAURI-RUST-AFE Responses-API plan-cap body.
const AFE_BODY: &str = "openai Responses API error: {\"error\":{\"type\":\
    \"usage_limit_reached\",\"message\":\"The usage limit has been reached\",\
    \"plan_type\":\"plus\",\"resets_at\":1750000000}}";

#[test]
fn context_window_matches_wrapped_500_body() {
    assert!(is_context_window_exceeded_message(
        "{\"error\":{\"code\":500,\"message\":\"Context size has been exceeded.\",\"type\":\"server_error\"}}"
    ));
}

#[test]
fn context_window_matches_established_phrasings() {
    for body in [
        "This model's maximum context length is 8192 tokens",
        "request exceeds the context window of this model",
        "context length exceeded",
        "too many tokens in the prompt",
        "token limit exceeded",
        "prompt is too long for the selected model",
        "input is too long",
    ] {
        assert!(
            is_context_window_exceeded_message(body),
            "should match context-overflow body: {body}"
        );
    }
}

#[test]
fn context_window_matches_lmstudio_n_keep_body() {
    let body = "lmstudio API error (400 Bad Request): {\"error\":\"The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 10978 >= n_ctx: 8192). Try to load the model with a larger context length, or provide a shorter input.\"}";
    assert!(is_context_window_exceeded_message(body));
    assert!(is_context_window_exceeded_message(
        "request rejected: prompt is greater than the context length of the loaded model"
    ));
    assert!(is_context_window_exceeded_message(
        "n_keep: 9000 >= n_ctx: 4096"
    ));
}

#[test]
fn context_window_ignores_unrelated_bodies() {
    for body in [
        "rate limit exceeded, retry after 30s",
        "Invalid request: model not found",
        "Insufficient budget",
        "tool call exceeded the allowed budget",
        // Only one of the paired n_keep/n_ctx tokens.
        "loaded model with n_ctx: 8192 and 32 layers",
    ] {
        assert!(
            !is_context_window_exceeded_message(body),
            "must NOT match unrelated body: {body}"
        );
    }
}

#[test]
fn context_window_token_rate_limits_are_not_overflow() {
    for body in [
        "Rate limit reached: too many tokens per minute (TPM) for this org",
        "rate_limit_exceeded: token limit exceeded, retry after 12s",
        "You have hit too many tokens per min; try again in 30s",
    ] {
        assert!(
            !is_context_window_exceeded_message(body),
            "TPM rate-limit must NOT match as context overflow: {body}"
        );
    }
    assert!(is_context_window_exceeded_message(
        "Request rejected: too many tokens in the input for this model"
    ));
}

#[test]
fn quota_exhausted_matches_verbatim_bodies() {
    assert!(body_indicates_quota_exhausted(C9A_BODY));
    assert!(body_indicates_quota_exhausted(AFE_BODY));
    assert!(body_indicates_quota_exhausted("usage_limit_reached"));
    assert!(body_indicates_quota_exhausted(
        "The usage limit has been reached"
    ));
}

#[test]
fn quota_exhausted_matches_common_phrasings() {
    for body in [
        "{\"reason\":\"MONTHLY_REQUEST_COUNT\"}",
        "You have reached the limit on your monthly requests",
        "monthly request quota reached",
        "monthly limit reached",
        "plan quota exceeded",
        "usage limit exceeded for this period",
    ] {
        assert!(
            body_indicates_quota_exhausted(body),
            "should match: {body:?}"
        );
    }
}

#[test]
fn quota_exhausted_ignores_unrelated_500_and_rate_limit() {
    for body in [
        "kiro API error (500 Internal Server Error): {\"error\":\
         {\"message\":\"upstream connection reset\",\"type\":\"server_error\"}}",
        "rate_limit_exceeded: too many requests, retry after 12s",
        "429 Too Many Requests",
        "context length exceeded: reduce the number of tokens",
    ] {
        assert!(
            !body_indicates_quota_exhausted(body),
            "should NOT match: {body:?}"
        );
    }
}

#[test]
fn rate_cap_matches_hxf_body_but_not_transient_or_context() {
    assert!(is_provider_rate_cap_exceeded_message(
        "groq API error (413 Payload Too Large): {\"error\":{\"message\":\"Request too large \
         for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on \
         tokens per minute (TPM): Limit 8000, Requested 42084.\",\"code\":\"rate_limit_exceeded\"}}"
    ));
    assert!(!is_provider_rate_cap_exceeded_message(
        "groq API error (429 Too Many Requests): Rate limit reached. Please try again in 2.5s."
    ));
    assert!(!is_provider_rate_cap_exceeded_message(
        "openai API error (400): This model's maximum context length is 8192 tokens"
    ));
    assert!(!is_provider_rate_cap_exceeded_message(
        "openai API error (413 Payload Too Large): request entity too large"
    ));
}

#[test]
fn insufficient_credits_matches_phrasings_and_ignores_unrelated() {
    for body in [
        "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 4096",
        "Insufficient credits",
        "insufficient balance",
        "insufficient funds",
        "Payment Required",
    ] {
        assert!(
            body_indicates_insufficient_credits(body),
            "should match: {body:?}"
        );
    }
    assert!(!body_indicates_insufficient_credits(
        "{\"error\":{\"message\":\"some unrelated condition\"}}"
    ));
    // Quota and credits are distinct buckets: the 500-wrapped C9A body is quota.
    assert!(!body_indicates_insufficient_credits(C9A_BODY));
}

#[test]
fn local_and_hosted_ollama_body_matchers() {
    assert!(body_indicates_no_model_loaded(
        "{\"error\":\"No models loaded. Please load a model in the developer page\"}"
    ));
    assert!(!body_indicates_no_model_loaded("model not found"));

    assert!(body_indicates_ollama_cloud_internal_error(
        "{\"error\":\"Internal Server Error (ref: 3f2b1c7e-1111-2222-3333-444455556666)\"}"
    ));
    // A local daemon 500 has no `ref:` UUID.
    assert!(!body_indicates_ollama_cloud_internal_error(
        "{\"error\":\"Internal Server Error\"}"
    ));
}

#[test]
fn policy_moderation_and_upstream_body_matchers() {
    assert!(body_indicates_provider_access_policy_denied(
        "{\"error\":{\"type\":\"access_terminated_error\"}}"
    ));
    assert!(body_indicates_provider_access_policy_denied(
        "This endpoint is currently only available for Coding Agents"
    ));
    assert!(!body_indicates_provider_access_policy_denied("forbidden"));

    assert!(body_indicates_moderation_rejection(
        "{\"error\":\"Message rejected by Ombudsman\",\"score\":80}"
    ));
    assert!(body_indicates_moderation_rejection("{\"score\": 3}"));
    assert!(!body_indicates_moderation_rejection(
        "invalid request: score must be positive"
    ));

    assert!(body_indicates_custom_openai_upstream_bad_request(
        "{\"error\":{\"message\":\"Bad request to upstream provider\",\"type\":\"upstream_error\",\"status\":400}}"
    ));
    assert!(!body_indicates_custom_openai_upstream_bad_request(
        "Bad request to upstream provider"
    ));
}

#[test]
fn auth_key_error_body_matcher() {
    for body in [
        "{\"type\":\"authentication_error\"}",
        "{\"code\":\"invalid_api_key\"}",
        "Incorrect API key provided",
        "no api key supplied",
        "Invalid or missing API key",
    ] {
        assert!(
            body_indicates_auth_key_error(body),
            "should match: {body:?}"
        );
    }
    assert!(!body_indicates_auth_key_error("quota exceeded"));
    // Provider-specific clauses (e.g. OpenRouter "user not found") stay in the host.
    assert!(!body_indicates_auth_key_error("User not found."));
}
