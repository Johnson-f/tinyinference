//! String-level predicates and extractors over a flattened provider error.
//!
//! Hosts collapse typed provider errors into a `String` at process
//! boundaries; these helpers re-detect the canonical shapes from that text
//! without depending on any product copy.

/// String-flat mirror of
/// the host's observability `is_empty_provider_response_message`.
///
/// The typed `AgentError::EmptyProviderResponse` is collapsed to a `String`
/// at the native-bus boundary before reaching this layer, so we re-detect
/// the same canonical phrase the agent harness emits. Anchored on
/// `"model returned an empty response"` (the verbatim user-facing string from
/// `AgentError::EmptyProviderResponse`) — NOT the looser `"empty response"`,
/// so internal fall-through phrases (`"summarizer returned empty response"`,
/// `"provider returned an empty response; returning empty extraction"`) are
/// not misclassified. Keep the anchor in lockstep with the observability
/// mirror.
///
/// Caller passes the already-lowercased error string.
pub fn is_empty_provider_response_text(lower: &str) -> bool {
    lower.contains("model returned an empty response")
}

/// Detect a malformed tool-history rejection (orphaned / mismatched
/// `role:'tool'` message). This is the *poisoned history* shape the de-poison
/// guard recovers from — NOT a model/parameter mismatch — so it earns the
/// "we cleared it, resend" copy instead of "try a different model".
///
/// Anchored on the managed backend's `validateToolMessageOrdering` strings
/// (verified against tinyhumansai/backend `chatCompletions.ts` — "role 'tool' …
/// matching tool_call", "does not match any tool_call from the preceding
/// assistant message"), the raw upstream jinja variant ("tool role … no
/// previous assistant message with a tool call"), and the equivalent BYO
/// provider phrasings. Caller passes the already-lowercased error string.
pub fn is_malformed_tool_history_text(lower: &str) -> bool {
    let tool_role = lower.contains("role 'tool'") || lower.contains("tool role");
    let about_tool_call = lower.contains("tool call") || lower.contains("tool_call");
    (tool_role && about_tool_call)
        || lower.contains("does not match any tool_call from the preceding assistant message")
}

/// Detect a transport-level connection drop with no provider status / managed
/// `errorCode` — the residue that otherwise falls to the generic `inference`
/// catch-all (issue #3714 bucket #1).
///
/// Anchored on the canonical reqwest/hyper shapes for a severed or never-opened
/// connection (stale keep-alive reused after sleep/wake, network change, raw
/// mid-stream SSE drop). Intentionally does NOT match `"timed out"` (the
/// dedicated `timeout` arm owns that) nor any `4xx/5xx` status (those arms claim
/// their shapes earlier). Caller passes the already-lowercased error string.
pub fn is_connection_dropped_text(lower: &str) -> bool {
    const DROP_MARKERS: &[&str] = &[
        "connection closed before message completed", // hyper IncompleteMessage
        "error reading a body from connection",
        "connection reset",
        "connection refused",
        "connection aborted",
        "broken pipe",
        "unexpected end of file",
        "unexpected eof",
        "error sending request",
        "tcp connect error",
        "dns error",
        "failed to lookup address",
    ];
    DROP_MARKERS.iter().any(|marker| lower.contains(marker))
}

/// Detect an un-claimed provider 4xx (generic client-side request rejection).
///
/// Mirrors the status tokens emitted by `inference::provider::ops::api_error`
/// (`"<provider> API error (400 Bad Request): …"`). Ordered AFTER the
/// provider-config-rejection and model-unavailable arms in
/// the host's inference-error classifier, so
/// only 4xx shapes those arms did not claim reach this predicate.
///
/// Caller passes the already-lowercased error string.
pub fn is_provider_request_rejected_text(lower: &str) -> bool {
    // Match only when the 4xx status appears inside a provider error envelope
    // (`<provider> API error (4xx …)`, emitted by
    // `inference::provider::ops::api_error`). Matching a bare "400"/"404"
    // anywhere would misclassify unrelated errors that merely contain those
    // digits (token counts, byte offsets, timestamps). Per CodeRabbit review
    // on PR #3199. The `returned http 4xx` forms are tinyinference's
    // `ProviderError` Display (`<provider> returned HTTP 400: …`), which is
    // how a failure reported inside a stream reaches classification (#6724).
    const PROVIDER_4XX_MARKERS: &[&str] = &[
        "api error (400",
        "api error (404",
        "api error (409",
        "api error (422",
        "returned http 400",
        "returned http 404",
        "returned http 409",
        "returned http 422",
    ];
    PROVIDER_4XX_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Whether a model-availability error body describes a **transient** upstream
/// outage rather than a user misconfiguration (#5503).
///
/// The `model_unavailable` arm matches on the bare word `unavailable`, which a
/// provider emits for BOTH "you picked a model I don't host" (config, terminal)
/// and "this model is temporarily down / overloaded right now" (transient,
/// retryable). Only the second class carries one of these temporary-outage
/// markers, so it's the safe discriminator: a terminal endpoint rejection like
/// `"model unavailable on this endpoint"` (a 404 for a model that endpoint
/// doesn't host) carries none of them and stays on the config verdict.
///
/// Deliberately does NOT key on the bare word `unavailable` — that's the very
/// ambiguity being disambiguated. Caller passes the already-lowercased string.
pub fn is_transient_unavailability_text(lower: &str) -> bool {
    const TRANSIENT_MARKERS: &[&str] = &[
        "temporarily",
        "temporary",
        "currently unavailable",
        "currently overloaded",
        "overloaded",
        "try again later",
        "try again in a",
        "please retry",
    ];
    TRANSIENT_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Extract a Retry-After / retry_after seconds hint from a free-form
/// error string. Mirrors [`super::parse_retry_after_ms`] but operates on
/// the already-flattened `String` that reaches the channel-classifier
/// layer.
///
/// Returns `Some(n)` when a non-negative integer or fractional value
/// follows one of the canonical headers; fractional values are
/// rounded up so the user is never told to retry sooner than the
/// upstream actually allows.
pub fn parse_retry_after_secs(err: &str) -> Option<u64> {
    // Normalise quoted JSON-key wrappers ("retry_after": 30) by
    // stripping double quotes before scanning for prefixes
    // (CodeRabbit review on #2371). A serialised provider body like
    // `{"retry_after": 30}` would otherwise miss every prefix and
    // the user would lose the retry hint the provider supplied.
    let normalized = err.to_ascii_lowercase().replace('"', "");
    for prefix in &[
        "retry-after:",
        "retry_after:",
        "retry-after ",
        "retry_after ",
        // Managed backend (#870) emits the structured `retryAfter` field
        // (camelCase). After lower-casing + quote-stripping above it
        // collapses to `retryafter: 30` / `retryafter 30`, so the
        // separator-bearing prefixes here let the same parser surface the
        // structured field the spec asks us to prefer (F5).
        "retryafter:",
        "retryafter ",
    ] {
        if let Some(pos) = normalized.find(prefix) {
            let after = &normalized[pos + prefix.len()..];
            let num_str: String = after
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if let Ok(secs) = num_str.parse::<f64>()
                && secs.is_finite()
                && secs >= 0.0
            {
                return Some(secs.ceil() as u64);
            }
        }
    }
    None
}

/// Pull the structured provider error message out of a raw error string.
///
/// Provider error chains from OpenAI/Anthropic/OpenRouter/etc. arrive looking
/// like `custom_openai API error (404 Not Found): {"error":{"message":"...","type":"..."}}`.
/// We extract the `error.message` value so the UI can show the *real* reason
/// — e.g. "Project ... does not have access to model `gpt-5.5`" — instead of
/// a generic apology.
///
/// Returns `None` for transport-level failures (DNS, TLS, connect refused)
/// where there is no provider body to quote — those have no actionable
/// detail and the raw error text can leak internal infrastructure URLs,
/// which the chat surface deliberately does not expose to end users.
pub fn extract_provider_error_detail(err: &str) -> Option<String> {
    const MAX_DETAIL_CHARS: usize = 300;

    // Find the first `"message"` JSON field anywhere in the error chain.
    let key = "\"message\"";
    let idx = err.find(key)?;
    let after_key = &err[idx + key.len()..];
    // Skip whitespace and the colon to the opening quote of the value.
    let after_colon = after_key.trim_start_matches(|c: char| c != '"');
    let stripped = after_colon.strip_prefix('"')?;

    // Manual unescape over the standard JSON escape set. `\uXXXX` is
    // deliberately left alone rather than decoded: this is an error-display
    // path, and an unhandled sequence should stay visible as the literal
    // `\uXXXX` instead of silently losing a character.
    let mut out = String::new();
    let mut chars = stripped.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                let trimmed = out.trim();
                if trimmed.is_empty() {
                    return None;
                }
                let sanitized = tinyinference_core::sanitize::sanitize_api_error(trimmed);
                return Some(truncate_with_ellipsis(&sanitized, MAX_DETAIL_CHARS));
            }
            '\\' => {
                if let Some(esc) = chars.next() {
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        other => {
                            out.push('\\');
                            out.push(other);
                        }
                    }
                }
            }
            other => out.push(other),
        }
    }

    None
}

/// Append the upstream provider detail to a user-facing message, if a useful
/// one can be extracted. Keeps the friendly summary first and the verbatim
/// provider reason below as a quotable block.
pub fn with_provider_detail(summary: &str, err: &str) -> String {
    match extract_provider_error_detail(err) {
        Some(detail) => format!("{summary}\n\n> {detail}"),
        None => summary.to_string(),
    }
}

/// Best-effort extraction of the provider name from an error string.
///
/// `inference::provider::ops::api_error` formats upstream failures as
/// `"<provider> API error (<status>): <body>"`, e.g.
/// `"openrouter API error (429 Too Many Requests): ..."`. We pull the
/// leading word and lowercase it so the wire value is stable across
/// providers' own capitalisation.
///
/// Returns `None` when:
/// - The error string doesn't carry the `" API error"` infix.
/// - The candidate word contains characters that wouldn't appear in a
///   provider name (slashes, colons, etc. — guards against transport
///   error prefixes that happen to be followed by " API error").
pub fn extract_provider_name(err: &str) -> Option<String> {
    const INFIX: &str = " API error";
    let idx = err.find(INFIX)?;
    let prefix = err[..idx].trim_end();
    let candidate = prefix
        .rsplit_once(char::is_whitespace)
        .map_or(prefix, |(_, last)| last);
    if candidate.is_empty()
        || !candidate
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return None;
    }
    Some(candidate.to_ascii_lowercase())
}

/// Detect the reliable-provider aggregate that fires once every
/// configured `model_fallbacks` entry has been tried.
///
/// `reliable.rs::format_failure_aggregate` always opens with
/// `"All providers/models failed. Attempts:"`. When that marker is
/// present the FE should NOT offer a fallback retry — there is none
/// left to try.
pub fn is_fallback_chain_exhausted(err: &str) -> bool {
    err.contains("All providers/models failed")
}

/// Truncate to at most `max_chars` characters (UTF-8 safe), trimming trailing
/// whitespace and appending `...` when truncated.
fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}...", s[..idx].trim_end()),
        None => s.to_string(),
    }
}

/// Whether a lowercased flattened error reads as a rate limit (`rate limit`
/// wording or a bare `429`). The broadest of the status-ish arms: a host must
/// test its own, more specific rate caps first.
pub fn is_rate_limit_text(lower: &str) -> bool {
    lower.contains("rate limit") || lower.contains("429")
}

/// Whether a lowercased flattened error reads as a request timeout.
pub fn is_timeout_text(lower: &str) -> bool {
    lower.contains("timeout") || lower.contains("timed out")
}

/// Whether a lowercased flattened error reads as an authentication failure
/// (`401`, `unauthorized`, or a mention of the API key).
pub fn is_auth_error_text(lower: &str) -> bool {
    lower.contains("401") || lower.contains("unauthorized") || lower.contains("api key")
}

/// Whether a lowercased flattened error reads as a payment / balance failure
/// (`402`, `payment required`, `insufficient balance`).
pub fn is_payment_required_text(lower: &str) -> bool {
    lower.contains("402")
        || lower.contains("payment required")
        || lower.contains("insufficient balance")
}

/// Whether a lowercased flattened error reads as a provider-side 5xx outage
/// (`500`, `internal server`, `service unavailable`, `503`).
pub fn is_server_error_text(lower: &str) -> bool {
    lower.contains("500")
        || lower.contains("internal server")
        || lower.contains("service unavailable")
        || lower.contains("503")
}

/// Whether a lowercased flattened error reads as a context-length overflow:
/// the word `context` together with a length/limit/exceed/token word.
pub fn is_context_length_text(lower: &str) -> bool {
    lower.contains("context")
        && (lower.contains("length")
            || lower.contains("limit")
            || lower.contains("exceed")
            || lower.contains("token"))
}

/// Whether a lowercased flattened error reads as a model that is missing or
/// unavailable (`model` plus not found / unavailable / does not exist / does
/// not have access). Pair with [`is_transient_unavailability_text`] to tell a
/// stale model pin from a temporary outage.
pub fn is_model_unavailable_text(lower: &str) -> bool {
    lower.contains("model")
        && (lower.contains("not found")
            || lower.contains("unavailable")
            || lower.contains("does not exist")
            || lower.contains("does not have access"))
}

/// Whether a lowercased flattened error says the model cannot take image input.
pub fn is_vision_unsupported_text(lower: &str) -> bool {
    lower.contains("does not support vision") || lower.contains("capability=vision")
}

/// Whether a lowercased flattened error is the Codex OAuth refresh-failure
/// sentinel (`codex authentication token is expired`).
pub fn is_codex_token_expired_text(lower: &str) -> bool {
    lower.contains("codex authentication token is expired")
}
