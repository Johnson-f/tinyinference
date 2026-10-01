//! Provider authentication failure classification.

/// Returns whether an error body indicates an expired OpenAI OAuth session.
///
/// This distinguishes ChatGPT/Codex subscription token expiry from ordinary
/// API-key rejection. HTTP status and host-provider exclusions remain the
/// responsibility of the caller because they are routing policy.
#[must_use]
pub fn is_openai_oauth_session_expired_message(message: &str) -> bool {
    const OAUTH_EXPIRY_MARKERS: &[&str] = &[
        "token_expired",
        "authentication token is expired",
        "please try signing in again",
    ];
    let lower = message.to_ascii_lowercase();
    OAUTH_EXPIRY_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
