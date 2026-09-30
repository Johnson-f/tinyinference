//! Budget-exhaustion classification for provider and backend error bodies.
//!
//! Budget exhaustion is reported by hosted backends and managed inference
//! providers as a deterministic user state, not a defect. Every layer that sees
//! such a body (HTTP errors, agent loop guards, schedulers, telemetry)
//! classifies it through this one predicate. See also
//! [`crate::config_rejection`], which is checked first for overlapping phrases.

/// Return whether a provider message represents deterministic exhausted-budget
/// user state rather than a product defect.
pub fn is_budget_exhausted_message(message: &str) -> bool {
    const PHRASES: &[&str] = &[
        "insufficient budget",
        "budget exceeded",
        "add credits",
        "insufficient balance",
        "no remaining credits",
        "credit balance is too low",
    ];
    let lower = message.to_ascii_lowercase();
    PHRASES.iter().any(|phrase| lower.contains(phrase))
}

#[cfg(test)]
#[path = "billing_tests.rs"]
mod tests;
