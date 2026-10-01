//! Budget-exhaustion classification for provider and backend error bodies.
//!
//! Budget exhaustion is reported by hosted backends and managed inference
//! providers as a deterministic user state, not a defect. Every layer that sees
//! such a body (HTTP errors, agent loop guards, schedulers, telemetry)
//! classifies it through this one predicate. See also
//! [`crate::config_rejection`], which is checked first for overlapping phrases.

/// Phrases that mark a deterministic exhausted-budget state. Matched as
/// lowercase substrings by [`BudgetMatch::Billing`] (and as the shared tail of
/// [`BudgetMatch::Managed`]).
const BILLING_PHRASES: &[&str] = &[
    "insufficient budget",
    "budget exceeded",
    "add credits",
    "insufficient balance",
    "no remaining credits",
    "credit balance is too low",
];

/// Word sequences matched by [`BudgetMatch::Strict`], whole-word only.
const STRICT_NEEDLES: &[&str] = &[
    "budget exceeded",
    "budget exceeds",
    "top up",
    "add credits",
    "out of credits",
    "no remaining credits",
];

/// How eagerly [`is_budget_message`] reads a message. The three hosts that
/// used to carry their own phrase lists each needed a different trade-off
/// between missing a real budget error and mislabelling a real defect, so the
/// strictness is an explicit argument instead of three copies of the table.
///
/// | mode | normalisation | matches |
/// | --- | --- | --- |
/// | `Billing` | ASCII lowercase | substring of any [`BILLING_PHRASES`] entry |
/// | `Managed` | lowercase, runs of `-`, `_` and whitespace folded to one space | `budget`..`exceed`, `top up`, `add`..`credits`, `out of credits`, `no remaining credits` (the first and third allow any text between the words), **or** `Billing` on the original message |
/// | `Strict` | lowercase, every non-alphanumeric byte to a space | a [`STRICT_NEEDLES`] entry as a whole-word sequence, so `stop updating` never reads as `top up` |
///
/// `Strict` deliberately does **not** include the `Billing`-only phrases
/// (`insufficient budget`, `insufficient balance`, `credit balance is too low`):
/// a false positive there hides a real fatal error from telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetMatch {
    /// Provider/backend billing phrases only.
    Billing,
    /// `Billing` plus the looser top-up / out-of-credits wording the chat
    /// surface offers actionable copy for.
    Managed,
    /// Whole-word needles only; for callers where a false positive is costly.
    Strict,
}

/// Return whether `message` signals an exhausted inference budget under the
/// given [`BudgetMatch`] strictness. The single phrase matcher behind every
/// budget classification in the host.
pub fn is_budget_message(message: &str, mode: BudgetMatch) -> bool {
    match mode {
        BudgetMatch::Billing => matches_billing(message),
        BudgetMatch::Managed => matches_managed(message) || matches_billing(message),
        BudgetMatch::Strict => matches_strict(message),
    }
}

fn matches_billing(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    BILLING_PHRASES.iter().any(|phrase| lower.contains(phrase))
}

fn matches_managed(message: &str) -> bool {
    let mut normalized = String::with_capacity(message.len());
    let mut in_run = false;
    for c in message.trim().to_ascii_lowercase().chars() {
        if c == '-' || c == '_' || c.is_whitespace() {
            if !in_run {
                normalized.push(' ');
                in_run = true;
            }
        } else {
            normalized.push(c);
            in_run = false;
        }
    }
    let gap = |first: &str, second: &str| {
        normalized
            .find(first)
            .is_some_and(|at| normalized[at + first.len()..].contains(second))
    };
    gap("budget", "exceed")
        || normalized.contains("top up")
        || gap("add", "credits")
        || normalized.contains("out of credits")
        || normalized.contains("no remaining credits")
}

fn matches_strict(message: &str) -> bool {
    let normalized: String = message
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect();
    let words: Vec<&str> = normalized.split_whitespace().collect();
    STRICT_NEEDLES.iter().any(|needle| {
        let needle_words: Vec<&str> = needle.split_whitespace().collect();
        words
            .windows(needle_words.len())
            .any(|window| window == needle_words.as_slice())
    })
}

/// Return whether a provider message represents deterministic exhausted-budget
/// user state rather than a product defect ([`BudgetMatch::Billing`]).
pub fn is_budget_exhausted_message(message: &str) -> bool {
    is_budget_message(message, BudgetMatch::Billing)
}

#[cfg(test)]
#[path = "billing_tests.rs"]
mod tests;
