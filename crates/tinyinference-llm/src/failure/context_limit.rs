//! Extracting the context limit a provider states in an overflow error.
//!
//! When a request overflows, most providers say how big the window really is:
//!
//! - OpenAI, DeepSeek, OpenRouter and vLLM: `"This model's maximum context
//!   length is 131072 tokens. However, you requested 140000 tokens …"` (OpenRouter
//!   says `"This endpoint's …"`).
//! - Anthropic: `"prompt is too long: 208000 tokens > 200000 maximum"`.
//! - Newer vLLM: `"… is longer than the maximum model length of 32768"`.
//! - Mistral: `"… too large for model with 32768 maximum context length"`.
//! - Gemini: `"… exceeds the maximum number of tokens allowed (1048576)"`.
//! - llama.cpp / LM Studio: `"(n_keep: 10978 >= n_ctx: 8192)"` and
//!   `"exceeds the available context size (8192 tokens)"`.
//!
//! That number is the authoritative window for the endpoint that answered, so a
//! host can record it as a correction (see
//! [`crate::model::discover::record_overflow_error`]) instead of retrying
//! against a guessed one.

/// The smallest value accepted as a context window. Anything below this is
/// almost certainly a different number in the message (a status, a count of
/// messages), not a window.
const MIN_PLAUSIBLE_WINDOW: u64 = 512;
/// The largest value accepted as a context window (100M tokens).
const MAX_PLAUSIBLE_WINDOW: u64 = 100_000_000;

/// Phrases after which the stated limit is the first number that follows.
///
/// Matched against the lowercased message. Each anchor is specific enough that
/// the next number is the window, not the requested size.
const LIMIT_FOLLOWS: &[&str] = &[
    "maximum context length is",
    "maximum context length of",
    "max context length is",
    "context length is only",
    "maximum model length of",
    "maximum model length is",
    "max_model_len",
    "maximum number of tokens allowed",
    "available context size",
    "n_ctx:",
    "n_ctx =",
    "context window of",
    "context window is",
    "context_length of",
];

/// Phrases before which the stated limit is the last number that precedes.
const LIMIT_PRECEDES: &[&str] = &["maximum context length", "maximum\""];

/// Extracts the context window a provider states in an overflow error body.
///
/// Returns `None` when the text states no recognisable limit, including when it
/// is not a context-overflow message at all. The caller does not need to check
/// [`crate::failure::is_context_window_exceeded_message`] first, but recording
/// the result is only meaningful for overflow errors.
#[must_use]
pub fn parse_context_limit_from_error(message: &str) -> Option<u64> {
    let lower = message.to_ascii_lowercase();

    // Anthropic: "prompt is too long: 208000 tokens > 200000 maximum". The
    // window is the number after `>`.
    if let Some(start) = lower.find("prompt is too long") {
        let tail = &lower[start..];
        if let Some(gt) = tail.find('>') {
            if let Some(limit) = first_number(&tail[gt + 1..], 24) {
                return plausible(limit);
            }
        }
    }

    for anchor in LIMIT_FOLLOWS {
        if let Some(start) = lower.find(anchor) {
            if let Some(limit) = first_number(&lower[start + anchor.len()..], 24) {
                if let Some(limit) = plausible(limit) {
                    return Some(limit);
                }
            }
        }
    }

    // Mistral: "too large for model with 32768 maximum context length". The
    // window is the number right before the phrase.
    for anchor in LIMIT_PRECEDES {
        if let Some(end) = lower.find(anchor) {
            if let Some(limit) = last_number(&lower[..end], 16) {
                if let Some(limit) = plausible(limit) {
                    return Some(limit);
                }
            }
        }
    }

    None
}

fn plausible(value: u64) -> Option<u64> {
    (MIN_PLAUSIBLE_WINDOW..=MAX_PLAUSIBLE_WINDOW)
        .contains(&value)
        .then_some(value)
}

/// The first integer (digits, optionally grouped with `,` or `_`) that starts
/// within `window` bytes of the start of `text`.
fn first_number(text: &str, window: usize) -> Option<u64> {
    let start = text
        .char_indices()
        .take_while(|(index, _)| *index <= window)
        .find(|(_, character)| character.is_ascii_digit())
        .map(|(index, _)| index)?;
    parse_grouped(&text[start..])
}

/// The last integer that ends within `window` bytes of the end of `text`.
fn last_number(text: &str, window: usize) -> Option<u64> {
    let trimmed = text.trim_end();
    let tail_start = trimmed.len().saturating_sub(window);
    let tail_start = (tail_start..=trimmed.len())
        .find(|index| trimmed.is_char_boundary(*index))
        .unwrap_or(trimmed.len());
    let tail = &trimmed[tail_start..];
    let end = tail.rfind(|character: char| character.is_ascii_digit())? + 1;
    let begin = tail[..end]
        .rfind(|character: char| !(character.is_ascii_digit() || character == ','))
        .map_or(0, |index| index + 1);
    parse_grouped(&tail[begin..end])
}

/// Parses a leading integer, accepting `,` / `_` digit grouping.
fn parse_grouped(text: &str) -> Option<u64> {
    let mut digits = String::new();
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if character.is_ascii_digit() {
            digits.push(character);
        } else if (character == ',' || character == '_')
            && chars.peek().is_some_and(char::is_ascii_digit)
        {
            continue;
        } else {
            break;
        }
    }
    if digits.is_empty() || digits.len() > 12 {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
#[path = "context_limit_tests.rs"]
mod tests;
