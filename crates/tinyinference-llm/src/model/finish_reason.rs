//! One spelling for the finish reason providers report in several.
//!
//! OpenAI-compatible endpoints report `length` when the output cap cut a
//! reply off, Anthropic reports `max_tokens`, Gemini (through a gateway)
//! `MAX_TOKENS`, and the OpenAI Responses API `max_output_tokens`. A consumer
//! comparing against one spelling misses truncation on the others, so every
//! adapter routes the value through [`canonical_finish_reason`] before it
//! reaches [`ModelResponse::finish_reason`]. The provider's own value is still
//! in the response's `raw` payload for anyone who needs it.
//!
//! [`ModelResponse::finish_reason`]: crate::model::ModelResponse::finish_reason

/// The canonical finish reason for a reply the output limit cut off.
pub const FINISH_REASON_LENGTH: &str = "length";

/// Maps a provider's spelling of an output-limit stop to
/// [`FINISH_REASON_LENGTH`]. Every other value passes through unchanged.
#[must_use]
pub fn canonical_finish_reason(raw: String) -> String {
    if is_length_spelling(&raw) {
        FINISH_REASON_LENGTH.to_string()
    } else {
        raw
    }
}

/// The spellings providers use for "the output limit ended the reply".
fn is_length_spelling(raw: &str) -> bool {
    ["length", "max_tokens", "max_output_tokens"]
        .iter()
        .any(|spelling| raw.eq_ignore_ascii_case(spelling))
}

#[cfg(test)]
#[path = "finish_reason_tests.rs"]
mod tests;
