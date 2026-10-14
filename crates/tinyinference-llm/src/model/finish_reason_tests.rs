use super::*;

#[test]
fn every_output_limit_spelling_becomes_length() {
    for raw in ["length", "max_tokens", "MAX_TOKENS", "max_output_tokens"] {
        assert_eq!(
            canonical_finish_reason(raw.to_string()),
            FINISH_REASON_LENGTH,
            "{raw}"
        );
    }
}

#[test]
fn other_finish_reasons_pass_through_unchanged() {
    for raw in [
        "stop",
        "end_turn",
        "tool_use",
        "tool_calls",
        "stop_sequence",
        "content_filter",
        "pause_turn",
        "incomplete",
    ] {
        assert_eq!(canonical_finish_reason(raw.to_string()), raw);
    }
}
