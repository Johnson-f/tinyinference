use super::*;

/// Drives a sequence of content deltas through the stream machine and
/// returns the accumulated `(visible, reasoning)`.
fn run_stream(config: &ReasoningTagExtraction, deltas: &[&str]) -> (String, String) {
    let mut machine = ReasoningTagStream::new(config);
    let mut visible = String::new();
    let mut reasoning = String::new();
    for delta in deltas {
        machine.push(delta, &mut visible, &mut reasoning);
    }
    machine.finish(&mut visible, &mut reasoning);
    (visible, reasoning)
}

#[test]
fn potential_start_index_finds_full_and_partial_matches() {
    assert_eq!(potential_start_index("ab<think>c", "<think>"), Some(2));
    // Partial tag straddling the end is held from its first byte.
    assert_eq!(potential_start_index("hello<thi", "<think>"), Some(5));
    assert_eq!(potential_start_index("<", "<think>"), Some(0));
    // A lone trailing `<` with a non-matching continuation is not held.
    assert_eq!(potential_start_index("<h", "<think>"), None);
    assert_eq!(potential_start_index("plain text", "<think>"), None);
    // `<thinker>` is not `<think>` and must not be held back.
    assert_eq!(potential_start_index("<thinker>", "<think>"), None);
}

#[test]
fn tag_fully_inside_one_delta() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = run_stream(&cfg, &["Hi<think>ponder</think>there"]);
    assert_eq!(visible, "Hithere");
    assert_eq!(reasoning, "ponder");
}

#[test]
fn tag_split_across_deltas_at_every_boundary() {
    let cfg = ReasoningTagExtraction::default();
    let source = "before<think>secret</think>after";
    let open = "<think>";
    // Split the opening tag at every interior byte boundary.
    for cut in 0..=open.len() {
        let prefix = format!("before{}", &open[..cut]);
        let suffix = format!("{}secret</think>after", &open[cut..]);
        let (visible, reasoning) = run_stream(&cfg, &[&prefix, &suffix]);
        assert_eq!(visible, "beforeafter", "cut at {cut}");
        assert_eq!(reasoning, "secret", "cut at {cut}");
    }
    // Sanity: the whole thing in one delta agrees.
    let (visible, reasoning) = run_stream(&cfg, &[source]);
    assert_eq!(visible, "beforeafter");
    assert_eq!(reasoning, "secret");
}

#[test]
fn closing_tag_split_across_deltas() {
    let cfg = ReasoningTagExtraction::default();
    let close = "</think>";
    for cut in 0..=close.len() {
        let first = format!("<think>reason{}", &close[..cut]);
        let second = format!("{}visible", &close[cut..]);
        let (visible, reasoning) = run_stream(&cfg, &[&first, &second]);
        assert_eq!(visible, "visible", "cut at {cut}");
        assert_eq!(reasoning, "reason", "cut at {cut}");
    }
}

#[test]
fn lone_angle_bracket_is_not_held_forever() {
    let cfg = ReasoningTagExtraction::default();
    // A `<` that turns out not to open a tag is released once disproven.
    let (visible, reasoning) = run_stream(&cfg, &["a < b < c"]);
    assert_eq!(visible, "a < b < c");
    assert_eq!(reasoning, "");

    // `<` held at a delta boundary, then disproven by the next delta.
    let (visible, reasoning) = run_stream(&cfg, &["value <", "= 3"]);
    assert_eq!(visible, "value <= 3");
    assert_eq!(reasoning, "");
}

#[test]
fn thinker_lookalike_tag_is_not_extracted() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = run_stream(&cfg, &["use <thinker> here"]);
    assert_eq!(visible, "use <thinker> here");
    assert_eq!(reasoning, "");

    // Even when split so `<think` is a real prefix mid-stream.
    let (visible, reasoning) = run_stream(&cfg, &["use <think", "er> here"]);
    assert_eq!(visible, "use <thinker> here");
    assert_eq!(reasoning, "");
}

#[test]
fn think_section_spanning_many_deltas() {
    let cfg = ReasoningTagExtraction::default();
    let deltas = [
        "Answer: ",
        "<think>",
        "step one, ",
        "step two, ",
        "step three",
        "</think>",
        "42",
    ];
    let (visible, reasoning) = run_stream(&cfg, &deltas);
    assert_eq!(visible, "Answer: 42");
    assert_eq!(reasoning, "step one, step two, step three");
}

#[test]
fn text_after_think_is_visible() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = run_stream(&cfg, &["<think>hmm</think>", "the final answer"]);
    assert_eq!(visible, "the final answer");
    assert_eq!(reasoning, "hmm");
}

#[test]
fn start_with_reasoning_mode_stream() {
    // DeepSeek-R1 template: output begins mid-reasoning, only a closing tag.
    let cfg = ReasoningTagExtraction::default().with_start_with_reasoning(true);
    let (visible, reasoning) = run_stream(&cfg, &["chain of ", "thought</think>", "final answer"]);
    assert_eq!(visible, "final answer");
    assert_eq!(reasoning, "chain of thought");
}

#[test]
fn start_with_reasoning_without_closing_tag_is_all_reasoning() {
    let cfg = ReasoningTagExtraction::default().with_start_with_reasoning(true);
    let (visible, reasoning) = run_stream(&cfg, &["still ", "thinking"]);
    assert_eq!(visible, "");
    assert_eq!(reasoning, "still thinking");
}

#[test]
fn unclosed_think_streams_remainder_as_reasoning() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = run_stream(&cfg, &["ok <think>never closed"]);
    assert_eq!(visible, "ok ");
    assert_eq!(reasoning, "never closed");
}

#[test]
fn non_streaming_extraction_strips_surrounding_whitespace() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) =
        extract_reasoning(&cfg, "<think>deliberate</think>\n\nThe answer is 42.");
    assert_eq!(visible, "The answer is 42.");
    assert_eq!(reasoning, "deliberate");
}

#[test]
fn non_streaming_plain_text_is_preserved_verbatim() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = extract_reasoning(&cfg, "line one\nline two");
    assert_eq!(visible, "line one\nline two");
    assert_eq!(reasoning, "");
}

#[test]
fn non_streaming_multiple_sections_join_reasoning_with_separator() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = extract_reasoning(&cfg, "a<think>r1</think>b<think>r2</think>c");
    // Visible segments concatenate (consistent with the streamed deltas);
    // reasoning sections join with the separator.
    assert_eq!(visible, "abc");
    assert_eq!(reasoning, "r1\nr2");
}

#[test]
fn non_streaming_start_with_reasoning() {
    let cfg = ReasoningTagExtraction::default().with_start_with_reasoning(true);
    let (visible, reasoning) = extract_reasoning(&cfg, "reasoning here</think>\n\nvisible answer");
    assert_eq!(visible, "visible answer");
    assert_eq!(reasoning, "reasoning here");
}

/// Captured live from `exaone-deep:2.4b` through Ollama's OpenAI-compatible
/// `/v1/chat/completions` (2026-09-10): the model inlines its entire chain of
/// thought in `<thought>…</thought>` and sends **no** `reasoning` /
/// `reasoning_content` side channel at all. In the real capture the visible
/// answer was 11 bytes (`\boxed{4}`) after 4990 bytes of reasoning.
#[test]
fn default_extracts_thought_tag_from_live_exaone_capture() {
    let cfg = ReasoningTagExtraction::default();
    let content = "<thought>\nOkay, the user is asking me what 2 plus 2 is. Addition combines two\n\
             quantities, so 2 plus 2 means adding them together. That gives 4.\n\
             </thought>\n\n\\boxed{4}";
    let (visible, reasoning) = extract_reasoning(&cfg, content);
    assert_eq!(visible, "\\boxed{4}");
    assert!(
        reasoning.contains("Addition combines two"),
        "chain of thought must land on the reasoning channel, got {reasoning:?}"
    );
    assert!(
        !visible.contains("Okay, the user is asking"),
        "chain of thought leaked into the visible answer: {visible:?}"
    );
}

/// The same capture, streamed. The opening tag really does arrive split as
/// three deltas (`<`, `thought`, `>`), so this also pins the partial-tag
/// buffering across a multi-candidate tag set.
#[test]
fn default_extracts_thought_tag_split_across_deltas() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = run_stream(
        &cfg,
        &[
            "<",
            "thought",
            ">",
            "\n",
            "Okay",
            ", 2 plus 2 is 4.",
            "</thought>",
            "\n\n",
            "4",
        ],
    );
    // Streamed deltas are emitted verbatim; only the terminal response
    // (recomputed through `extract_reasoning`) trims tag-adjacent space.
    assert_eq!(visible, "\n\n4");
    assert_eq!(reasoning, "\nOkay, 2 plus 2 is 4.");
}

/// The conventions the default is meant to cover, each opened and closed by
/// its own name.
#[test]
fn default_covers_common_reasoning_tag_names() {
    let cfg = ReasoningTagExtraction::default();
    for tag in ["think", "thinking", "thought", "reasoning"] {
        let content = format!("<{tag}>hidden</{tag}>answer");
        let (visible, reasoning) = extract_reasoning(&cfg, &content);
        assert_eq!(visible, "answer", "tag {tag}");
        assert_eq!(reasoning, "hidden", "tag {tag}");

        let (visible, reasoning) = run_stream(&cfg, &[&format!("<{tag}>hidden</{tag}>"), "answer"]);
        assert_eq!(visible, "answer", "streamed tag {tag}");
        assert_eq!(reasoning, "hidden", "streamed tag {tag}");
    }
}

/// A tag outside the set is ordinary text — widening must not turn every
/// angle-bracketed word into reasoning.
#[test]
fn tag_names_outside_the_default_set_stay_visible() {
    let cfg = ReasoningTagExtraction::default();
    let (visible, reasoning) = extract_reasoning(&cfg, "<answer>42</answer>");
    assert_eq!(visible, "<answer>42</answer>");
    assert_eq!(reasoning, "");

    let (visible, reasoning) = run_stream(&cfg, &["<ans", "wer>42</answer>"]);
    assert_eq!(visible, "<answer>42</answer>");
    assert_eq!(reasoning, "");
}

/// Near-misses must be released, not swallowed. `<thin ` is a live prefix of
/// `<think>` *and* `<thinking>`; once disproven both must let it go, and a
/// prefix of the longest candidate must not strand the shorter ones.
#[test]
fn near_miss_prefixes_are_released_not_swallowed() {
    let cfg = ReasoningTagExtraction::default();
    for text in [
        "<thin ", "<thinke", "<thought", "<reason ", "a < b", "<think",
    ] {
        let (visible, reasoning) = run_stream(&cfg, &[text]);
        assert_eq!(visible, text, "prose {text:?} must survive verbatim");
        assert_eq!(reasoning, "", "prose {text:?} produced reasoning");
    }

    // Held at a delta boundary, then disproven by the next delta.
    let (visible, reasoning) = run_stream(&cfg, &["value <", "think about it"]);
    assert_eq!(visible, "value <think about it");
    assert_eq!(reasoning, "");

    // `<thinking>` must not be closed by `</think>`: the opener decides the
    // closer, so the mismatched close is ordinary reasoning text.
    let (visible, reasoning) = extract_reasoning(&cfg, "<thinking>a</think>b</thinking>tail");
    assert_eq!(visible, "tail");
    assert_eq!(reasoning, "a</think>b");
}

#[test]
fn custom_tag_name_is_honored() {
    let cfg = ReasoningTagExtraction::new("reasoning");
    let (visible, reasoning) = run_stream(&cfg, &["a<reasoning>", "b</reasoning>c"]);
    assert_eq!(visible, "ac");
    assert_eq!(reasoning, "b");
}
