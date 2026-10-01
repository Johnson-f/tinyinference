//! Streaming-safe extraction of inline `<think>…</think>`-style reasoning tags.
//!
//! Reasoning models served through OpenAI-compatible local runtimes (qwen3 and
//! deepseek-r1 distills via Ollama `/v1`, LM Studio, llama.cpp) frequently emit
//! their chain-of-thought **inline** in the normal `content` string, wrapped in
//! a reasoning tag, instead of on the `reasoning_content` / `reasoning`
//! side-channel the adapter already normalizes (see
//! [`reasoning_value_text`](super::reasoning_value_text)). Left untouched, that
//! chain-of-thought leaks straight into the visible assistant text.
//!
//! The tag name is not standardized and the runtime does not report it: qwen3
//! and deepseek-r1 use `<think>`, EXAONE Deep uses `<thought>`, and others use
//! `<thinking>` or `<reasoning>`. Since nothing downstream re-inspects the text,
//! a name this module does not match is unrecoverable — so the default accepts
//! the common set (see [`ReasoningTagExtraction::default`]) rather than one
//! name. A section is closed by the closing tag of the name that opened it, so
//! the wider set cannot make one convention terminate another.
//!
//! This module moves the tagged text onto the reasoning channel
//! ([`ContentBlock::Thinking`](crate::message::ContentBlock::Thinking))
//! and strips the tags from the visible text. It provides two entry points that
//! share the same tag-matching logic so the streamed deltas and the final
//! response agree on what is reasoning and what is visible:
//!
//! * [`ReasoningTagStream`] — an incremental state machine for the SSE path. It
//!   buffers and holds back any trailing bytes that *could* be the prefix of an
//!   opening or closing tag ([`potential_start_index`], the Vercel AI SDK's
//!   `getPotentialStartIndex` trick) so a tag split across deltas is neither
//!   leaked as visible text nor mangled.
//! * [`extract_reasoning`] — a whole-string extractor for the non-streaming
//!   path (and the authoritative terminal streamed response).
//!
//! # Side-channel interaction
//!
//! Inline-tag reasoning and side-channel reasoning both feed the single leading
//! `Thinking` block. When a response carries both, side-channel reasoning leads
//! and inline-extracted reasoning follows, joined by the configured separator.
//! In practice a given model uses one convention or the other, not both.

/// Options controlling inline `<tag>…</tag>` reasoning extraction on the
/// OpenAI-compatible provider. Construct with [`ReasoningTagExtraction::new`] or
/// [`ReasoningTagExtraction::default`] (the `think` tag) and pass to
/// [`OpenAiModel::with_reasoning_tag_extraction`](super::OpenAiModel::with_reasoning_tag_extraction).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningTagExtraction {
    /// Tag names without angle brackets, e.g. `think` → `<think>` / `</think>`.
    /// Any one of them opens a reasoning section; the section is closed by the
    /// closing tag of *the name that opened it*, so `<thinking>a</think>b`
    /// stays open until `</thinking>`. Models inline their chain of thought
    /// under several conventions (`<think>` for qwen3 / deepseek-r1,
    /// `<thought>` for EXAONE Deep) and the runtime does not tell us which, so
    /// the default accepts the common set rather than one name.
    tag_names: Vec<String>,
    /// Separator inserted between multiple extracted reasoning sections (and
    /// between side-channel and inline reasoning). Defaults to a newline.
    separator: String,
    /// When `true`, the output BEGINS mid-reasoning with no opening tag and only
    /// a closing `</tag>` appears — the DeepSeek-R1 template mode (the AI SDK's
    /// `startWithReasoning`). Everything before the first closing tag is
    /// reasoning. Defaults to `false`.
    start_with_reasoning: bool,
}

impl Default for ReasoningTagExtraction {
    fn default() -> Self {
        Self {
            tag_names: ["think", "thinking", "thought", "reasoning"]
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            separator: "\n".to_string(),
            start_with_reasoning: false,
        }
    }
}

impl ReasoningTagExtraction {
    /// Extraction for a custom tag name (no angle brackets), newline separator,
    /// opening-tag-gated (not DeepSeek mode).
    pub fn new(tag_name: impl Into<String>) -> Self {
        Self::for_tags([tag_name])
    }

    /// Extraction for an explicit set of tag names (no angle brackets). Any of
    /// them opens a section; each is closed by its own closing tag.
    pub fn for_tags<I, T>(tag_names: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            tag_names: tag_names.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Overrides the separator joining multiple reasoning sections.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Enables DeepSeek-R1 template mode: the stream begins mid-reasoning with no
    /// opening tag, and only a closing `</tag>` marks the end of reasoning.
    pub fn with_start_with_reasoning(mut self, enabled: bool) -> Self {
        self.start_with_reasoning = enabled;
        self
    }

    /// The literal opening tags, e.g. `<think>`, positionally aligned with
    /// [`Self::closing_tags`].
    fn opening_tags(&self) -> Vec<String> {
        self.tag_names
            .iter()
            .map(|name| format!("<{name}>"))
            .collect()
    }

    /// The literal closing tags, e.g. `</think>`, positionally aligned with
    /// [`Self::opening_tags`].
    fn closing_tags(&self) -> Vec<String> {
        self.tag_names
            .iter()
            .map(|name| format!("</{name}>"))
            .collect()
    }

    /// The separator joining reasoning sections.
    pub(super) fn separator(&self) -> &str {
        &self.separator
    }
}

/// Finds where `searched` begins within `text`, as either a full occurrence or a
/// partial prefix at the very end of `text`.
///
/// Returns the byte offset of the earliest position at which `searched` could
/// start:
/// * `Some(idx)` where `text[idx..]` fully contains `searched` (a complete tag),
///   or where `text[idx..]` is a non-empty proper prefix of `searched` (a
///   partial tag straddling the end that must be held back for more input);
/// * `None` when no suffix of `text` could begin `searched` — the whole `text`
///   is safe to release.
///
/// This is the Rust port of the Vercel AI SDK's `getPotentialStartIndex`. It
/// only ever returns positions on UTF-8 character boundaries (the tags are
/// ASCII, and it scans `char_indices`), so callers may slice `text` at the
/// result safely.
pub(super) fn potential_start_index(text: &str, searched: &str) -> Option<usize> {
    if searched.is_empty() {
        return None;
    }
    if let Some(idx) = text.find(searched) {
        return Some(idx);
    }
    // No full occurrence: look for a suffix of `text` that is a prefix of
    // `searched`, scanning shortest-suffix-first (end inward) to match the AI
    // SDK. Iterate char-boundary starts so any returned index is sliceable.
    let starts: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    for &i in starts.iter().rev() {
        let suffix = &text[i..];
        if searched.starts_with(suffix) {
            return Some(i);
        }
    }
    None
}

/// Earliest position in `text` at which any of `tags` could begin.
///
/// Returns `(idx, Some((i, len)))` when `tags[i]` occurs in full at `idx`, and
/// `(idx, None)` when `text` ends in a partial tag that is still undecided and
/// must be held back for more input. `None` when no suffix of `text` could
/// begin any tag — the whole `text` is safe to release.
///
/// Two distinct tags can never match completely at the same index: every tag is
/// `<name>` or `</name>`, and the terminating `>` means no tag literal is a
/// prefix of another (`<think>` does not prefix `<thinking>`). A complete match
/// therefore wins over an undecided partial at the same index without ambiguity.
fn first_tag_match(text: &str, tags: &[String]) -> Option<(usize, Option<(usize, usize)>)> {
    let mut best: Option<(usize, Option<(usize, usize)>)> = None;
    for (i, tag) in tags.iter().enumerate() {
        let Some(idx) = potential_start_index(text, tag) else {
            continue;
        };
        let resolved = (idx + tag.len() <= text.len()).then_some((i, tag.len()));
        let better = match best {
            None => true,
            // Earliest candidate wins; on a tie a resolved tag beats a partial.
            Some((best_idx, best_resolved)) => {
                idx < best_idx || (idx == best_idx && best_resolved.is_none() && resolved.is_some())
            }
        };
        if better {
            best = Some((idx, resolved));
        }
    }
    best
}

/// Incremental extractor for the streamed content path.
///
/// Feed each `content` delta to [`push`](Self::push); it appends any resolved
/// visible / reasoning text to the supplied buffers and retains any trailing
/// partial-tag bytes internally until the next delta (or [`finish`](Self::finish)
/// at end of stream) resolves them.
#[derive(Clone, Debug)]
pub(super) struct ReasoningTagStream {
    openings: Vec<String>,
    closings: Vec<String>,
    /// Bytes received but not yet released (may end in a partial tag).
    buffer: String,
    /// The closing tags that can end the current reasoning section — the
    /// matching opener's own tag, or every tag in `start_with_reasoning` mode
    /// where no opener introduced the section. `None` while scanning visible
    /// text, so this doubles as "am I inside reasoning".
    awaiting_close: Option<Vec<String>>,
}

impl ReasoningTagStream {
    /// Builds a stream extractor from the configured tags. In DeepSeek mode the
    /// machine starts already inside a reasoning section.
    pub(super) fn new(config: &ReasoningTagExtraction) -> Self {
        let closings = config.closing_tags();
        Self {
            openings: config.opening_tags(),
            // No opener ran, so any closing tag ends the section.
            awaiting_close: config.start_with_reasoning.then(|| closings.clone()),
            closings,
            buffer: String::new(),
        }
    }

    /// Folds one content delta into the machine, appending resolved text to
    /// `visible` / `reasoning`. A trailing partial tag is held in `buffer`.
    pub(super) fn push(&mut self, delta: &str, visible: &mut String, reasoning: &mut String) {
        self.buffer.push_str(delta);
        loop {
            // Scoped so the borrow of the tag list ends before `self` is
            // mutated below; the result is all `Copy`.
            let found = {
                let tags: &[String] = match self.awaiting_close.as_ref() {
                    Some(closings) => closings,
                    None => &self.openings,
                };
                first_tag_match(&self.buffer, tags)
            };
            match found {
                // Neither a complete tag nor a partial-tag suffix: everything
                // buffered is safe to release.
                None => {
                    let released = std::mem::take(&mut self.buffer);
                    self.emit(&released, visible, reasoning);
                    break;
                }
                Some((idx, resolved)) => {
                    // Text before the (partial or full) tag belongs to the
                    // current channel. Emitted before the state flips below, so
                    // it lands on the channel that was open when it arrived.
                    let before = self.buffer[..idx].to_string();
                    self.emit(&before, visible, reasoning);
                    match resolved {
                        // Complete tag: drop it, flip channel, keep scanning the
                        // remainder. Entering reasoning pins the closer to the
                        // tag that opened it, so `<thinking>a</think>` stays open.
                        Some((i, len)) => {
                            self.buffer = self.buffer[idx + len..].to_string();
                            self.awaiting_close = match self.awaiting_close {
                                Some(_) => None,
                                None => Some(vec![self.closings[i].clone()]),
                            };
                        }
                        // Partial tag at the buffer tail: hold it for more input.
                        None => {
                            self.buffer = self.buffer[idx..].to_string();
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Releases any buffered partial-tag tail at end of stream into the current
    /// channel — a partial tag that never completed is real content.
    ///
    /// The streaming provider path does not call this: it recomputes the
    /// authoritative split from the raw accumulated content in
    /// [`into_response`](super::OpenAiStreamAcc), which also applies the
    /// non-streaming trimming so the terminal response matches exactly. This is
    /// the state machine's flush contract, exercised by the unit tests.
    #[cfg(test)]
    pub(super) fn finish(&mut self, visible: &mut String, reasoning: &mut String) {
        let released = std::mem::take(&mut self.buffer);
        self.emit(&released, visible, reasoning);
    }

    fn emit(&self, text: &str, visible: &mut String, reasoning: &mut String) {
        if text.is_empty() {
            return;
        }
        if self.awaiting_close.is_some() {
            reasoning.push_str(text);
        } else {
            visible.push_str(text);
        }
    }
}

/// Extracts inline reasoning from a complete `content` string.
///
/// Returns `(visible_text, reasoning_text)`. Reasoning sections are trimmed and
/// joined with the configured separator; the visible text has the tagged
/// sections (and the whitespace that surrounded them) removed. When the content
/// contains no reasoning at all, it is returned verbatim as the visible text so
/// plain responses are preserved byte-for-byte.
pub(super) fn extract_reasoning(
    config: &ReasoningTagExtraction,
    content: &str,
) -> (String, String) {
    let openings = config.opening_tags();
    let closings = config.closing_tags();
    // Mirrors `ReasoningTagStream::awaiting_close`: `Some` means inside a
    // reasoning section, holding the closing tags that can end it.
    let mut awaiting_close: Option<Vec<String>> =
        config.start_with_reasoning.then(|| closings.clone());
    let mut rest = content;
    let mut visible_parts: Vec<&str> = Vec::new();
    let mut reasoning_parts: Vec<&str> = Vec::new();

    loop {
        let tags: &[String] = match awaiting_close.as_ref() {
            Some(closing) => closing,
            None => &openings,
        };
        // Earliest full occurrence of any candidate. No partial handling here:
        // the whole string is already in hand.
        let hit = tags
            .iter()
            .enumerate()
            .filter_map(|(i, tag)| rest.find(tag.as_str()).map(|idx| (idx, i, tag.len())))
            .min_by_key(|&(idx, _, _)| idx);
        match hit {
            Some((idx, i, len)) => {
                let (before, after) = rest.split_at(idx);
                if awaiting_close.is_some() {
                    reasoning_parts.push(before);
                } else {
                    visible_parts.push(before);
                }
                rest = &after[len..];
                awaiting_close = match awaiting_close {
                    Some(_) => None,
                    None => Some(vec![closings[i].clone()]),
                };
            }
            None => {
                if awaiting_close.is_some() {
                    reasoning_parts.push(rest);
                } else {
                    visible_parts.push(rest);
                }
                break;
            }
        }
    }

    // No reasoning was ever entered: return the content untouched so plain text
    // is preserved exactly (no whitespace normalization).
    if !config.start_with_reasoning && reasoning_parts.is_empty() {
        return (content.to_string(), String::new());
    }

    // Visible text: concatenate the surviving segments (matching the streaming
    // deltas, which carry no separator) and trim the whitespace that bordered
    // the removed sections at the ends.
    let visible = visible_parts.concat().trim().to_string();
    // Reasoning: join distinct sections with the configured separator, trimming
    // each so tag-adjacent whitespace does not bloat the thinking block.
    let reasoning = reasoning_parts
        .iter()
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(&config.separator);

    (visible, reasoning)
}

#[cfg(test)]
#[path = "reasoning_tags_tests.rs"]
mod tests;
