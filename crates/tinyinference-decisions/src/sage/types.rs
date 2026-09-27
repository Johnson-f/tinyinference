//! Public Sage decision request and result types.

use serde::{Deserialize, Serialize};

/// Content judged by Sage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionContent {
    /// Plain document text.
    Text(String),
    /// An explicitly typed document, image, or sortable list.
    Structured(StructuredContent),
}

impl From<String> for DecisionContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for DecisionContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

/// Structured Sage content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StructuredContent {
    /// Explicitly typed text.
    Text {
        /// Text to judge.
        value: String,
    },
    /// A base64 `data:` URI image, optionally accompanied by text.
    Image {
        /// PNG, JPEG, or WebP base64 data URI.
        media: String,
        /// Context accompanying the image.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// Items to rank with a Sort question.
    List {
        /// Items in their original order.
        value: Vec<ListItem>,
    },
}

/// One item in a sortable list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListItem {
    /// Stable item identifier returned in the sorted result.
    pub id: String,
    /// Item text.
    pub content: String,
}

/// A question Sage can answer about content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionQuestion {
    /// A yes, no, or uncertain decision.
    #[serde(rename = "yesno")]
    YesNo {
        /// Stable question identifier.
        id: String,
        /// Question and decision rules.
        instructions: String,
    },
    /// Select one option, or return uncertain.
    Choice {
        /// Stable question identifier.
        id: String,
        /// Question and decision rules.
        instructions: String,
        /// Candidate options.
        options: Vec<ChoiceOption>,
    },
    /// Score a fixed five-level rubric.
    Scale {
        /// Stable question identifier.
        id: String,
        /// Question and decision rules.
        instructions: String,
        /// Exactly one level for each integer from zero through four.
        levels: Vec<ScaleLevel>,
    },
    /// Rank list items.
    Sort {
        /// Stable question identifier.
        id: String,
        /// Ranking criterion.
        instructions: String,
    },
    /// Judge each tag independently.
    Tags {
        /// Stable question identifier.
        id: String,
        /// Optional overall instructions.
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        /// Labels to judge.
        tags: Vec<TagSpec>,
    },
}

impl DecisionQuestion {
    /// Returns the stable question identifier.
    pub fn id(&self) -> &str {
        match self {
            Self::YesNo { id, .. }
            | Self::Choice { id, .. }
            | Self::Scale { id, .. }
            | Self::Sort { id, .. }
            | Self::Tags { id, .. } => id,
        }
    }
}

/// An option offered to a Choice question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    /// Unique option identifier.
    pub option: String,
    /// Optional description of when the option applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// One level in a five-level Scale rubric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleLevel {
    /// Rubric level, from zero through four.
    pub level: u8,
    /// Meaning of this level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A label offered to a Tags question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TagSpec {
    /// Stable tag identifier.
    pub id: String,
    /// Optional display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// When Sage should perform a reasoning pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Reason when Sage's first pass requests it.
    #[default]
    Auto,
    /// Use only the first pass.
    Off,
    /// Always perform the reasoning pass.
    On,
}

/// Accuracy and latency trade for Choice questions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyMode {
    /// Score each option in its own pass.
    #[default]
    Quality,
    /// Use the faster single-pass score.
    Fast,
}

/// When optional web grounding should run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundingTrigger {
    /// Search when the first pass is below the configured confidence floor.
    #[default]
    LowConfidence,
    /// Always search.
    Always,
    /// Do not search.
    Never,
}

/// Optional web grounding controls.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroundingConfig {
    /// Search trigger; Sage defaults to low confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<GroundingTrigger>,
    /// Decision threshold from zero through one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_floor: Option<f64>,
    /// Maximum search results, from one through twenty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u8>,
    /// Maximum added context tokens, from one through 8000.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u16>,
    /// Whether search sources should be returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_sources: Option<bool>,
}

/// One `/decide` call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Document, image, or list to judge.
    pub content: DecisionContent,
    /// Question about the content.
    pub question: DecisionQuestion,
    /// Optional search controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<GroundingConfig>,
    /// Reasoning policy; omitted to use Sage's default.
    #[serde(default, skip_serializing_if = "is_auto")]
    pub reasoning: ReasoningMode,
    /// Choice accuracy and latency policy; omitted to use Sage's default.
    #[serde(default, skip_serializing_if = "is_quality")]
    pub latency_mode: LatencyMode,
}

impl DecisionRequest {
    /// Creates a decision request with Sage's default reasoning and latency modes.
    pub fn new(content: impl Into<DecisionContent>, question: DecisionQuestion) -> Self {
        Self {
            content: content.into(),
            question,
            grounding: None,
            reasoning: ReasoningMode::Auto,
            latency_mode: LatencyMode::Quality,
        }
    }
}

/// A question in one batch group; grounding belongs to each question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchQuestion {
    /// Decision question.
    #[serde(flatten)]
    pub question: DecisionQuestion,
    /// Optional search controls for this question.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<GroundingConfig>,
}

/// Questions sharing one document or image in a batch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchGroup {
    /// Shared content.
    pub content: DecisionContent,
    /// Questions about the shared content.
    pub questions: Vec<BatchQuestion>,
}

/// One `/decide/batch` call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchDecisionRequest {
    /// Content groups in request order.
    pub requests: Vec<BatchGroup>,
    /// Shared reasoning policy.
    #[serde(default, skip_serializing_if = "is_auto")]
    pub reasoning: ReasoningMode,
    /// Shared Choice accuracy and latency policy.
    #[serde(default, skip_serializing_if = "is_quality")]
    pub latency_mode: LatencyMode,
}

fn is_auto(value: &ReasoningMode) -> bool {
    *value == ReasoningMode::Auto
}

fn is_quality(value: &LatencyMode) -> bool {
    *value == LatencyMode::Quality
}

/// A typed Sage answer, discriminated by its question kind.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionResponse {
    /// Yes/no result.
    #[serde(rename = "yesno")]
    YesNo {
        /// Question identifier.
        id: String,
        /// Verdict and calibrated probability.
        result: YesNoResult,
        /// Per-answer metadata.
        meta: AnswerMeta,
        /// Search details when grounding was requested.
        grounding_meta: Option<GroundingMeta>,
    },
    /// Choice result.
    Choice {
        /// Question identifier.
        id: String,
        /// Chosen option and option scores.
        result: ChoiceResult,
        /// Per-answer metadata.
        meta: AnswerMeta,
        /// Search details when grounding was requested.
        grounding_meta: Option<GroundingMeta>,
    },
    /// Scale result.
    Scale {
        /// Question identifier.
        id: String,
        /// Rubric score.
        result: ScaleResult,
        /// Per-answer metadata.
        meta: AnswerMeta,
        /// Search details when grounding was requested.
        grounding_meta: Option<GroundingMeta>,
    },
    /// Sort result.
    Sort {
        /// Question identifier.
        id: String,
        /// Ranked item ids.
        result: SortResult,
        /// Per-answer metadata.
        meta: AnswerMeta,
        /// Search details when grounding was requested.
        grounding_meta: Option<GroundingMeta>,
    },
    /// Tags result.
    Tags {
        /// Question identifier.
        id: String,
        /// Independent tag verdicts.
        result: TagsResult,
        /// Per-answer metadata.
        meta: AnswerMeta,
        /// Search details when grounding was requested.
        grounding_meta: Option<GroundingMeta>,
    },
}

impl DecisionResponse {
    /// Returns the question identifier echoed by Sage.
    pub fn id(&self) -> &str {
        match self {
            Self::YesNo { id, .. }
            | Self::Choice { id, .. }
            | Self::Scale { id, .. }
            | Self::Sort { id, .. }
            | Self::Tags { id, .. } => id,
        }
    }
}

/// A yes/no verdict; `None` means Sage is unsure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YesNoAnswer {
    /// Yes.
    Yes,
    /// No.
    No,
}

/// A yes/no decision and calibrated probability of yes.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct YesNoResult {
    /// Sage verdict, or `None` when unsure.
    pub answer: Option<YesNoAnswer>,
    /// Calibrated probability of yes, including on uncertain answers.
    pub probability: f64,
}

/// A Choice answer and each option's independent probability.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ChoiceResult {
    /// Chosen option, or `None` when too close to call.
    pub chosen: Option<String>,
    /// Confidence in the chosen option, or `None` when undecided.
    pub probability: Option<f64>,
    /// Option probabilities in request order; they need not sum to one.
    pub probabilities: Vec<OptionProbability>,
}

/// One option's independent probability.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct OptionProbability {
    /// Option identifier.
    pub option: String,
    /// Probability assigned to this option.
    pub probability: f64,
}

/// Expected rubric level and confidence in the distribution.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ScaleResult {
    /// Expected value from zero through four.
    pub expectation: f64,
    /// Confidence in the level distribution.
    pub confidence: f64,
}

/// Sorted item ids and optional confidence.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SortResult {
    /// Item ids in descending rank order.
    pub sorted: Vec<String>,
    /// Ranking confidence, when available.
    pub confidence: Option<f64>,
}

/// Independent tag verdicts.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TagsResult {
    /// Tag answers in request order.
    pub tags: Vec<TagResult>,
}

/// One tag's probability and verdict.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TagResult {
    /// Tag identifier.
    pub id: String,
    /// Probability that the tag applies.
    pub probability: f64,
    /// Verdict; `None` means unsure for text content.
    pub applies: Option<bool>,
}

/// Per-question timing, usage, compute, and reasoning metadata.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct AnswerMeta {
    /// Model identifier reported by Sage.
    pub model: String,
    /// Server-side latency in milliseconds.
    pub latency_ms: Option<f64>,
    /// Number of questions represented, when reported.
    pub question_count: Option<u64>,
    /// Compute path, such as `fanout_candidates` or `packed`.
    pub compute_mode: Option<String>,
    /// Usage attributed to this answer.
    pub usage: Option<DecisionUsage>,
    /// Reasoning-pass accounting, when available.
    pub reasoning: Option<ReasoningMeta>,
}

/// Sage's billed token and image usage.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct DecisionUsage {
    /// Billed input tokens.
    pub billed_input_tokens: u64,
    /// Rendered input tokens, when supplied.
    pub rendered_tokens: Option<u64>,
    /// Images used in this request.
    #[serde(default)]
    pub image_count: u64,
    /// Image tokens consumed.
    #[serde(default)]
    pub image_tokens: u64,
}

/// Sage's optional reasoning-pass accounting.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ReasoningMeta {
    /// Whether the first pass requested reasoning.
    pub fired: bool,
    /// Whether reasoning actually ran.
    pub ran: bool,
    /// Whether it completed before the budget, when run.
    pub finished: Option<bool>,
    /// Generated reasoning tokens; not billed.
    pub tokens: Option<u64>,
    /// First-pass gate margin in nats.
    pub margin: Option<f64>,
    /// Budget limit reason, when reasoning did not complete.
    pub limited: Option<String>,
}

/// Grounding search activity and returned sources.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GroundingMeta {
    /// Whether web search ran.
    pub triggered: bool,
    /// Why search did or did not run.
    pub trigger_reason: Option<String>,
    /// Search queries used.
    #[serde(default)]
    pub queries: Vec<String>,
    /// Returned sources.
    #[serde(default)]
    pub sources: Vec<GroundingSource>,
    /// Search context tokens added.
    pub added_context_tokens: Option<u64>,
    /// Search duration in milliseconds.
    pub search_ms: Option<f64>,
}

/// A source returned by Sage's web grounding.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GroundingSource {
    /// Source URL, when available.
    pub url: Option<String>,
    /// Source title, when available.
    pub title: Option<String>,
    /// Extracted source snippet, when available.
    pub snippet: Option<String>,
}

/// Results for a batch call, aligned to request groups and question order.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BatchDecisionResponse {
    /// Group results in request order.
    pub results: Vec<BatchGroupResult>,
    /// Overall batch accounting.
    pub meta: BatchMeta,
}

/// Answers for one batch group.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BatchGroupResult {
    /// Individual answers in question order.
    pub answers: Vec<BatchAnswer>,
}

/// An individual answer or error inside a successful batch call.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BatchAnswer {
    /// Whether this question succeeded.
    pub ok: bool,
    /// Typed answer if the question succeeded.
    pub result: Option<DecisionResponse>,
    /// Per-question error if it failed.
    pub error: Option<String>,
}

/// Overall batch metadata.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BatchMeta {
    /// Model identifier.
    pub model: String,
    /// Content groups processed.
    pub request_count: u64,
    /// Questions processed.
    pub question_count: u64,
    /// Server-side latency in milliseconds.
    pub latency_ms: Option<f64>,
    /// Combined usage, when available.
    pub usage: Option<DecisionUsage>,
}

/// Dry-run billed-token estimate.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct UsageEstimate {
    /// Minimum estimated billed input tokens.
    pub min_input_tokens: u64,
    /// Maximum estimated billed input tokens.
    pub max_input_tokens: u64,
    /// Whether the estimate is exact.
    pub exact: bool,
}

/// One Sage model in the public `/models` catalog.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct SageModel {
    /// Model identifier.
    pub id: String,
    /// Human-readable model name.
    pub name: String,
    /// Model version, when reported.
    pub version: Option<String>,
    /// Whether this model is ready to serve.
    pub is_ready: Option<bool>,
    /// Supported input modalities.
    #[serde(default)]
    pub input_modalities: Vec<String>,
    /// Maximum context length, when reported.
    pub context_length: Option<u64>,
}
