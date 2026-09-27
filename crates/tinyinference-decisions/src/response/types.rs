//! Stable response payload definitions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One decoded System One response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationResponse {
    /// Model that performed the evaluation.
    pub model: String,
    /// Answers keyed by the caller's question ids.
    pub answers: BTreeMap<String, Answer>,
    /// Provider-reported token counts.
    pub usage: Usage,
}

/// Provider-reported token usage.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Usage {
    /// Input tokens, when reported.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Output tokens, when reported.
    #[serde(default)]
    pub output_tokens: Option<u64>,
}

/// One typed answer.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    /// Closed-set selection and distribution.
    Choice(ChoiceAnswer),
    /// Ordered score and level distribution.
    Score(ScoreAnswer),
    /// Probability of yes.
    Noul(NoulAnswer),
}

/// Answer to a Choice question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChoiceAnswer {
    /// Highest-probability option.
    pub choice: String,
    /// Probability for every requested option.
    pub probabilities: BTreeMap<String, f64>,
    /// Concentration of the distribution, not correctness probability.
    pub confidence: f64,
}

/// Answer to a Score question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScoreAnswer {
    /// Probability-weighted position along the scale.
    pub score: f64,
    /// Requested level descriptions keyed by zero-based index.
    pub legend: BTreeMap<String, Value>,
    /// Probability for every level.
    pub probabilities: BTreeMap<String, f64>,
    /// Concentration of the distribution, not correctness probability.
    pub confidence: f64,
}

/// Answer to a Noul question.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoulAnswer {
    /// Probability that the requested condition is true.
    pub noul: f64,
}
