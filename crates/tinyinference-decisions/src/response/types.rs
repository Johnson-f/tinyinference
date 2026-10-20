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
    /// Cached input tokens, when the provider reports them.
    #[serde(default, alias = "cached_tokens")]
    pub cached_input_tokens: Option<u64>,
    /// Provider-reported USD charge in integer micro-dollars. The provider's
    /// decimal `cost` is rounded to the nearest micro-dollar; invalid or
    /// unreported costs stay absent.
    #[serde(default, rename = "cost", deserialize_with = "deserialize_usd_micros")]
    pub cost_usd_micros: Option<i64>,
}

fn deserialize_usd_micros<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value.and_then(|value| value.as_number()?.as_f64()) else {
        return Ok(None);
    };
    if !value.is_finite() || value < 0.0 || value > (i64::MAX as f64 / 1_000_000.0) {
        return Ok(None);
    }
    Ok(Some((value * 1_000_000.0).round() as i64))
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
