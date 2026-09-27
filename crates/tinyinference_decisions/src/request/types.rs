//! Stable request payload definitions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One complete System One evaluation request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationRequest {
    /// Unstructured text or structured application state visible to every question.
    pub state: Value,
    /// System One model identifier, normally `jev-latest`.
    pub model: String,
    /// Independently evaluated questions keyed by caller-owned ids.
    pub questions: BTreeMap<String, Question>,
}

impl EvaluationRequest {
    /// Build a request using the stable Jev alias.
    #[must_use]
    pub fn jev(state: impl Into<Value>, questions: BTreeMap<String, Question>) -> Self {
        Self {
            state: state.into(),
            model: "jev-latest".to_owned(),
            questions,
        }
    }
}

/// A typed question evaluated independently against shared state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// Select exactly one option from a closed set.
    Choice(Choice),
    /// Place the state along an ordered descriptive scale.
    Score(Score),
    /// Estimate the probability that a yes/no condition holds.
    Noul(Noul),
}

/// A closed-set selection question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Choice {
    /// Complete meaning of the decision to make.
    pub instructions: Value,
    /// Option name to optional distinguishing description.
    pub criteria: BTreeMap<String, Option<Value>>,
}

/// An ordered-scale question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Score {
    /// Dimension being rated.
    pub instructions: Value,
    /// Ordered, standalone level descriptions.
    pub criteria: Vec<Value>,
}

/// A yes/no probability question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Noul {
    /// Condition whose probability of being true is requested.
    pub instructions: Value,
    /// Optional descriptions clarifying both outcomes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// Descriptions of the true and false outcomes of a [`Noul`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoulCriteria {
    /// What counts as true.
    pub r#true: Value,
    /// What counts as false.
    pub r#false: Value,
}
