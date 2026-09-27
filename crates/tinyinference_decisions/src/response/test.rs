//! Response wire and request-relative validation tests.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use crate::{Choice, EvaluationRequest, Noul, Question, Score};

fn request() -> EvaluationRequest {
    EvaluationRequest::jev(
        "state",
        BTreeMap::from([
            (
                "route".to_owned(),
                Question::Choice(Choice {
                    instructions: json!("route"),
                    criteria: BTreeMap::from([("a".to_owned(), None), ("b".to_owned(), None)]),
                }),
            ),
            (
                "quality".to_owned(),
                Question::Score(Score {
                    instructions: json!("quality"),
                    criteria: vec![json!("low"), json!("high")],
                }),
            ),
            (
                "safe".to_owned(),
                Question::Noul(Noul {
                    instructions: json!("safe"),
                    criteria: None,
                }),
            ),
        ]),
    )
}

fn response() -> EvaluationResponse {
    serde_json::from_value(json!({
        "model": "jev-latest",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.25, "b": 0.75},
                "confidence": 0.5
            },
            "quality": {
                "type": "score",
                "score": 0.8,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.2, "1": 0.8},
                "confidence": 0.6
            },
            "safe": {"type": "noul", "noul": 0.9}
        },
        "usage": {"input_tokens": 42, "output_tokens": 3}
    }))
    .unwrap()
}

#[test]
fn validates_all_three_answer_types() {
    response().validate_for(&request()).unwrap();
    let mut rounded = response();
    let Answer::Score(answer) = rounded.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    answer.score = 0.81;
    rounded.validate_for(&request()).unwrap();
}

#[test]
fn usage_fields_remain_optional() {
    let usage: Usage = serde_json::from_value(json!({})).unwrap();
    assert_eq!(usage, Usage::default());
}

#[test]
fn rejects_missing_extra_or_wrongly_typed_answers() {
    let mut missing = response();
    missing.answers.remove("safe");
    assert!(missing.validate_for(&request()).is_err());

    let mut wrong = response();
    wrong.answers.insert(
        "safe".to_owned(),
        Answer::Choice(ChoiceAnswer {
            choice: "a".to_owned(),
            probabilities: BTreeMap::from([("a".to_owned(), 1.0)]),
            confidence: 1.0,
        }),
    );
    assert!(wrong.validate_for(&request()).is_err());
}

#[test]
fn rejects_invalid_distributions_and_inconsistent_scores() {
    let mut distribution = response();
    let Answer::Choice(choice) = distribution.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.probabilities.insert("a".to_owned(), 0.75);
    assert!(distribution.validate_for(&request()).is_err());

    let mut score = response();
    let Answer::Score(answer) = score.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    answer.score = 0.1;
    assert!(score.validate_for(&request()).is_err());
}

#[test]
fn rejects_empty_model_extra_ids_and_nonmaximal_choice() {
    let mut empty_model = response();
    empty_model.model.clear();
    assert!(empty_model.validate_for(&request()).is_err());

    let mut wrong_model = response();
    wrong_model.model = "jev-other".into();
    assert!(wrong_model.validate_for(&request()).is_err());

    let mut extra = response();
    extra
        .answers
        .insert("extra".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
    assert!(extra.validate_for(&request()).is_err());

    let mut nonmaximal = response();
    let Answer::Choice(choice) = nonmaximal.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.choice = "a".into();
    assert!(nonmaximal.validate_for(&request()).is_err());
}

#[test]
fn openrouter_accepts_resolved_jev_models_only() {
    let mut resolved = response();
    resolved.model = "typesafe/jev-1.13-20260917".into();
    resolved.validate_for_openrouter(&request()).unwrap();

    let mut namespaced_latest = request();
    namespaced_latest.model = "~typesafe/jev-latest".into();
    resolved
        .validate_for_openrouter(&namespaced_latest)
        .unwrap();

    let mut unrelated = resolved;
    unrelated.model = "typesafe/other-1".into();
    assert!(unrelated.validate_for_openrouter(&request()).is_err());
}

#[test]
fn rejects_out_of_range_empty_and_mismatched_probability_payloads() {
    let mut confidence = response();
    let Answer::Choice(choice) = confidence.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.confidence = 1.1;
    assert!(confidence.validate_for(&request()).is_err());

    let mut empty = response();
    let Answer::Choice(choice) = empty.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.probabilities.clear();
    assert!(empty.validate_for(&request()).is_err());

    let mut labels = response();
    let Answer::Choice(choice) = labels.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.probabilities.remove("a");
    choice.probabilities.insert("c".into(), 0.25);
    assert!(labels.validate_for(&request()).is_err());

    let mut noul = response();
    let Answer::Noul(answer) = noul.answers.get_mut("safe").unwrap() else {
        panic!("fixture answer should be a noul")
    };
    answer.noul = f64::NAN;
    assert!(noul.validate_for(&request()).is_err());
}

#[test]
fn rejects_nonfinite_score_and_mismatched_legend() {
    let mut nonfinite = response();
    let Answer::Score(score) = nonfinite.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.score = f64::INFINITY;
    assert!(nonfinite.validate_for(&request()).is_err());

    let mut outside = response();
    let Answer::Score(score) = outside.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.probabilities = BTreeMap::from([("0".into(), 0.98), ("1".into(), 0.02)]);
    score.score = -0.01;
    assert!(outside.validate_for(&request()).is_err());

    let mut legend = response();
    let Answer::Score(score) = legend.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.legend.remove("1");
    assert!(legend.validate_for(&request()).is_err());

    let mut reversed = response();
    let Answer::Score(score) = reversed.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.legend.insert("0".into(), json!("high"));
    score.legend.insert("1".into(), json!("low"));
    assert!(reversed.validate_for(&request()).is_err());
}

#[test]
fn accepts_the_exact_score_rounding_boundary() {
    let mut boundary = response();
    let Answer::Score(score) = boundary.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.probabilities = BTreeMap::from([("0".into(), 0.97), ("1".into(), 0.03)]);
    score.score = 0.05;
    boundary.validate_for(&request()).unwrap();
}

#[test]
fn many_option_distributions_tolerate_per_option_rounding() {
    // 21 options rounded to two decimals summing to 0.99: what the OpenRouter
    // endpoint answers. Rejected before the size-aware tolerance.
    let mut probabilities = std::collections::BTreeMap::new();
    for i in 0..20 {
        probabilities.insert(format!("o{i}"), 0.04);
    }
    probabilities.insert("o20".to_owned(), 0.19);
    assert!((probabilities.values().sum::<f64>() - 0.99).abs() < 1e-9);
    assert!(validate_distribution(&probabilities, "choice").is_ok());

    // Two options are still held tightly.
    let two = std::collections::BTreeMap::from([("a".to_owned(), 0.6), ("b".to_owned(), 0.39)]);
    assert!(validate_distribution(&two, "choice").is_err());
}
