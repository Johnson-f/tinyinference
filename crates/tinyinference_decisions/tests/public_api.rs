//! Public surface smoke tests.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;

use serde_json::json;
use tinyinference_decisions::{Choice, EvaluationRequest, Question};

#[test]
fn public_types_build_a_valid_jev_request() {
    let request = EvaluationRequest::jev(
        "route this",
        BTreeMap::from([(
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("Who should handle this?"),
                criteria: BTreeMap::from([
                    ("planner".to_owned(), None),
                    ("reviewer".to_owned(), None),
                ]),
            }),
        )]),
    );
    request.validate().expect("public request should validate");
}
