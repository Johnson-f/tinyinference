//! Public surface smoke tests.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::future::Future;

use serde_json::json;
use tinyinference_decisions::{Choice, Client, EvaluationRequest, EvaluationResult, Question};

#[allow(dead_code)]
fn evaluate_returns_crate_result<'a>(
    client: &'a Client,
    request: &'a EvaluationRequest,
) -> impl Future<Output = tinyinference_decisions::Result<EvaluationResult>> + 'a {
    client.evaluate(request)
}

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
