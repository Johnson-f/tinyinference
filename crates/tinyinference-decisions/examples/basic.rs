//! Evaluate one ticket when `TYPESAFE_API_KEY` is configured.

use std::collections::BTreeMap;

use serde_json::json;
use tinyinference_decisions::{Choice, Client, EvaluationRequest, Question};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = EvaluationRequest::jev(
        json!({"ticket": "I was charged twice. Please fix this."}),
        BTreeMap::from([(
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("Which team should handle this ticket?"),
                criteria: BTreeMap::from([
                    ("billing".to_owned(), Some(json!("payments and refunds"))),
                    ("technical".to_owned(), Some(json!("bugs and outages"))),
                    ("other".to_owned(), None),
                ]),
            }),
        )]),
    );
    let result = Client::from_env()?.evaluate(&request).await?;
    let answer = result.response.answers.get("route").ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "response omitted route answer",
        )
    })?;
    println!("{answer:?}");
    Ok(())
}
