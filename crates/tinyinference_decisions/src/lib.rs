//! Typed Rust access to `TypeSafe` AI's System One API and Jev model.
//!
//! A request supplies text or structured state plus independent [`Question`]s.
//! Jev returns typed choices, ordinal scores, and yes/no probabilities for code
//! to compose. This crate validates both sides of that wire contract and owns
//! only the HTTP wait; policy, thresholds, and actions stay with the caller.
//!
//! # Example
//!
//! ```no_run
//! use std::collections::BTreeMap;
//! use serde_json::json;
//! use tinyinference_decisions::{Choice, Client, EvaluationRequest, Question};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let criteria = BTreeMap::from([
//!     ("billing".to_owned(), Some(json!("payments and refunds"))),
//!     ("technical".to_owned(), Some(json!("bugs and outages"))),
//! ]);
//! let request = EvaluationRequest::jev(
//!     json!({"ticket": "I was charged twice"}),
//!     BTreeMap::from([(
//!         "route".to_owned(),
//!         Question::Choice(Choice {
//!             instructions: json!("Which team should handle this ticket?"),
//!             criteria,
//!         }),
//!     )]),
//! );
//! let result = Client::from_env()?.evaluate(&request).await?;
//! println!("{:?}", result.response.answers["route"]);
//! # Ok(())
//! # }
//! ```
//!
//! The crate deliberately does not execute a selected action, infer permission
//! from confidence, or hide a retry behind an unbounded loop.

mod client;
mod error;
mod request;
mod response;

pub use client::{
    Client, ClientConfig, EvaluationFailure, EvaluationResult, Provider, RetryPolicy,
};
pub use error::{Error, Result};
pub use request::{Choice, EvaluationRequest, Noul, NoulCriteria, Question, Score};
pub use response::{Answer, ChoiceAnswer, EvaluationResponse, NoulAnswer, ScoreAnswer, Usage};
