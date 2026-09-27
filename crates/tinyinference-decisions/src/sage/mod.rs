//! Levanto Sage's typed decision API.
//!
//! Sage evaluates one question with [`SageClient::decide`] or groups of
//! questions with [`SageClient::decide_batch`]. Uncertain verdicts are valid
//! answers represented by `None` in the relevant result field.

mod client;
mod types;

pub use client::SageClient;
pub use types::*;

#[cfg(test)]
mod test;
