//! Read-only access to the user's Ollama / OpenAI-compatible endpoint:
//! health probes, installed-model listing, and diagnostics.
//!
//! Nothing here pulls a model or starts, stops, or locates a runtime binary.
mod diagnostics;
mod health;
mod util;

pub(in crate::service) use health::OllamaHealthStatus;
// Re-export free functions that form the public/crate API of this module.
pub use util::test_ollama_connection;

#[cfg(test)]
#[path = "../ollama_admin_tests.rs"]
mod tests;
