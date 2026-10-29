//! Provider model-catalog types and response parsing.

mod fetch;
mod listing;
mod parsing;
mod types;

use crate::providers::openai::codex::OPENAI_CODEX_MODEL_HINTS;

pub use fetch::{ListModelsError, list_models, validate_openrouter_key};
pub use listing::{append_query_param, endpoint_hosts_models_listing, is_openrouter_provider};
pub use parsing::{merge_openai_codex_model_hints, parse_models_response};
pub use types::ModelInfo;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

#[cfg(test)]
#[path = "fetch_tests.rs"]
mod fetch_tests;
