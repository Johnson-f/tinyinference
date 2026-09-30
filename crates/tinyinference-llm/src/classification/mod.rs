//! Provider-neutral classification helpers.

pub mod emoji;

pub use emoji::{extract_first_emoji, is_emoji_start, is_regional_indicator};
