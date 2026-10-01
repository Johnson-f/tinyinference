//! Errors produced by local inference operations.

use thiserror::Error;

/// Result returned by local inference APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized local inference failure.
#[derive(Debug, Error)]
pub enum Error {
    /// A caller supplied an invalid local-runtime identifier or option.
    #[error("invalid local inference input: {0}")]
    InvalidInput(String),
    /// No local vision model was configured for a vision request.
    #[error("no local vision model is configured: {0}")]
    VisionModelNotConfigured(String),
    /// The configured model cannot accept image input.
    #[error("configured model is not vision-capable: {0}")]
    VisionModelUnsupported(String),
}
