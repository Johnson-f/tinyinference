//! Crate-wide error and result types.

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No API key was configured.
    #[error("TYPESAFE_API_KEY must be set or supplied explicitly")]
    MissingApiKey,
    /// Client configuration is not usable.
    #[error("invalid client configuration: {reason}")]
    InvalidConfig {
        /// Stable explanation of the rejected field.
        reason: String,
    },
    /// The request violates the local System One contract.
    #[error("invalid request: {reason}")]
    InvalidRequest {
        /// Stable explanation of the rejected value.
        reason: String,
    },
    /// Authentication was rejected.
    #[error("provider authentication failed")]
    Authentication,
    /// The provider rejected the request shape.
    #[error("provider rejected the request")]
    Unprocessable,
    /// The account or endpoint rate limit was reached.
    #[error("provider rate limit exceeded")]
    RateLimited,
    /// The provider reported temporary overload.
    #[error("provider service overloaded")]
    Overloaded,
    /// The endpoint returned another unsuccessful status.
    #[error("provider request failed with status {status}")]
    HttpStatus {
        /// Returned HTTP status code.
        status: u16,
    },
    /// The request timed out.
    #[error("provider request timed out")]
    Timeout,
    /// The HTTP transport failed before a response was available.
    #[error("provider transport failed")]
    Transport {
        /// Underlying transport failure.
        #[source]
        source: reqwest::Error,
    },
    /// The response body was not valid JSON for the declared wire shape.
    #[error("provider response could not be decoded")]
    Decode {
        /// Underlying JSON decoding failure.
        #[source]
        source: serde_json::Error,
    },
    /// The decoded response is inconsistent with the request.
    #[error("invalid response: {reason}")]
    InvalidResponse {
        /// Stable explanation of the contract violation.
        reason: String,
    },
}

impl Error {
    pub(crate) fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_response(reason: impl Into<String>) -> Self {
        Self::InvalidResponse {
            reason: reason.into(),
        }
    }
}

/// The crate's standard result type.
///
/// Use this alias in public signatures instead of spelling out
/// `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod test;
