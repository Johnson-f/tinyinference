//! Public reranking requests, results, and configuration.

use std::{fmt, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Cooperative cancellation shared with the caller; cancelling stops local
/// waiting, but cannot guarantee cancellation of provider processing or billing.
pub type RerankCancellation = crate::EmbeddingCancellation;

/// Candidate texts in the exact order addressed by returned result indices.
///
/// Keep IDs, metadata, and original search scores with the caller. Repeated
/// document text at different positions represents distinct candidates.
#[derive(Clone)]
pub struct RerankRequest {
    /// Question or search query used to score the documents.
    pub query: String,
    /// Original candidate text; empty or whitespace-only entries are invalid.
    pub documents: Vec<String>,
    /// Maximum results; `None` returns all, zero returns none, and values above
    /// the candidate count are clamped to that count.
    pub top_k: Option<usize>,
    /// Explicitly permit provider-side shortening of oversized input.
    pub truncate: bool,
    /// Caller-controlled cancellation signal.
    pub cancellation: RerankCancellation,
}

impl RerankRequest {
    /// Creates a request for all candidates with truncation disabled.
    pub fn new(query: impl Into<String>, documents: Vec<String>) -> Self {
        Self {
            query: query.into(),
            documents,
            top_k: None,
            truncate: false,
            cancellation: RerankCancellation::new(),
        }
    }
}

impl fmt::Debug for RerankRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankRequest")
            .field("documents", &self.documents.len())
            .field("top_k", &self.top_k)
            .field("truncate", &self.truncate)
            .field("cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// One result addressing the caller's original candidate list.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    /// Zero-based position in the submitted documents, not the sorted results.
    pub index: usize,
    /// Finite provider score; higher is better. This is neither a probability
    /// nor the vector similarity stored in [`crate::ScoredDoc::score`].
    pub relevance_score: f64,
}

/// Measured usage from the successful provider response only.
///
/// Failed or retried attempts may incur additional unreported usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankUsage {
    /// Provider-reported input tokens, without local estimates.
    pub input_tokens: u64,
}

/// Validated results in descending score order, ties ordered by original index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Exactly the effective requested result count; never a partial success.
    pub results: Vec<RerankResult>,
    /// Missing or invalid provider usage is unknown, not zero.
    pub usage: Option<RerankUsage>,
}

/// A provider-neutral reranking model, usable behind an `Arc<dyn Reranker>`.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Stable provider name.
    fn name(&self) -> &str;
    /// Configured provider model identifier.
    fn model_id(&self) -> &str;
    /// Reranks original candidate texts without modifying caller-owned metadata.
    ///
    /// Implementations must preserve original indices, return finite scores and
    /// exactly the effective result count, and respect cancellation. Empty
    /// candidates or zero results perform no provider call; pre-cancellation
    /// still wins. Nonempty requests require nonblank query and documents.
    ///
    /// # Errors
    /// Returns [`crate::Error::Validation`] for invalid inputs,
    /// [`crate::Error::Cancelled`] for cancellation, or
    /// [`crate::Error::Rerank`] for provider, deadline, or response failures.
    /// Fallback to original search order belongs to the caller.
    async fn rerank(&self, request: RerankRequest) -> crate::Result<RerankResponse>;
}

/// Voyage settings, validated when constructing [`super::VoyageReranker`].
#[derive(Clone)]
pub struct VoyageRerankConfig {
    /// Model ID; defaults to `rerank-3`. Nonblank custom IDs are accepted.
    pub model: String,
    /// API base URL without `/rerank`; defaults to Voyage's `/v1` URL.
    pub base_url: String,
    /// Total budget including request pacing, retries, and response reads.
    pub timeout: Duration,
    /// Extra attempts for 429/500/502/503/504, from zero to the core retry cap.
    /// Retries may repeat billable work; zero disables them.
    pub max_retries: u32,
    /// Maximum serialized request size, including JSON escaping.
    pub max_request_bytes: usize,
    /// Maximum response body size, enforced while reading.
    pub max_response_bytes: usize,
}

impl Default for VoyageRerankConfig {
    fn default() -> Self {
        Self {
            model: "rerank-3".into(),
            base_url: crate::VOYAGE_API_BASE.into(),
            timeout: Duration::from_secs(30),
            max_retries: tinyinference_core::MAX_RETRIES,
            max_request_bytes: 8 * 1024 * 1024,
            max_response_bytes: 2 * 1024 * 1024,
        }
    }
}

impl fmt::Debug for VoyageRerankConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoyageRerankConfig")
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("max_request_bytes", &self.max_request_bytes)
            .field("max_response_bytes", &self.max_response_bytes)
            .finish_non_exhaustive()
    }
}

/// Distinguishable failures of a reranking operation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RerankError {
    /// Credentials or access were rejected.
    #[error("reranking authentication rejected (HTTP {status})")]
    Authentication {
        /// Original 401 or 403 status.
        status: u16,
    },
    /// The provider still rate-limited the request after allowed retries.
    #[error("reranking rate limit exceeded")]
    RateLimited {
        /// Provider delay in milliseconds, bounded by the core retry policy.
        retry_after_ms: Option<u64>,
    },
    /// Another unsuccessful provider status.
    #[error("reranking provider returned HTTP {status}")]
    HttpStatus {
        /// Original HTTP status.
        status: u16,
    },
    /// The total operation deadline expired.
    #[error("reranking request timed out")]
    Timeout,
    /// The request or response was interrupted; provider billing may be unknown.
    #[error("reranking transport failed")]
    Transport,
    /// The response could not satisfy the requested result contract.
    #[error("invalid reranking response: {reason}")]
    InvalidResponse {
        /// Credential-safe explanation without provider body or document text.
        reason: &'static str,
    },
    /// The response exceeded the configured body limit.
    #[error("reranking response exceeded {limit} bytes")]
    ResponseTooLarge {
        /// Configured maximum response bytes.
        limit: usize,
    },
}
