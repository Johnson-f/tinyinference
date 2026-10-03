//! Provider-neutral reranking of candidate document texts.
//!
//! [`VoyageReranker`] accepts an explicit API key and defaults to `rerank-3`.
//! Candidates may come from any search system. Results address the original
//! text list; keep document IDs, metadata, and vector similarity scores with
//! the caller. Truncation is disabled unless explicitly requested.
//!
//! A total deadline includes pacing, HTTP requests, response reads, and retry
//! waits. Transport interruptions are not retried because the provider may
//! already have billed the request. Explicit temporary HTTP statuses may be
//! retried; configure zero retries to disable this. Reported usage covers only
//! the successful response and missing usage means unknown, never zero.
//!
//! # Rerank search hits using caller-owned source text
//!
//! [`crate::Retriever`] stores vectors and metadata, not document text. Resolve
//! texts by ID before sending any request; missing text should not silently
//! remove a candidate and change the index mapping.
//!
//! ```no_run
//! use std::collections::HashMap;
//! use tinyinference_embeddings::{
//!     Error, Result, Retriever, Reranker, RerankRequest, ScoredDoc,
//! };
//!
//! async fn search(
//!     retriever: &Retriever,
//!     reranker: &dyn Reranker,
//!     texts: &HashMap<String, String>,
//!     query: &str,
//! ) -> Result<Vec<(ScoredDoc, Option<f64>)>> {
//!     let hits = retriever.retrieve(query, 30).await?;
//!     let documents = hits.iter().map(|hit| {
//!         texts.get(&hit.id).cloned()
//!             .ok_or_else(|| Error::Validation("missing source document".into()))
//!     }).collect::<Result<Vec<_>>>()?;
//!     let mut request = RerankRequest::new(query, documents);
//!     request.top_k = Some(5);
//!     match reranker.rerank(request).await {
//!         Ok(response) => response.results.into_iter().map(|result| {
//!             let hit = hits.get(result.index).cloned()
//!                 .ok_or_else(|| Error::Validation("invalid reranking index".into()))?;
//!             Ok((hit, Some(result.relevance_score)))
//!         }).collect(),
//!         Err(Error::Cancelled) => Err(Error::Cancelled),
//!         // This caller elects to retain search order on provider failure.
//!         Err(Error::Rerank(_)) => Ok(hits.into_iter().take(5)
//!             .map(|hit| (hit, None)).collect()),
//!         Err(error) => Err(error),
//!     }
//! }
//! ```

mod types;
mod voyage;

pub use types::{
    RerankCancellation, RerankError, RerankRequest, RerankResponse, RerankResult, RerankUsage,
    Reranker, VoyageRerankConfig,
};
pub use voyage::VoyageReranker;

fn validate_request(request: &RerankRequest) -> crate::Result<usize> {
    let count = request.documents.len();
    let top_k = request.top_k.unwrap_or(count).min(count);
    if top_k == 0 {
        return Ok(0);
    }
    if request.query.trim().is_empty() {
        return Err(crate::Error::Validation(
            "reranking query must not be blank".into(),
        ));
    }
    if let Some(index) = request
        .documents
        .iter()
        .position(|text| text.trim().is_empty())
    {
        return Err(crate::Error::Validation(format!(
            "reranking document at index {index} must not be blank"
        )));
    }
    Ok(top_k)
}

fn validate_results(
    response: &mut RerankResponse,
    document_count: usize,
    top_k: usize,
) -> crate::Result<()> {
    if response.results.len() != top_k {
        return Err(RerankError::InvalidResponse {
            reason: "unexpected result count",
        }
        .into());
    }
    let mut seen = vec![false; document_count];
    for result in &response.results {
        let Some(already_seen) = seen.get_mut(result.index) else {
            return Err(RerankError::InvalidResponse {
                reason: "document index out of range",
            }
            .into());
        };
        if *already_seen {
            return Err(RerankError::InvalidResponse {
                reason: "duplicate document index",
            }
            .into());
        }
        *already_seen = true;
        if !result.relevance_score.is_finite() {
            return Err(RerankError::InvalidResponse {
                reason: "non-finite relevance score",
            }
            .into());
        }
    }
    response.results.sort_by(|a, b| {
        if a.relevance_score == b.relevance_score {
            a.index.cmp(&b.index)
        } else {
            b.relevance_score.total_cmp(&a.relevance_score)
        }
    });
    Ok(())
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
