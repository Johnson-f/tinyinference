use std::{fmt, io::Write, sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;

use super::{
    RerankCancellation, RerankError, RerankRequest, RerankResponse, RerankResult, RerankUsage,
    Reranker, VoyageRerankConfig,
};
use crate::{Error, Result};

/// A Voyage reranker with caller-supplied credentials and bounded operations.
///
/// The client never reads ambient credentials. Timeout or cancellation stops
/// local waiting; Voyage may still finish and bill work already submitted.
/// Automatic retries cover explicit 429/500/502/503/504 responses only, and may
/// repeat billed work. Configure zero retries to disable them.
pub struct VoyageReranker {
    config: VoyageRerankConfig,
    authorization: HeaderValue,
    client: reqwest::Client,
    transport: Arc<dyn HttpTransport>,
}

impl fmt::Debug for VoyageReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoyageReranker")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl VoyageReranker {
    /// Creates a reranker with default settings.
    ///
    /// # Errors
    /// Returns [`Error::Validation`] for invalid credentials or
    /// [`Error::Rerank`] if the HTTP client cannot be created.
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::with_options(api_key, VoyageRerankConfig::default())
    }

    /// Creates a reranker with validated settings and a reusable HTTP client.
    ///
    /// Credentialed endpoints require HTTPS except on loopback. Userinfo,
    /// queries, fragments, and redirects are disallowed.
    ///
    /// # Errors
    /// Returns [`Error::Validation`] for invalid credentials, endpoint, model,
    /// timeout, limits, or retry count; [`Error::Rerank`] for client setup failure.
    pub fn with_options(
        api_key: impl Into<String>,
        mut config: VoyageRerankConfig,
    ) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(Error::Validation("Voyage API key must not be blank".into()));
        }
        if config.model.trim().is_empty()
            || config.timeout.is_zero()
            || config.max_request_bytes == 0
            || config.max_response_bytes == 0
            || config.max_retries > tinyinference_core::MAX_RETRIES
            || Instant::now().checked_add(config.timeout).is_none()
        {
            return Err(Error::Validation(
                "invalid reranking model, timeout, byte limit, or retry count".into(),
            ));
        }
        let endpoint = crate::factory::validate_custom_endpoint(&config.base_url, true)?;
        let parsed = reqwest::Url::parse(&endpoint)
            .map_err(|_| Error::Validation("invalid reranking endpoint".into()))?;
        if !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(Error::Validation(
                "reranking endpoint must not contain userinfo, query, or fragment".into(),
            ));
        }
        config.base_url = parsed.as_str().trim_end_matches('/').to_owned();
        let mut authorization = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|_| Error::Validation("invalid Voyage API key header".into()))?;
        authorization.set_sensitive(true);
        let client = reqwest::Client::builder()
            .connect_timeout(config.timeout.min(Duration::from_secs(10)))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| RerankError::Transport)?;
        Ok(Self {
            config,
            authorization,
            transport: Arc::new(ReqwestTransport {
                client: client.clone(),
            }),
            client,
        })
    }

    async fn call(
        &self,
        request: &RerankRequest,
        top_k: usize,
        deadline: Instant,
    ) -> Result<RerankResponse> {
        let wire = WireRequest {
            query: &request.query,
            documents: &request.documents,
            model: &self.config.model,
            top_k,
            return_documents: false,
            truncation: request.truncate,
        };
        let mut body = BoundedBody {
            bytes: Vec::new(),
            limit: self.config.max_request_bytes,
        };
        serde_json::to_writer(&mut body, &wire).map_err(|_| {
            Error::Validation("reranking request exceeds serialized byte limit".into())
        })?;
        let outgoing = self
            .client
            .post(format!("{}/rerank", self.config.base_url))
            .header(AUTHORIZATION, self.authorization.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body.bytes)
            .build()
            .map_err(|_| Error::Validation("could not construct reranking request".into()))?;
        ensure_active(&request.cancellation, deadline)?;
        let response = self
            .send_with_retry(outgoing, &request.cancellation, deadline)
            .await?;
        let bytes = read_body(response, self.config.max_response_bytes).await?;
        let wire: WireResponse =
            serde_json::from_slice(&bytes).map_err(|_| RerankError::InvalidResponse {
                reason: "malformed result payload",
            })?;
        let usage = wire
            .usage
            .as_ref()
            .and_then(|usage| usage.get("total_tokens"))
            .and_then(Value::as_u64)
            .map(|input_tokens| RerankUsage { input_tokens });
        let mut response = RerankResponse {
            results: wire.data,
            usage,
        };
        super::validate_results(&mut response, request.documents.len(), top_k)?;
        Ok(response)
    }

    async fn send_with_retry(
        &self,
        outgoing: reqwest::Request,
        cancellation: &RerankCancellation,
        deadline: Instant,
    ) -> Result<reqwest::Response> {
        let mut retries = 0;
        loop {
            crate::rate_limit::acquire(&self.config.base_url).await;
            ensure_active(cancellation, deadline)?;
            let attempt = outgoing.try_clone().ok_or_else(|| {
                Error::Validation("reranking request body cannot be replayed".into())
            })?;
            let response = self.transport.send(attempt).await?;
            if response.status().is_success() {
                return Ok(response);
            }
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            drop(response);
            if matches!(status, 429 | 500 | 502 | 503 | 504) && retries < self.config.max_retries {
                let delay =
                    tinyinference_core::backoff_ms_for_attempt(retries, retry_after.as_deref());
                retries += 1;
                tokio::time::sleep(Duration::from_millis(delay)).await;
                continue;
            }
            return Err(match status {
                401 | 403 => RerankError::Authentication { status },
                429 => RerankError::RateLimited {
                    retry_after_ms: tinyinference_core::parse_retry_after_ms(
                        retry_after.as_deref(),
                    ),
                },
                _ => RerankError::HttpStatus { status },
            }
            .into());
        }
    }

    async fn execute(&self, request: &RerankRequest, deadline: Instant) -> Result<RerankResponse> {
        if request.documents.is_empty() || request.top_k == Some(0) {
            return Ok(RerankResponse {
                results: vec![],
                usage: None,
            });
        }
        if request.documents.len() > 1_000 {
            return Err(Error::Validation(
                "Voyage reranking accepts at most 1000 documents".into(),
            ));
        }
        let top_k = super::validate_request(request)?;
        ensure_active(&request.cancellation, deadline)?;
        self.call(request, top_k, deadline).await
    }
}

#[async_trait]
impl Reranker for VoyageReranker {
    fn name(&self) -> &str {
        "voyage"
    }
    fn model_id(&self) -> &str {
        &self.config.model
    }

    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse> {
        if request.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let deadline = Instant::now()
            .checked_add(self.config.timeout)
            .ok_or_else(|| Error::Validation("reranking timeout is too large".into()))?;
        let result = tokio::select! {
            biased;
            () = request.cancellation.cancelled() => return Err(Error::Cancelled),
            () = tokio::time::sleep_until(deadline) => return Err(RerankError::Timeout.into()),
            result = self.execute(&request, deadline) => result,
        };
        ensure_active(&request.cancellation, deadline)?;
        result
    }
}

fn ensure_active(cancellation: &RerankCancellation, deadline: Instant) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(Error::Cancelled)
    } else if Instant::now() >= deadline {
        Err(RerankError::Timeout.into())
    } else {
        Ok(())
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    query: &'a str,
    documents: &'a [String],
    model: &'a str,
    top_k: usize,
    return_documents: bool,
    truncation: bool,
}

#[derive(Deserialize)]
struct WireResponse {
    data: Vec<RerankResult>,
    #[serde(default)]
    usage: Option<Value>,
}

struct BoundedBody {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedBody {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("request body exceeds limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[async_trait]
trait HttpTransport: Send + Sync {
    async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response>;
}

struct ReqwestTransport {
    client: reqwest::Client,
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response> {
        self.client
            .execute(request)
            .await
            .map_err(|_| RerankError::Transport.into())
    }
}

async fn read_body(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(RerankError::ResponseTooLarge { limit }.into());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| RerankError::Transport)? {
        if chunk.len() > limit.saturating_sub(body.len()) {
            return Err(RerankError::ResponseTooLarge { limit }.into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
#[path = "voyage_tests.rs"]
mod tests;
