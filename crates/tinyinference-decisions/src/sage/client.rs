//! Levanto Sage HTTP transport and request validation.

use std::time::Duration;

use reqwest::{StatusCode, Url};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::types::{
    BatchDecisionRequest, BatchDecisionResponse, DecisionContent, DecisionQuestion,
    DecisionRequest, DecisionResponse, GroundingConfig, SageModel, StructuredContent,
    UsageEstimate,
};
use crate::{Error, Result};

const DEFAULT_BASE_URL: &str = "https://sage.levanto.ai/";
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// A client for Levanto Sage's typed decision endpoints.
pub struct SageClient {
    http: reqwest::Client,
    base_url: Url,
    api_key: String,
}

impl std::fmt::Debug for SageClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SageClient")
            .field(
                "base_url",
                &format!(
                    "{}/[REDACTED]",
                    self.base_url.origin().ascii_serialization()
                ),
            )
            .field("api_key", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl SageClient {
    /// Creates a client targeting Sage's public endpoint.
    ///
    /// # Errors
    /// Returns [`Error::InvalidConfig`] if the API key is empty.
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a client targeting a custom Sage-compatible endpoint.
    ///
    /// Plain HTTP is accepted only for loopback test servers. The base URL
    /// must be the API root and must not contain credentials or a query.
    ///
    /// # Errors
    /// Returns [`Error::InvalidConfig`] for an invalid key or URL.
    pub fn with_base_url(api_key: impl Into<String>, base_url: &str) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(Error::InvalidConfig {
                reason: "API key must not be empty".into(),
            });
        }
        let mut base_url = Url::parse(base_url).map_err(|error| Error::InvalidConfig {
            reason: format!("invalid Sage base URL: {error}"),
        })?;
        if !base_url.username().is_empty()
            || base_url.password().is_some()
            || base_url.query().is_some()
            || base_url.fragment().is_some()
        {
            return Err(Error::InvalidConfig {
                reason: "Sage base URL must not contain credentials, a query, or a fragment".into(),
            });
        }
        let host = base_url.host_str().unwrap_or_default();
        if base_url.scheme() != "https"
            && !(base_url.scheme() == "http"
                && (host == "localhost" || host == "127.0.0.1" || host == "[::1]"))
        {
            return Err(Error::InvalidConfig {
                reason: "Sage credentials require HTTPS except for loopback servers".into(),
            });
        }
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|source| Error::Transport { source })?;
        Ok(Self {
            http,
            base_url,
            api_key,
        })
    }

    /// Checks whether Sage is ready to serve decisions.
    ///
    /// HTTP 503 means Sage is still loading and returns `false`.
    ///
    /// # Errors
    /// Returns a transport or HTTP error for other failures.
    pub async fn ready(&self) -> Result<bool> {
        let response = self
            .http
            .get(self.url("ready"))
            .send()
            .await
            .map_err(|source| Error::Transport { source })?;
        if response.status() == StatusCode::SERVICE_UNAVAILABLE {
            return Ok(false);
        }
        if response.status().is_success() {
            return Ok(true);
        }
        Err(http_error(response))
    }

    /// Lists the models advertised by Sage's public catalog.
    ///
    /// # Errors
    /// Returns a transport, HTTP, or response decoding error.
    pub async fn models(&self) -> Result<Vec<SageModel>> {
        #[derive(serde::Deserialize)]
        struct Catalog {
            data: Vec<SageModel>,
        }
        let response = self
            .http
            .get(self.url("models"))
            .send()
            .await
            .map_err(|source| Error::Transport { source })?;
        Ok(read_response::<Catalog>(response).await?.data)
    }

    /// Asks Sage one typed question.
    ///
    /// An uncertain answer is returned as a successful result with a `None`
    /// verdict in its kind-specific result.
    ///
    /// # Errors
    /// Returns a validation, transport, HTTP, or response decoding error.
    pub async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        validate_question(
            &request.content,
            &request.question,
            request.grounding.as_ref(),
        )?;
        self.post("decide", request).await
    }

    /// Asks multiple questions, preserving group and question order.
    ///
    /// Individual question failures appear inside [`BatchDecisionResponse`].
    ///
    /// # Errors
    /// Returns a validation, transport, HTTP, or response decoding error for
    /// the overall call.
    pub async fn decide_batch(
        &self,
        request: &BatchDecisionRequest,
    ) -> Result<BatchDecisionResponse> {
        validate_batch(request)?;
        self.post("decide/batch", request).await
    }

    /// Estimates billed input tokens for one decision without running it.
    ///
    /// # Errors
    /// Returns a validation, transport, HTTP, or response decoding error.
    pub async fn estimate_decision(&self, request: &DecisionRequest) -> Result<UsageEstimate> {
        validate_question(
            &request.content,
            &request.question,
            request.grounding.as_ref(),
        )?;
        self.estimate(request).await
    }

    /// Estimates billed input tokens for a batch without running it.
    ///
    /// # Errors
    /// Returns a validation, transport, HTTP, or response decoding error.
    pub async fn estimate_batch(&self, request: &BatchDecisionRequest) -> Result<UsageEstimate> {
        validate_batch(request)?;
        self.estimate(request).await
    }

    async fn estimate<T: Serialize + ?Sized>(&self, request: &T) -> Result<UsageEstimate> {
        #[derive(serde::Deserialize)]
        struct Estimate {
            usage: UsageEstimate,
        }
        Ok(self
            .post::<_, Estimate>("usage/estimate", request)
            .await?
            .usage)
    }

    async fn post<T: Serialize + ?Sized, R: DeserializeOwned>(
        &self,
        path: &str,
        request: &T,
    ) -> Result<R> {
        let response = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await
            .map_err(|source| Error::Transport { source })?;
        read_response(response).await
    }

    fn url(&self, path: &str) -> Url {
        self.base_url
            .join(path)
            .expect("static Sage endpoint path is valid")
    }
}

async fn read_response<T: DeserializeOwned>(mut response: reqwest::Response) -> Result<T> {
    if !response.status().is_success() {
        return Err(http_error(response));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|source| Error::Transport { source })?
    {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(Error::ResponseTooLarge {
                limit: MAX_RESPONSE_BYTES,
            });
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body).map_err(|source| Error::Decode { source })
}

fn http_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    match status {
        400 | 422 => Error::Unprocessable,
        401 | 403 => Error::Authentication,
        429 => Error::RateLimited,
        503 => Error::Overloaded,
        _ => Error::HttpStatus { status },
    }
}

fn validate_batch(request: &BatchDecisionRequest) -> Result<()> {
    if request.requests.is_empty() {
        return Err(Error::invalid_request("batch requires at least one group"));
    }
    for group in &request.requests {
        if group.questions.is_empty() {
            return Err(Error::invalid_request(
                "batch group requires at least one question",
            ));
        }
        for question in &group.questions {
            validate_question(
                &group.content,
                &question.question,
                question.grounding.as_ref(),
            )?;
        }
    }
    Ok(())
}

fn validate_question(
    content: &DecisionContent,
    question: &DecisionQuestion,
    grounding: Option<&GroundingConfig>,
) -> Result<()> {
    if question.id().trim().is_empty() {
        return Err(Error::invalid_request("question id must not be empty"));
    }
    let image = matches!(
        content,
        DecisionContent::Structured(StructuredContent::Image { .. })
    );
    let list = matches!(
        content,
        DecisionContent::Structured(StructuredContent::List { .. })
    );
    if image && grounding.is_some() {
        return Err(Error::invalid_request("image content cannot use grounding"));
    }
    if let Some(config) = grounding {
        if config
            .confidence_floor
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err(Error::invalid_request(
                "grounding confidence floor must be in 0..=1",
            ));
        }
        if config
            .max_results
            .is_some_and(|value| !(1..=20).contains(&value))
        {
            return Err(Error::invalid_request(
                "grounding max results must be in 1..=20",
            ));
        }
        if config
            .max_context_tokens
            .is_some_and(|value| !(1..=8000).contains(&value))
        {
            return Err(Error::invalid_request(
                "grounding context tokens must be in 1..=8000",
            ));
        }
    }
    match question {
        DecisionQuestion::YesNo { .. } => {}
        DecisionQuestion::Choice { options, .. } => {
            let max = if image { 20 } else { 120 };
            if !(2..=max).contains(&options.len()) {
                return Err(Error::invalid_request(format!(
                    "choice requires 2..={max} options"
                )));
            }
        }
        DecisionQuestion::Scale { levels, .. } => {
            if levels.len() != 5
                || (0..5).any(|level| !levels.iter().any(|item| item.level == level))
            {
                return Err(Error::invalid_request(
                    "scale requires exactly levels 0 through 4",
                ));
            }
        }
        DecisionQuestion::Sort { .. } => {
            if !list || grounding.is_some() {
                return Err(Error::invalid_request(
                    "sort requires list content without grounding",
                ));
            }
        }
        DecisionQuestion::Tags { tags, .. } => {
            if !(1..=120).contains(&tags.len()) {
                return Err(Error::invalid_request("tags requires 1..=120 tags"));
            }
        }
    }
    if list {
        if !matches!(question, DecisionQuestion::Sort { .. }) {
            return Err(Error::invalid_request(
                "list content requires a sort question",
            ));
        }
        let DecisionContent::Structured(StructuredContent::List { value }) = content else {
            unreachable!()
        };
        if !(2..=120).contains(&value.len()) {
            return Err(Error::invalid_request("sort requires 2..=120 list items"));
        }
    }
    if let DecisionContent::Structured(StructuredContent::Image { media, .. }) = content
        && ![
            "data:image/png;base64,",
            "data:image/jpeg;base64,",
            "data:image/webp;base64,",
        ]
        .iter()
        .any(|prefix| media.starts_with(prefix) && media.len() > prefix.len())
    {
        return Err(Error::invalid_request(
            "image must be a PNG, JPEG, or WebP base64 data URI",
        ));
    }
    Ok(())
}
