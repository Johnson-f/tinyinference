//! Uniform construction and listing discovery for resolved provider specs.
//!
//! Hosts own saved configuration, routing policy and access gates. This factory
//! consumes their resolved spec and obtains credentials through a callback;
//! it neither loads host configuration nor implements managed gateway policy.

use std::sync::Arc;

use crate::catalog::append_query_param;
use crate::model::ChatModel;
use crate::model::discover::{DiscoveryRequest, pinned_openrouter_providers};
use crate::{Error, Result};
use tinyinference_providers::route::{ProviderRoute, RouteKind};

use super::anthropic::{AnthropicConfig, endpoint_is_anthropic_messages};
use super::openai::codex::OpenAiCodexRouting;
use super::openai::{AuthStyle, OpenAiConfig, endpoint_is_openrouter};
use super::{ProviderKind, ProviderSpec};

/// Resolves a credential for an already selected provider.
///
/// Implementations can consult a keyring, OAuth profile, or environment. Errors
/// are returned unchanged. Empty credentials are permitted only when the spec
/// does not require an API key.
pub trait CredentialResolver: Send + Sync {
    /// Reads the credential without making an inference request.
    ///
    /// # Errors
    /// Returns the credential source's failure when lookup cannot complete.
    fn resolve(&self, spec: &ProviderSpec) -> Result<String>;
}

impl<F> CredentialResolver for F
where
    F: Fn(&ProviderSpec) -> Result<String> + Send + Sync,
{
    fn resolve(&self, spec: &ProviderSpec) -> Result<String> {
        self(spec)
    }
}

/// Local servers that use only the OpenAI-compatible inference transport.
///
/// The resolved spec retains its provider slug, endpoint and model identity.
/// These families have no automatic model listing or native context setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCompatibleKind {
    /// MLX's compatible model server.
    Mlx,
    /// OMLX's compatible model server.
    Omlx,
    /// An explicitly local, otherwise unclassified compatible server.
    OpenAi,
}

/// How an Ollama context option reaches the server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OllamaContextMode {
    /// Apply num_ctx through a native warm-up before inference, preserving the
    /// registry's default behavior. Failed native setup prevents inference.
    #[default]
    Native,
    /// Keep the compatible request options without issuing a native warm-up.
    /// Ollama may ignore num_ctx on this wire; no confirmed limit is advertised.
    CompatibleOnly,
}

/// Optional adapter settings applied after host policy and route resolution.
///
/// Header/query values can contain secrets; Debug omits their values. The registry never adds host identity or telemetry.
#[derive(Clone, Default)]
pub struct ProviderBuildOptions<'a> {
    /// Authentication override. Defaults to Anthropic for that family, no
    /// authentication for local families or empty optional credentials, and Bearer otherwise.
    pub auth_style: Option<AuthStyle>,
    /// Fixed per-route temperature.
    pub temperature_override: Option<f64>,
    /// Model glob patterns that reject temperature.
    pub temperature_unsupported_models: &'a [String],
    /// Native tool capability override; local adapters default to false.
    pub native_tool_calling: Option<bool>,
    /// Vision capability override; local adapters default to false.
    pub vision: Option<bool>,
    /// Whether to fold system messages into the first user message.
    pub merge_system_into_user: bool,
    /// Additional request headers.
    pub extra_headers: &'a [(String, String)],
    /// Additional request query parameters.
    pub extra_query_params: &'a [(String, String)],
    /// Host client identity, when required by an endpoint.
    pub user_agent: Option<&'a str>,
    /// Options merged beneath request-specific options.
    pub default_provider_options: Option<serde_json::Value>,
    /// Optional Ollama context setting. Native mode applies it before the first
    /// inference request; a failed setup prevents inference. CompatibleOnly
    /// retains the option on the compatible wire without native setup.
    pub num_ctx: Option<u32>,
    /// Explicit local-compatible family for a Compatible spec. Preserves its
    /// provider/profile/cache identity, conservative tools/vision, and disables
    /// automatic listings and Anthropic cache markers.
    pub local_compatible: Option<LocalCompatibleKind>,
    /// Ollama context setup policy. Defaults to native setup; CompatibleOnly
    /// preserves hosts that historically used the compatible wire exclusively.
    pub ollama_context_mode: OllamaContextMode,
    /// Resolved Codex routing. OAuth listings use the Codex `/models` endpoint;
    /// the host supplies `client_version` through `extra_query_params`, and any
    /// user-agent identity through `user_agent`.
    pub codex: Option<&'a OpenAiCodexRouting>,
    /// HTTP client carrying host TLS/proxy policy. When set it is authoritative
    /// and CA/builder settings below are ignored.
    pub http: Option<reqwest::Client>,
    /// Factory for host-configured HTTP builders, used when no finished client
    /// is injected. The closure should return a fresh builder for every call.
    pub http_builder: Option<ProviderHttpClientBuilder<'a>>,
    /// Additional trusted CA certificates in PEM format. Applied to the
    /// configured builder, preserving its proxy/timeouts, or to a default
    /// builder when none is supplied.
    pub ca_bundle: Option<&'a str>,
}

/// A factory for host-configured HTTP builders.
///
/// The closure returns a fresh builder so provider construction can preserve
/// host proxy, timeout, and TLS policy while applying optional CA roots.
pub type ProviderHttpClientBuilder<'a> = Arc<dyn Fn() -> reqwest::ClientBuilder + Send + Sync + 'a>;

impl ProviderBuildOptions<'_> {
    fn http_client(&self) -> Result<Option<reqwest::Client>> {
        if let Some(client) = &self.http {
            return Ok(Some(client.clone()));
        }
        let Some(builder) = self
            .http_builder
            .as_ref()
            .map(|factory| factory())
            .or_else(|| self.ca_bundle.is_some().then(reqwest::Client::builder))
        else {
            return Ok(None);
        };
        let builder = match self.ca_bundle {
            Some(pem) => super::http::client_builder_with_ca_bundle(builder, pem)?,
            None => builder,
        };
        builder
            .build()
            .map(Some)
            .map_err(|_| Error::Validation("could not build provider HTTP client".into()))
    }
}

impl std::fmt::Debug for ProviderBuildOptions<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderBuildOptions")
            .field("temperature_override", &self.temperature_override)
            .field("native_tool_calling", &self.native_tool_calling)
            .field("vision", &self.vision)
            .field("local_compatible", &self.local_compatible)
            .field("ollama_context_mode", &self.ollama_context_mode)
            .field(
                "header_names",
                &self
                    .extra_headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field(
                "query_names",
                &self
                    .extra_query_params
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("codex_oauth", &self.codex.map(|route| route.using_oauth))
            .field("ca_bundle", &self.ca_bundle.map(|_| "<PEM>"))
            .field(
                "http_builder",
                &self.http_builder.as_ref().map(|_| "<builder factory>"),
            )
            .finish_non_exhaustive()
    }
}

/// Constructs third-party and local adapters from a host-resolved spec.
///
/// ProviderKind::TinyHumans is treated as an ordinary compatible endpoint:
/// managed credentials, billing and endpoint guards belong to the host's
/// managed gateway extension, which should bypass this factory.
pub struct ProviderRegistry<R> {
    credentials: R,
}

impl<R> std::fmt::Debug for ProviderRegistry<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderRegistry")
            .finish_non_exhaustive()
    }
}

impl<R: CredentialResolver> ProviderRegistry<R> {
    /// Creates a factory using the supplied credential source.
    pub fn new(credentials: R) -> Self {
        Self { credentials }
    }

    /// Builds a model using default adapter settings.
    ///
    /// # Errors
    /// Returns validation failures for empty models/endpoints or missing required
    /// credentials, and propagates credential resolver errors.
    pub fn build(&self, spec: &ProviderSpec) -> Result<Arc<dyn ChatModel<()>>> {
        self.build_with(spec, ProviderBuildOptions::default())
    }

    /// Builds a model with host-resolved adapter settings.
    ///
    /// First-party Anthropic endpoints use Messages when native tools are not
    /// explicitly disabled. Other endpoints use Chat Completions; Codex OAuth
    /// routing selects Responses and omits max_output_tokens.
    ///
    /// # Errors
    /// Returns the same configuration and credential errors as [Self::build],
    /// and rejects local_compatible settings for non-Compatible specs.
    pub fn build_with(
        &self,
        spec: &ProviderSpec,
        options: ProviderBuildOptions<'_>,
    ) -> Result<Arc<dyn ChatModel<()>>> {
        validate(spec)?;
        if options.local_compatible.is_some() && spec.kind != ProviderKind::Compatible {
            return Err(Error::Validation(
                "local_compatible requires a Compatible provider spec".into(),
            ));
        }
        let key = self.resolve_key(spec)?;
        let http = options.http_client()?;
        let endpoint = effective_endpoint(spec, &options).trim();
        let model = spec.model.trim();
        let auth = effective_auth(spec, &options, &key);
        if options.local_compatible.is_none()
            && matches!(auth, AuthStyle::Anthropic)
            && options.native_tool_calling != Some(false)
            && endpoint_is_anthropic_messages(endpoint)
        {
            return Ok(
                super::anthropic::config::build_anthropic_model_with_overrides(
                    AnthropicConfig {
                        endpoint,
                        api_key: &key,
                        model,
                        temperature_override: options.temperature_override,
                        temperature_unsupported_models: options.temperature_unsupported_models,
                        extra_headers: options.extra_headers,
                    },
                    http.clone(),
                    &spec.provider,
                    options.vision,
                ),
            );
        }
        let compatible_local = options.local_compatible.is_some()
            || (spec.kind == ProviderKind::Ollama
                && options.ollama_context_mode == OllamaContextMode::CompatibleOnly);
        let local = is_local(&spec.kind) || compatible_local;
        let oauth = options.codex.is_some_and(|routing| routing.using_oauth);
        let route_options = if local {
            options
                .num_ctx
                .map(|num_ctx| serde_json::json!({"options": {"num_ctx": num_ctx}}))
        } else {
            openrouter_default_provider_options(endpoint)
        };
        let provider_options =
            merge_route_options(route_options, options.default_provider_options.clone());
        let native_num_ctx = if spec.kind == ProviderKind::Ollama && !compatible_local {
            provider_options
                .as_ref()
                .and_then(|value| value.get("options"))
                .and_then(|value| value.get("num_ctx"))
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            Error::Validation(
                                "Ollama num_ctx must be a positive 32-bit integer".into(),
                            )
                        })
                })
                .transpose()?
        } else {
            None
        };
        let native_options = provider_options.clone();
        let extra_headers = effective_headers(&options);
        let extra_query_params = effective_query_params(&options);
        let config = OpenAiConfig {
            provider_name: &spec.provider,
            endpoint,
            api_key: &key,
            auth_style: auth.clone(),
            model,
            temperature_unsupported_models: options.temperature_unsupported_models,
            temperature_override: options.temperature_override,
            merge_system_into_user: options.merge_system_into_user,
            extra_headers: &extra_headers,
            native_tool_calling: options.native_tool_calling.or(local.then_some(false)),
            vision: options.vision.or(local.then_some(false)),
            default_provider_options: provider_options,
            responses_api_primary: oauth,
            responses_omit_max_output_tokens: oauth,
            extra_query_params: &extra_query_params,
            user_agent: options.user_agent,
            explicit_cache_control: !local && endpoint_is_openrouter(endpoint),
        };
        if compatible_local {
            return Ok(super::openai::config::build_openai_model_with(
                config,
                false,
                http.clone(),
            ));
        }
        if local {
            let mut resolved = spec.clone();
            resolved.base_url = endpoint.to_owned();
            resolved.model = model.to_owned();
            let model =
                super::openai::OpenAiModel::from_spec(resolved, &key)?.with_auth_style(auth);
            let mut model =
                super::openai::config::configure_openai_model(model, config, http.clone());
            let num_ctx = native_num_ctx;
            if let Some(num_ctx) = num_ctx {
                model = model.with_local_num_ctx(num_ctx);
                if let Some(native_options) = native_options {
                    model = model.with_default_provider_options(native_options);
                }
            }
            return Ok(Arc::new(LocalRegistryModel {
                model,
                apply_context: num_ctx.is_some(),
                ready: tokio::sync::OnceCell::new(),
            }));
        }
        Ok(super::openai::config::build_openai_model_with(
            config, true, http,
        ))
    }

    /// Resolves discovery for a typed route using host-resolved provider data.
    ///
    /// Route kinds without a provider listing, including managed, local, and
    /// subprocess routes, return `None` without consulting credentials. A
    /// cloud route uses the supplied spec's endpoint and credential resolver;
    /// endpoints are never inferred from the route's product name. The
    /// existing spec-and-options method remains [`Self::discovery_request`].
    ///
    /// # Errors
    /// Returns validation or credential errors from the existing spec-based
    /// discovery path.
    pub fn discovery_request_for_route(
        &self,
        route: ProviderRoute,
        spec: Option<ProviderSpec>,
    ) -> Result<Option<DiscoveryRequest>> {
        self.discovery_request_for_route_with_options(route, spec, &ProviderBuildOptions::default())
    }

    /// Resolves discovery for a typed route using host-resolved provider data.
    ///
    /// Route kinds without a provider listing, including managed, local, and
    /// subprocess routes, return `None` without consulting credentials. A
    /// cloud route uses the supplied spec's endpoint and credential resolver;
    /// endpoints are never inferred from the route's product name.
    ///
    /// # Errors
    /// Returns validation or credential errors from the existing spec-based
    /// discovery path.
    pub fn discovery_request_for_route_with_options(
        &self,
        route: ProviderRoute,
        spec: Option<ProviderSpec>,
        options: &ProviderBuildOptions<'_>,
    ) -> Result<Option<DiscoveryRequest>> {
        if !matches!(route.kind(), RouteKind::Cloud { .. }) {
            return Ok(None);
        }
        let Some(mut spec) = spec else {
            return Ok(None);
        };
        if spec.provider != route.provider {
            return Ok(None);
        }
        if !route.model.trim().is_empty() {
            spec.model = route.model;
        }
        self.discovery_request(&spec, options)
    }

    /// Builds a listing request with the same endpoint, model and credentials
    /// as construction. Ollama enables native /api/show limit discovery. Other
    /// local families return None; Codex OAuth carries its listing URL and
    /// protocol headers in the returned request.
    ///
    /// Extra headers are retained. For Codex OAuth listings, extra query
    /// parameters and the supplied user-agent are also carried to the listing
    /// request; other providers do not copy inference query parameters.
    ///
    /// # Errors
    /// Returns configuration/credential errors for discoverable routes. Local
    /// families other than Ollama do not consult credentials.
    pub fn discovery_request(
        &self,
        spec: &ProviderSpec,
        options: &ProviderBuildOptions<'_>,
    ) -> Result<Option<DiscoveryRequest>> {
        if (is_local(&spec.kind) && spec.kind != ProviderKind::Ollama)
            || options.local_compatible.is_some()
        {
            return Ok(None);
        }
        validate(spec)?;
        let key = self.resolve_key(spec)?;
        let endpoint = effective_endpoint(spec, options).trim();
        let model = spec.model.trim();
        let mut request = DiscoveryRequest::new(endpoint, model);
        if let Some(route) = options.codex.filter(|route| route.using_oauth) {
            let mut listing_url = format!("{}/models", endpoint.trim_end_matches('/'));
            for (name, value) in effective_query_params(options) {
                listing_url = append_query_param(&listing_url, &name, &value);
            }
            request = request.with_listing_url(listing_url);
            for (name, value) in route.headers() {
                request = request.with_header(name, value);
            }
            if let Some(user_agent) = options.user_agent {
                request = request.with_header("User-Agent", user_agent);
            }
        }
        if spec.kind == ProviderKind::Ollama {
            request = request.with_ollama_native(true);
        }
        if !key.trim().is_empty() {
            request = match effective_auth(spec, options, &key) {
                AuthStyle::None => request,
                AuthStyle::Bearer => {
                    request.with_header("Authorization", format!("Bearer {}", key.trim()))
                }
                AuthStyle::XApiKey => request.with_header("x-api-key", key.trim()),
                AuthStyle::Anthropic => request
                    .with_header("x-api-key", key.trim())
                    .with_header("anthropic-version", "2023-06-01"),
                AuthStyle::Custom(name) => request.with_header(name, key.trim()),
            };
        }
        for (name, value) in effective_headers(options) {
            request = with_replaced_header(request, &name, &value);
        }
        if let Some(provider_options) = options
            .default_provider_options
            .clone()
            .or_else(|| openrouter_default_provider_options(endpoint))
        {
            request = request.with_pinned_providers(pinned_openrouter_providers(&provider_options));
        }
        Ok(Some(request))
    }

    fn resolve_key(&self, spec: &ProviderSpec) -> Result<String> {
        let key = self.credentials.resolve(spec)?;
        if spec.requires_api_key && key.trim().is_empty() {
            return Err(Error::Validation(format!(
                "provider '{}' requires an API key",
                spec.provider
            )));
        }
        Ok(key.trim().to_string())
    }
}

fn with_replaced_header(
    mut request: DiscoveryRequest,
    name: &str,
    value: &str,
) -> DiscoveryRequest {
    request
        .headers
        .retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
    request.with_header(name, value)
}

fn effective_headers(options: &ProviderBuildOptions<'_>) -> Vec<(String, String)> {
    let mut headers = options
        .codex
        .into_iter()
        .flat_map(|route| route.headers())
        .collect::<Vec<_>>();
    for (name, value) in options.extra_headers {
        if let Some(existing) = headers
            .iter_mut()
            .find(|(existing, _)| existing.eq_ignore_ascii_case(name))
        {
            *existing = (name.clone(), value.clone());
        } else {
            headers.push((name.clone(), value.clone()));
        }
    }
    headers
}

fn effective_query_params(options: &ProviderBuildOptions<'_>) -> Vec<(String, String)> {
    let mut query = options.extra_query_params.to_vec();
    if let Some(route) = options.codex.filter(|route| route.using_oauth)
        && !query.iter().any(|(name, _)| name == "client_version")
    {
        query.extend(
            route.client_version_query(&super::openai::codex::openai_codex_client_version()),
        );
    }
    query
}

/// Why a cloud slug could not resolve to a concrete model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlugModelError {
    /// The requested model and its fallback were both empty.
    MissingModel,
    /// The requested model was a hint and its fallback was absent, blank, or
    /// itself another hint.
    NoConcreteDefault,
}

impl std::fmt::Display for SlugModelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModel => formatter.write_str("no model configured"),
            Self::NoConcreteDefault => formatter.write_str("no concrete default model configured"),
        }
    }
}

impl std::error::Error for SlugModelError {}

/// Resolves a cloud slug model using the host's abstract-model predicate.
///
/// Concrete requested model identifiers are returned byte-for-byte, including
/// namespaced values such as `qwen3:14b`. Empty requests use the configured
/// default. Hint requests map to a trimmed concrete default, and the predicate
/// also validates that fallback. This preserves the host's existing trim
/// behavior while keeping hint semantics owned by the host.
///
/// # Errors
/// Returns [`SlugModelError::MissingModel`] for an empty request with no
/// fallback, or [`SlugModelError::NoConcreteDefault`] when a hint cannot map to
/// a concrete fallback.
pub fn resolve_slug_model(
    requested: &str,
    default: Option<&str>,
    is_hint: impl Fn(&str) -> bool,
) -> std::result::Result<String, SlugModelError> {
    let effective = if requested.trim().is_empty() {
        default.unwrap_or_default()
    } else {
        requested
    };
    if effective.trim().is_empty() {
        return if requested.trim().is_empty() {
            Err(SlugModelError::MissingModel)
        } else {
            Err(SlugModelError::NoConcreteDefault)
        };
    }
    if is_hint(effective) {
        let fallback = default
            .map(str::trim)
            .filter(|candidate| !candidate.is_empty() && !is_hint(candidate))
            .ok_or(SlugModelError::NoConcreteDefault)?;
        return Ok(fallback.to_owned());
    }
    Ok(effective.to_owned())
}

/// Stable price ordering for direct OpenRouter routes; request-specific
/// options can override it. Other endpoints receive no default routing option.
pub fn openrouter_default_provider_options(endpoint: &str) -> Option<serde_json::Value> {
    endpoint_is_openrouter(endpoint).then(|| serde_json::json!({"provider": {"sort": "price"}}))
}

fn validate(spec: &ProviderSpec) -> Result<()> {
    if spec.model.trim().is_empty() {
        return Err(Error::Validation("provider model must not be empty".into()));
    }
    if spec.base_url.trim().is_empty() {
        return Err(Error::Validation(
            "provider endpoint must not be empty".into(),
        ));
    }
    Ok(())
}

fn is_local(kind: &ProviderKind) -> bool {
    matches!(
        kind,
        ProviderKind::Ollama | ProviderKind::LmStudio | ProviderKind::LlamaCpp | ProviderKind::Vllm
    )
}

fn effective_endpoint<'a>(
    spec: &'a ProviderSpec,
    options: &'a ProviderBuildOptions<'_>,
) -> &'a str {
    options
        .codex
        .map_or(&spec.base_url, |route| &route.endpoint)
}

// Configuring Ollama requires its native endpoint; the compatible wire ignores
// num_ctx. One successful setup is shared by all concurrent calls on this model.
struct LocalRegistryModel {
    model: super::openai::OpenAiModel,
    apply_context: bool,
    ready: tokio::sync::OnceCell<()>,
}

impl LocalRegistryModel {
    async fn prepare(&self) -> Result<()> {
        if self.apply_context {
            self.ready
                .get_or_try_init(|| async { self.model.warm_up().await })
                .await?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl ChatModel<()> for LocalRegistryModel {
    fn profile(&self) -> Option<&crate::model::ModelProfile> {
        <super::openai::OpenAiModel as ChatModel<()>>::profile(&self.model)
    }
    fn supports_input(
        &self,
        modality: crate::model::InputModality,
        mime: &str,
        source: crate::model::InputSource,
    ) -> bool {
        <super::openai::OpenAiModel as ChatModel<()>>::supports_input(
            &self.model,
            modality,
            mime,
            source,
        )
    }
    fn cache_identity(&self) -> Option<String> {
        <super::openai::OpenAiModel as ChatModel<()>>::cache_identity(&self.model)
    }
    fn tracks_physical_attempts(&self) -> bool {
        <super::openai::OpenAiModel as ChatModel<()>>::tracks_physical_attempts(&self.model)
    }
    async fn invoke(
        &self,
        state: &(),
        request: crate::model::ModelRequest,
    ) -> Result<crate::model::ModelResponse> {
        self.prepare().await?;
        self.model.invoke(state, request).await
    }
    async fn stream(
        &self,
        state: &(),
        request: crate::model::ModelRequest,
    ) -> Result<crate::model::ModelStream> {
        self.prepare().await?;
        self.model.stream(state, request).await
    }
    async fn fetch_deferred(
        &self,
        handle: &crate::model::DeferredHandle,
    ) -> Result<crate::model::DeferredStatus> {
        <super::openai::OpenAiModel as ChatModel<()>>::fetch_deferred(&self.model, handle).await
    }
}

// Route defaults form a base; host settings override only the fields supplied.
fn merge_route_options(
    defaults: Option<serde_json::Value>,
    overrides: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match (defaults, overrides) {
        (Some(serde_json::Value::Object(mut base)), Some(serde_json::Value::Object(over))) => {
            for (key, value) in over {
                let previous = base.remove(&key);
                base.insert(
                    key,
                    merge_route_options(previous, Some(value)).expect("override is present"),
                );
            }
            Some(serde_json::Value::Object(base))
        }
        (base, None | Some(serde_json::Value::Null)) => base,
        (_, over) => over,
    }
}

fn effective_auth(spec: &ProviderSpec, options: &ProviderBuildOptions<'_>, key: &str) -> AuthStyle {
    options.auth_style.clone().unwrap_or_else(|| {
        if is_local(&spec.kind)
            || options.local_compatible.is_some()
            || (!spec.requires_api_key && key.is_empty())
        {
            AuthStyle::None
        } else if spec.kind == ProviderKind::Anthropic {
            AuthStyle::Anthropic
        } else {
            AuthStyle::Bearer
        }
    })
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
