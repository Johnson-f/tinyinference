//! Context-window precedence and remembered provider-sourced windows.
//!
//! Hosts supply configuration overrides, a resolved discovery request, local
//! profile fallback and static hints. This module neither selects providers nor
//! reads configuration, credentials or platform pricing catalogs.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::discover::{
    DiscoveryRequest, LimitSource, ModelLimitsCache, ModelListingFetcher, ReqwestListingFetcher,
    discover_model_limits_with_guard,
};
use crate::model::context_window_for_model_id;
use crate::model::workload;
use tinyinference_providers::route::LocalProviderKind;

/// A managed default hint supplied by the host that owns that model mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedDefaultWindow {
    /// The exact managed model id receiving this hint.
    pub model: String,
    /// Its context window in tokens.
    pub window: u64,
}

/// Host-owned facts used to resolve one concrete model route.
///
/// `endpoint` is the base URL used by the selected model; a local Ollama
/// route accepts either its native root or its `/v1` OpenAI-compatible base.
/// A supplied `discovery` request takes precedence and can carry host-owned
/// headers, listing URLs, pinned-provider data, and timeouts. `catalog_window`
/// is a host catalog hint; managed model defaults are supplied separately so
/// this generic crate never embeds a managed provider's model id.
#[derive(Clone, Default)]
pub struct ContextWindowRouteInput {
    /// A non-zero user override.
    pub config_override: Option<u64>,
    /// Inference endpoint, used to build a default limits request when needed.
    pub endpoint: Option<String>,
    /// Host-built request for a route with custom discovery requirements.
    pub discovery: Option<DiscoveryRequest>,
    /// Host-owned pricing or model catalog hint.
    pub catalog_window: Option<u64>,
    /// Optional host-owned managed default model mapping.
    pub managed_default: Option<ManagedDefaultWindow>,
    /// Context window selected by the host's local runtime profile logic.
    pub local_window: Option<u64>,
}

impl std::fmt::Debug for ContextWindowRouteInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContextWindowRouteInput")
            .field("config_override", &self.config_override)
            .field("endpoint", &self.endpoint.as_ref().map(|_| "[redacted]"))
            .field("discovery", &self.discovery.as_ref().map(|_| "[redacted]"))
            .field("catalog_window", &self.catalog_window)
            .field(
                "managed_default",
                &self.managed_default.as_ref().map(|hint| hint.window),
            )
            .field("local_window", &self.local_window)
            .finish()
    }
}

/// Where a resolved context window came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSource {
    /// An explicit host configuration override.
    ConfigOverride,
    /// Provider listing or pinned endpoint metadata.
    ProviderReported,
    /// A smaller window learned from a provider overflow response.
    LearnedFromOverflow,
    /// Local runtime profile fallback.
    LocalProfile,
    /// A host-supplied static model hint.
    StaticGuess,
    /// No usable window was known.
    Unknown,
}

impl WindowSource {
    /// Stable label for host diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::ConfigOverride => "config_override",
            Self::ProviderReported => "provider_reported",
            Self::LearnedFromOverflow => "learned_from_overflow",
            Self::LocalProfile => "local_profile",
            Self::StaticGuess => "static_guess",
            Self::Unknown => "unknown",
        }
    }
}

/// A context window and its provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedWindow {
    /// Token window, or None when unknown.
    pub window: Option<u64>,
    /// Source selected by precedence.
    pub source: WindowSource,
    /// Positive provider-reported output ceiling, including partial discovery.
    pub max_output_tokens: Option<u64>,
}

/// Inputs already resolved by the host's routing/configuration adapter.
/// Debug output omits discovery details, which may contain credentials.
#[derive(Clone, Default)]
pub struct ContextWindowInput<'a> {
    /// A non-zero explicit override takes precedence without discovery.
    pub config_override: Option<u64>,
    /// Listing request for the endpoint/model actually used by inference.
    pub discovery: Option<&'a DiscoveryRequest>,
    /// Static model estimate, including any host-specific tier/catalog hints.
    pub static_window: Option<u64>,
    /// Whether a local runtime profile was selected. A local profile can
    /// remain unknown, distinct from an unknown external model.
    pub has_local_profile: bool,
    /// Local fallback used only when no static estimate is known.
    pub local_window: Option<u64>,
}

impl std::fmt::Debug for ContextWindowInput<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContextWindowInput")
            .field("config_override", &self.config_override)
            .field("discovery", &self.discovery.map(|_| "[redacted]"))
            .field("static_window", &self.static_window)
            .field("has_local_profile", &self.has_local_profile)
            .field("local_window", &self.local_window)
            .finish()
    }
}

/// Instance-owned route/model and discovery-identity precedence resolution.
///
/// Provider-sourced output caps are remembered independently of context windows.
/// Only provider-sourced and learned windows are remembered. Explicit host
/// overrides and fallback guesses are evaluated anew on every call. Keys are
/// trimmed and ASCII-folded. A discovery identity additionally includes its
/// endpoint, target model and cache variant (listing/probe/provider/credential
/// scope), preventing fallback across different resolved requests. Debug output
/// reports counts only, omitting discovery identities that may contain secrets.
#[derive(Default)]
pub struct ContextWindowResolver {
    state: Mutex<ResolverState>,
}

#[derive(Debug, Default)]
struct ResolverState {
    generation: u64,
    remembered: HashMap<RememberedKey, RememberedLimits>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RememberedKey {
    route: (String, String),
    discovery: Option<(String, String, String)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RememberedLimits {
    window: Option<(u64, WindowSource)>,
    max_output_tokens: Option<u64>,
}

impl std::fmt::Debug for ContextWindowResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        formatter
            .debug_struct("ContextWindowResolver")
            .field("generation", &state.generation)
            .field("remembered_count", &state.remembered.len())
            .finish_non_exhaustive()
    }
}

impl ContextWindowResolver {
    /// Creates empty state. The host owns the resolver's runtime lifetime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the smallest remembered provider-sourced window for a route/model.
    ///
    /// This backward-compatible identity-free projection is conservative across
    /// observed discovery variants. It does not select an endpoint or credential;
    /// [`Self::resolve`] with discovery uses only that exact request identity.
    pub fn remembered_window(&self, provider: &str, model: &str) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .projection(&key(provider, model))
            .window
            .map(|(window, _)| window)
    }

    /// Clears remembered limits when the host replaces runtime identity.
    /// Pending discovery may finish, but cannot repopulate the newer identity
    /// or publish stale results into the caller's shared discovery cache.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.generation = state.generation.wrapping_add(1);
        state.remembered.clear();
    }

    /// Resolves a concrete provider route using host-owned configuration and
    /// transport policy, then the same precedence and identity memory as
    /// [`Self::resolve`].
    ///
    /// The host supplies the endpoint or a complete discovery request, its
    /// explicit/catalog/managed model hints, local profile result, shared
    /// cache, and a factory for the HTTP client configured with its CA and
    /// proxy policy. A local Ollama route forces the native `/api/show` probe
    /// and strips a route temperature suffix from the probed model id. Other
    /// endpoints use the existing Ollama URL auto-detection. The factory is
    /// called only when discovery can run; overrides and routes without an
    /// endpoint do not construct a client.
    ///
    /// This route-level method emits the final source diagnostics and warns
    /// once per external model when the result is a static guess or unknown.
    /// It never logs the endpoint or discovery headers.
    pub async fn resolve_for_route<F>(
        &self,
        provider: &str,
        model: &str,
        local_kind: Option<LocalProviderKind>,
        route: ContextWindowRouteInput,
        cache: &ModelLimitsCache,
        fetcher_factory: F,
    ) -> ResolvedWindow
    where
        F: FnOnce() -> reqwest::Client,
    {
        let prepared = PreparedRoute::new(model, local_kind, route);
        let needs_fetcher = prepared.config_override.is_none_or(|window| window == 0)
            && prepared.discovery.is_some();
        let fetcher = needs_fetcher.then(|| ReqwestListingFetcher::new(fetcher_factory()));
        let unavailable = UnavailableRouteFetcher;
        let fetcher: &dyn ModelListingFetcher = fetcher
            .as_ref()
            .map_or(&unavailable, |fetcher| fetcher as &dyn ModelListingFetcher);
        self.resolve_prepared_route(provider, model, local_kind, prepared, fetcher, cache)
            .await
    }

    /// Resolves a concrete provider route with a host-supplied listing fetcher.
    ///
    /// This has the same route normalization, precedence, identity memory and
    /// diagnostics as [`Self::resolve_for_route`], while letting hosts inject a
    /// fake or policy-configured [`ModelListingFetcher`] directly. It is useful
    /// when a host already owns a fetcher and must keep tests offline.
    pub async fn resolve_for_route_with_fetcher(
        &self,
        provider: &str,
        model: &str,
        local_kind: Option<LocalProviderKind>,
        route: ContextWindowRouteInput,
        cache: &ModelLimitsCache,
        fetcher: &dyn ModelListingFetcher,
    ) -> ResolvedWindow {
        let prepared = PreparedRoute::new(model, local_kind, route);
        self.resolve_prepared_route(provider, model, local_kind, prepared, fetcher, cache)
            .await
    }

    async fn resolve_prepared_route(
        &self,
        provider: &str,
        model: &str,
        local_kind: Option<LocalProviderKind>,
        prepared: PreparedRoute,
        fetcher: &dyn ModelListingFetcher,
        cache: &ModelLimitsCache,
    ) -> ResolvedWindow {
        let input = ContextWindowInput {
            config_override: prepared.config_override,
            discovery: prepared.discovery.as_ref(),
            static_window: prepared.static_window,
            has_local_profile: local_kind.is_some(),
            local_window: prepared.local_window,
        };
        let resolved = self.resolve(provider, model, input, fetcher, cache).await;

        let route_label = provider.split_once(':').map_or(provider, |(slug, _)| slug);
        let route_label = if !route_label.is_empty()
            && route_label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
        {
            route_label
        } else {
            "provider"
        };
        if local_kind.is_none()
            && matches!(
                resolved.source,
                WindowSource::StaticGuess | WindowSource::Unknown
            )
        {
            warn_static_guess(model, resolved.window, route_label);
        }
        if prepared.discovery.is_some()
            && matches!(
                resolved.source,
                WindowSource::ProviderReported | WindowSource::LearnedFromOverflow
            )
        {
            tracing::info!(
                route = route_label,
                model,
                context_window = resolved.window,
                max_output_tokens = ?resolved.max_output_tokens,
                source = resolved.source.label(),
                "[model_context] context window from provider"
            );
        }
        tracing::debug!(
            route = route_label,
            model,
            context_window = ?resolved.window,
            max_output_tokens = ?resolved.max_output_tokens,
            source = resolved.source.label(),
            "[model_context] resolved context window"
        );
        resolved
    }

    /// Resolves override > discovered/learned > remembered > static/local > unknown.
    ///
    /// Discovery failures retain provenance and output ceilings only for the
    /// matching endpoint/model/cache variant. Without a discovery request, the
    /// route/model lookup projects the smallest observed window and output cap;
    /// callers requiring exact routed facts must supply the resolved request.
    /// Partial discovery can supply an output ceiling independently of the window.
    /// A positive override performs no network discovery, but preserves a positive
    /// output ceiling already in the matching discovery-cache slot or remembered
    /// state. It never remembers the explicit context override.
    /// Discovery bounds/caching come from the request and `ModelLimitsCache`. No diagnostics
    /// or telemetry are emitted: hosts use the returned source label.
    pub async fn resolve(
        &self,
        provider: &str,
        model: &str,
        input: ContextWindowInput<'_>,
        fetcher: &dyn ModelListingFetcher,
        cache: &ModelLimitsCache,
    ) -> ResolvedWindow {
        let route_key = remembered_key(provider, model, input.discovery);
        let (generation, remembered_cap) = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            let cached_cap = if input.config_override.is_some_and(|window| window > 0) {
                // Cache-only lookup under the same generation guard as discovery:
                // no fetch and no remembered-window write for explicit overrides.
                input.discovery.and_then(|request| {
                    cache
                        .get_variant(&request.endpoint, &request.model, &request.cache_variant())
                        .effective()
                        .and_then(|limits| limits.max_output_tokens)
                        .filter(|tokens| *tokens > 0)
                })
            } else {
                None
            };
            (
                state.generation,
                cached_cap.or(state.lookup(&route_key).max_output_tokens),
            )
        };
        if let Some(window) = input.config_override.filter(|window| *window > 0) {
            return ResolvedWindow {
                window: Some(window),
                source: WindowSource::ConfigOverride,
                max_output_tokens: remembered_cap,
            };
        }
        let limits = if let Some(request) = input.discovery {
            discover_model_limits_with_guard(fetcher, cache, request, &|operation| {
                let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
                if state.generation != generation {
                    return false;
                }
                operation();
                true
            })
            .await
        } else {
            None
        };
        let max_output_tokens = limits
            .as_ref()
            .and_then(|limits| limits.max_output_tokens)
            .filter(|tokens| *tokens > 0);
        let discovered_window = limits.as_ref().and_then(|limits| {
            limits
                .context_window
                .filter(|window| *window > 0)
                .map(|window| {
                    let source = match limits.source {
                        LimitSource::LearnedFromOverflow => WindowSource::LearnedFromOverflow,
                        _ => WindowSource::ProviderReported,
                    };
                    (window, source)
                })
        });
        let remembered = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.generation == generation {
                if discovered_window.is_some() || max_output_tokens.is_some() {
                    let entry = state.remembered.entry(route_key.clone()).or_default();
                    entry.window = discovered_window.or(entry.window);
                    entry.max_output_tokens = max_output_tokens.or(entry.max_output_tokens);
                }
                Some(state.lookup(&route_key))
            } else {
                None
            }
        };
        let max_output_tokens =
            max_output_tokens.or(remembered.and_then(|limits| limits.max_output_tokens));
        if let Some((window, source)) =
            discovered_window.or(remembered.and_then(|limits| limits.window))
        {
            return ResolvedWindow {
                window: Some(window),
                source,
                max_output_tokens,
            };
        }
        let (window, source) =
            if let Some(window) = input.static_window.filter(|window| *window > 0) {
                (Some(window), WindowSource::StaticGuess)
            } else if input.has_local_profile {
                (
                    input.local_window.filter(|window| *window > 0),
                    WindowSource::LocalProfile,
                )
            } else {
                (None, WindowSource::Unknown)
            };
        ResolvedWindow {
            window,
            source,
            max_output_tokens,
        }
    }
}

struct PreparedRoute {
    config_override: Option<u64>,
    discovery: Option<DiscoveryRequest>,
    static_window: Option<u64>,
    local_window: Option<u64>,
}

impl PreparedRoute {
    fn new(
        model: &str,
        local_kind: Option<LocalProviderKind>,
        route: ContextWindowRouteInput,
    ) -> Self {
        let ollama = local_kind == Some(LocalProviderKind::Ollama);
        let probe_model = if ollama {
            model.split_once('@').map_or(model, |(id, _)| id).trim()
        } else {
            model.trim()
        };
        let mut discovery = route.discovery.or_else(|| {
            route.endpoint.map(|endpoint| {
                let endpoint = if ollama {
                    ollama_openai_endpoint(&endpoint)
                } else {
                    endpoint
                };
                DiscoveryRequest::new(endpoint, probe_model)
            })
        });
        if ollama && let Some(request) = discovery.as_mut() {
            request.model = probe_model.to_owned();
            request.ollama_native = Some(true);
        }

        let managed_window = route.managed_default.as_ref().and_then(|hint| {
            (hint.model.trim() == model.trim() && hint.window > 0).then_some(hint.window)
        });
        let static_window = managed_window
            .or_else(|| workload::context_window_for_model(model))
            .or(route.catalog_window.filter(|window| *window > 0))
            .or_else(|| context_window_for_model_id(model));

        Self {
            config_override: route.config_override,
            discovery,
            static_window,
            local_window: route.local_window,
        }
    }
}

fn warn_static_guess(model: &str, window: Option<u64>, route: &str) {
    let first = first_static_guess_for_model(model);
    if first {
        tracing::warn!(
            model,
            route,
            context_window = ?window,
            "[model_context] context window is a static guess — the provider did not report one"
        );
    } else {
        tracing::debug!(
            model,
            route,
            context_window = ?window,
            "[model_context] context window is a static guess"
        );
    }
}

fn first_static_guess_for_model(model: &str) -> bool {
    static WARNED: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    WARNED
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(model.to_owned())
}

fn ollama_openai_endpoint(endpoint: &str) -> String {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.ends_with("/v1") {
        endpoint.to_owned()
    } else {
        format!("{endpoint}/v1")
    }
}

struct UnavailableRouteFetcher;

#[async_trait::async_trait]
impl ModelListingFetcher for UnavailableRouteFetcher {
    async fn get_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
    ) -> crate::Result<serde_json::Value> {
        Err(crate::Error::Catalog(
            "route discovery is unavailable".to_owned(),
        ))
    }
}

impl ResolverState {
    fn lookup(&self, key: &RememberedKey) -> RememberedLimits {
        if key.discovery.is_some() {
            self.remembered.get(key).copied().unwrap_or_default()
        } else {
            self.projection(&key.route)
        }
    }

    fn projection(&self, route: &(String, String)) -> RememberedLimits {
        let mut result = RememberedLimits::default();
        for limits in self
            .remembered
            .iter()
            .filter_map(|(key, limits)| (key.route == *route).then_some(limits))
        {
            if let Some(window) = limits.window
                && result.window.is_none_or(|current| window.0 < current.0)
            {
                result.window = Some(window);
            }
            if let Some(cap) = limits.max_output_tokens {
                result.max_output_tokens =
                    Some(result.max_output_tokens.map_or(cap, |old| old.min(cap)));
            }
        }
        result
    }
}

fn remembered_key(
    provider: &str,
    model: &str,
    discovery: Option<&DiscoveryRequest>,
) -> RememberedKey {
    RememberedKey {
        route: key(provider, model),
        discovery: discovery.map(|request| {
            (
                request
                    .endpoint
                    .trim()
                    .trim_end_matches('/')
                    .to_ascii_lowercase(),
                request
                    .model
                    .trim()
                    .trim_end_matches('/')
                    .to_ascii_lowercase(),
                request.cache_variant(),
            )
        }),
    }
}

fn key(provider: &str, model: &str) -> (String, String) {
    (
        provider.trim().to_ascii_lowercase(),
        model.trim().to_ascii_lowercase(),
    )
}

#[cfg(test)]
#[path = "context_window_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "context_window_review_tests.rs"]
mod review_tests;

#[cfg(test)]
#[path = "context_window_route_tests.rs"]
mod route_tests;
