//! Provider-route grammar shared by hosts and provider factories.

/// Provider kinds that can run on a host-local inference runtime.
///
/// This mirrors the provider names owned by `tinyinference-local` without
/// making this lower-level crate depend on that crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalProviderKind {
    /// Ollama native runtime.
    Ollama,
    /// LM Studio runtime.
    LmStudio,
    /// MLX-compatible local server.
    Mlx,
    /// OMLX local server.
    Omlx,
    /// Generic local OpenAI-compatible endpoint.
    LocalOpenai,
}

impl LocalProviderKind {
    /// Returns the canonical provider slug for this local runtime.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "lmstudio",
            Self::Mlx => "mlx",
            Self::Omlx => "omlx",
            Self::LocalOpenai => "local-openai",
        }
    }

    fn parse(slug: &str) -> Option<Self> {
        match slug.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Self::LmStudio),
            "mlx" | "mlx-server" | "mlx_lm" => Some(Self::Mlx),
            "omlx" | "omlx-server" => Some(Self::Omlx),
            "local-openai" | "local_openai" | "custom-openai" | "custom_openai" | "llamacpp"
            | "llama.cpp" | "vllm" => Some(Self::LocalOpenai),
            _ => None,
        }
    }
}

/// The semantic class of a parsed provider route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteKind {
    /// A route served by the host's managed inference gateway.
    Managed,
    /// A route handled by a local inference runtime.
    Local(LocalProviderKind),
    /// A route handled by the Claude Code command-line provider.
    ClaudeCode,
    /// A route handled by the Claude Agent SDK provider.
    ClaudeAgentSdk,
    /// A caller-configured cloud provider slug.
    Cloud {
        /// Provider slug as written in the route.
        slug: String,
    },
    /// A host routing sentinel such as `cloud` that still needs resolution.
    ///
    /// A sentinel is not a concrete provider route and is therefore not
    /// classified as external by [`ProviderRoute::is_external`]. Hosts should
    /// resolve it before constructing a provider or recording egress.
    Sentinel,
}

/// A provider identifier and its case-sensitive model selection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRoute {
    /// Host-selected provider slug; no case normalization is applied.
    pub provider: String,
    /// Model identifier without a successfully parsed temperature suffix.
    pub model: String,
    /// Finite per-route temperature override, when supplied.
    pub temperature: Option<f64>,
}

impl ProviderRoute {
    /// Parses `<provider>:<model>[@<temperature>]`.
    ///
    /// Splits at the first colon so model namespaces and tags survive. Routes
    /// without a colon are left to host sentinel/default handling. Empty
    /// models are retained for the provider builder to validate.
    #[must_use]
    pub fn parse(route: &str) -> Option<Self> {
        let (provider, model) = route.trim().split_once(':')?;
        let (model, temperature) = split_model_and_temperature(model);
        Some(Self {
            provider: provider.trim().to_owned(),
            model,
            temperature,
        })
    }

    /// Classifies this route without depending on host-specific configuration.
    #[must_use]
    pub fn kind(&self) -> RouteKind {
        let slug = self.provider.trim();
        let normalized = slug.to_ascii_lowercase();
        match normalized.as_str() {
            "openhuman" => RouteKind::Managed,
            "cloud" | "" => RouteKind::Sentinel,
            "claude-code" => RouteKind::ClaudeCode,
            "claude_agent_sdk" | "claude-agent-sdk" => RouteKind::ClaudeAgentSdk,
            _ => LocalProviderKind::parse(slug).map_or_else(
                || RouteKind::Cloud {
                    slug: self.provider.clone(),
                },
                RouteKind::Local,
            ),
        }
    }

    /// Returns whether inference for this route leaves the host device.
    ///
    /// Local runtimes are not external. Managed and cloud routes, along with
    /// the Claude subprocess providers, are external. An unresolved sentinel
    /// returns `false` because it has no concrete destination yet; hosts must
    /// resolve it before construction and egress accounting.
    #[must_use]
    pub fn is_external(&self) -> bool {
        matches!(
            self.kind(),
            RouteKind::Managed
                | RouteKind::ClaudeCode
                | RouteKind::ClaudeAgentSdk
                | RouteKind::Cloud { .. }
        )
    }
}

/// Splits a `custom:<endpoint>` provider value into its slug and endpoint.
///
/// Other values are returned unchanged with no endpoint. The endpoint portion
/// is preserved verbatim, including any URL path or query.
#[must_use]
pub fn split_custom_provider(provider: &str) -> (String, Option<String>) {
    if let Some(endpoint) = provider.strip_prefix("custom:") {
        ("custom".to_owned(), Some(endpoint.to_owned()))
    } else {
        (provider.to_owned(), None)
    }
}

/// Builds the stable provider string from a slug and optional custom endpoint.
#[must_use]
pub fn custom_provider_string(slug: &str, endpoint: Option<&str>) -> String {
    match endpoint {
        Some(endpoint) => format!("{slug}:{endpoint}"),
        None => slug.to_owned(),
    }
}

/// Converts a bare local model name to the default Ollama route.
///
/// Names containing a colon are treated as explicit provider routes and
/// returned unchanged. The bare `openhuman` sentinel also stays unchanged;
/// other names are routed to Ollama. This follows the `/v1` host behavior,
/// where a colon-bearing model string is considered already qualified.
#[must_use]
pub fn bare_model_route(name: &str) -> String {
    if name.contains(':') || name == "openhuman" {
        name.to_owned()
    } else {
        format!("ollama:{name}")
    }
}

/// Returns the model id and provider owner represented by a route.
///
/// Explicit routes keep their provider prefix and model namespace/tag, remove
/// a trailing suffix accepted by Rust's `f64` parser, and use their provider
/// slug as `owned_by`. Unqualified model ids are owned by the managed
/// `openhuman` route. The suffix handling matches the existing `/v1` model
/// catalog behavior, including values such as `NaN` and `inf`.
#[must_use]
pub fn route_owner(route: &str) -> (String, String) {
    let route = route.trim();
    let (id, owned_by) = match route.split_once(':') {
        Some((provider, model)) => (
            format!("{}:{}", provider.trim(), strip_temperature_suffix(model)),
            provider.trim(),
        ),
        None => (strip_temperature_suffix(route).to_owned(), "openhuman"),
    };
    (id, owned_by.to_owned())
}

fn strip_temperature_suffix(model: &str) -> &str {
    let trimmed = model.trim();
    let Some((head, tail)) = trimmed.rsplit_once('@') else {
        return trimmed;
    };
    if tail.parse::<f64>().is_ok() {
        head.trim()
    } else {
        trimmed
    }
}

/// Parse a `<model>[@<temp>]` tail into `(model, override)`.
///
/// Tolerates whitespace around the components. Returns `temperature = None`
/// when the suffix is absent or unparseable — the model text is taken as-is.
pub fn split_model_and_temperature(raw: &str) -> (String, Option<f64>) {
    let trimmed = raw.trim();
    if let Some(at_pos) = trimmed.rfind('@') {
        let head = trimmed[..at_pos].trim();
        let tail = trimmed[at_pos + 1..].trim();
        if !head.is_empty()
            && let Ok(parsed) = tail.parse::<f64>()
            && parsed.is_finite()
        {
            return (head.to_string(), Some(parsed));
        }
    }
    (trimmed.to_string(), None)
}

#[cfg(test)]
#[path = "route_tests.rs"]
mod tests;
