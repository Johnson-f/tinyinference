//! Token usage accounting types.

use serde::{Deserialize, Serialize};

/// Provider usage with an exact USD charge and normalized token counts.
///
/// `charged_amount_usd` retains provider precision for hosts that persist
/// fractional micro-dollar values. The normalized [`Usage`] projection keeps
/// the existing integer-micros representation for compatibility.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BilledUsage {
    /// Token counts and optional provider metadata.
    pub usage: Usage,
    /// Provider-reported or estimated amount in USD. Zero when unavailable.
    pub charged_amount_usd: f64,
    /// Whether the value is a local estimate rather than a provider charge.
    pub cost_is_estimate: bool,
    /// Whether a provider explicitly reported the charge, including zero.
    pub charge_reported: bool,
}

impl BilledUsage {
    /// Creates usage from input and output counts with no known charge.
    pub fn from_counts(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            usage: Usage::new(input_tokens, output_tokens),
            ..Self::default()
        }
    }

    /// Returns cached input tokens.
    pub fn cached_input_tokens(&self) -> u64 {
        self.usage.cache_read_tokens
    }

    /// Returns the context window, or zero when it is unknown.
    pub fn context_window(&self) -> u64 {
        self.usage.context_window_tokens.unwrap_or(0)
    }

    /// Sets cached input tokens.
    pub fn with_cached_input_tokens(mut self, tokens: u64) -> Self {
        self.usage.cache_read_tokens = tokens;
        self
    }

    /// Sets the context window; zero means unknown.
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.usage.context_window_tokens = (tokens > 0).then_some(tokens);
        self
    }

    /// Sets cache-creation tokens.
    pub fn with_cache_creation_tokens(mut self, tokens: u64) -> Self {
        self.usage.cache_creation_tokens = tokens;
        self
    }

    /// Sets reasoning tokens.
    pub fn with_reasoning_tokens(mut self, tokens: u64) -> Self {
        self.usage.reasoning_tokens = tokens;
        self
    }

    /// Sets an exact provider-reported USD charge. Zero is a known free call.
    pub fn with_charged_usd(mut self, usd: f64) -> Self {
        self.charged_amount_usd = usd;
        self.cost_is_estimate = false;
        self.charge_reported = true;
        self
    }

    /// Sets a provider-reported charge when present; `None` leaves it unknown.
    pub fn with_reported_charge(self, usd: Option<f64>) -> Self {
        match usd {
            Some(usd) => self.with_charged_usd(usd),
            None => self,
        }
    }

    /// Sets a local USD estimate, which is never projected as a provider charge.
    pub fn with_estimated_usd(mut self, usd: f64) -> Self {
        self.charged_amount_usd = usd;
        self.cost_is_estimate = true;
        self.charge_reported = false;
        self
    }
}

impl std::ops::Deref for BilledUsage {
    type Target = Usage;

    fn deref(&self) -> &Usage {
        &self.usage
    }
}

/// A provider-reported charge expressed in integer micro-units.
///
/// The inference boundary carries the provider's measured charge but deliberately
/// does not calculate prices. Hosts retain pricing policy and convert their
/// provider DTOs into this fixed-point representation at the adapter boundary.
/// The host selects the unit currency; a call sequence combined into one
/// [`Usage`] must therefore have a common currency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChargedAmount {
    /// Charge in one-millionth host-selected currency units.
    pub micros: i64,
}

impl ChargedAmount {
    /// Creates an amount from integer micro-units.
    #[must_use]
    pub fn new(micros: i64) -> Self {
        Self { micros }
    }

    /// Creates a USD amount from integer micro-dollars.
    #[must_use]
    pub fn usd_micros(micros: i64) -> Self {
        Self::new(micros)
    }
}

/// Normalized token usage for a single model call.
///
/// Providers expose different breakdowns; fields default to zero so partial
/// data still produces a valid record. Detail fields (cache, reasoning) do not
/// need to sum to the totals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt/input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Completion/output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Total tokens (input + output) as reported by the provider when known.
    #[serde(default)]
    pub total_tokens: u64,
    /// Input tokens served from a provider prompt/KV cache.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Input tokens written into a provider prompt/KV cache.
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Reasoning/thinking output tokens when the provider exposes them.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Provider-reported charge for this call, when the provider exposes one.
    ///
    /// This is measured metadata, not a local price calculation. Hosts own
    /// pricing policy and may leave it absent for local or unmetered providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charged_amount: Option<ChargedAmount>,
    /// The model context window used for this call, when the provider reports
    /// it or the host resolved it authoritatively.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window_tokens: Option<u64>,
}

/// Aggregate usage across many calls, tracking both the call count and the
/// summed [`Usage`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageTotals {
    /// Number of accumulated calls.
    #[serde(default)]
    pub calls: u64,
    /// Summed usage across all accumulated calls.
    #[serde(default)]
    pub usage: Usage,
}
