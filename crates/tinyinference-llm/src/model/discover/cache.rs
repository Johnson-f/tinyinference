//! In-process cache of discovered and learned model limits.
//!
//! Keyed by `(endpoint, model)`, both normalized (trimmed, trailing `/`
//! dropped, lowercased) so the chat adapter and the discovery caller agree on
//! the key without coordinating.
//!
//! Each key holds two independent facts:
//!
//! - **discovered**: what the provider's listing said, with a TTL. A failed
//!   discovery is cached too (as "nothing found") with a shorter TTL, so an
//!   endpoint without a listing is not re-probed on every turn.
//! - **learned**: the window the provider stated in a context-overflow error.
//!   It is a correction from the endpoint that actually answered, so it is
//!   kept longer and only ever lowers the effective window.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::types::{LimitSource, ModelLimits};

/// How long a discovered limit stays fresh.
pub const DEFAULT_DISCOVERED_TTL: Duration = Duration::from_secs(60 * 60);
/// How long a failed discovery is remembered before re-probing.
pub const DEFAULT_NEGATIVE_TTL: Duration = Duration::from_secs(5 * 60);
/// How long a limit learned from an overflow error is kept.
pub const DEFAULT_LEARNED_TTL: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Clone, Debug)]
struct Discovered {
    limits: Option<ModelLimits>,
    at: Instant,
}

#[derive(Clone, Debug)]
struct Learned {
    context_window: u64,
    at: Instant,
}

#[derive(Clone, Debug, Default)]
struct Entry {
    /// Discovered facts keyed by request variant (see
    /// [`DiscoveryRequest::cache_variant`](super::DiscoveryRequest::cache_variant)):
    /// the same model resolves differently under a different listing URL or
    /// pinned-provider set, so those must not share a slot. The learned
    /// overflow window below is shared by every variant.
    discovered: HashMap<String, Discovered>,
    learned: Option<Learned>,
}

/// The cached state for one `(endpoint, model)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CachedLimits {
    /// `Some(Some(_))`: a fresh discovery found limits. `Some(None)`: a fresh
    /// discovery found nothing (do not re-probe yet). `None`: no fresh
    /// discovery, so the caller should probe.
    pub discovered: Option<Option<ModelLimits>>,
    /// A fresh window learned from an overflow error.
    pub learned_context_window: Option<u64>,
}

impl CachedLimits {
    /// The best provider-sourced limits: the discovered ones, lowered by a
    /// learned overflow window when that is smaller (or standing in for them
    /// when discovery found nothing).
    #[must_use]
    pub fn effective(&self) -> Option<ModelLimits> {
        let discovered = self.discovered.clone().flatten();
        match (discovered, self.learned_context_window) {
            (Some(limits), Some(learned))
                if limits.context_window.is_none_or(|window| learned < window) =>
            {
                Some(ModelLimits {
                    context_window: Some(learned),
                    max_output_tokens: limits.max_output_tokens,
                    input_modalities: limits.input_modalities,
                    source: LimitSource::LearnedFromOverflow,
                })
            }
            (Some(limits), _) => Some(limits),
            (None, Some(learned)) => Some(ModelLimits {
                context_window: Some(learned),
                max_output_tokens: None,
                input_modalities: None,
                source: LimitSource::LearnedFromOverflow,
            }),
            (None, None) => None,
        }
    }
}

/// A TTL cache of model limits keyed by `(endpoint, model)`.
#[derive(Debug)]
pub struct ModelLimitsCache {
    entries: Mutex<HashMap<(String, String), Entry>>,
    discovered_ttl: Duration,
    negative_ttl: Duration,
    learned_ttl: Duration,
}

impl Default for ModelLimitsCache {
    fn default() -> Self {
        Self::new(
            DEFAULT_DISCOVERED_TTL,
            DEFAULT_NEGATIVE_TTL,
            DEFAULT_LEARNED_TTL,
        )
    }
}

/// Normalizes a cache key part.
fn normalize(part: &str) -> String {
    part.trim().trim_end_matches('/').to_ascii_lowercase()
}

impl ModelLimitsCache {
    /// A cache with explicit TTLs for discovered, failed and learned entries.
    #[must_use]
    pub fn new(discovered_ttl: Duration, negative_ttl: Duration, learned_ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            discovered_ttl,
            negative_ttl,
            learned_ttl,
        }
    }

    fn key(endpoint: &str, model: &str) -> (String, String) {
        (normalize(endpoint), normalize(model))
    }

    fn with_entries<R>(&self, f: impl FnOnce(&mut HashMap<(String, String), Entry>) -> R) -> R {
        // A poisoned lock only means another thread panicked mid-update; the
        // map is still a valid cache, so keep using it.
        let mut guard = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut guard)
    }

    /// The fresh cached state for `(endpoint, model)` across every discovery
    /// variant: the smallest fresh discovered window, lowered by any learned
    /// overflow window. Input modalities intersect fresh variant facts; any
    /// unknown variant makes the aggregate unknown. Use [`Self::get_variant`]
    /// for one request's slot.
    #[must_use]
    pub fn get(&self, endpoint: &str, model: &str) -> CachedLimits {
        self.get_at(endpoint, model, Instant::now())
    }

    /// [`Self::get`] evaluated at `now` (lets tests advance time without
    /// sleeping).
    #[must_use]
    pub fn get_at(&self, endpoint: &str, model: &str, now: Instant) -> CachedLimits {
        let key = Self::key(endpoint, model);
        self.with_entries(|entries| {
            let Some(entry) = entries.get(&key) else {
                return CachedLimits::default();
            };
            // Across variants: the smallest fresh found window (conservative),
            // else a fresh negative, else nothing.
            let mut best: Option<ModelLimits> = None;
            let mut negative = false;
            let mut modalities: Option<Vec<String>> = None;
            let mut saw_modalities = false;
            let mut unknown_modalities = false;
            for discovered in entry.discovered.values() {
                match self.fresh(discovered, now) {
                    Some(Some(limits)) => {
                        if let Some(values) = &limits.input_modalities {
                            if saw_modalities {
                                if let Some(common) = &mut modalities {
                                    common.retain(|value| values.contains(value));
                                }
                            } else {
                                modalities = Some(values.clone());
                                saw_modalities = true;
                            }
                        } else {
                            unknown_modalities = true;
                        }
                        let smaller = best.as_ref().is_none_or(|current| {
                            limits.context_window.unwrap_or(u64::MAX)
                                < current.context_window.unwrap_or(u64::MAX)
                        });
                        if smaller {
                            best = Some(limits);
                        }
                    }
                    Some(None) => {
                        negative = true;
                        unknown_modalities = true;
                    }
                    None => {}
                }
            }
            if let Some(best) = &mut best {
                best.input_modalities = if unknown_modalities {
                    None
                } else {
                    modalities.map(|mut values| {
                        values.sort();
                        values.dedup();
                        values
                    })
                };
            }
            CachedLimits {
                discovered: best.map(Some).or(negative.then_some(None)),
                learned_context_window: self.fresh_learned(entry, now),
            }
        })
    }

    fn fresh(&self, discovered: &Discovered, now: Instant) -> Option<Option<ModelLimits>> {
        let ttl = if discovered.limits.is_some() {
            self.discovered_ttl
        } else {
            self.negative_ttl
        };
        (now.saturating_duration_since(discovered.at) < ttl).then(|| discovered.limits.clone())
    }

    fn fresh_learned(&self, entry: &Entry, now: Instant) -> Option<u64> {
        entry
            .learned
            .as_ref()
            .filter(|learned| now.saturating_duration_since(learned.at) < self.learned_ttl)
            .map(|learned| learned.context_window)
    }

    /// [`Self::get`] for one discovery `variant`.
    #[must_use]
    pub fn get_variant(&self, endpoint: &str, model: &str, variant: &str) -> CachedLimits {
        self.get_variant_at(endpoint, model, variant, Instant::now())
    }

    /// [`Self::get_variant`] evaluated at `now`.
    #[must_use]
    pub fn get_variant_at(
        &self,
        endpoint: &str,
        model: &str,
        variant: &str,
        now: Instant,
    ) -> CachedLimits {
        let key = Self::key(endpoint, model);
        self.with_entries(|entries| {
            let Some(entry) = entries.get(&key) else {
                return CachedLimits::default();
            };
            let discovered = entry
                .discovered
                .get(variant)
                .and_then(|discovered| self.fresh(discovered, now));
            let learned_context_window = self.fresh_learned(entry, now);
            CachedLimits {
                discovered,
                learned_context_window,
            }
        })
    }

    /// Records a discovery result (`None` = nothing found).
    pub fn insert_discovered(&self, endpoint: &str, model: &str, limits: Option<ModelLimits>) {
        self.insert_discovered_at(endpoint, model, limits, Instant::now());
    }

    /// [`Self::insert_discovered`] stamped at `at`.
    pub fn insert_discovered_at(
        &self,
        endpoint: &str,
        model: &str,
        limits: Option<ModelLimits>,
        at: Instant,
    ) {
        self.insert_discovered_variant_at(endpoint, model, "", limits, at);
    }

    /// Records a discovery result for one discovery `variant`.
    pub fn insert_discovered_variant(
        &self,
        endpoint: &str,
        model: &str,
        variant: &str,
        limits: Option<ModelLimits>,
    ) {
        self.insert_discovered_variant_at(endpoint, model, variant, limits, Instant::now());
    }

    /// [`Self::insert_discovered_variant`] stamped at `at`.
    pub fn insert_discovered_variant_at(
        &self,
        endpoint: &str,
        model: &str,
        variant: &str,
        limits: Option<ModelLimits>,
        at: Instant,
    ) {
        let key = Self::key(endpoint, model);
        self.with_entries(|entries| {
            entries
                .entry(key)
                .or_default()
                .discovered
                .insert(variant.to_string(), Discovered { limits, at });
        });
    }

    /// Records a window the provider stated in an overflow error.
    ///
    /// A smaller learned window replaces a larger one; a larger one replaces
    /// a smaller one only once the smaller has expired, so one odd error
    /// cannot widen a window another endpoint proved smaller.
    pub fn insert_learned(&self, endpoint: &str, model: &str, context_window: u64) {
        self.insert_learned_at(endpoint, model, context_window, Instant::now());
    }

    /// [`Self::insert_learned`] stamped at `at`.
    pub fn insert_learned_at(&self, endpoint: &str, model: &str, context_window: u64, at: Instant) {
        let key = Self::key(endpoint, model);
        let learned_ttl = self.learned_ttl;
        self.with_entries(|entries| {
            let entry = entries.entry(key).or_default();
            let keep_existing = entry.learned.as_ref().is_some_and(|existing| {
                at.saturating_duration_since(existing.at) < learned_ttl
                    && existing.context_window <= context_window
            });
            if !keep_existing {
                entry.learned = Some(Learned { context_window, at });
            }
        });
    }

    /// Drops every entry.
    pub fn clear(&self) {
        self.with_entries(HashMap::clear);
    }
}

/// The process-wide cache shared by discovery callers and the chat adapters
/// that learn limits from overflow errors.
#[must_use]
pub fn model_limits_cache() -> &'static ModelLimitsCache {
    static CACHE: OnceLock<ModelLimitsCache> = OnceLock::new();
    CACHE.get_or_init(ModelLimitsCache::default)
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
