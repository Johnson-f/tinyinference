use super::*;

const ENDPOINT: &str = "https://openrouter.ai/api/v1";
const MODEL: &str = "deepseek/deepseek-v4.1-flash";

fn limits(window: u64) -> ModelLimits {
    ModelLimits {
        context_window: Some(window),
        max_output_tokens: Some(8_192),
        source: LimitSource::ProviderListing,
    }
}

fn cache() -> ModelLimitsCache {
    ModelLimitsCache::new(
        Duration::from_secs(100),
        Duration::from_secs(10),
        Duration::from_secs(1_000),
    )
}

#[test]
fn discovered_entry_expires_after_ttl() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_discovered_at(ENDPOINT, MODEL, Some(limits(1_048_576)), t0);

    let fresh = cache.get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(99));
    assert_eq!(fresh.discovered, Some(Some(limits(1_048_576))));
    assert_eq!(fresh.effective().unwrap().context_window, Some(1_048_576));

    let stale = cache.get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(100));
    assert_eq!(stale.discovered, None);
    assert_eq!(stale.effective(), None);
}

#[test]
fn negative_entry_uses_shorter_ttl() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_discovered_at(ENDPOINT, MODEL, None, t0);
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(9))
            .discovered,
        Some(None)
    );
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(10))
            .discovered,
        None
    );
}

#[test]
fn keys_are_normalized() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_discovered_at(
        "HTTPS://OpenRouter.ai/api/v1/",
        MODEL,
        Some(limits(5_000)),
        t0,
    );
    assert!(
        cache
            .get_at(ENDPOINT, &MODEL.to_uppercase(), t0)
            .discovered
            .is_some()
    );
}

#[test]
fn learned_window_lowers_discovered_window() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_discovered_at(ENDPOINT, MODEL, Some(limits(1_048_576)), t0);
    cache.insert_learned_at(ENDPOINT, MODEL, 163_840, t0);
    let effective = cache.get_at(ENDPOINT, MODEL, t0).effective().unwrap();
    assert_eq!(effective.context_window, Some(163_840));
    assert_eq!(effective.max_output_tokens, Some(8_192));
    assert_eq!(effective.source, LimitSource::LearnedFromOverflow);
}

#[test]
fn learned_window_never_raises_discovered_window() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_discovered_at(ENDPOINT, MODEL, Some(limits(128_000)), t0);
    cache.insert_learned_at(ENDPOINT, MODEL, 200_000, t0);
    let effective = cache.get_at(ENDPOINT, MODEL, t0).effective().unwrap();
    assert_eq!(effective.context_window, Some(128_000));
    assert_eq!(effective.source, LimitSource::ProviderListing);
}

#[test]
fn learned_window_stands_alone_without_discovery() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_learned_at(ENDPOINT, MODEL, 131_072, t0);
    let state = cache.get_at(ENDPOINT, MODEL, t0);
    assert_eq!(state.discovered, None);
    assert_eq!(state.effective().unwrap().context_window, Some(131_072));
    // Expires on its own TTL.
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(1_000))
            .learned_context_window,
        None
    );
}

#[test]
fn smaller_learned_window_is_kept_until_it_expires() {
    let cache = cache();
    let t0 = Instant::now();
    cache.insert_learned_at(ENDPOINT, MODEL, 100_000, t0);
    cache.insert_learned_at(ENDPOINT, MODEL, 150_000, t0 + Duration::from_secs(1));
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(2))
            .learned_context_window,
        Some(100_000)
    );
    cache.insert_learned_at(ENDPOINT, MODEL, 90_000, t0 + Duration::from_secs(3));
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(4))
            .learned_context_window,
        Some(90_000)
    );
    cache.insert_learned_at(ENDPOINT, MODEL, 150_000, t0 + Duration::from_secs(2_000));
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, t0 + Duration::from_secs(2_001))
            .learned_context_window,
        Some(150_000)
    );
}

#[test]
fn clear_drops_everything() {
    let cache = cache();
    cache.insert_discovered(ENDPOINT, MODEL, Some(limits(1)));
    cache.clear();
    assert_eq!(cache.get(ENDPOINT, MODEL), CachedLimits::default());
}
