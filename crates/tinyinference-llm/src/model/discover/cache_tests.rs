use super::*;

const ENDPOINT: &str = "https://openrouter.ai/api/v1";
const MODEL: &str = "deepseek/deepseek-v4.1-flash";

fn limits(window: u64) -> ModelLimits {
    ModelLimits {
        context_window: Some(window),
        max_output_tokens: Some(8_192),
        input_modalities: None,
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

fn variant(window: u64, modalities: Option<&[&str]>) -> ModelLimits {
    ModelLimits {
        input_modalities: modalities
            .map(|values| values.iter().map(|value| (*value).to_owned()).collect()),
        ..limits(window)
    }
}

#[test]
fn aggregate_modalities_intersect_independently_of_window_and_ties() {
    for windows in [(32_000, 64_000), (64_000, 32_000), (32_000, 32_000)] {
        let cache = cache();
        let now = Instant::now();
        cache.insert_discovered_variant_at(
            ENDPOINT,
            MODEL,
            "image",
            Some(variant(windows.0, Some(&["text", "image"]))),
            now,
        );
        cache.insert_discovered_variant_at(
            ENDPOINT,
            MODEL,
            "text",
            Some(variant(windows.1, Some(&["text"]))),
            now,
        );
        let aggregate = cache.get_at(ENDPOINT, MODEL, now).effective().unwrap();
        assert_eq!(aggregate.context_window, Some(windows.0.min(windows.1)));
        assert_eq!(aggregate.input_modalities, Some(vec!["text".into()]));
        assert_eq!(
            cache
                .get_variant_at(ENDPOINT, MODEL, "image", now)
                .effective()
                .unwrap()
                .input_modalities,
            Some(vec!["text".into(), "image".into()])
        );
    }
}

#[test]
fn aggregate_modalities_preserve_unknown_negative_empty_and_expiration() {
    let cache = cache();
    let now = Instant::now();
    cache.insert_discovered_variant_at(
        ENDPOINT,
        MODEL,
        "image",
        Some(variant(32_000, Some(&["image", "text"]))),
        now,
    );
    cache.insert_discovered_variant_at(
        ENDPOINT,
        MODEL,
        "unknown",
        Some(variant(64_000, None)),
        now,
    );
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, now)
            .effective()
            .unwrap()
            .input_modalities,
        None
    );
    cache.insert_discovered_variant_at(
        ENDPOINT,
        MODEL,
        "unknown",
        Some(variant(64_000, Some(&[]))),
        now,
    );
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, now)
            .effective()
            .unwrap()
            .input_modalities,
        Some(vec![])
    );
    cache.insert_discovered_variant_at(ENDPOINT, MODEL, "unknown", None, now);
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, now)
            .effective()
            .unwrap()
            .input_modalities,
        None
    );
    assert_eq!(
        cache
            .get_at(ENDPOINT, MODEL, now + Duration::from_secs(10))
            .effective()
            .unwrap()
            .input_modalities,
        Some(vec!["image".into(), "text".into()])
    );
}
