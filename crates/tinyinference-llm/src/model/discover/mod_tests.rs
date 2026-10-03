use super::*;

#[test]
fn record_overflow_error_stores_stated_limit() {
    let cache = ModelLimitsCache::default();
    let recorded = record_overflow_error_in(
        &cache,
        "https://api.deepseek.com/v1/",
        "deepseek-chat",
        "This model's maximum context length is 131072 tokens. However, you requested 140000 tokens",
    );
    assert_eq!(recorded, Some(131_072));
    let effective = cache
        .get("https://api.deepseek.com/v1", "deepseek-chat")
        .effective()
        .unwrap();
    assert_eq!(effective.context_window, Some(131_072));
    assert_eq!(effective.source, LimitSource::LearnedFromOverflow);
}

#[test]
fn record_overflow_error_ignores_messages_without_a_limit() {
    let cache = ModelLimitsCache::default();
    assert_eq!(
        record_overflow_error_in(&cache, "e", "m", "context length exceeded"),
        None
    );
    assert_eq!(cache.get("e", "m"), CachedLimits::default());
}

#[test]
fn global_cache_round_trip() {
    let endpoint = "https://global-cache-test.example/v1";
    assert_eq!(cached_model_limits(endpoint, "m"), None);
    record_overflow_error(endpoint, "m", "prompt is too long: 250000 tokens > 200000 maximum");
    assert_eq!(
        cached_model_limits(endpoint, "m").and_then(|limits| limits.context_window),
        Some(200_000)
    );
}
