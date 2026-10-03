use super::*;
use serde_json::json;

fn openrouter_listing() -> Value {
    json!({
        "data": [
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "name": "DeepSeek: DeepSeek V4.1 Flash",
                "context_length": 1_048_576,
                "top_provider": {
                    "context_length": 1_048_576,
                    "max_completion_tokens": 65_536,
                    "is_moderated": false
                }
            },
            {
                "id": "deepseek/deepseek-chat",
                "context_length": 163_840,
                "top_provider": { "context_length": 163_840, "max_completion_tokens": null }
            },
            { "id": "no-limits/model" }
        ]
    })
}

#[test]
fn parses_openrouter_listing_entry() {
    let limits = parse_model_limits(&openrouter_listing(), "deepseek/deepseek-v4.1-flash")
        .expect("listed model");
    assert_eq!(limits.context_window, Some(1_048_576));
    assert_eq!(limits.max_output_tokens, Some(65_536));
    assert_eq!(limits.source, LimitSource::ProviderListing);
}

#[test]
fn openrouter_listing_tolerates_route_prefix() {
    let limits = parse_model_limits(
        &openrouter_listing(),
        "openrouter/deepseek/deepseek-v4.1-flash",
    )
    .expect("prefixed id matches");
    assert_eq!(limits.context_window, Some(1_048_576));
}

#[test]
fn falls_back_to_top_provider_context() {
    let body = json!({ "data": [{
        "id": "a/b",
        "top_provider": { "context_length": 32_000, "max_completion_tokens": 4_096 }
    }]});
    let limits = parse_model_limits(&body, "a/b").unwrap();
    assert_eq!(limits.context_window, Some(32_000));
    assert_eq!(limits.max_output_tokens, Some(4_096));
}

#[test]
fn listing_skips_entries_without_limits() {
    let listed = parse_listing_limits(&openrouter_listing());
    let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        ids,
        ["deepseek/deepseek-v4.1-flash", "deepseek/deepseek-chat"]
    );
    assert!(parse_model_limits(&openrouter_listing(), "no-limits/model").is_none());
    assert!(parse_model_limits(&openrouter_listing(), "missing/model").is_none());
}

#[test]
fn parses_openai_compatible_single_model_record() {
    let body = json!({ "id": "gpt-x", "object": "model", "context_window": 400_000 });
    let limits = parse_model_limits(&body, "gpt-x").unwrap();
    assert_eq!(limits.context_window, Some(400_000));
    assert_eq!(limits.max_output_tokens, None);
    // A record for another model is not this model's limit.
    assert!(parse_model_limits(&body, "gpt-y").is_none());
}

#[test]
fn parses_vllm_max_model_len() {
    let body = json!({
        "object": "list",
        "data": [{ "id": "Qwen/Qwen3-32B", "object": "model", "max_model_len": 32_768 }]
    });
    let limits = parse_model_limits(&body, "Qwen/Qwen3-32B").unwrap();
    assert_eq!(limits.context_window, Some(32_768));
}

#[test]
fn parses_other_window_keys_and_quoted_numbers() {
    for (key, value) in [
        ("context_length", json!(8_192)),
        ("max_context_length", json!(16_384)),
        ("max_input_tokens", json!(200_000)),
        ("context_window", json!("131072")),
    ] {
        let body = json!({ "data": [{ "id": "m", key: value }] });
        let limits = parse_model_limits(&body, "m").unwrap_or_else(|| panic!("{key}"));
        assert!(limits.context_window.is_some(), "{key}");
    }
    let body = json!({ "data": [{ "id": "m", "context_length": 0 }] });
    assert!(parse_model_limits(&body, "m").is_none());
}

#[test]
fn single_record_under_data_object() {
    let body = json!({ "data": { "id": "m", "context_length": 65_536 } });
    assert_eq!(
        parse_model_limits(&body, "m").unwrap().context_window,
        Some(65_536)
    );
}

#[test]
fn exact_id_wins_over_prefix_tolerant_match() {
    let body = json!({ "data": [
        { "id": "vendor/model", "context_length": 100_000 },
        { "id": "model", "context_length": 50_000 }
    ]});
    assert_eq!(
        parse_model_limits(&body, "model").unwrap().context_window,
        Some(50_000)
    );
}

#[test]
fn model_ids_match_rules() {
    assert!(model_ids_match("deepseek/x", "DeepSeek/X"));
    assert!(model_ids_match("deepseek/x", "openrouter/deepseek/x"));
    assert!(model_ids_match("openrouter/deepseek/x", "deepseek/x"));
    assert!(!model_ids_match("deepseek/x", "deepseek/y"));
    assert!(!model_ids_match("", "x"));
}

fn openrouter_endpoints() -> Value {
    json!({ "data": {
        "id": "deepseek/deepseek-v4.1-flash",
        "endpoints": [
            {
                "name": "DeepSeek | deepseek/deepseek-v4.1-flash",
                "provider_name": "DeepSeek",
                "tag": "deepseek",
                "context_length": 1_048_576,
                "max_completion_tokens": 65_536
            },
            {
                "name": "DeepInfra | deepseek/deepseek-v4.1-flash",
                "provider_name": "DeepInfra",
                "tag": "deepinfra/fp8",
                "context_length": 163_840,
                "max_completion_tokens": 16_384
            }
        ]
    }})
}

#[test]
fn pinned_provider_uses_its_endpoint_limit() {
    let limits =
        parse_openrouter_endpoint_limits(&openrouter_endpoints(), &["deepinfra".to_string()])
            .unwrap();
    assert_eq!(limits.context_window, Some(163_840));
    assert_eq!(limits.max_output_tokens, Some(16_384));
    assert_eq!(
        limits.source,
        LimitSource::ProviderEndpoint {
            provider: "DeepInfra".to_string()
        }
    );
}

#[test]
fn several_pinned_providers_take_the_minimum() {
    let limits = parse_openrouter_endpoint_limits(
        &openrouter_endpoints(),
        &["DeepSeek".to_string(), "deep-infra".to_string()],
    )
    .unwrap();
    assert_eq!(limits.context_window, Some(163_840));
}

#[test]
fn unmatched_pin_yields_none() {
    assert!(
        parse_openrouter_endpoint_limits(&openrouter_endpoints(), &["together".to_string()])
            .is_none()
    );
}

#[test]
fn pinned_providers_from_routing_options() {
    assert_eq!(
        pinned_openrouter_providers(&json!({ "provider": { "only": ["DeepInfra"] } })),
        ["DeepInfra"]
    );
    assert_eq!(
        pinned_openrouter_providers(
            &json!({ "provider": { "order": ["a", "b"], "allow_fallbacks": false } })
        ),
        ["a", "b"]
    );
    // Order with fallbacks on does not pin.
    assert!(pinned_openrouter_providers(&json!({ "provider": { "order": ["a"] } })).is_empty());
    // OpenHuman's default `sort` does not pin.
    assert!(pinned_openrouter_providers(&json!({ "provider": { "sort": "price" } })).is_empty());
    assert!(pinned_openrouter_providers(&json!({})).is_empty());
}
