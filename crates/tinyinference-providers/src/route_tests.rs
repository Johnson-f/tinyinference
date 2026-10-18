use super::*;

#[test]
fn route_preserves_namespaces_tags_and_case_with_a_temperature_suffix() {
    let route = ProviderRoute::parse("  My-Provider : Org/Model:Tag @ 0.7 ").unwrap();
    assert_eq!(route.provider, "My-Provider");
    assert_eq!(route.model, "Org/Model:Tag");
    assert_eq!(route.temperature, Some(0.7));
}

#[test]
fn route_leaves_host_sentinels_and_empty_models_for_host_and_builder() {
    assert!(ProviderRoute::parse("cloud").is_none());
    assert!(ProviderRoute::parse("openhuman").is_none());
    assert_eq!(ProviderRoute::parse("ollama:").unwrap().model, "");
}

#[test]
fn temperature_parser_preserves_absent_invalid_and_non_finite_suffixes() {
    for raw in [
        " model ",
        " model@bad ",
        " model@NaN ",
        " model@inf ",
        " @0.7 ",
    ] {
        assert_eq!(split_model_and_temperature(raw), (raw.trim().into(), None));
    }
}

#[test]
fn temperature_parser_uses_the_last_at_sign_and_trims_components() {
    assert_eq!(
        split_model_and_temperature(" org@revision/model @ -0.5 "),
        ("org@revision/model".into(), Some(-0.5))
    );
    assert_eq!(
        split_model_and_temperature("model@0"),
        ("model".into(), Some(0.0))
    );
}

#[test]
fn route_kind_classifies_managed_local_subprocess_and_cloud_routes() {
    for (raw, expected) in [
        ("openhuman:model", RouteKind::Managed),
        (
            "ollama:qwen3:14b",
            RouteKind::Local(LocalProviderKind::Ollama),
        ),
        (
            "lm-studio:model",
            RouteKind::Local(LocalProviderKind::LmStudio),
        ),
        ("mlx:model", RouteKind::Local(LocalProviderKind::Mlx)),
        ("omlx:model", RouteKind::Local(LocalProviderKind::Omlx)),
        (
            "local_openai:llama",
            RouteKind::Local(LocalProviderKind::LocalOpenai),
        ),
        (
            "llamacpp:model",
            RouteKind::Local(LocalProviderKind::LocalOpenai),
        ),
        (
            "llama.cpp:model",
            RouteKind::Local(LocalProviderKind::LocalOpenai),
        ),
        (
            "vllm:model",
            RouteKind::Local(LocalProviderKind::LocalOpenai),
        ),
        ("claude-code:sonnet", RouteKind::ClaudeCode),
        ("claude_agent_sdk:sonnet", RouteKind::ClaudeAgentSdk),
        ("claude-agent-sdk:sonnet", RouteKind::ClaudeAgentSdk),
        (
            "My-Provider:model",
            RouteKind::Cloud {
                slug: "My-Provider".to_owned(),
            },
        ),
        (
            "openai:model",
            RouteKind::Cloud {
                slug: "openai".to_owned(),
            },
        ),
        ("cloud:model", RouteKind::Sentinel),
    ] {
        assert_eq!(ProviderRoute::parse(raw).unwrap().kind(), expected, "{raw}");
    }
}

#[test]
fn external_route_classification_excludes_managed_and_sentinel_routes() {
    for (raw, expected) in [
        ("openhuman:model", true),
        ("cloud:model", false),
        ("ollama:qwen3", false),
        ("lmstudio:model", false),
        ("mlx:model", false),
        ("omlx:model", false),
        ("local-openai:model", false),
        ("local_openai:model", false),
        ("custom-openai:model", false),
        ("llamacpp:model", false),
        ("llama.cpp:model", false),
        ("vllm:model", false),
        ("claude-code:sonnet", true),
        ("claude_agent_sdk:sonnet", true),
        ("claude-agent-sdk:sonnet", true),
        ("openai:gpt-4.1", true),
    ] {
        assert_eq!(
            ProviderRoute::parse(raw).unwrap().is_external(),
            expected,
            "{raw}"
        );
    }
}

#[test]
fn custom_provider_helpers_round_trip_endpoint_and_plain_slug() {
    assert_eq!(
        split_custom_provider("custom:https://models.example/v1"),
        (
            "custom".to_owned(),
            Some("https://models.example/v1".to_owned())
        )
    );
    assert_eq!(split_custom_provider("openai"), ("openai".to_owned(), None));
    assert_eq!(
        custom_provider_string("custom", Some("https://models.example/v1")),
        "custom:https://models.example/v1"
    );
    assert_eq!(custom_provider_string("openai", None), "openai");
}

#[test]
fn bare_model_routes_default_to_ollama_and_owner_uses_route_slug() {
    assert_eq!(bare_model_route("llama3"), "ollama:llama3");
    assert_eq!(bare_model_route("openhuman"), "openhuman");
    assert_eq!(bare_model_route("openai:gpt-4.1"), "openai:gpt-4.1");
    assert_eq!(
        route_owner("openrouter:org/model"),
        ("openrouter:org/model".to_owned(), "openrouter".to_owned())
    );
    assert_eq!(
        route_owner("bare-model"),
        ("bare-model".to_owned(), "openhuman".to_owned())
    );
    assert_eq!(
        route_owner("acme:org/model:tag@0.7"),
        ("acme:org/model:tag".to_owned(), "acme".to_owned())
    );
    assert_eq!(
        route_owner("acme:model@beta"),
        ("acme:model@beta".to_owned(), "acme".to_owned())
    );
    assert_eq!(
        route_owner("gpt-4o@0.7"),
        ("gpt-4o".to_owned(), "openhuman".to_owned())
    );
    assert_eq!(
        route_owner("gpt-4o@NaN"),
        ("gpt-4o".to_owned(), "openhuman".to_owned())
    );
    assert_eq!(
        route_owner("gpt-4o@beta"),
        ("gpt-4o@beta".to_owned(), "openhuman".to_owned())
    );
}
