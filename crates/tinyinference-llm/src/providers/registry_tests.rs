use super::*;

fn registry() -> ProviderRegistry<impl CredentialResolver> {
    ProviderRegistry::new(|_: &ProviderSpec| Ok("test-key".to_string()))
}

#[test]
fn slug_model_fallback_keeps_concrete_namespaced_ids_and_remaps_hints() {
    assert_eq!(
        resolve_slug_model(" qwen3:14b ", Some("provider-default"), |model| {
            model.trim().starts_with("hint:")
        })
        .unwrap(),
        " qwen3:14b "
    );
    assert_eq!(
        resolve_slug_model("  ", Some(" provider-default "), |model| {
            model.trim().starts_with("hint:")
        })
        .unwrap(),
        " provider-default "
    );
    assert_eq!(
        resolve_slug_model("hint:fast", Some(" gpt-5 "), |model| {
            model.trim().starts_with("hint:")
        })
        .unwrap(),
        "gpt-5"
    );
    assert_eq!(
        resolve_slug_model("hint:fast", None, |model| model.starts_with("hint:")),
        Err(SlugModelError::NoConcreteDefault)
    );
    assert_eq!(
        resolve_slug_model("hint:fast", Some("  "), |model| model.starts_with("hint:")),
        Err(SlugModelError::NoConcreteDefault)
    );
    assert_eq!(
        resolve_slug_model("hint:fast", Some("hint:slow"), |model| model
            .starts_with("hint:")),
        Err(SlugModelError::NoConcreteDefault)
    );
    assert_eq!(
        resolve_slug_model("", Some("hint:slow"), |model| model.starts_with("hint:")),
        Err(SlugModelError::NoConcreteDefault)
    );
    assert_eq!(
        resolve_slug_model("", None, |model| model.starts_with("hint:")),
        Err(SlugModelError::MissingModel)
    );
}

#[test]
fn typed_cloud_discovery_reuses_spec_endpoint_model_and_credentials() {
    let registry = registry();
    let route = tinyinference_providers::route::ProviderRoute::parse("openai:qwen3:14b").unwrap();
    let mut spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    spec.provider = "openai".into();
    spec.base_url = "https://custom.example.test/v1".into();
    spec.model = "configured-default".into();
    let request = registry
        .discovery_request_for_route(route, Some(spec.clone()))
        .unwrap()
        .unwrap();
    assert_eq!(request.endpoint, spec.base_url);
    assert_eq!(request.model, "qwen3:14b");
    assert_eq!(
        request.headers,
        vec![("Authorization".into(), "Bearer test-key".into())]
    );
}

#[test]
fn typed_routes_without_cloud_listings_bypass_credentials() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| -> Result<String> {
        panic!("route without a cloud listing must not consult credentials")
    });
    for route in [
        "openhuman:gpt-5",
        "ollama:qwen3:14b",
        "claude-code:sonnet",
        "cloud:gpt-5",
    ] {
        assert!(
            registry
                .discovery_request_for_route(
                    tinyinference_providers::route::ProviderRoute::parse(route).unwrap(),
                    Some(ProviderSpec::for_kind(ProviderKind::OpenAi)),
                )
                .unwrap()
                .is_none()
        );
    }
    let route = tinyinference_providers::route::ProviderRoute::parse("openai:gpt-5").unwrap();
    assert!(
        registry
            .discovery_request_for_route(route.clone(), None)
            .unwrap()
            .is_none()
    );
    let mut mismatched = ProviderSpec::for_kind(ProviderKind::OpenAi);
    mismatched.provider = "other".into();
    assert!(
        registry
            .discovery_request_for_route(route, Some(mismatched))
            .unwrap()
            .is_none()
    );
}

#[test]
fn builds_cloud_and_local_profiles_from_resolved_specs() {
    for kind in [
        ProviderKind::OpenAi,
        ProviderKind::OpenRouter,
        ProviderKind::Anthropic,
        ProviderKind::Ollama,
    ] {
        let spec = ProviderSpec::for_kind(kind.clone());
        let model = registry().build(&spec).unwrap();
        let profile = model.profile().unwrap();
        assert_eq!(profile.model.as_deref(), Some(spec.model.as_str()));
        assert_eq!(profile.tool_calling, kind != ProviderKind::Ollama);
        assert!(
            model
                .cache_identity()
                .is_none_or(|identity| !identity.contains("test-key"))
        );
    }
}

#[test]
fn explicit_auth_and_text_tools_keep_anthropic_on_compatible_adapter() {
    let spec = ProviderSpec::for_kind(ProviderKind::Anthropic);
    let model = registry()
        .build_with(
            &spec,
            ProviderBuildOptions {
                native_tool_calling: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!model.profile().unwrap().tool_calling);
    let spec = ProviderSpec {
        provider: "private".into(),
        ..spec
    };
    let model = registry()
        .build_with(
            &spec,
            ProviderBuildOptions {
                auth_style: Some(AuthStyle::Bearer),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        model.profile().unwrap().provider.as_deref(),
        Some("private")
    );
}

#[test]
fn discovery_uses_same_endpoint_model_auth_and_pinned_routes() {
    let spec = ProviderSpec::for_kind(ProviderKind::OpenRouter);
    let options = ProviderBuildOptions {
        default_provider_options: Some(serde_json::json!({"provider":{"only":["DeepInfra"]}})),
        ..Default::default()
    };
    let request = registry()
        .discovery_request(&spec, &options)
        .unwrap()
        .unwrap();
    assert_eq!(request.endpoint, spec.base_url);
    assert_eq!(request.model, spec.model);
    assert_eq!(
        request.headers,
        vec![("Authorization".into(), "Bearer test-key".into())]
    );
    assert_eq!(request.pinned_providers, vec!["DeepInfra"]);
    let spec = ProviderSpec::for_kind(ProviderKind::Anthropic);
    let request = registry()
        .discovery_request(&spec, &Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        request.headers,
        vec![
            ("x-api-key".into(), "test-key".into()),
            ("anthropic-version".into(), "2023-06-01".into()),
        ]
    );
}

#[test]
fn local_discovery_does_not_read_credentials() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| -> Result<String> {
        panic!("discovery must not consult credentials when no listing exists")
    });
    let spec = ProviderSpec::for_kind(ProviderKind::LmStudio);
    assert!(
        registry
            .discovery_request(&spec, &Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn discovery_extra_headers_replace_builtin_authentication_case_insensitively() {
    let spec = ProviderSpec::for_kind(ProviderKind::Anthropic);
    let headers = vec![
        ("X-API-KEY".into(), "override-key".into()),
        ("Anthropic-Version".into(), "override-version".into()),
    ];
    let request = registry()
        .discovery_request(
            &spec,
            &ProviderBuildOptions {
                extra_headers: &headers,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        request.headers,
        vec![
            ("X-API-KEY".into(), "override-key".into()),
            ("Anthropic-Version".into(), "override-version".into()),
        ]
    );
}

#[test]
fn build_and_discovery_trim_model_and_endpoint_values() {
    let mut spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    spec.model = " gpt-test\n".into();
    spec.base_url = " https://api.openai.com/v1 \n".into();
    let model = registry().build(&spec).unwrap();
    assert_eq!(model.profile().unwrap().model.as_deref(), Some("gpt-test"));
    let request = registry()
        .discovery_request(&spec, &Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(request.model, "gpt-test");
    assert_eq!(request.endpoint, "https://api.openai.com/v1");
}

#[test]
fn invalid_specs_and_missing_required_keys_fail_before_construction() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok(String::new()));
    let mut spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    assert!(matches!(registry.build(&spec), Err(Error::Validation(_))));
    spec.requires_api_key = false;
    assert!(registry.build(&spec).is_ok());
    spec.model.clear();
    assert!(matches!(registry.build(&spec), Err(Error::Validation(_))));
    spec.model = "gpt-test".into();
    spec.base_url.clear();
    assert!(matches!(registry.build(&spec), Err(Error::Validation(_))));
}

#[test]
fn credential_errors_are_returned_without_rewriting() {
    let registry =
        ProviderRegistry::new(|_: &ProviderSpec| Err(Error::Unsupported("keyring offline".into())));
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    assert!(
        matches!(registry.build(&spec), Err(Error::Unsupported(message)) if message == "keyring offline")
    );
}

#[test]
fn price_ordering_is_only_supplied_to_openrouter() {
    assert_eq!(
        openrouter_default_provider_options("https://openrouter.ai/api/v1"),
        Some(serde_json::json!({"provider":{"sort":"price"}}))
    );
    assert_eq!(
        openrouter_default_provider_options("https://api.openai.com/v1"),
        None
    );
}

#[test]
fn debugging_registry_options_never_discloses_transport_secrets() {
    let headers = vec![("Authorization".into(), "secret-header".into())];
    let query = vec![("token".into(), "secret-query".into())];
    let route = OpenAiCodexRouting {
        endpoint: "https://user:secret-password@example.com?token=secret-url".into(),
        using_oauth: true,
        account_id: Some("secret-account".into()),
    };
    let debug = format!(
        "{:?}",
        ProviderBuildOptions {
            extra_headers: &headers,
            extra_query_params: &query,
            codex: Some(&route),
            user_agent: Some("secret-user-agent"),
            ..Default::default()
        }
    );
    assert!(!debug.contains("secret"));
    assert!(debug.contains("Authorization"));
}

#[test]
fn injected_http_client_takes_precedence_over_ca_bundle_and_builder() {
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    let http = reqwest::Client::new();
    let model = registry()
        .build_with(
            &spec,
            ProviderBuildOptions {
                http: Some(http),
                ca_bundle: Some("malformed secret PEM"),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        model.profile().unwrap().model.as_deref(),
        Some("gpt-4.1-mini")
    );
}

#[test]
fn client_builder_factory_can_borrow_host_transport_settings() {
    let proxy_url = String::from("http://127.0.0.1:19444");
    let builder = std::sync::Arc::new(|| {
        reqwest::Client::builder().proxy(reqwest::Proxy::http(&proxy_url).unwrap())
    });
    let model = registry()
        .build_with(
            &ProviderSpec::for_kind(ProviderKind::OpenAi),
            ProviderBuildOptions {
                http_builder: Some(builder),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(model.profile().is_some());
}

#[test]
fn invalid_ca_bundle_fails_without_leaking_input() {
    let secret = "malformed secret PEM";
    let result = registry().build_with(
        &ProviderSpec::for_kind(ProviderKind::OpenAi),
        ProviderBuildOptions {
            ca_bundle: Some(secret),
            ..Default::default()
        },
    );
    let error = match result {
        Ok(_) => panic!("invalid CA bundle must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::Validation(_)));
    assert!(!error.to_string().contains(secret));
}

#[test]
fn resolved_credentials_are_trimmed_for_both_inference_and_discovery() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok("\n test-key \r\n".into()));
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    assert_eq!(registry.resolve_key(&spec).unwrap(), "test-key");
    let request = registry
        .discovery_request(&spec, &Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        request.headers,
        vec![("Authorization".into(), "Bearer test-key".into())]
    );
}

#[test]
fn native_anthropic_preserves_the_resolved_provider_identity() {
    let spec = ProviderSpec {
        provider: "my-anthropic".into(),
        ..ProviderSpec::for_kind(ProviderKind::Anthropic)
    };
    let model = registry().build(&spec).unwrap();
    assert_eq!(
        model.profile().unwrap().provider.as_deref(),
        Some("my-anthropic")
    );
}

#[test]
fn native_anthropic_applies_the_vision_override() {
    let spec = ProviderSpec::for_kind(ProviderKind::Anthropic);
    let model = registry()
        .build_with(
            &spec,
            ProviderBuildOptions {
                vision: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!model.profile().unwrap().modalities.image_in);
}

#[test]
fn local_registry_profiles_leave_unprobed_context_windows_unknown() {
    for kind in [
        ProviderKind::Ollama,
        ProviderKind::LmStudio,
        ProviderKind::LlamaCpp,
        ProviderKind::Vllm,
    ] {
        let mut spec = ProviderSpec::for_kind(kind);
        if spec.model.is_empty() {
            spec.model = "local-model".into();
        }
        let model = registry().build(&spec).unwrap();
        assert_eq!(
            model.profile().unwrap().max_input_tokens,
            None,
            "{} must not borrow a hosted model window",
            spec.provider
        );
    }
}

#[test]
fn ollama_registry_exposes_native_model_limit_discovery() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok(String::new()));
    let mut spec = ProviderSpec::for_kind(ProviderKind::Ollama).with_model("CaseSensitive:latest");
    spec.base_url = "http://localhost:11434/v1".into();
    let request = registry
        .discovery_request(&spec, &Default::default())
        .unwrap()
        .expect("Ollama has native /api/show model-limit discovery");
    assert!(request.ollama_native_enabled());
    assert_eq!(request.model, "CaseSensitive:latest");
    assert_eq!(request.endpoint, spec.base_url);
}

#[test]
fn explicit_local_compatible_profiles_match_the_existing_compatible_builder() {
    for (kind, slug) in [
        (LocalCompatibleKind::Mlx, "mlx"),
        (LocalCompatibleKind::Omlx, "omlx"),
        (LocalCompatibleKind::OpenAi, "local-openai"),
    ] {
        for model_id in [
            "CaseSensitive:latest",
            "gpt-4o-mini",
            "claude-3-5-sonnet-latest",
        ] {
            let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok("local-key".into()));
            let mut spec = ProviderSpec::for_kind(ProviderKind::Compatible).with_model(model_id);
            spec.provider = slug.into();
            spec.base_url = "http://localhost:8080/v1".into();
            spec.requires_api_key = false;
            let model = registry
                .build_with(
                    &spec,
                    ProviderBuildOptions {
                        local_compatible: Some(kind),
                        auth_style: Some(AuthStyle::Bearer),
                        ..Default::default()
                    },
                )
                .unwrap();
            let previous = super::super::openai::build_local_runtime_chat_model(
                slug,
                &spec.base_url,
                "local-key",
                AuthStyle::Bearer,
                model_id,
                &[],
                None,
                None,
            );
            assert_eq!(
                serde_json::to_value(model.profile()).unwrap(),
                serde_json::to_value(previous.profile()).unwrap()
            );
            assert_eq!(model.cache_identity(), previous.cache_identity());
            assert_eq!(model.profile().unwrap().provider.as_deref(), Some(slug));
            if model_id == "CaseSensitive:latest" {
                assert_eq!(model.profile().unwrap().max_input_tokens, None);
            }
        }
    }
}

#[test]
fn explicit_local_compatible_discovery_never_reads_credentials() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| -> Result<String> {
        panic!("local compatible servers have no listing contract")
    });
    let spec = ProviderSpec::for_kind(ProviderKind::Compatible).with_model("local");
    for kind in [
        LocalCompatibleKind::Mlx,
        LocalCompatibleKind::Omlx,
        LocalCompatibleKind::OpenAi,
    ] {
        assert!(
            registry
                .discovery_request(
                    &spec,
                    &ProviderBuildOptions {
                        local_compatible: Some(kind),
                        ..Default::default()
                    }
                )
                .unwrap()
                .is_none()
        );
    }
}

struct NativeOllamaLimits;
#[async_trait::async_trait]
impl crate::model::discover::ModelListingFetcher for NativeOllamaLimits {
    async fn get_json(&self, _: &str, _: &[(String, String)]) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"data":[]}))
    }
    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        assert_eq!(url, "http://localhost:19321/api/show");
        assert_eq!(body["model"], "CaseSensitive:latest");
        assert!(headers.contains(&("x-local".into(), "scoped".into())));
        Ok(serde_json::json!({"model_info":{"llama.context_length":32768}}))
    }
}
#[tokio::test]
async fn ollama_registry_native_request_discovers_actual_limits_on_nonstandard_ports() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok(String::new()));
    let mut spec = ProviderSpec::for_kind(ProviderKind::Ollama).with_model("CaseSensitive:latest");
    spec.base_url = "http://localhost:19321/v1".into();
    let headers = [("x-local".into(), "scoped".into())];
    let request = registry
        .discovery_request(
            &spec,
            &ProviderBuildOptions {
                extra_headers: &headers,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    let limits = crate::model::discover::discover_model_limits_with(
        &NativeOllamaLimits,
        &Default::default(),
        &request,
    )
    .await
    .unwrap();
    assert_eq!(limits.context_window, Some(32768));
}

#[test]
fn local_compatible_settings_reject_a_cloud_provider_family() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok("key".into()));
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    let error = match registry.build_with(
        &spec,
        ProviderBuildOptions {
            local_compatible: Some(LocalCompatibleKind::Mlx),
            ..Default::default()
        },
    ) {
        Ok(_) => panic!("cloud family must not be relabeled by local-compatible options"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requires a Compatible"));
}

#[test]
fn hosted_route_defaults_are_replaced_per_top_level_key() {
    let price = Some(serde_json::json!({"provider":{"sort":"price"}}));
    assert_eq!(
        replace_route_defaults(
            price.clone(),
            Some(serde_json::json!({"provider":{"order":["wafer"]},"top_k":7}))
        ),
        Some(serde_json::json!({"provider":{"order":["wafer"]},"top_k":7}))
    );
    assert_eq!(
        replace_route_defaults(price.clone(), Some(serde_json::json!({"top_k":7}))),
        Some(serde_json::json!({"provider":{"sort":"price"},"top_k":7}))
    );
    assert_eq!(
        replace_route_defaults(price.clone(), Some(serde_json::json!({"provider":null}))),
        Some(serde_json::json!({}))
    );
    assert_eq!(replace_route_defaults(price.clone(), None), price);
    assert_eq!(
        replace_route_defaults(price.clone(), Some(serde_json::Value::Null)),
        price
    );
    assert_eq!(
        replace_route_defaults(None, Some(serde_json::json!({"provider":null,"top_k":7}))),
        Some(serde_json::json!({"top_k":7}))
    );
    assert_eq!(
        replace_route_defaults(price, Some(serde_json::json!("not an object"))),
        Some(serde_json::json!("not an object"))
    );
}

#[test]
fn discovery_pins_the_caller_order_instead_of_the_price_default() {
    let spec = ProviderSpec::for_kind(ProviderKind::OpenRouter);
    let options = ProviderBuildOptions {
        default_provider_options: Some(
            serde_json::json!({"provider":{"order":["wafer"],"allow_fallbacks":false}}),
        ),
        ..Default::default()
    };
    let request = registry()
        .discovery_request(&spec, &options)
        .unwrap()
        .unwrap();
    assert_eq!(request.pinned_providers, vec!["wafer"]);
    let request = registry()
        .discovery_request(&spec, &Default::default())
        .unwrap()
        .unwrap();
    assert!(request.pinned_providers.is_empty());
}
