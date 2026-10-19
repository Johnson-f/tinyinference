//! Public API checks for provider overrides as consumed by an external host.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tinyinference_embeddings::{EmbeddingModel, MockEmbeddingModel};
use tinyinference_llm::model::ChatModel;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::providers::overrides::{
    CatalogEntry, CatalogRequest, ChatModelEntry, ChatModelRequest, EmbeddingRequest,
    ProviderOverrides, ReadinessRequest,
};

struct HostContext<'a> {
    borrowed_config: &'a str,
    // Deliberately non-Clone, like a host context that borrows configuration.
    _ownership_marker: std::marker::PhantomData<&'a mut ()>,
}

#[test]
fn synchronous_factories_accept_borrowed_non_clone_host_contexts() {
    let config = String::from("borrowed-host-config");
    let context = HostContext {
        borrowed_config: &config,
        _ownership_marker: std::marker::PhantomData,
    };
    let overrides = ProviderOverrides::new(
        |id| {
            (id == "hosted").then(|| ChatModelEntry {
                build: Arc::new(|request: &ChatModelRequest<'_, HostContext<'_>>| {
                    assert_eq!(request.context.borrowed_config, "borrowed-host-config");
                    Ok((
                        Arc::new(MockModel::constant("borrowed")) as Arc<dyn ChatModel<()>>,
                        "borrowed-model".into(),
                    ))
                }),
                probe: None,
            })
        },
        |id| {
            (id == "hosted").then(|| {
                Arc::new(|request: &EmbeddingRequest<'_, HostContext<'_>>| {
                    assert_eq!(request.context.borrowed_config, "borrowed-host-config");
                    Ok(Box::new(MockEmbeddingModel::new(request.dimensions))
                        as Box<dyn EmbeddingModel>)
                }) as _
            })
        },
        |_| None,
    );

    let chat = overrides.build_chat(
        "hosted",
        &ChatModelRequest {
            role: "chat",
            model_override: None,
            default_model: None,
            native_tool_calling: false,
            thread_id: None,
            context: &context,
        },
        |_| panic!("registered chat factory should be used"),
    );
    assert_eq!(chat.unwrap().1, "borrowed-model");

    let embedding = overrides.build_embedding(
        "hosted",
        &EmbeddingRequest {
            model: "embed",
            dimensions: 3,
            context: &context,
        },
        |_| panic!("registered embedding factory should be used"),
    );
    assert_eq!(embedding.unwrap().dimensions(), 3);
}

#[tokio::test]
async fn independent_live_family_lookups_override_only_their_builtin() {
    let global_context = HostContext {
        borrowed_config: "global-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let scoped_context = HostContext {
        borrowed_config: "scoped-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let chat_enabled = Arc::new(AtomicBool::new(true));
    let chat_enabled_lookup = Arc::clone(&chat_enabled);
    let overrides = ProviderOverrides::new(
        move |id| {
            (id == "shared" && chat_enabled_lookup.load(Ordering::SeqCst)).then(|| ChatModelEntry {
                build: Arc::new(|request: &ChatModelRequest<'_, HostContext<'_>>| {
                    assert_eq!(request.context.borrowed_config, "global-config");
                    if request.model_override == Some("fail") {
                        return Err(tinyinference_llm::Error::Validation(
                            "override construction failed".into(),
                        ));
                    }
                    Ok((
                        Arc::new(MockModel::constant("chat override")) as Arc<dyn ChatModel<()>>,
                        "chat-model".into(),
                    ))
                }),
                probe: Some(Arc::new(
                    |request: ReadinessRequest<HostContext<'static>>| {
                        Box::pin(async move {
                            assert_eq!(request.role, "chat");
                            assert_eq!(request.model, "chat-model");
                            assert_eq!(request.context.borrowed_config, "readiness-config");
                            Ok(())
                        })
                    },
                )),
            })
        },
        |id| {
            (id == "shared").then(|| {
                Arc::new(|request: &EmbeddingRequest<'_, HostContext<'_>>| {
                    assert_eq!(request.context.borrowed_config, "scoped-config");
                    Ok(Box::new(MockEmbeddingModel::new(request.dimensions))
                        as Box<dyn EmbeddingModel>)
                }) as _
            })
        },
        |id| {
            (id == "shared").then(|| CatalogEntry {
                list: Arc::new(|request: CatalogRequest<HostContext<'static>>| {
                    Box::pin(async move {
                        assert_eq!(request.context.borrowed_config, "catalog-config");
                        Ok(serde_json::json!({"models": []}))
                    })
                }),
                limits: Some(Arc::new(|model, context: &HostContext<'static>| {
                    (model == "catalog-model" && context.borrowed_config == "catalog-config").then(
                        || {
                            tinyinference_llm::model::discover::DiscoveryRequest::new(
                                "https://provider.invalid/v1",
                                model,
                            )
                        },
                    )
                })),
            })
        },
    );
    let _cloned_overrides = overrides.clone();

    let chat = overrides.build_chat(
        "shared",
        &ChatModelRequest {
            role: "chat",
            model_override: Some("chosen-model"),
            default_model: None,
            native_tool_calling: true,
            thread_id: Some("thread"),
            context: &global_context,
        },
        |_| panic!("chat override must run before builtin"),
    );
    assert_eq!(chat.unwrap().1, "chat-model");

    let override_error = overrides.build_chat(
        "shared",
        &ChatModelRequest {
            role: "chat",
            model_override: Some("fail"),
            default_model: None,
            native_tool_calling: true,
            thread_id: None,
            context: &global_context,
        },
        |_| panic!("a failed override must not fall through to builtin"),
    );
    assert!(override_error.is_err());

    let embedding = overrides.build_embedding(
        "shared",
        &EmbeddingRequest {
            model: "embed-model",
            dimensions: 8,
            context: &scoped_context,
        },
        |_| panic!("embedding override must run before builtin"),
    );
    assert_eq!(embedding.unwrap().dimensions(), 8);

    let fallback_context = HostContext {
        borrowed_config: "fallback-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let fallback = overrides.build_embedding(
        "unregistered",
        &EmbeddingRequest {
            model: "embed-model",
            dimensions: 4,
            context: &fallback_context,
        },
        |_| Ok(Box::new(MockEmbeddingModel::new(4)) as Box<dyn EmbeddingModel>),
    );
    assert_eq!(fallback.unwrap().dimensions(), 4);

    chat_enabled.store(false, Ordering::SeqCst);
    let chat_fallback = overrides.build_chat(
        "shared",
        &ChatModelRequest {
            role: "chat",
            model_override: None,
            default_model: Some("builtin-model"),
            native_tool_calling: false,
            thread_id: None,
            context: &fallback_context,
        },
        |_| {
            Ok((
                Arc::new(MockModel::constant("builtin chat")) as Arc<dyn ChatModel<()>>,
                "builtin-model".into(),
            ))
        },
    );
    assert_eq!(chat_fallback.unwrap().1, "builtin-model");
    chat_enabled.store(true, Ordering::SeqCst);

    let chat_entry = overrides.chat_entry("shared").unwrap();
    let probe = chat_entry.probe.unwrap();
    (probe)(ReadinessRequest {
        role: "chat".into(),
        model: "chat-model".into(),
        context: HostContext {
            borrowed_config: "readiness-config",
            _ownership_marker: std::marker::PhantomData,
        },
    })
    .await
    .unwrap();

    let catalog_context = HostContext {
        borrowed_config: "catalog-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let limits = overrides.catalog_entry("shared").unwrap().limits.unwrap();
    assert!(limits("catalog-model", &catalog_context).is_some());
    let listing = overrides
        .list_catalog(
            "shared",
            CatalogRequest {
                context: catalog_context,
            },
            |_| panic!("catalog override must run before builtin"),
        )
        .await
        .unwrap();
    assert_eq!(listing["models"], serde_json::json!([]));

    let builtin_listing = overrides
        .list_catalog(
            "unregistered",
            CatalogRequest {
                context: HostContext {
                    borrowed_config: "builtin-catalog-config",
                    _ownership_marker: std::marker::PhantomData,
                },
            },
            |_| Box::pin(async { Ok(serde_json::json!({"builtin": true})) }),
        )
        .await
        .unwrap();
    assert_eq!(builtin_listing["builtin"], true);
}
