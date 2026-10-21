use super::*;
use crate::model::{InputModality, InputSource};
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Mutex;

#[derive(Default)]
struct RequestRecorder {
    requests: Mutex<Vec<ModelRequest>>,
    profile: Option<ModelProfile>,
}

impl RequestRecorder {
    fn with_profile(profile: Option<ModelProfile>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            profile,
        }
    }
}

#[async_trait]
impl ChatModel<()> for RequestRecorder {
    async fn invoke(&self, _: &(), request: ModelRequest) -> Result<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        Ok(ModelResponse::assistant("ok"))
    }

    fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    fn supports_input(&self, modality: InputModality, _: &str, source: InputSource) -> bool {
        modality == InputModality::Image && source == InputSource::Base64
    }

    fn cache_identity(&self) -> Option<String> {
        Some("fixture-model".to_owned())
    }
}

async fn assert_request_defaults_preserve_optional_profile(
    model: ProfileOverrideModel<()>,
    inner: &RequestRecorder,
    expected_profile: Option<ModelProfile>,
) {
    assert_eq!(model.profile(), expected_profile.as_ref());
    assert!(model.supports_input(InputModality::Image, "image/png", InputSource::Base64));
    assert_eq!(model.cache_identity().as_deref(), Some("fixture-model"));

    model.invoke(&(), ModelRequest::default()).await.unwrap();
    model
        .invoke(
            &(),
            ModelRequest::default()
                .with_model("invoke-explicit-model")
                .with_temperature(0.9),
        )
        .await
        .unwrap();
    model
        .stream(&(), ModelRequest::default())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    model
        .stream(
            &(),
            ModelRequest::default()
                .with_model("stream-explicit-model")
                .with_temperature(0.8),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let requests = inner.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].model.as_deref(), Some("default-model"));
    assert_eq!(requests[0].temperature, Some(0.25));
    assert_eq!(requests[1].model.as_deref(), Some("invoke-explicit-model"));
    assert_eq!(requests[1].temperature, Some(0.9));
    assert_eq!(requests[2].model.as_deref(), Some("default-model"));
    assert_eq!(requests[2].temperature, Some(0.25));
    assert_eq!(requests[3].model.as_deref(), Some("stream-explicit-model"));
    assert_eq!(requests[3].temperature, Some(0.8));
}

#[tokio::test]
async fn request_defaults_preserve_missing_profiles_and_explicit_call_values() {
    let inner = Arc::new(RequestRecorder::default());
    let model = ProfileOverrideModel::new_preserving_profile(inner.clone()).with_defaults(
        ModelRequestDefaults {
            model: Some("default-model".to_owned()),
            temperature: Some(0.25),
        },
    );
    assert_request_defaults_preserve_optional_profile(model, &inner, None).await;
}

#[tokio::test]
async fn request_defaults_keep_explicit_profiles_and_explicit_call_values() {
    let inner_profile = ModelProfile {
        provider: Some("fixture".to_owned()),
        model: Some("profile-model".to_owned()),
        ..ModelProfile::default()
    };
    let inner = Arc::new(RequestRecorder::with_profile(Some(inner_profile.clone())));
    let model = ProfileOverrideModel::new_preserving_profile(inner.clone()).with_defaults(
        ModelRequestDefaults {
            model: Some("default-model".to_owned()),
            temperature: Some(0.25),
        },
    );
    assert_request_defaults_preserve_optional_profile(model, &inner, Some(inner_profile)).await;
}

#[tokio::test]
async fn explicit_profile_override_still_wins_over_the_inner_profile() {
    let inner = Arc::new(RequestRecorder::with_profile(Some(ModelProfile {
        provider: Some("inner".to_owned()),
        ..ModelProfile::default()
    })));
    let profile = ModelProfile {
        provider: Some("override".to_owned()),
        model: Some("profile-model".to_owned()),
        ..ModelProfile::default()
    };
    let model = ProfileOverrideModel::new(inner.clone(), profile.clone()).with_defaults(
        ModelRequestDefaults {
            model: Some("default-model".to_owned()),
            temperature: Some(0.25),
        },
    );
    assert_request_defaults_preserve_optional_profile(model, &inner, Some(profile)).await;
}

#[test]
fn without_reasoning_sets_the_existing_metadata_hint_without_dropping_metadata() {
    let request = ModelRequest {
        metadata: serde_json::json!({"caller": "kept"}),
        ..ModelRequest::default()
    };

    let request = without_reasoning(request);
    assert_eq!(request.metadata["caller"], "kept");
    assert_eq!(
        request.metadata[WITHOUT_REASONING_METADATA_KEY],
        serde_json::Value::Bool(true)
    );
}

#[test]
fn without_reasoning_replaces_non_object_metadata_with_the_hint() {
    let request = ModelRequest {
        metadata: serde_json::Value::Null,
        ..ModelRequest::default()
    };

    let request = without_reasoning(request);
    assert_eq!(
        request.metadata,
        serde_json::json!({WITHOUT_REASONING_METADATA_KEY: true})
    );
}
