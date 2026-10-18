use super::*;

#[test]
fn without_reasoning_sets_the_existing_metadata_hint_without_dropping_metadata() {
    let mut request = ModelRequest::default();
    request.metadata = serde_json::json!({"caller": "kept"});

    let request = without_reasoning(request);
    assert_eq!(request.metadata["caller"], "kept");
    assert_eq!(
        request.metadata[WITHOUT_REASONING_METADATA_KEY],
        serde_json::Value::Bool(true)
    );
}

#[test]
fn without_reasoning_replaces_non_object_metadata_with_the_hint() {
    let mut request = ModelRequest::default();
    request.metadata = serde_json::Value::Null;

    let request = without_reasoning(request);
    assert_eq!(
        request.metadata,
        serde_json::json!({WITHOUT_REASONING_METADATA_KEY: true})
    );
}
