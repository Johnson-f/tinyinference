use super::*;

#[test]
fn identity_and_defaults_match_host_contract() {
    let model = CohereEmbeddingModel::new("key");
    assert_eq!(model.name(), "cohere");
    assert_eq!(model.model_id(), COHERE_DEFAULT_MODEL);
    assert_eq!(model.dimensions(), COHERE_DEFAULT_DIMENSIONS);
    assert_eq!(
        model.signature(),
        "provider=cohere;model=embed-english-v3.0;dims=1024"
    );
}

#[tokio::test]
async fn empty_batch_short_circuits_before_key_validation() {
    let model = CohereEmbeddingModel::new("");
    assert!(model.embed(&[]).await.unwrap().is_empty());
}

#[tokio::test]
async fn empty_batch_reports_no_usage() {
    let model = CohereEmbeddingModel::new("");
    let (vectors, usage) = model.embed_with_usage(&[]).await.unwrap();
    assert!(vectors.is_empty());
    assert!(usage.is_none());
}

#[test]
fn billed_units_carry_the_batch_usage() {
    let payload: CohereResponse = serde_json::from_str(
        r#"{"embeddings":{"float":[[1.0]]},"meta":{"billed_units":{"input_tokens":77}}}"#,
    )
    .unwrap();
    assert_eq!(payload.usage(), Some(EmbeddingUsage::new(77)));
}

#[test]
fn a_response_without_billed_units_reports_no_usage() {
    // Every link is optional, and a missing one must not fail an embed
    // whose vectors parsed.
    for body in [
        r#"{"embeddings":{"float":[[1.0]]}}"#,
        r#"{"embeddings":{"float":[[1.0]]},"meta":{}}"#,
        r#"{"embeddings":{"float":[[1.0]]},"meta":{"billed_units":{}}}"#,
        r#"{"embeddings":{"float":[[1.0]]},"meta":{"billed_units":{"input_tokens":0}}}"#,
    ] {
        let payload: CohereResponse = serde_json::from_str(body).unwrap();
        assert!(payload.usage().is_none(), "expected no usage for {body}");
    }
}

#[tokio::test]
async fn missing_key_fails_before_network() {
    let model = CohereEmbeddingModel::new("").with_base_url("http://127.0.0.1:1");
    let error = model.embed(&["hello".into()]).await.unwrap_err();
    assert!(error.to_string().contains("API key not set"));
}

#[tokio::test]
async fn zero_dimensions_fail_before_network() {
    let model = CohereEmbeddingModel::new("key")
        .with_dimensions(0)
        .with_base_url("http://127.0.0.1:1");
    let error = model.embed(&["hello".into()]).await.unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
}
