use super::*;

fn missing_bearer() -> BearerResolver {
    Arc::new(|| {
        Err(Error::Validation(
            "No backend session for cloud embeddings".into(),
        ))
    })
}

#[test]
fn identity_matches_host_contract() {
    let model = CloudEmbeddingModel::new(
        "https://api.example/openai/v1/",
        DEFAULT_CLOUD_MODEL,
        DEFAULT_CLOUD_DIMENSIONS,
        missing_bearer(),
    );
    assert_eq!(model.name(), "cloud");
    assert_eq!(
        model.signature(),
        "provider=cloud;model=embedding-v1;dims=1024"
    );
}

#[tokio::test]
async fn validation_precedes_bearer_resolution() {
    let model = CloudEmbeddingModel::new(
        "https://api.example/openai/v1",
        DEFAULT_CLOUD_MODEL,
        DEFAULT_CLOUD_DIMENSIONS,
        missing_bearer(),
    );
    assert!(model.embed(&[]).await.unwrap().is_empty());
    let error = model.embed(&[" ".into()]).await.unwrap_err();
    assert!(error.to_string().contains("empty/whitespace"));
}
