use super::*;

#[test]
fn defaults_and_identity_match_host() {
    let model = OllamaEmbeddingModel::default();
    assert_eq!(model.base_url(), DEFAULT_OLLAMA_URL);
    assert_eq!(model.model_id(), DEFAULT_OLLAMA_MODEL);
    assert_eq!(model.dimensions(), DEFAULT_OLLAMA_DIMENSIONS);
    assert_eq!(model.signature(), "provider=ollama;model=bge-m3;dims=1024");
}

#[test]
fn validates_root_url_and_real_model() {
    assert!(OllamaEmbeddingModel::try_new("http://host:11434/api", "m", 1).is_err());
    assert!(OllamaEmbeddingModel::try_new("http://user:p@host:11434", "m", 1).is_err());
    assert!(OllamaEmbeddingModel::try_new("http://host:11434", "local-v1", 1).is_err());
}

#[tokio::test]
async fn blank_inputs_are_position_safe_without_network() {
    let model = OllamaEmbeddingModel::default();
    let vectors = model.embed(&[" ".into(), "\n".into()]).await.unwrap();
    assert_eq!(vectors, vec![Vec::<f32>::new(), Vec::new()]);
}

#[test]
fn recognizes_only_nan_encoding_failures() {
    assert!(is_nan_encode_error("unsupported value: NaN"));
    assert!(!is_nan_encode_error("model crashed"));
}
