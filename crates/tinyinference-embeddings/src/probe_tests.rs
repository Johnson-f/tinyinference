use super::*;

#[test]
fn probe_url_appends_path_before_query_parameters() {
    let url = embeddings_probe_url("https://host.example/v1?api-version=2026").unwrap();
    assert_eq!(url.path(), "/v1/embeddings");
    assert_eq!(url.query(), Some("api-version=2026"));
}

#[test]
fn final_dimensions_use_the_proven_response_width() {
    assert_eq!(final_probe_dims("text-embedding-3-large", 1024, 3072), 3072);
    assert_eq!(final_probe_dims("bge-m3", 1024, 768), 768);
    assert_eq!(final_probe_dims("bge-m3", 1024, 0), 1024);
}

#[test]
fn probe_requires_one_vector_for_its_one_input() {
    let error = validate_probe_vectors("bge-m3", 1024, &[vec![0.0], vec![1.0]])
        .expect_err("two vectors must be rejected");
    assert!(error.to_string().contains("expected 1, got 2"));
}
