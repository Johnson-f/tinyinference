use super::is_openai_oauth_session_expired_message;

#[test]
fn detects_oauth_expiry_markers_without_matching_api_key_failures() {
    assert!(is_openai_oauth_session_expired_message(
        r#"{"error":{"code":"token_expired"}}"#
    ));
    assert!(is_openai_oauth_session_expired_message(
        "Provided authentication token is expired"
    ));
    assert!(!is_openai_oauth_session_expired_message(
        "Incorrect API key provided"
    ));
}
