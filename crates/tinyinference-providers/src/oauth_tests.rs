use super::*;

fn unsigned_jwt(payload: serde_json::Value) -> String {
    let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
    let payload = URL_SAFE_NO_PAD.encode(payload.to_string());
    format!("{header}.{payload}.")
}

#[test]
fn codex_config_contains_the_public_client_contract() {
    let config = openai_codex_config("http://127.0.0.1:1455/auth/callback");
    assert_eq!(config.client_id, OPENAI_CODEX_CLIENT_ID);
    assert_eq!(config.authorize_url, OPENAI_AUTHORIZE_URL);
    assert_eq!(config.token_url, OPENAI_TOKEN_URL);
    assert!(config.scopes.iter().any(|scope| scope == "offline_access"));
}

#[test]
fn codex_cli_parser_normalizes_tokens_and_reads_jwt_metadata() {
    let access_token = unsigned_jwt(serde_json::json!({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        "exp": 2_000_000_000_i64,
    }));
    let bytes = serde_json::to_vec(&serde_json::json!({
        "tokens": {
            "access_token": access_token,
            "refresh_token": " refresh ",
            "id_token": " id "
        }
    }))
    .expect("fixture");
    let imported = parse_openai_codex_auth_json(&bytes).expect("parse");
    assert_eq!(imported.account_id.as_deref(), Some("acct_123"));
    assert_eq!(imported.expires_at_unix, Some(2_000_000_000));
    assert_eq!(imported.token.refresh_token.as_deref(), Some("refresh"));
    assert_eq!(imported.token.id_token.as_deref(), Some("id"));
}

#[test]
fn codex_cli_parser_rejects_missing_access_token() {
    let error = parse_openai_codex_auth_json(br#"{"tokens":{}}"#).unwrap_err();
    assert!(error.contains("access token"));
}

#[test]
fn debug_output_redacts_oauth_credentials() {
    let token = OAuthTokenSet {
        access_token: "access-secret".to_string(),
        refresh_token: Some("refresh-secret".to_string()),
        id_token: Some("id-secret".to_string()),
        expires_in: 3600,
        issued_at: 1,
    };
    let debug = format!("{token:?}");
    assert!(!debug.contains("access-secret"));
    assert!(!debug.contains("refresh-secret"));
    assert!(!debug.contains("id-secret"));
    assert!(debug.contains("[REDACTED]"));
}
