use super::*;

#[test]
fn truncation_limit_includes_the_suffix() {
    let sanitized = sanitize_api_error(&"x".repeat(MAX_API_ERROR_CHARS + 50));
    assert_eq!(sanitized.chars().count(), MAX_API_ERROR_CHARS);
    assert!(sanitized.ends_with("..."));
}

#[test]
fn secret_scrubbing_preserves_unicode_boundaries() {
    assert_eq!(
        scrub_secret_patterns("é before sk-secret after 🚀"),
        "é before [REDACTED] after 🚀"
    );
}

#[test]
fn test_scrub_credentials_utf8() {
    // Regex requires at least 8 chars for the value
    // The [a-zA-Z0-9_\-\.]{8,} part of the regex does NOT match emoji
    // So we must use quotes to hit the "([^"]{8,})" part
    let input = "api_key: \"🦀🦀🦀🦀🦀🦀🦀🦀\"";
    let output = scrub_credentials(input);
    // Should preserve 4 crabs and then redact
    assert!(output.contains("🦀🦀🦀🦀*[REDACTED]"));
}

#[test]
fn test_scrub_credentials_short_val() {
    let input = "api_key: 12345678";
    let output = scrub_credentials(input);
    assert!(output.contains("api_key: 1234*[REDACTED]"));
}

#[test]
fn scrubbed_json_keeps_its_quoted_key_and_parses() {
    let input = r#"{"token":"example-secret-value","status":"ready"}"#;
    let output = scrub_credentials(input);
    let parsed: serde_json::Value = serde_json::from_str(&output).expect("valid JSON after scrub");
    assert_eq!(parsed["status"], "ready");
    assert!(!output.contains("example-secret-value"));
    assert!(output.contains("*[REDACTED]"));
}

// #4453: bare, unlabelled secrets that show up in env dumps / API responses.

#[test]
fn scrubs_bare_aws_access_key() {
    let out = scrub_credentials("config dump AKIAIOSFODNN7EXAMPLE trailing text");
    assert!(
        !out.contains("AKIAIOSFODNN7EXAMPLE"),
        "bare AWS access key must be redacted: {out}"
    );
    assert!(
        out.contains("[REDACTED]"),
        "redaction marker present: {out}"
    );
}

#[test]
fn scrubs_bare_openai_key() {
    let out = scrub_credentials("response body sk-abcdefghij1234567890ABCDEFGHIJ end");
    assert!(
        !out.contains("abcdefghij1234567890ABCDEFGHIJ"),
        "openai secret body must be redacted: {out}"
    );
    assert!(
        out.contains("sk-"),
        "the sk- scheme is kept for context: {out}"
    );
    assert!(
        out.contains("[REDACTED]"),
        "redaction marker present: {out}"
    );
}

#[test]
fn scrubs_space_separated_bearer_token() {
    let out = scrub_credentials("Authorization: Bearer abcDEF1234567890ghijklmnop done");
    assert!(
        !out.contains("abcDEF1234567890ghijklmnop"),
        "space-separated bearer token must be redacted: {out}"
    );
    assert!(
        out.contains("Bearer"),
        "the scheme word is kept for context: {out}"
    );
    assert!(
        out.contains("[REDACTED]"),
        "redaction marker present: {out}"
    );
}

#[test]
fn scrubs_complete_basic_authorization_credentials() {
    let out = scrub_credentials("Authorization: Basic dXNlcjpzdXBlclNlY3JldA==");
    assert!(!out.contains("dXNlcjpzdXBlclNlY3JldA=="));
    assert!(out.contains("Basic *[REDACTED]"));
}

#[test]
fn scrubs_plus_and_escaped_quotes_in_labelled_values() {
    let plus = scrub_credentials("api_key=abcd+efgh/ijklmnop==");
    assert!(!plus.contains("efgh/ijklmnop"));

    let escaped = scrub_credentials(r#"token="abcd\"efghijklmnop""#);
    assert!(!escaped.contains("efghijklmnop"));
}

#[test]
fn redacts_short_labelled_values_and_keeps_json_escape_pairs_valid() {
    let short = scrub_credentials(r#"{"api_key":"short"}"#);
    assert!(!short.contains("short"));

    let escaped = scrub_credentials(r#"{"token":"abc\\secret-value"}"#);
    let parsed: serde_json::Value =
        serde_json::from_str(&escaped).expect("escaped JSON remains valid");
    assert_eq!(parsed["token"], "abc\\*[REDACTED]");
    assert!(!escaped.contains("secret-value"));
}

#[test]
fn test_scrub_credentials() {
    let input = "API_KEY=sk-1234567890abcdef; token: 1234567890; password=\"secret123456\"";
    let scrubbed = scrub_credentials(input);
    assert!(scrubbed.contains("API_KEY=sk-1*[REDACTED]"));
    assert!(scrubbed.contains("token: 1234*[REDACTED]"));
    assert!(scrubbed.contains("password=\"secr*[REDACTED]\""));
    assert!(!scrubbed.contains("abcdef"));
    assert!(!scrubbed.contains("secret123456"));
}

#[test]
fn scrub_credentials_is_idempotent() {
    let once = scrub_credentials("token=aB3dEfGh1234 password: hunter2hunter2");
    assert_eq!(scrub_credentials(&once), once);
}

#[test]
fn scrub_credentials_empty_input() {
    assert_eq!(scrub_credentials(""), "");
}

#[test]
fn scrub_credentials_no_sensitive_data() {
    let input = "normal text without any secrets";
    assert_eq!(scrub_credentials(input), input);
}

#[test]
fn scrub_credentials_short_values_are_redacted() {
    let input = r#"api_key="short""#;
    let output = scrub_credentials(input);
    assert_eq!(output, r#"api_key="shor*[REDACTED]""#);
    assert!(!output.contains("short"));
}
