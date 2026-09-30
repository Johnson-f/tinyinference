//! Secret scrubbing and bounded provider-error formatting.

use regex::Regex;
use std::sync::LazyLock;

/// Maximum number of characters retained from a provider API error.
pub const MAX_API_ERROR_CHARS: usize = 200;
const TRANSPORT_ERROR_MAX_CHARS: usize = 1200;

/// Redact credentials carried by a URL while retaining its routing shape.
///
/// Userinfo and fragments are removed. Query parameter names remain visible for
/// diagnostics, but every value is replaced so presigned URLs and API keys can
/// never reach logs or error strings.
pub fn redact_url(input: &str) -> String {
    let Ok(mut url) = url::Url::parse(input.trim()) else {
        return "[REDACTED INVALID URL]".to_string();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_fragment(None);
    let names = url
        .query_pairs()
        .map(|(name, _)| name.into_owned())
        .collect::<Vec<_>>();
    url.set_query(None);
    if !names.is_empty() {
        let mut query = url.query_pairs_mut();
        for name in names {
            query.append_pair(&name, "[REDACTED]");
        }
    }
    url.to_string()
}

fn truncate_with_suffix(input: &str, max_chars: usize, suffix: &str) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let suffix_chars = suffix.chars().count();
    if suffix_chars >= max_chars {
        return suffix.chars().take(max_chars).collect();
    }
    let mut truncated: String = input.chars().take(max_chars - suffix_chars).collect();
    truncated.push_str(suffix);
    truncated
}

fn is_secret_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':')
}

fn token_end(input: &str, from: usize) -> usize {
    let mut end = from;
    for (i, c) in input[from..].char_indices() {
        if is_secret_char(c) {
            end = from + i + c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Scrub known secret-like token prefixes from provider error strings.
pub fn scrub_secret_patterns(input: &str) -> String {
    const PREFIXES: [&str; 7] = [
        "sk-",
        "xoxb-",
        "xoxp-",
        "ghp_",
        "gho_",
        "ghu_",
        "github_pat_",
    ];

    let mut scrubbed = input.to_string();

    for prefix in PREFIXES {
        let mut search_from = 0;
        while let Some(rel) = scrubbed[search_from..].find(prefix) {
            let start = search_from + rel;
            let content_start = start + prefix.len();
            let end = token_end(&scrubbed, content_start);

            if end == content_start {
                search_from = content_start;
                continue;
            }

            scrubbed.replace_range(start..end, "[REDACTED]");
            search_from = start + "[REDACTED]".len();
        }
    }

    scrubbed
}

/// Key/value credential shapes: `token: "…"`, `api_key=…`, `bearer: …`, etc.
static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(token|api[_-]?key|password|secret|user[_-]?key|bearer|credential)["']?\s*[:=]\s*(?:"((?:\\.|[^"\\]){8,})"|'((?:\\.|[^'\\]){8,})'|([a-zA-Z0-9_+./=\-]{8,}))"#).unwrap()
});

/// Bare AWS access-key IDs — `AKIA…`/`ASIA…` followed by 16 base32 chars — which
/// appear naked in env dumps and config reads with no surrounding key name.
static AWS_ACCESS_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b").unwrap());

/// Bare OpenAI-style secret keys — `sk-…` (incl. `sk-proj-…`) with a long token
/// body. Not necessarily attached to a `key:` label in raw API responses.
static OPENAI_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(sk-[A-Za-z0-9_\-]{16,})\b").unwrap());

/// Space-separated bearer tokens as they appear in HTTP auth headers
/// (`Authorization: Bearer <token>`) — the KV regex only catches `bearer:`/`=`.
static BEARER_SPACE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(Bearer)\s+([A-Za-z0-9_\-\.=+/]{16,})").unwrap());

/// Authorization headers can carry Basic credentials, which are encoded but
/// remain recoverable and must be removed in full.
static AUTHORIZATION_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(authorization["']?\s*[:=]\s*["']?)(basic|bearer)\s+([A-Za-z0-9_\-\.=+/]{8,})"#,
    )
    .unwrap()
});

/// Preserve the first 4 chars of `val` for context, returning the redacted
/// prefix (empty when the value is too short to safely reveal any of it).
fn redact_prefix(val: &str) -> &str {
    if val.chars().count() > 4 {
        match val.char_indices().nth(4) {
            Some((idx, _)) => &val[..idx],
            None => val,
        }
    } else {
        ""
    }
}

/// Scrub credentials from tool output to prevent accidental exfiltration.
///
/// Complements [`scrub_secret_patterns`], which redacts known token *prefixes*
/// from provider errors; this one covers labelled key/value pairs and a few
/// bare secret shapes.
///
/// Replaces known credential patterns with a redacted placeholder while preserving
/// a small prefix for context.
///
/// Covers labelled key/value pairs plus bare secrets that show up unlabelled in
/// env dumps, config reads and API responses: AWS access-key IDs (`AKIA…`/
/// `ASIA…`), OpenAI-style `sk-…` keys, and space-separated `Bearer <token>`
/// auth headers.
pub fn scrub_credentials(input: &str) -> String {
    let stage_kv = SENSITIVE_KV_REGEX.replace_all(input, |caps: &regex::Captures<'_>| {
        let full_match = &caps[0];
        let value = caps
            .get(2)
            .or(caps.get(3))
            .or(caps.get(4))
            .expect("sensitive key-value match has a value");
        // Replace only the value span. Rebuilding the key from captures used
        // to add a second opening quote to JSON (`""token": ...`), making
        // tool results impossible to parse after redaction.
        let start = value.start() - caps.get(0).expect("full match").start();
        let end = value.end() - caps.get(0).expect("full match").start();
        format!(
            "{}{}*[REDACTED]{}",
            &full_match[..start],
            redact_prefix(value.as_str()),
            &full_match[end..]
        )
    });

    // Bare AWS access-key IDs: keep the 4-char `AKIA`/`ASIA` prefix for context.
    let stage_aws = AWS_ACCESS_KEY_REGEX.replace_all(&stage_kv, |caps: &regex::Captures<'_>| {
        format!("{}*[REDACTED]", redact_prefix(&caps[1]))
    });

    // Bare `sk-…` keys: keep the `sk-` scheme, redact the secret body.
    let stage_openai = OPENAI_KEY_REGEX.replace_all(&stage_aws, |_caps: &regex::Captures<'_>| {
        "sk-*[REDACTED]".to_string()
    });

    // Space-separated `Bearer <token>`: keep the scheme word, redact the token.
    let stage_authorization = AUTHORIZATION_REGEX
        .replace_all(&stage_openai, |caps: &regex::Captures<'_>| {
            format!("{}{} *[REDACTED]", &caps[1], &caps[2])
        });

    BEARER_SPACE_REGEX
        .replace_all(&stage_authorization, |caps: &regex::Captures<'_>| {
            format!("{} *[REDACTED]", &caps[1])
        })
        .to_string()
}

/// Sanitize API error text by scrubbing secrets and truncating length.
pub fn sanitize_api_error(input: &str) -> String {
    let scrubbed = scrub_secret_patterns(input);
    truncate_with_suffix(&scrubbed, MAX_API_ERROR_CHARS, "...")
}

/// Full `source()` chain for connection / TLS failures (scrubbed, longer than API body snippets).
pub fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut parts: Vec<String> = vec![err.to_string()];
    let mut src = std::error::Error::source(err);
    while let Some(e) = src {
        parts.push(e.to_string());
        src = std::error::Error::source(e);
    }
    let joined = parts.join(" | ");
    let scrubbed = scrub_secret_patterns(&joined);
    truncate_with_suffix(&scrubbed, TRANSPORT_ERROR_MAX_CHARS, "…")
}

/// Cause chain from [`anyhow::Error`] (e.g. responses fallback), scrubbed and length-limited.
pub fn format_anyhow_chain(err: &anyhow::Error) -> String {
    let joined = err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    let scrubbed = scrub_secret_patterns(&joined);
    truncate_with_suffix(&scrubbed, TRANSPORT_ERROR_MAX_CHARS, "…")
}

#[cfg(test)]
mod tests {
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
        let parsed: serde_json::Value =
            serde_json::from_str(&output).expect("valid JSON after scrub");
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
    fn scrub_credentials_empty_input() {
        assert_eq!(scrub_credentials(""), "");
    }

    #[test]
    fn scrub_credentials_no_sensitive_data() {
        let input = "normal text without any secrets";
        assert_eq!(scrub_credentials(input), input);
    }

    #[test]
    fn scrub_credentials_short_values_not_redacted() {
        // Values shorter than 8 chars are not redacted.
        let input = r#"api_key="short""#;
        assert_eq!(scrub_credentials(input), input);
    }
}
