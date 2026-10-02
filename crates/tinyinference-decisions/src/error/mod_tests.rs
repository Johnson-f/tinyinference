//! Error rendering behavior.

use super::*;

#[test]
fn messages_do_not_render_secrets_or_response_bodies() {
    let errors = [
        Error::MissingApiKey,
        Error::Authentication,
        Error::Unprocessable,
        Error::RateLimited,
        Error::Overloaded,
        Error::HttpStatus { status: 503 },
        Error::Timeout,
    ];
    for error in errors {
        let rendered = error.to_string();
        assert!(!rendered.contains("Bearer"));
        assert!(!rendered.contains("apikey_"));
    }
}

#[test]
fn contextual_errors_name_their_category() {
    assert_eq!(
        Error::invalid_request("questions must not be empty").to_string(),
        "invalid request: questions must not be empty"
    );
    assert_eq!(
        Error::invalid_response("answer missing").to_string(),
        "invalid response: answer missing"
    );
}
