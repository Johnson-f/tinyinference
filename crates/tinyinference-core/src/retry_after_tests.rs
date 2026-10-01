use super::*;

#[test]
fn parses_delta_seconds_and_caps() {
    assert_eq!(parse_retry_after_ms(Some(" 5 ")), Some(5_000));
    assert_eq!(parse_retry_after_ms(Some("999")), Some(MAX_BACKOFF_MS));
    assert_eq!(parse_retry_after_ms(Some("-1")), None);
}

#[test]
fn parses_http_dates() {
    let now = httpdate::parse_http_date("Wed, 21 Oct 2015 07:27:55 GMT").unwrap();
    assert_eq!(
        parse_retry_after_ms_at(Some("Wed, 21 Oct 2015 07:28:00 GMT"), now),
        Some(5_000)
    );
}

#[test]
fn falls_back_to_bounded_exponential_backoff() {
    assert_eq!(backoff_ms_for_attempt(0, None), 1_000);
    assert_eq!(backoff_ms_for_attempt(2, None), 4_000);
    assert_eq!(backoff_ms_for_attempt(20, None), MAX_BACKOFF_MS);
}
