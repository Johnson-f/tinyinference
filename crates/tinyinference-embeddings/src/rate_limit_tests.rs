use super::*;

#[test]
fn loopback_detection_is_fail_closed() {
    assert!(is_loopback_url("http://localhost:11434"));
    assert!(is_loopback_url("http://127.0.0.1:8080"));
    assert!(is_loopback_url("http://[::1]:8080"));
    assert!(!is_loopback_url("https://api.openai.com"));
    assert!(!is_loopback_url("not a url"));
}

#[test]
fn bucket_math_paces_without_bursting() {
    let mut tokens = 1.0;
    assert!(refill_and_take(&mut tokens, 1.0, 0.0).is_none());
    let wait = refill_and_take(&mut tokens, 1.0, 0.25).unwrap();
    assert!((wait.as_secs_f64() - 0.75).abs() < 1e-6);
}

#[tokio::test]
async fn disabled_and_loopback_limits_never_block() {
    acquire_with_limit("https://api.example.com", 0).await;
    acquire_with_limit("http://127.0.0.1:1", 1).await;
}
