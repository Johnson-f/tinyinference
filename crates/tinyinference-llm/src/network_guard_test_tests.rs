use super::*;
use std::sync::Mutex;

// The guard is process-wide `static` state, so tests that flip it must
// not interleave with each other; serialize them with a mutex rather
// than relying on cargo test's default single-process, multi-thread
// execution to happen to avoid collisions.
static GUARD_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn denies_and_allows_round_trip() {
    let _lock = GUARD_TEST_LOCK.lock().unwrap();
    allow_network_models();
    assert!(!network_models_denied());
    assert!(ensure_network_models_allowed().is_ok());

    deny_network_models();
    assert!(network_models_denied());
    assert!(ensure_network_models_allowed().is_err());

    allow_network_models();
    assert!(!network_models_denied());
    assert!(ensure_network_models_allowed().is_ok());
}

#[test]
fn deny_is_idempotent() {
    let _lock = GUARD_TEST_LOCK.lock().unwrap();
    deny_network_models();
    deny_network_models();
    assert!(network_models_denied());
    allow_network_models();
}
