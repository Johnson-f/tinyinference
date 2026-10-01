use super::*;

#[test]
fn timeline_is_nonempty_and_scales() {
    let short = synthetic_viseme_timeline("hi");
    let long = synthetic_viseme_timeline("the quick brown fox");
    assert_eq!(short[0].viseme, "sil");
    assert!(long.last().unwrap().end_ms > short.last().unwrap().end_ms);
}
