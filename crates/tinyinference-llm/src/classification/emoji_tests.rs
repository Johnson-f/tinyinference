use super::*;

#[test]
fn extract_emoji_from_simple_string() {
    assert_eq!(extract_first_emoji("👍"), Some("👍".to_string()));
    assert_eq!(extract_first_emoji("🔥"), Some("🔥".to_string()));
    assert_eq!(extract_first_emoji("❤️"), Some("❤️".to_string()));
}

#[test]
fn extract_emoji_with_surrounding_text() {
    assert_eq!(extract_first_emoji("Sure! 😂"), Some("😂".to_string()));
    assert_eq!(
        extract_first_emoji("I think 👀 fits here"),
        Some("👀".to_string())
    );
}

#[test]
fn extract_none_when_no_emoji() {
    assert_eq!(extract_first_emoji("NONE"), None);
    assert_eq!(extract_first_emoji("no reaction"), None);
    assert_eq!(extract_first_emoji(""), None);
}

#[test]
fn extract_flag_emoji_keeps_pair_together() {
    assert_eq!(extract_first_emoji("🇺🇸"), Some("🇺🇸".to_string()));
    assert_eq!(
        extract_first_emoji("🇬🇧 Great Britain"),
        Some("🇬🇧".to_string())
    );
}

#[test]
fn is_emoji_start_recognizes_common_emojis() {
    assert!(is_emoji_start('👍'));
    assert!(is_emoji_start('🔥'));
    assert!(is_emoji_start('😂'));
    assert!(is_emoji_start('⭐'));
    assert!(!is_emoji_start('A'));
    assert!(!is_emoji_start('1'));
}

#[test]
fn zwj_sequences_stay_together() {
    assert_eq!(
        extract_first_emoji("👩\u{200D}💻 coding"),
        Some("👩\u{200D}💻".to_string())
    );
}
