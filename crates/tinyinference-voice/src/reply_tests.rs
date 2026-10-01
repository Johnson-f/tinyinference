use super::*;
use serde_json::json;

#[test]
fn normalize_canonical_shape() {
    let raw = json!({
        "audio_base64": "AAA=",
        "audio_mime": "audio/mpeg",
        "visemes": [
            { "viseme": "sil", "start_ms": 0, "end_ms": 100 },
            { "viseme": "aa", "start_ms": 100, "end_ms": 250 },
        ],
    });
    let r = normalize_response(&raw);
    assert_eq!(r.audio_base64, "AAA=");
    assert_eq!(r.audio_mime, "audio/mpeg");
    assert_eq!(r.visemes.len(), 2);
    assert_eq!(r.visemes[1].viseme, "aa");
    assert_eq!(r.visemes[1].end_ms, 250);
}

#[test]
fn normalize_accepts_cues_and_short_keys() {
    let raw = json!({
        "audio": "BBB=",
        "mime": "audio/wav",
        "cues": [{ "v": "PP", "t": 0, "d": 80 }],
    });
    let r = normalize_response(&raw);
    assert_eq!(r.audio_base64, "BBB=");
    assert_eq!(r.audio_mime, "audio/wav");
    assert_eq!(
        r.visemes,
        vec![VisemeFrame {
            viseme: "PP".into(),
            start_ms: 0,
            end_ms: 80
        }]
    );
}

#[test]
fn normalize_accepts_seconds_keys() {
    // The cloud backend ships per-frame timing as seconds (`startSeconds`/
    // `endSeconds`); the parser must convert to ms rather than dropping it
    // (which collapsed every frame to start=0/end=80 and froze the mouth).
    let raw = json!({
        "audio_base64": "AAA=",
        "visemes": [
            { "viseme": "sil", "startSeconds": 0.0, "endSeconds": 0.12 },
            { "viseme": "aa", "startSeconds": 0.12, "endSeconds": 0.45 },
            // A gap before the next cue (0.45 → 0.90) is a real pause: the
            // mouth rests there. Preserved because we keep the true ends.
            { "viseme": "PP", "startSeconds": 0.9, "endSeconds": 1.05 },
        ],
        "alignment": [
            { "char": "h", "startSeconds": 0.0, "endSeconds": 0.05 },
        ],
    });
    let r = normalize_response(&raw);
    assert_eq!(r.visemes.len(), 3);
    assert_eq!(
        r.visemes[0],
        VisemeFrame {
            viseme: "sil".into(),
            start_ms: 0,
            end_ms: 120
        }
    );
    assert_eq!(
        r.visemes[1],
        VisemeFrame {
            viseme: "aa".into(),
            start_ms: 120,
            end_ms: 450
        }
    );
    assert_eq!(
        r.visemes[2],
        VisemeFrame {
            viseme: "PP".into(),
            start_ms: 900,
            end_ms: 1050
        }
    );
    let alignment = r.alignment.expect("alignment present");
    assert_eq!(alignment.len(), 1);
    assert_eq!(alignment[0].start_ms, 0);
    assert_eq!(alignment[0].end_ms, 50);
}

#[test]
fn normalize_accepts_short_seconds_duration_aliases() {
    let raw = json!({
        "audio_base64": "AAA=",
        "visemes": [
            { "viseme": "aa", "startSec": 1.2, "durationSec": 0.15 },
        ],
    });
    let r = normalize_response(&raw);
    assert_eq!(
        r.visemes,
        vec![VisemeFrame {
            viseme: "aa".into(),
            start_ms: 1200,
            end_ms: 1350
        }]
    );
}

#[test]
fn normalize_drops_malformed_cues() {
    let raw = json!({
        "audio_base64": "CCC=",
        "visemes": [
            { "viseme": "aa", "start_ms": 0, "end_ms": 100 },
            { "viseme": "",   "start_ms": 100, "end_ms": 200 },
            { "viseme": "PP", "start_ms": 200, "end_ms": 200 },
        ],
    });
    let r = normalize_response(&raw);
    assert_eq!(r.visemes.len(), 1);
    assert_eq!(r.visemes[0].viseme, "aa");
}

#[test]
fn normalize_passes_through_alignment() {
    let raw = json!({
        "audio_base64": "DDD=",
        "alignment": [{ "char": "h", "start_ms": 0, "end_ms": 50 }],
    });
    let r = normalize_response(&raw);
    assert_eq!(r.alignment.as_deref().unwrap()[0].char, "h");
}

#[test]
fn reply_speech_serializes_to_the_ui_shape() {
    let speech = ReplySpeech {
        audio_base64: "AAA=".into(),
        audio_mime: "audio/mpeg".into(),
        visemes: vec![VisemeFrame {
            viseme: "aa".into(),
            start_ms: 1,
            end_ms: 2,
        }],
        alignment: None,
    };
    assert_eq!(
        serde_json::to_value(&speech).unwrap(),
        json!({
            "audio_base64": "AAA=",
            "audio_mime": "audio/mpeg",
            "visemes": [{"viseme": "aa", "start_ms": 1, "end_ms": 2}],
        })
    );
    let with_alignment = ReplySpeech {
        alignment: Some(vec![AlignmentFrame {
            char: "h".into(),
            start_ms: 0,
            end_ms: 5,
        }]),
        ..speech
    };
    assert_eq!(
        serde_json::to_value(&with_alignment).unwrap()["alignment"],
        json!([{"char": "h", "start_ms": 0, "end_ms": 5}])
    );
}
