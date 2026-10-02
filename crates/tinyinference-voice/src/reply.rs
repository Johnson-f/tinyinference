//! Reply-speech response normalization: turns the tolerant JSON a speech
//! backend returns (several key spellings, ms or seconds timings) into one
//! typed shape with an Oculus-15 viseme timeline for lip-sync.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::VisemeFrame;

/// Char-level timing returned by some backends (e.g. ElevenLabs alignment).
/// Not directly rendered, but kept so the UI can derive a fallback timeline
/// when the backend does not ship visemes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlignmentFrame {
    /// The character this timing covers.
    pub char: String,
    /// Start of the character in milliseconds.
    pub start_ms: u64,
    /// End of the character in milliseconds.
    pub end_ms: u64,
}

/// Normalized reply-speech response handed to a UI — audio, its MIME type, a
/// viseme timeline and optional char-level alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplySpeech {
    /// Base64-encoded audio bytes.
    pub audio_base64: String,
    /// MIME type of the audio.
    pub audio_mime: String,
    /// Viseme timeline for lip-sync.
    pub visemes: Vec<VisemeFrame>,
    /// Optional char-level alignment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alignment: Option<Vec<AlignmentFrame>>,
}

/// Translate the backend's tolerant response shape into the UI contract.
/// Accepts `visemes` / `cues` / `viseme_cues`, and per-frame
/// `start_ms`+`end_ms` or `time_ms`+`duration_ms`.
pub fn normalize_response(raw: &Value) -> ReplySpeech {
    let audio_base64 = raw
        .get("audio_base64")
        .or_else(|| raw.get("audio"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let audio_mime = raw
        .get("audio_mime")
        .or_else(|| raw.get("mime"))
        .and_then(Value::as_str)
        .unwrap_or("audio/mpeg")
        .to_string();

    let cues = raw
        .get("visemes")
        .or_else(|| raw.get("cues"))
        .or_else(|| raw.get("viseme_cues"));
    let visemes = cues
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(parse_cue).collect::<Vec<_>>())
        .unwrap_or_default();

    let alignment = raw
        .get("alignment")
        .or_else(|| raw.get("characters"))
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(parse_alignment).collect::<Vec<_>>());

    ReplySpeech {
        audio_base64,
        audio_mime,
        visemes,
        alignment,
    }
}

fn parse_cue(v: &Value) -> Option<VisemeFrame> {
    let viseme = v
        .get("viseme")
        .or_else(|| v.get("v"))
        .or_else(|| v.get("code"))
        .and_then(Value::as_str)?
        .to_string();
    if viseme.is_empty() {
        return None;
    }
    // Accept millisecond keys, or seconds keys (`startSeconds`/`endSeconds`, the
    // shape the cloud backend actually ships) converted to ms. Without the
    // seconds keys every frame collapsed to start=0/end=80 — the mascot mouth
    // then froze on the first viseme because the whole track had no real timing.
    let start = read_ms(
        v,
        &["start_ms", "time_ms", "t"],
        &["startSeconds", "start_seconds", "startSec"],
    )
    .unwrap_or(0);
    let end = read_ms(v, &["end_ms"], &["endSeconds", "end_seconds", "endSec"])
        .or_else(|| {
            let t = read_ms(
                v,
                &["time_ms", "t"],
                &["startSeconds", "start_seconds", "startSec"],
            )?;
            let d = read_ms(
                v,
                &["duration_ms", "d"],
                &["durationSeconds", "duration_seconds", "durationSec"],
            )?;
            Some(t + d)
        })
        .unwrap_or(start + 80);
    if end <= start {
        return None;
    }
    Some(VisemeFrame {
        viseme,
        start_ms: start,
        end_ms: end,
    })
}

fn parse_alignment(v: &Value) -> Option<AlignmentFrame> {
    let ch = v.get("char").and_then(Value::as_str)?.to_string();
    let start = read_ms(v, &["start_ms"], &["startSeconds", "start_seconds"])?;
    let end = read_ms(v, &["end_ms"], &["endSeconds", "end_seconds"])?;
    if end <= start {
        return None;
    }
    Some(AlignmentFrame {
        char: ch,
        start_ms: start,
        end_ms: end,
    })
}

fn read_u64(v: &Value, keys: &[&str]) -> Option<u64> {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(Value::as_u64) {
            return Some(n);
        }
        if let Some(f) = v.get(*k).and_then(Value::as_f64)
            && f.is_finite()
            && f >= 0.0
        {
            return Some(f as u64);
        }
    }
    None
}

/// Read a time value in milliseconds. Tries `ms_keys` first (integer or float
/// ms), then `sec_keys` interpreted as seconds and converted to ms. This lets
/// the parser tolerate both the `*_ms` contract and the backend's
/// `*Seconds` shape without the caller caring which arrived.
fn read_ms(v: &Value, ms_keys: &[&str], sec_keys: &[&str]) -> Option<u64> {
    if let Some(ms) = read_u64(v, ms_keys) {
        return Some(ms);
    }
    for k in sec_keys {
        if let Some(f) = v.get(*k).and_then(Value::as_f64)
            && f.is_finite()
            && f >= 0.0
        {
            return Some((f * 1000.0).round() as u64);
        }
        if let Some(n) = v.get(*k).and_then(Value::as_u64) {
            return Some(n.saturating_mul(1000));
        }
    }
    None
}

#[cfg(test)]
#[path = "reply_tests.rs"]
mod tests;
