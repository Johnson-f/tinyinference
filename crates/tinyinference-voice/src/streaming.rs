//! PCM16 streaming buffer mechanics.

/// Input sample rate expected by the streaming protocol.
pub const AUDIO_SAMPLE_RATE: usize = 16_000;
/// Maximum retained sliding window: fifteen seconds.
pub const MAX_STREAM_BUFFER_SAMPLES: usize = AUDIO_SAMPLE_RATE * 15;
/// Maximum complete recording: five minutes.
pub const MAX_FULL_AUDIO_SAMPLES: usize = AUDIO_SAMPLE_RATE * 60 * 5;

/// Decode a little-endian PCM16 frame, rejecting incomplete samples.
#[must_use]
pub fn decode_pcm16le_frame(data: &[u8]) -> Option<Vec<i16>> {
    if !data.len().is_multiple_of(2) {
        return None;
    }
    let (samples, remainder) = data.as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    Some(samples.iter().copied().map(i16::from_le_bytes).collect())
}

/// Append samples to the sliding and full buffers without exceeding the cap.
pub fn append_stream_samples(
    audio_buf: &mut Vec<i16>,
    full_audio_buf: &mut Vec<i16>,
    samples: &[i16],
) -> bool {
    if full_audio_buf.len().saturating_add(samples.len()) > MAX_FULL_AUDIO_SAMPLES {
        return false;
    }
    full_audio_buf.extend_from_slice(samples);
    audio_buf.extend_from_slice(samples);
    if audio_buf.len() > MAX_STREAM_BUFFER_SAMPLES {
        audio_buf.drain(..audio_buf.len() - MAX_STREAM_BUFFER_SAMPLES);
    }
    true
}

/// Return whether a JSON command requests the end of the recording.
#[must_use]
pub fn is_stop_command(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| value.get("type")?.as_str().map(ToOwned::to_owned))
        .is_some_and(|kind| kind == "stop")
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
