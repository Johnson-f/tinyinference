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
mod tests {
    use super::*;

    #[test]
    fn decodes_pcm16_and_rejects_odd_frames() {
        assert_eq!(decode_pcm16le_frame(&[1, 0, 255, 255]), Some(vec![1, -1]));
        assert!(decode_pcm16le_frame(&[1]).is_none());
    }

    #[test]
    fn enforces_full_cap_and_sliding_window() {
        let mut window = vec![0; MAX_STREAM_BUFFER_SAMPLES];
        let mut full = vec![0; MAX_FULL_AUDIO_SAMPLES - 1];
        assert!(append_stream_samples(&mut window, &mut full, &[1]));
        assert_eq!(window.len(), MAX_STREAM_BUFFER_SAMPLES);
        assert!(!append_stream_samples(&mut window, &mut full, &[2]));
    }

    #[test]
    fn recognizes_only_stop_commands() {
        assert!(is_stop_command(r#"{"type":"stop"}"#));
        assert!(!is_stop_command(r#"{"type":"continue"}"#));
        assert!(!is_stop_command("invalid"));
    }

    #[test]
    fn keeps_full_audio_and_trims_only_the_sliding_window() {
        let mut window = vec![0; MAX_STREAM_BUFFER_SAMPLES - 2];
        let mut full = vec![1, 2];
        assert!(append_stream_samples(&mut window, &mut full, &[3, 4, 5, 6]));
        assert_eq!(full, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(window.len(), MAX_STREAM_BUFFER_SAMPLES);
        assert_eq!(&window[window.len() - 4..], &[3, 4, 5, 6]);
    }

    #[test]
    fn accepts_chunks_up_to_the_exact_full_cap() {
        let mut window = Vec::new();
        let mut full = Vec::new();
        let chunk = vec![0i16; 1_024];
        while full.len() + chunk.len() <= MAX_FULL_AUDIO_SAMPLES {
            assert!(append_stream_samples(&mut window, &mut full, &chunk));
        }
        assert!(full.len() <= MAX_FULL_AUDIO_SAMPLES);
        let remainder = vec![0i16; MAX_FULL_AUDIO_SAMPLES - full.len()];
        assert!(append_stream_samples(&mut window, &mut full, &remainder));
        assert_eq!(full.len(), MAX_FULL_AUDIO_SAMPLES);
        assert_eq!(window.len(), MAX_STREAM_BUFFER_SAMPLES);
        assert!(!append_stream_samples(&mut window, &mut full, &[1]));
    }

    #[test]
    fn rejects_a_single_chunk_that_crosses_the_cap_without_partial_writes() {
        let mut window = Vec::new();
        let mut full = Vec::new();
        assert!(append_stream_samples(
            &mut window,
            &mut full,
            &vec![0i16; MAX_FULL_AUDIO_SAMPLES - 1],
        ));
        let window_before = window.clone();

        assert!(!append_stream_samples(&mut window, &mut full, &[7, 8]));
        assert_eq!(full.len(), MAX_FULL_AUDIO_SAMPLES - 1);
        assert_eq!(
            window, window_before,
            "a rejected chunk must not be windowed"
        );
    }

    #[test]
    fn rejects_input_once_the_full_cap_is_reached_and_leaves_the_window_alone() {
        let mut window = Vec::new();
        let mut full = vec![0i16; MAX_FULL_AUDIO_SAMPLES];
        assert!(!append_stream_samples(&mut window, &mut full, &[1, 2, 3]));
        assert_eq!(full.len(), MAX_FULL_AUDIO_SAMPLES);
        assert!(window.is_empty());
    }

    #[test]
    fn a_chunk_larger_than_the_window_keeps_only_the_newest_samples() {
        let mut window = Vec::new();
        let mut full = Vec::new();
        let chunk: Vec<i16> = (0..MAX_STREAM_BUFFER_SAMPLES as i32 + 10)
            .map(|i| (i % 1_000) as i16)
            .collect();
        assert!(append_stream_samples(&mut window, &mut full, &chunk));
        assert_eq!(full.len(), chunk.len());
        assert_eq!(window, chunk[10..]);
    }
}
