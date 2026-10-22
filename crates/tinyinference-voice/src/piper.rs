//! Local Piper text-to-speech execution.

use std::path::Path;

use std::ffi::OsString;

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use tokio::io::AsyncWriteExt;

use crate::{PiperSpeech, VisemeFrame};

/// Run Piper with explicit binary and voice-model paths.
pub async fn synthesize(
    piper_binary: &Path,
    voice_model: &Path,
    text: &str,
) -> Result<PiperSpeech, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("text is required".to_string());
    }
    let output_path = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|error| format!("failed to create piper output file: {error}"))?
        .into_temp_path();
    synthesize_to_file(piper_binary, voice_model, text, &output_path, Some(1.15)).await?;
    let bytes = tokio::fs::read(&output_path)
        .await
        .map_err(|error| format!("failed to read piper output: {error}"))?;
    Ok(PiperSpeech {
        audio_base64: BASE64.encode(bytes),
        audio_mime: "audio/wav".to_string(),
        visemes: synthetic_viseme_timeline(text),
    })
}

/// Runs Piper and writes its WAV output to the caller's path.
///
/// When `length_scale` is `None`, Piper's own default is used. The caller
/// owns the output path and can read or stream the generated WAV directly.
///
/// # Errors
///
/// Returns an error when text is empty, Piper cannot be started, writing its
/// input fails, or Piper exits unsuccessfully.
pub async fn synthesize_to_file(
    piper_binary: &Path,
    voice_model: &Path,
    text: &str,
    output_path: &Path,
    length_scale: Option<f32>,
) -> Result<(), String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("text is required".to_string());
    }
    let mut child = tokio::process::Command::new(piper_binary)
        .args(piper_arguments(voice_model, output_path, length_scale))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to launch piper: {error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|error| format!("failed to write text to piper stdin: {error}"))?;
    }
    let output = child
        .wait_with_output()
        .await
        .map_err(|error| format!("failed to wait on piper: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "piper failed (exit={:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

fn piper_arguments(
    voice_model: &Path,
    output_path: &Path,
    length_scale: Option<f32>,
) -> Vec<OsString> {
    let mut arguments = vec![
        "--model".into(),
        voice_model.as_os_str().to_owned(),
        "--output_file".into(),
        output_path.as_os_str().to_owned(),
    ];
    if let Some(length_scale) = length_scale {
        arguments.push("--length-scale".into());
        arguments.push(length_scale.to_string().into());
    }
    arguments
}

/// Build a neutral fallback viseme timeline from text length.
#[must_use]
pub fn synthetic_viseme_timeline(text: &str) -> Vec<VisemeFrame> {
    let duration = (text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count()
        .max(1) as u64)
        * 80;
    vec![
        VisemeFrame {
            viseme: "sil".to_string(),
            start_ms: 0,
            end_ms: 40,
        },
        VisemeFrame {
            viseme: "aa".to_string(),
            start_ms: 40,
            end_ms: duration.max(80),
        },
    ]
}

#[cfg(test)]
#[path = "piper_tests.rs"]
mod tests;
