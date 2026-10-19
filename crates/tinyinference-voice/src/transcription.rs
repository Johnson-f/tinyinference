//! Provider-neutral transcription helpers for audio files and byte buffers.

use std::path::Path;

use async_trait::async_trait;

use crate::mime::mime_for_extension;

/// An STT adapter supplied by the host or a provider crate.
///
/// The host owns provider selection, credentials, HTTP policy, and response
/// policy. Implementations receive the complete audio payload and its file
/// metadata so they can adapt it to any provider protocol.
#[async_trait]
pub trait TranscriptionProvider: Send + Sync {
    /// Transcribes audio bytes and returns provider-neutral text.
    async fn transcribe(
        &self,
        audio: &[u8],
        mime_type: &str,
        file_name: &str,
    ) -> Result<String, String>;
}

/// Reads and transcribes a file using a caller-provided provider.
///
/// The audio is read once, an empty file is rejected, and MIME is inferred
/// from the lowercased file extension. Files without a UTF-8 name are passed
/// as `audio.wav`; unknown or absent extensions use the standard WAV MIME
/// fallback.
///
/// # Errors
///
/// Returns an error when the file cannot be read or is empty, or the provider
/// fails.
pub async fn transcribe_file<P>(path: impl AsRef<Path>, provider: &P) -> Result<String, String>
where
    P: TranscriptionProvider + ?Sized,
{
    let path = path.as_ref();
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| "audio.wav".to_string());
    transcribe_path(path, Some(&file_name), provider).await
}

async fn transcribe_path<P>(
    path: &Path,
    file_name: Option<&str>,
    provider: &P,
) -> Result<String, String>
where
    P: TranscriptionProvider + ?Sized,
{
    let audio = tokio::fs::read(path)
        .await
        .map_err(|error| format!("[voice-stt] failed to read audio file: {error}"))?;
    if audio.is_empty() {
        return Err("[voice-stt] audio file is empty".to_string());
    }

    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_default()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let mime = mime_for_extension(&extension);
    provider
        .transcribe(&audio, mime, file_name.unwrap_or("audio.wav"))
        .await
}

/// Writes audio bytes to a unique temporary file, transcribes them, and
/// removes the file on success, error, or cancellation.
///
/// `extension` may include a leading dot and surrounding whitespace. An
/// omitted extension defaults to `webm`; explicit values must contain only
/// ASCII letters and digits after normalization, so blank and dots-only values
/// are rejected.
///
/// # Errors
///
/// Returns an error when the audio is empty, the extension is invalid, the
/// temporary file cannot be created or written, or the provider fails.
pub async fn transcribe_bytes<P>(
    audio: &[u8],
    extension: Option<&str>,
    provider: &P,
) -> Result<String, String>
where
    P: TranscriptionProvider + ?Sized,
{
    transcribe_bytes_in(audio, extension, None, provider).await
}

async fn transcribe_bytes_in<P>(
    audio: &[u8],
    extension: Option<&str>,
    temp_dir: Option<&Path>,
    provider: &P,
) -> Result<String, String>
where
    P: TranscriptionProvider + ?Sized,
{
    if audio.is_empty() {
        return Err("[voice-stt] audio is empty".to_string());
    }
    let extension = sanitize_extension(extension)?;
    let suffix = format!(".{extension}");
    let mut builder = tempfile::Builder::new();
    builder.suffix(&suffix);
    let temporary_file = match temp_dir {
        Some(directory) => builder.tempfile_in(directory),
        None => builder.tempfile(),
    }
    .map_err(|error| format!("[voice-stt] failed to create temporary audio file: {error}"))?;
    // Keep only TempPath so its synchronous Drop cleanup still runs if this
    // async function is cancelled while writing or awaiting the provider.
    let temporary_path = temporary_file.into_temp_path();
    tokio::fs::write(&temporary_path, audio)
        .await
        .map_err(|error| format!("[voice-stt] failed to write temporary audio file: {error}"))?;

    let file_name = format!("audio.{extension}");
    let result = transcribe_path(&temporary_path, Some(&file_name), provider).await;
    drop(temporary_path);
    result
}

fn sanitize_extension(extension: Option<&str>) -> Result<String, String> {
    let Some(extension) = extension else {
        return Ok("webm".to_string());
    };
    let extension = extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    if extension.is_empty() {
        return Err("[voice-stt] invalid audio extension".to_string());
    }
    if !extension
        .chars()
        .all(|character| character.is_ascii_alphanumeric())
    {
        return Err("[voice-stt] invalid audio extension".to_string());
    }
    Ok(extension)
}

#[cfg(test)]
#[path = "transcription_tests.rs"]
mod tests;
