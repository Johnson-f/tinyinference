//! HTTP clients for third-party speech-to-text APIs: OpenAI-compatible,
//! Deepgram, and ElevenLabs Scribe.
//!
//! The caller supplies the [`reqwest::Client`], so proxy, TLS and timeout
//! policy stay with the host. Errors are plain strings prefixed `[voice-stt]`.

#[cfg(feature = "schemars")]
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use crate::mime::extension_for_mime;

/// API style for STT requests. Different providers use incompatible request
/// shapes; the factory dispatches based on this discriminator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum SttApiStyle {
    /// OpenAI-compatible: multipart POST to `/audio/transcriptions`.
    #[default]
    OpenaiAudio,
    /// Deepgram: POST binary audio to `/listen?model=<model>`.
    Deepgram,
    /// ElevenLabs Scribe: multipart POST to `/speech-to-text` with `model_id`
    /// and an `xi-api-key` header.
    ElevenLabs,
}

/// A configured third-party STT endpoint.
#[derive(Debug, Clone)]
pub struct ExternalSttClient {
    client: reqwest::Client,
    model: String,
    endpoint: String,
    api_key: String,
    api_style: SttApiStyle,
}

impl ExternalSttClient {
    /// Builds a client for `endpoint` using the caller's HTTP client.
    pub fn new(
        client: reqwest::Client,
        model: impl Into<String>,
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        api_style: SttApiStyle,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            api_style,
        }
    }

    /// The configured model id.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Transcribes `audio_bytes` of the given `mime` type, dispatching on the
    /// configured API style.
    pub async fn transcribe(
        &self,
        audio_bytes: &[u8],
        mime: &str,
        file_name: Option<&str>,
        language: Option<&str>,
    ) -> Result<String, String> {
        match self.api_style {
            SttApiStyle::OpenaiAudio => {
                self.transcribe_openai_compat(audio_bytes, mime, file_name, language)
                    .await
            }
            SttApiStyle::Deepgram => self.transcribe_deepgram(audio_bytes, mime, language).await,
            SttApiStyle::ElevenLabs => {
                self.transcribe_elevenlabs(audio_bytes, mime, file_name, language)
                    .await
            }
        }
    }

    async fn transcribe_openai_compat(
        &self,
        audio_bytes: &[u8],
        mime: &str,
        file_name: Option<&str>,
        language: Option<&str>,
    ) -> Result<String, String> {
        let url = format!(
            "{}/audio/transcriptions",
            self.endpoint.trim_end_matches('/')
        );
        let ext = extension_for_mime(mime);
        let default_fname = format!("audio.{ext}");
        let fname = file_name.unwrap_or(&default_fname);

        let file_part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
            .file_name(fname.to_string())
            .mime_str(mime)
            .map_err(|e| format!("[voice-stt] mime error: {e}"))?;

        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", file_part);

        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("[voice-stt] external STT request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("[voice-stt] external STT error {status}: {body}"));
        }

        #[derive(Deserialize)]
        struct TranscriptionResp {
            text: String,
        }
        let parsed: TranscriptionResp = resp
            .json()
            .await
            .map_err(|e| format!("[voice-stt] failed to parse response: {e}"))?;
        Ok(parsed.text)
    }

    async fn transcribe_deepgram(
        &self,
        audio_bytes: &[u8],
        mime: &str,
        language: Option<&str>,
    ) -> Result<String, String> {
        let mut url = format!(
            "{}/listen?model={}",
            self.endpoint.trim_end_matches('/'),
            self.model
        );
        if let Some(lang) = language {
            url.push_str(&format!("&language={lang}"));
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", mime)
            .body(audio_bytes.to_vec())
            .send()
            .await
            .map_err(|e| format!("[voice-stt] deepgram request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("[voice-stt] deepgram error {status}: {body}"));
        }

        #[derive(Deserialize)]
        struct DeepgramChannel {
            alternatives: Vec<DeepgramAlt>,
        }
        #[derive(Deserialize)]
        struct DeepgramAlt {
            transcript: String,
        }
        #[derive(Deserialize)]
        struct DeepgramResult {
            channels: Vec<DeepgramChannel>,
        }
        #[derive(Deserialize)]
        struct DeepgramResp {
            results: DeepgramResult,
        }

        let parsed: DeepgramResp = resp
            .json()
            .await
            .map_err(|e| format!("[voice-stt] deepgram parse error: {e}"))?;

        let text = parsed
            .results
            .channels
            .first()
            .and_then(|ch| ch.alternatives.first())
            .map(|a| a.transcript.clone())
            .unwrap_or_default();
        Ok(text)
    }

    async fn transcribe_elevenlabs(
        &self,
        audio_bytes: &[u8],
        mime: &str,
        file_name: Option<&str>,
        language: Option<&str>,
    ) -> Result<String, String> {
        let url = format!("{}/speech-to-text", self.endpoint.trim_end_matches('/'));
        let ext = extension_for_mime(mime);
        let default_fname = format!("audio.{ext}");
        let fname = file_name.unwrap_or(&default_fname);

        let file_part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
            .file_name(fname.to_string())
            .mime_str(mime)
            .map_err(|e| format!("[voice-stt] mime error: {e}"))?;
        let mut form = reqwest::multipart::Form::new()
            .text("model_id", self.model.clone())
            .part("file", file_part);

        if let Some(lang) = language {
            form = form.text("language_code", lang.to_string());
        }

        let resp = self
            .client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("[voice-stt] elevenlabs request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("[voice-stt] elevenlabs error {status}: {body}"));
        }

        #[derive(Deserialize)]
        struct TranscriptionResp {
            text: String,
        }
        let parsed: TranscriptionResp = resp
            .json()
            .await
            .map_err(|e| format!("[voice-stt] elevenlabs parse error: {e}"))?;
        Ok(parsed.text)
    }
}

#[cfg(test)]
#[path = "external_stt_test.rs"]
mod tests;
