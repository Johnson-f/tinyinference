//! HTTP clients for third-party text-to-speech APIs: OpenAI-compatible and
//! ElevenLabs.
//!
//! The caller supplies the [`reqwest::Client`], so proxy, TLS and timeout
//! policy stay with the host. Errors are plain strings prefixed `[voice-tts]`.

#[cfg(feature = "schemars")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// API style for TTS requests.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum TtsApiStyle {
    /// OpenAI-compatible: POST JSON `{ model, voice, input }` to `/audio/speech`.
    #[default]
    OpenaiAudio,
    /// ElevenLabs: POST JSON `{ text, model_id }` to `/text-to-speech/<voice_id>`.
    ElevenLabs,
}

/// A configured third-party TTS endpoint.
#[derive(Debug, Clone)]
pub struct ExternalTtsClient {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    api_style: TtsApiStyle,
}

impl ExternalTtsClient {
    /// Builds a client for `endpoint` using the caller's HTTP client.
    pub fn new(
        client: reqwest::Client,
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        api_style: TtsApiStyle,
    ) -> Self {
        Self {
            client,
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            api_style,
        }
    }

    /// Synthesizes `text` with `voice`, returning the audio bytes and the
    /// response content type, dispatching on the configured API style.
    pub async fn synthesize(&self, text: &str, voice: &str) -> Result<(Vec<u8>, String), String> {
        match self.api_style {
            TtsApiStyle::OpenaiAudio => self.synthesize_openai_compat(text, voice).await,
            TtsApiStyle::ElevenLabs => self.synthesize_elevenlabs(text, voice).await,
        }
    }

    async fn synthesize_openai_compat(
        &self,
        text: &str,
        voice: &str,
    ) -> Result<(Vec<u8>, String), String> {
        let url = format!("{}/audio/speech", self.endpoint.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": "tts-1",
            "voice": voice,
            "input": text,
        });

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| format!("[voice-tts] external TTS request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("[voice-tts] external TTS error {status}: {body}"));
        }

        let content_type = content_type_or_mpeg(&resp);
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("[voice-tts] failed to read audio: {e}"))?;

        Ok((bytes.to_vec(), content_type))
    }

    async fn synthesize_elevenlabs(
        &self,
        text: &str,
        voice_id: &str,
    ) -> Result<(Vec<u8>, String), String> {
        let url = format!(
            "{}/text-to-speech/{}",
            self.endpoint.trim_end_matches('/'),
            voice_id
        );

        let body = serde_json::json!({
            "text": text,
            "model_id": "eleven_multilingual_v2",
        });

        let resp = self
            .client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| format!("[voice-tts] elevenlabs request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("[voice-tts] elevenlabs error {status}: {body}"));
        }

        let content_type = content_type_or_mpeg(&resp);
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("[voice-tts] failed to read elevenlabs audio: {e}"))?;

        Ok((bytes.to_vec(), content_type))
    }
}

fn content_type_or_mpeg(resp: &reqwest::Response) -> String {
    resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("audio/mpeg")
        .to_string()
}

#[cfg(test)]
#[path = "external_tts_tests.rs"]
mod tests;
