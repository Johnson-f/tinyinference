use super::*;
use axum::{
    Router,
    body::{Body, Bytes},
    http::{HeaderMap, Response, StatusCode},
    routing::post,
};
use std::sync::{Arc, Mutex};

type Seen = Arc<Mutex<(Vec<(String, String)>, String)>>;

async fn serve(
    path: &'static str,
    status: StatusCode,
    content_type: Option<&'static str>,
) -> (String, Seen) {
    let seen: Seen = Arc::new(Mutex::new((Vec::new(), String::new())));
    let captured = Arc::clone(&seen);
    let app = Router::new().route(
        path,
        post(move |headers: HeaderMap, body: Bytes| {
            let captured = Arc::clone(&captured);
            async move {
                let mut seen = captured.lock().unwrap();
                seen.0 = headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                seen.1 = String::from_utf8_lossy(&body).to_string();
                let mut builder = Response::builder().status(status);
                if let Some(ct) = content_type {
                    builder = builder.header("content-type", ct);
                }
                builder.body(Body::from("AUDIO")).unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), seen)
}

fn header(seen: &Seen, name: &str) -> Option<String> {
    seen.lock()
        .unwrap()
        .0
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
}

#[test]
fn api_style_serializes_lowercase() {
    assert_eq!(
        serde_json::to_string(&TtsApiStyle::OpenaiAudio).unwrap(),
        "\"openaiaudio\""
    );
    assert_eq!(
        serde_json::from_str::<TtsApiStyle>("\"elevenlabs\"").unwrap(),
        TtsApiStyle::ElevenLabs
    );
    assert_eq!(TtsApiStyle::default(), TtsApiStyle::OpenaiAudio);
}

#[tokio::test]
async fn openai_compat_posts_json_with_bearer_and_returns_content_type() {
    let (base, seen) = serve("/audio/speech", StatusCode::OK, Some("audio/wav")).await;
    let client = ExternalTtsClient::new(
        reqwest::Client::new(),
        format!("{base}/"),
        "sk-test",
        TtsApiStyle::OpenaiAudio,
    );
    let (bytes, mime) = client.synthesize("hi there", "alloy").await.unwrap();
    assert_eq!(bytes, b"AUDIO");
    assert_eq!(mime, "audio/wav");
    assert_eq!(
        header(&seen, "authorization").as_deref(),
        Some("Bearer sk-test")
    );
    let body: serde_json::Value = serde_json::from_str(&seen.lock().unwrap().1).unwrap();
    assert_eq!(
        body,
        serde_json::json!({"model": "tts-1", "voice": "alloy", "input": "hi there"})
    );
}

#[tokio::test]
async fn elevenlabs_posts_to_voice_path_and_defaults_content_type() {
    let (base, seen) = serve("/text-to-speech/voice123", StatusCode::OK, None).await;
    let client = ExternalTtsClient::new(
        reqwest::Client::new(),
        base,
        "xi-key",
        TtsApiStyle::ElevenLabs,
    );
    let (bytes, mime) = client.synthesize("hello", "voice123").await.unwrap();
    assert_eq!(bytes, b"AUDIO");
    assert_eq!(mime, "audio/mpeg");
    assert_eq!(header(&seen, "xi-api-key").as_deref(), Some("xi-key"));
    let body: serde_json::Value = serde_json::from_str(&seen.lock().unwrap().1).unwrap();
    assert_eq!(
        body,
        serde_json::json!({"text": "hello", "model_id": "eleven_multilingual_v2"})
    );
}

#[tokio::test]
async fn non_success_status_reports_provider_error_text() {
    let (base, _) = serve("/audio/speech", StatusCode::TOO_MANY_REQUESTS, None).await;
    let client =
        ExternalTtsClient::new(reqwest::Client::new(), base, "k", TtsApiStyle::OpenaiAudio);
    let err = client.synthesize("x", "v").await.unwrap_err();
    assert_eq!(
        err,
        "[voice-tts] external TTS error 429 Too Many Requests: AUDIO"
    );
}
