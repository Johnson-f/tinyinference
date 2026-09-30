use super::*;
use axum::{
    Router,
    body::Bytes,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Seen {
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

async fn serve(
    path: &'static str,
    status: StatusCode,
    reply: &'static str,
) -> (String, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let captured = Arc::clone(&seen);
    let app = Router::new().route(
        path,
        post(move |headers: HeaderMap, body: Bytes| {
            let captured = Arc::clone(&captured);
            async move {
                let mut seen = captured.lock().unwrap();
                seen.path = path.to_string();
                seen.headers = headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                seen.body = String::from_utf8_lossy(&body).to_string();
                (status, reply)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), seen)
}

fn header(seen: &Seen, name: &str) -> Option<String> {
    seen.headers
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
}

#[test]
fn api_style_serializes_lowercase() {
    assert_eq!(
        serde_json::to_string(&SttApiStyle::OpenaiAudio).unwrap(),
        "\"openaiaudio\""
    );
    assert_eq!(
        serde_json::to_string(&SttApiStyle::Deepgram).unwrap(),
        "\"deepgram\""
    );
    assert_eq!(
        serde_json::from_str::<SttApiStyle>("\"elevenlabs\"").unwrap(),
        SttApiStyle::ElevenLabs
    );
    assert_eq!(SttApiStyle::default(), SttApiStyle::OpenaiAudio);
}

#[tokio::test]
async fn elevenlabs_uses_scribe_endpoint_request_shape_and_authentication() {
    let (base, seen) = serve(
        "/speech-to-text",
        StatusCode::OK,
        r#"{"text":"transcribed by scribe"}"#,
    )
    .await;
    let client = ExternalSttClient::new(
        reqwest::Client::new(),
        "scribe_v1",
        base,
        "test-elevenlabs-key",
        SttApiStyle::ElevenLabs,
    );
    let text = client
        .transcribe(&[1, 2, 3], "audio/wav", Some("clip.wav"), Some("en"))
        .await
        .unwrap();
    assert_eq!(text, "transcribed by scribe");
    let seen = seen.lock().unwrap();
    assert_eq!(
        header(&seen, "xi-api-key").as_deref(),
        Some("test-elevenlabs-key")
    );
    assert!(seen.body.contains("name=\"model_id\""));
    assert!(seen.body.contains("scribe_v1"));
    assert!(seen.body.contains("name=\"language_code\""));
    assert!(seen.body.contains("filename=\"clip.wav\""));
}

#[tokio::test]
async fn openai_compat_posts_multipart_with_bearer_and_default_filename() {
    let (base, seen) = serve(
        "/audio/transcriptions",
        StatusCode::OK,
        r#"{"text":"hello"}"#,
    )
    .await;
    let client = ExternalSttClient::new(
        reqwest::Client::new(),
        "whisper-1",
        format!("{base}/"),
        "sk-test",
        SttApiStyle::OpenaiAudio,
    );
    assert_eq!(client.model(), "whisper-1");
    let text = client
        .transcribe(&[1], "audio/mpeg", None, None)
        .await
        .unwrap();
    assert_eq!(text, "hello");
    let seen = seen.lock().unwrap();
    assert_eq!(
        header(&seen, "authorization").as_deref(),
        Some("Bearer sk-test")
    );
    assert!(seen.body.contains("filename=\"audio.mp3\""));
    assert!(!seen.body.contains("name=\"language\""));
}

#[tokio::test]
async fn deepgram_posts_binary_with_token_auth_and_reads_first_transcript() {
    let (base, seen) = serve(
        "/listen",
        StatusCode::OK,
        r#"{"results":{"channels":[{"alternatives":[{"transcript":"dg text"}]}]}}"#,
    )
    .await;
    let client = ExternalSttClient::new(
        reqwest::Client::new(),
        "nova-2",
        base,
        "dg-key",
        SttApiStyle::Deepgram,
    );
    let text = client
        .transcribe(&[9, 9], "audio/wav", None, Some("en"))
        .await
        .unwrap();
    assert_eq!(text, "dg text");
    let seen = seen.lock().unwrap();
    assert_eq!(
        header(&seen, "authorization").as_deref(),
        Some("Token dg-key")
    );
    assert_eq!(header(&seen, "content-type").as_deref(), Some("audio/wav"));
}

#[tokio::test]
async fn non_success_status_reports_provider_error_text() {
    let (base, _) = serve("/audio/transcriptions", StatusCode::UNAUTHORIZED, "nope").await;
    let client = ExternalSttClient::new(
        reqwest::Client::new(),
        "m",
        base,
        "k",
        SttApiStyle::OpenaiAudio,
    );
    let err = client
        .transcribe(&[1], "audio/wav", None, None)
        .await
        .unwrap_err();
    assert_eq!(err, "[voice-stt] external STT error 401 Unauthorized: nope");
}
