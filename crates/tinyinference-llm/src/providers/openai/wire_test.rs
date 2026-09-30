//! Request-body and header tests for the OpenAI-compatible provider.
//!
//! Ported from OpenHuman's `tests/inference_provider_e2e.rs`, which drove
//! `OpenAiModel` against a wiremock server and no OpenHuman code. The
//! translation tests assert on the JSON body directly; the wire tests use a
//! one-shot local socket to check what is actually sent over HTTP.

use std::io::{Read, Write};
use std::net::TcpListener;

use serde_json::{Value, json};

use super::*;
use crate::message::Message;
use crate::model::{ChatModel, ModelRequest};

fn body_for(model: &OpenAiModel, request: &ModelRequest) -> Value {
    serde_json::to_value(model.translate_request(request).unwrap()).unwrap()
}

fn request(model: &str, temperature: f64) -> ModelRequest {
    ModelRequest::new(vec![Message::user("hi")])
        .with_model(model)
        .with_temperature(temperature)
}

fn unsupported_patterns() -> Vec<String> {
    ["o1*", "o3*", "o4*", "gpt-5*"]
        .into_iter()
        .map(String::from)
        .collect()
}

#[test]
fn temperature_is_sent_for_models_that_support_it() {
    let model = OpenAiModel::new("k").with_temperature_unsupported_models(unsupported_patterns());
    let body = body_for(&model, &request("gpt-4o-mini", 0.7));
    assert_eq!(body["temperature"].as_f64(), Some(0.7), "body={body}");
}

#[test]
fn temperature_is_omitted_for_unsupported_model_families() {
    let model = OpenAiModel::new("k").with_temperature_unsupported_models(unsupported_patterns());
    for name in [
        "o1-preview",
        "o3-mini",
        "o4-preview",
        "gpt-5",
        "gpt-5-turbo",
    ] {
        let body = body_for(&model, &request(name, 0.7));
        assert!(
            body.get("temperature").is_none(),
            "temperature must be absent for model={name}; body={body}"
        );
        assert_eq!(body["model"], json!(name));
    }
}

#[test]
fn temperature_is_kept_when_no_unsupported_patterns_are_configured() {
    let model = OpenAiModel::new("k");
    let body = body_for(&model, &request("o1-preview", 0.4));
    assert_eq!(body["temperature"].as_f64(), Some(0.4), "body={body}");
}

#[test]
fn request_model_overrides_the_configured_default_model() {
    let model = OpenAiModel::new("k").with_model("gpt-4.1-mini");
    let body = body_for(&model, &request("claude-3-sonnet", 0.5));
    assert_eq!(body["model"], json!("claude-3-sonnet"));

    let no_override = ModelRequest::new(vec![Message::user("hi")]);
    assert_eq!(
        body_for(&model, &no_override)["model"],
        json!("gpt-4.1-mini")
    );
}

/// Serves one canned chat completion and returns the raw request it received.
fn serve_once() -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = sock.read(&mut tmp).unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&buf[..split]).to_ascii_lowercase();
            let len = head
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")?
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
                .unwrap_or(0);
            if buf.len() >= split + 4 + len {
                break;
            }
        }
        let payload = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15 }
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
            payload.len()
        );
        sock.write_all(response.as_bytes()).unwrap();
        String::from_utf8_lossy(&buf).into_owned()
    });
    (base, handle)
}

fn header_value(raw: &str, name: &str) -> Option<String> {
    raw.split("\r\n\r\n").next()?.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

#[tokio::test]
async fn bearer_auth_is_sent_as_authorization_header_and_body_carries_the_model() {
    let (base, server) = serve_once();
    let model = OpenAiModel::new("secret-key")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::Bearer);

    let response = model.invoke(&(), request("gpt-4o", 0.7)).await.unwrap();
    assert_eq!(response.text(), "Hello!");

    let raw = server.join().unwrap();
    assert!(raw.starts_with("POST /v1/chat/completions "), "{raw}");
    assert_eq!(
        header_value(&raw, "authorization").as_deref(),
        Some("Bearer secret-key")
    );
    let body: Value = serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["model"], json!("gpt-4o"));
}

#[tokio::test]
async fn anthropic_auth_sends_key_and_version_but_no_authorization_header() {
    let (base, server) = serve_once();
    let model = OpenAiModel::new("sk-ant-test")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::Anthropic);

    model
        .invoke(&(), request("claude-3-haiku", 0.5))
        .await
        .unwrap();

    let raw = server.join().unwrap();
    assert_eq!(
        header_value(&raw, "x-api-key").as_deref(),
        Some("sk-ant-test")
    );
    assert_eq!(
        header_value(&raw, "anthropic-version").as_deref(),
        Some("2023-06-01")
    );
    assert_eq!(header_value(&raw, "authorization"), None, "{raw}");
}

#[tokio::test]
async fn no_auth_style_sends_no_credentials() {
    let (base, server) = serve_once();
    let model = OpenAiModel::new("")
        .with_provider("ollama")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::None);

    let response = model.invoke(&(), request("llama3", 0.7)).await.unwrap();
    assert_eq!(response.text(), "Hello!");

    let raw = server.join().unwrap();
    assert_eq!(header_value(&raw, "authorization"), None, "{raw}");
    assert_eq!(header_value(&raw, "x-api-key"), None, "{raw}");
}
