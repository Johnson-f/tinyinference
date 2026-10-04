use super::*;
use crate::message::{ImageRef, MediaRef, UserMessage};
use crate::providers::{anthropic::AnthropicModel, openai::OpenAiModel};

async fn assert_rejected(model: &dyn ChatModel<()>, request: ModelRequest, expected: &str) {
    match model.invoke(&(), request.clone()).await {
        Err(Error::Validation(message)) => assert!(message.contains(expected), "{message}"),
        other => panic!("expected validation before I/O, got {other:?}"),
    }
    match model.stream(&(), request).await {
        Err(Error::Validation(message)) => assert!(message.contains(expected), "{message}"),
        _ => panic!("expected validation before streaming I/O"),
    }
}

#[tokio::test]
async fn invalid_inline_media_is_rejected_before_public_transport_io() {
    let chat = OpenAiModel::compatible("key", "http://127.0.0.1:1/v1", "gpt-4.1");
    let responses = OpenAiModel::compatible("key", "http://127.0.0.1:1/v1", "gpt-4.1")
        .with_responses_api_primary()
        .with_responses_document_input(true);
    let anthropic =
        AnthropicModel::with_base_url("key", "http://127.0.0.1:1").with_insecure_http(true);
    for data in ["", "%%%", "Q", "QR=="] {
        let image = ContentBlock::Image(ImageRef {
            url: format!("data:image/png;base64,{data}"),
            mime_type: None,
        });
        let audio = ContentBlock::Audio(MediaRef::base64(data, "audio/wav"));
        let pdf = ContentBlock::Document(MediaRef::base64(data, "application/pdf"));
        for (model, block) in [
            (&chat as &dyn ChatModel<()>, image.clone()),
            (&chat, audio),
            (&responses, image.clone()),
            (&responses, pdf.clone()),
            (&anthropic, image),
            (&anthropic, pdf),
        ] {
            assert_rejected(
                model,
                ModelRequest::new(vec![Message::User(UserMessage {
                    content: vec![block],
                })]),
                "Base64",
            )
            .await;
        }
    }
}

#[tokio::test]
async fn assistant_media_is_rejected_before_public_chat_transport_io() {
    let model = OpenAiModel::compatible("key", "http://127.0.0.1:1/v1", "gpt-4.1");
    for block in [
        ContentBlock::Image(ImageRef {
            url: "data:image/png;base64,QQ==".into(),
            mime_type: None,
        }),
        ContentBlock::Audio(MediaRef::base64("QQ==", "audio/wav")),
        ContentBlock::Video(MediaRef::base64("QQ==", "video/mp4")),
        ContentBlock::Document(MediaRef::base64("QQ==", "application/pdf")),
    ] {
        let mut message = Message::assistant("visible text");
        if let Message::Assistant(assistant) = &mut message {
            assistant.content.push(block);
        }
        assert_rejected(
            &model,
            ModelRequest::new(vec![message]),
            "assistant messages",
        )
        .await;
    }
}

/// Captures complete outgoing JSON on loopback and returns synthetic responses.
fn wire_fixture(
    count: usize,
    reply: &'static str,
) -> (String, std::thread::JoinHandle<Vec<Value>>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let mut captured = Vec::new();
        for _ in 0..count {
            let (mut socket, _) = listener.accept().unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0; 4096];
            let (start, length) = loop {
                let read = socket.read(&mut buffer).unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&buffer[..read]);
                if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&bytes[..split]).unwrap();
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap();
                    break (split + 4, length);
                }
            };
            while bytes.len() < start + length {
                let read = socket.read(&mut buffer).unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body: Value = serde_json::from_slice(&bytes[start..start + length]).unwrap();
            let (content_type, payload) = if body["stream"] == true && body.get("input").is_some() {
                (
                    "text/event-stream",
                    "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output_text\":\"ok\"}}\n\n",
                )
            } else if body["stream"] == true {
                (
                    "text/event-stream",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                )
            } else {
                ("application/json", reply)
            };
            captured.push(body);
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}", payload.len()).unwrap();
        }
        captured
    });
    (endpoint, server)
}

const CHAT_REPLY: &str =
    r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;

#[tokio::test]
async fn valid_inline_media_reaches_public_provider_wire() {
    let image = ContentBlock::Image(ImageRef {
        url: "data:image/png;base64,QQ==".into(),
        mime_type: None,
    });
    let audio = ContentBlock::Audio(MediaRef::base64("QQ==", "audio/wav"));
    let pdf = ContentBlock::Document(MediaRef::base64("QQ==", "application/pdf"));
    let request = |blocks| ModelRequest::new(vec![Message::User(UserMessage { content: blocks })]);

    let (endpoint, server) = wire_fixture(1, CHAT_REPLY);
    let model = OpenAiModel::compatible("key", endpoint, "gpt-4.1");
    assert_eq!(
        model
            .invoke(&(), request(vec![image.clone(), audio]))
            .await
            .unwrap()
            .text(),
        "ok"
    );
    let bodies = server.join().unwrap();
    assert_eq!(
        bodies[0]["messages"][0]["content"][0]["image_url"]["url"],
        "data:image/png;base64,QQ=="
    );
    assert_eq!(
        bodies[0]["messages"][0]["content"][1]["input_audio"]["data"],
        "QQ=="
    );

    let (endpoint, server) = wire_fixture(1, r#"{"output_text":"ok"}"#);
    let model = OpenAiModel::compatible("key", endpoint, "gpt-4.1")
        .with_responses_api_primary()
        .with_responses_document_input(true);
    assert_eq!(
        model
            .invoke(&(), request(vec![image.clone(), pdf.clone()]))
            .await
            .unwrap()
            .text(),
        "ok"
    );
    let bodies = server.join().unwrap();
    assert_eq!(
        bodies[0]["input"][0]["content"][0]["image_url"],
        "data:image/png;base64,QQ=="
    );
    assert_eq!(
        bodies[0]["input"][0]["content"][1]["file_data"],
        "data:application/pdf;base64,QQ=="
    );

    let (endpoint, server) = wire_fixture(
        1,
        r#"{"id":"msg","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1},"stop_reason":"end_turn"}"#,
    );
    let model = AnthropicModel::with_base_url("key", endpoint).with_insecure_http(true);
    assert_eq!(
        model
            .invoke(&(), request(vec![image, pdf]))
            .await
            .unwrap()
            .text(),
        "ok"
    );
    let bodies = server.join().unwrap();
    assert_eq!(
        bodies[0]["messages"][0]["content"][0]["source"]["data"],
        "QQ=="
    );
    assert_eq!(
        bodies[0]["messages"][0]["content"][1]["source"]["data"],
        "QQ=="
    );
}

#[tokio::test]
async fn assistant_text_thinking_and_tools_keep_public_chat_wire_semantics() {
    use futures::StreamExt;
    let (endpoint, server) = wire_fixture(2, CHAT_REPLY);
    let model = OpenAiModel::compatible("key", endpoint, "gpt-4.1");
    let mut text = Message::assistant("visible text");
    let mut tools = Message::assistant("");
    if let Message::Assistant(assistant) = &mut text {
        assistant
            .content
            .push(ContentBlock::thinking("private reasoning"));
        assistant.content.push(ContentBlock::RedactedThinking {
            data: "opaque".into(),
        });
    }
    if let Message::Assistant(assistant) = &mut tools {
        assistant.tool_calls.push(ToolCall::new(
            "call-1",
            "lookup",
            serde_json::json!({"key":"value"}),
        ));
    }
    let request = ModelRequest::new(vec![text, tools]);
    assert_eq!(
        model.invoke(&(), request.clone()).await.unwrap().text(),
        "ok"
    );
    let items: Vec<_> = model.stream(&(), request).await.unwrap().collect().await;
    assert!(
        items
            .iter()
            .any(|item| matches!(item, ModelStreamItem::Completed(_)))
    );
    for body in server.join().unwrap() {
        assert_eq!(body["messages"][0]["content"], "visible text");
        assert!(body["messages"][1]["content"].is_null());
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call-1");
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            r#"{"key":"value"}"#
        );
    }
}
