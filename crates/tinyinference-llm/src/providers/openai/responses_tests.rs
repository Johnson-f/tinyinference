use super::*;
use crate::message::Message;
use serde_json::json;

#[test]
fn build_input_folds_system_into_instructions_and_keys_roles() {
    let messages = vec![
        Message::system("be terse"),
        Message::system("and correct"),
        Message::user("hi"),
        Message::assistant("hello"),
        Message::user("  "), // empty → skipped
    ];
    let (instructions, input) = build_responses_input(&messages);
    assert_eq!(instructions.as_deref(), Some("be terse\n\nand correct"));
    assert_eq!(input.len(), 2);
    assert_eq!(input[0].role, "user");
    assert_eq!(input[0].content[0].kind, "input_text");
    assert_eq!(input[0].content[0].text, "hi");
    // Assistant items must use `output_text`, not `input_text`.
    assert_eq!(input[1].role, "assistant");
    assert_eq!(input[1].content[0].kind, "output_text");
    assert_eq!(input[1].content[0].text, "hello");
}

#[test]
fn build_input_skips_custom_messages() {
    let messages = vec![
        Message::user("hi"),
        Message::Custom(crate::message::CustomMessage {
            kind: "compaction".into(),
            payload: json!({"summary": "..."}),
            display: Some("Compacted".into()),
        }),
        Message::assistant("hello"),
    ];
    let (_, input) = build_responses_input(&messages);
    assert_eq!(input.len(), 2);
    assert_eq!(input[0].content[0].text, "hi");
    assert_eq!(input[1].content[0].text, "hello");
}

#[test]
fn extract_text_prefers_output_text_then_scans_content() {
    let with_convenience = ResponsesResponse {
        status: None,
        incomplete_details: None,
        output: Vec::new(),
        output_text: Some("  final  ".to_string()),
        usage: None,
    };
    assert_eq!(
        extract_responses_text(&with_convenience).as_deref(),
        Some("final")
    );

    let via_content = ResponsesResponse {
        status: None,
        incomplete_details: None,
        output: vec![ResponsesOutput {
            content: vec![
                ResponsesContent {
                    kind: Some("reasoning".into()),
                    text: Some("...".into()),
                },
                ResponsesContent {
                    kind: Some("output_text".into()),
                    text: Some("answer".into()),
                },
            ],
            ..ResponsesOutput::default()
        }],
        output_text: None,
        usage: None,
    };
    assert_eq!(
        extract_responses_text(&via_content).as_deref(),
        Some("answer")
    );

    let empty = ResponsesResponse {
        status: None,
        incomplete_details: None,
        output: Vec::new(),
        output_text: None,
        usage: None,
    };
    assert_eq!(extract_responses_text(&empty), None);
}

#[test]
fn parse_maps_text_and_usage_onto_model_response() {
    let body = json!({
        "output_text": "the answer",
        "usage": { "input_tokens": 12, "output_tokens": 5 }
    });
    let resp = parse_responses_response(body);
    assert_eq!(resp.text(), "the answer");
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    let usage = resp.usage.expect("usage mapped");
    assert_eq!(usage.input_tokens, 12);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 17);
}

#[test]
fn parse_tolerates_a_body_without_output() {
    let resp = parse_responses_response(json!({ "id": "resp_1" }));
    assert_eq!(resp.text(), "");
}

#[test]
fn parse_preserves_incomplete_reason() {
    let resp = parse_responses_response(json!({
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "output_text": "partial"
    }));
    assert_eq!(resp.text(), "partial");
    assert_eq!(resp.finish_reason.as_deref(), Some("max_output_tokens"));
}

#[test]
fn parse_responses_wire_reads_sse_completed_event() {
    let sse = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output_text\":\"hi\"}}\n\n";
    let value = parse_responses_wire(sse).expect("sse");
    assert_eq!(value["output_text"], "hi");
}

#[test]
fn parse_responses_wire_uses_output_text_done_when_completed_output_is_empty() {
    let sse = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\nevent: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"text\":\"ok\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n";
    let value = parse_responses_wire(sse).expect("sse");
    assert_eq!(value["output_text"], "ok");
}

#[tokio::test]
async fn responses_invoke_avoids_codex_stream_required_400() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use crate::model::{ChatModel, ModelRequest};
    use crate::providers::openai::OpenAiModel;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = sock.read(&mut tmp).unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&buf[..split]).unwrap_or("");
                let content_len = headers
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(k, v)| {
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .unwrap_or(0);
                let start = split + 4;
                while buf.len() < start + content_len {
                    let n = sock.read(&mut tmp).unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                break;
            }
        }
        let req = String::from_utf8_lossy(&buf);
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        let stream_true = body.contains("\"stream\":true") || body.contains("\"stream\": true");
        let (status, content_type, payload) = if stream_true {
            let sse = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output_text\":\"ok\"}}\n\n";
            ("200 OK", "text/event-stream", sse.to_string())
        } else {
            (
                "400 Bad Request",
                "application/json",
                r#"{"detail":"Stream must be set to true"}"#.to_string(),
            )
        };
        let resp = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        sock.write_all(resp.as_bytes()).unwrap();
    });

    let model = OpenAiModel::compatible("k", format!("http://{addr}/v1"), "gpt-5.6-luna")
        .with_responses_api_primary();
    let response = model
        .invoke(&(), ModelRequest::new(vec![Message::user("hi")]))
        .await
        .expect("codex-shaped mock must succeed when stream is true");
    assert_eq!(response.text(), "ok");
    server.join().unwrap();
}
