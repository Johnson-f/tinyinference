use super::*;
use crate::model::FINISH_REASON_LENGTH;

fn call(arguments: &str) -> NativeToolCall {
    NativeToolCall {
        id: "call-1".into(),
        name: "now".into(),
        arguments: arguments.into(),
        extra_content: None,
    }
}

// Zero-argument calls arrive as `arguments: ""` from several providers.
#[test]
fn empty_or_blank_arguments_mean_an_empty_object() {
    for arguments in ["", "  \n"] {
        let response = ChatResponse {
            tool_calls: vec![call(arguments)],
            ..Default::default()
        };
        let converted = response_to_model_response(&response, &ModelRequest::default());
        let call = &converted.message.tool_calls[0];
        assert!(!call.is_invalid(), "{arguments:?} marked invalid");
        assert_eq!(call.arguments, serde_json::json!({}));
    }
}

// A length cut must not be reported as a clean stop.
#[test]
fn reported_finish_reason_is_kept() {
    let response = ChatResponse::<BilledUsage, NativeToolCall> {
        text: Some("cut off mid".into()),
        finish_reason: Some("max_tokens".into()),
        ..Default::default()
    };
    let converted = response_to_model_response(&response, &ModelRequest::default());
    assert_eq!(
        converted.finish_reason.as_deref(),
        Some(FINISH_REASON_LENGTH)
    );

    let unreported = ChatResponse {
        tool_calls: vec![call("{}")],
        ..Default::default()
    };
    let converted = response_to_model_response(&unreported, &ModelRequest::default());
    assert_eq!(converted.finish_reason.as_deref(), Some("tool_calls"));
}
