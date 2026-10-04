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
