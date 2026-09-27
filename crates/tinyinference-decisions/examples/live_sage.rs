//! Live Sage smoke example using synthetic content.
//!
//! Run with `SAGE_API_KEY` set. Each decision call uses the account's allowance.

use tinyinference_decisions::sage::{
    BatchDecisionRequest, BatchGroup, BatchQuestion, ChoiceOption, DecisionContent,
    DecisionQuestion, DecisionRequest, DecisionResponse, ReasoningMode, SageClient,
    StructuredContent,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("SAGE_API_KEY")?;
    let sage = SageClient::new(key)?;

    println!("ready: {}", sage.ready().await?);
    println!("models: {}", sage.models().await?.len());

    let request = DecisionRequest {
        reasoning: ReasoningMode::On,
        ..DecisionRequest::new(
            "A customer requests a cash refund 40 days after purchase. The policy allows cash refunds within 30 days.",
            DecisionQuestion::YesNo {
                id: "cash_refund".into(),
                instructions: "Under the policy, should we issue a cash refund?".into(),
            },
        )
    };
    let estimate = sage.estimate_decision(&request).await?;
    println!("estimate exact: {}", estimate.exact);
    if let DecisionResponse::YesNo { result, meta, .. } = sage.decide(&request).await? {
        println!("refund verdict: {:?}", result.answer);
        println!(
            "reasoning ran: {:?}",
            meta.reasoning.as_ref().map(|reasoning| reasoning.ran)
        );
    }

    let image = DecisionRequest::new(
        DecisionContent::Structured(StructuredContent::Image {
            media: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/J5sAAAAASUVORK5CYII=".into(),
            text: None,
        }),
        DecisionQuestion::YesNo {
            id: "person_visible".into(),
            instructions: "Does the image show a person?".into(),
        },
    );
    if let DecisionResponse::YesNo { result, meta, .. } = sage.decide(&image).await? {
        println!("image verdict: {:?}", result.answer);
        println!(
            "images billed: {:?}",
            meta.usage.as_ref().map(|usage| usage.image_count)
        );
    }

    let batch = BatchDecisionRequest {
        requests: vec![BatchGroup {
            content: "Customer ticket: the invoice has a duplicate charge.".into(),
            questions: vec![
                BatchQuestion {
                    question: DecisionQuestion::YesNo {
                        id: "billing_issue".into(),
                        instructions: "Is this a billing issue?".into(),
                    },
                    grounding: None,
                },
                BatchQuestion {
                    question: DecisionQuestion::Choice {
                        id: "route".into(),
                        instructions: "Which team should handle this?".into(),
                        options: ["billing", "technical"]
                            .into_iter()
                            .map(|option| ChoiceOption {
                                option: option.into(),
                                description: None,
                            })
                            .collect(),
                    },
                    grounding: None,
                },
            ],
        }],
        reasoning: ReasoningMode::Auto,
        latency_mode: Default::default(),
    };
    let result = sage.decide_batch(&batch).await?;
    println!("batch questions: {}", result.meta.question_count);
    println!("batch answers: {}", result.results[0].answers.len());
    Ok(())
}
