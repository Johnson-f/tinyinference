use axum::Router;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use serde_json::{Value, json};

use super::{
    BatchAnswer, BatchDecisionRequest, BatchGroup, BatchQuestion, ChoiceOption, DecisionContent,
    DecisionQuestion, DecisionRequest, DecisionResponse, GroundingConfig, LatencyMode,
    ReasoningMode, SageClient, ScaleLevel, StructuredContent, TagSpec,
};
use crate::Error;

fn yesno() -> DecisionQuestion {
    DecisionQuestion::YesNo {
        id: "review".into(),
        instructions: "Needs review?".into(),
    }
}

#[test]
fn request_shapes_match_sage_wire_contract() {
    let mut request = DecisionRequest::new("claim", yesno());
    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        json!({
            "content": "claim",
            "question": {"kind":"yesno", "id":"review", "instructions":"Needs review?"}
        })
    );
    request.content = DecisionContent::Structured(StructuredContent::Image {
        media: "data:image/png;base64,aGVsbG8=".into(),
        text: Some("screenshot".into()),
    });
    request.reasoning = ReasoningMode::On;
    request.latency_mode = LatencyMode::Fast;
    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["content"]["kind"], "image");
    assert_eq!(value["content"]["text"], "screenshot");
    assert_eq!(value["reasoning"], "on");
    assert_eq!(value["latency_mode"], "fast");

    let batch = BatchDecisionRequest {
        requests: vec![BatchGroup {
            content: "claim".into(),
            questions: vec![BatchQuestion {
                question: yesno(),
                grounding: Some(GroundingConfig::default()),
            }],
        }],
        reasoning: ReasoningMode::Auto,
        latency_mode: LatencyMode::Quality,
    };
    let value = serde_json::to_value(&batch).unwrap();
    assert_eq!(value["requests"][0]["questions"][0]["kind"], "yesno");
    assert_eq!(value["requests"][0]["questions"][0]["grounding"], json!({}));
    assert!(value["requests"][0].get("grounding").is_none());
    let decoded: BatchDecisionRequest = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, batch);
}

#[test]
fn typed_results_preserve_uncertainty_reasoning_and_usage() {
    let yes: DecisionResponse = serde_json::from_value(json!({
        "id":"review", "kind":"yesno", "result":{"answer":null,"probability":0.51},
        "meta":{"model":"levanto-sage-v1.1", "reasoning":{"fired":true,"ran":true,"finished":false,"tokens":42,"limited":"timeout"},
                "usage":{"billed_input_tokens":20,"image_count":1,"image_tokens":850}}
    })).unwrap();
    let DecisionResponse::YesNo { result, meta, .. } = yes else {
        panic!("wrong result kind")
    };
    assert_eq!(result.answer, None);
    assert_eq!(meta.reasoning.unwrap().limited.as_deref(), Some("timeout"));
    assert_eq!(meta.usage.unwrap().image_tokens, 850);

    let choice: DecisionResponse = serde_json::from_value(json!({
        "id":"route", "kind":"choice", "result":{"chosen":null,"probability":null,
        "probabilities":[{"option":"a","probability":0.72},{"option":"b","probability":0.70}]},
        "meta":{"model":"levanto-sage-v1.1", "compute_mode":"fanout_candidates"}
    }))
    .unwrap();
    let DecisionResponse::Choice { result, meta, .. } = choice else {
        panic!("wrong result kind")
    };
    assert_eq!(result.chosen, None);
    assert_eq!(result.probabilities.len(), 2);
    assert_eq!(meta.compute_mode.as_deref(), Some("fanout_candidates"));

    let tags: DecisionResponse = serde_json::from_value(json!({
        "id":"labels", "kind":"tags", "result":{"tags":[{"id":"spam","probability":0.5,"applies":null}]},
        "meta":{"model":"levanto-sage-v1.1"}
    })).unwrap();
    let DecisionResponse::Tags { result, .. } = tags else {
        panic!("wrong result kind")
    };
    assert_eq!(result.tags[0].applies, None);
}

#[test]
fn all_result_kinds_and_batch_partial_errors_decode() {
    let scale: DecisionResponse = serde_json::from_value(json!({
        "id":"score","kind":"scale","result":{"expectation":2.4,"confidence":0.71},
        "meta":{"model":"levanto-sage-v1.1"}
    }))
    .unwrap();
    assert!(matches!(scale, DecisionResponse::Scale { .. }));
    let sort: DecisionResponse = serde_json::from_value(json!({
        "id":"rank","kind":"sort","result":{"sorted":["b","a"],"confidence":null},
        "meta":{"model":"levanto-sage-v1.1"}
    }))
    .unwrap();
    assert!(matches!(sort, DecisionResponse::Sort { .. }));
    let batch: super::BatchDecisionResponse = serde_json::from_value(json!({
        "results":[{"answers":[
            {"ok":true,"result":{"id":"review","kind":"yesno","result":{"answer":"yes","probability":0.93},"meta":{"model":"levanto-sage-v1.1"}}},
            {"ok":false,"error":"bad question"}
        ]}],
        "meta":{"model":"levanto-sage-v1.1","request_count":1,"question_count":2}
    })).unwrap();
    assert!(matches!(
        batch.results[0].answers[0],
        BatchAnswer {
            ok: true,
            result: Some(_),
            ..
        }
    ));
    assert_eq!(
        batch.results[0].answers[1].error.as_deref(),
        Some("bad question")
    );
}

#[tokio::test]
async fn client_uses_auth_and_endpoints_and_classifies_errors() {
    async fn decide(
        headers: HeaderMap,
        axum::Json(body): axum::Json<Value>,
    ) -> (StatusCode, axum::Json<Value>) {
        if headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            != Some("Bearer test-key")
        {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(json!({"detail":"bad key"})),
            );
        }
        if body["question"]["id"] == "bad" {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({"detail":"invalid question"})),
            );
        }
        if body["question"]["id"] == "invalid_shape" {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                axum::Json(json!({"detail":"invalid shape"})),
            );
        }
        if body["question"]["id"] == "exhausted" {
            return (
                StatusCode::PAYMENT_REQUIRED,
                axum::Json(json!({"detail":"allowance exhausted"})),
            );
        }
        if body["question"]["id"] == "loading" {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(json!({"detail":"loading"})),
            );
        }
        (
            StatusCode::OK,
            axum::Json(json!({
                "id":"review", "kind":"yesno", "result":{"answer":"yes","probability":0.94},
                "meta":{"model":"levanto-sage-v1.1"}
            })),
        )
    }
    async fn batch() -> axum::Json<Value> {
        axum::Json(
            json!({"results":[{"answers":[{"ok":false,"error":"unavailable"}]}],
            "meta":{"model":"levanto-sage-v1.1","request_count":1,"question_count":1}}),
        )
    }
    async fn estimate() -> axum::Json<Value> {
        axum::Json(json!({"usage":{"min_input_tokens":12,"max_input_tokens":12,"exact":true}}))
    }
    async fn models() -> axum::Json<Value> {
        axum::Json(json!({"data":[{"id":"levanto-sage","name":"Levanto Sage","is_ready":true}]}))
    }
    let app = Router::new()
        .route("/ready", get(|| async { StatusCode::OK }))
        .route("/models", get(models))
        .route("/decide", post(decide))
        .route("/decide/batch", post(batch))
        .route("/usage/estimate", post(estimate));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = SageClient::with_base_url("test-key", &format!("http://{address}/")).unwrap();
    assert!(client.ready().await.unwrap());
    assert_eq!(client.models().await.unwrap()[0].id, "levanto-sage");
    assert!(matches!(
        client
            .decide(&DecisionRequest::new("claim", yesno()))
            .await
            .unwrap(),
        DecisionResponse::YesNo { .. }
    ));
    assert_eq!(
        client
            .estimate_decision(&DecisionRequest::new("claim", yesno()))
            .await
            .unwrap()
            .min_input_tokens,
        12
    );
    let bad = DecisionRequest::new(
        "claim",
        DecisionQuestion::YesNo {
            id: "bad".into(),
            instructions: "?".into(),
        },
    );
    assert!(matches!(
        client.decide(&bad).await.unwrap_err(),
        Error::Unprocessable
    ));
    let invalid_shape = DecisionRequest::new(
        "claim",
        DecisionQuestion::YesNo {
            id: "invalid_shape".into(),
            instructions: "?".into(),
        },
    );
    assert!(matches!(
        client.decide(&invalid_shape).await.unwrap_err(),
        Error::Unprocessable
    ));
    for (id, expected_status) in [("exhausted", 402), ("loading", 503)] {
        let request = DecisionRequest::new(
            "claim",
            DecisionQuestion::YesNo {
                id: id.into(),
                instructions: "?".into(),
            },
        );
        let error = client.decide(&request).await.unwrap_err();
        assert!(matches!(
            (expected_status, error),
            (402, Error::HttpStatus { status: 402 }) | (503, Error::Overloaded)
        ));
    }
    let wrong_key = SageClient::with_base_url("wrong-key", &format!("http://{address}/")).unwrap();
    assert!(matches!(
        wrong_key
            .decide(&DecisionRequest::new("claim", yesno()))
            .await,
        Err(Error::Authentication)
    ));
    let batch_request = BatchDecisionRequest {
        requests: vec![BatchGroup {
            content: "claim".into(),
            questions: vec![BatchQuestion {
                question: yesno(),
                grounding: None,
            }],
        }],
        reasoning: ReasoningMode::Auto,
        latency_mode: LatencyMode::Quality,
    };
    assert!(!client.decide_batch(&batch_request).await.unwrap().results[0].answers[0].ok);
    assert_eq!(
        client
            .estimate_batch(&batch_request)
            .await
            .unwrap()
            .max_input_tokens,
        12
    );
    server.abort();
}

#[tokio::test]
async fn invalid_combinations_are_rejected_before_network() {
    let client = SageClient::new("test-key").unwrap();
    let image = DecisionContent::Structured(StructuredContent::Image {
        media: "data:image/png;base64,AAAA".into(),
        text: None,
    });
    let mut request = DecisionRequest::new(image.clone(), yesno());
    request.grounding = Some(GroundingConfig::default());
    assert!(matches!(
        client.decide(&request).await,
        Err(Error::InvalidRequest { .. })
    ));

    let options = (0..21)
        .map(|index| ChoiceOption {
            option: index.to_string(),
            description: None,
        })
        .collect();
    request = DecisionRequest::new(
        image,
        DecisionQuestion::Choice {
            id: "pick".into(),
            instructions: "Pick".into(),
            options,
        },
    );
    assert!(matches!(
        client.decide(&request).await,
        Err(Error::InvalidRequest { .. })
    ));

    request = DecisionRequest::new(
        "text",
        DecisionQuestion::Scale {
            id: "score".into(),
            instructions: "Score".into(),
            levels: (0..4)
                .map(|level| ScaleLevel {
                    level,
                    description: None,
                })
                .collect(),
        },
    );
    assert!(matches!(
        client.decide(&request).await,
        Err(Error::InvalidRequest { .. })
    ));

    request = DecisionRequest::new(
        "text",
        DecisionQuestion::Tags {
            id: "tags".into(),
            instructions: None,
            tags: Vec::<TagSpec>::new(),
        },
    );
    assert!(matches!(
        client.decide(&request).await,
        Err(Error::InvalidRequest { .. })
    ));
}

#[test]
fn debug_does_not_expose_credential() {
    let client = SageClient::new("super-secret-key").unwrap();
    let debug = format!("{client:?}");
    assert!(!debug.contains("super-secret-key"));
    assert!(debug.contains("[REDACTED]"));
}
