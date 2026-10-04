use super::*;

use crate::model::ModelResponse;
use serde_json::json;

#[test]
fn old_responses_read_without_inventing_execution_or_rich_output() {
    let response: ModelResponse = serde_json::from_value(json!({
        "message":{"content":[{"text":"hello"}],"tool_calls":[]}
    }))
    .unwrap();
    assert!(response.output.is_empty());
    assert!(response.execution.is_none());
    assert_eq!(response.text(), "hello");
    let encoded = serde_json::to_value(response).unwrap();
    assert!(encoded.get("execution").is_none());
    assert!(encoded.get("output").is_none());
}

#[test]
fn future_statuses_survive_serialization_without_becoming_success() {
    let status: ExecutionStatus = serde_json::from_value(json!("waiting_for_review")).unwrap();
    assert!(!status.is_terminal());
    assert_eq!(
        serde_json::to_value(status).unwrap(),
        json!("waiting_for_review")
    );
}

#[test]
fn deferred_cancellation_serializes_with_or_without_a_snapshot() {
    use crate::model::DeferredStatus;
    for response in [None, Some(Box::new(ModelResponse::assistant("partial")))] {
        let status = DeferredStatus::Cancelled { response };
        let value = serde_json::to_value(status).unwrap();
        assert_eq!(value["status"], "cancelled");
        assert!(matches!(
            serde_json::from_value::<DeferredStatus>(value).unwrap(),
            DeferredStatus::Cancelled { .. }
        ));
    }
}
