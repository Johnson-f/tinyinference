# TinyInference Decisions

`tinyinference_decisions` provides a typed Rust client for TypeSafe AI's System One API and Jev decision model. A request supplies shared state and independent Choice, Score, and Noul questions. Jev returns typed answers alongside latency, attempts, usage, and request metadata.

The client keeps execution and policy outside the model. A Choice selects only from caller supplied values, a Score rates one described dimension, and a Noul reports the probability of a yes/no condition. Callers own confidence thresholds, escalation, state transitions, and side effects.

```rust,no_run
use std::collections::BTreeMap;
use serde_json::json;
use tinyinference_decisions::{Choice, Client, EvaluationRequest, Question};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let request = EvaluationRequest::jev(
    json!({"ticket": "I was charged twice"}),
    BTreeMap::from([("route".to_owned(), Question::Choice(Choice {
        instructions: json!("Which team should handle this ticket?"),
        criteria: BTreeMap::from([
            ("billing".to_owned(), None),
            ("technical".to_owned(), None),
        ]),
    }))]),
);
let result = Client::from_env()?.evaluate(&request).await?;
println!("{:?}", result.response.answers["route"]);
# Ok(())
# }
```

The API key is read from `TYPESAFE_API_KEY` or supplied through `ClientConfig`. Keys are redacted from `Debug` output and never included in errors. The client supports TypeSafe and OpenRouter endpoints, bounded retries, request and response validation, and explicit per-call failure metadata.

For OpenRouter, construct `ClientConfig::openrouter("<key>")`. Tiny Humans proxy users can use `ClientConfig::tinyhumans_openrouter("<key>")`. Custom endpoints can be configured with `.with_endpoint_url(...)`.

The live example spends a real API call:

```sh
TYPESAFE_API_KEY='<key>' cargo run -p tinyinference_decisions --example basic
```

The crate is part of the TinyInference workspace and is licensed GPL-3.0-only.
