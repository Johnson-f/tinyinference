use super::*;

use crate::{Error, Result, ScoredDoc};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[tokio::test]
async fn empty_candidates_need_no_query_or_provider_call() {
    let reranker = VoyageReranker::new("test-key").unwrap();
    let response = reranker
        .rerank(RerankRequest::new("", Vec::new()))
        .await
        .unwrap();
    assert_eq!(
        response,
        RerankResponse {
            results: vec![],
            usage: None
        }
    );
}

#[test]
fn results_have_a_stable_serializable_shape() {
    let response = RerankResponse {
        results: vec![RerankResult {
            index: 2,
            relevance_score: 0.75,
        }],
        usage: Some(RerankUsage { input_tokens: 17 }),
    };
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({
            "results":[{"index":2,"relevance_score":0.75}], "usage":{"input_tokens":17}
        })
    );
    fn send_sync<T: Send + Sync>() {}
    send_sync::<VoyageReranker>();
    send_sync::<RerankRequest>();
    send_sync::<Arc<dyn Reranker>>();
}

struct FixedReranker {
    documents: Vec<String>,
    reply: Mutex<Option<Result<RerankResponse>>>,
}

#[async_trait::async_trait]
impl Reranker for FixedReranker {
    fn name(&self) -> &str {
        "fixture"
    }
    fn model_id(&self) -> &str {
        "fixed"
    }
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse> {
        assert_eq!(request.documents, self.documents);
        self.reply.lock().unwrap().take().expect("only one call")
    }
}

async fn rerank_hits(
    model: &dyn Reranker,
    hits: Vec<ScoredDoc>,
    texts: &HashMap<String, String>,
) -> Result<Vec<(ScoredDoc, Option<f64>)>> {
    let documents = hits
        .iter()
        .map(|hit| {
            texts
                .get(&hit.id)
                .cloned()
                .ok_or_else(|| Error::Validation("missing source document".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    match model.rerank(RerankRequest::new("query", documents)).await {
        Ok(response) => response
            .results
            .into_iter()
            .map(|result| {
                let hit = hits
                    .get(result.index)
                    .cloned()
                    .ok_or_else(|| Error::Validation("invalid result index".into()))?;
                Ok((hit, Some(result.relevance_score)))
            })
            .collect(),
        Err(Error::Cancelled) => Err(Error::Cancelled),
        Err(Error::Rerank(_)) => Ok(hits.into_iter().map(|hit| (hit, None)).collect()),
        Err(error) => Err(error),
    }
}

fn reverse_rankings(documents: Vec<String>) -> FixedReranker {
    FixedReranker {
        documents,
        reply: Mutex::new(Some(Ok(RerankResponse {
            results: vec![
                RerankResult {
                    index: 1,
                    relevance_score: 0.9,
                },
                RerankResult {
                    index: 0,
                    relevance_score: 0.1,
                },
            ],
            usage: None,
        }))),
    }
}

#[tokio::test]
async fn external_search_hits_keep_ids_metadata_and_original_scores() {
    let hits = vec![
        ScoredDoc {
            id: "a".into(),
            score: 0.8,
            metadata: json!({"source":"one"}),
        },
        ScoredDoc {
            id: "b".into(),
            score: 0.6,
            metadata: json!({"source":"two"}),
        },
    ];
    let texts = HashMap::from([("a".into(), "first".into()), ("b".into(), "second".into())]);
    let model = reverse_rankings(vec!["first".into(), "second".into()]);
    let result = rerank_hits(&model, hits.clone(), &texts).await.unwrap();
    assert_eq!(
        result,
        vec![(hits[1].clone(), Some(0.9)), (hits[0].clone(), Some(0.1))]
    );
}

#[tokio::test]
async fn retriever_hits_use_explicit_source_text_and_missing_text_prevents_reranking() {
    let retriever = crate::Retriever::new(
        Arc::new(crate::MockEmbeddingModel::new(16)),
        Arc::new(crate::InMemoryVectorStore::new()),
    );
    retriever
        .index(vec![
            ("a".into(), "alpha".into(), json!({"source":"one"})),
            ("b".into(), "beta".into(), json!({"source":"two"})),
        ])
        .await
        .unwrap();
    let hits = retriever.retrieve("alpha", 2).await.unwrap();
    assert_eq!(hits[0].id, "a");
    let texts = HashMap::from([("a".into(), "alpha".into()), ("b".into(), "beta".into())]);
    let model = reverse_rankings(vec!["alpha".into(), "beta".into()]);
    let result = rerank_hits(&model, hits.clone(), &texts).await.unwrap();
    assert_eq!(
        result,
        vec![(hits[1].clone(), Some(0.9)), (hits[0].clone(), Some(0.1))]
    );
    let unused = reverse_rankings(vec![]);
    assert!(matches!(
        rerank_hits(&unused, hits, &HashMap::new()).await,
        Err(Error::Validation(_))
    ));
    assert!(unused.reply.lock().unwrap().is_some());
}

#[tokio::test]
async fn caller_can_fall_back_on_provider_failure_but_preserves_cancellation() {
    let hit = ScoredDoc {
        id: "a".into(),
        score: 0.8,
        metadata: json!({"source":"one"}),
    };
    let texts = HashMap::from([("a".into(), "first".into())]);
    let failed = FixedReranker {
        documents: vec!["first".into()],
        reply: Mutex::new(Some(Err(RerankError::Transport.into()))),
    };
    assert_eq!(
        rerank_hits(&failed, vec![hit.clone()], &texts)
            .await
            .unwrap(),
        vec![(hit.clone(), None)]
    );
    let cancelled = FixedReranker {
        documents: vec!["first".into()],
        reply: Mutex::new(Some(Err(Error::Cancelled))),
    };
    assert!(matches!(
        rerank_hits(&cancelled, vec![hit], &texts).await,
        Err(Error::Cancelled)
    ));
}
