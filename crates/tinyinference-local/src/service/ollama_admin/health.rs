
use super::super::LocalAiService;

/// Fine-grained result of a health probe against an Ollama endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::service) enum OllamaHealthStatus {
    /// 2-second fast probe succeeded.
    Running,
    /// 2-second probe timed out but an 8-second retry succeeded — server is
    /// alive but under load.
    Degraded,
    /// Both probes failed (connection refused, non-2xx, or 8s timeout).
    Stopped,
}

impl LocalAiService {
    /// Two-phase health probe against the given base URL (#6032).
    ///
    /// Fast path (2 s): if `/api/tags` succeeds → `Running`.
    /// Slow path (8 s): if the fast probe timed out, retry once with more
    /// headroom → `Degraded` (alive but busy). Anything else → `Stopped`.
    pub(in crate::service) async fn ollama_health_status_at(
        &self,
        base_url: &str,
    ) -> OllamaHealthStatus {
        tracing::debug!(
            target: "local_ai::ollama_admin",
            %base_url,
            "[local_ai:ollama_admin] ollama_health_status_at: fast probe"
        );
        let fast = self
            .http
            .get(format!("{base_url}/api/tags"))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await;
        match fast {
            Ok(r) if r.status().is_success() => return OllamaHealthStatus::Running,
            Err(ref e) if e.is_timeout() => {
                tracing::debug!(
                    target: "local_ai::ollama_admin",
                    %base_url,
                    "[local_ai:ollama_admin] ollama_health_status_at: fast probe timed out; retrying with 8s"
                );
                let slow = self
                    .http
                    .get(format!("{base_url}/api/tags"))
                    .timeout(std::time::Duration::from_secs(8))
                    .send()
                    .await;
                if matches!(slow, Ok(ref r) if r.status().is_success()) {
                    tracing::debug!(
                        target: "local_ai::ollama_admin",
                        %base_url,
                        "[local_ai:ollama_admin] ollama_health_status_at: degraded (slow)"
                    );
                    return OllamaHealthStatus::Degraded;
                }
            }
            _ => {}
        }
        tracing::debug!(
            target: "local_ai::ollama_admin",
            %base_url,
            "[local_ai:ollama_admin] ollama_health_status_at: stopped"
        );
        OllamaHealthStatus::Stopped
    }

    /// Quick check that the Ollama runner can actually exec models against the given URL.
    pub(in crate::service) async fn ollama_runner_ok_at(&self, base_url: &str) -> bool {
        let resp = self
            .http
            .get(format!("{base_url}/api/tags"))
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => {
                // Tags endpoint works — but the runner error only shows up on model exec.
                // Do a lightweight pull-status check (won't download, just checks).
                let check = self
                    .http
                    .post(format!("{base_url}/api/show"))
                    .json(&serde_json::json!({"name": "___nonexistent_probe___"}))
                    .timeout(std::time::Duration::from_secs(3))
                    .send()
                    .await;
                match check {
                    Ok(r) => {
                        let status = r.status().as_u16();
                        let body = r.text().await.unwrap_or_default();
                        // 404 = model not found — runner is fine. 500 with fork/exec = broken.
                        if status == 500 && body.contains("fork/exec") {
                            log::warn!("[local_ai] ollama runner broken: {body}");
                            return false;
                        }
                        true
                    }
                    Err(_) => true, // network error, assume ok
                }
            }
            _ => false,
        }
    }
}
