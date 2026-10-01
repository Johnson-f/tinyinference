use crate::ollama::{ollama_base_url, ollama_base_url_from_override};
use crate::service::RuntimeConfig as Config;

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

    /// Check Ollama health against the given base URL.
    pub(in crate::service) async fn ollama_healthy_at(&self, base_url: &str) -> bool {
        tracing::debug!(
            target: "local_ai::ollama_admin",
            %base_url,
            "[local_ai:ollama_admin] ollama_healthy_at: checking"
        );
        self.http
            .get(format!("{base_url}/api/tags"))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Backward-compat wrapper — resolves the URL from env vars only (no config).
    /// Prefer [`ollama_healthy_at`] when a `Config` is available.
    pub(in crate::service) async fn ollama_healthy(&self) -> bool {
        self.ollama_healthy_at(&ollama_base_url()).await
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

    pub(in crate::service) async fn has_model(&self, model: &str) -> Result<bool, String> {
        self.has_model_at(&ollama_base_url(), model).await
    }

    pub(in crate::service) async fn has_model_for_config(
        &self,
        config: &Config,
        model: &str,
    ) -> Result<bool, String> {
        self.has_model_at(
            &ollama_base_url_from_override(config.local_ai.base_url.as_deref()),
            model,
        )
        .await
    }

    pub(in crate::service) async fn has_model_at(
        &self,
        base_url: &str,
        model: &str,
    ) -> Result<bool, String> {
        use crate::ollama::OllamaTagsResponse;
        // Issue the /api/tags GET directly. We previously short-circuited via
        // ollama_healthy(), but that doubled the number of /api/tags round-trips
        // on healthy polls (one probe + one tags fetch). With three has_model()
        // calls per assets_status poll (chat, vision, embedding) that was 6
        // network calls instead of 3. The 500ms connect_timeout on the shared
        // reqwest client (set in bootstrap.rs) bounds the cost when the server
        // is down — the connect failure surfaces as Err, same as ollama_healthy()
        // would have surfaced as `false`.
        log::debug!("[local_ai] has_model_at: checking for model `{model}` at {base_url}");
        let response = self
            .http
            .get(format!("{base_url}/api/tags"))
            // Per-request timeout matches list_models (5s). The shared client's
            // connect_timeout only bounds the TCP handshake; without this a
            // hung server (accepted connection, no response body) would block
            // assets_status polls indefinitely.
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| format!("ollama tags request failed: {e}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = body.trim();
            return Err(format!(
                "ollama tags failed with status {}{}",
                status,
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            ));
        }
        let payload: OllamaTagsResponse = response
            .json()
            .await
            .map_err(|e| format!("ollama tags parse failed: {e}"))?;

        let target = model.to_ascii_lowercase();
        Ok(payload.models.iter().any(|m| {
            let name = m.name.to_ascii_lowercase();
            name == target || name.starts_with(&(target.clone() + ":"))
        }))
    }
}
