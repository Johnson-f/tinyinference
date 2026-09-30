use crate::ollama::{OllamaTagsResponse, validate_ollama_url};

pub(super) fn lm_studio_models_error_means_unreachable(error: &str) -> bool {
    error.starts_with("lm studio models request failed:")
}

pub(crate) fn interrupted_pull_settle_window_secs(
    observed_bytes: bool,
    settle_window_secs: u64,
) -> u64 {
    if observed_bytes {
        settle_window_secs.max(1)
    } else {
        0
    }
}

/// Kill a process by PID using `sysinfo`'s cross-platform `Process::kill`.
///
/// Used by `reclaim_orphan_if_ours` where we no longer have the original
/// `tokio::process::Child` handle (the spawning openhuman crashed) but
/// recorded the PID in the spawn marker.
pub(crate) fn kill_pid_by_id(pid: u32) {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let target = Pid::from_u32(pid);
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::Some(&[target]), true);
    match sys.process(target) {
        Some(proc) => {
            if proc.kill() {
                log::info!("[local_ai] killed reclaimed ollama orphan pid={pid}");
            } else {
                // sysinfo's kill returns false if the platform refused
                // (permissions, race with exit). The next ollama_healthy()
                // check will reveal whether the daemon is actually gone.
                log::warn!("[local_ai] sysinfo Process::kill returned false for pid={pid}");
            }
        }
        None => {
            log::debug!("[local_ai] kill_pid_by_id: pid={pid} no longer present");
        }
    }
}

/// Confirm that a live PID still runs the executable recorded in its marker.
/// A marker PID can be reused after a crash, so health on the configured URL
/// alone is not evidence that the process is ours.
pub(crate) fn pid_matches_binary(pid: u32, binary_path: &str) -> bool {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let target = Pid::from_u32(pid);
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::Some(&[target]), true);
    let Some(executable) = sys.process(target).and_then(|process| process.exe()) else {
        return false;
    };
    let recorded = std::path::Path::new(binary_path);
    match (
        std::fs::canonicalize(executable),
        std::fs::canonicalize(recorded),
    ) {
        (Ok(actual), Ok(expected)) => actual == expected,
        _ => executable == recorded,
    }
}

/// Test connectivity to a user-supplied Ollama URL.
///
/// Validates the URL via [`validate_ollama_url`], then issues a GET to
/// `{normalized_url}/api/tags` with a 3-second timeout.
/// Returns a JSON object with `reachable`, optional `error`, and
/// `models_count` when reachable.
pub async fn test_ollama_connection(url: &str) -> Result<serde_json::Value, String> {
    let normalized = validate_ollama_url(url)?;
    log::debug!("[local_ai] test_ollama_connection: testing url={normalized}");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    match client.get(format!("{normalized}/api/tags")).send().await {
        Ok(resp) if resp.status().is_success() => {
            let models_count = resp
                .json::<OllamaTagsResponse>()
                .await
                .map(|t| t.models.len())
                .unwrap_or(0);
            log::debug!(
                "[local_ai] test_ollama_connection: reachable url={normalized} models={models_count}"
            );
            Ok(serde_json::json!({
                "reachable": true,
                "error": null,
                "models_count": models_count,
            }))
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("server responded with status {status}: {}", body.trim());
            log::debug!(
                "[local_ai] test_ollama_connection: unreachable url={normalized} err={err}"
            );
            Ok(serde_json::json!({
                "reachable": false,
                "error": err,
                "models_count": null,
            }))
        }
        Err(e) => {
            let err = e.to_string();
            log::debug!(
                "[local_ai] test_ollama_connection: connection failed url={normalized} err={err}"
            );
            Ok(serde_json::json!({
                "reachable": false,
                "error": err,
                "models_count": null,
            }))
        }
    }
}
