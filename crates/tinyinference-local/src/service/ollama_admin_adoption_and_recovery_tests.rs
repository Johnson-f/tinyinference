//! Ownership-contract tests for the Ollama daemon lifecycle: adopting a daemon
//! we did not spawn, and recovering from a spawn marker left by a crash.
//!
//! Ported from OpenHuman's `tests/ollama_lifecycle_e2e.rs`. The owned-spawn
//! flow lives in `ollama_admin_lm_studio_and_lifecycle_tests.rs`.

use super::*;
use crate::spawn_marker::{OllamaSpawnMarker, pid_is_alive, read_marker_at, write_marker_at};

fn sleep_command() -> tokio::process::Command {
    let mut cmd = if cfg!(windows) {
        let mut c = tokio::process::Command::new("powershell");
        c.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 30"]);
        c
    } else {
        let mut c = tokio::process::Command::new("sleep");
        c.arg("30");
        c
    };
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd
}

fn tmp_config(tmp: &tempfile::TempDir) -> Config {
    let mut config = Config::default();
    config.workspace_dir = tmp.path().to_path_buf();
    config.config_path = tmp.path().join("config.toml");
    config.shared_root_dir = tmp.path().to_path_buf();
    config
}

/// Spawn and reap a short-lived process so its PID is known to be dead.
fn dead_pid() -> u32 {
    let mut child = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", "exit 0"])
            .spawn()
            .expect("spawn cmd")
    } else {
        std::process::Command::new("true")
            .spawn()
            .expect("spawn true")
    };
    let pid = child.id();
    let _ = child.wait();
    pid
}

async fn wait_until_dead(pid: u32) -> bool {
    for _ in 0..40 {
        if !pid_is_alive(pid) {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test]
async fn external_adoption_shutdown_leaves_external_process_running() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp_config(&tmp);
    let service = LocalAiService::new(&config);

    assert!(
        !service.has_owned_ollama(),
        "fresh service must have no owned child"
    );

    // Stand-in for an Ollama the user started outside the app.
    let mut external = sleep_command().spawn().expect("spawn external stub");
    let external_pid = external.id().expect("external pid");

    service.shutdown_owned_ollama(&config).await;

    let marker_path = crate::service::paths::ollama_spawn_marker_path(&config);
    assert!(
        !marker_path.exists(),
        "adopting an external daemon must not create a spawn marker"
    );
    assert!(
        pid_is_alive(external_pid),
        "external daemon pid {external_pid} must survive a no-op shutdown"
    );

    let _ = external.kill().await;
    let _ = external.wait().await;
}

#[tokio::test]
async fn start_and_wait_adopts_healthy_daemon_without_owning_it() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    let app = Router::new().route("/api/tags", get(|| async { Json(json!({ "models": [] })) }));
    let base = spawn_mock(app).await;

    let mut config = tmp_config(&tmp);
    config.local_ai.base_url = Some(base);
    let service = LocalAiService::new(&config);

    // The binary path is never touched because a healthy daemon short-circuits
    // the spawn.
    service
        .start_and_wait_for_server(&config, std::path::Path::new("/nonexistent/ollama"))
        .await
        .expect("a healthy daemon must be adopted");

    assert!(
        !service.has_owned_ollama(),
        "an adopted daemon must not be tracked as owned"
    );
    assert!(
        !crate::service::paths::ollama_spawn_marker_path(&config).exists(),
        "adopting must not write a spawn marker"
    );
}

#[tokio::test]
async fn reclaim_orphan_clears_stale_marker_with_dead_pid() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    let mut config = tmp_config(&tmp);
    config.local_ai.base_url = Some("http://127.0.0.1:1".to_string());
    let service = LocalAiService::new(&config);

    let marker_path = crate::service::paths::ollama_spawn_marker_path(&config);
    let pid = dead_pid();
    write_marker_at(
        &marker_path,
        &OllamaSpawnMarker::new(pid, std::path::Path::new("test-stub")),
    )
    .expect("write marker");
    assert!(marker_path.exists());

    service.reclaim_orphan_if_ours(&config).await;

    assert!(
        !marker_path.exists(),
        "a marker whose pid is dead must be cleared"
    );
}

#[tokio::test]
async fn reclaim_orphan_without_marker_is_noop() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp_config(&tmp);
    let service = LocalAiService::new(&config);

    service.reclaim_orphan_if_ours(&config).await;

    assert!(!crate::service::paths::ollama_spawn_marker_path(&config).exists());
}

#[tokio::test]
async fn reclaim_orphan_keeps_marker_and_process_when_daemon_not_healthy() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    let mut config = tmp_config(&tmp);
    // Nothing listens here, so the alive pid may be a recycled, unrelated
    // process; it must not be killed.
    config.local_ai.base_url = Some("http://127.0.0.1:1".to_string());
    let service = LocalAiService::new(&config);

    let mut bystander = sleep_command().spawn().expect("spawn stub");
    let pid = bystander.id().expect("pid");
    let marker_path = crate::service::paths::ollama_spawn_marker_path(&config);
    write_marker_at(
        &marker_path,
        &OllamaSpawnMarker::new(pid, std::path::Path::new("test-stub")),
    )
    .expect("write marker");

    service.reclaim_orphan_if_ours(&config).await;

    assert!(
        pid_is_alive(pid),
        "an alive pid with an unhealthy daemon must not be killed"
    );
    assert_eq!(
        read_marker_at(&marker_path).map(|m| m.pid),
        Some(pid),
        "the marker must be left for a later reclaim"
    );

    let _ = bystander.kill().await;
    let _ = bystander.wait().await;
}

#[tokio::test]
async fn reclaim_orphan_kills_our_previous_daemon_when_healthy() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    let app = Router::new().route("/api/tags", get(|| async { Json(json!({ "models": [] })) }));
    let base = spawn_mock(app).await;
    let mut config = tmp_config(&tmp);
    config.local_ai.base_url = Some(base);
    let service = LocalAiService::new(&config);

    // Stand-in for the orphan a crashed previous session left behind.
    let mut orphan = sleep_command().spawn().expect("spawn stub");
    let pid = orphan.id().expect("pid");
    let marker_path = crate::service::paths::ollama_spawn_marker_path(&config);
    write_marker_at(
        &marker_path,
        &OllamaSpawnMarker::new(pid, std::path::Path::new("test-stub")),
    )
    .expect("write marker");

    service.reclaim_orphan_if_ours(&config).await;

    assert!(
        !marker_path.exists(),
        "reclaiming must clear the spawn marker"
    );
    // Reap so a killed-but-unreaped zombie does not read as alive.
    let _ = orphan.wait().await;
    assert!(
        wait_until_dead(pid).await,
        "the recorded orphan pid {pid} must be killed"
    );
}

#[tokio::test]
async fn stale_marker_does_not_break_diagnostics() {
    let _guard = crate::service::inference_test_guard();
    let tmp = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", "http://127.0.0.1:1");
    }
    let config = tmp_config(&tmp);

    let marker_path = crate::service::paths::ollama_spawn_marker_path(&config);
    write_marker_at(
        &marker_path,
        &OllamaSpawnMarker::new(dead_pid(), std::path::Path::new("test-stub")),
    )
    .expect("write marker");

    let service = LocalAiService::new(&config);
    let diag = service
        .diagnostics(&config)
        .await
        .expect("diagnostics must succeed with a stale spawn marker");

    assert_eq!(diag["ollama_running"], false);
    let issues = diag["issues"].as_array().cloned().unwrap_or_default();
    assert!(
        !issues.is_empty(),
        "an unreachable server must surface issues"
    );
    // Diagnostics is read-only: the marker is only consumed by the bootstrap.
    assert!(marker_path.exists());

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}
