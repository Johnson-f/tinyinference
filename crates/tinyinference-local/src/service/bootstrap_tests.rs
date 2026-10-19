#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::os::unix::fs::PermissionsExt;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::collections::BTreeMap;

use axum::{Json, Router, routing::get};
use serde_json::json;

use super::*;

#[test]
fn autosummary_debounce_blocks_repeated_calls_inside_window() {
    let mut config = Config::default();
    config.local_ai.autosummary_debounce_ms = 60_000;
    let service = LocalAiService::new(&config);

    assert!(service.should_run_memory_autosummary(&config));
    assert!(!service.should_run_memory_autosummary(&config));
}

/// Mock Ollama that answers `/api/tags` and counts every request that would
/// mutate the runtime (`/api/pull`, `/api/create`, `/api/delete`) or any
/// unknown path.
async fn spawn_counting_ollama(forbidden_hits: Arc<AtomicUsize>) -> String {
    let fallback_hits = Arc::clone(&forbidden_hits);
    let app = Router::new()
        .route(
            "/api/tags",
            get(|| async { Json(json!({ "models": [{ "name": "gemma3:1b-it-qat" }] })) }),
        )
        .fallback(move || {
            let hits = Arc::clone(&fallback_hits);
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                axum::http::StatusCode::NOT_FOUND
            }
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://127.0.0.1:{}", addr.port())
}

/// A loopback listener that accepts requests and closes them without an HTTP
/// response, keeping the endpoint bound for the duration of the test.
async fn spawn_rejecting_endpoint() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            drop(stream);
        }
    });
    format!("http://127.0.0.1:{}", addr.port())
}

async fn spawn_counting_rejecting_endpoint(hits: Arc<AtomicUsize>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            hits.fetch_add(1, Ordering::SeqCst);
            drop(stream);
        }
    });
    format!("http://127.0.0.1:{}", addr.port())
}

async fn spawn_latched_rejecting_endpoint(
    hits: Arc<AtomicUsize>,
) -> (
    String,
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            hits.fetch_add(1, Ordering::SeqCst);
            let _ = accepted_tx.send(());
            let _ = release_rx.await;
            drop(stream);
        }
        while let Ok((stream, _)) = listener.accept().await {
            hits.fetch_add(1, Ordering::SeqCst);
            drop(stream);
        }
    });
    (
        format!("http://127.0.0.1:{}", addr.port()),
        accepted_rx,
        release_tx,
        task,
    )
}

async fn spawn_latched_endpoint() -> (
    String,
    tokio::sync::oneshot::Receiver<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            let _ = accepted_tx.send(());
            let _stream = stream;
            std::future::pending::<()>().await;
        }
    });
    (
        format!("http://127.0.0.1:{}", addr.port()),
        accepted_rx,
        task,
    )
}

fn opted_in(base_url: &str) -> Config {
    let mut config = Config::default();
    config.local_ai.runtime_enabled = true;
    config.local_ai.opt_in_confirmed = true;
    config.local_ai.provider = "ollama".to_string();
    config.local_ai.base_url = Some(base_url.to_string());
    config
}

/// A local TCP port with nothing listening on it.
fn unbound_base_url() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn bootstrap_without_opt_in_is_disabled_and_probes_nothing() {
    let forbidden = Arc::new(AtomicUsize::new(0));
    let base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let mut config = opted_in(&base);
    config.local_ai.opt_in_confirmed = false;
    let service = LocalAiService::new(&config);

    service.bootstrap(&config).await;

    assert_eq!(service.status().state, "disabled");
    assert_eq!(forbidden.load(Ordering::SeqCst), 0);
}

/// Regression: bootstrap is a read-only probe. Against a healthy endpoint it
/// reports `ready` and never asks the runtime to pull a model.
#[tokio::test]
async fn bootstrap_against_reachable_endpoint_is_ready_without_pulling() {
    let forbidden = Arc::new(AtomicUsize::new(0));
    let base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let mut config = opted_in(&base);
    // Models the endpoint does NOT serve: the old bootstrap would have pulled
    // them. The probe must not.
    config.local_ai.chat_model_id = "gemma3:4b-it-qat".to_string();
    config.local_ai.embedding_model_id = "bge-m3".to_string();
    config.local_ai.vision_model_id = "moondream:1.8b-v2-q4_K_S".to_string();
    let service = LocalAiService::new(&config);

    service.bootstrap(&config).await;

    let status = service.status();
    assert_eq!(status.state, "ready");
    assert_eq!(status.warning, None);
    assert_eq!(status.error_category, None);
    assert_eq!(
        forbidden.load(Ordering::SeqCst),
        0,
        "bootstrap must only GET /api/tags — no /api/pull or other mutating call"
    );
}

#[tokio::test]
async fn ready_status_is_retained_until_reset_to_idle() {
    let _guard = crate::service::inference_test_guard();

    let forbidden = Arc::new(AtomicUsize::new(0));
    let reachable_base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let reachable = opted_in(&reachable_base);
    let unreachable_base = spawn_rejecting_endpoint().await;
    let unreachable = opted_in(&unreachable_base);
    let service = LocalAiService::new(&reachable);

    service.bootstrap(&reachable).await;
    assert_eq!(service.status().state, "ready");

    // A ready state remains authoritative until the host explicitly resets it.
    service.bootstrap(&unreachable).await;
    assert_eq!(service.status().state, "ready");

    service.reset_to_idle(&unreachable);
    service.bootstrap(&unreachable).await;
    assert_eq!(service.status().state, "unreachable");
    assert_eq!(forbidden.load(Ordering::SeqCst), 0);
}

/// Regression: with nothing listening, bootstrap reports `unreachable` and
/// does not try to start a runtime (it returns promptly and the endpoint is
/// still not listening afterwards).
#[tokio::test]
async fn bootstrap_with_rejecting_endpoint_is_unreachable_promptly() {
    let base = spawn_rejecting_endpoint().await;
    let config = opted_in(&base);
    let service = LocalAiService::new(&config);

    let started = std::time::Instant::now();
    service.bootstrap(&config).await;
    let elapsed = started.elapsed();

    let status = service.status();
    assert_eq!(status.state, "unreachable");
    assert_eq!(status.error_category.as_deref(), Some("server"));
    assert!(
        status
            .warning
            .as_deref()
            .is_some_and(|w| w.contains("not reachable")),
        "warning must tell the user to start their runtime: {:?}",
        status.warning
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "probe must not wait on a spawned runtime (took {elapsed:?})"
    );
}

#[tokio::test]
async fn rapid_bootstrap_burst_does_not_queue_unreachable_probes() {
    let hits = Arc::new(AtomicUsize::new(0));
    let config = opted_in(&spawn_counting_rejecting_endpoint(Arc::clone(&hits)).await);
    let service = Arc::new(LocalAiService::new(&config));
    let mut first_task = None;
    for _ in 0..8 {
        if let Some(task) = service.schedule_bootstrap(config.clone()) {
            assert!(
                first_task.replace(task).is_none(),
                "only one probe is scheduled"
            );
        }
    }
    first_task
        .expect("first poll schedules a probe")
        .await
        .unwrap();

    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "a burst of status polls must not queue redundant unreachable probes"
    );

    service
        .schedule_bootstrap(config)
        .expect("a fresh poll can retry after the unreachable probe completes")
        .await
        .unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn concurrent_unreachable_bootstraps_share_probe_and_later_retry() {
    let _guard = crate::service::inference_test_guard();
    let hits = Arc::new(AtomicUsize::new(0));
    let (base, accepted, release, server) =
        spawn_latched_rejecting_endpoint(Arc::clone(&hits)).await;
    let config = opted_in(&base);
    let service = Arc::new(LocalAiService::new(&config));
    let callers = 8;
    let barrier = Arc::new(tokio::sync::Barrier::new(callers + 1));
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut tasks = Vec::new();
    for _ in 0..callers {
        let service = Arc::clone(&service);
        let config = config.clone();
        let barrier = Arc::clone(&barrier);
        let started_tx = started_tx.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            started_tx.send(()).unwrap();
            service.bootstrap(&config).await;
        }));
    }
    barrier.wait().await;
    drop(started_tx);
    for _ in 0..callers {
        started_rx.recv().await.unwrap();
    }
    accepted.await.unwrap();
    for _ in 0..callers {
        tokio::task::yield_now().await;
    }
    release.send(()).unwrap();
    for task in tasks {
        task.await.unwrap();
    }
    server.abort();

    assert_eq!(service.status().state, "unreachable");
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    let retry_hits = Arc::new(AtomicUsize::new(0));
    let retry_config = opted_in(&spawn_counting_rejecting_endpoint(Arc::clone(&retry_hits)).await);
    service.bootstrap(&retry_config).await;
    assert_eq!(retry_hits.load(Ordering::SeqCst), 1);
}

async fn wait_for_bootstrap_waiters(service: &LocalAiService, expected: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let waiters = service
                .bootstrap_flight
                .lock()
                .as_ref()
                .map_or(0, tokio::sync::watch::Sender::receiver_count);
            if waiters == expected {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all callers should join the current bootstrap generation");
}

#[tokio::test]
async fn cancelling_foreground_bootstrap_releases_waiters_to_retry() {
    let _guard = crate::service::inference_test_guard();
    let hits = Arc::new(AtomicUsize::new(0));
    let (base, accepted, release, server) =
        spawn_latched_rejecting_endpoint(Arc::clone(&hits)).await;
    let config = opted_in(&base);
    let service = Arc::new(LocalAiService::new(&config));

    let foreground_service = Arc::clone(&service);
    let foreground_config = config.clone();
    let foreground = tokio::spawn(async move {
        foreground_service.ensure_ready(&foreground_config).await;
    });
    accepted.await.unwrap();

    let waiter_service = Arc::clone(&service);
    let waiter_config = config.clone();
    let waiter = tokio::spawn(async move {
        waiter_service.ensure_ready(&waiter_config).await;
    });
    wait_for_bootstrap_waiters(&service, 1).await;
    foreground.abort();
    assert!(foreground.await.unwrap_err().is_cancelled());
    release.send(()).unwrap();
    waiter.await.unwrap();
    server.abort();

    assert_eq!(service.status().state, "unreachable");
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn foreground_ensure_ready_joins_scheduled_background_probe() {
    let _guard = crate::service::inference_test_guard();
    let hits = Arc::new(AtomicUsize::new(0));
    let (base, accepted, release, server) =
        spawn_latched_rejecting_endpoint(Arc::clone(&hits)).await;
    let config = opted_in(&base);
    let service = Arc::new(LocalAiService::new(&config));
    let foreground_service = Arc::clone(&service);
    let foreground_config = config.clone();
    let foreground = tokio::spawn(async move {
        foreground_service.ensure_ready(&foreground_config).await;
    });
    accepted.await.unwrap();

    assert!(
        service.schedule_bootstrap(config).is_none(),
        "a background poll must join rather than queue behind a foreground probe"
    );
    release.send(()).unwrap();
    foreground.await.unwrap();
    server.abort();
    assert_eq!(service.status().state, "unreachable");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn ensure_ready_waits_for_scheduled_background_probe() {
    let _guard = crate::service::inference_test_guard();
    let hits = Arc::new(AtomicUsize::new(0));
    let (base, accepted, release, server) =
        spawn_latched_rejecting_endpoint(Arc::clone(&hits)).await;
    let config = opted_in(&base);
    let service = Arc::new(LocalAiService::new(&config));
    let background = service
        .schedule_bootstrap(config.clone())
        .expect("background probe should be scheduled");
    accepted.await.unwrap();
    let foreground_service = Arc::clone(&service);
    let foreground = tokio::spawn(async move {
        foreground_service.ensure_ready(&config).await;
    });
    wait_for_bootstrap_waiters(&service, 1).await;

    release.send(()).unwrap();
    foreground.await.unwrap();
    background.await.unwrap();
    server.abort();
    assert_eq!(service.status().state, "unreachable");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn opt_out_waiter_applies_its_own_disabled_gate_after_probe() {
    let _guard = crate::service::inference_test_guard();
    let hits = Arc::new(AtomicUsize::new(0));
    let (base, accepted, release, server) =
        spawn_latched_rejecting_endpoint(Arc::clone(&hits)).await;
    let enabled_config = opted_in(&base);
    let service = Arc::new(LocalAiService::new(&enabled_config));
    let leader_service = Arc::clone(&service);
    let leader_config = enabled_config.clone();
    let leader = tokio::spawn(async move {
        leader_service.ensure_ready(&leader_config).await;
    });
    accepted.await.unwrap();

    let mut disabled_config = enabled_config.clone();
    disabled_config.local_ai.opt_in_confirmed = false;
    let waiter_service = Arc::clone(&service);
    let waiter = tokio::spawn(async move {
        waiter_service.ensure_ready(&disabled_config).await;
    });
    wait_for_bootstrap_waiters(&service, 1).await;
    release.send(()).unwrap();
    leader.await.unwrap();
    waiter.await.unwrap();
    server.abort();

    assert_eq!(service.status().state, "disabled");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[test]
fn status_with_provider_overlays_one_snapshot_without_mutating_cached_status() {
    let mut original_config = Config::default();
    original_config.local_ai.provider = "ollama".to_string();
    let service = LocalAiService::new(&original_config);
    let mut current_config = original_config.clone();
    current_config.local_ai.provider = "lm_studio".to_string();

    let snapshot = service.status_with_provider(&current_config);

    assert_eq!(snapshot.provider, "lm_studio");
    assert_eq!(service.status().provider, "ollama");
}

#[tokio::test]
async fn unpolled_scheduled_bootstrap_cancellation_releases_single_flight_guard() {
    let config = opted_in(&unbound_base_url());
    let service = Arc::new(LocalAiService::new(&config));
    let task = service
        .schedule_bootstrap(config)
        .expect("first poll schedules a probe");
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    let hits = Arc::new(AtomicUsize::new(0));
    let retry_config = opted_in(&spawn_counting_rejecting_endpoint(Arc::clone(&hits)).await);
    service
        .schedule_bootstrap(retry_config)
        .expect("cancellation before first poll releases the guard")
        .await
        .unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancelled_scheduled_bootstrap_releases_single_flight_guard() {
    let (blocked_base, accepted, server) = spawn_latched_endpoint().await;
    let config = opted_in(&blocked_base);
    let service = Arc::new(LocalAiService::new(&config));
    let task = service
        .schedule_bootstrap(config)
        .expect("first poll schedules a probe");
    accepted.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    server.abort();

    let hits = Arc::new(AtomicUsize::new(0));
    let retry_config = opted_in(&spawn_counting_rejecting_endpoint(Arc::clone(&hits)).await);
    service
        .schedule_bootstrap(retry_config)
        .expect("cancellation releases the scheduling guard")
        .await
        .unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

/// Linux x86_64 process probes use seccomp to observe forbidden starts
/// synchronously. Other platforms do not claim this syscall-level guarantee.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[tokio::test]
async fn bootstrap_does_not_launch_local_runtime_processes() {
    let _guard = crate::service::inference_test_guard();

    let scripts = tempfile::tempdir().expect("scripts");
    let marker = scripts.path().join("runtime-launched.marker");
    for name in [
        "ollama",
        "lms",
        "mlx_lm.server",
        "mlx_lm.generate",
        "piper",
        "python",
        "python3",
    ] {
        let executable = scripts.path().join(name);
        std::fs::write(
            &executable,
            "#!/bin/sh\n: > \"$TINYINFERENCE_SPAWN_MARKER\"\nexit 0\n",
        )
        .expect("write runtime stub");
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755))
            .expect("make runtime stub executable");
    }

    // Prove that the immediately-exiting stub would leave a trace if a launch
    // occurred; a closed endpoint alone cannot distinguish that from no launch.
    let canary = Command::new(scripts.path().join("ollama"))
        .env("TINYINFERENCE_SPAWN_MARKER", &marker)
        .status()
        .expect("run marker canary");
    assert!(canary.success());
    assert!(marker.exists(), "the fake runtime must create its marker");
    std::fs::remove_file(&marker).expect("clear canary marker");

    let healthy_base = spawn_counting_ollama(Arc::new(AtomicUsize::new(0))).await;
    run_process_probe_child(scripts.path(), &marker, &healthy_base, "ready").await;

    let rejecting_base = spawn_rejecting_endpoint().await;
    run_process_probe_child(scripts.path(), &marker, &rejecting_base, "unreachable").await;
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
async fn run_process_probe_child(
    scripts_dir: &std::path::Path,
    spawn_marker: &std::path::Path,
    base: &str,
    expected_state: &str,
) {
    let completion_marker = scripts_dir.join(format!("child-completed-{expected_state}.marker"));
    let output = tokio::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "service::bootstrap::tests::bootstrap_process_probe_child",
            "--ignored",
            "--nocapture",
        ])
        .env("PATH", scripts_dir)
        .env("TINYINFERENCE_BOOTSTRAP_BASE_URL", base)
        .env("TINYINFERENCE_SPAWN_MARKER", spawn_marker)
        .env("TINYINFERENCE_CHILD_COMPLETED", &completion_marker)
        .env("TINYINFERENCE_EXPECTED_STATE", expected_state)
        .env_remove("OPENHUMAN_OLLAMA_BASE_URL")
        .env_remove("OLLAMA_BIN")
        .env_remove("PIPER_BIN")
        .env_remove("WHISPER_BIN")
        .output()
        .await
        .expect("run bootstrap probe in isolated child");

    assert!(
        output.status.success(),
        "child bootstrap test failed for expected {expected_state}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        completion_marker.exists(),
        "the exact child test must run through diagnostics and bootstrap"
    );
    assert!(
        !spawn_marker.exists(),
        "bootstrap must not launch a runtime when the endpoint is {expected_state}"
    );
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn install_process_creation_guard() {
    use seccompiler::{
        BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
        SeccompRule,
    };
    use std::convert::TryInto;

    let arch = std::env::consts::ARCH
        .try_into()
        .expect("supported Linux x86_64");

    let deny_process_creation = SeccompFilter::new(
        [
            (libc::SYS_fork, vec![]),
            (libc::SYS_vfork, vec![]),
            (libc::SYS_execve, vec![]),
            (libc::SYS_execveat, vec![]),
            (
                libc::SYS_clone,
                vec![
                    SeccompRule::new(vec![
                        SeccompCondition::new(
                            0,
                            SeccompCmpArgLen::Dword,
                            SeccompCmpOp::MaskedEq(libc::CLONE_THREAD as u64),
                            0,
                        )
                        .expect("valid clone flags condition"),
                    ])
                    .expect("valid clone rule"),
                ],
            ),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
        SeccompAction::Allow,
        SeccompAction::KillProcess,
        arch,
    )
    .expect("build process-creation seccomp filter");
    let deny_process_creation: BpfProgram = deny_process_creation
        .try_into()
        .expect("compile process-creation seccomp filter");
    seccompiler::apply_filter(&deny_process_creation).expect("install process-creation filter");

    // libc may probe clone3 before falling back to clone for threads. Return
    // ENOSYS there; the clone rule above permits only CLONE_THREAD clones.
    let disable_clone3 = SeccompFilter::new(
        [(libc::SYS_clone3, vec![])].into_iter().collect(),
        SeccompAction::Allow,
        SeccompAction::Errno(libc::ENOSYS as u32),
        arch,
    )
    .expect("build clone3 fallback filter");
    let disable_clone3: BpfProgram = disable_clone3
        .try_into()
        .expect("compile clone3 fallback filter");
    seccompiler::apply_filter(&disable_clone3).expect("install clone3 fallback filter");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[tokio::test]
#[ignore = "invoked by bootstrap_does_not_launch_local_runtime_processes"]
async fn bootstrap_process_probe_child() {
    let base = std::env::var("TINYINFERENCE_BOOTSTRAP_BASE_URL")
        .expect("parent test passes the mock endpoint");
    let config = opted_in(&base);
    let service = LocalAiService::new(&config);

    install_process_creation_guard();
    std::thread::spawn(|| ())
        .join()
        .expect("process guard must continue to allow threads");

    service
        .diagnostics(&config)
        .await
        .expect("diagnostics should succeed against the mock endpoint");
    service.bootstrap(&config).await;

    assert_eq!(
        service.status().state,
        std::env::var("TINYINFERENCE_EXPECTED_STATE").expect("parent passes expected state")
    );
    std::fs::write(
        std::env::var("TINYINFERENCE_CHILD_COMPLETED").expect("parent passes completion marker"),
        "completed",
    )
    .expect("write child completion marker");
}

/// An unreachable endpoint is probed again on the next call, so a runtime the
/// user starts later is picked up without a restart.
#[tokio::test]
async fn unreachable_endpoint_is_reprobed_on_next_bootstrap() {
    let config = opted_in(&unbound_base_url());
    let service = LocalAiService::new(&config);
    service.bootstrap(&config).await;
    assert_eq!(service.status().state, "unreachable");

    let forbidden = Arc::new(AtomicUsize::new(0));
    let base = spawn_counting_ollama(Arc::clone(&forbidden)).await;
    let config = opted_in(&base);
    service.bootstrap(&config).await;

    assert_eq!(service.status().state, "ready");
    assert_eq!(forbidden.load(Ordering::SeqCst), 0);
}

/// OpenAI-compatible runtimes (LM Studio) are probed with `GET /v1/models`.
#[tokio::test]
async fn bootstrap_probes_openai_compatible_endpoint_with_v1_models() {
    let hits = Arc::new(AtomicUsize::new(0));
    let models_hits = Arc::clone(&hits);
    let app = Router::new().route(
        "/v1/models",
        get(move || {
            let hits = Arc::clone(&models_hits);
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                Json(json!({ "data": [{ "id": "local-model" }] }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let mut config = opted_in(&format!("http://127.0.0.1:{}/v1", addr.port()));
    config.local_ai.provider = "lm_studio".to_string();
    let service = LocalAiService::new(&config);

    service.bootstrap(&config).await;

    assert_eq!(service.status().state, "ready");
    assert_eq!(service.status().provider, "lm_studio");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[test]
fn vision_mode_follows_configured_vision_model() {
    let mut config = Config::default();
    let service = LocalAiService::new(&config);
    assert_eq!(service.status().vision_mode, "disabled");
    assert_eq!(service.status().vision_state, "disabled");

    config.local_ai.vision_model_id = "moondream:1.8b-v2-q4_K_S".to_string();
    let service = LocalAiService::new(&config);
    assert_eq!(service.status().vision_mode, "ondemand");
    assert_eq!(service.status().vision_state, "idle");
}
