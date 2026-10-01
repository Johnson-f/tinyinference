use super::*;

fn temp_config() -> (tempfile::TempDir, Config) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = Config::default();
    config.workspace_dir = dir.path().join("workspace");
    config.config_path = dir.path().join("config.toml");
    (dir, config)
}

#[test]
fn resolve_tts_voice_path_appends_onnx_for_voice_ids() {
    // The installer drop-zone (`bin/piper/voices/<id>.onnx`) is probed
    // FIRST by `resolve_tts_voice_path`, and lives under the shared
    // root (`~/.openhuman/`) — not the temp config. If a sibling
    // install_piper test runs in parallel with the default voice id
    // and leaves a stub there, this test sees that file and the
    // assertion fails. Serialise via the shared install guard and
    // wipe the installer path so the legacy `models/local-ai/tts/`
    // candidate is the only match.
    let _g = shared_install_lock();
    let (_tmp, mut config) = temp_config();
    config.local_ai.tts_voice_id = "en_US-lessac-medium".to_string();
    let installer_onnx = workspace_piper_voice_paths(&config, "en_US-lessac-medium")
        .map(|(onnx, _)| onnx)
        .expect("installer onnx path");
    let _ = std::fs::remove_file(&installer_onnx);
    let model_path = workspace_local_models_dir(&config)
        .join("tts")
        .join("en_US-lessac-medium.onnx");
    std::fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
    std::fs::write(&model_path, b"stub").expect("write");

    let resolved = resolve_tts_voice_path(&config).expect("resolve tts");
    assert_eq!(resolved, model_path.display().to_string());
}

#[test]
fn workspace_piper_voice_paths_returns_onnx_pair() {
    let (_tmp, config) = temp_config();
    let (onnx, json) =
        workspace_piper_voice_paths(&config, "en_US-lessac-medium").expect("voice paths");
    assert!(onnx.to_string_lossy().ends_with("en_US-lessac-medium.onnx"));
    assert!(
        json.to_string_lossy()
            .ends_with("en_US-lessac-medium.onnx.json")
    );
    // Empty voice id is rejected so the caller can fail fast.
    assert!(workspace_piper_voice_paths(&config, "").is_none());
    assert!(workspace_piper_voice_paths(&config, "   ").is_none());
}

#[test]
fn workspace_piper_binary_candidates_include_flat_layout() {
    let (_tmp, config) = temp_config();
    let candidates = workspace_piper_binary_candidates(&config);
    let suffix = if cfg!(windows) { "piper.exe" } else { "piper" };
    assert!(
        candidates.iter().any(|p| p.ends_with(suffix)),
        "flat-layout piper binary must be a candidate"
    );
}

/// Serialise with sibling install_piper tests that
/// write into the same shared `~/.openhuman/bin/...` directory. Uses
/// the existing module-wide guard so all readers/writers go through
/// one critical section.
fn shared_install_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::service::inference_test_guard()
}

#[test]
fn resolve_piper_binary_with_config_prefers_workspace_install() {
    let _g = shared_install_lock();
    let (_tmp, config) = temp_config();
    let target = workspace_piper_binary_candidates(&config)
        .into_iter()
        .next()
        .expect("at least one candidate");
    let _ = std::fs::remove_dir_all(workspace_piper_dir(&config));
    std::fs::create_dir_all(target.parent().expect("parent")).expect("mkdir");
    std::fs::write(&target, b"stub").expect("write stub");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755))
            .expect("mark executable");
    }
    let resolved = resolve_piper_binary_with_config(&config).expect("workspace resolve");
    assert_eq!(resolved, target);
    let _ = std::fs::remove_dir_all(workspace_piper_dir(&config));
}

#[cfg(unix)]
#[test]
fn resolve_piper_binary_with_config_skips_non_executable_workspace_copy() {
    let _g = shared_install_lock();
    let (_tmp, config) = temp_config();
    let target = workspace_piper_binary_candidates(&config)
        .into_iter()
        .next()
        .expect("at least one candidate");
    let _ = std::fs::remove_dir_all(workspace_piper_dir(&config));
    std::fs::create_dir_all(target.parent().expect("parent")).expect("mkdir");
    std::fs::write(&target, b"stub").expect("write stub");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644))
            .expect("mark non-executable");
    }
    let resolved = resolve_piper_binary_with_config(&config);
    assert_ne!(
        resolved.as_deref(),
        Some(target.as_path()),
        "a non-executable workspace copy must not pin piper resolution"
    );
    let _ = std::fs::remove_dir_all(workspace_piper_dir(&config));
}

#[test]
fn standard_unix_bin_dirs_membership_is_platform_correct() {
    let dirs = standard_unix_bin_dirs();
    if cfg!(windows) {
        assert!(dirs.is_empty(), "Windows relies on PATH; no standard dirs");
    } else {
        // Homebrew dirs are the whole point — a GUI app's minimal PATH
        // omits them, so they MUST be probed explicitly (issue #3425).
        assert!(
            dirs.contains(&PathBuf::from("/opt/homebrew/bin")),
            "Apple Silicon Homebrew dir must be probed"
        );
        assert!(
            dirs.contains(&PathBuf::from("/usr/local/bin")),
            "Intel Homebrew / common /usr/local dir must be probed"
        );
    }
}

#[test]
fn resolve_binary_in_dirs_finds_first_match_in_order() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let first = tmp.path().join("first");
    let second = tmp.path().join("second");
    std::fs::create_dir_all(&first).expect("mkdir first");
    std::fs::create_dir_all(&second).expect("mkdir second");
    // Only the second dir holds the binary → it is returned.
    let bin = second.join("piper");
    std::fs::write(&bin, b"stub").expect("write stub");
    let found = resolve_binary_in_dirs("piper", &[first.clone(), second.clone()]);
    assert_eq!(found, Some(bin.clone()));

    // When both hold it, the earlier dir wins (precedence is positional).
    let bin_first = first.join("piper");
    std::fs::write(&bin_first, b"stub").expect("write stub");
    let found = resolve_binary_in_dirs("piper", &[first, second]);
    assert_eq!(found, Some(bin_first));
}

#[test]
fn resolve_binary_in_dirs_returns_none_when_absent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let found = resolve_binary_in_dirs("piper", &[tmp.path().to_path_buf()]);
    assert!(found.is_none(), "missing binary must resolve to None");
}
