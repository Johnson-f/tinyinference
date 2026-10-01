//! Workspace paths for user-supplied local voice assets (Piper binary and
//! voices). Nothing here downloads or installs anything; these helpers only
//! resolve files the user already placed on disk.

use std::path::PathBuf;

use crate::service::RuntimeConfig as Config;

use crate::models as model_ids;

/// Returns the per-user config directory (parent of config.toml).
pub fn config_root_dir(config: &Config) -> PathBuf {
    config
        .config_path
        .parent()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| config.workspace_dir.clone())
}

/// Returns the root directory under which local-AI artifacts (binaries,
/// model files) are written and resolved.
///
/// Default callers see the shared `~/.openhuman/` root, which avoids
/// duplicating multi-GB model files across users on a single machine.
///
/// When `OPENHUMAN_WORKSPACE` is **explicitly** set (test/dev parallel
/// sessions, multi-workspace deployments, isolated CI runs), the
/// shared-root contract no longer applies — those callers want full
/// isolation, including their own copy of any installed binaries. Honor
/// the override by returning the workspace dir directly.
fn shared_root_dir(config: &Config) -> PathBuf {
    if std::env::var_os("OPENHUMAN_WORKSPACE").is_some() {
        return config_root_dir(config);
    }
    if config.shared_root_dir.as_os_str().is_empty() {
        config_root_dir(config)
    } else {
        config.shared_root_dir.clone()
    }
}

pub(crate) fn workspace_local_models_dir(config: &Config) -> PathBuf {
    shared_root_dir(config).join("models").join("local-ai")
}

/// Standard Unix locations a CLI binary may live in that are **not**
/// guaranteed to be on the `PATH` a GUI app inherits. A macOS app launched
/// from Finder/Dock gets the minimal launchd `PATH`
/// (`/usr/bin:/bin:/usr/sbin:/sbin`), so Homebrew dirs (`/opt/homebrew/bin`
/// on Apple Silicon, `/usr/local/bin` on Intel) are invisible even when the
/// user installed the binary there and it runs fine from a terminal — the
/// exact symptom in issue #3425. Probe these explicitly as a last resort.
///
/// Windows resolution relies entirely on the `PATH` scan, so this is empty
/// there (the in-app installer drops its binaries into the workspace anyway).
fn standard_unix_bin_dirs() -> Vec<PathBuf> {
    if cfg!(windows) {
        return Vec::new();
    }
    [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    .iter()
    .map(PathBuf::from)
    .collect()
}

/// Return the first of `dirs` that holds `bin_name` as a regular file.
/// Shared by the `PATH` scan and the standard-dir fallback so both agree on
/// what "found" means.
fn resolve_binary_in_dirs(bin_name: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    dirs.iter()
        .map(|dir| dir.join(bin_name))
        .find(|candidate| candidate.is_file())
}

pub fn resolve_piper_binary() -> Option<PathBuf> {
    // Precedence: env override > PATH lookup > standard Unix dirs.
    if let Some(from_env) = std::env::var("PIPER_BIN")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        let path = PathBuf::from(from_env);
        if path.is_file() {
            return Some(path);
        }
    }

    let bin_name = if cfg!(windows) { "piper.exe" } else { "piper" };
    if let Some(from_path) = std::env::var_os("PATH").and_then(|path_var| {
        let dirs: Vec<PathBuf> = std::env::split_paths(&path_var).collect();
        resolve_binary_in_dirs(bin_name, &dirs)
    }) {
        return Some(from_path);
    }

    // Last resort: GUI-app PATH omits Homebrew dirs (see
    // `standard_unix_bin_dirs`). Probe them so a `brew install piper` binary
    // is found even when launched from Finder.
    if let Some(from_std) = resolve_binary_in_dirs(bin_name, &standard_unix_bin_dirs()) {
        log::debug!(
            "[voice-install:piper] resolved binary from standard dir {}",
            from_std.display()
        );
        return Some(from_std);
    }
    None
}

/// Config-aware piper resolution: an executable the user placed under the
/// workspace Piper dir first, then `PIPER_BIN`, then `PATH`.
///
/// The workspace dir is only probed, never populated: Piper is installed by
/// the user.
pub fn resolve_piper_binary_with_config(config: &Config) -> Option<PathBuf> {
    if let Some(workspace) = workspace_piper_binary_candidates(config)
        .into_iter()
        .find(|candidate| candidate.is_file() && is_executable_file(candidate))
    {
        log::debug!(
            "[voice:piper] resolved workspace binary {}",
            workspace.display()
        );
        return Some(workspace);
    }
    resolve_piper_binary()
}

/// Whether `path` carries an execute bit for anybody. A non-executable
/// workspace copy is skipped so `PIPER_BIN` / `PATH` stay reachable.
///
/// Windows has no execute bit, so every regular file qualifies there.
#[cfg(unix)]
fn is_executable_file(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(_path: &std::path::Path) -> bool {
    true
}

/// Workspace dir for Piper artifacts.
pub fn workspace_piper_dir(config: &Config) -> PathBuf {
    shared_root_dir(config).join("bin").join("piper")
}

/// On-disk paths for a Piper voice — returns the `.onnx` and
/// `.onnx.json` sidecar in that order. Returns `None` if the voice id
/// is empty (no fallback — the caller must validate up front).
pub(crate) fn workspace_piper_voice_paths(
    config: &Config,
    voice_id: &str,
) -> Option<(PathBuf, PathBuf)> {
    let trimmed = voice_id.trim();
    if trimmed.is_empty() {
        return None;
    }
    let base = workspace_piper_dir(config).join("voices").join(trimmed);
    Some((
        base.with_extension("onnx"),
        base.with_extension("onnx.json"),
    ))
}

/// All candidate paths where the workspace-installed Piper binary might
/// land. Windows zips drop `piper.exe` in a `piper/` subdir; tar.gz
/// archives on Linux/macOS sometimes flatten to the install root.
pub(crate) fn workspace_piper_binary_candidates(config: &Config) -> Vec<PathBuf> {
    let root = workspace_piper_dir(config);
    let bin_name = if cfg!(windows) { "piper.exe" } else { "piper" };
    vec![
        root.join(bin_name),
        root.join("piper").join(bin_name),
        root.join("bin").join(bin_name),
    ]
}

pub fn resolve_tts_voice_path(config: &Config) -> Result<String, String> {
    let voice_id = model_ids::effective_tts_voice_id(config);
    let path = PathBuf::from(&voice_id);
    if path.is_file() {
        return Ok(path.display().to_string());
    }
    let filename = if voice_id.ends_with(".onnx") {
        voice_id.clone()
    } else {
        format!("{voice_id}.onnx")
    };
    // Installer drop-zone — `install_piper` writes
    // `bin/piper/voices/<id>.onnx`. Probed FIRST because legacy paths
    // may contain stale stubs from earlier workspaces (a 4-byte legacy
    // stub used to win over a 63 MB installer copy and crash Piper with
    // STATUS_STACK_BUFFER_OVERRUN).
    let installer_onnx_path =
        workspace_piper_voice_paths(config, voice_id.trim_end_matches(".onnx"))
            .map(|(onnx, _)| onnx);
    if let Some(p) = &installer_onnx_path
        && p.is_file()
    {
        return Ok(p.display().to_string());
    }
    // Legacy path used by the original voice pipeline. Still checked so
    // pre-installer setups keep working.
    let legacy = workspace_local_models_dir(config)
        .join("tts")
        .join(&filename);
    if legacy.is_file() {
        return Ok(legacy.display().to_string());
    }
    let installer_display = installer_onnx_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(no installer path resolvable)".to_string());
    Err(format!(
        "TTS voice model not found. Expected '{}' (installer) or '{}' (legacy)",
        installer_display,
        legacy.display()
    ))
}

#[cfg(test)]
#[path = "paths_tests.rs"]
mod tests;
