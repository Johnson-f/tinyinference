use super::*;

#[test]
fn timeline_is_nonempty_and_scales() {
    let short = synthetic_viseme_timeline("hi");
    let long = synthetic_viseme_timeline("the quick brown fox");
    assert_eq!(short[0].viseme, "sil");
    assert!(long.last().unwrap().end_ms > short.last().unwrap().end_ms);
}

#[test]
fn explicit_output_path_and_optional_length_scale_are_preserved() {
    let args = piper_arguments(
        std::path::Path::new("voice.onnx"),
        std::path::Path::new("chosen.wav"),
        None,
    );
    let args = args
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        args,
        vec!["--model", "voice.onnx", "--output_file", "chosen.wav"]
    );

    let args = piper_arguments(
        std::path::Path::new("voice.onnx"),
        std::path::Path::new("chosen.wav"),
        Some(1.15),
    );
    assert_eq!(args[4].to_string_lossy(), "--length-scale");
    assert_eq!(args[5].to_string_lossy(), "1.15");
}

#[cfg(unix)]
fn fake_piper(directory: &std::path::Path, exit_code: i32) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let executable = directory.join("fake-piper");
    std::fs::write(
        &executable,
        format!(
            r#"#!/bin/sh
set -eu
printf '%s' "$*" > "$PIPER_ARGS_FILE"
cat > "$PIPER_STDIN_FILE"
output=''
previous=''
for argument in "$@"; do
  if [ "$previous" = "--output_file" ]; then output="$argument"; fi
  previous="$argument"
done
[ -n "$output" ]
printf 'fake wav bytes' > "$output"
printf 'fake piper error' >&2
exit {exit_code}
"#
        ),
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
    executable
}

#[cfg(unix)]
#[tokio::test]
async fn file_api_runs_piper_with_trimmed_stdin_and_preserves_caller_output() {
    let directory = tempfile::tempdir().unwrap();
    let binary = fake_piper(directory.path(), 0);
    let voice = directory.path().join("voice.onnx");
    let output = directory.path().join("chosen.wav");
    let args = directory.path().join("args.txt");
    let stdin = directory.path().join("stdin.txt");
    unsafe {
        std::env::set_var("PIPER_ARGS_FILE", &args);
        std::env::set_var("PIPER_STDIN_FILE", &stdin);
    }

    let result = synthesize_to_file(&binary, &voice, "  speak this  ", &output, None).await;

    unsafe {
        std::env::remove_var("PIPER_ARGS_FILE");
        std::env::remove_var("PIPER_STDIN_FILE");
    }
    result.unwrap();
    assert_eq!(std::fs::read_to_string(stdin).unwrap(), "speak this");
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "fake wav bytes");
    assert!(!std::fs::read_to_string(args).unwrap().contains("--length-scale"));
}

#[cfg(unix)]
#[tokio::test]
async fn legacy_api_keeps_base64_visemes_and_explicit_default_length_scale() {
    let directory = tempfile::tempdir().unwrap();
    let binary = fake_piper(directory.path(), 0);
    let voice = directory.path().join("voice.onnx");
    let args = directory.path().join("args.txt");
    let stdin = directory.path().join("stdin.txt");
    unsafe {
        std::env::set_var("PIPER_ARGS_FILE", &args);
        std::env::set_var("PIPER_STDIN_FILE", &stdin);
    }

    let speech = synthesize(&binary, &voice, " legacy text ").await;

    unsafe {
        std::env::remove_var("PIPER_ARGS_FILE");
        std::env::remove_var("PIPER_STDIN_FILE");
    }
    let speech = speech.unwrap();
    assert_eq!(speech.audio_base64, "ZmFrZSB3YXYgYnl0ZXM=");
    assert_eq!(speech.audio_mime, "audio/wav");
    assert_eq!(speech.visemes, synthetic_viseme_timeline("legacy text"));
    assert!(std::fs::read_to_string(args)
        .unwrap()
        .contains("--length-scale 1.15"));
    assert_eq!(std::fs::read_to_string(stdin).unwrap(), "legacy text");
}

#[cfg(unix)]
#[tokio::test]
async fn file_api_reports_piper_child_failure() {
    let directory = tempfile::tempdir().unwrap();
    let binary = fake_piper(directory.path(), 7);
    let output = directory.path().join("failed.wav");

    let error = synthesize_to_file(
        &binary,
        std::path::Path::new("voice.onnx"),
        "hello",
        &output,
        None,
    )
    .await
    .unwrap_err();

    assert!(error.contains("piper failed (exit=Some(7))"));
    assert!(error.contains("fake piper error"));
}
