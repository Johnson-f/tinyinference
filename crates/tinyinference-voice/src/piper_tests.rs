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
