//! Audio container MIME type and file-extension mapping.

/// MIME hint sent to a speech-to-text backend for a given file extension. The
/// backend forwards the blob to its STT provider, which sniffs the container;
/// a wrong hint only costs a re-sniff, so an unknown extension falls back to
/// WAV.
pub fn mime_for_extension(ext: &str) -> &'static str {
    match ext {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" | "mp4" => "audio/mp4",
        "ogg" | "opus" => "audio/ogg",
        "webm" => "audio/webm",
        "flac" => "audio/flac",
        _ => "audio/wav",
    }
}

/// File extension used when uploading audio of the given MIME type. Unknown
/// types fall back to `wav`.
pub fn extension_for_mime(mime: &str) -> &str {
    match mime {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/ogg" => "ogg",
        "audio/webm" => "webm",
        "audio/flac" => "flac",
        "audio/mp4" | "audio/m4a" => "m4a",
        _ => "wav",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_to_mime_covers_known_containers_and_defaults_to_wav() {
        assert_eq!(mime_for_extension("mp3"), "audio/mpeg");
        assert_eq!(mime_for_extension("m4a"), "audio/mp4");
        assert_eq!(mime_for_extension("mp4"), "audio/mp4");
        assert_eq!(mime_for_extension("opus"), "audio/ogg");
        assert_eq!(mime_for_extension("webm"), "audio/webm");
        assert_eq!(mime_for_extension("flac"), "audio/flac");
        assert_eq!(mime_for_extension("xyz"), "audio/wav");
    }

    #[test]
    fn mime_to_extension_covers_known_types_and_defaults_to_wav() {
        assert_eq!(extension_for_mime("audio/x-wav"), "wav");
        assert_eq!(extension_for_mime("audio/mp3"), "mp3");
        assert_eq!(extension_for_mime("audio/m4a"), "m4a");
        assert_eq!(extension_for_mime("audio/unknown"), "wav");
    }
}
