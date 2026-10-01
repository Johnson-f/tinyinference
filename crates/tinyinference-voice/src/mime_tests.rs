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
