use super::*;
use std::sync::{Arc, Mutex};

struct CaptureProvider {
    captured: Mutex<Option<(Vec<u8>, String, String)>>,
    result: Result<String, String>,
}

impl Default for CaptureProvider {
    fn default() -> Self {
        Self {
            captured: Mutex::new(None),
            result: Ok(String::new()),
        }
    }
}

#[async_trait]
impl TranscriptionProvider for CaptureProvider {
    async fn transcribe(
        &self,
        audio: &[u8],
        mime_type: &str,
        file_name: &str,
    ) -> Result<String, String> {
        *self.captured.lock().unwrap() =
            Some((audio.to_vec(), mime_type.to_string(), file_name.to_string()));
        self.result.clone()
    }
}

#[tokio::test]
async fn transcribe_file_passes_audio_mime_and_original_filename() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("sample.OPUS");
    tokio::fs::write(&path, b"voice").await.unwrap();
    let provider = CaptureProvider {
        result: Ok("recognized".to_string()),
        ..CaptureProvider::default()
    };

    let result = transcribe_file(&path, &provider).await.unwrap();

    assert_eq!(result, "recognized");
    assert_eq!(
        provider.captured.lock().unwrap().as_ref().unwrap(),
        &(
            b"voice".to_vec(),
            "audio/ogg".to_string(),
            "sample.OPUS".to_string()
        )
    );
}

#[tokio::test]
async fn transcribe_file_uses_wav_mime_for_missing_or_unknown_extensions() {
    let directory = tempfile::tempdir().unwrap();
    let no_extension = directory.path().join("sample");
    let unknown_extension = directory.path().join("sample.unknown");
    tokio::fs::write(&no_extension, b"voice").await.unwrap();
    tokio::fs::write(&unknown_extension, b"voice")
        .await
        .unwrap();
    let provider = CaptureProvider::default();

    transcribe_file(&no_extension, &provider).await.unwrap();
    assert_eq!(
        provider.captured.lock().unwrap().as_ref().unwrap(),
        &(
            b"voice".to_vec(),
            "audio/wav".to_string(),
            "sample".to_string()
        )
    );
    transcribe_file(&unknown_extension, &provider)
        .await
        .unwrap();
    assert_eq!(
        provider.captured.lock().unwrap().as_ref().unwrap(),
        &(
            b"voice".to_vec(),
            "audio/wav".to_string(),
            "sample.unknown".to_string()
        )
    );
}

#[cfg(unix)]
#[tokio::test]
async fn transcribe_file_uses_audio_wav_when_filename_is_not_utf8() {
    use std::os::unix::ffi::OsStrExt;

    let directory = tempfile::tempdir().unwrap();
    let path = directory
        .path()
        .join(std::ffi::OsStr::from_bytes(b"\xff.wav"));
    tokio::fs::write(&path, b"voice").await.unwrap();
    let provider = CaptureProvider::default();

    transcribe_file(&path, &provider).await.unwrap();

    assert_eq!(
        provider.captured.lock().unwrap().as_ref().unwrap().2,
        "audio.wav"
    );
}

#[tokio::test]
async fn transcribe_bytes_normalizes_extension_and_removes_temporary_audio() {
    let directory = tempfile::tempdir().unwrap();
    let provider = CaptureProvider {
        result: Ok("recognized".to_string()),
        ..CaptureProvider::default()
    };

    let result = transcribe_bytes_in(b"voice", Some(" .WEBM "), Some(directory.path()), &provider)
        .await
        .unwrap();

    assert_eq!(result, "recognized");
    assert_eq!(
        provider.captured.lock().unwrap().as_ref().unwrap(),
        &(
            b"voice".to_vec(),
            "audio/webm".to_string(),
            "audio.webm".to_string()
        )
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn transcribe_bytes_defaults_extension_and_cleans_up_provider_errors() {
    let directory = tempfile::tempdir().unwrap();
    let provider = CaptureProvider {
        result: Err("provider failed".to_string()),
        ..CaptureProvider::default()
    };

    assert_eq!(
        transcribe_bytes_in(b"voice", None, Some(directory.path()), &provider)
            .await
            .unwrap_err(),
        "provider failed"
    );
    assert_eq!(
        provider.captured.lock().unwrap().as_ref().unwrap().1,
        "audio/webm"
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn transcribe_bytes_rejects_empty_audio_and_traversal_extensions() {
    let provider = CaptureProvider::default();
    assert!(
        transcribe_bytes(&[], None, &provider)
            .await
            .unwrap_err()
            .contains("empty")
    );
    assert!(
        transcribe_bytes(b"voice", Some("../wav"), &provider)
            .await
            .unwrap_err()
            .contains("extension")
    );
    for extension in ["", "   ", "...", " . "] {
        assert!(
            transcribe_bytes(b"voice", Some(extension), &provider)
                .await
                .unwrap_err()
                .contains("extension"),
            "explicit extension {extension:?} should be rejected"
        );
    }
}

struct PendingProvider {
    entered: Arc<tokio::sync::Notify>,
    wait: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl TranscriptionProvider for PendingProvider {
    async fn transcribe(
        &self,
        _audio: &[u8],
        _mime_type: &str,
        _file_name: &str,
    ) -> Result<String, String> {
        self.entered.notify_one();
        self.wait.notified().await;
        Ok(String::new())
    }
}

#[tokio::test]
async fn cancelling_transcribe_bytes_removes_temporary_audio() {
    let directory = tempfile::tempdir().unwrap();
    let entered = Arc::new(tokio::sync::Notify::new());
    let provider = Arc::new(PendingProvider {
        entered: Arc::clone(&entered),
        wait: Arc::new(tokio::sync::Notify::new()),
    });
    let provider_for_task = Arc::clone(&provider);
    let directory_path = directory.path().to_owned();
    let task = tokio::spawn(async move {
        transcribe_bytes_in(
            b"voice",
            Some("wav"),
            Some(&directory_path),
            &*provider_for_task,
        )
        .await
    });

    entered.notified().await;
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[test]
fn cancelling_queued_temp_write_does_not_recreate_audio_after_cleanup() {
    use std::future::Future;
    use std::sync::mpsc;
    use std::task::Poll;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let directory_path = directory.path().to_owned();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let blocker = runtime.spawn_blocking(move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    started_rx.recv().unwrap();

    runtime.block_on(async {
        let provider = CaptureProvider::default();
        let mut future = Box::pin(transcribe_bytes_in(
            b"queued voice",
            Some("wav"),
            Some(&directory_path),
            &provider,
        ));

        // First poll creates the tempfile and queues Tokio's blocking write
        // behind the latched worker, then returns Pending.
        let pending = std::future::poll_fn(|context| {
            Poll::Ready(matches!(future.as_mut().poll(context), Poll::Pending))
        })
        .await;
        assert!(pending, "write should be queued behind the blocked worker");
        assert_eq!(std::fs::read_dir(&directory_path).unwrap().count(), 1);

        drop(future);
        release_tx.send(()).unwrap();
        blocker.await.unwrap();
        // This sentinel is submitted after the queued file write; when it
        // completes, the blocking queue has drained through the write.
        tokio::task::spawn_blocking(|| {}).await.unwrap();
    });

    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}
