//! The speech model on this machine: where it is looked for, and its
//! download into the Cutlass `Models` folder.
//!
//! The model is NVIDIA's Parakeet, from its Hugging Face page at a fixed
//! revision, checked against the SHA-256 that page publishes. The user is
//! asked first: it is 2.5 GB under the NVIDIA Open Model License. A stopped
//! download keeps its pieces and the next one continues from them.

use std::path::{Path, PathBuf};
use std::time::Duration;

use cutlass_download::{
    Checksum, DownloadError, Downloader, Fetcher, Http, Job, Priority, Progress as Fetched,
};

use super::{Shell, ShellError};
use crate::jobs::Progress;

/// One model file and where it comes from.
pub(crate) struct ModelFile {
    /// The folder inside `Models`.
    pub folder: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

/// The speech model the transcript and `speech.start` use.
pub(crate) const SPEECH: ModelFile = ModelFile {
    folder: "parakeet-unified-en-0.6b",
    file: "parakeet-unified-en-0.6b.nemo",
    url: "https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/\
          fe53cd885760c96b6a5f51a0bfd362cb4584a98b/parakeet-unified-en-0.6b.nemo",
    sha256: "ec23ed9150c8fde49072c3e2d61678ab903dbcef389d658db833420cbc1da35b",
    bytes: 2_474_055_680,
};

/// What the user is asked before the download starts.
pub const SPEECH_MODEL_QUESTION: &str = "Transcripts and the assistant's hearing need the \
    speech model, NVIDIA Parakeet (2.5 GB, from Hugging Face, under the NVIDIA Open Model \
    License). Download it into Cutlass's Models folder?";

/// The folder holding the speech model: the one in `Models`, or in a debug
/// build the checkout's `local-files` copy when `Models` has none.
pub(super) fn speech_model_dir() -> PathBuf {
    let installed = Shell::home().models().join(SPEECH.folder);
    if installed.join(SPEECH.file).is_file() {
        return installed;
    }
    #[cfg(debug_assertions)]
    {
        let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files")
            .join(SPEECH.folder);
        if checkout.join(SPEECH.file).is_file() {
            return checkout;
        }
    }
    installed
}

/// Whether the speech model is on this machine.
pub(super) fn speech_model_here() -> bool {
    speech_model_dir().join(SPEECH.file).is_file()
}

/// Fetch `model` into `models/<folder>/<file>`: written beside it as
/// `.partial` and renamed once every byte is in and the checksum matches.
pub(crate) fn fetch_model(
    model: &ModelFile,
    models: &Path,
    progress: &Progress,
) -> Result<PathBuf, String> {
    let folder = models.join(model.folder);
    std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let target = folder.join(model.file);
    let partial = folder.join(format!("{}.partial", model.file));
    let checksum = Checksum::parse(model.sha256).map_err(|error| error.to_string())?;
    let mut fetcher = Fetcher::new(Http::new());
    let id = fetcher
        .start(&Job {
            url: model.url.to_owned(),
            destination: partial.clone(),
            checksum: Some(checksum),
            priority: Priority::Visible,
        })
        .map_err(|error| error.to_string())?;
    loop {
        if progress.cancelled() {
            // The pieces stay; the next download continues from them.
            let _ = fetcher.pause(id);
            return Err("the download was stopped".to_owned());
        }
        match fetcher.progress(id) {
            Ok(Fetched::Partial { bytes_written }) => {
                if model.bytes > 0 {
                    progress.fraction(bytes_written as f32 / model.bytes as f32);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(Fetched::Finished { .. }) => break,
            Ok(Fetched::Paused { .. } | Fetched::Cancelled { .. }) => {
                return Err("the download stopped".to_owned());
            }
            Err(DownloadError::Failed { message }) => return Err(message),
            Err(error) => return Err(error.to_string()),
        }
    }
    std::fs::rename(&partial, &target).map_err(|error| error.to_string())?;
    Ok(target)
}

impl Shell {
    /// Download the speech model as a job. Ask the user first
    /// ([`SPEECH_MODEL_QUESTION`]).
    pub fn download_speech_model(&mut self) -> Result<(), ShellError> {
        self.wants_speech_model = false;
        if speech_model_here() {
            return Ok(());
        }
        let models = Shell::home().models();
        self.start_job("Download the speech model", move |progress| {
            progress.stage("Downloading the speech model");
            let result = fetch_model(&SPEECH, &models, progress);
            Box::new(move |shell: &mut Shell| match result {
                Ok(_) => {
                    shell.set_notice("The speech model is ready");
                    Ok(())
                }
                Err(message) => Err(ShellError::Unavailable(format!(
                    "the speech model did not download: {message}"
                ))),
            })
        })
    }

    /// Whether a transcript or hearing found no speech model, so the window
    /// should offer the download.
    #[must_use]
    pub fn wants_speech_model(&self) -> bool {
        self.wants_speech_model
    }

    /// The user said not now.
    pub fn decline_speech_model(&mut self) {
        self.wants_speech_model = false;
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    use super::{ModelFile, fetch_model};
    use crate::jobs::Progress;

    /// A one-file server that answers ranges, as Hugging Face's does.
    fn serve(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.nemo", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let body = body.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut stream = stream;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        let mut range = None;
                        loop {
                            let mut header = String::new();
                            if reader.read_line(&mut header).unwrap_or(0) == 0 {
                                return;
                            }
                            let header = header.trim_end().to_ascii_lowercase();
                            if header.is_empty() {
                                break;
                            }
                            if let Some(value) = header.strip_prefix("range: bytes=") {
                                let (start, end) = value.split_once('-').unwrap();
                                range = Some((
                                    start.parse::<usize>().unwrap(),
                                    end.parse::<usize>().unwrap() + 1,
                                ));
                            }
                        }
                        let (start, end) = range.unwrap_or((0, body.len()));
                        let end = end.min(body.len());
                        let head = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                             Content-Range: bytes {start}-{}/{}\r\n\r\n",
                            end - start,
                            end - 1,
                            body.len()
                        );
                        if stream.write_all(head.as_bytes()).is_err()
                            || stream.write_all(&body[start..end]).is_err()
                        {
                            return;
                        }
                    }
                });
            }
        });
        url
    }

    #[test]
    fn a_model_lands_in_its_folder_only_when_its_checksum_matches() {
        let body: Vec<u8> = (0..3_000_000u32).map(|index| (index % 253) as u8).collect();
        let url: &'static str = Box::leak(serve(body.clone()).into_boxed_str());
        let sha: &'static str = Box::leak(
            cutlass_download::Checksum::of(&body)
                .to_hex()
                .into_boxed_str(),
        );
        let models = std::env::temp_dir().join(format!("cutlass-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&models);
        let model = ModelFile {
            folder: "test-model",
            file: "model.nemo",
            url,
            sha256: sha,
            bytes: body.len() as u64,
        };
        let progress = Progress::default();
        let path = fetch_model(&model, &models, &progress).unwrap();
        assert_eq!(path, models.join("test-model").join("model.nemo"));
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert!(
            !models
                .join("test-model")
                .join("model.nemo.partial")
                .exists()
        );

        let wrong = ModelFile {
            file: "wrong.nemo",
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..model
        };
        let error = fetch_model(&wrong, &models, &progress).unwrap_err();
        assert!(error.contains("checksum"), "{error}");
        assert!(!models.join("test-model").join("wrong.nemo").exists());
        let _ = std::fs::remove_dir_all(&models);
    }
}
