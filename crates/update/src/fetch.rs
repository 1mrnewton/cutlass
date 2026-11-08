//! The feed and the package over a [`Transport`].

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use cutlass_download::{
    DownloadError, Downloader, Fetcher, Job, Priority, Progress, Transport, hash_file,
};

use crate::{FEED_LIMIT, Offer, Platform, UpdateError, Version, offer, parse_checksum};

/// A small text file at `url`, refused past `limit` bytes.
pub fn get_text(transport: &dyn Transport, url: &str, limit: usize) -> Result<String, UpdateError> {
    let mut bytes = Vec::new();
    let mut over = false;
    transport
        .get(url, None, &mut |chunk| {
            if bytes.len() + chunk.len() > limit {
                over = true;
                return false;
            }
            bytes.extend_from_slice(chunk);
            true
        })
        .map_err(UpdateError::Network)?;
    if over {
        return Err(UpdateError::Feed(format!(
            "{url} is larger than {limit} bytes"
        )));
    }
    String::from_utf8(bytes).map_err(|_| UpdateError::Feed(format!("{url} is not text")))
}

/// Read the feed at `feed_url` and say whether it offers something newer
/// than `running` for `platform`.
pub fn check(
    transport: &dyn Transport,
    feed_url: &str,
    running: &Version,
    platform: Platform,
) -> Result<Option<Offer>, UpdateError> {
    let json = get_text(transport, feed_url, FEED_LIMIT)?;
    offer(&json, running, platform)
}

/// Fetch the offer's package into `folder/<version>/<name>` and check it
/// against its `.sha256`. A package already there with that checksum is
/// kept; other versions' folders are removed. `progress` gets the share
/// done; `stop` returning true pauses the download, whose pieces the next
/// try continues from.
pub fn download<T: Transport>(
    transport: T,
    offer: &Offer,
    folder: &Path,
    progress: &mut dyn FnMut(f32),
    stop: &dyn Fn() -> bool,
) -> Result<PathBuf, UpdateError> {
    let sum = get_text(&transport, &offer.checksum.url, 4096)?;
    let checksum = parse_checksum(&sum)?;
    let version = offer.version.to_string();
    let here = folder.join(&version);
    clear_others(folder, &version);
    fs::create_dir_all(&here).map_err(|error| io(&here, &error))?;
    let target = here.join(file_name(&offer.package.name)?);
    if target.is_file() {
        match hash_file(&target) {
            Ok((found, _)) if found == checksum => return Ok(target),
            _ => {
                let _ = fs::remove_file(&target);
            }
        }
    }
    let partial = here.join(format!("{}.partial", offer.package.name));
    let mut fetcher = Fetcher::new(transport);
    let id = fetcher
        .start(&Job {
            url: offer.package.url.clone(),
            destination: partial.clone(),
            checksum: Some(checksum),
            priority: Priority::Visible,
        })
        .map_err(|error| UpdateError::Network(error.to_string()))?;
    loop {
        if stop() {
            let _ = fetcher.pause(id);
            return Err(UpdateError::Stopped);
        }
        match fetcher.progress(id) {
            Ok(Progress::Partial { bytes_written }) => {
                if offer.package.size > 0 {
                    progress((bytes_written as f32 / offer.package.size as f32).min(1.0));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(Progress::Finished { .. }) => break,
            Ok(Progress::Paused { .. } | Progress::Cancelled { .. }) => {
                return Err(UpdateError::Stopped);
            }
            Err(DownloadError::Failed { message }) => return Err(UpdateError::Network(message)),
            Err(error) => return Err(UpdateError::Network(error.to_string())),
        }
    }
    drop(fetcher);
    fs::rename(&partial, &target).map_err(|error| io(&target, &error))?;
    progress(1.0);
    Ok(target)
}

/// The asset's name as one plain file name, so a hostile feed cannot write
/// outside the updates folder.
fn file_name(name: &str) -> Result<&str, UpdateError> {
    let plain =
        !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0']);
    if plain {
        Ok(name)
    } else {
        Err(UpdateError::Feed(format!("{name:?} is not a file name")))
    }
}

/// Remove the folders of other versions under `folder`.
fn clear_others(folder: &Path, keep: &str) {
    let Ok(entries) = fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() != keep && entry.path().is_dir() {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

pub(crate) fn io(path: &Path, error: &std::io::Error) -> UpdateError {
    UpdateError::Install(format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use cutlass_download::{Head, Transport};

    use super::{file_name, get_text};
    use crate::UpdateError;

    /// Serves one body in chunks, or fails.
    struct Canned {
        body: Result<Vec<u8>, String>,
        asked: Mutex<Vec<String>>,
    }

    impl Transport for Canned {
        fn head(&self, _url: &str) -> Result<Head, String> {
            Err("no".to_owned())
        }

        fn get(
            &self,
            url: &str,
            _range: Option<(u64, u64)>,
            sink: &mut dyn FnMut(&[u8]) -> bool,
        ) -> Result<(), String> {
            self.asked.lock().unwrap().push(url.to_owned());
            let body = self.body.clone()?;
            for chunk in body.chunks(7) {
                if !sink(chunk) {
                    break;
                }
            }
            Ok(())
        }
    }

    fn canned(body: Result<&[u8], &str>) -> Canned {
        Canned {
            body: body.map(<[u8]>::to_vec).map_err(str::to_owned),
            asked: Mutex::new(Vec::new()),
        }
    }

    #[test]
    fn text_is_read_whole_up_to_the_limit() {
        let transport = canned(Ok(b"{\"tag_name\":\"v1.0.0\"}"));
        assert_eq!(
            get_text(&transport, "https://feed", 100).unwrap(),
            "{\"tag_name\":\"v1.0.0\"}"
        );
        assert_eq!(transport.asked.lock().unwrap()[0], "https://feed");
        assert!(matches!(
            get_text(&transport, "https://feed", 10),
            Err(UpdateError::Feed(message)) if message.contains("larger")
        ));
        let binary = canned(Ok(&[0xff, 0xfe, 0x00]));
        assert!(matches!(
            get_text(&binary, "u", 100),
            Err(UpdateError::Feed(_))
        ));
        let offline = canned(Err("connection refused"));
        assert!(matches!(
            get_text(&offline, "u", 100),
            Err(UpdateError::Network(message)) if message == "connection refused"
        ));
    }

    #[test]
    fn a_package_name_that_leaves_the_folder_is_refused() {
        assert_eq!(file_name("Cutlass.dmg").unwrap(), "Cutlass.dmg");
        for name in [
            "",
            ".",
            "..",
            "../Cutlass.dmg",
            "a/b",
            "a\\b",
            "C:x",
            "a\0b",
        ] {
            assert!(file_name(name).is_err(), "{name:?}");
        }
    }
}
