//! Updates: the daily look at the release feed, the offer the window shows,
//! and the download and install that end in a restart.
//!
//! With the switch off in Settings nothing asks the feed. The package lands
//! in the Cutlass cache's `Updates` folder and counts only when its SHA-256
//! matches the release's `.sha256`.

use std::path::Path;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_download::{Http, Transport};
use cutlass_update::{Installed, Offer, Platform, Relaunch, UpdateError, Version};

use super::{Shell, ShellError};
use crate::jobs::Progress;

/// The version this build is.
pub const RUNNING_VERSION: &str = env!("CARGO_PKG_VERSION");

type Checked = Result<Option<Offer>, UpdateError>;

/// What the shell keeps about updates.
pub(crate) struct Updates {
    /// Where the feed is read. A test points it at a loopback server.
    feed: String,
    /// A check running, and whether the user asked for it.
    checking: Option<(mpsc::Receiver<Checked>, bool)>,
    offer: Option<Offer>,
    restart: Option<Relaunch>,
    /// Where an install goes, in place of the running app's own. Tests set it.
    installed: Option<Installed>,
}

impl Default for Updates {
    fn default() -> Self {
        Self {
            // Tests never reach GitHub: nothing listens on port 9.
            feed: if cfg!(test) {
                "http://127.0.0.1:9/feed"
            } else {
                cutlass_update::FEED_URL
            }
            .to_owned(),
            checking: None,
            offer: None,
            restart: None,
            installed: None,
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// Download `offer` into `folder` and install it over `installed`, reporting
/// to `progress`. The command to run once this process quits comes back.
pub(crate) fn fetch_and_install<T: Transport>(
    transport: T,
    offer: &Offer,
    installed: &Installed,
    folder: &Path,
    progress: &Progress,
) -> Result<Relaunch, UpdateError> {
    progress.stage(&format!("Downloading Cutlass {}", offer.version));
    let package = cutlass_update::download(
        transport,
        offer,
        folder,
        &mut |share| progress.fraction(share),
        &|| progress.cancelled(),
    )?;
    progress.stage("Installing the update");
    let relaunch =
        cutlass_update::install(installed, &package, &offer.version, std::process::id())?;
    // The Windows installer runs after the app quits; the others are in place.
    if !matches!(installed, Installed::Folder(_)) {
        let _ = std::fs::remove_file(&package);
    }
    Ok(relaunch)
}

impl Shell {
    /// At launch: look for an update when the switch is on and a day has
    /// passed since the last look.
    pub fn check_updates_at_launch(&mut self) {
        let updates = &self.settings.updates;
        if updates.check && cutlass_update::due(updates.last_check, now()) {
            self.start_update_check(false);
        }
    }

    /// The Check for updates command. With the switch off nothing is asked.
    pub fn check_for_updates(&mut self) {
        if !self.settings.updates.check {
            self.set_notice("Updates are off. Turn on Check for updates in Settings.");
            return;
        }
        if self.updates.checking.is_none() {
            self.start_update_check(true);
        }
    }

    fn start_update_check(&mut self, asked: bool) {
        let (Some(platform), Some(running)) =
            (Platform::current(), Version::parse(RUNNING_VERSION))
        else {
            if asked {
                self.set_notice("Cutlass does not update itself on this system");
            }
            return;
        };
        let started = now();
        self.update_settings(|settings| settings.updates.last_check = started);
        let feed = self.updates.feed.clone();
        let (send, receive) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("update check".to_owned())
            .spawn(move || {
                let _ = send.send(cutlass_update::check(
                    &Http::new(),
                    &feed,
                    &running,
                    platform,
                ));
            });
        match spawned {
            Ok(_) => self.updates.checking = Some((receive, asked)),
            Err(error) => cutlass_logger::warn!("update check: {error}"),
        }
    }

    /// Take a finished check: an offer for the window, or a notice when the
    /// user asked. True while one still runs.
    pub fn poll_update_check(&mut self) -> bool {
        let Some((receive, asked)) = self.updates.checking.as_ref() else {
            return false;
        };
        let asked = *asked;
        let result = match receive.try_recv() {
            Err(mpsc::TryRecvError::Empty) => return true,
            Ok(result) => result,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err(UpdateError::Network("the update check stopped".to_owned()))
            }
        };
        self.updates.checking = None;
        match result {
            Ok(Some(offer)) => {
                cutlass_logger::info!("update: Cutlass {} is available", offer.version);
                self.updates.offer = Some(offer);
            }
            Ok(None) if asked => {
                self.set_notice(format!("Cutlass {RUNNING_VERSION} is the latest version"));
            }
            Ok(None) => {}
            Err(error) => {
                cutlass_logger::warn!("update check: {error}");
                if asked {
                    self.set_notice(format!("Could not check for updates: {error}"));
                }
            }
        }
        false
    }

    /// The release the window offers, until the user answers.
    #[must_use]
    pub fn update_offer(&self) -> Option<&Offer> {
        self.updates.offer.as_ref()
    }

    /// Show `offer` as if a check had found it.
    #[cfg(test)]
    pub(crate) fn offer_update(&mut self, offer: Offer) {
        self.updates.offer = Some(offer);
    }

    /// The user said later.
    pub fn decline_update(&mut self) {
        self.updates.offer = None;
    }

    /// Download the offered release as a job and install it. Once it lands,
    /// [`Self::take_restart`] has the command that starts the new version.
    pub fn install_update(&mut self) -> Result<(), ShellError> {
        let Some(offer) = self.updates.offer.take() else {
            return Ok(());
        };
        let installed = match self.updates.installed.clone() {
            Some(installed) => installed,
            None => Platform::current()
                .ok_or_else(|| UpdateError::NotInstalled("no package for this system".to_owned()))
                .and_then(Installed::current)
                .map_err(|error| ShellError::Unavailable(error.to_string()))?,
        };
        let folder = Shell::home().updates();
        let label = format!("Update to Cutlass {}", offer.version);
        self.start_job(&label, move |progress| {
            let result = fetch_and_install(Http::new(), &offer, &installed, &folder, progress);
            Box::new(move |shell: &mut Shell| match result {
                Ok(relaunch) => {
                    shell.updates.restart = Some(relaunch);
                    Ok(())
                }
                Err(error) => Err(ShellError::Unavailable(format!(
                    "the update did not install: {error}"
                ))),
            })
        })
    }

    /// The command that starts the new version once this one quits, once an
    /// update is in place. The window starts it and closes.
    pub fn take_restart(&mut self) -> Option<Relaunch> {
        self.updates.restart.take()
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use cutlass_settings::Settings;
    use cutlass_update::{Installed, Platform, Version};

    use super::{RUNNING_VERSION, fetch_and_install};
    use crate::jobs::Progress;
    use crate::shell::Shell;

    type Routes = Vec<(String, Vec<u8>)>;

    /// A loopback server whose routes `make` builds from its own address,
    /// answering ranges; it counts the requests.
    fn serve(make: impl FnOnce(&str) -> Routes) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let routes = Arc::new(make(&base));
        let count = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&count);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let routes = Arc::clone(&routes);
                let counted = Arc::clone(&counted);
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut stream = stream;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        counted.fetch_add(1, Ordering::SeqCst);
                        let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
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
                        let body = routes
                            .iter()
                            .find(|(route, _)| *route == path)
                            .map(|(_, body)| body.clone())
                            .unwrap_or_default();
                        let (head, bytes) = match range {
                            Some((start, end)) => {
                                let end = end.min(body.len());
                                let head = format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                                     Content-Range: bytes {start}-{}/{}\r\n\r\n",
                                    end - start,
                                    end - 1,
                                    body.len()
                                );
                                (head, body[start..end].to_vec())
                            }
                            None => (
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                                    body.len()
                                ),
                                body,
                            ),
                        };
                        if stream.write_all(head.as_bytes()).is_err()
                            || stream.write_all(&bytes).is_err()
                        {
                            return;
                        }
                    }
                });
            }
        });
        (base, count)
    }

    /// `/feed` for release `tag`, its package for this platform, and the
    /// package's `.sha256`.
    fn release(base: &str, tag: &str, package: &[u8]) -> Routes {
        let name = Platform::current().unwrap().asset_name();
        let json = format!(
            r#"{{"tag_name":"{tag}","body":"Faster export\n- Fixes","assets":[
                {{"name":"{name}","size":{},"browser_download_url":"{base}/{name}"}},
                {{"name":"{name}.sha256","size":64,"browser_download_url":"{base}/{name}.sha256"}}]}}"#,
            package.len()
        );
        vec![
            ("/feed".to_owned(), json.into_bytes()),
            (format!("/{name}"), package.to_vec()),
            (
                format!("/{name}.sha256"),
                format!("{}  {name}\n", cutlass_download::Checksum::of(package)).into_bytes(),
            ),
        ]
    }

    fn new_shell(settings: Settings) -> Shell {
        let (decoders, encoders) = crate::register_codecs();
        Shell::new(decoders, encoders, settings)
    }

    fn wait(shell: &mut Shell) {
        let began = Instant::now();
        while shell.poll_update_check() {
            assert!(
                began.elapsed() < Duration::from_secs(20),
                "the check never ended"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A port nothing listens on, so a stray check fails instead of leaving
    /// the machine.
    fn nowhere(shell: &mut Shell) {
        "http://127.0.0.1:9/feed".clone_into(&mut shell.updates.feed);
    }

    #[test]
    fn with_the_switch_off_nothing_is_asked() {
        let mut settings = Settings::default();
        settings.updates.check = false;
        let mut shell = new_shell(settings);
        let (base, count) = serve(|base| release(base, "v99.0.0", b"package"));
        shell.updates.feed = format!("{base}/feed");
        shell.check_updates_at_launch();
        assert!(!shell.poll_update_check());
        shell.check_for_updates();
        assert!(!shell.poll_update_check());
        assert!(shell.notice().contains("Settings"), "{}", shell.notice());
        assert_eq!(shell.settings().updates.last_check, 0);
        assert!(shell.update_offer().is_none());
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(count.load(Ordering::SeqCst), 0, "the feed was asked");
    }

    #[test]
    fn launch_looks_at_most_once_a_day() {
        let mut settings = Settings::default();
        settings.updates.last_check = super::now() - 60;
        let mut shell = new_shell(settings);
        nowhere(&mut shell);
        shell.check_updates_at_launch();
        assert!(!shell.poll_update_check(), "it looked a minute ago");

        let mut settings = Settings::default();
        settings.updates.last_check = super::now() - 2 * 24 * 60 * 60;
        let mut shell = new_shell(settings);
        nowhere(&mut shell);
        shell.check_updates_at_launch();
        assert!(shell.settings().updates.last_check >= super::now() - 5);
        wait(&mut shell);
        // A check nobody asked for fails quietly.
        assert!(shell.notice().is_empty(), "{}", shell.notice());
        assert!(shell.update_offer().is_none());
    }

    #[test]
    fn a_newer_release_is_offered_and_the_latest_says_so() {
        let mut shell = new_shell(Settings::default());
        let (base, _) = serve(|base| release(base, "v99.0.0", b"package"));
        shell.updates.feed = format!("{base}/feed");
        shell.check_for_updates();
        wait(&mut shell);
        let offer = shell.update_offer().unwrap();
        assert_eq!(offer.version, Version::parse("99.0.0").unwrap());
        assert_eq!(offer.notes_head(4), ["Faster export", "- Fixes"]);
        shell.decline_update();
        assert!(shell.update_offer().is_none());

        let (base, _) = serve(|base| release(base, &format!("v{RUNNING_VERSION}"), b"x"));
        shell.updates.feed = format!("{base}/feed");
        shell.check_for_updates();
        wait(&mut shell);
        assert!(shell.update_offer().is_none());
        assert!(
            shell.notice().contains("is the latest version"),
            "{}",
            shell.notice()
        );

        nowhere(&mut shell);
        shell.check_for_updates();
        wait(&mut shell);
        assert!(
            shell.notice().starts_with("Could not check for updates"),
            "{}",
            shell.notice()
        );
    }

    #[test]
    fn a_bare_binary_does_not_download_an_update() {
        let mut shell = new_shell(Settings::default());
        let (base, count) = serve(|base| release(base, "v99.0.0", b"package"));
        shell.updates.feed = format!("{base}/feed");
        shell.check_for_updates();
        wait(&mut shell);
        assert!(shell.update_offer().is_some());
        let asked = count.load(Ordering::SeqCst);
        // The test binary is not an installed app.
        let error = shell.install_update().unwrap_err().to_string();
        assert!(error.contains("updates need"), "{error}");
        assert!(shell.job().is_none());
        assert_eq!(
            count.load(Ordering::SeqCst),
            asked,
            "the package was fetched"
        );
        assert!(shell.take_restart().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn the_job_downloads_checks_and_puts_the_package_in_place() {
        let dir =
            std::env::temp_dir().join(format!("cutlass-desktop-update-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("Cutlass.AppImage");
        std::fs::write(&image, b"old").unwrap();
        let package: Vec<u8> = (0..1_500_000u32).map(|index| (index % 249) as u8).collect();
        let (base, _) = serve(|base| release(base, "v99.0.0", &package));
        let running = Version::parse("0.0.1").unwrap();
        let platform = Platform::current().unwrap();
        let offer = cutlass_update::check(
            &cutlass_download::Http::new(),
            &format!("{base}/feed"),
            &running,
            platform,
        )
        .unwrap()
        .unwrap();
        let progress = Progress::default();
        let relaunch = fetch_and_install(
            cutlass_download::Http::new(),
            &offer,
            &Installed::AppImage(image.clone()),
            &dir.join("Updates"),
            &progress,
        )
        .unwrap();
        assert_eq!(std::fs::read(&image).unwrap(), package);
        assert_eq!(relaunch.args.last().unwrap(), image.as_os_str());
        assert!(
            !dir.join("Updates/99.0.0")
                .join(&offer.package.name)
                .exists(),
            "the installed package is removed"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
