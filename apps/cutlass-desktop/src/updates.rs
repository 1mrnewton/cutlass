//! Launch-screen update nudge: the Rust half of `UpdateBackend`.
//!
//! Anonymous `CloudClient::latest_version` on a worker thread — never on
//! the UI thread. Best-effort; failures are silent.

use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::CloudClient;
use slint::ComponentHandle;
use tracing::{info, warn};

use crate::UpdateBackend;

enum Command {
    Check,
    OpenUpdate,
}

#[derive(Clone)]
pub struct UpdatesHandle {
    tx: Sender<Command>,
}

impl UpdatesHandle {
    pub fn check(&self) {
        let _ = self.tx.send(Command::Check);
    }

    pub fn open_update(&self) {
        let _ = self.tx.send(Command::OpenUpdate);
    }
}

pub struct UpdatesWorker {
    handle: UpdatesHandle,
    _join: JoinHandle<()>,
}

impl UpdatesWorker {
    pub fn spawn(backend_weak: slint::Weak<crate::AppWindow>) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-updates".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak);
                while let Ok(command) = rx.recv() {
                    worker.run(command);
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            handle: UpdatesHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> UpdatesHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    update_url: String,
}

impl Worker {
    fn new(backend_weak: slint::Weak<crate::AppWindow>) -> Self {
        Self {
            backend_weak,
            update_url: String::new(),
        }
    }

    fn run(&mut self, command: Command) {
        match command {
            Command::Check => self.check_for_update(),
            Command::OpenUpdate => {
                if !self.update_url.is_empty() {
                    open_in_browser(&self.update_url);
                }
            }
        }
    }

    fn check_for_update(&mut self) {
        let client = CloudClient::new(&crate::cloud::base_url(), None);
        let latest = match client.latest_version() {
            Ok(latest) => latest,
            Err(e) => {
                info!("update check skipped: {e}");
                return;
            }
        };
        if !version_is_newer(&latest.version, env!("CARGO_PKG_VERSION")) {
            return;
        }
        info!(
            "update available: {} (running {})",
            latest.version,
            env!("CARGO_PKG_VERSION")
        );
        self.update_url = latest.download_url.clone();
        let version = latest.version.clone();
        self.publish(move |b| {
            b.set_update_available(true);
            b.set_update_version(version.as_str().into());
        });
    }

    fn publish(&self, f: impl FnOnce(UpdateBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<UpdateBackend>());
            }
        }) {
            warn!("update UI update failed: {e}");
        }
    }
}

/// `major.minor.patch` triple, ignoring any pre-release suffix
/// (`0.5.3-alpha.0` → `(0, 5, 3)`).
fn version_triple(version: &str) -> Option<(u64, u64, u64)> {
    let core = version.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

/// Whether `remote` is a strictly newer release than `local`. Unparseable
/// versions never nudge (a bad catalog entry must not spam every user).
fn version_is_newer(remote: &str, local: &str) -> bool {
    match (version_triple(remote), version_triple(local)) {
        (Some(r), Some(l)) => r > l,
        _ => false,
    }
}

fn open_in_browser(url: &str) {
    if let Err(error) = crate::external::open_web_url(url) {
        warn!("failed to open browser: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_triples() {
        assert_eq!(version_triple("0.5.3-alpha.0"), Some((0, 5, 3)));
        assert_eq!(version_triple("1.2.3"), Some((1, 2, 3)));
        assert_eq!(version_triple("1.2"), Some((1, 2, 0)));
        assert_eq!(version_triple("nope"), None);
    }

    #[test]
    fn newer_version_detection() {
        assert!(version_is_newer("0.6.0", "0.5.3-alpha.0"));
        assert!(version_is_newer("0.5.4", "0.5.3"));
        assert!(
            !version_is_newer("0.5.3", "0.5.3-alpha.0"),
            "same triple never nudges"
        );
        assert!(!version_is_newer("0.5.2", "0.5.3"));
        assert!(!version_is_newer("garbage", "0.5.3"));
    }
}
