//! Cutlass account: the Rust half of `AccountBackend`.
//!
//! One worker thread owns everything network-flavored about the account —
//! OAuth sign-in (system browser + loopback redirect), token refresh,
//! balance/pack fetches, Polar checkout hand-off, and the startup update
//! check. The session token lives in the OS keychain
//! (`cutlass_cloud::token_store`); nothing here writes secrets to disk.
//!
//! Threading mirrors `cloud.rs`: commands in over a channel, results
//! hopped to the UI thread with `invoke_from_event_loop`.

use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::dto::CreditPack;
use cutlass_cloud::token_store::{self, StoredSession};
use cutlass_cloud::{CloudClient, auth};
use slint::{ComponentHandle, ModelRc, VecModel};
use tracing::{info, warn};

use crate::{AccountBackend, CreditPackRow};

/// How long the loopback listener waits for the browser redirect before
/// giving up (the user may need to complete 2FA at the provider).
const SIGN_IN_TIMEOUT: Duration = Duration::from_secs(300);

enum Command {
    /// Startup: restore the keychain session (refreshing if stale) and run
    /// the update check.
    Init,
    SignIn {
        provider: String,
    },
    SignOut,
    RefreshBalance,
    BuyPack {
        index: usize,
    },
    OpenUpdate,
}

#[derive(Clone)]
pub struct AccountHandle {
    tx: Sender<Command>,
}

impl AccountHandle {
    pub fn init(&self) {
        let _ = self.tx.send(Command::Init);
    }

    pub fn sign_in(&self, provider: String) {
        let _ = self.tx.send(Command::SignIn { provider });
    }

    pub fn sign_out(&self) {
        let _ = self.tx.send(Command::SignOut);
    }

    pub fn refresh_balance(&self) {
        let _ = self.tx.send(Command::RefreshBalance);
    }

    pub fn buy_pack(&self, index: usize) {
        let _ = self.tx.send(Command::BuyPack { index });
    }

    pub fn open_update(&self) {
        let _ = self.tx.send(Command::OpenUpdate);
    }
}

pub struct AccountWorker {
    handle: AccountHandle,
    _join: JoinHandle<()>,
}

impl AccountWorker {
    pub fn spawn(backend_weak: slint::Weak<crate::AppWindow>) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-account".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak);
                while let Ok(command) = rx.recv() {
                    worker.run(command);
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            handle: AccountHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> AccountHandle {
        self.handle.clone()
    }
}

/// Backend base URL: `[account] base_url` in config.toml, then the
/// `CUTLASS_API_BASE` env override, then production.
pub fn base_url() -> String {
    let from_settings = cutlass_settings::load(&cutlass_settings::default_config_path())
        .map(|s| s.account.base_url)
        .unwrap_or_default();
    if !from_settings.is_empty() {
        return from_settings;
    }
    std::env::var("CUTLASS_API_BASE")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| cutlass_cloud::DEFAULT_BASE_URL.to_string())
}

/// A fresh access token from the keychain session, refreshing (and
/// re-storing) when stale. The shared entry point for every surface that
/// talks to the backend as the signed-in user from its own thread (the
/// managed chat provider, generation fallbacks).
pub fn managed_access_token() -> Result<String, String> {
    let mut session =
        token_store::load().ok_or("Not signed in — sign in under Settings > Account.")?;
    if session.needs_refresh() {
        let pair = auth::refresh(&base_url(), &session.refresh_token)
            .map_err(|e| format!("Session expired — sign in again. ({e})"))?;
        session = StoredSession::from_pair(&pair);
        if let Err(e) = token_store::store(&session) {
            warn!("keychain store after refresh failed: {e}");
        }
    }
    Ok(session.access_token)
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    base_url: String,
    session: Option<StoredSession>,
    packs: Vec<CreditPack>,
    update_url: String,
}

impl Worker {
    fn new(backend_weak: slint::Weak<crate::AppWindow>) -> Self {
        Self {
            backend_weak,
            base_url: base_url(),
            session: None,
            packs: Vec::new(),
            update_url: String::new(),
        }
    }

    fn run(&mut self, command: Command) {
        match command {
            Command::Init => {
                self.restore_session();
                self.check_for_update();
            }
            Command::SignIn { provider } => self.sign_in(&provider),
            Command::SignOut => self.sign_out(),
            Command::RefreshBalance => self.fetch_account_state(),
            Command::BuyPack { index } => self.buy_pack(index),
            Command::OpenUpdate => {
                if !self.update_url.is_empty() {
                    open_in_browser(&self.update_url);
                }
            }
        }
    }

    // --- session lifecycle ------------------------------------------------

    fn restore_session(&mut self) {
        let Some(mut session) = token_store::load() else {
            return;
        };
        if session.needs_refresh() {
            match auth::refresh(&self.base_url, &session.refresh_token) {
                Ok(pair) => {
                    session = StoredSession::from_pair(&pair);
                    if let Err(e) = token_store::store(&session) {
                        warn!("keychain store after refresh failed: {e}");
                    }
                }
                Err(e) => {
                    // A dead refresh token means the session is over; a
                    // network blip means try again later — either way the
                    // safe UI state is signed-out (the token stays in the
                    // keychain for the next launch unless it was rejected).
                    if matches!(e, cutlass_cloud::CloudError::Status { .. }) {
                        info!("stored session rejected, signing out: {e}");
                        token_store::clear();
                    } else {
                        warn!("session refresh failed (offline?): {e}");
                    }
                    return;
                }
            }
        }
        self.session = Some(session);
        self.fetch_account_state();
    }

    fn sign_in(&mut self, provider: &str) {
        self.publish(|b| {
            b.set_status("signing-in".into());
            b.set_error("".into());
        });
        let result = auth::start_sign_in(&self.base_url, provider)
            .and_then(|(authorize_url, pending)| {
                open_in_browser(&authorize_url);
                pending.wait(SIGN_IN_TIMEOUT)
            })
            .map(|pair| StoredSession::from_pair(&pair));
        match result {
            Ok(session) => {
                if let Err(e) = token_store::store(&session) {
                    warn!("keychain store failed (session won't survive restart): {e}");
                }
                self.session = Some(session);
                info!("signed in via {provider}");
                self.fetch_account_state();
            }
            Err(e) => {
                warn!("sign-in failed: {e}");
                let message = sign_in_error_message(&e);
                self.publish(move |b| {
                    b.set_status("signed-out".into());
                    b.set_error(message.as_str().into());
                });
            }
        }
    }

    fn sign_out(&mut self) {
        if let Some(session) = self.session.take() {
            // Server revocation is best-effort; the keychain wipe is what
            // actually signs out.
            if let Err(e) = auth::sign_out(&self.base_url, &session.refresh_token) {
                warn!("server sign-out failed (token revoked locally anyway): {e}");
            }
        }
        token_store::clear();
        self.packs.clear();
        self.publish(|b| {
            b.set_status("signed-out".into());
            b.set_email("".into());
            b.set_provider("".into());
            b.set_credits(0);
            b.set_balance_known(false);
            b.set_packs(ModelRc::default());
            b.set_error("".into());
        });
    }

    /// Refresh the access token if stale, then return a client for the
    /// account routes. `None` means signed out.
    fn authed_client(&mut self) -> Option<auth::AuthedClient> {
        let session = self.session.as_mut()?;
        if session.needs_refresh() {
            match auth::refresh(&self.base_url, &session.refresh_token) {
                Ok(pair) => {
                    *session = StoredSession::from_pair(&pair);
                    if let Err(e) = token_store::store(session) {
                        warn!("keychain store after refresh failed: {e}");
                    }
                }
                Err(e) => {
                    warn!("token refresh failed: {e}");
                    return None;
                }
            }
        }
        Some(auth::AuthedClient::new(
            &self.base_url,
            &session.access_token,
        ))
    }

    // --- account state (identity + balance + packs) ------------------------

    fn fetch_account_state(&mut self) {
        let Some(client) = self.authed_client() else {
            return;
        };
        let me = match client.me() {
            Ok(me) => me,
            Err(e) => {
                warn!("GET /v1/me failed: {e}");
                let message = format!("Couldn't load the account: {e}");
                self.publish(move |b| b.set_error(message.as_str().into()));
                return;
            }
        };
        let balance = client.balance();
        let packs = client.packs().map(|p| p.packs).unwrap_or_default();
        self.packs = packs.clone();

        let email = if me.email.is_empty() {
            me.display_name.clone()
        } else {
            me.email.clone()
        };
        let provider = me.provider.clone();
        self.publish(move |b| {
            b.set_status("signed-in".into());
            b.set_email(email.as_str().into());
            b.set_provider(provider.as_str().into());
            b.set_error("".into());
            match &balance {
                Ok(balance) => {
                    b.set_credits(balance.credits.min(i32::MAX as i64) as i32);
                    b.set_balance_known(true);
                }
                Err(_) => b.set_balance_known(false),
            }
            let rows: Vec<CreditPackRow> = packs
                .iter()
                .map(|p| CreditPackRow {
                    id: p.id.as_str().into(),
                    label: format!("{} — {} credits · {}", p.name, p.credits, p.price_display)
                        .as_str()
                        .into(),
                })
                .collect();
            b.set_packs(ModelRc::new(VecModel::from(rows)));
        });
    }

    fn buy_pack(&mut self, index: usize) {
        let Some(pack) = self.packs.get(index).cloned() else {
            return;
        };
        let Some(client) = self.authed_client() else {
            return;
        };
        match client.checkout(&pack.id) {
            Ok(response) => {
                info!("opening Polar checkout for pack {}", pack.id);
                open_in_browser(&response.checkout_url);
            }
            Err(e) => {
                warn!("checkout failed: {e}");
                let message = format!("Couldn't start checkout: {e}");
                self.publish(move |b| b.set_error(message.as_str().into()));
            }
        }
    }

    // --- update nudge -------------------------------------------------------

    fn check_for_update(&mut self) {
        let client = CloudClient::new(&self.base_url, None);
        let latest = match client.latest_version() {
            Ok(latest) => latest,
            // Silent: the nudge is best-effort and most launches are offline
            // from the backend's point of view during development.
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

    // --- UI publishing ------------------------------------------------------

    fn publish(&self, f: impl FnOnce(AccountBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<AccountBackend>());
            }
        }) {
            warn!("account UI update failed: {e}");
        }
    }
}

fn sign_in_error_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Cancelled => "Sign-in timed out — try again.".into(),
        CloudError::Network(_) => {
            "Couldn't reach the Cutlass service — check your connection.".into()
        }
        CloudError::Status { status, .. } => format!("Sign-in was rejected ({status})."),
        _ => "Sign-in failed — try again.".into(),
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

/// Open a URL in the default browser, off the UI thread. The URL is
/// always one of ours or one the backend vouched for (authorize URL,
/// checkout URL, download page).
fn open_in_browser(url: &str) {
    let spawn = |program: &str, args: &[&str]| {
        if let Err(e) = std::process::Command::new(program).args(args).spawn() {
            warn!("failed to open browser: {e}");
        }
    };
    #[cfg(target_os = "macos")]
    spawn("open", &[url]);
    #[cfg(target_os = "windows")]
    spawn("cmd", &["/C", "start", "", url]);
    #[cfg(all(unix, not(target_os = "macos")))]
    spawn("xdg-open", &[url]);
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
