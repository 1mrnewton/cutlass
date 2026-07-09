//! The authed half of the cloud client: OAuth sign-in (PKCE + loopback
//! redirect), token refresh, and the account routes (identity, balance,
//! ledger, credit packs, Polar checkout).
//!
//! Sign-in shape (the OAuth-only decision): the app never sees provider
//! passwords. It generates a PKCE verifier, asks the backend for the
//! provider authorize URL, opens it in the **system browser**, and catches
//! the redirect on a localhost listener; the backend swaps the code for a
//! [`TokenPair`] (short-lived access JWT + rotating refresh token) which
//! the desktop stores in the OS keychain ([`crate::token_store`]) — never
//! in a file.
//!
//! Everything here blocks and belongs on a worker thread, like the rest of
//! the crate.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::time::Duration;

use crate::dto::{
    Balance, CheckoutRequest, CheckoutResponse, LedgerPage, Me, OauthCallbackRequest,
    OauthStartRequest, OauthStartResponse, PacksResponse, RefreshRequest, TokenPair,
};
use crate::error::CloudError;

/// URL-safe base64 without padding (RFC 4648 §5) — PKCE's encoding.
fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[n as usize & 63] as char);
        }
    }
    out
}

/// A fresh PKCE pair: `(verifier, S256 challenge)`.
fn pkce_pair() -> Result<(String, String), CloudError> {
    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed)
        .map_err(|e| CloudError::Protocol(format!("PKCE entropy unavailable: {e}")))?;
    let verifier = base64url(&seed);
    let challenge = base64url(&crate::download::sha256(verifier.as_bytes()));
    Ok((verifier, challenge))
}

/// A sign-in in flight: the browser is open on the provider's consent page
/// and the loopback listener is waiting for the redirect.
pub struct PendingSignIn {
    listener: TcpListener,
    verifier: String,
    state: String,
    base_url: String,
    agent: ureq::Agent,
}

/// Start an OAuth sign-in: binds the loopback listener, asks the backend
/// for the provider authorize URL, and returns it with the pending flow.
/// The caller opens the URL in the system browser, then calls
/// [`PendingSignIn::wait`] (on a worker thread — it blocks).
pub fn start_sign_in(
    base_url: &str,
    provider: &str,
) -> Result<(String, PendingSignIn), CloudError> {
    let base_url = base_url.trim_end_matches('/').to_string();
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| CloudError::Io(std::io::Error::other(format!("loopback listener: {e}"))))?;
    let port = listener
        .local_addr()
        .map_err(|e| CloudError::Io(std::io::Error::other(format!("loopback listener: {e}"))))?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");
    let (verifier, challenge) = pkce_pair()?;

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build();
    let url = format!("{base_url}/v1/auth/oauth/start");
    let response = agent
        .post(&url)
        .send_json(
            serde_json::to_value(OauthStartRequest {
                provider: provider.to_string(),
                code_challenge: challenge,
                redirect_uri,
            })
            .expect("start request serializes"),
        )
        .map_err(|e| CloudError::from_ureq(&url, e))?;
    let start: OauthStartResponse = response
        .into_json()
        .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))?;

    Ok((
        start.authorize_url,
        PendingSignIn {
            listener,
            verifier,
            state: start.state,
            base_url,
            agent,
        },
    ))
}

impl PendingSignIn {
    /// Block until the provider redirects back (or `timeout` elapses), then
    /// swap the code for tokens at the backend. The browser tab gets a tiny
    /// "return to Cutlass" page either way.
    pub fn wait(self, timeout: Duration) -> Result<TokenPair, CloudError> {
        self.listener.set_nonblocking(false).map_err(|e| {
            CloudError::Io(std::io::Error::other(format!("loopback listener: {e}")))
        })?;
        // A deadline via read timeout on accept isn't portable; poll accept
        // in nonblocking mode instead so a user who closes the browser tab
        // doesn't hang the worker forever.
        self.listener.set_nonblocking(true).map_err(|e| {
            CloudError::Io(std::io::Error::other(format!("loopback listener: {e}")))
        })?;
        let deadline = std::time::Instant::now() + timeout;
        let (mut stream, code, returned_state) = loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let mut stream = stream;
                    stream.set_nonblocking(false).map_err(|e| {
                        CloudError::Io(std::io::Error::other(format!("callback stream: {e}")))
                    })?;
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                    match read_callback_params(&mut stream) {
                        Some((code, state)) => break (stream, code, state),
                        // A stray request (favicon probe); answer and keep
                        // waiting for the real callback.
                        None => {
                            let _ = respond_html(&mut stream, "Waiting for sign-in…");
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return Err(CloudError::Cancelled);
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => {
                    return Err(CloudError::Io(std::io::Error::other(format!(
                        "callback accept: {e}"
                    ))));
                }
            }
        };

        if returned_state != self.state {
            let _ = respond_html(&mut stream, "Sign-in failed — please try again.");
            return Err(CloudError::Protocol("OAuth state mismatch".into()));
        }
        let _ = respond_html(
            &mut stream,
            "You're signed in — you can close this tab and return to Cutlass.",
        );

        let url = format!("{}/v1/auth/oauth/callback", self.base_url);
        let response = self
            .agent
            .post(&url)
            .send_json(
                serde_json::to_value(OauthCallbackRequest {
                    state: self.state,
                    code,
                    code_verifier: self.verifier,
                })
                .expect("callback request serializes"),
            )
            .map_err(|e| CloudError::from_ureq(&url, e))?;
        response
            .into_json()
            .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))
    }
}

/// Parse `GET /callback?code=…&state=…` off the socket; `None` when the
/// request isn't the callback (wrong path or missing params).
fn read_callback_params(stream: &mut std::net::TcpStream) -> Option<(String, String)> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    // "GET /callback?code=x&state=y HTTP/1.1"
    let path = request_line.split_whitespace().nth(1)?;
    let query = path.strip_prefix("/callback?")?;
    let mut code = None;
    let mut state = None;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=')?;
        match k {
            "code" => code = Some(percent_decode(v)),
            "state" => state = Some(percent_decode(v)),
            _ => {}
        }
    }
    Some((code?, state?))
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                if let Some(v) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    out.push(v);
                    i += 3;
                    continue;
                }
                out.push(b'%');
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn respond_html(stream: &mut std::net::TcpStream, message: &str) -> std::io::Result<()> {
    let body = format!(
        "<!doctype html><meta charset=\"utf-8\"><title>Cutlass</title>\
         <body style=\"font-family:system-ui;background:#141414;color:#eee;\
         display:grid;place-items:center;height:100vh;margin:0\">\
         <p>{message}</p></body>"
    );
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

// ---------------------------------------------------------------------------
// Authed client
// ---------------------------------------------------------------------------

/// Blocking client over the backend's account routes; every request sends
/// the bearer access token.
pub struct AuthedClient {
    base_url: String,
    agent: ureq::Agent,
    access_token: String,
}

impl AuthedClient {
    pub fn new(base_url: &str, access_token: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout(Duration::from_secs(30))
                .build(),
            access_token: access_token.to_string(),
        }
    }

    fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, CloudError> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .agent
            .get(&url)
            .set("Authorization", &format!("Bearer {}", self.access_token))
            .call()
            .map_err(|e| CloudError::from_ureq(&url, e))?;
        response
            .into_json()
            .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))
    }

    fn post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<T, CloudError> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.access_token))
            .send_json(body)
            .map_err(|e| CloudError::from_ureq(&url, e))?;
        response
            .into_json()
            .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))
    }

    /// `GET /v1/me`.
    pub fn me(&self) -> Result<Me, CloudError> {
        self.get("/v1/me")
    }

    /// `GET /v1/credits/balance`.
    pub fn balance(&self) -> Result<Balance, CloudError> {
        self.get("/v1/credits/balance")
    }

    /// `GET /v1/credits/packs` — the purchasable credit packs.
    pub fn packs(&self) -> Result<PacksResponse, CloudError> {
        self.get("/v1/credits/packs")
    }

    /// `POST /v1/credits/checkout` — a Polar checkout URL for `pack_id`,
    /// to open in the system browser.
    pub fn checkout(&self, pack_id: &str) -> Result<CheckoutResponse, CloudError> {
        self.post(
            "/v1/credits/checkout",
            serde_json::to_value(CheckoutRequest {
                pack_id: pack_id.to_string(),
            })
            .expect("checkout request serializes"),
        )
    }

    /// `GET /v1/credits/history`.
    pub fn history(&self, cursor: Option<&str>) -> Result<LedgerPage, CloudError> {
        match cursor {
            Some(cursor) => self.get(&format!("/v1/credits/history?cursor={cursor}")),
            None => self.get("/v1/credits/history"),
        }
    }

    /// `POST /v1/generate/{image|video|tts}` — start a managed job. A 402
    /// surfaces as [`CloudError::Status`] (out of credits / spend cap).
    pub fn generate(
        &self,
        kind: &str,
        request: &crate::dto::GenerateRequest,
    ) -> Result<crate::dto::Job, CloudError> {
        self.post(
            &format!("/v1/generate/{kind}"),
            serde_json::to_value(request).expect("generate request serializes"),
        )
    }

    /// `GET /v1/jobs/{id}` — poll a managed job.
    pub fn job(&self, id: &str) -> Result<crate::dto::Job, CloudError> {
        self.get(&format!("/v1/jobs/{id}"))
    }

    /// The bearer token, for surfaces that speak to the backend through
    /// other clients (the managed chat provider in `cutlass-ai`).
    pub fn access_token(&self) -> &str {
        &self.access_token
    }
}

/// `POST /v1/auth/refresh` — swap a refresh token for a fresh pair (the
/// old refresh token is invalidated server-side).
pub fn refresh(base_url: &str, refresh_token: &str) -> Result<TokenPair, CloudError> {
    let base_url = base_url.trim_end_matches('/');
    let url = format!("{base_url}/v1/auth/refresh");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build();
    let response = agent
        .post(&url)
        .send_json(
            serde_json::to_value(RefreshRequest {
                refresh_token: refresh_token.to_string(),
            })
            .expect("refresh request serializes"),
        )
        .map_err(|e| CloudError::from_ureq(&url, e))?;
    response
        .into_json()
        .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))
}

/// `POST /v1/auth/signout` — revoke the refresh token server-side.
/// Best-effort: the local keychain wipe is what actually signs out.
pub fn sign_out(base_url: &str, refresh_token: &str) -> Result<(), CloudError> {
    let base_url = base_url.trim_end_matches('/');
    let url = format!("{base_url}/v1/auth/signout");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(Duration::from_secs(15))
        .build();
    agent
        .post(&url)
        .send_json(
            serde_json::to_value(RefreshRequest {
                refresh_token: refresh_token.to_string(),
            })
            .expect("signout request serializes"),
        )
        .map_err(|e| CloudError::from_ureq(&url, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64url_matches_rfc_vectors() {
        // RFC 4648 §10 vectors, translated to the URL-safe alphabet.
        assert_eq!(base64url(b""), "");
        assert_eq!(base64url(b"f"), "Zg");
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foo"), "Zm9v");
        assert_eq!(base64url(b"foobar"), "Zm9vYmFy");
        // URL-safe characters, no padding.
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
    }

    #[test]
    fn pkce_challenge_matches_rfc_7636_appendix_b() {
        // Appendix B fixes the verifier and expected S256 challenge.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let challenge = base64url(&crate::download::sha256(verifier.as_bytes()));
        assert_eq!(challenge, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn pkce_pairs_are_unique() {
        let (v1, c1) = pkce_pair().unwrap();
        let (v2, c2) = pkce_pair().unwrap();
        assert_ne!(v1, v2);
        assert_ne!(c1, c2);
        assert_eq!(v1.len(), 43, "32 bytes base64url = 43 chars");
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("a%2Fb+c"), "a/b c");
        assert_eq!(percent_decode("plain"), "plain");
        // Malformed escapes pass through rather than panicking.
        assert_eq!(percent_decode("bad%zz"), "bad%zz");
        assert_eq!(percent_decode("tail%2"), "tail%2");
    }

    #[test]
    fn callback_parser_extracts_code_and_state() {
        // Simulate the request line off a socketless reader by testing the
        // pure parts: path parsing via a real loopback pair.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_callback_params(&mut stream)
        });
        let mut client = std::net::TcpStream::connect(addr).unwrap();
        write!(
            client,
            "GET /callback?code=abc%2F1&state=xyz HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        .unwrap();
        let parsed = join.join().unwrap();
        assert_eq!(parsed, Some(("abc/1".to_string(), "xyz".to_string())));
    }
}
