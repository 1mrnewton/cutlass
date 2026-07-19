//! The typed HTTP client for `cutlass-backend`'s anonymous surface.
//!
//! Auth-requiring routes (credits, generation) join in the accounts
//! workstream; nothing here sends a token. Catalog responses are
//! ETag-cached on disk so Library sections repaint instantly offline and
//! revalidate in the background (the "editor never blocks on the backend"
//! principle — callers still run this on worker threads).

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::dto::{CatalogResponse, LatestVersion, StockKind, StockSearchResponse};
use crate::error::CloudError;

/// Blocking client over the backend's anonymous routes.
pub struct CloudClient {
    base_url: String,
    agent: ureq::Agent,
    /// Where ETag-validated catalog responses persist; `None` disables
    /// disk caching (tests).
    cache_dir: Option<PathBuf>,
}

impl CloudClient {
    /// `base_url` without a trailing slash, e.g. `https://api.cutlass.sh`.
    /// `cache_dir` is the app-data catalog cache (created on demand).
    pub fn new(base_url: &str, cache_dir: Option<PathBuf>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout(Duration::from_secs(30))
                .build(),
            cache_dir,
        }
    }

    /// `GET /v1/app/latest-version` — the launch-screen update nudge.
    /// Never cached (it exists to notice new releases).
    pub fn latest_version(&self) -> Result<LatestVersion, CloudError> {
        let url = format!("{}/v1/app/latest-version", self.base_url);
        let response = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| CloudError::from_ureq(&url, e))?;
        parse_json(&url, response)
    }

    /// `GET /v1/stock/search` — normalized stock metadata. Not disk-cached
    /// (queries are ad hoc; the server holds the shared response cache).
    pub fn stock_search(
        &self,
        query: &str,
        kind: StockKind,
        page: u32,
    ) -> Result<StockSearchResponse, CloudError> {
        let url = format!("{}/v1/stock/search", self.base_url);
        let page_string = page.to_string();
        let response = self
            .agent
            .get(&url)
            .query("q", query)
            .query("kind", kind.as_str())
            .query("page", &page_string)
            .call()
            .map_err(|e| CloudError::from_ureq(&url, e))?;
        parse_json(&url, response)
    }

    /// `GET /v1/templates` — the template gallery.
    pub fn templates(&self) -> Result<CatalogResponse, CloudError> {
        self.catalog("/v1/templates", "templates.json")
    }

    /// `GET /v1/assets/text-presets`.
    pub fn text_presets(&self) -> Result<CatalogResponse, CloudError> {
        self.catalog("/v1/assets/text-presets", "text-presets.json")
    }

    /// `GET /v1/assets/sfx`.
    pub fn sfx(&self) -> Result<CatalogResponse, CloudError> {
        self.catalog("/v1/assets/sfx", "sfx.json")
    }

    /// `GET /v1/assets/luts`.
    pub fn luts(&self) -> Result<CatalogResponse, CloudError> {
        self.catalog("/v1/assets/luts", "luts.json")
    }

    /// `GET /v1/assets/lottie` — file-backed Lottie animations.
    pub fn lottie(&self) -> Result<CatalogResponse, CloudError> {
        self.catalog("/v1/assets/lottie", "lottie.json")
    }

    /// `GET /v1/assets/skills` — agent skill packs.
    pub fn skills(&self) -> Result<CatalogResponse, CloudError> {
        self.catalog("/v1/assets/skills", "skills.json")
    }

    /// A catalog GET with ETag revalidation against the disk cache: 304
    /// (or any network failure with a cache on disk) serves the cached
    /// body, so browsing keeps working offline with stale-but-usable data.
    fn catalog(&self, path: &str, cache_name: &str) -> Result<CatalogResponse, CloudError> {
        let url = format!("{}{}", self.base_url, path);
        let cached = self.read_cache(cache_name);

        let mut request = self.agent.get(&url);
        if let Some((etag, _)) = &cached
            && !etag.is_empty()
        {
            request = request.set("If-None-Match", etag);
        }

        match request.call() {
            Ok(response) if response.status() == 304 => {
                let (_, body) = cached.expect("304 implies a cached ETag was sent");
                serde_json::from_str(&body)
                    .map_err(|e| CloudError::Protocol(format!("{url}: stale cache: {e}")))
            }
            Ok(response) => {
                let etag = response.header("ETag").unwrap_or_default().to_string();
                let body = response
                    .into_string()
                    .map_err(|e| CloudError::Network(format!("{url}: reading body: {e}")))?;
                let parsed: CatalogResponse = serde_json::from_str(&body)
                    .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))?;
                self.write_cache(cache_name, &etag, &body);
                Ok(parsed)
            }
            Err(err) => {
                // Offline (or server trouble) with a cache on disk: serve
                // stale rather than blank out the Library section.
                if let Some((_, body)) = cached
                    && let Ok(parsed) = serde_json::from_str(&body)
                {
                    tracing::debug!("serving stale catalog {cache_name} after: {err}");
                    return Ok(parsed);
                }
                Err(CloudError::from_ureq(&url, err))
            }
        }
    }

    /// Cached (etag, body) for a catalog, if present and well-formed.
    /// Format: first line the ETag (may be empty), rest the JSON body.
    fn read_cache(&self, name: &str) -> Option<(String, String)> {
        let path = self.cache_path(name)?;
        let raw = std::fs::read_to_string(path).ok()?;
        let (etag, body) = raw.split_once('\n')?;
        Some((etag.to_string(), body.to_string()))
    }

    fn write_cache(&self, name: &str, etag: &str, body: &str) {
        let Some(path) = self.cache_path(name) else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Single-line ETag header then the body; atomic rename so a crash
        // can't leave a torn cache a later session would trust.
        let tmp = path.with_extension("tmp");
        let content = format!("{}\n{}", etag.replace('\n', ""), body);
        if std::fs::write(&tmp, content).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }

    fn cache_path(&self, name: &str) -> Option<PathBuf> {
        Some(self.cache_dir.as_ref()?.join("catalogs").join(name))
    }

    /// The configured backend base URL (for UI display).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The catalog cache directory, when disk caching is on.
    pub fn cache_dir(&self) -> Option<&Path> {
        self.cache_dir.as_deref()
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(
    url: &str,
    response: ureq::Response,
) -> Result<T, CloudError> {
    let body = response
        .into_string()
        .map_err(|e| CloudError::Network(format!("{url}: reading body: {e}")))?;
    serde_json::from_str(&body).map_err(|e| CloudError::Protocol(format!("{url}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_write_and_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let client = CloudClient::new("https://example.invalid", Some(dir.path().to_path_buf()));
        client.write_cache("templates.json", "W/\"abc\"", r#"{"entries":[]}"#);
        let (etag, body) = client.read_cache("templates.json").expect("cached");
        assert_eq!(etag, "W/\"abc\"");
        assert_eq!(body, r#"{"entries":[]}"#);
    }

    #[test]
    fn offline_with_cache_serves_stale_catalog() {
        let dir = tempfile::tempdir().unwrap();
        // .invalid TLD guarantees resolution failure — the offline path.
        let client = CloudClient::new("https://example.invalid", Some(dir.path().to_path_buf()));
        client.write_cache("templates.json", "", r#"{"entries":[]}"#);
        let catalog = client.templates().expect("stale cache served offline");
        assert!(catalog.entries.is_empty());
    }

    #[test]
    fn offline_without_cache_is_a_network_error() {
        let dir = tempfile::tempdir().unwrap();
        let client = CloudClient::new("https://example.invalid", Some(dir.path().to_path_buf()));
        match client.templates() {
            Err(CloudError::Network(_)) => {}
            other => panic!("expected Network error, got {other:?}"),
        }
    }
}
