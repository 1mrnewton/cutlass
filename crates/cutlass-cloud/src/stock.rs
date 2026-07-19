//! The stock-search seam: one trait, two ways to reach the providers.
//!
//! - [`BackendStockProvider`]: anonymous, routes through `cutlass-backend`
//!   (the server holds the Pexels/Pixabay keys and caches responses).
//! - [`DirectStockProvider`]: BYOK — the user configured their own stock
//!   keys, so even search skips the backend entirely.
//!
//! Both normalize into the shared [`StockItem`] DTO; the Library UI cannot
//! tell them apart. File downloads always go direct to the provider CDN
//! either way (`download` module).

use std::time::Duration;

use crate::client::CloudClient;
use crate::dto::{StockFile, StockItem, StockKind, StockProviderId, StockSearchResponse};
use crate::error::CloudError;

/// Where stock search results come from. Implementations are blocking;
/// callers run them on worker threads.
pub trait StockProvider: Send + Sync {
    fn search(
        &self,
        query: &str,
        kind: StockKind,
        page: u32,
    ) -> Result<StockSearchResponse, CloudError>;
}

/// Anonymous search through the backend.
pub struct BackendStockProvider {
    client: CloudClient,
}

impl BackendStockProvider {
    pub fn new(client: CloudClient) -> Self {
        Self { client }
    }
}

impl StockProvider for BackendStockProvider {
    fn search(
        &self,
        query: &str,
        kind: StockKind,
        page: u32,
    ) -> Result<StockSearchResponse, CloudError> {
        self.client.stock_search(query, kind, page)
    }
}

/// BYOK: user-supplied Pexels and/or Pixabay keys, providers called
/// directly. Mirrors the normalization the backend performs so the UI
/// sees identical [`StockItem`]s.
pub struct DirectStockProvider {
    pexels_key: Option<String>,
    pixabay_key: Option<String>,
    agent: ureq::Agent,
}

/// Results per provider page. Both providers accept this range.
const PER_PAGE: u32 = 24;

impl DirectStockProvider {
    /// At least one key should be present; a keyless instance returns
    /// empty results rather than erroring (the caller decides routing).
    pub fn new(pexels_key: Option<String>, pixabay_key: Option<String>) -> Self {
        Self {
            pexels_key,
            pixabay_key,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout(Duration::from_secs(30))
                .build(),
        }
    }

    pub fn has_any_key(&self) -> bool {
        self.pexels_key.is_some() || self.pixabay_key.is_some()
    }

    fn search_pexels(
        &self,
        key: &str,
        query: &str,
        kind: StockKind,
        page: u32,
    ) -> Result<(Vec<StockItem>, bool), CloudError> {
        let (url, is_video) = match kind {
            StockKind::Video => ("https://api.pexels.com/videos/search", true),
            StockKind::Photo => ("https://api.pexels.com/v1/search", false),
            // Pexels has no public audio API.
            StockKind::Audio => return Ok((vec![], false)),
        };
        let page_string = page.to_string();
        let per_page_string = PER_PAGE.to_string();
        let response = self
            .agent
            .get(url)
            .set("Authorization", key)
            .query("query", query)
            .query("page", &page_string)
            .query("per_page", &per_page_string)
            .call()
            .map_err(|e| CloudError::from_ureq(url, e))?;
        let body: serde_json::Value = response
            .into_json()
            .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))?;

        let has_more = body.get("next_page").is_some_and(|v| !v.is_null());
        let items = if is_video {
            body["videos"]
                .as_array()
                .map(|list| list.iter().filter_map(pexels_video_item).collect())
                .unwrap_or_default()
        } else {
            body["photos"]
                .as_array()
                .map(|list| list.iter().filter_map(pexels_photo_item).collect())
                .unwrap_or_default()
        };
        Ok((items, has_more))
    }

    fn search_pixabay(
        &self,
        key: &str,
        query: &str,
        kind: StockKind,
        page: u32,
    ) -> Result<(Vec<StockItem>, bool), CloudError> {
        let url = match kind {
            StockKind::Video => "https://pixabay.com/api/videos/",
            StockKind::Photo => "https://pixabay.com/api/",
            // Pixabay's public API does not cover audio (verified; the
            // catalog's curated CC0 packs are the audio fallback).
            StockKind::Audio => return Ok((vec![], false)),
        };
        let page_string = page.to_string();
        let per_page_string = PER_PAGE.to_string();
        let response = self
            .agent
            .get(url)
            .query("key", key)
            .query("q", query)
            .query("page", &page_string)
            .query("per_page", &per_page_string)
            .query("safesearch", "true")
            .call()
            .map_err(|e| CloudError::from_ureq(url, e))?;
        let body: serde_json::Value = response
            .into_json()
            .map_err(|e| CloudError::Protocol(format!("{url}: {e}")))?;

        let total = body["totalHits"].as_u64().unwrap_or(0);
        let has_more = u64::from(page * PER_PAGE) < total;
        let items = body["hits"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter_map(|hit| pixabay_item(hit, kind))
                    .collect()
            })
            .unwrap_or_default();
        Ok((items, has_more))
    }
}

impl StockProvider for DirectStockProvider {
    fn search(
        &self,
        query: &str,
        kind: StockKind,
        page: u32,
    ) -> Result<StockSearchResponse, CloudError> {
        let mut items = Vec::new();
        let mut has_more = false;
        let mut first_error: Option<CloudError> = None;

        if let Some(key) = &self.pexels_key {
            match self.search_pexels(key, query, kind, page) {
                Ok((mut found, more)) => {
                    items.append(&mut found);
                    has_more |= more;
                }
                Err(e) => first_error = Some(e),
            }
        }
        if let Some(key) = &self.pixabay_key {
            match self.search_pixabay(key, query, kind, page) {
                Ok((mut found, more)) => {
                    items.append(&mut found);
                    has_more |= more;
                }
                Err(e) => first_error = first_error.or(Some(e)),
            }
        }

        // One provider failing shouldn't blank the other's results; only
        // a totally empty outcome surfaces the error.
        if items.is_empty()
            && let Some(e) = first_error
        {
            return Err(e);
        }
        Ok(StockSearchResponse {
            items,
            page,
            has_more,
        })
    }
}

// --- Pexels normalization ---------------------------------------------------

fn pexels_video_item(v: &serde_json::Value) -> Option<StockItem> {
    let id = v["id"].as_u64()?;
    let author = v["user"]["name"].as_str().unwrap_or("").to_string();
    let mut files: Vec<StockFile> = v["video_files"]
        .as_array()?
        .iter()
        .filter_map(|f| {
            Some(StockFile {
                url: f["link"].as_str()?.to_string(),
                quality: f["quality"].as_str().unwrap_or("").to_string(),
                width: f["width"].as_u64().unwrap_or(0) as u32,
                height: f["height"].as_u64().unwrap_or(0) as u32,
                content_type: f["file_type"].as_str().unwrap_or("").to_string(),
                size_bytes: f["size"].as_u64().unwrap_or(0),
            })
        })
        .collect();
    // Best (largest) rendition first — what "download" grabs by default.
    files.sort_by_key(|f| std::cmp::Reverse(u64::from(f.width) * u64::from(f.height)));
    Some(StockItem {
        id: id.to_string(),
        provider: StockProviderId::Pexels,
        kind: StockKind::Video,
        width: v["width"].as_u64().unwrap_or(0) as u32,
        height: v["height"].as_u64().unwrap_or(0) as u32,
        duration_seconds: v["duration"].as_f64(),
        thumbnail_url: v["image"].as_str().unwrap_or("").to_string(),
        files,
        attribution: format!("Video by {author} on Pexels"),
        author,
        license: "Pexels License".into(),
        source_url: v["url"].as_str().unwrap_or("").to_string(),
    })
}

fn pexels_photo_item(p: &serde_json::Value) -> Option<StockItem> {
    let id = p["id"].as_u64()?;
    let author = p["photographer"].as_str().unwrap_or("").to_string();
    let src = &p["src"];
    let mut files = Vec::new();
    for (label, key) in [("original", "original"), ("large", "large2x")] {
        if let Some(url) = src[key].as_str() {
            files.push(StockFile {
                url: url.to_string(),
                quality: label.to_string(),
                width: 0,
                height: 0,
                content_type: "image/jpeg".into(),
                size_bytes: 0,
            });
        }
    }
    Some(StockItem {
        id: id.to_string(),
        provider: StockProviderId::Pexels,
        kind: StockKind::Photo,
        width: p["width"].as_u64().unwrap_or(0) as u32,
        height: p["height"].as_u64().unwrap_or(0) as u32,
        duration_seconds: None,
        thumbnail_url: src["medium"].as_str().unwrap_or("").to_string(),
        files,
        attribution: format!("Photo by {author} on Pexels"),
        author,
        license: "Pexels License".into(),
        source_url: p["url"].as_str().unwrap_or("").to_string(),
    })
}

// --- Pixabay normalization --------------------------------------------------

fn pixabay_item(hit: &serde_json::Value, kind: StockKind) -> Option<StockItem> {
    let id = hit["id"].as_u64()?;
    let author = hit["user"].as_str().unwrap_or("").to_string();
    let (files, thumbnail, width, height, duration) = match kind {
        StockKind::Video => {
            let videos = &hit["videos"];
            let mut files: Vec<StockFile> = ["large", "medium", "small"]
                .iter()
                .filter_map(|tier| {
                    let v = &videos[*tier];
                    Some(StockFile {
                        url: v["url"].as_str()?.to_string(),
                        quality: (*tier).to_string(),
                        width: v["width"].as_u64().unwrap_or(0) as u32,
                        height: v["height"].as_u64().unwrap_or(0) as u32,
                        content_type: "video/mp4".into(),
                        size_bytes: v["size"].as_u64().unwrap_or(0),
                    })
                })
                .collect();
            files.retain(|f| !f.url.is_empty());
            let thumb = videos["medium"]["thumbnail"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let (w, h) = files.first().map(|f| (f.width, f.height)).unwrap_or((0, 0));
            (files, thumb, w, h, hit["duration"].as_f64())
        }
        StockKind::Photo => {
            let mut files = Vec::new();
            for (label, key) in [("large", "largeImageURL"), ("web", "webformatURL")] {
                if let Some(url) = hit[key].as_str() {
                    files.push(StockFile {
                        url: url.to_string(),
                        quality: label.to_string(),
                        width: 0,
                        height: 0,
                        content_type: "image/jpeg".into(),
                        size_bytes: 0,
                    });
                }
            }
            let thumb = hit["previewURL"].as_str().unwrap_or("").to_string();
            let w = hit["imageWidth"].as_u64().unwrap_or(0) as u32;
            let h = hit["imageHeight"].as_u64().unwrap_or(0) as u32;
            (files, thumb, w, h, None)
        }
        StockKind::Audio => return None,
    };
    Some(StockItem {
        id: id.to_string(),
        provider: StockProviderId::Pixabay,
        kind,
        width,
        height,
        duration_seconds: duration,
        thumbnail_url: thumbnail,
        files,
        attribution: format!("{author} on Pixabay"),
        author,
        license: "Pixabay Content License".into(),
        source_url: hit["pageURL"].as_str().unwrap_or("").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyless_direct_provider_returns_empty_not_error() {
        let provider = DirectStockProvider::new(None, None);
        assert!(!provider.has_any_key());
        let out = provider.search("sunset", StockKind::Video, 1).unwrap();
        assert!(out.items.is_empty());
        assert!(!out.has_more);
    }

    #[test]
    fn pexels_video_normalizes_and_sorts_best_first() {
        let raw = serde_json::json!({
            "id": 857191,
            "width": 1920, "height": 1080, "duration": 10,
            "url": "https://www.pexels.com/video/857191/",
            "image": "https://images.pexels.com/videos/857191/thumb.jpg",
            "user": {"name": "Joey"},
            "video_files": [
                {"link": "https://cdn/sd.mp4", "quality": "sd",
                 "width": 640, "height": 360, "file_type": "video/mp4", "size": 100},
                {"link": "https://cdn/hd.mp4", "quality": "hd",
                 "width": 1920, "height": 1080, "file_type": "video/mp4", "size": 900}
            ]
        });
        let item = pexels_video_item(&raw).expect("normalizes");
        assert_eq!(item.provider, StockProviderId::Pexels);
        assert_eq!(item.files[0].quality, "hd");
        assert_eq!(item.attribution, "Video by Joey on Pexels");
        assert_eq!(item.duration_seconds, Some(10.0));
    }

    #[test]
    fn pixabay_video_normalizes_tiers() {
        let raw = serde_json::json!({
            "id": 125, "pageURL": "https://pixabay.com/videos/id-125/",
            "duration": 12, "user": "creator",
            "videos": {
                "large": {"url": "https://cdn/large.mp4", "width": 1920,
                          "height": 1080, "size": 900},
                "medium": {"url": "https://cdn/med.mp4", "width": 1280,
                           "height": 720, "size": 500,
                           "thumbnail": "https://cdn/thumb.jpg"},
                "small": {"url": "", "width": 0, "height": 0, "size": 0}
            }
        });
        let item = pixabay_item(&raw, StockKind::Video).expect("normalizes");
        assert_eq!(item.provider, StockProviderId::Pixabay);
        assert_eq!(item.files.len(), 2); // empty small URL dropped
        assert_eq!(item.files[0].quality, "large");
        assert_eq!(item.thumbnail_url, "https://cdn/thumb.jpg");
    }
}
