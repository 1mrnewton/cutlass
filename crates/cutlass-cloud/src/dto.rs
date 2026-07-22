//! Request/response DTOs for every `cutlass-backend` route — the contract
//! source of truth shared by the editor and the backend (which consumes
//! this crate as a git dependency; its contract tests fail CI on drift).
//!
//! Compatibility rules (the "old clients keep working" principle):
//!
//! - **Unknown-field tolerant everywhere**: no `deny_unknown_fields`,
//!   ever. The backend may add optional fields within `/v1` and shipped
//!   builds keep parsing.
//! - **Additive-only within `/v1`**: renaming/removing a field or changing
//!   a type means `/v2`, not an edit here.
//! - Enums that may grow (asset kinds, providers) carry an `Other`
//!   catch-all so new server values degrade gracefully instead of failing
//!   the whole response.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared envelope
// ---------------------------------------------------------------------------

/// Error body every non-2xx `/v1` response carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Stable machine-readable code (`rate_limited`, `insufficient_credits`,
    /// `not_found`, `invalid_request`, `upstream_failed`, `internal`).
    pub code: String,
    /// Human-readable message, safe to surface in the UI.
    pub message: String,
    /// Seconds to wait before retrying, when the server knows (429s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/// `GET /v1/app/latest-version` — the launch-screen update nudge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatestVersion {
    /// Semver of the newest released build, e.g. `0.5.3-alpha.0`.
    pub version: String,
    /// Where the "update available" chip links.
    pub download_url: String,
    /// Optional one-line release note for the tooltip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// Stock search (metadata only — files download direct from provider CDNs)
// ---------------------------------------------------------------------------

/// What kind of stock media a search targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StockKind {
    Video,
    Photo,
    Audio,
}

impl StockKind {
    pub fn as_str(self) -> &'static str {
        match self {
            StockKind::Video => "video",
            StockKind::Photo => "photo",
            StockKind::Audio => "audio",
        }
    }
}

/// Which upstream a stock item came from. `Other` keeps old clients
/// parsing when a new provider joins server-side.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StockProviderId {
    Pexels,
    Pixabay,
    #[serde(other)]
    Other,
}

/// `GET /v1/stock/search` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockSearchResponse {
    pub items: Vec<StockItem>,
    /// 1-based page echoed back.
    pub page: u32,
    /// Whether asking for `page + 1` is worthwhile.
    pub has_more: bool,
}

/// One stock result, normalized across providers. Carries **everything
/// needed at download time** (file URLs, attribution, license) so there is
/// no second metadata round-trip and nothing breaks if a provider changes
/// its detail API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockItem {
    /// Provider-scoped id, opaque to the client.
    pub id: String,
    pub provider: StockProviderId,
    pub kind: StockKind,
    /// Pixel size of the largest file, when known (0 for audio).
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    /// Duration in seconds for video/audio; `None` for photos.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Small preview image (grid thumbnail), direct provider CDN URL.
    pub thumbnail_url: String,
    /// Downloadable files by quality, best first. Direct CDN URLs — the
    /// client never routes bytes through the backend.
    pub files: Vec<StockFile>,
    /// Display name of the creator.
    #[serde(default)]
    pub author: String,
    /// Ready-to-display attribution line ("Video by X on Pexels").
    #[serde(default)]
    pub attribution: String,
    /// Short license note ("Pexels License", "Pixabay Content License").
    #[serde(default)]
    pub license: String,
    /// Provider page for the item (attribution link target).
    #[serde(default)]
    pub source_url: String,
}

/// One downloadable rendition of a stock item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockFile {
    /// Direct provider-CDN URL (keyless).
    pub url: String,
    /// Human label ("1080p", "4K", "original", "mp3").
    #[serde(default)]
    pub quality: String,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    /// MIME type when the provider reports one.
    #[serde(default)]
    pub content_type: String,
    /// File size in bytes when the provider reports one (0 = unknown).
    #[serde(default)]
    pub size_bytes: u64,
}

// ---------------------------------------------------------------------------
// Asset catalog (templates, text presets, SFX, LUTs, skills)
// ---------------------------------------------------------------------------

/// What a catalog entry is. `Other` keeps old clients parsing when a new
/// kind ships server-side (they skip entries they don't understand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Template,
    TextPreset,
    Sfx,
    Lut,
    Lottie,
    Skill,
    #[serde(other)]
    Other,
}

/// `GET /v1/templates`, `/v1/assets/*` responses share this shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogResponse {
    pub entries: Vec<CatalogEntry>,
}

/// One catalog asset. Files live on the CDN; the backend serves metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Stable catalog id (permanent once shipped — projects may store it).
    pub id: String,
    pub kind: AssetKind,
    pub name: String,
    /// Browse category (mirrors `TemplateCategory` for templates).
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// CDN URL of the asset file (template bundle, `.cube`, audio file,
    /// preset JSON, skill archive).
    pub file_url: String,
    /// CDN URL of a preview (image or short clip), when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview_url: Option<String>,
    /// File size in bytes (0 = unknown).
    #[serde(default)]
    pub size_bytes: u64,
    /// SHA-256 of the file, hex — verified after download.
    #[serde(default)]
    pub checksum_sha256: String,
    /// Templates only: the minimum project-schema version an app must
    /// support to open this template. Older apps refuse gracefully.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_schema_version: Option<u32>,
    /// Attribution fields, reserved for community submissions; first-party
    /// entries say "Cutlass" / "CC0".
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub license: String,
    /// Templates: seconds of timeline; SFX: seconds of audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Templates: how many media slots a user fills.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_count: Option<u32>,
}

// ---------------------------------------------------------------------------
// Text presets (payload of a `TextPreset` catalog entry's file)
// ---------------------------------------------------------------------------

/// The text-preset catalog file: styled title recipes rendered by the
/// existing text + look-animation pipeline. **Bundled-OFL-fonts-only**:
/// `font_family` must name a font that ships with Cutlass, because the
/// renderer falls back silently on missing named fonts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPresetCatalog {
    pub presets: Vec<TextPreset>,
}

/// One animated-text preset: a `TextStyle` subset plus catalog animation
/// ids, all resolvable by the shipped editor with no new render tech.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPreset {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub category: String,
    /// Bundled (OFL) font family name; empty = default sans.
    #[serde(default)]
    pub font_family: String,
    pub font_size: f32,
    /// RGBA fill.
    pub fill: [u8; 4],
    /// Look-animation catalog ids (must exist in the app's catalogs;
    /// unknown ids are skipped, never errors).
    #[serde(default)]
    pub animation_in: Option<String>,
    #[serde(default)]
    pub animation_out: Option<String>,
    #[serde(default)]
    pub animation_combo: Option<String>,
    /// Sample text shown in the Library tile.
    #[serde(default)]
    pub sample_text: String,
}

// ---------------------------------------------------------------------------
// Generation jobs (BYOK fal path; shared shape with the backend contract)
// ---------------------------------------------------------------------------

/// Generation request payload (prompt + optional model / duration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    /// Provider-recognizable model id; empty = client default.
    #[serde(default)]
    pub model: String,
    /// Seconds, for video/TTS where the provider accepts it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Client idempotency key.
    pub idempotency_key: String,
}

/// Job status. `Other` tolerates new server states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    #[serde(other)]
    Other,
}

/// Generation job status / result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub status: JobStatus,
    /// Provider URL of the result once succeeded — the app downloads it
    /// directly (no bytes through the backend) and imports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_url: Option<String>,
    /// Reserved for contract compatibility with the backend DTO; BYOK
    /// always reports `0`.
    #[serde(default)]
    pub credits_charged: i64,
    /// Failure detail when `status == Failed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_item_tolerates_unknown_fields_and_providers() {
        // A future server adds fields and a new provider; an old client
        // must keep parsing (the compat principle, contract-tested).
        let json = r#"{
            "id": "123", "provider": "shutterstock_free", "kind": "video",
            "width": 1920, "height": 1080, "duration_seconds": 12.5,
            "thumbnail_url": "https://cdn/x.jpg",
            "files": [{"url": "https://cdn/x.mp4", "quality": "1080p",
                       "width": 1920, "height": 1080, "brand_new_field": 1}],
            "author": "A", "attribution": "Video by A", "license": "L",
            "source_url": "https://p/x", "brand_new_field": true
        }"#;
        let item: StockItem = serde_json::from_str(json).expect("tolerant parse");
        assert_eq!(item.provider, StockProviderId::Other);
        assert_eq!(item.files.len(), 1);
    }

    #[test]
    fn catalog_entry_defaults_optional_fields() {
        let json = r#"{
            "id": "tpl-1", "kind": "template", "name": "Vlog intro",
            "file_url": "https://cdn/tpl-1.cutlassb"
        }"#;
        let entry: CatalogEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(entry.kind, AssetKind::Template);
        assert!(entry.tags.is_empty());
        assert_eq!(entry.min_schema_version, None);
    }

    #[test]
    fn unknown_asset_kind_degrades_to_other() {
        let json = r#"{"id": "x", "kind": "hologram", "name": "X",
                       "file_url": "https://cdn/x"}"#;
        let entry: CatalogEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(entry.kind, AssetKind::Other);
    }

    #[test]
    fn dto_roundtrips() {
        let resp = StockSearchResponse {
            items: vec![],
            page: 1,
            has_more: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: StockSearchResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.page, 1);

        let job = Job {
            id: "j".into(),
            status: JobStatus::Succeeded,
            result_url: Some("https://p/out.mp4".into()),
            credits_charged: 50,
            error: None,
        };
        let json = serde_json::to_string(&job).unwrap();
        let back: Job = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, JobStatus::Succeeded);
    }
}
