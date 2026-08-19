//! Shared cloud endpoint configuration.
//!
//! The catalog workers (templates, text presets, SFX, LUTs, Lottie) and the
//! update check all talk to the same anonymous backend; this module owns
//! how its base URL resolves.

/// Backend (API) base URL: `[cloud] base_url` in config.toml, then the
/// `CUTLASS_API_BASE` env override, then production. Shared by the asset
/// catalogs and the update check.
pub fn base_url() -> String {
    let from_settings = cutlass_settings::load(&cutlass_settings::default_config_path())
        .map(|s| s.cloud.base_url)
        .unwrap_or_default();
    if !from_settings.is_empty() {
        return from_settings;
    }
    std::env::var("CUTLASS_API_BASE")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| cutlass_cloud::DEFAULT_BASE_URL.to_string())
}
