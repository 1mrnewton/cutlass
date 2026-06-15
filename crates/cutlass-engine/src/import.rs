//! Register a file in the media pool using demux-only probing.

use std::path::Path;

use cutlass_cache::{CacheSpec, FrameCache, SourceFingerprint};
use cutlass_models::MediaSource;
use cutlass_probe::probe;
use tracing::debug;

use crate::error::EngineError;

/// Probe a media file and register it with the frame cache.
///
/// Audio-only sources (probe reports zero dimensions) and still images skip
/// cache registration: the frame cache stores video YUV for scrubbing, while
/// stills decode once into the in-memory RGBA cache (see `DecoderPool`).
pub fn import_media(path: &Path, cache: &FrameCache) -> Result<MediaSource, EngineError> {
    let probed = probe(path)?;

    if probed.width > 0 && !probed.is_image {
        let fingerprint = SourceFingerprint::from_path(path)?;
        // The preview cache stores frames at the reduced preview resolution
        // (see PREVIEW_MAX_HEIGHT), so the spec must describe *those* dims:
        // flipping the cap changes the spec and invalidates the stale index.
        let (width, height) = crate::composite::preview_scaled_dims(probed.width, probed.height);
        let spec = CacheSpec {
            width,
            height,
            pixfmt: "yuv420p".into(),
        };
        cache
            .register_source(fingerprint, spec)
            .map_err(cutlass_cache::DiskCacheError::from)?;
    }

    let media = probed.to_media_source(path);

    debug!(
        path = %path.display(),
        width = media.width,
        height = media.height,
        duration_ticks = media.duration.value,
        has_audio = media.has_audio,
        is_image = media.is_image,
        codec = %probed.video_codec,
        "imported media"
    );

    Ok(media)
}
