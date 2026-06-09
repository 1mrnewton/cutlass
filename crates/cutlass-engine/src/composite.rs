//! Timeline layer resolution and CPU decode helpers for GPU compositing.

use std::time::Duration;

use cutlass_cache::{FrameCache, SourceFingerprint};
use cutlass_compositor::{CompositeLayer, CompositorConfig};
use cutlass_models::{Clip, Generator, ModelError, Project, RationalTime, TrackKind};
use tracing::debug;

use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::{decoded_to_rgba, pack_yuv420p, unpack_yuv420p, RgbaFrame};

const DEFAULT_WIDTH: u32 = 1920;
const DEFAULT_HEIGHT: u32 = 1080;

/// Output canvas size: max media dimensions on the timeline, or 1920×1080 fallback.
/// Width and height are forced even for downstream H.264 export.
pub fn composite_canvas_size(project: &Project) -> (u32, u32) {
    let mut max_w = 0u32;
    let mut max_h = 0u32;

    for track in project.timeline().tracks_ordered() {
        if track.kind != TrackKind::Video {
            continue;
        }
        for clip in track.clips() {
            if let Some(media_id) = clip.media()
                && let Some(media) = project.media(media_id)
            {
                max_w = max_w.max(media.width);
                max_h = max_h.max(media.height);
            }
        }
    }

    if max_w == 0 || max_h == 0 {
        (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    } else {
        (to_even(max_w), to_even(max_h))
    }
}

/// Resolve enabled video layers at `time`, bottom to top.
pub fn resolve_layers(
    project: &Project,
    cache: &FrameCache,
    pool: &mut DecoderPool,
    time: RationalTime,
    canvas: &CompositorConfig,
) -> Result<Vec<CompositeLayer>, EngineError> {
    let mut layers = Vec::new();

    for track in project.timeline().tracks_ordered() {
        if track.kind != TrackKind::Video || !track.enabled {
            continue;
        }
        let Some(clip) = track.clip_at(time)? else {
            continue;
        };

        match &clip.content {
            cutlass_models::ClipSource::Media { .. } => {
                let frame = decode_media_rgba(project, cache, pool, clip, time)?;
                let bytes = resize_rgba(&frame, canvas.width, canvas.height)?;
                layers.push(CompositeLayer::Rgba { bytes });
            }
            cutlass_models::ClipSource::Generated(generator) => match generator {
                Generator::SolidColor { rgba } => {
                    layers.push(CompositeLayer::Solid { rgba: *rgba });
                }
                Generator::Text { .. } | Generator::Shape { .. } | Generator::Adjustment => {
                    debug!(?generator, "skipping unsupported generator for composite");
                }
            },
        }
    }

    Ok(layers)
}

fn decode_media_rgba(
    project: &Project,
    cache: &FrameCache,
    pool: &mut DecoderPool,
    clip: &Clip,
    time: RationalTime,
) -> Result<RgbaFrame, EngineError> {
    let source_time = clip
        .source_time_at(time)?
        .ok_or_else(|| EngineError::Preview("timeline position outside clip".into()))?;

    let media_id = clip
        .media()
        .ok_or_else(|| EngineError::Preview("clip has no backing media".into()))?;
    let media = project
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;

    let fingerprint = SourceFingerprint::from_path(media.path())?;
    let source_id = fingerprint.id();
    let target = rational_time_to_duration(source_time);

    let (decoder, index) = pool.decoder_and_index(media_id, media.path())?;
    let pts = index.duration_to_ticks(target);

    if let Some(packed) = cache.get(source_id, pts) {
        let decoded = unpack_yuv420p(&packed, media.width, media.height)?;
        return decoded_to_rgba(&decoded);
    }

    let decoded = decoder
        .seek_to_frame(target)?
        .ok_or_else(|| EngineError::Preview("decoder returned no frame".into()))?;

    if let Ok(packed) = pack_yuv420p(&decoded) {
        cache.cache_frame(source_id, decoded.pts_ticks, packed);
    }

    decoded_to_rgba(&decoded)
}

fn resize_rgba(frame: &RgbaFrame, dst_w: u32, dst_h: u32) -> Result<Vec<u8>, EngineError> {
    if frame.width == dst_w && frame.height == dst_h {
        return Ok(frame.bytes.clone());
    }
    if frame.width == 0 || frame.height == 0 {
        return Err(EngineError::Preview("cannot resize empty frame".into()));
    }

    let src_w = frame.width as f32;
    let src_h = frame.height as f32;
    let dst_w_f = dst_w as f32;
    let dst_h_f = dst_h as f32;
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_x = (x as f32 + 0.5) * src_w / dst_w_f - 0.5;
            let src_y = (y as f32 + 0.5) * src_h / dst_h_f - 0.5;
            let px = sample_bilinear(&frame.bytes, frame.width, frame.height, src_x, src_y);
            let i = ((y * dst_w + x) * 4) as usize;
            out[i..i + 4].copy_from_slice(&px);
        }
    }
    Ok(out)
}

fn sample_bilinear(bytes: &[u8], w: u32, h: u32, x: f32, y: f32) -> [u8; 4] {
    let x = x.clamp(0.0, (w.saturating_sub(1)) as f32);
    let y = y.clamp(0.0, (h.saturating_sub(1)) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let p = |px: u32, py: u32| -> [f32; 4] {
        let i = ((py * w + px) * 4) as usize;
        [
            bytes[i] as f32,
            bytes[i + 1] as f32,
            bytes[i + 2] as f32,
            bytes[i + 3] as f32,
        ]
    };

    let c00 = p(x0, y0);
    let c10 = p(x1, y0);
    let c01 = p(x0, y1);
    let c11 = p(x1, y1);

    let mut out = [0u8; 4];
    for ch in 0..4 {
        let top = c00[ch] * (1.0 - tx) + c10[ch] * tx;
        let bot = c01[ch] * (1.0 - tx) + c11[ch] * tx;
        let v = top * (1.0 - ty) + bot * ty;
        out[ch] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn to_even(v: u32) -> u32 {
    if v.is_multiple_of(2) { v } else { v + 1 }
}

fn rational_time_to_duration(time: RationalTime) -> Duration {
    let num = i128::from(time.rate.num);
    let den = i128::from(time.rate.den);
    if num <= 0 || den <= 0 || time.value <= 0 {
        return Duration::ZERO;
    }
    let nanos = (i128::from(time.value) * 1_000_000_000 * den) / num;
    Duration::from_nanos(nanos.clamp(0, i128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::Rational;

    #[test]
    fn composite_canvas_size_defaults_without_media() {
        let project = Project::new("test", Rational::FPS_24);
        assert_eq!(composite_canvas_size(&project), (1920, 1080));
    }

    #[test]
    fn resize_identity_is_copy() {
        let frame = RgbaFrame::new(2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .unwrap();
        let out = resize_rgba(&frame, 2, 2).unwrap();
        assert_eq!(out, frame.bytes);
    }
}
