//! Relight, as eval sees it: which depth video a clip's light reads, and
//! the light over a clip at one moment.
//!
//! [`relight_ask`] names the depth video: the stretch of the clip's file its
//! timeline shows, at the project's frame rate. The job reads that stretch
//! through [`crate::EnhanceFrames`], guesses how near each pixel is with
//! [`RELIGHT_MODEL`], and writes a grey picture a frame, nearer lighter,
//! where `cutlass_store::relight_file` says. Once it is there, a clip with
//! a Relight preset is lit from it in preview and export alike; until then
//! it draws unlit.

use std::path::{Path, PathBuf};

use cutlass_effects::{Lighting, lighting};
use cutlass_model::{Draft, MaterialRef, Micros, Segment, SegmentId, TimeRange};
use cutlass_picture::CpuBytes;
use cutlass_store::{RelightKey, find_relight};

use crate::enhance::{EnhanceAsk, EnhanceError, clip_stretch};

/// The model depth videos are made with, read at its short side. A depth
/// video made by another is not used.
pub const RELIGHT_MODEL: &str = "depth-anything-v2-small-364";

/// What a depth video is made from: a stretch of one file at a frame rate.
#[derive(Clone, Debug, PartialEq)]
pub struct RelightAsk {
    /// The clip's file on disk.
    pub source: PathBuf,
    /// The stretch the clip shows, in the file's own time.
    pub range: TimeRange,
    /// Pictures per second of the file's time in the depth video.
    pub rate: f64,
}

impl RelightAsk {
    /// The name its depth video is kept under.
    #[must_use]
    pub fn key(&self) -> RelightKey {
        RelightKey {
            source: self.source.clone(),
            model: RELIGHT_MODEL.into(),
            rate: self.rate,
        }
    }

    /// The same stretch as [`crate::EnhanceFrames`] reads it: the clip's
    /// own frames, upright.
    #[must_use]
    pub fn frames(&self) -> EnhanceAsk {
        EnhanceAsk {
            source: self.source.clone(),
            range: self.range.clone(),
            rate: self.rate,
            level: 0,
            denoise: 0,
            interpolate: 0,
            motion_blur: None,
        }
    }
}

/// What relighting clip `segment_id` reads. A relative file stays inside
/// `root`.
pub fn relight_ask(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Result<RelightAsk, EnhanceError> {
    let (source, range, rate, _) = clip_stretch(draft, root, segment_id, "relit")?;
    Ok(RelightAsk {
        source,
        range,
        rate,
    })
}

/// The depth video that lights `segment_id` and where in the file it
/// starts, when the clip has a Relight preset and its depth is in the
/// project.
pub(crate) fn relight_depth(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Option<(PathBuf, Micros)> {
    let root = root?;
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    video.relight()?;
    let ask = relight_ask(draft, Some(root), segment_id).ok()?;
    let (path, held) = find_relight(root, &ask.key(), &ask.range)?;
    Some((path, held.start))
}

/// The light over `segment` at `time`, its keyed settings as keyed then,
/// when it has a Relight preset this build knows and some strength. Moving lights are timed from the clip's
/// start.
pub(crate) fn light_at(draft: &Draft, segment: &Segment, time: Micros) -> Option<Lighting> {
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    // A keyed strength, color or angle lights the clip as keyed at `time`.
    let relight =
        cutlass_model::relight_at(&video.relight()?, segment, &draft.keyframe_graph_list, time);
    let seconds = time.0.saturating_sub(segment.target.start.0) as f64 / 1e6;
    lighting(&relight, seconds).filter(|lit| lit.strength > 0.0)
}

/// Light `picture` from `depth`, a grey picture of how near each pixel is,
/// stretched to the picture's size. A depth of no size leaves it unlit.
pub(crate) fn relit(picture: CpuBytes, depth: &CpuBytes, lit: &Lighting) -> CpuBytes {
    let (w, h) = (picture.width(), picture.height());
    let Some(near) = stretched_grey(depth, w, h) else {
        return picture;
    };
    let mut rgba = picture.rgba().to_vec();
    cutlass_effects::relight_rgba(&mut rgba, w, h, &near, lit);
    CpuBytes::new(w, h, rgba).unwrap_or(picture)
}

/// The red of `depth` at `width` by `height`, bilinear, centres matched.
fn stretched_grey(depth: &CpuBytes, width: u32, height: u32) -> Option<Vec<u8>> {
    let (dw, dh) = (depth.width() as usize, depth.height() as usize);
    let (w, h) = (width as usize, height as usize);
    let src = depth.rgba();
    if dw == 0 || dh == 0 || w == 0 || h == 0 || src.len() < dw * dh * 4 {
        return None;
    }
    let axis = |to: usize, from: usize| -> Vec<(usize, usize, f32)> {
        (0..to)
            .map(|i| {
                let at = ((i as f32 + 0.5) * from as f32 / to as f32 - 0.5).max(0.0);
                let lo = (at as usize).min(from - 1);
                let hi = (lo + 1).min(from - 1);
                (lo, hi, at - lo as f32)
            })
            .collect()
    };
    let (xs, ys) = (axis(w, dw), axis(h, dh));
    let red = |x: usize, y: usize| f32::from(src[(y * dw + x) * 4]);
    let mut out = Vec::with_capacity(w * h);
    for &(y0, y1, fy) in &ys {
        for &(x0, x1, fx) in &xs {
            let top = red(x0, y0) + (red(x1, y0) - red(x0, y0)) * fx;
            let bottom = red(x0, y1) + (red(x1, y1) - red(x0, y1)) * fx;
            out.push((top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests;
