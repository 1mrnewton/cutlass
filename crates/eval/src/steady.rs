//! Steadied clips at render: each frame of a clip whose material has a
//! `stable` level is moved by the correction its kept camera path plans.
//!
//! The path is read once from the project's `Resources/video_stable` and
//! planned at the clip's level, then kept with the open clips, so preview
//! and export read the same file and draw the same frames. A clip with no
//! kept path yet (still measuring) or no project folder is drawn as it is.
//! The correction moves the oriented picture before the clip's crop and
//! transform, as one more map on the GPU and one resample on the CPU.

use std::path::Path;
use std::sync::Arc;

use cutlass_engine::{map_timeline, placement_of};
use cutlass_gpu::Affine;
use cutlass_model::{Draft, MaterialRef, Micros, Segment, VideoStable};
use cutlass_motion::{Correction, Plan, Strength, plan};
use cutlass_picture::CpuBytes;
use cutlass_store::{StableKey, load_camera_path};

use crate::media::OpenClips;
use crate::stable::{StableAsk, stable_ask};

/// A clip's plan, and what it was planned from.
pub(crate) struct Steady {
    ask: StableAsk,
    strength: Strength,
    plan: Plan,
}

impl Steady {
    /// The correction at `source`, a time in the file, between the
    /// planned frames around it.
    fn at(&self, source: Micros) -> Correction {
        let frames = &self.plan.frames;
        let Some(last) = frames.len().checked_sub(1) else {
            return Correction::NONE;
        };
        let place = (source.0 - self.ask.range.start.0) as f64 * self.ask.rate / 1_000_000.0;
        let place = if place.is_finite() {
            place.clamp(0.0, last as f64)
        } else {
            0.0
        };
        let below = place.floor() as usize;
        let above = (below + 1).min(last);
        let share = place - below as f64;
        let (a, b) = (frames[below], frames[above]);
        let mix = |x: f64, y: f64| x + (y - x) * share;
        Correction {
            x: mix(a.x, b.x),
            y: mix(a.y, b.y),
            turn: mix(a.turn, b.turn),
            scale: mix(a.scale, b.scale),
        }
    }
}

/// The strength a stored level asks for.
pub(crate) fn strength_of(level: i64) -> Strength {
    match level {
        VideoStable::MINIMUM_CUT => Strength::MinimumCut,
        VideoStable::MOST_STABLE => Strength::MostStable,
        _ => Strength::Recommended,
    }
}

/// The correction for `segment` at timeline `time`, or `None` when the
/// clip is not steadied or has no kept path yet.
pub(crate) fn steady_at(
    clips: &mut OpenClips,
    draft: &Draft,
    segment: &Segment,
    time: Micros,
) -> Option<Correction> {
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let level = video.stable.as_ref().filter(|stable| stable.is_on())?.level;
    let strength = strength_of(level);
    let root = clips.root()?.to_path_buf();
    let ask = stable_ask(draft, Some(&root), &segment.id).ok()?;
    let kept = clips
        .steadies()
        .get(&segment.id)
        .filter(|kept| kept.ask == ask && kept.strength == strength)
        .cloned();
    let kept = match kept {
        Some(kept) => kept,
        None => {
            let kept = Arc::new(load(&root, ask, strength)?);
            clips.steadies().insert(segment.id.clone(), kept.clone());
            kept
        }
    };
    let source = map_timeline(&placement_of(draft, segment).ok()?, time).ok()?;
    Some(kept.at(source))
}

/// The plan for `ask` at `strength`, from the path kept in `project`.
fn load(project: &Path, ask: StableAsk, strength: Strength) -> Option<Steady> {
    let key = StableKey {
        source: ask.source.clone(),
        start: ask.range.start,
        duration: ask.range.duration,
        rate: ask.rate,
    };
    let path = load_camera_path(project, &key)?;
    let plan = plan(&path, strength).ok()?;
    Some(Steady {
        ask,
        strength,
        plan,
    })
}

/// Steadied pixel to oriented-source pixel, for a picture `size` big:
/// the inverse of `correction`, which grows and turns the picture about
/// its middle and then slides it by a share of its size.
pub(crate) fn steady_map(correction: &Correction, size: (u32, u32)) -> Option<Affine> {
    let (w, h) = (f64::from(size.0), f64::from(size.1));
    let Correction { x, y, turn, scale } = *correction;
    if ![x, y, turn, scale].iter().all(|v| v.is_finite()) || scale <= 1e-9 || w <= 0.0 || h <= 0.0 {
        return None;
    }
    let (middle_x, middle_y) = (w / 2.0, h / 2.0);
    let (cos, sin) = (turn.cos() / scale, turn.sin() / scale);
    // p = m + R(-turn) / scale * (q - m - shift)
    let (sx, sy) = (middle_x + x * w, middle_y + y * h);
    Some(Affine([
        cos,
        sin,
        middle_x - (cos * sx + sin * sy),
        -sin,
        cos,
        middle_y - (-sin * sx + cos * sy),
    ]))
}

/// `outer` after `inner`.
pub(crate) fn then(inner: Affine, outer: Affine) -> Affine {
    let [a, b, c, d, e, f] = outer.0;
    let [p, q, r, s, t, u] = inner.0;
    Affine([
        a * p + b * s,
        a * q + b * t,
        a * r + b * u + c,
        d * p + e * s,
        d * q + e * t,
        d * r + e * u + f,
    ])
}

/// `picture` sampled through `map` into a `size` picture: each pixel's
/// middle mapped, floored to the source pixel, clear outside it, as the
/// GPU pass samples.
pub(crate) fn resample(picture: &CpuBytes, map: Affine, size: (u32, u32)) -> Option<CpuBytes> {
    let (src_w, src_h) = (picture.width(), picture.height());
    let src = picture.rgba();
    let (out_w, out_h) = size;
    let mut out = vec![0_u8; cutlass_picture::rgba_len(out_w, out_h)?];
    let stride = out_w as usize * 4;
    for (y, row) in out.chunks_exact_mut(stride.max(1)).enumerate() {
        for (x, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (u, v) = map.apply(x as f64 + 0.5, y as f64 + 0.5);
            if !(u >= 0.0 && v >= 0.0 && u < f64::from(src_w) && v < f64::from(src_h)) {
                continue;
            }
            let at = (v as usize * src_w as usize + u as usize) * 4;
            if let Some(from) = src.get(at..at + 4) {
                pixel.copy_from_slice(from);
            }
        }
    }
    CpuBytes::new(out_w, out_h, out).ok()
}

#[cfg(test)]
mod tests;
