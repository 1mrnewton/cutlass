//! Auto reframe, as eval sees it: what a clip's subject search reads, and
//! the window that cuts the clip at each frame.
//!
//! [`reframe_ask`] names the search: the clip's whole file at
//! [`REFRAME_RATE`] samples a second, so a trim or split reuses it. The job
//! finds the subject in each sample and keeps the track where
//! `cutlass_store::save_subject_path` says. At draw time the track is
//! planned for the clip's shape and settings by
//! `cutlass_motion::plan_reframe`, once per clip, and the window at each
//! frame takes the place of the clip's crop, in preview and export alike.
//! Until the track is there the clip keeps its own crop.

use std::path::Path;
use std::sync::Arc;

use cutlass_engine::{map_timeline, placement_of};
use cutlass_model::{Crop, Draft, MaterialRef, Micros, Reframe, Segment, SegmentId, TimeRange};
use cutlass_motion::{Follow, plan_reframe, window_size};
use cutlass_store::{ReframeKey, load_subject_path};

use crate::enhance::{EnhanceAsk, EnhanceError};
use crate::media::OpenClips;
use crate::stable::{StableError, stable_ask};

/// The model and reading the subject is found with. A track found another
/// way is not used.
pub const REFRAME_MODEL: &str = "depth-anything-v2-small-252-popout-1";

/// Samples a second the subject is looked for at.
pub const REFRAME_RATE: f64 = 5.0;

/// What a clip's subject search reads: its whole file at a rate.
#[derive(Clone, Debug, PartialEq)]
pub struct ReframeAsk {
    pub source: std::path::PathBuf,
    /// The whole file, in its own time.
    pub range: TimeRange,
    pub rate: f64,
}

impl ReframeAsk {
    /// The name its track is kept under.
    #[must_use]
    pub fn key(&self) -> ReframeKey {
        ReframeKey {
            source: self.source.clone(),
            range: self.range.clone(),
            model: REFRAME_MODEL.into(),
            rate: self.rate,
        }
    }

    /// The same samples as [`crate::EnhanceFrames`] reads them, upright.
    #[must_use]
    pub fn frames(&self) -> EnhanceAsk {
        EnhanceAsk {
            source: self.source.clone(),
            range: self.range.clone(),
            rate: self.rate,
            level: 0,
            denoise: 0,
            deflicker: 0,
            interpolate: 0,
            motion_blur: None,
            keyed_blur: Vec::new(),
        }
    }
}

/// What reframing clip `segment_id` searches. A relative file stays inside
/// `root`.
pub fn reframe_ask(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Result<ReframeAsk, EnhanceError> {
    let file = stable_ask(draft, root, segment_id).map_err(|error| match error {
        StableError::Clip { reason } => EnhanceError::Clip {
            reason: reason
                .replace("steadied", "reframed")
                .replace("to steady", "to reframe"),
        },
        other => EnhanceError::Frame {
            reason: other.to_string(),
        },
    })?;
    Ok(ReframeAsk {
        source: file.source,
        range: file.range,
        rate: REFRAME_RATE,
    })
}

/// A clip's planned window, and what it was planned from. An empty plan is
/// a clip with no track yet.
pub(crate) struct Reframed {
    ask: ReframeAsk,
    reframe: Reframe,
    canvas: (u32, u32),
    /// The window's width and height, as shares of the frame.
    size: (f64, f64),
    middles: Vec<(f64, f64)>,
}

impl Reframed {
    /// The window's middle at `source`, a time in the file, between the
    /// samples around it.
    fn at(&self, source: Micros) -> Option<(f64, f64)> {
        let last = self.middles.len().checked_sub(1)?;
        let place = (source.0 - self.ask.range.start.0) as f64 * self.ask.rate / 1_000_000.0;
        let place = if place.is_finite() {
            place.clamp(0.0, last as f64)
        } else {
            0.0
        };
        let below = place.floor() as usize;
        let above = (below + 1).min(last);
        let share = place - below as f64;
        let (a, b) = (self.middles[below], self.middles[above]);
        Some((a.0 + (b.0 - a.0) * share, a.1 + (b.1 - a.1) * share))
    }
}

/// The window cutting `segment` at timeline `time` in a `canvas` wide and
/// high, as a crop of its upright picture, or `None` when the clip is not
/// reframed or has no track yet.
pub(crate) fn crop_at(
    clips: &mut OpenClips,
    draft: &Draft,
    segment: &Segment,
    time: Micros,
    canvas: (u32, u32),
) -> Option<Crop> {
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let reframe = video.reframe()?;
    let root = clips.root()?.to_path_buf();
    let ask = reframe_ask(draft, Some(&root), &segment.id).ok()?;
    let kept = clips
        .reframes()
        .get(&segment.id)
        .filter(|kept| kept.ask == ask && kept.reframe == reframe && kept.canvas == canvas)
        .cloned();
    let kept = match kept {
        Some(kept) => kept,
        None => {
            let kept = Arc::new(plan(&root, ask, reframe, canvas));
            clips.reframes().insert(segment.id.clone(), kept.clone());
            kept
        }
    };
    let source = map_timeline(&placement_of(draft, segment).ok()?, time).ok()?;
    let (x, y) = kept.at(source)?;
    let (w, h) = kept.size;
    Some(window_crop(x, y, w, h))
}

/// The window `w` by `h` about (`x`, `y`) as a crop quad.
fn window_crop(x: f64, y: f64, w: f64, h: f64) -> Crop {
    let (left, right) = (x - w / 2.0, x + w / 2.0);
    let (top, bottom) = (y - h / 2.0, y + h / 2.0);
    Crop {
        upper_left_x: left,
        upper_left_y: top,
        upper_right_x: right,
        upper_right_y: top,
        lower_left_x: left,
        lower_left_y: bottom,
        lower_right_x: right,
        lower_right_y: bottom,
    }
}

/// The plan for `ask`'s kept track at `reframe`'s shape and settings, or
/// an empty one when there is no track or no shape.
fn plan(project: &Path, ask: ReframeAsk, reframe: Reframe, canvas: (u32, u32)) -> Reframed {
    let mut reframed = Reframed {
        ask,
        reframe,
        canvas,
        size: (1.0, 1.0),
        middles: Vec::new(),
    };
    let Some(path) = load_subject_path(project, &reframed.ask.key()) else {
        return reframed;
    };
    let Some(window) = reframed.reframe.aspect(canvas) else {
        return reframed;
    };
    let frame = f64::from(path.width) / f64::from(path.height);
    reframed.size = window_size(frame, window);
    let follow = Follow::from_settings(reframed.reframe.stability, reframed.reframe.speed);
    reframed.middles = plan_reframe(&path.subjects, reframed.ask.rate, reframed.size, follow);
    reframed
}

#[cfg(test)]
mod tests;
