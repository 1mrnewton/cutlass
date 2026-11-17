//! Remove BG on the selected clip: store its settings, and make its matte
//! as a job when the project has none for it yet. The matte goes into the
//! project's `Resources/video_matte`, and the preview and export cut the
//! clip with it as soon as the job lands.
//!
//! Auto removal needs nothing from the user. Custom removal is told what to
//! keep on some frames, by clicks (Select object) or by painting (Paint by
//! hand), and follows that through the clip. The edge, the outline and
//! their values need no new matte, so they show at once; trimming the clip
//! keeps its matte, and a wider trim or a new pick makes a new one.

use std::path::{Path, PathBuf};

use cutlass_effects::{Brush, Matting, MattingMode, Pick, PickClick, Stroke, segment_matting};
use cutlass_eval::{MatteAsk, matte_ask, matte_ask_with};
use cutlass_model::{MaterialId, MaterialRef, Micros, SegmentId};
use cutlass_session::Request;
use cutlass_store::find_matte;
use cutlass_vision::{Compute, EnhanceJobError, Follower, Matter, follow_clip, matte_clip};

use super::enhance::model_dir;
use super::{Landing, Shell, ShellError};

/// The job's label, which the inspector looks for to show its progress.
pub(crate) const MATTE_JOB: &str = "Remove BG";

/// The folder in `Models` holding Auto removal's model, and its weights.
pub(super) const AUTO_FOLDER: &str = "matting";
pub(super) const AUTO_WEIGHTS: &str = "birefnet_lite.bpk";

/// The folder in `Models` holding Custom removal's follower, and its
/// weights.
pub(super) const FOLLOW_FOLDER: &str = "follow";
pub(super) const FOLLOW_WEIGHTS: &str = "sam2.1_hiera_tiny.bpk";

/// The smallest and largest brush, as a share of the frame's short side.
const BRUSH_SMALLEST: f64 = 0.005;
const BRUSH_LARGEST: f64 = 0.1;

/// Auto removal's model, loaded from `dir`.
fn load_matter(dir: &Path, compute: Compute) -> Result<Box<dyn Matter + Send>, String> {
    let _ = (dir, compute);
    Err("Auto removal's model is not in this build yet".to_owned())
}

/// Custom removal's follower, loaded from `dir`.
fn load_follower(dir: &Path, compute: Compute) -> Result<Box<dyn Follower + Send>, String> {
    let _ = (dir, compute);
    Err("Custom removal's model is not in this build yet".to_owned())
}

/// Brush strokes being painted on one frame of a clip, before they are
/// applied.
#[derive(Clone, Debug, PartialEq)]
pub struct Painting {
    pub segment_id: SegmentId,
    /// The timeline time painting started at.
    pub time: Micros,
    /// The source time of the frame painted on.
    source: Micros,
    /// The strokes so far, in unit coordinates of the frame.
    pub brushes: Vec<Brush>,
    /// The stroke being drawn.
    pub live: Option<Brush>,
    /// A smart brush finds the edge of what it touches.
    pub smart: bool,
    /// Paint the subject in, or erase it.
    pub keep: bool,
    /// The brush's size, 0 to 1.
    pub size: f64,
    /// Each stroke in `brushes` as the preview draws it.
    pub trails: Vec<Trail>,
    /// The stroke being drawn, as the preview draws it.
    pub live_trail: Option<Trail>,
}

/// A brush stroke in canvas pixels, for drawing over the preview.
#[derive(Clone, Debug, PartialEq)]
pub struct Trail {
    /// The path on the canvas.
    pub points: Vec<(f64, f64)>,
    /// The brush's radius on the canvas.
    pub radius: f64,
    /// Paint the subject in, or erase it.
    pub keep: bool,
}

impl Painting {
    /// The brush's radius, as a share of the frame's short side.
    #[must_use]
    pub fn radius(&self) -> f64 {
        BRUSH_SMALLEST + (BRUSH_LARGEST - BRUSH_SMALLEST) * self.size.clamp(0.0, 1.0)
    }
}

/// Whether Auto removal's model is on this machine.
#[must_use]
pub fn auto_model_here() -> bool {
    model_dir(AUTO_FOLDER, AUTO_WEIGHTS)
        .join(AUTO_WEIGHTS)
        .is_file()
}

/// Whether Custom removal's follower is on this machine.
#[must_use]
pub fn follow_model_here() -> bool {
    model_dir(FOLLOW_FOLDER, FOLLOW_WEIGHTS)
        .join(FOLLOW_WEIGHTS)
        .is_file()
}

impl Shell {
    /// The selected clip, when it is a movie that can be cut out.
    fn matting_segment(&self) -> Result<SegmentId, ShellError> {
        let refusal = "only a video clip can have its background removed";
        let segment_id = self.picture_segment(refusal)?;
        let draft = self.session_draft()?;
        let movie = super::placed(draft, &segment_id).is_some_and(|(segment, _)| {
            matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(video)) if video.kind != "photo"
            )
        });
        if !movie {
            return Err(ShellError::Unavailable(refusal.to_owned()));
        }
        Ok(segment_id)
    }

    /// The selected clip's Remove BG settings, when it has some.
    #[must_use]
    pub fn matting(&self) -> Option<Matting> {
        let draft = self.draft()?;
        let segment_id = self.selected()?;
        let (segment, _) = super::placed(draft, segment_id)?;
        segment_matting(draft, segment)
    }

    /// Whether the selected clip's matte is in the project.
    #[must_use]
    pub fn matte_made(&self) -> bool {
        let (Some(draft), Some(root), Some(segment_id)) =
            (self.draft(), self.path.as_deref(), self.selected())
        else {
            return false;
        };
        matte_ask(draft, Some(root), segment_id)
            .ok()
            .is_some_and(|ask| find_matte(root, &ask.key(), &ask.range).is_some())
    }

    /// How far the Remove BG job is, 0 to 1, while one runs.
    #[must_use]
    pub fn matte_progress(&self) -> Option<f32> {
        self.job()
            .filter(|job| job.label == MATTE_JOB)
            .map(|job| job.fraction.unwrap_or(0.0))
    }

    /// Turn Auto removal on the selected clip on, keeping its edge and
    /// outline and dropping Custom removal's picks, or off, taking Remove
    /// BG off the clip. The matte is made as a job after.
    pub fn set_auto_removal(&mut self, on: bool) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.refuse_locked(&segment_id)?;
        let now = self.matting();
        if !on {
            return match now {
                Some(matting) if matting.mode == MattingMode::Auto => {
                    self.put_matting(segment_id, None)
                }
                _ => Ok(()),
            };
        }
        let matting = Matting {
            mode: MattingMode::Auto,
            picks: Vec::new(),
            ..now.unwrap_or_default()
        };
        self.put_matting(segment_id, Some(matting))
    }

    /// Take Custom removal off the selected clip, with its picks.
    pub fn clear_custom_removal(&mut self) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.refuse_locked(&segment_id)?;
        match self.matting() {
            Some(matting) if matting.mode == MattingMode::Custom => {
                self.put_matting(segment_id, None)
            }
            _ => Ok(()),
        }
    }

    /// Set how much the selected clip's edge is softened and pulled in,
    /// each 0 to 1. No new matte is made.
    pub fn set_matting_edge(&mut self, feather: f64, clean_up: f64) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.refuse_locked(&segment_id)?;
        let Some(matting) = self.matting() else {
            return Err(ShellError::Unavailable(
                "turn Auto removal or Custom removal on first".to_owned(),
            ));
        };
        self.put_matting(
            segment_id,
            Some(Matting {
                feather: feather.clamp(0.0, 1.0),
                clean_up: clean_up.clamp(0.0, 1.0),
                ..matting
            }),
        )
    }

    /// Outline what is kept on the selected clip with `stroke`, or take
    /// the outline off with `None`. No new matte is made.
    pub fn set_matting_stroke(&mut self, stroke: Option<Stroke>) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.refuse_locked(&segment_id)?;
        let Some(matting) = self.matting() else {
            return Err(ShellError::Unavailable(
                "turn Auto removal or Custom removal on first".to_owned(),
            ));
        };
        self.put_matting(segment_id, Some(Matting { stroke, ..matting }))
    }

    /// Make the selected clip's matte again when the project lacks it, as
    /// after a wider trim or a file that changed.
    pub fn make_matte(&mut self) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.ensure_matte(&segment_id)
    }

    /// Store `matting` on `segment_id` as one undo step, then make its
    /// matte when the project lacks it.
    fn put_matting(
        &mut self,
        segment_id: SegmentId,
        matting: Option<Matting>,
    ) -> Result<(), ShellError> {
        self.put_matting_with(segment_id, matting, Vec::new())
    }

    /// As [`Self::put_matting`], with `before` in the same undo step. When
    /// `matting` needs a matte the project lacks and its model is not on
    /// this machine, nothing is stored.
    fn put_matting_with(
        &mut self,
        segment_id: SegmentId,
        matting: Option<Matting>,
        mut before: Vec<Request>,
    ) -> Result<(), ShellError> {
        let make = match &matting {
            Some(matting) if matting.ready() => self.needs_matte(&segment_id, matting)?,
            _ => false,
        };
        let material_id = MaterialId::new(self.fresh("matting"));
        before.push(Request::SetMatting {
            segment_id: segment_id.clone(),
            material_id,
            matting,
        });
        self.apply_requests(&before)?;
        if make {
            self.ensure_matte(&segment_id)?;
        }
        Ok(())
    }

    /// Whether `segment_id` with `matting` needs a matte the project lacks.
    /// Refused when it does and the model to make it is not here.
    fn needs_matte(&self, segment_id: &SegmentId, matting: &Matting) -> Result<bool, ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before removing a background".to_owned())
        })?;
        let draft = self.session_draft()?;
        let ask = matte_ask_with(draft, Some(&root), segment_id, matting.clone())
            .map_err(|error| ShellError::Unavailable(error.to_string()))?;
        if find_matte(&root, &ask.key(), &ask.range).is_some() {
            return Ok(false);
        }
        let here = match matting.mode {
            MattingMode::Auto => auto_model_here(),
            MattingMode::Custom => follow_model_here(),
        };
        if !here {
            return Err(ShellError::Unavailable(format!(
                "the {} model is not on this machine",
                match matting.mode {
                    MattingMode::Auto => "Auto removal",
                    MattingMode::Custom => "Custom removal",
                }
            )));
        }
        Ok(true)
    }

    /// Start a job making `segment_id`'s matte when the project lacks it
    /// and none is running.
    fn ensure_matte(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before removing a background".to_owned())
        })?;
        let draft = self.session_draft()?;
        let ask = matte_ask(draft, Some(&root), segment_id)
            .map_err(|error| ShellError::Unavailable(error.to_string()))?;
        if find_matte(&root, &ask.key(), &ask.range).is_some() || self.matte_progress().is_some() {
            return Ok(());
        }
        self.start_matte(root, ask)
    }

    /// Make `ask`'s matte on a job, which redraws the preview when it
    /// lands. The GPU is used when this build has one; the CPU is slower.
    fn start_matte(&mut self, root: PathBuf, ask: MatteAsk) -> Result<(), ShellError> {
        let custom = ask.matting.mode == MattingMode::Custom;
        let (folder, weights) = if custom {
            (FOLLOW_FOLDER, FOLLOW_WEIGHTS)
        } else {
            (AUTO_FOLDER, AUTO_WEIGHTS)
        };
        let dir = model_dir(folder, weights);
        if !dir.join(weights).is_file() {
            return Err(ShellError::Unavailable(format!(
                "the {MATTE_JOB} model is not on this machine"
            )));
        }
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let kernels = Shell::home().cache().join("kernels");
        let compute = Compute::best();
        self.start_job(MATTE_JOB, move |progress| {
            progress.stage("Loading the model");
            if let Err(error) = cutlass_vision::keep_kernels_in(&kernels) {
                cutlass_logger::warn!("kernel cache: {error}");
            }
            let mut tell = |done: f32| {
                progress.fraction(done);
                !progress.cancelled()
            };
            let made = if custom {
                load_follower(&dir, compute)
                    .map_err(Some)
                    .and_then(|mut follower| {
                        progress.stage("Following the subject");
                        follow_clip(
                            &readers,
                            &encoders,
                            follower.as_mut(),
                            &ask,
                            &root,
                            &mut tell,
                        )
                        .map(|_| ())
                        .map_err(stopped_or)
                    })
            } else {
                load_matter(&dir, compute)
                    .map_err(Some)
                    .and_then(|mut matter| {
                        progress.stage("Finding the subject");
                        matte_clip(&readers, &encoders, matter.as_mut(), &ask, &root, &mut tell)
                            .map(|_| ())
                            .map_err(stopped_or)
                    })
            };
            let landing: Landing = Box::new(move |shell: &mut Shell| match made {
                Ok(()) => {
                    shell.preview_cache.borrow_mut().clear();
                    Ok(())
                }
                Err(None) => Ok(()),
                Err(Some(reason)) => {
                    cutlass_logger::error!("{MATTE_JOB}: {reason}");
                    Err(ShellError::Unavailable(format!(
                        "the background could not be removed: {reason}"
                    )))
                }
            });
            landing
        })
    }

    /// Add `pick` to the selected clip's Custom removal, replacing one on
    /// the same frame, and follow the picks through the clip. `before` goes
    /// in the same undo step.
    fn add_pick(
        &mut self,
        segment_id: SegmentId,
        pick: Pick,
        before: Vec<Request>,
    ) -> Result<(), ShellError> {
        let draft = self.session_draft()?;
        let now = super::placed(draft, &segment_id)
            .and_then(|(segment, _)| segment_matting(draft, segment));
        let mut matting = match now {
            Some(matting) if matting.mode == MattingMode::Custom => matting,
            Some(matting) => Matting {
                mode: MattingMode::Custom,
                picks: Vec::new(),
                ..matting
            },
            None => Matting {
                mode: MattingMode::Custom,
                ..Matting::default()
            },
        };
        merge_pick(&mut matting.picks, pick);
        if !matting.is_valid() {
            return Err(ShellError::Unavailable(
                "too many picks on this clip".to_owned(),
            ));
        }
        self.put_matting_with(segment_id, Some(matting), before)
    }

    /// Follow the object being picked through its clip: its clicks become a
    /// Custom removal pick on that frame, and the one-frame selection made
    /// while picking comes off, in one undo step. Picking stops.
    pub fn follow_picked(&mut self) -> Result<(), ShellError> {
        let Some(picking) = self.picking.clone() else {
            return Err(ShellError::Unavailable("not picking an object".to_owned()));
        };
        if picking.clicks.is_empty() {
            self.picking = None;
            return Ok(());
        }
        self.refuse_busy()?;
        let pick = Pick {
            frame: picking.source(),
            clicks: picking
                .clicks
                .iter()
                .map(|click| {
                    let (x, y) = unit_of_clip(click.clip);
                    PickClick {
                        x,
                        y,
                        keep: click.keep,
                    }
                })
                .collect(),
            brushes: Vec::new(),
        };
        let off = Request::SetSelection {
            segment_id: picking.segment_id.clone(),
            material_id: MaterialId::new(self.fresh("selection")),
            selection: None,
        };
        self.add_pick(picking.segment_id.clone(), pick, vec![off])?;
        self.picking = None;
        Ok(())
    }

    /// The strokes being painted, while Paint by hand is on.
    #[must_use]
    pub fn painting(&self) -> Option<&Painting> {
        self.painting.as_ref()
    }

    /// Start painting on the selected clip's frame under the playhead, with
    /// a plain brush that keeps.
    pub fn start_painting(&mut self) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.refuse_locked(&segment_id)?;
        let time = self.playhead();
        let source = {
            let draft = self.session_draft()?;
            let (segment, _) = super::placed(draft, &segment_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            cutlass_engine::placement_of(draft, segment)
                .ok()
                .and_then(|map| cutlass_engine::map_timeline(&map, time).ok())
                .ok_or_else(|| {
                    ShellError::Unavailable("move the playhead onto the clip".to_owned())
                })?
        };
        self.picking = None;
        self.pen = None;
        self.painting = Some(Painting {
            segment_id,
            time,
            source,
            brushes: Vec::new(),
            live: None,
            smart: false,
            keep: true,
            size: 0.3,
            trails: Vec::new(),
            live_trail: None,
        });
        Ok(())
    }

    /// Stop painting, leaving strokes not applied behind.
    pub fn stop_painting(&mut self) {
        self.painting = None;
    }

    /// Choose the brush: `smart` finds edges, `keep` paints in rather than
    /// erasing.
    pub fn set_brush(&mut self, smart: bool, keep: bool) {
        if let Some(painting) = self.painting.as_mut() {
            painting.smart = smart;
            painting.keep = keep;
        }
    }

    /// Set the brush's size, 0 to 1.
    pub fn set_brush_size(&mut self, size: f64) {
        if let Some(painting) = self.painting.as_mut() {
            painting.size = size.clamp(0.0, 1.0);
        }
    }

    /// Where canvas pixel `x`, `y` falls on the painted frame, in unit
    /// coordinates, and the brush's radius in canvas pixels there.
    fn brush_at(&self, x: f64, y: f64) -> Option<([f64; 2], f64)> {
        let painting = self.painting.as_ref()?;
        let draft = self.draft()?;
        let (segment, _) = super::placed(draft, &painting.segment_id)?;
        let canvas = (
            u32::try_from(draft.canvas_config.width.max(1)).unwrap_or(1),
            u32::try_from(draft.canvas_config.height.max(1)).unwrap_or(1),
        );
        let at = |x: f64, y: f64| {
            cutlass_eval::clip_point(draft, segment, painting.time, canvas, x, y).map(unit_of_clip)
        };
        let (u, v) = at(x, y)?;
        // The radius in frame pixels over how many frame pixels one canvas
        // pixel spans there.
        let frame = match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Video(video)) => (
                f64::from(video.width.max(1)),
                f64::from(video.height.max(1)),
            ),
            _ => (1.0, 1.0),
        };
        let span = at(x + 1.0, y).map_or(0.0, |(u1, v1)| {
            ((u1 - u) * frame.0).hypot((v1 - v) * frame.1)
        });
        let on_canvas = if span > 0.0 {
            painting.radius() * frame.0.min(frame.1) / span
        } else {
            0.0
        };
        Some(([u, v], on_canvas))
    }

    /// The brush's radius in canvas pixels at canvas pixel `x`, `y`, while
    /// painting.
    #[must_use]
    pub fn brush_on_canvas(&self, x: f64, y: f64) -> Option<f64> {
        self.brush_at(x, y).map(|(_, radius)| radius)
    }

    /// Paint at canvas pixel `x`, `y`: the first call starts a stroke, and
    /// each later one carries it on until [`Self::end_stroke`].
    pub fn paint_at(&mut self, x: f64, y: f64) -> Result<(), ShellError> {
        if self.painting.is_none() {
            return Err(ShellError::Unavailable("not painting".to_owned()));
        }
        self.session_draft()?;
        let Some((point, on_canvas)) = self.brush_at(x, y) else {
            return Ok(());
        };
        if let Some(painting) = self.painting.as_mut() {
            let radius = painting.radius();
            match (painting.live.as_mut(), painting.live_trail.as_mut()) {
                (Some(live), Some(trail))
                    if live.points.len() < cutlass_effects::matting::MAX_BRUSH_POINTS =>
                {
                    if live.points.last() != Some(&point) {
                        live.points.push(point);
                        trail.points.push((x, y));
                    }
                }
                (Some(_), _) => {}
                (None, _) => {
                    painting.live = Some(Brush {
                        points: vec![point],
                        radius,
                        keep: painting.keep,
                        smart: painting.smart,
                    });
                    painting.live_trail = Some(Trail {
                        points: vec![(x, y)],
                        radius: on_canvas,
                        keep: painting.keep,
                    });
                }
            }
        }
        Ok(())
    }

    /// Finish the stroke being painted.
    pub fn end_stroke(&mut self) {
        if let Some(painting) = self.painting.as_mut() {
            let trail = painting.live_trail.take();
            if let Some(live) = painting.live.take()
                && painting.brushes.len() < cutlass_effects::matting::MAX_BRUSHES
            {
                painting.brushes.push(live);
                painting.trails.push(trail.unwrap_or(Trail {
                    points: Vec::new(),
                    radius: 0.0,
                    keep: true,
                }));
            }
        }
    }

    /// Take the last stroke back.
    pub fn undo_stroke(&mut self) {
        if let Some(painting) = self.painting.as_mut() {
            painting.brushes.pop();
            painting.trails.pop();
        }
    }

    /// Apply the painted strokes: they become a Custom removal pick on the
    /// painted frame, followed through the clip. Painting stops.
    pub fn apply_painting(&mut self) -> Result<(), ShellError> {
        self.end_stroke();
        let Some(painting) = self.painting.clone() else {
            return Err(ShellError::Unavailable("not painting".to_owned()));
        };
        if painting.brushes.is_empty() {
            self.painting = None;
            return Ok(());
        }
        self.refuse_busy()?;
        let pick = Pick {
            frame: painting.source,
            clicks: Vec::new(),
            brushes: painting.brushes,
        };
        self.add_pick(painting.segment_id, pick, Vec::new())?;
        self.painting = None;
        Ok(())
    }
}

/// Map a point of a clip's picture, -1 to 1 with y up, to unit coordinates
/// of its frame, 0 to 1 with y down.
fn unit_of_clip((u, v): (f64, f64)) -> (f64, f64) {
    (
        ((u + 1.0) * 0.5).clamp(0.0, 1.0),
        ((1.0 - v) * 0.5).clamp(0.0, 1.0),
    )
}

/// Put `pick` among `picks` in source time order. A pick on a frame that
/// already has one adds its clicks and strokes to it.
fn merge_pick(picks: &mut Vec<Pick>, pick: Pick) {
    match picks.binary_search_by_key(&pick.frame, |kept| kept.frame) {
        Ok(at) => {
            let kept = &mut picks[at];
            kept.clicks.extend(pick.clicks);
            kept.clicks
                .truncate(cutlass_effects::matting::MAX_PICK_CLICKS);
            kept.brushes.extend(pick.brushes);
            kept.brushes.truncate(cutlass_effects::matting::MAX_BRUSHES);
        }
        Err(at) => picks.insert(at, pick),
    }
}

fn stopped_or(error: EnhanceJobError) -> Option<String> {
    match error {
        EnhanceJobError::Stopped => None,
        error => Some(error.to_string()),
    }
}

#[cfg(test)]
mod tests;
