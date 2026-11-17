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
use super::speech_model::{ModelFile, fetch_model};
use super::{Landing, Shell, ShellError};

/// The job's label, which the inspector looks for to show its progress.
pub(crate) const MATTE_JOB: &str = "Remove BG";

/// The model download's label.
const MODEL_JOB: &str = "Download the Remove BG model";

/// The folder in `Models` holding Auto removal's model, and its weights.
pub(super) const AUTO_FOLDER: &str = "matting";
pub(super) const AUTO_WEIGHTS: &str = cutlass_vision::MATTING_WEIGHTS;

/// The folder in `Models` holding Custom removal's follower, and its
/// weights.
pub(super) const FOLLOW_FOLDER: &str = "follow";
pub(super) const FOLLOW_WEIGHTS: &str = cutlass_vision::FOLLOW_WEIGHTS;

/// The square Auto removal's model reads frames at, as it was trained.
/// 768 takes about half the time, but on a busy room it kept a blanket and
/// part of a couch that 1024 takes off.
pub(super) const AUTO_SIDE: u32 = cutlass_vision::MATTING_SIDE;

/// Auto removal's model as BiRefNet_lite's authors publish it (MIT),
/// pinned to a revision and its SHA-256; converted on this machine.
pub(super) const AUTO_DOWNLOAD: ModelFile = ModelFile {
    folder: AUTO_FOLDER,
    file: "birefnet_lite.safetensors",
    url: "https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/\
          aa62cd87eafb9cc43056d08ef3615a14628b831d/model.safetensors",
    sha256: "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815",
    bytes: 177_634_392,
};

/// Custom removal's follower as Meta publishes it (Apache-2.0), pinned the
/// same way.
pub(super) const FOLLOW_DOWNLOAD: ModelFile = ModelFile {
    folder: FOLLOW_FOLDER,
    file: "sam2.1_hiera_tiny.safetensors",
    url: "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/\
          de431c4043854a71d8101e17995dfe596bf101a5/model.safetensors",
    sha256: "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7",
    bytes: 155_908_064,
};

/// What the page says before a Remove BG model downloads; the button
/// under it is the yes.
#[must_use]
pub fn matte_model_question(custom: bool) -> &'static str {
    if custom {
        "Following an object through the clip needs SAM 2.1 tiny (156 MB, from Hugging \
         Face, Apache-2.0), downloaded into Cutlass's Models folder."
    } else {
        "Auto removal needs BiRefNet_lite (178 MB, from Hugging Face, MIT), downloaded \
         into Cutlass's Models folder."
    }
}

/// Turn the downloaded `published` weights into the `.bpk` the model loads,
/// in `folder`, and drop the download.
fn install_weights(published: &Path, folder: &Path, custom: bool) -> Result<(), String> {
    let converted = if custom {
        cutlass_vision::convert_sam2(published, &folder.join(FOLLOW_WEIGHTS))
    } else {
        cutlass_vision::convert_birefnet(published, &folder.join(AUTO_WEIGHTS))
    };
    converted.map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(published);
    Ok(())
}

/// The smallest and largest brush, as a share of the frame's short side.
const BRUSH_SMALLEST: f64 = 0.005;
const BRUSH_LARGEST: f64 = 0.1;

/// Auto removal's model, BiRefNet_lite, loaded from `dir`.
fn load_matter(dir: &Path, compute: Compute) -> Result<Box<dyn Matter + Send>, String> {
    let model = cutlass_vision::Matting::load_at(dir, compute, AUTO_SIDE)
        .map_err(|error| error.to_string())?;
    Ok(Box::new(model))
}

/// Custom removal's follower, SAM 2.1 tiny, loaded from `dir`.
fn load_follower(dir: &Path, compute: Compute) -> Result<Box<dyn Follower + Send>, String> {
    let model =
        cutlass_vision::FollowModel::load(dir, compute).map_err(|error| error.to_string())?;
    Ok(Box::new(model))
}

/// Add the matte each clip in `draft` and its compound clips draws with to
/// `keep`. `false` when that cannot be told for some clip.
fn used_mattes(draft: &cutlass_model::Draft, root: &Path, keep: &mut Vec<PathBuf>) -> bool {
    for segment in draft.tracks.iter().flat_map(|track| &track.segments) {
        let Some(matting) = segment_matting(draft, segment) else {
            continue;
        };
        if !matting.ready() {
            continue;
        }
        // A file that is gone may not even name a matte: keep them all.
        let Ok(ask) = matte_ask(draft, Some(root), &segment.id) else {
            return false;
        };
        if !ask.source.is_file() {
            return false;
        }
        if let Some((path, _)) = find_matte(root, &ask.key(), &ask.range) {
            keep.push(path);
        }
    }
    draft
        .materials
        .drafts
        .iter()
        .all(|compound| match &compound.draft {
            Some(inner) => used_mattes(inner, root, keep),
            None => compound.draft_file_path.is_empty(),
        })
}

/// One Custom removal pick as the page lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickRow {
    /// The source time of the picked frame.
    pub frame: Micros,
    /// Where the timeline shows that frame, when it does.
    pub at: Option<Micros>,
    pub clicks: usize,
    pub strokes: usize,
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
    /// The selected clip, when it is a movie or a photo that can be cut
    /// out.
    fn matting_segment(&self) -> Result<SegmentId, ShellError> {
        let refusal = "only a video or photo clip can have its background removed";
        let segment_id = self.picture_segment(refusal)?;
        let draft = self.session_draft()?;
        let picture = super::placed(draft, &segment_id).is_some_and(|(segment, _)| {
            matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(_))
            )
        });
        if !picture {
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

    /// How far a Remove BG model's download is, 0 to 1, while one runs.
    #[must_use]
    pub fn matte_model_progress(&self) -> Option<f32> {
        self.job()
            .filter(|job| job.label == MODEL_JOB)
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
        // Asked for by hand: a matte that failed before is tried again.
        if let (Some(draft), Some(root)) = (self.draft(), self.path.as_deref())
            && let Ok(ask) = matte_ask(draft, Some(root), &segment_id)
        {
            self.matte_failed.remove(&failure_key(&ask));
        }
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
        if find_matte(&root, &ask.key(), &ask.range).is_some()
            || self.matte_progress().is_some()
            || self.matte_failed.contains(&failure_key(&ask))
        {
            return Ok(());
        }
        self.start_matte(root, ask)
    }

    /// Download Remove BG's model, Custom removal's follower when `custom`,
    /// as a job: fetched, checked, and converted for this machine. Ask the
    /// user first ([`matte_model_question`]).
    pub fn download_matte_model(&mut self, custom: bool) -> Result<(), ShellError> {
        let here = if custom {
            follow_model_here()
        } else {
            auto_model_here()
        };
        if here {
            return Ok(());
        }
        let models = Shell::home().models();
        let model = if custom {
            &FOLLOW_DOWNLOAD
        } else {
            &AUTO_DOWNLOAD
        };
        self.start_job(MODEL_JOB, move |progress| {
            progress.stage("Downloading the model");
            let result = fetch_model(model, &models, progress).and_then(|published| {
                progress.stage("Preparing the model");
                install_weights(&published, &models.join(model.folder), custom)
            });
            Box::new(move |shell: &mut Shell| match result {
                Ok(()) => {
                    shell.set_notice("The Remove BG model is ready");
                    // A clip that waited for the model gets its matte.
                    shell.matte_swept = None;
                    Ok(())
                }
                Err(message) => Err(ShellError::Unavailable(format!(
                    "the Remove BG model did not download: {message}"
                ))),
            })
        })
    }

    /// Start the matte of the first clip with Remove BG the project lacks
    /// one for, when no job runs: after a wider trim, an edit from the
    /// assistant, or a project opened on another machine. Looked for once a
    /// revision; a matte whose job failed is not tried again.
    pub(super) fn sweep_mattes(&mut self) {
        if self.job.is_some() || self.picking.is_some() || self.painting.is_some() {
            return;
        }
        let (Some(revision), Some(root)) = (self.revision(), self.path.clone()) else {
            return;
        };
        if self.matte_swept == Some(revision) {
            return;
        }
        self.matte_swept = Some(revision);
        let Some(ask) = self.missing_matte(&root) else {
            return;
        };
        let tried = failure_key(&ask);
        if let Err(error) = self.start_matte(root, ask) {
            self.matte_failed.insert(tried);
            cutlass_logger::warn!("{MATTE_JOB}: {error}");
        }
    }

    /// The first clip's matte the project lacks whose model is here.
    fn missing_matte(&self, root: &Path) -> Option<MatteAsk> {
        let draft = self.draft()?;
        draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter_map(|segment| {
                let matting = segment_matting(draft, segment)?;
                let here = match matting.mode {
                    MattingMode::Auto => auto_model_here(),
                    MattingMode::Custom => follow_model_here(),
                };
                if !here {
                    return None;
                }
                matte_ask(draft, Some(root), &segment.id).ok()
            })
            .find(|ask| {
                find_matte(root, &ask.key(), &ask.range).is_none()
                    && !self.matte_failed.contains(&failure_key(ask))
            })
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
        let tried = failure_key(&ask);
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
                    shell.matte_failed.insert(tried);
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

    /// Take out the project's mattes that no clip draws with now, inside
    /// compound clips too: those left by earlier trims, picks and models.
    /// Done when a project opens, since undo does not reach back past
    /// that; a matte needed again is made again. When a clip's file cannot
    /// be read or a compound clip's own draft is not in the project, no
    /// matte is taken out, so one that only waits for its file stays.
    pub(super) fn prune_mattes(&self) {
        let (Some(draft), Some(root)) = (self.draft(), self.path.as_deref()) else {
            return;
        };
        let mut keep = Vec::new();
        if !used_mattes(draft, root, &mut keep) {
            return;
        }
        match cutlass_store::remove_mattes_but(root, &keep) {
            Ok(0) => {}
            Ok(removed) => cutlass_logger::info!("took out {removed} unused mattes"),
            Err(error) => cutlass_logger::warn!("unused mattes stay: {error}"),
        }
    }

    /// The selected clip's Custom removal picks in the order they are in
    /// the clip: the frame each is on, where the timeline shows it (`None`
    /// when a trim or speed change left it out), and how much is on it.
    #[must_use]
    pub fn picks_listed(&self) -> Vec<PickRow> {
        let (Some(draft), Some(segment_id)) = (self.draft(), self.selected()) else {
            return Vec::new();
        };
        let Some((segment, _)) = super::placed(draft, segment_id) else {
            return Vec::new();
        };
        let Some(matting) =
            segment_matting(draft, segment).filter(|matting| matting.mode == MattingMode::Custom)
        else {
            return Vec::new();
        };
        let map = cutlass_engine::placement_of(draft, segment).ok();
        matting
            .picks
            .iter()
            .map(|pick| PickRow {
                frame: pick.frame,
                at: map
                    .as_ref()
                    .and_then(|map| cutlass_engine::timeline_for_source(map, pick.frame)),
                clicks: pick.clicks.len(),
                strokes: pick.brushes.len(),
            })
            .collect()
    }

    /// Take the selected clip's pick on source frame `frame` out, as one
    /// undo step. Its matte is made again from the picks left; with none
    /// left, Custom removal comes off.
    pub fn remove_pick(&mut self, frame: Micros) -> Result<(), ShellError> {
        let segment_id = self.matting_segment()?;
        self.refuse_locked(&segment_id)?;
        let Some(mut matting) = self
            .matting()
            .filter(|matting| matting.mode == MattingMode::Custom)
        else {
            return Err(ShellError::Unavailable(
                "the clip has no Custom removal".to_owned(),
            ));
        };
        let before = matting.picks.len();
        matting.picks.retain(|pick| pick.frame != frame);
        if matting.picks.len() == before {
            return Err(ShellError::Unavailable("no pick on that frame".to_owned()));
        }
        let matting = (!matting.picks.is_empty()).then_some(matting);
        self.put_matting(segment_id, matting)
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

/// What a matte is made from, as one string, to remember a failed job by.
fn failure_key(ask: &MatteAsk) -> String {
    format!("{:?} {:?}", ask.key(), ask.range)
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
