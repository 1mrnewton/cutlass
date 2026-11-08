//! Enhance quality, Reduce image noise and Optical flow on the selected
//! clip, and Optical flow on every clip at once: store the level, and make
//! the clip's copy as a job when the project has none for it yet. A clip
//! with more than one on has one copy, denoised, filled, then enhanced.
//! The copy goes into the project's `Resources/video_enhance`,
//! `Resources/video_denoise` when it is denoised but not enhanced, or
//! `Resources/video_interpolate` when it is only filled, and the preview
//! and export draw the clip from it as soon as the job lands. Trimming or
//! splitting the clip keeps its copy; a wider trim, another level or, with
//! Optical flow, another speed makes a new one.

use std::path::PathBuf;

use cutlass_eval::{EnhanceAsk, enhance_ask};
use cutlass_model::{Algorithm, Draft, MaterialRef, SegmentId, TrackAttribute};
use cutlass_session::Request;
use cutlass_store::find_enhanced;
use cutlass_vision::{
    Compute, CopyModels, DENOISE_WEIGHTS, Denoiser, ENHANCE_WEIGHTS, EnhanceJobError, Enhancer,
    INTERPOLATE_WEIGHTS, Interpolator, enhance_clip,
};

use super::{Landing, Shell, ShellError};

/// The jobs' labels, which the inspector looks for to show their progress.
/// A copy that is enhanced runs as the first, one denoised but not
/// enhanced as the second, one only filled, and Apply to all, as the
/// third.
pub(crate) const ENHANCE_JOB: &str = "Enhance quality";
pub(crate) const DENOISE_JOB: &str = "Reduce image noise";
pub(crate) const INTERPOLATE_JOB: &str = "Optical flow";

/// The folders in `Models` holding each model.
const ENHANCE_FOLDER: &str = "enhance";
const DENOISE_FOLDER: &str = "denoise";
const INTERPOLATE_FOLDER: &str = "interpolate";

/// The folder holding a model: the one in `Models`, or in a debug build
/// the checkout's `local-files` copy when `Models` has none.
pub(super) fn model_dir(folder: &str, weights: &str) -> PathBuf {
    let installed = Shell::home().models().join(folder);
    if installed.join(weights).is_file() {
        return installed;
    }
    #[cfg(debug_assertions)]
    {
        let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/models")
            .join(folder);
        if checkout.join(weights).is_file() {
            return checkout;
        }
    }
    installed
}

/// Which of the levels a change sets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Copied {
    Enhance,
    Denoise,
    Interpolate,
}

/// A clip's three levels: Enhance quality, Reduce image noise and the
/// Optical flow frame rate.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Levels {
    enhance: i64,
    denoise: i64,
    interpolate: i64,
}

impl Levels {
    fn any(self) -> bool {
        self.enhance > 0 || self.denoise > 0 || self.interpolate > 0
    }
}

/// Clip `segment_id`'s levels in `draft`, all 0 when it is not a movie.
fn levels_of(draft: &Draft, segment_id: &SegmentId) -> Levels {
    let Some((segment, _)) = super::placed(draft, segment_id) else {
        return Levels::default();
    };
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return Levels::default();
    };
    let level = |algorithm: Option<&Algorithm>| algorithm.map_or(0, |algorithm| algorithm.level);
    Levels {
        enhance: level(video.enhance()),
        denoise: level(video.denoise()),
        interpolate: level(video.interpolate()),
    }
}

/// Every movie clip on an unlocked lane, in lane order.
fn movie_clips(draft: &Draft) -> Vec<SegmentId> {
    draft
        .tracks
        .iter()
        .filter(|track| !track.attribute.contains(TrackAttribute::LOCK))
        .flat_map(|track| &track.segments)
        .filter(|segment| {
            matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(video)) if video.kind != "photo"
            )
        })
        .map(|segment| segment.id.clone())
        .collect()
}

impl Shell {
    /// The selected clip, when it is a movie that can be enhanced.
    fn enhance_segment(&self) -> Result<SegmentId, ShellError> {
        let refusal = "only a video clip can be enhanced or denoised";
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

    /// Enhance the selected clip at `level` (1 HD, 2 UHD), or stop at 0. A
    /// clip with no copy yet is enhanced as a job first; the level is
    /// stored straight away, in one undo step, and the clip shows enhanced
    /// once the job lands.
    pub fn set_enhance(&mut self, level: i64) -> Result<(), ShellError> {
        self.set_copied(Copied::Enhance, level)
    }

    /// Reduce the selected clip's noise at `level` (1 Weak, 2 Strong), or
    /// stop at 0, as [`Self::set_enhance`] does.
    pub fn set_denoise(&mut self, level: i64) -> Result<(), ShellError> {
        self.set_copied(Copied::Denoise, level)
    }

    /// Fill the selected clip with frames made between its own to `fps`
    /// frames a second (30, 50 or 60) however slow it plays, or stop at 0,
    /// as [`Self::set_enhance`] does.
    pub fn set_interpolate(&mut self, fps: i64) -> Result<(), ShellError> {
        self.set_copied(Copied::Interpolate, fps)
    }

    /// Optical flow at `fps`, or off at 0, on every movie clip on an
    /// unlocked lane, in one undo step, and one job making the copies the
    /// project lacks. A clip whose copy cannot be asked for, such as one
    /// whose file is gone, still takes the setting and draws its own file.
    pub fn set_interpolate_all(&mut self, fps: i64) -> Result<(), ShellError> {
        let fps = fps.max(0);
        if fps > 0 && !Algorithm::FRAME_RATES.contains(&fps) {
            return Err(ShellError::Edit(cutlass_session::EditError::FlowRate));
        }
        let draft = self.session_draft()?;
        let clips = movie_clips(draft);
        if clips.is_empty() {
            return Err(ShellError::Unavailable(
                "the project has no video clip to fill".to_owned(),
            ));
        }
        let wanted: Vec<(SegmentId, Levels)> = clips
            .iter()
            .map(|id| {
                let levels = Levels {
                    interpolate: fps,
                    ..levels_of(draft, id)
                };
                (id.clone(), levels)
            })
            .collect();
        let made = self.make_copies(&wanted);
        if fps > 0 {
            made?;
        } else if let Err(error) = made {
            cutlass_logger::warn!("the clips' copies were not started: {error}");
        }
        let requests: Vec<Request> = clips
            .into_iter()
            .map(|segment_id| Request::SetInterpolate { segment_id, fps })
            .collect();
        self.apply_requests(&requests)
    }

    fn set_copied(&mut self, which: Copied, level: i64) -> Result<(), ShellError> {
        let segment_id = self.enhance_segment()?;
        self.refuse_locked(&segment_id)?;
        let level = level.max(0);
        if which == Copied::Interpolate && level > 0 && !Algorithm::FRAME_RATES.contains(&level) {
            return Err(ShellError::Edit(cutlass_session::EditError::FlowRate));
        }
        let mut levels = levels_of(self.session_draft()?, &segment_id);
        match which {
            Copied::Enhance => levels.enhance = level,
            Copied::Denoise => levels.denoise = level,
            Copied::Interpolate => levels.interpolate = level,
        }
        if levels.any() {
            let made = self.make_copies(&[(segment_id.clone(), levels)]);
            // Turning one off never fails: the clip draws its own file
            // until the copy of what stays on is made.
            if level > 0 {
                made?;
            } else if let Err(error) = made {
                cutlass_logger::warn!("the clip's copy was not started: {error}");
            }
        }
        self.apply(match which {
            Copied::Enhance => Request::SetEnhance { segment_id, level },
            Copied::Denoise => Request::SetDenoise { segment_id, level },
            Copied::Interpolate => Request::SetInterpolate {
                segment_id,
                fps: level,
            },
        })
    }

    /// Start one job making each clip's copy at its levels, for the clips
    /// with a level on whose copy the project lacks. One clip is refused
    /// when its copy cannot be asked for; of several, such a clip is left
    /// out, and refused only when none is left.
    fn make_copies(&mut self, clips: &[(SegmentId, Levels)]) -> Result<(), ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before enhancing a clip".to_owned())
        })?;
        let draft = self.session_draft()?;
        let wanted: Vec<_> = clips.iter().filter(|(_, levels)| levels.any()).collect();
        let mut asks: Vec<EnhanceAsk> = Vec::new();
        let (mut asked, mut refused) = (0, None);
        for (segment_id, levels) in &wanted {
            let ask = enhance_ask(
                draft,
                Some(&root),
                segment_id,
                levels.enhance,
                levels.denoise,
                levels.interpolate,
            );
            match ask {
                Ok(ask) => {
                    asked += 1;
                    if !asks.contains(&ask)
                        && find_enhanced(&root, &ask.key(), &ask.range).is_none()
                    {
                        asks.push(ask);
                    }
                }
                Err(error) => {
                    cutlass_logger::warn!("{segment_id}: {error}");
                    refused = Some(ShellError::Unavailable(error.to_string()));
                }
            }
        }
        if let Some(error) = refused
            && (wanted.len() == 1 || asked == 0)
        {
            return Err(error);
        }
        if asks.is_empty() {
            return Ok(());
        }
        self.start_copies(root, asks)
    }

    /// Make each of `asks`' copies on one job, which redraws the preview
    /// when it lands.
    fn start_copies(&mut self, root: PathBuf, asks: Vec<EnhanceAsk>) -> Result<(), ShellError> {
        let label = match asks.as_slice() {
            [ask] if ask.level > 0 => ENHANCE_JOB,
            [ask] if ask.denoise > 0 => DENOISE_JOB,
            _ => INTERPOLATE_JOB,
        };
        let enhance = asks.iter().any(|ask| ask.level > 0);
        let denoise = asks.iter().any(|ask| ask.denoise > 0);
        let interpolate = asks.iter().any(|ask| ask.interpolate > 0);
        if Compute::best() != Compute::Gpu {
            return Err(ShellError::Unavailable(format!(
                "{label} needs a GPU this build can use"
            )));
        }
        let enhance_dir = model_dir(ENHANCE_FOLDER, ENHANCE_WEIGHTS);
        let denoise_dir = model_dir(DENOISE_FOLDER, DENOISE_WEIGHTS);
        let interpolate_dir = model_dir(INTERPOLATE_FOLDER, INTERPOLATE_WEIGHTS);
        let missing = |name: &str| {
            ShellError::Unavailable(format!("the {name} model is not on this machine"))
        };
        if enhance && !enhance_dir.join(ENHANCE_WEIGHTS).is_file() {
            return Err(missing(ENHANCE_JOB));
        }
        if denoise && !denoise_dir.join(DENOISE_WEIGHTS).is_file() {
            return Err(missing(DENOISE_JOB));
        }
        if interpolate && !interpolate_dir.join(INTERPOLATE_WEIGHTS).is_file() {
            return Err(missing(INTERPOLATE_JOB));
        }
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let kernels = Shell::home().cache().join("kernels");
        self.start_job(label, move |progress| {
            progress.stage("Loading the model");
            if let Err(error) = cutlass_vision::keep_kernels_in(&kernels) {
                cutlass_logger::warn!("kernel cache: {error}");
            }
            let loaded = (|| {
                let enhancer = enhance
                    .then(|| Enhancer::load(&enhance_dir, Compute::Gpu))
                    .transpose()?;
                let denoiser = denoise
                    .then(|| Denoiser::load(&denoise_dir, Compute::Gpu))
                    .transpose()?;
                let interpolator = interpolate
                    .then(|| Interpolator::load(&interpolate_dir, Compute::Gpu))
                    .transpose()?;
                Ok::<_, cutlass_vision::VisionError>((enhancer, denoiser, interpolator))
            })();
            let made = loaded.map_err(|error| Some(error.to_string())).and_then(
                |(enhancer, denoiser, interpolator)| {
                    progress.stage("Processing");
                    let models = CopyModels {
                        enhancer: enhancer.as_ref(),
                        denoiser: denoiser.as_ref(),
                        interpolator: interpolator.as_ref(),
                    };
                    let share = 1.0 / asks.len() as f32;
                    for (index, ask) in asks.iter().enumerate() {
                        enhance_clip(&readers, &encoders, models, ask, &root, &mut |done| {
                            progress.fraction((index as f32 + done) * share);
                            !progress.cancelled()
                        })
                        .map_err(|error| match error {
                            EnhanceJobError::Stopped => None,
                            error => Some(error.to_string()),
                        })?;
                    }
                    Ok(())
                },
            );
            let landing: Landing = Box::new(move |shell: &mut Shell| match made {
                Ok(()) => {
                    shell.preview_cache.borrow_mut().clear();
                    Ok(())
                }
                Err(None) => Ok(()),
                Err(Some(reason)) => {
                    cutlass_logger::error!("{label}: {reason}");
                    Err(ShellError::Unavailable(format!(
                        "the clip's copy could not be made: {reason}"
                    )))
                }
            });
            landing
        })
    }

    /// Make clip `segment_id`'s copy for its speed when Optical flow is on
    /// and the project lacks it. The speed is already stored, so a copy
    /// that cannot start is only logged; the clip draws its own file.
    pub(super) fn refill_copy(&mut self, segment_id: &SegmentId) {
        let Some(draft) = self.draft() else {
            return;
        };
        let levels = levels_of(draft, segment_id);
        if levels.interpolate <= 0 {
            return;
        }
        if let Err(error) = self.make_copies(&[(segment_id.clone(), levels)]) {
            cutlass_logger::warn!("the clip's copy for its new speed was not started: {error}");
        }
    }

    /// The selected clip's algorithm `kind` level, 0 when it is off.
    fn copied_level(&self, kind: &str) -> i64 {
        let Some(draft) = self.draft() else {
            return 0;
        };
        let Some(segment_id) = self.selected() else {
            return 0;
        };
        super::placed(draft, segment_id)
            .and_then(
                |(segment, _)| match draft.materials.get(segment.material_id.as_str()) {
                    Some(MaterialRef::Video(video)) => video.algorithm(kind),
                    _ => None,
                },
            )
            .map_or(0, |algorithm: &Algorithm| algorithm.level)
    }

    /// The selected clip's Enhance quality level, 0 when it is off.
    #[must_use]
    pub fn enhance_level(&self) -> i64 {
        self.copied_level(Algorithm::QUALITY_ENHANCE)
    }

    /// The selected clip's Reduce image noise level, 0 when it is off.
    #[must_use]
    pub fn denoise_level(&self) -> i64 {
        self.copied_level(Algorithm::NOISE_REDUCTION)
    }

    /// The selected clip's Optical flow frame rate, 0 when it is off.
    #[must_use]
    pub fn interpolate_level(&self) -> i64 {
        self.copied_level(Algorithm::COMPLEMENT_FRAME)
    }

    /// How far the job labelled `label` is, 0 to 1, while it runs.
    fn copy_progress(&self, label: &str) -> Option<f32> {
        self.job()
            .filter(|job| job.label == label)
            .map(|job| job.fraction.unwrap_or(0.0))
    }

    /// How far the enhance job is, 0 to 1, while one runs.
    #[must_use]
    pub fn enhance_progress(&self) -> Option<f32> {
        self.copy_progress(ENHANCE_JOB)
    }

    /// How far a job making a copy that is only denoised is, 0 to 1.
    #[must_use]
    pub fn denoise_progress(&self) -> Option<f32> {
        self.copy_progress(DENOISE_JOB)
    }

    /// How far a job filling clips that are not denoised or enhanced, or
    /// filling every clip, is, 0 to 1.
    #[must_use]
    pub fn interpolate_progress(&self) -> Option<f32> {
        self.copy_progress(INTERPOLATE_JOB)
    }
}

#[cfg(test)]
mod tests;
