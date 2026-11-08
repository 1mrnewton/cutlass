//! Enhance quality and Reduce image noise on the selected clip: store the
//! level, and make the clip's copy as a job when the project has none for
//! it yet. A clip with both on has one copy, denoised then enhanced. The
//! copy goes into the project's `Resources/video_enhance`, or
//! `Resources/video_denoise` when it is only denoised, and the preview and
//! export draw the clip from it as soon as the job lands. Trimming or
//! splitting the clip keeps its copy; a wider trim or another level makes
//! a new one.

use std::path::PathBuf;

use cutlass_eval::{EnhanceAsk, enhance_ask};
use cutlass_model::{Algorithm, MaterialRef, SegmentId};
use cutlass_session::Request;
use cutlass_store::find_enhanced;
use cutlass_vision::{
    Compute, CopyModels, DENOISE_WEIGHTS, Denoiser, ENHANCE_WEIGHTS, EnhanceJobError, Enhancer,
    enhance_clip,
};

use super::{Landing, Shell, ShellError};

/// The jobs' labels, which the inspector looks for to show their progress.
/// A copy that is enhanced runs as the first, one only denoised as the
/// second.
pub(crate) const ENHANCE_JOB: &str = "Enhance quality";
pub(crate) const DENOISE_JOB: &str = "Reduce image noise";

/// The folders in `Models` holding each model.
const ENHANCE_FOLDER: &str = "enhance";
const DENOISE_FOLDER: &str = "denoise";

/// The folder holding a model: the one in `Models`, or in a debug build
/// the checkout's `local-files` copy when `Models` has none.
fn model_dir(folder: &str, weights: &str) -> PathBuf {
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

/// Which of the two levels a change sets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Copied {
    Enhance,
    Denoise,
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

    fn set_copied(&mut self, which: Copied, level: i64) -> Result<(), ShellError> {
        let segment_id = self.enhance_segment()?;
        self.refuse_locked(&segment_id)?;
        let level = level.max(0);
        let (mut enhance, mut denoise) = (self.enhance_level(), self.denoise_level());
        match which {
            Copied::Enhance => enhance = level,
            Copied::Denoise => denoise = level,
        }
        if enhance > 0 || denoise > 0 {
            let made = self.make_copy(&segment_id, enhance, denoise);
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
        })
    }

    /// Start the job making the clip's copy at these levels, unless the
    /// project has it.
    fn make_copy(
        &mut self,
        segment_id: &SegmentId,
        enhance: i64,
        denoise: i64,
    ) -> Result<(), ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before enhancing a clip".to_owned())
        })?;
        let ask = enhance_ask(
            self.session_draft()?,
            Some(&root),
            segment_id,
            enhance,
            denoise,
        )
        .map_err(|error| ShellError::Unavailable(error.to_string()))?;
        if find_enhanced(&root, &ask.key(), &ask.range).is_some() {
            return Ok(());
        }
        self.start_copy(root, ask)
    }

    /// Make `ask`'s copy on a job, which redraws the preview when it lands.
    fn start_copy(&mut self, root: PathBuf, ask: EnhanceAsk) -> Result<(), ShellError> {
        let label = if ask.level > 0 {
            ENHANCE_JOB
        } else {
            DENOISE_JOB
        };
        if Compute::best() != Compute::Gpu {
            return Err(ShellError::Unavailable(format!(
                "{label} needs a GPU this build can use"
            )));
        }
        let enhance_dir = model_dir(ENHANCE_FOLDER, ENHANCE_WEIGHTS);
        let denoise_dir = model_dir(DENOISE_FOLDER, DENOISE_WEIGHTS);
        let missing = |name: &str| {
            ShellError::Unavailable(format!("the {name} model is not on this machine"))
        };
        if ask.level > 0 && !enhance_dir.join(ENHANCE_WEIGHTS).is_file() {
            return Err(missing(ENHANCE_JOB));
        }
        if ask.denoise > 0 && !denoise_dir.join(DENOISE_WEIGHTS).is_file() {
            return Err(missing(DENOISE_JOB));
        }
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let kernels = Shell::home().cache().join("kernels");
        self.start_job(label, move |progress| {
            progress.stage("Loading the model");
            if let Err(error) = cutlass_vision::keep_kernels_in(&kernels) {
                cutlass_logger::warn!("kernel cache: {error}");
            }
            let enhancer = (ask.level > 0)
                .then(|| Enhancer::load(&enhance_dir, Compute::Gpu))
                .transpose();
            let denoiser = (ask.denoise > 0)
                .then(|| Denoiser::load(&denoise_dir, Compute::Gpu))
                .transpose();
            let made = enhancer
                .and_then(|enhancer| denoiser.map(|denoiser| (enhancer, denoiser)))
                .map_err(|error| Some(error.to_string()))
                .and_then(|(enhancer, denoiser)| {
                    progress.stage("Processing");
                    let models = CopyModels {
                        enhancer: enhancer.as_ref(),
                        denoiser: denoiser.as_ref(),
                    };
                    enhance_clip(&readers, &encoders, models, &ask, &root, &mut |done| {
                        progress.fraction(done);
                        !progress.cancelled()
                    })
                    .map_err(|error| match error {
                        EnhanceJobError::Stopped => None,
                        error => Some(error.to_string()),
                    })
                });
            let landing: Landing = Box::new(move |shell: &mut Shell| match made {
                Ok(_) => {
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
}

#[cfg(test)]
mod tests;
