//! Enhance quality on the selected clip: store its level, and make the
//! clip's enhanced copy as a job when the project has none for it yet.
//! The copy goes into the project's `Resources/video_enhance`, and the
//! preview and export draw the clip from it as soon as the job lands.
//! Trimming or splitting an enhanced clip keeps its copy; a wider trim or
//! another level makes a new one.

use std::path::PathBuf;

use cutlass_eval::{EnhanceAsk, enhance_ask};
use cutlass_model::{Algorithm, MaterialRef, SegmentId};
use cutlass_session::Request;
use cutlass_store::find_enhanced;
use cutlass_vision::{Compute, ENHANCE_WEIGHTS, EnhanceJobError, Enhancer, enhance_clip};

use super::{Landing, Shell, ShellError};

/// The job's label, which the inspector looks for to show its progress.
pub(crate) const ENHANCE_JOB: &str = "Enhance quality";

/// The folder in `Models` holding the Enhance quality model.
const ENHANCE_FOLDER: &str = "enhance";

/// The folder holding the Enhance quality model: the one in `Models`, or in
/// a debug build the checkout's `local-files` copy when `Models` has none.
fn enhance_model_dir() -> PathBuf {
    let installed = Shell::home().models().join(ENHANCE_FOLDER);
    if installed.join(ENHANCE_WEIGHTS).is_file() {
        return installed;
    }
    #[cfg(debug_assertions)]
    {
        let checkout =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/enhance");
        if checkout.join(ENHANCE_WEIGHTS).is_file() {
            return checkout;
        }
    }
    installed
}

impl Shell {
    /// The selected clip, when it is a movie that can be enhanced.
    fn enhance_segment(&self) -> Result<SegmentId, ShellError> {
        let refusal = "only a video clip can be enhanced";
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
        let segment_id = self.enhance_segment()?;
        self.refuse_locked(&segment_id)?;
        if level > 0 {
            let root = self.path.clone().ok_or_else(|| {
                ShellError::Unavailable("save the project before enhancing a clip".to_owned())
            })?;
            let ask = enhance_ask(self.session_draft()?, Some(&root), &segment_id, level, 0)
                .map_err(|error| ShellError::Unavailable(error.to_string()))?;
            if find_enhanced(&root, &ask.key(), &ask.range).is_none() {
                self.start_enhance(root, ask)?;
            }
        }
        self.apply(Request::SetEnhance {
            segment_id,
            level: level.max(0),
        })
    }

    /// Make `ask`'s copy on a job, which redraws the preview when it lands.
    fn start_enhance(&mut self, root: PathBuf, ask: EnhanceAsk) -> Result<(), ShellError> {
        if Compute::best() != Compute::Gpu {
            return Err(ShellError::Unavailable(
                "Enhance quality needs a GPU this build can use".to_owned(),
            ));
        }
        let weights = enhance_model_dir();
        if !weights.join(ENHANCE_WEIGHTS).is_file() {
            return Err(ShellError::Unavailable(
                "the Enhance quality model is not on this machine".to_owned(),
            ));
        }
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let kernels = Shell::home().cache().join("kernels");
        self.start_job(ENHANCE_JOB, move |progress| {
            progress.stage("Loading the model");
            if let Err(error) = cutlass_vision::keep_kernels_in(&kernels) {
                cutlass_logger::warn!("kernel cache: {error}");
            }
            let made = Enhancer::load(&weights, Compute::Gpu)
                .map_err(|error| Some(error.to_string()))
                .and_then(|enhancer| {
                    progress.stage("Processing");
                    enhance_clip(&readers, &encoders, &enhancer, &ask, &root, &mut |done| {
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
                    cutlass_logger::error!("enhance quality: {reason}");
                    Err(ShellError::Unavailable(format!(
                        "the clip could not be enhanced: {reason}"
                    )))
                }
            });
            landing
        })
    }

    /// The selected clip's Enhance quality level, 0 when it is off.
    #[must_use]
    pub fn enhance_level(&self) -> i64 {
        let Some(draft) = self.draft() else {
            return 0;
        };
        let Some(segment_id) = self.selected() else {
            return 0;
        };
        super::placed(draft, segment_id)
            .and_then(
                |(segment, _)| match draft.materials.get(segment.material_id.as_str()) {
                    Some(MaterialRef::Video(video)) => video.enhance(),
                    _ => None,
                },
            )
            .map_or(0, |enhance: &Algorithm| enhance.level)
    }

    /// How far the enhance job is, 0 to 1, while one runs.
    #[must_use]
    pub fn enhance_progress(&self) -> Option<f32> {
        self.job()
            .filter(|job| job.label == ENHANCE_JOB)
            .map(|job| job.fraction.unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests;
