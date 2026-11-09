//! Auto reframe on the selected clip: store its shape and settings, and
//! find where its subject is as a job when the project has no track for
//! its file yet. The track goes into the project's
//! `Resources/video_reframe`, and the preview and export cut the clip
//! around its subject as soon as the job lands. Another shape or setting,
//! a trim or a split reuses the track, since it covers the whole file.

use cutlass_eval::{ReframeAsk, reframe_ask};
use cutlass_model::{MaterialRef, Reframe, SegmentId};
use cutlass_session::Request;
use cutlass_store::{load_subject_path, save_subject_path};
use cutlass_vision::{Compute, DEPTH_WEIGHTS, DepthModel, EnhanceJobError, find_subjects};

use super::enhance::model_dir;
use super::relight::RELIGHT_FOLDER;
use super::{Landing, Shell, ShellError};

/// The job's label, which the inspector looks for to show its progress.
pub(crate) const REFRAME_JOB: &str = "Auto reframe";

impl Shell {
    /// The selected clip, when it is a movie that can be reframed.
    fn reframe_segment(&self) -> Result<SegmentId, ShellError> {
        let refusal = "only a video clip can be reframed";
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

    /// Cut the selected clip to `ratio`, one of [`Reframe::RATIOS`], around
    /// its subject with `stability` (1 Flexible, 2 Default, 3 Stable) and
    /// `speed` (1 Fast, 2 Default, 3 Slow), or stop with an empty ratio. A
    /// clip whose subject the project has not found yet is searched as a
    /// job first; the setting is stored straight away, in one undo step,
    /// and the cut shows once the job lands.
    pub fn set_reframe(
        &mut self,
        ratio: &str,
        stability: i64,
        speed: i64,
    ) -> Result<(), ShellError> {
        let segment_id = self.reframe_segment()?;
        self.refuse_locked(&segment_id)?;
        if !ratio.is_empty() {
            self.find_subject(&segment_id)?;
        }
        self.apply(Request::SetReframe {
            segment_id,
            ratio: ratio.to_owned(),
            stability,
            speed,
        })
    }

    /// Start a job finding clip `segment_id`'s subject when the project
    /// has no track for it.
    fn find_subject(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before reframing a clip".to_owned())
        })?;
        let draft = self.session_draft()?;
        let ask = reframe_ask(draft, Some(&root), segment_id)
            .map_err(|error| ShellError::Unavailable(error.to_string()))?;
        if self.reframe_progress().is_some() || load_subject_path(&root, &ask.key()).is_some() {
            return Ok(());
        }
        self.start_subjects(root, ask)
    }

    /// Find `ask`'s subject on a job, which keeps the track and redraws the
    /// preview when it lands. The GPU is used when this build has one.
    fn start_subjects(
        &mut self,
        root: std::path::PathBuf,
        ask: ReframeAsk,
    ) -> Result<(), ShellError> {
        let dir = model_dir(RELIGHT_FOLDER, DEPTH_WEIGHTS);
        if !dir.join(DEPTH_WEIGHTS).is_file() {
            return Err(ShellError::Unavailable(format!(
                "the {REFRAME_JOB} model is not on this machine"
            )));
        }
        let readers = self.readers.clone();
        let kernels = Shell::home().cache().join("kernels");
        let compute = Compute::best();
        self.start_job(REFRAME_JOB, move |progress| {
            progress.stage("Loading the model");
            if let Err(error) = cutlass_vision::keep_kernels_in(&kernels) {
                cutlass_logger::warn!("kernel cache: {error}");
            }
            let found = DepthModel::load(&dir, compute)
                .map_err(|error| Some(error.to_string()))
                .and_then(|model| {
                    progress.stage("Processing");
                    find_subjects(&readers, &model, &ask, &mut |done| {
                        progress.fraction(done);
                        !progress.cancelled()
                    })
                    .map_err(|error| match error {
                        EnhanceJobError::Stopped => None,
                        error => Some(error.to_string()),
                    })
                })
                .and_then(|path| {
                    save_subject_path(&root, &ask.key(), &path)
                        .map_err(|error| Some(error.to_string()))
                });
            let landing: Landing = Box::new(move |shell: &mut Shell| match found {
                Ok(()) => {
                    shell.preview_cache.borrow_mut().clear();
                    Ok(())
                }
                Err(None) => Ok(()),
                Err(Some(reason)) => {
                    cutlass_logger::error!("{REFRAME_JOB}: {reason}");
                    Err(ShellError::Unavailable(format!(
                        "the clip's subject could not be found: {reason}"
                    )))
                }
            });
            landing
        })
    }

    /// The selected clip's Auto reframe, `None` when it is off.
    #[must_use]
    pub fn reframe(&self) -> Option<Reframe> {
        let draft = self.draft()?;
        let segment_id = self.selected()?;
        let (segment, _) = super::placed(draft, segment_id)?;
        match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Video(video)) => video.reframe(),
            _ => None,
        }
    }

    /// How far the Auto reframe job is, 0 to 1, while one runs.
    #[must_use]
    pub fn reframe_progress(&self) -> Option<f32> {
        self.job()
            .filter(|job| job.label == REFRAME_JOB)
            .map(|job| job.fraction.unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests;
