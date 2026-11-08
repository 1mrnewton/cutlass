//! Steadying the selected clip: store its level, and measure its camera
//! once as a job when the project keeps no path for it yet. The path is
//! written to the project's `Resources/video_stable` by the job, and the
//! preview draws the clip steadied as soon as it lands. Changing the level
//! of a measured clip plans again without measuring.

use cutlass_eval::{StableAsk, StableError, measure_camera, stable_ask};
use cutlass_model::{MaterialRef, SegmentId, VideoStable};
use cutlass_session::Request;
use cutlass_store::{StableKey, load_camera_path, save_camera_path};

use super::{Landing, Shell, ShellError};

/// The job's label, which the inspector looks for to show its progress.
pub(crate) const STABILIZE_JOB: &str = "Stabilize";

fn key(ask: &StableAsk) -> StableKey {
    StableKey {
        source: ask.source.clone(),
        start: ask.range.start,
        duration: ask.range.duration,
        rate: ask.rate,
    }
}

impl Shell {
    /// The selected clip, when it is a movie that can be steadied.
    fn stable_segment(&self) -> Result<SegmentId, ShellError> {
        let segment_id = self.picture_segment("only a video clip can be steadied")?;
        let draft = self.session_draft()?;
        let movie = super::placed(draft, &segment_id).is_some_and(|(segment, _)| {
            matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(video)) if video.kind != "photo"
            )
        });
        if !movie {
            return Err(ShellError::Unavailable(
                "only a video clip can be steadied".to_owned(),
            ));
        }
        Ok(segment_id)
    }

    /// Steady the selected clip at `level` (1 cuts least, 2 recommended,
    /// 3 most stable), or stop steadying it at 0. A clip the project has
    /// not measured yet is measured as a job first; the level is stored
    /// straight away, in one undo step, and the clip shows steadied once
    /// the job lands.
    pub fn set_stable(&mut self, level: i64) -> Result<(), ShellError> {
        let segment_id = self.stable_segment()?;
        self.refuse_locked(&segment_id)?;
        if level > 0 {
            let root = self.path.clone().ok_or_else(|| {
                ShellError::Unavailable("save the project before steadying a clip".to_owned())
            })?;
            let ask = stable_ask(self.session_draft()?, Some(&root), &segment_id)
                .map_err(|error| ShellError::Unavailable(error.to_string()))?;
            if load_camera_path(&root, &key(&ask)).is_none() {
                let readers = self.readers.clone();
                self.start_job(STABILIZE_JOB, move |progress| {
                    progress.stage("Processing");
                    let measured = measure_camera(&readers, &ask, &mut |done| {
                        progress.fraction(done);
                        !progress.cancelled()
                    })
                    .map_err(|error| match error {
                        StableError::Stopped => None,
                        error => Some(error.to_string()),
                    })
                    .and_then(|path| {
                        save_camera_path(&root, &key(&ask), &path)
                            .map_err(|error| Some(error.to_string()))
                    });
                    let landing: Landing = Box::new(move |shell: &mut Shell| match measured {
                        Ok(()) => {
                            shell.preview_cache.borrow_mut().clear();
                            Ok(())
                        }
                        Err(None) => Ok(()),
                        Err(Some(reason)) => Err(ShellError::Unavailable(format!(
                            "the clip could not be steadied: {reason}"
                        ))),
                    });
                    landing
                })?;
            }
        }
        self.apply(Request::SetStable {
            segment_id,
            level: level.max(0),
        })
    }

    /// The selected clip's level, 0 when it is not steadied.
    #[must_use]
    pub fn stable_level(&self) -> i64 {
        let Some(draft) = self.draft() else {
            return 0;
        };
        let Some(segment_id) = self.selected() else {
            return 0;
        };
        super::placed(draft, segment_id)
            .and_then(
                |(segment, _)| match draft.materials.get(segment.material_id.as_str()) {
                    Some(MaterialRef::Video(video)) => video.stable.as_ref(),
                    _ => None,
                },
            )
            .filter(|stable| stable.is_on())
            .map_or(0, |stable: &VideoStable| stable.level)
    }

    /// How far the steadying job is, 0 to 1, while one runs.
    #[must_use]
    pub fn stable_progress(&self) -> Option<f32> {
        self.job()
            .filter(|job| job.label == STABILIZE_JOB)
            .map(|job| job.fraction.unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests;
