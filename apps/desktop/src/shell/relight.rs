//! Relight on the selected clip: store its light, and guess the depth of
//! its frames as a job when the project has none for it yet. The depth goes
//! into the project's `Resources/video_relight`, and the preview and export
//! light the clip from it as soon as the job lands. Another preset, its
//! strength, or the Custom light's color and angle need no new depth, so
//! they show at once; trimming or splitting the clip keeps its depth, and a
//! wider trim makes a new one.

use cutlass_eval::{RelightAsk, relight_ask};
use cutlass_model::{MaterialRef, Relight, SegmentId};
use cutlass_session::Request;
use cutlass_store::find_relight;
use cutlass_vision::{Compute, DEPTH_WEIGHTS, DepthModel, EnhanceJobError, relight_clip};

use super::enhance::model_dir;
use super::{KeyedSetting, Landing, Shell, ShellError};

/// The job's label, which the inspector looks for to show its progress.
pub(crate) const RELIGHT_JOB: &str = "Relight";

/// The folder in `Models` holding the depth model.
pub(super) const RELIGHT_FOLDER: &str = "relight";

impl Shell {
    /// The selected clip, when it is a movie that can be relit.
    fn relight_segment(&self) -> Result<SegmentId, ShellError> {
        let refusal = "only a video clip can be relit";
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

    /// Light the selected clip with `preset` at `strength` (0 to 1), the
    /// Custom light taking `color` (`#rrggbb`) from `angle` degrees, or
    /// turn it off with an empty preset. A clip whose depth the project
    /// lacks has it guessed as a job first; the light is stored straight
    /// away, in one undo step, and shows once the job lands.
    pub fn set_relight(
        &mut self,
        preset: &str,
        strength: f64,
        color: &str,
        angle: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.relight_segment()?;
        self.refuse_locked(&segment_id)?;
        if !preset.is_empty() {
            self.make_depth(&segment_id)?;
        }
        let now = self.relight();
        let mut requests = vec![Request::SetRelight {
            segment_id,
            preset: preset.to_owned(),
            strength,
            color: color.to_owned(),
            angle,
        }];
        // A keyed strength, color or angle plays its keys, so a change to
        // one keys it here.
        if let Some(now) = now.filter(|_| !preset.is_empty()) {
            let parts = [
                (
                    KeyedSetting::RelightStrength,
                    now.strength != strength,
                    vec![strength],
                ),
                (
                    KeyedSetting::RelightColor,
                    !now.color.eq_ignore_ascii_case(color),
                    relight_rgb(color).to_vec(),
                ),
                (KeyedSetting::RelightAngle, now.angle != angle, vec![angle]),
            ];
            for (setting, changed, values) in parts {
                if changed && let Some(key) = self.key_setting_here(setting, values) {
                    requests.push(key);
                }
            }
        }
        self.apply_requests(&requests)
    }

    /// Start a job guessing clip `segment_id`'s depth when the project
    /// lacks it.
    fn make_depth(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before relighting a clip".to_owned())
        })?;
        let draft = self.session_draft()?;
        let ask = relight_ask(draft, Some(&root), segment_id)
            .map_err(|error| ShellError::Unavailable(error.to_string()))?;
        if find_relight(&root, &ask.key(), &ask.range).is_some()
            || self.relight_progress().is_some()
        {
            return Ok(());
        }
        self.start_depth(root, ask)
    }

    /// Guess `ask`'s depth on a job, which redraws the preview when it
    /// lands. The GPU is used when this build has one; the CPU is slower.
    fn start_depth(&mut self, root: std::path::PathBuf, ask: RelightAsk) -> Result<(), ShellError> {
        let dir = model_dir(RELIGHT_FOLDER, DEPTH_WEIGHTS);
        if !dir.join(DEPTH_WEIGHTS).is_file() {
            return Err(ShellError::Unavailable(format!(
                "the {RELIGHT_JOB} model is not on this machine"
            )));
        }
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let kernels = Shell::home().cache().join("kernels");
        let compute = Compute::best();
        self.start_job(RELIGHT_JOB, move |progress| {
            progress.stage("Loading the model");
            if let Err(error) = cutlass_vision::keep_kernels_in(&kernels) {
                cutlass_logger::warn!("kernel cache: {error}");
            }
            let made = DepthModel::load(&dir, compute)
                .map_err(|error| Some(error.to_string()))
                .and_then(|model| {
                    progress.stage("Processing");
                    relight_clip(&readers, &encoders, &model, &ask, &root, &mut |done| {
                        progress.fraction(done);
                        !progress.cancelled()
                    })
                    .map(|_| ())
                    .map_err(|error| match error {
                        EnhanceJobError::Stopped => None,
                        error => Some(error.to_string()),
                    })
                });
            let landing: Landing = Box::new(move |shell: &mut Shell| match made {
                Ok(()) => {
                    shell.preview_cache.borrow_mut().clear();
                    Ok(())
                }
                Err(None) => Ok(()),
                Err(Some(reason)) => {
                    cutlass_logger::error!("{RELIGHT_JOB}: {reason}");
                    Err(ShellError::Unavailable(format!(
                        "the clip's depth could not be read: {reason}"
                    )))
                }
            });
            landing
        })
    }

    /// The selected clip's light, its keyed settings as keyed at the
    /// playhead; `None` when it is not relit.
    #[must_use]
    pub fn relight(&self) -> Option<Relight> {
        let draft = self.draft()?;
        let segment_id = self.selected()?;
        let (segment, _) = super::placed(draft, segment_id)?;
        match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Video(video)) => Some(cutlass_model::relight_at(
                &video.relight()?,
                segment,
                &draft.keyframe_graph_list,
                self.playhead,
            )),
            _ => None,
        }
    }

    /// How far the Relight job is, 0 to 1, while one runs.
    #[must_use]
    pub fn relight_progress(&self) -> Option<f32> {
        self.job()
            .filter(|job| job.label == RELIGHT_JOB)
            .map(|job| job.fraction.unwrap_or(0.0))
    }
}

/// The Custom light's `color` (`#rrggbb`) as red, green and blue from 0
/// to 1; an empty or unreadable color is the light's own.
pub(super) fn relight_rgb(color: &str) -> [f64; 3] {
    let body = color.trim().trim_start_matches('#');
    let channel = |at: usize| {
        body.get(at..at + 2)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .map(|byte| f64::from(byte) / 255.0)
    };
    match (channel(0), channel(2), channel(4)) {
        (Some(r), Some(g), Some(b)) if body.len() == 6 => [r, g, b],
        _ => cutlass_effects::CUSTOM_COLOR.map(|c| f64::from(c.powf(1.0 / 2.2))),
    }
}

#[cfg(test)]
mod tests;
