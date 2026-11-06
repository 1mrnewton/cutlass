//! Tracking on the selected clip: follow a box from the playhead through
//! the clip as a job, then make other clips follow it in one undo step.

use cutlass_eval::{TrackAsk, TrackError, track_clip};
use cutlass_model::{MaterialId, SegmentId, TrackMode};
use cutlass_session::Request;

use super::{Landing, Shell, ShellError};

impl Shell {
    /// Track `corners`, a box in the selected video clip's own frame (-1 to
    /// 1, y up) at the playhead, through the clip, and make `followers`
    /// follow it. The job shows its progress and can be cancelled; the
    /// result lands as one `Track` request, and is dropped if the project
    /// changed meanwhile.
    pub fn track_object(
        &mut self,
        corners: [(f64, f64); 4],
        mode: TrackMode,
        followers: Vec<SegmentId>,
    ) -> Result<(), ShellError> {
        if followers.is_empty() {
            return Err(ShellError::Unavailable(
                "choose a clip to follow the target".to_owned(),
            ));
        }
        let target = self.picture_segment("tracking belongs on a video clip")?;
        for id in &followers {
            self.refuse_locked(id)?;
        }
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let draft = session.draft().clone();
        let revision = session.revision();
        let readers = self.readers.clone();
        let root = self.path.clone();
        let tracking_id = MaterialId::new(self.fresh("tracking"));
        let ask = TrackAsk {
            segment_id: target.clone(),
            at: self.playhead(),
            corners,
            mode,
        };
        self.start_job("Track", move |progress| {
            progress.stage("Tracking");
            let path = track_clip(&readers, &draft, root.as_deref(), &ask, &mut |done| {
                progress.fraction(done);
                !progress.cancelled()
            });
            let landing: Landing = Box::new(move |shell: &mut Shell| {
                let path = match path {
                    Ok(path) => path,
                    Err(TrackError::Stopped) => return Ok(()),
                    Err(error) => return Err(ShellError::Unavailable(error.to_string())),
                };
                if shell.revision() != Some(revision) {
                    return Err(ShellError::Unavailable(
                        "the project changed while the clip was tracked".to_owned(),
                    ));
                }
                shell.apply(Request::Track {
                    target,
                    followers,
                    tracking_id,
                    anchor: ask.at,
                    path,
                })
            });
            landing
        })
    }
}
