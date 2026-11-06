//! Tracking on the selected clip: draw a box on the preview, choose what
//! follows it, follow the box through the clip as a job, then make the
//! followers follow it in one undo step.

use cutlass_eval::{TrackAsk, TrackError, TrackSpan, track_clip};
use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, SegmentId, TrackAttribute, TrackMode, TrackType,
    VIDEO_TRACKING,
};
use cutlass_session::Request;

use super::{Landing, Shell, ShellError};

/// The smallest box side, in canvas pixels.
const MIN_BOX: f64 = 8.0;

/// A track being set up: the clip, the frame, the mode, the box and the
/// clips that will follow it.
#[derive(Clone, Debug, PartialEq)]
pub struct Tracking {
    pub segment_id: SegmentId,
    /// The timeline time the box is drawn at.
    pub time: Micros,
    pub mode: TrackMode,
    /// The box's top left and bottom right in canvas pixels.
    pub canvas_box: Option<((f64, f64), (f64, f64))>,
    pub followers: Vec<SegmentId>,
}

/// A clip that can follow the target, and its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FollowerChoice {
    pub segment_id: SegmentId,
    pub name: String,
}

/// A target's tracked path in canvas pixels: each frame's time and box.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackedPath {
    pub frames: Vec<(Micros, [(f64, f64); 4])>,
}

impl TrackedPath {
    /// The box at the frame nearest `time`, when `time` is on the path.
    #[must_use]
    pub fn at(&self, time: Micros) -> Option<[(f64, f64); 4]> {
        let (first, last) = (self.frames.first()?.0, self.frames.last()?.0);
        if time.0 < first.0 || time.0 > last.0 {
            return None;
        }
        self.frames
            .iter()
            .min_by_key(|(at, _)| at.0.abs_diff(time.0))
            .map(|(_, corners)| *corners)
    }
}

impl Shell {
    /// The track being set up, for the inspector and the preview.
    pub fn tracking(&self) -> Option<&Tracking> {
        self.tracking.as_ref()
    }

    /// Whether the selected clip can be tracked: a video clip under the
    /// playhead, unlocked.
    pub fn track_enabled(&self) -> bool {
        self.track_target().is_ok()
    }

    /// Start setting up a track on the selected clip, at the playhead.
    pub fn start_tracking(&mut self) -> Result<(), ShellError> {
        let segment_id = self.track_target()?;
        self.picking = None;
        self.tracking = Some(Tracking {
            segment_id,
            time: self.playhead(),
            mode: TrackMode::Object,
            canvas_box: None,
            followers: Vec::new(),
        });
        Ok(())
    }

    pub fn stop_tracking(&mut self) {
        self.tracking = None;
    }

    pub fn set_track_mode(&mut self, mode: TrackMode) {
        if let Some(tracking) = self.tracking.as_mut() {
            tracking.mode = mode;
        }
    }

    /// Set the box from two opposite corners in canvas pixels, on the frame
    /// under the playhead.
    pub fn set_track_box(&mut self, a: (f64, f64), b: (f64, f64)) -> Result<(), ShellError> {
        let time = self.playhead();
        let segment_id = match self.tracking.as_ref() {
            Some(tracking) => tracking.segment_id.clone(),
            None => return Err(ShellError::Unavailable("not setting up a track".to_owned())),
        };
        let on_clip = self
            .session_draft()
            .ok()
            .and_then(|draft| super::placed(draft, &segment_id))
            .is_some_and(|(segment, _)| {
                (segment.target.start.0..segment.target.end().0).contains(&time.0)
            });
        if !on_clip {
            return Err(ShellError::Unavailable(
                "move the playhead onto the clip".to_owned(),
            ));
        }
        let Some(tracking) = self.tracking.as_mut() else {
            return Err(ShellError::Unavailable("not setting up a track".to_owned()));
        };
        tracking.time = time;
        let low = (a.0.min(b.0), a.1.min(b.1));
        let high = (a.0.max(b.0), a.1.max(b.1));
        if high.0 - low.0 < MIN_BOX || high.1 - low.1 < MIN_BOX {
            return Err(ShellError::Unavailable(
                "draw a larger box around the target".to_owned(),
            ));
        }
        tracking.canvas_box = Some((low, high));
        Ok(())
    }

    /// Add `id` to the followers, or take it off.
    pub fn toggle_follower(&mut self, id: &SegmentId) {
        if let Some(tracking) = self.tracking.as_mut() {
            if let Some(at) = tracking.followers.iter().position(|f| f == id) {
                tracking.followers.remove(at);
            } else {
                tracking.followers.push(id.clone());
            }
        }
    }

    /// The clips that can follow the track's target: video, sticker and
    /// text clips on unlocked rows that overlap it.
    pub fn follower_choices(&self) -> Vec<FollowerChoice> {
        let (Some(tracking), Ok(draft)) = (self.tracking.as_ref(), self.session_draft()) else {
            return Vec::new();
        };
        follower_choices(draft, &tracking.segment_id)
    }

    /// Whether the track is ready to run: a box and a follower.
    pub fn track_ready(&self) -> bool {
        self.tracking
            .as_ref()
            .is_some_and(|t| t.canvas_box.is_some() && !t.followers.is_empty())
    }

    /// Run the track set up so far as a job.
    pub fn run_tracking(&mut self) -> Result<(), ShellError> {
        let Some(tracking) = self.tracking.clone() else {
            return Err(ShellError::Unavailable("not setting up a track".to_owned()));
        };
        let Some((low, high)) = tracking.canvas_box else {
            return Err(ShellError::Unavailable(
                "draw a box around the target on the preview".to_owned(),
            ));
        };
        let draft = self.session_draft()?;
        let Some(segment) = super::placed(draft, &tracking.segment_id).map(|(s, _)| s) else {
            self.tracking = None;
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        let canvas = (
            u32::try_from(draft.canvas_config.width.max(1)).unwrap_or(1),
            u32::try_from(draft.canvas_config.height.max(1)).unwrap_or(1),
        );
        let mut corners = [(0.0, 0.0); 4];
        for (corner, (x, y)) in corners.iter_mut().zip([
            (low.0, low.1),
            (high.0, low.1),
            (high.0, high.1),
            (low.0, high.1),
        ]) {
            *corner = cutlass_eval::clip_point(draft, segment, tracking.time, canvas, x, y)
                .ok_or_else(|| ShellError::Unavailable("the clip is not drawn there".to_owned()))?;
        }
        let (u, v) = corners
            .iter()
            .fold((0.0, 0.0), |s, p| (s.0 + p.0 / 4.0, s.1 + p.1 / 4.0));
        if !((-1.0..=1.0).contains(&u) && (-1.0..=1.0).contains(&v)) {
            return Err(ShellError::Unavailable(
                "draw the box on the clip".to_owned(),
            ));
        }
        self.track_object(
            tracking.segment_id,
            tracking.time,
            corners,
            tracking.mode,
            tracking.followers,
        )?;
        self.tracking = None;
        Ok(())
    }

    /// The selected clip's tracked path in canvas pixels: the path it was
    /// tracked along, or the one it follows.
    pub fn tracked_path(&self) -> Option<TrackedPath> {
        let id = self.selected()?;
        let draft = self.session_draft().ok()?;
        let (segment, _) = super::placed(draft, id)?;
        let record = draft.materials.video_trackings.iter().find(|raw| {
            raw.kind == VIDEO_TRACKING
                && (raw.fields.get("segment_id").and_then(|v| v.as_str()) == Some(id.as_str())
                    || segment.extra_material_refs.contains(&raw.id))
        })?;
        let (w, h) = (
            f64::from(draft.canvas_config.width.max(1)),
            f64::from(draft.canvas_config.height.max(1)),
        );
        let frames = record
            .fields
            .get("path")?
            .as_array()?
            .iter()
            .filter_map(|row| {
                let row = row.as_array()?;
                let time = Micros(row.first()?.as_i64()?);
                let mut corners = [(0.0, 0.0); 4];
                for (i, corner) in corners.iter_mut().enumerate() {
                    let x = row.get(1 + 2 * i)?.as_f64()?;
                    let y = row.get(2 + 2 * i)?.as_f64()?;
                    *corner = ((x + 1.0) / 2.0 * w, (1.0 - y) / 2.0 * h);
                }
                Some((time, corners))
            })
            .collect::<Vec<_>>();
        (!frames.is_empty()).then_some(TrackedPath { frames })
    }

    /// Track `corners`, a box in `target`'s own frame (-1 to 1, y up) at
    /// timeline time `at`, through the clip, and make `followers` follow
    /// it. The job shows its progress and can be cancelled; the result
    /// lands as one `Track` request, and is dropped if the project changed
    /// meanwhile.
    pub fn track_object(
        &mut self,
        target: SegmentId,
        at: Micros,
        corners: [(f64, f64); 4],
        mode: TrackMode,
        followers: Vec<SegmentId>,
    ) -> Result<(), ShellError> {
        if followers.is_empty() {
            return Err(ShellError::Unavailable(
                "choose a clip to follow the target".to_owned(),
            ));
        }
        self.refuse_locked(&target)?;
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
            at,
            corners,
            mode,
            span: TrackSpan::Both,
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

    /// The selected clip as a track target at the playhead.
    fn track_target(&self) -> Result<SegmentId, ShellError> {
        let segment_id = self.picture_segment("tracking belongs on a video clip")?;
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        let Some((segment, kind)) = super::placed(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if *kind != TrackType::Video
            || !matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(video)) if video.kind != "photo"
            )
        {
            return Err(ShellError::Unavailable(
                "tracking belongs on a video clip".to_owned(),
            ));
        }
        let time = self.playhead();
        if !(segment.target.start.0..segment.target.end().0).contains(&time.0) {
            return Err(ShellError::Unavailable(
                "move the playhead onto the clip".to_owned(),
            ));
        }
        Ok(segment_id)
    }
}

/// The clips in `draft` that can follow `target`.
fn follower_choices(draft: &Draft, target: &SegmentId) -> Vec<FollowerChoice> {
    let Some((target_segment, _)) = super::placed(draft, target) else {
        return Vec::new();
    };
    let (start, end) = (target_segment.target.start.0, target_segment.target.end().0);
    draft
        .tracks
        .iter()
        .filter(|track| {
            matches!(
                track.kind,
                TrackType::Video | TrackType::Sticker | TrackType::Text
            ) && track.attribute.0 & TrackAttribute::LOCK == 0
        })
        .flat_map(|track| &track.segments)
        .filter(|segment| {
            &segment.id != target && segment.target.start.0 < end && segment.target.end().0 > start
        })
        .map(|segment| FollowerChoice {
            segment_id: segment.id.clone(),
            name: super::clip_name(draft, segment),
        })
        .collect()
}

#[cfg(test)]
mod tests;
