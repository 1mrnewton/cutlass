//! Tracking on the selected clip: draw a box on the preview, choose what
//! follows it, follow the box through the clip as a job, then make the
//! followers follow it in one undo step. A clip's mask can follow instead,
//! to hide or pick out what moves. A tracked frame is fixed by
//! drawing the box again on it: the clip is tracked away from the track's
//! anchor from there, and the new path replaces the old one on that side.

use cutlass_eval::{TrackAsk, TrackError, TrackSpan, track_clip};
use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, RawMaterial, SegmentId, TrackAttribute, TrackMode,
    TrackPath, TrackSample, TrackType, VIDEO_TRACKING,
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
    /// The clips whose mask follows it instead.
    pub masks: Vec<SegmentId>,
    /// The track being fixed, when this one fixes a frame of it.
    pub fixing: Option<Fixing>,
    /// Which way from the drawn frame to track a new track. A fix always
    /// tracks away from the track's anchor.
    pub direction: TrackSpan,
}

/// A track a new box fixes: its anchor and its path in canvas units.
#[derive(Clone, Debug, PartialEq)]
pub struct Fixing {
    pub anchor: Micros,
    pub path: TrackPath,
}

/// The assistant's tracking jobs: each one's path by id, `None` while it
/// runs.
#[derive(Clone, Debug, Default)]
pub struct AgentTracks {
    started: u64,
    paths: std::collections::BTreeMap<String, Option<Result<TrackPath, String>>>,
}

impl AgentTracks {
    /// A fresh job id.
    pub fn next_id(&mut self) -> String {
        self.started = self.started.saturating_add(1);
        format!("track-{}", self.started)
    }

    /// Job `id`'s path: `None` while it runs.
    pub fn path(&self, id: &str) -> Result<Option<TrackPath>, String> {
        match self.paths.get(id) {
            Some(None) => Ok(None),
            Some(Some(Ok(path))) => Ok(Some(path.clone())),
            Some(Some(Err(error))) => Err(error.clone()),
            None => Err(format!("no tracking job {id}")),
        }
    }
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
        self.pen = None;
        self.tracking = Some(Tracking {
            segment_id,
            time: self.playhead(),
            mode: TrackMode::Object,
            canvas_box: None,
            followers: Vec::new(),
            masks: Vec::new(),
            fixing: None,
            direction: TrackSpan::Both,
        });
        Ok(())
    }

    /// Whether the selected clip's track can be fixed at the playhead: it
    /// was tracked, and the playhead is on the clip.
    pub fn fix_enabled(&self) -> bool {
        self.fix_setup().is_ok()
    }

    /// Start fixing the selected clip's track at the playhead: the mode
    /// and the followers are the track's, and the box is the tracked one
    /// at the playhead, ready to be drawn again.
    pub fn start_fixing(&mut self) -> Result<(), ShellError> {
        let tracking = self.fix_setup()?;
        self.picking = None;
        self.tracking = Some(tracking);
        Ok(())
    }

    /// The selected clip's track, set up to be fixed at the playhead.
    fn fix_setup(&self) -> Result<Tracking, ShellError> {
        let segment_id = self.track_target()?;
        let draft = self.session_draft()?;
        let record = draft
            .materials
            .video_trackings
            .iter()
            .find(|raw| {
                raw.kind == VIDEO_TRACKING
                    && raw.fields.get("segment_id").and_then(|v| v.as_str())
                        == Some(segment_id.as_str())
            })
            .ok_or_else(|| ShellError::Unavailable("the clip has no track to fix".to_owned()))?;
        self.record_setup(segment_id, record)
    }

    /// Track `record` of target `segment_id`, set up to be fixed at the
    /// playhead: its mode, the clips that follow it, and the tracked box
    /// there.
    fn record_setup(
        &self,
        segment_id: SegmentId,
        record: &RawMaterial,
    ) -> Result<Tracking, ShellError> {
        let unavailable = || ShellError::Unavailable("the clip has no track to fix".to_owned());
        let draft = self.session_draft()?;
        let path = record_path(record).ok_or_else(unavailable)?;
        let anchor = record
            .fields
            .get("anchor")
            .and_then(|v| v.as_i64())
            .map_or_else(
                || path.samples.first().map_or(Micros(0), |s| s.time),
                Micros,
            );
        let masks: Vec<SegmentId> = record
            .fields
            .get("masks")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
            .map(SegmentId::new)
            .collect();
        let followers = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| {
                segment.extra_material_refs.contains(&record.id) && !masks.contains(&segment.id)
            })
            .map(|segment| segment.id.clone())
            .collect();
        let time = self.playhead();
        let canvas = canvas_size(draft);
        let canvas_box = pixels(&path, canvas).at(time).map(|corners| {
            let (mut low, mut high) = (corners[0], corners[0]);
            for (x, y) in corners {
                low = (low.0.min(x), low.1.min(y));
                high = (high.0.max(x), high.1.max(y));
            }
            (low, high)
        });
        Ok(Tracking {
            segment_id,
            time,
            mode: path.mode,
            canvas_box,
            followers,
            masks,
            fixing: Some(Fixing { anchor, path }),
            direction: TrackSpan::Both,
        })
    }

    pub fn stop_tracking(&mut self) {
        self.tracking = None;
    }

    /// Which way a new track runs from the drawn frame.
    pub fn set_track_direction(&mut self, direction: TrackSpan) {
        if let Some(tracking) = self.tracking.as_mut() {
            tracking.direction = direction;
        }
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

    /// Add `id` to the followers, or take it off. A clip follows by its
    /// pose or by its mask, so it leaves the masks.
    pub fn toggle_follower(&mut self, id: &SegmentId) {
        if let Some(tracking) = self.tracking.as_mut() {
            tracking.masks.retain(|m| m != id);
            if let Some(at) = tracking.followers.iter().position(|f| f == id) {
                tracking.followers.remove(at);
            } else {
                tracking.followers.push(id.clone());
            }
        }
    }

    /// Have `id`'s mask follow, or stop it; it leaves the followers.
    pub fn toggle_mask_follower(&mut self, id: &SegmentId) {
        if let Some(tracking) = self.tracking.as_mut() {
            tracking.followers.retain(|f| f != id);
            if let Some(at) = tracking.masks.iter().position(|m| m == id) {
                tracking.masks.remove(at);
            } else {
                tracking.masks.push(id.clone());
            }
        }
    }

    /// The clips whose mask can follow the target: video and sticker clips
    /// on unlocked rows that overlap it, the target first.
    pub fn mask_choices(&self) -> Vec<FollowerChoice> {
        let (Some(tracking), Ok(draft)) = (self.tracking.as_ref(), self.session_draft()) else {
            return Vec::new();
        };
        mask_choices(draft, &tracking.segment_id)
    }

    /// The clips that can follow the track's target: video, sticker and
    /// text clips on unlocked rows that overlap it.
    pub fn follower_choices(&self) -> Vec<FollowerChoice> {
        let (Some(tracking), Ok(draft)) = (self.tracking.as_ref(), self.session_draft()) else {
            return Vec::new();
        };
        follower_choices(draft, &tracking.segment_id)
    }

    /// Whether the track is ready to run: a box and a follower or a mask.
    pub fn track_ready(&self) -> bool {
        self.tracking.as_ref().is_some_and(|t| {
            t.canvas_box.is_some() && !(t.followers.is_empty() && t.masks.is_empty())
        })
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
        let fps = draft.fps;
        let target = tracking.segment_id;
        let followers = tracking.followers;
        let masks = tracking.masks;
        match tracking.fixing {
            None => self.track_job(
                TrackAsk {
                    segment_id: target,
                    at: tracking.time,
                    corners,
                    mode: tracking.mode,
                    span: tracking.direction,
                },
                followers,
                masks,
                None,
            )?,
            Some(fixing) => {
                let span = fix_span(tracking.time, fixing.anchor, fps);
                self.track_job(
                    TrackAsk {
                        segment_id: target,
                        at: tracking.time,
                        corners,
                        mode: tracking.mode,
                        span,
                    },
                    followers,
                    masks,
                    Some(fixing),
                )?;
            }
        }
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
        let path = record_path(record)?;
        Some(pixels(&path, canvas_size(draft)))
    }

    /// Track `corners`, a box in `target`'s own frame (-1 to 1, y up) at
    /// timeline time `at`, through the clip, and make `followers` follow
    /// it and the masks of `masks`. The job shows its progress and can be cancelled; the result
    /// lands as one `Track` request, and is dropped if the project changed
    /// meanwhile.
    pub fn track_object(
        &mut self,
        target: SegmentId,
        at: Micros,
        corners: [(f64, f64); 4],
        mode: TrackMode,
        followers: Vec<SegmentId>,
        masks: Vec<SegmentId>,
    ) -> Result<(), ShellError> {
        let ask = TrackAsk {
            segment_id: target,
            at,
            corners,
            mode,
            span: TrackSpan::Both,
        };
        self.track_job(ask, followers, masks, None)
    }

    /// Run `ask` as a job and land it as one `Track` request; when it
    /// fixes a track, join the new path to the old one first.
    fn track_job(
        &mut self,
        ask: TrackAsk,
        followers: Vec<SegmentId>,
        masks: Vec<SegmentId>,
        fixing: Option<Fixing>,
    ) -> Result<(), ShellError> {
        let target = ask.segment_id.clone();
        if followers.is_empty() && masks.is_empty() {
            return Err(ShellError::Unavailable(
                "choose a clip or a mask to follow the target".to_owned(),
            ));
        }
        self.refuse_locked(&target)?;
        for id in followers.iter().chain(&masks) {
            self.refuse_locked(id)?;
        }
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let draft = session.draft().clone();
        let revision = session.revision();
        let readers = self.readers.clone();
        let root = self.path.clone();
        let tracking_id = MaterialId::new(self.fresh("tracking"));
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
                let (anchor, path) = match fixing {
                    Some(fixing) if ask.span != TrackSpan::Both => {
                        (fixing.anchor, joined(&fixing.path, path, ask.at, ask.span))
                    }
                    _ => (ask.at, path),
                };
                let draft = shell.session_draft()?;
                let masks = masks
                    .into_iter()
                    .map(|id| {
                        clip_path(draft, &id, &path)
                            .map(|own| (id, own))
                            .ok_or_else(|| {
                                ShellError::Unavailable(
                                    "a clip whose mask follows is not drawn along the track"
                                        .to_owned(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shell.apply(Request::Track {
                    target,
                    followers,
                    tracking_id,
                    anchor,
                    path,
                    masks,
                })
            });
            landing
        })
    }

    /// Start the assistant's tracking job `id` on `ask`: the box in canvas
    /// units is turned into the clip's own frame, and the path is kept for
    /// [`Self::agent_track`]. The draft is not written here; the
    /// assistant's registry writes the path.
    pub fn start_agent_track(
        &mut self,
        id: String,
        ask: &cutlass_agent::TrackAsk,
    ) -> Result<(), ShellError> {
        let at = Micros(ask.at_us);
        let draft = self.session_draft()?;
        let Some((segment, kind)) = super::placed(draft, &ask.target) else {
            return Err(ShellError::Unavailable(format!(
                "no clip {} on the timeline",
                ask.target.as_str()
            )));
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
        if !(segment.target.start.0..segment.target.end().0).contains(&at.0) {
            return Err(ShellError::Unavailable(
                "at_us is not on the clip".to_owned(),
            ));
        }
        let (w, h) = canvas_size(draft);
        let mut corners = [(0.0, 0.0); 4];
        for (corner, &(x, y)) in corners.iter_mut().zip(&ask.corners) {
            let (px, py) = ((x + 1.0) / 2.0 * w, (1.0 - y) / 2.0 * h);
            *corner = cutlass_eval::clip_point(draft, segment, at, (w as u32, h as u32), px, py)
                .ok_or_else(|| ShellError::Unavailable("the clip is not drawn there".to_owned()))?;
        }
        let (u, v) = corners
            .iter()
            .fold((0.0, 0.0), |s, p| (s.0 + p.0 / 4.0, s.1 + p.1 / 4.0));
        if !((-1.0..=1.0).contains(&u) && (-1.0..=1.0).contains(&v)) {
            return Err(ShellError::Unavailable(
                "the box is not on the clip".to_owned(),
            ));
        }
        self.refuse_locked(&ask.target)?;
        let job_ask = TrackAsk {
            segment_id: ask.target.clone(),
            at,
            corners,
            mode: ask.mode,
            span: TrackSpan::Both,
        };
        let draft = draft.clone();
        let readers = self.readers.clone();
        let root = self.path.clone();
        let landed = id.clone();
        self.start_job("Track", move |progress| {
            progress.stage("Tracking");
            let path = track_clip(&readers, &draft, root.as_deref(), &job_ask, &mut |done| {
                progress.fraction(done);
                !progress.cancelled()
            })
            .map_err(|error| error.to_string());
            let landing: Landing = Box::new(move |shell: &mut Shell| {
                shell.agent_tracks.paths.insert(landed, Some(path));
                Ok(())
            });
            landing
        })?;
        self.agent_tracks.paths.insert(id, None);
        Ok(())
    }

    /// The assistant's tracking job `id`: its path, or `None` while it
    /// runs.
    pub fn agent_track(&mut self, id: &str) -> Result<Option<TrackPath>, ShellError> {
        self.agent_tracks_now()
            .path(id)
            .map_err(ShellError::Unavailable)
    }

    /// The assistant's tracking jobs as they stand: a job that is no
    /// longer running and left no path was stopped.
    pub fn agent_tracks_now(&mut self) -> AgentTracks {
        self.poll_job();
        if !self.job().is_some_and(|job| job.label == "Track") {
            for path in self.agent_tracks.paths.values_mut() {
                path.get_or_insert_with(|| Err("the tracking job was stopped".to_owned()));
            }
        }
        self.agent_tracks.clone()
    }

    /// A fresh id for an assistant tracking job.
    pub fn agent_tracks_next_id(&mut self) -> String {
        self.agent_tracks.next_id()
    }

    /// Take the job ids `tracks` handed out, so the next ones are fresh.
    pub fn keep_track_ids(&mut self, tracks: &AgentTracks) {
        self.agent_tracks.started = self.agent_tracks.started.max(tracks.started);
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

/// What a clip follows: the video clip tracked, its name, and whether the
/// clip's mask follows rather than the clip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Following {
    pub target: SegmentId,
    pub target_name: String,
    pub by_mask: bool,
}

impl Shell {
    /// What the selected clip follows, when it follows a track.
    pub fn following(&self) -> Option<Following> {
        let id = self.selected()?;
        let draft = self.session_draft().ok()?;
        let record = followed_record(draft, id)?;
        let target = SegmentId::new(record.fields.get("segment_id")?.as_str()?);
        let target_name = super::placed(draft, &target).map_or_else(
            || "a removed clip".to_owned(),
            |(s, _)| super::clip_name(draft, s),
        );
        let by_mask = record
            .fields
            .get("masks")
            .and_then(|v| v.as_array())
            .is_some_and(|ids| ids.iter().any(|v| v.as_str() == Some(id.as_str())));
        Some(Following {
            target,
            target_name,
            by_mask,
        })
    }

    /// The name of the clip the track being set up follows.
    pub fn tracking_target_name(&self) -> Option<String> {
        let tracking = self.tracking.as_ref()?;
        let draft = self.session_draft().ok()?;
        let (segment, _) = super::placed(draft, &tracking.segment_id)?;
        Some(super::clip_name(draft, segment))
    }

    /// The video clip the selected text, sticker or shape is drawn over at
    /// the playhead: the nearest one below it.
    pub fn follow_target(&self) -> Result<SegmentId, ShellError> {
        let id = self
            .selected()
            .ok_or_else(|| ShellError::Unavailable("select a text, sticker or shape".to_owned()))?
            .clone();
        let draft = self.session_draft()?;
        video_below(draft, &id, self.playhead())
    }

    /// Start setting up a track for the selected text, sticker or shape:
    /// the video under it is tracked and the item follows.
    pub fn start_following(&mut self) -> Result<(), ShellError> {
        let target = self.follow_target()?;
        let follower = self.selected().cloned().ok_or_else(nothing_selected)?;
        self.refuse_locked(&follower)?;
        self.picking = None;
        self.pen = None;
        self.tracking = Some(Tracking {
            segment_id: target,
            time: self.playhead(),
            mode: TrackMode::Object,
            canvas_box: None,
            followers: vec![follower],
            masks: Vec::new(),
            fixing: None,
            direction: TrackSpan::Both,
        });
        Ok(())
    }

    /// Start setting up a track for the selected video clip's mask: the
    /// clip itself is tracked and its mask follows.
    pub fn start_mask_following(&mut self) -> Result<(), ShellError> {
        let target = self.track_target()?;
        self.picking = None;
        self.pen = None;
        self.tracking = Some(Tracking {
            segment_id: target.clone(),
            time: self.playhead(),
            mode: TrackMode::Object,
            canvas_box: None,
            followers: Vec::new(),
            masks: vec![target],
            fixing: None,
            direction: TrackSpan::Both,
        });
        Ok(())
    }

    /// Start fixing the track the selected clip follows, at the playhead.
    pub fn start_fixing_following(&mut self) -> Result<(), ShellError> {
        let id = self.selected().cloned().ok_or_else(nothing_selected)?;
        self.refuse_locked(&id)?;
        let draft = self.session_draft()?;
        let record = followed_record(draft, &id)
            .ok_or_else(|| ShellError::Unavailable("the clip follows no track".to_owned()))?;
        let target = record
            .fields
            .get("segment_id")
            .and_then(|v| v.as_str())
            .map(SegmentId::new)
            .ok_or_else(|| ShellError::Unavailable("the clip follows no track".to_owned()))?;
        let on_target = super::placed(draft, &target).is_some_and(|(segment, _)| {
            (segment.target.start.0..segment.target.end().0).contains(&self.playhead().0)
        });
        if !on_target {
            return Err(ShellError::Unavailable(
                "move the playhead onto the tracked video".to_owned(),
            ));
        }
        let record = record.clone();
        let tracking = self.record_setup(target, &record)?;
        self.picking = None;
        self.pen = None;
        self.tracking = Some(tracking);
        Ok(())
    }

    /// Stop the selected clip following its track, in one undo step.
    pub fn stop_following(&mut self) -> Result<(), ShellError> {
        let id = self.selected().cloned().ok_or_else(nothing_selected)?;
        if self
            .tracking
            .as_ref()
            .is_some_and(|t| t.followers.contains(&id) || t.masks.contains(&id))
        {
            self.tracking = None;
        }
        self.apply(Request::Untrack { segment_id: id })
    }
}

fn nothing_selected() -> ShellError {
    ShellError::Unavailable("select a text, sticker or shape".to_owned())
}

/// The `video_tracking` record clip `id` lists.
fn followed_record<'a>(draft: &'a Draft, id: &SegmentId) -> Option<&'a RawMaterial> {
    let (segment, _) = super::placed(draft, id)?;
    draft
        .materials
        .video_trackings
        .iter()
        .find(|raw| raw.kind == VIDEO_TRACKING && segment.extra_material_refs.contains(&raw.id))
}

/// The video clip drawn nearest below clip `id` at `time`: clips are drawn
/// in order of render index, then row, then place on the row.
fn video_below(draft: &Draft, id: &SegmentId, time: Micros) -> Result<SegmentId, ShellError> {
    let key = |t: usize, s: usize| (draft.tracks[t].segments[s].render_index, t, s);
    let mut own = None;
    for (t, track) in draft.tracks.iter().enumerate() {
        if let Some(s) = track.segments.iter().position(|segment| &segment.id == id) {
            own = Some((t, s));
        }
    }
    let (t, s) =
        own.ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
    if !matches!(
        draft.tracks[t].kind,
        TrackType::Text | TrackType::Sticker | TrackType::Video
    ) {
        return Err(ShellError::Unavailable(
            "only a text, sticker or shape can follow".to_owned(),
        ));
    }
    let on = |segment: &cutlass_model::Segment| {
        (segment.target.start.0..segment.target.end().0).contains(&time.0)
    };
    if !on(&draft.tracks[t].segments[s]) {
        return Err(ShellError::Unavailable(
            "move the playhead onto the clip".to_owned(),
        ));
    }
    let mine = key(t, s);
    draft
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, track)| track.kind == TrackType::Video)
        .flat_map(|(ti, track)| {
            track
                .segments
                .iter()
                .enumerate()
                .map(move |(si, seg)| (ti, si, seg))
        })
        .filter(|(ti, si, segment)| {
            &segment.id != id
                && on(segment)
                && key(*ti, *si) < mine
                && matches!(
                    draft.materials.get(segment.material_id.as_str()),
                    Some(MaterialRef::Video(video)) if video.kind != "photo"
                )
        })
        .max_by_key(|(ti, si, _)| key(*ti, *si))
        .map(|(_, _, segment)| segment.id.clone())
        .ok_or_else(|| {
            ShellError::Unavailable("put it over a video clip at the playhead".to_owned())
        })
}

/// Which way to track a fix at `time` of a track anchored at `anchor`:
/// away from the anchor, or both ways on the anchor's own frame.
fn fix_span(time: Micros, anchor: Micros, fps: f64) -> TrackSpan {
    let fps = if fps.is_finite() && fps >= 1.0 {
        fps
    } else {
        30.0
    };
    let half_frame = (500_000.0 / fps).round() as i64;
    if time.0.abs_diff(anchor.0) as i64 <= half_frame {
        TrackSpan::Both
    } else if time.0 > anchor.0 {
        TrackSpan::After
    } else {
        TrackSpan::Before
    }
}

/// `old` with `new` in place of its samples on `span`'s side of `at`, the
/// fixed frame. `new` runs from `at` that way.
fn joined(old: &TrackPath, new: TrackPath, at: Micros, span: TrackSpan) -> TrackPath {
    let keep = |sample: &&TrackSample| match span {
        TrackSpan::After => sample.time.0 < at.0,
        TrackSpan::Before => sample.time.0 > at.0,
        TrackSpan::Both => false,
    };
    let kept = old.samples.iter().filter(keep).copied();
    let samples = match span {
        TrackSpan::Before => new.samples.into_iter().chain(kept).collect(),
        TrackSpan::After | TrackSpan::Both => kept.chain(new.samples).collect(),
    };
    TrackPath {
        mode: new.mode,
        samples,
    }
}

/// A `video_tracking` record's path, in canvas units.
fn record_path(record: &RawMaterial) -> Option<TrackPath> {
    let mode = match record.fields.get("tracker_type").and_then(|v| v.as_i64()) {
        Some(1) => TrackMode::Plane,
        _ => TrackMode::Object,
    };
    let samples = record
        .fields
        .get("path")?
        .as_array()?
        .iter()
        .filter_map(|row| {
            let row = row.as_array()?;
            let time = Micros(row.first()?.as_i64()?);
            let mut corners = [(0.0, 0.0); 4];
            for (i, corner) in corners.iter_mut().enumerate() {
                *corner = (row.get(1 + 2 * i)?.as_f64()?, row.get(2 + 2 * i)?.as_f64()?);
            }
            Some(TrackSample { time, corners })
        })
        .collect::<Vec<_>>();
    (!samples.is_empty()).then_some(TrackPath { mode, samples })
}

/// `path` in canvas pixels on a `canvas`-sized canvas.
fn pixels(path: &TrackPath, canvas: (f64, f64)) -> TrackedPath {
    let (w, h) = canvas;
    TrackedPath {
        frames: path
            .samples
            .iter()
            .map(|sample| {
                (
                    sample.time,
                    sample
                        .corners
                        .map(|(x, y)| ((x + 1.0) / 2.0 * w, (1.0 - y) / 2.0 * h)),
                )
            })
            .collect(),
    }
}

fn canvas_size(draft: &Draft) -> (f64, f64) {
    (
        f64::from(draft.canvas_config.width.max(1)),
        f64::from(draft.canvas_config.height.max(1)),
    )
}

/// `path`, in canvas units, in clip `id`'s own frame over its span: where
/// its mask is drawn.
pub fn clip_path(draft: &Draft, id: &SegmentId, path: &TrackPath) -> Option<TrackPath> {
    let (segment, _) = super::placed(draft, id)?;
    cutlass_eval::path_in_clip(draft, segment, path)
}

/// The clips in `draft` whose mask can follow `target`: the target first.
fn mask_choices(draft: &Draft, target: &SegmentId) -> Vec<FollowerChoice> {
    let Some((target_segment, _)) = super::placed(draft, target) else {
        return Vec::new();
    };
    let (start, end) = (target_segment.target.start.0, target_segment.target.end().0);
    let mut choices: Vec<FollowerChoice> = draft
        .tracks
        .iter()
        .filter(|track| {
            matches!(track.kind, TrackType::Video | TrackType::Sticker)
                && track.attribute.0 & TrackAttribute::LOCK == 0
        })
        .flat_map(|track| &track.segments)
        .filter(|segment| segment.target.start.0 < end && segment.target.end().0 > start)
        .map(|segment| FollowerChoice {
            segment_id: segment.id.clone(),
            name: super::clip_name(draft, segment),
        })
        .collect();
    if let Some(at) = choices.iter().position(|c| &c.segment_id == target) {
        let own = choices.remove(at);
        choices.insert(0, own);
    }
    choices
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
