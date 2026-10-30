//! Compact draft reads and one session combo.
//!
//! A query returns a revision and leaves it unchanged. A write is one undo
//! step, or a dry run that reports the revision it would store. A request id
//! repeats the first result for the life of this value.

use std::collections::HashMap;
use std::fmt;

use cutlass_model::{Draft, MaterialId, Micros, Segment, SegmentId, TimeRange, TrackId};
use cutlass_session::{EditError, Request, Session};

/// A compact read of the open draft.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DraftQuery {
    /// Canvas, frame rate, background, color space, duration, and name.
    Project,
    /// One track and the segments on it.
    Track {
        /// The track to read.
        track_id: TrackId,
    },
    /// One segment, found on any track.
    Segment {
        /// The segment to read.
        segment_id: SegmentId,
    },
    /// Segments whose targets overlap `[start, end)`.
    Range {
        /// Inclusive start of the half-open span.
        start: Micros,
        /// Exclusive end of the half-open span.
        end: Micros,
    },
    /// The whole draft.
    Full,
}

/// Canvas, rate, and duration at one revision.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// The draft name.
    pub name: String,
    /// The stored draft duration.
    pub duration: Micros,
    /// The stored frame rate.
    pub fps: f64,
    /// Canvas width in pixels.
    pub width: i32,
    /// Canvas height in pixels.
    pub height: i32,
    /// The canvas ratio word.
    pub ratio: String,
    /// The canvas background, when one is stored.
    pub background: Option<String>,
    /// The stored color space.
    pub color_space: i32,
}

/// One segment at one revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// The segment id.
    pub id: SegmentId,
    /// The material the segment points at.
    pub material_id: MaterialId,
    /// Target start.
    pub start: Micros,
    /// Target duration.
    pub duration: Micros,
    /// Source start, when the segment stores a source range.
    pub source_start: Option<Micros>,
    /// Source duration, when the segment stores a source range.
    pub source_duration: Option<Micros>,
}

/// One track and its segments at one revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// The track id.
    pub id: TrackId,
    /// The track kind word from `TrackType::as_str`.
    pub kind: String,
    /// The track name.
    pub name: String,
    /// Segments on the track, in track order.
    pub segments: Vec<SegmentSnapshot>,
}

/// Segments that overlap a half-open span, at one revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// Overlapping segments, in track order.
    pub segments: Vec<SegmentSnapshot>,
}

/// The whole draft at one revision.
#[derive(Clone, Debug, PartialEq)]
pub struct FullSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// The open draft.
    pub draft: Box<Draft>,
}

/// The view a draft query returned.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryView {
    /// Project settings, without the segment list.
    Project(ProjectSnapshot),
    /// One track.
    Track(TrackSnapshot),
    /// One segment.
    Segment(SegmentSnapshot),
    /// The segments that overlap a span.
    Range(RangeSnapshot),
    /// The whole draft.
    Full(FullSnapshot),
}

impl fmt::Display for QueryView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Project(project) => {
                write!(
                    formatter,
                    "revision={} name={} canvas={}x{} fps={} duration={} ratio={} color_space={}",
                    project.revision,
                    project.name,
                    project.width,
                    project.height,
                    project.fps,
                    project.duration.0,
                    project.ratio,
                    project.color_space,
                )?;
                if let Some(background) = &project.background {
                    write!(formatter, " background={background}")?;
                }
                Ok(())
            }
            Self::Track(track) => {
                write!(
                    formatter,
                    "revision={} track={} kind={} name={}",
                    track.revision, track.id, track.kind, track.name
                )?;
                for segment in &track.segments {
                    write!(formatter, " {}", segment.id)?;
                }
                Ok(())
            }
            Self::Segment(segment) => {
                write!(
                    formatter,
                    "revision={} segment={} material={} start={} duration={}",
                    segment.revision,
                    segment.id,
                    segment.material_id,
                    segment.start.0,
                    segment.duration.0,
                )?;
                if let Some(source_start) = segment.source_start {
                    write!(formatter, " source_start={}", source_start.0)?;
                }
                if let Some(source_duration) = segment.source_duration {
                    write!(formatter, " source_duration={}", source_duration.0)?;
                }
                Ok(())
            }
            Self::Range(range) => {
                write!(formatter, "revision={} range", range.revision)?;
                for segment in &range.segments {
                    write!(formatter, " {}", segment.id)?;
                }
                Ok(())
            }
            Self::Full(full) => {
                let draft = &*full.draft;
                write!(
                    formatter,
                    "revision={} full id={} name={} duration={} fps={} color_space={} canvas={}x{} ratio={}",
                    full.revision,
                    draft.id,
                    draft.name,
                    draft.duration.0,
                    draft.fps,
                    draft.color_space,
                    draft.canvas_config.width,
                    draft.canvas_config.height,
                    draft.canvas_config.ratio,
                )?;
                if let Some(background) = &draft.canvas_config.background {
                    write!(formatter, " background={background}")?;
                }
                write!(
                    formatter,
                    " video_mute={} maintrack_adsorb={}",
                    draft.config.video_mute, draft.config.maintrack_adsorb
                )?;
                for track in &draft.tracks {
                    write!(
                        formatter,
                        " track={} kind={}",
                        track.id,
                        track.kind.as_str()
                    )?;
                    for segment in &track.segments {
                        write!(
                            formatter,
                            " segment={} material={} start={} duration={}",
                            segment.id,
                            segment.material_id,
                            segment.target.start.0,
                            segment.target.duration.0
                        )?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Whether a combo is written onto the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyMode {
    /// Write the combo. It is one undo step.
    Commit,
    /// Report the revision a commit would write.
    DryRun,
}

/// One combo, the revision it was read from, and whether to commit.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyInput {
    /// An idempotency key. Empty and missing keys are not remembered.
    pub request_id: Option<String>,
    /// The revision the caller read.
    pub expected_revision: u64,
    /// The requests. The list is one combo.
    pub requests: Vec<Request>,
    /// Commit the combo, or preview it.
    pub mode: ApplyMode,
}

/// The revision after a combo and whether that combo was committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplyResult {
    /// The revision the combo reached, or the current revision when nothing ran.
    pub revision: u64,
    /// Set when the combo was written onto the session.
    pub committed: bool,
}

/// Why a draft tool did not return a view or commit a combo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DraftError {
    /// The caller read a different revision than the open draft.
    RevisionMismatch {
        /// The revision the caller sent.
        expected: u64,
        /// The revision the session holds.
        actual: u64,
    },
    /// One request failed, and the combo was not written.
    RequestFailed {
        /// The index of the first request whose prefix fails.
        index: usize,
        /// [`request_name`] of that request.
        name: &'static str,
        /// The session error for that prefix.
        cause: EditError,
    },
    /// Undo was asked for and the session has no undo step.
    EmptyUndo,
    /// Redo was asked for and the session has no redo step.
    EmptyRedo,
    /// The track id is not on the draft.
    MissingTrack {
        /// The id that was asked for.
        id: TrackId,
    },
    /// The segment id is not on any track.
    MissingSegment {
        /// The id that was asked for.
        id: SegmentId,
    },
    /// The range ends before it starts, or its duration does not fit.
    InvalidRange,
    /// A failed combo could not be named again from the open draft.
    UnnamedFailure,
}

impl fmt::Display for DraftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RevisionMismatch { expected, actual } => write!(
                formatter,
                "revision {actual} does not match {expected}, so nothing was written"
            ),
            Self::RequestFailed { index, name, .. } => write!(
                formatter,
                "request {index} {name} failed, and the draft is unchanged"
            ),
            Self::EmptyUndo => write!(formatter, "nothing to undo"),
            Self::EmptyRedo => write!(formatter, "nothing to redo"),
            Self::MissingTrack { id } => write!(formatter, "missing track {id}"),
            Self::MissingSegment { id } => write!(formatter, "missing segment {id}"),
            Self::InvalidRange => write!(formatter, "the range ends before it starts"),
            Self::UnnamedFailure => write!(
                formatter,
                "the draft is unchanged, and the failed request could not be named"
            ),
        }
    }
}

impl std::error::Error for DraftError {}

/// Draft queries and combos for one open session.
#[derive(Debug)]
pub struct DraftTools {
    session: Session,
    idempotent: HashMap<String, Result<ApplyResult, DraftError>>,
}

impl DraftTools {
    /// Tools for `session`. Request ids last as long as this value.
    #[must_use]
    pub fn new(session: Session) -> Self {
        Self {
            session,
            idempotent: HashMap::new(),
        }
    }

    /// The session these tools read and write.
    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Read one view without changing the revision.
    pub fn query(&self, query: &DraftQuery) -> Result<QueryView, DraftError> {
        let revision = self.session.revision();
        let draft = self.session.draft();
        match query {
            DraftQuery::Project => Ok(project_view(revision, draft)),
            DraftQuery::Track { track_id } => track_view(revision, draft, track_id),
            DraftQuery::Segment { segment_id } => segment_view(revision, draft, segment_id),
            DraftQuery::Range { start, end } => range_view(revision, draft, *start, *end),
            DraftQuery::Full => Ok(full_view(revision, draft)),
        }
    }

    /// Apply one combo, or preview it, and remember a non-empty request id.
    pub fn apply(&mut self, input: ApplyInput) -> Result<ApplyResult, DraftError> {
        let ApplyInput {
            request_id,
            expected_revision,
            requests,
            mode,
        } = input;
        let key = idempotency_key(request_id);
        if let Some(key) = &key
            && let Some(cached) = self.idempotent.get(key)
        {
            return cached.clone();
        }
        let result = self.apply_fresh(expected_revision, &requests, mode);
        if let Some(key) = key {
            self.idempotent.insert(key, result.clone());
        }
        result
    }

    /// Restore the draft from before the last committed combo.
    pub fn undo(&mut self) -> Result<u64, DraftError> {
        self.session.undo().map_err(map_undo)
    }

    /// Reapply the draft that undo removed.
    pub fn redo(&mut self) -> Result<u64, DraftError> {
        self.session.redo().map_err(map_redo)
    }

    fn apply_fresh(
        &mut self,
        expected: u64,
        requests: &[Request],
        mode: ApplyMode,
    ) -> Result<ApplyResult, DraftError> {
        match mode {
            ApplyMode::DryRun => preview(&self.session, expected, requests),
            ApplyMode::Commit => self.commit(expected, requests),
        }
    }

    fn commit(&mut self, expected: u64, requests: &[Request]) -> Result<ApplyResult, DraftError> {
        let outcome = self.session.apply(expected, requests);
        match outcome {
            Ok(revision) => Ok(ApplyResult {
                revision,
                committed: !requests.is_empty(),
            }),
            Err(error) => Err(explain(&self.session, expected, requests, error)),
        }
    }
}

/// The stable name of one session request.
#[must_use]
pub fn request_name(request: &Request) -> &'static str {
    match request {
        Request::Place { .. } => "place",
        Request::Move { .. } => "move",
        Request::Trim { .. } => "trim",
        Request::SetWords { .. } => "set_words",
        Request::SetTrackAttribute { .. } => "set_track_attribute",
        Request::SetConstantSpeed { .. } => "set_constant_speed",
        Request::SetCurveSpeed { .. } => "set_curve_speed",
        Request::SetTransition { .. } => "set_transition",
        Request::Split { .. } => "split",
        Request::Link { .. } => "link",
        Request::Remove { .. } => "remove",
        Request::SetReverse { .. } => "set_reverse",
        Request::SetFreeze { .. } => "set_freeze",
        Request::SetVolume { .. } => "set_volume",
        Request::SetFade { .. } => "set_fade",
        Request::SetCrop { .. } => "set_crop",
        Request::SetTextContent { .. } => "set_text_content",
        Request::SetKeyframes { .. } => "set_keyframes",
    }
}

fn idempotency_key(request_id: Option<String>) -> Option<String> {
    request_id.filter(|id| !id.is_empty())
}

fn project_view(revision: u64, draft: &Draft) -> QueryView {
    QueryView::Project(ProjectSnapshot {
        revision,
        name: draft.name.clone(),
        duration: draft.duration,
        fps: draft.fps,
        width: draft.canvas_config.width,
        height: draft.canvas_config.height,
        ratio: draft.canvas_config.ratio.clone(),
        background: draft.canvas_config.background.clone(),
        color_space: draft.color_space,
    })
}

fn track_view(revision: u64, draft: &Draft, track_id: &TrackId) -> Result<QueryView, DraftError> {
    let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) else {
        return Err(DraftError::MissingTrack {
            id: track_id.clone(),
        });
    };
    Ok(QueryView::Track(TrackSnapshot {
        revision,
        id: track.id.clone(),
        kind: track.kind.as_str().to_owned(),
        name: track.name.clone(),
        segments: track
            .segments
            .iter()
            .map(|segment| segment_snapshot(revision, segment))
            .collect(),
    }))
}

fn segment_view(
    revision: u64,
    draft: &Draft,
    segment_id: &SegmentId,
) -> Result<QueryView, DraftError> {
    for track in &draft.tracks {
        for segment in &track.segments {
            if &segment.id == segment_id {
                return Ok(QueryView::Segment(segment_snapshot(revision, segment)));
            }
        }
    }
    Err(DraftError::MissingSegment {
        id: segment_id.clone(),
    })
}

fn range_view(
    revision: u64,
    draft: &Draft,
    start: Micros,
    end: Micros,
) -> Result<QueryView, DraftError> {
    if end < start {
        return Err(DraftError::InvalidRange);
    }
    let Some(duration) = end.0.checked_sub(start.0) else {
        return Err(DraftError::InvalidRange);
    };
    let span = TimeRange::new(start.0, duration);
    let mut segments = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if segment.target.duration.0 <= 0 {
                continue;
            }
            if matches!(segment.target.overlaps(&span), Ok(true)) {
                segments.push(segment_snapshot(revision, segment));
            }
        }
    }
    Ok(QueryView::Range(RangeSnapshot { revision, segments }))
}

fn full_view(revision: u64, draft: &Draft) -> QueryView {
    QueryView::Full(FullSnapshot {
        revision,
        draft: Box::new(draft.clone()),
    })
}

fn segment_snapshot(revision: u64, segment: &Segment) -> SegmentSnapshot {
    let source = segment.source_range.as_ref();
    SegmentSnapshot {
        revision,
        id: segment.id.clone(),
        material_id: segment.material_id.clone(),
        start: segment.target.start,
        duration: segment.target.duration,
        source_start: source.map(|range| range.start),
        source_duration: source.map(|range| range.duration),
    }
}

fn preview(
    session: &Session,
    expected: u64,
    requests: &[Request],
) -> Result<ApplyResult, DraftError> {
    let outcome = {
        let mut clone = session.clone();
        clone.apply(expected, requests)
    };
    match outcome {
        Ok(revision) => Ok(ApplyResult {
            revision,
            committed: false,
        }),
        Err(error) => Err(explain(session, expected, requests, error)),
    }
}

fn explain(session: &Session, expected: u64, requests: &[Request], error: EditError) -> DraftError {
    match error {
        EditError::RevisionMismatch { expected, actual } => {
            DraftError::RevisionMismatch { expected, actual }
        }
        _ => name_failed_request(session, expected, requests),
    }
}

fn name_failed_request(session: &Session, expected: u64, requests: &[Request]) -> DraftError {
    let Some(index) = failing_index(session, expected, requests) else {
        return DraftError::UnnamedFailure;
    };
    let Some(request) = requests.get(index) else {
        return DraftError::UnnamedFailure;
    };
    let mut probe = session.clone();
    match probe.apply(expected, &requests[..=index]) {
        Ok(_) | Err(EditError::RevisionMismatch { .. }) => DraftError::UnnamedFailure,
        Err(cause) => DraftError::RequestFailed {
            index,
            name: request_name(request),
            cause,
        },
    }
}

/// How a prefix applied on a clone of the open session.
enum PrefixClass {
    Success,
    Failed,
    Mismatch,
}

fn classify(session: &Session, expected: u64, requests: &[Request]) -> PrefixClass {
    let mut clone = session.clone();
    match clone.apply(expected, requests) {
        Ok(_) => PrefixClass::Success,
        Err(EditError::RevisionMismatch { .. }) => PrefixClass::Mismatch,
        Err(_) => PrefixClass::Failed,
    }
}

/// The first index whose prefix fails while the shorter prefix succeeds.
fn failing_index(session: &Session, expected: u64, requests: &[Request]) -> Option<usize> {
    if requests.is_empty() {
        return None;
    }
    let mut low = 0;
    let mut high = requests.len();
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        match classify(session, expected, &requests[..mid]) {
            PrefixClass::Success => low = mid,
            PrefixClass::Failed => high = mid,
            PrefixClass::Mismatch => return None,
        }
    }
    let index = high.checked_sub(1)?;
    match classify(session, expected, &requests[..index]) {
        PrefixClass::Success => Some(index),
        PrefixClass::Failed | PrefixClass::Mismatch => None,
    }
}

fn map_undo(error: EditError) -> DraftError {
    match error {
        EditError::EmptyUndo => DraftError::EmptyUndo,
        _ => DraftError::UnnamedFailure,
    }
}

fn map_redo(error: EditError) -> DraftError {
    match error {
        EditError::EmptyRedo => DraftError::EmptyRedo,
        _ => DraftError::UnnamedFailure,
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialId, Micros, Segment, SegmentId, TimeRange, Track, TrackId};
    use cutlass_session::{EditError, Request, Session};

    use super::{
        ApplyInput, ApplyMode, ApplyResult, DraftError, DraftQuery, DraftTools, QueryView,
        request_name,
    };

    fn open_tools() -> (DraftTools, TrackId) {
        let mut draft = Draft::empty("D");
        let track_id = TrackId::new("main");
        draft.tracks[0].id = track_id.clone();
        (DraftTools::new(Session::open(draft)), track_id)
    }

    fn place(track_id: &TrackId, id: &str, start: i64, duration: i64) -> Request {
        Request::Place {
            track_id: track_id.clone(),
            segment: Box::new(Segment::new(id, "media", TimeRange::new(start, duration))),
        }
    }

    fn apply_input(
        request_id: Option<&str>,
        expected_revision: u64,
        requests: Vec<Request>,
        mode: ApplyMode,
    ) -> ApplyInput {
        ApplyInput {
            request_id: request_id.map(str::to_owned),
            expected_revision,
            requests,
            mode,
        }
    }

    #[test]
    fn query_project_is_compact_and_includes_revision() {
        let (mut tools, track) = open_tools();
        let view = tools.query(&DraftQuery::Project).unwrap();
        let QueryView::Project(project) = &view else {
            panic!("project query returned {view}");
        };
        assert_eq!(project.revision, 0);
        assert_eq!(project.name, "");
        assert_eq!(project.duration, Micros(0));
        assert_eq!(project.fps, 30.0);
        assert_eq!(project.width, 1920);
        assert_eq!(project.height, 1080);
        assert_eq!(project.ratio, "original");
        assert_eq!(project.background, None);
        assert_eq!(project.color_space, 0);
        let text = view.to_string();
        assert!(text.contains("revision=0"));
        assert!(text.contains("1920"));
        assert!(text.contains("1080"));
        assert!(text.contains("fps="));
        assert!(!text.contains("target_timerange"));

        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track, "only-in-full", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        let view = tools.query(&DraftQuery::Project).unwrap();
        let QueryView::Project(project) = &view else {
            panic!("project query returned {view}");
        };
        assert_eq!(project.revision, 1);
        assert_eq!(project.duration, Micros(1_000_000));
        assert!(!view.to_string().contains("only-in-full"));
        assert_eq!(tools.session().revision(), 1);
    }

    #[test]
    fn query_track_returns_that_track() {
        let (mut tools, track_id) = open_tools();
        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track_id, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        let view = tools
            .query(&DraftQuery::Track {
                track_id: track_id.clone(),
            })
            .unwrap();
        let QueryView::Track(track) = &view else {
            panic!("track query returned {view}");
        };
        assert_eq!(track.revision, 1);
        assert_eq!(track.id, track_id);
        assert_eq!(track.kind, "video");
        assert_eq!(track.name, "");
        assert_eq!(track.segments.len(), 1);
        let segment = &track.segments[0];
        assert_eq!(segment.revision, 1);
        assert_eq!(segment.id, SegmentId::new("a"));
        assert_eq!(segment.material_id, MaterialId::new("media"));
        assert_eq!(segment.start, Micros(0));
        assert_eq!(segment.duration, Micros(1_000_000));
        assert_eq!(segment.source_start, None);
        assert_eq!(segment.source_duration, None);
        assert!(view.to_string().contains("revision=1"));
        assert!(view.to_string().contains("kind=video"));
    }

    #[test]
    fn query_missing_track_is_an_error() {
        let (tools, _) = open_tools();
        let id = TrackId::new("missing");
        let error = tools
            .query(&DraftQuery::Track {
                track_id: id.clone(),
            })
            .unwrap_err();
        assert_eq!(error, DraftError::MissingTrack { id });
        assert_eq!(tools.session().revision(), 0);
    }

    #[test]
    fn query_segment_returns_range_and_revision() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let other = TrackId::new("other");
        draft.tracks.push(Track::main(other.clone()));
        let mut tools = DraftTools::new(Session::open(draft));
        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&other, "a", 250_000, 750_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        let view = tools
            .query(&DraftQuery::Segment {
                segment_id: SegmentId::new("a"),
            })
            .unwrap();
        let QueryView::Segment(segment) = &view else {
            panic!("segment query returned {view}");
        };
        assert_eq!(segment.revision, 1);
        assert_eq!(segment.id, SegmentId::new("a"));
        assert_eq!(segment.material_id, MaterialId::new("media"));
        assert_eq!(segment.start, Micros(250_000));
        assert_eq!(segment.duration, Micros(750_000));
        assert_eq!(segment.source_start, None);
        assert_eq!(segment.source_duration, None);
        let text = view.to_string();
        assert!(text.contains("revision=1"));
        assert!(text.contains("segment=a"));
        assert_eq!(tools.session().revision(), 1);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn query_missing_segment_is_an_error() {
        let (tools, _) = open_tools();
        let id = SegmentId::new("missing");
        let error = tools
            .query(&DraftQuery::Segment {
                segment_id: id.clone(),
            })
            .unwrap_err();
        assert_eq!(error, DraftError::MissingSegment { id });
        assert_eq!(tools.session().revision(), 0);
    }

    #[test]
    fn query_range_includes_overlapping_segments_only() {
        let (mut tools, track) = open_tools();
        tools
            .apply(apply_input(
                None,
                0,
                vec![
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 1_000_000, 1_000_000),
                ],
                ApplyMode::Commit,
            ))
            .unwrap();
        let excluded = tools
            .query(&DraftQuery::Range {
                start: Micros(1_000_000),
                end: Micros(2_000_000),
            })
            .unwrap();
        let QueryView::Range(range) = &excluded else {
            panic!("range query returned {excluded}");
        };
        assert_eq!(range.revision, 1);
        assert_eq!(
            range
                .segments
                .iter()
                .map(|segment| segment.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b"]
        );

        let included = tools
            .query(&DraftQuery::Range {
                start: Micros(500_000),
                end: Micros(1_500_000),
            })
            .unwrap();
        let QueryView::Range(range) = &included else {
            panic!("range query returned {included}");
        };
        assert_eq!(
            range
                .segments
                .iter()
                .map(|segment| segment.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        let error = tools
            .query(&DraftQuery::Range {
                start: Micros(2),
                end: Micros(1),
            })
            .unwrap_err();
        assert_eq!(error, DraftError::InvalidRange);
        assert_eq!(tools.session().revision(), 1);
    }

    #[test]
    fn query_full_returns_the_draft() {
        let (mut tools, track) = open_tools();
        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        let view = tools.query(&DraftQuery::Full).unwrap();
        let QueryView::Full(full) = &view else {
            panic!("full query returned {view}");
        };
        assert_eq!(full.revision, 1);
        assert_eq!(&*full.draft, tools.session().draft());
        let text = view.to_string();
        assert!(text.contains("revision=1"));
        assert!(text.split_whitespace().any(|word| word == "full"));
        assert!(text.contains("segment=a"));
        assert!(text.contains("material=media"));
        assert!(text.contains("start=0"));
        assert!(text.contains("duration=1000000"));
        assert!(text.contains("canvas=1920x1080"));
        assert!(!text.contains("target_timerange"));
        assert!(!text.contains("extra_material_refs"));
    }

    #[test]
    fn query_does_not_change_the_revision() {
        let (mut tools, track) = open_tools();
        let before = tools.session().draft().clone();
        tools.query(&DraftQuery::Project).unwrap();
        tools.query(&DraftQuery::Full).unwrap();
        tools
            .query(&DraftQuery::Track {
                track_id: track.clone(),
            })
            .unwrap();
        tools
            .query(&DraftQuery::Range {
                start: Micros(0),
                end: Micros(1_000_000),
            })
            .unwrap();
        assert_eq!(
            tools
                .query(&DraftQuery::Segment {
                    segment_id: SegmentId::new("missing"),
                })
                .unwrap_err(),
            DraftError::MissingSegment {
                id: SegmentId::new("missing"),
            }
        );
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.session().draft(), &before);

        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        let before = tools.session().draft().clone();
        tools
            .query(&DraftQuery::Segment {
                segment_id: SegmentId::new("a"),
            })
            .unwrap();
        tools.query(&DraftQuery::Full).unwrap();
        tools.query(&DraftQuery::Project).unwrap();
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.session().draft(), &before);
    }

    #[test]
    fn apply_one_combo_is_one_undo() {
        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let result = tools
            .apply(apply_input(
                None,
                0,
                vec![
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 1_000_000, 500_000),
                ],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(
            result,
            ApplyResult {
                revision: 1,
                committed: true,
            }
        );
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 2);
        assert_eq!(tools.undo().unwrap(), 2);
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);
    }

    #[test]
    fn revision_mismatch_writes_nothing() {
        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let error = tools
            .apply(apply_input(
                None,
                9,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap_err();
        assert_eq!(
            error,
            DraftError::RevisionMismatch {
                expected: 9,
                actual: 0,
            }
        );
        assert!(error.to_string().contains("nothing was written"));
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);

        let error = tools
            .apply(apply_input(None, 9, Vec::new(), ApplyMode::Commit))
            .unwrap_err();
        assert_eq!(
            error,
            DraftError::RevisionMismatch {
                expected: 9,
                actual: 0,
            }
        );
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.session().revision(), 0);
    }

    #[test]
    fn failed_request_rolls_back_and_names_the_request() {
        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let requests = vec![
            place(&track, "a", 0, 1_000_000),
            place(&track, "a", 1_000_000, 1_000_000),
        ];
        assert_eq!(request_name(&requests[1]), "place");
        let error = tools
            .apply(apply_input(None, 0, requests, ApplyMode::Commit))
            .unwrap_err();
        assert_eq!(
            error,
            DraftError::RequestFailed {
                index: 1,
                name: "place",
                cause: EditError::DuplicateSegment,
            }
        );
        assert!(error.to_string().contains("request 1 place failed"));
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);

        let (mut tools, track) = open_tools();
        let error = tools
            .apply(apply_input(
                Some("dry-fail"),
                0,
                vec![
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "a", 1_000_000, 1_000_000),
                ],
                ApplyMode::DryRun,
            ))
            .unwrap_err();
        assert_eq!(
            error,
            DraftError::RequestFailed {
                index: 1,
                name: "place",
                cause: EditError::DuplicateSegment,
            }
        );
        let again = tools
            .apply(apply_input(
                Some("dry-fail"),
                0,
                vec![place(&track, "b", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap_err();
        assert_eq!(again, error);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);
    }

    #[test]
    fn dry_run_returns_the_revision_it_would_write() {
        let (mut tools, track) = open_tools();
        let result = tools
            .apply(apply_input(
                Some("preview"),
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::DryRun,
            ))
            .unwrap();
        assert_eq!(
            result,
            ApplyResult {
                revision: 1,
                committed: false,
            }
        );
        assert_eq!(tools.session().revision(), 0);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);

        let again = tools
            .apply(apply_input(
                Some("preview"),
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(again, result);
        assert!(!again.committed);
        assert_eq!(tools.session().revision(), 0);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn request_id_returns_the_first_result() {
        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let first = tools
            .apply(apply_input(
                Some("step"),
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(
            first,
            ApplyResult {
                revision: 1,
                committed: true,
            }
        );
        let second = tools
            .apply(apply_input(
                Some("step"),
                0,
                vec![place(&track, "b", 1_000_000, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(second, first);
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
        assert_eq!(
            tools.session().draft().tracks[0].segments[0].id.as_str(),
            "a"
        );
        assert_eq!(tools.undo().unwrap(), 2);
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);

        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let error = tools
            .apply(apply_input(
                Some("stale"),
                4,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap_err();
        assert_eq!(
            error,
            DraftError::RevisionMismatch {
                expected: 4,
                actual: 0,
            }
        );
        let again = tools
            .apply(apply_input(
                Some("stale"),
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap_err();
        assert_eq!(again, error);
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);
    }

    #[test]
    fn undo_and_redo_call_the_session() {
        let (mut tools, track) = open_tools();
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);
        assert_eq!(tools.redo().unwrap_err(), DraftError::EmptyRedo);
        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.undo().unwrap(), 2);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
        assert_eq!(tools.session().draft().duration, Micros(0));
        assert_eq!(tools.redo().unwrap(), 3);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
        assert_eq!(
            tools.session().draft().tracks[0].segments[0].id,
            SegmentId::new("a")
        );
        assert_eq!(tools.redo().unwrap_err(), DraftError::EmptyRedo);
    }

    #[test]
    fn empty_apply_does_not_move_the_revision() {
        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let committed = tools
            .apply(apply_input(None, 0, Vec::new(), ApplyMode::Commit))
            .unwrap();
        assert_eq!(
            committed,
            ApplyResult {
                revision: 0,
                committed: false,
            }
        );
        let preview = tools
            .apply(apply_input(None, 0, Vec::new(), ApplyMode::DryRun))
            .unwrap();
        assert_eq!(
            preview,
            ApplyResult {
                revision: 0,
                committed: false,
            }
        );
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.session().draft(), &original);
        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(tools.undo().unwrap(), 2);
        assert_eq!(tools.session().draft(), &original);
        assert_eq!(tools.undo().unwrap_err(), DraftError::EmptyUndo);
    }
}
