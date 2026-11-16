//! Compact draft reads and one session combo.
//!
//! A query returns a revision and leaves it unchanged. A write is one undo
//! step, or a dry run that reports the revision it would store. A request id
//! repeats the first result for the life of this value.

use std::collections::HashMap;
use std::fmt;

use cutlass_model::{Draft, MaterialId, Micros, Segment, SegmentId, TimeRange, Track, TrackId};
use cutlass_session::{EditError, Request, Session};
use serde_json::Value;

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
    /// Every material in the library: the media, texts and the rest.
    Materials,
    /// One material's stored record.
    Material {
        /// The material to read.
        material_id: MaterialId,
    },
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
    /// Where the last clip on the timeline ends.
    pub end: Micros,
    /// Each track's id, kind word and how many segments it holds, in order.
    pub tracks: Vec<(TrackId, String, usize)>,
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
    /// The blend into the next clip, when the segment carries one: its
    /// effect id (or its material id when it has none) and its length.
    pub transition: Option<(String, Micros)>,
    /// The track the segment is on.
    pub track: TrackId,
    /// That track's kind word.
    pub kind: String,
    /// What the segment stores beyond a plain clip, and its materials; only
    /// a segment query reads it.
    pub detail: Option<SegmentDetail>,
}

/// What one segment stores, as JSON text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentDetail {
    /// The segment's fields that differ from a plain clip's (position,
    /// scale, volume, speed, keyframes and so on); empty when none do.
    pub changed: String,
    /// The material the segment plays and each one it refers to: the pool
    /// it is in and its record, without its empty, zero and false values.
    pub materials: Vec<(String, String)>,
}

/// The materials in the library at one revision, one line each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialsSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// Each material's pool, id and the fields that name it.
    pub lines: Vec<String>,
    /// How many more there are than `lines` shows.
    pub more: usize,
}

/// One material's record at one revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialSnapshot {
    /// The revision this view was read from.
    pub revision: u64,
    /// The pool the material is in.
    pub pool: String,
    /// The record as JSON, without its empty, zero and false values.
    pub record: String,
    /// The segments that play or refer to it.
    pub used_by: Vec<SegmentId>,
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
    /// The library.
    Materials(MaterialsSnapshot),
    /// One material.
    Material(MaterialSnapshot),
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
                write!(formatter, " end={} tracks=", project.end.0)?;
                for (index, (id, kind, count)) in project.tracks.iter().enumerate() {
                    let comma = if index == 0 { "" } else { "," };
                    // A track without an id, from an older project, is `""`.
                    let id = if id.as_str().is_empty() {
                        "\"\""
                    } else {
                        id.as_str()
                    };
                    write!(formatter, "{comma}{id}:{kind}:{count}")?;
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
                    write!(
                        formatter,
                        " {}@{}+{}",
                        segment.id, segment.start.0, segment.duration.0
                    )?;
                }
                Ok(())
            }
            Self::Segment(segment) => {
                write!(
                    formatter,
                    "revision={} segment={} track={} kind={} material={} start={} duration={}",
                    segment.revision,
                    segment.id,
                    shown_track(&segment.track),
                    segment.kind,
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
                if let Some((effect, duration)) = &segment.transition {
                    write!(formatter, " transition={effect}+{}", duration.0)?;
                }
                if let Some(detail) = &segment.detail {
                    if !detail.changed.is_empty() {
                        write!(formatter, "\nset={}", detail.changed)?;
                    }
                    for (pool, record) in &detail.materials {
                        write!(formatter, "\n{pool}={record}")?;
                    }
                }
                Ok(())
            }
            Self::Range(range) => {
                write!(formatter, "revision={} range", range.revision)?;
                for segment in &range.segments {
                    write!(
                        formatter,
                        " {}@{}+{}:{}",
                        segment.id,
                        segment.start.0,
                        segment.duration.0,
                        shown_track(&segment.track)
                    )?;
                }
                Ok(())
            }
            Self::Materials(materials) => {
                write!(formatter, "revision={} materials", materials.revision)?;
                for line in &materials.lines {
                    write!(formatter, "\n{line}")?;
                }
                if materials.more > 0 {
                    write!(
                        formatter,
                        "\nand {} more; \"material <id>\" shows one",
                        materials.more
                    )?;
                }
                Ok(())
            }
            Self::Material(material) => {
                write!(
                    formatter,
                    "revision={} material {}={}\nused_by=",
                    material.revision, material.pool, material.record
                )?;
                for (index, id) in material.used_by.iter().enumerate() {
                    let comma = if index == 0 { "" } else { "," };
                    write!(formatter, "{comma}{id}")?;
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
                        if let Some((effect, duration)) = transition_of(draft, segment) {
                            write!(formatter, " transition={effect}+{}", duration.0)?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// A track id as a query writes it: one without an id, from an older
/// project, is `""`.
pub(crate) fn shown_track(id: &TrackId) -> &str {
    if id.as_str().is_empty() {
        "\"\""
    } else {
        id.as_str()
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
    /// The material id is not in the library.
    MissingMaterial {
        /// The id that was asked for.
        id: MaterialId,
    },
    /// The range ends before it starts, or its duration does not fit.
    InvalidRange,
    /// A failed combo could not be named again from the open draft.
    UnnamedFailure,
    /// The request id already answered a different combo.
    RequestIdReused {
        /// The id that was sent again.
        id: String,
    },
}

impl fmt::Display for DraftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RevisionMismatch { expected, actual } => write!(
                formatter,
                "revision {actual} does not match {expected}, so nothing was written"
            ),
            Self::RequestFailed { index, name, cause } => write!(
                formatter,
                "request {index} {name} failed: {cause}. The draft is unchanged"
            ),
            Self::EmptyUndo => write!(formatter, "nothing to undo"),
            Self::EmptyRedo => write!(formatter, "nothing to redo"),
            Self::MissingTrack { id } => write!(formatter, "missing track {id}"),
            Self::MissingSegment { id } => write!(formatter, "missing segment {id}"),
            Self::MissingMaterial { id } => write!(
                formatter,
                "missing material {id}; draft.query \"materials\" lists them"
            ),
            Self::InvalidRange => write!(formatter, "the range ends before it starts"),
            Self::RequestIdReused { id } => write!(
                formatter,
                "request id {id} already committed a different combo, so nothing was written; \
                 give each new combo a new request id"
            ),
            Self::UnnamedFailure => write!(
                formatter,
                "the draft is unchanged, and the failed request could not be named"
            ),
        }
    }
}

impl std::error::Error for DraftError {}

/// The combo each request id committed and what it answered, so the same
/// combo sent again under that id, as a client does when an answer is lost,
/// is answered the same way and not written twice. A dry run or a refused
/// combo changed nothing and is not kept, so a retry runs it afresh. A host
/// that builds its tools anew for each call hands this from one to the next.
#[derive(Debug, Default)]
pub struct Remembered(HashMap<String, Committed>);

/// A combo a request id committed, and its answer.
#[derive(Clone, Debug)]
struct Committed {
    expected_revision: u64,
    requests: Vec<Request>,
    result: ApplyResult,
}

/// The most request ids kept; past it the oldest are forgotten together.
const REMEMBERED_LIMIT: usize = 1024;

/// Draft queries and combos for one open session.
#[derive(Debug)]
pub struct DraftTools {
    session: Session,
    idempotent: HashMap<String, Committed>,
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

    /// The request ids answered so far, for the tools that come next.
    #[must_use]
    pub fn take_remembered(&mut self) -> Remembered {
        Remembered(std::mem::take(&mut self.idempotent))
    }

    /// Carry on the request ids earlier tools answered.
    pub fn set_remembered(&mut self, remembered: Remembered) {
        self.idempotent = remembered.0;
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
            DraftQuery::Materials => Ok(materials_view(revision, draft)),
            DraftQuery::Material { material_id } => material_view(revision, draft, material_id),
        }
    }

    /// Apply one combo, or preview it. A non-empty request id that commits
    /// is remembered with its combo: the same combo under it again is
    /// answered as before, and another combo under it is refused.
    pub fn apply(&mut self, input: ApplyInput) -> Result<ApplyResult, DraftError> {
        let ApplyInput {
            request_id,
            expected_revision,
            requests,
            mode,
        } = input;
        let key = idempotency_key(request_id);
        if let Some(key) = &key
            && let Some(kept) = self.idempotent.get(key)
        {
            if kept.expected_revision == expected_revision && kept.requests == requests {
                return Ok(kept.result);
            }
            return Err(DraftError::RequestIdReused { id: key.clone() });
        }
        let result = self.apply_fresh(expected_revision, &requests, mode)?;
        if let Some(key) = key
            && result.committed
        {
            if self.idempotent.len() >= REMEMBERED_LIMIT {
                self.idempotent.clear();
            }
            self.idempotent.insert(
                key,
                Committed {
                    expected_revision,
                    requests,
                    result,
                },
            );
        }
        Ok(result)
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
        Request::AddCaption { .. } => "add_caption",
        Request::RecaptionClip { .. } => "recaption_clip",
        Request::AddSubtitles { .. } => "add_subtitles",
        Request::InsertCaption { .. } => "insert_caption",
        Request::SetCaptionTemplate { .. } => "set_caption_template",
        Request::ExtractAudio { .. } => "extract_audio",
        Request::RecoverAudio { .. } => "recover_audio",
        Request::SetTrackAttribute { .. } => "set_track_attribute",
        Request::SetConstantSpeed { .. } => "set_constant_speed",
        Request::SetCurveSpeed { .. } => "set_curve_speed",
        Request::SetChangePitch { .. } => "set_change_pitch",
        Request::SetTransition { .. } => "set_transition",
        Request::RemoveTransition { .. } => "remove_transition",
        Request::SetGroup { .. } => "set_group",
        Request::PasteAttributes { .. } => "paste_attributes",
        Request::Split { .. } => "split",
        Request::Link { .. } => "link",
        Request::Remove { .. } => "remove",
        Request::SetReverse { .. } => "set_reverse",
        Request::SetFreeze { .. } => "set_freeze",
        Request::SetActive { .. } => "set_active",
        Request::SetMirror { .. } => "set_mirror",
        Request::SetVerticalFlip { .. } => "set_vertical_flip",
        Request::SetRotation { .. } => "set_rotation",
        Request::SetPivot { .. } => "set_pivot",
        Request::SetPosition { .. } => "set_position",
        Request::SetScale { .. } => "set_scale",
        Request::SetUniformScale { .. } => "set_uniform_scale",
        Request::SetScaleAxes { .. } => "set_scale_axes",
        Request::SetOpacity { .. } => "set_opacity",
        Request::SetVolume { .. } => "set_volume",
        Request::SetFade { .. } => "set_fade",
        Request::SetCrop { .. } => "set_crop",
        Request::SetStable { .. } => "set_stable",
        Request::SetEnhance { .. } => "set_enhance",
        Request::SetDenoise { .. } => "set_denoise",
        Request::SetDeflicker { .. } => "set_deflicker",
        Request::SetInterpolate { .. } => "set_interpolate",
        Request::SetRelight { .. } => "set_relight",
        Request::SetReframe { .. } => "set_reframe",
        Request::SetMotionBlur { .. } => "set_motion_blur",
        Request::SetTextContent { .. } => "set_text_content",
        Request::SetTextSize { .. } => "set_text_size",
        Request::SetTextColor { .. } => "set_text_color",
        Request::SetTextAlign { .. } => "set_text_align",
        Request::SetTextPattern { .. } => "set_text_pattern",
        Request::SetTextRangePattern { .. } => "set_text_range_pattern",
        Request::SetLetterSpacing { .. } => "set_letter_spacing",
        Request::SetLineSpacing { .. } => "set_line_spacing",
        Request::SetLineWidth { .. } => "set_line_width",
        Request::SetTextFont { .. } => "set_text_font",
        Request::SetTextFontAxes { .. } => "set_text_font_axes",
        Request::SetTextOutline { .. } => "set_text_outline",
        Request::SetTextShadow { .. } => "set_text_shadow",
        Request::SetTextShadowOpacity { .. } => "set_text_shadow_opacity",
        Request::SetTextCurve { .. } => "set_text_curve",
        Request::SetTextBackground { .. } => "set_text_background",
        Request::SetTextBackgroundBox { .. } => "set_text_background_box",
        Request::SetTextBubble { .. } => "set_text_bubble",
        Request::SetCaptionStyle { .. } => "set_caption_style",
        Request::SetCaptionKeyword { .. } => "set_caption_keyword",
        Request::SetCaptionStagger { .. } => "set_caption_stagger",
        Request::SetTextGlow { .. } => "set_text_glow",
        Request::AddText { .. } => "add_text",
        Request::AddShape { .. } => "add_shape",
        Request::DrawShape { .. } => "draw_shape",
        Request::SetShapeSize { .. } => "set_shape_size",
        Request::SetShapeFill { .. } => "set_shape_fill",
        Request::SetShapeStroke { .. } => "set_shape_stroke",
        Request::SetShapeCorners { .. } => "set_shape_corners",
        Request::SetShapeShadow { .. } => "set_shape_shadow",
        Request::AddEffect { .. } => "add_effect",
        Request::BindEffect { .. } => "bind_effect",
        Request::SetBlend { .. } => "set_blend",
        Request::SetBlendOn { .. } => "set_blend_on",
        Request::SetAdjust { .. } => "set_adjust",
        Request::AddAdjustment { .. } => "add_adjustment",
        Request::AddFilter { .. } => "add_filter",
        Request::SetHsl { .. } => "set_hsl",
        Request::SetCurve { .. } => "set_curve",
        Request::SetWheel { .. } => "set_wheel",
        Request::SetLut { .. } => "set_lut",
        Request::SetChroma { .. } => "set_chroma",
        Request::SetMask { .. } => "set_mask",
        Request::SetSelection { .. } => "set_selection",
        Request::Track { .. } => "track",
        Request::Untrack { .. } => "untrack",
        Request::SetColorMask { .. } => "set_color_mask",
        Request::SetEffectStrength { .. } => "set_effect_strength",
        Request::SetEffectControl { .. } => "set_effect_control",
        Request::SetKeyframes { .. } => "set_keyframes",
        Request::AddKeyframe { .. } => "add_keyframe",
        Request::RemoveKeyframe { .. } => "remove_keyframe",
        Request::KeyValues { .. } => "key_values",
        Request::UnkeyValues { .. } => "unkey_values",
        Request::SetKeyframeGraph { .. } => "set_keyframe_graph",
        Request::SetAnimation { .. } => "set_animation",
        Request::KeyPose { .. } => "key_pose",
        Request::Import { .. } => "import",
        Request::ReplaceMedia { .. } => "replace_media",
        Request::LinkMedia { .. } => "link_media",
        Request::CreateCompound { .. } => "create_compound",
        Request::Ungroup { .. } => "ungroup",
        Request::SetInside { .. } => "set_inside",
        Request::InsertTrack { .. } => "insert_track",
        Request::InsertTrackAt { .. } => "insert_track_at",
        Request::RemoveTrack { .. } => "remove_track",
        Request::RenameTrack { .. } => "rename_track",
        Request::AddTimeMark { .. } => "add_time_mark",
        Request::RemoveTimeMark { .. } => "remove_time_mark",
        Request::AddBeat { .. } => "add_beat",
        Request::RemoveBeat { .. } => "remove_beat",
        Request::SetMagnet { .. } => "set_magnet",
        Request::SetCanvasBackground { .. } => "set_canvas_background",
        Request::SetCanvasSize { .. } => "set_canvas_size",
        Request::SetLetterbox { .. } => "set_letterbox",
        Request::SetLetterboxBlur { .. } => "set_letterbox_blur",
        Request::SetFrameRate { .. } => "set_frame_rate",
        Request::SetDropFrame { .. } => "set_drop_frame",
        Request::SetColorSpace { .. } => "set_color_space",
        Request::SetProjectName { .. } => "set_project_name",
        Request::SetCover { .. } => "set_cover",
        Request::SetFreeLayer { .. } => "set_free_layer",
        Request::Arrange { .. } => "arrange",
    }
}

fn idempotency_key(request_id: Option<String>) -> Option<String> {
    request_id.filter(|id| !id.trim().is_empty())
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
        end: Micros(
            draft
                .tracks
                .iter()
                .flat_map(|track| &track.segments)
                .map(|segment| {
                    segment
                        .target
                        .start
                        .0
                        .saturating_add(segment.target.duration.0)
                })
                .max()
                .unwrap_or_default(),
        ),
        tracks: draft
            .tracks
            .iter()
            .map(|track| {
                (
                    track.id.clone(),
                    track.kind.as_str().to_owned(),
                    track.segments.len(),
                )
            })
            .collect(),
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
            .map(|segment| segment_snapshot(revision, draft, track, segment))
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
                let mut snapshot = segment_snapshot(revision, draft, track, segment);
                snapshot.detail = Some(segment_detail(draft, segment));
                return Ok(QueryView::Segment(snapshot));
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
                segments.push(segment_snapshot(revision, draft, track, segment));
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

fn segment_snapshot(
    revision: u64,
    draft: &Draft,
    track: &Track,
    segment: &Segment,
) -> SegmentSnapshot {
    let source = segment.source_range.as_ref();
    SegmentSnapshot {
        revision,
        id: segment.id.clone(),
        material_id: segment.material_id.clone(),
        start: segment.target.start,
        duration: segment.target.duration,
        source_start: source.map(|range| range.start),
        source_duration: source.map(|range| range.duration),
        transition: transition_of(draft, segment),
        track: track.id.clone(),
        kind: track.kind.as_str().to_owned(),
        detail: None,
    }
}

/// The longest JSON text one line of a view carries.
const RECORD_LIMIT: usize = 4000;

/// The most materials the library view lists.
const MATERIALS_LIMIT: usize = 300;

/// What `segment` stores beyond a plain clip, and the records of the
/// materials it plays and refers to.
fn segment_detail(draft: &Draft, segment: &Segment) -> SegmentDetail {
    let mut plain = Segment::new(
        segment.id.as_str(),
        segment.material_id.as_str(),
        segment.target.clone(),
    );
    plain.source_range.clone_from(&segment.source_range);
    let changed = match (serde_json::to_value(segment), serde_json::to_value(&plain)) {
        (Ok(stored), Ok(plain)) => difference(&stored, &plain)
            .map(|value| cut(&value.to_string()))
            .unwrap_or_default(),
        _ => String::new(),
    };
    let pools = pools(draft);
    let mut materials = Vec::new();
    for id in std::iter::once(&segment.material_id).chain(&segment.extra_material_refs) {
        if let Some((pool, record)) = find_material(&pools, id.as_str()) {
            materials.push((
                pool.to_owned(),
                cut(&shown_record(pool, record).to_string()),
            ));
        }
    }
    SegmentDetail { changed, materials }
}

/// The material pools as JSON: each pool's name and its records.
fn pools(draft: &Draft) -> Vec<(String, Vec<Value>)> {
    let Ok(Value::Object(map)) = serde_json::to_value(&draft.materials) else {
        return Vec::new();
    };
    map.into_iter()
        .filter_map(|(pool, records)| match records {
            Value::Array(records) if !records.is_empty() => Some((pool, records)),
            _ => None,
        })
        .collect()
}

fn find_material<'a>(pools: &'a [(String, Vec<Value>)], id: &str) -> Option<(&'a str, &'a Value)> {
    pools.iter().find_map(|(pool, records)| {
        records
            .iter()
            .find(|record| record.get("id").and_then(Value::as_str) == Some(id))
            .map(|record| (pool.as_str(), record))
    })
}

/// The parts of `stored` that differ from `plain`, keeping the shape.
fn difference(stored: &Value, plain: &Value) -> Option<Value> {
    match (stored, plain) {
        (Value::Object(stored), Value::Object(plain)) => {
            let mut out = serde_json::Map::new();
            for (key, value) in stored {
                match plain.get(key) {
                    Some(other) => {
                        if let Some(changed) = difference(value, other) {
                            out.insert(key.clone(), changed);
                        }
                    }
                    None => {
                        out.insert(key.clone(), value.clone());
                    }
                }
            }
            (!out.is_empty()).then_some(Value::Object(out))
        }
        _ => (stored != plain).then(|| stored.clone()),
    }
}

/// Values that are stored empty and still say something, by pool and key,
/// with what they mean: text with no color is drawn white.
const MEANT: &[(&str, &str, &str)] = &[("texts", "text_color", "#ffffff")];

/// `record` as a view writes it: [`trimmed`], with the [`MEANT`] values
/// written out, so a reader is not left to guess what a missing one is.
fn shown_record(pool: &str, record: &Value) -> Value {
    let mut shown = trimmed(record);
    if let Some(map) = shown.as_object_mut() {
        for (in_pool, key, meaning) in MEANT {
            if *in_pool == pool && !map.contains_key(*key) {
                map.insert((*key).to_owned(), Value::String((*meaning).to_owned()));
            }
        }
    }
    shown
}

/// `record` without its empty, zero and false values, which say nothing.
fn trimmed(record: &Value) -> Value {
    match record {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), trimmed(value)))
                .filter(|(_, value)| !says_nothing(value))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(trimmed).collect()),
        other => other.clone(),
    }
}

fn says_nothing(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(on) => !on,
        Value::Number(number) => number.as_f64() == Some(0.0),
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
    }
}

/// `text`, cut to [`RECORD_LIMIT`] with a note when it is longer.
fn cut(text: &str) -> String {
    if text.len() <= RECORD_LIMIT {
        return text.to_owned();
    }
    let mut end = RECORD_LIMIT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} (cut, {} characters in all)", &text[..end], text.len())
}

/// The fields that name a material in the library view, in this order.
const NAMING: &[&str] = &[
    "type",
    "name",
    "path",
    "content",
    "effect_id",
    "resource_id",
    "duration",
    "width",
    "height",
    "has_audio",
];

fn materials_view(revision: u64, draft: &Draft) -> QueryView {
    let mut lines = Vec::new();
    let mut more = 0;
    for (pool, records) in pools(draft) {
        for record in &records {
            if lines.len() == MATERIALS_LIMIT {
                more += 1;
                continue;
            }
            let id = record.get("id").and_then(Value::as_str).unwrap_or_default();
            let mut line = format!("{pool} {id}");
            for key in NAMING {
                let Some(value) = record.get(*key).filter(|value| !says_nothing(value)) else {
                    continue;
                };
                let text = match value {
                    Value::String(text) if text.chars().count() > 80 => {
                        Value::String(format!("{}...", text.chars().take(80).collect::<String>()))
                            .to_string()
                    }
                    Value::String(text) if text.contains(char::is_whitespace) => value.to_string(),
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                line.push_str(&format!(" {key}={text}"));
            }
            lines.push(line);
        }
    }
    QueryView::Materials(MaterialsSnapshot {
        revision,
        lines,
        more,
    })
}

fn material_view(
    revision: u64,
    draft: &Draft,
    material_id: &MaterialId,
) -> Result<QueryView, DraftError> {
    let pools = pools(draft);
    let Some((pool, record)) = find_material(&pools, material_id.as_str()) else {
        return Err(DraftError::MissingMaterial {
            id: material_id.clone(),
        });
    };
    let used_by = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .filter(|segment| {
            &segment.material_id == material_id || segment.extra_material_refs.contains(material_id)
        })
        .map(|segment| segment.id.clone())
        .collect();
    Ok(QueryView::Material(MaterialSnapshot {
        revision,
        pool: pool.to_owned(),
        record: cut(&shown_record(pool, record).to_string()),
        used_by,
    }))
}

/// The blend `segment` carries into the next clip: the transition its
/// material refs name, as its effect id (or material id) and length.
fn transition_of(draft: &Draft, segment: &Segment) -> Option<(String, Micros)> {
    segment.extra_material_refs.iter().find_map(|reference| {
        draft
            .materials
            .transitions
            .iter()
            .find(|transition| &transition.id == reference)
            .map(|transition| {
                let effect = if transition.effect_id.is_empty() {
                    transition.id.as_str().to_owned()
                } else {
                    transition.effect_id.clone()
                };
                (effect, transition.duration)
            })
    })
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
    use cutlass_model::{
        Draft, MaterialId, MaterialVideo, Micros, Segment, SegmentId, TimeRange, Track, TrackId,
        TrackType,
    };
    use cutlass_session::{EditError, ImportedMedium, Request, Session};
    use serde_json::Value;

    use super::{
        ApplyInput, ApplyMode, ApplyResult, DraftError, DraftQuery, DraftTools, QueryView,
        request_name,
    };

    fn open_tools() -> (DraftTools, TrackId) {
        let mut draft = Draft::empty("D");
        draft.materials.videos.push(MaterialVideo::new("media"));
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

    #[test]
    fn a_segment_query_shows_what_the_clip_stores_and_its_materials() {
        let (mut tools, track) = open_tools();
        tools
            .apply(apply_input(
                None,
                0,
                vec![
                    place(&track, "a", 0, 1_000_000),
                    Request::SetPosition {
                        segment_id: SegmentId::new("a"),
                        x: 0.5,
                        y: -0.25,
                    },
                    Request::SetVolume {
                        segment_id: SegmentId::new("a"),
                        volume: 0.4,
                    },
                    Request::AddText {
                        track_id: TrackId::new("titles"),
                        segment_id: SegmentId::new("title"),
                        material_id: MaterialId::new("title-text"),
                        start: Micros(0),
                        duration: Micros(1_000_000),
                        content: "Summer trip".to_owned(),
                        face: "Inter.ttf".to_owned(),
                    },
                ],
                ApplyMode::Commit,
            ))
            .unwrap();
        let read = |id: &str| {
            tools
                .query(&DraftQuery::Segment {
                    segment_id: SegmentId::new(id),
                })
                .unwrap()
                .to_string()
        };
        let clip = read("a");
        let mut lines = clip.lines();
        assert_eq!(
            lines.next().unwrap(),
            "revision=1 segment=a track=main kind=video material=media start=0 duration=1000000"
        );
        // Only what differs from a plain clip is listed.
        let set: Value =
            serde_json::from_str(lines.next().unwrap().strip_prefix("set=").unwrap()).unwrap();
        assert_eq!(set["volume"], 0.4);
        assert_eq!(set["clip"]["transform"]["x"], 0.5);
        assert_eq!(set["clip"]["transform"]["y"], -0.25);
        assert!(
            set.get("id").is_none() && set.get("speed").is_none(),
            "{set}"
        );
        assert!(lines.next().unwrap().starts_with("videos={"));
        // A title's words, size and face are in its text record.
        let title = read("title");
        assert!(title.contains("track=titles kind=text"), "{title}");
        let record = title
            .lines()
            .find_map(|line| line.strip_prefix("texts="))
            .unwrap();
        let record: Value = serde_json::from_str(record).unwrap();
        assert_eq!(record["content"], "Summer trip");
        assert_eq!(record["font_size"], 80.0);
        assert_eq!(record["font_path"], "Inter.ttf");
        // New text stores no color and is drawn white: the view says so.
        assert_eq!(record["text_color"], "#ffffff");
        tools
            .apply(apply_input(
                None,
                1,
                vec![Request::SetTextColor {
                    segment_id: SegmentId::new("title"),
                    color: "#ffcc00".to_owned(),
                    face: String::new(),
                }],
                ApplyMode::Commit,
            ))
            .unwrap();
        let colored = tools
            .query(&DraftQuery::Material {
                material_id: MaterialId::new("title-text"),
            })
            .unwrap()
            .to_string();
        assert!(colored.contains(r##""text_color":"#ffcc00""##), "{colored}");
        // A plain clip has nothing set, and a track view stays one line.
        let (mut tools, track) = open_tools();
        tools
            .apply(apply_input(
                None,
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        let plain = tools
            .query(&DraftQuery::Segment {
                segment_id: SegmentId::new("a"),
            })
            .unwrap()
            .to_string();
        assert!(!plain.contains("set="), "{plain}");
        let lane = tools
            .query(&DraftQuery::Track { track_id: track })
            .unwrap()
            .to_string();
        assert_eq!(lane.lines().count(), 1, "{lane}");
    }

    #[test]
    fn the_library_lists_materials_and_one_reads_in_full() {
        let (mut tools, track) = open_tools();
        let mut movie = MaterialVideo::new("movie");
        movie.path = "/media/a b.mp4".into();
        movie.width = 1920;
        movie.height = 1080;
        movie.duration = Micros(8_000_000);
        tools
            .apply(apply_input(
                None,
                0,
                vec![
                    Request::Import {
                        medium: ImportedMedium::Video(Box::new(movie)),
                    },
                    place(&track, "a", 0, 1_000_000),
                    Request::AddText {
                        track_id: TrackId::new("titles"),
                        segment_id: SegmentId::new("title"),
                        material_id: MaterialId::new("title-text"),
                        start: Micros(0),
                        duration: Micros(1_000_000),
                        content: "Hello there".to_owned(),
                        face: "Inter.ttf".to_owned(),
                    },
                ],
                ApplyMode::Commit,
            ))
            .unwrap();
        let library = tools.query(&DraftQuery::Materials).unwrap().to_string();
        let mut lines = library.lines();
        assert_eq!(lines.next().unwrap(), "revision=1 materials");
        assert!(
            library.lines().any(|line| line
                == "videos movie path=\"/media/a b.mp4\" duration=8000000 width=1920 height=1080"),
            "{library}"
        );
        assert!(
            library
                .lines()
                .any(|line| line.starts_with("texts title-text type=text content=\"Hello there\"")),
            "{library}"
        );
        let one = tools
            .query(&DraftQuery::Material {
                material_id: MaterialId::new("media"),
            })
            .unwrap()
            .to_string();
        assert!(one.starts_with("revision=1 material videos={"), "{one}");
        assert!(one.ends_with("\nused_by=a"), "{one}");
        assert_eq!(
            tools
                .query(&DraftQuery::Material {
                    material_id: MaterialId::new("ghost"),
                })
                .unwrap_err(),
            DraftError::MissingMaterial {
                id: MaterialId::new("ghost")
            }
        );
        // A range names where each clip is, not only its id.
        let range = tools
            .query(&DraftQuery::Range {
                start: Micros(0),
                end: Micros(500_000),
            })
            .unwrap()
            .to_string();
        assert_eq!(
            range,
            "revision=1 range title@0+1000000:titles a@0+1000000:main"
        );
    }

    #[test]
    fn a_long_record_is_cut_with_its_length() {
        let long = "x".repeat(super::RECORD_LIMIT + 10);
        let cut = super::cut(&long);
        assert!(cut.len() < long.len() + 40);
        assert!(cut.ends_with(&format!("(cut, {} characters in all)", long.len())));
        assert_eq!(super::cut("short"), "short");
    }

    #[test]
    fn request_ids_carry_from_one_set_of_tools_to_the_next() {
        let (mut tools, track) = open_tools();
        let first = tools
            .apply(apply_input(
                Some("once"),
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        // A host that builds tools for each call hands the ids on.
        let remembered = tools.take_remembered();
        let mut next = DraftTools::new(tools.session().clone());
        next.set_remembered(remembered);
        let again = next
            .apply(apply_input(
                Some("once"),
                0,
                vec![place(&track, "a", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(again, first);
        assert_eq!(next.session().revision(), 1, "the combo ran once");
        // Without them the same call is a stale write, refused.
        let mut fresh = DraftTools::new(tools.session().clone());
        assert!(
            fresh
                .apply(apply_input(
                    Some("once"),
                    0,
                    vec![place(&track, "a", 0, 1_000_000)],
                    ApplyMode::Commit,
                ))
                .is_err()
        );
    }

    #[test]
    fn import_is_named() {
        let request = Request::Import {
            medium: ImportedMedium::Video(Box::new(MaterialVideo::new("clip"))),
        };
        assert_eq!(request_name(&request), "import");
    }

    #[test]
    fn insert_track_is_named() {
        let request = Request::InsertTrack {
            track_id: TrackId::new("lane"),
            kind: TrackType::Audio,
        };
        assert_eq!(request_name(&request), "insert_track");
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
        // Each track's id, kind and size, so a model can name them.
        let text = view.to_string();
        assert!(text.contains(" end=1000000"), "{text}");
        assert!(text.contains(" tracks=main:video:1"), "{text}");
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
        draft.materials.videos.push(MaterialVideo::new("media"));
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
    fn queries_show_the_blend_a_clip_carries_into_the_next() {
        let (mut tools, track) = open_tools();
        let blend = |id: &str, effect_id: &str| Request::SetTransition {
            segment_id: SegmentId::new("a"),
            transition: Box::new(cutlass_model::MaterialTransition {
                id: MaterialId::new(id),
                effect_id: effect_id.to_owned(),
                duration: Micros(500_000),
                ..cutlass_model::MaterialTransition::default()
            }),
        };
        tools
            .apply(apply_input(
                None,
                0,
                vec![
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 1_000_000, 1_000_000),
                    blend("tr", "cutlass-dissolve"),
                ],
                ApplyMode::Commit,
            ))
            .unwrap();
        let segment = |tools: &DraftTools, id: &str| {
            tools
                .query(&DraftQuery::Segment {
                    segment_id: SegmentId::new(id),
                })
                .unwrap()
        };
        let view = segment(&tools, "a");
        let QueryView::Segment(outgoing) = &view else {
            panic!("segment query returned {view}");
        };
        assert_eq!(
            outgoing.transition,
            Some(("cutlass-dissolve".to_owned(), Micros(500_000)))
        );
        let first_line = view.to_string();
        let first_line = first_line.lines().next().unwrap_or_default();
        assert!(
            first_line.ends_with(" transition=cutlass-dissolve+500000"),
            "{view}"
        );
        let incoming = segment(&tools, "b").to_string();
        assert!(!incoming.contains("transition="), "{incoming}");
        let full = tools.query(&DraftQuery::Full).unwrap().to_string();
        assert!(
            full.contains(
                "segment=a material=media start=0 duration=1000000 \
                 transition=cutlass-dissolve+500000 segment=b"
            ),
            "{full}"
        );
        assert_eq!(full.matches("transition=").count(), 1, "{full}");

        // A blend with no effect id is named by its material id, which
        // never holds a space, so the words still split.
        tools
            .apply(apply_input(
                None,
                1,
                vec![blend("tr", "")],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert!(
            segment(&tools, "a")
                .to_string()
                .lines()
                .next()
                .unwrap_or_default()
                .ends_with(" transition=tr+500000")
        );
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
        // The text says why, so the caller can mend the request.
        assert!(
            error
                .to_string()
                .contains("clip id is empty or already used"),
            "{error}"
        );
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
        // The refused dry run is not kept: a fixed combo under its id runs.
        let fixed = tools
            .apply(apply_input(
                Some("dry-fail"),
                0,
                vec![place(&track, "b", 0, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert!(fixed.committed);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
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
        // The commit after a dry run under the same id writes.
        assert_eq!(
            again,
            ApplyResult {
                revision: 1,
                committed: true,
            }
        );
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
    }

    #[test]
    fn request_id_answers_its_combo_again_and_refuses_another() {
        let (mut tools, track) = open_tools();
        let original = tools.session().draft().clone();
        let combo = || vec![place(&track, "a", 0, 1_000_000)];
        let first = tools
            .apply(apply_input(Some("step"), 0, combo(), ApplyMode::Commit))
            .unwrap();
        assert_eq!(
            first,
            ApplyResult {
                revision: 1,
                committed: true,
            }
        );
        // The same combo again, as after a lost answer: not written twice.
        let again = tools
            .apply(apply_input(Some("step"), 0, combo(), ApplyMode::Commit))
            .unwrap();
        assert_eq!(again, first);
        assert_eq!(tools.session().revision(), 1);
        // Another combo under the same id is refused, not dropped unseen.
        let other = tools
            .apply(apply_input(
                Some("step"),
                1,
                vec![place(&track, "b", 1_000_000, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap_err();
        assert_eq!(
            other,
            DraftError::RequestIdReused {
                id: "step".to_owned()
            }
        );
        assert!(other.to_string().contains("new request id"), "{other}");
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
        assert_eq!(tools.undo().unwrap(), 2);
        assert_eq!(tools.session().draft(), &original);
    }

    #[test]
    fn request_id_keeps_no_dry_run_and_no_refusal() {
        let (mut tools, track) = open_tools();
        let combo = || vec![place(&track, "a", 0, 1_000_000)];
        // A refused combo is not kept: the retry on the right revision runs.
        let stale = tools
            .apply(apply_input(Some("try"), 4, combo(), ApplyMode::Commit))
            .unwrap_err();
        assert_eq!(
            stale,
            DraftError::RevisionMismatch {
                expected: 4,
                actual: 0,
            }
        );
        // Nor is a dry run: the commit after it under the same id writes.
        let dry = tools
            .apply(apply_input(Some("try"), 0, combo(), ApplyMode::DryRun))
            .unwrap();
        assert!(!dry.committed);
        assert_eq!(tools.session().revision(), 0);
        let written = tools
            .apply(apply_input(Some("try"), 0, combo(), ApplyMode::Commit))
            .unwrap();
        assert_eq!(
            written,
            ApplyResult {
                revision: 1,
                committed: true,
            }
        );
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
        // A blank id is no id.
        let blank = tools
            .apply(apply_input(
                Some("  "),
                1,
                vec![place(&track, "b", 1_000_000, 1_000_000)],
                ApplyMode::Commit,
            ))
            .unwrap();
        assert_eq!(blank.revision, 2);
        assert!(tools.take_remembered().0.len() == 1);
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
