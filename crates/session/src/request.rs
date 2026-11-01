use cutlass_model::{
    CommonKeyframeGroup, Crop, FreezeInfo, MaterialAudio, MaterialId, MaterialTransition,
    MaterialVideo, Micros, Segment, SegmentId, SpeedPoint, TrackId, TrackType, WordColumns,
};

/// A media record stored in the library. A segment places it later.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportedMedium {
    /// A picture, including a still stored as a video material.
    Video(Box<MaterialVideo>),
    /// A sound. Its track type is supplied by the import, not by lane category.
    Audio(Box<MaterialAudio>),
}

/// Which edge of a segment a trim moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimEdge {
    /// The timeline start. The end stays put.
    Start,
    /// The timeline end. The start stays put.
    End,
}

/// One change. A combo is a slice of these.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Insert `segment` on `track_id`, ordered by its target start.
    Place {
        track_id: TrackId,
        segment: Box<Segment>,
    },
    /// Keep the duration. `start` is the new target start. `track_id` may be the same track.
    Move {
        segment_id: SegmentId,
        track_id: TrackId,
        start: Micros,
    },
    /// Move one target edge. A source range maps through the segment speed.
    Trim {
        segment_id: SegmentId,
        edge: TrimEdge,
        time: Micros,
    },
    /// Replace the recognized words and the current words on one text material.
    ///
    /// Word times are milliseconds. The caption string becomes those words
    /// joined by spaces. Both word arrays receive this result.
    SetWords {
        material_id: MaterialId,
        words: WordColumns,
    },
    /// Place one caption for `source_id` and store its words.
    ///
    /// The caption uses the source clip's timeline range. `track_id` is created
    /// when it is missing. Word times are milliseconds on that range.
    AddCaption {
        source_id: SegmentId,
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        words: WordColumns,
    },
    /// Place the picture's sound on an audio clip linked to `source_id`.
    ///
    /// `track_id` is created when it is missing. The picture stays where it is.
    ExtractAudio {
        source_id: SegmentId,
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
    },
    /// Turn one track-attribute bit on or off. Mute and solo are not gains.
    SetTrackAttribute {
        track_id: TrackId,
        bit: i32,
        on: bool,
    },
    /// Set a constant speed. Target duration becomes the source duration mapped by `speed`.
    SetConstantSpeed { segment_id: SegmentId, speed: f64 },
    /// Store a curve. `y` is speed and is linear in `x`. Target duration is the integral.
    SetCurveSpeed {
        segment_id: SegmentId,
        speed_id: MaterialId,
        points: Vec<SpeedPoint>,
    },
    /// Store a blend on the outgoing segment. Target ranges still only touch.
    SetTransition {
        segment_id: SegmentId,
        transition: Box<MaterialTransition>,
    },
    /// Split `segment_id` at `at`. The time must sit outside every blend window on the track.
    Split {
        segment_id: SegmentId,
        at: Micros,
        right_id: SegmentId,
    },
    /// Store a linkage pair. `owner` is the segment the request names.
    Link {
        owner: SegmentId,
        dependent: SegmentId,
    },
    /// Remove one segment. An owner takes its dependents with it when follow is on.
    Remove { segment_id: SegmentId },
    /// Store the reverse flag and the path on the video material. This does not seek.
    SetReverse {
        segment_id: SegmentId,
        reverse: bool,
        path: String,
    },
    /// Store a freeze mapping on the video material. This does not seek.
    SetFreeze {
        segment_id: SegmentId,
        freeze: Option<FreezeInfo>,
    },
    /// Store a horizontal mirror on the segment. The clip stays where it is.
    SetMirror {
        segment_id: SegmentId,
        horizontal: bool,
    },
    /// Store the clip rotation in degrees. The clip stays where it is.
    SetRotation { segment_id: SegmentId, degrees: f64 },
    /// Store the picture position. `x` and `y` are fractions of half the canvas.
    SetPosition {
        segment_id: SegmentId,
        x: f64,
        y: f64,
    },
    /// Store the picture scale on the uniform value and on both axes.
    SetScale { segment_id: SegmentId, scale: f64 },
    /// Store the picture opacity. `1` is solid and `0` is clear.
    SetOpacity { segment_id: SegmentId, opacity: f64 },
    /// Set the segment gain. This is not mute, solo, or a fade.
    SetVolume { segment_id: SegmentId, volume: f64 },
    /// Store fade-in and fade-out durations. The segment volume stays.
    SetFade {
        segment_id: SegmentId,
        fade_id: MaterialId,
        fade_in: Micros,
        fade_out: Micros,
    },
    /// Store the crop quad. Ratio and scale stay.
    SetCrop { segment_id: SegmentId, crop: Crop },
    /// Replace the text material's string. Word columns stay as stored.
    SetTextContent {
        segment_id: SegmentId,
        content: String,
    },
    /// Store the text size. An empty face path is filled when `face` is not.
    SetTextSize {
        segment_id: SegmentId,
        size: f64,
        face: String,
    },
    /// Store the text color. An empty string is white. An empty face path is filled.
    SetTextColor {
        segment_id: SegmentId,
        color: String,
        face: String,
    },
    /// Store left, center, or right. An empty line width takes `line_width`.
    SetTextAlign {
        segment_id: SegmentId,
        alignment: i32,
        face: String,
        line_width: f64,
    },
    /// Store extra space between letters, as a fraction of the text size.
    SetLetterSpacing {
        segment_id: SegmentId,
        spacing: f64,
        face: String,
    },
    /// Store extra space between lines, as a fraction of the text size.
    SetLineSpacing {
        segment_id: SegmentId,
        spacing: f64,
        face: String,
    },
    /// Store the wrap width in pixels. Zero keeps the run on one line.
    SetLineWidth {
        segment_id: SegmentId,
        width: f64,
        face: String,
    },
    /// Replace the segment's keyframe groups.
    SetKeyframes {
        segment_id: SegmentId,
        keyframes: Vec<CommonKeyframeGroup>,
    },
    /// Add a material to the library. This does not place a segment.
    Import { medium: ImportedMedium },
    /// Add an empty lane. Video and mixed lanes are overlays above the main track.
    /// Audio is inserted under the main track. This does not place a segment.
    InsertTrack { track_id: TrackId, kind: TrackType },
    /// Store a marker. `None` is the ruler, at a timeline time. `Some` is a clip,
    /// and the mark is an offset from that clip's target start.
    AddTimeMark {
        segment_id: Option<SegmentId>,
        at: Micros,
        color: String,
        material_id: MaterialId,
    },
    /// Remove one marker. `start` is the timeline time on the ruler, or the
    /// offset stored on a clip.
    RemoveTimeMark {
        segment_id: Option<SegmentId>,
        start: Micros,
    },
    /// Store a beat on an audio segment. `source` is the source time.
    AddBeat {
        segment_id: SegmentId,
        source: Micros,
        material_id: MaterialId,
    },
    /// Remove one beat. `source` is the stored source time.
    RemoveBeat {
        segment_id: SegmentId,
        source: Micros,
    },
    /// Turn main-track packing on or off. Turning it on closes gaps.
    SetMagnet { on: bool },
    /// Store the project canvas color. An empty string clears it.
    SetCanvasBackground { color: String },
}
