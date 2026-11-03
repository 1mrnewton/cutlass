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
    /// Store a top-to-bottom flip on the segment. The clip stays where it is.
    SetVerticalFlip {
        segment_id: SegmentId,
        vertical: bool,
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
    /// Store the face file name on a text clip.
    SetTextFont { segment_id: SegmentId, face: String },
    /// Ring the letters in `color`, `width` ems wide. An empty color
    /// takes the outline off.
    SetTextOutline {
        segment_id: SegmentId,
        color: String,
        width: f64,
    },
    /// Cast a shadow in `color`, `distance` hundredths of an em toward
    /// `angle` degrees, blurred by `blur` from 0 to 1. An empty color takes
    /// the shadow off.
    SetTextShadow {
        segment_id: SegmentId,
        color: String,
        distance: f64,
        angle: f64,
        blur: f64,
    },
    /// Make the letters glow in `color`, `strength` and `range` from 0 to 1.
    /// The glow is a bloom effect the clip points at; `glow_id` names a new
    /// one when the clip has none of its own. An empty color takes it off.
    SetTextGlow {
        segment_id: SegmentId,
        glow_id: MaterialId,
        color: String,
        strength: f64,
        range: f64,
    },
    /// Fill a box behind the letters in `color` at `alpha`, its corners
    /// rounded by `round` from 0 to 1. An empty color takes the box off.
    SetTextBackground {
        segment_id: SegmentId,
        color: String,
        alpha: f64,
        round: f64,
    },
    /// Place a new text clip saying `content` in `face`, from `start` for
    /// `duration`. `track_id` is created as a text row when it is missing.
    AddText {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        start: Micros,
        duration: Micros,
        content: String,
        face: String,
    },
    /// Place the bundled effect `effect_id`, such as `recipe/shake`, from
    /// `start` for `duration`. `track_id` is created as an effect row when
    /// it is missing.
    AddEffect {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        start: Micros,
        duration: Micros,
        effect_id: String,
    },
    /// Replace the segment's keyframe groups.
    SetKeyframes {
        segment_id: SegmentId,
        keyframes: Vec<CommonKeyframeGroup>,
    },
    /// Key the clip's place, size, turn, and opacity as they are at
    /// `offset` from its start. A key there takes those values again.
    AddKeyframe {
        segment_id: SegmentId,
        offset: Micros,
    },
    /// Remove every key at `offset` from the clip's start.
    RemoveKeyframe {
        segment_id: SegmentId,
        offset: Micros,
    },
    /// Ease the span that ends at the key `offset` from the clip's start
    /// along the Graphs panel curve named `graph`, or make it straight with
    /// `None`.
    SetKeyframeGraph {
        segment_id: SegmentId,
        offset: Micros,
        graph: Option<String>,
    },
    /// Store the picture's place, size, and turn, as the three requests
    /// above do, and key them at `offset` from the clip's start.
    KeyPose {
        segment_id: SegmentId,
        offset: Micros,
        x: f64,
        y: f64,
        scale: f64,
        rotation: f64,
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
    /// Store the project canvas size and ratio. A side is from 1 through 7680.
    SetCanvasSize {
        width: i32,
        height: i32,
        ratio: String,
    },
    /// Store the color around a fitted picture. An empty string clears it.
    SetLetterbox {
        segment_id: SegmentId,
        color: String,
        canvas_id: MaterialId,
    },
    /// Cover the canvas behind a picture with a blurred copy of it, or
    /// with `None` remove the backdrop and its color, so the project color
    /// shows. `blur` is a fraction of the short side, from 0 through 1.
    SetLetterboxBlur {
        segment_id: SegmentId,
        blur: Option<f64>,
        canvas_id: MaterialId,
    },
    /// Store the project frame rate. `drop_frame` is a timecode label.
    SetFrameRate { fps: f64, drop_frame: bool },
    /// Store the drop-frame timecode label. The frame rate stays.
    SetDropFrame { on: bool },
    /// Store the project color space. The preview is not retinted.
    SetColorSpace { space: i32 },
    /// Store the project name. Surrounding whitespace is dropped.
    SetProjectName { name: String },
}
