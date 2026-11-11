use serde::{Deserialize, Serialize};

use cutlass_model::{
    AnimType, CommonKeyframeGroup, Crop, FreezeInfo, MaterialAudio, MaterialId, MaterialTransition,
    MaterialVideo, Micros, Segment, SegmentId, ShapeKind, ShapePoint, SpeedPoint, TimeRange,
    TrackId, TrackType, WordColumns,
};

/// A media record stored in the library. A segment places it later.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportedMedium {
    /// A picture, including a still stored as a video material.
    Video(Box<MaterialVideo>),
    /// A sound. Its track type is supplied by the import, not by lane category.
    Audio(Box<MaterialAudio>),
}

/// One timed caption a subtitle file brings in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubtitleCue {
    pub segment_id: SegmentId,
    pub material_id: MaterialId,
    /// Where the caption sits on the timeline, in microseconds.
    pub target: TimeRange,
    /// Word times are milliseconds from the start of `target`.
    pub words: WordColumns,
}

/// Which edge of a segment a trim moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrimEdge {
    /// The timeline start. The end stays put.
    Start,
    /// The timeline end. The start stays put.
    End,
}

/// Where Arrange moves a clip among the picture clips it overlaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arrange {
    /// Above the next clip over it.
    Forward,
    /// Below the next clip under it.
    Backward,
    /// Above every clip it overlaps.
    Front,
    /// Below every clip it overlaps.
    Back,
}

/// One change. A combo is a slice of these.
///
/// As JSON a request names its kind in `op`, in snake case, beside its
/// fields: `{"op":"move","segment_id":"a","track_id":"t","start":0}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
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
    /// The caption uses the source clip's timeline range, or `span` inside
    /// it, so one transcript can be placed as a caption per line. `track_id`
    /// is created when it is missing. Word times are milliseconds on the
    /// caption's own range.
    AddCaption {
        source_id: SegmentId,
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        words: WordColumns,
        /// Boxed, so a request stays small.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        span: Option<Box<TimeRange>>,
        /// The font file the caption is set in, as `AddText`'s `face`.
        #[serde(default)]
        face: String,
    },
    /// Place subtitle cues as captions on a new text lane `track_id`.
    ///
    /// The lane is flagged as subtitles and each cue becomes a caption clip
    /// with its own words. An existing `track_id`, no cues, or cues that
    /// overlap are refused, and the draft stays as it was.
    AddSubtitles {
        track_id: TrackId,
        cues: Vec<SubtitleCue>,
        /// The font file the captions are set in, as `AddText`'s `face`.
        #[serde(default)]
        face: String,
    },
    /// Place one caption on the lane of caption `like`, looking like it.
    ///
    /// The new caption copies `like`'s text look and its place on the
    /// canvas, then takes `cue`'s words. `like` must be a caption on an
    /// unlocked text lane, and the cue must fit beside the lane's clips.
    InsertCaption { like: SegmentId, cue: SubtitleCue },
    /// Give the clip `source_id` was heard from new caption lines: every
    /// caption linked to it on `like`'s lane comes off, and each line is
    /// placed there looking like `like`, linked to `source_id`. `like` must
    /// be one of those captions.
    RecaptionClip {
        source_id: SegmentId,
        like: SegmentId,
        lines: Vec<SubtitleCue>,
        /// The font file a line takes when `like` has none.
        #[serde(default)]
        face: String,
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
    /// Give a picture whose sound was extracted its own sound back, and
    /// take the clip that sound went to off the timeline when it is still
    /// linked to the picture.
    RecoverAudio { segment_id: SegmentId },
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
    /// Let the clip's pitch follow its speed (`on`), or keep the pitch as
    /// the speed changes.
    SetChangePitch { segment_id: SegmentId, on: bool },
    /// Store a blend on the outgoing segment. Target ranges still only touch.
    SetTransition {
        segment_id: SegmentId,
        transition: Box<MaterialTransition>,
    },
    /// Give the clips one group, so they move together, or take them out
    /// of their groups with an empty `group_id`.
    SetGroup {
        segment_ids: Vec<SegmentId>,
        group_id: String,
    },
    /// Paste what Copy attributes took onto each clip that can take it:
    /// the placement onto clips that draw, the volume onto clips that
    /// sound. At least one clip must take something.
    PasteAttributes {
        segment_ids: Vec<SegmentId>,
        attributes: Box<crate::ClipAttributes>,
    },
    /// Take the blend off the outgoing segment, and its material off the
    /// draft. A segment with no blend is refused.
    RemoveTransition { segment_id: SegmentId },
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
    /// Activate or deactivate a clip. A deactivated clip stays on its row
    /// and keeps its edits, but is not drawn and not heard.
    SetActive { segment_id: SegmentId, active: bool },
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
    /// Steady a video clip at `level` (1 cuts least, 2 recommended, 3 most
    /// stable), or stop at 0 or below. The kept camera path stays.
    SetStable { segment_id: SegmentId, level: i64 },
    /// Enhance a video clip's quality at `level` (1 HD, 2 UHD), or stop at
    /// 0 or below. The enhanced copy is made apart from the draft.
    SetEnhance { segment_id: SegmentId, level: i64 },
    /// Reduce a video clip's noise at `level` (1 Weak, 2 Strong), or stop
    /// at 0 or below. The denoised copy is made apart from the draft.
    SetDenoise { segment_id: SegmentId, level: i64 },
    /// Fill a video clip with frames made between its own, to `fps` frames
    /// a second (30, 50 or 60) however slow it plays, or stop at 0 or
    /// below. The filled copy is made apart from the draft.
    SetInterpolate { segment_id: SegmentId, fps: i64 },
    /// Light a video clip again with the preset `preset` at `strength` (0
    /// to 1), or stop with an empty preset. The Custom light takes `color`
    /// (`#rrggbb`, empty for its default) and `angle` (degrees, 0 from the
    /// right, 90 from above). The depth it is lit from is made apart from
    /// the draft.
    SetRelight {
        segment_id: SegmentId,
        preset: String,
        strength: f64,
        color: String,
        angle: f64,
    },
    /// Cut a video clip to `ratio`, one of `Reframe::RATIOS`, following
    /// its subject, holding as still as `stability` (1 Flexible, 2
    /// Default, 3 Stable) and following as fast as `speed` (1 Fast, 2
    /// Default, 3 Slow); or stop with an empty ratio. Where the subject is
    /// is found apart from the draft.
    SetReframe {
        segment_id: SegmentId,
        ratio: String,
        stability: i64,
        speed: i64,
    },
    /// Blur a video clip along how things move through it, or stop when
    /// `on` is false. `blur` is the blur's length as a share of a frame's
    /// time and `blend` the share of the sharp frame mixed back in, both 0
    /// to 1; `direction` is 0 both ways, 1 forward or 2 backward; `times`
    /// is 1, 2, 4 or 6 frames made across each blur. The blurred copy is
    /// made apart from the draft.
    SetMotionBlur {
        segment_id: SegmentId,
        on: bool,
        blur: f64,
        blend: f64,
        direction: i64,
        times: i64,
    },
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
    /// Store left, center, or right across, or top, middle, or bottom down a
    /// vertical column (3 to 5). An empty line width takes `line_width`, and
    /// so does turning the text vertical or back.
    SetTextAlign {
        segment_id: SegmentId,
        alignment: i32,
        face: String,
        line_width: f64,
    },
    /// Turn bold, italic and underline on or off over the whole text. An
    /// empty face path is filled.
    SetTextPattern {
        segment_id: SegmentId,
        bold: bool,
        italic: bool,
        underline: bool,
        face: String,
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
    /// Shape the title's background as the [`cutlass_model::Bubble`] keyed
    /// `bubble`, or a plain box again when it is empty. A title with no
    /// background yet gets a white one, and white letters turn black.
    SetTextBubble {
        segment_id: SegmentId,
        bubble: String,
    },
    /// Draw the caption in the style keyed `style` (one of
    /// [`cutlass_model::CAPTION_STYLE_KEYS`]), or plain when it is empty.
    SetCaptionStyle {
        segment_id: SegmentId,
        style: String,
    },
    /// Record the caption template the caption is styled with, by its id
    /// and name, or take the record off when `resource_id` is empty. Its
    /// look is set by the requests that go with it.
    SetCaptionTemplate {
        segment_id: SegmentId,
        resource_id: String,
        name: String,
    },
    /// Mark word `word` of the caption as a keyword, drawn in the style's
    /// spoken look while it shows, or take the mark off.
    SetCaptionKeyword {
        segment_id: SegmentId,
        word: usize,
        on: bool,
    },
    /// Start each letter of a caption word `millis` after the one before
    /// it, from 0 to [`crate::MAX_CAPTION_STAGGER_MS`].
    SetCaptionStagger { segment_id: SegmentId, millis: i64 },
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
    /// Place a new `kind` shape from `start` for `duration` on `track_id`,
    /// a sticker row made when it is missing.
    AddShape {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        start: Micros,
        duration: Micros,
        kind: ShapeKind,
    },
    /// Place a shape drawn with the pen through `points`, in canvas pixels
    /// from the canvas middle with y up, from `start` for `duration` on
    /// `track_id`. The clip sits where the drawing was. A closed path is
    /// filled; an open one is a line.
    DrawShape {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        start: Micros,
        duration: Micros,
        points: Vec<ShapePoint>,
        closed: bool,
    },
    /// Make the shape `width` by `height` canvas pixels. Its points are
    /// stretched to the new size; the clip's scale stays.
    SetShapeSize {
        segment_id: SegmentId,
        width: f64,
        height: f64,
    },
    /// Fill the shape with `color` at `opacity` from 0 to 1. An empty color
    /// takes the fill off.
    SetShapeFill {
        segment_id: SegmentId,
        color: String,
        opacity: f64,
    },
    /// Stroke the shape's outline `width` canvas pixels wide in `color` at
    /// `opacity`, solid (0), dashed (1) or dotted (2). A width of 0 takes
    /// the stroke off.
    SetShapeStroke {
        segment_id: SegmentId,
        color: String,
        width: f64,
        opacity: f64,
        style: i32,
    },
    /// Round the shape's corners by `radius` canvas pixels: one value for
    /// every corner, or one per corner. Zeros take the rounding off.
    SetShapeCorners {
        segment_id: SegmentId,
        radius: Vec<f64>,
    },
    /// Cast a `color` shadow `distance` canvas pixels away at `angle`
    /// degrees, at `opacity`. An opacity of 0 takes the shadow off.
    SetShapeShadow {
        segment_id: SegmentId,
        color: String,
        distance: f64,
        angle: f64,
        opacity: f64,
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
    /// Run the effect clip `segment_id` over the picture clip `clip` alone,
    /// or with `None` over every picture below it again.
    BindEffect {
        segment_id: SegmentId,
        clip: Option<SegmentId>,
    },
    /// Blend the clip with the picture below by `mode`, a blend package id
    /// such as `blend-multiply`. Normal or empty takes the blend off.
    /// `material_id` names the new `mix_mode` material.
    SetBlend {
        segment_id: SegmentId,
        material_id: MaterialId,
        mode: String,
    },
    /// Put an adjustment layer from `start` for `duration` on `track_id`,
    /// an adjust row made when it is missing. Its sliders, set with
    /// [`Request::SetAdjust`], run over every picture below it.
    AddAdjustment {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        start: Micros,
        duration: Micros,
    },
    /// Put the bundled filter `filter_id`, such as `look/warm`, from `start`
    /// for `duration` on `track_id`, a filter row made when it is missing.
    /// It runs over every picture below it; its strength is set with
    /// [`Request::SetEffectStrength`].
    AddFilter {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
        start: Micros,
        duration: Micros,
        filter_id: String,
    },
    /// Set HSL colour `color`, 0 red to 7 magenta, on the clip: `hue`,
    /// `saturation` and `lightness` from -1 to 1. All zero takes it off.
    SetHsl {
        segment_id: SegmentId,
        material_id: MaterialId,
        color: u8,
        hue: f64,
        saturation: f64,
        lightness: f64,
    },
    /// Set the clip's curve `channel` (`rgb`, `red`, `green` or `blue`) to
    /// `points`, each `[x, y]` from 0 to 1. None, or the diagonal, is the
    /// straight line.
    SetCurve {
        segment_id: SegmentId,
        material_id: MaterialId,
        channel: String,
        points: Vec<[f64; 2]>,
    },
    /// Set the clip's colour wheel `wheel` (`lift`, `gamma` or `gain`): the
    /// puck at `x`, `y` inside the unit circle, the ring at `luma`.
    SetWheel {
        segment_id: SegmentId,
        material_id: MaterialId,
        wheel: String,
        x: f64,
        y: f64,
        luma: f64,
    },
    /// Put a LUT on the clip: `cube` is a `.cube` file's text, kept on the
    /// clip under `name`. `None` keeps the clip's table and sets its
    /// `strength`, from 0 to 1; an empty text takes the LUT off.
    SetLut {
        segment_id: SegmentId,
        material_id: MaterialId,
        name: String,
        cube: Option<String>,
        strength: f64,
    },
    /// Key `color` (`#rrggbb`) out of the clip with `strength` and
    /// `shadow`, each 0 to 1. An empty colour takes the key off.
    SetChroma {
        segment_id: SegmentId,
        material_id: MaterialId,
        color: String,
        strength: f64,
        shadow: f64,
    },
    /// Mask the clip with `mask`, in the clip's own frame, or take its mask
    /// off with `None`.
    SetMask {
        segment_id: SegmentId,
        material_id: MaterialId,
        mask: Option<cutlass_effects::Mask>,
    },
    /// Cut the video clip down to `selection`, the object click-to-select
    /// found, or take its selection off with `None`.
    SetSelection {
        segment_id: SegmentId,
        material_id: MaterialId,
        selection: Option<cutlass_effects::Selection>,
    },
    /// Make `followers` (video, sticker or text clips) follow the video clip
    /// `target` along `path`, the target's corners on the canvas frame by
    /// frame. Each follower keeps its offset from the target at `anchor`,
    /// the time the box was drawn at. The path is kept as the
    /// `video_tracking` record `tracking_id`. A clip in `masks` has its
    /// mask follow instead.
    Track {
        target: SegmentId,
        followers: Vec<SegmentId>,
        tracking_id: MaterialId,
        anchor: Micros,
        path: cutlass_model::TrackPath,
        /// Clips whose mask follows instead, each with the path in its own
        /// frame (-1 to 1, y up).
        #[serde(default)]
        masks: Vec<(SegmentId, cutlass_model::TrackPath)>,
    },
    /// Stop `segment_id` following the track it lists, and take off the
    /// keys the track wrote: its pose keys, or its mask's when its mask
    /// followed.
    Untrack { segment_id: SegmentId },
    /// Limit the clip's colour work, or an adjustment layer, to `mask`, or
    /// take that limit off with `None`.
    SetColorMask {
        segment_id: SegmentId,
        material_id: MaterialId,
        mask: Option<cutlass_effects::Mask>,
    },
    /// Set the adjust slider `key`, its reference name such as
    /// `temperature`, to `value`: -1 to 1, or 0 to 1 for fade, sharpen and
    /// grain. Zero takes it off. `material_id` names a new slider material.
    SetAdjust {
        segment_id: SegmentId,
        material_id: MaterialId,
        key: String,
        value: f64,
    },
    /// Store `value`, from 0 to 1, as the strength of the effect, filter,
    /// or adjustment the clip uses. When another clip uses that material too,
    /// a copy named `material_id` takes its place on this clip only.
    SetEffectStrength {
        segment_id: SegmentId,
        material_id: MaterialId,
        value: f64,
    },
    /// Set the control `control` (such as `speed`) of the recipe effect the
    /// clip uses to `value`, inside the control's range. When another clip
    /// uses that material too, a copy named `material_id` takes its place on
    /// this clip only.
    SetEffectControl {
        segment_id: SegmentId,
        material_id: MaterialId,
        control: String,
        value: f64,
    },
    /// Replace the segment's keyframe groups.
    SetKeyframes {
        segment_id: SegmentId,
        keyframes: Vec<CommonKeyframeGroup>,
    },
    /// Key the clip's place, size, turn, and opacity as they are at
    /// `offset` from its start. A key there takes those values again.
    /// Named `properties` (pose names such as `KFTypePositionX` or
    /// `KFTypeAlpha`) key only those parts and leave the rest of the pose
    /// alone; none keys the whole pose.
    AddKeyframe {
        segment_id: SegmentId,
        offset: Micros,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        properties: Vec<String>,
    },
    /// Key each named setting at `offset` from the clip's start: the
    /// volume (`KFTypeVolume`) or the mask (`KFTypeCommonMask*`), in the
    /// setting's own units. A key there takes the new value.
    KeyValues {
        segment_id: SegmentId,
        offset: Micros,
        values: Vec<(String, f64)>,
    },
    /// Take the keys on the named settings at `offset` off.
    UnkeyValues {
        segment_id: SegmentId,
        offset: Micros,
        properties: Vec<String>,
    },
    /// Remove the pose keys at `offset` from the clip's start.
    RemoveKeyframe {
        segment_id: SegmentId,
        offset: Micros,
    },
    /// Ease the span that ends at the key `offset` from the clip's start
    /// along the Graphs panel curve named `graph`, or make it straight with
    /// `None`. Only the keys on `properties` are eased; none eases every
    /// pose key, as before.
    SetKeyframeGraph {
        segment_id: SegmentId,
        offset: Micros,
        graph: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        properties: Vec<String>,
    },
    /// Give the clip the in or out animation whose resource id is
    /// `animation`, lasting `duration`, or clear that side with `None`.
    SetAnimation {
        segment_id: SegmentId,
        kind: AnimType,
        animation: Option<String>,
        duration: Micros,
    },
    /// Change the picture's place, size, and turn at `offset` from the
    /// clip's start. Each part that changed is keyed there when it already
    /// has keys and stored, as the three requests above do, when it has
    /// none. A part that did not change is left alone.
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
    /// Make the clips `segment_ids` one compound clip `segment_id` playing
    /// a draft of their own, stored as the material `material_id`.
    CreateCompound {
        segment_ids: Vec<SegmentId>,
        segment_id: SegmentId,
        material_id: MaterialId,
    },
    /// Put the clips the compound clip `segment_id` shows back on the
    /// timeline in its place.
    Ungroup { segment_id: SegmentId },
    /// Store `inside` as the draft the compound material `material_id`
    /// plays. A compound clip that showed all of it follows its length.
    SetInside {
        material_id: MaterialId,
        inside: Box<cutlass_model::Draft>,
    },
    /// Play `material_id` in `segment_id` in place of its media, keeping
    /// the clip's edits. A picture takes a picture, a sound a sound.
    ReplaceMedia {
        segment_id: SegmentId,
        material_id: MaterialId,
    },
    /// Point a material at the file stored at `path`, for media that has
    /// moved. Every clip of the material plays the new file in place of
    /// the old, which must reach as far as they do.
    LinkMedia {
        material_id: MaterialId,
        path: String,
        width: i32,
        height: i32,
        duration: Micros,
    },
    /// Add an empty lane. Video and mixed lanes are overlays above the main track.
    /// Audio is inserted under the main track. This does not place a segment.
    InsertTrack { track_id: TrackId, kind: TrackType },
    /// Add an empty lane at `index` in the track list, which is the row order.
    /// Audio goes under the main track and every other kind above it; an
    /// index outside the kind's block, or past the end, is refused.
    InsertTrackAt {
        track_id: TrackId,
        kind: TrackType,
        index: usize,
    },
    /// Take a lane and its clips off the draft. The main track and a locked
    /// track stay. Each clip leaves as a delete would take it.
    RemoveTrack { track_id: TrackId },
    /// Name a lane. An empty or blank name goes back to the default name.
    RenameTrack { track_id: TrackId, name: String },
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
    /// Stack picture clips by their own layer (on) or by their tracks
    /// (off, which puts every clip back on its track's layer).
    SetFreeLayer { on: bool },
    /// Move a picture clip up or down among those it overlaps. Needs
    /// Free layer on.
    Arrange {
        segment_id: SegmentId,
        step: Arrange,
    },
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Micros, Segment, SegmentId, TimeRange, TrackId};

    use super::{Request, TrimEdge};

    #[test]
    fn a_request_reads_from_json_with_its_op() {
        let requests: Vec<Request> = serde_json::from_str(
            r#"[
                {"op":"move","segment_id":"a","track_id":"t","start":250000},
                {"op":"trim","segment_id":"a","edge":"end","time":900000}
            ]"#,
        )
        .unwrap();
        assert_eq!(
            requests,
            [
                Request::Move {
                    segment_id: SegmentId::new("a"),
                    track_id: TrackId::new("t"),
                    start: Micros(250_000),
                },
                Request::Trim {
                    segment_id: SegmentId::new("a"),
                    edge: TrimEdge::End,
                    time: Micros(900_000),
                },
            ]
        );
        assert!(serde_json::from_str::<Request>(r#"{"op":"fly"}"#).is_err());
        assert!(serde_json::from_str::<Request>(r#"{"op":"move","segment_id":"a"}"#).is_err());
    }

    #[test]
    fn a_placed_segment_survives_a_round_trip() {
        let request = Request::Place {
            track_id: TrackId::new("main"),
            segment: Box::new(Segment::new("s", "m", TimeRange::new(0, 1_000_000))),
        };
        let text = serde_json::to_string(&request).unwrap();
        assert!(text.starts_with(r#"{"op":"place""#), "{text}");
        assert_eq!(serde_json::from_str::<Request>(&text).unwrap(), request);
    }
}
