//! The agent-facing wire format: the JSON surface the LLM sees and emits.
//!
//! Deliberately *not* serde derives on `cutlass-commands` — the wire layer is
//! shaped for LLM ergonomics (times in fractional seconds, ids as plain
//! integers, flat tagged objects) and keeps internal refactors from silently
//! changing the prompt-visible schema. Lowering to real engine commands (and
//! every guardrail) lives in [`crate::validate`].
//!
//! The vocabulary is closed by construction: project commands (open / save /
//! export / import) are not representable here, and [`WireGenerator`] carries
//! only the generator kinds the compositor actually renders — the phantom
//! sticker/effect/filter/adjustment variants cannot be expressed.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Bumped whenever the prompt-visible tool surface changes shape.
/// The snapshot test in `tests/tool_schema.rs` makes drift a reviewed diff.
///
/// 2: M2 keyframe commands (`set_param_keyframe`, `remove_param_keyframe`,
///    `set_param_constant`).
/// 3: M1 clip speed (`set_clip_speed`).
/// 4: M1 clip audio mix (`set_clip_audio`).
/// 5: M1 timeline markers (`add_marker`, `remove_marker`, `set_marker`).
/// 6: M1 crop + flip (`set_clip_crop`).
pub const TOOL_SCHEMA_VERSION: u32 = 6;

/// Track lane categories the agent may create or target.
///
/// The engine has more kinds (effect / filter / adjustment lanes); they are
/// placeholders that render nothing today, so the agent cannot create them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireTrackKind {
    /// Footage and other imported picture media.
    Video,
    /// Imported sound media.
    Audio,
    /// Titles and captions.
    Text,
    /// Graphic overlays: solid colors and shapes.
    Sticker,
}

/// Geometry of a generated shape clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireShape {
    Rectangle,
    Ellipse,
}

/// Synthetic clip content the agent may create.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireGenerator {
    /// A title / text layer (rendered with the default style; styling is
    /// preserved when replacing the text of an existing text clip).
    Text {
        /// The text to display.
        content: String,
    },
    /// A solid color fill covering the canvas.
    Solid {
        /// Fill color as `[red, green, blue, alpha]`, each 0-255.
        rgba: [u8; 4],
    },
    /// A filled vector shape centered on the canvas.
    Shape {
        shape: WireShape,
        /// Fill color as `[red, green, blue, alpha]`, each 0-255.
        rgba: [u8; 4],
    },
}

/// Add a track to the timeline stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddTrack {
    pub kind: WireTrackKind,
    /// Display name, e.g. "V2" or "Music".
    pub name: String,
    /// Stack position (0 = bottom layer, composited first). Omit to add on
    /// top of the stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

/// Place a trimmed range of imported media on a video or audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddClip {
    /// Target track id.
    pub track: u64,
    /// Media pool id of the source file.
    pub media: u64,
    /// In-point within the source media, in seconds.
    pub source_start: f64,
    /// Length of the source range to use, in seconds.
    pub source_duration: f64,
    /// Where the clip begins on the timeline, in seconds.
    pub start: f64,
}

/// Place a generated clip (text, solid color, shape) on a matching track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddGenerated {
    /// Target track id. Text goes on text tracks; solids and shapes go on
    /// sticker (overlay) tracks.
    pub track: u64,
    pub generator: WireGenerator,
    /// Where the clip begins on the timeline, in seconds.
    pub start: f64,
    /// Clip length on the timeline, in seconds.
    pub duration: f64,
}

/// Replace a generated clip's content (edit a title's text, recolor a
/// shape). Rejected for media-backed clips. Replacing the text of a text
/// clip keeps its current styling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetGenerator {
    /// The generated clip to modify.
    pub clip: u64,
    pub generator: WireGenerator,
}

/// Change a clip's placement on the canvas. Omitted fields keep their
/// current value. Rejected for clips on audio tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipTransform {
    pub clip: u64,
    /// Horizontal offset of the content center from the canvas center, as a
    /// fraction of canvas width (+ is right; 0.5 puts the center on the
    /// right edge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_x: Option<f64>,
    /// Vertical offset of the content center from the canvas center, as a
    /// fraction of canvas height (+ is down).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_y: Option<f64>,
    /// Uniform scale; 1.0 fits the content inside the canvas (100%).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Clockwise rotation in degrees about the content center.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f64>,
    /// Layer opacity, 0.0 (transparent) to 1.0 (opaque).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

/// Crop a clip to a sub-region of its frame and/or mirror it. Crop values
/// are the fractions trimmed off each edge (left 0.25 removes the left
/// quarter); the kept region aspect-fits the canvas exactly like the full
/// frame did, so cropping never moves the layer. Omitted fields keep
/// their current value. Rejected for clips on audio tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipCrop {
    pub clip: u64,
    /// Fraction of the frame width trimmed off the left edge (0–1). Omit
    /// to keep the current value; 0 restores the edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left: Option<f64>,
    /// Fraction of the frame height trimmed off the top edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top: Option<f64>,
    /// Fraction of the frame width trimmed off the right edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<f64>,
    /// Fraction of the frame height trimmed off the bottom edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bottom: Option<f64>,
    /// Mirror the content left-right. Omit to keep the current state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_h: Option<bool>,
    /// Mirror the content top-bottom. Omit to keep the current state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_v: Option<bool>,
}

/// An animatable clip property the keyframe commands can address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireClipParam {
    /// Canvas placement of the content center (vec2: use the `position`
    /// argument, not `value`).
    Position,
    /// Uniform scale (1.0 = fit inside the canvas).
    Scale,
    /// Clockwise rotation in degrees.
    Rotation,
    /// Layer opacity 0.0–1.0.
    Opacity,
}

/// Interpolation toward the next keyframe. (Custom bezier curves exist in
/// the engine but are inspector-only; the agent surface stays simple.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireEasing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

/// Add or replace a keyframe on one animatable clip property, making the
/// property animate over time. The first keyframe on a property turns its
/// fixed value into a curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetParamKeyframe {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline position of the keyframe, in seconds. Must fall inside the
    /// clip.
    pub at: f64,
    /// New value for `scale` / `rotation` / `opacity`. Ignored for
    /// `position`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position` (fractions of canvas size from center,
    /// +x right, +y down). Ignored for scalar params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// Interpolation toward the next keyframe. Defaults to linear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub easing: Option<WireEasing>,
}

/// Remove the keyframe at exactly a timeline position on one property.
/// Removing the last keyframe freezes the property at that value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveParamKeyframe {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline position of the keyframe to remove, in seconds.
    pub at: f64,
}

/// Set one animatable property to a fixed value, removing all its
/// keyframes (stops the animation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetParamConstant {
    pub clip: u64,
    pub param: WireClipParam,
    /// New value for `scale` / `rotation` / `opacity`. Ignored for
    /// `position`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position`. Ignored for scalar params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
}

/// Change a media clip's constant playback speed and/or direction. The clip
/// keeps its timeline start and source footage; its timeline length
/// re-derives from the speed (a 2x clip takes half the time). Audio of
/// retimed clips is muted. Not valid for generated clips (text/solid/shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipSpeed {
    /// The media clip to retime.
    pub clip: u64,
    /// Playback rate multiplier: 2.0 plays at double speed (half as long on
    /// the timeline), 0.5 is half-speed slow motion. Allowed range 0.05 to
    /// 100. Omit to keep the current speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Play the clip's footage backwards. Omit to keep the current
    /// direction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversed: Option<bool>,
}

/// Set a clip's audio mix: a constant volume gain plus linear fade-in/out
/// ramps. Volume 1.0 is unchanged, 0.0 mutes, 2.0 doubles (max 10). Fades
/// are seconds of ramp at the clip's head/tail. Audio lives on audio-lane
/// clips; for a video clip, target its linked audio companion instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipAudio {
    /// The audio-lane media clip to adjust.
    pub clip: u64,
    /// Gain multiplier: 0.0 mutes, 1.0 keeps the recorded level, up to a
    /// maximum of 10. Omit to keep the current volume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// Fade-in duration in seconds from the clip's start (0 removes the
    /// fade). Omit to keep the current fade-in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_in: Option<f64>,
    /// Fade-out duration in seconds ending at the clip's end (0 removes the
    /// fade). Omit to keep the current fade-out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_out: Option<f64>,
}

/// Split a clip at a timeline position into two abutting clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SplitClip {
    pub clip: u64,
    /// Timeline position of the cut, in seconds. Must fall strictly inside
    /// the clip.
    pub at: f64,
}

/// Re-place / trim a clip to a new timeline range. Trimming the head of a
/// media clip advances its source in-point to match (like dragging a trim
/// handle).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrimClip {
    pub clip: u64,
    /// New timeline start of the clip, in seconds.
    pub start: f64,
    /// New clip length, in seconds.
    pub duration: f64,
}

/// Move a clip to a track at a new start time, keeping its duration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MoveClip {
    pub clip: u64,
    /// Destination track id (may be the clip's current track).
    pub to_track: u64,
    /// New timeline start, in seconds.
    pub start: f64,
}

/// Remove a clip, leaving a gap where it sat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveClip {
    pub clip: u64,
}

/// Remove a track and any clips still on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveTrack {
    pub track: u64,
}

/// Toggle whether a visual track contributes to the composite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTrackEnabled {
    pub track: u64,
    pub enabled: bool,
}

/// Toggle whether an audio track is silenced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTrackMuted {
    pub track: u64,
    pub muted: bool,
}

/// Toggle whether a track's clips are editable (selection / move / trim).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTrackLocked {
    pub track: u64,
    pub locked: bool,
}

/// Remove a clip and slide later clips on its track left to close the gap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RippleDelete {
    pub clip: u64,
}

/// Shift every clip on a track that starts at or after `from` by `delta`
/// seconds (negative shifts left). Rejected if a left shift would collide
/// with an earlier clip or push a clip before time 0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShiftClips {
    pub track: u64,
    /// Clips starting at or after this timeline position (seconds) shift.
    pub from: f64,
    /// Signed shift amount in seconds.
    pub delta: f64,
}

/// Insert a trimmed range of media at a timeline position, first shifting
/// every clip at or after that position right to make room.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RippleInsert {
    /// Target track id (video or audio).
    pub track: u64,
    /// Media pool id of the source file.
    pub media: u64,
    /// In-point within the source media, in seconds.
    pub source_start: f64,
    /// Length of the source range to use, in seconds.
    pub source_duration: f64,
    /// Timeline position of the insert, in seconds.
    pub at: f64,
}

/// Put two or more clips into one link group: linked clips select, move,
/// and trim together. Replaces any previous links on those clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinkClips {
    /// Ids of the clips to link (at least two).
    pub clips: Vec<u64>,
}

/// Marker flag colors (the editor's fixed palette).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireMarkerColor {
    Teal,
    Blue,
    Purple,
    Pink,
    Red,
    Orange,
    Yellow,
    Green,
}

/// Drop a named, colored marker on the timeline ruler — an anchor for
/// navigation and for aligning edits ("cut at the marker", beat sync).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddMarker {
    /// Timeline position of the marker, in seconds.
    pub at: f64,
    /// Short label shown beside the flag (e.g. "Drop", "Beat 1"). Omit for
    /// an unnamed marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Flag color. Omit to cycle the palette automatically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<WireMarkerColor>,
}

/// Remove a timeline marker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveMarker {
    /// The marker to remove.
    pub marker: u64,
}

/// Move, rename, or recolor an existing timeline marker. Omitted fields
/// keep their current value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetMarker {
    /// The marker to change.
    pub marker: u64,
    /// New timeline position in seconds. Omit to keep the position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<f64>,
    /// New label ("" clears it). Omit to keep the name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// New flag color. Omit to keep the color.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<WireMarkerColor>,
}

/// Every timeline edit the agent may request, as one tagged value.
///
/// Tool calls arrive as `(name, arguments)` pairs and convert through
/// [`WireCommand::from_tool_call`]; serialized plans (dry-run previews,
/// eval fixtures) use the `command`-tagged JSON representation directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum WireCommand {
    AddTrack(AddTrack),
    AddClip(AddClip),
    AddGenerated(AddGenerated),
    SetGenerator(SetGenerator),
    SetClipTransform(SetClipTransform),
    SetClipCrop(SetClipCrop),
    SetParamKeyframe(SetParamKeyframe),
    RemoveParamKeyframe(RemoveParamKeyframe),
    SetParamConstant(SetParamConstant),
    SetClipSpeed(SetClipSpeed),
    SetClipAudio(SetClipAudio),
    SplitClip(SplitClip),
    TrimClip(TrimClip),
    MoveClip(MoveClip),
    RemoveClip(RemoveClip),
    RemoveTrack(RemoveTrack),
    SetTrackEnabled(SetTrackEnabled),
    SetTrackMuted(SetTrackMuted),
    SetTrackLocked(SetTrackLocked),
    RippleDelete(RippleDelete),
    ShiftClips(ShiftClips),
    RippleInsert(RippleInsert),
    LinkClips(LinkClips),
    AddMarker(AddMarker),
    RemoveMarker(RemoveMarker),
    SetMarker(SetMarker),
}

impl WireCommand {
    /// Rewrite clip/track/marker references through the given maps (ids
    /// absent from a map pass through unchanged).
    ///
    /// This is what makes plan replay work: a plan is recorded against a
    /// sandbox where `add_track`/`split_clip` allocated sandbox-local ids;
    /// when the live engine replays the plan, each created entity gets a
    /// fresh id, and later steps that referenced the sandbox id must be
    /// remapped onto the real one.
    pub fn remap_ids(
        &mut self,
        clip_map: &std::collections::HashMap<u64, u64>,
        track_map: &std::collections::HashMap<u64, u64>,
        marker_map: &std::collections::HashMap<u64, u64>,
    ) {
        let clip = |id: &mut u64| {
            if let Some(mapped) = clip_map.get(id) {
                *id = *mapped;
            }
        };
        let track = |id: &mut u64| {
            if let Some(mapped) = track_map.get(id) {
                *id = *mapped;
            }
        };
        let marker = |id: &mut u64| {
            if let Some(mapped) = marker_map.get(id) {
                *id = *mapped;
            }
        };
        match self {
            WireCommand::AddTrack(_) => {}
            WireCommand::AddClip(a) => track(&mut a.track),
            WireCommand::AddGenerated(a) => track(&mut a.track),
            WireCommand::SetGenerator(a) => clip(&mut a.clip),
            WireCommand::SetClipTransform(a) => clip(&mut a.clip),
            WireCommand::SetClipCrop(a) => clip(&mut a.clip),
            WireCommand::SetParamKeyframe(a) => clip(&mut a.clip),
            WireCommand::RemoveParamKeyframe(a) => clip(&mut a.clip),
            WireCommand::SetParamConstant(a) => clip(&mut a.clip),
            WireCommand::SetClipSpeed(a) => clip(&mut a.clip),
            WireCommand::SetClipAudio(a) => clip(&mut a.clip),
            WireCommand::SplitClip(a) => clip(&mut a.clip),
            WireCommand::TrimClip(a) => clip(&mut a.clip),
            WireCommand::MoveClip(a) => {
                clip(&mut a.clip);
                track(&mut a.to_track);
            }
            WireCommand::RemoveClip(a) => clip(&mut a.clip),
            WireCommand::RemoveTrack(a) => track(&mut a.track),
            WireCommand::SetTrackEnabled(a) => track(&mut a.track),
            WireCommand::SetTrackMuted(a) => track(&mut a.track),
            WireCommand::SetTrackLocked(a) => track(&mut a.track),
            WireCommand::RippleDelete(a) => clip(&mut a.clip),
            WireCommand::ShiftClips(a) => track(&mut a.track),
            WireCommand::RippleInsert(a) => track(&mut a.track),
            WireCommand::LinkClips(a) => a.clips.iter_mut().for_each(clip),
            WireCommand::AddMarker(_) => {}
            WireCommand::RemoveMarker(a) => marker(&mut a.marker),
            WireCommand::SetMarker(a) => marker(&mut a.marker),
        }
    }
}

/// One LLM tool: name, model-facing description, and a JSON Schema for its
/// arguments.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: serde_json::Value,
}

fn spec<T: JsonSchema>(name: &'static str, description: &'static str) -> ToolSpec {
    let parameters = serde_json::to_value(schemars::schema_for!(T))
        .expect("tool argument schemas are plain data and always serialize");
    ToolSpec {
        name,
        description,
        parameters,
    }
}

/// The read-only tool: returns the current project summary + editor
/// context. Not a [`WireCommand`] — the agent loop answers it without
/// touching dispatch.
pub fn describe_project_spec() -> ToolSpec {
    ToolSpec {
        name: "describe_project",
        description: "Get the current state of the project: tracks, clips with ids and \
                      times in seconds, the media pool, and the user's selection and \
                      playhead. Call this whenever you are unsure about ids or timing.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {},
        }),
    }
}

macro_rules! tools {
    ($( $name:literal => $variant:ident ( $args:ty ), $desc:literal; )+) => {
        /// The full tool surface, in stable order.
        pub fn tool_specs() -> Vec<ToolSpec> {
            vec![ $( spec::<$args>($name, $desc) ),+ ]
        }

        impl WireCommand {
            /// The tool name this command arrives under.
            pub fn tool_name(&self) -> &'static str {
                match self {
                    $( WireCommand::$variant(_) => $name, )+
                }
            }

            /// Decode a provider tool call. Unknown names and malformed
            /// arguments come back as model-readable messages.
            pub fn from_tool_call(
                name: &str,
                arguments: serde_json::Value,
            ) -> Result<WireCommand, String> {
                match name {
                    $(
                        $name => serde_json::from_value::<$args>(arguments)
                            .map(WireCommand::$variant)
                            .map_err(|e| format!("invalid arguments for {name}: {e}")),
                    )+
                    other => Err(format!(
                        "unknown tool '{other}'; available tools: {}",
                        [$($name),+].join(", ")
                    )),
                }
            }
        }
    };
}

tools! {
    "add_track" => AddTrack(AddTrack),
        "Add a track to the timeline stack (video, audio, text, or sticker overlay lane).";
    "add_clip" => AddClip(AddClip),
        "Place a trimmed range of an imported media file on a video or audio track. Times are in seconds.";
    "add_generated" => AddGenerated(AddGenerated),
        "Place a generated clip (text title, solid color, or shape) on a matching track. Times are in seconds.";
    "set_generator" => SetGenerator(SetGenerator),
        "Replace a generated clip's content: change a title's text (styling preserved) or recolor a solid/shape. Not valid for media clips.";
    "set_clip_transform" => SetClipTransform(SetClipTransform),
        "Change a clip's placement on the canvas: position, scale, rotation, opacity. Omitted fields keep their current value. Not valid on audio tracks.";
    "set_clip_crop" => SetClipCrop(SetClipCrop),
        "Crop a clip to a sub-region of its frame (fractions trimmed off each edge; 0 restores an edge) and/or mirror it with flip_h / flip_v. Omitted fields keep their current value. Not valid on audio tracks.";
    "set_param_keyframe" => SetParamKeyframe(SetParamKeyframe),
        "Add or replace a keyframe on a clip property (position, scale, rotation, opacity) at a timeline position in seconds, animating it over time. Use 'value' for scalar params, 'position' for position.";
    "remove_param_keyframe" => RemoveParamKeyframe(RemoveParamKeyframe),
        "Remove the keyframe at a timeline position (seconds) on a clip property. Removing the last keyframe freezes the property at that value.";
    "set_param_constant" => SetParamConstant(SetParamConstant),
        "Set a clip property to a fixed value and remove all its keyframes (stops its animation).";
    "set_clip_speed" => SetClipSpeed(SetClipSpeed),
        "Change a media clip's playback speed (2.0 = double speed, 0.5 = slow motion) and/or play it in reverse. The clip's timeline length re-derives from the speed; its audio is muted while retimed. Not valid for generated clips.";
    "set_clip_audio" => SetClipAudio(SetClipAudio),
        "Set an audio-lane clip's volume (0.0 mutes, 1.0 unchanged, 2.0 doubles) and/or fade-in/fade-out durations in seconds. Omitted fields keep their current value. For a video clip, target its linked audio companion clip.";
    "split_clip" => SplitClip(SplitClip),
        "Split a clip at a timeline position (seconds) into two abutting clips.";
    "trim_clip" => TrimClip(TrimClip),
        "Re-place / trim a clip to a new timeline start and duration in seconds. Trimming a media clip's head advances its source in-point.";
    "move_clip" => MoveClip(MoveClip),
        "Move a clip to a track at a new start time (seconds), keeping its duration.";
    "remove_clip" => RemoveClip(RemoveClip),
        "Remove a clip, leaving a gap where it sat.";
    "remove_track" => RemoveTrack(RemoveTrack),
        "Remove a track and any clips still on it.";
    "set_track_enabled" => SetTrackEnabled(SetTrackEnabled),
        "Show or hide a visual track in the composite.";
    "set_track_muted" => SetTrackMuted(SetTrackMuted),
        "Mute or unmute an audio track.";
    "set_track_locked" => SetTrackLocked(SetTrackLocked),
        "Lock or unlock a track's clips against editing.";
    "ripple_delete" => RippleDelete(RippleDelete),
        "Remove a clip and slide later clips on its track left to close the gap.";
    "shift_clips" => ShiftClips(ShiftClips),
        "Shift every clip on a track starting at/after a position by a signed number of seconds.";
    "ripple_insert" => RippleInsert(RippleInsert),
        "Insert a trimmed range of media at a timeline position, shifting later clips right to make room. Times are in seconds.";
    "link_clips" => LinkClips(LinkClips),
        "Link two or more clips so they select, move, and trim together (replaces their previous links).";
    "add_marker" => AddMarker(AddMarker),
        "Drop a named, colored marker on the timeline ruler at a position in seconds. Omit color to cycle the palette.";
    "remove_marker" => RemoveMarker(RemoveMarker),
        "Remove a ruler marker by id.";
    "set_marker" => SetMarker(SetMarker),
        "Move, rename, or recolor a ruler marker. Omitted fields keep their current value.";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_json_round_trips() {
        let cmd = WireCommand::TrimClip(TrimClip {
            clip: 12,
            start: 14.0,
            duration: 4.0,
        });
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "command": "trim_clip",
                "clip": 12,
                "start": 14.0,
                "duration": 4.0,
            })
        );
        let back: WireCommand = serde_json::from_value(json).unwrap();
        assert_eq!(back, cmd);
    }

    #[test]
    fn from_tool_call_decodes_arguments() {
        let cmd = WireCommand::from_tool_call(
            "split_clip",
            serde_json::json!({ "clip": 7, "at": 12.4 }),
        )
        .unwrap();
        assert_eq!(
            cmd,
            WireCommand::SplitClip(SplitClip { clip: 7, at: 12.4 })
        );
        assert_eq!(cmd.tool_name(), "split_clip");
    }

    #[test]
    fn from_tool_call_rejects_unknown_tool_and_bad_args() {
        let err = WireCommand::from_tool_call("save_project", serde_json::json!({})).unwrap_err();
        assert!(err.contains("unknown tool 'save_project'"));
        assert!(err.contains("add_clip"));

        let err = WireCommand::from_tool_call(
            "trim_clip",
            serde_json::json!({ "clip": "not-a-number" }),
        )
        .unwrap_err();
        assert!(err.contains("invalid arguments for trim_clip"));
    }

    #[test]
    fn generator_wire_format_is_tagged_lowercase() {
        let shape = WireGenerator::Shape {
            shape: WireShape::Ellipse,
            rgba: [255, 0, 0, 255],
        };
        assert_eq!(
            serde_json::to_value(&shape).unwrap(),
            serde_json::json!({ "type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255] })
        );
    }

    #[test]
    fn remap_ids_rewrites_only_mapped_references() {
        let clip_map = std::collections::HashMap::from([(10u64, 99u64)]);
        let track_map = std::collections::HashMap::from([(2u64, 7u64)]);
        let marker_map = std::collections::HashMap::from([(4u64, 40u64)]);

        let mut mv = WireCommand::MoveClip(MoveClip {
            clip: 10,
            to_track: 2,
            start: 1.0,
        });
        mv.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            mv,
            WireCommand::MoveClip(MoveClip {
                clip: 99,
                to_track: 7,
                start: 1.0,
            })
        );

        // Unmapped ids pass through; link lists remap element-wise.
        let mut link = WireCommand::LinkClips(LinkClips {
            clips: vec![10, 11],
        });
        link.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            link,
            WireCommand::LinkClips(LinkClips {
                clips: vec![99, 11],
            })
        );

        // Marker references follow the marker map (sandbox add_marker ids
        // land on the live engine's ids during plan replay).
        let mut set = WireCommand::SetMarker(SetMarker {
            marker: 4,
            at: Some(2.0),
            name: None,
            color: None,
        });
        set.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            set,
            WireCommand::SetMarker(SetMarker {
                marker: 40,
                at: Some(2.0),
                name: None,
                color: None,
            })
        );
    }

    #[test]
    fn tool_specs_cover_every_command_with_object_schemas() {
        let specs = tool_specs();
        assert_eq!(specs.len(), 26);
        for spec in &specs {
            assert!(!spec.description.is_empty(), "{} missing description", spec.name);
            assert_eq!(
                spec.parameters.get("type").and_then(|t| t.as_str()),
                Some("object"),
                "{} schema is not an object",
                spec.name
            );
        }
    }
}
