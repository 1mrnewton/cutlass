//! Gesture-level editor intents, composed into engine command batches.
//!
//! The shells speak in gestures ("append these picks", "ripple-trim this main
//! clip", "drop a text at the playhead"); the engine speaks in atomic
//! [`EditCommand`]s. The translation — seconds→ticks, host-lane selection,
//! ripple choreography, undo grouping — is editor *logic* shared by iOS and
//! Android, so it lives here in Rust. Every multi-command intent runs inside
//! one history group: it undoes as a single step and rolls back atomically on
//! mid-flight failure.
//!
//! Wire shape: `{"intent": "add_text", ...}` (snake_case tags and fields,
//! seconds as `f64`). Responses are small JSON objects with the created ids;
//! shells refresh the full `ui_state` after every intent anyway.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineError};
use cutlass_models::{
    AudioRole, Clip, ClipId, ClipParam, ClipSource, ClipTransform, Easing, Generator, MediaId,
    ModelError, Param, ParamValue, Project, Rational, RationalTime, Scale2, TimeRange, TrackId,
    TrackKind,
};

/// Which clip edge a trim gesture drags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrimEdge {
    Leading,
    Trailing,
}

/// Lane flavor for generated full-canvas bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectLaneKind {
    Effect,
    Filter,
    Adjustment,
}

impl EffectLaneKind {
    fn track_kind(self) -> TrackKind {
        match self {
            EffectLaneKind::Effect => TrackKind::Effect,
            EffectLaneKind::Filter => TrackKind::Filter,
            EffectLaneKind::Adjustment => TrackKind::Adjustment,
        }
    }

    fn generator(self) -> Generator {
        match self {
            EffectLaneKind::Effect => Generator::Effect,
            EffectLaneKind::Filter => Generator::Filter,
            EffectLaneKind::Adjustment => Generator::Adjustment,
        }
    }
}

/// A gesture-level operation. See the module docs for the wire shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "intent", rename_all = "snake_case")]
pub enum Intent {
    /// Import each file and append it to the end of the main track (creating
    /// the main track on first use). Audio-only files land on an audio lane
    /// instead.
    AppendMain { paths: Vec<PathBuf> },
    /// Import a file and ripple-insert it at the main-track clip boundary
    /// nearest `at_seconds` (later clips shift right).
    InsertMain { path: PathBuf, at_seconds: f64 },
    /// Split any clip at an absolute timeline position.
    Split { clip: ClipId, seconds: f64 },
    /// Trim a *main-track* clip with ripple semantics: the track stays
    /// gapless, later clips slide. `delta_seconds` is the signed change of
    /// the dragged edge (leading: + trims content in; trailing: + extends).
    RippleTrimMain {
        clip: ClipId,
        edge: TrimEdge,
        delta_seconds: f64,
    },
    /// Re-place a free lane clip (start and/or length), non-ripple.
    TrimLane {
        clip: ClipId,
        start_seconds: f64,
        length_seconds: f64,
    },
    /// Move a lane clip (or lift a main clip) to a lane at `start_seconds`:
    /// a specific lane when `track` is given, otherwise the first same-kind
    /// lane with a free span (a new lane is created when none fits).
    MoveLane {
        clip: ClipId,
        #[serde(default)]
        track: Option<TrackId>,
        start_seconds: f64,
    },
    /// Insert a clip into the main track at slot `index` (post-removal
    /// indexing when the clip already lives on main — a reorder).
    InsertIntoMain { clip: ClipId, index: usize },
    /// Drop a text clip at the playhead on a text lane.
    AddText {
        text: String,
        at_seconds: f64,
        #[serde(default = "default_overlay_seconds")]
        duration_seconds: f64,
    },
    /// Drop a sticker clip. `asset` is a catalog id
    /// ([`cutlass_models::sticker_catalog`]); omitted (older callers) means
    /// the first bundled sticker.
    AddSticker {
        at_seconds: f64,
        #[serde(default = "default_overlay_seconds")]
        duration_seconds: f64,
        #[serde(default)]
        asset: Option<String>,
    },
    /// Drop an effect / filter / adjustment bar.
    AddEffect {
        kind: EffectLaneKind,
        at_seconds: f64,
        #[serde(default = "default_overlay_seconds")]
        duration_seconds: f64,
    },
    /// Import a file and drop it as picture-in-picture on an overlay video
    /// lane (half scale, upper half of the canvas — the CapCut drop pose).
    AddPip { path: PathBuf, at_seconds: f64 },
    /// Import a file and drop it on an audio lane. `role` tags the clip with
    /// the picker tab it came from (music / sfx / voiceover).
    AddAudio {
        path: PathBuf,
        at_seconds: f64,
        #[serde(default)]
        role: Option<AudioRole>,
    },
    /// Duplicate a clip right after itself (ripple on main, free slot on a
    /// lane). Copies content, speed, transform, audio mix, crop, effect ids,
    /// and look fields (mask/chroma/stabilize/filter/adjust/animations/role);
    /// keyframed animation flattens to its start value, speed curves reset.
    Duplicate { clip: ClipId },
    /// Swap a media clip's source for a new file, keeping its slot. The
    /// replacement must be at least as long as the clip needs (images are
    /// exempt: one frame repeats).
    ReplaceMedia { clip: ClipId, path: PathBuf },
    /// CapCut "extract audio": place the clip's audio on an audio lane,
    /// linked to the original (whose own lane goes silent via linkage).
    ExtractAudio { clip: ClipId },
    /// CapCut "freeze": extract the main-track clip's frame under the
    /// playhead, write it to `png_path`, import it as a still, and
    /// ripple-insert it at the playhead (splitting the clip when mid-clip).
    Freeze {
        clip: ClipId,
        seconds: f64,
        /// Destination for the extracted frame — the shell owns media
        /// placement (project media dir), Rust owns the pixels.
        png_path: PathBuf,
        #[serde(default = "default_freeze_seconds")]
        duration_seconds: f64,
    },
    /// Retime a media clip. `speed` is the positive rate multiplier.
    SetSpeed {
        clip: ClipId,
        speed: f64,
        #[serde(default)]
        reversed: bool,
    },
    /// Apply a speed-curve preset from the catalog (`None` restores constant
    /// speed). The clip's duration re-derives from the curve in one undo step.
    SetSpeedPreset {
        clip: ClipId,
        #[serde(default)]
        preset: Option<String>,
    },
    /// Volume slider + fade handles. `volume: None` keeps the current gain.
    SetAudio {
        clip: ClipId,
        #[serde(default)]
        volume: Option<f32>,
        #[serde(default)]
        fade_in_seconds: f64,
        #[serde(default)]
        fade_out_seconds: f64,
    },
    /// Canvas placement in UI coordinates (`pos` 0..1, 0.5 = center). With
    /// `at_seconds` the edit composes with animation (keyframe write on
    /// animated properties); without it every property flattens to constant.
    SetTransform {
        clip: ClipId,
        pos_x: f32,
        pos_y: f32,
        scale: f32,
        rotation_degrees: f32,
        opacity: f32,
        #[serde(default)]
        at_seconds: Option<f64>,
    },
    /// Stamp (or, when one already sits there, remove) a transform keyframe
    /// at an absolute timeline position — the timeline's diamond toggle.
    ToggleTransformKeyframe { clip: ClipId, seconds: f64 },
    /// Set, change, or clear (`transition_id: None`) the transition at a
    /// main-track clip's right junction.
    SetTransition {
        clip: ClipId,
        #[serde(default)]
        transition_id: Option<String>,
        #[serde(default)]
        duration_seconds: f64,
    },
}

fn default_overlay_seconds() -> f64 {
    3.0
}

fn default_freeze_seconds() -> f64 {
    3.0
}

/// Execute one intent. Multi-command intents run in a history group: one undo
/// step, atomic rollback on failure. Returns the intent's response payload.
pub fn run(engine: &mut Engine, intent: Intent) -> Result<serde_json::Value, EngineError> {
    match intent {
        Intent::AppendMain { paths } => grouped(engine, |e| append_main(e, &paths)),
        Intent::InsertMain { path, at_seconds } => {
            grouped(engine, |e| insert_main(e, &path, at_seconds))
        }
        Intent::Split { clip, seconds } => split(engine, clip, seconds),
        Intent::RippleTrimMain {
            clip,
            edge,
            delta_seconds,
        } => grouped(engine, |e| ripple_trim_main(e, clip, edge, delta_seconds)),
        Intent::TrimLane {
            clip,
            start_seconds,
            length_seconds,
        } => trim_lane(engine, clip, start_seconds, length_seconds),
        Intent::MoveLane {
            clip,
            track,
            start_seconds,
        } => grouped(engine, |e| move_lane(e, clip, track, start_seconds)),
        Intent::InsertIntoMain { clip, index } => {
            grouped(engine, |e| insert_into_main(e, clip, index))
        }
        Intent::AddText {
            text,
            at_seconds,
            duration_seconds,
        } => grouped(engine, |e| {
            add_generated(
                e,
                TrackKind::Text,
                Generator::text(text.clone()),
                at_seconds,
                duration_seconds,
            )
        }),
        Intent::AddSticker {
            at_seconds,
            duration_seconds,
            asset,
        } => grouped(engine, |e| {
            let asset = asset
                .clone()
                .or_else(|| {
                    cutlass_models::sticker_catalog()
                        .first()
                        .map(|s| s.id.to_owned())
                })
                .unwrap_or_default();
            add_generated(
                e,
                TrackKind::Sticker,
                Generator::Sticker { asset },
                at_seconds,
                duration_seconds,
            )
        }),
        Intent::AddEffect {
            kind,
            at_seconds,
            duration_seconds,
        } => grouped(engine, |e| {
            add_generated(
                e,
                kind.track_kind(),
                kind.generator(),
                at_seconds,
                duration_seconds,
            )
        }),
        Intent::AddPip { path, at_seconds } => grouped(engine, |e| add_pip(e, &path, at_seconds)),
        Intent::AddAudio {
            path,
            at_seconds,
            role,
        } => grouped(engine, |e| add_audio(e, &path, at_seconds, role)),
        Intent::Duplicate { clip } => grouped(engine, |e| duplicate(e, clip)),
        Intent::ReplaceMedia { clip, path } => grouped(engine, |e| replace_media(e, clip, &path)),
        Intent::ExtractAudio { clip } => extract_audio(engine, clip),
        Intent::Freeze {
            clip,
            seconds,
            png_path,
            duration_seconds,
        } => grouped(engine, |e| {
            freeze(e, clip, seconds, &png_path, duration_seconds)
        }),
        Intent::SetSpeed {
            clip,
            speed,
            reversed,
        } => set_speed(engine, clip, speed, reversed),
        Intent::SetSpeedPreset { clip, preset } => set_speed_preset(engine, clip, preset),
        Intent::SetAudio {
            clip,
            volume,
            fade_in_seconds,
            fade_out_seconds,
        } => set_audio(engine, clip, volume, fade_in_seconds, fade_out_seconds),
        Intent::SetTransform {
            clip,
            pos_x,
            pos_y,
            scale,
            rotation_degrees,
            opacity,
            at_seconds,
        } => set_transform(
            engine,
            clip,
            [pos_x, pos_y],
            scale,
            rotation_degrees,
            opacity,
            at_seconds,
        ),
        Intent::ToggleTransformKeyframe { clip, seconds } => {
            grouped(engine, |e| toggle_transform_keyframe(e, clip, seconds))
        }
        Intent::SetTransition {
            clip,
            transition_id,
            duration_seconds,
        } => grouped(engine, |e| {
            set_transition(e, clip, transition_id.clone(), duration_seconds)
        }),
    }
}

// --- tick math --------------------------------------------------------------

/// Nearest frame tick at `rate` for a wall-clock position (never negative).
pub fn frame_tick(seconds: f64, rate: Rational) -> i64 {
    (seconds.max(0.0) * rate.as_f64()).round() as i64
}

fn at(seconds: f64, rate: Rational) -> RationalTime {
    RationalTime::new(frame_tick(seconds, rate), rate)
}

/// A UI speed multiplier as the exact rational the model stores (milli
/// precision, clamped to the model's speed range).
pub fn speed_to_rational(speed: f64) -> Rational {
    let clamped = speed.clamp(0.05, 100.0);
    Rational::new((clamped * 1000.0).round() as i32, 1000)
}

fn invalid(msg: impl Into<String>) -> EngineError {
    ModelError::InvalidParam(msg.into()).into()
}

// --- engine plumbing ---------------------------------------------------------

/// Run `f` inside one history group; commit on success, roll back on failure.
fn grouped<T>(
    engine: &mut Engine,
    f: impl FnOnce(&mut Engine) -> Result<T, EngineError>,
) -> Result<T, EngineError> {
    engine.begin_group();
    match f(engine) {
        Ok(value) => {
            engine.commit_group();
            Ok(value)
        }
        Err(err) => {
            engine.rollback_group();
            Err(err)
        }
    }
}

fn apply_created(engine: &mut Engine, command: EditCommand) -> Result<ClipId, EngineError> {
    match engine.apply(Command::Edit(command))? {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => Ok(id),
        other => Err(invalid(format!("expected a created clip, got {other:?}"))),
    }
}

fn apply_created_track(engine: &mut Engine, command: EditCommand) -> Result<TrackId, EngineError> {
    match engine.apply(Command::Edit(command))? {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => Ok(id),
        other => Err(invalid(format!("expected a created track, got {other:?}"))),
    }
}

fn apply_imported(engine: &mut Engine, path: &std::path::Path) -> Result<MediaId, EngineError> {
    match engine.apply(Command::Project(ProjectCommand::Import {
        path: path.to_path_buf(),
    }))? {
        ApplyOutcome::Imported { media } => Ok(media),
        other => Err(invalid(format!(
            "expected an import outcome, got {other:?}"
        ))),
    }
}

fn clip_snapshot(project: &Project, clip: ClipId) -> Result<(TrackId, Clip), EngineError> {
    let track = project
        .timeline()
        .track_of(clip)
        .ok_or(ModelError::UnknownClip(clip))?;
    let snapshot = project
        .clip(clip)
        .ok_or(ModelError::UnknownClip(clip))?
        .clone();
    Ok((track, snapshot))
}

// --- lane selection (the shared "host lane" policy) ---------------------------

/// Where a clip of `kind` spanning `range` should land.
#[derive(Debug, PartialEq, Eq)]
pub enum HostLanePlan {
    Existing(TrackId),
    /// Create a lane at this stack index (`None` = top of the stack).
    Create {
        index: Option<usize>,
    },
}

/// First non-main lane of `kind` (scanning UI rows top to bottom) whose span
/// at `range` is free; otherwise a plan to create one at the kind's default
/// position — video just above main, generated kinds at the bottom of the
/// overlay block, audio at the bottom of the stack.
pub fn host_lane_plan(
    project: &Project,
    kind: TrackKind,
    range: TimeRange,
    exclude: Option<ClipId>,
) -> Result<HostLanePlan, EngineError> {
    let main = crate::ui_state::main_track(project);
    let timeline = project.timeline();
    for id in crate::ui_state::lane_rows(project) {
        if Some(id) == main {
            continue;
        }
        let Some(track) = timeline.track(id) else {
            continue;
        };
        if track.kind == kind && !track.has_overlap(range, exclude)? {
            return Ok(HostLanePlan::Existing(id));
        }
    }
    Ok(HostLanePlan::Create {
        index: new_lane_index(project, kind),
    })
}

/// Stack position (0 = bottom) for a brand-new lane of `kind`, mirroring the
/// shells' row conventions: video overlays go directly above main; text /
/// sticker / effect kinds slot in *below* the existing overlay block (bottom
/// row of that group in the UI); audio appends and the timeline's audio-floor
/// invariant sinks it.
fn new_lane_index(project: &Project, kind: TrackKind) -> Option<usize> {
    let order = project.timeline().order();
    let position = |id: TrackId| order.iter().position(|t| *t == id);
    match kind {
        TrackKind::Video => {
            let main = crate::ui_state::main_track(project)?;
            position(main).map(|p| p + 1)
        }
        TrackKind::Audio => None,
        _ => project
            .timeline()
            .tracks_ordered()
            .filter(|t| {
                matches!(
                    t.kind,
                    TrackKind::Text
                        | TrackKind::Sticker
                        | TrackKind::Effect
                        | TrackKind::Filter
                        | TrackKind::Adjustment
                )
            })
            .filter_map(|t| position(t.id))
            .min(),
    }
}

fn default_lane_name(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "Overlay",
        TrackKind::Audio => "Audio",
        TrackKind::Text => "Text",
        TrackKind::Sticker => "Sticker",
        TrackKind::Effect => "Effects",
        TrackKind::Filter => "Filters",
        TrackKind::Adjustment => "Adjust",
    }
}

/// Resolve a [`host_lane_plan`], creating the lane when needed.
fn ensure_host_lane(
    engine: &mut Engine,
    kind: TrackKind,
    range: TimeRange,
    exclude: Option<ClipId>,
) -> Result<TrackId, EngineError> {
    match host_lane_plan(engine.project(), kind, range, exclude)? {
        HostLanePlan::Existing(id) => Ok(id),
        HostLanePlan::Create { index } => apply_created_track(
            engine,
            EditCommand::AddTrack {
                kind,
                name: default_lane_name(kind).into(),
                index,
                pinned: false,
            },
        ),
    }
}

/// The main track, created (at the bottom of the stack) on first use.
fn ensure_main(engine: &mut Engine) -> Result<TrackId, EngineError> {
    if let Some(main) = crate::ui_state::main_track(engine.project()) {
        return Ok(main);
    }
    apply_created_track(
        engine,
        EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: "Main".into(),
            index: Some(0),
            pinned: false,
        },
    )
}

// --- import + placement intents ----------------------------------------------

fn append_main(engine: &mut Engine, paths: &[PathBuf]) -> Result<serde_json::Value, EngineError> {
    let main = ensure_main(engine)?;
    let rate = engine.project().timeline().frame_rate;
    let mut clips = Vec::new();
    let mut media_ids = Vec::new();

    for path in paths {
        let media_id = apply_imported(engine, path)?;
        media_ids.push(media_id.raw());
        let media = engine
            .project()
            .media(media_id)
            .ok_or(ModelError::UnknownMedia(media_id))?;
        let source = media.full_range();

        let clip = if media.is_audio_only() {
            // Music/voiceover picks can't sit on the main picture track.
            let main_end = main_track_end(engine.project(), main);
            let len = cutlass_models::resample(source.duration, rate).value.max(1);
            let range = TimeRange::at_rate(main_end, len, rate);
            let lane = ensure_host_lane(engine, TrackKind::Audio, range, None)?;
            apply_created(
                engine,
                EditCommand::AddClip {
                    track: lane,
                    media: media_id,
                    source,
                    start: RationalTime::new(main_end, rate),
                },
            )?
        } else {
            let main_end = main_track_end(engine.project(), main);
            apply_created(
                engine,
                EditCommand::AddClip {
                    track: main,
                    media: media_id,
                    source,
                    start: RationalTime::new(main_end, rate),
                },
            )?
        };
        clips.push(clip.raw());
    }

    Ok(serde_json::json!({ "clips": clips, "media": media_ids }))
}

fn main_track_end(project: &Project, main: TrackId) -> i64 {
    project
        .timeline()
        .track(main)
        .map(|t| t.content_end())
        .unwrap_or(0)
}

/// The main-track clip boundary tick nearest `t` (a ripple insert must land
/// between clips, never inside one).
pub fn nearest_main_boundary(project: &Project, main: TrackId, tick: i64) -> i64 {
    let Some(track) = project.timeline().track(main) else {
        return 0;
    };
    let mut boundaries = vec![0i64];
    boundaries.extend(track.clips_ordered().iter().map(|c| c.timeline.end_tick()));
    boundaries
        .into_iter()
        .min_by_key(|b| (b - tick).abs())
        .unwrap_or(0)
}

fn insert_main(
    engine: &mut Engine,
    path: &std::path::Path,
    at_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let main = ensure_main(engine)?;
    let rate = engine.project().timeline().frame_rate;
    let media_id = apply_imported(engine, path)?;
    let media = engine
        .project()
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;
    let source = media.full_range();
    let tick = nearest_main_boundary(engine.project(), main, frame_tick(at_seconds, rate));
    let clip = apply_created(
        engine,
        EditCommand::RippleInsert {
            track: main,
            media: media_id,
            source,
            at: RationalTime::new(tick, rate),
        },
    )?;
    Ok(serde_json::json!({ "clip": clip.raw(), "media": media_id.raw() }))
}

fn add_generated(
    engine: &mut Engine,
    kind: TrackKind,
    generator: Generator,
    at_seconds: f64,
    duration_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let start = frame_tick(at_seconds, rate);
    let len = frame_tick(duration_seconds.max(0.1), rate).max(1);
    let range = TimeRange::at_rate(start, len, rate);
    let lane = ensure_host_lane(engine, kind, range, None)?;
    let clip = apply_created(
        engine,
        EditCommand::AddGenerated {
            track: lane,
            generator,
            timeline: range,
        },
    )?;
    Ok(serde_json::json!({ "clip": clip.raw(), "track": lane.raw() }))
}

fn add_pip(
    engine: &mut Engine,
    path: &std::path::Path,
    at_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let media_id = apply_imported(engine, path)?;
    let media = engine
        .project()
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;
    if media.is_audio_only() {
        return Err(invalid("picture-in-picture needs visual media"));
    }
    let source = media.full_range();
    let start = frame_tick(at_seconds, rate);
    let len = cutlass_models::resample(source.duration, rate).value.max(1);
    let range = TimeRange::at_rate(start, len, rate);
    let lane = ensure_host_lane(engine, TrackKind::Video, range, None)?;
    let clip = apply_created(
        engine,
        EditCommand::AddClip {
            track: lane,
            media: media_id,
            source,
            start: RationalTime::new(start, rate),
        },
    )?;
    // The CapCut drop pose: half scale, sitting in the upper half.
    engine.apply(Command::Edit(EditCommand::SetClipTransform {
        clip,
        transform: ClipTransform {
            position: [0.0, -0.18],
            anchor_point: [0.5, 0.5],
            scale: 0.5.into(),
            rotation: 0.0,
            opacity: 1.0,
        },
        at: None,
    }))?;
    Ok(serde_json::json!({ "clip": clip.raw(), "track": lane.raw(), "media": media_id.raw() }))
}

fn add_audio(
    engine: &mut Engine,
    path: &std::path::Path,
    at_seconds: f64,
    role: Option<AudioRole>,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let media_id = apply_imported(engine, path)?;
    let media = engine
        .project()
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;
    if !media.has_audio {
        return Err(invalid("file has no audio stream"));
    }
    let source = media.full_range();
    let start = frame_tick(at_seconds, rate);
    let len = cutlass_models::resample(source.duration, rate).value.max(1);
    let range = TimeRange::at_rate(start, len, rate);
    let lane = ensure_host_lane(engine, TrackKind::Audio, range, None)?;
    let clip = apply_created(
        engine,
        EditCommand::AddClip {
            track: lane,
            media: media_id,
            source,
            start: RationalTime::new(start, rate),
        },
    )?;
    if role.is_some() {
        engine.apply(Command::Edit(EditCommand::SetAudioRole { clip, role }))?;
    }
    Ok(serde_json::json!({ "clip": clip.raw(), "track": lane.raw(), "media": media_id.raw() }))
}

// --- structural intents --------------------------------------------------------

fn split(
    engine: &mut Engine,
    clip: ClipId,
    seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let new = apply_created(
        engine,
        EditCommand::SplitClip {
            clip,
            at: at(seconds, rate),
        },
    )?;
    Ok(serde_json::json!({ "clip": new.raw() }))
}

fn require_main_clip(project: &Project, clip: ClipId) -> Result<(TrackId, Clip), EngineError> {
    let (track, snapshot) = clip_snapshot(project, clip)?;
    if Some(track) != crate::ui_state::main_track(project) {
        return Err(invalid("clip is not on the main track"));
    }
    Ok((track, snapshot))
}

fn ripple_trim_main(
    engine: &mut Engine,
    clip: ClipId,
    edge: TrimEdge,
    delta_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let (track, snapshot) = require_main_clip(engine.project(), clip)?;
    let s = snapshot.timeline.start.value;
    let d = snapshot.timeline.duration.value;

    // Clamp shrink so at least one frame survives; growth is validated by
    // TrimClip against the source window.
    let dt = frame_tick(delta_seconds.abs(), rate) * delta_seconds.signum() as i64;
    let dt = match edge {
        TrimEdge::Leading => dt.min(d - 1),
        TrimEdge::Trailing => dt.max(-(d - 1)),
    };
    if dt == 0 {
        return Ok(serde_json::json!({ "clip": clip.raw() }));
    }

    let shift = |from: i64, delta: i64| EditCommand::ShiftClips {
        track,
        from: RationalTime::new(from, rate),
        delta: RationalTime::new(delta, rate),
    };
    let trim = |start: i64, len: i64| EditCommand::TrimClip {
        clip,
        timeline: TimeRange::at_rate(start, len, rate),
    };

    // Ordering rule: free space before consuming it. Shrinks trim first and
    // close the gap after; extensions open space first and trim into it.
    let commands: [EditCommand; 2] = match (edge, dt > 0) {
        (TrimEdge::Leading, true) => [trim(s + dt, d - dt), shift(s + dt, -dt)],
        (TrimEdge::Leading, false) => [shift(s, -dt), trim(s, d - dt)],
        (TrimEdge::Trailing, true) => [shift(s + d, dt), trim(s, d + dt)],
        (TrimEdge::Trailing, false) => [trim(s, d + dt), shift(s + d, dt)],
    };
    for command in commands {
        engine.apply(Command::Edit(command))?;
    }
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

fn trim_lane(
    engine: &mut Engine,
    clip: ClipId,
    start_seconds: f64,
    length_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let start = frame_tick(start_seconds, rate);
    let len = frame_tick(length_seconds, rate).max(1);
    engine.apply(Command::Edit(EditCommand::TrimClip {
        clip,
        timeline: TimeRange::at_rate(start, len, rate),
    }))?;
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

fn move_lane(
    engine: &mut Engine,
    clip: ClipId,
    track: Option<TrackId>,
    start_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let (current_track, snapshot) = clip_snapshot(engine.project(), clip)?;
    let start = frame_tick(start_seconds, rate);
    let range = TimeRange::at_rate(start, snapshot.timeline.duration.value.max(1), rate);

    let target = match track {
        Some(target) => target,
        None => {
            let kind = engine
                .project()
                .timeline()
                .track(current_track)
                .ok_or(ModelError::UnknownTrack(current_track))?
                .kind;
            ensure_host_lane(engine, kind, range, Some(clip))?
        }
    };
    // Lifting a clip off the magnetic main track closes the hole it leaves
    // (later clips slide left), like every ripple edit there.
    let from_main = Some(current_track) == crate::ui_state::main_track(engine.project())
        && current_track != target;
    engine.apply(Command::Edit(EditCommand::MoveClip {
        clip,
        to_track: target,
        start: RationalTime::new(start, rate),
    }))?;
    if from_main {
        let s = snapshot.timeline.start.value;
        let d = snapshot.timeline.duration.value;
        engine.apply(Command::Edit(EditCommand::ShiftClips {
            track: current_track,
            from: RationalTime::new(s + d, rate),
            delta: RationalTime::new(-d, rate),
        }))?;
    }
    Ok(serde_json::json!({ "clip": clip.raw(), "track": target.raw() }))
}

fn insert_into_main(
    engine: &mut Engine,
    clip: ClipId,
    index: usize,
) -> Result<serde_json::Value, EngineError> {
    let main = ensure_main(engine)?;
    let rate = engine.project().timeline().frame_rate;
    let (current_track, snapshot) = clip_snapshot(engine.project(), clip)?;
    let len = snapshot.timeline.duration.value;

    let track = engine
        .project()
        .timeline()
        .track(main)
        .ok_or(ModelError::UnknownTrack(main))?;
    let on_main = current_track == main;

    // Boundaries of the (post-removal, for reorders) slot layout.
    let others: Vec<i64> = track
        .clips_ordered()
        .iter()
        .filter(|c| c.id != clip)
        .map(|c| c.timeline.duration.value)
        .collect();
    let index = index.min(others.len());
    let target: i64 = others.iter().take(index).sum();

    let move_to = |start: i64| EditCommand::MoveClip {
        clip,
        to_track: main,
        start: RationalTime::new(start, rate),
    };
    let shift = |from: i64, delta: i64| EditCommand::ShiftClips {
        track: main,
        from: RationalTime::new(from, rate),
        delta: RationalTime::new(delta, rate),
    };

    if on_main {
        let s = snapshot.timeline.start.value;
        if s == target {
            return Ok(serde_json::json!({ "clip": clip.raw() }));
        }
        // Park at the end, close the old gap, open the new slot, land.
        let park = track.content_end();
        engine.apply(Command::Edit(move_to(park)))?;
        engine.apply(Command::Edit(shift(s, -len)))?;
        engine.apply(Command::Edit(shift(target, len)))?;
        engine.apply(Command::Edit(move_to(target)))?;
    } else {
        engine.apply(Command::Edit(shift(target, len)))?;
        engine.apply(Command::Edit(move_to(target)))?;
    }
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

// --- clip content intents -------------------------------------------------------

fn duplicate(engine: &mut Engine, clip: ClipId) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let main = crate::ui_state::main_track(engine.project());
    let (track_id, snapshot) = clip_snapshot(engine.project(), clip)?;
    let track_kind = engine
        .project()
        .timeline()
        .track(track_id)
        .ok_or(ModelError::UnknownTrack(track_id))?
        .kind;

    let s = snapshot.timeline.start.value;
    let d = snapshot.timeline.duration.value;
    let copy_start = s + d;

    let new = match &snapshot.content {
        ClipSource::Generated(generator) => {
            let range = TimeRange::at_rate(copy_start, d, rate);
            let lane = if !engine
                .project()
                .timeline()
                .track(track_id)
                .ok_or(ModelError::UnknownTrack(track_id))?
                .has_overlap(range, None)?
            {
                track_id
            } else {
                ensure_host_lane(engine, track_kind, range, None)?
            };
            apply_created(
                engine,
                EditCommand::AddGenerated {
                    track: lane,
                    generator: generator.clone(),
                    timeline: range,
                },
            )?
        }
        ClipSource::Media { media, source } => {
            // The copy is placed at source-derived length first, then retimed
            // back to `d`; reserve the larger of the two extents.
            let placed_len = cutlass_models::resample(source.duration, rate).value.max(1);
            let reserve = placed_len.max(d);

            let new = if Some(track_id) == main {
                engine.apply(Command::Edit(EditCommand::ShiftClips {
                    track: track_id,
                    from: RationalTime::new(copy_start, rate),
                    delta: RationalTime::new(reserve, rate),
                }))?;
                apply_created(
                    engine,
                    EditCommand::AddClip {
                        track: track_id,
                        media: *media,
                        source: *source,
                        start: RationalTime::new(copy_start, rate),
                    },
                )?
            } else {
                let range = TimeRange::at_rate(copy_start, reserve, rate);
                let lane = if !engine
                    .project()
                    .timeline()
                    .track(track_id)
                    .ok_or(ModelError::UnknownTrack(track_id))?
                    .has_overlap(range, None)?
                {
                    track_id
                } else {
                    ensure_host_lane(engine, track_kind, range, None)?
                };
                apply_created(
                    engine,
                    EditCommand::AddClip {
                        track: lane,
                        media: *media,
                        source: *source,
                        start: RationalTime::new(copy_start, rate),
                    },
                )?
            };

            // Re-derive the copy's duration (speed / reverse), then trim any
            // over-reservation back out of the main track.
            if snapshot.speed != Rational::new(1, 1) || snapshot.reversed {
                engine.apply(Command::Edit(EditCommand::SetClipSpeed {
                    clip: new,
                    speed: snapshot.speed,
                    reversed: snapshot.reversed,
                }))?;
            }
            if Some(track_id) == main && reserve > d {
                engine.apply(Command::Edit(EditCommand::ShiftClips {
                    track: track_id,
                    from: RationalTime::new(copy_start + reserve, rate),
                    delta: RationalTime::new(d - reserve, rate),
                }))?;
            }

            // Carry the mix + framing over (audio applies to media clips only).
            engine.apply(Command::Edit(EditCommand::SetClipAudio {
                clip: new,
                volume: Some(snapshot.volume.sample(0)),
                fade_in: RationalTime::new(snapshot.fade_in, rate),
                fade_out: RationalTime::new(snapshot.fade_out, rate),
            }))?;
            let crop = snapshot.crop.sample(0);
            if !crop.is_full() || snapshot.flip_h || snapshot.flip_v {
                engine.apply(Command::Edit(EditCommand::SetClipCrop {
                    clip: new,
                    crop,
                    flip_h: snapshot.flip_h,
                    flip_v: snapshot.flip_v,
                    at: None,
                }))?;
            }
            new
        }
    };

    // Visual placement + effect chain (animation flattens to its start value).
    if track_kind.is_visual() {
        let pose = snapshot.transform.sample(0);
        if !pose.is_identity() {
            engine.apply(Command::Edit(EditCommand::SetClipTransform {
                clip: new,
                transform: pose,
                at: None,
            }))?;
        }
    }
    for effect in &snapshot.effects {
        engine.apply(Command::Edit(EditCommand::AddEffect {
            clip: new,
            effect_id: effect.effect_id.clone(),
        }))?;
    }
    copy_look(engine, &snapshot, new)?;

    Ok(serde_json::json!({ "clip": new.raw() }))
}

/// Carry the Phase I look fields onto a duplicated clip. Each setter runs only
/// when the source had the property, so validation never trips on defaults.
fn copy_look(engine: &mut Engine, snapshot: &Clip, new: ClipId) -> Result<(), EngineError> {
    if snapshot.mask.is_some() {
        engine.apply(Command::Edit(EditCommand::SetClipMask {
            clip: new,
            mask: snapshot.mask.clone(),
        }))?;
    }
    if snapshot.chroma_key.is_some() {
        engine.apply(Command::Edit(EditCommand::SetClipChroma {
            clip: new,
            chroma: snapshot.chroma_key.clone(),
        }))?;
    }
    if snapshot.stabilize.is_some() {
        engine.apply(Command::Edit(EditCommand::SetClipStabilize {
            clip: new,
            stabilize: snapshot.stabilize,
        }))?;
    }
    if snapshot.filter.is_some() {
        engine.apply(Command::Edit(EditCommand::SetClipFilter {
            clip: new,
            filter: snapshot.filter.clone(),
        }))?;
    }
    if !snapshot.adjust.is_neutral() {
        engine.apply(Command::Edit(EditCommand::SetClipAdjustments {
            clip: new,
            adjust: snapshot.adjust.clone(),
        }))?;
    }
    for (slot, animation) in [
        (cutlass_models::AnimationSlot::In, &snapshot.animation_in),
        (cutlass_models::AnimationSlot::Out, &snapshot.animation_out),
        (
            cutlass_models::AnimationSlot::Combo,
            &snapshot.animation_combo,
        ),
    ] {
        if animation.is_some() {
            engine.apply(Command::Edit(EditCommand::SetClipAnimation {
                clip: new,
                slot,
                animation: animation.clone(),
            }))?;
        }
    }
    if snapshot.audio_role.is_some() {
        engine.apply(Command::Edit(EditCommand::SetAudioRole {
            clip: new,
            role: snapshot.audio_role,
        }))?;
    }
    Ok(())
}

fn replace_media(
    engine: &mut Engine,
    clip: ClipId,
    path: &std::path::Path,
) -> Result<serde_json::Value, EngineError> {
    let (_, snapshot) = clip_snapshot(engine.project(), clip)?;
    let ClipSource::Media { .. } = snapshot.content else {
        return Err(invalid("replace targets a media clip"));
    };

    let media_id = apply_imported(engine, path)?;
    let media = engine
        .project()
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;

    // Source ticks the clip consumes at its current speed, at the new media's
    // native rate.
    let needed = snapshot
        .scale_by_speed(
            cutlass_models::resample(snapshot.timeline.duration, media.frame_rate).value,
        )
        .max(1);
    if !media.is_image && needed > media.duration.value {
        return Err(EngineError::Import(format!(
            "replacement media is shorter than the clip needs ({} < {} ticks)",
            media.duration.value, needed
        )));
    }
    engine.apply(Command::Edit(EditCommand::SetClipMedia {
        clip,
        media: media_id,
        source: TimeRange::at_rate(0, needed, media.frame_rate),
    }))?;
    Ok(serde_json::json!({ "clip": clip.raw(), "media": media_id.raw() }))
}

fn extract_audio(engine: &mut Engine, clip: ClipId) -> Result<serde_json::Value, EngineError> {
    let audio_clip = match engine.apply(Command::Edit(EditCommand::ExtractAudio {
        clip,
        to_track: None,
    }))? {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => {
            return Err(invalid(format!(
                "expected an extracted audio clip, got {other:?}"
            )));
        }
    };
    let lane = engine
        .project()
        .timeline()
        .track_of(audio_clip)
        .ok_or(ModelError::UnknownClip(audio_clip))?;
    Ok(serde_json::json!({ "clip": audio_clip.raw(), "track": lane.raw() }))
}

fn freeze(
    engine: &mut Engine,
    clip: ClipId,
    seconds: f64,
    png_path: &std::path::Path,
    duration_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let (track, snapshot) = require_main_clip(engine.project(), clip)?;
    let ClipSource::Media { media, .. } = snapshot.content else {
        return Err(invalid("freeze frame targets a media clip"));
    };
    let entry = engine
        .project()
        .media(media)
        .ok_or(ModelError::UnknownMedia(media))?
        .clone();
    if entry.is_image {
        return Err(invalid("clip is already a still"));
    }

    // Which source frame sits under the playhead (speed/reverse honoured)?
    // Edge hits clamp to the nearest interior frame so a freeze at a cut
    // boundary still has content to freeze.
    let s = snapshot.timeline.start.value;
    let d = snapshot.timeline.duration.value;
    let local = (frame_tick(seconds, rate) - s).clamp(0, d);
    let probe_tick = s + local.min(d - 1);
    let source_time = snapshot
        .source_time_at(RationalTime::new(probe_tick, rate))?
        .ok_or_else(|| invalid("playhead is outside the clip"))?;

    // Extract the frame at native size before touching the timeline, so a
    // decode failure leaves no half-applied edit behind.
    let frame = render_media_frame(&entry, source_time)?;
    write_freeze_png(png_path, &frame)?;
    let still = apply_imported(engine, png_path)?;
    let still_window = engine
        .project()
        .media(still)
        .ok_or(ModelError::UnknownMedia(still))?
        .duration
        .value;

    // Mid-clip freezes split first; edge hits insert flush at the boundary.
    let mut split = None;
    let insert_at = if local == 0 {
        s
    } else if local >= d {
        s + d
    } else {
        split = Some(apply_created(
            engine,
            EditCommand::SplitClip {
                clip,
                at: RationalTime::new(s + local, rate),
            },
        )?);
        s + local
    };

    let len = frame_tick(duration_seconds.max(0.1), cutlass_models::STILL_TICK_RATE)
        .clamp(1, still_window);
    let freeze_clip = apply_created(
        engine,
        EditCommand::RippleInsert {
            track,
            media: still,
            source: TimeRange::at_rate(0, len, cutlass_models::STILL_TICK_RATE),
            at: RationalTime::new(insert_at, rate),
        },
    )?;

    // The still inherits the source clip's look at the frozen instant:
    // framing flattens to a constant, the effect chain carries by id.
    let pose = snapshot.transform.sample(local);
    if !pose.is_identity() {
        engine.apply(Command::Edit(EditCommand::SetClipTransform {
            clip: freeze_clip,
            transform: pose,
            at: None,
        }))?;
    }
    for effect in &snapshot.effects {
        engine.apply(Command::Edit(EditCommand::AddEffect {
            clip: freeze_clip,
            effect_id: effect.effect_id.clone(),
        }))?;
    }

    Ok(serde_json::json!({
        "clip": freeze_clip.raw(),
        "media": still.raw(),
        "split": split.map(|c| c.raw()),
    }))
}

/// Render one frame of `media` at native size through the normal
/// resolve/render path (a scratch single-clip project, the thumbnailer
/// trick), so orientation and aspect handling can never drift from preview.
fn render_media_frame(
    media: &cutlass_models::MediaSource,
    source_time: RationalTime,
) -> Result<cutlass_core::RgbaImage, EngineError> {
    let rate = media.frame_rate;
    let frames = media.duration.value.max(1);
    let mut scratch = Project::new("freeze", rate);
    let id = scratch.add_media(media.clone());
    let track = scratch.add_track(TrackKind::Video, "Media");
    scratch.add_clip(
        track,
        id,
        TimeRange::at_rate(0, frames, rate),
        RationalTime::new(0, rate),
    )?;
    let mut renderer = cutlass_render::Renderer::new_headless()?;
    let tick = source_time.value.clamp(0, frames - 1);
    Ok(renderer.render_frame(&scratch, RationalTime::new(tick, rate))?)
}

/// Write a tightly-packed RGBA8 frame as an 8-bit PNG.
fn write_freeze_png(
    path: &std::path::Path,
    image: &cutlass_core::RgbaImage,
) -> Result<(), EngineError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), image.width, image.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| EngineError::Import(format!("freeze png: {e}")))?;
    writer
        .write_image_data(&image.pixels)
        .map_err(|e| EngineError::Import(format!("freeze png: {e}")))?;
    Ok(())
}

// --- property intents -----------------------------------------------------------

fn set_speed(
    engine: &mut Engine,
    clip: ClipId,
    speed: f64,
    reversed: bool,
) -> Result<serde_json::Value, EngineError> {
    engine.apply(Command::Edit(EditCommand::SetClipSpeed {
        clip,
        speed: speed_to_rational(speed),
        reversed,
    }))?;
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

fn set_speed_preset(
    engine: &mut Engine,
    clip: ClipId,
    preset: Option<String>,
) -> Result<serde_json::Value, EngineError> {
    let curve = match preset.as_deref() {
        Some(id) => Some(
            cutlass_models::speed_preset(id)
                .ok_or_else(|| invalid(format!("unknown speed preset '{id}'")))?,
        ),
        None => None,
    };
    engine.apply(Command::Edit(EditCommand::SetSpeedCurve { clip, curve }))?;
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

fn set_audio(
    engine: &mut Engine,
    clip: ClipId,
    volume: Option<f32>,
    fade_in_seconds: f64,
    fade_out_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    engine.apply(Command::Edit(EditCommand::SetClipAudio {
        clip,
        volume,
        fade_in: at(fade_in_seconds, rate),
        fade_out: at(fade_out_seconds, rate),
    }))?;
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

fn set_transform(
    engine: &mut Engine,
    clip: ClipId,
    pos: [f32; 2],
    scale: f32,
    rotation_degrees: f32,
    opacity: f32,
    at_seconds: Option<f64>,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    engine.apply(Command::Edit(EditCommand::SetClipTransform {
        clip,
        transform: ClipTransform {
            // UI position is 0..1 with 0.5 at the canvas center; the model
            // stores an offset from the center.
            position: [pos[0] - 0.5, pos[1] - 0.5],
            anchor_point: [0.5, 0.5],
            scale: Scale2::uniform(scale),
            rotation: rotation_degrees,
            opacity,
        },
        at: at_seconds.map(|s| at(s, rate)),
    }))?;
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

/// The transform properties the diamond toggle stamps.
const KEYFRAME_PARAMS: [ClipParam; 4] = [
    ClipParam::Position,
    ClipParam::Scale,
    ClipParam::Rotation,
    ClipParam::Opacity,
];

fn toggle_transform_keyframe(
    engine: &mut Engine,
    clip: ClipId,
    seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let (_, snapshot) = clip_snapshot(engine.project(), clip)?;
    let tick = frame_tick(seconds, rate);
    if !snapshot.timeline.contains(RationalTime::new(tick, rate))? {
        return Err(invalid("keyframe position is outside the clip"));
    }
    let rel = tick - snapshot.timeline.start.value;

    let has_kf = |param: &Param<f32>| param.keyframes().iter().any(|kf| kf.tick == rel);
    let has_kf2 = |param: &Param<[f32; 2]>| param.keyframes().iter().any(|kf| kf.tick == rel);
    let has_kf_scale = |param: &Param<Scale2>| param.keyframes().iter().any(|kf| kf.tick == rel);

    let existing: Vec<ClipParam> = KEYFRAME_PARAMS
        .into_iter()
        .filter(|param| match param {
            ClipParam::Position => has_kf2(&snapshot.transform.position),
            ClipParam::Scale => has_kf_scale(&snapshot.transform.scale),
            ClipParam::Rotation => has_kf(&snapshot.transform.rotation),
            ClipParam::Opacity => has_kf(&snapshot.transform.opacity),
            _ => false,
        })
        .collect();

    let position = RationalTime::new(tick, rate);
    if existing.is_empty() {
        // Stamp the current pose across all four properties.
        let pose = snapshot.transform.sample(rel);
        let scale_value = if pose.scale.is_uniform() {
            ParamValue::Scalar(pose.scale.x)
        } else {
            ParamValue::Vec2([pose.scale.x, pose.scale.y])
        };
        let stamps: [(ClipParam, ParamValue); 4] = [
            (ClipParam::Position, ParamValue::Vec2(pose.position)),
            (ClipParam::Scale, scale_value),
            (ClipParam::Rotation, ParamValue::Scalar(pose.rotation)),
            (ClipParam::Opacity, ParamValue::Scalar(pose.opacity)),
        ];
        for (param, value) in stamps {
            engine.apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip,
                param,
                at: position,
                value,
                easing: Easing::Linear,
            }))?;
        }
        Ok(serde_json::json!({ "clip": clip.raw(), "keyframed": true }))
    } else {
        for param in existing {
            engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
                clip,
                param,
                at: position,
            }))?;
        }
        Ok(serde_json::json!({ "clip": clip.raw(), "keyframed": false }))
    }
}

fn set_transition(
    engine: &mut Engine,
    clip: ClipId,
    transition_id: Option<String>,
    duration_seconds: f64,
) -> Result<serde_json::Value, EngineError> {
    let rate = engine.project().timeline().frame_rate;
    let (track_id, _) = clip_snapshot(engine.project(), clip)?;
    let existing = engine
        .project()
        .timeline()
        .track(track_id)
        .and_then(|t| t.transition_at(clip).map(|t| t.transition_id.clone()));

    match (existing, transition_id) {
        (Some(_), None) => {
            engine.apply(Command::Edit(EditCommand::RemoveTransition { clip }))?;
        }
        (None, None) => {}
        (existing, Some(id)) => {
            if existing.as_deref() != Some(id.as_str()) {
                if existing.is_some() {
                    engine.apply(Command::Edit(EditCommand::RemoveTransition { clip }))?;
                }
                engine.apply(Command::Edit(EditCommand::AddTransition {
                    clip,
                    transition_id: id,
                }))?;
            }
            if duration_seconds > 0.0 {
                engine.apply(Command::Edit(EditCommand::SetTransition {
                    clip,
                    duration: frame_tick(duration_seconds, rate).max(1),
                }))?;
            }
        }
    }
    Ok(serde_json::json!({ "clip": clip.raw() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::MediaSource;

    const FPS: Rational = Rational::FPS_30;

    #[test]
    fn frame_tick_rounds_to_nearest_frame_and_clamps() {
        assert_eq!(frame_tick(0.0, FPS), 0);
        assert_eq!(frame_tick(-2.0, FPS), 0);
        assert_eq!(frame_tick(1.0, FPS), 30);
        assert_eq!(frame_tick(0.016, FPS), 0); // 0.48 frames rounds down
        assert_eq!(frame_tick(0.017, FPS), 1); // 0.51 frames rounds up
    }

    #[test]
    fn speed_maps_to_milli_rational_and_clamps() {
        assert_eq!(speed_to_rational(2.0), Rational::new(2000, 1000));
        assert_eq!(speed_to_rational(0.3333), Rational::new(333, 1000));
        assert_eq!(speed_to_rational(0.0), Rational::new(50, 1000));
        assert_eq!(speed_to_rational(500.0), Rational::new(100_000, 1000));
    }

    #[test]
    fn extract_audio_delegates_once_and_rejects_a_second_request() {
        let mut project = Project::new("extract", FPS);
        let media = project.add_media(MediaSource::new(
            "/tmp/extract-mobile.mp4",
            1920,
            1080,
            FPS,
            300,
            true,
        ));
        let main = project.add_track(TrackKind::Video, "Main");
        let video = project
            .add_clip(
                main,
                media,
                TimeRange::at_rate(30, 90, FPS),
                RationalTime::new(60, FPS),
            )
            .unwrap();
        let mut engine =
            Engine::with_project(cutlass_engine::EngineConfig::default(), project).expect("engine");

        let first = run(&mut engine, Intent::ExtractAudio { clip: video }).unwrap();
        assert!(first["clip"].is_u64());
        assert!(first["track"].is_u64());
        assert_eq!(engine.project().timeline().clip_count(), 2);
        assert!(!engine.project().timeline().carries_own_audio(video));

        let before = serde_json::to_vec(engine.project()).unwrap();
        let error = run(&mut engine, Intent::ExtractAudio { clip: video })
            .expect_err("already-extracted audio must be strict");
        assert!(error.to_string().contains("already has extracted audio"));
        assert_eq!(serde_json::to_vec(engine.project()).unwrap(), before);

        assert!(engine.undo(), "the first extraction is one history entry");
        assert_eq!(engine.project().timeline().clip_count(), 1);
        assert!(engine.project().timeline().carries_own_audio(video));
        assert!(!engine.undo(), "the rejected retry records no history");
    }

    fn project_with_main() -> (Project, TrackId) {
        let mut project = Project::new("t", FPS);
        let main = project.add_track(TrackKind::Video, "Main");
        (project, main)
    }

    #[test]
    fn nearest_boundary_snaps_between_clips() {
        let (mut project, main) = project_with_main();
        let media = project.add_media(MediaSource::new("/tmp/a.mp4", 640, 360, FPS, 300, false));
        project
            .add_clip(
                main,
                media,
                TimeRange::at_rate(0, 60, FPS),
                RationalTime::new(0, FPS),
            )
            .unwrap();
        project
            .add_clip(
                main,
                media,
                TimeRange::at_rate(0, 60, FPS),
                RationalTime::new(60, FPS),
            )
            .unwrap();

        assert_eq!(nearest_main_boundary(&project, main, 0), 0);
        assert_eq!(nearest_main_boundary(&project, main, 25), 0);
        assert_eq!(nearest_main_boundary(&project, main, 45), 60);
        assert_eq!(nearest_main_boundary(&project, main, 100), 120);
        assert_eq!(nearest_main_boundary(&project, main, 10_000), 120);
    }

    #[test]
    fn host_lane_prefers_free_existing_lane_of_kind() {
        let (mut project, _main) = project_with_main();
        let text = project.add_track(TrackKind::Text, "T1");
        project
            .add_generated(
                text,
                Generator::text("busy"),
                TimeRange::at_rate(0, 90, FPS),
            )
            .unwrap();

        // Overlapping span → a new lane below the overlay block.
        let plan = host_lane_plan(
            &project,
            TrackKind::Text,
            TimeRange::at_rate(30, 30, FPS),
            None,
        )
        .unwrap();
        let text_pos = project.timeline().order().iter().position(|t| *t == text);
        assert_eq!(plan, HostLanePlan::Create { index: text_pos });

        // A free span reuses the lane.
        let plan = host_lane_plan(
            &project,
            TrackKind::Text,
            TimeRange::at_rate(90, 30, FPS),
            None,
        )
        .unwrap();
        assert_eq!(plan, HostLanePlan::Existing(text));
    }

    #[test]
    fn new_video_lane_goes_directly_above_main() {
        let (project, main) = project_with_main();
        let plan = host_lane_plan(
            &project,
            TrackKind::Video,
            TimeRange::at_rate(0, 30, FPS),
            None,
        )
        .unwrap();
        let main_pos = project
            .timeline()
            .order()
            .iter()
            .position(|t| *t == main)
            .unwrap();
        assert_eq!(
            plan,
            HostLanePlan::Create {
                index: Some(main_pos + 1)
            }
        );
    }

    #[test]
    fn new_audio_lane_appends_and_sinks_to_the_floor() {
        let (project, _main) = project_with_main();
        let plan = host_lane_plan(
            &project,
            TrackKind::Audio,
            TimeRange::at_rate(0, 30, FPS),
            None,
        )
        .unwrap();
        assert_eq!(plan, HostLanePlan::Create { index: None });
    }

    #[test]
    fn intent_wire_format_roundtrips() {
        let json = r#"{"intent":"add_text","text":"Hi","at_seconds":1.5}"#;
        let intent: Intent = serde_json::from_str(json).unwrap();
        match &intent {
            Intent::AddText {
                text,
                at_seconds,
                duration_seconds,
            } => {
                assert_eq!(text, "Hi");
                assert_eq!(*at_seconds, 1.5);
                assert_eq!(*duration_seconds, 3.0, "duration defaults to 3s");
            }
            other => panic!("unexpected intent {other:?}"),
        }

        let json =
            r#"{"intent":"ripple_trim_main","clip":7,"edge":"leading","delta_seconds":-0.5}"#;
        let intent: Intent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            intent,
            Intent::RippleTrimMain {
                edge: TrimEdge::Leading,
                ..
            }
        ));

        let json = r#"{"intent":"move_lane","clip":3,"start_seconds":2.0}"#;
        let intent: Intent = serde_json::from_str(json).unwrap();
        assert!(matches!(intent, Intent::MoveLane { track: None, .. }));
    }
}
