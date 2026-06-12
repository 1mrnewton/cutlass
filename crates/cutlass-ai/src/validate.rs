//! Validation + lowering: wire commands → engine commands, against a live
//! project snapshot.
//!
//! This is the guardrail chokepoint. Every rejection carries a
//! model-readable message (including the ids that *do* exist) so the agent
//! loop can feed it straight back as a tool result and let the model correct
//! course. Checks here are for good error messages and the whitelist; the
//! engine remains the authority (overlaps, source bounds, rate math) and
//! re-validates everything on apply.

use cutlass_commands::{Command, EditCommand};
use cutlass_models::{
    Clip, ClipId, ClipParam, ClipTransform, Easing, Generator, MediaId, ParamValue, Project,
    Rational, RationalTime, TimeRange, TrackId, TrackKind,
};

use crate::wire::{
    WireClipParam, WireCommand, WireEasing, WireGenerator, WireShape, WireTrackKind,
};

/// A wire command the project as it stands cannot accept. The message is
/// written for the model: it names the problem and the valid alternatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub message: String,
}

impl Rejection {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Rejection {}

/// Validate `command` against `project` and lower it to an engine command.
pub fn validate(command: &WireCommand, project: &Project) -> Result<Command, Rejection> {
    let edit = match command {
        WireCommand::AddTrack(args) => EditCommand::AddTrack {
            kind: track_kind(args.kind),
            name: args.name.clone(),
            index: args.index.map(|i| i as usize),
        },
        WireCommand::AddClip(args) => {
            let track = track_ref(project, args.track)?;
            let media = media_ref(project, args.media)?;
            require_media_lane(track.kind, args.track)?;
            let (source, start) = media_placement(
                project,
                media,
                args.source_start,
                args.source_duration,
                args.start,
                "start",
            )?;
            EditCommand::AddClip {
                track: track.id,
                media: media.id,
                source,
                start,
            }
        }
        WireCommand::AddGenerated(args) => {
            let track = track_ref(project, args.track)?;
            let generator = lower_generator(&args.generator, None);
            require_generator_lane(track, &generator)?;
            let timeline = timeline_range(project, args.start, args.duration)?;
            EditCommand::AddGenerated {
                track: track.id,
                generator,
                timeline,
            }
        }
        WireCommand::SetGenerator(args) => {
            let clip = clip_ref(project, args.clip)?;
            let Some(current) = generated_content(clip) else {
                return Err(Rejection::new(format!(
                    "clip {} is a media clip; set_generator only works on generated \
                     clips (text, solid, shape)",
                    args.clip
                )));
            };
            EditCommand::SetGenerator {
                clip: clip.id,
                generator: lower_generator(&args.generator, Some(current)),
            }
        }
        WireCommand::SetClipTransform(args) => {
            let clip = clip_ref(project, args.clip)?;
            // Omitted properties keep their current value — sampled at the
            // clip start for animated params (the agent edits whole-clip
            // placement; keyframe-level edits get their own commands).
            let current = clip.transform.sample(0);
            let transform = ClipTransform {
                position: [
                    args.position_x.map_or(current.position[0], |v| v as f32),
                    args.position_y.map_or(current.position[1], |v| v as f32),
                ],
                scale: args.scale.map_or(current.scale, |v| v as f32),
                rotation: args.rotation.map_or(current.rotation, |v| v as f32),
                opacity: args.opacity.map_or(current.opacity, |v| v as f32),
            };
            transform
                .validate()
                .map_err(|e| Rejection::new(format!("invalid transform: {e}")))?;
            EditCommand::SetClipTransform {
                clip: clip.id,
                transform,
                at: None,
            }
        }
        WireCommand::SetParamKeyframe(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = keyframe_position(project, clip, args.at)?;
            let value = param_value(args.param, args.value, args.position)?;
            EditCommand::SetParamKeyframe {
                clip: clip.id,
                param: clip_param(args.param),
                at,
                value,
                easing: easing(args.easing),
            }
        }
        WireCommand::RemoveParamKeyframe(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = keyframe_position(project, clip, args.at)?;
            EditCommand::RemoveParamKeyframe {
                clip: clip.id,
                param: clip_param(args.param),
                at,
            }
        }
        WireCommand::SetParamConstant(args) => {
            let clip = clip_ref(project, args.clip)?;
            let value = param_value(args.param, args.value, args.position)?;
            EditCommand::SetParamConstant {
                clip: clip.id,
                param: clip_param(args.param),
                value,
            }
        }
        WireCommand::SplitClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = timeline_time(project, args.at, "at")?;
            let tl = clip.timeline;
            if at.value <= tl.start.value || at.value >= tl.end_tick() {
                let rate = timeline_rate(project);
                return Err(Rejection::new(format!(
                    "split position {:.3}s is not strictly inside clip {} \
                     ({:.3}s to {:.3}s)",
                    args.at,
                    args.clip,
                    ticks_to_seconds(tl.start.value, rate),
                    ticks_to_seconds(tl.end_tick(), rate),
                )));
            }
            EditCommand::SplitClip { clip: clip.id, at }
        }
        WireCommand::TrimClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let timeline = timeline_range(project, args.start, args.duration)?;
            EditCommand::TrimClip {
                clip: clip.id,
                timeline,
            }
        }
        WireCommand::MoveClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let track = track_ref(project, args.to_track)?;
            if !track.kind.accepts_content(&clip.content) {
                return Err(Rejection::new(format!(
                    "clip {} cannot live on track {} ({} lane)",
                    args.clip,
                    args.to_track,
                    kind_name(track.kind),
                )));
            }
            let start = timeline_time(project, args.start, "start")?;
            require_non_negative(args.start, "start")?;
            EditCommand::MoveClip {
                clip: clip.id,
                to_track: track.id,
                start,
            }
        }
        WireCommand::RemoveClip(args) => EditCommand::RemoveClip {
            clip: clip_ref(project, args.clip)?.id,
        },
        WireCommand::RemoveTrack(args) => EditCommand::RemoveTrack {
            track: track_ref(project, args.track)?.id,
        },
        WireCommand::SetTrackEnabled(args) => EditCommand::SetTrackEnabled {
            track: track_ref(project, args.track)?.id,
            enabled: args.enabled,
        },
        WireCommand::SetTrackMuted(args) => EditCommand::SetTrackMuted {
            track: track_ref(project, args.track)?.id,
            muted: args.muted,
        },
        WireCommand::SetTrackLocked(args) => EditCommand::SetTrackLocked {
            track: track_ref(project, args.track)?.id,
            locked: args.locked,
        },
        WireCommand::RippleDelete(args) => EditCommand::RippleDelete {
            clip: clip_ref(project, args.clip)?.id,
        },
        WireCommand::ShiftClips(args) => {
            let track = track_ref(project, args.track)?;
            require_non_negative(args.from, "from")?;
            let from = timeline_time(project, args.from, "from")?;
            let delta = timeline_time_signed(project, args.delta, "delta")?;
            if delta.value == 0 {
                return Err(Rejection::new(format!(
                    "delta of {:+.4}s rounds to zero frames at the timeline rate; \
                     nothing would move",
                    args.delta
                )));
            }
            EditCommand::ShiftClips {
                track: track.id,
                from,
                delta,
            }
        }
        WireCommand::RippleInsert(args) => {
            let track = track_ref(project, args.track)?;
            let media = media_ref(project, args.media)?;
            require_media_lane(track.kind, args.track)?;
            let (source, at) = media_placement(
                project,
                media,
                args.source_start,
                args.source_duration,
                args.at,
                "at",
            )?;
            EditCommand::RippleInsert {
                track: track.id,
                media: media.id,
                source,
                at,
            }
        }
        WireCommand::LinkClips(args) => {
            if args.clips.len() < 2 {
                return Err(Rejection::new(
                    "link_clips needs at least two clip ids".to_string(),
                ));
            }
            let mut clips = Vec::with_capacity(args.clips.len());
            for &raw in &args.clips {
                clips.push(clip_ref(project, raw)?.id);
            }
            EditCommand::LinkClips { clips }
        }
    };
    Ok(Command::Edit(edit))
}

// --- entity lookups (errors list what exists) ------------------------------

const MAX_LISTED_IDS: usize = 32;

fn list_ids(mut ids: Vec<u64>) -> String {
    if ids.is_empty() {
        return "none".to_string();
    }
    ids.sort_unstable();
    let extra = ids.len().saturating_sub(MAX_LISTED_IDS);
    let mut out = ids
        .into_iter()
        .take(MAX_LISTED_IDS)
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if extra > 0 {
        out.push_str(&format!(" (and {extra} more)"));
    }
    out
}

fn clip_ref(project: &Project, raw: u64) -> Result<&Clip, Rejection> {
    project.clip(ClipId::from_raw(raw)).ok_or_else(|| {
        let existing = project
            .timeline()
            .tracks_ordered()
            .flat_map(|t| t.clips())
            .map(|c| c.id.raw())
            .collect();
        Rejection::new(format!(
            "clip {raw} does not exist; clips on the timeline: {}",
            list_ids(existing)
        ))
    })
}

fn track_ref(project: &Project, raw: u64) -> Result<&cutlass_models::Track, Rejection> {
    project.timeline().track(TrackId::from_raw(raw)).ok_or_else(|| {
        let existing = project
            .timeline()
            .tracks_ordered()
            .map(|t| t.id.raw())
            .collect();
        Rejection::new(format!(
            "track {raw} does not exist; tracks on the timeline: {}",
            list_ids(existing)
        ))
    })
}

fn media_ref(project: &Project, raw: u64) -> Result<&cutlass_models::MediaSource, Rejection> {
    project.media(MediaId::from_raw(raw)).ok_or_else(|| {
        let existing = project.media_iter().map(|m| m.id.raw()).collect();
        Rejection::new(format!(
            "media {raw} does not exist; media in the pool: {}",
            list_ids(existing)
        ))
    })
}

// --- kind rules -------------------------------------------------------------

fn track_kind(kind: WireTrackKind) -> TrackKind {
    match kind {
        WireTrackKind::Video => TrackKind::Video,
        WireTrackKind::Audio => TrackKind::Audio,
        WireTrackKind::Text => TrackKind::Text,
        WireTrackKind::Sticker => TrackKind::Sticker,
    }
}

fn kind_name(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
        TrackKind::Text => "text",
        TrackKind::Sticker => "sticker",
        TrackKind::Effect => "effect",
        TrackKind::Filter => "filter",
        TrackKind::Adjustment => "adjustment",
    }
}

fn require_media_lane(kind: TrackKind, raw: u64) -> Result<(), Rejection> {
    if matches!(kind, TrackKind::Video | TrackKind::Audio) {
        Ok(())
    } else {
        Err(Rejection::new(format!(
            "track {raw} is a {} lane; media clips need a video or audio track",
            kind_name(kind)
        )))
    }
}

fn require_generator_lane(
    track: &cutlass_models::Track,
    generator: &Generator,
) -> Result<(), Rejection> {
    let content = cutlass_models::ClipSource::Generated(generator.clone());
    if track.kind.accepts_content(&content) {
        return Ok(());
    }
    let needed = match generator {
        Generator::Text { .. } => "a text track",
        _ => "a sticker (overlay) track",
    };
    Err(Rejection::new(format!(
        "track {} is a {} lane; this generator needs {needed}",
        track.id.raw(),
        kind_name(track.kind),
    )))
}

/// Lower a wire generator. When replacing the content of an existing text
/// clip, the current style is preserved (the agent edits words, not looks).
fn lower_generator(wire: &WireGenerator, current: Option<&Generator>) -> Generator {
    match wire {
        WireGenerator::Text { content } => {
            let style = match current {
                Some(Generator::Text { style, .. }) => style.clone(),
                _ => Default::default(),
            };
            Generator::Text {
                content: content.clone(),
                style,
            }
        }
        WireGenerator::Solid { rgba } => Generator::SolidColor { rgba: *rgba },
        WireGenerator::Shape { shape, rgba } => Generator::Shape {
            shape: match shape {
                WireShape::Rectangle => cutlass_models::Shape::Rectangle,
                WireShape::Ellipse => cutlass_models::Shape::Ellipse,
            },
            rgba: *rgba,
        },
    }
}

fn generated_content(clip: &Clip) -> Option<&Generator> {
    match &clip.content {
        cutlass_models::ClipSource::Generated(g) => Some(g),
        cutlass_models::ClipSource::Media { .. } => None,
    }
}

// --- seconds → ticks ---------------------------------------------------------

fn timeline_rate(project: &Project) -> Rational {
    project.timeline().frame_rate
}

fn clip_param(param: WireClipParam) -> ClipParam {
    match param {
        WireClipParam::Position => ClipParam::Position,
        WireClipParam::Scale => ClipParam::Scale,
        WireClipParam::Rotation => ClipParam::Rotation,
        WireClipParam::Opacity => ClipParam::Opacity,
    }
}

fn easing(easing: Option<WireEasing>) -> Easing {
    match easing {
        None | Some(WireEasing::Linear) => Easing::Linear,
        Some(WireEasing::EaseIn) => Easing::EaseIn,
        Some(WireEasing::EaseOut) => Easing::EaseOut,
        Some(WireEasing::EaseInOut) => Easing::EaseInOut,
    }
}

/// Build the typed parameter value from the wire's `value` / `position`
/// fields, rejecting the wrong shape with a message naming the right one.
fn param_value(
    param: WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
) -> Result<ParamValue, Rejection> {
    match param {
        WireClipParam::Position => position
            .map(|p| ParamValue::Vec2([p[0] as f32, p[1] as f32]))
            .ok_or_else(|| {
                Rejection::new("param 'position' needs the 'position' argument as [x, y]")
            }),
        WireClipParam::Scale | WireClipParam::Rotation | WireClipParam::Opacity => value
            .map(|v| ParamValue::Scalar(v as f32))
            .ok_or_else(|| {
                Rejection::new(format!(
                    "param '{param:?}' needs the 'value' argument (a number)",
                )
                .to_lowercase())
            }),
    }
}

/// A keyframe's timeline position in seconds, pre-checked against the
/// clip's extent so the model gets a message naming where the clip sits.
fn keyframe_position(
    project: &Project,
    clip: &Clip,
    seconds: f64,
) -> Result<RationalTime, Rejection> {
    let at = timeline_time(project, seconds, "at")?;
    let tl = clip.timeline;
    if at.value < tl.start.value || at.value >= tl.end_tick() {
        let rate = timeline_rate(project);
        return Err(Rejection::new(format!(
            "keyframe position {seconds:.3}s is outside clip {} ({:.3}s to {:.3}s)",
            clip.id.raw(),
            ticks_to_seconds(tl.start.value, rate),
            ticks_to_seconds(tl.end_tick(), rate),
        )));
    }
    Ok(at)
}

fn ticks_to_seconds(ticks: i64, rate: Rational) -> f64 {
    ticks as f64 * rate.seconds_per_frame()
}

fn seconds_to_ticks(seconds: f64, rate: Rational, what: &str) -> Result<i64, Rejection> {
    if !seconds.is_finite() {
        return Err(Rejection::new(format!("{what} must be a finite number")));
    }
    let ticks = seconds * f64::from(rate.num) / f64::from(rate.den);
    if !(-(2f64.powi(53))..=2f64.powi(53)).contains(&ticks) {
        return Err(Rejection::new(format!("{what} of {seconds}s is out of range")));
    }
    Ok(ticks.round() as i64)
}

fn require_non_negative(seconds: f64, what: &str) -> Result<(), Rejection> {
    if seconds < 0.0 {
        return Err(Rejection::new(format!(
            "{what} must not be negative (got {seconds}s)"
        )));
    }
    Ok(())
}

/// A non-negative timeline position, frame-snapped to the project rate.
fn timeline_time(project: &Project, seconds: f64, what: &str) -> Result<RationalTime, Rejection> {
    let ticks = seconds_to_ticks(seconds, timeline_rate(project), what)?;
    Ok(RationalTime::new(ticks, timeline_rate(project)))
}

/// A signed timeline delta, frame-snapped to the project rate.
fn timeline_time_signed(
    project: &Project,
    seconds: f64,
    what: &str,
) -> Result<RationalTime, Rejection> {
    let ticks = seconds_to_ticks(seconds, timeline_rate(project), what)?;
    Ok(RationalTime::new(ticks, timeline_rate(project)))
}

/// A timeline range from `start`/`duration` seconds; duration must survive
/// frame snapping with at least one frame.
fn timeline_range(project: &Project, start: f64, duration: f64) -> Result<TimeRange, Rejection> {
    require_non_negative(start, "start")?;
    if duration <= 0.0 {
        return Err(Rejection::new(format!(
            "duration must be positive (got {duration}s)"
        )));
    }
    let rate = timeline_rate(project);
    let start_ticks = seconds_to_ticks(start, rate, "start")?;
    let duration_ticks = seconds_to_ticks(duration, rate, "duration")?.max(1);
    Ok(TimeRange::at_rate(start_ticks, duration_ticks, rate))
}

/// Source range (at the media's native rate) + timeline position for
/// placing media. Pre-checks bounds so the model gets a message naming the
/// media's actual extent.
fn media_placement(
    project: &Project,
    media: &cutlass_models::MediaSource,
    source_start: f64,
    source_duration: f64,
    timeline_seconds: f64,
    timeline_what: &str,
) -> Result<(TimeRange, RationalTime), Rejection> {
    require_non_negative(source_start, "source_start")?;
    if source_duration <= 0.0 {
        return Err(Rejection::new(format!(
            "source_duration must be positive (got {source_duration}s)"
        )));
    }
    require_non_negative(timeline_seconds, timeline_what)?;

    let rate = media.frame_rate;
    let start_ticks = seconds_to_ticks(source_start, rate, "source_start")?;
    let duration_ticks = seconds_to_ticks(source_duration, rate, "source_duration")?.max(1);
    if start_ticks + duration_ticks > media.duration.value {
        return Err(Rejection::new(format!(
            "source range {:.3}s + {:.3}s exceeds media {} which is {:.3}s long",
            source_start,
            source_duration,
            media.id.raw(),
            ticks_to_seconds(media.duration.value, rate),
        )));
    }
    let source = TimeRange::at_rate(start_ticks, duration_ticks, rate);
    let at = timeline_time(project, timeline_seconds, timeline_what)?;
    Ok((source, at))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire;
    use cutlass_models::MediaSource;

    const R24: Rational = Rational::FPS_24;

    /// 24 fps project: one video track with a 10 s media clip at 0 s, one
    /// text track with a title from 2 s to 5 s, and a 60 s media source.
    fn fixture() -> (Project, u64, u64, u64, u64, u64) {
        let mut project = Project::new("fixture", R24);
        let media = project.add_media(MediaSource::new(
            "/tmp/agent-fixture.mp4",
            1920,
            1080,
            R24,
            60 * 24,
            true,
        ));
        let video = project.add_track(TrackKind::Video, "V1");
        let text = project.add_track(TrackKind::Text, "Titles");
        let clip = project
            .add_clip(
                video,
                media,
                TimeRange::at_rate(0, 240, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        let title = project
            .add_generated(
                text,
                Generator::text("INTRO"),
                TimeRange::at_rate(48, 72, R24),
            )
            .unwrap();
        (
            project,
            media.raw(),
            video.raw(),
            text.raw(),
            clip.raw(),
            title.raw(),
        )
    }

    fn lower(project: &Project, cmd: WireCommand) -> EditCommand {
        match validate(&cmd, project).expect("command should validate") {
            Command::Edit(edit) => edit,
            other => panic!("expected edit, got {other:?}"),
        }
    }

    fn reject(project: &Project, cmd: WireCommand) -> String {
        validate(&cmd, project).expect_err("command should be rejected").message
    }

    #[test]
    fn trim_clip_converts_seconds_to_frame_ticks() {
        let (project, _, _, _, clip, _) = fixture();
        let edit = lower(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: 4.0,
                duration: 6.0,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::TrimClip {
                clip: ClipId::from_raw(clip),
                timeline: TimeRange::at_rate(96, 144, R24),
            }
        );
    }

    #[test]
    fn fractional_seconds_snap_to_nearest_frame() {
        let (project, _, _, _, clip, _) = fixture();
        // 1.02 s at 24 fps = 24.48 frames -> 24; duration 0.01 s -> 0.24
        // frames -> clamps to 1 frame.
        let edit = lower(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: 1.02,
                duration: 0.01,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::TrimClip {
                clip: ClipId::from_raw(clip),
                timeline: TimeRange::at_rate(24, 1, R24),
            }
        );
    }

    #[test]
    fn add_clip_uses_media_rate_for_source_and_timeline_rate_for_start() {
        let mut project = Project::new("mixed-rates", R24);
        let media = project.add_media(MediaSource::new(
            "/tmp/30fps.mp4",
            1920,
            1080,
            Rational::FPS_30,
            300,
            true,
        ));
        let track = project.add_track(TrackKind::Video, "V1");

        let edit = lower(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: track.raw(),
                media: media.raw(),
                source_start: 1.0,
                source_duration: 4.0,
                start: 2.0,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::AddClip {
                track,
                media,
                source: TimeRange::at_rate(30, 120, Rational::FPS_30),
                start: RationalTime::new(48, R24),
            }
        );
    }

    #[test]
    fn add_clip_rejects_out_of_bounds_source_with_media_extent() {
        let (project, media, video, _, _, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: video,
                media,
                source_start: 55.0,
                source_duration: 10.0,
                start: 0.0,
            }),
        );
        assert!(msg.contains("exceeds media"), "{msg}");
        assert!(msg.contains("60.000s"), "{msg}");
    }

    #[test]
    fn unknown_ids_list_existing_ones() {
        let (project, _, video, text, clip, title) = fixture();

        let msg = reject(
            &project,
            WireCommand::RemoveClip(wire::RemoveClip { clip: 999 }),
        );
        assert!(msg.contains("clip 999 does not exist"), "{msg}");
        assert!(msg.contains(&clip.to_string()), "{msg}");
        assert!(msg.contains(&title.to_string()), "{msg}");

        let msg = reject(
            &project,
            WireCommand::RemoveTrack(wire::RemoveTrack { track: 999 }),
        );
        assert!(msg.contains("track 999 does not exist"), "{msg}");
        assert!(msg.contains(&video.to_string()), "{msg}");
        assert!(msg.contains(&text.to_string()), "{msg}");

        let msg = reject(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: video,
                media: 999,
                source_start: 0.0,
                source_duration: 1.0,
                start: 0.0,
            }),
        );
        assert!(msg.contains("media 999 does not exist"), "{msg}");
    }

    #[test]
    fn generators_must_match_lane_kind() {
        let (project, _, video, text, _, _) = fixture();

        let msg = reject(
            &project,
            WireCommand::AddGenerated(wire::AddGenerated {
                track: video,
                generator: WireGenerator::Text {
                    content: "hi".into(),
                },
                start: 0.0,
                duration: 2.0,
            }),
        );
        assert!(msg.contains("needs a text track"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::AddGenerated(wire::AddGenerated {
                track: text,
                generator: WireGenerator::Solid {
                    rgba: [0, 0, 0, 255],
                },
                start: 0.0,
                duration: 2.0,
            }),
        );
        assert!(msg.contains("sticker (overlay) track"), "{msg}");
    }

    #[test]
    fn media_clips_cannot_land_on_generator_lanes() {
        let (project, media, _, text, _, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: text,
                media,
                source_start: 0.0,
                source_duration: 1.0,
                start: 0.0,
            }),
        );
        assert!(msg.contains("media clips need a video or audio track"), "{msg}");
    }

    #[test]
    fn set_generator_preserves_text_style_and_rejects_media_clips() {
        let (mut project, _, _, _, clip, title) = fixture();

        // Give the title a non-default style, then replace its content.
        let styled = Generator::Text {
            content: "INTRO".into(),
            style: cutlass_models::TextStyle {
                size: 120.0,
                ..Default::default()
            },
        };
        project
            .set_generator(ClipId::from_raw(title), styled.clone())
            .unwrap();

        let edit = lower(
            &project,
            WireCommand::SetGenerator(wire::SetGenerator {
                clip: title,
                generator: WireGenerator::Text {
                    content: "OUTRO".into(),
                },
            }),
        );
        match edit {
            EditCommand::SetGenerator {
                generator: Generator::Text { content, style },
                ..
            } => {
                assert_eq!(content, "OUTRO");
                assert_eq!(style.size, 120.0, "existing style must be preserved");
            }
            other => panic!("unexpected lowering: {other:?}"),
        }

        let msg = reject(
            &project,
            WireCommand::SetGenerator(wire::SetGenerator {
                clip,
                generator: WireGenerator::Text {
                    content: "nope".into(),
                },
            }),
        );
        assert!(msg.contains("is a media clip"), "{msg}");
    }

    #[test]
    fn transform_merges_with_current_values() {
        let (mut project, _, _, _, clip, _) = fixture();
        project
            .set_transform(
                ClipId::from_raw(clip),
                ClipTransform {
                    position: [0.25, 0.0],
                    scale: 0.5,
                    rotation: 10.0,
                    opacity: 0.8,
                },
                None,
            )
            .unwrap();

        let edit = lower(
            &project,
            WireCommand::SetClipTransform(wire::SetClipTransform {
                clip,
                position_x: None,
                position_y: Some(-0.1),
                scale: None,
                rotation: None,
                opacity: Some(1.0),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipTransform {
                clip: ClipId::from_raw(clip),
                transform: ClipTransform {
                    position: [0.25, -0.1],
                    scale: 0.5,
                    rotation: 10.0,
                    opacity: 1.0,
                },
                at: None,
            }
        );

        let msg = reject(
            &project,
            WireCommand::SetClipTransform(wire::SetClipTransform {
                clip,
                position_x: None,
                position_y: None,
                scale: Some(0.0),
                rotation: None,
                opacity: None,
            }),
        );
        assert!(msg.contains("invalid transform"), "{msg}");
    }

    #[test]
    fn split_outside_clip_names_its_extent() {
        let (project, _, _, _, clip, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::SplitClip(wire::SplitClip { clip, at: 10.0 }),
        );
        assert!(msg.contains("not strictly inside clip"), "{msg}");
        assert!(msg.contains("0.000s"), "{msg}");
        assert!(msg.contains("10.000s"), "{msg}");
    }

    #[test]
    fn shift_rejects_sub_frame_delta_and_link_needs_two() {
        let (project, _, video, _, clip, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::ShiftClips(wire::ShiftClips {
                track: video,
                from: 0.0,
                delta: 0.001,
            }),
        );
        assert!(msg.contains("rounds to zero frames"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::LinkClips(wire::LinkClips { clips: vec![clip] }),
        );
        assert!(msg.contains("at least two"), "{msg}");
    }

    #[test]
    fn non_finite_and_negative_times_are_rejected() {
        let (project, _, _, _, clip, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: f64::NAN,
                duration: 1.0,
            }),
        );
        assert!(msg.contains("finite"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: -1.0,
                duration: 1.0,
            }),
        );
        assert!(msg.contains("must not be negative"), "{msg}");
    }
}
