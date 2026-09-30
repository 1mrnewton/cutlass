use super::*;

/// The clip's speed as a display/scale float (1.0 for degenerate rationals,
/// which the model rejects anyway).
pub(super) fn speed_factor(speed: EngineRational) -> f32 {
    if speed.num <= 0 || speed.den <= 0 {
        return 1.0;
    }
    speed.num as f32 / speed.den as f32
}

/// Retime badge for the timeline card: `2x` / `0.5x` (trailing zeros
/// trimmed), with ` R` appended when reversed — a reversed 1× clip shows
/// just `R`. A speed ramp (M2 curve) shows its *effective* average rate with
/// a `~` prefix (`~1.4x`) so it reads as varying, not constant. Empty ⇔
/// forward 1× with no ramp (no badge).
pub(super) fn speed_label(clip: &EngineClip) -> String {
    if !clip.is_retimed() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    // A ramp's effective rate is the base speed times the curve's average;
    // the `~` marks that the instantaneous rate varies across the clip.
    let ramped = clip.has_speed_curve();
    let factor = speed_factor(clip.speed)
        * if ramped {
            clip.speed_curve_average() as f32
        } else {
            1.0
        };
    if ramped || (factor - 1.0).abs() > f32::EPSILON {
        let mut s = format!("{factor:.2}");
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        parts.push(format!("{}{s}x", if ramped { "~" } else { "" }));
    }
    if clip.reversed {
        parts.push("R".into());
    }
    parts.join(" ")
}

/// `time` as seconds, exact rational division in floating point.
pub(super) fn time_to_seconds(time: EngineTime) -> f64 {
    if time.rate.num <= 0 || time.rate.den <= 0 {
        return 0.0;
    }
    time.value as f64 * f64::from(time.rate.den) / f64::from(time.rate.num)
}

/// Clip badge: CapCut-style `3.4s` under a minute, `M:SS` (or `H:MM:SS`)
/// from there up.
pub(super) fn clip_duration_label(duration: EngineTime) -> String {
    let secs = time_to_seconds(duration).max(0.0);
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let whole = secs.round() as i64;
        let (h, m, s) = (whole / 3600, (whole / 60) % 60, whole % 60);
        if h > 0 {
            format!("{h}:{m:02}:{s:02}")
        } else {
            format!("{m}:{s:02}")
        }
    }
}

/// Trim headroom for generated clips, which have no source bounds. Big enough
/// to never clamp, small enough that `clip end + room` can't overflow `i32`.
pub(super) const UNBOUNDED_ROOM: i32 = i32::MAX / 4;

/// How far (sequence ticks) each clip edge can extend before running out of
/// source media: `(head, tail)`. Head room is the media before the in-point,
/// tail room the media after the out-point, both projected to the sequence
/// rate *conservatively* (see [`room_to_sequence_ticks`]) so the trim ghost
/// never offers an extension `Project::trim_clip` would reject.
pub(super) fn trim_rooms(project: &EngineProject, clip: &EngineClip) -> (i32, i32) {
    let tl_rate = project.timeline().frame_rate;
    match &clip.content {
        ClipSource::Media { media, source } => {
            let Some(media) = project.media(*media) else {
                return (0, 0);
            };
            // Stills extend freely: the one frame repeats, and the pool
            // duration is a default placement length, not material bounds
            // (the engine relaxes `trim_clip` the same way).
            if media.is_image {
                return (UNBOUNDED_ROOM, UNBOUNDED_ROOM);
            }
            let head_media = source.start.value;
            let tail_media = media.duration.value - source.end_tick();
            (
                room_to_sequence_ticks(head_media, media.frame_rate, tl_rate),
                room_to_sequence_ticks(tail_media, media.frame_rate, tl_rate),
            )
        }
        ClipSource::Generated(_) => (UNBOUNDED_ROOM, UNBOUNDED_ROOM),
    }
}

/// Largest number of sequence ticks an edge may extend such that the engine's
/// media-rate resample of that delta stays within `room_media` ticks.
///
/// `Project::trim_clip` re-derives the source delta by resampling the
/// timeline delta (round-to-nearest), so a naive media→sequence conversion
/// can overshoot by a tick and get the commit rejected. Convert, then verify
/// by round-tripping and step down until it fits; when the rates differ, keep
/// one extra media tick in reserve for the duration-resample's own rounding.
pub(super) fn room_to_sequence_ticks(
    room_media: i64,
    media_rate: EngineRational,
    tl_rate: EngineRational,
) -> i32 {
    let mut room = room_media.max(0);
    if !rate_eq(media_rate, tl_rate) {
        room = (room - 1).max(0);
    }
    let mut ticks = resample(EngineTime::new(room, media_rate), tl_rate).value;
    while ticks > 0 && resample(EngineTime::new(ticks, tl_rate), media_rate).value > room {
        ticks -= 1;
    }
    clamp_i32(ticks)
}

/// `(lane label, text-generator content)` for a clip.
pub(super) fn clip_labels(project: &EngineProject, clip: &EngineClip) -> (String, String) {
    match &clip.content {
        ClipSource::Media { media, .. } => {
            let name = project
                .media(*media)
                .map(media_name)
                .unwrap_or_else(|| format!("Clip {}", clip.id.raw()));
            (name, String::new())
        }
        ClipSource::Generated(generator) => match generator {
            Generator::Text { content, .. } => ("Text".to_owned(), content.clone()),
            Generator::SolidColor { .. } => ("Solid".to_owned(), String::new()),
            Generator::Shape { .. } => ("Shape".to_owned(), String::new()),
            Generator::Sticker { asset } => (
                cutlass_models::sticker_spec(asset)
                    .map_or_else(|| "Sticker".to_owned(), |s| s.label.to_owned()),
                String::new(),
            ),
            Generator::Lottie { path, .. } => (
                std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map_or_else(|| "Animation".to_owned(), str::to_owned),
                String::new(),
            ),
            Generator::Effect => ("Effect".to_owned(), String::new()),
            Generator::Filter => ("Filter".to_owned(), String::new()),
            Generator::Adjustment => ("Adjustment".to_owned(), String::new()),
        },
    }
}

pub(super) fn clip_shape_size(clip: &EngineClip) -> (f32, f32) {
    match &clip.content {
        // Geometry became animatable (`Param`) on this branch; the inspector
        // edits the clip-start sample, like every other projected property.
        ClipSource::Generated(Generator::Shape { width, height, .. }) => {
            (width.sample(0), height.sample(0))
        }
        _ => (0.0, 0.0),
    }
}

/// `(generator-kind tag, fill color)` for the timeline card. The tag selects
/// the card's preview rendering (see `panels/timeline/clip.slint`); the color
/// is the solid/shape fill (transparent for everything else).
pub(super) fn clip_generator_visual(clip: &EngineClip) -> (&'static str, Color) {
    let transparent = Color::from_argb_u8(0, 0, 0, 0);
    match &clip.content {
        ClipSource::Generated(Generator::Text { .. }) => ("text", transparent),
        ClipSource::Generated(Generator::SolidColor { rgba }) => ("solid", rgba_color(*rgba)),
        ClipSource::Generated(Generator::Shape { shape, rgba, .. }) => {
            // The shape family grew on this branch; every non-ellipse figure
            // reads fine on the card as the generic cornered swatch.
            let tag = match shape {
                cutlass_models::Shape::Ellipse => "ellipse",
                _ => "rect",
            };
            (tag, rgba_color(rgba.sample(0)))
        }
        // Only catalog-backed stickers tag as composited (drives preview
        // hit-testing); a legacy empty asset draws nothing, so no tag.
        ClipSource::Generated(Generator::Sticker { asset })
            if cutlass_models::sticker_spec(asset).is_some() =>
        {
            ("sticker", transparent)
        }
        // File-backed Lotties composite like stickers (preview hit-testing).
        ClipSource::Generated(Generator::Lottie { .. }) => ("sticker", transparent),
        // Canvas-pass lane bars (effect / filter / adjustment): not drawn as
        // layer quads — tags drive inspector gating for no-op blend/styles/
        // motion-blur controls (see resolve CanvasPass guards).
        ClipSource::Generated(Generator::Effect) => ("effect", transparent),
        ClipSource::Generated(Generator::Filter) => ("filter", transparent),
        ClipSource::Generated(Generator::Adjustment) => ("adjustment", transparent),
        _ => ("", transparent),
    }
}

pub(super) fn rgba_color(rgba: [u8; 4]) -> Color {
    Color::from_argb_u8(rgba[3], rgba[0], rgba[1], rgba[2])
}

/// Project a clip's text styling into the Slint `TextStyle`. Non-text clips
/// (and text clips written before styling existed) get the engine default
/// look, so the inspector always has a coherent style to edit.
pub(super) fn clip_text_style(clip: &EngineClip) -> TextClipStyle {
    let style = match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style.clone(),
        _ => EngineTextStyle::default(),
    };
    text_style_to_ui(&style)
}

/// Convert an engine `TextStyle` to the Slint struct. Effect opacities are
/// pulled out of their rgba alpha into a dedicated 0..=1 control, and the
/// swatch colors are made opaque so the picker preview reads cleanly.
pub(super) fn text_style_to_ui(style: &EngineTextStyle) -> TextClipStyle {
    let opaque = |rgba: [u8; 4]| Color::from_rgb_u8(rgba[0], rgba[1], rgba[2]);
    let alpha01 = |rgba: [u8; 4]| rgba[3] as f32 / 255.0;
    let stroke = style.stroke.clone().unwrap_or_default();
    let background = style.background.clone().unwrap_or_default();
    let shadow = style.shadow.clone().unwrap_or_default();
    TextClipStyle {
        font: style.font.clone().into(),
        size: style.size.sample(0),
        bold: style.bold,
        italic: style.italic,
        underline: style.underline,
        case: text_case_to_int(style.case),
        fill: {
            let fill = style.fill.sample(0);
            Color::from_argb_u8(fill[3], fill[0], fill[1], fill[2])
        },
        letter_spacing: style.letter_spacing.sample(0),
        line_spacing: style.line_spacing.sample(0),
        align_h: align_h_to_int(style.align_h),
        align_v: align_v_to_int(style.align_v),
        wrap: style.wrap,
        stroke_enabled: style.stroke.is_some(),
        stroke_color: opaque(stroke.rgba.sample(0)),
        stroke_width: stroke.width.sample(0),
        background_enabled: style.background.is_some(),
        background_color: opaque(background.rgba.sample(0)),
        background_opacity: alpha01(background.rgba.sample(0)),
        background_radius: background.radius.sample(0),
        shadow_enabled: style.shadow.is_some(),
        shadow_color: opaque(shadow.rgba.sample(0)),
        shadow_opacity: alpha01(shadow.rgba.sample(0)),
        shadow_blur: shadow.blur.sample(0),
        shadow_distance: shadow.distance.sample(0),
    }
}

pub(super) fn text_case_to_int(case: TextCase) -> i32 {
    match case {
        TextCase::Normal => 0,
        TextCase::Upper => 1,
        TextCase::Lower => 2,
        TextCase::Title => 3,
    }
}

pub(super) fn align_h_to_int(align: TextAlignH) -> i32 {
    match align {
        TextAlignH::Left => 0,
        TextAlignH::Center => 1,
        TextAlignH::Right => 2,
    }
}

pub(super) fn align_v_to_int(align: TextAlignV) -> i32 {
    match align {
        TextAlignV::Top => 0,
        TextAlignV::Middle => 1,
        TextAlignV::Bottom => 2,
    }
}

/// The engine's composite canvas size, as Slint lengths. Delegating to the
/// renderer's `canvas_size` (rather than mirroring it) keeps preview
/// hit-test geometry pixel-identical to the composited frame by
/// construction — including the M1 aspect presets.
pub(super) fn canvas_size(project: &EngineProject) -> (f32, f32) {
    let (w, h) = cutlass_render::canvas_size(project);
    (w as f32, h as f32)
}

/// `CanvasAspect` as the preset index the canvas dialog's ratio list uses.
pub(super) fn aspect_to_index(aspect: cutlass_models::CanvasAspect) -> i32 {
    cutlass_models::CanvasAspect::ALL
        .iter()
        .position(|a| *a == aspect)
        .map_or(0, |i| i as i32)
}

/// Lane kinds the UI surfaces today. Filter lanes are still phantom until
/// their engine lands (v1 roadmap M0 "hide phantom kinds", M5): the model
/// keeps them — they round-trip through save/load untouched and composite
/// nothing — but the projection skips them so users never see lanes that do
/// nothing. Adjustment lanes became real in M4; effect lanes surfaced with
/// standalone effect segments (CapCut effects-as-track-clips).
pub(super) fn kind_visible(kind: EngineKind) -> bool {
    kind != EngineKind::Filter
}

pub(super) fn track_kind(kind: EngineKind) -> TrackKind {
    match kind {
        EngineKind::Video => TrackKind::Video,
        EngineKind::Audio => TrackKind::Audio,
        EngineKind::Text => TrackKind::Text,
        EngineKind::Sticker => TrackKind::Sticker,
        EngineKind::Effect => TrackKind::Effect,
        EngineKind::Filter => TrackKind::Filter,
        EngineKind::Adjustment => TrackKind::Adjustment,
    }
}

pub(super) fn clip_capabilities(
    project: &EngineProject,
    clip: &EngineClip,
    kind: EngineKind,
) -> ClipCapabilities {
    let caps = EngineCaps::for_clip(project, clip, kind);
    ClipCapabilities {
        has_transform: caps.has_transform,
        has_crop: caps.has_crop,
        has_audio: caps.has_audio,
        has_speed: caps.has_speed,
        has_text: caps.has_text,
        has_shape: caps.has_shape,
        has_effects: caps.has_effects,
        has_filter_adjust: caps.has_filter_adjust,
        can_split: caps.can_split,
        can_reverse: caps.can_reverse,
        can_ripple_delete: caps.can_ripple_delete,
        can_extract_audio: caps.can_extract_audio,
    }
}

/// One color per lane kind (the engine has no per-track color). Matches the
/// palette the UI previously hardcoded in `editor-store.slint`.
pub(super) fn kind_color(kind: EngineKind) -> Color {
    let (r, g, b) = match kind {
        EngineKind::Video => (0x4A, 0x6F, 0xA5),
        EngineKind::Audio => (0xC9, 0x98, 0x46),
        EngineKind::Text => (0x5E, 0x8B, 0x7E),
        EngineKind::Sticker => (0xBF, 0x6F, 0x4A),
        EngineKind::Effect => (0x7B, 0x68, 0xA6),
        EngineKind::Filter => (0x4A, 0x8C, 0x8C),
        EngineKind::Adjustment => (0x6C, 0x5B, 0x7B),
    };
    Color::from_rgb_u8(r, g, b)
}

pub(super) fn marker_to_slint(marker: &EngineMarker) -> TimelineMarker {
    let [r, g, b, a] = marker.color.rgba();
    TimelineMarker {
        id: marker.id.raw().to_string().into(),
        tick: clamp_i32(marker.tick.value),
        name: marker.name.clone().into(),
        color: Color::from_argb_u8(a, r, g, b),
        color_name: marker.color.token().into(),
    }
}

pub(super) fn rational(rate: cutlass_models::Rational) -> Rational {
    Rational {
        num: rate.num,
        den: rate.den,
    }
}

pub(super) fn rational_time(time: EngineTime) -> RationalTime {
    RationalTime {
        // Slint's time model is `i32`; clamp the engine's `i64` ticks. Realistic
        // projects stay well inside `i32` (≈24 days at 1000 fps).
        value: clamp_i32(time.value),
        rate: rational(time.rate),
    }
}

pub(super) fn time_range(range: EngineRange) -> TimeRange {
    TimeRange {
        start: rational_time(range.start),
        duration: rational_time(range.duration),
    }
}

/// The single choke point projecting engine `i64` ticks into Slint's `i32`
/// time model (keyframes roadmap Phase 4 — tick audit). Every tick that
/// crosses the boundary (`rational_time`, markers, keyframe + speed-ramp
/// rows) routes through here so an out-of-range value **saturates** instead
/// of wrapping — a clip parked past the bound clamps to the edge of the
/// addressable timeline rather than teleporting to a negative tick.
///
/// ## Timeline-length bound
///
/// `i32::MAX` ticks is the hard ceiling. In wall-clock time that is
/// `i32::MAX / fps` seconds: ≈ 20.7 hours at 30 fps, ≈ 8.3 hours at 72 fps,
/// ≈ 24.8 days at 1000 fps. Real projects stay orders of magnitude inside
/// it; the clamp only exists so a pathological/corrupt tick can never alias
/// to a bogus on-screen position. Promoting the Slint model to `i64` is the
/// long-term fix (tracked in `timeline-roadmap.md`).
pub(super) fn clamp_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

pub(super) fn model<T: Clone + 'static>(items: Vec<T>) -> ModelRc<T> {
    ModelRc::from(Rc::new(VecModel::from(items)))
}
