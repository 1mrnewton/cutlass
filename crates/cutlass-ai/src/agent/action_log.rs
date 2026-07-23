use cutlass_commands::EditOutcome;

use crate::wire::{self, WireCommand};

fn secs(v: f64) -> String {
    format!("{v:.2}s")
}

fn rgba(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3])
}

/// Human percent plus the raw wire multiplier the model must send next turn.
/// e.g. `150% (=1.5)` — percent alone was echoed as `value: 150` and rejected.
fn percent_with_raw(v: f64) -> String {
    format!("{:.0}% (={v})", v * 100.0)
}

fn param_name(param: &wire::WireClipParam) -> String {
    match param {
        wire::WireClipParam::Position => "position".into(),
        wire::WireClipParam::AnchorPoint => "anchor point".into(),
        wire::WireClipParam::Scale => "scale".into(),
        wire::WireClipParam::Rotation => "rotation".into(),
        wire::WireClipParam::Opacity => "opacity".into(),
        wire::WireClipParam::Crop => "crop".into(),
        wire::WireClipParam::Volume => "volume".into(),
        wire::WireClipParam::Pan => "pan".into(),
        wire::WireClipParam::Speed => "speed".into(),
        wire::WireClipParam::Effect { index, param } => format!("effect {index} {param}"),
        wire::WireClipParam::Shape { param } => format!("shape {param:?}").to_lowercase(),
        wire::WireClipParam::Text { param } => format!("text {param:?}").to_lowercase(),
        wire::WireClipParam::Look { param } => format!("look {param:?}").to_lowercase(),
        wire::WireClipParam::Style { param } => format!("style {param:?}").to_lowercase(),
    }
}

/// The keyframed value in editor language: "scale 150% (=1.5)", "rotation 90°",
/// "[0.25, -0.10]". Falls back to "?" when the call omitted the value (the
/// validation rejection carries the real message).
fn param_value_phrase(
    param: &wire::WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
    color: Option<[u8; 4]>,
    rect: Option<[f64; 4]>,
) -> String {
    match param {
        wire::WireClipParam::Position | wire::WireClipParam::AnchorPoint => position
            .map(|p| format!("[{:.2}, {:.2}]", p[0], p[1]))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Crop => rect
            .map(|r| format!("[{:.2}, {:.2}, {:.2}, {:.2}]", r[0], r[1], r[2], r[3]))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Scale => {
            if let Some(p) = position {
                format!("[{}, {}]", p[0], p[1])
            } else if let Some(v) = value {
                percent_with_raw(v)
            } else {
                "?".into()
            }
        }
        wire::WireClipParam::Opacity | wire::WireClipParam::Volume => {
            value.map(percent_with_raw).unwrap_or_else(|| "?".into())
        }
        wire::WireClipParam::Pan => value
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Rotation => value
            .map(|v| format!("{v:.0}°"))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Shape {
            param: wire::WireShapeParam::Fill | wire::WireShapeParam::StrokeColor,
        }
        | wire::WireClipParam::Text {
            param:
                wire::WireTextParam::Fill
                | wire::WireTextParam::StrokeColor
                | wire::WireTextParam::ShadowColor
                | wire::WireTextParam::BackgroundColor,
        }
        | wire::WireClipParam::Style {
            param:
                wire::WireStyleParam::ShadowColor
                | wire::WireStyleParam::GlowColor
                | wire::WireStyleParam::OutlineColor
                | wire::WireStyleParam::BackgroundColor,
        } => color.map(rgba).unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Style {
            param: wire::WireStyleParam::ShadowOffset,
        }
        | wire::WireClipParam::Look {
            param: wire::WireLookParam::MaskCenter | wire::WireLookParam::MaskSize,
        } => position
            .map(|p| format!("[{:.2}, {:.2}]", p[0], p[1]))
            .unwrap_or_else(|| "?".into()),
        _ => value.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
    }
}

fn generator_phrase(generator: &wire::WireGenerator) -> String {
    match generator {
        wire::WireGenerator::Text { content } => format!("text '{content}'"),
        wire::WireGenerator::Solid { rgba: c } => format!("solid {}", rgba(*c)),
        wire::WireGenerator::Shape {
            shape,
            rgba: c,
            width,
            height,
        } => {
            let name = match shape {
                wire::WireShape::Rectangle => "rectangle",
                wire::WireShape::Ellipse => "ellipse",
            };
            let size = match (width, height) {
                (Some(w), Some(h)) => format!(" {w:.0}×{h:.0} ref px"),
                _ => String::new(),
            };
            format!("{} {}{}", rgba(*c), name, size)
        }
    }
}

/// One transcript line per command: what happened, in editor language.
/// `outcome` is `None` for dry-run (planned, not applied).
pub fn describe_action(command: &WireCommand, outcome: Option<&EditOutcome>) -> String {
    let mut line = match command {
        WireCommand::AddTrack(a) => format!("added {:?} track '{}'", a.kind, a.name).to_lowercase(),
        WireCommand::AddClip(a) => format!(
            "placed media {} ({}–{} of source) at {} on track {}",
            a.media,
            secs(a.source_start),
            secs(a.source_start + a.source_duration),
            secs(a.start),
            a.track,
        ),
        WireCommand::ExtractAudio(a) => {
            format!(
                "extracted audio from clip {} onto track {}",
                a.clip, a.track
            )
        }
        WireCommand::DuplicateClip(a) => format!(
            "duplicated clip {} onto track {} at {}",
            a.clip,
            a.to_track,
            secs(a.start),
        ),
        WireCommand::AddGenerated(a) => format!(
            "added {} at {} for {} on track {}",
            generator_phrase(&a.generator),
            secs(a.start),
            secs(a.duration),
            a.track,
        ),
        WireCommand::SetGenerator(a) => format!(
            "changed clip {} to {}",
            a.clip,
            generator_phrase(&a.generator)
        ),
        WireCommand::SetClipTransform(a) => {
            let mut parts = Vec::new();
            if a.position_x.is_some() || a.position_y.is_some() {
                parts.push("position".to_string());
            }
            if let Some(s) = a.scale {
                parts.push(match s {
                    wire::WireScale::Uniform(u) => format!("scale {}", percent_with_raw(u)),
                    wire::WireScale::Axes([x, y]) => format!("scale: [{x}, {y}]"),
                });
            }
            if let Some(r) = a.rotation {
                parts.push(format!("rotation {r:.0}°"));
            }
            if let Some(o) = a.opacity {
                parts.push(format!("opacity {}", percent_with_raw(o)));
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetClipCrop(a) => {
            let mut parts = Vec::new();
            let edges: Vec<String> = [
                ("left", a.left),
                ("top", a.top),
                ("right", a.right),
                ("bottom", a.bottom),
            ]
            .iter()
            .filter_map(|(name, v)| v.map(|v| format!("{name} {}", percent_with_raw(v))))
            .collect();
            if !edges.is_empty() {
                parts.push(format!("cropped {}", edges.join(", ")));
            }
            if let Some(h) = a.flip_h {
                parts.push(
                    if h {
                        "flipped horizontally"
                    } else {
                        "unflipped horizontally"
                    }
                    .into(),
                );
            }
            if let Some(v) = a.flip_v {
                parts.push(
                    if v {
                        "flipped vertically"
                    } else {
                        "unflipped vertically"
                    }
                    .into(),
                );
            }
            if parts.is_empty() {
                parts.push("framing unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetParamKeyframe(a) => format!(
            "keyframed clip {} {} = {} at {}",
            a.clip,
            param_name(&a.param),
            param_value_phrase(&a.param, a.value, a.position, a.rgba, a.rect),
            secs(a.at),
        ),
        WireCommand::RemoveParamKeyframe(a) => format!(
            "removed clip {} {} keyframe at {}",
            a.clip,
            param_name(&a.param),
            secs(a.at),
        ),
        WireCommand::SetParamConstant(a) => format!(
            "set clip {} {} to {} (animation cleared)",
            a.clip,
            param_name(&a.param),
            param_value_phrase(&a.param, a.value, a.position, a.rgba, a.rect),
        ),
        WireCommand::ApplyEasingPreset(a) => format!(
            "applied {} easing preset on clip {} {} from {}",
            match a.preset {
                crate::wire::WireEasingPreset::BounceOut => "bounce_out",
                crate::wire::WireEasingPreset::ElasticOut => "elastic_out",
                crate::wire::WireEasingPreset::BackOut => "back_out",
            },
            a.clip,
            param_name(&a.param),
            secs(a.from_tick),
        ),
        WireCommand::SetClipSpeed(a) => {
            let mut parts = Vec::new();
            if let Some(s) = a.speed {
                parts.push(format!("speed {s}x"));
            }
            if let Some(r) = a.reversed {
                parts.push(if r {
                    "reversed".into()
                } else {
                    "forward".to_string()
                });
            }
            if parts.is_empty() {
                parts.push("retiming unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetSpeedCurve(a) => match &a.preset {
            Some(preset) => format!("applied {preset} speed ramp to clip {}", a.clip),
            None => format!("cleared speed ramp on clip {}", a.clip),
        },
        WireCommand::SetClipPitch(a) => format!(
            "set clip {} pitch to {}",
            a.clip,
            if a.preserve_pitch {
                "preserved"
            } else {
                "follow speed"
            }
        ),
        WireCommand::SetDenoise(a) => format!(
            "turned noise reduction {} on clip {}",
            if a.denoise { "on" } else { "off" },
            a.clip
        ),
        WireCommand::SetClipMask(a) => match &a.mask {
            Some(mask) => format!(
                "set {} mask on clip {}",
                match mask.kind {
                    crate::wire::WireMaskKind::Linear => "linear",
                    crate::wire::WireMaskKind::Mirror => "mirror",
                    crate::wire::WireMaskKind::Circle => "circle",
                    crate::wire::WireMaskKind::Rectangle => "rectangle",
                    crate::wire::WireMaskKind::Heart => "heart",
                    crate::wire::WireMaskKind::Star => "star",
                },
                a.clip
            ),
            None => format!("cleared mask on clip {}", a.clip),
        },
        WireCommand::SetClipChroma(a) => match &a.chroma {
            Some(_) => format!("set chroma key on clip {}", a.clip),
            None => format!("cleared chroma key on clip {}", a.clip),
        },
        WireCommand::SetClipStabilize(a) => match a.level {
            Some(level) => format!(
                "set {} stabilization on clip {}",
                match level {
                    crate::wire::WireStabilizeLevel::Recommended => "recommended",
                    crate::wire::WireStabilizeLevel::Smooth => "smooth",
                    crate::wire::WireStabilizeLevel::MaxSmooth => "max smooth",
                },
                a.clip
            ),
            None => format!("cleared stabilization on clip {}", a.clip),
        },
        WireCommand::SetClipFilter(a) => match &a.filter {
            Some(filter) => format!("set {} filter on clip {}", filter.id, a.clip),
            None => format!("cleared filter on clip {}", a.clip),
        },
        WireCommand::SetClipBlendMode(a) => format!(
            "set clip {} blend mode to {}",
            a.clip,
            match a.mode {
                crate::wire::WireBlendMode::Normal => "normal",
                crate::wire::WireBlendMode::Darken => "darken",
                crate::wire::WireBlendMode::Multiply => "multiply",
                crate::wire::WireBlendMode::ColorBurn => "color_burn",
                crate::wire::WireBlendMode::Lighten => "lighten",
                crate::wire::WireBlendMode::Screen => "screen",
                crate::wire::WireBlendMode::ColorDodge => "color_dodge",
                crate::wire::WireBlendMode::Add => "add",
                crate::wire::WireBlendMode::Overlay => "overlay",
                crate::wire::WireBlendMode::SoftLight => "soft_light",
                crate::wire::WireBlendMode::HardLight => "hard_light",
                crate::wire::WireBlendMode::Difference => "difference",
                crate::wire::WireBlendMode::Exclusion => "exclusion",
            }
        ),
        WireCommand::SetMotionBlur(a) => {
            if a.enabled {
                let mut parts = vec![format!("enable motion blur on clip {}", a.clip)];
                if let Some(s) = a.shutter_deg {
                    parts.push(format!("shutter={s}"));
                }
                if let Some(n) = a.samples {
                    parts.push(format!("samples={n}"));
                }
                parts.join(" ")
            } else {
                format!("disable motion blur on clip {}", a.clip)
            }
        }
        WireCommand::SetClipLayerStyles(a) => {
            let mut blocks = Vec::new();
            if a.styles.shadow.is_some() {
                blocks.push("shadow");
            }
            if a.styles.glow.is_some() {
                blocks.push("glow");
            }
            if a.styles.outline.is_some() {
                blocks.push("outline");
            }
            if a.styles.background.is_some() {
                blocks.push("background");
            }
            if blocks.is_empty() {
                format!("cleared layer styles on clip {}", a.clip)
            } else {
                format!(
                    "set layer styles ({}) on clip {}",
                    blocks.join(", "),
                    a.clip
                )
            }
        }
        WireCommand::SetClipAdjustments(a) => {
            let mut parts = Vec::new();
            if let Some(v) = a.brightness {
                parts.push(format!("brightness {v:.2}"));
            }
            if let Some(v) = a.contrast {
                parts.push(format!("contrast {v:.2}"));
            }
            if let Some(v) = a.saturation {
                parts.push(format!("saturation {v:.2}"));
            }
            if let Some(v) = a.exposure {
                parts.push(format!("exposure {v:.2}"));
            }
            if let Some(v) = a.temperature {
                parts.push(format!("temperature {v:.2}"));
            }
            if let Some(v) = a.tint {
                parts.push(format!("tint {v:.2}"));
            }
            if let Some(v) = a.hue {
                parts.push(format!("hue {v:.2}"));
            }
            if let Some(v) = a.highlights {
                parts.push(format!("highlights {v:.2}"));
            }
            if let Some(v) = a.shadows {
                parts.push(format!("shadows {v:.2}"));
            }
            if let Some(v) = a.sharpness {
                parts.push(format!("sharpness {v:.2}"));
            }
            if let Some(v) = a.vignette {
                parts.push(format!("vignette {v:.2}"));
            }
            if parts.is_empty() {
                parts.push("adjustments unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetClipAnimation(a) => {
            let slot = match a.slot {
                crate::wire::WireAnimationSlot::In => "in",
                crate::wire::WireAnimationSlot::Out => "out",
                crate::wire::WireAnimationSlot::Combo => "combo",
            };
            match &a.animation {
                Some(id) => format!("set {} animation on clip {} ({} slot)", id, a.clip, slot),
                None => format!("cleared {} animation on clip {}", slot, a.clip),
            }
        }
        WireCommand::SetAudioRole(a) => match a.role {
            Some(role) => format!(
                "tagged clip {} as {}",
                a.clip,
                match role {
                    crate::wire::WireAudioRole::Music => "music",
                    crate::wire::WireAudioRole::Sfx => "sfx",
                    crate::wire::WireAudioRole::Voiceover => "voiceover",
                    crate::wire::WireAudioRole::Extracted => "extracted",
                }
            ),
            None => format!("cleared audio role on clip {}", a.clip),
        },
        WireCommand::SetClipAudio(a) => {
            let mut parts = Vec::new();
            if let Some(v) = a.volume {
                parts.push(if v == 0.0 {
                    "muted".to_string()
                } else {
                    format!("volume {}", percent_with_raw(v))
                });
            }
            if let Some(f) = a.fade_in {
                parts.push(format!("fade in {}", secs(f)));
            }
            if let Some(f) = a.fade_out {
                parts.push(format!("fade out {}", secs(f)));
            }
            if parts.is_empty() {
                parts.push("audio unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::AddEffect(a) => format!("added {} effect to clip {}", a.effect, a.clip),
        WireCommand::RemoveEffect(a) => {
            format!("removed effect {} from clip {}", a.index, a.clip)
        }
        WireCommand::MoveEffect(a) => format!(
            "moved effect {} to {} on clip {}",
            a.from_index, a.to_index, a.clip
        ),
        WireCommand::SetEffectParam(a) => {
            let shown = if let Some(c) = a.rgba {
                rgba(c)
            } else if let Some(p) = a.position {
                format!("[{:.2}, {:.2}]", p[0], p[1])
            } else {
                a.value.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
            };
            format!(
                "set clip {} effect {} {} = {}",
                a.clip, a.index, a.param, shown
            )
        }
        WireCommand::AddTransition(a) => {
            format!("added {} transition after clip {}", a.transition, a.clip)
        }
        WireCommand::RemoveTransition(a) => {
            format!("removed transition after clip {}", a.clip)
        }
        WireCommand::SetTransition(a) => {
            format!("set transition after clip {} to {}s", a.clip, a.seconds)
        }
        WireCommand::SplitClip(a) => format!("split clip {} at {}", a.clip, secs(a.at)),
        WireCommand::TrimClip(a) => format!(
            "trimmed clip {} to {}–{}",
            a.clip,
            secs(a.start),
            secs(a.start + a.duration)
        ),
        WireCommand::MoveClip(a) => {
            format!(
                "moved clip {} to {} on track {}",
                a.clip,
                secs(a.start),
                a.to_track
            )
        }
        WireCommand::RemoveClip(a) => format!("removed clip {}", a.clip),
        WireCommand::RemoveTrack(a) => format!("removed track {}", a.track),
        WireCommand::SetTrackEnabled(a) => format!(
            "{} track {}",
            if a.enabled { "showed" } else { "hid" },
            a.track
        ),
        WireCommand::SetTrackMuted(a) => format!(
            "{} track {}",
            if a.muted { "muted" } else { "unmuted" },
            a.track
        ),
        WireCommand::SetTrackLocked(a) => format!(
            "{} track {}",
            if a.locked { "locked" } else { "unlocked" },
            a.track
        ),
        WireCommand::RippleDelete(a) => {
            format!(
                "ripple-deleted clip {} (later clips closed the gap)",
                a.clip
            )
        }
        WireCommand::ShiftClips(a) => format!(
            "shifted clips on track {} from {} by {:+.2}s",
            a.track,
            secs(a.from),
            a.delta
        ),
        WireCommand::RippleInsert(a) => format!(
            "ripple-inserted media {} at {} on track {} (later clips moved right)",
            a.media,
            secs(a.at),
            a.track
        ),
        WireCommand::LinkClips(a) => format!(
            "linked clips {}",
            a.clips
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        WireCommand::UnlinkClips(a) => format!(
            "unlinked complete groups touched by clips {}",
            a.clips
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        WireCommand::AddMarker(a) => {
            let name = match &a.name {
                Some(name) if !name.is_empty() => format!(" '{name}'"),
                _ => String::new(),
            };
            let color = a
                .color
                .map(|c| format!(" ({c:?})").to_lowercase())
                .unwrap_or_default();
            format!("added marker{name} at {}{color}", secs(a.at))
        }
        WireCommand::RemoveMarker(a) => format!("removed marker {}", a.marker),
        WireCommand::SetMarker(a) => {
            let mut parts = Vec::new();
            if let Some(at) = a.at {
                parts.push(format!("moved to {}", secs(at)));
            }
            if let Some(name) = &a.name {
                parts.push(format!("named '{name}'"));
            }
            if let Some(color) = a.color {
                parts.push(format!("colored {color:?}").to_lowercase());
            }
            if parts.is_empty() {
                parts.push("unchanged".into());
            }
            format!("set marker {} {}", a.marker, parts.join(", "))
        }
        WireCommand::SetCanvas(a) => {
            let mut parts = Vec::new();
            if let Some(aspect) = a.aspect {
                parts.push(format!("aspect {}", aspect.name()));
            }
            if let Some([r, g, b]) = a.background {
                parts.push(format!("background rgb({r}, {g}, {b})"));
            }
            if parts.is_empty() {
                parts.push("unchanged".into());
            }
            format!("set canvas {}", parts.join(", "))
        }
    };
    match outcome {
        Some(EditOutcome::Created(id)) => line.push_str(&format!(" (new clip {})", id.raw())),
        Some(EditOutcome::CreatedTrack(id)) => line.push_str(&format!(" (track {})", id.raw())),
        Some(EditOutcome::CreatedMarker(id)) => line.push_str(&format!(" (marker {})", id.raw())),
        _ => {}
    }
    line
}
