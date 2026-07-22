use super::*;

// --- entity lookups (errors list what exists) ------------------------------

const MAX_LISTED_IDS: usize = 32;

pub(super) fn list_ids(mut ids: Vec<u64>) -> String {
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

pub(super) fn clip_ref(project: &Project, raw: u64) -> Result<&Clip, Rejection> {
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

/// Comma-separated catalog effect ids, for rejection messages.
pub(super) fn effect_ids() -> String {
    cutlass_models::effect_catalog()
        .iter()
        .map(|s| s.id)
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn transition_ids() -> String {
    cutlass_models::transition_catalog()
        .iter()
        .map(|s| s.id)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether a clip butts directly against a following clip on its track (the
/// next clip starts exactly where this one ends) — the precondition for a
/// transition at its right junction.
pub(super) fn has_right_neighbor(project: &Project, clip: &Clip) -> bool {
    let timeline = project.timeline();
    let Some(track) = timeline.track_of(clip.id).and_then(|id| timeline.track(id)) else {
        return false;
    };
    let end = clip.timeline.end_tick();
    track
        .clips()
        .any(|c| c.id != clip.id && c.timeline.start.value == end)
}

/// Reject when the clip carries no transition at its right junction.
pub(super) fn require_transition(
    project: &Project,
    clip: &Clip,
    raw_clip: u64,
) -> Result<(), Rejection> {
    let timeline = project.timeline();
    let has = timeline
        .track_of(clip.id)
        .and_then(|id| timeline.track(id))
        .is_some_and(|t| t.transition_at(clip.id).is_some());
    if !has {
        return Err(Rejection::new(format!(
            "clip {raw_clip} has no transition at its right cut"
        )));
    }
    Ok(())
}

/// Resolve a wire effect-chain index against a clip, rejecting out-of-range
/// indices with the clip's current chain length.
pub(super) fn effect_index(clip: &Clip, index: u32, raw_clip: u64) -> Result<usize, Rejection> {
    let index = usize::try_from(index).map_err(|_| {
        Rejection::new(format!(
            "clip {raw_clip} has no effect at index {index} (chain length {})",
            clip.effects.len()
        ))
    })?;
    if index >= clip.effects.len() {
        return Err(Rejection::new(format!(
            "clip {raw_clip} has no effect at index {index} (chain length {})",
            clip.effects.len()
        )));
    }
    Ok(index)
}

/// Resolve `(effect index, param name)` to the catalog slot and its spec.
pub(super) fn effect_param_slot(
    clip: &Clip,
    index: u32,
    param: &str,
    raw_clip: u64,
) -> Result<(usize, usize, &'static cutlass_models::EffectParamSpec), Rejection> {
    let effect = effect_index(clip, index, raw_clip)?;
    let instance = &clip.effects[effect];
    let spec = cutlass_models::effect_spec(&instance.effect_id).ok_or_else(|| {
        Rejection::new(format!(
            "effect '{}' on clip {raw_clip} is not in the catalog",
            instance.effect_id
        ))
    })?;
    let slot = spec
        .params
        .iter()
        .position(|p| p.name == param)
        .ok_or_else(|| {
            let names: Vec<&str> = spec.params.iter().map(|p| p.name).collect();
            Rejection::new(format!(
                "effect '{}' has no parameter '{}'; parameters: {}",
                instance.effect_id,
                param,
                names.join(", ")
            ))
        })?;
    Ok((effect, slot, &spec.params[slot]))
}

/// Reject clips on an audio lane, which have no frame to operate on.
pub(super) fn reject_audio_lane(
    project: &Project,
    clip: &Clip,
    why: &str,
    raw_clip: u64,
) -> Result<(), Rejection> {
    let timeline = project.timeline();
    let on_audio = timeline
        .track_of(clip.id)
        .and_then(|id| timeline.track(id))
        .is_some_and(|t| t.kind == TrackKind::Audio);
    if on_audio {
        return Err(Rejection::new(format!(
            "clip {raw_clip} is on an audio lane; {why}"
        )));
    }
    Ok(())
}

/// Reject transitions on lanes whose clips don't composite as frames: audio
/// has no frame, and effect/filter/adjustment segments resolve to canvas-wide
/// passes the renderer can't nest inside a transition (mirrors the model's
/// `TrackKind::supports_transitions`).
pub(super) fn reject_non_transition_lane(
    project: &Project,
    clip: &Clip,
    raw_clip: u64,
) -> Result<(), Rejection> {
    let timeline = project.timeline();
    let kind = timeline
        .track_of(clip.id)
        .and_then(|id| timeline.track(id))
        .map(|t| t.kind);
    match kind {
        Some(kind) if !kind.supports_transitions() => Err(Rejection::new(format!(
            "clip {raw_clip} is on a {kind:?} lane; transitions need a lane that composites frames (video, sticker, or text)"
        ))),
        _ => Ok(()),
    }
}

pub(super) fn track_ref(project: &Project, raw: u64) -> Result<&cutlass_models::Track, Rejection> {
    project
        .timeline()
        .track(TrackId::from_raw(raw))
        .ok_or_else(|| {
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

pub(super) fn media_ref(
    project: &Project,
    raw: u64,
) -> Result<&cutlass_models::MediaSource, Rejection> {
    project.media(MediaId::from_raw(raw)).ok_or_else(|| {
        let existing = project.media_iter().map(|m| m.id.raw()).collect();
        Rejection::new(format!(
            "media {raw} does not exist; media in the pool: {}",
            list_ids(existing)
        ))
    })
}

pub(super) fn marker_ref(project: &Project, raw: u64) -> Result<&Marker, Rejection> {
    project
        .timeline()
        .marker(MarkerId::from_raw(raw))
        .ok_or_else(|| {
            let existing = project
                .timeline()
                .markers()
                .iter()
                .map(|m| m.id.raw())
                .collect();
            Rejection::new(format!(
                "marker {raw} does not exist; markers on the timeline: {}",
                list_ids(existing)
            ))
        })
}

pub(super) fn marker_color(color: WireMarkerColor) -> MarkerColor {
    match color {
        WireMarkerColor::Teal => MarkerColor::Teal,
        WireMarkerColor::Blue => MarkerColor::Blue,
        WireMarkerColor::Purple => MarkerColor::Purple,
        WireMarkerColor::Pink => MarkerColor::Pink,
        WireMarkerColor::Red => MarkerColor::Red,
        WireMarkerColor::Orange => MarkerColor::Orange,
        WireMarkerColor::Yellow => MarkerColor::Yellow,
        WireMarkerColor::Green => MarkerColor::Green,
    }
}

pub(super) fn canvas_aspect(aspect: WireCanvasAspect) -> CanvasAspect {
    match aspect {
        WireCanvasAspect::Auto => CanvasAspect::Auto,
        WireCanvasAspect::Wide16x9 => CanvasAspect::Wide16x9,
        WireCanvasAspect::Tall9x16 => CanvasAspect::Tall9x16,
        WireCanvasAspect::Square1x1 => CanvasAspect::Square1x1,
        WireCanvasAspect::Portrait4x5 => CanvasAspect::Portrait4x5,
        WireCanvasAspect::Cinema21x9 => CanvasAspect::Cinema21x9,
    }
}

// --- kind rules -------------------------------------------------------------

pub(super) fn track_kind(kind: WireTrackKind) -> TrackKind {
    match kind {
        WireTrackKind::Video => TrackKind::Video,
        WireTrackKind::Audio => TrackKind::Audio,
        WireTrackKind::Text => TrackKind::Text,
        WireTrackKind::Sticker => TrackKind::Sticker,
    }
}

pub(super) fn kind_name(kind: TrackKind) -> &'static str {
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

pub(super) fn require_media_lane(kind: TrackKind, raw: u64) -> Result<(), Rejection> {
    if matches!(kind, TrackKind::Video | TrackKind::Audio) {
        Ok(())
    } else {
        Err(Rejection::new(format!(
            "track {raw} is a {} lane; media clips need a video or audio track",
            kind_name(kind)
        )))
    }
}

pub(super) fn require_generator_lane(
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
