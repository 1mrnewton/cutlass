//! Caption groups and per-cue metadata, projected for the caption inspector,
//! the cue list, and the timeline's CC badge.

use cutlass_models::{
    CaptionFileFormat, CaptionGroup, CaptionHighlightMode, CaptionSource, Clip as EngineClip,
    Timeline,
};
use slint::{Color, ModelRc};

use super::helpers::{model, rgba_color, text_style_to_ui};
use crate::CaptionGroupView;

/// The caption fields of one clip: group id, 1-based line number, speaker,
/// confidence, and whether it can drive the word highlight.
///
/// Ordinary text clips project an empty group id, which is what every consumer
/// tests — the inspector dispatch, the cue list, and the timeline badge.
pub(super) struct CaptionCueProjection {
    pub group: String,
    pub index: i32,
    pub speaker: String,
    pub confidence: f32,
    pub has_words: bool,
    pub style_override: bool,
}

pub(super) fn clip_caption(clip: &EngineClip) -> CaptionCueProjection {
    let Some(cue) = clip.caption.as_ref() else {
        return CaptionCueProjection {
            group: String::new(),
            index: 0,
            speaker: String::new(),
            confidence: 0.0,
            has_words: false,
            style_override: false,
        };
    };
    CaptionCueProjection {
        group: cue.group.raw().to_string(),
        // The model numbers cues from zero; the cue list reads "1." for the
        // first line.
        index: cue.index.saturating_add(1) as i32,
        speaker: cue.speaker.clone().unwrap_or_default(),
        confidence: cue.confidence.unwrap_or(0.0),
        has_words: cue.has_word_timings(),
        style_override: cue.style_override,
    }
}

/// Every caption group, ordered by its first cue so the caption list reads in
/// timeline order rather than id order.
pub(super) fn caption_groups(timeline: &Timeline) -> ModelRc<CaptionGroupView> {
    let mut groups: Vec<(i64, CaptionGroupView)> = timeline
        .caption_groups_ordered()
        .into_iter()
        .map(|group| {
            (
                first_cue_tick(timeline, group),
                group_to_slint(timeline, group),
            )
        })
        .collect();
    groups.sort_by_key(|(tick, _)| *tick);
    model(groups.into_iter().map(|(_, view)| view).collect())
}

/// Timeline tick of the group's earliest cue (`i64::MAX` for an empty group, so
/// it sorts last rather than jumping to the front).
fn first_cue_tick(timeline: &Timeline, group: &CaptionGroup) -> i64 {
    timeline
        .caption_cues(group.id)
        .into_iter()
        .map(|clip| clip.timeline.start.value)
        .min()
        .unwrap_or(i64::MAX)
}

fn group_to_slint(timeline: &Timeline, group: &CaptionGroup) -> CaptionGroupView {
    let highlight = group.highlight.as_ref();
    CaptionGroupView {
        id: group.id.raw().to_string().into(),
        label: group.label.clone().into(),
        track_id: group.track.raw().to_string().into(),
        template: group.template.clone().unwrap_or_default().into(),
        source_label: source_label(&group.source).into(),
        cue_count: timeline.caption_cues(group.id).len() as i32,
        max_chars_per_line: i32::from(group.layout.max_chars_per_line),
        max_lines: i32::from(group.layout.max_lines),
        min_duration_ms: clamp_ms(group.layout.min_duration_ms),
        max_duration_ms: clamp_ms(group.layout.max_duration_ms),
        min_gap_ms: clamp_ms(group.layout.min_gap_ms),
        safe_area_bottom: group.layout.safe_area_bottom,
        highlight_mode: highlight.map_or(0, |h| match h.mode {
            CaptionHighlightMode::Off => 0,
            CaptionHighlightMode::Word => 1,
            CaptionHighlightMode::Line => 2,
        }),
        highlight_fill: highlight.map_or(Color::default(), |h| rgba_color(h.fill)),
        highlight_plate_enabled: highlight.is_some_and(|h| h.plate.is_some()),
        highlight_plate: highlight
            .and_then(|h| h.plate)
            .map_or(Color::default(), rgba_color),
        highlight_scale: highlight.map_or(1.0, |h| h.scale),
        style: text_style_to_ui(&group.style.text),
    }
}

/// Where the captions came from, for the group header.
fn source_label(source: &CaptionSource) -> String {
    match source {
        CaptionSource::Manual => "Manual".to_owned(),
        CaptionSource::Imported { format } => {
            let format = match format {
                CaptionFileFormat::Srt => "SRT",
                CaptionFileFormat::Vtt => "VTT",
            };
            format!("Imported ({format})")
        }
        CaptionSource::Auto { language, .. } => match language {
            Some(language) => format!("Auto ({language})"),
            None => "Auto".to_owned(),
        },
    }
}

fn clamp_ms(ms: u32) -> i32 {
    i32::try_from(ms).unwrap_or(i32::MAX)
}
