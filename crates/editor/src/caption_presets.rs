//! Caption presets every shell offers: a caption style, the caption
//! animation that plays in the loop slot, a letter stagger, and the base
//! look of the words. A keyword is drawn in the style's spoken look.
//!
//! A preset is the app's own; the reference editor's presets are
//! downloaded packages. Picking one records it on the caption as its
//! `caption_template_info`.

use cutlass_model::{ANIMATION_PRESETS, AnimType, Draft, Micros, SegmentId};
use cutlass_session::Request;

use crate::caption_list::{caption_rows, main_caption_lane};

/// One caption preset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptionPreset {
    /// Stored as the caption's preset resource id.
    pub id: &'static str,
    pub name: &'static str,
    /// A key of [`cutlass_model::CAPTION_STYLE_KEYS`].
    pub style: &'static str,
    /// The loop preset the caption animation plays, if any.
    pub animation: Option<&'static str>,
    /// Milliseconds each letter starts after the one before it.
    pub stagger_ms: i64,
    /// The words' color.
    pub color: &'static str,
    /// The outline color and width in ems; an empty color has none.
    pub outline: (&'static str, f64),
    /// The groups the Presets page files it under, as
    /// [`PRESET_CATEGORIES`] names them.
    pub categories: &'static [&'static str],
}

const WHITE: &str = "#FFFFFF";
const INK: &str = "#111111";
const YELLOW: &str = "#FFD400";
const PINK: &str = "#FF4FA3";
const LIME: &str = "#B6FF3B";
const CYAN: &str = "#3EE6FF";
const ORANGE: &str = "#FF8A00";

/// The groups the Presets page offers, in its order.
pub const PRESET_CATEGORIES: &[&str] = &[
    "Trending",
    "Classic",
    "Word",
    "Highlight",
    "Glow",
    "Bold",
    "Soft",
];

/// The presets, in the order the Presets page shows them.
pub const CAPTION_PRESETS: &[CaptionPreset] = &[
    CaptionPreset {
        id: "cutlass-caption-clean",
        name: "Clean",
        style: "classic",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.06),
        categories: &["Trending", "Classic"],
    },
    CaptionPreset {
        id: "cutlass-caption-karaoke",
        name: "Sing along",
        style: "karaoke",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.08),
        categories: &["Trending", "Highlight"],
    },
    CaptionPreset {
        id: "cutlass-caption-marker",
        name: "Marker",
        style: "highlight",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: ("", 0.0),
        categories: &["Highlight"],
    },
    CaptionPreset {
        id: "cutlass-caption-pop",
        name: "Pop up",
        style: "pop",
        animation: Some("cutlass-anim-loop-pulse"),
        stagger_ms: 30,
        color: WHITE,
        outline: (INK, 0.1),
        categories: &["Trending", "Bold"],
    },
    CaptionPreset {
        id: "cutlass-caption-reveal",
        name: "Reveal",
        style: "word-by-word",
        animation: None,
        stagger_ms: 40,
        color: WHITE,
        outline: (INK, 0.06),
        categories: &["Word"],
    },
    CaptionPreset {
        id: "cutlass-caption-neon",
        name: "Neon float",
        style: "neon",
        animation: Some("cutlass-anim-loop-float"),
        stagger_ms: 0,
        color: WHITE,
        outline: ("", 0.0),
        categories: &["Glow"],
    },
    CaptionPreset {
        id: "cutlass-caption-punch",
        name: "Punch",
        style: "bold-box",
        animation: Some("cutlass-anim-loop-shake"),
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.08),
        categories: &["Bold"],
    },
    CaptionPreset {
        id: "cutlass-caption-glide",
        name: "Glide",
        style: "rise",
        animation: Some("cutlass-anim-loop-breathe"),
        stagger_ms: 60,
        color: WHITE,
        outline: (INK, 0.06),
        categories: &["Soft"],
    },
    CaptionPreset {
        id: "cutlass-caption-sunny",
        name: "Sunny",
        style: "classic",
        animation: None,
        stagger_ms: 0,
        color: YELLOW,
        outline: (INK, 0.08),
        categories: &["Trending", "Classic"],
    },
    CaptionPreset {
        id: "cutlass-caption-subtitle",
        name: "Subtitle",
        style: "classic",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: ("", 0.0),
        categories: &["Classic", "Soft"],
    },
    CaptionPreset {
        id: "cutlass-caption-headline",
        name: "Headline",
        style: "one-word",
        animation: Some("cutlass-anim-loop-pulse"),
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.12),
        categories: &["Word", "Bold"],
    },
    CaptionPreset {
        id: "cutlass-caption-spotlight",
        name: "Spotlight",
        style: "one-word",
        animation: None,
        stagger_ms: 0,
        color: YELLOW,
        outline: (INK, 0.1),
        categories: &["Trending", "Word"],
    },
    CaptionPreset {
        id: "cutlass-caption-typewriter",
        name: "Typewriter",
        style: "word-by-word",
        animation: None,
        stagger_ms: 80,
        color: WHITE,
        outline: ("", 0.0),
        categories: &["Word", "Soft"],
    },
    CaptionPreset {
        id: "cutlass-caption-duet",
        name: "Duet",
        style: "karaoke",
        animation: None,
        stagger_ms: 0,
        color: PINK,
        outline: (INK, 0.08),
        categories: &["Highlight"],
    },
    CaptionPreset {
        id: "cutlass-caption-chorus",
        name: "Chorus",
        style: "karaoke",
        animation: Some("cutlass-anim-loop-breathe"),
        stagger_ms: 0,
        color: LIME,
        outline: (INK, 0.08),
        categories: &["Highlight"],
    },
    CaptionPreset {
        id: "cutlass-caption-ink-marker",
        name: "Ink marker",
        style: "highlight",
        animation: None,
        stagger_ms: 0,
        color: INK,
        outline: ("", 0.0),
        categories: &["Highlight", "Classic"],
    },
    CaptionPreset {
        id: "cutlass-caption-night-glow",
        name: "Night glow",
        style: "neon",
        animation: None,
        stagger_ms: 0,
        color: CYAN,
        outline: ("", 0.0),
        categories: &["Trending", "Glow"],
    },
    CaptionPreset {
        id: "cutlass-caption-flicker",
        name: "Flicker",
        style: "neon",
        animation: Some("cutlass-anim-loop-flicker"),
        stagger_ms: 0,
        color: PINK,
        outline: ("", 0.0),
        categories: &["Glow"],
    },
    CaptionPreset {
        id: "cutlass-caption-bounce",
        name: "Bounce",
        style: "pop",
        animation: Some("cutlass-anim-loop-float"),
        stagger_ms: 40,
        color: YELLOW,
        outline: (INK, 0.1),
        categories: &["Bold"],
    },
    CaptionPreset {
        id: "cutlass-caption-block",
        name: "Block",
        style: "bold-box",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.06),
        categories: &["Bold", "Classic"],
    },
    CaptionPreset {
        id: "cutlass-caption-drift",
        name: "Drift",
        style: "fade",
        animation: Some("cutlass-anim-loop-float"),
        stagger_ms: 30,
        color: WHITE,
        outline: ("", 0.0),
        categories: &["Soft"],
    },
    CaptionPreset {
        id: "cutlass-caption-whisper",
        name: "Whisper",
        style: "fade",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.04),
        categories: &["Soft"],
    },
    CaptionPreset {
        id: "cutlass-caption-lift",
        name: "Lift",
        style: "rise",
        animation: None,
        stagger_ms: 40,
        color: YELLOW,
        outline: (INK, 0.08),
        categories: &["Word"],
    },
    CaptionPreset {
        id: "cutlass-caption-swing",
        name: "Swing",
        style: "pop",
        animation: Some("cutlass-anim-loop-swing"),
        stagger_ms: 20,
        color: ORANGE,
        outline: (INK, 0.1),
        categories: &["Trending", "Bold"],
    },
];

/// The preset stored as `id`, if it is one.
#[must_use]
pub fn caption_preset(id: &str) -> Option<&'static CaptionPreset> {
    CAPTION_PRESETS.iter().find(|preset| preset.id == id)
}

/// The requests that style one caption as `preset`.
fn styled(segment_id: &SegmentId, preset: &CaptionPreset) -> Vec<Request> {
    let duration = preset
        .animation
        .and_then(|id| {
            ANIMATION_PRESETS
                .iter()
                .find(|preset| preset.resource_id == id && preset.kind == AnimType::Loop)
        })
        .map_or(Micros(0), |preset| preset.duration);
    vec![
        Request::SetCaptionStyle {
            segment_id: segment_id.clone(),
            style: preset.style.to_owned(),
        },
        Request::SetCaptionStagger {
            segment_id: segment_id.clone(),
            millis: preset.stagger_ms,
        },
        Request::SetTextColor {
            segment_id: segment_id.clone(),
            color: preset.color.to_owned(),
            face: String::new(),
        },
        Request::SetTextOutline {
            segment_id: segment_id.clone(),
            color: preset.outline.0.to_owned(),
            width: preset.outline.1,
        },
        Request::SetAnimation {
            segment_id: segment_id.clone(),
            kind: AnimType::Caption,
            animation: preset.animation.map(str::to_owned),
            duration,
        },
        Request::SetCaptionTemplate {
            segment_id: segment_id.clone(),
            resource_id: preset.id.to_owned(),
            name: preset.name.to_owned(),
        },
    ]
}

/// The requests that style caption `segment_id` as `preset`, as one undo
/// step. With `all` and the caption on a subtitle lane, every caption on
/// that lane takes it.
#[must_use]
pub fn caption_preset_requests(
    draft: &Draft,
    segment_id: &SegmentId,
    preset: &CaptionPreset,
    all: bool,
) -> Vec<Request> {
    let clips: Vec<SegmentId> = if all && main_caption_lane(draft, segment_id).is_some() {
        caption_rows(draft, segment_id)
            .into_iter()
            .map(|row| row.segment_id)
            .collect()
    } else {
        vec![segment_id.clone()]
    };
    clips.iter().flat_map(|clip| styled(clip, preset)).collect()
}

/// The requests that take caption `segment_id`'s preset off: the plain
/// style, no stagger and no caption animation. The letters keep their
/// color and outline. With `all` and the caption on a subtitle lane, every
/// caption on that lane is plain again.
#[must_use]
pub fn plain_caption_requests(draft: &Draft, segment_id: &SegmentId, all: bool) -> Vec<Request> {
    let clips: Vec<SegmentId> = if all && main_caption_lane(draft, segment_id).is_some() {
        caption_rows(draft, segment_id)
            .into_iter()
            .map(|row| row.segment_id)
            .collect()
    } else {
        vec![segment_id.clone()]
    };
    clips
        .into_iter()
        .flat_map(|clip| {
            [
                Request::SetCaptionStyle {
                    segment_id: clip.clone(),
                    style: String::new(),
                },
                Request::SetCaptionStagger {
                    segment_id: clip.clone(),
                    millis: 0,
                },
                Request::SetAnimation {
                    segment_id: clip.clone(),
                    kind: AnimType::Caption,
                    animation: None,
                    duration: Micros(0),
                },
                Request::SetCaptionTemplate {
                    segment_id: clip,
                    resource_id: String::new(),
                    name: String::new(),
                },
            ]
        })
        .collect()
}

/// The preset caption `segment_id` was styled with, by its id.
#[must_use]
pub fn caption_preset_of(draft: &Draft, segment_id: &SegmentId) -> Option<String> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    let text = draft
        .materials
        .texts
        .iter()
        .find(|text| text.id == segment.material_id)?;
    text.caption_template_info
        .get()
        .map(|info| info.resource_id.clone())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use cutlass_model::{CAPTION_STYLE_KEYS, TrackFlag, segment_animations};
    use cutlass_session::Session;

    use super::*;

    fn session() -> Session {
        let srt =
            "00:00:00,000 --> 00:00:02,000\none two\n\n00:00:03,000 --> 00:00:05,000\nthree\n";
        let mut ids = 0;
        let request = crate::subtitle_request(
            srt,
            crate::SubtitleFormat::Srt,
            "sample.ttf",
            &mut |p: &str| {
                ids += 1;
                format!("{p}-{ids}")
            },
        )
        .unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[request]).unwrap();
        session
    }

    fn clips(session: &Session) -> Vec<SegmentId> {
        session
            .draft()
            .tracks
            .iter()
            .find(|track| track.flag == TrackFlag::TextSubtitle)
            .unwrap()
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect()
    }

    #[test]
    fn every_preset_names_a_style_and_a_loop_animation_once() {
        let mut ids = HashSet::new();
        for preset in CAPTION_PRESETS {
            assert!(ids.insert(preset.id), "{}", preset.id);
            assert!(CAPTION_STYLE_KEYS.contains(&preset.style), "{}", preset.id);
            if let Some(animation) = preset.animation {
                assert!(
                    ANIMATION_PRESETS
                        .iter()
                        .any(|p| p.resource_id == animation && p.kind == AnimType::Loop),
                    "{animation}"
                );
            }
            assert!((0..=cutlass_session::MAX_CAPTION_STAGGER_MS).contains(&preset.stagger_ms));
            assert_eq!(caption_preset(preset.id), Some(preset));
        }
        assert_eq!(caption_preset("nope"), None);
    }

    #[test]
    fn a_preset_styles_every_main_caption_in_one_step() {
        let mut session = session();
        let ids = clips(&session);
        let before = session.draft().clone();
        let pop = caption_preset("cutlass-caption-pop").unwrap();
        let requests = caption_preset_requests(session.draft(), &ids[0], pop, true);
        session.apply(1, &requests).unwrap();
        let draft = session.draft();
        for id in &ids {
            assert_eq!(caption_preset_of(draft, id).as_deref(), Some(pop.id));
            let segment = draft
                .tracks
                .iter()
                .flat_map(|t| &t.segments)
                .find(|s| &s.id == id)
                .unwrap();
            let text = draft
                .materials
                .texts
                .iter()
                .find(|t| t.id == segment.material_id)
                .unwrap();
            assert_eq!(text.caption_style, "pop");
            assert_eq!(text.caption_stagger_ms, 30);
            assert_eq!(text.border_color, "#111111");
            let row = segment_animations(draft, segment).unwrap();
            assert_eq!(row.animations[0].kind, "caption");
            assert_eq!(row.animations[0].resource_id, "cutlass-anim-loop-pulse");
        }
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);

        // One at a time, a preset reaches its own caption only; one with
        // no animation takes a caption animation off.
        let requests = caption_preset_requests(session.draft(), &ids[0], pop, false);
        session.apply(session.revision(), &requests).unwrap();
        let clean = caption_preset("cutlass-caption-clean").unwrap();
        let requests = caption_preset_requests(session.draft(), &ids[0], clean, false);
        session.apply(session.revision(), &requests).unwrap();
        let draft = session.draft();
        assert_eq!(caption_preset_of(draft, &ids[0]).as_deref(), Some(clean.id));
        assert_eq!(caption_preset_of(draft, &ids[1]), None);
        assert!(draft.materials.material_animations.is_empty());
    }

    #[test]
    fn none_takes_the_preset_off_and_keeps_the_letters() {
        let mut session = session();
        let ids = clips(&session);
        let pop = caption_preset("cutlass-caption-pop").unwrap();
        let revision = session.revision();
        let requests = caption_preset_requests(session.draft(), &ids[0], pop, true);
        session.apply(revision, &requests).unwrap();
        let revision = session.revision();
        let requests = plain_caption_requests(session.draft(), &ids[0], true);
        session.apply(revision, &requests).unwrap();
        for id in &ids {
            assert_eq!(caption_preset_of(session.draft(), id), None);
            let segment = session
                .draft()
                .tracks
                .iter()
                .flat_map(|track| &track.segments)
                .find(|segment| &segment.id == id)
                .unwrap();
            let text = session
                .draft()
                .materials
                .texts
                .iter()
                .find(|text| text.id == segment.material_id)
                .unwrap();
            assert!(text.caption_style.is_empty());
            assert_eq!(text.caption_stagger_ms, 0);
            assert_eq!(text.text_color, pop.color);
            assert!(
                segment_animations(session.draft(), segment)
                    .is_none_or(|row| row.animations.iter().all(|a| a.kind != "caption"))
            );
        }
    }
}
