//! Caption templates every shell offers: a caption style, the caption
//! animation that plays in the loop slot, a letter stagger, and the base
//! look of the words. A keyword is drawn in the style's spoken look.
//!
//! A template is the app's own; the reference editor's templates are
//! downloaded packages. Picking one records it on the caption as its
//! `caption_template_info`.

use cutlass_model::{ANIMATION_PRESETS, AnimType, Draft, Micros, SegmentId};
use cutlass_session::Request;

use crate::caption_list::{caption_rows, main_caption_lane};

/// One caption template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptionTemplate {
    /// Stored as the caption's template resource id.
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
}

const WHITE: &str = "#FFFFFF";
const INK: &str = "#111111";

/// The templates, in the order the Templates page shows them.
pub const CAPTION_TEMPLATES: [CaptionTemplate; 8] = [
    CaptionTemplate {
        id: "cutlass-caption-clean",
        name: "Clean",
        style: "classic",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.06),
    },
    CaptionTemplate {
        id: "cutlass-caption-karaoke",
        name: "Sing along",
        style: "karaoke",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.08),
    },
    CaptionTemplate {
        id: "cutlass-caption-marker",
        name: "Marker",
        style: "highlight",
        animation: None,
        stagger_ms: 0,
        color: WHITE,
        outline: ("", 0.0),
    },
    CaptionTemplate {
        id: "cutlass-caption-pop",
        name: "Pop up",
        style: "pop",
        animation: Some("cutlass-anim-loop-pulse"),
        stagger_ms: 30,
        color: WHITE,
        outline: (INK, 0.1),
    },
    CaptionTemplate {
        id: "cutlass-caption-reveal",
        name: "Reveal",
        style: "word-by-word",
        animation: None,
        stagger_ms: 40,
        color: WHITE,
        outline: (INK, 0.06),
    },
    CaptionTemplate {
        id: "cutlass-caption-neon",
        name: "Neon float",
        style: "neon",
        animation: Some("cutlass-anim-loop-float"),
        stagger_ms: 0,
        color: WHITE,
        outline: ("", 0.0),
    },
    CaptionTemplate {
        id: "cutlass-caption-punch",
        name: "Punch",
        style: "bold-box",
        animation: Some("cutlass-anim-loop-shake"),
        stagger_ms: 0,
        color: WHITE,
        outline: (INK, 0.08),
    },
    CaptionTemplate {
        id: "cutlass-caption-glide",
        name: "Glide",
        style: "rise",
        animation: Some("cutlass-anim-loop-breathe"),
        stagger_ms: 60,
        color: WHITE,
        outline: (INK, 0.06),
    },
];

/// The template stored as `id`, if it is one.
#[must_use]
pub fn caption_template(id: &str) -> Option<&'static CaptionTemplate> {
    CAPTION_TEMPLATES.iter().find(|template| template.id == id)
}

/// The requests that style one caption as `template`.
fn styled(segment_id: &SegmentId, template: &CaptionTemplate) -> Vec<Request> {
    let duration = template
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
            style: template.style.to_owned(),
        },
        Request::SetCaptionStagger {
            segment_id: segment_id.clone(),
            millis: template.stagger_ms,
        },
        Request::SetTextColor {
            segment_id: segment_id.clone(),
            color: template.color.to_owned(),
            face: String::new(),
        },
        Request::SetTextOutline {
            segment_id: segment_id.clone(),
            color: template.outline.0.to_owned(),
            width: template.outline.1,
        },
        Request::SetAnimation {
            segment_id: segment_id.clone(),
            kind: AnimType::Caption,
            animation: template.animation.map(str::to_owned),
            duration,
        },
        Request::SetCaptionTemplate {
            segment_id: segment_id.clone(),
            resource_id: template.id.to_owned(),
            name: template.name.to_owned(),
        },
    ]
}

/// The requests that style caption `segment_id` as `template`, as one undo
/// step. With `all` and the caption on a subtitle lane, every caption on
/// that lane takes it.
#[must_use]
pub fn caption_template_requests(
    draft: &Draft,
    segment_id: &SegmentId,
    template: &CaptionTemplate,
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
    clips
        .iter()
        .flat_map(|clip| styled(clip, template))
        .collect()
}

/// The template caption `segment_id` was styled with, by its id.
#[must_use]
pub fn caption_template_of(draft: &Draft, segment_id: &SegmentId) -> Option<String> {
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
        let request = crate::subtitle_request(srt, crate::SubtitleFormat::Srt, &mut |p: &str| {
            ids += 1;
            format!("{p}-{ids}")
        })
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
    fn every_template_names_a_style_and_a_loop_preset_once() {
        let mut ids = HashSet::new();
        for template in &CAPTION_TEMPLATES {
            assert!(ids.insert(template.id), "{}", template.id);
            assert!(
                CAPTION_STYLE_KEYS.contains(&template.style),
                "{}",
                template.id
            );
            if let Some(animation) = template.animation {
                assert!(
                    ANIMATION_PRESETS
                        .iter()
                        .any(|p| p.resource_id == animation && p.kind == AnimType::Loop),
                    "{animation}"
                );
            }
            assert!((0..=cutlass_session::MAX_CAPTION_STAGGER_MS).contains(&template.stagger_ms));
            assert_eq!(caption_template(template.id), Some(template));
        }
        assert_eq!(caption_template("nope"), None);
    }

    #[test]
    fn a_template_styles_every_main_caption_in_one_step() {
        let mut session = session();
        let ids = clips(&session);
        let before = session.draft().clone();
        let pop = caption_template("cutlass-caption-pop").unwrap();
        let requests = caption_template_requests(session.draft(), &ids[0], pop, true);
        session.apply(1, &requests).unwrap();
        let draft = session.draft();
        for id in &ids {
            assert_eq!(caption_template_of(draft, id).as_deref(), Some(pop.id));
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

        // One at a time, a template reaches its own caption only; one with
        // no animation takes a caption animation off.
        let requests = caption_template_requests(session.draft(), &ids[0], pop, false);
        session.apply(session.revision(), &requests).unwrap();
        let clean = caption_template("cutlass-caption-clean").unwrap();
        let requests = caption_template_requests(session.draft(), &ids[0], clean, false);
        session.apply(session.revision(), &requests).unwrap();
        let draft = session.draft();
        assert_eq!(
            caption_template_of(draft, &ids[0]).as_deref(),
            Some(clean.id)
        );
        assert_eq!(caption_template_of(draft, &ids[1]), None);
        assert!(draft.materials.material_animations.is_empty());
    }
}
