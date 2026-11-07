//! What the text panel's Templates, Bubble and Effects pages offer, found
//! by a search and a category, and a small project that draws each one as
//! its tile's picture.

use cutlass_model::{
    AnimType, Bubble, Draft, MaterialId, MaterialText, Micros, Segment, SegmentId, TimeRange,
    Track, TrackFlag, TrackType,
};
use cutlass_session::{Request, Session};

use crate::caption_templates::CaptionTemplate;
use crate::inspector::TextLook;
use crate::text_effect::{TextEffect, text_effect_requests};

/// A bubble with its box color and its letters' color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BubbleLook {
    pub id: &'static str,
    pub name: &'static str,
    pub bubble: Bubble,
    /// The box's color.
    pub plate: &'static str,
    /// The letters' color.
    pub ink: &'static str,
    /// The groups the Bubble page files it under, as
    /// [`BUBBLE_CATEGORIES`] names them.
    pub categories: &'static [&'static str],
}

/// The groups the Bubble page offers, in its order.
pub const BUBBLE_CATEGORIES: &[&str] = &["Speech", "Labels", "Shout"];

macro_rules! bubble_look {
    ($id:literal, $name:literal, $bubble:ident, $plate:literal, $ink:literal, $group:literal) => {
        BubbleLook {
            id: $id,
            name: $name,
            bubble: Bubble::$bubble,
            plate: $plate,
            ink: $ink,
            categories: &[$group],
        }
    };
}

/// The bubbles, in the order the Bubble page shows them after None.
pub const BUBBLE_LOOKS: &[BubbleLook] = &[
    bubble_look!(
        "speech-white",
        "Say",
        Speech,
        "#FFFFFF",
        "#111111",
        "Speech"
    ),
    bubble_look!(
        "speech-black",
        "Say dark",
        Speech,
        "#111111",
        "#FFFFFF",
        "Speech"
    ),
    bubble_look!(
        "speech-yellow",
        "Say sunny",
        Speech,
        "#FFD400",
        "#111111",
        "Speech"
    ),
    bubble_look!(
        "speech-pink",
        "Say pink",
        Speech,
        "#FF4FA3",
        "#FFFFFF",
        "Speech"
    ),
    bubble_look!(
        "thought-white",
        "Think",
        Thought,
        "#FFFFFF",
        "#111111",
        "Speech"
    ),
    bubble_look!(
        "thought-black",
        "Think dark",
        Thought,
        "#111111",
        "#FFFFFF",
        "Speech"
    ),
    bubble_look!(
        "thought-sky",
        "Think sky",
        Thought,
        "#8FD3FF",
        "#111111",
        "Speech"
    ),
    bubble_look!(
        "thought-lilac",
        "Think lilac",
        Thought,
        "#C9A7FF",
        "#111111",
        "Speech"
    ),
    bubble_look!("pill-white", "Tag", Pill, "#FFFFFF", "#111111", "Labels"),
    bubble_look!(
        "pill-black",
        "Tag dark",
        Pill,
        "#111111",
        "#FFFFFF",
        "Labels"
    ),
    bubble_look!(
        "pill-lime",
        "Tag lime",
        Pill,
        "#B6FF3B",
        "#111111",
        "Labels"
    ),
    bubble_look!("pill-red", "Tag red", Pill, "#FF2E2E", "#FFFFFF", "Labels"),
    bubble_look!(
        "banner-cream",
        "Ribbon",
        Banner,
        "#FFF3D6",
        "#111111",
        "Labels"
    ),
    bubble_look!(
        "banner-black",
        "Ribbon dark",
        Banner,
        "#111111",
        "#FFFFFF",
        "Labels"
    ),
    bubble_look!(
        "banner-gold",
        "Ribbon gold",
        Banner,
        "#FFB800",
        "#111111",
        "Labels"
    ),
    bubble_look!(
        "banner-blue",
        "Ribbon blue",
        Banner,
        "#2E6BFF",
        "#FFFFFF",
        "Labels"
    ),
    bubble_look!("burst-yellow", "Boom", Burst, "#FFD400", "#111111", "Shout"),
    bubble_look!(
        "burst-red",
        "Boom red",
        Burst,
        "#FF2E2E",
        "#FFFFFF",
        "Shout"
    ),
    bubble_look!(
        "burst-white",
        "Boom white",
        Burst,
        "#FFFFFF",
        "#111111",
        "Shout"
    ),
    bubble_look!(
        "burst-violet",
        "Boom violet",
        Burst,
        "#7A2BFF",
        "#FFFFFF",
        "Shout"
    ),
];

/// The requests that give `segment_id` the bubble `look`: its shape, box
/// and letters. `face` is the shell's face, for the letters' color.
#[must_use]
pub fn bubble_look_requests(look: &BubbleLook, segment_id: &SegmentId, face: &str) -> Vec<Request> {
    vec![
        Request::SetTextBubble {
            segment_id: segment_id.clone(),
            bubble: look.bubble.key().to_owned(),
        },
        Request::SetTextBackground {
            segment_id: segment_id.clone(),
            color: look.plate.to_owned(),
            alpha: 1.0,
            round: 0.0,
        },
        Request::SetTextColor {
            segment_id: segment_id.clone(),
            color: look.ink.to_owned(),
            face: face.to_owned(),
        },
    ]
}

/// The bubble look a title wears, by its shape and box color.
#[must_use]
pub fn bubble_look_of(look: &TextLook) -> Option<&'static BubbleLook> {
    BUBBLE_LOOKS.iter().find(|bubble| {
        look.bubble == bubble.bubble.key()
            && look.background_color.eq_ignore_ascii_case(bubble.plate)
    })
}

/// An item is shown when it is in `category` (any, when empty) and every
/// word of `query` starts a word of its name or one of its categories,
/// ignoring case.
#[must_use]
pub fn catalog_shows(name: &str, categories: &[&str], query: &str, category: &str) -> bool {
    if !category.is_empty() && !categories.contains(&category) {
        return false;
    }
    let words: Vec<String> = std::iter::once(name)
        .chain(categories.iter().copied())
        .flat_map(str::split_whitespace)
        .map(str::to_lowercase)
        .collect();
    query
        .split_whitespace()
        .map(str::to_lowercase)
        .all(|asked| words.iter().any(|word| word.starts_with(&asked)))
}

/// What one tile shows.
#[derive(Clone, Copy, Debug)]
pub enum CatalogTile<'a> {
    Template(&'a CaptionTemplate),
    Effect(&'a TextEffect),
    Bubble(&'a BubbleLook),
}

/// The side of a tile's picture, in pixels.
pub const TILE_SIDE: i32 = 160;

const TILE_CLIP: &str = "tile";
/// A caption's tile is drawn while its second word is said, so a style
/// that marks the spoken word shows it.
const TILE_WORDS: [(&str, i64, i64); 2] = [("Say", 0, 500), ("it", 500, 2_000)];
const TILE_TIME: Micros = Micros(900_000);

/// A small project with one short text in `face`, styled as `tile`, and
/// the time to draw it at. A template's text is a caption, so its style
/// marks the spoken word; an effect's and a bubble's is a title.
#[must_use]
pub fn tile_session(tile: CatalogTile<'_>, face: &str) -> (Session, Micros) {
    let mut draft = Draft::empty("tile");
    draft.canvas_config.width = TILE_SIDE;
    draft.canvas_config.height = TILE_SIDE;
    let mut text = MaterialText::new(TILE_CLIP);
    text.font_path = face.to_owned();
    text.text_color = "#FFFFFF".to_owned();
    let caption = matches!(tile, CatalogTile::Template(_));
    if caption {
        text.content = "Say it".to_owned();
        text.words.text = TILE_WORDS
            .iter()
            .map(|(word, ..)| (*word).to_owned())
            .collect();
        text.words.start_time = TILE_WORDS.iter().map(|(_, start, _)| *start).collect();
        text.words.end_time = TILE_WORDS.iter().map(|(.., end)| *end).collect();
        text.font_size = 14.0;
    } else {
        text.content = if matches!(tile, CatalogTile::Bubble(_)) {
            "Hi!"
        } else {
            "Abc"
        }
        .to_owned();
        text.font_size = 16.0;
    }
    draft.materials.texts.push(text);
    let mut track = Track::main(TILE_CLIP);
    track.kind = TrackType::Text;
    if caption {
        track.flag = TrackFlag::TextSubtitle;
    }
    track.segments.push(Segment::new(
        TILE_CLIP,
        TILE_CLIP,
        TimeRange::new(0, 2_000_000),
    ));
    draft.tracks = vec![track];
    let mut session = Session::open(draft);
    let id = SegmentId::new(TILE_CLIP);
    let requests = match tile {
        CatalogTile::Template(template) => {
            let mut requests = crate::caption_templates::caption_template_requests(
                session.draft(),
                &id,
                template,
                false,
            );
            // A still tile has no loop to play.
            requests.retain(|request| {
                !matches!(
                    request,
                    Request::SetAnimation {
                        kind: AnimType::Caption,
                        ..
                    }
                )
            });
            requests
        }
        CatalogTile::Effect(effect) => {
            text_effect_requests(effect, &id, MaterialId::new("tile-glow"), face)
        }
        CatalogTile::Bubble(bubble) => bubble_look_requests(bubble, &id, face),
    };
    let revision = session.revision();
    // A request the tile's text refuses leaves it plain.
    let _ = session.apply(revision, &requests);
    (session, TILE_TIME)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::caption_templates::{CAPTION_TEMPLATES, TEMPLATE_CATEGORIES};
    use crate::inspector::text_look;
    use crate::text_effect::{EFFECT_CATEGORIES, TEXT_EFFECTS};

    #[test]
    fn every_item_has_a_name_an_id_and_categories_its_page_offers() {
        let mut ids = HashSet::new();
        for template in CAPTION_TEMPLATES {
            assert!(ids.insert(template.id), "{}", template.id);
            assert!(!template.categories.is_empty(), "{}", template.id);
            assert!(
                template
                    .categories
                    .iter()
                    .all(|c| TEMPLATE_CATEGORIES.contains(c))
            );
        }
        for effect in TEXT_EFFECTS {
            assert!(ids.insert(effect.id), "{}", effect.id);
            assert!(!effect.categories.is_empty(), "{}", effect.id);
            assert!(
                effect
                    .categories
                    .iter()
                    .all(|c| EFFECT_CATEGORIES.contains(c))
            );
        }
        for bubble in BUBBLE_LOOKS {
            assert!(ids.insert(bubble.id), "{}", bubble.id);
            assert!(
                bubble
                    .categories
                    .iter()
                    .all(|c| BUBBLE_CATEGORIES.contains(c))
            );
        }
        // Each category has something in it.
        for category in TEMPLATE_CATEGORIES {
            assert!(
                CAPTION_TEMPLATES
                    .iter()
                    .any(|t| t.categories.contains(category))
            );
        }
        for category in EFFECT_CATEGORIES {
            assert!(TEXT_EFFECTS.iter().any(|e| e.categories.contains(category)));
        }
        for category in BUBBLE_CATEGORIES {
            assert!(BUBBLE_LOOKS.iter().any(|b| b.categories.contains(category)));
        }
    }

    #[test]
    fn a_search_matches_the_starts_of_words_in_the_name_or_its_categories() {
        let cats = &["Glow", "Trending"];
        assert!(catalog_shows("Gold glow", cats, "", ""));
        assert!(catalog_shows("Gold glow", cats, "GLO", ""));
        assert!(catalog_shows("Gold glow", cats, "gold gl", ""));
        assert!(catalog_shows("Gold glow", cats, "trend", "Glow"));
        assert!(!catalog_shows("Gold glow", cats, "old", ""));
        assert!(!catalog_shows("Gold glow", cats, "", "Neon"));
        assert!(!catalog_shows("Gold glow", cats, "gold neon", ""));
    }

    #[test]
    fn a_bubble_look_is_one_step_and_reads_back() {
        let mut draft = Draft::empty("draft");
        let mut text = MaterialText::new("words");
        text.content = "Hi".to_owned();
        draft.materials.texts.push(text);
        draft.tracks[0]
            .segments
            .push(Segment::new("title", "words", TimeRange::new(0, 1_000_000)));
        let mut session = Session::open(draft);
        let id = SegmentId::new("title");
        for bubble in BUBBLE_LOOKS {
            let revision = session.revision();
            session
                .apply(revision, &bubble_look_requests(bubble, &id, "face.ttf"))
                .unwrap();
            let draft = session.draft();
            let look = text_look(draft, &draft.tracks[0].segments[0]).unwrap();
            assert_eq!(bubble_look_of(&look).map(|b| b.id), Some(bubble.id));
            assert_eq!(draft.materials.texts[0].text_color, bubble.ink);
        }
    }

    #[test]
    fn a_tile_project_wears_its_item() {
        let id = SegmentId::new(TILE_CLIP);
        let segment = |session: &Session| session.draft().tracks[0].segments[0].clone();
        for template in CAPTION_TEMPLATES {
            let (session, _) = tile_session(CatalogTile::Template(template), "face.ttf");
            assert_eq!(
                crate::caption_template_of(session.draft(), &id).as_deref(),
                Some(template.id)
            );
        }
        for effect in TEXT_EFFECTS {
            let (session, _) = tile_session(CatalogTile::Effect(effect), "face.ttf");
            let look = text_look(session.draft(), &segment(&session)).unwrap();
            assert_eq!(crate::text_effect_of(&look).map(|e| e.id), Some(effect.id));
        }
        for bubble in BUBBLE_LOOKS {
            let (session, _) = tile_session(CatalogTile::Bubble(bubble), "face.ttf");
            let look = text_look(session.draft(), &segment(&session)).unwrap();
            assert_eq!(bubble_look_of(&look).map(|b| b.id), Some(bubble.id));
        }
    }
}
