//! The effects shelf: the bundled effects as cards, and the requests that
//! place one on an effect row or over one clip.
//!
//! An effect on a row runs over every picture below it. An effect dropped
//! on a picture clip takes that clip's span, is bound to it so it runs over
//! that clip alone, and is linked to it so it moves with it. Every screen
//! places an effect through here.

use std::collections::HashMap;

use cutlass_effects::{bundled, bundled_recipe};
use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, Segment, SegmentId, Track, TrackAttribute, TrackId,
    TrackType,
};
use cutlass_session::Request;

/// How long an effect added at the playhead or on a row lasts.
pub const EFFECT_LENGTH: Micros = Micros(3_000_000);

/// A card's id starts with this, so a drop can tell an effect from a file.
pub const EFFECT_CARD_PREFIX: &str = "effect:";

/// The one card on the adjustments shelf: an adjustment layer, its sliders
/// at zero, that runs over every picture below it.
pub const ADJUSTMENT_CARD: &str = "adjustment:basic";

/// A filter card's id starts with this, then the look's `effect_id`.
pub const FILTER_CARD_PREFIX: &str = "filter:";

/// The bundled filters as `(card id, name)`, in shelf order.
#[must_use]
pub fn filter_cards() -> Vec<(String, &'static str)> {
    cutlass_effects::looks()
        .iter()
        .map(|look| {
            (
                format!("{FILTER_CARD_PREFIX}{}", look.effect_id()),
                look.name.as_str(),
            )
        })
        .collect()
}

/// The filters shelf's categories that hold at least one filter, in shelf
/// order.
#[must_use]
pub fn filter_categories() -> Vec<&'static str> {
    cutlass_effects::FILTER_CATEGORIES
        .iter()
        .copied()
        .filter(|category| {
            cutlass_effects::looks()
                .iter()
                .any(|look| look.category == *category)
        })
        .collect()
}

/// The bundled filter a card id names.
#[must_use]
pub fn filter_of_card(card_id: &str) -> Option<&'static cutlass_effects::Look> {
    cutlass_effects::bundled_look(card_id.strip_prefix(FILTER_CARD_PREFIX)?)
}

/// One card on the effects shelf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectCard {
    /// The library card id.
    pub id: String,
    /// What [`Request::AddEffect`] takes, such as `recipe/shake`.
    pub effect_id: String,
    pub name: String,
    /// The shelf category, one of [`cutlass_effects::EFFECT_CATEGORIES`].
    pub category: String,
}

/// Every effect the app can run, in shelf order.
#[must_use]
pub fn effect_cards() -> Vec<EffectCard> {
    bundled()
        .iter()
        .map(|recipe| {
            let effect_id = recipe.effect_id();
            EffectCard {
                id: format!("{EFFECT_CARD_PREFIX}{effect_id}"),
                effect_id,
                name: recipe.name.clone(),
                category: recipe.category.clone(),
            }
        })
        .collect()
}

/// The shelf's categories that hold at least one effect, in shelf order.
#[must_use]
pub fn effect_categories() -> Vec<&'static str> {
    let cards = effect_cards();
    cutlass_effects::EFFECT_CATEGORIES
        .iter()
        .copied()
        .filter(|category| cards.iter().any(|card| card.category == *category))
        .collect()
}

/// The effect a card id names, when it is one the app can run.
#[must_use]
pub fn effect_of_card(card_id: &str) -> Option<String> {
    let effect_id = card_id.strip_prefix(EFFECT_CARD_PREFIX)?;
    bundled_recipe(effect_id).map(cutlass_effects::Recipe::effect_id)
}

/// Where a new effect goes and what its ids are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectPlace {
    /// An effect row; a missing one is added.
    pub track_id: TrackId,
    pub segment_id: SegmentId,
    pub material_id: MaterialId,
    pub start: Micros,
    pub duration: Micros,
    /// The picture clip the effect runs over alone, if any.
    pub on: Option<SegmentId>,
}

/// The span of a picture clip an effect can go on: a clip on a video,
/// sticker or text row with a positive length. A title or a shape takes an
/// effect as a movie does.
#[must_use]
pub fn effect_target(draft: &Draft, clip: &SegmentId) -> Option<(Micros, Micros)> {
    draft
        .tracks
        .iter()
        .filter(|track| {
            matches!(
                track.kind,
                TrackType::Video | TrackType::Sticker | TrackType::Text
            )
        })
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == clip && segment.target.duration.0 > 0)
        .map(|segment| (segment.target.start, segment.target.duration))
}

/// The effect row a clip-long effect over `clip` goes on: the first one
/// that holds only effects riding on clips and has room for the clip's
/// span, or `None` when a new row must open. A row with an effect over
/// every picture is left alone, so this effect never shows on it.
#[must_use]
pub fn effect_row_for(draft: &Draft, clip: &SegmentId) -> Option<TrackId> {
    let (start, duration) = effect_target(draft, clip)?;
    let end = start.0.max(0).saturating_add(duration.0);
    draft
        .tracks
        .iter()
        .find(|track| {
            carries_clip_effects(draft, track)
                && !track.attribute.contains(TrackAttribute::LOCK)
                && track.segments.iter().all(|segment| {
                    let from = segment.target.start.0;
                    let to = from.saturating_add(segment.target.duration.0);
                    to <= start.0.max(0) || from >= end
                })
        })
        .map(|track| track.id.clone())
}

/// The effect clip runs over one clip the draft still has, so the timeline
/// draws it on that clip and not on its own row. An effect bound to a clip
/// that is gone stays on its row, where it can still be found and removed.
#[must_use]
pub fn rides_on_clip(draft: &Draft, segment: &Segment) -> bool {
    segment
        .bound_clip()
        .is_some_and(|clip| effect_target(draft, &clip).is_some())
}

/// An effect row that holds clips, every one of them riding on a clip: the
/// timeline draws no row for it.
#[must_use]
pub fn carries_clip_effects(draft: &Draft, track: &Track) -> bool {
    track.kind == TrackType::Effect
        && !track.segments.is_empty()
        && track
            .segments
            .iter()
            .all(|segment| rides_on_clip(draft, segment))
}

/// Every clip's own effects, by the clip they run over, each list in the
/// order [`effects_on`] gives. One pass over the effect rows, for a
/// timeline that draws them on their clips.
#[must_use]
pub fn effects_by_clip(draft: &Draft) -> HashMap<SegmentId, Vec<ClipEffect>> {
    let mut by_clip: HashMap<SegmentId, Vec<ClipEffect>> = HashMap::new();
    for track in draft
        .tracks
        .iter()
        .filter(|track| track.kind == TrackType::Effect)
    {
        let mut bound: Vec<(SegmentId, &Segment)> = track
            .segments
            .iter()
            .filter(|segment| rides_on_clip(draft, segment))
            .filter_map(|segment| Some((segment.bound_clip()?, segment)))
            .collect();
        bound.sort_by_key(|(_, segment)| segment.target.start);
        for (clip, segment) in bound {
            by_clip.entry(clip).or_default().push(ClipEffect {
                segment_id: segment.id.clone(),
                name: effect_title(draft, segment),
            });
        }
    }
    by_clip
}

/// The requests that place `effect_id` as one undo step. An effect on a
/// clip is bound to it, and linked to it unless the clip already follows
/// another clip.
#[must_use]
pub fn effect_requests(draft: &Draft, effect_id: &str, place: &EffectPlace) -> Vec<Request> {
    let mut requests = vec![Request::AddEffect {
        track_id: place.track_id.clone(),
        segment_id: place.segment_id.clone(),
        material_id: place.material_id.clone(),
        start: Micros(place.start.0.max(0)),
        duration: place.duration,
        effect_id: effect_id.to_owned(),
    }];
    if let Some(clip) = &place.on {
        requests.push(Request::BindEffect {
            segment_id: place.segment_id.clone(),
            clip: Some(clip.clone()),
        });
        if !follows_another(draft, clip) {
            requests.push(Request::Link {
                owner: clip.clone(),
                dependent: place.segment_id.clone(),
            });
        }
    }
    requests
}

/// An effect that runs over one clip alone: its own clip on an effect row,
/// and the name it shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipEffect {
    pub segment_id: SegmentId,
    pub name: String,
}

/// The name an effect, filter or adjustment clip shows: the one its
/// material stores, or the material's id when it stores none.
#[must_use]
pub fn effect_title(draft: &Draft, segment: &Segment) -> String {
    let name = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::VideoEffect(effect)) => effect.name.clone(),
        Some(MaterialRef::Effect(effect)) => effect.name.clone(),
        Some(MaterialRef::Plugin(effect)) => effect.plugin_id.clone(),
        Some(MaterialRef::Other(raw)) => raw
            .fields
            .get("name")
            .and_then(|name| name.as_str())
            .unwrap_or_default()
            .to_owned(),
        _ => String::new(),
    };
    if name.is_empty() {
        segment.material_id.as_str().to_owned()
    } else {
        name
    }
}

/// The effects that run over `clip` alone, in the order their rows stand
/// and then by where they start.
#[must_use]
pub fn effects_on(draft: &Draft, clip: &SegmentId) -> Vec<ClipEffect> {
    draft
        .tracks
        .iter()
        .filter(|track| track.kind == TrackType::Effect)
        .flat_map(|track| {
            let mut bound: Vec<&Segment> = track
                .segments
                .iter()
                .filter(|segment| segment.bound_clip().as_ref() == Some(clip))
                .collect();
            bound.sort_by_key(|segment| segment.target.start);
            bound
        })
        .map(|segment| ClipEffect {
            segment_id: segment.id.clone(),
            name: effect_title(draft, segment),
        })
        .collect()
}

/// The clip is linked as another clip's dependent, so it cannot own one.
fn follows_another(draft: &Draft, clip: &SegmentId) -> bool {
    draft
        .relationships
        .iter()
        .filter(|relationship| relationship.kind == "linkage")
        .flat_map(|relationship| &relationship.id_to_id)
        .any(|pair| &pair.dependent == clip)
}

#[cfg(test)]
mod filter_tests {
    use super::{filter_cards, filter_of_card};

    #[test]
    fn every_filter_card_names_its_look() {
        let cards = filter_cards();
        assert_eq!(cards.len(), cutlass_effects::looks().len());
        for (id, name) in cards {
            assert_eq!(
                filter_of_card(&id).map(|look| look.name.as_str()),
                Some(name)
            );
        }
        assert!(filter_of_card("effect:look/warm").is_none());
        assert_eq!(
            super::filter_categories(),
            cutlass_effects::FILTER_CATEGORIES.to_vec()
        );
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Segment, TimeRange};
    use cutlass_session::Session;

    use super::*;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.push(Segment::new(
            "clip",
            "m",
            TimeRange::new(1_000_000, 2_000_000),
        ));
        draft
    }

    #[test]
    fn every_card_names_a_bundled_effect() {
        let cards = effect_cards();
        assert_eq!(cards.len(), bundled().len());
        assert!(cards.len() >= 40);
        for card in &cards {
            assert_eq!(
                effect_of_card(&card.id).as_deref(),
                Some(card.effect_id.as_str())
            );
        }
        assert_eq!(effect_of_card("effect:recipe/none"), None);
        assert_eq!(effect_of_card("recipe/shake"), None);
        // Only categories with an effect in them, in shelf order.
        let categories = effect_categories();
        assert_eq!(categories.first(), Some(&"Basic"));
        for card in &cards {
            assert!(
                categories.contains(&card.category.as_str()),
                "{}",
                card.name
            );
        }
    }

    #[test]
    fn an_effect_on_a_clip_takes_its_span_binds_and_follows_it() {
        let draft = draft();
        let clip = SegmentId::new("clip");
        let (start, duration) = effect_target(&draft, &clip).unwrap();
        assert_eq!(effect_row_for(&draft, &clip), None);
        let place = EffectPlace {
            track_id: TrackId::new("fx"),
            segment_id: SegmentId::new("shake"),
            material_id: MaterialId::new("m-shake"),
            start,
            duration,
            on: Some(clip.clone()),
        };
        let card = &effect_cards()[0];
        let mut session = Session::open(draft.clone());
        session
            .apply(0, &effect_requests(&draft, &card.effect_id, &place))
            .unwrap();
        let draft = session.draft();
        let effect = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == "shake")
            .unwrap();
        assert_eq!(effect.target, TimeRange::new(1_000_000, 2_000_000));
        assert_eq!(effect.bound_clip(), Some(clip.clone()));
        assert!(
            draft
                .relationships
                .iter()
                .flat_map(|relationship| &relationship.id_to_id)
                .any(|pair| pair.owner == clip && pair.dependent.as_str() == "shake")
        );
        assert_eq!(
            effect_row_for(draft, &clip),
            None,
            "that row is now full there"
        );
        assert_eq!(effect_target(draft, &SegmentId::new("shake")), None);
    }

    #[test]
    fn a_title_and_a_shape_take_an_effect_and_a_sound_does_not() {
        let mut draft = draft();
        for (row, kind, clip) in [
            ("titles", TrackType::Text, "title"),
            ("shapes", TrackType::Sticker, "shape"),
            ("sounds", TrackType::Audio, "sound"),
        ] {
            let mut track = Track::main(row);
            track.kind = kind;
            track
                .segments
                .push(Segment::new(clip, "m", TimeRange::new(500_000, 1_500_000)));
            draft.tracks.push(track);
        }
        let span = Some((Micros(500_000), Micros(1_500_000)));
        assert_eq!(effect_target(&draft, &SegmentId::new("title")), span);
        assert_eq!(effect_target(&draft, &SegmentId::new("shape")), span);
        assert_eq!(effect_target(&draft, &SegmentId::new("sound")), None);
        // Dropped on the title, the effect is bound to it and follows it.
        let place = EffectPlace {
            track_id: TrackId::new("fx"),
            segment_id: SegmentId::new("blur"),
            material_id: MaterialId::new("m-blur"),
            start: Micros(500_000),
            duration: Micros(1_500_000),
            on: Some(SegmentId::new("title")),
        };
        let card = &effect_cards()[0];
        let mut session = Session::open(draft.clone());
        session
            .apply(0, &effect_requests(&draft, &card.effect_id, &place))
            .unwrap();
        let bound = session
            .draft()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == "blur")
            .and_then(Segment::bound_clip);
        assert_eq!(bound, Some(SegmentId::new("title")));
    }

    #[test]
    fn a_clip_lists_the_effects_bound_to_it_by_name_and_no_others() {
        let draft = draft();
        let clip = SegmentId::new("clip");
        assert!(effects_on(&draft, &clip).is_empty());
        let (start, duration) = effect_target(&draft, &clip).unwrap();
        let cards = effect_cards();
        let mut session = Session::open(draft);
        // Two on the clip, each on a row of its own, and one over every
        // picture, which is not the clip's.
        for (index, (row, id, on)) in [
            ("fx1", "first", Some(clip.clone())),
            ("fx2", "second", Some(clip.clone())),
            ("fx3", "all", None),
        ]
        .into_iter()
        .enumerate()
        {
            let place = EffectPlace {
                track_id: TrackId::new(row),
                segment_id: SegmentId::new(id),
                material_id: MaterialId::new(format!("m-{id}")),
                start,
                duration,
                on,
            };
            let draft = session.draft().clone();
            let revision = session.revision();
            session
                .apply(
                    revision,
                    &effect_requests(&draft, &cards[index].effect_id, &place),
                )
                .unwrap();
        }
        let draft = session.draft();
        let listed = effects_on(draft, &clip);
        let mut ids: Vec<&str> = listed
            .iter()
            .map(|effect| effect.segment_id.as_str())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, ["first", "second"]);
        for effect in &listed {
            let card = cards
                .iter()
                .find(|card| card.name == effect.name)
                .map(|card| card.name.as_str());
            assert_eq!(card, Some(effect.name.as_str()));
        }
        assert!(effects_on(draft, &SegmentId::new("gone")).is_empty());
        // A clip whose material is missing still has a name to show.
        let orphan = Segment::new("o", "lost", TimeRange::new(0, 1));
        assert_eq!(effect_title(draft, &orphan), "lost");
    }

    /// An effect row's id and its `(effect, bound clip, start, length)`.
    type EffectRow<'a> = (&'a str, &'a [(&'a str, Option<&'a str>, i64, i64)]);

    /// `draft()` with effect rows.
    fn with_rows(rows: &[EffectRow<'_>]) -> Draft {
        let mut draft = draft();
        for (id, effects) in rows {
            let mut track = Track::main(*id);
            track.kind = TrackType::Effect;
            for (effect, on, start, length) in *effects {
                let mut segment = Segment::new(*effect, "m", TimeRange::new(*start, *length));
                segment.bind_to(on.map(SegmentId::new).as_ref());
                track.segments.push(segment);
            }
            draft.tracks.push(track);
        }
        draft
    }

    #[test]
    fn an_effect_on_a_clip_goes_on_a_row_of_clip_effects_with_room() {
        let clip = SegmentId::new("clip");
        // A row with an effect over every picture is never taken, even
        // with room, so the clip's effect does not show on it.
        let draft = with_rows(&[("all", &[("free", None, 9_000_000, 1)])]);
        assert_eq!(effect_row_for(&draft, &clip), None);
        // A row of clip effects with room is taken; a full one is not.
        let draft = with_rows(&[
            ("full", &[("a", Some("clip"), 1_500_000, 100)]),
            ("room", &[("b", Some("clip"), 3_000_000, 0)]),
        ]);
        assert_eq!(
            effect_row_for(&draft, &clip).as_ref().map(TrackId::as_str),
            Some("room")
        );
        // A locked one is not.
        let mut locked = draft.clone();
        locked.tracks[2].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(effect_row_for(&locked, &clip), None);
    }

    #[test]
    fn an_effect_over_every_picture_passes_over_rows_of_clip_effects() {
        let draft = with_rows(&[
            ("riding", &[("a", Some("clip"), 1_000_000, 2_000_000)]),
            ("all", &[("b", None, 1_000_000, 2_000_000)]),
        ]);
        let at = Micros(5_000_000);
        assert_eq!(
            crate::free_row(&draft, TrackType::Effect, at, EFFECT_LENGTH)
                .as_ref()
                .map(TrackId::as_str),
            Some("all")
        );
        assert_eq!(
            crate::append_placement(&draft, TrackType::Effect).lane,
            crate::AppendLane::Track(TrackId::new("all"))
        );
        let only_riding = with_rows(&[("riding", &[("a", Some("clip"), 1_000_000, 2_000_000)])]);
        assert_eq!(
            crate::free_row(&only_riding, TrackType::Effect, at, EFFECT_LENGTH),
            None
        );
        assert_eq!(
            crate::append_placement(&only_riding, TrackType::Effect).lane,
            crate::AppendLane::New(TrackType::Effect)
        );
    }

    #[test]
    fn effects_by_clip_lists_each_clips_own_like_effects_on() {
        let mut draft = with_rows(&[
            (
                "one",
                &[
                    ("late", Some("clip"), 2_000_000, 1),
                    ("early", Some("clip"), 1_000_000, 1),
                    ("all", None, 0, 1),
                ],
            ),
            (
                "two",
                &[("other", Some("title"), 0, 1), ("gone", Some("lost"), 0, 1)],
            ),
        ]);
        let mut titles = Track::main("titles");
        titles.kind = TrackType::Text;
        titles
            .segments
            .push(Segment::new("title", "m", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(titles);
        let by_clip = effects_by_clip(&draft);
        let ids = |clip: &str| -> Vec<String> {
            by_clip
                .get(&SegmentId::new(clip))
                .into_iter()
                .flatten()
                .map(|effect| effect.segment_id.as_str().to_owned())
                .collect()
        };
        assert_eq!(ids("clip"), ["early", "late"]);
        assert_eq!(ids("title"), ["other"]);
        assert_eq!(
            by_clip.len(),
            2,
            "an effect on a missing clip rides on nothing"
        );
        for clip in ["clip", "title"] {
            assert_eq!(
                by_clip[&SegmentId::new(clip)],
                effects_on(&draft, &SegmentId::new(clip))
            );
        }
        assert!(rides_on_clip(&draft, &draft.tracks[1].segments[0]));
        assert!(!rides_on_clip(&draft, &draft.tracks[1].segments[2]));
        assert!(!rides_on_clip(&draft, &draft.tracks[2].segments[1]));
        assert!(effects_by_clip(&Draft::empty("D")).is_empty());
    }
}
