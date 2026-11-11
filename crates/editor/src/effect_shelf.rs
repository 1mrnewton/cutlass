//! The effects shelf: the bundled effects as cards, and the requests that
//! place one on an effect row or over one clip.
//!
//! An effect on a row runs over every picture below it. An effect dropped
//! on a picture clip takes that clip's span, is bound to it so it runs over
//! that clip alone, and is linked to it so it moves with it. Every screen
//! places an effect through here.

use cutlass_effects::{bundled, bundled_recipe};
use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, Segment, SegmentId, TrackId, TrackType,
};
use cutlass_session::Request;

use crate::append::free_row;

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

/// The bundled effects, in shelf order.
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

/// The effect a card id names, when it is one the app ships.
#[must_use]
pub fn effect_of_card(card_id: &str) -> Option<String> {
    let effect_id = card_id.strip_prefix(EFFECT_CARD_PREFIX)?;
    bundled_recipe(effect_id).map(|recipe| recipe.effect_id())
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

/// The span of a picture clip an effect can go on: a clip on a video row
/// with a positive length.
#[must_use]
pub fn effect_target(draft: &Draft, clip: &SegmentId) -> Option<(Micros, Micros)> {
    draft
        .tracks
        .iter()
        .filter(|track| track.kind == TrackType::Video)
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == clip && segment.target.duration.0 > 0)
        .map(|segment| (segment.target.start, segment.target.duration))
}

/// The effect row a clip-long effect over `clip` goes on: the first one
/// with room for the clip's span, or `None` when a new row must open.
#[must_use]
pub fn effect_row_for(draft: &Draft, clip: &SegmentId) -> Option<TrackId> {
    let (start, duration) = effect_target(draft, clip)?;
    free_row(draft, TrackType::Effect, start, duration)
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
}
