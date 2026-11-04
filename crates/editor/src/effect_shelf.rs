//! The effects shelf: the bundled effects as cards, and the requests that
//! place one on an effect row or over one clip.
//!
//! An effect on a row runs over every picture below it. An effect dropped
//! on a picture clip takes that clip's span, is bound to it so it runs over
//! that clip alone, and is linked to it so it moves with it. Every screen
//! places an effect through here.

use cutlass_effects::{bundled, bundled_recipe};
use cutlass_model::{Draft, MaterialId, Micros, SegmentId, TrackId, TrackType};
use cutlass_session::Request;

use crate::append::free_row;

/// How long an effect added at the playhead or on a row lasts.
pub const EFFECT_LENGTH: Micros = Micros(3_000_000);

/// A card's id starts with this, so a drop can tell an effect from a file.
pub const EFFECT_CARD_PREFIX: &str = "effect:";

/// One card on the effects shelf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectCard {
    /// The library card id.
    pub id: String,
    /// What [`Request::AddEffect`] takes, such as `recipe/shake`.
    pub effect_id: String,
    pub name: String,
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
            }
        })
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
    fn ten_cards_each_naming_a_bundled_effect() {
        let cards = effect_cards();
        assert_eq!(cards.len(), 10);
        for card in &cards {
            assert_eq!(
                effect_of_card(&card.id).as_deref(),
                Some(card.effect_id.as_str())
            );
        }
        assert_eq!(effect_of_card("effect:recipe/none"), None);
        assert_eq!(effect_of_card("recipe/shake"), None);
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
}
