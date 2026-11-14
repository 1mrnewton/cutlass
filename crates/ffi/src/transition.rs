//! Transitions on the phone: the shelf with each one's preview video, the
//! cuts between touching clips on the main row, and the requests that put
//! a blend on one, make it longer or shorter, copy it to every cut, or
//! take it off.

use cutlass_editor::{
    FreshIds, TransitionMiss, blend_of, offer_of, transition_categories, transition_everywhere,
    transition_length_request, transition_limits, transition_of_card, transition_offers,
    transition_on_cut,
};
use cutlass_model::{Draft, MaterialId, Micros, SegmentId};
use cutlass_session::Request;

use crate::error::FfiError;

/// A transition the shelf shows. `id` is what
/// [`crate::PhoneHost::set_transition`] takes; `preview` is the file name
/// of its looping preview video in the app bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneTransition {
    pub id: String,
    pub name: String,
    pub category: String,
    pub preview: String,
}

/// Every transition the app ships, in shelf order.
#[must_use]
pub fn phone_transitions() -> Vec<PhoneTransition> {
    transition_offers()
        .iter()
        .map(|offer| PhoneTransition {
            id: offer.card_id(),
            name: offer.name.to_owned(),
            category: offer.category.to_owned(),
            preview: format!("transition-{}.mp4", offer.id),
        })
        .collect()
}

/// The shelf's categories that hold a transition, in shelf order.
#[must_use]
pub fn phone_transition_categories() -> Vec<String> {
    transition_categories()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// A cut between two touching clips on the main row, and its blend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneCut {
    /// The clip before the cut, which names its blend.
    pub outgoing_id: SegmentId,
    pub at: Micros,
    /// The shelf card of its blend; none for a plain cut or a blend the
    /// shelf does not offer.
    pub transition: Option<String>,
    /// The blend's name; empty for a plain cut.
    pub name: String,
    /// How long the blend lasts and the range it can be set to; zero for
    /// a plain cut.
    pub duration: Micros,
    pub shortest: Micros,
    pub longest: Micros,
}

/// Every cut on the main row, earliest first.
#[must_use]
pub fn phone_cuts(draft: &Draft) -> Vec<PhoneCut> {
    let Some(main) = draft.tracks.iter().find(|track| track.is_main()) else {
        return Vec::new();
    };
    let mut ordered: Vec<_> = main
        .segments
        .iter()
        .filter(|segment| segment.target.duration.0 > 0)
        .collect();
    ordered.sort_by_key(|segment| segment.target.start.0);
    ordered
        .windows(2)
        .filter(|pair| pair[1].target.start == pair[0].target.end())
        .map(|pair| {
            let outgoing = pair[0];
            let blend = blend_of(draft, outgoing);
            let (shortest, longest) =
                transition_limits(draft, &outgoing.id).unwrap_or((Micros(0), Micros(0)));
            PhoneCut {
                outgoing_id: outgoing.id.clone(),
                at: outgoing.target.end(),
                transition: blend.and_then(offer_of).map(|offer| offer.card_id()),
                name: blend.map(|blend| blend.name.clone()).unwrap_or_default(),
                duration: blend.map_or(Micros(0), |blend| blend.duration),
                shortest,
                longest,
            }
        })
        .collect()
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

fn missed(miss: TransitionMiss) -> FfiError {
    refused(match miss {
        TransitionMiss::Locked => "the main row is locked",
        TransitionMiss::TooShort => "the clips are too short for a transition",
        TransitionMiss::NotMain => "a transition goes between clips on the main row",
        TransitionMiss::BadScale | TransitionMiss::OffLanes | TransitionMiss::NoCut => {
            "no clip touches that one on the main row"
        }
    })
}

/// The request that puts transition `card_id` on the cut after
/// `outgoing_id`, fitted to the clips.
pub(crate) fn set_request(
    draft: &Draft,
    outgoing_id: &SegmentId,
    card_id: &str,
) -> Result<Request, FfiError> {
    let offer =
        transition_of_card(card_id).ok_or_else(|| refused("that transition is not in the app"))?;
    let material_id = MaterialId::new(FreshIds::new(draft, 0).make("transition"));
    transition_on_cut(draft, offer, outgoing_id, material_id).map_err(missed)
}

/// The request that makes the blend after `outgoing_id` last `duration`,
/// held to its range; none when it lasts that long already.
pub(crate) fn length_request(
    draft: &Draft,
    outgoing_id: &SegmentId,
    duration: Micros,
) -> Result<Option<Request>, FfiError> {
    if transition_limits(draft, outgoing_id).is_none() {
        return Err(refused("that cut has no transition"));
    }
    Ok(transition_length_request(draft, outgoing_id, duration))
}

/// The requests that copy the blend after `outgoing_id` to every other
/// cut of its row.
pub(crate) fn everywhere_requests(
    draft: &Draft,
    outgoing_id: &SegmentId,
) -> Result<Vec<Request>, FfiError> {
    if transition_limits(draft, outgoing_id).is_none() {
        return Err(refused("that cut has no transition"));
    }
    let mut ids = FreshIds::new(draft, 0);
    Ok(transition_everywhere(draft, outgoing_id, &mut |stem| {
        ids.make(stem)
    }))
}

/// The request that takes the blend off the cut after `outgoing_id`.
pub(crate) fn remove_request(draft: &Draft, outgoing_id: &SegmentId) -> Result<Request, FfiError> {
    let has = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == outgoing_id)
        .and_then(|segment| blend_of(draft, segment))
        .is_some();
    if !has {
        return Err(refused("that cut has no transition"));
    }
    Ok(Request::RemoveTransition {
        segment_id: outgoing_id.clone(),
    })
}
