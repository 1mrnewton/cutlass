//! Filters, adjust sliders and effects over one clip, for the phone.
//!
//! A filter is a layer on a filter row that runs over every picture below
//! it. Picked for a clip, it takes the clip's span and is linked to the
//! clip so it moves with it; the clip keeps one such filter, and picking
//! another swaps it. Picked with nothing selected, it lasts
//! [`cutlass_editor::EFFECT_LENGTH`] from the playhead. Adjust sliders sit
//! on a picture clip itself, or on an adjustment layer.

use cutlass_editor::{
    EFFECT_LENGTH, EffectPlace, FreshIds, effect_requests, effect_row_for, effect_target, free_row,
};
use cutlass_effects::{ADJUSTMENT, AdjustKey, FILTER, looks, segment_adjust};
use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, Segment, SegmentId, TrackId, TrackType,
};
use cutlass_session::Request;

use crate::clip::{locate, unlocked};
use crate::error::FfiError;

/// A filter the picker shows. `id` is what the filter tools take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneFilter {
    pub id: String,
    pub name: String,
    /// One of the filter categories, for the picker's tabs.
    pub category: String,
}

/// The filters that ship in the app, in picker order.
#[must_use]
pub fn phone_filters() -> Vec<PhoneFilter> {
    looks()
        .iter()
        .map(|look| PhoneFilter {
            id: look.effect_id(),
            name: look.name.clone(),
            category: look.category.clone(),
        })
        .collect()
}

/// One adjust slider and where it stands.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneAdjust {
    /// What [`crate::PhoneHost::set_adjust`] takes, such as `temperature`.
    pub key: String,
    pub label: String,
    /// -1, or 0 for a slider that only adds.
    pub min: f64,
    pub value: f64,
}

/// The filter a picture clip keeps: the filter layer linked to it, its
/// filter id, and its strength from 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneClipFilter {
    pub segment_id: SegmentId,
    pub filter_id: String,
    pub strength: f64,
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

/// Every adjust slider of `segment_id`, a picture clip or an adjustment
/// layer. `None` for any other clip.
#[must_use]
pub fn adjust_sliders(draft: &Draft, segment_id: &SegmentId) -> Option<Vec<PhoneAdjust>> {
    let (segment, track) = locate(draft, segment_id)?;
    if !takes_adjust(track.kind.clone()) {
        return None;
    }
    let adjust = segment_adjust(draft, segment);
    Some(
        AdjustKey::ALL
            .iter()
            .map(|key| PhoneAdjust {
                key: key.kind().to_owned(),
                label: key.label().to_owned(),
                min: f64::from(key.min()),
                value: f64::from(adjust.get(*key)),
            })
            .collect(),
    )
}

fn takes_adjust(kind: TrackType) -> bool {
    matches!(kind, TrackType::Video | TrackType::Adjust)
}

/// The filter layers linked to `clip` as its dependents.
fn linked_filters<'a>(draft: &'a Draft, clip: &SegmentId) -> Vec<&'a Segment> {
    let dependents: Vec<&SegmentId> = draft
        .relationships
        .iter()
        .filter(|relationship| relationship.kind == "linkage")
        .flat_map(|relationship| &relationship.id_to_id)
        .filter(|pair| &pair.owner == clip)
        .map(|pair| &pair.dependent)
        .collect();
    draft
        .tracks
        .iter()
        .filter(|track| track.kind == TrackType::Filter)
        .flat_map(|track| &track.segments)
        .filter(|segment| dependents.contains(&&segment.id))
        .collect()
}

/// The filter material a filter layer uses: its filter id and strength.
fn filter_of(draft: &Draft, segment: &Segment) -> Option<(String, f64)> {
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Effect(effect)) if effect.kind == FILTER => {
            Some((effect.effect_id.clone(), effect.value.unwrap_or(1.0)))
        }
        _ => None,
    }
}

/// The filter `clip` keeps, if any. A filter layer itself answers for
/// its own filter.
#[must_use]
pub fn clip_filter(draft: &Draft, clip: &SegmentId) -> Option<PhoneClipFilter> {
    let (segment, track) = locate(draft, clip)?;
    let layer = if track.kind == TrackType::Filter {
        segment
    } else {
        linked_filters(draft, clip).into_iter().next()?
    };
    let (filter_id, strength) = filter_of(draft, layer)?;
    Some(PhoneClipFilter {
        segment_id: layer.id.clone(),
        filter_id,
        strength,
    })
}

/// The requests that give picture clip `clip` the filter `filter_id` over
/// its span, linked to it, taking off the filter it kept; an empty id
/// only takes it off. Also the new layer's id.
pub(crate) fn clip_filter_requests(
    draft: &Draft,
    clip: &SegmentId,
    filter_id: &str,
) -> Result<(Vec<Request>, Option<SegmentId>), FfiError> {
    let (segment, track) = unlocked(draft, clip)?;
    let mut requests = Vec::new();
    let mut freed: Option<TrackId> = None;
    match track.kind {
        TrackType::Filter => {
            // A filter layer picked on its own swaps in place.
            requests.push(Request::Remove {
                segment_id: clip.clone(),
            });
            freed = Some(track.id.clone());
        }
        TrackType::Video => {
            for old in linked_filters(draft, clip) {
                requests.push(Request::Remove {
                    segment_id: old.id.clone(),
                });
                if freed.is_none() {
                    freed = row_of(draft, &old.id);
                }
            }
        }
        _ => return Err(refused("a filter belongs on a picture")),
    }
    if filter_id.is_empty() {
        return Ok((requests, None));
    }
    if cutlass_effects::bundled_look(filter_id).is_none() {
        return Err(refused("that filter is not in the app"));
    }
    let start = segment.target.start;
    let duration = segment.target.duration;
    let mut ids = FreshIds::new(draft, 0);
    let track_id = freed
        .or_else(|| free_row(draft, TrackType::Filter, start, duration))
        .unwrap_or_else(|| TrackId::new(ids.make("track")));
    let segment_id = SegmentId::new(ids.make("seg"));
    requests.push(Request::AddFilter {
        track_id,
        segment_id: segment_id.clone(),
        material_id: MaterialId::new(ids.make("filter")),
        start,
        duration,
        filter_id: filter_id.to_owned(),
    });
    if track.kind == TrackType::Video {
        requests.push(Request::Link {
            owner: clip.clone(),
            dependent: segment_id.clone(),
        });
    }
    Ok((requests, Some(segment_id)))
}

fn row_of(draft: &Draft, segment_id: &SegmentId) -> Option<TrackId> {
    locate(draft, segment_id).map(|(_, track)| track.id.clone())
}

/// The request that puts a filter layer `filter_id` at `at` for
/// [`EFFECT_LENGTH`], and its id.
pub(crate) fn filter_at_request(
    draft: &Draft,
    filter_id: &str,
    at: Micros,
) -> Result<(Request, SegmentId), FfiError> {
    if cutlass_effects::bundled_look(filter_id).is_none() {
        return Err(refused("that filter is not in the app"));
    }
    let start = Micros(at.0.max(0));
    let mut ids = FreshIds::new(draft, 0);
    let track_id = free_row(draft, TrackType::Filter, start, EFFECT_LENGTH)
        .unwrap_or_else(|| TrackId::new(ids.make("track")));
    let segment_id = SegmentId::new(ids.make("seg"));
    Ok((
        Request::AddFilter {
            track_id,
            segment_id: segment_id.clone(),
            material_id: MaterialId::new(ids.make("filter")),
            start,
            duration: EFFECT_LENGTH,
            filter_id: filter_id.to_owned(),
        },
        segment_id,
    ))
}

/// The request that puts an adjustment layer at `at` for
/// [`EFFECT_LENGTH`], and its id.
pub(crate) fn adjustment_request(draft: &Draft, at: Micros) -> (Request, SegmentId) {
    let start = Micros(at.0.max(0));
    let mut ids = FreshIds::new(draft, 0);
    let track_id = free_row(draft, TrackType::Adjust, start, EFFECT_LENGTH)
        .unwrap_or_else(|| TrackId::new(ids.make("track")));
    let segment_id = SegmentId::new(ids.make("seg"));
    (
        Request::AddAdjustment {
            track_id,
            segment_id: segment_id.clone(),
            material_id: MaterialId::new(ids.make(ADJUSTMENT)),
            start,
            duration: EFFECT_LENGTH,
        },
        segment_id,
    )
}

/// The request that sets slider `key` of `segment_id` to `value`.
pub(crate) fn adjust_request(
    draft: &Draft,
    segment_id: &SegmentId,
    key: &str,
    value: f64,
) -> Result<Request, FfiError> {
    let (_, track) = unlocked(draft, segment_id)?;
    if !takes_adjust(track.kind.clone()) {
        return Err(refused("adjust belongs on a picture or an adjustment"));
    }
    let slider = AdjustKey::from_kind(key).ok_or_else(|| refused("that is not a slider"))?;
    if !value.is_finite() || value < f64::from(slider.min()) || value > 1.0 {
        return Err(refused("the slider does not reach that far"));
    }
    Ok(Request::SetAdjust {
        segment_id: segment_id.clone(),
        material_id: MaterialId::new(FreshIds::new(draft, 0).make("adjust")),
        key: key.to_owned(),
        value,
    })
}

/// The requests that run effect `effect_id` over picture clip `clip`
/// alone for its span, and the new effect clip's id.
pub(crate) fn clip_effect_requests(
    draft: &Draft,
    clip: &SegmentId,
    effect_id: &str,
) -> Result<(Vec<Request>, SegmentId), FfiError> {
    unlocked(draft, clip)?;
    let (start, duration) =
        effect_target(draft, clip).ok_or_else(|| refused("an effect belongs on a picture"))?;
    if cutlass_effects::bundled_recipe(effect_id).is_none() {
        return Err(refused("that effect is not in the app"));
    }
    let mut ids = FreshIds::new(draft, 0);
    let track_id = effect_row_for(draft, clip).unwrap_or_else(|| TrackId::new(ids.make("track")));
    let place = EffectPlace {
        track_id,
        segment_id: SegmentId::new(ids.make("seg")),
        material_id: MaterialId::new(ids.make("effect")),
        start,
        duration,
        on: Some(clip.clone()),
    };
    Ok((effect_requests(draft, effect_id, &place), place.segment_id))
}
