//! A clip's list of masks: add one, change one, take one off, or move one
//! (see `cutlass_effects::mask`).
//!
//! A place in the list counts the clip's records of that kind, clip masks
//! or colour masks, in the order the clip lists them. The first mask keys
//! under the plain mask key names and each later one under names marked
//! with its record, so the keys follow a mask that moves in the list, that
//! another clip stops sharing, or that comes first when the one before it
//! goes.

use cutlass_effects::{COLOR_MASK, MASK, Mask, MaskArt, art_fits};
use cutlass_model::{
    Draft, MASK_KEYS, MaterialId, MaterialRef, RawMaterial, Segment, SegmentId, TrackType, mask_key,
};

use super::{EditError, ensure_unlocked};

/// The most masks of one kind a clip lists.
pub const MAX_MASKS: usize = 8;

/// The record kind and the tracks it may sit on, for a clip mask or a
/// colour mask.
fn kind(color: bool) -> (&'static str, &'static [TrackType]) {
    if color {
        (
            COLOR_MASK,
            &[
                TrackType::Video,
                TrackType::Sticker,
                TrackType::Adjust,
                TrackType::Effect,
            ],
        )
    } else {
        (MASK, &[TrackType::Video, TrackType::Sticker])
    }
}

/// The record the keys of the mask of `kind` at `index`, record `id`, are
/// named for: none for a clip's first clip mask, which keys under the plain
/// names; a colour mask always keys under its record.
pub(super) fn key_record<'a>(kind: &str, index: usize, id: &'a str) -> Option<&'a str> {
    (kind != MASK || index > 0).then_some(id)
}

/// Put `mask` at the end of `segment_id`'s masks, as record `material_id`,
/// drawn from `art` when it is a text, brush or pen mask.
pub fn add_mask(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    mask: &Mask,
    art: Option<&MaskArt>,
    color: bool,
) -> Result<(), EditError> {
    let (kind, tracks) = kind(color);
    let (track, at) = target(draft, segment_id, tracks)?;
    if !mask.is_valid() || !art_fits(mask.shape, art) {
        return Err(EditError::Mask);
    }
    if records(draft, &draft.tracks[track].segments[at], kind).len() >= MAX_MASKS {
        return Err(EditError::MaskList);
    }
    fresh(draft, material_id)?;
    draft.materials.common_mask.push(RawMaterial {
        id: material_id.clone(),
        kind: kind.to_owned(),
        fields: fields(mask, art),
    });
    draft.tracks[track].segments[at]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

/// Make the mask at `index` on `segment_id` into `mask`, drawn from `art`;
/// with no `art` a drawn mask keeps what it was drawn from while its shape
/// stays. A record another clip shares is left to that clip; this clip
/// gets its own, named `material_id`, in the same place, and its keys come
/// along.
pub fn edit_mask(
    draft: &mut Draft,
    segment_id: &SegmentId,
    index: usize,
    material_id: &MaterialId,
    mask: &Mask,
    art: Option<&MaskArt>,
    color: bool,
) -> Result<(), EditError> {
    let (kind, tracks) = kind(color);
    let (track, at) = target(draft, segment_id, tracks)?;
    if !mask.is_valid() || !art_fits(mask.shape, art) {
        return Err(EditError::Mask);
    }
    let listed = records(draft, &draft.tracks[track].segments[at], kind);
    let Some(id) = listed.get(index).cloned() else {
        return Err(EditError::MaskList);
    };
    let kept = art.cloned().or_else(|| {
        draft
            .materials
            .common_mask
            .iter()
            .find(|raw| raw.id == id)
            .and_then(|raw| MaskArt::from_fields(&raw.fields))
            .filter(|art| art.shape() == mask.shape)
    });
    let written = fields(mask, kept.as_ref());
    if !shared(draft, &id) {
        if let Some(raw) = draft
            .materials
            .common_mask
            .iter_mut()
            .find(|raw| raw.id == id)
        {
            raw.fields = written;
        }
        return Ok(());
    }
    fresh(draft, material_id)?;
    draft.materials.common_mask.push(RawMaterial {
        id: material_id.clone(),
        kind: kind.to_owned(),
        fields: written,
    });
    let segment = &mut draft.tracks[track].segments[at];
    if let Some(slot) = segment
        .extra_material_refs
        .iter_mut()
        .find(|other| **other == id)
    {
        *slot = material_id.clone();
    }
    if let Some(old) = key_record(kind, index, id.as_str()) {
        rename_keys(segment, Some(old), Some(material_id.as_str()));
    }
    Ok(())
}

/// Take the mask at `index` off `segment_id`, with its keys. When the first
/// goes, the next becomes first and takes the plain key names.
pub fn remove_mask(
    draft: &mut Draft,
    segment_id: &SegmentId,
    index: usize,
    color: bool,
) -> Result<(), EditError> {
    let (kind, tracks) = kind(color);
    let (track, at) = target(draft, segment_id, tracks)?;
    let listed = records(draft, &draft.tracks[track].segments[at], kind);
    let Some(id) = listed.get(index).cloned() else {
        return Err(EditError::MaskList);
    };
    let alone = !shared(draft, &id);
    let segment = &mut draft.tracks[track].segments[at];
    segment.extra_material_refs.retain(|other| *other != id);
    drop_keys(segment, key_record(kind, index, id.as_str()));
    if kind == MASK
        && index == 0
        && let Some(next) = listed.get(1)
    {
        rename_keys(segment, Some(next.as_str()), None);
    }
    if alone {
        draft.materials.common_mask.retain(|raw| raw.id != id);
    }
    Ok(())
}

/// Move the mask at `from` on `segment_id` to place `to`, the others
/// keeping their order.
pub fn move_mask(
    draft: &mut Draft,
    segment_id: &SegmentId,
    from: usize,
    to: usize,
    color: bool,
) -> Result<(), EditError> {
    let (kind, tracks) = kind(color);
    let (track, at) = target(draft, segment_id, tracks)?;
    let mut listed = records(draft, &draft.tracks[track].segments[at], kind);
    if from >= listed.len() || to >= listed.len() {
        return Err(EditError::MaskList);
    }
    if from == to {
        return Ok(());
    }
    let old_first = listed[0].clone();
    let moving = listed.remove(from);
    listed.insert(to, moving);
    let segment = &mut draft.tracks[track].segments[at];
    // The clip's records of this kind take their new order, in the places
    // the kind held among the clip's records.
    let mut order = listed.iter();
    for slot in &mut segment.extra_material_refs {
        if listed.contains(slot)
            && let Some(id) = order.next()
        {
            *slot = id.clone();
        }
    }
    let new_first = &listed[0];
    if kind == MASK && *new_first != old_first {
        // The old first's plain keys take its name, then the new first's
        // named keys become the plain ones.
        rename_keys(segment, None, Some(old_first.as_str()));
        rename_keys(segment, Some(new_first.as_str()), None);
    }
    Ok(())
}

/// How many masks of its kind `segment_id` lists, once it is known the
/// clip may take them.
pub(super) fn count(
    draft: &Draft,
    segment_id: &SegmentId,
    color: bool,
) -> Result<usize, EditError> {
    let (kind, tracks) = kind(color);
    let (track, at) = target(draft, segment_id, tracks)?;
    Ok(records(draft, &draft.tracks[track].segments[at], kind).len())
}

/// The clip's records of `kind`, in its order.
pub(super) fn records(draft: &Draft, segment: &Segment, kind: &str) -> Vec<MaterialId> {
    segment
        .extra_material_refs
        .iter()
        .filter(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Other(raw)) if raw.kind == kind
            )
        })
        .cloned()
        .collect()
}

fn target(
    draft: &Draft,
    segment_id: &SegmentId,
    tracks: &[TrackType],
) -> Result<(usize, usize), EditError> {
    let (track, at) = super::locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track].attribute)?;
    if !tracks.contains(&draft.tracks[track].kind) {
        return Err(EditError::EffectTarget);
    }
    Ok((track, at))
}

/// A record's fields for `mask` drawn from `art`.
fn fields(mask: &Mask, art: Option<&MaskArt>) -> serde_json::Map<String, serde_json::Value> {
    let mut fields = mask.to_fields();
    if let Some(art) = art {
        art.write(&mut fields);
    }
    fields
}

fn shared(draft: &Draft, id: &MaterialId) -> bool {
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .filter(|segment| segment.extra_material_refs.contains(id))
        .count()
        > 1
}

fn fresh(draft: &Draft, material_id: &MaterialId) -> Result<(), EditError> {
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        Err(EditError::DuplicateMaterial)
    } else {
        Ok(())
    }
}

/// Rename the mask keys of record `from` to record `to` (`None` for the
/// plain names), dropping any keys `to` already had.
fn rename_keys(segment: &mut Segment, from: Option<&str>, to: Option<&str>) {
    drop_keys(segment, to);
    for name in MASK_KEYS {
        let (old, new) = (mask_key(name, from), mask_key(name, to));
        for group in &mut segment.common_keyframes {
            if group.property_type == old {
                group.property_type.clone_from(&new);
            }
        }
    }
}

/// Take off the mask keys of record `record` (`None` for the plain names).
fn drop_keys(segment: &mut Segment, record: Option<&str>) {
    let names = MASK_KEYS.map(|name| mask_key(name, record));
    segment
        .common_keyframes
        .retain(|group| !names.contains(&group.property_type));
}

#[cfg(test)]
mod tests;
