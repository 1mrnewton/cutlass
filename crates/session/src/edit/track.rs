use cutlass_model::{Draft, TrackAttribute, TrackId};

use super::EditError;

pub fn set_track_attribute(
    draft: &mut Draft,
    track_id: &TrackId,
    bit: i32,
    on: bool,
) -> Result<(), EditError> {
    if !known(bit) {
        return Err(EditError::UnknownAttribute);
    }
    {
        let Some(track) = draft.tracks.iter_mut().find(|track| &track.id == track_id) else {
            return Err(EditError::MissingTrack);
        };
        if on {
            track.attribute.0 |= bit;
        } else {
            track.attribute.0 &= !bit;
        }
    }
    if bit == TrackAttribute::SOLO {
        apply_solo(draft);
    }
    Ok(())
}

fn known(bit: i32) -> bool {
    matches!(
        bit,
        TrackAttribute::MUTE
            | TrackAttribute::HIDE
            | TrackAttribute::LOCK
            | TrackAttribute::SOLO
            | TrackAttribute::MUTE_BY_SOLO
    )
}

/// Solo is not a gain. Other tracks receive muted-by-solo. A solo track does not.
fn apply_solo(draft: &mut Draft) {
    let any_solo = draft
        .tracks
        .iter()
        .any(|track| track.attribute.contains(TrackAttribute::SOLO));
    for track in &mut draft.tracks {
        let muted = any_solo && !track.attribute.contains(TrackAttribute::SOLO);
        if muted {
            track.attribute.0 |= TrackAttribute::MUTE_BY_SOLO;
        } else {
            track.attribute.0 &= !TrackAttribute::MUTE_BY_SOLO;
        }
    }
}
