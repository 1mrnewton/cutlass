//! Effects on the phone: the bundled list with each one's preview video,
//! and adding one at the playhead.

use cutlass_effects::bundled;
use cutlass_model::{Draft, MaterialId, TrackType};
use cutlass_session::{Micros, Request, SegmentId};

use crate::media::{fresh, room};

/// How long a new effect clip lasts.
pub const NEW_EFFECT_LENGTH: Micros = Micros(3_000_000);

/// An effect the picker shows. `id` is what [`crate::PhoneHost::add_effect`]
/// takes; `preview` is the file name of its looping preview video in the
/// app bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneEffect {
    pub id: String,
    pub name: String,
    pub preview: String,
}

/// The effects that ship in the app, in picker order.
#[must_use]
pub fn phone_effects() -> Vec<PhoneEffect> {
    bundled()
        .iter()
        .map(|recipe| PhoneEffect {
            id: recipe.effect_id(),
            name: recipe.name.clone(),
            preview: format!("{}.mp4", recipe.id),
        })
        .collect()
}

/// The request that adds `effect_id` at `at` for [`NEW_EFFECT_LENGTH`], and
/// the new clip's id. It goes on the first effect row with room, else on a
/// new one.
pub(crate) fn add_request(draft: &Draft, effect_id: &str, at: Micros) -> (Request, SegmentId) {
    let (track_id, segment_id, taken) = room(draft, TrackType::Effect, at, NEW_EFFECT_LENGTH);
    let request = Request::AddEffect {
        track_id,
        segment_id: segment_id.clone(),
        material_id: MaterialId::new(fresh(&taken, "effect")),
        start: Micros(at.0.max(0)),
        duration: NEW_EFFECT_LENGTH,
        effect_id: effect_id.to_owned(),
    };
    (request, segment_id)
}
