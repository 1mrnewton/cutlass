//! Text clips on the phone: add one at the playhead, change its words or
//! its face, and read them back. Faces come from the folders the shell
//! names; the draft keeps a face's file name.

use cutlass_model::{Draft, MaterialId, MaterialRef, TrackType};
use cutlass_session::{Micros, Request, SegmentId};

use crate::media::{fresh, room};

/// How long a new text clip lasts.
pub const NEW_TEXT_LENGTH: Micros = Micros(3_000_000);

/// A face the phone can set text in. `name` is what the draft stores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneFont {
    pub name: String,
    pub label: String,
}

/// What a text clip says and the face it says it in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneText {
    pub content: String,
    pub face: String,
}

/// The faces in `faces`, each a name and its label for people.
#[must_use]
pub fn fonts(faces: Vec<(String, String)>) -> Vec<PhoneFont> {
    faces
        .into_iter()
        .map(|(name, label)| PhoneFont { name, label })
        .collect()
}

/// The request that adds `content` in `face` at `at` for
/// [`NEW_TEXT_LENGTH`], and the new clip's id. It goes on the first text
/// row with room, else on a new one.
pub(crate) fn add_request(
    draft: &Draft,
    content: &str,
    face: &str,
    at: Micros,
) -> (Request, SegmentId) {
    let (track_id, segment_id, taken) = room(draft, TrackType::Text, at, NEW_TEXT_LENGTH);
    let at = Micros(at.0.max(0));
    let material_id = MaterialId::new(fresh(&taken, "text"));
    let request = Request::AddText {
        track_id,
        segment_id: segment_id.clone(),
        material_id,
        start: at,
        duration: NEW_TEXT_LENGTH,
        content: content.to_owned(),
        face: face.to_owned(),
    };
    (request, segment_id)
}

/// What `segment_id` says, or `None` when it is not a text clip.
#[must_use]
pub fn text_of(draft: &Draft, segment_id: &SegmentId) -> Option<PhoneText> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    match draft.materials.get(segment.material_id.as_str())? {
        MaterialRef::Text(text) => Some(PhoneText {
            content: text.content.clone(),
            face: text.font_path.clone(),
        }),
        _ => None,
    }
}
