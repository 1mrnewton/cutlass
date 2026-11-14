//! Stickers, shapes and freehand drawing for the phone: the shelves, and
//! the requests that put one at the playhead on the first sticker row with
//! room, else on a new one.

use cutlass_editor::{
    FreshIds, STICKER_LENGTH, StickerFile, StickerPlace, free_row, freehand, shape_cards,
    shape_of_card, sticker_length, sticker_requests,
};
use cutlass_model::{Draft, MaterialId, Micros, SegmentId, TrackId, TrackType};
use cutlass_session::Request;

use crate::error::FfiError;
use crate::preview::Picture;

/// A sticker the shelf shows. `id` is what [`crate::PhoneHost::add_sticker`]
/// takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneSticker {
    pub id: String,
    pub name: String,
    /// It plays round as an animation.
    pub animated: bool,
}

/// Every sticker the app ships, in shelf order.
#[must_use]
pub fn phone_stickers() -> Vec<PhoneSticker> {
    cutlass_stickers::STICKERS
        .iter()
        .map(|sticker| PhoneSticker {
            id: sticker.id.to_owned(),
            name: sticker.name.to_owned(),
            animated: sticker.kind() == cutlass_stickers::Kind::Animation,
        })
        .collect()
}

/// Sticker `id` `side` pixels square for its shelf tile; an animation
/// shows its middle frame.
#[must_use]
pub fn sticker_poster(id: &str, side: u32) -> Option<Picture> {
    let side = side.clamp(1, 512);
    let rgba = cutlass_stickers::sticker(id)?.rgba(side).ok()?;
    Some(Picture {
        width: side,
        height: side,
        rgba,
    })
}

/// A shape the shelf shows. `id` is what [`crate::PhoneHost::add_shape`]
/// takes; `name` is the shape's own, such as `Circle`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneShape {
    pub id: String,
    pub name: String,
}

/// Every shape, in shelf order.
#[must_use]
pub fn phone_shapes() -> Vec<PhoneShape> {
    shape_cards()
        .into_iter()
        .map(|(id, kind)| PhoneShape {
            id,
            name: kind.name().to_owned(),
        })
        .collect()
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

/// The first sticker row free from `start` for `length`, or a new one.
fn sticker_row(
    draft: &Draft,
    ids: &mut FreshIds<'_>,
    start: Micros,
    length: Micros,
) -> (TrackId, bool) {
    free_row(draft, TrackType::Sticker, start, length).map_or_else(
        || (TrackId::new(ids.make("track")), true),
        |row| (row, false),
    )
}

/// The requests that place sticker `id` at `at` and its clip's id. Its
/// file is written into the project at `project` the first time.
pub(crate) fn sticker_at_requests(
    draft: &Draft,
    project: &std::path::Path,
    id: &str,
    at: Micros,
    linkage: bool,
) -> Result<(Vec<Request>, SegmentId), FfiError> {
    let sticker =
        cutlass_stickers::sticker(id).ok_or_else(|| refused("that sticker is not in the app"))?;
    let stored = sticker
        .write_into(project)
        .map_err(|reason| FfiError::Refused { reason })?;
    let (width, height) = sticker
        .size()
        .map_err(|reason| FfiError::Refused { reason })?;
    let start = Micros(at.0.max(0));
    let length = sticker_length(sticker.seconds());
    let mut ids = FreshIds::new(draft, 0);
    let (track_id, new_lane) = sticker_row(draft, &mut ids, start, length);
    let place = StickerPlace {
        track_id,
        new_lane,
        segment_id: SegmentId::new(ids.make("seg")),
        material_id: MaterialId::new(ids.make("mat")),
        start,
        length,
    };
    let file = StickerFile {
        stored: &stored,
        width,
        height,
    };
    Ok((
        sticker_requests(draft, file, &place, linkage),
        place.segment_id,
    ))
}

/// The request that places shape `card_id` at `at`, and its clip's id.
pub(crate) fn shape_at_request(
    draft: &Draft,
    card_id: &str,
    at: Micros,
) -> Result<(Vec<Request>, SegmentId), FfiError> {
    let kind = shape_of_card(card_id).ok_or_else(|| refused("that shape is not on the shelf"))?;
    let start = Micros(at.0.max(0));
    let mut ids = FreshIds::new(draft, 0);
    // A shape opens its sticker row itself when the row is new.
    let (track_id, _) = sticker_row(draft, &mut ids, start, STICKER_LENGTH);
    let segment_id = SegmentId::new(ids.make("seg"));
    let requests = vec![Request::AddShape {
        track_id,
        segment_id: segment_id.clone(),
        material_id: MaterialId::new(ids.make("shape")),
        start,
        duration: STICKER_LENGTH,
        kind,
    }];
    Ok((requests, segment_id))
}

/// The request that places a freehand stroke through `trace`, canvas
/// pixels from the top left, y down, at `at`, and its clip's id.
pub(crate) fn stroke_request(
    draft: &Draft,
    trace: &[[f64; 2]],
    at: Micros,
) -> Result<(Vec<Request>, SegmentId), FfiError> {
    let canvas = (draft.canvas_config.width, draft.canvas_config.height);
    let points = freehand(trace, canvas).ok_or_else(|| refused("the stroke is too short"))?;
    let start = Micros(at.0.max(0));
    let mut ids = FreshIds::new(draft, 0);
    // A shape opens its sticker row itself when the row is new.
    let (track_id, _) = sticker_row(draft, &mut ids, start, STICKER_LENGTH);
    let segment_id = SegmentId::new(ids.make("seg"));
    let requests = vec![Request::DrawShape {
        track_id,
        segment_id: segment_id.clone(),
        material_id: MaterialId::new(ids.make("shape")),
        start,
        duration: STICKER_LENGTH,
        points,
        closed: false,
    }];
    Ok((requests, segment_id))
}
