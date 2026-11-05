//! Free layer: picture clips stacked by their own layer instead of by
//! their tracks.
//!
//! A picture is drawn in the order of its `render_index`, then its track,
//! then its place on the track. With Free layer off every clip keeps its
//! track kind's own layer, so the tracks give the order. With it on,
//! Arrange moves a clip above or below the clips it overlaps in time.

use cutlass_model::{Draft, SegmentId, TrackType};

use super::speed::locate;
use super::{EditError, ensure_unlocked};
use crate::request::Arrange;

/// How far a kind's layers reach either side of its own floor.
const BAND: i32 = 1_000;

fn is_picture(kind: &TrackType) -> bool {
    matches!(
        kind,
        TrackType::Video | TrackType::Sticker | TrackType::Text
    )
}

fn floor(kind: &TrackType) -> i32 {
    kind.clone().render_floor().unwrap_or(0)
}

/// Turn Free layer on or off. Off puts every picture clip back on its
/// track kind's layer, so the tracks give the order again.
pub fn set_free_layer(draft: &mut Draft, on: bool) {
    draft.config.free_layer = on;
    if on {
        return;
    }
    for track in &mut draft.tracks {
        if !is_picture(&track.kind) {
            continue;
        }
        let floor = floor(&track.kind);
        for segment in &mut track.segments {
            segment.render_index = floor;
        }
    }
}

/// Move the clip one step or all the way up or down among the picture
/// clips of its kind it overlaps in time. Needs Free layer on.
pub fn arrange(draft: &mut Draft, segment_id: &SegmentId, step: Arrange) -> Result<(), EditError> {
    if !draft.config.free_layer {
        return Err(EditError::FreeLayer);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if !is_picture(&track.kind) {
        return Err(EditError::FreeLayer);
    }
    let floor = floor(&track.kind);
    let band = (floor - BAND)..=(floor + BAND - 1);
    let own = &track.segments[segment_index];
    let mine = (own.render_index, track_index, segment_index);
    let span = own.target.clone();
    // The clips it could pass, by the key the frame draws them in.
    let mut others: Vec<(i32, usize, usize)> = Vec::new();
    for (index, other_track) in draft.tracks.iter().enumerate() {
        if !is_picture(&other_track.kind) {
            continue;
        }
        for (at, segment) in other_track.segments.iter().enumerate() {
            let overlaps =
                segment.target.start.0 < span.end().0 && span.start.0 < segment.target.end().0;
            if (index, at) != (track_index, segment_index)
                && overlaps
                && band.contains(&segment.render_index)
            {
                others.push((segment.render_index, index, at));
            }
        }
    }
    if others.is_empty() {
        return Ok(());
    }
    // The stack in drawing order, bottom first; the clip moves within it
    // and the stack is numbered again from its lowest layer, so the others
    // keep their order.
    let mut stack = others;
    stack.push(mine);
    stack.sort_unstable();
    let base = stack[0].0;
    let at = stack.iter().position(|key| *key == mine).unwrap_or(0);
    let moved = stack.remove(at);
    let to = match step {
        Arrange::Forward => (at + 1).min(stack.len()),
        Arrange::Backward => at.saturating_sub(1),
        Arrange::Front => stack.len(),
        Arrange::Back => 0,
    };
    if to == at {
        return Ok(());
    }
    stack.insert(to, moved);
    for (place, (_, track, segment)) in stack.into_iter().enumerate() {
        let layer = base.saturating_add(i32::try_from(place).unwrap_or(i32::MAX));
        draft.tracks[track].segments[segment].render_index =
            layer.clamp(*band.start(), *band.end());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Segment, TimeRange, Track};

    use crate::edit::EditError;
    use crate::request::{Arrange, Request};
    use crate::session::Session;

    /// A main clip and two overlays over it, the second above the first.
    fn stacked() -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "m", TimeRange::new(0, 3_000_000)));
        for id in ["low", "high"] {
            let mut track = Track::main(id);
            track.flag = cutlass_model::TrackFlag::SubVideo;
            track
                .segments
                .push(Segment::new(id, "m", TimeRange::new(0, 1_000_000)));
            draft.tracks.push(track);
        }
        Session::open(draft)
    }

    /// The segment ids in the order the frame draws them, bottom first.
    fn order(session: &Session) -> Vec<String> {
        let mut keys = Vec::new();
        for (track, row) in session.draft().tracks.iter().enumerate() {
            for (at, segment) in row.segments.iter().enumerate() {
                keys.push(((segment.render_index, track, at), segment.id.to_string()));
            }
        }
        keys.sort();
        keys.into_iter().map(|(_, id)| id).collect()
    }

    fn apply(session: &mut Session, request: Request) -> Result<u64, EditError> {
        let revision = session.revision();
        session.apply(revision, &[request])
    }

    fn arrange(id: &str, step: Arrange) -> Request {
        Request::Arrange {
            segment_id: id.into(),
            step,
        }
    }

    #[test]
    fn free_layer_lets_a_clip_move_above_and_below_the_ones_it_overlaps() {
        let mut session = stacked();
        assert_eq!(order(&session), ["main", "low", "high"]);
        assert_eq!(
            apply(&mut session, arrange("low", Arrange::Front)),
            Err(EditError::FreeLayer),
            "off, the tracks give the order"
        );
        apply(&mut session, Request::SetFreeLayer { on: true }).unwrap();
        assert!(session.draft().config.free_layer);
        apply(&mut session, arrange("low", Arrange::Front)).unwrap();
        assert_eq!(order(&session), ["main", "high", "low"]);
        apply(&mut session, arrange("main", Arrange::Front)).unwrap();
        assert_eq!(order(&session), ["high", "low", "main"]);
        apply(&mut session, arrange("main", Arrange::Backward)).unwrap();
        assert_eq!(order(&session), ["high", "main", "low"]);
        apply(&mut session, arrange("low", Arrange::Back)).unwrap();
        assert_eq!(order(&session), ["low", "high", "main"]);
        apply(&mut session, arrange("high", Arrange::Forward)).unwrap();
        assert_eq!(order(&session), ["low", "main", "high"]);
        // At the top already, Front changes nothing.
        let before = session.draft().clone();
        apply(&mut session, arrange("high", Arrange::Front)).unwrap();
        assert_eq!(session.draft().tracks, before.tracks);
        // One undo each; off puts the tracks back in charge.
        session.undo().unwrap();
        assert_eq!(order(&session), ["low", "main", "high"]);
        apply(&mut session, Request::SetFreeLayer { on: false }).unwrap();
        assert_eq!(order(&session), ["main", "low", "high"]);
        session.undo().unwrap();
        assert_eq!(order(&session), ["low", "main", "high"]);
    }
}
