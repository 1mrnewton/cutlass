use cutlass_model::{Draft, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store the picture's position. `x` and `y` are fractions of half the canvas.
/// Positive `y` is up. The clip stays where it is on the timeline.
pub fn set_position(
    draft: &mut Draft,
    segment_id: &SegmentId,
    x: f64,
    y: f64,
) -> Result<(), EditError> {
    if !x.is_finite() || !y.is_finite() {
        return Err(EditError::Position);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let clip = &mut draft.tracks[track_index].segments[segment_index].clip;
    clip.transform.x = x;
    clip.transform.y = y;
    Ok(())
}

/// Store the picture's scale. Uniform scale and both axes take `scale`.
/// The uniform switch stays as it was.
pub fn set_scale(draft: &mut Draft, segment_id: &SegmentId, scale: f64) -> Result<(), EditError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(EditError::Scale);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    segment.uniform_scale.value = scale;
    segment.clip.scale.x = scale;
    segment.clip.scale.y = scale;
    Ok(())
}

/// Store the picture's opacity. `1` is solid and `0` is clear.
/// Position, scale, rotation, and flip stay as they were.
pub fn set_opacity(
    draft: &mut Draft,
    segment_id: &SegmentId,
    opacity: f64,
) -> Result<(), EditError> {
    if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
        return Err(EditError::Opacity);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index].clip.alpha = opacity;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Segment, TimeRange, TrackAttribute, TrackId};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> (Session, TrackId) {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(10, 1_000_000));
        segment.volume = 0.4;
        segment.clip.rotation = 90.0;
        segment.clip.flip.horizontal = true;
        segment.uniform_scale.on = true;
        session_with(draft, segment)
    }

    fn session_with(draft: Draft, segment: Segment) -> (Session, TrackId) {
        let track = draft.tracks[0].id.clone();
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        (session, track)
    }

    #[test]
    fn a_position_moves_the_picture_and_leaves_the_clip_in_place() {
        let (mut session, _) = open();
        session
            .apply(
                1,
                &[Request::SetPosition {
                    segment_id: "a".into(),
                    x: 0.25,
                    y: -0.5,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.clip.transform.x, 0.25);
        assert_eq!(segment.clip.transform.y, -0.5);
        assert_eq!(segment.clip.rotation, 90.0);
        assert!(segment.clip.flip.horizontal);
        assert_eq!(segment.uniform_scale.value, 1.0);
        assert_eq!(segment.volume, 0.4);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(10, 1_000_000)));
        session.undo().unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].clip.transform.x, 0.0);
        assert_eq!(session.draft().tracks[0].segments[0].clip.transform.y, 0.0);
    }

    #[test]
    fn a_scale_changes_both_axes_and_keeps_the_uniform_switch() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.clip.rotation = 90.0;
        segment.uniform_scale.on = false;
        let (mut session, _) = session_with(draft, segment);
        session
            .apply(
                1,
                &[Request::SetScale {
                    segment_id: "a".into(),
                    scale: 1.5,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.clip.scale.x, 1.5);
        assert_eq!(segment.clip.scale.y, 1.5);
        assert!(!segment.uniform_scale.on);
        assert_eq!(segment.uniform_scale.value, 1.5);
        assert_eq!(segment.clip.transform.x, 0.0);
        assert_eq!(segment.clip.rotation, 90.0);
        session.undo().unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].clip.scale.x, 1.0);
        assert!(!session.draft().tracks[0].segments[0].uniform_scale.on);
    }

    #[test]
    fn a_bad_position_or_scale_and_a_locked_lane_leave_the_draft() {
        let (mut session, track) = open();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[Request::SetPosition {
                    segment_id: "a".into(),
                    x: f64::NAN,
                    y: 0.0,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Position);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::SetScale {
                    segment_id: "a".into(),
                    scale: 0.0,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Scale);
        assert_eq!(session.draft(), &before);
        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: track,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let locked = session.draft().clone();
        let error = session
            .apply(
                2,
                &[Request::SetPosition {
                    segment_id: "a".into(),
                    x: 1.0,
                    y: 1.0,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &locked);
    }

    #[test]
    fn an_opacity_dims_the_picture_and_leaves_the_rest() {
        let (mut session, track) = open();
        session
            .apply(
                1,
                &[Request::SetOpacity {
                    segment_id: "a".into(),
                    opacity: 0.5,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.clip.alpha, 0.5);
        assert_eq!(segment.clip.rotation, 90.0);
        assert!(segment.clip.flip.horizontal);
        assert_eq!(segment.uniform_scale.value, 1.0);
        assert_eq!(segment.clip.transform.x, 0.0);
        assert_eq!(segment.volume, 0.4);
        session.undo().unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].clip.alpha, 1.0);

        session
            .apply(
                3,
                &[Request::SetOpacity {
                    segment_id: "a".into(),
                    opacity: 0.0,
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].clip.alpha, 0.0);
        let clear = session.draft().clone();
        let error = session
            .apply(
                4,
                &[Request::SetOpacity {
                    segment_id: "a".into(),
                    opacity: 1.1,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Opacity);
        let error = session
            .apply(
                4,
                &[Request::SetOpacity {
                    segment_id: "a".into(),
                    opacity: f64::NAN,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Opacity);
        assert_eq!(session.draft(), &clear);

        session
            .apply(
                4,
                &[Request::SetTrackAttribute {
                    track_id: track,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                5,
                &[Request::SetOpacity {
                    segment_id: "a".into(),
                    opacity: 1.0,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().tracks[0].segments[0].clip.alpha, 0.0);
    }
}
