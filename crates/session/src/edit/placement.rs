use cutlass_model::{Draft, SegmentId};

use super::locate;
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

/// Store the picture's scale. Uniform scale and the width take `scale`.
/// With uniform scale on the height takes it too; with it off the height
/// keeps its share of the width, so a stretched picture stays stretched.
/// The uniform switch stays as it was.
pub fn set_scale(draft: &mut Draft, segment_id: &SegmentId, scale: f64) -> Result<(), EditError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(EditError::Scale);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    let stretch = if segment.uniform_scale.on {
        1.0
    } else {
        stretch_of(segment.clip.scale.x, segment.clip.scale.y)
    };
    segment.uniform_scale.value = scale;
    segment.clip.scale.x = scale;
    segment.clip.scale.y = scale * stretch;
    Ok(())
}

/// How tall a picture is drawn for its width: its height scale over its
/// width scale, 1 when that does not read as a size.
#[must_use]
pub fn stretch_of(x: f64, y: f64) -> f64 {
    let stretch = (y / x).abs();
    if stretch.is_finite() && stretch > 0.0 {
        stretch
    } else {
        1.0
    }
}

/// Turn uniform scale on or off. Off, each axis takes the size uniform
/// scale drew; on, the uniform value and the height take the width, so the
/// picture keeps its width. Asking for the switch it has changes nothing.
pub fn set_uniform_scale(
    draft: &mut Draft,
    segment_id: &SegmentId,
    on: bool,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    if segment.uniform_scale.on == on {
        return Ok(());
    }
    if on {
        let width = segment.clip.scale.x.abs();
        let width = if width.is_finite() && width > 0.0 {
            width
        } else {
            1.0
        };
        segment.uniform_scale.value = width;
        segment.clip.scale.x = width;
        segment.clip.scale.y = width;
    } else {
        let size = segment.uniform_scale.value;
        let size = if size.is_finite() && size > 0.0 {
            size
        } else {
            1.0
        };
        segment.clip.scale.x = size;
        segment.clip.scale.y = size;
    }
    segment.uniform_scale.on = on;
    Ok(())
}

/// Store the picture's width and height scales, each on its own. They are
/// what is drawn with uniform scale off; a negative one flips that axis.
/// The switch stays as it was.
pub fn set_scale_axes(
    draft: &mut Draft,
    segment_id: &SegmentId,
    x: f64,
    y: f64,
) -> Result<(), EditError> {
    if !(x.is_finite() && y.is_finite() && x != 0.0 && y != 0.0) {
        return Err(EditError::Scale);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    segment.clip.scale.x = x;
    segment.clip.scale.y = y;
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
        let mut draft = crate::edit::test_draft("D");
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
    fn a_scale_with_uniform_off_keeps_the_stretch() {
        let mut draft = crate::edit::test_draft("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.uniform_scale.on = false;
        segment.clip.scale.x = 1.0;
        segment.clip.scale.y = 2.0;
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
        assert_eq!((segment.clip.scale.x, segment.clip.scale.y), (1.5, 3.0));
        assert!(!segment.uniform_scale.on);
    }

    #[test]
    fn uniform_scale_turns_off_and_on_and_keeps_the_size() {
        let mut draft = crate::edit::test_draft("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.uniform_scale.on = true;
        segment.uniform_scale.value = 0.5;
        // Axes left from an earlier stretch are not what uniform drew.
        segment.clip.scale.x = 3.0;
        segment.clip.scale.y = 4.0;
        let (mut session, _) = session_with(draft, segment);
        let switch = |session: &mut Session, revision: u64, on: bool| {
            session
                .apply(
                    revision,
                    &[Request::SetUniformScale {
                        segment_id: "a".into(),
                        on,
                    }],
                )
                .unwrap();
        };
        switch(&mut session, 1, false);
        let segment = &session.draft().tracks[0].segments[0];
        assert!(!segment.uniform_scale.on);
        assert_eq!((segment.clip.scale.x, segment.clip.scale.y), (0.5, 0.5));
        session
            .apply(
                2,
                &[Request::SetScaleAxes {
                    segment_id: "a".into(),
                    x: 0.75,
                    y: 1.25,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!((segment.clip.scale.x, segment.clip.scale.y), (0.75, 1.25));
        assert!(!segment.uniform_scale.on);
        // On again, the width drives both.
        switch(&mut session, 3, true);
        let segment = &session.draft().tracks[0].segments[0];
        assert!(segment.uniform_scale.on);
        assert_eq!(segment.uniform_scale.value, 0.75);
        assert_eq!((segment.clip.scale.x, segment.clip.scale.y), (0.75, 0.75));
        // The switch it has already is no change.
        let before = session.draft().clone();
        switch(&mut session, 4, true);
        assert_eq!(session.draft(), &before);
        // The switch, the axes, the switch back, and the one that changed
        // nothing are four steps.
        for _ in 0..4 {
            session.undo().unwrap();
        }
        let segment = &session.draft().tracks[0].segments[0];
        assert!(segment.uniform_scale.on);
        assert_eq!((segment.clip.scale.x, segment.clip.scale.y), (3.0, 4.0));
    }

    #[test]
    fn a_bad_axis_scale_or_a_locked_lane_leaves_the_draft() {
        let (mut session, track) = open();
        let before = session.draft().clone();
        // A negative axis flips, so only none or a broken one is refused.
        for (x, y) in [
            (0.0, 1.0),
            (1.0, 0.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            let error = session
                .apply(
                    1,
                    &[Request::SetScaleAxes {
                        segment_id: "a".into(),
                        x,
                        y,
                    }],
                )
                .unwrap_err();
            assert_eq!(error, EditError::Scale);
            assert_eq!(session.draft(), &before);
        }
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
                &[Request::SetUniformScale {
                    segment_id: "a".into(),
                    on: false,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &locked);
    }

    #[test]
    fn a_scale_changes_both_axes_and_keeps_the_uniform_switch() {
        let mut draft = crate::edit::test_draft("D");
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
