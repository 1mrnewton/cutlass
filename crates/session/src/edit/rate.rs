use cutlass_model::{Draft, FrameRate};

use super::EditError;

/// Store the project frame rate and the drop-frame timecode label.
///
/// The clock has to be one the draft can step. Clip times stay microseconds.
/// The canvas stays.
pub fn set_frame_rate(draft: &mut Draft, fps: f64, drop_frame: bool) -> Result<(), EditError> {
    if FrameRate::from_stored_fps(fps).is_none() {
        return Err(EditError::FrameRate);
    }
    draft.fps = fps;
    draft.is_drop_frame_timecode = drop_frame;
    Ok(())
}

/// Store the drop-frame timecode label. The frame rate stays.
pub fn set_drop_frame(draft: &mut Draft, on: bool) -> Result<(), EditError> {
    draft.is_drop_frame_timecode = on;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, FrameRate, Micros};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_frame_rate_steps_the_clock_and_drop_frame_is_a_label() {
        let mut session = Session::open(Draft::empty("D"));
        let width = session.draft().canvas_config.width;
        session
            .apply(
                0,
                &[Request::SetFrameRate {
                    fps: 29.97,
                    drop_frame: true,
                }],
            )
            .unwrap();
        let draft = session.draft();
        assert_eq!(draft.frame_rate(), Some(FrameRate::FPS_29_97));
        assert!(draft.is_drop_frame_timecode);
        assert_eq!(draft.timecode(Micros(0)).unwrap().format(), "00:00:00;00");
        assert_eq!(draft.canvas_config.width, width);

        session
            .apply(1, &[Request::SetDropFrame { on: false }])
            .unwrap();
        assert_eq!(session.draft().frame_rate(), Some(FrameRate::FPS_29_97));
        assert!(!session.draft().is_drop_frame_timecode);
        assert_eq!(session.draft().timecode(Micros(1_000_000)), None);
        session.undo().unwrap();
        assert!(session.draft().is_drop_frame_timecode);

        let before = session.draft().clone();
        let error = session
            .apply(
                3,
                &[Request::SetFrameRate {
                    fps: 23.98,
                    drop_frame: false,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::FrameRate);
        let error = session
            .apply(
                3,
                &[Request::SetFrameRate {
                    fps: f64::NAN,
                    drop_frame: true,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::FrameRate);
        assert_eq!(session.draft(), &before);
    }
}
