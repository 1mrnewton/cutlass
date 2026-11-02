use cutlass_model::Draft;

use super::EditError;

/// The spaces the player keeps. Any other int is not a project color space.
const PROJECT_COLOR_SPACES: [i32; 5] = [-1, 0, 2, 67, 2084];

/// Store the project color space.
///
/// `-1` is unknown, `0` is none, `2` is SDR 2020, `67` is HLG, and `2084` is PQ.
/// The canvas and the frame rate stay. This does not retint the preview.
pub fn set_color_space(draft: &mut Draft, space: i32) -> Result<(), EditError> {
    if !PROJECT_COLOR_SPACES.contains(&space) {
        return Err(EditError::ColorSpace);
    }
    draft.color_space = space;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::Draft;

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_color_space_is_stored_and_any_other_int_is_refused() {
        let mut session = Session::open(Draft::empty("D"));
        let width = session.draft().canvas_config.width;
        let fps = session.draft().fps;
        for (expected, space) in [0_u64, 1, 2, 3, 4].into_iter().zip([2, 67, 2084, -1, 0]) {
            session
                .apply(expected, &[Request::SetColorSpace { space }])
                .unwrap();
            assert_eq!(session.draft().color_space, space);
        }
        assert_eq!(session.draft().canvas_config.width, width);
        assert_eq!(session.draft().fps, fps);
        session.undo().unwrap();
        assert_eq!(session.draft().color_space, -1);

        let before = session.draft().clone();
        for space in [1, 3, i32::MAX] {
            let error = session
                .apply(6, &[Request::SetColorSpace { space }])
                .unwrap_err();
            assert_eq!(error, EditError::ColorSpace);
        }
        assert_eq!(session.draft(), &before);
    }
}
