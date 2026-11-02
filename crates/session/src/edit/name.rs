use cutlass_model::Draft;

/// Store the project name. Surrounding whitespace is dropped.
///
/// An empty name stays empty. The canvas, the frame rate, and the color
/// space stay.
pub fn set_project_name(draft: &mut Draft, name: &str) {
    draft.name = name.trim().to_owned();
}

#[cfg(test)]
mod tests {
    use cutlass_model::Draft;

    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_project_name_is_stored_and_the_frame_stays() {
        let mut session = Session::open(Draft::empty("D"));
        let width = session.draft().canvas_config.width;
        let space = session.draft().color_space;
        let fps = session.draft().fps;
        session
            .apply(
                0,
                &[Request::SetProjectName {
                    name: "  Cut  ".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().name, "Cut");
        assert_eq!(session.draft().canvas_config.width, width);
        assert_eq!(session.draft().color_space, space);
        assert_eq!(session.draft().fps, fps);

        session
            .apply(
                1,
                &[Request::SetProjectName {
                    name: " \n ".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().name, "");
        session.undo().unwrap();
        assert_eq!(session.draft().name, "Cut");
        assert_eq!(session.draft().canvas_config.width, width);
    }
}
