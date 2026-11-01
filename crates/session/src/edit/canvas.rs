use cutlass_model::{CanvasRatio, Draft};

use super::EditError;

/// The largest canvas side the preview keeps. A side outside 1 through this is refused.
const MAX_CANVAS_SIDE: i32 = 7_680;

/// Store the color under every layer.
///
/// An empty string clears it, and the preview stays clear. Six hex digits,
/// with an optional `#`, are opaque. Eight hex digits are `AARRGGBB`, with
/// an optional `#` or `0x`. The canvas size and ratio stay.
pub fn set_background(draft: &mut Draft, color: &str) -> Result<(), EditError> {
    let color = color.trim();
    if color.is_empty() {
        draft.canvas_config.background = None;
        return Ok(());
    }
    if !canvas_color(color) {
        return Err(EditError::CanvasColor);
    }
    draft.canvas_config.background = Some(color.to_owned());
    Ok(())
}

/// Store the frame size and the ratio name.
///
/// Each side is from 1 through 7680. The ratio is one of the project shapes,
/// including `original` and `custom`. The background and the frame rate stay.
pub fn set_size(draft: &mut Draft, width: i32, height: i32, ratio: &str) -> Result<(), EditError> {
    if !(1..=MAX_CANVAS_SIDE).contains(&width) || !(1..=MAX_CANVAS_SIDE).contains(&height) {
        return Err(EditError::CanvasSize);
    }
    if CanvasRatio::parse(ratio).as_str() != ratio {
        return Err(EditError::CanvasRatio);
    }
    draft.canvas_config.width = width;
    draft.canvas_config.height = height;
    draft.canvas_config.ratio = ratio.to_owned();
    Ok(())
}

/// A color the preview paints as the project canvas.
///
/// Eight digits are read as `AARRGGBB` before a six-digit `RRGGBB` color.
fn canvas_color(text: &str) -> bool {
    if packed(text) {
        return true;
    }
    let hex = text.strip_prefix('#').unwrap_or(text);
    hex.len() == 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn packed(text: &str) -> bool {
    let hex = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .or_else(|| text.strip_prefix('#'))
        .unwrap_or(text);
    hex.len() == 8 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use cutlass_model::Draft;

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_canvas_color_sits_under_the_frame_and_an_empty_string_clears_it() {
        let mut session = Session::open(Draft::empty("D"));
        let size = session.draft().canvas_config.clone();
        session
            .apply(
                0,
                &[Request::SetCanvasBackground {
                    color: "  #336699  ".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().canvas_config.background.as_deref(),
            Some("#336699")
        );
        assert_eq!(session.draft().canvas_config.width, size.width);
        assert_eq!(session.draft().canvas_config.height, size.height);
        assert_eq!(session.draft().canvas_config.ratio, size.ratio);

        session
            .apply(
                1,
                &[Request::SetCanvasBackground {
                    color: String::new(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().canvas_config.background, None);
        session.undo().unwrap();
        assert_eq!(
            session.draft().canvas_config.background.as_deref(),
            Some("#336699")
        );

        session
            .apply(
                3,
                &[Request::SetCanvasBackground {
                    color: "0xFF336699".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().canvas_config.background.as_deref(),
            Some("0xFF336699")
        );

        let before = session.draft().clone();
        let error = session
            .apply(
                4,
                &[Request::SetCanvasBackground {
                    color: "white".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasColor);
        let error = session
            .apply(
                4,
                &[Request::SetCanvasBackground {
                    color: "0x336699".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasColor);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_canvas_size_sets_the_frame_and_a_side_past_7680_is_refused() {
        let mut session = Session::open(Draft::empty("D"));
        let fps = session.draft().fps;
        session
            .apply(
                0,
                &[Request::SetCanvasBackground {
                    color: "#336699".to_owned(),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetCanvasSize {
                    width: 1_080,
                    height: 1_920,
                    ratio: "9:16".to_owned(),
                }],
            )
            .unwrap();
        let canvas = &session.draft().canvas_config;
        assert_eq!(canvas.width, 1_080);
        assert_eq!(canvas.height, 1_920);
        assert_eq!(canvas.ratio, "9:16");
        assert_eq!(canvas.background.as_deref(), Some("#336699"));
        assert_eq!(session.draft().fps, fps);
        session.undo().unwrap();
        assert_eq!(session.draft().canvas_config.width, 1_920);
        assert_eq!(session.draft().canvas_config.ratio, "original");

        session
            .apply(
                3,
                &[Request::SetCanvasSize {
                    width: 7_680,
                    height: 1,
                    ratio: "custom".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().canvas_config.width, 7_680);

        let before = session.draft().clone();
        let error = session
            .apply(
                4,
                &[Request::SetCanvasSize {
                    width: 0,
                    height: 1080,
                    ratio: "16:9".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasSize);
        let error = session
            .apply(
                4,
                &[Request::SetCanvasSize {
                    width: 7_681,
                    height: 1080,
                    ratio: "16:9".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasSize);
        let error = session
            .apply(
                4,
                &[Request::SetCanvasSize {
                    width: 1920,
                    height: 1080,
                    ratio: "wide".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasRatio);
        assert_eq!(session.draft(), &before);
    }
}
