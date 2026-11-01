use cutlass_model::Draft;

use super::EditError;

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
}
