//! Wire the Slint `ColorUtil` global to [`crate::color_math`].

use slint::{Color, ComponentHandle, SharedString};

use crate::color_math;
use crate::{AppWindow, ColorUtil, HsvComponents};

pub(crate) fn wire_color(app: &AppWindow) {
    let util = app.global::<ColorUtil>();

    util.on_hex_to_color(|hex| match color_math::parse_hex(hex.as_str()) {
        Some([r, g, b, a]) => Color::from_argb_u8(a, r, g, b),
        None => Color::from_argb_u8(0, 0, 0, 0),
    });

    util.on_is_valid_hex(|hex| color_math::parse_hex(hex.as_str()).is_some());

    util.on_color_to_hex(|c| {
        SharedString::from(color_math::format_hex(
            c.red(),
            c.green(),
            c.blue(),
            c.alpha(),
        ))
    });

    util.on_rgb_to_hsv(|r, g, b| {
        let (h, s, v) = color_math::rgb_to_hsv(r, g, b);
        HsvComponents { h, s, v }
    });

    util.on_rgb_to_hsv_preserving(|r, g, b, prev_h, prev_s| {
        let (h, s, v) = color_math::rgb_to_hsv_preserving(r, g, b, prev_h, prev_s);
        HsvComponents { h, s, v }
    });

    util.on_hsv_to_color(|h, s, v, a| {
        let (r, g, b) = color_math::hsv_to_rgb(h, s, v);
        Color::from_argb_u8(
            (a.clamp(0.0, 1.0) * 255.0).round() as u8,
            (r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
    });
}
