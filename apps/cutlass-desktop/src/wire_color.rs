//! Wire the Slint `ColorUtil` global to [`crate::color_math`], and keep the
//! app-wide recent-colors list in sync with `~/.cutlass/config.toml`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use slint::{Color, ComponentHandle, ModelRc, VecModel};

use crate::color_math;
use crate::{AppStore, AppWindow, ColorUtil, HsvComponents};

pub(crate) fn wire_color(
    app: &AppWindow,
    config_path: PathBuf,
    recent_colors: &[cutlass_settings::Rgba],
) {
    let util = app.global::<ColorUtil>();

    util.on_hex_to_color(|hex| match color_math::parse_hex(hex.as_str()) {
        Some([r, g, b, a]) => Color::from_argb_u8(a, r, g, b),
        None => Color::from_argb_u8(0, 0, 0, 0),
    });

    util.on_is_valid_hex(|hex| color_math::parse_hex(hex.as_str()).is_some());

    util.on_color_to_hex(|c| {
        slint::SharedString::from(color_math::format_hex(
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

    // Seed the shell model from settings (most-recent-first).
    let initial: Vec<Color> = recent_colors
        .iter()
        .copied()
        .map(|[r, g, b, a]| Color::from_argb_u8(a, r, g, b))
        .collect();
    let model = Rc::new(VecModel::from(initial));
    app.global::<AppStore>()
        .set_recent_colors(ModelRc::from(model.clone()));

    let live = Rc::new(RefCell::new(recent_colors.to_vec()));
    let app_weak = app.as_weak();
    util.on_record_recent(move |c| {
        let rgba = [c.red(), c.green(), c.blue(), c.alpha()];
        let mut list = live.borrow_mut();
        cutlass_settings::push_recent(&mut list, rgba);

        // Refresh the Slint model in place so every bound picker updates.
        model.set_vec(
            list.iter()
                .copied()
                .map(|[r, g, b, a]| Color::from_argb_u8(a, r, g, b))
                .collect::<Vec<_>>(),
        );
        if let Some(app) = app_weak.upgrade() {
            app.global::<AppStore>()
                .set_recent_colors(ModelRc::from(model.clone()));
        }

        if let Err(error) = persist_recent_colors_at(&config_path, &list) {
            tracing::warn!(%error, "recent colors could not be saved");
        }
    });
}

fn persist_recent_colors_at(
    path: &std::path::Path,
    colors: &[cutlass_settings::Rgba],
) -> Result<(), String> {
    let mut settings = cutlass_settings::load(path)
        .map_err(|error| format!("recent colors not saved; settings file unreadable: {error}"))?;
    settings.appearance.recent_colors = colors.to_vec();
    cutlass_settings::save(path, &settings)
        .map_err(|error| format!("recent colors could not be written: {error}"))
}
