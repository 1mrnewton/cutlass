//! The right-click plate shared by the library and the timeline.
//! `docs/desktop-layout.md`.

use std::sync::Arc;

use eframe::egui::{
    self, AtomKind, Button, Color32, FontFamily, FontId, IntoSizedResult, RichText, SizedAtomKind,
    Stroke, TextStyle, vec2,
};

use crate::icons::{self, Icon};
use crate::theme;

/// Outer width of the plate, including the 6px padding on each side.
const PLATE: f32 = 248.0;
const PAD: f32 = 6.0;
const INNER: f32 = PLATE - PAD * 2.0;
const ROW: f32 = 32.0;

pub(crate) fn context_menu(response: &egui::Response, add: impl FnOnce(&mut egui::Ui)) {
    egui::Popup::context_menu(response)
        .style(look)
        .info(stack())
        .show(|ui| {
            widen(ui);
            add(ui);
        });
}

pub(crate) fn row(ui: &mut egui::Ui, label: &str, enabled: bool) -> bool {
    press(ui, label, None, None, enabled)
}

pub(crate) fn row_key(ui: &mut egui::Ui, label: &str, key: &str, enabled: bool) -> bool {
    press(ui, label, Some(key), None, enabled)
}

pub(crate) fn row_badge(ui: &mut egui::Ui, label: &str, badge: &str, enabled: bool) -> bool {
    press(ui, label, None, Some(badge), enabled)
}

/// A row that opens the same plate on hover. The chevron is the trailing mark.
pub(crate) fn submenu(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui)) {
    let button =
        labeled(label, theme::INK, None).right_text(icons::mark(Icon::Chevron, 14.0, theme::MUTED));
    egui::menu::SubMenuButton::from_button(button).ui(ui, |ui| {
        widen(ui);
        add(ui);
    });
}

/// A 1px line, white at 8%, with 6px above and below. The plate's padding
/// already insets 6px, and another 2px makes the 8px inset.
pub(crate) fn rule(ui: &mut egui::Ui) {
    let width = if ui.is_sizing_pass() {
        0.0
    } else {
        ui.available_width()
    };
    let (rect, _) = ui.allocate_exact_size(vec2(width, 13.0), egui::Sense::hover());
    if ui.is_sizing_pass() || rect.width() <= 4.0 {
        return;
    }
    let inset = 2.0;
    ui.painter().hline(
        (rect.left() + inset)..=(rect.right() - inset),
        rect.center().y,
        Stroke::new(1.0, Color32::from_white_alpha(20)),
    );
}

fn press(
    ui: &mut egui::Ui,
    label: &str,
    key: Option<&str>,
    badge: Option<&str>,
    enabled: bool,
) -> bool {
    let ink = if enabled { theme::INK } else { theme::MENU_OFF };
    let mut button = labeled(label, ink, badge);
    if let Some(key) = key {
        let color = if enabled {
            theme::MENU_KEY
        } else {
            theme::MENU_KEY_OFF
        };
        button = button.right_text(RichText::new(key).size(12.0).color(color));
    }
    let clicked = ui.add_enabled(enabled, button).clicked();
    if clicked {
        ui.close();
    }
    clicked
}

fn labeled<'a>(label: &'a str, ink: Color32, badge: Option<&str>) -> Button<'a> {
    let text = RichText::new(label).color(ink);
    match badge {
        Some(badge) => Button::new((text, badge_mark(badge.to_owned()))).gap(8.0),
        None => Button::new(text),
    }
}

fn badge_mark(text: String) -> egui::Atom<'static> {
    AtomKind::closure(move |ui, _| {
        let font = FontId::new(10.0, FontFamily::Proportional);
        let galley = ui
            .painter()
            .layout_no_wrap(text.clone(), font, Color32::WHITE);
        let size = vec2(galley.size().x + 12.0, 16.0);
        IntoSizedResult {
            intrinsic_size: size,
            sized: SizedAtomKind::Paint {
                paint: Arc::new(move |ui, args| {
                    let rect = egui::Rect::from_center_size(args.rect.center(), size);
                    ui.painter().rect_filled(rect, 8.0, theme::MENU_BADGE);
                    let pos =
                        egui::pos2(rect.left() + 6.0, rect.center().y - galley.size().y * 0.5);
                    ui.painter().galley(pos, galley.clone(), Color32::WHITE);
                }),
                size,
            },
        }
    })
    .into()
}

fn widen(ui: &mut egui::Ui) {
    ui.set_min_width(INNER);
}

fn look(style: &mut egui::Style) {
    style.spacing.button_padding = vec2(10.0, 0.0);
    style.spacing.interact_size.y = ROW;
    style.spacing.item_spacing.y = 0.0;
    style.spacing.icon_spacing = 16.0;
    style.spacing.menu_margin = egui::Margin::same(6);
    style.visuals.menu_corner_radius = egui::CornerRadius::same(12);
    style.visuals.window_fill = theme::MENU;
    style.visuals.window_stroke = Stroke::NONE;
    style.visuals.popup_shadow = egui::Shadow {
        offset: [0, 16],
        blur: 40,
        spread: 0,
        color: Color32::from_black_alpha(115),
    };
    style.visuals.disabled_alpha = 1.0;
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(13.0, FontFamily::Proportional),
    );
    row_visual(
        &mut style.visuals.widgets.inactive,
        Color32::TRANSPARENT,
        theme::INK,
    );
    row_visual(
        &mut style.visuals.widgets.hovered,
        theme::MENU_HOVER,
        theme::INK,
    );
    row_visual(
        &mut style.visuals.widgets.active,
        theme::MENU_HOVER,
        theme::INK,
    );
    row_visual(
        &mut style.visuals.widgets.open,
        theme::MENU_HOVER,
        theme::INK,
    );
    row_visual(
        &mut style.visuals.widgets.noninteractive,
        Color32::TRANSPARENT,
        theme::MENU_OFF,
    );
}

fn row_visual(visual: &mut egui::style::WidgetVisuals, fill: Color32, ink: Color32) {
    visual.corner_radius = egui::CornerRadius::same(8);
    visual.bg_stroke = Stroke::NONE;
    visual.fg_stroke = Stroke::new(0.0, ink);
    visual.expansion = 0.0;
    visual.weak_bg_fill = fill;
}

fn stack() -> egui::UiStackInfo {
    // Half of the 12px horizontal padding is added to this gap. -12 leaves the
    // submenu overlapping the row by 6px.
    egui::UiStackInfo::new(egui::UiKind::Menu).with_tag_value(
        egui::menu::MenuConfig::MENU_CONFIG_TAG,
        egui::menu::MenuConfig::new().style(look).submenu_gap(-12.0),
    )
}
