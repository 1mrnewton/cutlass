//! A name above a control, and the same name in the accessibility tree.
//! `docs/desktop-layout.md`.

use eframe::egui::{self, Color32};

use crate::library::card_time;

/// Positions to try after the centered tip above the control.
const ELSEWHERE: &[egui::RectAlign] = &[
    egui::RectAlign::TOP_START,
    egui::RectAlign::TOP_END,
    egui::RectAlign::BOTTOM,
    egui::RectAlign::BOTTOM_START,
    egui::RectAlign::BOTTOM_END,
];

/// The name, centered above `response`. It drops below when the top does not fit.
/// The same words name the control for a screen reader.
pub(crate) fn show(response: &egui::Response, text: &str) {
    show_enabled(response, text, true);
}

/// Like [`show`], and `enabled` is false when the control cannot run.
pub(crate) fn show_enabled(response: &egui::Response, text: &str, enabled: bool) {
    control(response, egui::accesskit::Role::Button, text, enabled, None);
    hover(response, text);
}

/// An on/off control. The hover text stays the short name. The tree also says whether it is on.
pub(crate) fn show_switch(response: &egui::Response, text: &str, on: bool, enabled: bool) {
    control(
        response,
        egui::accesskit::Role::Switch,
        text,
        enabled,
        Some(on),
    );
    hover(response, text);
}

/// One choice in a group, such as a library shelf.
pub(crate) fn show_radio(response: &egui::Response, text: &str, on: bool) {
    control(
        response,
        egui::accesskit::Role::RadioButton,
        text,
        true,
        Some(on),
    );
    hover(response, text);
}

/// A clip. `on` means it is selected. The hover text includes the time span.
pub(crate) fn show_selected(response: &egui::Response, text: &str, on: bool) {
    control(
        response,
        egui::accesskit::Role::Button,
        text,
        true,
        Some(on),
    );
    hover(response, text);
}

/// A marker or a beat. It is named, and a click still belongs to the timeline.
pub(crate) fn show_image(response: &egui::Response, text: &str) {
    image(response, text);
    hover(response, text);
}

/// A named picture, such as the playhead. It has no hover tip.
pub(crate) fn image(response: &egui::Response, text: &str) {
    control(response, egui::accesskit::Role::Image, text, true, None);
}

/// A word that is already painted, so it has a name and no second tip.
pub(crate) fn name(response: &egui::Response, text: &str) {
    control(response, egui::accesskit::Role::Button, text, true, None);
}

/// A section that folds. `open` is whether the fields are shown.
pub(crate) fn disclosure(response: &egui::Response, text: &str, open: bool) {
    control(
        response,
        egui::accesskit::Role::DisclosureTriangle,
        text,
        true,
        None,
    );
    response.ctx.accesskit_node_builder(response.id, |node| {
        node.set_expanded(open);
    });
}

/// Name a panel. The node already exists. This only sets its label.
pub(crate) fn region(ui: &egui::Ui, name: &str) {
    ui.ctx().accesskit_node_builder(ui.unique_id(), |node| {
        node.set_label(name);
    });
}

/// Painted text a screen reader would otherwise miss.
pub(crate) fn read(ui: &egui::Ui, rect: egui::Rect, id: egui::Id, text: &str) {
    let response = ui.interact(rect, id, egui::Sense::hover());
    control(&response, egui::accesskit::Role::Label, text, true, None);
}

/// A refused edit. A screen reader hears it when it appears.
pub(crate) fn alert(response: &egui::Response, text: &str) {
    control(response, egui::accesskit::Role::Alert, text, true, None);
    response.ctx.accesskit_node_builder(response.id, |node| {
        node.set_live(egui::accesskit::Live::Polite);
    });
}

fn control(
    response: &egui::Response,
    role: egui::accesskit::Role,
    text: &str,
    enabled: bool,
    on: Option<bool>,
) {
    if text.is_empty() {
        return;
    }
    response.widget_info(|| {
        if let Some(on) = on {
            egui::WidgetInfo::selected(role, enabled, on, text)
        } else {
            egui::WidgetInfo::labeled(role, enabled, text)
        }
    });
}

fn hover(response: &egui::Response, text: &str) {
    if text.is_empty() {
        return;
    }
    let mut tip = egui::Tooltip::for_enabled(response);
    tip.popup = tip
        .popup
        .align(egui::RectAlign::TOP)
        .align_alternatives(ELSEWHERE)
        .gap(6.0)
        .frame(pill());
    tip.show(|ui| {
        ui.add(
            egui::Label::new(egui::RichText::new(text).size(12.0).color(Color32::WHITE))
                .wrap_mode(egui::TextWrapMode::Extend),
        );
    });
}

/// `name · start–end`, or just the span when the file has no name.
pub(crate) fn span(name: &str, start: i64, end: i64, timescale: u32) -> String {
    let range = format!(
        "{}–{}",
        card_time(start, timescale),
        card_time(end, timescale)
    );
    if name.is_empty() {
        range
    } else {
        format!("{name} · {range}")
    }
}

/// `Marker · time` or `Beat · time`, in `language`. The clock stays `M:SS.t`.
pub(crate) fn mark(
    language: crate::words::Language,
    beat: bool,
    time: i64,
    timescale: u32,
) -> String {
    let kind = crate::words::say(
        language,
        if beat {
            crate::words::Phrase::Beat
        } else {
            crate::words::Phrase::Marker
        },
    );
    format!("{kind} · {}", card_time(time, timescale))
}

fn pill() -> egui::Frame {
    egui::Frame::NONE
        .fill(Color32::from_rgb(0x3a, 0x3a, 0x3a))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(8, 4))
        .shadow(egui::Shadow {
            offset: [0, 6],
            blur: 16,
            spread: 0,
            color: Color32::from_black_alpha(89),
        })
}

#[cfg(test)]
mod tests {
    use eframe::egui;

    use super::{mark, span};
    use crate::words::Language;

    #[test]
    fn a_clip_tip_names_the_file_and_its_span() {
        assert_eq!(
            span("v.mp4", 0, 1_500_000, 1_000_000),
            "v.mp4 · 0:00.0–0:01.5"
        );
        assert_eq!(span("", 0, 1_500_000, 1_000_000), "0:00.0–0:01.5");
    }

    #[test]
    fn a_mark_tip_names_the_kind_and_the_time() {
        assert_eq!(
            mark(Language::English, false, 2_000_000, 1_000_000),
            "Marker · 0:02.0"
        );
        assert_eq!(
            mark(Language::English, true, 2_000_000, 1_000_000),
            "Beat · 0:02.0"
        );
        assert_eq!(
            mark(Language::French, false, 2_000_000, 1_000_000),
            "Marqueur · 0:02.0"
        );
    }

    #[test]
    fn a_control_keeps_its_hover_name_in_the_accessibility_tree() {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..Default::default()
        };
        let mut output = ctx.run_ui(raw, |ui| {
            let play = ui.interact(
                egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(36.0, 36.0)),
                ui.scope_id().with("play"),
                egui::Sense::click(),
            );
            super::show(&play, "Play");
            let magnet = ui.interact(
                egui::Rect::from_min_size(egui::pos2(50.0, 10.0), egui::vec2(26.0, 26.0)),
                ui.scope_id().with("magnet"),
                egui::Sense::click(),
            );
            super::show_switch(&magnet, "Main track magnet on", true, true);
        });
        let update = output
            .platform_output
            .accesskit_update
            .take()
            .expect("the tree is built when accesskit is on");
        output.textures_delta.clear();
        let play = update
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some("Play"))
            .expect("play");
        assert_eq!(play.1.role(), egui::accesskit::Role::Button);
        let magnet = update
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some("Main track magnet on"))
            .expect("magnet");
        assert_eq!(magnet.1.role(), egui::accesskit::Role::Switch);
        assert_eq!(magnet.1.toggled(), Some(egui::accesskit::Toggled::True));
    }
}
