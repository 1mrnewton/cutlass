//! A name above a control. `docs/desktop-layout.md`.

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
pub(crate) fn show(response: &egui::Response, text: &str) {
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

/// `Marker · time` or `Beat · time`.
pub(crate) fn mark(beat: bool, time: i64, timescale: u32) -> String {
    let kind = if beat { "Beat" } else { "Marker" };
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
    use super::{mark, span};

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
        assert_eq!(mark(false, 2_000_000, 1_000_000), "Marker · 0:02.0");
        assert_eq!(mark(true, 2_000_000, 1_000_000), "Beat · 0:02.0");
    }
}
