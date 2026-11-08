//! A video clip's AI stylize tab: AI effects from a prompt, and styles,
//! laid out as the reference lays them out.
//!
//! None of it edits the clip yet, so every control is dimmed and marked
//! "Later".

use egui::{Align, Align2, CornerRadius, FontId, Layout, RichText, Sense, Ui, Vec2};

use crate::{tokens, ui};

use super::audio::{later_section, name_tile};
use super::inspector::{BOX, InspectorState, ROW_GAP, none_tile};

/// The kinds of style, by our own names.
const KINDS: [&str; 12] = [
    "Trending",
    "Painting",
    "Photo",
    "Particle",
    "Expressions",
    "Hair",
    "Motion",
    "Split",
    "Portrait",
    "Face swap",
    "Scene swap",
    "Video effects",
];

/// The styles shown under Trending, by our own names.
const TRENDING: [&str; 6] = ["Surreal", "Dreamy", "Cyber", "Doll", "Camera shake", "Zoom"];

pub(super) fn ai_stylize_tab(ui: &mut Ui, state: &mut InspectorState) {
    later_section(
        ui,
        state,
        "ai-effects",
        "AI effects",
        Some("Describe a look and the clip is redrawn in it."),
        Some(&mut |ui: &mut Ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Prompt").size(12.0));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add(ui::Button::new("Examples").text_size(12.0));
                });
            });
            ui.add_space(4.0);
            prompt_box(ui);
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add(ui::Button::new("Generate").text_size(12.0));
                    ui.add(ui::Button::new("Random").text_size(12.0));
                });
            });
            ui.add_space(ROW_GAP);
        }),
    );
    later_section(
        ui,
        state,
        "ai-style",
        "Style",
        Some("Turn the clip into a style in one step."),
        Some(&mut |ui: &mut Ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::splat(6.0);
                for (index, label) in KINDS.iter().enumerate() {
                    ui.add(ui::Choice::new(
                        index == 0,
                        RichText::new(*label).size(12.0),
                    ));
                }
            });
            ui.add_space(ROW_GAP);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::splat(8.0);
                none_tile(ui, true);
                for name in TRENDING {
                    name_tile(ui, name, false);
                }
            });
            ui.add_space(ROW_GAP);
        }),
    );
}

/// An empty prompt box with its hint.
fn prompt_box(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 84.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(8), BOX);
    ui.painter().text(
        rect.left_top() + Vec2::new(10.0, 10.0),
        Align2::LEFT_TOP,
        "Describe the look you want",
        FontId::proportional(12.0),
        tokens::FAINT,
    );
    ui.add_space(ROW_GAP);
}
