//! A video clip's Remove BG page: Auto removal, Stroke, Custom removal and
//! Chroma key, laid out as the reference lays them out.
//!
//! Chroma key and click-to-select already edit the clip. Every other
//! control is drawn where it will live, dimmed, with a "Later" mark.

use egui::{RichText, Ui, Vec2};

use crate::shell::{Inspection, Shell};
use crate::{tokens, ui};

use super::audio::{later_section, later_slider, name_tile, right_button};
use super::inspector::{
    InspectorState, LATER, ROW_GAP, body, color_row, later_badge, none_tile, note, row_label,
    section, send, slider,
};

/// The outlines Stroke can draw around what is kept.
const STROKES: [&str; 7] = [
    "Paper",
    "Glow",
    "Hand-drawn",
    "Straight",
    "Solid",
    "Offset",
    "Dotted",
];

/// The tools Custom removal paints with.
const BRUSHES: [&str; 4] = ["Smart brush", "Smart eraser", "Brush", "Eraser"];

/// A key's strength when it is first switched on, or reset.
const KEY_STRENGTH: f64 = 0.4;

pub(super) fn remove_bg_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    later_section(
        ui,
        state,
        "auto-removal",
        "Auto removal",
        Some("Take the background off from behind the person or object."),
        None,
    );
    later_section(
        ui,
        state,
        "stroke",
        "Stroke",
        Some("Outline what is kept."),
        Some(&mut |ui: &mut Ui| {
            stroke_tiles(ui);
            later_slider(ui, "Size", 35.0);
            later_slider(ui, "Spacing", 25.0);
            later_slider(ui, "Object texture", 60.0);
        }),
    );
    custom_removal(shell, ui, state, inspection);
    chroma_key(shell, ui, state, inspection);
}

/// None, then each outline, by name.
fn stroke_tiles(ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(8.0);
        none_tile(ui, true);
        for name in STROKES {
            name_tile(ui, name, false);
        }
    });
    ui.add_space(ROW_GAP);
}

/// Custom removal: click the object on the preview and the clip is cut
/// down to it; the switch takes the selection off. The brushes that paint
/// the selection by hand come later.
fn custom_removal(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let picking = shell.picking().is_some();
    let header = section(
        ui,
        state,
        "custom-removal",
        "Custom removal",
        Some((
            inspection.selected || picking,
            inspection.selected || picking || shell.pick_enabled(),
        )),
        false,
    );
    if header.toggled {
        if picking {
            shell.stop_picking();
        } else if inspection.selected {
            send(shell, Shell::clear_selection);
        } else {
            send(shell, Shell::start_picking);
        }
    }
    if !header.open {
        return;
    }
    body(ui, |ui| {
        if picking {
            note(
                ui,
                "Click the object on the preview. Shift-click to take a part out.",
            );
        } else if !shell.vision_model_ready() {
            note(ui, "The selection model is not on this machine yet.");
        }
        ui.horizontal(|ui| {
            row_label(ui, "Select");
            if picking {
                if ui.add(ui::Button::new("Done").text_size(12.0)).clicked() {
                    shell.stop_picking();
                }
            } else if ui
                .add_enabled(
                    shell.pick_enabled(),
                    ui::Button::new("Select object").text_size(12.0),
                )
                .on_disabled_hover_text("Put the playhead on an unlocked video clip.")
                .clicked()
            {
                send(shell, Shell::start_picking);
            }
            if ui
                .add_enabled(
                    inspection.selected,
                    ui::Button::new("Clear").text_size(12.0),
                )
                .clicked()
            {
                send(shell, Shell::clear_selection);
            }
        });
        ui.add_space(ROW_GAP);
        ui.add_enabled_ui(false, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Paint by hand").size(12.0));
                later_badge(ui);
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::splat(6.0);
                for brush in BRUSHES {
                    ui.add(ui::Button::new(brush).text_size(11.0));
                }
            });
            ui.add_space(ROW_GAP);
            later_slider(ui, "Size", 20.0);
            right_button(ui, "Apply");
        })
        .response
        .on_disabled_hover_text(LATER);
    });
}

/// Chroma key: the switch keys green out or takes the key off; the colour,
/// intensity and shadow edit it. Feather edge and Clean up edge come later.
fn chroma_key(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    let key = inspection.chroma;
    let header = section(
        ui,
        state,
        "chroma",
        "Chroma key",
        Some((key.is_some(), true)),
        key.is_some(),
    );
    if header.toggled {
        if key.is_some() {
            send(shell, |shell| shell.set_chroma("", 0.0, 0.0));
        } else {
            state.open.insert("chroma".to_owned());
            send(shell, |shell| {
                shell.set_chroma("#00ff00", KEY_STRENGTH, 0.0)
            });
        }
    }
    let Some(key) = key else {
        // Off, the controls show dimmed at the values the switch starts
        // with, as in the reference.
        if header.open {
            body(ui, |ui| {
                ui.add_enabled_ui(false, |ui| {
                    ui.horizontal(|ui| {
                        row_label(ui, "Color picker");
                        ui.label(RichText::new("Green").size(12.0).color(tokens::MUTED));
                    });
                    ui.add_space(ROW_GAP);
                    later_slider(ui, "Intensity", KEY_STRENGTH * 100.0);
                    later_slider(ui, "Shadow", 0.0);
                    later_slider(ui, "Feather edge", 0.0);
                    later_slider(ui, "Clean up edge", 0.0);
                })
                .response
                .on_disabled_hover_text("Switch Chroma key on to key out a colour.");
            });
        }
        return;
    };
    if header.reset {
        let hex = key.hex();
        send(shell, |shell| shell.set_chroma(&hex, KEY_STRENGTH, 0.0));
    }
    if !header.open && !state.open.contains("chroma") {
        return;
    }
    body(ui, |ui| {
        let (strength, shadow) = (f64::from(key.strength), f64::from(key.shadow));
        if let Some(color) = color_row(ui, state, "chroma-color", "Color picker", &key.hex()) {
            send(shell, |shell| shell.set_chroma(&color, strength, shadow));
        }
        ui.horizontal(|ui| {
            row_label(ui, "");
            for (name, hex) in [("Green", "#00ff00"), ("Blue", "#0000ff")] {
                if ui.add(ui::Button::new(name).text_size(12.0)).clicked() {
                    send(shell, |shell| shell.set_chroma(hex, strength, shadow));
                }
            }
        });
        ui.add_space(ROW_GAP);
        let hex = key.hex();
        if let Some(value) = slider(
            ui,
            state,
            "chroma-strength",
            "Intensity",
            (strength * 100.0).round(),
            0.0..=100.0,
            |v| format!("{v:.0}"),
        ) {
            send(shell, |shell| shell.set_chroma(&hex, value / 100.0, shadow));
        }
        if let Some(value) = slider(
            ui,
            state,
            "chroma-shadow",
            "Shadow",
            (shadow * 100.0).round(),
            0.0..=100.0,
            |v| format!("{v:.0}"),
        ) {
            send(shell, |shell| {
                shell.set_chroma(&hex, strength, value / 100.0)
            });
        }
        for label in ["Feather edge", "Clean up edge"] {
            ui.add_enabled_ui(false, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(label).size(12.0));
                    later_badge(ui);
                });
                later_slider(ui, "", 0.0);
            })
            .response
            .on_disabled_hover_text(LATER);
        }
    });
}
