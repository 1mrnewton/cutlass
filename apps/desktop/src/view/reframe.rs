//! A video clip's Auto reframe section: the shape to cut to, how still the
//! cut holds and how fast it follows, and Apply.
//!
//! The choices wait until Apply, since a clip's first Apply searches its
//! file for the subject as a job, whose progress shows here; the box turns
//! it on with the choices shown and off at once.

use egui::{Rect, Stroke, Ui, Vec2};

use cutlass_model::Reframe;

use crate::shell::Shell;
use crate::{tokens, ui};

use super::inspector::{InspectorState, body, label_row, note, section, send};

/// What each choice is called.
const STABILITY: [(i64, &str); 3] = [
    (Reframe::FLEXIBLE, "Flexible"),
    (Reframe::DEFAULT, "Default"),
    (Reframe::STABLE, "Stable"),
];
const SPEED: [(i64, &str); 3] = [
    (Reframe::FAST, "Fast"),
    (Reframe::DEFAULT, "Default"),
    (Reframe::SLOW, "Slow"),
];

/// The shape a fresh Auto reframe starts at: upright, for phones.
const FIRST_RATIO: &str = "9:16";

/// What a ratio is called.
fn ratio_name(ratio: &str) -> &str {
    if ratio == Reframe::ORIGINAL {
        "Original"
    } else {
        ratio
    }
}

/// The choices shown: those waiting for Apply, else the clip's own, else
/// the first ones.
fn shown(state: &InspectorState, worn: Option<&Reframe>) -> (String, i64, i64) {
    let field = |key: &str| state.fields.get(key).filter(|value| !value.is_empty());
    let ratio = field("reframe ratio")
        .cloned()
        .or_else(|| worn.map(|cut| cut.ratio.clone()))
        .unwrap_or_else(|| FIRST_RATIO.to_owned());
    let number = |key: &str, own: Option<i64>| {
        field(key)
            .and_then(|value| value.parse().ok())
            .or(own)
            .unwrap_or(Reframe::DEFAULT)
    };
    (
        ratio,
        number("reframe stability", worn.map(|cut| cut.stability)),
        number("reframe speed", worn.map(|cut| cut.speed)),
    )
}

/// The Auto reframe section of a video clip's Basic page.
pub(super) fn auto_reframe(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let worn = shell.reframe();
    let header = section(
        ui,
        state,
        "reframe",
        "Auto reframe",
        Some((worn.is_some(), true)),
        false,
    );
    let (ratio, stability, speed) = shown(state, worn.as_ref());
    if header.toggled {
        if worn.is_some() {
            send(shell, |shell| shell.set_reframe("", stability, speed));
        } else {
            send(shell, move |shell| {
                shell.set_reframe(&ratio, stability, speed)
            });
        }
        return;
    }
    if !header.open {
        return;
    }
    let progress = shell.reframe_progress();
    let mut apply = false;
    body(ui, |ui| {
        if let Some(done) = progress {
            note(ui, &format!("Finding the subject {:.1}%", done * 100.0));
        }
        let mut picked_ratio = None;
        label_row(ui, "Aspect ratio", |ui| {
            egui::ComboBox::from_id_salt("reframe-ratio")
                .width(ui.available_width())
                .selected_text(ratio_name(&ratio))
                .show_ui(ui, |ui| {
                    for (index, choice) in Reframe::RATIOS.iter().enumerate() {
                        // Original, then the wide shapes, then the tall.
                        if index == 1 || *choice == "9:16" {
                            ui.separator();
                        }
                        let response =
                            ui.add(ui::Choice::new(*choice == ratio, ratio_name(choice)).wide());
                        if *choice != Reframe::ORIGINAL
                            && let Some(aspect) = Reframe::aspect_of(choice, (1, 1))
                        {
                            shape_mark(ui, response.rect, aspect as f32);
                        }
                        if response.clicked() {
                            picked_ratio = Some(*choice);
                        }
                    }
                });
        });
        if let Some(choice) = picked_ratio {
            state
                .fields
                .insert("reframe ratio".into(), choice.to_owned());
        }
        for (key, label, choices, current) in [
            (
                "reframe stability",
                "Image stabilization",
                STABILITY,
                stability,
            ),
            ("reframe speed", "Camera moving speed", SPEED, speed),
        ] {
            let mut picked = None;
            let name = choices
                .iter()
                .find(|(value, _)| *value == current)
                .map_or("Default", |(_, name)| *name);
            label_row(ui, label, |ui| {
                egui::ComboBox::from_id_salt(key)
                    .width(ui.available_width())
                    .selected_text(name)
                    .show_ui(ui, |ui| {
                        for (value, name) in choices {
                            if ui
                                .add(ui::Choice::new(value == current, name).wide())
                                .clicked()
                            {
                                picked = Some(value);
                            }
                        }
                    });
            });
            if let Some(value) = picked {
                state.fields.insert(key.into(), value.to_string());
            }
        }
        ui.add_space(6.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let waiting = worn.as_ref().is_none_or(|cut| {
                (cut.ratio.as_str(), cut.stability, cut.speed) != (ratio.as_str(), stability, speed)
            });
            apply = ui
                .add_enabled(waiting, ui::Button::new("Apply").text_size(12.0))
                .on_hover_text("Cut the clip to this shape around its subject")
                .clicked();
        });
    });
    if apply {
        for key in ["reframe ratio", "reframe stability", "reframe speed"] {
            state.fields.remove(key);
        }
        send(shell, move |shell| {
            shell.set_reframe(&ratio, stability, speed)
        });
    }
}

/// A small outline of a shape `aspect` wide over high at the right of a
/// choice's `row`.
fn shape_mark(ui: &Ui, row: Rect, aspect: f32) {
    let tall = 14.0;
    let size = if aspect >= 1.0 {
        Vec2::new((tall * aspect).min(28.0), (28.0 / aspect).min(tall))
    } else {
        Vec2::new(tall * aspect.max(0.3), tall)
    };
    let middle = egui::pos2(row.right() - 22.0, row.center().y);
    ui.painter().rect_stroke(
        Rect::from_center_size(middle, size),
        egui::CornerRadius::same(1),
        Stroke::new(1.0, tokens::MUTED),
        egui::StrokeKind::Inside,
    );
}
