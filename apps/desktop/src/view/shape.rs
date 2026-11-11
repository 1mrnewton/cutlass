//! A shape's inspector, laid out as the reference lays it out: Vector
//! image, Animation and Tracking.
//!
//! Vector image is the clip's Transform with the shape's own size first,
//! then Fill, Stroke, Corner radius and Shadow, each switched on or off by
//! its box. Each number row keys on its own; Tracking comes later.

use cutlass_model::{
    MaterialShape, SHAPE_FILL, SHAPE_SHADOW, SHAPE_STROKE_WIDTH, ShapeKind, ShapeStyleKey as Key,
};
use egui::{ComboBox, RichText, Ui};

use crate::shell::{Inspection, KeyedSetting, Shell};
use crate::tokens;

use super::inspector::{
    InspectorState, LATER, ROW_GAP, animation_tab, body, color_row, later_badge, opacity_slider,
    pair_row, placement, row_label, section, send, setting_row, slider, tabs,
};

/// The stroke's looks, in the order `border_line_style` numbers them.
const STROKE_STYLES: [&str; 3] = ["Solid", "Dashed", "Dotted"];

/// Sections that start open, as in the reference.
const SECTIONS: [&str; 4] = [
    "shape-fill",
    "shape-stroke",
    "shape-corners",
    "shape-shadow",
];

pub(super) fn shape_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let tab = state.tab;
    tabs(
        ui,
        &["Vector image", "Animation", "Tracking"],
        &mut state.tab,
    );
    if tab != state.tab {
        state.page = 0;
    }
    match state.tab {
        1 => {
            animation_tab(
                shell,
                ui,
                state,
                &["In", "Out", "Loop"],
                &[
                    &["Pop", "Spin", "Fade"],
                    &["Pop", "Spin", "Fade"],
                    &["Pop", "Spin", "Fade"],
                ],
            );
            return;
        }
        2 => {
            tracking_tab(ui);
            return;
        }
        _ => {}
    }
    let Some(shape) = shell.shape() else {
        return;
    };
    for key in SECTIONS {
        if state.open.insert(format!("{key} seen")) {
            state.open.insert(key.to_owned());
        }
    }
    placement(
        shell,
        ui,
        state,
        inspection,
        "Transform",
        Some("Rotate"),
        true,
    );
    body(ui, |ui| opacity_slider(shell, ui, state, inspection));
    let kind = shape.shape_kind();
    // A line, an arrow or an open pen path has no inside to fill.
    let line = shape.is_open();
    if !line {
        fill(shell, ui, state, &shape);
    }
    stroke(shell, ui, state, &shape, line);
    // A circle's sides are already curves, and a line has no corners.
    if !line && kind != Some(ShapeKind::Circle) {
        corners(shell, ui, state, &shape);
    }
    shadow(shell, ui, state, &shape);
}

/// The shape's width and height in canvas pixels; a line has a length.
pub(super) fn size_row(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    shape: &MaterialShape,
) {
    let (width, height) = shape.size();
    let line = shape.is_straight();
    let (new_w, new_h) = setting_row(shell, ui, KeyedSetting::ShapeStyle(Key::Size), true, |ui| {
        if line {
            pair_row(ui, state, "Size", ("W", width.round()), ("", f64::NAN))
        } else {
            pair_row(
                ui,
                state,
                "Size",
                ("W", width.round()),
                ("H", height.round()),
            )
        }
    });
    if new_w.is_some() || new_h.is_some() {
        let width = new_w.unwrap_or(width).max(1.0);
        let height = new_h.unwrap_or(height).max(1.0);
        send(shell, |shell| shell.set_shape_size(width, height));
    }
}

fn fill(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, shape: &MaterialShape) {
    let on = !shape.color.is_empty();
    let header = section(ui, state, "shape-fill", "Fill", Some((on, true)), true);
    if header.toggled {
        send(shell, Shell::toggle_shape_fill);
    }
    if header.reset {
        send(shell, |shell| shell.set_shape_fill(SHAPE_FILL, 1.0));
    }
    if !header.open {
        return;
    }
    let opacity = shape.global_alpha;
    body(ui, |ui| {
        ui.add_enabled_ui(on, |ui| {
            let shown = if on { &shape.color } else { SHAPE_FILL };
            if let Some(color) = color_row(ui, state, "shape-fill-color", "Color", shown) {
                send(shell, |shell| shell.set_shape_fill(&color, opacity));
            }
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::FillOpacity),
                true,
                |ui| percent(ui, state, "shape-fill-opacity", "Opacity", opacity),
            );
            if let Some(value) = changed {
                let color = shape.color.clone();
                send(shell, |shell| shell.set_shape_fill(&color, value));
            }
        });
    });
}

fn stroke(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    shape: &MaterialShape,
    line: bool,
) {
    let on = shape.border_width > 0.0;
    // A line is its stroke, so it cannot be switched off.
    let check = if line { None } else { Some((on, true)) };
    let header = section(ui, state, "shape-stroke", "Stroke", check, true);
    if header.toggled {
        send(shell, Shell::toggle_shape_stroke);
    }
    if header.reset {
        let width = if line || on { SHAPE_STROKE_WIDTH } else { 0.0 };
        send(shell, |shell| {
            shell.set_shape_stroke(SHAPE_FILL, width, 1.0, 0)
        });
    }
    if !header.open {
        return;
    }
    let (color, width, opacity, style) = (
        shape.border_color.clone(),
        shape.border_width,
        shape.border_alpha,
        shape.border_line_style,
    );
    body(ui, |ui| {
        ui.add_enabled_ui(on, |ui| {
            ui.horizontal(|ui| {
                row_label(ui, "Style");
                let current = usize::try_from(style)
                    .unwrap_or(0)
                    .min(STROKE_STYLES.len() - 1);
                let mut picked = current;
                ComboBox::from_id_salt("shape-stroke-style")
                    .width(ui.available_width())
                    .selected_text(STROKE_STYLES[current])
                    .show_ui(ui, |ui| {
                        for (index, name) in STROKE_STYLES.iter().enumerate() {
                            ui.selectable_value(&mut picked, index, *name);
                        }
                    })
                    .response
                    .widget_info(|| {
                        egui::WidgetInfo::labeled(egui::Role::ComboBox, true, "Stroke style")
                    });
                if picked != current {
                    let style = i32::try_from(picked).unwrap_or(0);
                    let color = color.clone();
                    send(shell, |shell| {
                        shell.set_shape_stroke(&color, width, opacity, style)
                    });
                }
            });
            ui.add_space(ROW_GAP);
            if let Some(picked) = color_row(ui, state, "shape-stroke-color", "Color", &color) {
                send(shell, |shell| {
                    shell.set_shape_stroke(&picked, width, opacity, style)
                });
            }
            let shown = if on { width } else { SHAPE_STROKE_WIDTH };
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::StrokeWidth),
                true,
                |ui| {
                    slider(
                        ui,
                        state,
                        "shape-stroke-width",
                        "Weight",
                        shown.round(),
                        1.0..=50.0,
                        |v| format!("{v:.0}"),
                    )
                },
            );
            if let Some(value) = changed {
                let color = color.clone();
                send(shell, |shell| {
                    shell.set_shape_stroke(&color, value, opacity, style)
                });
            }
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::StrokeOpacity),
                true,
                |ui| percent(ui, state, "shape-stroke-opacity", "Opacity", opacity),
            );
            if let Some(value) = changed {
                let color = color.clone();
                send(shell, |shell| {
                    shell.set_shape_stroke(&color, width, value, style)
                });
            }
        });
    });
}

fn corners(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, shape: &MaterialShape) {
    let radius = shape.roundness.iter().copied().fold(0.0_f64, f64::max);
    let on = radius > 0.0;
    let header = section(
        ui,
        state,
        "shape-corners",
        "Corner radius",
        Some((on, true)),
        true,
    );
    let (width, height) = shape.size();
    let most = (width.min(height) / 2.0).floor().max(1.0);
    if header.toggled {
        // On, the corners take a fifth of the most they can.
        let radius = if on {
            0.0
        } else {
            (most / 5.0).round().max(1.0)
        };
        send(shell, |shell| shell.set_shape_corners(vec![radius]));
    }
    if header.reset {
        send(shell, |shell| shell.set_shape_corners(vec![0.0]));
    }
    if !header.open {
        return;
    }
    body(ui, |ui| {
        ui.add_enabled_ui(on, |ui| {
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::Corners),
                true,
                |ui| {
                    slider(
                        ui,
                        state,
                        "shape-corner-radius",
                        "Corner radius",
                        radius.round(),
                        0.0..=most,
                        |v| format!("{v:.0}"),
                    )
                },
            );
            if let Some(value) = changed {
                send(shell, |shell| shell.set_shape_corners(vec![value]));
            }
        });
        ui.add_enabled_ui(false, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Each corner").size(12.0));
                later_badge(ui);
            });
        })
        .response
        .on_disabled_hover_text(LATER);
        ui.add_space(ROW_GAP);
    });
}

fn shadow(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, shape: &MaterialShape) {
    let on = shape.shadow_alpha > 0.0;
    let header = section(ui, state, "shape-shadow", "Shadow", Some((on, true)), true);
    if header.toggled {
        send(shell, Shell::toggle_shape_shadow);
    }
    let (start_distance, start_angle, start_opacity) = SHAPE_SHADOW;
    if header.reset && on {
        send(shell, |shell| {
            shell.set_shape_shadow("#000000", start_distance, start_angle, start_opacity)
        });
    }
    if !header.open {
        return;
    }
    // Off, the controls show where the switch starts them.
    let (color, distance, angle, opacity) = if on {
        (
            shape.shadow_color.clone(),
            shape.shadow_distance,
            shape.shadow_angle,
            shape.shadow_alpha,
        )
    } else {
        (
            "#000000".to_owned(),
            start_distance,
            start_angle,
            start_opacity,
        )
    };
    body(ui, |ui| {
        ui.add_enabled_ui(on, |ui| {
            if let Some(picked) = color_row(ui, state, "shape-shadow-color", "Color", &color) {
                send(shell, |shell| {
                    shell.set_shape_shadow(&picked, distance, angle, opacity)
                });
            }
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::ShadowDistance),
                true,
                |ui| {
                    slider(
                        ui,
                        state,
                        "shape-shadow-distance",
                        "Distance",
                        distance.round(),
                        0.0..=100.0,
                        |v| format!("{v:.0}"),
                    )
                },
            );
            if let Some(value) = changed {
                let color = color.clone();
                send(shell, |shell| {
                    shell.set_shape_shadow(&color, value, angle, opacity)
                });
            }
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::ShadowAngle),
                true,
                |ui| {
                    slider(
                        ui,
                        state,
                        "shape-shadow-angle",
                        "Angle",
                        angle.round(),
                        -180.0..=180.0,
                        |v| format!("{v:.0}°"),
                    )
                },
            );
            if let Some(value) = changed {
                let color = color.clone();
                send(shell, |shell| {
                    shell.set_shape_shadow(&color, distance, value, opacity)
                });
            }
            let changed = setting_row(
                shell,
                ui,
                KeyedSetting::ShapeStyle(Key::ShadowOpacity),
                true,
                |ui| percent(ui, state, "shape-shadow-opacity", "Opacity", opacity),
            );
            if let Some(value) = changed {
                let color = color.clone();
                send(shell, |shell| {
                    shell.set_shape_shadow(&color, distance, angle, value.max(0.01))
                });
            }
        });
    });
}

/// A 0 to 1 amount as a percent slider; the new amount, 0 to 1.
fn percent(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    stored: f64,
) -> Option<f64> {
    slider(
        ui,
        state,
        key,
        label,
        (stored * 100.0).round(),
        0.0..=100.0,
        |v| format!("{v:.0}%"),
    )
    .map(|value| value / 100.0)
}

/// Tracking sits on the item that follows, as in the reference; it comes
/// later.
fn tracking_tab(ui: &mut Ui) {
    ui.add_space(8.0);
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Tracking").size(13.0).color(tokens::TEXT));
            later_badge(ui);
        });
        ui.add_space(ROW_GAP);
        ui.horizontal(|ui| {
            super::inspector::none_tile(ui, true);
            super::audio::name_tile(ui, "Motion tracking", false);
        });
    })
    .response
    .on_disabled_hover_text(LATER);
}
