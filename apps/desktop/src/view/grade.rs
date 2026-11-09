//! The colour pages of an Adjust tab: HSL, curves and colour wheels.
//!
//! Each control sends one request through the shell as it moves; the view
//! holds edits while the pointer is down, so one drag is one undo step.

use egui::{Color32, CornerRadius, Pos2, RichText, Sense, Shape, Stroke, Ui, Vec2, pos2};

use cutlass_effects::{CurveChannel, Grade, HSL_COLORS, Wheel, WheelSetting, curve_table};

use crate::shell::{Inspection, Shell};
use crate::{tokens, ui};

use super::audio::{choice_row, later_value};
use super::inspector::{InspectorState, LATER, later_badge, note, send, slider};

/// Where each HSL colour's swatch sits on the hue circle, in degrees.
const SWATCH_HUES: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 275.0, 310.0];
const GRID: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);

/// The HSL page: a swatch for each colour, then its three sliders.
pub(super) fn hsl_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        for (index, name) in HSL_COLORS.iter().enumerate() {
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(22.0), Sense::click());
            let set = inspection.grade.hsl[index]
                .iter()
                .any(|value| *value != 0.0);
            ui.painter()
                .circle_filled(rect.center(), 9.0, hue_color(SWATCH_HUES[index]));
            if state.hsl_color == index {
                ui.painter()
                    .circle_stroke(rect.center(), 11.0, Stroke::new(2.0, Color32::WHITE));
            } else if set {
                ui.painter()
                    .circle_stroke(rect.center(), 11.0, Stroke::new(1.0, tokens::MUTED));
            }
            if response.on_hover_text(*name).clicked() {
                state.hsl_color = index;
            }
        }
    });
    ui.add_space(6.0);
    let color = state.hsl_color.min(7);
    note(ui, HSL_COLORS[color]);
    let stored = inspection.grade.hsl[color];
    let mut values = stored.map(f64::from);
    let mut moved = false;
    for (index, label) in ["Hue", "Saturation", "Lightness"].into_iter().enumerate() {
        let shown = (values[index] * 100.0).round();
        let key = format!("hsl-{color}-{index}");
        if let Some(value) = slider(ui, state, &key, label, shown, -100.0..=100.0, |v| {
            format!("{v:.0}")
        }) {
            values[index] = value / 100.0;
            moved = true;
        }
    }
    if moved {
        #[allow(clippy::cast_possible_truncation)]
        let color = color as u8;
        send(shell, |shell| shell.set_hsl(color, values));
    }
    ui.add_space(6.0);
    if ui
        .add_enabled(
            stored.iter().any(|value| *value != 0.0),
            ui::Button::new("Reset color").text_size(12.0),
        )
        .clicked()
    {
        #[allow(clippy::cast_possible_truncation)]
        let color = color as u8;
        send(shell, |shell| shell.set_hsl(color, [0.0; 3]));
    }
}

/// The curves' names, in [`CurveChannel::ALL`] order.
const CURVE_NAMES: [&str; 4] = ["Luma", "Red (R)", "Green (G)", "Blue (B)"];

/// The Curves page: every channel's curve, one under another, each with
/// its own reset, and a reset for all of them. A click adds a point, a
/// drag moves one, a double-click takes one away.
pub(super) fn curves_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    ui.add_space(6.0);
    let bent = |channel| !inspection.grade.curve(channel).is_empty();
    let any = CurveChannel::ALL.into_iter().any(bent);
    if curve_header(ui, "All", any) {
        state.curve_drag = None;
        send(shell, |shell| {
            for channel in CurveChannel::ALL {
                shell.set_curve(channel, &[])?;
            }
            Ok(())
        });
    }
    for (index, channel) in CurveChannel::ALL.into_iter().enumerate() {
        if curve_header(ui, CURVE_NAMES[index], bent(channel)) {
            state.curve_drag = None;
            send(shell, |shell| shell.set_curve(channel, &[]));
        }
        curve_graph(shell, ui, state, inspection, index);
        ui.add_space(8.0);
    }
}

/// A curve's name and its reset; true when the reset is clicked.
fn curve_header(ui: &mut Ui, name: &str, reset: bool) -> bool {
    let mut clicked = false;
    ui.horizontal(|ui| {
        ui.set_min_height(32.0);
        ui.label(RichText::new(name).size(13.0).color(Color32::WHITE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            clicked = ui
                .add_enabled(
                    reset,
                    ui::Button::icon("reset")
                        .min_size(Vec2::splat(24.0))
                        .tip(format!("Reset {name}")),
                )
                .clicked();
        });
    });
    clicked
}

/// One channel's curve to drag, the panel's width across.
fn curve_graph(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    index: usize,
) {
    let channel = CurveChannel::ALL[index];
    let stored = curve_points(&inspection.grade, channel);
    let width = ui.available_width().max(40.0);
    let height = (width * 0.66).max(40.0);
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(width, height), Sense::click_and_drag());
    let to_screen = |(x, y): (f32, f32)| pos2(rect.min.x + x * width, rect.max.y - y * height);
    let to_curve = |pos: Pos2| {
        (
            ((pos.x - rect.min.x) / width).clamp(0.0, 1.0),
            ((rect.max.y - pos.y) / height).clamp(0.0, 1.0),
        )
    };
    // A drag belongs to the curve it started on.
    let dragging = state.curve_channel == index;
    let mut points = state
        .curve_drag
        .as_ref()
        .filter(|_| dragging)
        .map_or_else(|| stored.clone(), |(_, points)| points.clone());
    let near = |points: &[(f32, f32)], pos: Pos2| {
        points
            .iter()
            .position(|point| to_screen(*point).distance(pos) < 10.0)
    };
    let mut changed = false;
    if response.double_clicked()
        && let Some(pos) = response.interact_pointer_pos()
        && let Some(at) = near(&points, pos)
        && points.len() > 2
    {
        points.remove(at);
        changed = true;
    } else if response.drag_started()
        && let Some(pos) = ui
            .input(|input| input.pointer.press_origin())
            .or_else(|| response.interact_pointer_pos())
    {
        let at = near(&points, pos).unwrap_or_else(|| {
            points.push(to_curve(pos));
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            near(&points, pos).unwrap_or_default()
        });
        state.curve_channel = index;
        state.curve_drag = Some((at, points.clone()));
    } else if response.dragged()
        && dragging
        && let Some(pos) = response.interact_pointer_pos()
        && let Some((at, _)) = state.curve_drag
        && at < points.len()
    {
        // A point stays between its neighbours.
        let (mut x, y) = to_curve(pos);
        let low = if at == 0 {
            0.0
        } else {
            points[at - 1].0 + 0.01
        };
        let high = if at + 1 == points.len() {
            1.0
        } else {
            points[at + 1].0 - 0.01
        };
        x = x.clamp(low, high.max(low));
        points[at] = (x, y);
        state.curve_drag = Some((at, points.clone()));
        changed = true;
    } else if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
        && near(&points, pos).is_none()
        && points.len() < cutlass_effects::MAX_CURVE_POINTS
    {
        points.push(to_curve(pos));
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        changed = true;
    }
    if response.drag_stopped() && dragging {
        state.curve_drag = None;
    }
    if changed {
        send(shell, |shell| shell.set_curve(channel, &points));
    }

    let painter = ui.painter_at(rect.expand(6.0));
    painter.rect_filled(
        rect,
        CornerRadius::same(4),
        Color32::from_rgb(0x0b, 0x0b, 0x0d),
    );
    for step in 1..4 {
        #[allow(clippy::cast_precision_loss)]
        let t = step as f32 / 4.0;
        painter.line_segment(
            [to_screen((t, 0.0)), to_screen((t, 1.0))],
            Stroke::new(1.0, GRID),
        );
        painter.line_segment(
            [to_screen((0.0, t)), to_screen((1.0, t))],
            Stroke::new(1.0, GRID),
        );
    }
    let color = channel_color(channel);
    let table = curve_table(&points);
    let line: Vec<Pos2> = table
        .iter()
        .enumerate()
        .map(|(at, y)| {
            #[allow(clippy::cast_precision_loss)]
            let x = at as f32 / 255.0;
            to_screen((x, *y))
        })
        .collect();
    painter.add(Shape::line(line, Stroke::new(1.5, color)));
    for point in &points {
        painter.circle_filled(to_screen(*point), 4.5, color);
    }
}

/// The Color wheel page: shadows, middle grey and highlights (lift, gamma
/// and gain), each a puck to drag inside a hue ring and a slider for its
/// brightness, with a reset each. A double-click on a wheel
/// puts its puck back in the middle.
pub(super) fn wheels_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    ui.add_space(10.0);
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Wheels").size(12.0));
            later_badge(ui);
        });
        choice_row(ui, None, "Primary");
    })
    .response
    .on_disabled_hover_text(LATER);
    later_value(ui, "Intensity", 100.0);
    for wheel in Wheel::ALL {
        let setting = inspection.grade.wheel(wheel);
        let name = wheel_name(wheel);
        let mut reset = false;
        ui.horizontal(|ui| {
            ui.set_min_height(28.0);
            ui.label(RichText::new(name).size(13.0).color(Color32::WHITE));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                reset = ui
                    .add_enabled(
                        setting != WheelSetting::default(),
                        ui::Button::icon("reset")
                            .min_size(Vec2::splat(24.0))
                            .tip(format!("Reset {name}")),
                    )
                    .clicked();
            });
        });
        if reset {
            send(shell, |shell| {
                shell.set_wheel(wheel, WheelSetting::default())
            });
            continue;
        }
        let side = 120.0;
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag());
        let centre = rect.center();
        let radius = side / 2.0 - 6.0;
        let painter = ui.painter_at(rect.expand(2.0));
        for step in 0..72 {
            #[allow(clippy::cast_precision_loss)]
            let (from, to) = (step as f32 * 5.0, (step + 1) as f32 * 5.0);
            let at = |degrees: f32, r: f32| {
                let angle = degrees.to_radians();
                pos2(centre.x + angle.cos() * r, centre.y - angle.sin() * r)
            };
            painter.add(Shape::convex_polygon(
                vec![
                    at(from, radius),
                    at(to, radius),
                    at(to, radius - 6.0),
                    at(from, radius - 6.0),
                ],
                hue_color(from),
                Stroke::NONE,
            ));
        }
        painter.circle_filled(centre, radius - 7.0, Color32::from_rgb(0x1e, 0x1e, 0x1e));
        let puck = pos2(
            centre.x + setting.x * (radius - 7.0),
            centre.y - setting.y * (radius - 7.0),
        );
        painter.circle_stroke(centre, 2.0, Stroke::new(1.0, GRID));
        painter.circle_stroke(puck, 6.0, Stroke::new(2.0, Color32::WHITE));
        let mut next = None;
        if response.double_clicked() {
            next = Some(WheelSetting {
                x: 0.0,
                y: 0.0,
                luma: setting.luma,
            });
        } else if (response.dragged() || response.clicked())
            && let Some(pos) = response.interact_pointer_pos()
        {
            let reach = radius - 7.0;
            next = Some(
                WheelSetting {
                    x: (pos.x - centre.x) / reach,
                    y: (centre.y - pos.y) / reach,
                    luma: setting.luma,
                }
                .held(),
            );
        }
        let shown = (f64::from(setting.luma) * 100.0).round();
        let key = format!("wheel-{}", wheel.key());
        if let Some(value) = slider(ui, state, &key, "Brightness", shown, -100.0..=100.0, |v| {
            format!("{v:.0}")
        }) {
            #[allow(clippy::cast_possible_truncation)]
            let luma = (value / 100.0) as f32;
            next = Some(WheelSetting {
                luma,
                ..next.unwrap_or(setting)
            });
        }
        if let Some(next) = next.filter(|next| *next != setting) {
            send(shell, |shell| shell.set_wheel(wheel, next));
        }
        ui.add_space(8.0);
    }
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Offset").size(13.0));
            later_badge(ui);
        });
        ui.label(
            RichText::new("Shift every tone at once.")
                .size(12.0)
                .color(tokens::MUTED),
        );
    })
    .response
    .on_disabled_hover_text(LATER);
}

/// A wheel's name as the reference shows it.
fn wheel_name(wheel: Wheel) -> &'static str {
    match wheel {
        Wheel::Lift => "Shadows",
        Wheel::Gamma => "Middle grey",
        Wheel::Gain => "Highlights",
    }
}

/// The points the curve editor shows: the stored ones, or the two ends of
/// the straight line.
fn curve_points(grade: &Grade, channel: CurveChannel) -> Vec<(f32, f32)> {
    let stored = grade.curve(channel);
    if stored.len() < 2 {
        vec![(0.0, 0.0), (1.0, 1.0)]
    } else {
        stored.to_vec()
    }
}

fn channel_color(channel: CurveChannel) -> Color32 {
    match channel {
        CurveChannel::Rgb => Color32::WHITE,
        CurveChannel::Red => Color32::from_rgb(0xf0, 0x55, 0x55),
        CurveChannel::Green => Color32::from_rgb(0x55, 0xe0, 0x70),
        CurveChannel::Blue => Color32::from_rgb(0x5a, 0x8c, 0xff),
    }
}

/// Full colour at `hue` degrees.
fn hue_color(hue: f32) -> Color32 {
    let sector = hue.rem_euclid(360.0) / 60.0;
    let second = 1.0 - (sector.rem_euclid(2.0) - 1.0).abs();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (r, g, b) = match sector as u32 {
        0 => (1.0, second, 0.0),
        1 => (second, 1.0, 0.0),
        2 => (0.0, 1.0, second),
        3 => (0.0, second, 1.0),
        4 => (second, 0.0, 1.0),
        _ => (1.0, 0.0, second),
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = |value: f32| (value * 255.0).round() as u8;
    Color32::from_rgb(byte(r), byte(g), byte(b))
}

#[cfg(test)]
mod tests {
    use super::{curve_points, hue_color};
    use cutlass_effects::{CurveChannel, Grade};
    use egui::Color32;

    #[test]
    fn an_empty_curve_shows_its_two_ends() {
        let mut grade = Grade::default();
        assert_eq!(
            curve_points(&grade, CurveChannel::Red),
            [(0.0, 0.0), (1.0, 1.0)]
        );
        grade.curves[1] = vec![(0.0, 0.2), (1.0, 0.8)];
        assert_eq!(
            curve_points(&grade, CurveChannel::Red),
            [(0.0, 0.2), (1.0, 0.8)]
        );
        assert_eq!(hue_color(0.0), Color32::from_rgb(255, 0, 0));
        assert_eq!(hue_color(240.0), Color32::from_rgb(0, 0, 255));
    }
}
