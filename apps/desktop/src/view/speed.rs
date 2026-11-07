//! The Speed tab: a constant speed, or a curve picked from the presets and
//! shaped by dragging its points, and whether the pitch follows the speed.
//!
//! A curve point is held in the view while it is dragged and sent once the
//! pointer lets go, so one drag is one undo step and the timeline moves
//! once.

use egui::{Color32, CornerRadius, Pos2, Rect, RichText, Sense, Shape, Stroke, Ui, Vec2, pos2};

use cutlass_model::{SpeedPoint, SpeedPreset, curve_speed_at};

use crate::shell::{Inspection, Shell};
use crate::{tokens, ui};

use super::inspector::{InspectorState, body, note, pages, send, slider, switch_row};

const GRID: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
const LINE: Color32 = Color32::from_rgb(0x2a, 0xd4, 0xe0);
/// The slowest and fastest a curve point runs.
const SLOWEST: f64 = 0.1;
const FASTEST: f64 = 10.0;
/// The most points a curve holds.
const MOST_POINTS: usize = 10;
/// What Custom starts from: even speed, with points to drag.
const EVEN: [(f64, f64); 5] = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (1.0, 1.0)];

/// The tab: Normal for a constant speed, Curve for a curve.
pub(super) fn speed_tab(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    pages(ui, &["Normal", "Curve"], &mut state.page);
    if state.page == 1 {
        curve_page(shell, ui, state, inspection);
    } else {
        normal_page(shell, ui, state, inspection);
    }
    body(ui, |ui| {
        if switch_row(ui, "Change pitch", inspection.change_pitch, true)
            .on_hover_text("Let the voice go higher when faster and lower when slower")
            .clicked()
        {
            let on = !inspection.change_pitch;
            send(shell, |shell| shell.set_change_pitch(on));
        }
    });
}

fn normal_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    body(ui, |ui| {
        if inspection.speed_curve.is_some() {
            note(ui, "The clip runs on a curve; a speed here replaces it.");
        }
        let speed = (inspection.speed * 10.0).round() / 10.0;
        if let Some(value) = slider(ui, state, "speed", "Speed", speed, 0.1..=4.0, |v| {
            format!("{v:.1}×")
        }) {
            send(shell, |shell| shell.set_speed(value));
        }
    });
}

/// Which tile a curve shows as: a preset, or Custom for any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pick {
    None,
    Preset(SpeedPreset),
    Custom,
}

fn picked(curve: Option<&[SpeedPoint]>) -> Pick {
    match curve {
        None => Pick::None,
        Some(points) => SpeedPreset::of(points).map_or(Pick::Custom, Pick::Preset),
    }
}

fn curve_page(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    let curve = inspection.speed_curve.as_deref();
    let now = picked(curve);
    body(ui, |ui| {
        let mut chosen = None;
        ui.horizontal_wrapped(|ui| {
            let tiles = std::iter::once(Pick::None)
                .chain(SpeedPreset::ALL.into_iter().map(Pick::Preset))
                .chain(std::iter::once(Pick::Custom));
            for pick in tiles {
                let label = match pick {
                    Pick::None => "None",
                    Pick::Preset(preset) => preset.label(),
                    Pick::Custom => "Custom",
                };
                if ui
                    .add(ui::Choice::new(
                        now == pick,
                        RichText::new(label).size(12.0),
                    ))
                    .clicked()
                {
                    chosen = Some(pick);
                }
            }
        });
        match chosen {
            Some(pick) if pick == now => {}
            Some(Pick::None) => {
                state.speed_drag = None;
                let speed = inspection.speed;
                send(shell, |shell| shell.set_speed(speed));
            }
            Some(Pick::Preset(preset)) => {
                state.speed_drag = None;
                send(shell, |shell| shell.set_speed_curve(&preset.points()));
            }
            Some(Pick::Custom) => {
                state.speed_drag = None;
                let even: Vec<SpeedPoint> =
                    EVEN.iter().map(|&(x, y)| SpeedPoint::new(x, y)).collect();
                send(shell, |shell| shell.set_speed_curve(&even));
            }
            None => {}
        }
        let Some(curve) = curve else {
            ui.add_space(6.0);
            note(
                ui,
                "Pick a curve to speed the clip up and slow it down as it plays.",
            );
            return;
        };
        ui.add_space(8.0);
        if let Some(points) = curve_graph(ui, state, curve) {
            send(shell, |shell| shell.set_speed_curve(&points));
        }
        ui.add_space(4.0);
        #[allow(clippy::cast_precision_loss)]
        let seconds = inspection.duration.0 as f64 / 1_000_000.0;
        note(
            ui,
            &format!(
                "Length {seconds:.1}s. Drag a point; click to add one, double-click to take one away."
            ),
        );
    });
}

/// Where a speed sits up the graph, 0 at the slowest and 1 at the fastest,
/// on a log scale so 1× is in the middle.
fn height_of(speed: f64) -> f64 {
    (speed.clamp(SLOWEST, FASTEST).log10() + 1.0) / 2.0
}

fn speed_of(height: f64) -> f64 {
    10_f64.powf(height.clamp(0.0, 1.0) * 2.0 - 1.0)
}

/// The curve to drag. Answers the points to send once a change is done.
fn curve_graph(
    ui: &mut Ui,
    state: &mut InspectorState,
    stored: &[SpeedPoint],
) -> Option<Vec<SpeedPoint>> {
    let width = ui.available_width().min(280.0);
    let size = Vec2::new(width, 140.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let frame = Frame { rect };
    let mut points: Vec<(f64, f64)> = state.speed_drag.as_ref().map_or_else(
        || stored.iter().map(|point| (point.x, point.y)).collect(),
        |(_, points)| points.clone(),
    );
    let near = |points: &[(f64, f64)], pos: Pos2| {
        points
            .iter()
            .position(|point| frame.screen(*point).distance(pos) < 10.0)
    };
    let mut done = false;
    if response.double_clicked()
        && let Some(pos) = response.interact_pointer_pos()
        && let Some(index) = near(&points, pos)
        && index > 0
        && index + 1 < points.len()
    {
        points.remove(index);
        state.speed_drag = None;
        done = true;
    } else if response.drag_started()
        && let Some(pos) = ui
            .input(|input| input.pointer.press_origin())
            .or_else(|| response.interact_pointer_pos())
        && let Some(index) = near(&points, pos)
    {
        state.speed_drag = Some((index, points.clone()));
    } else if response.dragged()
        && let Some(pos) = response.interact_pointer_pos()
        && let Some((index, _)) = state.speed_drag
        && index < points.len()
    {
        // The ends stay at the clip's ends; a point stays between its
        // neighbours.
        let (x, y) = frame.curve(pos);
        let x = if index == 0 {
            0.0
        } else if index + 1 == points.len() {
            1.0
        } else {
            x.clamp(points[index - 1].0 + 0.01, points[index + 1].0 - 0.01)
        };
        points[index] = (x, y);
        state.speed_drag = Some((index, points.clone()));
    } else if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
        && near(&points, pos).is_none()
        && points.len() < MOST_POINTS
    {
        let (x, _) = frame.curve(pos);
        if x > 0.0 && x < 1.0 {
            let shared: Vec<SpeedPoint> =
                points.iter().map(|&(x, y)| SpeedPoint::new(x, y)).collect();
            points.push((x, curve_speed_at(&shared, x)));
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            done = true;
        }
    }
    if response.drag_stopped() && state.speed_drag.take().is_some() {
        done = true;
    }
    draw(ui, &frame, &points);
    done.then(|| points.iter().map(|&(x, y)| SpeedPoint::new(x, y)).collect())
}

/// The graph's place on screen.
struct Frame {
    rect: Rect,
}

impl Frame {
    #[allow(clippy::cast_possible_truncation)]
    fn screen(&self, (x, y): (f64, f64)) -> Pos2 {
        pos2(
            self.rect.min.x + x as f32 * self.rect.width(),
            self.rect.max.y - height_of(y) as f32 * self.rect.height(),
        )
    }

    fn curve(&self, pos: Pos2) -> (f64, f64) {
        let x = f64::from((pos.x - self.rect.min.x) / self.rect.width()).clamp(0.0, 1.0);
        let up = f64::from((self.rect.max.y - pos.y) / self.rect.height());
        let speed = (speed_of(up) * 10.0).round() / 10.0;
        (x, speed.clamp(SLOWEST, FASTEST))
    }
}

fn draw(ui: &Ui, frame: &Frame, points: &[(f64, f64)]) {
    let painter = ui.painter_at(frame.rect.expand(2.0));
    painter.rect_filled(
        frame.rect,
        CornerRadius::same(4),
        Color32::from_rgb(0x1e, 0x1e, 0x1e),
    );
    for (speed, label) in [(0.1, "0.1×"), (1.0, "1×"), (10.0, "10×")] {
        let y = frame.screen((0.0, speed)).y;
        let stroke = if speed == 1.0 {
            Stroke::new(1.0, tokens::MUTED)
        } else {
            Stroke::new(1.0, GRID)
        };
        painter.line_segment(
            [pos2(frame.rect.min.x, y), pos2(frame.rect.max.x, y)],
            stroke,
        );
        painter.text(
            pos2(
                frame.rect.min.x + 4.0,
                y + if speed == 10.0 { 2.0 } else { -2.0 },
            ),
            if speed == 10.0 {
                egui::Align2::LEFT_TOP
            } else {
                egui::Align2::LEFT_BOTTOM
            },
            label,
            egui::FontId::proportional(10.0),
            tokens::FAINT,
        );
    }
    let line: Vec<Pos2> = points.iter().map(|point| frame.screen(*point)).collect();
    painter.add(Shape::line(line, Stroke::new(2.0, LINE)));
    for point in points {
        painter.circle_filled(frame.screen(*point), 4.5, Color32::WHITE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_graph_puts_one_times_in_the_middle_and_reads_back_what_it_drew() {
        assert!((height_of(1.0) - 0.5).abs() < 1e-9);
        assert!((height_of(0.1)).abs() < 1e-9);
        assert!((height_of(100.0) - 1.0).abs() < 1e-9);
        let frame = Frame {
            rect: Rect::from_min_size(pos2(10.0, 20.0), Vec2::new(200.0, 100.0)),
        };
        let (x, y) = frame.curve(frame.screen((0.5, 2.0)));
        assert!((x - 0.5).abs() < 1e-3 && (y - 2.0).abs() < 1e-9, "{x} {y}");
        assert_eq!(picked(None), Pick::None);
        assert_eq!(
            picked(Some(&SpeedPreset::Bullet.points())),
            Pick::Preset(SpeedPreset::Bullet)
        );
        let even: Vec<SpeedPoint> = EVEN.iter().map(|&(x, y)| SpeedPoint::new(x, y)).collect();
        assert_eq!(picked(Some(&even)), Pick::Custom);
    }
}
