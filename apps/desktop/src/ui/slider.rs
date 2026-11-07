//! A slider: a thin rail, the part up to the value filled, and a knob.
//!
//! Press anywhere on the rail to jump there and keep dragging. The knob
//! keeps one size; hover only brightens the fill. Values land on a step
//! about one point of rail wide, so a drag gives round numbers.

use egui::{Color32, CornerRadius, Rect, Response, Sense, Ui, Widget, pos2, vec2};

use crate::tokens;

/// A slider over `range`, as wide as asked.
///
/// ```ignore
/// ui.add(ui::Slider::new(&mut opacity, 0.0..=100.0).width(160.0));
/// ```
#[must_use = "add the slider with `ui.add`"]
pub struct Slider<'a> {
    value: &'a mut f64,
    range: std::ops::RangeInclusive<f64>,
    width: Option<f32>,
}

impl<'a> Slider<'a> {
    pub fn new(value: &'a mut f64, range: std::ops::RangeInclusive<f64>) -> Self {
        Self {
            value,
            range,
            width: None,
        }
    }

    /// The rail's width. Without one the slider fills the row.
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }
}

/// The power of ten nearest one point of rail, so drags land on round
/// values.
fn step_for(span: f64, rail: f32) -> f64 {
    let per_point = span / f64::from(rail.max(1.0));
    if per_point <= 0.0 || !per_point.is_finite() {
        return 0.0;
    }
    10f64.powf(per_point.log10().floor())
}

/// Where along the rail, from 0 to 1, `value` sits.
fn fraction(value: f64, low: f64, high: f64) -> f32 {
    if high <= low {
        return 0.0;
    }
    (((value - low) / (high - low)).clamp(0.0, 1.0)) as f32
}

impl Widget for Slider<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let width = self.width.unwrap_or_else(|| ui.available_width());
        let size = vec2(width, tokens::SLIDER_HEIGHT);
        let (rect, mut response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let (low, high) = (*self.range.start(), *self.range.end());
        let knob = tokens::SLIDER_KNOB;
        // The knob's middle travels from one end to the other, never past.
        let left = rect.left() + knob;
        let right = rect.right() - knob;
        let rail = (right - left).max(1.0);

        if let Some(pointer) = response.interact_pointer_pos()
            && (response.dragged() || response.clicked() || response.drag_started())
        {
            let along = f64::from(((pointer.x - left) / rail).clamp(0.0, 1.0));
            let mut value = low + along * (high - low);
            let step = step_for(high - low, rail);
            if step > 0.0 {
                value = (value / step).round() * step;
            }
            let value = value.clamp(low, high);
            if value != *self.value {
                *self.value = value;
                response.mark_changed();
            }
        }

        if ui.is_rect_visible(rect) {
            let enabled = ui.is_enabled();
            let lit = enabled && (response.hovered() || response.dragged());
            let painter = ui.painter();
            let mid = rect.center().y;
            let half = tokens::SLIDER_RAIL / 2.0;
            let whole = Rect::from_min_max(pos2(left, mid - half), pos2(right, mid + half));
            let radius = CornerRadius::same(half.ceil() as u8);
            painter.rect_filled(whole, radius, tokens::SLIDER_TRACK);
            let x = left + rail * fraction(*self.value, low, high);
            let fill = if !enabled {
                tokens::FAINT
            } else if lit {
                tokens::EXPORT_HOVER
            } else {
                tokens::EXPORT
            };
            painter.rect_filled(whole.with_max_x(x), radius, fill);
            let ink = if enabled {
                Color32::WHITE
            } else {
                tokens::MUTED
            };
            painter.circle_filled(pos2(x, mid), knob, ink);
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, Modifiers, PointerButton, Pos2};

    #[test]
    fn steps_are_about_one_point_and_round() {
        assert_eq!(step_for(100.0, 100.0), 1.0);
        assert_eq!(step_for(100.0, 160.0), 0.1);
        assert_eq!(step_for(1000.0, 150.0), 1.0);
        assert!((step_for(3.9, 150.0) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn the_fraction_clamps_to_the_rail() {
        assert_eq!(fraction(50.0, 0.0, 100.0), 0.5);
        assert_eq!(fraction(-5.0, 0.0, 100.0), 0.0);
        assert_eq!(fraction(500.0, 0.0, 100.0), 1.0);
        assert_eq!(fraction(1.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn a_press_jumps_and_a_drag_follows() {
        let ctx = egui::Context::default();
        let mut value = 0.0;
        let rect = std::cell::Cell::new(Rect::NOTHING);
        let frame = |value: &mut f64, events: Vec<Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let wide = 100.0 + 2.0 * tokens::SLIDER_KNOB;
                    rect.set(ui.add(Slider::new(value, 0.0..=100.0).width(wide)).rect);
                },
            );
            output.textures_delta.clear();
        };
        frame(&mut value, vec![]);
        let at = |x: f32| {
            let rect = rect.get();
            pos2(rect.left() + tokens::SLIDER_KNOB + x, rect.center().y)
        };
        let press = |pos, pressed| Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed,
            modifiers: Modifiers::NONE,
        };
        frame(&mut value, vec![Event::PointerMoved(at(25.0))]);
        frame(&mut value, vec![press(at(25.0), true)]);
        frame(&mut value, vec![Event::PointerMoved(at(40.0))]);
        assert_eq!(value, 40.0);
        frame(&mut value, vec![Event::PointerMoved(at(500.0))]);
        assert_eq!(value, 100.0);
        frame(&mut value, vec![press(at(500.0), false)]);
        assert_eq!(rect.get().height(), tokens::SLIDER_HEIGHT);
    }
}
