//! On and off: a check box with its label, and a switch.
//!
//! Both change the `bool` they are given and mark the response changed, as
//! egui's checkbox does. Hover changes a color, never the size. A switch's
//! knob slides rather than jumps.

use egui::{
    Align2, Color32, CornerRadius, FontId, Response, Sense, Stroke, StrokeKind, Ui, Vec2, Widget,
    pos2, vec2,
};

use crate::tokens;

/// A check box, with its label clickable too.
///
/// ```ignore
/// ui.add(ui::Check::new(&mut settings.memory, "Remember across conversations"));
/// ```
#[must_use = "add the check box with `ui.add`"]
pub struct Check<'a> {
    on: &'a mut bool,
    label: String,
}

impl<'a> Check<'a> {
    /// An empty label draws the box alone.
    pub fn new(on: &'a mut bool, label: impl Into<String>) -> Self {
        Self {
            on,
            label: label.into(),
        }
    }
}

/// A check box's fill, edge and mark colors.
fn check_look(on: bool, enabled: bool, hovered: bool) -> (Color32, Color32, Color32) {
    match (on, enabled) {
        (true, true) => {
            let fill = if hovered {
                tokens::EXPORT_HOVER
            } else {
                tokens::EXPORT
            };
            (fill, fill, tokens::EXPORT_INK)
        }
        (true, false) => (tokens::ACTIVE, tokens::ACTIVE, tokens::FAINT),
        (false, _) => {
            let edge = if enabled && hovered {
                tokens::MUTED
            } else {
                tokens::FAINT
            };
            (Color32::TRANSPARENT, edge, Color32::TRANSPARENT)
        }
    }
}

impl Widget for Check<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let font = FontId::proportional(tokens::TYPE_SIZE);
        let galley = (!self.label.is_empty())
            .then(|| ui.painter().layout_no_wrap(self.label, font, tokens::TEXT));
        let text = galley.as_ref().map_or(Vec2::ZERO, |galley| galley.size());
        let gap = if galley.is_some() {
            tokens::TOGGLE_GAP
        } else {
            0.0
        };
        let size = vec2(
            tokens::CHECK_SIZE + gap + text.x,
            tokens::TOGGLE_ROW.max(text.y),
        );
        let (rect, mut response) = ui.allocate_exact_size(size, Sense::click());
        if response.clicked() {
            *self.on = !*self.on;
            response.mark_changed();
        }
        if ui.is_rect_visible(rect) {
            let enabled = ui.is_enabled();
            let (fill, edge, mark) = check_look(*self.on, enabled, response.hovered());
            let square = egui::Rect::from_min_size(
                pos2(rect.left(), rect.center().y - tokens::CHECK_SIZE / 2.0),
                Vec2::splat(tokens::CHECK_SIZE),
            );
            let painter = ui.painter();
            painter.rect(
                square,
                CornerRadius::same(tokens::CHECK_RADIUS),
                fill,
                Stroke::new(tokens::HAIRLINE, edge),
                StrokeKind::Inside,
            );
            if *self.on {
                let c = square.center();
                painter.line(
                    vec![
                        c + vec2(-4.0, 0.0),
                        c + vec2(-1.0, 3.0),
                        c + vec2(4.5, -3.0),
                    ],
                    Stroke::new(1.8, mark),
                );
            }
            if let Some(galley) = galley {
                let ink = if enabled { tokens::TEXT } else { tokens::FAINT };
                let at = pos2(
                    square.right() + gap,
                    rect.center().y - galley.size().y / 2.0,
                );
                painter.galley(at, galley, ink);
            }
        }
        response
    }
}

/// A switch: a pill with a knob that slides to the right when on.
///
/// ```ignore
/// ui.add(ui::Switch::new(&mut uniform));
/// ```
#[must_use = "add the switch with `ui.add`"]
pub struct Switch<'a> {
    on: &'a mut bool,
}

impl<'a> Switch<'a> {
    pub fn new(on: &'a mut bool) -> Self {
        Self { on }
    }
}

/// A switch's track color.
fn switch_track(on: bool, enabled: bool, hovered: bool) -> Color32 {
    match (on, enabled, hovered) {
        (true, true, true) => tokens::EXPORT_HOVER,
        (true, true, false) => tokens::EXPORT,
        (true, false, _) => tokens::ACTIVE,
        (false, true, true) => tokens::SWITCH_OFF_HOVER,
        (false, _, _) => tokens::SWITCH_OFF,
    }
}

impl Widget for Switch<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let size = vec2(tokens::SWITCH_WIDTH, tokens::SWITCH_HEIGHT);
        let (rect, mut response) = ui.allocate_exact_size(size, Sense::click());
        if response.clicked() {
            *self.on = !*self.on;
            response.mark_changed();
        }
        if ui.is_rect_visible(rect) {
            let enabled = ui.is_enabled();
            let along =
                ui.ctx()
                    .animate_bool_with_time(response.id, *self.on, tokens::SWITCH_SLIDE);
            let radius = rect.height() / 2.0;
            let painter = ui.painter();
            painter.rect_filled(
                rect,
                CornerRadius::same(radius as u8),
                switch_track(*self.on, enabled, response.hovered()),
            );
            let knob = radius - tokens::SWITCH_KNOB_INSET;
            let x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), along);
            let ink = if enabled {
                Color32::WHITE
            } else {
                tokens::MUTED
            };
            painter.circle_filled(pos2(x, rect.center().y), knob, ink);
        }
        response
    }
}

/// A label at the left and a switch at the right of the row.
pub fn switch_row(ui: &mut Ui, label: &str, on: &mut bool) -> Response {
    ui.horizontal(|ui| {
        let ink = if ui.is_enabled() {
            tokens::TEXT
        } else {
            tokens::FAINT
        };
        let (rect, _) = ui.allocate_exact_size(
            vec2(
                (ui.available_width() - tokens::SWITCH_WIDTH - 8.0).max(0.0),
                tokens::TOGGLE_ROW,
            ),
            Sense::hover(),
        );
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(12.0),
            ink,
        );
        ui.add(Switch::new(on))
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, Modifiers, PointerButton, Pos2, Rect};

    fn click_frames(add: impl Fn(&mut Ui) -> Response) -> (bool, Rect) {
        let ctx = egui::Context::default();
        let changed = std::cell::Cell::new(false);
        let rect = std::cell::Cell::new(Rect::NOTHING);
        let frame = |events: Vec<Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let response = add(ui);
                    changed.set(changed.get() | response.changed());
                    rect.set(response.rect);
                },
            );
            output.textures_delta.clear();
        };
        frame(vec![]);
        let at = rect.get().right_center() - vec2(3.0, 0.0);
        let press = |pressed| Event::PointerButton {
            pos: at,
            button: PointerButton::Primary,
            pressed,
            modifiers: Modifiers::NONE,
        };
        frame(vec![Event::PointerMoved(at)]);
        frame(vec![press(true)]);
        frame(vec![press(false)]);
        (changed.get(), rect.get())
    }

    #[test]
    fn clicking_the_label_flips_the_check_box() {
        let on = std::cell::Cell::new(false);
        let (changed, rect) = click_frames(|ui| {
            let mut value = on.get();
            let response = ui.add(Check::new(&mut value, "Loop at the end"));
            on.set(value);
            response
        });
        assert!(changed && on.get());
        assert!(rect.width() > tokens::CHECK_SIZE + 40.0, "{rect:?}");
    }

    #[test]
    fn a_switch_flips_and_keeps_its_size() {
        let on = std::cell::Cell::new(true);
        let (changed, rect) = click_frames(|ui| {
            let mut value = on.get();
            let response = ui.add(Switch::new(&mut value));
            on.set(value);
            response
        });
        assert!(changed && !on.get());
        assert_eq!(
            rect.size(),
            vec2(tokens::SWITCH_WIDTH, tokens::SWITCH_HEIGHT)
        );
    }

    #[test]
    fn looks_follow_the_state() {
        assert_eq!(check_look(true, true, false).0, tokens::EXPORT);
        assert_eq!(check_look(false, true, true).1, tokens::MUTED);
        assert_eq!(check_look(false, false, true).1, tokens::FAINT);
        assert_eq!(switch_track(false, true, false), tokens::SWITCH_OFF);
        assert_eq!(switch_track(true, false, false), tokens::ACTIVE);
    }
}
