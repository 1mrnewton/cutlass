//! Buttons: a label, an icon, or both, in one of three kinds.
//!
//! A button is an egui widget, so `ui.add(...)` and `ui.add_enabled(...)`
//! work as they do for egui's own. A disabled button looks faint and ignores
//! clicks.

use egui::{
    Align2, Color32, CornerRadius, FontId, Rect, Response, Sense, Ui, Vec2, Widget, pos2, vec2,
};

use crate::{icons, tokens};

/// How loud a button is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonKind {
    /// The one action a place is for, such as Export.
    Primary,
    /// An ordinary button on a fill.
    Secondary,
    /// No fill until hovered or on, for tool strips and rails.
    Ghost,
}

/// A button.
///
/// ```ignore
/// if ui.add(ui::Button::new("Save")).clicked() { /* ... */ }
/// ui.add(ui::Button::primary("Export"));
/// ui.add(ui::Button::icon("split").tip("Split").on(active));
/// ```
#[must_use = "add the button with `ui.add`"]
pub struct Button {
    kind: ButtonKind,
    label: Option<String>,
    icon: Option<String>,
    tip: Option<String>,
    on: bool,
    min_size: Vec2,
    text_size: f32,
    icon_size: f32,
    radius: u8,
    full_width: bool,
}

impl Button {
    fn with(kind: ButtonKind, label: Option<String>, icon: Option<String>) -> Self {
        Self {
            kind,
            label,
            icon,
            tip: None,
            on: false,
            min_size: Vec2::ZERO,
            text_size: tokens::TYPE_SIZE,
            icon_size: 16.0,
            radius: match kind {
                ButtonKind::Primary => tokens::EXPORT_RADIUS,
                _ => tokens::ICON_RADIUS,
            },
            full_width: false,
        }
    }

    /// An ordinary text button.
    pub fn new(label: impl Into<String>) -> Self {
        Self::with(ButtonKind::Secondary, Some(label.into()), None)
    }

    /// The accent button for the one action a place is for.
    pub fn primary(label: impl Into<String>) -> Self {
        Self::with(ButtonKind::Primary, Some(label.into()), None)
    }

    /// A text button with no fill until hovered.
    pub fn ghost(label: impl Into<String>) -> Self {
        Self::with(ButtonKind::Ghost, Some(label.into()), None)
    }

    /// An icon button with no fill until hovered, for tool strips.
    pub fn icon(name: impl Into<String>) -> Self {
        Self::with(ButtonKind::Ghost, None, Some(name.into()))
    }

    pub fn kind(mut self, kind: ButtonKind) -> Self {
        self.kind = kind;
        self
    }

    /// Put an icon before the label.
    pub fn with_icon(mut self, name: impl Into<String>) -> Self {
        self.icon = Some(name.into());
        self
    }

    /// Shown on hover, and on hover while disabled too.
    pub fn tip(mut self, tip: impl Into<String>) -> Self {
        self.tip = Some(tip.into());
        self
    }

    /// Draw it as switched on, for toggles and the current tab or shelf.
    pub fn on(mut self, on: bool) -> Self {
        self.on = on;
        self
    }

    pub fn min_size(mut self, size: Vec2) -> Self {
        self.min_size = size;
        self
    }

    pub fn text_size(mut self, size: f32) -> Self {
        self.text_size = size;
        self
    }

    pub fn icon_size(mut self, size: f32) -> Self {
        self.icon_size = size;
        self
    }

    pub fn radius(mut self, radius: u8) -> Self {
        self.radius = radius;
        self
    }

    /// Take the whole width the layout offers.
    pub fn full_width(mut self) -> Self {
        self.full_width = true;
        self
    }
}

/// A button's fill and ink for one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Look {
    fill: Color32,
    ink: Color32,
}

fn look(kind: ButtonKind, on: bool, enabled: bool, hovered: bool, pressed: bool) -> Look {
    if !enabled {
        let fill = match kind {
            ButtonKind::Ghost if !on => Color32::TRANSPARENT,
            ButtonKind::Ghost => tokens::ACTIVE,
            _ => tokens::RAISED,
        };
        return Look {
            fill,
            ink: tokens::FAINT,
        };
    }
    match kind {
        ButtonKind::Primary => Look {
            fill: if hovered || pressed {
                tokens::EXPORT_HOVER
            } else {
                tokens::EXPORT
            },
            ink: tokens::EXPORT_INK,
        },
        ButtonKind::Secondary => Look {
            fill: if pressed {
                tokens::PRESSED
            } else if on || hovered {
                tokens::ACTIVE
            } else {
                tokens::RAISED
            },
            ink: tokens::TEXT,
        },
        ButtonKind::Ghost => Look {
            fill: if pressed || on {
                tokens::ACTIVE
            } else if hovered {
                tokens::RAISED
            } else {
                Color32::TRANSPARENT
            },
            ink: if on || hovered {
                tokens::TEXT
            } else {
                tokens::MUTED
            },
        },
    }
}

impl Widget for Button {
    fn ui(self, ui: &mut Ui) -> Response {
        let font = FontId::proportional(self.text_size);
        let galley = self.label.map(|label| {
            ui.painter()
                .layout_no_wrap(label, font, Color32::PLACEHOLDER)
        });
        let icon = self.icon.as_ref().map(|_| self.icon_size).unwrap_or(0.0);
        let gap = if galley.is_some() && self.icon.is_some() {
            tokens::BUTTON_GAP
        } else {
            0.0
        };
        let text = galley.as_ref().map_or(Vec2::ZERO, |galley| galley.size());
        let content = vec2(icon + gap + text.x, icon.max(text.y));
        let pad = if galley.is_some() {
            vec2(tokens::BUTTON_PAD_X, tokens::BUTTON_PAD_Y)
        } else {
            Vec2::splat(tokens::BUTTON_PAD_Y)
        };
        let mut size = (content + 2.0 * pad)
            .max(self.min_size)
            .max(vec2(0.0, tokens::BUTTON_HEIGHT));
        if galley.is_none() {
            // An icon button is square unless told otherwise.
            size.x = size.x.max(size.y);
        }
        if self.full_width {
            size.x = size.x.max(ui.available_width());
        }
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());
        if ui.is_rect_visible(rect) {
            let look = look(
                self.kind,
                self.on,
                ui.is_enabled(),
                response.hovered(),
                response.is_pointer_button_down_on(),
            );
            let painter = ui.painter();
            if look.fill != Color32::TRANSPARENT {
                painter.rect_filled(rect, CornerRadius::same(self.radius), look.fill);
            }
            let mut x = rect.center().x - content.x / 2.0;
            if let Some(name) = &self.icon {
                let mark =
                    Rect::from_min_size(pos2(x, rect.center().y - icon / 2.0), Vec2::splat(icon));
                if let Err(error) = icons::paint(painter, mark, name, look.ink) {
                    painter.text(
                        mark.center(),
                        Align2::CENTER_CENTER,
                        error.to_string(),
                        FontId::proportional(9.0),
                        tokens::FAINT,
                    );
                }
                x += icon + gap;
            }
            if let Some(galley) = galley {
                let at = pos2(x, rect.center().y - galley.size().y / 2.0);
                painter.galley(at, galley, look.ink);
            }
        }
        match self.tip {
            Some(tip) => response
                .on_hover_text(tip.clone())
                .on_disabled_hover_text(tip),
            None => response,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Pos2;

    fn run(add: impl FnMut(&mut Ui)) -> egui::FullOutput {
        let ctx = egui::Context::default();
        let mut add = add;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                ..Default::default()
            },
            |ui| add(ui),
        );
        output.textures_delta.clear();
        output
    }

    fn fills(output: &egui::FullOutput) -> Vec<Color32> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) => Some(rect.fill),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn each_kind_reads_its_colors_from_the_tokens() {
        let rest = |kind| look(kind, false, true, false, false);
        assert_eq!(rest(ButtonKind::Primary).fill, tokens::EXPORT);
        assert_eq!(rest(ButtonKind::Primary).ink, tokens::EXPORT_INK);
        assert_eq!(rest(ButtonKind::Secondary).fill, tokens::RAISED);
        assert_eq!(rest(ButtonKind::Ghost).fill, Color32::TRANSPARENT);
        assert_eq!(rest(ButtonKind::Ghost).ink, tokens::MUTED);
    }

    #[test]
    fn hover_on_and_press_each_show() {
        let primary = look(ButtonKind::Primary, false, true, true, false);
        assert_eq!(primary.fill, tokens::EXPORT_HOVER);
        let ghost = look(ButtonKind::Ghost, false, true, true, false);
        assert_eq!((ghost.fill, ghost.ink), (tokens::RAISED, tokens::TEXT));
        let on = look(ButtonKind::Ghost, true, true, false, false);
        assert_eq!((on.fill, on.ink), (tokens::ACTIVE, tokens::TEXT));
        let held = look(ButtonKind::Secondary, false, true, true, true);
        assert_eq!(held.fill, tokens::PRESSED);
    }

    #[test]
    fn a_disabled_button_is_faint_and_does_not_light_up() {
        for kind in [
            ButtonKind::Primary,
            ButtonKind::Secondary,
            ButtonKind::Ghost,
        ] {
            let off = look(kind, false, false, true, true);
            assert_eq!(off.ink, tokens::FAINT);
            assert_ne!(off.fill, tokens::EXPORT_HOVER);
            assert_ne!(off.fill, tokens::PRESSED);
        }
    }

    #[test]
    fn sizes_fit_the_content_and_respect_the_floor() {
        let mut text = Rect::NOTHING;
        let mut icon = Rect::NOTHING;
        let mut sized = Rect::NOTHING;
        let mut wide = Rect::NOTHING;
        run(|ui| {
            text = ui.add(Button::new("Save")).rect;
            icon = ui.add(Button::icon("play")).rect;
            sized = ui
                .add(Button::primary("Export").min_size(vec2(72.0, 26.0)))
                .rect;
            wide = ui.add(Button::icon("play").full_width()).rect;
        });
        assert!(text.height() >= tokens::BUTTON_HEIGHT);
        assert!(text.width() > 2.0 * tokens::BUTTON_PAD_X);
        assert_eq!(icon.width(), icon.height());
        assert!(sized.width() >= 72.0 && sized.height() >= 26.0);
        assert_eq!(wide.width(), 400.0);
    }

    #[test]
    fn a_primary_button_paints_the_export_fill() {
        let output = run(|ui| {
            ui.add(Button::primary("Export"));
            ui.add(Button::new("Cancel"));
            ui.add(Button::icon("play"));
        });
        let painted = fills(&output);
        assert!(painted.contains(&tokens::EXPORT));
        assert!(painted.contains(&tokens::RAISED));
        // A ghost at rest paints no fill.
        assert_eq!(painted.len(), 2);
    }

    #[test]
    fn a_disabled_button_ignores_a_click() {
        let ctx = egui::Context::default();
        let mut clicked = false;
        let mut rect = Rect::NOTHING;
        let frame = |events: Vec<egui::Event>, clicked: &mut bool, rect: &mut Rect| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |ui| {
                    let response = ui.add_enabled(false, Button::new("Undo"));
                    *rect = response.rect;
                    *clicked |= response.clicked();
                },
            );
            output.textures_delta.clear();
        };
        frame(vec![], &mut clicked, &mut rect);
        let at = rect.center();
        let press = |down| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: down,
            modifiers: Default::default(),
        };
        frame(
            vec![egui::Event::PointerMoved(at), press(true)],
            &mut clicked,
            &mut rect,
        );
        frame(vec![press(false)], &mut clicked, &mut rect);
        assert!(!clicked);
    }
}
