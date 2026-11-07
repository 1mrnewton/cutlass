//! Choices: one option of several, and the well that holds a row of them.
//!
//! A [`Choice`] is drawn raised when it is the one picked. Put a row of them
//! in [`Segmented`] and they read as one control; alone, as in a list of
//! fonts, they read as chips. Hover changes the fill, never the size.

use egui::{
    Color32, CornerRadius, FontId, FontSelection, InnerResponse, Response, Sense, Ui, Vec2, Widget,
    WidgetText, pos2, vec2,
};

use crate::tokens;

/// One option. Click it to pick it; the caller keeps which one is picked.
///
/// ```ignore
/// ui::Segmented::show(ui, |ui| {
///     if ui.add(ui::Choice::new(!mov, "MP4")).clicked() { mov = false; }
///     if ui.add(ui::Choice::new(mov, "MOV")).clicked() { mov = true; }
/// });
/// ```
#[must_use = "add the choice with `ui.add`"]
pub struct Choice {
    on: bool,
    label: WidgetText,
    tip: Option<String>,
    wide: bool,
}

impl Choice {
    /// `on` is whether this is the option picked now.
    pub fn new(on: bool, label: impl Into<WidgetText>) -> Self {
        Self {
            on,
            label: label.into(),
            tip: None,
            wide: false,
        }
    }

    /// Take the whole width the layout offers, with the label at the
    /// left, as a row in a dropdown does.
    pub fn wide(mut self) -> Self {
        self.wide = true;
        self
    }

    /// Shown on hover, and on hover while disabled too.
    pub fn tip(mut self, tip: impl Into<String>) -> Self {
        self.tip = Some(tip.into());
        self
    }
}

/// A choice's fill and ink for one frame.
fn look(on: bool, enabled: bool, hovered: bool, pressed: bool) -> (Color32, Color32) {
    if !enabled {
        let fill = if on {
            tokens::CHOICE_HOVER
        } else {
            Color32::TRANSPARENT
        };
        return (fill, tokens::FAINT);
    }
    let fill = if on || pressed {
        tokens::CHOICE_ON
    } else if hovered {
        tokens::CHOICE_HOVER
    } else {
        Color32::TRANSPARENT
    };
    let ink = if on || hovered {
        tokens::TEXT
    } else {
        tokens::MUTED
    };
    (fill, ink)
}

impl Widget for Choice {
    fn ui(self, ui: &mut Ui) -> Response {
        let galley = self.label.into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            FontSelection::FontId(FontId::proportional(tokens::CHOICE_TEXT)),
        );
        let mut size = (galley.size() + vec2(2.0 * tokens::CHOICE_PAD_X, 0.0))
            .max(vec2(0.0, tokens::CHOICE_HEIGHT));
        if self.wide {
            size.x = size.x.max(ui.available_width());
        }
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());
        if ui.is_rect_visible(rect) {
            let (fill, ink) = look(
                self.on,
                ui.is_enabled(),
                response.hovered(),
                response.is_pointer_button_down_on(),
            );
            if fill != Color32::TRANSPARENT {
                ui.painter()
                    .rect_filled(rect, CornerRadius::same(tokens::RADIUS_SM), fill);
            }
            let left = if self.wide {
                rect.left() + tokens::CHOICE_PAD_X
            } else {
                rect.center().x - galley.size().x / 2.0
            };
            let at = pos2(left, rect.center().y - galley.size().y / 2.0);
            ui.painter().galley(at, galley, ink);
        }
        match self.tip {
            Some(tip) => response
                .on_hover_text(tip.clone())
                .on_disabled_hover_text(tip),
            None => response,
        }
    }
}

/// A sunken well that holds a row of [`Choice`]s, so they read as one
/// control.
pub struct Segmented;

impl Segmented {
    /// Lay `add` out in a row inside the well.
    pub fn show<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R> {
        crate::ui::Surface::inset()
            .radius(tokens::RADIUS_MD)
            .padding(tokens::SEGMENT_PAD)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::splat(tokens::SEGMENT_GAP);
                    add(ui)
                })
                .inner
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Pos2, Rect, Shape};

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
                Shape::Rect(rect) => Some(rect.fill),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_picked_choice_is_raised_and_the_others_are_bare() {
        assert_eq!(
            look(true, true, false, false),
            (tokens::CHOICE_ON, tokens::TEXT)
        );
        assert_eq!(
            look(false, true, false, false),
            (Color32::TRANSPARENT, tokens::MUTED)
        );
        assert_eq!(look(false, true, true, false).0, tokens::CHOICE_HOVER);
        assert_eq!(look(true, false, false, false).1, tokens::FAINT);
    }

    #[test]
    fn a_row_sits_in_one_well_with_the_picked_one_raised() {
        let mut sizes = Vec::new();
        let output = run(|ui| {
            Segmented::show(ui, |ui| {
                sizes.push(ui.add(Choice::new(true, "MP4")).rect.size());
                sizes.push(ui.add(Choice::new(false, "MOV")).rect.size());
            });
        });
        let painted = fills(&output);
        assert_eq!(painted, vec![tokens::INSET, tokens::CHOICE_ON]);
        assert!(sizes.iter().all(|size| size.y == tokens::CHOICE_HEIGHT));
    }
}
