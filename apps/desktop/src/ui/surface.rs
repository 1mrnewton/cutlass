//! Backgrounds: the window gap, panels, bars, cards, wells, dialogs, and the
//! scrim.
//!
//! Pick a kind, adjust it if a spot truly needs to, then either `show` content
//! inside it or `paint` it into a rect you already laid out.

use egui::{Color32, CornerRadius, InnerResponse, Margin, Painter, Rect, Shape, Stroke, Ui};

use crate::tokens;

/// A background with a fill, an optional edge, rounded corners, and padding.
///
/// ```ignore
/// ui::Surface::raised().show(ui, |ui| ui.label("A card"));
/// ui::Surface::panel().paint(ui.painter(), rect);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "call `show` or `paint` to draw the surface"]
pub struct Surface {
    fill: Color32,
    stroke: Stroke,
    radius: CornerRadius,
    padding: Margin,
}

impl Surface {
    const fn flat(fill: Color32) -> Self {
        Self {
            fill,
            stroke: Stroke::NONE,
            radius: CornerRadius::ZERO,
            padding: Margin::ZERO,
        }
    }

    /// The window behind every region, which shows as the gaps between them.
    pub const fn gap() -> Self {
        Self::flat(tokens::GAP)
    }

    /// The library, preview, inspector, and timeline.
    pub const fn panel() -> Self {
        Self::flat(tokens::PANEL)
    }

    /// The project bar across the top.
    pub const fn bar() -> Self {
        Self::flat(tokens::PROJECT_BAR)
    }

    /// The dim layer behind a dialog or a drop.
    pub const fn scrim() -> Self {
        Self::flat(tokens::SCRIM)
    }

    /// A card, bubble, or field that sits on a panel.
    pub const fn raised() -> Self {
        Self {
            fill: tokens::RAISED,
            stroke: Stroke::NONE,
            radius: CornerRadius::same(tokens::RADIUS_LG),
            padding: Margin::same(10),
        }
    }

    /// A well sunk into a panel that holds controls, such as a row of tabs.
    pub const fn inset() -> Self {
        Self {
            fill: tokens::INSET,
            stroke: Stroke::NONE,
            radius: CornerRadius::same(tokens::RADIUS_LG),
            padding: Margin::same(4),
        }
    }

    /// An edge with no fill, in the given color.
    pub fn outline(color: Color32) -> Self {
        Self {
            fill: Color32::TRANSPARENT,
            stroke: Stroke::new(tokens::HAIRLINE, color),
            radius: CornerRadius::same(tokens::RADIUS_LG),
            padding: Margin::same(10),
        }
    }

    /// A dialog's body.
    pub const fn dialog() -> Self {
        Self {
            fill: tokens::PROJECT_BAR,
            stroke: Stroke::NONE,
            radius: CornerRadius::same(tokens::DIALOG_RADIUS),
            padding: Margin::ZERO,
        }
    }

    pub const fn fill(mut self, fill: Color32) -> Self {
        self.fill = fill;
        self
    }

    pub fn stroke(mut self, color: Color32) -> Self {
        self.stroke = Stroke::new(tokens::HAIRLINE, color);
        self
    }

    pub const fn radius(mut self, radius: u8) -> Self {
        self.radius = CornerRadius::same(radius);
        self
    }

    pub fn padding(mut self, padding: impl Into<Margin>) -> Self {
        self.padding = padding.into();
        self
    }

    /// Lay `add` out inside the surface, which grows to fit it.
    pub fn show<R>(self, ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R> {
        egui::Frame::new()
            .fill(self.fill)
            .stroke(self.stroke)
            .corner_radius(self.radius)
            .inner_margin(self.padding)
            .show(ui, add)
    }

    /// The surface as one shape over `rect`, for a painter slot filled in
    /// once the size is known.
    pub fn shape(&self, rect: Rect) -> Shape {
        Shape::Rect(egui::epaint::RectShape::new(
            rect,
            self.radius,
            self.fill,
            self.stroke,
            egui::StrokeKind::Inside,
        ))
    }

    /// Fill `rect`, which the caller has already laid out. Padding is not
    /// used here.
    pub fn paint(&self, painter: &Painter, rect: Rect) {
        painter.add(self.shape(rect));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Pos2, vec2};

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

    fn rects(output: &egui::FullOutput) -> Vec<egui::epaint::RectShape> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                Shape::Rect(rect) => Some(rect.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn kinds_read_their_colors_from_the_tokens() {
        assert_eq!(Surface::gap().fill, tokens::GAP);
        assert_eq!(Surface::panel().fill, tokens::PANEL);
        assert_eq!(Surface::bar().fill, tokens::PROJECT_BAR);
        assert_eq!(Surface::scrim().fill, tokens::SCRIM);
        assert_eq!(Surface::raised().fill, tokens::RAISED);
        assert_eq!(Surface::inset().fill, tokens::INSET);
        assert_eq!(
            Surface::dialog().radius,
            CornerRadius::same(tokens::DIALOG_RADIUS)
        );
        let outline = Surface::outline(tokens::MUTED);
        assert_eq!(outline.fill, Color32::TRANSPARENT);
        assert_eq!(outline.stroke.color, tokens::MUTED);
    }

    #[test]
    fn paint_fills_exactly_the_rect() {
        let rect = Rect::from_min_size(Pos2::new(10.0, 20.0), vec2(100.0, 50.0));
        let output = run(|ui| Surface::panel().paint(ui.painter(), rect));
        let painted = rects(&output);
        assert_eq!(painted.len(), 1);
        assert_eq!(painted[0].rect, rect);
        assert_eq!(painted[0].fill, tokens::PANEL);
        assert_eq!(painted[0].corner_radius, CornerRadius::ZERO);
    }

    #[test]
    fn show_pads_the_content_and_wraps_it() {
        let mut content = Rect::NOTHING;
        let mut outer = Rect::NOTHING;
        let output = run(|ui| {
            let shown = Surface::raised().padding(12).show(ui, |ui| {
                content = ui.label("card").rect;
            });
            outer = shown.response.rect;
        });
        assert!((content.min.x - outer.min.x - 12.0).abs() < 1e-3);
        assert!((content.min.y - outer.min.y - 12.0).abs() < 1e-3);
        let card = rects(&output)
            .into_iter()
            .find(|rect| rect.fill == tokens::RAISED)
            .expect("the card's fill");
        assert_eq!(card.corner_radius, CornerRadius::same(tokens::RADIUS_LG));
    }
}
