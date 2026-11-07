//! Text fields: one line or several, with an optional hint and a leading
//! icon or unit.
//!
//! A field keeps one size whatever its state. Hovering and typing change the
//! color of its edge, never its width or height. Clicking anywhere in the
//! box, its padding included, puts the caret in it.

use egui::{
    Align, Color32, CornerRadius, FontId, Id, IdSalt, Layout, Rect, Response, RichText, Sense,
    Stroke, StrokeKind, TextEdit, Ui, UiBuilder, Vec2, vec2,
};

use crate::{icons, tokens};

/// What sits before the text.
#[derive(Clone, Debug, PartialEq)]
enum Lead {
    Icon(String),
    Unit(String),
}

/// A text field.
///
/// ```ignore
/// let field = ui::Field::new(&mut query).hint("Search").icon("search").show(ui);
/// if field.committed { /* Enter, or a click elsewhere */ }
/// ```
#[must_use = "call `show` to draw the field"]
pub struct Field<'t> {
    text: &'t mut String,
    rows: Option<usize>,
    hint: Option<String>,
    width: Option<f32>,
    lead: Option<Lead>,
    text_size: f32,
    ink: Color32,
    fill: Color32,
    focus: bool,
    scrub: bool,
    salt: Option<IdSalt>,
}

/// What the user did with a field this frame.
pub struct FieldResponse {
    /// The text's own response: `changed`, `has_focus`, `lost_focus`.
    pub response: Response,
    /// The user left the field to keep the text: Enter on one line, or a
    /// click elsewhere.
    pub committed: bool,
    /// The user left the field with Escape, to throw the edit away.
    pub cancelled: bool,
    /// While the lead of a [`Field::scrub`] field is dragged: the points
    /// moved sideways since the press, right positive, a tenth as many with
    /// Shift held. `None` when it is not being dragged.
    pub scrub: Option<f32>,
}

impl<'t> Field<'t> {
    /// A one-line field.
    pub fn new(text: &'t mut String) -> Self {
        Self {
            text,
            rows: None,
            hint: None,
            width: None,
            lead: None,
            text_size: tokens::TYPE_SIZE,
            ink: tokens::TEXT,
            fill: tokens::FIELD,
            focus: false,
            scrub: false,
            salt: None,
        }
    }

    /// A field of at least `rows` lines that grows as the text does. Enter
    /// starts a new line, so only leaving the field commits.
    pub fn multiline(text: &'t mut String, rows: usize) -> Self {
        Self {
            rows: Some(rows.max(1)),
            ..Self::new(text)
        }
    }

    /// Faint text shown while the field is empty.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// A fixed width. Without one the field fills the row.
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    /// An icon before the text.
    pub fn icon(mut self, name: impl Into<String>) -> Self {
        self.lead = Some(Lead::Icon(name.into()));
        self
    }

    /// A short unit or label before the text, such as `X` or `%`.
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        let unit = unit.into();
        self.lead = (!unit.is_empty()).then_some(Lead::Unit(unit));
        self
    }

    pub fn text_size(mut self, size: f32) -> Self {
        self.text_size = size;
        self
    }

    pub fn ink(mut self, ink: Color32) -> Self {
        self.ink = ink;
        self
    }

    pub fn fill(mut self, fill: Color32) -> Self {
        self.fill = fill;
        self
    }

    /// Put the caret here when the field shows, as a rename does.
    pub fn focus(mut self) -> Self {
        self.focus = true;
        self
    }

    /// Let the icon or unit be dragged sideways to change the value, as a
    /// number field's label is. The field reports the drag in
    /// [`FieldResponse::scrub`]; a field with no lead has nothing to drag.
    pub fn scrub(mut self) -> Self {
        self.scrub = true;
        self
    }

    /// Keep the field's state under this salt, for fields drawn in a loop
    /// or only some frames.
    pub fn id_salt(mut self, salt: impl std::hash::Hash + std::fmt::Debug) -> Self {
        self.salt = Some(IdSalt::new(salt));
        self
    }

    /// Lay the field out in the row.
    pub fn show(self, ui: &mut Ui) -> FieldResponse {
        let id = match &self.salt {
            Some(salt) => ui.make_persistent_id(salt),
            None => ui.next_auto_id().with("cutlass-field"),
        };
        let pad = vec2(tokens::FIELD_PAD_X, tokens::FIELD_PAD_Y);
        let row = ui.fonts_mut(|fonts| fonts.row_height(&FontId::proportional(self.text_size)));
        let line = tokens::FIELD_HEIGHT.max(row + 2.0 * pad.y);
        let height = match self.rows {
            None => line,
            Some(rows) => {
                let least = rows as f32 * row + 2.0 * pad.y;
                let grown = ui.data(|data| data.get_temp::<f32>(id.with("height")));
                least.max(grown.unwrap_or(0.0))
            }
        };
        let width = self.width.unwrap_or_else(|| ui.available_width());
        let (rect, boxed) = ui.allocate_exact_size(vec2(width, height), Sense::click());
        self.draw(ui, id, rect, boxed, row)
    }

    /// Place the field over `rect`, which the caller has already laid out.
    pub fn show_in(self, ui: &mut Ui, rect: Rect) -> FieldResponse {
        let id = match &self.salt {
            Some(salt) => ui.make_persistent_id(salt),
            None => ui.next_auto_id().with("cutlass-field"),
        };
        let row = ui.fonts_mut(|fonts| fonts.row_height(&FontId::proportional(self.text_size)));
        let boxed = ui.interact(rect, id.with("box"), Sense::click());
        self.draw(ui, id, rect, boxed, row)
    }

    fn draw(self, ui: &mut Ui, id: Id, rect: Rect, boxed: Response, row: f32) -> FieldResponse {
        let enabled = ui.is_enabled();
        let focused = ui.memory(|memory| memory.has_focus(id));
        let hovered = enabled && ui.rect_contains_pointer(rect);
        if ui.is_rect_visible(rect) {
            ui.painter().rect(
                rect,
                CornerRadius::same(tokens::RADIUS_MD),
                self.fill,
                Stroke::new(tokens::HAIRLINE, edge(enabled, hovered, focused)),
                StrokeKind::Inside,
            );
        }

        let pad = vec2(tokens::FIELD_PAD_X, tokens::FIELD_PAD_Y);
        let single = self.rows.is_none();
        let inner = if single {
            // One line sits in the middle however tall the box is.
            let top = rect.center().y - row * 0.5;
            Rect::from_min_max(
                egui::pos2(rect.left() + pad.x, top),
                egui::pos2(rect.right() - pad.x, top + row),
            )
        } else {
            rect.shrink2(pad)
        };
        let mut content = ui.new_child(
            UiBuilder::new()
                .max_rect(inner)
                .layout(Layout::left_to_right(Align::Min)),
        );
        let ink = if enabled { self.ink } else { tokens::FAINT };
        // The lead's width, known before it is drawn, so its drag area and
        // its color are settled first.
        let unit_font = FontId::proportional(12.0);
        let lead_width = match &self.lead {
            Some(Lead::Icon(_)) => row.min(14.0),
            Some(Lead::Unit(unit)) => ui.fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(unit.clone(), unit_font.clone(), tokens::MUTED)
                    .size()
                    .x
            }),
            None => 0.0,
        };
        let lead_gap = 6.0;
        // Only lead_gap between the lead and the text.
        content.spacing_mut().item_spacing.x = 0.0;
        let mut scrub = None;
        let mut lit = false;
        if self.scrub && enabled && self.lead.is_some() {
            // From the box's left edge to halfway into the gap after the
            // lead, so the padding drags too.
            let hit = Rect::from_min_max(
                rect.left_top(),
                egui::pos2(inner.left() + lead_width + lead_gap * 0.5, rect.bottom()),
            );
            let handle = ui
                .interact(hit, id.with("scrub"), Sense::drag())
                .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
            if handle.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                let moved = handle.total_drag_delta().map_or(0.0, |delta| delta.x);
                let fine = ui.input(|input| input.modifiers.shift);
                scrub = Some(if fine { moved * 0.1 } else { moved });
            }
            lit = handle.hovered() || handle.dragged();
        }
        let lead_ink = if !enabled {
            tokens::FAINT
        } else if lit {
            tokens::TEXT
        } else {
            tokens::MUTED
        };
        if self.lead.is_some() {
            let (mark, _) = content.allocate_exact_size(Vec2::new(lead_width, row), Sense::hover());
            match &self.lead {
                Some(Lead::Icon(name)) => {
                    let mark = Rect::from_center_size(mark.center(), Vec2::splat(lead_width));
                    // A missing icon leaves its gap empty rather than
                    // failing the field.
                    let _ = icons::paint(content.painter(), mark, name, lead_ink);
                }
                Some(Lead::Unit(unit)) => {
                    content.painter().text(
                        mark.left_center(),
                        egui::Align2::LEFT_CENTER,
                        unit,
                        unit_font,
                        lead_ink,
                    );
                }
                None => {}
            }
            content.add_space(lead_gap);
        }

        let font = FontId::proportional(self.text_size);
        let mut edit = if single {
            TextEdit::singleline(self.text)
        } else {
            TextEdit::multiline(self.text).desired_rows(self.rows.unwrap_or(1))
        };
        edit = edit
            .id(id)
            .font(font)
            .text_color(ink)
            .frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .desired_width(content.available_width());
        if let Some(hint) = &self.hint {
            edit = edit.hint_text(RichText::new(hint).color(tokens::FAINT));
        }
        let response = content.add(edit);

        if let Some(rows) = self.rows {
            let least = rows as f32 * row + 2.0 * pad.y;
            let wanted = (response.rect.bottom() + pad.y - rect.top()).max(least);
            let had = ui.data(|data| data.get_temp::<f32>(id.with("height")));
            if had != Some(wanted) {
                ui.data_mut(|data| data.insert_temp(id.with("height"), wanted));
                if (wanted - rect.height()).abs() > 0.5 {
                    // The box is a frame behind its text; draw it again at
                    // the new size before anyone sees the old one.
                    ui.ctx().request_discard("a text field changed height");
                }
            }
        }

        // After the text has had its turn: it lets go of focus on a click
        // outside it, and a click in the padding is one.
        if enabled && boxed.clicked() {
            response.request_focus();
        }
        if self.focus && enabled && !response.has_focus() && !response.lost_focus() {
            response.request_focus();
        }
        let left = response.lost_focus();
        let escaped = left && ui.input(|input| input.key_pressed(egui::Key::Escape));
        FieldResponse {
            committed: left && !escaped,
            cancelled: escaped,
            scrub,
            response,
        }
    }
}

/// The edge color for a field's state.
fn edge(enabled: bool, hovered: bool, focused: bool) -> Color32 {
    if !enabled {
        Color32::TRANSPARENT
    } else if focused {
        tokens::FIELD_EDGE_FOCUS
    } else if hovered {
        tokens::FIELD_EDGE_HOVER
    } else {
        Color32::TRANSPARENT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, Key, Modifiers, PointerButton, Pos2, Shape, pos2};

    struct Run {
        ctx: egui::Context,
        text: String,
        rows: Option<usize>,
    }

    struct Frame {
        output: egui::FullOutput,
        committed: bool,
        cancelled: bool,
        focused: bool,
    }

    impl Run {
        fn new(text: &str) -> Self {
            Self {
                ctx: egui::Context::default(),
                text: text.to_owned(),
                rows: None,
            }
        }

        fn frame(&mut self, events: Vec<Event>) -> Frame {
            let mut committed = false;
            let mut cancelled = false;
            let mut focused = false;
            let rows = self.rows;
            let text = &mut self.text;
            let mut output = self.ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let field = match rows {
                        None => Field::new(text),
                        Some(rows) => Field::multiline(text, rows),
                    };
                    let shown = field.hint("Name").width(200.0).show(ui);
                    committed |= shown.committed;
                    cancelled |= shown.cancelled;
                    focused = shown.response.has_focus();
                },
            );
            output.textures_delta.clear();
            Frame {
                output,
                committed,
                cancelled,
                focused,
            }
        }

        fn click(&mut self, at: Pos2) -> Frame {
            self.frame(vec![Event::PointerMoved(at)]);
            self.frame(vec![press(at, true)]);
            self.frame(vec![press(at, false)])
        }

        fn key(&mut self, key: Key) -> Frame {
            self.frame(vec![Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            }])
        }
    }

    fn press(at: Pos2, pressed: bool) -> Event {
        Event::PointerButton {
            pos: at,
            button: PointerButton::Primary,
            pressed,
            modifiers: Modifiers::NONE,
        }
    }

    /// The field's box: the first rounded rect painted.
    fn box_shape(output: &egui::FullOutput) -> egui::epaint::RectShape {
        output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                Shape::Rect(rect) if rect.fill == tokens::FIELD => Some(rect.clone()),
                _ => None,
            })
            .expect("the field's box")
    }

    #[test]
    fn one_line_is_the_field_height_and_the_width_asked_for() {
        let mut run = Run::new("");
        let frame = run.frame(vec![]);
        let shape = box_shape(&frame.output);
        assert_eq!(shape.rect.height(), tokens::FIELD_HEIGHT);
        assert_eq!(shape.rect.width(), 200.0);
        assert_eq!(shape.corner_radius, CornerRadius::same(tokens::RADIUS_MD));
    }

    #[test]
    fn hover_and_focus_change_the_edge_color_not_the_size() {
        let mut run = Run::new("clip");
        let rest = box_shape(&run.frame(vec![]).output);
        assert_eq!(rest.stroke.color, Color32::TRANSPARENT);

        let inside = rest.rect.center();
        let hovered = box_shape(&run.frame(vec![Event::PointerMoved(inside)]).output);
        assert_eq!(hovered.stroke.color, tokens::FIELD_EDGE_HOVER);
        assert_eq!(hovered.rect, rest.rect);

        run.click(inside);
        let focused = box_shape(&run.frame(vec![]).output);
        assert_eq!(focused.stroke.color, tokens::FIELD_EDGE_FOCUS);
        assert_eq!(focused.rect, rest.rect);
        assert_eq!(focused.stroke.width, rest.stroke.width);
    }

    #[test]
    fn a_click_in_the_padding_puts_the_caret_in_the_field() {
        let mut run = Run::new("clip");
        let rect = box_shape(&run.frame(vec![]).output).rect;
        // Just inside the left edge, left of where the text starts.
        run.click(pos2(rect.left() + 2.0, rect.center().y));
        assert!(run.frame(vec![]).focused);
    }

    #[test]
    fn enter_commits_and_escape_cancels() {
        let mut run = Run::new("clip");
        let rect = box_shape(&run.frame(vec![]).output).rect;
        run.click(rect.center());
        run.frame(vec![Event::Text("s".to_owned())]);
        assert_eq!(run.text, "clips");
        let left = run.key(Key::Enter);
        assert!(left.committed && !left.cancelled);

        run.click(rect.center());
        let left = run.key(Key::Escape);
        assert!(left.cancelled && !left.committed);
    }

    #[test]
    fn several_lines_grow_with_the_text() {
        let mut run = Run::new("one");
        run.rows = Some(2);
        let short = box_shape(&run.frame(vec![]).output).rect;
        run.text = "one\ntwo\nthree\nfour\nfive".to_owned();
        run.frame(vec![]);
        let tall = box_shape(&run.frame(vec![]).output).rect;
        assert!(tall.height() > short.height() + 20.0, "{short:?} {tall:?}");
        assert_eq!(tall.width(), short.width());
    }

    #[test]
    fn dragging_the_unit_scrubs_and_clicking_it_does_not_type() {
        let ctx = egui::Context::default();
        let mut text = "10".to_owned();
        let mut frame = |events: Vec<Event>| {
            let mut scrub = None;
            let mut rect = Rect::NOTHING;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let shown = Field::new(&mut text)
                        .unit("X")
                        .scrub()
                        .width(120.0)
                        .show(ui);
                    scrub = shown.scrub;
                    rect = shown.response.rect;
                },
            );
            output.textures_delta.clear();
            (scrub, rect)
        };
        let (_, text_rect) = frame(vec![]);
        let unit = pos2(text_rect.left() - 8.0, text_rect.center().y);
        frame(vec![Event::PointerMoved(unit)]);
        frame(vec![press(unit, true)]);
        let (scrub, _) = frame(vec![Event::PointerMoved(unit + vec2(30.0, 0.0))]);
        assert_eq!(scrub, Some(30.0));
        let (scrub, _) = frame(vec![Event::PointerMoved(unit + vec2(20.0, 4.0))]);
        assert_eq!(scrub, Some(20.0));
        let (scrub, _) = frame(vec![press(unit + vec2(20.0, 4.0), false)]);
        assert_eq!(scrub, None);
        assert_eq!(text, "10");
    }

    #[test]
    fn a_disabled_field_shows_no_edge() {
        assert_eq!(edge(false, true, true), Color32::TRANSPARENT);
    }
}
