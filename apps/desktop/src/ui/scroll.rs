//! A scroll view with Cutlass's own scroll bar.
//!
//! egui's bars are hidden. The bar is painted here, floats over the content
//! at one fixed width, and only changes color on hover or drag.

use egui::{CornerRadius, Id, IdSalt, Pos2, Rect, ScrollArea, Sense, Ui, Vec2b, scroll_area};

use crate::tokens;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Vertical,
    Horizontal,
}

impl Axis {
    /// The index of this axis in a `Vec2` or `Pos2`.
    fn along(self) -> usize {
        match self {
            Self::Horizontal => 0,
            Self::Vertical => 1,
        }
    }

    fn across(self) -> usize {
        1 - self.along()
    }
}

/// A scrolling region.
///
/// ```ignore
/// ui::Scroll::vertical("library-cards")
///     .auto_shrink([false, false])
///     .show(ui, |ui| { /* cards */ });
/// ```
#[must_use = "call `show` to add the scroll view"]
pub struct Scroll {
    axis: Axis,
    salt: IdSalt,
    auto_shrink: Vec2b,
    max_height: Option<f32>,
    stick_to_bottom: bool,
    bar: bool,
    margin: egui::Margin,
}

impl Scroll {
    /// Scrolls up and down. The salt keeps the offset across frames.
    pub fn vertical(salt: impl egui::AsIdSalt) -> Self {
        Self::new(Axis::Vertical, salt)
    }

    /// Scrolls sideways. The salt keeps the offset across frames.
    pub fn horizontal(salt: impl egui::AsIdSalt) -> Self {
        Self::new(Axis::Horizontal, salt)
    }

    fn new(axis: Axis, salt: impl egui::AsIdSalt) -> Self {
        Self {
            axis,
            salt: IdSalt::new(salt),
            auto_shrink: Vec2b::TRUE,
            max_height: None,
            stick_to_bottom: false,
            bar: true,
            margin: egui::Margin::ZERO,
        }
    }

    /// Shrink to the content on each axis, as `ScrollArea::auto_shrink`.
    pub fn auto_shrink(mut self, shrink: impl Into<Vec2b>) -> Self {
        self.auto_shrink = shrink.into();
        self
    }

    pub fn max_height(mut self, height: f32) -> Self {
        self.max_height = Some(height);
        self
    }

    /// Keep the end in view while content grows, as a chat log does.
    pub fn stick_to_bottom(mut self, stick: bool) -> Self {
        self.stick_to_bottom = stick;
        self
    }

    /// Pad the content inside the scroll view. The bar stays on the outer
    /// edge, so a padded panel gives the view its whole rect and pads here.
    pub fn content_margin(mut self, margin: impl Into<egui::Margin>) -> Self {
        self.margin = margin.into();
        self
    }

    /// Scroll without drawing a bar, for a strip like tabs.
    pub fn without_bar(mut self) -> Self {
        self.bar = false;
        self
    }

    pub fn show<R>(self, ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
        let key = ui.make_persistent_id((self.salt, "cutlass-scroll"));
        let pending: Option<f32> = ui.data_mut(|data| data.remove_temp(key));
        let mut area = match self.axis {
            Axis::Vertical => ScrollArea::vertical(),
            Axis::Horizontal => ScrollArea::horizontal(),
        }
        .id_salt(self.salt)
        .auto_shrink(self.auto_shrink)
        .stick_to_bottom(self.stick_to_bottom)
        .content_margin(self.margin)
        .scroll_bar_visibility(scroll_area::ScrollBarVisibility::AlwaysHidden);
        if let Some(height) = self.max_height {
            area = area.max_height(height);
        }
        if let Some(offset) = pending {
            area = match self.axis {
                Axis::Vertical => area.vertical_scroll_offset(offset),
                Axis::Horizontal => area.horizontal_scroll_offset(offset),
            };
        }
        let output = area.show(ui, add);
        if self.bar {
            let a = self.axis.along();
            bar(
                ui,
                key,
                self.axis,
                output.inner_rect,
                output.content_size[a],
                output.state.clamped_offset()[a],
            );
        }
        output.inner
    }
}

/// Where the bar and its thumb sit for one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bar {
    track: Rect,
    thumb: Rect,
    /// How far the thumb can move along the track.
    travel: f32,
    /// How far the content can scroll.
    range: f32,
}

impl Bar {
    /// The bar for a viewport, or none when the content fits.
    fn place(axis: Axis, view: Rect, content: f32, offset: f32) -> Option<Self> {
        let (a, c) = (axis.along(), axis.across());
        let length = view.size()[a];
        if content <= length + 0.5 {
            return None;
        }
        let mut min = view.min;
        let mut max = view.max;
        min[a] += tokens::SCROLL_INSET;
        max[a] -= tokens::SCROLL_INSET;
        max[c] -= tokens::SCROLL_EDGE;
        min[c] = max[c] - tokens::SCROLL_BAR;
        let track = Rect::from_min_max(min, max);
        let span = track.size()[a];
        let thumb_length = (span * length / content)
            .max(tokens::SCROLL_MIN_THUMB)
            .min(span);
        let travel = span - thumb_length;
        let range = content - length;
        let mut bar = Self {
            track,
            thumb: track,
            travel,
            range,
        };
        bar.thumb = bar.thumb_at(axis, offset);
        Some(bar)
    }

    fn thumb_at(&self, axis: Axis, offset: f32) -> Rect {
        let a = axis.along();
        let start = self.track.min[a] + self.travel * (offset / self.range).clamp(0.0, 1.0);
        let length = self.track.size()[a] - self.travel;
        let mut thumb = self.track;
        thumb.min[a] = start;
        thumb.max[a] = start + length;
        thumb
    }

    /// The content offset that puts the thumb's leading edge at `start`.
    fn offset_for(&self, axis: Axis, start: f32) -> f32 {
        if self.travel <= 0.0 {
            return 0.0;
        }
        let fraction = (start - self.track.min[axis.along()]) / self.travel;
        fraction.clamp(0.0, 1.0) * self.range
    }
}

fn bar(ui: &Ui, key: Id, axis: Axis, view: Rect, content: f32, offset: f32) {
    let Some(mut bar) = Bar::place(axis, view, content, offset) else {
        return;
    };
    let a = axis.along();
    let grab_key = key.with("grab");
    let response = ui.interact(bar.track, key.with("bar"), Sense::click_and_drag());
    let pointer: Option<Pos2> = response.interact_pointer_pos();

    // Where along the thumb the pointer holds it. A press off the thumb
    // centers the thumb on the pointer.
    let grab_for = |at: Pos2, thumb: Rect| {
        if thumb.contains(at) {
            at[a] - thumb.min[a]
        } else {
            thumb.size()[a] / 2.0
        }
    };

    let mut moved = None;
    if let Some(at) = pointer {
        if response.drag_started() {
            ui.data_mut(|data| data.insert_temp(grab_key, grab_for(at, bar.thumb)));
        }
        if response.dragged() {
            let grab = ui
                .data(|data| data.get_temp::<f32>(grab_key))
                .unwrap_or_else(|| grab_for(at, bar.thumb));
            moved = Some(bar.offset_for(axis, at[a] - grab));
        } else if response.clicked() && !bar.thumb.contains(at) {
            moved = Some(bar.offset_for(axis, at[a] - bar.thumb.size()[a] / 2.0));
        }
    }
    if response.drag_stopped() {
        ui.data_mut(|data| data.remove::<f32>(grab_key));
    }
    if let Some(offset) = moved {
        ui.data_mut(|data| data.insert_temp(key, offset));
        bar.thumb = bar.thumb_at(axis, offset);
        ui.ctx().request_repaint();
    }

    // At rest the bar is hidden. It fades in while the pointer is over the
    // area or the content has just moved, and only its opacity changes.
    let now = ui.input(|input| input.time);
    let moved_at = ui.data_mut(|data| {
        let last = data.get_temp::<(f32, f64)>(key.with("moved"));
        let at = match last {
            Some((was, at)) if (was - offset).abs() < 0.5 => at,
            Some(_) => now,
            None => f64::NEG_INFINITY,
        };
        data.insert_temp(key.with("moved"), (offset, at));
        at
    });
    let lingering = now - moved_at < tokens::SCROLL_LINGER;
    if lingering {
        ui.ctx()
            .request_repaint_after_secs((tokens::SCROLL_LINGER - (now - moved_at)) as f32);
    }
    let shown =
        lingering || response.hovered() || response.dragged() || ui.rect_contains_pointer(view);
    let opacity = ui
        .ctx()
        .animate_bool_with_time(key.with("fade"), shown, tokens::SCROLL_FADE);
    if opacity <= 0.0 {
        return;
    }

    let color = if response.dragged() {
        tokens::SCROLL_THUMB_DRAG
    } else if response.hovered() {
        tokens::SCROLL_THUMB_HOVER
    } else {
        tokens::SCROLL_THUMB
    };
    let round = CornerRadius::same((tokens::SCROLL_BAR / 2.0) as u8);
    ui.painter().rect_filled(
        bar.track,
        round,
        tokens::SCROLL_TRACK.gamma_multiply(opacity),
    );
    let mut thumb = bar.thumb;
    thumb.min[axis.across()] += tokens::SCROLL_THUMB_INSET;
    thumb.max[axis.across()] -= tokens::SCROLL_THUMB_INSET;
    ui.painter()
        .rect_filled(thumb, round, color.gamma_multiply(opacity));
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Pos2, Vec2, pos2, vec2};

    fn view() -> Rect {
        Rect::from_min_size(Pos2::ZERO, vec2(200.0, 100.0))
    }

    #[test]
    fn no_bar_when_the_content_fits() {
        assert!(Bar::place(Axis::Vertical, view(), 100.0, 0.0).is_none());
        assert!(Bar::place(Axis::Horizontal, view(), 150.0, 0.0).is_none());
    }

    #[test]
    fn the_bar_is_one_fixed_width_at_the_far_edge() {
        let bar = Bar::place(Axis::Vertical, view(), 400.0, 0.0).unwrap();
        assert_eq!(bar.track.width(), tokens::SCROLL_BAR);
        assert_eq!(bar.thumb.width(), tokens::SCROLL_BAR);
        assert_eq!(bar.track.right(), 200.0 - tokens::SCROLL_EDGE);

        let bar = Bar::place(Axis::Horizontal, view(), 800.0, 0.0).unwrap();
        assert_eq!(bar.track.height(), tokens::SCROLL_BAR);
        assert_eq!(bar.track.bottom(), 100.0 - tokens::SCROLL_EDGE);
    }

    #[test]
    fn the_thumb_follows_the_offset_and_back() {
        let span = 100.0 - 2.0 * tokens::SCROLL_INSET;
        let top = Bar::place(Axis::Vertical, view(), 400.0, 0.0).unwrap();
        assert_eq!(top.thumb.top(), tokens::SCROLL_INSET);
        assert!((top.thumb.height() - span / 4.0).abs() < 1e-3);

        let end = Bar::place(Axis::Vertical, view(), 400.0, 300.0).unwrap();
        assert!((end.thumb.bottom() - top.track.bottom()).abs() < 1e-3);

        let half = Bar::place(Axis::Vertical, view(), 400.0, 150.0).unwrap();
        assert!((half.offset_for(Axis::Vertical, half.thumb.top()) - 150.0).abs() < 1e-3);
    }

    #[test]
    fn a_long_list_keeps_a_thumb_you_can_grab() {
        let bar = Bar::place(Axis::Vertical, view(), 100_000.0, 0.0).unwrap();
        assert_eq!(bar.thumb.height(), tokens::SCROLL_MIN_THUMB);
    }

    #[test]
    fn dragging_past_the_ends_clamps() {
        let bar = Bar::place(Axis::Vertical, view(), 400.0, 0.0).unwrap();
        assert_eq!(bar.offset_for(Axis::Vertical, -50.0), 0.0);
        assert_eq!(bar.offset_for(Axis::Vertical, 500.0), 300.0);
    }

    fn frame(ctx: &egui::Context, events: Vec<egui::Event>) -> egui::FullOutput {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                events,
                ..Default::default()
            },
            |ui| {
                Scroll::vertical("test")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for row in 0..100 {
                            ui.label(format!("row {row}"));
                        }
                    });
            },
        );
        output.textures_delta.clear();
        output
    }

    fn thumb_width(output: &egui::FullOutput, color: egui::Color32) -> Option<f32> {
        output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) if rect.fill == color => Some(rect.rect.width()),
                _ => None,
            })
    }

    /// The thumb's painted width: the bar less its inset on both sides.
    fn painted() -> f32 {
        tokens::SCROLL_BAR - 2.0 * tokens::SCROLL_THUMB_INSET
    }

    /// Frames enough for a fade to finish.
    fn settle(ctx: &egui::Context, events: Vec<egui::Event>) -> egui::FullOutput {
        frame(ctx, events);
        let mut last = frame(ctx, vec![]);
        for _ in 0..30 {
            last = frame(ctx, vec![]);
        }
        last
    }

    /// Whether any part of the bar is painted: the list draws only text.
    fn any_thumb(output: &egui::FullOutput) -> bool {
        output
            .shapes
            .iter()
            .any(|clipped| matches!(&clipped.shape, egui::Shape::Rect(rect) if rect.fill.a() > 0))
    }

    #[test]
    fn the_bar_hides_at_rest_and_shows_over_the_area() {
        let ctx = egui::Context::default();
        let rest = settle(&ctx, vec![]);
        assert!(!any_thumb(&rest), "no bar while nothing happens");

        let inside = Pos2::new(100.0, 100.0);
        let over_area = settle(&ctx, vec![egui::Event::PointerMoved(inside)]);
        assert_eq!(
            thumb_width(&over_area, tokens::SCROLL_THUMB),
            Some(painted())
        );
        assert_eq!(
            thumb_width(&over_area, tokens::SCROLL_TRACK),
            Some(tokens::SCROLL_BAR)
        );

        let gone = settle(&ctx, vec![egui::Event::PointerGone]);
        assert!(!any_thumb(&gone), "the bar fades once the pointer leaves");
    }

    #[test]
    fn hover_changes_the_color_not_the_width() {
        let ctx = egui::Context::default();
        let inside = settle(
            &ctx,
            vec![egui::Event::PointerMoved(Pos2::new(100.0, 100.0))],
        );
        let rest = thumb_width(&inside, tokens::SCROLL_THUMB).expect("a shown thumb");
        assert_eq!(rest, painted());

        let over = Pos2::new(400.0 - tokens::SCROLL_EDGE - 2.0, 20.0);
        let hovered = settle(&ctx, vec![egui::Event::PointerMoved(over)]);
        assert_eq!(
            thumb_width(&hovered, tokens::SCROLL_THUMB_HOVER),
            Some(painted())
        );
    }

    #[test]
    fn a_margin_pads_the_content_and_leaves_the_bar_on_the_edge() {
        let ctx = egui::Context::default();
        let mut first = Rect::NOTHING;
        let mut view = Rect::NOTHING;
        for _ in 0..2 {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
                    ..Default::default()
                },
                |ui| {
                    view = ui.max_rect();
                    Scroll::vertical("padded")
                        .auto_shrink([false, false])
                        .content_margin(16)
                        .show(ui, |ui| {
                            for row in 0..100 {
                                let label = ui.label(format!("row {row}"));
                                if row == 0 {
                                    first = label.rect;
                                }
                            }
                        });
                },
            );
            output.textures_delta.clear();
        }
        assert_eq!(first.left(), view.left() + 16.0);
        let bar = Bar::place(Axis::Vertical, view, 2000.0, 0.0).unwrap();
        assert_eq!(bar.track.right(), view.right() - tokens::SCROLL_EDGE);
    }

    #[test]
    fn dragging_the_thumb_scrolls() {
        let ctx = egui::Context::default();
        frame(&ctx, vec![]);
        let x = 400.0 - tokens::SCROLL_EDGE - 2.0;
        let press = |down, at: Pos2| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: down,
            modifiers: Default::default(),
        };
        let start = pos2(x, 10.0);
        let end = start + Vec2::new(0.0, 120.0);
        frame(
            &ctx,
            vec![egui::Event::PointerMoved(start), press(true, start)],
        );
        frame(
            &ctx,
            vec![egui::Event::PointerMoved(start + vec2(0.0, 20.0))],
        );
        frame(&ctx, vec![egui::Event::PointerMoved(end)]);
        frame(&ctx, vec![press(false, end)]);
        frame(&ctx, vec![]);
        let mut offset = 0.0;
        let mut last = ctx.run_ui(Default::default(), |ui| {
            let id = ui.make_persistent_id(IdSalt::new(IdSalt::new("test")));
            offset = scroll_area::State::load(ui.ctx(), id)
                .expect("the scroll view kept its state")
                .clamped_offset()
                .y;
        });
        last.textures_delta.clear();
        assert!(offset > 100.0, "the list scrolled to {offset}");
    }
}
