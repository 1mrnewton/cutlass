//! The 6px gaps between the library, the preview, the inspector, and the timeline.
//!
//! Each gap is the drag handle. `docs/desktop-layout.md`.

use eframe::egui::{self, CursorIcon, Pos2, Rect, Sense, WidgetInfo, pos2};

use crate::theme::{GAP_PX, TIMELINE_MIN, TIMELINE_RESERVE};
use crate::words::{self, Phrase};

const LIBRARY_DEFAULT: f32 = 368.0;
const LIBRARY_MIN: f32 = 220.0;
const INSPECTOR_DEFAULT: f32 = 300.0;
const INSPECTOR_MIN: f32 = 240.0;
const TIMELINE_DEFAULT: f32 = 420.0;

/// Outer sizes of the three resizable regions, including their gap margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Panes {
    library: f32,
    inspector: f32,
    timeline: f32,
}

impl Panes {
    pub(crate) fn new() -> Self {
        Self {
            library: LIBRARY_DEFAULT,
            inspector: INSPECTOR_DEFAULT,
            timeline: TIMELINE_DEFAULT,
        }
    }

    pub(crate) fn library(self) -> f32 {
        self.library
    }

    pub(crate) fn inspector(self) -> f32 {
        self.inspector
    }

    pub(crate) fn timeline(self) -> f32 {
        self.timeline
    }

    /// Keep every pane inside the body. The preview is whatever width remains.
    pub(crate) fn fit(&mut self, width: f32, height: f32) {
        self.timeline = timeline_height(self.timeline, height);
        self.library = library_width(self.library, width, self.inspector);
        self.inspector = inspector_width(self.inspector, width, self.library);
    }
}

/// Apply a drag that started on a gap, then clamp to the body.
///
/// The edge follows the pointer. A press and release in the same frame still
/// counts once the pointer has moved off the gap.
pub(crate) fn drag(ui: &egui::Ui, panes: &mut Panes, body: Rect) {
    let seams = seams(body, panes);
    if let Some(pos) = drag_point(ui, gap_id("library"), seams.library, true) {
        panes.library = library_width(pos.x - body.left(), body.width(), panes.inspector);
    }
    if let Some(pos) = drag_point(ui, gap_id("inspector"), seams.inspector, true) {
        panes.inspector = inspector_width(body.right() - pos.x, body.width(), panes.library);
    }
    if let Some(pos) = drag_point(ui, gap_id("timeline"), seams.timeline, false) {
        panes.timeline = timeline_height(body.bottom() - pos.y, body.height());
    }
    panes.fit(body.width(), body.height());
}

/// The three gap rects. Nothing is painted: a hover draws no line.
pub(crate) fn show(ui: &mut egui::Ui, body: Rect, panes: &Panes) {
    let seams = seams(body, panes);
    grab(
        ui,
        seams.library,
        gap_id("library"),
        words::of(ui, Phrase::ResizeLibrary),
        CursorIcon::ResizeHorizontal,
    );
    grab(
        ui,
        seams.inspector,
        gap_id("inspector"),
        words::of(ui, Phrase::ResizeInspector),
        CursorIcon::ResizeHorizontal,
    );
    grab(
        ui,
        seams.timeline,
        gap_id("timeline"),
        words::of(ui, Phrase::ResizeTimeline),
        CursorIcon::ResizeVertical,
    );
}

struct Seams {
    library: Rect,
    inspector: Rect,
    timeline: Rect,
}

fn seams(body: Rect, panes: &Panes) -> Seams {
    let gap = f32::from(GAP_PX);
    let timeline_top = body.bottom() - panes.timeline;
    let above = Rect::from_min_max(body.min, pos2(body.right(), timeline_top));
    Seams {
        library: Rect::from_min_max(
            pos2(body.left() + panes.library - gap, above.top()),
            pos2(body.left() + panes.library, above.bottom()),
        )
        .intersect(above),
        inspector: Rect::from_min_max(
            pos2(body.right() - panes.inspector, above.top()),
            pos2(body.right() - panes.inspector + gap, above.bottom()),
        )
        .intersect(above),
        timeline: Rect::from_min_max(
            pos2(body.left(), timeline_top),
            pos2(body.right(), timeline_top + gap),
        )
        .intersect(body),
    }
}

fn grab(ui: &mut egui::Ui, rect: Rect, id: egui::Id, name: &str, cursor: CursorIcon) {
    if !rect.is_positive() {
        return;
    }
    let response = ui.interact(rect, id, Sense::click_and_drag());
    if response.hovered() || response.dragged() {
        ui.ctx().set_cursor_icon(cursor);
    }
    response.widget_info(|| WidgetInfo::labeled(egui::Role::Splitter, true, name));
}

fn gap_id(name: &str) -> egui::Id {
    egui::Id::unique(("pane-gap", name))
}

#[derive(Clone, Copy, Default)]
struct PressOrigin {
    x: f32,
    y: f32,
}

/// Where the pointer is, once a press on this gap has moved.
///
/// The press is remembered until release. A move and a release that arrive in
/// one frame still move the edge, which is the whole drag egui would otherwise
/// drop.
fn drag_point(ui: &egui::Ui, id: egui::Id, rect: Rect, horizontal: bool) -> Option<Pos2> {
    let press_id = id.with("press-origin");
    let mut origin = ui
        .ctx()
        .data(|data| data.get_temp::<PressOrigin>(press_id))
        .map(|origin| pos2(origin.x, origin.y));
    let (press, release, latest) = ui.input(|input| {
        let mut press = None;
        let mut release = None;
        for event in &input.events {
            if let egui::Event::PointerButton {
                pos,
                button,
                pressed,
                ..
            } = event
                && *button == egui::PointerButton::Primary
            {
                if *pressed {
                    press = Some(*pos);
                } else {
                    release = Some(*pos);
                }
            }
        }
        (press, release, input.pointer.latest_pos())
    });
    if let Some(press) = press {
        origin = rect.contains(press).then_some(press);
    }
    let moved = origin.and_then(|origin| {
        let pos = release.or(latest)?;
        let distance = if horizontal {
            (pos.x - origin.x).abs()
        } else {
            (pos.y - origin.y).abs()
        };
        (distance > 1.0).then_some(pos)
    });
    if release.is_some() || press.is_some_and(|press| !rect.contains(press)) {
        origin = None;
    }
    ui.ctx().data_mut(|data| {
        if let Some(origin) = origin {
            data.insert_temp(
                press_id,
                PressOrigin {
                    x: origin.x,
                    y: origin.y,
                },
            );
        } else {
            data.remove_temp::<PressOrigin>(press_id);
        }
    });
    moved
}

fn timeline_height(requested: f32, available: f32) -> f32 {
    let max = (available - TIMELINE_RESERVE).max(0.0);
    let min = TIMELINE_MIN.min(max);
    requested.clamp(min, max)
}

fn library_width(requested: f32, available: f32, inspector: f32) -> f32 {
    let max = (available - inspector).max(0.0);
    let min = LIBRARY_MIN.min(max);
    requested.clamp(min, max)
}

fn inspector_width(requested: f32, available: f32, library: f32) -> f32 {
    let max = (available - library).max(0.0);
    let min = INSPECTOR_MIN.min(max);
    requested.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use eframe::egui::{self, Rect, pos2, vec2};

    use super::{LIBRARY_MIN, Panes, inspector_width, library_width, seams, timeline_height};

    fn body() -> Rect {
        Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 762.0))
    }

    #[test]
    fn the_gap_above_the_timeline_is_the_handle_and_the_preview_stays() {
        let panes = Panes::new();
        let seams = seams(body(), &panes);
        assert!((seams.timeline.top() - 342.0).abs() < f32::EPSILON);
        assert!((seams.timeline.height() - 6.0).abs() < f32::EPSILON);
        assert!((seams.timeline.width() - 1280.0).abs() < f32::EPSILON);
        assert!((timeline_height(10_000.0, 762.0) - 494.0).abs() < f32::EPSILON);
        assert!((timeline_height(1.0, 762.0) - 160.0).abs() < f32::EPSILON);
        assert_eq!(timeline_height(10_000.0, 200.0), 0.0);
    }

    #[test]
    fn a_side_gap_widens_its_panel_and_stops_before_the_other() {
        let mut panes = Panes::new();
        panes.library = library_width(panes.library + 80.0, 1280.0, panes.inspector);
        assert!((panes.library() - 448.0).abs() < f32::EPSILON);
        panes.inspector = inspector_width(panes.inspector + 40.0, 1280.0, panes.library);
        assert!((panes.inspector() - 340.0).abs() < f32::EPSILON);
        panes.library = library_width(10_000.0, 1280.0, panes.inspector);
        assert!((panes.library() - (1280.0 - panes.inspector())).abs() < f32::EPSILON);
        panes.inspector = inspector_width(10_000.0, 1280.0, panes.library);
        assert!((panes.inspector() - (1280.0 - panes.library())).abs() < f32::EPSILON);

        assert!((library_width(1.0, 1280.0, 300.0) - LIBRARY_MIN).abs() < f32::EPSILON);
        assert!((inspector_width(1.0, 400.0, 220.0) - 180.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dragging_the_gap_above_the_timeline_grows_it() {
        let ctx = egui::Context::default();
        let mut panes = Panes::new();
        let screen = Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 800.0));
        let mut time = 0.0;
        let mut pass = |events: Vec<egui::Event>, panes: &mut Panes| {
            time += 0.05;
            let raw = egui::RawInput {
                screen_rect: Some(screen),
                time: Some(time),
                events,
                ..Default::default()
            };
            let mut output = ctx.run_ui(raw, |ui| {
                egui::Panel::top("head").exact_size(38.0).show(ui, |_| {});
                let body = ui.available_rect_before_wrap();
                super::drag(ui, panes, body);
                egui::Panel::bottom("timeline")
                    .resizable(false)
                    .exact_size(panes.timeline())
                    .frame(crate::theme::gap_above())
                    .show_separator_line(false)
                    .show(ui, |ui| {
                        ui.allocate_rect(ui.max_rect(), egui::Sense::hover());
                    });
                egui::Panel::left("library")
                    .resizable(false)
                    .exact_size(panes.library())
                    .frame(crate::theme::gap_right())
                    .show_separator_line(false)
                    .show(ui, |ui| {
                        ui.allocate_rect(ui.max_rect(), egui::Sense::hover());
                    });
                egui::Panel::right("inspector")
                    .resizable(false)
                    .exact_size(panes.inspector())
                    .frame(crate::theme::gap_left())
                    .show_separator_line(false)
                    .show(ui, |ui| {
                        ui.allocate_rect(ui.max_rect(), egui::Sense::hover());
                    });
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.allocate_rect(ui.max_rect(), egui::Sense::hover());
                });
                super::show(ui, body, panes);
            });
            output.textures_delta.clear();
        };

        pass(vec![], &mut panes);
        let on_gap = pos2(640.0, 383.0);
        pass(
            vec![
                egui::Event::PointerMoved(on_gap),
                egui::Event::PointerButton {
                    pos: on_gap,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            &mut panes,
        );
        assert!((panes.timeline() - 420.0).abs() < f32::EPSILON);
        pass(
            vec![egui::Event::PointerMoved(pos2(640.0, 300.0))],
            &mut panes,
        );
        assert!(
            (panes.timeline() - 494.0).abs() < 1.0,
            "timeline {}",
            panes.timeline()
        );

        pass(
            vec![egui::Event::PointerButton {
                pos: pos2(640.0, 300.0),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
            &mut panes,
        );
        let on_library = pos2(365.0, 200.0);
        pass(
            vec![
                egui::Event::PointerMoved(on_library),
                egui::Event::PointerButton {
                    pos: on_library,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            &mut panes,
        );
        pass(
            vec![egui::Event::PointerMoved(pos2(480.0, 200.0))],
            &mut panes,
        );
        assert!(
            (panes.library() - 480.0).abs() < 1.0,
            "library {}",
            panes.library()
        );

        let mut release = |pos: egui::Pos2, panes: &mut Panes| {
            pass(
                vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                }],
                panes,
            );
        };
        release(pos2(480.0, 200.0), &mut panes);

        let on_library_again = pos2(477.0, 180.0);
        pass(
            vec![
                egui::Event::PointerMoved(on_library_again),
                egui::Event::PointerButton {
                    pos: on_library_again,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            &mut panes,
        );
        pass(
            vec![
                egui::Event::PointerMoved(pos2(560.0, 180.0)),
                egui::Event::PointerButton {
                    pos: pos2(560.0, 180.0),
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            &mut panes,
        );
        assert!(
            (panes.library() - 560.0).abs() < 1.0,
            "library after a release in the move frame {}",
            panes.library()
        );

        let on_inspector = pos2(983.0, 200.0);
        pass(
            vec![
                egui::Event::PointerMoved(on_inspector),
                egui::Event::PointerButton {
                    pos: on_inspector,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerMoved(pos2(900.0, 200.0)),
                egui::Event::PointerButton {
                    pos: pos2(900.0, 200.0),
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            &mut panes,
        );
        assert!(
            (panes.inspector() - 380.0).abs() < 1.0,
            "inspector {}",
            panes.inspector()
        );
    }

    #[test]
    fn each_side_gap_is_six_pixels_on_the_outer_edge() {
        let panes = Panes::new();
        let seams = seams(body(), &panes);
        assert!((seams.library.left() - 362.0).abs() < f32::EPSILON);
        assert!((seams.library.width() - 6.0).abs() < f32::EPSILON);
        assert!((seams.library.bottom() - 342.0).abs() < f32::EPSILON);
        assert!((seams.inspector.left() - 980.0).abs() < f32::EPSILON);
        assert!((seams.inspector.width() - 6.0).abs() < f32::EPSILON);
        assert!((seams.inspector.bottom() - seams.library.bottom()).abs() < f32::EPSILON);
    }
}
