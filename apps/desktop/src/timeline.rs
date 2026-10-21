//! The project's tracks, drawn bottom to top, with the ruler above them.

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, Sense, Stroke, pos2, vec2};

use cutlass_project::{Command, FrameRate, History, Project, SegmentId, Track, TrackId, TrackKind};

use crate::ruler;
use crate::theme;

const HEAD: f32 = 130.0;
const COVER: f32 = 72.0;
const RULER_HEIGHT: f32 = 44.0;
const MAIN_HEIGHT: f32 = 64.0;
const LANE_HEIGHT: f32 = 40.0;
const GAP: f32 = 8.0;

struct Lane {
    label: &'static str,
    main: bool,
    empty: bool,
    top: f32,
    height: f32,
}

pub fn delete_selected(
    history: &mut History,
    project: &mut Project,
    selected: &mut Vec<SegmentId>,
) -> bool {
    if selected.is_empty() {
        return false;
    }
    let mut commands = Vec::new();
    for id in selected.iter().copied() {
        let on_main = project
            .timeline()
            .main_track()
            .segments()
            .iter()
            .any(|segment| segment.id() == id);
        if on_main {
            commands.push(Command::RemoveSegment(id));
        } else {
            commands.insert(0, Command::RemoveSegment(id));
        }
    }
    if history.apply(project, &commands).is_err() {
        return false;
    }
    selected.clear();
    true
}

fn choose(selected: &mut Vec<SegmentId>, hit: Option<SegmentId>) {
    selected.clear();
    if let Some(id) = hit {
        selected.push(id);
    }
}

fn toggle_selected(selected: &mut Vec<SegmentId>, id: SegmentId) {
    if let Some(index) = selected.iter().position(|kept| *kept == id) {
        selected.remove(index);
    } else {
        selected.push(id);
    }
}

fn chosen(selected: &[SegmentId], id: SegmentId) -> bool {
    selected.contains(&id)
}

const BOX_SLOP: f32 = 4.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Picked {
    Vertical(usize),
    Horizontal(usize),
}

#[derive(Clone, Default)]
struct Guides {
    vertical: Vec<i64>,
    horizontal: Vec<f32>,
    selected: Option<Picked>,
    pull: Option<egui::Pos2>,
    row: Option<egui::Pos2>,
}

#[derive(Clone)]
struct BoxSelect {
    origin: egui::Pos2,
    additive: bool,
    kept: Vec<SegmentId>,
}

/// A clip following the pointer. The move is written when the pointer goes up.
pub struct ClipDrag {
    segment: SegmentId,
    origin: egui::Pos2,
    start_us: i64,
    from: TrackId,
}

/// An edge following the pointer. The trim is written when the pointer goes up.
pub struct TrimDrag {
    segment: SegmentId,
    edge: Edge,
    origin_x: f32,
}

#[derive(Clone, Copy)]
enum Edge {
    Start,
    End,
}

pub fn show(
    ui: &mut egui::Ui,
    project: &Project,
    playhead: &mut i64,
    selected: &mut Vec<SegmentId>,
    drag: &mut Option<ClipDrag>,
    trim: &mut Option<TrimDrag>,
    split: &mut bool,
) -> Vec<Command> {
    let full = ui.max_rect();
    let ruler_rect = Rect::from_min_size(full.min, vec2(full.width(), RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler_rect.left_bottom(), full.max);
    selected.retain(|id| project.timeline().segment(*id).is_some());
    if drag
        .as_ref()
        .is_some_and(|drag| project.timeline().segment(drag.segment).is_none())
    {
        *drag = None;
    }
    if trim
        .as_ref()
        .is_some_and(|trim| project.timeline().segment(trim.segment).is_none())
    {
        *trim = None;
    }
    let mut commands = Vec::new();
    let mut box_rect = None;
    let pointer = ui.input(|input| input.pointer.interact_pos());
    let snap_id = ui.scope_id().with("snap-on");
    let mut snap = ui
        .data(|data| data.get_temp::<bool>(snap_id))
        .unwrap_or(true);
    let guides_id = ui.scope_id().with("reference-lines");
    let mut guides = ui
        .data(|data| data.get_temp::<Guides>(guides_id))
        .unwrap_or_default();
    let lane = lane_left(ruler_rect);
    let lane_width = (ruler_rect.right() - lane).max(0.0);
    let duration = project.duration().value;
    let rate = project.settings().rate().rate();
    let zoom_id = ui.scope_id().with("timeline-zoom");
    let mut view = if duration > 0 {
        ui.data(|data| data.get_temp::<View>(zoom_id))
            .filter(|stored| stored.pixel_size.is_finite() && stored.pixel_size > 0.0)
            .unwrap_or_else(|| View::fit(lane_width, duration))
    } else {
        ui.data_mut(|data| data.remove::<View>(zoom_id));
        View::fit(lane_width, duration)
    };
    if duration > 0 && lane_width > 0.0 {
        clamp_view(&mut view, lane_width, duration, rate);
        let factor = f64::from(ui.input(|input| input.zoom_delta()));
        if let Some(point) = pointer.filter(|point| full.contains(*point)) {
            zoom_around(&mut view, lane, point.x, factor, lane_width, duration, rate);
        }
        let anchor = view.x(lane, *playhead).clamp(lane, lane + lane_width);
        match zoom_controls(ui, ruler_rect, true) {
            Some(ZoomStep::In) => {
                zoom_around(&mut view, lane, anchor, 2.0, lane_width, duration, rate);
            }
            Some(ZoomStep::Out) => {
                zoom_around(&mut view, lane, anchor, 0.5, lane_width, duration, rate);
            }
            Some(ZoomStep::Fit) => {
                view = View::fit(lane_width, duration);
                clamp_view(&mut view, lane_width, duration, rate);
            }
            None => {}
        }
        ui.data_mut(|data| data.insert_temp(zoom_id, view));
    } else {
        let _ = zoom_controls(ui, ruler_rect, false);
    }
    if tracks.height() > 0.0 {
        let pressed = ui.input(|input| input.pointer.primary_pressed());
        let released = ui.input(|input| input.pointer.primary_released());
        let shift = ui.input(|input| input.modifiers.shift);
        let box_id = ui.scope_id().with("box-select");
        if pressed && let Some(point) = pointer {
            let mark = playhead_mark(ruler_rect, project, *playhead, view);
            let on_buttons = switch_rects(ruler_rect)
                .iter()
                .chain(zoom_rects(ruler_rect).iter())
                .any(|button| button.contains(point));
            let on_ruler = ruler_rect.contains(point)
                && point.x >= lane_left(ruler_rect)
                && !mark.is_some_and(|mark| mark.contains(point));
            let on_corner =
                ruler_rect.contains(point) && point.x < lane_left(ruler_rect) && !on_buttons;
            if on_ruler {
                guides.pull = Some(point);
                guides.row = None;
                *drag = None;
                *trim = None;
            } else if on_corner {
                guides.row = Some(point);
                guides.pull = None;
                *drag = None;
                *trim = None;
            }
        }
        let from_ruler = released && (guides.pull.is_some() || guides.row.is_some());
        if released
            && let Some(origin) = guides.pull.take()
            && pointer.is_some_and(|point| point.y > ruler_rect.bottom() + 8.0)
        {
            let time = time_at(ruler_rect, project, origin.x, view);
            guides.selected = Some(Picked::Vertical(keep_line(&mut guides.vertical, time)));
            selected.clear();
        }
        if released
            && guides.row.take().is_some()
            && let Some(point) = pointer.filter(|point| point.y > ruler_rect.bottom() + 8.0)
        {
            let fraction = row_fraction(tracks, point.y);
            guides.selected = Some(Picked::Horizontal(keep_row(
                &mut guides.horizontal,
                fraction,
                tracks.height(),
            )));
            selected.clear();
        }
        let typing = ui.memory(|memory| memory.focused().is_some());
        let delete_line = ui.input(|input| {
            input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace)
        });
        if !typing && delete_line {
            match guides.selected {
                Some(Picked::Vertical(index)) if index < guides.vertical.len() => {
                    guides.vertical.remove(index);
                    guides.selected = None;
                }
                Some(Picked::Horizontal(index)) if index < guides.horizontal.len() => {
                    guides.horizontal.remove(index);
                    guides.selected = None;
                }
                _ => {}
            }
        }
        if pressed && let Some(point) = pointer.filter(|point| tracks.contains(*point)) {
            ui.data_mut(|data| data.remove::<BoxSelect>(box_id));
            guides.selected = None;
            if let Some(picked) = picked_guide(&guides, tracks, point, view) {
                guides.selected = Some(picked);
                *drag = None;
                *trim = None;
                guides.pull = None;
                guides.row = None;
                choose(selected, None);
            } else if *split {
                *drag = None;
                *trim = None;
                choose(selected, segment_at(project, tracks, point, view));
            } else if let Some(id) = shift
                .then(|| segment_at(project, tracks, point, view))
                .flatten()
            {
                *drag = None;
                *trim = None;
                toggle_selected(selected, id);
            } else if !shift && let Some(active) = trim_from(project, tracks, point, view) {
                choose(selected, Some(active.segment));
                *drag = None;
                *trim = Some(active);
            } else {
                *trim = None;
                let hit = if shift {
                    None
                } else {
                    segment_at(project, tracks, point, view)
                };
                if hit.is_none() {
                    let kept = selected.clone();
                    if !shift {
                        selected.clear();
                    }
                    *drag = None;
                    ui.data_mut(|data| {
                        data.insert_temp(
                            box_id,
                            BoxSelect {
                                origin: point,
                                additive: shift,
                                kept,
                            },
                        );
                    });
                } else {
                    choose(selected, hit);
                    *drag = hit.and_then(|id| drag_from(project, id, point));
                }
            }
        }
        if let Some(boxing) = ui.data(|data| data.get_temp::<BoxSelect>(box_id))
            && let Some(point) = pointer
            && boxing.origin.distance(point) >= BOX_SLOP
        {
            let hits = clips_in(
                project,
                tracks,
                Rect::from_two_pos(boxing.origin, point),
                view,
            );
            *selected = if boxing.additive {
                unite(&boxing.kept, hits)
            } else {
                hits
            };
            box_rect = Some(Rect::from_two_pos(boxing.origin, point));
        }
        if !ui.input(|input| input.pointer.primary_down()) {
            ui.data_mut(|data| data.remove::<BoxSelect>(box_id));
        }
        if released {
            if *split && !from_ruler {
                drag.take();
                trim.take();
                if ui.input(|input| input.pointer.primary_clicked())
                    && let Some(point) = pointer.filter(|point| tracks.contains(*point))
                    && let Some(command) = split_command(project, tracks, point, view)
                {
                    commands.push(command);
                }
            } else if let Some(active) = trim.take() {
                commands = trim_commands(
                    project,
                    tracks,
                    &active,
                    pointer,
                    &Align {
                        snap,
                        playhead: *playhead,
                        guides: &guides.vertical,
                        view,
                    },
                );
            } else if let Some(active) = drag.take()
                && let Some(command) = move_command(
                    project,
                    tracks,
                    &active,
                    pointer,
                    &Align {
                        snap,
                        playhead: *playhead,
                        guides: &guides.vertical,
                        view,
                    },
                )
            {
                commands.push(command);
            }
        }
    }
    paint_ruler(ui, ruler_rect, project, view);
    ruler_switches(
        ui,
        ruler_rect,
        project.duration().value > 0,
        split,
        &mut snap,
    );
    ui.data_mut(|data| data.insert_temp(snap_id, snap));
    ui.data_mut(|data| data.insert_temp(guides_id, guides.clone()));
    if *split && pointer.is_some_and(|point| tracks.contains(point)) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    if tracks.height() > 0.0 {
        for (track, lane) in project
            .timeline()
            .tracks()
            .iter()
            .zip(layout(project.timeline().tracks(), tracks))
        {
            let lane_rect = Rect::from_min_size(
                pos2(lane_left(tracks), lane.top),
                vec2((tracks.right() - lane_left(tracks)).max(0.0), lane.height),
            );
            ui.painter().rect_filled(
                lane_rect,
                0,
                if lane.main { theme::LANE } else { theme::PANEL },
            );
            if lane.empty {
                ui.painter().rect_stroke(
                    lane_rect,
                    0,
                    Stroke::new(1.0, theme::LANE_LINE),
                    egui::StrokeKind::Inside,
                );
                ui.painter().text(
                    lane_rect.left_center() + vec2(14.0, 0.0),
                    Align2::LEFT_CENTER,
                    "Drag a clip here",
                    FontId::new(12.0, FontFamily::Proportional),
                    theme::MUTED,
                );
            }
            ui.painter().text(
                pos2(tracks.left() + 12.0, lane_rect.center().y),
                Align2::LEFT_CENTER,
                lane.label,
                FontId::new(12.0, FontFamily::Proportional),
                theme::MUTED,
            );
            paint_segments(
                ui,
                track,
                lane_rect,
                lane.main,
                &ClipPaint {
                    project,
                    pixel_size: view.pixel_size,
                    origin_us: view.origin_us,
                    selected,
                    drag: drag.as_ref(),
                    trim: trim.as_ref(),
                    pointer,
                    snap,
                    playhead: *playhead,
                    guides: &guides.vertical,
                },
            );
        }
    }
    paint_guides(ui, tracks, project, &guides, pointer, view);
    paint_playhead(ui, ruler_rect, tracks, project, playhead, view);
    if snap
        && let Some(x) = snap_guide(
            project,
            tracks,
            drag.as_ref(),
            trim.as_ref(),
            pointer,
            &Align {
                snap,
                playhead: *playhead,
                guides: &guides.vertical,
                view,
            },
        )
    {
        let bottom = tracks.bottom().max(ruler_rect.bottom());
        ui.painter().vline(
            x,
            ruler_rect.bottom()..=bottom,
            Stroke::new(1.0, theme::EXPORT),
        );
    }
    if let Some(rect) = box_rect {
        ui.painter()
            .rect_filled(rect, 0.0, Color32::from_white_alpha(28));
        ui.painter().rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, Color32::WHITE),
            egui::StrokeKind::Outside,
        );
    }
    commands
}

fn lane_left(area: Rect) -> f32 {
    area.left() + HEAD + COVER
}

#[derive(Clone, Copy)]
struct View {
    pixel_size: f64,
    origin_us: i64,
}

impl View {
    fn fit(width: f32, duration_us: i64) -> Self {
        Self {
            pixel_size: ruler::opening_pixel_size(width, duration_us),
            origin_us: 0,
        }
    }

    fn x(self, left: f32, us: i64) -> f32 {
        if !(self.pixel_size.is_finite() && self.pixel_size > 0.0) {
            return left;
        }
        left + ((us - self.origin_us) as f64 * self.pixel_size) as f32
    }

    fn us_at(self, left: f32, x: f32) -> i64 {
        if !(self.pixel_size.is_finite() && self.pixel_size > 0.0) {
            return self.origin_us.max(0);
        }
        let raw = f64::from(x - left) / self.pixel_size;
        if !raw.is_finite() {
            return self.origin_us.max(0);
        }
        let delta = raw.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64;
        self.origin_us.saturating_add(delta).max(0)
    }
}

struct Align<'a> {
    snap: bool,
    playhead: i64,
    guides: &'a [i64],
    view: View,
}

fn clamp_view(view: &mut View, width: f32, duration_us: i64, rate: FrameRate) {
    let min = ruler::quarter_pixel_size(width, duration_us);
    let max = ruler::finest_pixel_size(rate);
    if view.pixel_size.is_finite() && max >= min && max > 0.0 {
        view.pixel_size = view.pixel_size.clamp(min, max);
    }
    if !(view.pixel_size.is_finite() && view.pixel_size > 0.0) {
        view.origin_us = 0;
        return;
    }
    let visible = (f64::from(width) / view.pixel_size).round();
    let visible = if visible.is_finite() && visible >= 0.0 {
        visible.min(i64::MAX as f64) as i64
    } else {
        0
    };
    let max_origin = duration_us.max(0).saturating_sub(visible).max(0);
    view.origin_us = view.origin_us.clamp(0, max_origin);
}

fn zoom_around(
    view: &mut View,
    left: f32,
    x: f32,
    factor: f64,
    width: f32,
    duration_us: i64,
    rate: FrameRate,
) {
    let old = view.pixel_size;
    if !(old.is_finite() && old > 0.0 && factor.is_finite() && factor > 0.0 && factor != 1.0) {
        return;
    }
    let time = view.us_at(left, x);
    view.pixel_size = old * factor;
    clamp_view(view, width, duration_us, rate);
    let delta = f64::from(x - left) / view.pixel_size;
    let delta = if delta.is_finite() {
        delta.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    } else {
        0
    };
    view.origin_us = time.saturating_sub(delta).max(0);
    clamp_view(view, width, duration_us, rate);
}

#[derive(Clone, Copy)]
enum ZoomStep {
    Out,
    Fit,
    In,
}

fn zoom_rects(ruler: Rect) -> [Rect; 3] {
    let top = ruler.center().y - 13.0;
    let x = ruler.left() + HEAD + 2.0;
    [
        Rect::from_min_size(pos2(x, top), vec2(18.0, 26.0)),
        Rect::from_min_size(pos2(x + 20.0, top), vec2(28.0, 26.0)),
        Rect::from_min_size(pos2(x + 50.0, top), vec2(18.0, 26.0)),
    ]
}

fn zoom_controls(ui: &mut egui::Ui, ruler: Rect, live: bool) -> Option<ZoomStep> {
    let rects = zoom_rects(ruler);
    let steps = [ZoomStep::Out, ZoomStep::Fit, ZoomStep::In];
    let labels = ["−", "Fit", "+"];
    let salts = ["zoom-out", "zoom-fit", "zoom-in"];
    let mut chosen = None;
    for index in 0..rects.len() {
        let response = ui.interact(
            rects[index],
            ui.scope_id().with(salts[index]),
            Sense::click(),
        );
        if live && response.clicked() {
            chosen = Some(steps[index]);
        }
        let color = if live { theme::INK } else { theme::FAINT };
        ui.painter().text(
            rects[index].center(),
            Align2::CENTER_CENTER,
            labels[index],
            FontId::new(11.0, FontFamily::Proportional),
            color,
        );
    }
    chosen
}

fn keep_line(lines: &mut Vec<i64>, time: i64) -> usize {
    if let Some(index) = lines.iter().position(|line| *line == time) {
        return index;
    }
    lines.push(time);
    lines.len() - 1
}

fn time_at(area: Rect, project: &Project, x: f32, view: View) -> i64 {
    let us = view.us_at(lane_left(area), x);
    let end = project.duration().value.max(us);
    ruler::seek(us, project.settings().rate().rate(), end)
}

fn playhead_mark(ruler: Rect, project: &Project, playhead: i64, view: View) -> Option<Rect> {
    if project.duration().value <= 0 {
        return None;
    }
    let left = lane_left(ruler);
    let width = (ruler.right() - left).max(0.0);
    if !(view.pixel_size.is_finite() && view.pixel_size > 0.0) {
        return None;
    }
    let x = view.x(left, playhead.max(0)).clamp(left, left + width);
    Some(Rect::from_center_size(
        pos2(x, ruler.bottom() - 6.0),
        vec2(11.0, 12.0),
    ))
}

fn line_gap(lines: &[i64], tracks: Rect, point: egui::Pos2, view: View) -> Option<(usize, f32)> {
    if !(view.pixel_size.is_finite() && view.pixel_size > 0.0) {
        return None;
    }
    let left = lane_left(tracks);
    lines.iter().enumerate().find_map(|(index, time)| {
        let x = view.x(left, *time);
        let gap = (point.x - x).abs();
        (gap <= 4.0).then_some((index, gap))
    })
}

fn row_fraction(tracks: Rect, y: f32) -> f32 {
    let height = tracks.height();
    if height <= 0.0 {
        return 0.0;
    }
    ((y - tracks.top()) / height).clamp(0.0, 1.0)
}

fn keep_row(lines: &mut Vec<f32>, fraction: f32, height: f32) -> usize {
    if height > 0.0
        && let Some(index) = lines
            .iter()
            .position(|line| (*line - fraction).abs() * height <= 1.0)
    {
        return index;
    }
    lines.push(fraction);
    lines.len() - 1
}

fn row_gap(lines: &[f32], tracks: Rect, point: egui::Pos2) -> Option<(usize, f32)> {
    let height = tracks.height();
    if height <= 0.0 {
        return None;
    }
    lines.iter().enumerate().find_map(|(index, fraction)| {
        let y = tracks.top() + fraction * height;
        let gap = (point.y - y).abs();
        (gap <= 4.0).then_some((index, gap))
    })
}

fn picked_guide(guides: &Guides, tracks: Rect, point: egui::Pos2, view: View) -> Option<Picked> {
    let vertical = line_gap(&guides.vertical, tracks, point, view);
    let horizontal = row_gap(&guides.horizontal, tracks, point);
    match (vertical, horizontal) {
        (Some((index, across)), Some((row, down))) => Some(if down < across {
            Picked::Horizontal(row)
        } else {
            Picked::Vertical(index)
        }),
        (Some((index, _)), None) => Some(Picked::Vertical(index)),
        (None, Some((row, _))) => Some(Picked::Horizontal(row)),
        (None, None) => None,
    }
}

fn paint_guides(
    ui: &egui::Ui,
    tracks: Rect,
    project: &Project,
    guides: &Guides,
    pointer: Option<egui::Pos2>,
    view: View,
) {
    let left = lane_left(tracks);
    if tracks.height() <= 0.0 {
        return;
    }
    if view.pixel_size.is_finite() && view.pixel_size > 0.0 {
        let draw = |time: i64, color: Color32| {
            ui.painter().vline(
                view.x(left, time),
                tracks.top()..=tracks.bottom(),
                Stroke::new(1.0, color),
            );
        };
        for (index, time) in guides.vertical.iter().enumerate() {
            let color = if guides.selected == Some(Picked::Vertical(index)) {
                theme::INK
            } else {
                theme::FAINT
            };
            draw(*time, color);
        }
        if let Some(origin) = guides.pull
            && pointer.is_some_and(|point| point.y > tracks.top())
        {
            draw(time_at(tracks, project, origin.x, view), theme::EXPORT);
        }
    }
    let draw_row = |fraction: f32, color: Color32| {
        let y = tracks.top() + fraction * tracks.height();
        ui.painter()
            .hline(tracks.left()..=tracks.right(), y, Stroke::new(1.0, color));
    };
    for (index, fraction) in guides.horizontal.iter().enumerate() {
        let color = if guides.selected == Some(Picked::Horizontal(index)) {
            theme::INK
        } else {
            theme::FAINT
        };
        draw_row(*fraction, color);
    }
    if let Some(point) = pointer.filter(|point| guides.row.is_some() && point.y > tracks.top()) {
        draw_row(row_fraction(tracks, point.y), theme::EXPORT);
    }
}

fn ruler_switches(
    ui: &mut egui::Ui,
    ruler: Rect,
    split_live: bool,
    split: &mut bool,
    snap: &mut bool,
) {
    let [split_button, snap_button] = switch_rects(ruler);
    ruler_switch(ui, split_button, "Split", "split-blade", split_live, split);
    ruler_switch(ui, snap_button, "Snap", "snap-switch", true, snap);
}

fn switch_rects(ruler: Rect) -> [Rect; 2] {
    let top = ruler.center().y - 13.0;
    [
        Rect::from_min_size(pos2(ruler.left() + 6.0, top), vec2(58.0, 26.0)),
        Rect::from_min_size(pos2(ruler.left() + 68.0, top), vec2(56.0, 26.0)),
    ]
}

fn ruler_switch(
    ui: &mut egui::Ui,
    button: Rect,
    label: &str,
    salt: &str,
    live: bool,
    on: &mut bool,
) {
    if !live {
        *on = false;
    }
    let response = ui.interact(button, ui.scope_id().with(salt), Sense::click());
    if live && response.clicked() {
        *on = !*on;
    }
    if *on {
        ui.painter().rect_filled(button, 6.0, theme::ACTIVE);
    }
    let color = if !live {
        theme::FAINT
    } else if *on {
        theme::EXPORT
    } else {
        theme::INK
    };
    ui.painter().text(
        button.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::new(12.0, FontFamily::Proportional),
        color,
    );
}

fn split_command(
    project: &Project,
    tracks: Rect,
    point: egui::Pos2,
    view: View,
) -> Option<Command> {
    let id = segment_at(project, tracks, point, view)?;
    let range = project.timeline().segment(id)?.timeline();
    if !(view.pixel_size.is_finite() && view.pixel_size > 0.0) {
        return None;
    }
    let at = ruler::seek(
        view.us_at(lane_left(tracks), point.x),
        project.settings().rate().rate(),
        project.duration().value,
    );
    if at <= range.start().value || at >= range.end().value {
        return None;
    }
    Some(Command::Split { segment: id, at })
}

fn clips_in(project: &Project, tracks: Rect, area: Rect, view: View) -> Vec<SegmentId> {
    if tracks.height() <= 0.0 {
        return Vec::new();
    }
    let width = (tracks.right() - lane_left(tracks)).max(0.0);
    let mut hits = Vec::new();
    for (track, lane) in project
        .timeline()
        .tracks()
        .iter()
        .zip(layout(project.timeline().tracks(), tracks))
    {
        let lane_rect =
            Rect::from_min_size(pos2(lane_left(tracks), lane.top), vec2(width, lane.height));
        for segment in track.segments() {
            let range = segment.timeline();
            let rect = clip_rect(
                lane_rect,
                lane.main,
                range.start().value,
                range.duration().value,
                view.pixel_size,
                view.origin_us,
            );
            if rect.intersects(area) {
                hits.push(segment.id());
            }
        }
    }
    hits
}

fn unite(kept: &[SegmentId], hits: Vec<SegmentId>) -> Vec<SegmentId> {
    let mut out = kept.to_vec();
    for id in hits {
        if !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

struct ClipPaint<'a> {
    project: &'a Project,
    pixel_size: f64,
    origin_us: i64,
    selected: &'a [SegmentId],
    drag: Option<&'a ClipDrag>,
    trim: Option<&'a TrimDrag>,
    pointer: Option<egui::Pos2>,
    snap: bool,
    playhead: i64,
    guides: &'a [i64],
}

fn paint_segments(ui: &mut egui::Ui, track: &Track, lane: Rect, main: bool, paint: &ClipPaint<'_>) {
    for segment in track.segments() {
        let range = segment.timeline();
        let rect = clip_rect(
            lane,
            main,
            range.start().value,
            range.duration().value,
            paint.pixel_size,
            paint.origin_us,
        );
        if rect.left() > lane.right() || rect.right() < lane.left() {
            continue;
        }
        let dragging = paint.drag.is_some_and(|drag| drag.segment == segment.id());
        let trimming = paint.trim.is_some_and(|trim| trim.segment == segment.id());
        let color = clip_color(track.kind());
        let resting = if dragging || trimming {
            color.gamma_multiply(0.35)
        } else {
            color
        };
        ui.painter().rect_filled(rect, 7.0, resting);
        if !dragging && !trimming && chosen(paint.selected, segment.id()) {
            stroke_clip(ui, rect);
        }
        if dragging {
            let offset = paint
                .drag
                .zip(paint.pointer)
                .map(|(drag, point)| drag_offset(paint, drag, point))
                .unwrap_or_default();
            let moved = rect.translate(offset);
            ui.painter().rect_filled(moved, 7.0, color);
            stroke_clip(ui, moved);
            paint_name(ui, paint.project, segment, moved);
        } else if trimming {
            let delta = paint
                .trim
                .zip(paint.pointer)
                .map(|(trim, point)| trim_offset(paint, trim, point.x))
                .unwrap_or(0.0);
            let edge = paint.trim.map(|trim| trim.edge).unwrap_or(Edge::End);
            let resized = trimmed_rect(rect, edge, delta);
            ui.painter().rect_filled(resized, 7.0, color);
            stroke_clip(ui, resized);
            paint_name(ui, paint.project, segment, resized);
        } else {
            paint_name(ui, paint.project, segment, rect);
        }
    }
}

fn stroke_clip(ui: &egui::Ui, rect: Rect) {
    ui.painter().rect_stroke(
        rect,
        7.0,
        Stroke::new(1.5, Color32::WHITE),
        egui::StrokeKind::Inside,
    );
}

fn paint_name(ui: &egui::Ui, project: &Project, segment: &cutlass_project::Segment, rect: Rect) {
    let name = project
        .asset(segment.asset_id())
        .map(|asset| file_name(asset.source()))
        .unwrap_or("");
    if name.is_empty() {
        return;
    }
    ui.painter().with_clip_rect(rect.shrink(6.0)).text(
        rect.left_center() + vec2(8.0, 0.0),
        Align2::LEFT_CENTER,
        name,
        FontId::new(11.0, FontFamily::Proportional),
        Color32::WHITE,
    );
}

fn drag_from(project: &Project, id: SegmentId, origin: egui::Pos2) -> Option<ClipDrag> {
    let segment = project.timeline().segment(id)?;
    let from = project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.segments().iter().any(|segment| segment.id() == id))?
        .id();
    Some(ClipDrag {
        segment: id,
        origin,
        start_us: segment.timeline().start().value,
        from,
    })
}

const SNAP_PX: f32 = 10.0;

fn snap_targets(project: &Project, playhead: i64, guides: &[i64]) -> Vec<i64> {
    let mut targets = vec![0, project.duration().value.max(0), playhead.max(0)];
    targets.extend(guides);
    for track in project.timeline().tracks() {
        for segment in track.segments() {
            let range = segment.timeline();
            targets.push(range.start().value);
            targets.push(range.end().value);
        }
    }
    targets.sort_unstable();
    targets.dedup();
    targets
}

fn gap(left: i64, right: i64) -> i64 {
    i64::try_from(left.abs_diff(right)).unwrap_or(i64::MAX)
}

fn snap_limit_us(pixel_size: f64) -> i64 {
    if !(pixel_size.is_finite() && pixel_size > 0.0) {
        return 0;
    }
    let limit = (f64::from(SNAP_PX) / pixel_size).round();
    if !limit.is_finite() || limit < 0.0 {
        return 0;
    }
    limit.min(i64::MAX as f64) as i64
}

/// The time to land on, and the guide time when a target is inside the pull.
fn snap_point(time: i64, targets: &[i64], pixel_size: f64) -> (i64, Option<i64>) {
    let limit = snap_limit_us(pixel_size);
    let mut best: Option<(i64, i64)> = None;
    for &target in targets {
        let distance = gap(time, target);
        let closer = best.is_none_or(|(best_distance, best_target)| {
            distance < best_distance || (distance == best_distance && target < best_target)
        });
        if distance <= limit && closer {
            best = Some((distance, target));
        }
    }
    match best {
        Some((_, target)) => (target, Some(target)),
        None => (time, None),
    }
}

/// A moved clip keeps its length. Either edge can pull, and the start is what moves.
fn snap_span(start: i64, duration: i64, targets: &[i64], pixel_size: f64) -> (i64, Option<i64>) {
    let limit = snap_limit_us(pixel_size);
    let mut best: Option<(i64, i64, i64, i64)> = None;
    for &target in targets {
        if target >= 0 {
            let distance = gap(start, target);
            consider(distance, target, target, start, &mut best);
        }
        if let Some(new_start) = target.checked_sub(duration).filter(|start| *start >= 0) {
            let end = start.saturating_add(duration);
            consider(gap(end, target), new_start, target, start, &mut best);
        }
    }
    match best {
        Some((distance, _, line, new_start)) if distance <= limit => (new_start, Some(line)),
        _ => (start, None),
    }
}

fn consider(
    distance: i64,
    new_start: i64,
    line: i64,
    raw: i64,
    best: &mut Option<(i64, i64, i64, i64)>,
) {
    let shift = gap(new_start, raw);
    let replace = best.is_none_or(|(best_distance, best_shift, best_line, _)| {
        distance < best_distance
            || (distance == best_distance && shift < best_shift)
            || (distance == best_distance && shift == best_shift && line < best_line)
    });
    if replace {
        *best = Some((distance, shift, line, new_start));
    }
}

fn snap_guide(
    project: &Project,
    tracks: Rect,
    drag: Option<&ClipDrag>,
    trim: Option<&TrimDrag>,
    pointer: Option<egui::Pos2>,
    align: &Align<'_>,
) -> Option<f32> {
    let point = pointer?;
    let pixel_size = align.view.pixel_size;
    if !(pixel_size.is_finite() && pixel_size > 0.0) {
        return None;
    }
    let targets = snap_targets(project, align.playhead, align.guides);
    let line = if let Some(drag) = drag {
        let duration = project
            .timeline()
            .segment(drag.segment)?
            .timeline()
            .duration()
            .value;
        let raw = moved_start(drag.start_us, point.x - drag.origin.x, pixel_size);
        snap_span(raw, duration, &targets, pixel_size).1
    } else if let Some(trim) = trim {
        let range = project.timeline().segment(trim.segment)?.timeline();
        let delta = delta_us(point.x - trim.origin_x, pixel_size);
        let edge = match trim.edge {
            Edge::End => range.end().value.saturating_add(delta),
            Edge::Start => range.start().value.saturating_add(delta),
        };
        snap_point(edge, &targets, pixel_size).1
    } else {
        None
    }?;
    Some(align.view.x(lane_left(tracks), line))
}

fn drag_offset(paint: &ClipPaint<'_>, drag: &ClipDrag, point: egui::Pos2) -> egui::Vec2 {
    let raw = point - drag.origin;
    if !(paint.snap && paint.pixel_size.is_finite() && paint.pixel_size > 0.0) {
        return raw;
    }
    let Some(segment) = paint.project.timeline().segment(drag.segment) else {
        return raw;
    };
    let raw_start = moved_start(drag.start_us, raw.x, paint.pixel_size);
    let (start, line) = snap_span(
        raw_start,
        segment.timeline().duration().value,
        &snap_targets(paint.project, paint.playhead, paint.guides),
        paint.pixel_size,
    );
    if line.is_none() {
        return raw;
    }
    let x = ((start - drag.start_us) as f64 * paint.pixel_size) as f32;
    vec2(x, raw.y)
}

fn trim_offset(paint: &ClipPaint<'_>, trim: &TrimDrag, pointer_x: f32) -> f32 {
    let raw = pointer_x - trim.origin_x;
    if !(paint.snap && paint.pixel_size.is_finite() && paint.pixel_size > 0.0) {
        return raw;
    }
    let Some(segment) = paint.project.timeline().segment(trim.segment) else {
        return raw;
    };
    let range = segment.timeline();
    let delta = delta_us(raw, paint.pixel_size);
    let edge = match trim.edge {
        Edge::End => range.end().value.saturating_add(delta),
        Edge::Start => range.start().value.saturating_add(delta),
    };
    let (snapped, line) = snap_point(
        edge,
        &snap_targets(paint.project, paint.playhead, paint.guides),
        paint.pixel_size,
    );
    if line.is_none() {
        return raw;
    }
    let adjust = snapped.saturating_sub(edge);
    raw + (adjust as f64 * paint.pixel_size) as f32
}

fn moved_start(start_us: i64, delta_px: f32, pixel_size: f64) -> i64 {
    if !(pixel_size.is_finite() && pixel_size > 0.0 && delta_px.is_finite()) {
        return start_us.max(0);
    }
    let delta = f64::from(delta_px) / pixel_size;
    if !delta.is_finite() {
        return start_us.max(0);
    }
    let delta = delta.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    start_us.saturating_add(delta).max(0)
}

fn move_command(
    project: &Project,
    tracks: Rect,
    drag: &ClipDrag,
    pointer: Option<egui::Pos2>,
    align: &Align<'_>,
) -> Option<Command> {
    let point = pointer?;
    let pixel_size = align.view.pixel_size;
    let mut start = moved_start(drag.start_us, point.x - drag.origin.x, pixel_size);
    if align.snap
        && let Some(segment) = project.timeline().segment(drag.segment)
    {
        start = snap_span(
            start,
            segment.timeline().duration().value,
            &snap_targets(project, align.playhead, align.guides),
            pixel_size,
        )
        .0;
    }
    let track = lane_under(project, tracks, point).filter(|id| *id != drag.from);
    if start == drag.start_us && track.is_none() {
        return None;
    }
    Some(Command::Move {
        segment: drag.segment,
        track,
        start,
    })
}

fn lane_under(project: &Project, tracks: Rect, point: egui::Pos2) -> Option<TrackId> {
    let width = (tracks.right() - lane_left(tracks)).max(0.0);
    for (track, lane) in project
        .timeline()
        .tracks()
        .iter()
        .zip(layout(project.timeline().tracks(), tracks))
        .rev()
    {
        let lane_rect =
            Rect::from_min_size(pos2(lane_left(tracks), lane.top), vec2(width, lane.height));
        if lane_rect.contains(point) {
            return Some(track.id());
        }
    }
    None
}

const EDGE_PX: f32 = 6.0;

fn trim_from(project: &Project, tracks: Rect, point: egui::Pos2, view: View) -> Option<TrimDrag> {
    if tracks.height() <= 0.0 {
        return None;
    }
    let width = (tracks.right() - lane_left(tracks)).max(0.0);
    for (track, lane) in project
        .timeline()
        .tracks()
        .iter()
        .zip(layout(project.timeline().tracks(), tracks))
        .rev()
    {
        let lane_rect =
            Rect::from_min_size(pos2(lane_left(tracks), lane.top), vec2(width, lane.height));
        for segment in track.segments().iter().rev() {
            let range = segment.timeline();
            let rect = clip_rect(
                lane_rect,
                lane.main,
                range.start().value,
                range.duration().value,
                view.pixel_size,
                view.origin_us,
            );
            if !rect.contains(point) {
                continue;
            }
            if rect.width() <= 16.0 {
                return None;
            }
            let edge = if point.x <= rect.left() + EDGE_PX {
                Some(Edge::Start)
            } else if point.x >= rect.right() - EDGE_PX {
                Some(Edge::End)
            } else {
                None
            };
            return edge.map(|edge| TrimDrag {
                segment: segment.id(),
                edge,
                origin_x: point.x,
            });
        }
    }
    None
}

fn trimmed_rect(rect: Rect, edge: Edge, delta_px: f32) -> Rect {
    let min_width = 8.0;
    match edge {
        Edge::Start => {
            let left = (rect.left() + delta_px).min(rect.right() - min_width);
            Rect::from_min_max(pos2(left, rect.top()), rect.max)
        }
        Edge::End => {
            let right = (rect.right() + delta_px).max(rect.left() + min_width);
            Rect::from_min_max(rect.min, pos2(right, rect.bottom()))
        }
    }
}

fn delta_us(delta_px: f32, pixel_size: f64) -> i64 {
    if !(pixel_size.is_finite() && pixel_size > 0.0 && delta_px.is_finite()) {
        return 0;
    }
    let delta = f64::from(delta_px) / pixel_size;
    if !delta.is_finite() {
        return 0;
    }
    delta.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64
}

fn trim_commands(
    project: &Project,
    _tracks: Rect,
    trim: &TrimDrag,
    pointer: Option<egui::Pos2>,
    align: &Align<'_>,
) -> Vec<Command> {
    let Some(point) = pointer else {
        return Vec::new();
    };
    let Some(segment) = project.timeline().segment(trim.segment) else {
        return Vec::new();
    };
    let Some(asset) = project.asset(segment.asset_id()) else {
        return Vec::new();
    };
    let pixel_size = align.view.pixel_size;
    let mut delta = delta_us(point.x - trim.origin_x, pixel_size);
    if align.snap {
        let range = segment.timeline();
        let edge = match trim.edge {
            Edge::End => range.end().value.saturating_add(delta),
            Edge::Start => range.start().value.saturating_add(delta),
        };
        let snapped = snap_point(
            edge,
            &snap_targets(project, align.playhead, align.guides),
            pixel_size,
        )
        .0;
        delta = delta.saturating_add(snapped.saturating_sub(edge));
    }
    let edge = match trim.edge {
        Edge::Start => cutlass_project::TrimEdge::Start,
        Edge::End => cutlass_project::TrimEdge::End,
    };
    cutlass_project::commands_for_trim(segment, asset, edge, delta).unwrap_or_default()
}

fn segment_at(project: &Project, tracks: Rect, point: egui::Pos2, view: View) -> Option<SegmentId> {
    if tracks.height() <= 0.0 {
        return None;
    }
    let lanes = layout(project.timeline().tracks(), tracks);
    for (track, lane) in project.timeline().tracks().iter().zip(lanes).rev() {
        let lane_rect = Rect::from_min_size(
            pos2(lane_left(tracks), lane.top),
            vec2((tracks.right() - lane_left(tracks)).max(0.0), lane.height),
        );
        for segment in track.segments().iter().rev() {
            let range = segment.timeline();
            let rect = clip_rect(
                lane_rect,
                lane.main,
                range.start().value,
                range.duration().value,
                view.pixel_size,
                view.origin_us,
            );
            if rect.contains(point) {
                return Some(segment.id());
            }
        }
    }
    None
}

fn clip_rect(
    lane: Rect,
    main: bool,
    start_us: i64,
    duration_us: i64,
    pixel_size: f64,
    origin_us: i64,
) -> Rect {
    let (inset, height) = if main { (8.0, 48.0) } else { (6.0, 28.0) };
    let width = (duration_us.max(0) as f64 * pixel_size).max(8.0) as f32;
    let x = lane.left() + ((start_us.max(0) - origin_us) as f64 * pixel_size) as f32;
    Rect::from_min_size(pos2(x, lane.top() + inset), vec2(width, height))
}

pub(crate) fn file_name(source: &str) -> &str {
    let trimmed = source.trim_end_matches(['/', '\\']);
    trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed)
}

fn clip_color(kind: TrackKind) -> Color32 {
    match kind {
        TrackKind::Video => theme::VIDEO,
        TrackKind::Audio => theme::AUDIO,
        TrackKind::Text => theme::TEXT,
        TrackKind::Sticker => theme::STICKER,
        TrackKind::Effect | TrackKind::PluginEffect => theme::EFFECT,
        TrackKind::Filter => theme::FILTER,
        TrackKind::Adjust => theme::ADJUST,
    }
}

fn paint_ruler(ui: &mut egui::Ui, ruler_rect: Rect, project: &Project, view: View) {
    let left = lane_left(ruler_rect);
    let width = (ruler_rect.right() - left).max(0.0);
    let bottom = ruler_rect.bottom();
    for mark in ruler::marks(
        project.settings().rate(),
        view.pixel_size,
        width,
        view.origin_us,
    ) {
        let x = view.x(left, mark.us);
        let height = if mark.major { 12.0 } else { 6.0 };
        ui.painter().vline(
            x,
            (bottom - height)..=bottom,
            Stroke::new(1.0, theme::MUTED),
        );
        if let Some(label) = mark.label {
            ui.painter().text(
                pos2(x + 4.0, bottom - height - 2.0),
                Align2::LEFT_BOTTOM,
                label,
                FontId::new(10.0, FontFamily::Proportional),
                theme::MUTED,
            );
        }
    }
}

fn paint_playhead(
    ui: &mut egui::Ui,
    ruler_rect: Rect,
    tracks: Rect,
    project: &Project,
    playhead: &mut i64,
    view: View,
) {
    let left = lane_left(ruler_rect);
    let width = (ruler_rect.right() - left).max(0.0);
    let duration = project.duration().value;
    let live = duration > 0;
    let rate = project.settings().rate().rate();
    if !live {
        *playhead = 0;
    }
    *playhead = ruler::seek(*playhead, rate, duration);
    let color = if live {
        Color32::WHITE
    } else {
        Color32::from_white_alpha(96)
    };
    let bottom = ruler_rect.bottom();
    let x_of = |us: i64| {
        if view.pixel_size > 0.0 {
            view.x(left, us).clamp(left, left + width)
        } else {
            left
        }
    };
    if live && view.pixel_size > 0.0 {
        let mark = Rect::from_center_size(pos2(x_of(*playhead), bottom - 6.0), vec2(11.0, 12.0));
        let response = ui.interact(mark, ui.scope_id().with("playhead"), Sense::drag());
        if response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            *playhead = ruler::seek(view.us_at(left, pointer.x), rate, duration);
        }
    }
    let x = x_of(*playhead);
    let mark = Rect::from_center_size(pos2(x, bottom - 6.0), vec2(11.0, 12.0));
    ui.painter().rect_filled(mark, 2.0, color);
    if tracks.bottom() >= bottom {
        ui.painter()
            .vline(x, bottom..=tracks.bottom(), Stroke::new(1.0, color));
    }
}

fn layout(tracks: &[Track], area: Rect) -> Vec<Lane> {
    let Some(main) = tracks.iter().position(Track::is_main) else {
        return Vec::new();
    };
    let height = |index: usize| {
        if tracks[index].is_main() {
            MAIN_HEIGHT
        } else {
            LANE_HEIGHT
        }
    };
    let mut tops = vec![0.0; tracks.len()];
    tops[main] = area.center().y - height(main) / 2.0;
    let mut below = tops[main] + height(main);
    for index in (0..main).rev() {
        below += GAP;
        tops[index] = below;
        below += height(index);
    }
    let mut above = tops[main];
    for (index, _) in tracks.iter().enumerate().skip(main + 1) {
        above -= GAP + height(index);
        tops[index] = above;
    }
    tracks
        .iter()
        .enumerate()
        .map(|(index, track)| Lane {
            label: label(track),
            main: track.is_main(),
            empty: track.is_main() && track.segments().is_empty(),
            top: tops[index],
            height: height(index),
        })
        .collect()
}

fn label(track: &Track) -> &'static str {
    if track.is_main() {
        return "Main";
    }
    match track.kind() {
        TrackKind::Video => "Video",
        TrackKind::Audio => "Audio",
        TrackKind::Text => "Text",
        TrackKind::Sticker => "Sticker",
        TrackKind::Effect | TrackKind::PluginEffect => "Effect",
        TrackKind::Filter => "Filter",
        TrackKind::Adjust => "Adjust",
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment, Speed, TimeRange,
    };
    use eframe::egui::{Rect, pos2, vec2};

    use super::{clip_rect, lane_left, layout, segment_at};

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "Untitled")
    }

    fn fitted(tracks: Rect, duration: i64) -> super::View {
        super::View::fit((tracks.right() - lane_left(tracks)).max(0.0), duration)
    }

    fn loose(tracks: Rect, duration: i64) -> super::Align<'static> {
        super::Align {
            snap: false,
            playhead: 0,
            guides: &[],
            view: fitted(tracks, duration),
        }
    }

    #[test]
    fn an_empty_cut_shows_main_in_the_middle() {
        let project = project();
        let lanes = layout(
            project.timeline().tracks(),
            Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 400.0)),
        );
        assert_eq!(lanes.len(), 1);
        assert_eq!(lanes[0].label, "Main");
        assert!(lanes[0].empty);
        assert!((lanes[0].top - 168.0).abs() < f32::EPSILON);
        assert!(
            (super::lane_left(Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 400.0))) - 202.0)
                .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn audio_sits_under_main_and_an_overlay_sits_above_it() {
        let mut history = History::new();
        let mut project = project();
        let video = Asset::video(
            "file:///v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("timescale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut project, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let (sound, picture) = {
            let stored = project.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(30_000, duration.timescale),
            )
            .expect("source");
            (
                Segment::media(stored, source, 0, Channels::Sound, Speed::ONE).expect("sound"),
                Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("picture"),
            )
        };
        history
            .apply(
                &mut project,
                &[
                    Command::Place {
                        track: None,
                        segment: sound,
                    },
                    Command::Place {
                        track: None,
                        segment: picture,
                    },
                ],
            )
            .expect("place");
        let lanes = layout(
            project.timeline().tracks(),
            Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 400.0)),
        );
        let main = lanes
            .iter()
            .find(|lane| lane.label == "Main")
            .expect("main");
        let audio = lanes
            .iter()
            .find(|lane| lane.label == "Audio")
            .expect("audio");
        let overlay = lanes
            .iter()
            .find(|lane| lane.label == "Video")
            .expect("overlay");
        assert!(audio.top > main.top);
        assert!(overlay.top < main.top);
        assert!(main.empty);
    }

    #[test]
    fn a_one_second_clip_fills_three_hundred_pixels_at_the_one_second_step() {
        let lane = Rect::from_min_size(pos2(202.0, 100.0), vec2(900.0, 64.0));
        let rect = super::clip_rect(lane, true, 0, 1_000_000, 3e-4, 0);
        assert!((rect.left() - 202.0).abs() < 0.01);
        assert!((rect.width() - 300.0).abs() < 0.01);
        assert!((rect.top() - 108.0).abs() < f32::EPSILON);
        assert!((rect.height() - 48.0).abs() < f32::EPSILON);
        let short = super::clip_rect(lane, false, 0, 1, 3e-4, 0);
        assert!((short.width() - 8.0).abs() < f32::EPSILON);
        assert!((short.height() - 28.0).abs() < f32::EPSILON);
        assert_eq!(super::file_name("file:///cuts/v.mp4"), "v.mp4");
    }

    #[test]
    fn a_point_on_a_clip_finds_it_and_a_point_on_the_head_does_not() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        let id = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Segment(id)] => *id,
            other => panic!("expected a segment, got {other:?}"),
        };
        let tracks = Rect::from_min_size(pos2(0.0, 44.0), vec2(800.0, 356.0));
        let lane = layout(cut.timeline().tracks(), tracks)
            .into_iter()
            .find(|lane| lane.main)
            .expect("main");
        let width = (tracks.right() - lane_left(tracks)).max(0.0);
        let pixel_size = crate::ruler::opening_pixel_size(width, cut.duration().value);
        let lane_rect =
            Rect::from_min_size(pos2(lane_left(tracks), lane.top), vec2(width, lane.height));
        let range = cut.timeline().segment(id).expect("segment").timeline();
        let rect = clip_rect(
            lane_rect,
            true,
            range.start().value,
            range.duration().value,
            pixel_size,
            0,
        );
        assert_eq!(
            segment_at(
                &cut,
                tracks,
                rect.center(),
                fitted(tracks, cut.duration().value)
            ),
            Some(id)
        );
        assert_eq!(
            segment_at(
                &cut,
                tracks,
                pos2(12.0, rect.center().y),
                fitted(tracks, cut.duration().value)
            ),
            None
        );
    }

    #[test]
    fn delete_removes_the_selected_clip_and_keeps_main() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        let id = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Segment(id)] => *id,
            other => panic!("expected a segment, got {other:?}"),
        };
        let mut selected = Vec::new();
        super::choose(&mut selected, Some(id));
        assert_eq!(selected, vec![id]);
        super::toggle_selected(&mut selected, id);
        assert!(selected.is_empty());
        super::toggle_selected(&mut selected, id);
        assert_eq!(selected, vec![id]);
        assert!(super::delete_selected(
            &mut history,
            &mut cut,
            &mut selected
        ));
        assert!(selected.is_empty());
        assert!(cut.timeline().segment(id).is_none());
        assert!(cut.timeline().main_track().is_main());
        assert!(cut.timeline().main_track().segments().is_empty());
        assert!(!super::delete_selected(
            &mut history,
            &mut cut,
            &mut selected
        ));
    }

    #[test]
    fn a_drag_writes_the_move_and_a_still_pointer_does_not() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let (id, from) = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Opened { segment, track }] => (*segment, *track),
            other => panic!("expected a new lane, got {other:?}"),
        };
        let tracks = Rect::from_min_size(pos2(0.0, 0.0), vec2(602.0, 400.0));
        let lane = layout(cut.timeline().tracks(), tracks)
            .into_iter()
            .find(|lane| !lane.main)
            .expect("overlay");
        let origin = pos2(lane_left(tracks) + 40.0, lane.top + 10.0);
        let drag = super::ClipDrag {
            segment: id,
            origin,
            start_us: 0,
            from,
        };
        assert!(
            super::move_command(
                &cut,
                tracks,
                &drag,
                Some(origin),
                &loose(tracks, cut.duration().value)
            )
            .is_none()
        );
        let pixel_size = crate::ruler::opening_pixel_size(400.0, cut.duration().value);
        let landed = super::moved_start(0, 150.0, pixel_size);
        let command = super::move_command(
            &cut,
            tracks,
            &drag,
            Some(pos2(origin.x + 150.0, origin.y)),
            &loose(tracks, cut.duration().value),
        )
        .expect("move");
        assert_eq!(
            command,
            Command::Move {
                segment: id,
                track: None,
                start: landed,
            }
        );
        history.apply(&mut cut, &[command]).expect("apply");
        assert_eq!(
            cut.timeline()
                .segment(id)
                .expect("segment")
                .timeline()
                .start()
                .value,
            landed
        );
        assert_eq!(super::moved_start(100, -10_000.0, pixel_size), 0);

        let main = layout(cut.timeline().tracks(), tracks)
            .into_iter()
            .find(|lane| lane.main)
            .expect("main");
        let onto_main = super::move_command(
            &cut,
            tracks,
            &super::ClipDrag {
                segment: id,
                origin,
                start_us: landed,
                from,
            },
            Some(pos2(origin.x, main.top + 10.0)),
            &loose(tracks, cut.duration().value),
        )
        .expect("lane");
        let main_id = cut.timeline().main_track().id();
        assert_eq!(
            onto_main,
            Command::Move {
                segment: id,
                track: Some(main_id),
                start: landed,
            }
        );
    }

    #[test]
    fn dragging_the_right_edge_shortens_the_clip() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let id = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Opened { segment, .. }] => *segment,
            other => panic!("expected a new lane, got {other:?}"),
        };
        let tracks = Rect::from_min_size(pos2(0.0, 0.0), vec2(602.0, 400.0));
        let lane = layout(cut.timeline().tracks(), tracks)
            .into_iter()
            .find(|lane| !lane.main)
            .expect("overlay");
        let width = (tracks.right() - lane_left(tracks)).max(0.0);
        let pixel_size = crate::ruler::opening_pixel_size(width, cut.duration().value);
        let range = cut.timeline().segment(id).expect("segment").timeline();
        let rect = clip_rect(
            Rect::from_min_size(pos2(lane_left(tracks), lane.top), vec2(width, lane.height)),
            false,
            range.start().value,
            range.duration().value,
            pixel_size,
            0,
        );
        let edge = pos2(rect.right() - 2.0, rect.center().y);
        let trim = super::trim_from(&cut, tracks, edge, fitted(tracks, cut.duration().value))
            .expect("edge");
        assert!(
            super::trim_from(
                &cut,
                tracks,
                rect.center(),
                fitted(tracks, cut.duration().value)
            )
            .is_none()
        );
        let before = range.duration().value;
        let commands = super::trim_commands(
            &cut,
            tracks,
            &trim,
            Some(pos2(edge.x - 80.0, edge.y)),
            &loose(tracks, cut.duration().value),
        );
        assert!(!commands.is_empty());
        history.apply(&mut cut, &commands).expect("trim");
        let after = cut.timeline().segment(id).expect("segment").timeline();
        assert_eq!(after.start().value, 0);
        assert!(after.duration().value < before);
    }

    #[test]
    fn a_click_in_the_middle_splits_and_a_click_on_the_edge_does_not() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let id = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Opened { segment, .. }] => *segment,
            other => panic!("expected a new lane, got {other:?}"),
        };
        let tracks = Rect::from_min_size(pos2(0.0, 0.0), vec2(602.0, 400.0));
        let lane = layout(cut.timeline().tracks(), tracks)
            .into_iter()
            .find(|lane| !lane.main)
            .expect("overlay");
        let width = (tracks.right() - lane_left(tracks)).max(0.0);
        let pixel_size = crate::ruler::opening_pixel_size(width, cut.duration().value);
        let range = cut.timeline().segment(id).expect("segment").timeline();
        let rect = clip_rect(
            Rect::from_min_size(pos2(lane_left(tracks), lane.top), vec2(width, lane.height)),
            false,
            range.start().value,
            range.duration().value,
            pixel_size,
            0,
        );
        let view = fitted(tracks, cut.duration().value);
        assert!(super::split_command(&cut, tracks, rect.left_center(), view).is_none());
        assert_eq!(super::clips_in(&cut, tracks, rect, view), vec![id]);
        assert!(
            super::clips_in(
                &cut,
                tracks,
                Rect::from_min_size(pos2(0.0, 0.0), vec2(12.0, 12.0)),
                view,
            )
            .is_empty()
        );
        assert_eq!(super::unite(&[id], vec![id]), vec![id]);
        let command = super::split_command(&cut, tracks, rect.center(), view).expect("split");
        let Command::Split { segment, at } = command else {
            panic!("expected a split, got {command:?}");
        };
        assert_eq!(segment, id);
        assert!(at > range.start().value && at < range.end().value);
        history.apply(&mut cut, &[command]).expect("apply");
        let left = cut.timeline().segment(id).expect("left").timeline();
        let right = cut
            .timeline()
            .tracks()
            .iter()
            .find(|track| !track.is_main())
            .expect("overlay")
            .segments()
            .iter()
            .find(|piece| piece.id() != id)
            .expect("right");
        assert!(left.duration().value < range.duration().value);
        assert_eq!(right.timeline().start().value, left.end().value);
        assert_eq!(left.start().value, 0);
    }

    #[test]
    fn a_near_edge_snaps_and_a_far_one_stays_put() {
        let pixel_size = 0.0003;
        let targets = [0, 500_000, 1_000_000];
        let (start, line) = super::snap_span(480_000, 200_000, &targets, pixel_size);
        assert_eq!(start, 500_000);
        assert_eq!(line, Some(500_000));
        let (start, line) = super::snap_span(400_000, 200_000, &targets, pixel_size);
        assert_eq!(start, 400_000);
        assert!(line.is_none());
        let (start, line) = super::snap_span(790_000, 200_000, &targets, pixel_size);
        assert_eq!(start, 800_000);
        assert_eq!(line, Some(1_000_000));
        assert_eq!(super::snap_point(10, &targets, pixel_size), (0, Some(0)));
        let cut = project();
        assert_eq!(super::snap_targets(&cut, 0, &[250_000]), vec![0, 250_000]);
    }

    #[test]
    fn a_vertical_line_sits_on_a_frame_and_a_click_near_it_selects_it() {
        let cut = project();
        let tracks = Rect::from_min_size(pos2(0.0, 44.0), vec2(600.0, 200.0));
        let left = super::lane_left(tracks);
        let view = fitted(tracks, cut.duration().value);
        assert_eq!(super::time_at(tracks, &cut, left, view), 0);
        assert!(super::time_at(tracks, &cut, left + 40.0, view) > 0);
        assert!(super::playhead_mark(tracks, &cut, 0, view).is_none());
        let mut lines = vec![0];
        assert_eq!(super::keep_line(&mut lines, 0), 0);
        assert_eq!(lines, vec![0]);
        assert_eq!(
            super::line_gap(&lines, tracks, pos2(left, 80.0), view).map(|(index, _)| index),
            Some(0)
        );
        assert_eq!(
            super::line_gap(&lines, tracks, pos2(left + 20.0, 80.0), view),
            None
        );
    }

    #[test]
    fn zooming_keeps_the_time_under_the_pointer() {
        let mut view = super::View {
            pixel_size: 0.0002,
            origin_us: 0,
        };
        super::zoom_around(
            &mut view,
            0.0,
            100.0,
            2.0,
            400.0,
            2_000_000,
            cutlass_project::FrameRate::Fps30,
        );
        assert!((view.pixel_size - 0.0004).abs() < 1e-12);
        assert_eq!(view.us_at(0.0, 100.0), 500_000);
        super::zoom_around(
            &mut view,
            0.0,
            100.0,
            0.01,
            400.0,
            2_000_000,
            cutlass_project::FrameRate::Fps30,
        );
        let quarter = crate::ruler::quarter_pixel_size(400.0, 2_000_000);
        assert!((view.pixel_size - quarter).abs() < 1e-12);
        assert_eq!(view.origin_us, 0);
    }

    #[test]
    fn a_horizontal_line_keeps_its_fraction_of_the_tracks() {
        let tracks = Rect::from_min_size(pos2(0.0, 44.0), vec2(600.0, 200.0));
        assert_eq!(super::row_fraction(tracks, 44.0), 0.0);
        assert_eq!(super::row_fraction(tracks, 144.0), 0.5);
        assert_eq!(super::row_fraction(tracks, 1_000.0), 1.0);
        let mut rows = Vec::new();
        assert_eq!(super::keep_row(&mut rows, 0.5, tracks.height()), 0);
        assert_eq!(super::keep_row(&mut rows, 0.502, tracks.height()), 0);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            super::row_gap(&rows, tracks, pos2(10.0, 144.0)),
            Some((0, 0.0))
        );
        assert_eq!(super::row_gap(&rows, tracks, pos2(10.0, 160.0)), None);
        let ruler = Rect::from_min_size(pos2(0.0, 0.0), vec2(600.0, 44.0));
        let cover = pos2(super::HEAD + 10.0, 22.0);
        assert!(cover.x < super::lane_left(ruler));
        assert!(
            super::switch_rects(ruler)
                .iter()
                .all(|button| !button.contains(cover))
        );
    }
}
