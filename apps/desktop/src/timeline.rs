//! The project's tracks, drawn bottom to top, with the ruler above them.

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, Sense, Stroke, pos2, vec2};

use cutlass_project::{
    Command, FrameRate, History, Project, SEQUENCE_TIMESCALE, SegmentId, Track, TrackId, TrackKind,
};

use crate::icons::{self, Icon};
use crate::ruler;
use crate::theme;
use crate::words::{self, Language, Phrase};

const HEAD: f32 = 130.0;
const COVER: f32 = 72.0;
const RULER_HEIGHT: f32 = 44.0;
const MAIN_HEIGHT: f32 = 64.0;
const LANE_HEIGHT: f32 = 40.0;
const GAP: f32 = 8.0;

struct Lane {
    main: bool,
    empty: bool,
    top: f32,
    height: f32,
}

pub fn delete_selected(
    history: &mut History,
    project: &mut Project,
    selected: &mut Vec<SegmentId>,
    lanes: &Lanes,
) -> bool {
    if selected.is_empty() {
        return false;
    }
    let mut commands = Vec::new();
    for id in selected.iter().copied() {
        if blocks_edit(project, lanes, id) {
            continue;
        }
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
    if commands.is_empty() {
        return false;
    }
    if history.apply(project, &commands).is_err() {
        return false;
    }
    selected.retain(|id| project.timeline().segment(*id).is_some());
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

/// A mark on a clip, measured from that clip's start so a move carries it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ClipStamp {
    segment: SegmentId,
    offset: i64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PickedStamp {
    Ruler(usize),
    Clip(usize),
    Beat(usize),
}

/// Hide, mute, and lock for one lane. View state, until the project stores them.
#[derive(Clone, Copy, PartialEq, Eq)]
struct LaneFlag {
    track: TrackId,
    locked: bool,
    hidden: bool,
    muted: bool,
}

/// Lane flags for this session. A missing lane is shown, heard, and open.
#[derive(Clone, Default)]
pub(crate) struct Lanes {
    rows: Vec<LaneFlag>,
}

fn lane_id() -> egui::Id {
    egui::Id::unique("timeline-lanes")
}

pub(crate) fn lane_memory(ui: &egui::Ui) -> Lanes {
    ui.data(|data| data.get_temp(lane_id())).unwrap_or_default()
}

pub(crate) fn segment_locked(ui: &egui::Ui, project: &Project, id: SegmentId) -> bool {
    segment_lane_locked(project, &lane_memory(ui), id)
}

pub(crate) fn hidden_tracks(ui: &egui::Ui) -> Vec<TrackId> {
    lane_memory(ui)
        .rows
        .into_iter()
        .filter(|row| row.hidden)
        .map(|row| row.track)
        .collect()
}

fn retain_lanes(project: &Project, lanes: &mut Lanes) {
    lanes
        .rows
        .retain(|row| project.timeline().track(row.track).is_some());
}

fn lane_row(lanes: &Lanes, track: TrackId) -> LaneFlag {
    lanes
        .rows
        .iter()
        .copied()
        .find(|row| row.track == track)
        .unwrap_or(LaneFlag {
            track,
            locked: false,
            hidden: false,
            muted: false,
        })
}

fn remember_lane(lanes: &mut Lanes, row: LaneFlag) {
    lanes.rows.retain(|item| item.track != row.track);
    if row.locked || row.hidden || row.muted {
        lanes.rows.push(row);
    }
}

fn segment_track(project: &Project, id: SegmentId) -> Option<TrackId> {
    project.timeline().tracks().iter().find_map(|track| {
        track
            .segments()
            .iter()
            .any(|segment| segment.id() == id)
            .then_some(track.id())
    })
}

fn segment_lane_locked(project: &Project, lanes: &Lanes, id: SegmentId) -> bool {
    segment_track(project, id).is_some_and(|track| lane_row(lanes, track).locked)
}

/// The one main clip this range sits on. Two owners, or none, do not follow.
fn sole_owner(
    range: cutlass_project::TimeRange,
    mains: &[(SegmentId, cutlass_project::TimeRange)],
) -> Option<SegmentId> {
    let mut found = None;
    for (id, main) in mains {
        if range.overlaps(*main) {
            if found.is_some() {
                return None;
            }
            found = Some(*id);
        }
    }
    found
}

/// A main edit can slide a clip on a locked lane when linkage or the magnet is on.
fn ripples_locked(project: &Project, lanes: &Lanes, id: SegmentId) -> bool {
    let on_main = project
        .timeline()
        .main_track()
        .segments()
        .iter()
        .any(|segment| segment.id() == id);
    if !on_main {
        return false;
    }
    let magnet = project.settings().magnet();
    let linkage = project.settings().linkage();
    if !magnet && !linkage {
        return false;
    }
    let mains: Vec<_> = project
        .timeline()
        .main_track()
        .segments()
        .iter()
        .map(|segment| (segment.id(), segment.timeline()))
        .collect();
    for track in project.timeline().tracks() {
        if track.is_main() || !lane_row(lanes, track.id()).locked {
            continue;
        }
        for segment in track.segments() {
            let owner = sole_owner(segment.timeline(), &mains);
            if (magnet && owner.is_some()) || (linkage && owner == Some(id)) {
                return true;
            }
        }
    }
    false
}

fn blocks_edit(project: &Project, lanes: &Lanes, id: SegmentId) -> bool {
    segment_lane_locked(project, lanes, id) || ripples_locked(project, lanes, id)
}

pub(crate) fn edit_refused(project: &Project, lanes: &Lanes, id: SegmentId) -> bool {
    blocks_edit(project, lanes, id)
}

fn blocks_command(project: &Project, lanes: &Lanes, command: &Command) -> bool {
    match command {
        Command::Move { segment, track, .. } => {
            blocks_edit(project, lanes, *segment)
                || track.is_some_and(|id| lane_row(lanes, id).locked)
        }
        Command::Split { segment, .. }
        | Command::RemoveSegment(segment)
        | Command::Trim { segment, .. }
        | Command::SetSpeed { segment, .. } => blocks_edit(project, lanes, *segment),
        Command::SetTransform { segment, .. } => segment_lane_locked(project, lanes, *segment),
        _ => false,
    }
}

/// Marker flags and beat notes. View state, until the project stores them.
#[derive(Clone, Default)]
struct Stamps {
    ruler: Vec<i64>,
    clips: Vec<ClipStamp>,
    beats: Vec<ClipStamp>,
    selected: Option<PickedStamp>,
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
    crate::tip::region(ui, words::of(ui, Phrase::RegionTimeline));
    // The panel stores the height this ui claims. Painting the lanes does not
    // claim it, so a drag would snap back to the minimum on release.
    ui.take_available_height();
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
    let stamps_id = ui.scope_id().with("timeline-marks");
    let mut stamps = ui
        .data(|data| data.get_temp::<Stamps>(stamps_id))
        .unwrap_or_default();
    retain_stamps(project, &mut stamps);
    let marker_id = ui.scope_id().with("marker-mode");
    let beat_id = ui.scope_id().with("beat-mode");
    let mut marker = ui
        .data(|data| data.get_temp::<bool>(marker_id))
        .unwrap_or(false);
    let mut beat = ui
        .data(|data| data.get_temp::<bool>(beat_id))
        .unwrap_or(false);
    if duration == 0 {
        marker = false;
        beat = false;
    }
    let mark_times = stamp_times(project, &stamps);
    let mut lanes = lane_memory(ui);
    retain_lanes(project, &mut lanes);
    let mut note = None;
    if let Some(key) = crate::keys::tool(ui) {
        let mut modes = Modes {
            split,
            snap: &mut snap,
            marker: &mut marker,
            beat: &mut beat,
        };
        let (command, refusal, zoom) = apply_shortcut(
            key,
            duration > 0,
            project,
            &lanes,
            &mut modes,
            words::active(ui),
        );
        if let Some(command) = command {
            commands.push(command);
        }
        if let Some(text) = refusal {
            note = Some(text);
        }
        if duration > 0
            && lane_width > 0.0
            && let Some(step) = zoom
        {
            let anchor = view.x(lane, *playhead).clamp(lane, lane + lane_width);
            match step {
                ZoomStep::In => {
                    zoom_around(&mut view, lane, anchor, 2.0, lane_width, duration, rate);
                }
                ZoomStep::Out => {
                    zoom_around(&mut view, lane, anchor, 0.5, lane_width, duration, rate);
                }
                ZoomStep::Fit => {
                    view = View::fit(lane_width, duration);
                    clamp_view(&mut view, lane_width, duration, rate);
                }
            }
            ui.data_mut(|data| data.insert_temp(zoom_id, view));
        }
    }
    if tracks.height() > 0.0 {
        let held_id = ui.scope_id().with("held-locked");
        let mut held_locked = ui.data(|data| data.get_temp(held_id)).unwrap_or(false);
        let pressed = ui.input(|input| input.pointer.primary_pressed());
        let released = ui.input(|input| input.pointer.primary_released());
        let shift = ui.input(|input| input.modifiers.shift);
        let box_id = ui.scope_id().with("box-select");
        if pressed && let Some(point) = pointer {
            held_locked = false;
            let mark = playhead_mark(ruler_rect, project, *playhead, view);
            let on_buttons = on_tools(ruler_rect, point);
            let on_ruler = ruler_rect.contains(point)
                && point.x >= lane_left(ruler_rect)
                && !mark.is_some_and(|mark| mark.contains(point));
            let on_corner =
                ruler_rect.contains(point) && point.x < lane_left(ruler_rect) && !on_buttons;
            if on_ruler {
                if let Some(index) = near_us(view, lane_left(ruler_rect), &stamps.ruler, point.x) {
                    stamps.selected = Some(PickedStamp::Ruler(index));
                    guides.selected = None;
                    selected.clear();
                    guides.pull = None;
                    guides.row = None;
                } else if !marker {
                    guides.pull = Some(point);
                    guides.row = None;
                    *drag = None;
                    *trim = None;
                } else {
                    guides.pull = None;
                    guides.row = None;
                    *drag = None;
                    *trim = None;
                }
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
            match stamps.selected {
                Some(PickedStamp::Ruler(index)) if index < stamps.ruler.len() => {
                    stamps.ruler.remove(index);
                    stamps.selected = None;
                }
                Some(PickedStamp::Clip(index)) if index < stamps.clips.len() => {
                    stamps.clips.remove(index);
                    stamps.selected = None;
                }
                Some(PickedStamp::Beat(index)) if index < stamps.beats.len() => {
                    stamps.beats.remove(index);
                    stamps.selected = None;
                }
                _ => match guides.selected {
                    Some(Picked::Vertical(index)) if index < guides.vertical.len() => {
                        guides.vertical.remove(index);
                        guides.selected = None;
                    }
                    Some(Picked::Horizontal(index)) if index < guides.horizontal.len() => {
                        guides.horizontal.remove(index);
                        guides.selected = None;
                    }
                    _ => {}
                },
            }
        }
        if pressed
            && let Some(point) =
                pointer.filter(|point| tracks.contains(*point) && point.x >= lane_left(tracks))
        {
            ui.data_mut(|data| data.remove::<BoxSelect>(box_id));
            guides.selected = None;
            stamps.selected = None;
            if let Some(picked) = picked_guide(&guides, tracks, point, view) {
                guides.selected = Some(picked);
                *drag = None;
                *trim = None;
                guides.pull = None;
                guides.row = None;
                choose(selected, None);
            } else if let Some(picked) = picked_stamp(project, &stamps, tracks, point, view) {
                stamps.selected = Some(picked);
                *drag = None;
                *trim = None;
                guides.pull = None;
                guides.row = None;
                choose(selected, None);
            } else if *split || marker {
                *drag = None;
                *trim = None;
                choose(selected, segment_at(project, tracks, point, view));
            } else if beat
                && let Some(id) = segment_at(project, tracks, point, view)
                && segment_kind(project, id) == Some(TrackKind::Audio)
            {
                *drag = None;
                *trim = None;
                choose(selected, Some(id));
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
                *trim = if blocks_edit(project, &lanes, active.segment) {
                    held_locked = true;
                    None
                } else {
                    Some(active)
                };
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
                    *drag = hit
                        .filter(|id| {
                            if blocks_edit(project, &lanes, *id) {
                                held_locked = true;
                                false
                            } else {
                                true
                            }
                        })
                        .and_then(|id| drag_from(project, id, point));
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
            let clicked = ui.input(|input| input.pointer.primary_clicked());
            let on_locked_clip = pointer.is_some_and(|point| {
                segment_at(project, tracks, point, view)
                    .is_some_and(|id| segment_lane_locked(project, &lanes, id))
            });
            if clicked && on_locked_clip && (marker || beat) {
                note = Some(words::of(ui, Phrase::TrackIsLocked));
            } else if clicked
                && marker
                && !on_locked_clip
                && let Some(point) = pointer.filter(|point| !on_tools(ruler_rect, *point))
                && drop_marker(
                    project,
                    tracks,
                    ruler_rect,
                    point,
                    *playhead,
                    view,
                    &mut stamps,
                )
            {
                guides.selected = None;
                selected.clear();
            } else if clicked
                && beat
                && !on_locked_clip
                && let Some(point) = pointer.filter(|point| on_audio(project, tracks, *point, view))
                && drop_beat(project, tracks, point, view, &mut stamps)
            {
                guides.selected = None;
                selected.clear();
            } else if *split && !from_ruler {
                drag.take();
                trim.take();
                if clicked && let Some(point) = pointer.filter(|point| tracks.contains(*point)) {
                    if let Some(command) = split_command(project, tracks, point, view) {
                        if blocks_command(project, &lanes, &command) {
                            note = Some(words::of(ui, Phrase::TrackIsLocked));
                        } else {
                            commands.push(command);
                        }
                    } else if segment_at(project, tracks, point, view).is_some() {
                        note = Some(words::of(ui, Phrase::CutTooClose));
                    }
                }
            } else if let Some(active) = trim.take() {
                let next = trim_commands(
                    project,
                    tracks,
                    &active,
                    pointer,
                    &Align {
                        snap,
                        playhead: *playhead,
                        guides: &guides.vertical,
                        marks: &mark_times,
                        view,
                    },
                );
                if next
                    .iter()
                    .all(|command| !blocks_command(project, &lanes, command))
                {
                    commands = next;
                } else {
                    note = Some(words::of(ui, Phrase::TrackIsLocked));
                }
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
                        marks: &mark_times,
                        view,
                    },
                )
            {
                if blocks_command(project, &lanes, &command) {
                    note = Some(words::of(ui, Phrase::TrackIsLocked));
                } else {
                    commands.push(command);
                }
            }
            if !clicked && held_locked && note.is_none() {
                note = Some(words::of(ui, Phrase::TrackIsLocked));
            }
            held_locked = false;
        }
        ui.data_mut(|data| data.insert_temp(held_id, held_locked));
    }
    paint_ruler(ui, ruler_rect, project, view);
    let live = project.duration().value > 0;
    if ruler_switches(ui, ruler_rect, live, split, &mut snap) {
        marker = false;
        beat = false;
    }
    if let Some(text) = project_switches(ui, ruler_rect, project, &lanes, &mut commands) {
        note = Some(text);
    }
    mark_switches(ui, ruler_rect, live, &mut marker, &mut beat, split);
    ui.data_mut(|data| data.insert_temp(snap_id, snap));
    ui.data_mut(|data| data.insert_temp(guides_id, guides.clone()));
    ui.data_mut(|data| data.insert_temp(marker_id, marker));
    ui.data_mut(|data| data.insert_temp(beat_id, beat));
    if *split && pointer.is_some_and(|point| tracks.contains(point) && point.x >= lane_left(tracks))
        || marker
            && pointer.is_some_and(|point| {
                ruler_rect.contains(point) && point.x >= lane_left(ruler_rect)
                    || tracks.contains(point) && point.x >= lane_left(tracks)
            })
        || beat && pointer.is_some_and(|point| on_audio(project, tracks, point, view))
    {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    let mut remove = None;
    let mut clip_tips = Vec::new();
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
                    words::of(ui, Phrase::DragClip),
                    FontId::new(12.0, FontFamily::Proportional),
                    theme::MUTED,
                );
            }
            let head = Rect::from_min_max(
                pos2(tracks.left(), lane.top),
                pos2(tracks.left() + HEAD, lane.top + lane.height),
            );
            let mut row = lane_row(&lanes, track.id());
            paint_lane_label(ui, head, words::say(words::active(ui), lane_phrase(track)));
            if let Some(text) = paint_lane_flags(ui, head, track.id(), &mut row) {
                note = Some(text);
            }
            let dim = row.hidden || (row.muted && track.kind() == TrackKind::Audio);
            remember_lane(&mut lanes, row);
            paint_segments(
                ui,
                track,
                lane_rect,
                lane.main,
                &mut remove,
                &mut ClipPaint {
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
                    marks: &mark_times,
                    dim,
                    lanes: &lanes,
                    tips: &mut clip_tips,
                },
            );
        }
        if let Some(id) = remove {
            if blocks_edit(project, &lanes, id) {
                note = Some(words::of(ui, Phrase::TrackIsLocked));
            } else {
                commands.push(Command::RemoveSegment(id));
            }
        }
    }
    ui.data_mut(|data| data.insert_temp(lane_id(), lanes));
    paint_stamps(ui, project, tracks, ruler_rect, &stamps, view);
    for (response, text, picked) in clip_tips {
        crate::tip::show_selected(&response, &text, picked);
    }
    ui.data_mut(|data| data.insert_temp(stamps_id, stamps));
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
                marks: &mark_times,
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
    if let Some(text) = note {
        show_note(ui, text);
    }
    paint_note(ui, ruler_rect);
    commands
}

const TOAST_HOLD: f64 = 1.2;
const TOAST_FADE: f64 = 0.6;

#[derive(Clone)]
struct Note {
    text: String,
    at: f64,
}

fn note_id() -> egui::Id {
    egui::Id::unique("timeline-note")
}

pub(crate) fn show_note(ui: &egui::Ui, text: impl Into<String>) {
    let at = ui.input(|input| input.time);
    ui.data_mut(|data| {
        data.insert_temp(
            note_id(),
            Note {
                text: text.into(),
                at,
            },
        );
    });
}

#[cfg(test)]
pub(crate) fn note_text(ctx: &egui::Context) -> Option<String> {
    ctx.data(|data| data.get_temp::<Note>(note_id()).map(|note| note.text))
}

pub(crate) fn apply_note(error: &cutlass_project::Error, language: Language) -> &'static str {
    let phrase = match error {
        cutlass_project::Error::Overlap | cutlass_project::Error::LinkageOverlap(_) => {
            Phrase::SpotTaken
        }
        cutlass_project::Error::WrongTrack => Phrase::WrongLane,
        cutlass_project::Error::SplitOutside => Phrase::CutTooClose,
        cutlass_project::Error::SourceOutsideAsset | cutlass_project::Error::RoundsToZero => {
            Phrase::TrimLeaves
        }
        _ => Phrase::EditMissed,
    };
    words::say(language, phrase)
}

fn toast_alpha(age: f64) -> Option<u8> {
    if !(0.0..TOAST_HOLD + TOAST_FADE).contains(&age) {
        return None;
    }
    if age < TOAST_HOLD {
        return Some(255);
    }
    let faded = (TOAST_HOLD + TOAST_FADE - age) / TOAST_FADE * 255.0;
    let rounded = faded.round();
    if !(0.0..=255.0).contains(&rounded) {
        return None;
    }
    Some(rounded as u8)
}

fn paint_note(ui: &egui::Ui, ruler: Rect) {
    let Some(note) = ui.data(|data| data.get_temp::<Note>(note_id())) else {
        return;
    };
    let age = ui.input(|input| input.time) - note.at;
    let Some(alpha) = toast_alpha(age) else {
        return;
    };
    let ink =
        Color32::from_rgba_unmultiplied(theme::GAP.r(), theme::GAP.g(), theme::GAP.b(), alpha);
    let galley = ui.painter().layout_no_wrap(
        note.text.to_owned(),
        FontId::new(12.0, FontFamily::Proportional),
        ink,
    );
    let size = galley.size() + vec2(24.0, 12.0);
    let center = pos2(ruler.center().x, ruler.bottom() + 18.0 + size.y * 0.5);
    let rect = Rect::from_center_size(center, size);
    ui.painter()
        .rect_filled(rect, 99.0, Color32::from_white_alpha(alpha));
    ui.painter().galley(
        pos2(rect.left() + 12.0, rect.center().y - galley.size().y * 0.5),
        galley,
        ink,
    );
    let response = ui.interact(rect, ui.scope_id().with("toast"), egui::Sense::hover());
    crate::tip::alert(&response, &note.text);
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
    marks: &'a [i64],
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    let labels = ["−", words::of(ui, Phrase::Fit), "+"];
    let names = [Phrase::ZoomOut, Phrase::ZoomToFit, Phrase::ZoomIn];
    let salts = ["zoom-out", "zoom-fit", "zoom-in"];
    let mut chosen = None;
    for index in 0..rects.len() {
        let response = ui.interact(
            rects[index],
            ui.scope_id().with(salts[index]),
            Sense::click(),
        );
        crate::tip::show_enabled(&response, words::of(ui, names[index]), live);
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

const MARK_PX: f32 = 6.0;

fn shown_us(start: i64, offset: i64, duration: i64) -> Option<i64> {
    if offset < 0 || duration <= 0 || offset >= duration {
        return None;
    }
    Some(start.saturating_add(offset))
}

fn offset_in_span(at: i64, start: i64, end: i64, rate: FrameRate) -> Option<i64> {
    if end <= start {
        return None;
    }
    let snapped = ruler::seek(at.clamp(start, end - 1), rate, end - 1).max(start);
    (snapped < end).then_some(snapped - start)
}

fn clip_us(project: &Project, stamp: ClipStamp) -> Option<i64> {
    let range = project.timeline().segment(stamp.segment)?.timeline();
    shown_us(range.start().value, stamp.offset, range.duration().value)
}

fn segment_kind(project: &Project, id: SegmentId) -> Option<TrackKind> {
    project.timeline().tracks().iter().find_map(|track| {
        track
            .segments()
            .iter()
            .any(|segment| segment.id() == id)
            .then_some(track.kind())
    })
}

fn on_audio(project: &Project, tracks: Rect, point: egui::Pos2, view: View) -> bool {
    segment_at(project, tracks, point, view)
        .is_some_and(|id| segment_kind(project, id) == Some(TrackKind::Audio))
}

fn retain_stamps(project: &Project, stamps: &mut Stamps) {
    let duration = project.duration().value;
    if duration <= 0 {
        stamps.ruler.clear();
    } else {
        stamps.ruler.retain(|us| (0..=duration).contains(us));
    }
    stamps
        .clips
        .retain(|stamp| clip_us(project, *stamp).is_some());
    stamps.beats.retain(|stamp| {
        clip_us(project, *stamp).is_some()
            && segment_kind(project, stamp.segment) == Some(TrackKind::Audio)
    });
    let live = match stamps.selected {
        Some(PickedStamp::Ruler(index)) => index < stamps.ruler.len(),
        Some(PickedStamp::Clip(index)) => index < stamps.clips.len(),
        Some(PickedStamp::Beat(index)) => index < stamps.beats.len(),
        None => true,
    };
    if !live {
        stamps.selected = None;
    }
}

fn stamp_times(project: &Project, stamps: &Stamps) -> Vec<i64> {
    let mut times = stamps.ruler.clone();
    for stamp in stamps.clips.iter().chain(stamps.beats.iter()) {
        if let Some(us) = clip_us(project, *stamp) {
            times.push(us);
        }
    }
    times
}

fn keep_time(times: &mut Vec<i64>, time: i64) -> usize {
    if let Some(index) = times.iter().position(|us| *us == time) {
        return index;
    }
    times.push(time);
    times.len() - 1
}

fn keep_clip(stamps: &mut Vec<ClipStamp>, segment: SegmentId, offset: i64) -> usize {
    if let Some(index) = stamps
        .iter()
        .position(|stamp| stamp.segment == segment && stamp.offset == offset)
    {
        return index;
    }
    stamps.push(ClipStamp { segment, offset });
    stamps.len() - 1
}

fn near_us(view: View, left: f32, times: &[i64], x: f32) -> Option<usize> {
    if !(view.pixel_size.is_finite() && view.pixel_size > 0.0) {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for (index, time) in times.iter().enumerate() {
        let gap = (x - view.x(left, *time)).abs();
        if gap <= MARK_PX && best.is_none_or(|(_, best_gap)| gap < best_gap) {
            best = Some((index, gap));
        }
    }
    best.map(|(index, _)| index)
}

fn picked_stamp(
    project: &Project,
    stamps: &Stamps,
    tracks: Rect,
    point: egui::Pos2,
    view: View,
) -> Option<PickedStamp> {
    let id = segment_at(project, tracks, point, view)?;
    let left = lane_left(tracks);
    let mut best: Option<(PickedStamp, f32)> = None;
    let mut consider = |kind_index: PickedStamp, us: i64| {
        let gap = (point.x - view.x(left, us)).abs();
        if gap <= MARK_PX && best.is_none_or(|(_, best_gap)| gap < best_gap) {
            best = Some((kind_index, gap));
        }
    };
    for (index, stamp) in stamps.clips.iter().enumerate() {
        if stamp.segment == id
            && let Some(us) = clip_us(project, *stamp)
        {
            consider(PickedStamp::Clip(index), us);
        }
    }
    for (index, stamp) in stamps.beats.iter().enumerate() {
        if stamp.segment == id
            && let Some(us) = clip_us(project, *stamp)
        {
            consider(PickedStamp::Beat(index), us);
        }
    }
    best.map(|(picked, _)| picked)
}

fn drop_marker(
    project: &Project,
    tracks: Rect,
    ruler: Rect,
    point: egui::Pos2,
    playhead: i64,
    view: View,
    stamps: &mut Stamps,
) -> bool {
    let head = playhead_mark(ruler, project, playhead, view);
    let on_ruler = ruler.contains(point)
        && point.x >= lane_left(ruler)
        && !head.is_some_and(|mark| mark.contains(point));
    if on_ruler {
        let left = lane_left(ruler);
        if let Some(index) = near_us(view, left, &stamps.ruler, point.x) {
            stamps.selected = Some(PickedStamp::Ruler(index));
        } else if project.duration().value > 0 {
            let time = ruler::seek(
                view.us_at(left, point.x),
                project.settings().rate().rate(),
                project.duration().value,
            );
            let index = keep_time(&mut stamps.ruler, time);
            stamps.selected = Some(PickedStamp::Ruler(index));
        } else {
            return false;
        }
        return true;
    }
    let Some(id) = segment_at(project, tracks, point, view) else {
        return false;
    };
    let Some(index) = place_clip(project, tracks, id, point, view, &mut stamps.clips) else {
        return false;
    };
    stamps.selected = Some(PickedStamp::Clip(index));
    true
}

fn drop_beat(
    project: &Project,
    tracks: Rect,
    point: egui::Pos2,
    view: View,
    stamps: &mut Stamps,
) -> bool {
    let Some(id) = segment_at(project, tracks, point, view) else {
        return false;
    };
    if segment_kind(project, id) != Some(TrackKind::Audio) {
        return false;
    }
    let Some(index) = place_clip(project, tracks, id, point, view, &mut stamps.beats) else {
        return false;
    };
    stamps.selected = Some(PickedStamp::Beat(index));
    true
}

fn place_clip(
    project: &Project,
    tracks: Rect,
    id: SegmentId,
    point: egui::Pos2,
    view: View,
    stamps: &mut Vec<ClipStamp>,
) -> Option<usize> {
    let segment = project.timeline().segment(id)?;
    let range = segment.timeline();
    let own: Vec<i64> = stamps
        .iter()
        .filter(|stamp| stamp.segment == id)
        .filter_map(|stamp| clip_us(project, *stamp))
        .collect();
    if let Some(local) = near_us(view, lane_left(tracks), &own, point.x) {
        return stamps
            .iter()
            .enumerate()
            .filter(|(_, stamp)| stamp.segment == id)
            .nth(local)
            .map(|(index, _)| index);
    }
    let offset = offset_in_span(
        time_at(tracks, project, point.x, view),
        range.start().value,
        range.end().value,
        project.settings().rate().rate(),
    )?;
    Some(keep_clip(stamps, id, offset))
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
) -> bool {
    let [split_button, snap_button] = switch_rects(ruler);
    let armed = ruler_switch(
        ui,
        split_button,
        words::of(ui, Phrase::Split),
        "split-blade",
        split_live,
        split,
    );
    ruler_switch(
        ui,
        snap_button,
        words::of(ui, Phrase::Snap),
        "snap-switch",
        true,
        snap,
    );
    armed
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
) -> bool {
    if !live {
        *on = false;
    }
    let response = ui.interact(button, ui.scope_id().with(salt), Sense::click());
    let was = *on;
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
    crate::tip::name_switch(&response, label, *on, live);
    *on && !was
}

fn mark_rects(ruler: Rect) -> [Rect; 2] {
    let top = ruler.center().y - 13.0;
    let beat = Rect::from_min_size(pos2(ruler.right() - 32.0, top), vec2(26.0, 26.0));
    let marker = Rect::from_min_size(pos2(beat.left() - 30.0, top), vec2(26.0, 26.0));
    [marker, beat]
}

fn on_tools(ruler: Rect, point: egui::Pos2) -> bool {
    switch_rects(ruler)
        .into_iter()
        .chain(zoom_rects(ruler))
        .chain(project_rects(ruler))
        .chain(mark_rects(ruler))
        .any(|button| button.contains(point))
}

fn mark_switches(
    ui: &mut egui::Ui,
    ruler: Rect,
    live: bool,
    marker: &mut bool,
    beat: &mut bool,
    split: &mut bool,
) {
    let [marker_rect, beat_rect] = mark_rects(ruler);
    if mode_switch(
        ui,
        marker_rect,
        Icon::Marker,
        words::of(ui, Phrase::Marker),
        "marker-mode",
        live,
        marker,
    ) {
        *beat = false;
        *split = false;
    }
    if mode_switch(
        ui,
        beat_rect,
        Icon::Beat,
        words::of(ui, Phrase::Beat),
        "beat-mode",
        live,
        beat,
    ) {
        *marker = false;
        *split = false;
    }
}

fn mode_switch(
    ui: &mut egui::Ui,
    button: Rect,
    icon: Icon,
    tip: &str,
    salt: &str,
    live: bool,
    on: &mut bool,
) -> bool {
    if !live {
        *on = false;
    }
    let response = ui.interact(button, ui.scope_id().with(salt), Sense::click());
    let was = *on;
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
    icons::paint(ui.painter(), button.center(), icon, 16.0, color);
    crate::tip::show_switch(&response, tip, *on, live);
    *on && !was
}

fn project_rects(ruler: Rect) -> [Rect; 2] {
    let [marker, _] = mark_rects(ruler);
    let linkage = Rect::from_min_size(pos2(marker.left() - 30.0, marker.top()), vec2(26.0, 26.0));
    let magnet = Rect::from_min_size(pos2(linkage.left() - 30.0, marker.top()), vec2(26.0, 26.0));
    [magnet, linkage]
}

fn main_locked(project: &Project, lanes: &Lanes) -> bool {
    lane_row(lanes, project.timeline().main_track().id()).locked
}

/// A locked lane holds a clip that follows one main clip, so packing would move it.
fn locked_follower(project: &Project, lanes: &Lanes) -> bool {
    let mains: Vec<_> = project
        .timeline()
        .main_track()
        .segments()
        .iter()
        .map(|segment| (segment.id(), segment.timeline()))
        .collect();
    project.timeline().tracks().iter().any(|track| {
        !track.is_main()
            && lane_row(lanes, track.id()).locked
            && track
                .segments()
                .iter()
                .any(|segment| sole_owner(segment.timeline(), &mains).is_some())
    })
}

fn magnet_refusal(project: &Project, lanes: &Lanes, language: Language) -> Option<&'static str> {
    let turning_on = !project.settings().magnet();
    if main_locked(project, lanes) {
        return Some(words::say(
            language,
            if turning_on {
                Phrase::MagnetLockedOn
            } else {
                Phrase::MagnetLockedOff
            },
        ));
    }
    if turning_on && project.settings().linkage() && locked_follower(project, lanes) {
        return Some(words::say(language, Phrase::MagnetFollower));
    }
    None
}

struct Modes<'a> {
    split: &'a mut bool,
    snap: &'a mut bool,
    marker: &'a mut bool,
    beat: &'a mut bool,
}

fn apply_shortcut(
    key: crate::keys::Shortcut,
    live: bool,
    project: &Project,
    lanes: &Lanes,
    modes: &mut Modes<'_>,
    language: Language,
) -> (Option<Command>, Option<&'static str>, Option<ZoomStep>) {
    use crate::keys::Shortcut;
    match key {
        Shortcut::Undo | Shortcut::Redo => (None, None, None),
        Shortcut::Select => {
            *modes.split = false;
            *modes.marker = false;
            *modes.beat = false;
            (None, None, None)
        }
        Shortcut::Split if live => {
            *modes.split = !*modes.split;
            if *modes.split {
                *modes.marker = false;
                *modes.beat = false;
            }
            (None, None, None)
        }
        Shortcut::Snap => {
            *modes.snap = !*modes.snap;
            (None, None, None)
        }
        Shortcut::Magnet => {
            if let Some(command) = magnet_toggle(project, lanes) {
                (Some(command), None, None)
            } else {
                (None, magnet_refusal(project, lanes, language), None)
            }
        }
        Shortcut::Linkage => (Some(linkage_toggle(project)), None, None),
        Shortcut::Marker if live => {
            *modes.marker = !*modes.marker;
            if *modes.marker {
                *modes.beat = false;
                *modes.split = false;
            }
            (None, None, None)
        }
        Shortcut::ZoomIn if live => (None, None, Some(ZoomStep::In)),
        Shortcut::ZoomOut if live => (None, None, Some(ZoomStep::Out)),
        Shortcut::ZoomFit if live => (None, None, Some(ZoomStep::Fit)),
        Shortcut::Split
        | Shortcut::Marker
        | Shortcut::ZoomIn
        | Shortcut::ZoomOut
        | Shortcut::ZoomFit => (None, None, None),
    }
}

fn magnet_toggle(project: &Project, lanes: &Lanes) -> Option<Command> {
    if magnet_refusal(project, lanes, Language::English).is_some() {
        return None;
    }
    Some(Command::SetMagnet(!project.settings().magnet()))
}

fn linkage_toggle(project: &Project) -> Command {
    Command::SetLinkage(!project.settings().linkage())
}

fn project_switches(
    ui: &mut egui::Ui,
    ruler: Rect,
    project: &Project,
    lanes: &Lanes,
    commands: &mut Vec<Command>,
) -> Option<&'static str> {
    let [magnet_rect, linkage_rect] = project_rects(ruler);
    let cover = Rect::from_min_max(
        pos2(magnet_rect.left() - 4.0, ruler.top()),
        ruler.right_bottom(),
    );
    ui.painter().rect_filled(cover, 0.0, theme::PANEL);
    let magnet_on = project.settings().magnet();
    let magnet_tip = if main_locked(project, lanes) {
        words::of(ui, Phrase::MagnetLocked)
    } else if magnet_on {
        words::of(ui, Phrase::MagnetOn)
    } else {
        words::of(ui, Phrase::MagnetOff)
    };
    let mut refused = None;
    if setting_switch(
        ui,
        magnet_rect,
        Icon::Magnet,
        magnet_tip,
        "magnet-switch",
        magnet_on,
    ) {
        if let Some(command) = magnet_toggle(project, lanes) {
            commands.push(command);
        } else {
            refused = magnet_refusal(project, lanes, words::active(ui));
        }
    }
    let linkage_on = project.settings().linkage();
    let linkage_tip = if linkage_on {
        words::of(ui, Phrase::LinkageOn)
    } else {
        words::of(ui, Phrase::LinkageOff)
    };
    if setting_switch(
        ui,
        linkage_rect,
        Icon::Linkage,
        linkage_tip,
        "linkage-switch",
        linkage_on,
    ) {
        commands.push(linkage_toggle(project));
    }
    refused
}

fn setting_switch(
    ui: &mut egui::Ui,
    button: Rect,
    icon: Icon,
    tip: &str,
    salt: &str,
    on: bool,
) -> bool {
    let response = ui.interact(button, ui.scope_id().with(salt), Sense::click());
    if on {
        ui.painter().rect_filled(button, 6.0, theme::ACTIVE);
    }
    let color = if on { theme::EXPORT } else { theme::INK };
    icons::paint(ui.painter(), button.center(), icon, 16.0, color);
    crate::tip::show_switch(&response, tip, on, true);
    response.clicked()
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
    selected: &'a mut Vec<SegmentId>,
    drag: Option<&'a ClipDrag>,
    trim: Option<&'a TrimDrag>,
    pointer: Option<egui::Pos2>,
    snap: bool,
    playhead: i64,
    guides: &'a [i64],
    marks: &'a [i64],
    dim: bool,
    lanes: &'a Lanes,
    tips: &'a mut Vec<(egui::Response, String, bool)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClipAction {
    Copy,
    Cut,
    Delete,
    Freeze,
    Reverse,
    Mirror,
    Rotate,
    Crop,
    Transcribe,
    SplitScenes,
}

fn picture_actions(kind: TrackKind) -> &'static [ClipAction] {
    match kind {
        TrackKind::Video => &[
            ClipAction::Freeze,
            ClipAction::Reverse,
            ClipAction::Mirror,
            ClipAction::Rotate,
            ClipAction::Crop,
            ClipAction::Transcribe,
            ClipAction::SplitScenes,
        ],
        TrackKind::Audio => &[ClipAction::Reverse, ClipAction::Transcribe],
        TrackKind::Text | TrackKind::Sticker => &[ClipAction::Mirror, ClipAction::Rotate],
        TrackKind::Effect | TrackKind::Filter | TrackKind::Adjust | TrackKind::PluginEffect => &[],
    }
}

fn clip_menu(kind: TrackKind) -> Vec<ClipAction> {
    let mut rows = vec![ClipAction::Copy, ClipAction::Cut, ClipAction::Delete];
    rows.extend(picture_actions(kind));
    rows
}

fn action_runs(action: ClipAction) -> bool {
    matches!(action, ClipAction::Delete)
}

fn action_label(action: ClipAction, language: Language) -> &'static str {
    let phrase = match action {
        ClipAction::Copy => Phrase::Copy,
        ClipAction::Cut => Phrase::Cut,
        ClipAction::Delete => Phrase::Delete,
        ClipAction::Freeze => Phrase::Freeze,
        ClipAction::Reverse => Phrase::Reverse,
        ClipAction::Mirror => Phrase::Mirror,
        ClipAction::Rotate => Phrase::Rotate,
        ClipAction::Crop => Phrase::Crop,
        ClipAction::Transcribe => Phrase::Transcribe,
        ClipAction::SplitScenes => Phrase::SplitScenes,
    };
    words::say(language, phrase)
}

fn paint_clip_menu(
    ui: &mut egui::Ui,
    kind: TrackKind,
    refused: bool,
    id: SegmentId,
    remove: &mut Option<SegmentId>,
) {
    let rows = clip_menu(kind);
    let pictures = picture_actions(kind).len();
    let split_at = rows.len() - pictures;
    for (index, action) in rows.iter().copied().enumerate() {
        if index == split_at && pictures > 0 {
            ui.separator();
        }
        let enabled = action_runs(action) && !refused;
        let color = if enabled { theme::INK } else { theme::FAINT };
        if ui
            .add_enabled(
                enabled,
                egui::Button::new(
                    egui::RichText::new(action_label(action, words::active(ui))).color(color),
                ),
            )
            .clicked()
        {
            if enabled && action == ClipAction::Delete {
                *remove = Some(id);
            }
            ui.close();
        }
    }
}

fn paint_segments(
    ui: &mut egui::Ui,
    track: &Track,
    lane: Rect,
    main: bool,
    remove: &mut Option<SegmentId>,
    paint: &mut ClipPaint<'_>,
) {
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
        let resting = if dragging || trimming || paint.dim {
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
        let response = ui.interact(
            rect,
            ui.scope_id().with(("clip-menu", segment.id())),
            Sense::click(),
        );
        let activated = response.clicked() && !response.clicked_by(egui::PointerButton::Primary);
        let from_menu = response.secondary_clicked() && !chosen(paint.selected, segment.id());
        if activated || from_menu {
            choose(paint.selected, Some(segment.id()));
        }
        let refused = blocks_edit(paint.project, paint.lanes, segment.id());
        let kind = track.kind();
        let id = segment.id();
        response.context_menu(|ui| paint_clip_menu(ui, kind, refused, id, remove));
        let range = segment.timeline();
        let name = paint
            .project
            .asset(segment.asset_id())
            .map(|asset| file_name(asset.source()))
            .unwrap_or("");
        let text = crate::tip::span(
            name,
            range.start().value,
            range.end().value,
            range.start().timescale.get(),
        );
        let picked = chosen(paint.selected, segment.id());
        paint.tips.push((response, text, picked));
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

fn snap_targets(project: &Project, playhead: i64, guides: &[i64], marks: &[i64]) -> Vec<i64> {
    let mut targets = vec![0, project.duration().value.max(0), playhead.max(0)];
    targets.extend(guides);
    targets.extend(marks);
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
    let targets = snap_targets(project, align.playhead, align.guides, align.marks);
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
        &snap_targets(paint.project, paint.playhead, paint.guides, paint.marks),
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
        &snap_targets(paint.project, paint.playhead, paint.guides, paint.marks),
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
            &snap_targets(project, align.playhead, align.guides, align.marks),
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
            &snap_targets(project, align.playhead, align.guides, align.marks),
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

fn paint_stamps(
    ui: &mut egui::Ui,
    project: &Project,
    tracks: Rect,
    ruler: Rect,
    stamps: &Stamps,
    view: View,
) {
    let left = lane_left(ruler);
    let covered = project_rects(ruler)[0].left() - 4.0;
    for (index, us) in stamps.ruler.iter().enumerate() {
        if view.x(left, *us) >= covered {
            continue;
        }
        let color = if stamps.selected == Some(PickedStamp::Ruler(index)) {
            theme::INK
        } else {
            theme::STICKER
        };
        let at = pos2(view.x(left, *us), ruler.top() + 14.0);
        icons::paint(ui.painter(), at, Icon::Marker, 14.0, color);
        let hit = Rect::from_center_size(at, vec2(16.0, 16.0));
        let response = ui.interact(
            hit,
            ui.scope_id().with(("ruler-mark", index)),
            Sense::hover(),
        );
        crate::tip::show_image(
            &response,
            &crate::tip::mark(words::active(ui), false, *us, SEQUENCE_TIMESCALE.get()),
        );
    }
    if tracks.height() <= 0.0 {
        return;
    }
    let lanes = layout(project.timeline().tracks(), tracks);
    for (track, lane) in project.timeline().tracks().iter().zip(lanes) {
        let lane_rect = Rect::from_min_size(
            pos2(lane_left(tracks), lane.top),
            vec2((tracks.right() - lane_left(tracks)).max(0.0), lane.height),
        );
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
            paint_bound(
                ui,
                &stamps.clips,
                stamps.selected,
                segment.id(),
                range.start().value,
                range.duration().value,
                StampPlace {
                    rect,
                    left,
                    view,
                    icon: Icon::Marker,
                    y: rect.top() + 10.0,
                    idle: theme::STICKER,
                    chosen: theme::INK,
                    pick: PickedStamp::Clip,
                },
            );
            if track.kind() == TrackKind::Audio {
                paint_bound(
                    ui,
                    &stamps.beats,
                    stamps.selected,
                    segment.id(),
                    range.start().value,
                    range.duration().value,
                    StampPlace {
                        rect,
                        left,
                        view,
                        icon: Icon::Beat,
                        y: rect.bottom() - 10.0,
                        idle: theme::INK,
                        chosen: theme::EXPORT,
                        pick: PickedStamp::Beat,
                    },
                );
            }
        }
    }
}

fn paint_bound(
    ui: &mut egui::Ui,
    stamps: &[ClipStamp],
    selected: Option<PickedStamp>,
    segment: SegmentId,
    start: i64,
    duration: i64,
    place: StampPlace,
) {
    for (index, stamp) in stamps.iter().enumerate() {
        if stamp.segment != segment {
            continue;
        }
        let Some(us) = shown_us(start, stamp.offset, duration) else {
            continue;
        };
        let x = place.view.x(place.left, us);
        let x = if place.rect.width() <= 16.0 {
            place.rect.center().x
        } else {
            x.clamp(place.rect.left() + 8.0, place.rect.right() - 8.0)
        };
        let color = if selected == Some((place.pick)(index)) {
            place.chosen
        } else {
            place.idle
        };
        let at = pos2(x, place.y);
        icons::paint(ui.painter(), at, place.icon, 13.0, color);
        let beat = matches!(place.icon, Icon::Beat);
        let hit = Rect::from_center_size(at, vec2(16.0, 16.0));
        let response = ui.interact(
            hit,
            ui.scope_id().with((
                if beat { "beat-mark" } else { "clip-mark" },
                segment,
                stamp.offset,
            )),
            Sense::hover(),
        );
        crate::tip::show_image(
            &response,
            &crate::tip::mark(words::active(ui), beat, us, SEQUENCE_TIMESCALE.get()),
        );
    }
}

struct StampPlace {
    rect: Rect,
    left: f32,
    view: View,
    icon: Icon,
    y: f32,
    idle: Color32,
    chosen: Color32,
    pick: fn(usize) -> PickedStamp,
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
        crate::tip::image(&response, words::of(ui, Phrase::Playhead));
    }
    let x = x_of(*playhead);
    let mark = Rect::from_center_size(pos2(x, bottom - 6.0), vec2(11.0, 12.0));
    ui.painter().rect_filled(mark, 2.0, color);
    if tracks.bottom() >= bottom {
        ui.painter()
            .vline(x, bottom..=tracks.bottom(), Stroke::new(1.0, color));
    }
}

fn paint_lane_label(ui: &mut egui::Ui, head: Rect, label: &str) {
    let hide = flag_rects(head)[0];
    let max_width = hide.left() - 8.0 - (head.left() + 12.0);
    if max_width <= 8.0 {
        return;
    }
    let mut job = egui::text::LayoutJob::simple(
        label.to_owned(),
        FontId::new(12.0, FontFamily::Proportional),
        theme::MUTED,
        max_width,
    );
    job.wrap.max_rows = 1;
    job.wrap.break_anywhere = true;
    let galley = ui.painter().layout_job(job);
    ui.painter().galley(
        pos2(head.left() + 12.0, head.center().y - galley.size().y * 0.5),
        galley,
        theme::MUTED,
    );
}

fn flag_rects(head: Rect) -> [Rect; 3] {
    let y = head.center().y;
    let lock = Rect::from_center_size(pos2(head.right() - 16.0, y), vec2(22.0, 22.0));
    let mute = Rect::from_center_size(pos2(lock.center().x - 24.0, y), vec2(22.0, 22.0));
    let hide = Rect::from_center_size(pos2(mute.center().x - 24.0, y), vec2(22.0, 22.0));
    [hide, mute, lock]
}

fn paint_lane_flags(
    ui: &mut egui::Ui,
    head: Rect,
    track: TrackId,
    row: &mut LaneFlag,
) -> Option<&'static str> {
    let [hide, mute, lock] = flag_rects(head);
    let hide_icon = if row.hidden {
        Icon::ViewOff
    } else {
        Icon::View
    };
    let hide_tip = words::of(
        ui,
        if row.hidden {
            Phrase::Hidden
        } else {
            Phrase::Hide
        },
    );
    if lane_button(
        ui,
        hide,
        hide_icon,
        hide_tip,
        ui.scope_id().with(("hide", track)),
        row.hidden,
    ) {
        row.hidden = !row.hidden;
    }
    let mute_icon = if row.muted {
        Icon::VolumeOff
    } else {
        Icon::Volume
    };
    let mute_tip = words::of(
        ui,
        if row.muted {
            Phrase::Muted
        } else {
            Phrase::Mute
        },
    );
    if lane_button(
        ui,
        mute,
        mute_icon,
        mute_tip,
        ui.scope_id().with(("mute", track)),
        row.muted,
    ) {
        row.muted = !row.muted;
    }
    let lock_tip = words::of(
        ui,
        if row.locked {
            Phrase::Locked
        } else {
            Phrase::Lock
        },
    );
    let mut locked_note = None;
    if lane_button(
        ui,
        lock,
        Icon::Lock,
        lock_tip,
        ui.scope_id().with(("lock", track)),
        row.locked,
    ) {
        row.locked = !row.locked;
        locked_note = Some(words::of(
            ui,
            if row.locked {
                Phrase::TrackLocked
            } else {
                Phrase::TrackUnlocked
            },
        ));
    }
    locked_note
}

fn lane_button(
    ui: &mut egui::Ui,
    rect: Rect,
    icon: Icon,
    tip: &str,
    id: egui::Id,
    engaged: bool,
) -> bool {
    let response = ui.interact(rect, id, Sense::click());
    let color = if engaged { theme::INK } else { theme::FAINT };
    icons::paint(ui.painter(), rect.center(), icon, 14.0, color);
    crate::tip::show_switch(&response, tip, engaged, true);
    response.clicked()
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
            main: track.is_main(),
            empty: track.is_main() && track.segments().is_empty(),
            top: tops[index],
            height: height(index),
        })
        .collect()
}

fn lane_phrase(track: &Track) -> Phrase {
    if track.is_main() {
        Phrase::Main
    } else {
        kind_phrase(track.kind())
    }
}

/// The name of a lane kind. The main lane has its own short name.
pub(crate) fn kind_name(kind: TrackKind, language: Language) -> &'static str {
    words::say(language, kind_phrase(kind))
}

fn kind_phrase(kind: TrackKind) -> Phrase {
    match kind {
        TrackKind::Video => Phrase::Video,
        TrackKind::Audio => Phrase::Audio,
        TrackKind::Text => Phrase::Text,
        TrackKind::Sticker => Phrase::Sticker,
        TrackKind::Effect | TrackKind::PluginEffect => Phrase::Effect,
        TrackKind::Filter => Phrase::Filter,
        TrackKind::Adjust => Phrase::Adjust,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment, Speed, TimeRange,
        TrackKind,
    };
    use eframe::egui::{Rect, pos2, vec2};

    use super::{Language, clip_rect, lane_left, layout, segment_at};

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
            marks: &[],
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
        assert!(lanes[0].main);
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
        let tracks = project.timeline().tracks();
        let lanes = layout(
            tracks,
            Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 400.0)),
        );
        let lane = |kind: TrackKind| {
            tracks
                .iter()
                .zip(&lanes)
                .find(|(track, _)| !track.is_main() && track.kind() == kind)
                .map(|(_, lane)| lane)
                .expect("lane")
        };
        let main = tracks
            .iter()
            .zip(&lanes)
            .find(|(track, _)| track.is_main())
            .map(|(_, lane)| lane)
            .expect("main");
        assert!(lane(TrackKind::Audio).top > main.top);
        assert!(lane(TrackKind::Video).top < main.top);
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
            &mut selected,
            &super::Lanes::default(),
        ));
        assert!(selected.is_empty());
        assert!(cut.timeline().segment(id).is_none());
        assert!(cut.timeline().main_track().is_main());
        assert!(cut.timeline().main_track().segments().is_empty());
        assert!(!super::delete_selected(
            &mut history,
            &mut cut,
            &mut selected,
            &super::Lanes::default(),
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
        assert_eq!(
            super::snap_targets(&cut, 0, &[250_000], &[400_000]),
            vec![0, 250_000, 400_000]
        );
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

    #[test]
    fn a_mark_keeps_its_distance_from_the_clip_start() {
        assert_eq!(
            super::shown_us(1_000_000, 250_000, 500_000),
            Some(1_250_000)
        );
        assert_eq!(super::shown_us(0, 500_000, 500_000), None);
        let frame = super::offset_in_span(500_000, 0, 1_000_000, cutlass_project::FrameRate::Fps30)
            .expect("frame");
        assert!(frame < 1_000_000);
        assert_eq!(
            super::offset_in_span(frame, 0, 1_000_000, cutlass_project::FrameRate::Fps30),
            Some(frame)
        );
        assert!(super::offset_in_span(0, 5, 5, cutlass_project::FrameRate::Fps30).is_none());
        let mut times = Vec::new();
        assert_eq!(super::keep_time(&mut times, 10), 0);
        assert_eq!(super::keep_time(&mut times, 10), 0);
        assert_eq!(times, vec![10]);
    }

    #[test]
    fn a_beat_is_audio_and_a_repeated_offset_is_the_same_mark() {
        let mut history = History::new();
        let mut cut = project();
        let sound = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        let picture = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            false,
        )
        .expect("video");
        let (audio_id, video_id) = match history
            .apply(
                &mut cut,
                &[Command::AddAsset(sound), Command::AddAsset(picture)],
            )
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(audio), Outcome::Asset(video)] => (*audio, *video),
            other => panic!("expected two assets, got {other:?}"),
        };
        let place = |cut: &mut Project, history: &mut History, asset| {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            let segment = Segment::media(stored, source, 0, stored.default_channels(), Speed::ONE)
                .expect("segment");
            match history
                .apply(
                    cut,
                    &[Command::Place {
                        track: None,
                        segment,
                    }],
                )
                .expect("place")
                .as_slice()
            {
                [Outcome::Segment(id) | Outcome::Opened { segment: id, .. }] => *id,
                other => panic!("expected a segment, got {other:?}"),
            }
        };
        let audio = place(&mut cut, &mut history, audio_id);
        let video = place(&mut cut, &mut history, video_id);
        assert_eq!(
            super::segment_kind(&cut, audio),
            Some(cutlass_project::TrackKind::Audio)
        );
        assert_ne!(
            super::segment_kind(&cut, video),
            Some(cutlass_project::TrackKind::Audio)
        );
        let mut beats = Vec::new();
        assert_eq!(super::keep_clip(&mut beats, audio, 100), 0);
        assert_eq!(super::keep_clip(&mut beats, audio, 100), 0);
        assert_eq!(beats.len(), 1);
        assert_eq!(super::clip_us(&cut, beats[0]), Some(100));
    }

    #[test]
    fn a_locked_lane_refuses_the_edit() {
        let mut history = History::new();
        let mut cut = project();
        let sound = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        let picture = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            false,
        )
        .expect("video");
        let (audio_asset, video_asset) = match history
            .apply(
                &mut cut,
                &[Command::AddAsset(sound), Command::AddAsset(picture)],
            )
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(audio), Outcome::Asset(video)] => (*audio, *video),
            other => panic!("expected two assets, got {other:?}"),
        };
        let main = cut.timeline().main_track().id();
        let video = {
            let stored = cut.asset(video_asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            let segment =
                Segment::media(stored, source, 0, Channels::Picture, Speed::ONE).expect("segment");
            match history
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
            }
        };
        let (audio, audio_track) = {
            let stored = cut.asset(audio_asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            let segment =
                Segment::media(stored, source, 0, Channels::Sound, Speed::ONE).expect("segment");
            match history
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
                other => panic!("expected a new track, got {other:?}"),
            }
        };
        let locked_audio = super::Lanes {
            rows: vec![super::LaneFlag {
                track: audio_track,
                locked: true,
                hidden: false,
                muted: false,
            }],
        };
        assert!(super::blocks_edit(&cut, &locked_audio, video));
        let mut selected = vec![video];
        assert!(!super::delete_selected(
            &mut history,
            &mut cut,
            &mut selected,
            &locked_audio,
        ));
        assert!(cut.timeline().segment(video).is_some());
        assert!(cut.timeline().segment(audio).is_some());
        let onto_main = Command::Move {
            segment: audio,
            track: Some(main),
            start: 0,
        };
        assert!(super::blocks_command(&cut, &locked_audio, &onto_main));
        let locked_main_only = super::Lanes {
            rows: vec![super::LaneFlag {
                track: main,
                locked: true,
                hidden: false,
                muted: false,
            }],
        };
        assert!(super::blocks_command(&cut, &locked_main_only, &onto_main));
        selected = vec![video, audio];
        assert!(super::delete_selected(
            &mut history,
            &mut cut,
            &mut selected,
            &locked_main_only,
        ));
        assert!(cut.timeline().segment(video).is_some());
        assert!(cut.timeline().segment(audio).is_none());
        assert_eq!(selected, vec![video]);
    }

    #[test]
    fn a_locked_main_refuses_the_magnet_and_linkage_still_toggles() {
        let mut history = History::new();
        let mut cut = project();
        let open = super::Lanes::default();
        let main = cut.timeline().main_track().id();
        let locked_main = super::Lanes {
            rows: vec![super::LaneFlag {
                track: main,
                locked: true,
                hidden: false,
                muted: false,
            }],
        };
        assert!(super::magnet_toggle(&cut, &locked_main).is_none());
        assert_eq!(
            super::magnet_refusal(&cut, &locked_main, Language::English),
            Some("Main track is locked. Couldn't turn off main track magnet.")
        );
        assert_eq!(super::linkage_toggle(&cut), Command::SetLinkage(false));
        let off = super::magnet_toggle(&cut, &open).expect("off");
        assert_eq!(off, Command::SetMagnet(false));
        history.apply(&mut cut, &[off]).expect("magnet");
        assert!(!cut.settings().magnet());
        assert!(cut.settings().linkage());
        assert_eq!(
            super::magnet_refusal(&cut, &locked_main, Language::English),
            Some("Main track is locked. Couldn't turn on main track magnet.")
        );

        let picture = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            false,
        )
        .expect("video");
        let sound = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        let (video_asset, audio_asset) = match history
            .apply(
                &mut cut,
                &[Command::AddAsset(picture), Command::AddAsset(sound)],
            )
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(video), Outcome::Asset(audio)] => (*video, *audio),
            other => panic!("expected two assets, got {other:?}"),
        };
        let stored = cut.asset(video_asset).expect("asset");
        let duration = stored.duration().expect("duration");
        let source = TimeRange::new(
            Timestamp::new(0, duration.timescale),
            Timestamp::new(duration.value, duration.timescale),
        )
        .expect("source");
        let segment =
            Segment::media(stored, source, 0, Channels::Picture, Speed::ONE).expect("segment");
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place");
        let stored = cut.asset(audio_asset).expect("asset");
        let duration = stored.duration().expect("duration");
        let source = TimeRange::new(
            Timestamp::new(0, duration.timescale),
            Timestamp::new(duration.value, duration.timescale),
        )
        .expect("source");
        let segment =
            Segment::media(stored, source, 0, Channels::Sound, Speed::ONE).expect("segment");
        let audio_track = match history
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
            [Outcome::Opened { track, .. }] => *track,
            other => panic!("expected a new track, got {other:?}"),
        };
        let locked_audio = super::Lanes {
            rows: vec![super::LaneFlag {
                track: audio_track,
                locked: true,
                hidden: false,
                muted: false,
            }],
        };
        assert!(super::magnet_toggle(&cut, &locked_audio).is_none());
        assert_eq!(
            super::magnet_refusal(&cut, &locked_audio, Language::English),
            Some("A locked lane would move. Couldn't turn on main track magnet.")
        );
        let unlink = super::linkage_toggle(&cut);
        history.apply(&mut cut, &[unlink]).expect("linkage");
        assert_eq!(
            super::magnet_toggle(&cut, &locked_audio),
            Some(Command::SetMagnet(true))
        );
    }

    #[test]
    fn a_video_menu_lists_the_picture_actions_and_only_delete_runs() {
        use cutlass_project::TrackKind;
        let video = super::clip_menu(TrackKind::Video);
        assert_eq!(
            video,
            vec![
                super::ClipAction::Copy,
                super::ClipAction::Cut,
                super::ClipAction::Delete,
                super::ClipAction::Freeze,
                super::ClipAction::Reverse,
                super::ClipAction::Mirror,
                super::ClipAction::Rotate,
                super::ClipAction::Crop,
                super::ClipAction::Transcribe,
                super::ClipAction::SplitScenes,
            ]
        );
        let audio = super::clip_menu(TrackKind::Audio);
        assert_eq!(
            &audio[3..],
            &[super::ClipAction::Reverse, super::ClipAction::Transcribe]
        );
        let text = super::clip_menu(TrackKind::Text);
        assert_eq!(
            &text[3..],
            &[super::ClipAction::Mirror, super::ClipAction::Rotate]
        );
        assert_eq!(
            super::clip_menu(TrackKind::Sticker),
            super::clip_menu(TrackKind::Text)
        );
        assert_eq!(
            super::clip_menu(TrackKind::Effect),
            vec![
                super::ClipAction::Copy,
                super::ClipAction::Cut,
                super::ClipAction::Delete,
            ]
        );
        assert!(super::action_runs(super::ClipAction::Delete));
        assert!(!super::action_runs(super::ClipAction::Copy));
        assert!(!super::action_runs(super::ClipAction::Freeze));
    }

    #[test]
    fn a_toast_holds_then_fades_and_names_the_refusal() {
        assert_eq!(super::toast_alpha(0.0), Some(255));
        assert_eq!(super::toast_alpha(1.0), Some(255));
        assert_eq!(super::toast_alpha(1.2), Some(255));
        let fading = super::toast_alpha(1.5).expect("fading");
        let later = super::toast_alpha(1.7).expect("later");
        assert!(later < fading && fading < 255, "{later} {fading}");
        assert_eq!(super::toast_alpha(1.8), None);
        assert_eq!(super::toast_alpha(-0.1), None);
        assert_eq!(
            super::apply_note(&cutlass_project::Error::Overlap, Language::English),
            "That spot is taken."
        );
        assert_eq!(
            super::apply_note(&cutlass_project::Error::WrongTrack, Language::English),
            "That lane can't hold this clip."
        );
        assert_eq!(
            super::apply_note(&cutlass_project::Error::SplitOutside, Language::English),
            "That cut is too close to the edge."
        );
        assert_eq!(
            super::apply_note(
                &cutlass_project::Error::SourceOutsideAsset,
                Language::English
            ),
            "That trim leaves the file."
        );
        assert_eq!(
            super::apply_note(&cutlass_project::Error::RoundsToZero, Language::English),
            "That trim leaves the file."
        );
        assert_eq!(
            super::apply_note(&cutlass_project::Error::NothingToUndo, Language::English),
            "That edit did not land."
        );
    }

    #[test]
    fn a_tool_key_does_what_its_button_does() {
        use crate::keys::Shortcut;
        let cut = project();
        let open = super::Lanes::default();
        let mut split = false;
        let mut snap = true;
        let mut marker = true;
        let mut beat = true;
        let press = |key,
                     live,
                     lanes: &super::Lanes,
                     split: &mut bool,
                     snap: &mut bool,
                     marker: &mut bool,
                     beat: &mut bool| {
            let mut modes = super::Modes {
                split,
                snap,
                marker,
                beat,
            };
            super::apply_shortcut(key, live, &cut, lanes, &mut modes, Language::English)
        };
        let (command, note, zoom) = press(
            Shortcut::Select,
            true,
            &open,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert!(command.is_none() && note.is_none() && zoom.is_none());
        assert!(!split && !marker && !beat && snap);

        let _ = press(
            Shortcut::Split,
            true,
            &open,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert!(split && !marker && !beat);
        let _ = press(
            Shortcut::Split,
            false,
            &open,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert!(split);

        let _ = press(
            Shortcut::Snap,
            true,
            &open,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert!(!snap);
        let _ = press(
            Shortcut::Marker,
            true,
            &open,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert!(marker && !split && !beat);

        let (command, note, _) = press(
            Shortcut::Linkage,
            true,
            &open,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert_eq!(command, Some(Command::SetLinkage(false)));
        assert!(note.is_none());
        assert_eq!(
            press(
                Shortcut::ZoomIn,
                true,
                &open,
                &mut split,
                &mut snap,
                &mut marker,
                &mut beat
            )
            .2,
            Some(super::ZoomStep::In)
        );
        assert!(
            press(
                Shortcut::ZoomFit,
                false,
                &open,
                &mut split,
                &mut snap,
                &mut marker,
                &mut beat
            )
            .2
            .is_none()
        );

        let main = cut.timeline().main_track().id();
        let locked = super::Lanes {
            rows: vec![super::LaneFlag {
                track: main,
                locked: true,
                hidden: false,
                muted: false,
            }],
        };
        let (command, note, _) = press(
            Shortcut::Magnet,
            true,
            &locked,
            &mut split,
            &mut snap,
            &mut marker,
            &mut beat,
        );
        assert!(command.is_none());
        assert_eq!(
            note,
            Some("Main track is locked. Couldn't turn off main track magnet.")
        );
    }
}
