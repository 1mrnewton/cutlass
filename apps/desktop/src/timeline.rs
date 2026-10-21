//! The project's tracks, drawn bottom to top, with the ruler above them.

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, Sense, Stroke, pos2, vec2};

use cutlass_project::{Command, History, Project, SegmentId, Track, TrackId, TrackKind};

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
    selected: &mut Option<SegmentId>,
) -> bool {
    let Some(id) = *selected else {
        return false;
    };
    if history
        .apply(project, &[Command::RemoveSegment(id)])
        .is_err()
    {
        return false;
    }
    *selected = None;
    true
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
    selected: &mut Option<SegmentId>,
    drag: &mut Option<ClipDrag>,
    trim: &mut Option<TrimDrag>,
    split: &mut bool,
) -> Vec<Command> {
    let full = ui.max_rect();
    let ruler_rect = Rect::from_min_size(full.min, vec2(full.width(), RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler_rect.left_bottom(), full.max);
    if selected.is_some_and(|id| project.timeline().segment(id).is_none()) {
        *selected = None;
    }
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
    let pointer = ui.input(|input| input.pointer.interact_pos());
    if tracks.height() > 0.0 {
        let pressed = ui.input(|input| input.pointer.primary_pressed());
        let released = ui.input(|input| input.pointer.primary_released());
        if pressed && let Some(point) = pointer.filter(|point| tracks.contains(*point)) {
            if *split {
                *drag = None;
                *trim = None;
                *selected = segment_at(project, tracks, point);
            } else if let Some(active) = trim_from(project, tracks, point) {
                *selected = Some(active.segment);
                *drag = None;
                *trim = Some(active);
            } else {
                *trim = None;
                *selected = segment_at(project, tracks, point);
                *drag = selected.and_then(|id| drag_from(project, id, point));
            }
        }
        if released {
            if *split {
                drag.take();
                trim.take();
                if ui.input(|input| input.pointer.primary_clicked())
                    && let Some(point) = pointer.filter(|point| tracks.contains(*point))
                    && let Some(command) = split_command(project, tracks, point)
                {
                    commands.push(command);
                }
            } else if let Some(active) = trim.take() {
                commands = trim_commands(project, tracks, &active, pointer);
            } else if let Some(active) = drag.take()
                && let Some(command) = move_command(project, tracks, &active, pointer)
            {
                commands.push(command);
            }
        }
    }
    paint_ruler(ui, ruler_rect, project);
    split_blade(ui, ruler_rect, project.duration().value > 0, split);
    if *split && pointer.is_some_and(|point| tracks.contains(point)) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    if tracks.height() > 0.0 {
        let pixel_size = ruler::opening_pixel_size(
            (tracks.right() - lane_left(tracks)).max(0.0),
            project.duration().value,
        );
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
                    pixel_size,
                    selected: *selected,
                    drag: drag.as_ref(),
                    trim: trim.as_ref(),
                    pointer,
                },
            );
        }
    }
    paint_playhead(ui, ruler_rect, tracks, project, playhead);
    commands
}

fn lane_left(area: Rect) -> f32 {
    area.left() + HEAD + COVER
}

fn split_blade(ui: &mut egui::Ui, ruler: Rect, live: bool, on: &mut bool) {
    if !live {
        *on = false;
    }
    let button = Rect::from_center_size(
        pos2(ruler.left() + HEAD / 2.0, ruler.center().y),
        vec2(72.0, 26.0),
    );
    let response = ui.interact(button, ui.scope_id().with("split-blade"), Sense::click());
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
        "Split",
        FontId::new(12.0, FontFamily::Proportional),
        color,
    );
}

fn split_command(project: &Project, tracks: Rect, point: egui::Pos2) -> Option<Command> {
    let id = segment_at(project, tracks, point)?;
    let range = project.timeline().segment(id)?.timeline();
    let width = (tracks.right() - lane_left(tracks)).max(0.0);
    let pixel_size = ruler::opening_pixel_size(width, project.duration().value);
    if !(pixel_size.is_finite() && pixel_size > 0.0) {
        return None;
    }
    let raw = f64::from(point.x - lane_left(tracks)) / pixel_size;
    if !raw.is_finite() {
        return None;
    }
    let us = raw.round().clamp(0.0, i64::MAX as f64) as i64;
    let at = ruler::seek(
        us,
        project.settings().rate().rate(),
        project.duration().value,
    );
    if at <= range.start().value || at >= range.end().value {
        return None;
    }
    Some(Command::Split { segment: id, at })
}

struct ClipPaint<'a> {
    project: &'a Project,
    pixel_size: f64,
    selected: Option<SegmentId>,
    drag: Option<&'a ClipDrag>,
    trim: Option<&'a TrimDrag>,
    pointer: Option<egui::Pos2>,
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
        if !dragging && !trimming && paint.selected == Some(segment.id()) {
            stroke_clip(ui, rect);
        }
        if dragging {
            let offset = paint
                .drag
                .zip(paint.pointer)
                .map(|(drag, point)| point - drag.origin)
                .unwrap_or_default();
            let moved = rect.translate(offset);
            ui.painter().rect_filled(moved, 7.0, color);
            stroke_clip(ui, moved);
            paint_name(ui, paint.project, segment, moved);
        } else if trimming {
            let delta = paint
                .trim
                .zip(paint.pointer)
                .map(|(trim, point)| point.x - trim.origin_x)
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
) -> Option<Command> {
    let point = pointer?;
    let width = (tracks.right() - lane_left(tracks)).max(0.0);
    let pixel_size = ruler::opening_pixel_size(width, project.duration().value);
    let start = moved_start(drag.start_us, point.x - drag.origin.x, pixel_size);
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

fn trim_from(project: &Project, tracks: Rect, point: egui::Pos2) -> Option<TrimDrag> {
    if tracks.height() <= 0.0 {
        return None;
    }
    let pixel_size = ruler::opening_pixel_size(
        (tracks.right() - lane_left(tracks)).max(0.0),
        project.duration().value,
    );
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
                pixel_size,
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
    tracks: Rect,
    trim: &TrimDrag,
    pointer: Option<egui::Pos2>,
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
    let width = (tracks.right() - lane_left(tracks)).max(0.0);
    let pixel_size = ruler::opening_pixel_size(width, project.duration().value);
    let edge = match trim.edge {
        Edge::Start => cutlass_project::TrimEdge::Start,
        Edge::End => cutlass_project::TrimEdge::End,
    };
    cutlass_project::commands_for_trim(
        segment,
        asset,
        edge,
        delta_us(point.x - trim.origin_x, pixel_size),
    )
    .unwrap_or_default()
}

fn segment_at(project: &Project, tracks: Rect, point: egui::Pos2) -> Option<SegmentId> {
    if tracks.height() <= 0.0 {
        return None;
    }
    let pixel_size = ruler::opening_pixel_size(
        (tracks.right() - lane_left(tracks)).max(0.0),
        project.duration().value,
    );
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
                pixel_size,
            );
            if rect.contains(point) {
                return Some(segment.id());
            }
        }
    }
    None
}

fn clip_rect(lane: Rect, main: bool, start_us: i64, duration_us: i64, pixel_size: f64) -> Rect {
    let (inset, height) = if main { (8.0, 48.0) } else { (6.0, 28.0) };
    let width = (duration_us.max(0) as f64 * pixel_size).max(8.0) as f32;
    let x = lane.left() + (start_us.max(0) as f64 * pixel_size) as f32;
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

fn paint_ruler(ui: &mut egui::Ui, ruler_rect: Rect, project: &Project) {
    let left = lane_left(ruler_rect);
    let width = (ruler_rect.right() - left).max(0.0);
    let pixel_size = ruler::opening_pixel_size(width, project.duration().value);
    let bottom = ruler_rect.bottom();
    for mark in ruler::marks(project.settings().rate(), pixel_size, width) {
        let x = left + (mark.us as f64 * pixel_size) as f32;
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
) {
    let left = lane_left(ruler_rect);
    let width = (ruler_rect.right() - left).max(0.0);
    let duration = project.duration().value;
    let live = duration > 0;
    let rate = project.settings().rate().rate();
    let pixel_size = ruler::opening_pixel_size(width, duration);
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
        if pixel_size > 0.0 {
            let drawn = left + (us as f64 * pixel_size) as f32;
            drawn.clamp(left, left + width)
        } else {
            left
        }
    };
    if live && pixel_size > 0.0 {
        let mark = Rect::from_center_size(pos2(x_of(*playhead), bottom - 6.0), vec2(11.0, 12.0));
        let response = ui.interact(mark, ui.scope_id().with("playhead"), Sense::drag());
        if response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let raw = f64::from(pointer.x - left) / pixel_size;
            let us = if raw.is_finite() {
                raw.round().clamp(0.0, i64::MAX as f64) as i64
            } else {
                0
            };
            *playhead = ruler::seek(us, rate, duration);
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
        let rect = super::clip_rect(lane, true, 0, 1_000_000, 3e-4);
        assert!((rect.left() - 202.0).abs() < 0.01);
        assert!((rect.width() - 300.0).abs() < 0.01);
        assert!((rect.top() - 108.0).abs() < f32::EPSILON);
        assert!((rect.height() - 48.0).abs() < f32::EPSILON);
        let short = super::clip_rect(lane, false, 0, 1, 3e-4);
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
        );
        assert_eq!(segment_at(&cut, tracks, rect.center()), Some(id));
        assert_eq!(segment_at(&cut, tracks, pos2(12.0, rect.center().y)), None);
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
        let mut selected = Some(id);
        assert!(super::delete_selected(
            &mut history,
            &mut cut,
            &mut selected
        ));
        assert!(selected.is_none());
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
        assert!(super::move_command(&cut, tracks, &drag, Some(origin)).is_none());
        let pixel_size = crate::ruler::opening_pixel_size(400.0, cut.duration().value);
        let landed = super::moved_start(0, 150.0, pixel_size);
        let command =
            super::move_command(&cut, tracks, &drag, Some(pos2(origin.x + 150.0, origin.y)))
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
        );
        let edge = pos2(rect.right() - 2.0, rect.center().y);
        let trim = super::trim_from(&cut, tracks, edge).expect("edge");
        assert!(super::trim_from(&cut, tracks, rect.center()).is_none());
        let before = range.duration().value;
        let commands = super::trim_commands(&cut, tracks, &trim, Some(pos2(edge.x - 80.0, edge.y)));
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
        );
        assert!(super::split_command(&cut, tracks, rect.left_center()).is_none());
        let command = super::split_command(&cut, tracks, rect.center()).expect("split");
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
}
