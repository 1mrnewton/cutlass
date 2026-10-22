//! The preview frame and the transport. The frame shows the project canvas.
//! One selected layer can be moved, scaled, and rotated. On Apple the top
//! picture is drawn in that box. `docs/preview.md`.

use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Pos2, Rect, Sense, Stroke, Vec2, pos2, vec2,
};

use cutlass_project::{Channels, Project, SegmentId, TrackId};

use crate::icons::{self, Icon};
use crate::ruler;
use crate::theme;
use crate::timeline::file_name;

const TRANSPORT: f32 = 52.0;

pub fn show(
    ui: &mut egui::Ui,
    project: &Project,
    playhead_us: i64,
    playing: bool,
    selected: &mut Vec<SegmentId>,
    picture: Option<egui::TextureId>,
) -> Vec<cutlass_project::Command> {
    crate::tip::region(
        ui,
        crate::words::of(ui, crate::words::Phrase::RegionPreview),
    );
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let transport = Rect::from_min_max(pos2(full.left(), full.bottom() - TRANSPORT), full.max);
    let frame = Rect::from_min_max(
        full.min + vec2(16.0, 16.0),
        pos2(full.right() - 16.0, transport.top() - 8.0),
    );
    let mut commands = Vec::new();
    if frame.height() > 0.0 && frame.width() > 0.0 {
        let hidden = crate::timeline::hidden_tracks(project);
        let plate = fitted_canvas(
            frame,
            project.settings().width(),
            project.settings().height(),
        );
        let canvas = ui.interact(
            frame,
            ui.scope_id().with("preview-canvas"),
            Sense::click_and_drag(),
        );
        let drag = plate.map(|plate| {
            gesture(
                ui,
                project,
                &Canvas {
                    playhead_us,
                    frame,
                    plate,
                },
                &canvas,
                selected,
                &mut commands,
            )
        });
        let drag = drag.flatten();
        if let Some(plate) = plate {
            let (red, green, blue) = project.settings().background().channels();
            ui.painter()
                .rect_filled(plate, 0.0, Color32::from_rgb(red, green, blue));
            if let Some(id) = picture
                && let Some(layer) = shown_box(project, playhead_us, &hidden, plate, drag.as_ref())
            {
                paint_picture(ui, id, &layer);
            }
            paint_selected(
                ui,
                project,
                playhead_us,
                &hidden,
                plate,
                selected,
                drag.as_ref(),
            );
            paint_guides(ui, plate, drag.as_ref());
            paint_handles(
                ui,
                project,
                &Canvas {
                    playhead_us,
                    frame,
                    plate,
                },
                &hidden,
                selected,
                drag.as_ref(),
            );
        }
        let rate = project.settings().rate();
        let time = ruler::timecode(playhead_us, rate);
        let label = match shot(project, playhead_us, &hidden) {
            Some(name) => format!("{name} · {time}"),
            None => format!(
                "{} · {time}",
                crate::words::of(ui, crate::words::Phrase::Empty)
            ),
        };
        ui.painter().text(
            frame.left_top() + vec2(12.0, 12.0),
            Align2::LEFT_TOP,
            &label,
            FontId::new(11.0, FontFamily::Monospace),
            theme::INK,
        );
        crate::tip::read(
            ui,
            Rect::from_min_size(frame.left_top() + vec2(12.0, 8.0), vec2(280.0, 18.0)),
            ui.scope_id().with("preview-label"),
            &label,
        );
        let top = plate.and_then(|plate| top_layer(project, playhead_us, &hidden, plate));
        canvas.widget_info(|| {
            egui::WidgetInfo::selected(
                egui::accesskit::Role::Button,
                true,
                top.is_some_and(|id| selected.contains(&id)),
                &label,
            )
        });
    }
    paint_transport(ui, transport, project, playhead_us, playing);
    commands
}

pub(crate) fn take_play(ui: &egui::Ui) -> bool {
    ui.data_mut(|data| data.remove_temp(play_id()))
        .unwrap_or(false)
}

fn play_id() -> egui::Id {
    egui::Id::unique("preview-play")
}

fn paint_transport(
    ui: &mut egui::Ui,
    rect: Rect,
    project: &Project,
    playhead_us: i64,
    playing: bool,
) {
    let rate = project.settings().rate();
    let live = project.duration().value > 0;
    let ink = if live { theme::INK } else { theme::MUTED };
    let time = format!(
        "{}  /  {}",
        ruler::timecode(playhead_us, rate),
        ruler::timecode(project.duration().value, rate)
    );
    ui.painter().text(
        pos2(rect.left() + 16.0, rect.center().y),
        Align2::LEFT_CENTER,
        &time,
        FontId::new(12.0, FontFamily::Monospace),
        ink,
    );
    crate::tip::read(
        ui,
        Rect::from_min_max(
            pos2(rect.left() + 12.0, rect.top()),
            pos2(rect.center().x - 28.0, rect.bottom()),
        ),
        ui.scope_id().with("transport-time"),
        &time,
    );

    let dim = if live { 1.0 } else { 0.35 };
    let play = Rect::from_center_size(rect.center(), Vec2::splat(36.0));
    let play_response = ui.interact(
        play,
        ui.scope_id().with("play"),
        if live { Sense::click() } else { Sense::hover() },
    );
    if play_response.clicked() {
        ui.data_mut(|data| data.insert_temp(play_id(), true));
    }
    let (disk, mark) = theme::face(
        theme::INK,
        Color32::from_rgb(0x12, 0x13, 0x0f),
        live && (playing || theme::hot(&play_response)),
    );
    ui.painter()
        .circle_filled(play.center(), 18.0, disk.gamma_multiply(dim));
    icons::paint(
        ui.painter(),
        play.center(),
        if playing { Icon::Pause } else { Icon::Play },
        16.0,
        mark.gamma_multiply(dim),
    );
    crate::tip::show_enabled(
        &play_response,
        crate::words::of(
            ui,
            if playing {
                crate::words::Phrase::Pause
            } else {
                crate::words::Phrase::Play
            },
        ),
        live,
    );

    let still = Rect::from_center_size(
        pos2(rect.right() - 28.0, rect.center().y),
        Vec2::splat(32.0),
    );
    let still_response = ui.interact(
        still,
        ui.scope_id().with("still"),
        if live { Sense::click() } else { Sense::hover() },
    );
    if still_response.clicked() {
        crate::timeline::show_note(ui, still_toast(crate::words::active(ui), playhead_us));
    }
    let (disk, mark) = theme::face(
        theme::BAR,
        theme::MUTED,
        live && theme::hot(&still_response),
    );
    ui.painter()
        .circle_filled(still.center(), 16.0, disk.gamma_multiply(dim));
    icons::paint(
        ui.painter(),
        still.center(),
        Icon::Camera,
        16.0,
        mark.gamma_multiply(dim),
    );
    crate::tip::show_enabled(
        &still_response,
        crate::words::of(ui, crate::words::Phrase::Still),
        live,
    );
}

/// The canvas, fitted inside `frame` and centered. The aspect is `width`/`height`.
fn fitted_canvas(frame: Rect, width: u32, height: u32) -> Option<Rect> {
    if width == 0 || height == 0 || frame.width() <= 0.0 || frame.height() <= 0.0 {
        return None;
    }
    let picture = width as f32 / height as f32;
    let (plate_width, plate_height) = if frame.width() / frame.height() > picture {
        let plate_height = frame.height();
        (plate_height * picture, plate_height)
    } else {
        let plate_width = frame.width();
        (plate_width, plate_width / picture)
    };
    Some(Rect::from_center_size(
        frame.center(),
        vec2(plate_width, plate_height),
    ))
}

/// A picture on the canvas. `rotation` is degrees, clockwise, because y points down.
#[derive(Clone, Copy)]
struct LayerBox {
    center: Pos2,
    size: Vec2,
    rotation: f32,
}

const GUIDE_PX: f32 = 6.0;
const HANDLE_PX: f32 = 6.0;
const ROTATE_GAP: f32 = 20.0;
const MIN_SCALE: f32 = 0.01;

#[derive(Clone, Copy, PartialEq)]
enum Grab {
    Move,
    Scale { x: i8, y: i8 },
    Rotate,
}

/// The placement under the pointer, written when the pointer goes up.
#[derive(Clone, Copy)]
struct Drag {
    segment: SegmentId,
    grab: Grab,
    position: (f64, f64),
    scale: (f64, f64),
    rotation: f64,
    opacity: f64,
    flip_horizontal: bool,
    flip_vertical: bool,
    origin: Pos2,
    start_position: (f64, f64),
    start_rotation: f64,
    base: Vec2,
    start_box: LayerBox,
    guide_x: bool,
    guide_y: bool,
}

fn drag_id() -> egui::Id {
    egui::Id::unique("preview-drag")
}

/// The fitted picture, scaled, then moved from the canvas center. `docs/project.md`.
fn layer_box(
    plate: Rect,
    canvas: (u32, u32),
    media: (u32, u32),
    position: (f64, f64),
    scale: (f64, f64),
    rotation: f64,
) -> Option<LayerBox> {
    let canvas_w = canvas.0 as f32;
    let canvas_h = canvas.1 as f32;
    let media_w = media.0 as f32;
    let media_h = media.1 as f32;
    if canvas_w <= 0.0 || canvas_h <= 0.0 || media_w <= 0.0 || media_h <= 0.0 {
        return None;
    }
    let (scale_x, scale_y) = (scale.0 as f32, scale.1 as f32);
    let (position_x, position_y) = (position.0 as f32, position.1 as f32);
    let rotation = rotation as f32;
    if !scale_x.is_finite()
        || !scale_y.is_finite()
        || scale_x <= 0.0
        || scale_y <= 0.0
        || !position_x.is_finite()
        || !position_y.is_finite()
        || !rotation.is_finite()
    {
        return None;
    }
    let media_shape = media_w / media_h;
    let (fitted_w, fitted_h) = if canvas_w / canvas_h > media_shape {
        let fitted_h = canvas_h;
        (fitted_h * media_shape, fitted_h)
    } else {
        let fitted_w = canvas_w;
        (fitted_w, fitted_w / media_shape)
    };
    Some(LayerBox {
        center: plate.center() + vec2(position_x * plate.width(), position_y * plate.height()),
        size: vec2(
            fitted_w * scale_x * plate.width() / canvas_w,
            fitted_h * scale_y * plate.height() / canvas_h,
        ),
        rotation,
    })
}

fn picture_corners(layer: &LayerBox) -> [Pos2; 4] {
    let (sin, cos) = (-layer.rotation).to_radians().sin_cos();
    let half = layer.size / 2.0;
    [
        vec2(-half.x, -half.y),
        vec2(half.x, -half.y),
        vec2(half.x, half.y),
        vec2(-half.x, half.y),
    ]
    .map(|local| {
        let dx = local.x * cos + local.y * sin;
        let dy = -local.x * sin + local.y * cos;
        layer.center + vec2(dx, dy)
    })
}

fn paint_picture(ui: &egui::Ui, id: egui::TextureId, layer: &LayerBox) {
    let mut mesh = egui::Mesh::with_texture(id);
    let uvs = [
        pos2(0.0, 0.0),
        pos2(1.0, 0.0),
        pos2(1.0, 1.0),
        pos2(0.0, 1.0),
    ];
    for (point, uv) in picture_corners(layer).into_iter().zip(uvs) {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: point,
            uv,
            color: Color32::WHITE,
        });
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    ui.painter().add(egui::Shape::mesh(mesh));
}

fn layer_contains(layer: &LayerBox, point: Pos2) -> bool {
    let (sin, cos) = (-layer.rotation).to_radians().sin_cos();
    let dx = point.x - layer.center.x;
    let dy = point.y - layer.center.y;
    let local_x = dx * cos - dy * sin;
    let local_y = dx * sin + dy * cos;
    local_x.abs() <= layer.size.x / 2.0 && local_y.abs() <= layer.size.y / 2.0
}

/// Source time at `playhead_us`. The start of the clip is the source start.
/// A later playhead follows the speed. A curve replaces the rational.
pub(crate) fn media_time(
    segment: &cutlass_project::Segment,
    playhead_us: i64,
) -> Option<cutlass_core::Timestamp> {
    let source = segment.source()?;
    let timeline = segment.timeline();
    if playhead_us < timeline.start().value || playhead_us >= timeline.end().value {
        return None;
    }
    let offset = playhead_us.checked_sub(timeline.start().value)?;
    let ticks = segment.source_offset(offset).ok()?;
    let mut value = source.start().value.checked_add(ticks)?;
    let end = source.end().value;
    if value >= end {
        if end <= source.start().value {
            return None;
        }
        value = end - 1;
    }
    Some(cutlass_core::Timestamp::new(
        value,
        source.start().timescale,
    ))
}

fn covers(segment: &cutlass_project::Segment, us: i64) -> bool {
    let range = segment.timeline();
    segment.visible() && range.start().value <= us && us < range.end().value
}

fn picture_box(
    project: &Project,
    segment: &cutlass_project::Segment,
    plate: Rect,
    drag: Option<&Drag>,
) -> Option<LayerBox> {
    if segment.channels() == Channels::Sound {
        return None;
    }
    let media = project.asset(segment.asset_id()?)?.visual_size()?;
    let transform = segment.transform();
    let (position, scale, rotation) = drag
        .filter(|drag| drag.segment == segment.id())
        .map(|drag| (drag.position, drag.scale, drag.rotation))
        .unwrap_or_else(|| {
            (
                transform.position(),
                transform.scale(),
                transform.rotation(),
            )
        });
    layer_box(
        plate,
        (project.settings().width(), project.settings().height()),
        media,
        position,
        scale,
        rotation,
    )
}

/// The topmost picture under the playhead. Sound-only, hidden, and invisible
/// segments are skipped.
pub(crate) fn picture_segment<'a>(
    project: &'a Project,
    playhead_us: i64,
    hidden: &[TrackId],
) -> Option<&'a cutlass_project::Segment> {
    project.timeline().tracks().iter().rev().find_map(|track| {
        if hidden.contains(&track.id()) {
            return None;
        }
        track.segments().iter().find(|segment| {
            covers(segment, playhead_us)
                && segment.channels() != Channels::Sound
                && segment
                    .asset_id()
                    .and_then(|id| project.asset(id))
                    .and_then(|asset| asset.visual_size())
                    .is_some()
        })
    })
}

fn shown_box(
    project: &Project,
    playhead_us: i64,
    hidden: &[TrackId],
    plate: Rect,
    drag: Option<&Drag>,
) -> Option<LayerBox> {
    let segment = picture_segment(project, playhead_us, hidden)?;
    picture_box(project, segment, plate, drag)
}

/// The topmost picture under the playhead, whether or not it covers `point`.
fn top_layer(
    project: &Project,
    playhead_us: i64,
    hidden: &[TrackId],
    plate: Rect,
) -> Option<SegmentId> {
    layers(project, playhead_us, hidden, plate)
        .next()
        .map(|(id, _)| id)
}

fn layer_at(
    project: &Project,
    playhead_us: i64,
    hidden: &[TrackId],
    plate: Rect,
    point: Pos2,
) -> Option<SegmentId> {
    layers(project, playhead_us, hidden, plate)
        .find(|(_, layer)| layer_contains(layer, point))
        .map(|(id, _)| id)
}

fn layers<'a>(
    project: &'a Project,
    playhead_us: i64,
    hidden: &'a [TrackId],
    plate: Rect,
) -> impl Iterator<Item = (SegmentId, LayerBox)> + 'a {
    project
        .timeline()
        .tracks()
        .iter()
        .rev()
        .flat_map(move |track| {
            let skipped = hidden.contains(&track.id());
            track.segments().iter().filter_map(move |segment| {
                if skipped || !covers(segment, playhead_us) {
                    return None;
                }
                picture_box(project, segment, plate, None).map(|layer| (segment.id(), layer))
            })
        })
}

fn pick(selected: &mut Vec<SegmentId>, hit: Option<SegmentId>, shift: bool) {
    match hit {
        Some(id) if shift => {
            if let Some(index) = selected.iter().position(|kept| *kept == id) {
                selected.remove(index);
            } else {
                selected.push(id);
            }
        }
        Some(id) => {
            selected.clear();
            selected.push(id);
        }
        None if !shift => selected.clear(),
        None => {}
    }
}

fn paint_selected(
    ui: &egui::Ui,
    project: &Project,
    playhead_us: i64,
    hidden: &[TrackId],
    plate: Rect,
    selected: &[SegmentId],
    drag: Option<&Drag>,
) {
    let stroke = Stroke::new(1.5, Color32::WHITE);
    for track in project.timeline().tracks() {
        if hidden.contains(&track.id()) {
            continue;
        }
        for segment in track.segments() {
            if !selected.contains(&segment.id()) || !covers(segment, playhead_us) {
                continue;
            }
            if let Some(layer) = picture_box(project, segment, plate, drag) {
                paint_edge(ui.painter(), &layer, stroke);
            }
        }
    }
}

fn paint_edge(painter: &egui::Painter, layer: &LayerBox, stroke: Stroke) {
    let inset = stroke.width / 2.0;
    let half = vec2(
        (layer.size.x / 2.0 - inset).max(0.0),
        (layer.size.y / 2.0 - inset).max(0.0),
    );
    let (sin, cos) = layer.rotation.to_radians().sin_cos();
    let corner = |x: f32, y: f32| {
        pos2(
            layer.center.x + x * cos - y * sin,
            layer.center.y + x * sin + y * cos,
        )
    };
    let points = [
        corner(-half.x, -half.y),
        corner(half.x, -half.y),
        corner(half.x, half.y),
        corner(-half.x, half.y),
    ];
    for index in 0..4 {
        painter.line_segment([points[index], points[(index + 1) % 4]], stroke);
    }
}

struct Canvas {
    playhead_us: i64,
    frame: Rect,
    plate: Rect,
}

fn gesture(
    ui: &mut egui::Ui,
    project: &Project,
    canvas: &Canvas,
    response: &egui::Response,
    selected: &mut Vec<SegmentId>,
    commands: &mut Vec<cutlass_project::Command>,
) -> Option<Drag> {
    let hidden = crate::timeline::hidden_tracks(project);
    let mut live = ui.data(|data| data.get_temp::<Drag>(drag_id()));
    if response.drag_started()
        && !ui.input(|input| input.modifiers.shift)
        && let Some(origin) = ui.input(|input| input.pointer.press_origin())
    {
        match begin_drag(project, canvas, &hidden, selected, origin) {
            Begin::Drag(drag) => {
                live = Some(drag);
                ui.data_mut(|data| data.insert_temp(drag_id(), drag));
            }
            Begin::Locked => {
                crate::timeline::show_note(
                    ui,
                    crate::words::of(ui, crate::words::Phrase::TrackIsLocked),
                );
            }
            Begin::None => {}
        }
    }
    if response.dragged()
        && let Some(point) = response.interact_pointer_pos()
        && let Some(drag) = live.as_mut()
    {
        follow(drag, canvas.plate, point);
        ui.data_mut(|data| data.insert_temp(drag_id(), *drag));
    }
    if response.drag_stopped()
        && let Some(drag) = live
    {
        if let Some(command) = finish(project, &drag) {
            commands.push(command);
        }
        ui.data_mut(|data| data.remove::<Drag>(drag_id()));
        return Some(drag);
    }
    if response.clicked()
        && let Some(point) = response.interact_pointer_pos()
    {
        let on_handle =
            chosen_layer(project, canvas, &hidden, selected, None).is_some_and(|(_, layer)| {
                matches!(
                    grab_at(&layer, canvas.frame, point),
                    Some(Grab::Scale { .. } | Grab::Rotate)
                )
            });
        if !on_handle {
            let shift = ui.input(|input| input.modifiers.shift);
            pick(
                selected,
                layer_at(project, canvas.playhead_us, &hidden, canvas.plate, point),
                shift,
            );
        }
    }
    live
}

enum Begin {
    Drag(Drag),
    Locked,
    None,
}

fn begin_drag(
    project: &Project,
    canvas: &Canvas,
    hidden: &[TrackId],
    selected: &[SegmentId],
    origin: Pos2,
) -> Begin {
    let Some((id, layer)) = chosen_layer(project, canvas, hidden, selected, None) else {
        return Begin::None;
    };
    let Some(grab) = grab_at(&layer, canvas.frame, origin) else {
        return Begin::None;
    };
    if crate::timeline::segment_locked(project, id) {
        return Begin::Locked;
    }
    let Some(segment) = project.timeline().segment(id) else {
        return Begin::None;
    };
    let Some(base) = unit_size(project, segment, canvas.plate) else {
        return Begin::None;
    };
    let transform = segment.transform();
    let (flip_horizontal, flip_vertical) = transform.flips();
    Begin::Drag(Drag {
        segment: id,
        grab,
        position: transform.position(),
        scale: transform.scale(),
        rotation: transform.rotation(),
        opacity: transform.opacity(),
        flip_horizontal,
        flip_vertical,
        origin,
        start_position: transform.position(),
        start_rotation: transform.rotation(),
        base,
        start_box: layer,
        guide_x: false,
        guide_y: false,
    })
}

fn follow(drag: &mut Drag, plate: Rect, pointer: Pos2) {
    match drag.grab {
        Grab::Move => {
            let (position, guide_x, guide_y) =
                moved(drag.start_position, plate, drag.origin, pointer);
            drag.position = position;
            drag.guide_x = guide_x;
            drag.guide_y = guide_y;
        }
        Grab::Scale { x, y } => {
            if let Some((position, scale)) =
                scaled(plate, &drag.start_box, drag.base, (x, y), pointer)
            {
                drag.position = position;
                drag.scale = scale;
            }
        }
        Grab::Rotate => {
            drag.rotation = turned(
                drag.start_rotation,
                drag.start_box.center,
                drag.origin,
                pointer,
            );
        }
    }
}

fn finish(project: &Project, drag: &Drag) -> Option<cutlass_project::Command> {
    let current = project.timeline().segment(drag.segment)?.transform();
    let next = cutlass_project::Transform::new(
        drag.position,
        drag.scale,
        drag.rotation,
        drag.opacity,
        drag.flip_horizontal,
        drag.flip_vertical,
    )
    .ok()?;
    if next == current {
        None
    } else {
        Some(cutlass_project::Command::SetTransform {
            segment: drag.segment,
            transform: next,
        })
    }
}

fn chosen_layer(
    project: &Project,
    canvas: &Canvas,
    hidden: &[TrackId],
    selected: &[SegmentId],
    drag: Option<&Drag>,
) -> Option<(SegmentId, LayerBox)> {
    let [id] = *selected else {
        return None;
    };
    for track in project.timeline().tracks() {
        if hidden.contains(&track.id()) {
            continue;
        }
        for segment in track.segments() {
            if segment.id() != id || !covers(segment, canvas.playhead_us) {
                continue;
            }
            return picture_box(project, segment, canvas.plate, drag).map(|layer| (id, layer));
        }
    }
    None
}

fn unit_size(project: &Project, segment: &cutlass_project::Segment, plate: Rect) -> Option<Vec2> {
    let media = project.asset(segment.asset_id()?)?.visual_size()?;
    layer_box(
        plate,
        (project.settings().width(), project.settings().height()),
        media,
        (0.0, 0.0),
        (1.0, 1.0),
        0.0,
    )
    .map(|layer| layer.size)
}

/// Position after a move. An axis within [`GUIDE_PX`] of the canvas center snaps there.
fn moved(start: (f64, f64), plate: Rect, from: Pos2, to: Pos2) -> ((f64, f64), bool, bool) {
    if plate.width() <= 0.0 || plate.height() <= 0.0 {
        return (start, false, false);
    }
    let x = start.0 + f64::from(to.x - from.x) / f64::from(plate.width());
    let y = start.1 + f64::from(to.y - from.y) / f64::from(plate.height());
    let guide_x = (x * f64::from(plate.width())).abs() <= f64::from(GUIDE_PX);
    let guide_y = (y * f64::from(plate.height())).abs() <= f64::from(GUIDE_PX);
    (
        (if guide_x { 0.0 } else { x }, if guide_y { 0.0 } else { y }),
        guide_x,
        guide_y,
    )
}

/// Scale from the opposite side. The pointer sets the moving edges. Scale stays positive.
fn scaled(
    plate: Rect,
    layer: &LayerBox,
    base: Vec2,
    handle: (i8, i8),
    pointer: Pos2,
) -> Option<((f64, f64), (f64, f64))> {
    if base.x <= 0.0 || base.y <= 0.0 || plate.width() <= 0.0 || plate.height() <= 0.0 {
        return None;
    }
    let anchor = local_point(
        layer,
        -f32::from(handle.0) * layer.size.x / 2.0,
        -f32::from(handle.1) * layer.size.y / 2.0,
    );
    let (sin, cos) = (-layer.rotation).to_radians().sin_cos();
    let dx = pointer.x - anchor.x;
    let dy = pointer.y - anchor.y;
    let local_x = dx * cos - dy * sin;
    let local_y = dx * sin + dy * cos;
    let mut width = layer.size.x;
    let mut height = layer.size.y;
    if handle.0 != 0 {
        width = local_x * f32::from(handle.0);
    }
    if handle.1 != 0 {
        height = local_y * f32::from(handle.1);
    }
    width = width.max(base.x * MIN_SCALE);
    height = height.max(base.y * MIN_SCALE);
    let toward_x = f32::from(handle.0) * width / 2.0;
    let toward_y = f32::from(handle.1) * height / 2.0;
    let (sin, cos) = layer.rotation.to_radians().sin_cos();
    let center = pos2(
        anchor.x + toward_x * cos - toward_y * sin,
        anchor.y + toward_x * sin + toward_y * cos,
    );
    Some((
        (
            f64::from((center.x - plate.center().x) / plate.width()),
            f64::from((center.y - plate.center().y) / plate.height()),
        ),
        (f64::from(width / base.x), f64::from(height / base.y)),
    ))
}

fn turned(start: f64, center: Pos2, from: Pos2, to: Pos2) -> f64 {
    start + pointer_angle(center, to) - pointer_angle(center, from)
}

fn pointer_angle(center: Pos2, point: Pos2) -> f64 {
    f64::from(point.y - center.y)
        .atan2(f64::from(point.x - center.x))
        .to_degrees()
}

fn local_point(layer: &LayerBox, x: f32, y: f32) -> Pos2 {
    let (sin, cos) = layer.rotation.to_radians().sin_cos();
    pos2(
        layer.center.x + x * cos - y * sin,
        layer.center.y + x * sin + y * cos,
    )
}

fn rotate_point(layer: &LayerBox, frame: Rect) -> Pos2 {
    let outside = local_point(layer, 0.0, -layer.size.y / 2.0 - ROTATE_GAP);
    if frame.contains(outside) {
        outside
    } else {
        local_point(layer, 0.0, -layer.size.y / 2.0 + ROTATE_GAP)
    }
}

fn handle_points(layer: &LayerBox, frame: Rect) -> Vec<(Grab, Pos2)> {
    let mut points = Vec::with_capacity(9);
    for y in [-1_i8, 0, 1] {
        for x in [-1_i8, 0, 1] {
            if x == 0 && y == 0 {
                continue;
            }
            points.push((
                Grab::Scale { x, y },
                local_point(
                    layer,
                    f32::from(x) * layer.size.x / 2.0,
                    f32::from(y) * layer.size.y / 2.0,
                ),
            ));
        }
    }
    points.push((Grab::Rotate, rotate_point(layer, frame)));
    points
}

fn grab_at(layer: &LayerBox, frame: Rect, point: Pos2) -> Option<Grab> {
    let mut best: Option<(f32, Grab)> = None;
    for (grab, at) in handle_points(layer, frame) {
        let distance = at.distance(point);
        if distance <= HANDLE_PX && best.is_none_or(|(was, _)| distance < was) {
            best = Some((distance, grab));
        }
    }
    if let Some((_, grab)) = best {
        return Some(grab);
    }
    layer_contains(layer, point).then_some(Grab::Move)
}

fn paint_guides(ui: &egui::Ui, plate: Rect, drag: Option<&Drag>) {
    let Some(drag) = drag else {
        return;
    };
    let stroke = Stroke::new(1.0, theme::EXPORT);
    if drag.guide_x {
        ui.painter().line_segment(
            [
                pos2(plate.center().x, plate.top()),
                pos2(plate.center().x, plate.bottom()),
            ],
            stroke,
        );
    }
    if drag.guide_y {
        ui.painter().line_segment(
            [
                pos2(plate.left(), plate.center().y),
                pos2(plate.right(), plate.center().y),
            ],
            stroke,
        );
    }
}

fn paint_handles(
    ui: &egui::Ui,
    project: &Project,
    canvas: &Canvas,
    hidden: &[TrackId],
    selected: &[SegmentId],
    drag: Option<&Drag>,
) {
    let Some((_, layer)) = chosen_layer(project, canvas, hidden, selected, drag) else {
        return;
    };
    let rotate = rotate_point(&layer, canvas.frame);
    let top = local_point(&layer, 0.0, -layer.size.y / 2.0);
    ui.painter()
        .line_segment([top, rotate], Stroke::new(1.0, Color32::WHITE));
    ui.painter().circle_filled(rotate, 5.0, Color32::WHITE);
    for (grab, at) in handle_points(&layer, canvas.frame) {
        if matches!(grab, Grab::Scale { .. }) {
            ui.painter().rect_filled(
                Rect::from_center_size(at, Vec2::splat(7.0)),
                0.0,
                Color32::WHITE,
            );
        }
    }
}

/// `Still · M:SS.t`, in `language`. The clock stays `M:SS.t`.
pub(crate) fn still_toast(language: crate::words::Language, playhead_us: i64) -> String {
    format!(
        "{} · {}",
        crate::words::say(language, crate::words::Phrase::Still),
        crate::library::card_time(playhead_us, 1_000_000)
    )
}

/// The topmost segment that covers `us`, named from its file. A hidden lane is skipped.
fn shot<'a>(project: &'a Project, us: i64, hidden: &[cutlass_project::TrackId]) -> Option<&'a str> {
    for track in project.timeline().tracks().iter().rev() {
        if hidden.contains(&track.id()) {
            continue;
        }
        for segment in track.segments() {
            let range = segment.timeline();
            if segment.visible() && range.start().value <= us && us < range.end().value {
                if let Some(words) = segment.words() {
                    return Some(words);
                }
                if let Some(name) = segment.sticker_name() {
                    return Some(name);
                }
                if let Some(plugin) = segment.attachment().map(|attachment| attachment.plugin()) {
                    return Some(plugin);
                }
                if segment.is_effect() {
                    return Some("Effect");
                }
                if segment.is_filter() {
                    return Some("Filter");
                }
                if segment.is_adjust() {
                    return Some("Adjust");
                }
                let asset = project.asset(segment.asset_id()?)?;
                return Some(file_name(asset.source()));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment, Speed, TimeRange,
    };

    use super::shot;

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "Untitled")
    }

    #[test]
    fn the_topmost_segment_under_the_playhead_names_the_frame() {
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
        let placed = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(placed).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place");
        assert_eq!(shot(&cut, 0, &[]), Some("v.mp4"));
        assert_eq!(shot(&cut, 1_000_000, &[]), None);
        assert_eq!(shot(&project(), 0, &[]), None);
        assert_eq!(shot(&cut, 0, &[main]), None);
    }

    #[test]
    fn the_canvas_fits_inside_the_frame_and_keeps_its_shape() {
        let wide = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 200.0),
        );
        let plate = super::fitted_canvas(wide, 1920, 1080).expect("plate");
        assert!((plate.height() - 200.0).abs() < 0.01);
        assert!((plate.width() - 200.0 * 16.0 / 9.0).abs() < 0.01);
        assert!((plate.center().y - 100.0).abs() < 0.01);

        let tall = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(10.0, 20.0),
            eframe::egui::vec2(200.0, 400.0),
        );
        let plate = super::fitted_canvas(tall, 1080, 1920).expect("plate");
        assert!((plate.width() - 200.0).abs() < 0.01);
        assert!((plate.height() - 200.0 * 16.0 / 9.0).abs() < 0.01);
        assert!((plate.center().x - 110.0).abs() < 0.01);

        let exact = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(320.0, 180.0),
        );
        let plate = super::fitted_canvas(exact, 1920, 1080).expect("plate");
        assert!((plate.width() - 320.0).abs() < 0.01);
        assert!((plate.height() - 180.0).abs() < 0.01);
        assert!(super::fitted_canvas(exact, 0, 1080).is_none());
    }

    #[test]
    fn a_layer_keeps_the_picture_shape_and_a_turn_moves_the_hit() {
        let plate = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(320.0, 180.0),
        );
        let full = super::layer_box(
            plate,
            (1920, 1080),
            (1920, 1080),
            (0.0, 0.0),
            (1.0, 1.0),
            0.0,
        )
        .expect("layer");
        assert!((full.size.x - 320.0).abs() < 0.01);
        assert!((full.size.y - 180.0).abs() < 0.01);
        assert!(super::layer_contains(&full, plate.center()));

        let tall = super::layer_box(
            plate,
            (1920, 1080),
            (1080, 1920),
            (0.0, 0.0),
            (1.0, 1.0),
            0.0,
        )
        .expect("layer");
        assert!((tall.size.y - 180.0).abs() < 0.01);
        assert!(tall.size.x < 200.0);
        assert!(super::layer_contains(&tall, plate.center()));
        assert!(!super::layer_contains(
            &tall,
            eframe::egui::pos2(20.0, 90.0)
        ));

        let moved = super::layer_box(
            plate,
            (1920, 1080),
            (1920, 1080),
            (0.5, 0.0),
            (1.0, 1.0),
            0.0,
        )
        .expect("layer");
        assert!((moved.center.x - 320.0).abs() < 0.01);

        let flat = super::layer_box(
            plate,
            (1920, 1080),
            (1920, 1080),
            (0.0, 0.0),
            (1.0, 0.1),
            0.0,
        )
        .expect("layer");
        let above = eframe::egui::pos2(160.0, 50.0);
        assert!(!super::layer_contains(&flat, above));
        let turned = super::layer_box(
            plate,
            (1920, 1080),
            (1920, 1080),
            (0.0, 0.0),
            (1.0, 0.1),
            90.0,
        )
        .expect("layer");
        assert!(super::layer_contains(&turned, above));
    }

    #[test]
    fn a_move_snaps_to_the_center_and_a_side_sets_the_scale() {
        let plate = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(320.0, 180.0),
        );
        let (position, guide_x, guide_y) = super::moved(
            (0.0, 0.0),
            plate,
            eframe::egui::pos2(160.0, 90.0),
            eframe::egui::pos2(164.0, 90.0),
        );
        assert_eq!(position, (0.0, 0.0));
        assert!(guide_x && guide_y);

        let (position, guide_x, guide_y) = super::moved(
            (0.0, 0.0),
            plate,
            eframe::egui::pos2(0.0, 90.0),
            eframe::egui::pos2(40.0, 90.0),
        );
        assert!((position.0 - 40.0 / 320.0).abs() < 1.0e-6);
        assert!(!guide_x && guide_y);

        let layer = super::layer_box(
            plate,
            (1920, 1080),
            (1920, 1080),
            (0.0, 0.0),
            (1.0, 1.0),
            0.0,
        )
        .expect("layer");
        let (position, scale) = super::scaled(
            plate,
            &layer,
            layer.size,
            (1, 0),
            eframe::egui::pos2(340.0, 90.0),
        )
        .expect("scale");
        assert!((scale.0 - 340.0 / 320.0).abs() < 1.0e-4);
        assert!((scale.1 - 1.0).abs() < 1.0e-4);
        assert!((position.0 - 10.0 / 320.0).abs() < 1.0e-4);
        assert!(position.1.abs() < 1.0e-4);

        let quarter = super::turned(
            0.0,
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::pos2(0.0, -10.0),
            eframe::egui::pos2(10.0, 0.0),
        );
        assert!((quarter - 90.0).abs() < 0.01);
    }

    #[test]
    fn a_click_selects_the_topmost_layer_and_a_miss_clears_it() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let cut = placed();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 300.0),
        );
        let mut selected = Vec::new();
        let mut time = 0.0;
        let (tree, _) = frame(&ctx, &cut, &mut selected, &mut time, screen, vec![]);
        let canvas = node_bounds(&tree, "v.mp4 · 00:00:00:00");
        let center = eframe::egui::pos2((canvas.0 + canvas.2) / 2.0, (canvas.1 + canvas.3) / 2.0);
        press(&ctx, &cut, &mut selected, &mut time, screen, center, false);
        let id = cut
            .timeline()
            .tracks()
            .iter()
            .flat_map(|track| track.segments())
            .next()
            .expect("segment")
            .id();
        assert_eq!(selected, vec![id]);

        press(&ctx, &cut, &mut selected, &mut time, screen, center, true);
        assert!(selected.is_empty());
        press(&ctx, &cut, &mut selected, &mut time, screen, center, false);
        let letterbox = eframe::egui::pos2((canvas.0 + canvas.2) / 2.0, canvas.1 + 2.0);
        press(
            &ctx,
            &cut,
            &mut selected,
            &mut time,
            screen,
            letterbox,
            false,
        );
        assert!(selected.is_empty());
    }

    #[test]
    fn dragging_the_layer_writes_its_new_place() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let cut = placed();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 300.0),
        );
        let mut selected = Vec::new();
        let mut time = 0.0;
        let (tree, _) = frame(&ctx, &cut, &mut selected, &mut time, screen, vec![]);
        let canvas = node_bounds(&tree, "v.mp4 · 00:00:00:00");
        let center = eframe::egui::pos2((canvas.0 + canvas.2) / 2.0, (canvas.1 + canvas.3) / 2.0);
        press(&ctx, &cut, &mut selected, &mut time, screen, center, false);
        assert_eq!(selected.len(), 1);
        let commands = pull(
            &ctx,
            &cut,
            &mut selected,
            &mut time,
            screen,
            center,
            center + eframe::egui::vec2(40.0, 0.0),
        );
        let frame_rect = eframe::egui::Rect::from_min_max(
            screen.min + eframe::egui::vec2(16.0, 16.0),
            eframe::egui::pos2(screen.right() - 16.0, screen.bottom() - 52.0 - 8.0),
        );
        let plate =
            super::fitted_canvas(frame_rect, cut.settings().width(), cut.settings().height())
                .expect("plate");
        match commands.as_slice() {
            [Command::SetTransform { transform, .. }] => {
                let (x, y) = transform.position();
                assert!((x - f64::from(40.0 / plate.width())).abs() < 1.0e-3);
                assert!(y.abs() < 1.0e-3);
                assert_eq!(transform.scale(), (1.0, 1.0));
            }
            other => panic!("expected a transform, got {other:?}"),
        }
    }

    #[test]
    fn a_hidden_or_silent_segment_is_not_a_layer() {
        let plate = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(320.0, 180.0),
        );
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
            Segment::media(stored, source, 0, Channels::Sound, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment,
                }],
            )
            .expect("place");
        assert!(super::layer_at(&cut, 0, &[], plate, plate.center()).is_none());
        assert_eq!(super::shot(&cut, 0, &[]), Some("v.mp4"));
        let picture = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Picture, Speed::ONE).expect("segment")
        };
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment: picture,
                }],
            )
            .expect("place");
        assert!(super::layer_at(&cut, 0, &[], plate, plate.center()).is_some());
        assert!(super::layer_at(&cut, 0, &[main], plate, plate.center()).is_none());

        let mut picture = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 2_000_000, Channels::Picture, Speed::ONE)
                .expect("segment")
        };
        picture.set_visible(false);
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment: picture,
                }],
            )
            .expect("place");
        assert!(super::layer_at(&cut, 2_000_000, &[], plate, plate.center()).is_none());
        assert!(super::shot(&cut, 2_000_000, &[]).is_none());
    }

    #[test]
    fn the_front_layer_wins_the_click() {
        let plate = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(320.0, 180.0),
        );
        let mut history = History::new();
        let mut cut = project();
        let back = video_asset("file:///cuts/back.mp4");
        let front = video_asset("file:///cuts/front.mp4");
        history
            .apply(
                &mut cut,
                &[Command::AddAsset(back), Command::AddAsset(front)],
            )
            .expect("add");
        let main = cut.timeline().main_track().id();
        place_at(&mut history, &mut cut, 0, Some(main), Channels::Both);
        place_at(&mut history, &mut cut, 1, None, Channels::Both);
        let hit = super::layer_at(&cut, 0, &[], plate, plate.center()).expect("layer");
        let source = cut
            .asset(
                cut.timeline()
                    .tracks()
                    .iter()
                    .find_map(|track| {
                        track
                            .segments()
                            .iter()
                            .find(|segment| segment.id() == hit)
                            .and_then(|segment| segment.asset_id())
                    })
                    .expect("segment"),
            )
            .expect("asset")
            .source();
        assert_eq!(crate::timeline::file_name(source), "front.mp4");
    }

    #[test]
    fn the_still_toast_names_the_time_in_either_language() {
        use crate::words::Language;
        assert_eq!(
            super::still_toast(Language::English, 1_500_000),
            "Still · 0:01.5"
        );
        assert_eq!(
            super::still_toast(Language::French, 1_500_000),
            "Image fixe · 0:01.5"
        );
    }

    #[test]
    fn the_camera_marks_a_still_when_the_cut_has_a_duration() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let cut = placed();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 300.0),
        );
        let mut time = 0.0;
        let mut pass = |project: &Project, events: Vec<eframe::egui::Event>| {
            time += 0.05;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, project, 1_500_000, false, &mut Vec::new(), None);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            tree.expect("the tree is built")
        };
        let empty = pass(&project(), vec![]);
        let camera = bounds(&empty, "Still");
        pass(
            &project(),
            vec![
                eframe::egui::Event::PointerMoved(camera),
                eframe::egui::Event::PointerButton {
                    pos: camera,
                    button: eframe::egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: eframe::egui::Modifiers::default(),
                },
            ],
        );
        pass(
            &project(),
            vec![eframe::egui::Event::PointerButton {
                pos: camera,
                button: eframe::egui::PointerButton::Primary,
                pressed: false,
                modifiers: eframe::egui::Modifiers::default(),
            }],
        );
        assert!(crate::timeline::note_text(&ctx).is_none());

        let tree = pass(&cut, vec![]);
        let camera = bounds(&tree, "Still");
        pass(
            &cut,
            vec![
                eframe::egui::Event::PointerMoved(camera),
                eframe::egui::Event::PointerButton {
                    pos: camera,
                    button: eframe::egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: eframe::egui::Modifiers::default(),
                },
            ],
        );
        pass(
            &cut,
            vec![eframe::egui::Event::PointerButton {
                pos: camera,
                button: eframe::egui::PointerButton::Primary,
                pressed: false,
                modifiers: eframe::egui::Modifiers::default(),
            }],
        );
        assert_eq!(
            crate::timeline::note_text(&ctx).as_deref(),
            Some("Still · 0:01.5")
        );
    }

    fn placed() -> Project {
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
        let id = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(id).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place");
        cut
    }

    fn video_asset(source: &str) -> Asset {
        Asset::video(
            source,
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video")
    }

    fn place_at(
        history: &mut History,
        cut: &mut Project,
        index: usize,
        track: Option<cutlass_project::TrackId>,
        channels: Channels,
    ) {
        let asset = cut.assets().get(index).expect("asset");
        let duration = asset.duration().expect("duration");
        let source = TimeRange::new(
            Timestamp::new(0, duration.timescale),
            Timestamp::new(duration.value, duration.timescale),
        )
        .expect("source");
        let segment = Segment::media(asset, source, 0, channels, Speed::ONE).expect("segment");
        history
            .apply(cut, &[Command::Place { track, segment }])
            .expect("place");
    }

    fn frame(
        ctx: &eframe::egui::Context,
        cut: &Project,
        selected: &mut Vec<cutlass_project::SegmentId>,
        time: &mut f64,
        screen: eframe::egui::Rect,
        events: Vec<eframe::egui::Event>,
    ) -> (
        eframe::egui::accesskit::TreeUpdate,
        Vec<cutlass_project::Command>,
    ) {
        *time += 0.05;
        let mut commands = Vec::new();
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(screen),
                time: Some(*time),
                events,
                ..Default::default()
            },
            |ui| {
                commands.clone_from(&super::show(ui, cut, 0, false, selected, None));
            },
        );
        let tree = output.platform_output.accesskit_update.take();
        output.textures_delta.clear();
        (tree.expect("the tree is built"), commands)
    }

    fn pull(
        ctx: &eframe::egui::Context,
        cut: &Project,
        selected: &mut Vec<cutlass_project::SegmentId>,
        time: &mut f64,
        screen: eframe::egui::Rect,
        from: eframe::egui::Pos2,
        to: eframe::egui::Pos2,
    ) -> Vec<cutlass_project::Command> {
        let modifiers = eframe::egui::Modifiers::default();
        frame(
            ctx,
            cut,
            selected,
            time,
            screen,
            vec![
                eframe::egui::Event::PointerMoved(from),
                eframe::egui::Event::PointerButton {
                    pos: from,
                    button: eframe::egui::PointerButton::Primary,
                    pressed: true,
                    modifiers,
                },
            ],
        );
        frame(
            ctx,
            cut,
            selected,
            time,
            screen,
            vec![eframe::egui::Event::PointerMoved(to)],
        );
        frame(
            ctx,
            cut,
            selected,
            time,
            screen,
            vec![eframe::egui::Event::PointerButton {
                pos: to,
                button: eframe::egui::PointerButton::Primary,
                pressed: false,
                modifiers,
            }],
        )
        .1
    }

    fn press(
        ctx: &eframe::egui::Context,
        cut: &Project,
        selected: &mut Vec<cutlass_project::SegmentId>,
        time: &mut f64,
        screen: eframe::egui::Rect,
        pos: eframe::egui::Pos2,
        shift: bool,
    ) {
        let modifiers = eframe::egui::Modifiers {
            shift,
            ..Default::default()
        };
        frame(
            ctx,
            cut,
            selected,
            time,
            screen,
            vec![
                eframe::egui::Event::ModifiersChanged(modifiers),
                eframe::egui::Event::PointerMoved(pos),
                eframe::egui::Event::PointerButton {
                    pos,
                    button: eframe::egui::PointerButton::Primary,
                    pressed: true,
                    modifiers,
                },
            ],
        );
        frame(
            ctx,
            cut,
            selected,
            time,
            screen,
            vec![
                eframe::egui::Event::ModifiersChanged(modifiers),
                eframe::egui::Event::PointerButton {
                    pos,
                    button: eframe::egui::PointerButton::Primary,
                    pressed: false,
                    modifiers,
                },
            ],
        );
    }

    fn node_bounds(
        tree: &eframe::egui::accesskit::TreeUpdate,
        label: &str,
    ) -> (f32, f32, f32, f32) {
        let (_, node) = tree
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some(label))
            .expect(label);
        let rect = node.bounds().expect("bounds");
        (
            rect.x0 as f32,
            rect.y0 as f32,
            rect.x1 as f32,
            rect.y1 as f32,
        )
    }

    fn bounds(tree: &eframe::egui::accesskit::TreeUpdate, label: &str) -> eframe::egui::Pos2 {
        let (_, node) = tree
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some(label))
            .expect(label);
        let rect = node.bounds().expect("bounds");
        eframe::egui::pos2(
            ((rect.x0 + rect.x1) / 2.0) as f32,
            ((rect.y0 + rect.y1) / 2.0) as f32,
        )
    }

    #[test]
    fn media_time_follows_the_playhead_and_the_speed() {
        let scale = NonZeroU32::new(30_000).expect("scale");
        let asset = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, scale),
            320,
            180,
            true,
        )
        .expect("video");
        let source = TimeRange::new(Timestamp::new(6_000, scale), Timestamp::new(24_000, scale))
            .expect("source");
        let segment =
            Segment::media(&asset, source, 1_000_000, Channels::Both, Speed::ONE).expect("segment");
        assert!(super::media_time(&segment, 0).is_none());
        let start = super::media_time(&segment, 1_000_000).expect("start");
        assert_eq!(start.value, 6_000);
        assert_eq!(start.timescale, scale);
        let mid = super::media_time(&segment, 1_400_000).expect("mid");
        assert_eq!(mid.value, 18_000);

        let fast = Segment::media(
            &asset,
            source,
            0,
            Channels::Both,
            Speed::new(2, 1).expect("speed"),
        )
        .expect("fast");
        let sped = super::media_time(&fast, 200_000).expect("sped");
        assert_eq!(sped.value, 18_000);
    }

    #[test]
    fn a_speed_curve_seeks_ahead_of_the_constant_ramp() {
        let scale = NonZeroU32::new(1_000_000).expect("scale");
        let asset = Asset::video(
            "file:///cuts/ramp.mp4",
            Timestamp::new(1_000_000, scale),
            16,
            16,
            true,
        )
        .expect("video");
        let mut history = History::new();
        let mut cut = project();
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(asset)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let source = TimeRange::new(Timestamp::new(0, scale), Timestamp::new(1_000_000, scale))
            .expect("source");
        let segment = Segment::media(
            cut.asset(asset).expect("asset"),
            source,
            0,
            Channels::Both,
            Speed::ONE,
        )
        .expect("segment");
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
        let curve = cutlass_project::SpeedCurve::new(vec![
            cutlass_project::SpeedPoint::new(0, 1, Speed::ONE).expect("start"),
            cutlass_project::SpeedPoint::new(1, 1, Speed::new(2, 1).expect("fast")).expect("end"),
        ])
        .expect("curve");
        history
            .apply(
                &mut cut,
                &[Command::SetCurve {
                    segment: id,
                    curve: Some(curve),
                }],
            )
            .expect("curve");
        let segment = cut.timeline().segment(id).expect("clip");
        let mid = super::media_time(segment, 346_573).expect("mid");
        assert_eq!(mid.value, 414_213);
        assert_eq!(mid.timescale, scale);
        assert_ne!(mid.value, 346_573);
    }

    #[test]
    fn a_nested_timeline_does_not_change_the_seek() {
        let scale = NonZeroU32::new(1_000_000).expect("scale");
        let asset = Asset::video(
            "file:///cuts/nest.mp4",
            Timestamp::new(1_000_000, scale),
            16,
            16,
            true,
        )
        .expect("video");
        let mut history = History::new();
        let mut cut = project();
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(asset)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let source = TimeRange::new(Timestamp::new(0, scale), Timestamp::new(1_000_000, scale))
            .expect("source");
        let segment = Segment::media(
            cut.asset(asset).expect("asset"),
            source,
            0,
            Channels::Both,
            Speed::ONE,
        )
        .expect("segment");
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
        let nest = match history
            .apply(&mut cut, &[Command::AddTimeline])
            .expect("timeline")
            .as_slice()
        {
            [Outcome::Timeline(nest)] => *nest,
            other => panic!("expected a timeline, got {other:?}"),
        };
        history
            .apply(
                &mut cut,
                &[Command::SetNest {
                    segment: id,
                    nest: Some(nest),
                }],
            )
            .expect("nest");
        let mid = {
            let segment = cut.timeline().segment(id).expect("clip");
            super::media_time(segment, 400_000).expect("mid")
        };
        assert_eq!(mid.value, 400_000);
        assert_eq!(mid.timescale, scale);
        let stored = cut.nest(nest).expect("stored");
        assert_eq!(stored.tracks().len(), 1);
        assert!(stored.main_track().segments().is_empty());
    }

    #[test]
    fn a_rotated_picture_stays_inside_its_box() {
        let flat = super::LayerBox {
            center: eframe::egui::pos2(100.0, 80.0),
            size: eframe::egui::vec2(40.0, 20.0),
            rotation: 0.0,
        };
        for corner in super::picture_corners(&flat) {
            assert!(super::layer_contains(&flat, corner));
        }
        let layer = super::LayerBox {
            center: eframe::egui::pos2(100.0, 80.0),
            size: eframe::egui::vec2(40.0, 20.0),
            rotation: 37.0,
        };
        let (sin, cos) = (-layer.rotation).to_radians().sin_cos();
        for corner in super::picture_corners(&layer) {
            let dx = corner.x - layer.center.x;
            let dy = corner.y - layer.center.y;
            let local_x = dx * cos - dy * sin;
            let local_y = dx * sin + dy * cos;
            assert!(
                (local_x.abs() - layer.size.x / 2.0).abs() < 1e-3,
                "local x {local_x}"
            );
            assert!(
                (local_y.abs() - layer.size.y / 2.0).abs() < 1e-3,
                "local y {local_y}"
            );
            let toward = eframe::egui::pos2(
                layer.center.x + (corner.x - layer.center.x) * 0.99,
                layer.center.y + (corner.y - layer.center.y) * 0.99,
            );
            assert!(super::layer_contains(&layer, toward));
        }
        assert!(!super::layer_contains(
            &layer,
            layer.center + eframe::egui::vec2(80.0, 80.0)
        ));
    }

    #[test]
    fn a_picture_is_painted_only_when_a_frame_is_ready() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let image = eframe::egui::ColorImage::filled([2, 2], eframe::egui::Color32::WHITE);
        let texture = ctx.load_texture("test-frame", image, eframe::egui::TextureOptions::LINEAR);
        let cut = placed();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 300.0),
        );
        let painted = |picture| {
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(1.0),
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, &cut, 0, false, &mut Vec::new(), picture);
                },
            );
            let found = output
                .shapes
                .iter()
                .any(|clipped| uses_texture(&clipped.shape, texture.id()));
            output.textures_delta.clear();
            found
        };
        assert!(painted(Some(texture.id())));
        assert!(!painted(None));
    }

    fn uses_texture(shape: &eframe::egui::Shape, id: eframe::egui::TextureId) -> bool {
        match shape {
            eframe::egui::Shape::Mesh(mesh) => mesh.texture_id == id,
            eframe::egui::Shape::Vec(shapes) => shapes.iter().any(|shape| uses_texture(shape, id)),
            _ => false,
        }
    }
}
