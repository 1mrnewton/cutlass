//! The preview frame and the transport. The frame shows the project canvas.
//! A decoded picture is not drawn yet. `docs/preview.md`.

use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Pos2, Rect, Sense, Stroke, Vec2, pos2, vec2,
};

use cutlass_project::{Channels, Project, SegmentId, TrackId};

use crate::icons::{self, Icon};
use crate::ruler;
use crate::theme;
use crate::timeline::file_name;

const TRANSPORT: f32 = 52.0;

pub fn show(ui: &mut egui::Ui, project: &Project, playhead_us: i64, selected: &mut Vec<SegmentId>) {
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
    if frame.height() > 0.0 && frame.width() > 0.0 {
        let hidden = crate::timeline::hidden_tracks(ui);
        let plate = fitted_canvas(
            frame,
            project.settings().width(),
            project.settings().height(),
        );
        if let Some(plate) = plate {
            let (red, green, blue) = project.settings().background().channels();
            ui.painter()
                .rect_filled(plate, 0.0, Color32::from_rgb(red, green, blue));
            paint_selected(ui, project, playhead_us, &hidden, plate, selected);
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
        let canvas = ui.interact(frame, ui.scope_id().with("preview-canvas"), Sense::click());
        let top = plate.and_then(|plate| top_layer(project, playhead_us, &hidden, plate));
        canvas.widget_info(|| {
            egui::WidgetInfo::selected(
                egui::accesskit::Role::Button,
                true,
                top.is_some_and(|id| selected.contains(&id)),
                &label,
            )
        });
        if canvas.clicked()
            && let Some(plate) = plate
            && let Some(point) = canvas.interact_pointer_pos()
        {
            let shift = ui.input(|input| input.modifiers.shift);
            pick(
                selected,
                layer_at(project, playhead_us, &hidden, plate, point),
                shift,
            );
        }
    }
    paint_transport(ui, transport, project, playhead_us);
}

fn paint_transport(ui: &mut egui::Ui, rect: Rect, project: &Project, playhead_us: i64) {
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
    let play_response = ui.interact(play, ui.scope_id().with("play"), Sense::hover());
    ui.painter()
        .circle_filled(play.center(), 18.0, theme::INK.gamma_multiply(dim));
    let mark = Color32::from_rgb(0x12, 0x13, 0x0f).gamma_multiply(dim);
    icons::paint(ui.painter(), play.center(), Icon::Play, 16.0, mark);
    crate::tip::show_enabled(
        &play_response,
        crate::words::of(ui, crate::words::Phrase::Play),
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
    ui.painter()
        .circle_filled(still.center(), 16.0, theme::BAR.gamma_multiply(dim));
    icons::paint(
        ui.painter(),
        still.center(),
        Icon::Camera,
        16.0,
        theme::MUTED.gamma_multiply(dim),
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
struct LayerBox {
    center: Pos2,
    size: Vec2,
    rotation: f32,
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

fn layer_contains(layer: &LayerBox, point: Pos2) -> bool {
    let (sin, cos) = (-layer.rotation).to_radians().sin_cos();
    let dx = point.x - layer.center.x;
    let dy = point.y - layer.center.y;
    let local_x = dx * cos - dy * sin;
    let local_y = dx * sin + dy * cos;
    local_x.abs() <= layer.size.x / 2.0 && local_y.abs() <= layer.size.y / 2.0
}

fn covers(segment: &cutlass_project::Segment, us: i64) -> bool {
    let range = segment.timeline();
    segment.visible() && range.start().value <= us && us < range.end().value
}

fn picture_box(
    project: &Project,
    segment: &cutlass_project::Segment,
    plate: Rect,
) -> Option<LayerBox> {
    if segment.channels() == Channels::Sound {
        return None;
    }
    let media = project.asset(segment.asset_id())?.visual_size()?;
    let transform = segment.transform();
    layer_box(
        plate,
        (project.settings().width(), project.settings().height()),
        media,
        transform.position(),
        transform.scale(),
        transform.rotation(),
    )
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
                picture_box(project, segment, plate).map(|layer| (segment.id(), layer))
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
            if let Some(layer) = picture_box(project, segment, plate) {
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
                let asset = project.asset(segment.asset_id())?;
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
        let tree = frame(&ctx, &cut, &mut selected, &mut time, screen, vec![]);
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
                            .map(|segment| segment.asset_id())
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
                |ui| super::show(ui, project, 1_500_000, &mut Vec::new()),
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
    ) -> eframe::egui::accesskit::TreeUpdate {
        *time += 0.05;
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(screen),
                time: Some(*time),
                events,
                ..Default::default()
            },
            |ui| super::show(ui, cut, 0, selected),
        );
        let tree = output.platform_output.accesskit_update.take();
        output.textures_delta.clear();
        tree.expect("the tree is built")
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
}
