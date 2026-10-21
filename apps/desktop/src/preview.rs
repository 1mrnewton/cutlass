//! The preview frame and the transport. The frame shows the project canvas.
//! A decoded picture is not drawn yet. `docs/preview.md`.

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, Sense, Vec2, pos2, vec2};

use cutlass_project::Project;

use crate::icons::{self, Icon};
use crate::ruler;
use crate::theme;
use crate::timeline::file_name;

const TRANSPORT: f32 = 52.0;

pub fn show(ui: &mut egui::Ui, project: &Project, playhead_us: i64) {
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
        if let Some(plate) = fitted_canvas(
            frame,
            project.settings().width(),
            project.settings().height(),
        ) {
            let (red, green, blue) = project.settings().background().channels();
            ui.painter()
                .rect_filled(plate, 0.0, Color32::from_rgb(red, green, blue));
        }
        let rate = project.settings().rate();
        let time = ruler::timecode(playhead_us, rate);
        let hidden = crate::timeline::hidden_tracks(ui);
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
            if range.start().value <= us && us < range.end().value {
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
                |ui| super::show(ui, project, 1_500_000),
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
