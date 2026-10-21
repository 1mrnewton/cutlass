//! The preview frame and the transport. The picture is not drawn yet. `docs/preview.md`.

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, Sense, Vec2, pos2, vec2};

use cutlass_project::Project;

use crate::icons::{self, Icon};
use crate::ruler;
use crate::theme;
use crate::timeline::file_name;

const TRANSPORT: f32 = 52.0;

pub fn show(ui: &mut egui::Ui, project: &Project, playhead_us: i64) {
    crate::tip::region(ui, "Preview");
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let transport = Rect::from_min_max(pos2(full.left(), full.bottom() - TRANSPORT), full.max);
    let frame = Rect::from_min_max(
        full.min + vec2(16.0, 16.0),
        pos2(full.right() - 16.0, transport.top() - 8.0),
    );
    if frame.height() > 0.0 && frame.width() > 0.0 {
        let rate = project.settings().rate();
        let time = ruler::timecode(playhead_us, rate);
        let hidden = crate::timeline::hidden_tracks(ui);
        let label = match shot(project, playhead_us, &hidden) {
            Some(name) => format!("{name} · {time}"),
            None => format!("Empty · {time}"),
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
    crate::tip::show_enabled(&play_response, "Play", live);

    let still = Rect::from_center_size(
        pos2(rect.right() - 28.0, rect.center().y),
        Vec2::splat(32.0),
    );
    let still_response = ui.interact(still, ui.scope_id().with("still"), Sense::hover());
    ui.painter()
        .circle_filled(still.center(), 16.0, theme::BAR.gamma_multiply(dim));
    icons::paint(
        ui.painter(),
        still.center(),
        Icon::Camera,
        16.0,
        theme::MUTED.gamma_multiply(dim),
    );
    crate::tip::show_enabled(&still_response, "Still", live);
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
}
