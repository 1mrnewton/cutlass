//! The project's tracks, drawn bottom to top, with the ruler above them.

use eframe::egui::{self, Align2, FontFamily, FontId, Rect, Stroke, pos2, vec2};

use cutlass_project::{Project, Track, TrackKind};

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

pub fn show(ui: &mut egui::Ui, project: &Project) {
    let full = ui.max_rect();
    let ruler_rect = Rect::from_min_size(full.min, vec2(full.width(), RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler_rect.left_bottom(), full.max);
    paint_ruler(ui, ruler_rect, project);
    if tracks.height() <= 0.0 {
        return;
    }
    for lane in layout(project.timeline().tracks(), tracks) {
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
    }
}

fn lane_left(area: Rect) -> f32 {
    area.left() + HEAD + COVER
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

    use super::layout;

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
}
