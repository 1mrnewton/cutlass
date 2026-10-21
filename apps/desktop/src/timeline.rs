//! The project's tracks, drawn bottom to top, with the ruler above them.

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, Sense, Stroke, pos2, vec2};

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

pub fn show(ui: &mut egui::Ui, project: &Project, playhead: &mut i64) {
    let full = ui.max_rect();
    let ruler_rect = Rect::from_min_size(full.min, vec2(full.width(), RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler_rect.left_bottom(), full.max);
    paint_ruler(ui, ruler_rect, project);
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
            paint_segments(ui, project, track, lane_rect, lane.main, pixel_size);
        }
    }
    paint_playhead(ui, ruler_rect, tracks, project, playhead);
}

fn lane_left(area: Rect) -> f32 {
    area.left() + HEAD + COVER
}

fn paint_segments(
    ui: &mut egui::Ui,
    project: &Project,
    track: &Track,
    lane: Rect,
    main: bool,
    pixel_size: f64,
) {
    for segment in track.segments() {
        let range = segment.timeline();
        let rect = clip_rect(
            lane,
            main,
            range.start().value,
            range.duration().value,
            pixel_size,
        );
        if rect.left() > lane.right() || rect.right() < lane.left() {
            continue;
        }
        ui.painter()
            .rect_filled(rect, 7.0, clip_color(track.kind()));
        let name = project
            .asset(segment.asset_id())
            .map(|asset| file_name(asset.source()))
            .unwrap_or("");
        if name.is_empty() {
            continue;
        }
        ui.painter().with_clip_rect(rect.shrink(6.0)).text(
            rect.left_center() + vec2(8.0, 0.0),
            Align2::LEFT_CENTER,
            name,
            FontId::new(11.0, FontFamily::Proportional),
            Color32::WHITE,
        );
    }
}

fn clip_rect(lane: Rect, main: bool, start_us: i64, duration_us: i64, pixel_size: f64) -> Rect {
    let (inset, height) = if main { (8.0, 48.0) } else { (6.0, 28.0) };
    let width = (duration_us.max(0) as f64 * pixel_size).max(8.0) as f32;
    let x = lane.left() + (start_us.max(0) as f64 * pixel_size) as f32;
    Rect::from_min_size(pos2(x, lane.top() + inset), vec2(width, height))
}

fn file_name(source: &str) -> &str {
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
}
