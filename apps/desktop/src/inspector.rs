//! The inspector. A selected clip shows its own rows. Otherwise the project. `docs/inspector.md`.

use eframe::egui::{self, RichText, ScrollArea, Sense, UiBuilder, vec2};

use cutlass_project::{
    AspectRatio, Channels, ColorSpace, FrameRate, Project, SegmentId, TrackKind,
};

use crate::library::card_time;
use crate::theme;
use crate::timeline::file_name;

pub fn show(ui: &mut egui::Ui, project: &Project, selected: Option<SegmentId>) {
    crate::tip::region(ui, "Inspector");
    if let Some(panel) = selected.and_then(|id| clip_panel(project, id)) {
        paint(ui, panel.title, &panel.rows, "clip");
        return;
    }
    paint(ui, "Details", &facts(project), "project");
}

struct Panel {
    title: &'static str,
    rows: Vec<(&'static str, String)>,
}

fn paint(ui: &mut egui::Ui, title: &str, rows: &[(&str, String)], salt: &str) {
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let padded = full.shrink2(vec2(16.0, 0.0));
    let mut details = ui.new_child(UiBuilder::new().max_rect(padded));
    details.set_clip_rect(padded);
    ScrollArea::vertical()
        .id_salt(("inspector", salt))
        .show(&mut details, |ui| {
            ui.set_min_width(ui.available_width());
            ui.add_space(14.0);
            ui.label(
                RichText::new(title)
                    .size(15.0)
                    .strong()
                    .color(theme::EXPORT),
            );
            ui.add_space(14.0);
            for (label, value) in rows {
                fact(ui, label, value);
            }
        });
}

fn fact(ui: &mut egui::Ui, label: &str, value: &str) {
    let width = ui.available_width();
    ui.horizontal(|ui| {
        ui.set_max_width(width);
        ui.spacing_mut().item_spacing.x = 12.0;
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(theme::MUTED)),
        );
        ui.vertical(|ui| {
            ui.set_width((width - 120.0).max(40.0));
            ui.add(egui::Label::new(RichText::new(value).size(13.0).color(theme::INK)).wrap());
        });
    });
    ui.add_space(4.0);
}

fn clip_panel(project: &Project, id: SegmentId) -> Option<Panel> {
    let track = project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.segments().iter().any(|segment| segment.id() == id))?;
    let segment = track.segments().iter().find(|segment| segment.id() == id)?;
    let asset = project.asset(segment.asset_id())?;
    let range = segment.timeline();
    let speed = segment.speed();
    let mut rows = vec![
        ("Name", file_name(asset.source()).to_string()),
        (
            "Start",
            card_time(range.start().value, range.start().timescale.get()),
        ),
        (
            "Duration",
            card_time(range.duration().value, range.duration().timescale.get()),
        ),
        (
            "Speed",
            format!("{}/{}", speed.numerator(), speed.denominator().get()),
        ),
    ];
    if segment.channels() != Channels::Picture {
        rows.push(("Volume", number(segment.volume())));
    }
    if segment.channels() != Channels::Sound {
        let (x, y) = segment.transform().position();
        let (width, height) = segment.transform().scale();
        rows.push(("Position", format!("{}, {}", number(x), number(y))));
        rows.push(("Size", format!("{} × {}", number(width), number(height))));
    }
    Some(Panel {
        title: kind_title(track.kind()),
        rows,
    })
}

fn number(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e6 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

fn kind_title(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "Video",
        TrackKind::Audio => "Audio",
        TrackKind::Text => "Text",
        TrackKind::Sticker => "Sticker",
        TrackKind::Effect | TrackKind::PluginEffect => "Effect",
        TrackKind::Filter => "Filter",
        TrackKind::Adjust => "Adjust",
    }
}

fn facts(project: &Project) -> Vec<(&'static str, String)> {
    let settings = project.settings();
    let mut rows = vec![
        ("Name", project.name().to_string()),
        ("Saved", "Not saved".to_string()),
        (
            "Color space",
            color_space(settings.color_space()).to_string(),
        ),
        ("Ratio", ratio(settings.ratio()).to_string()),
        (
            "Resolution",
            format!("{}×{}", settings.width(), settings.height()),
        ),
        ("Frame rate", frame_rate(settings.rate().rate()).to_string()),
    ];
    if settings.rate().rate().allows_drop_frame() {
        let value = if settings.rate().drop_frame() {
            "On"
        } else {
            "Off"
        };
        rows.push(("Drop frame", value.to_string()));
    }
    rows
}

fn color_space(space: ColorSpace) -> &'static str {
    match space {
        ColorSpace::Rec709 => "Rec. 709",
        ColorSpace::Rec2020Hlg => "Rec. 2020 HLG",
        ColorSpace::Rec2100Pq => "Rec. 2100 PQ",
    }
}

fn ratio(ratio: AspectRatio) -> &'static str {
    match ratio {
        AspectRatio::Original => "Original",
        AspectRatio::R16x9 => "16:9",
        AspectRatio::R9x16 => "9:16",
        AspectRatio::Square => "1:1",
        AspectRatio::R4x5 => "4:5",
        AspectRatio::R4x3 => "4:3",
    }
}

fn frame_rate(rate: FrameRate) -> &'static str {
    match rate {
        FrameRate::Fps23976 => "23.976 fps",
        FrameRate::Fps24 => "24 fps",
        FrameRate::Fps25 => "25 fps",
        FrameRate::Fps2997 => "29.97 fps",
        FrameRate::Fps30 => "30 fps",
        FrameRate::Fps50 => "50 fps",
        FrameRate::Fps5994 => "59.94 fps",
        FrameRate::Fps60 => "60 fps",
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, FrameRate, History, Outcome, Project, ProjectId, Segment,
        SequenceRate, Speed, TimeRange,
    };

    use super::{clip_panel, facts};

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "Untitled")
    }

    fn labels(project: &Project) -> Vec<&str> {
        facts(project).into_iter().map(|(label, _)| label).collect()
    }

    #[test]
    fn an_empty_cut_shows_the_starting_project() {
        let rows = facts(&project());
        assert_eq!(
            rows,
            vec![
                ("Name", "Untitled".to_string()),
                ("Saved", "Not saved".to_string()),
                ("Color space", "Rec. 709".to_string()),
                ("Ratio", "Original".to_string()),
                ("Resolution", "1920×1080".to_string()),
                ("Frame rate", "30 fps".to_string()),
            ]
        );
    }

    #[test]
    fn drop_frame_appears_only_for_the_two_rates_that_allow_it() {
        let mut history = History::new();
        let mut cut = project();
        history
            .apply(&mut cut, &[Command::SetRate(FrameRate::Fps2997.sequence())])
            .expect("rate");
        let rows = facts(&cut);
        assert_eq!(rows[5].1, "29.97 fps");
        assert_eq!(rows[6], ("Drop frame", "On".to_string()));

        history
            .apply(
                &mut cut,
                &[Command::SetRate(
                    SequenceRate::new(FrameRate::Fps2997, false).expect("drop can be off"),
                )],
            )
            .expect("drop off");
        assert_eq!(facts(&cut)[6], ("Drop frame", "Off".to_string()));

        history
            .apply(&mut cut, &[Command::SetRate(FrameRate::Fps24.sequence())])
            .expect("24");
        assert!(!labels(&cut).contains(&"Drop frame"));
    }

    #[test]
    fn a_selected_clip_shows_its_file_time_speed_volume_and_placement() {
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
        let panel = clip_panel(&cut, id).expect("panel");
        assert_eq!(panel.title, "Video");
        assert_eq!(
            panel.rows,
            vec![
                ("Name", "v.mp4".to_string()),
                ("Start", "0:00.0".to_string()),
                ("Duration", "0:01.0".to_string()),
                ("Speed", "1/1".to_string()),
                ("Volume", "1".to_string()),
                ("Position", "0, 0".to_string()),
                ("Size", "1 × 1".to_string()),
            ]
        );
    }
}
