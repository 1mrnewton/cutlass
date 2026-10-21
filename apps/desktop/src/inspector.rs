//! The project panel. Clip panels come later. `docs/inspector.md`.

use eframe::egui::{self, RichText, ScrollArea, Sense, UiBuilder, vec2};

use cutlass_project::{AspectRatio, ColorSpace, FrameRate, Project};

use crate::theme;

pub fn show(ui: &mut egui::Ui, project: &Project) {
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let padded = full.shrink2(vec2(16.0, 0.0));
    let mut details = ui.new_child(UiBuilder::new().max_rect(padded));
    details.set_clip_rect(padded);
    ScrollArea::vertical()
        .id_salt("project-details")
        .show(&mut details, |ui| {
            ui.set_min_width(ui.available_width());
            ui.add_space(14.0);
            ui.label(
                RichText::new("Details")
                    .size(15.0)
                    .strong()
                    .color(theme::EXPORT),
            );
            ui.add_space(14.0);
            for (label, value) in facts(project) {
                fact(ui, label, &value);
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
    use cutlass_project::{Command, FrameRate, History, Project, ProjectId, SequenceRate};

    use super::facts;

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
}
