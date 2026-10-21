//! The library rail and the project's media bin. `docs/library.md`.

use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Rect, ScrollArea, Sense, Stroke, UiBuilder, pos2,
    vec2,
};

use cutlass_project::{Asset, AssetId, AssetKind, Project};

use crate::icons::{self, Icon};
use crate::theme;
use crate::timeline::file_name;

const RAIL: f32 = 76.0;
const CARD_W: f32 = 144.0;
const CARD_H: f32 = 96.0;
const CARD_GAP: f32 = 8.0;

/// One shelf on the rail. Media is the project bin. The others are later.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Shelf {
    Media,
    Audio,
    Text,
    Stickers,
    Effects,
    Transitions,
    Filters,
    Adjustments,
    Templates,
}

impl Shelf {
    const ALL: [Self; 9] = [
        Self::Media,
        Self::Audio,
        Self::Text,
        Self::Stickers,
        Self::Effects,
        Self::Transitions,
        Self::Filters,
        Self::Adjustments,
        Self::Templates,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Media => "Media",
            Self::Audio => "Audio",
            Self::Text => "Text",
            Self::Stickers => "Stickers",
            Self::Effects => "Effects",
            Self::Transitions => "Transitions",
            Self::Filters => "Filters",
            Self::Adjustments => "Adjustments",
            Self::Templates => "Templates",
        }
    }

    const fn icon(self) -> Icon {
        match self {
            Self::Media => Icon::Media,
            Self::Audio => Icon::Audio,
            Self::Text => Icon::Text,
            Self::Stickers => Icon::Stickers,
            Self::Effects => Icon::Effects,
            Self::Transitions => Icon::Transitions,
            Self::Filters => Icon::Filters,
            Self::Adjustments => Icon::Adjustments,
            Self::Templates => Icon::Templates,
        }
    }
}

pub fn show(
    ui: &mut egui::Ui,
    project: &Project,
    shelf: &mut Shelf,
    selected: &mut Option<AssetId>,
) {
    if selected.is_some_and(|id| project.asset(id).is_none()) {
        *selected = None;
    }
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let rail = Rect::from_min_size(full.min, vec2(RAIL, full.height()));
    let hairline = Rect::from_min_size(rail.right_top(), vec2(1.0, full.height()));
    ui.painter().rect_filled(hairline, 0.0, theme::GAP);
    paint_rail(ui, rail, shelf);
    if *shelf == Shelf::Media {
        let assets = Rect::from_min_max(hairline.right_top(), full.max).shrink(12.0);
        paint_bin(ui, assets, project, selected);
    }
}

fn paint_rail(ui: &mut egui::Ui, rect: Rect, shelf: &mut Shelf) {
    let inner = Rect::from_min_max(
        pos2(rect.left() + 8.0, rect.top()),
        pos2(rect.right() - 8.0, rect.bottom()),
    );
    let mut column = ui.new_child(UiBuilder::new().max_rect(inner));
    ScrollArea::vertical()
        .id_salt("library-rail")
        .show(&mut column, |ui| {
            ui.add_space(8.0);
            ui.spacing_mut().item_spacing.y = 2.0;
            for tool in Shelf::ALL {
                if rail_button(ui, tool, *shelf == tool).clicked() {
                    *shelf = tool;
                }
            }
        });
}

fn rail_button(ui: &mut egui::Ui, shelf: Shelf, active: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 44.0), Sense::click());
    if active {
        ui.painter().rect_filled(rect, 10.0, theme::ACTIVE);
    }
    let color = if active { theme::INK } else { theme::FAINT };
    icons::paint(ui.painter(), rect.center(), shelf.icon(), 20.0, color);
    crate::tip::show(&response, shelf.label());
    response
}

fn paint_bin(ui: &mut egui::Ui, rect: Rect, project: &Project, selected: &mut Option<AssetId>) {
    let mut column = ui.new_child(UiBuilder::new().max_rect(rect));
    ScrollArea::vertical()
        .id_salt("library-cards")
        .auto_shrink([false, false])
        .show(&mut column, |ui| {
            let width = ui.available_width();
            let cols = columns(width);
            let assets = project.assets();
            let rows = assets.len().div_ceil(cols).max(1);
            let height = (rows as f32 * (CARD_H + CARD_GAP)).max(ui.available_height());
            let (area, background) = ui.allocate_exact_size(vec2(width, height), Sense::click());
            let mut picked = false;
            for (index, asset) in assets.iter().enumerate() {
                let origin = card_origin(area.min, index, cols);
                let card = Rect::from_min_size(origin, vec2(CARD_W, CARD_H));
                let response = ui.interact(card, ui.scope_id().with(asset.id()), Sense::click());
                if response.clicked() {
                    *selected = Some(asset.id());
                    picked = true;
                }
                crate::tip::show(&response, file_name(asset.source()));
                paint_card(
                    ui,
                    card,
                    asset,
                    *selected == Some(asset.id()),
                    on_timeline(project, asset.id()),
                );
            }
            if background.clicked() && !picked {
                *selected = None;
            }
        });
}

fn paint_card(ui: &egui::Ui, rect: Rect, asset: &Asset, selected: bool, placed: bool) {
    let swatch = swatch(asset.kind());
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 12.0, swatch);
    let ink = ink_on(swatch);
    painter.text(
        rect.left_bottom()
            + vec2(
                8.0,
                if asset.duration().is_some() {
                    -22.0
                } else {
                    -8.0
                },
            ),
        Align2::LEFT_BOTTOM,
        file_name(asset.source()),
        FontId::new(12.0, FontFamily::Proportional),
        ink,
    );
    if let Some(duration) = asset.duration() {
        painter.text(
            rect.left_bottom() + vec2(8.0, -8.0),
            Align2::LEFT_BOTTOM,
            card_time(duration.value, duration.timescale.get()),
            FontId::new(11.0, FontFamily::Proportional),
            Color32::from_white_alpha(220),
        );
    }
    if placed {
        painter.text(
            rect.left_top() + vec2(8.0, 8.0),
            Align2::LEFT_TOP,
            "On timeline",
            FontId::new(10.0, FontFamily::Proportional),
            ink,
        );
    }
    if selected {
        ui.painter().rect_stroke(
            rect,
            12.0,
            Stroke::new(1.5, Color32::WHITE),
            egui::StrokeKind::Inside,
        );
    }
}

/// How many 144px cards fit. One column does not stretch the card.
fn columns(width: f32) -> usize {
    if width <= 0.0 {
        return 1;
    }
    let count = ((width + CARD_GAP) / (CARD_W + CARD_GAP)).floor();
    if count < 1.0 { 1 } else { count as usize }
}

fn card_origin(area: egui::Pos2, index: usize, cols: usize) -> egui::Pos2 {
    let cols = cols.max(1);
    area + vec2(
        (index % cols) as f32 * (CARD_W + CARD_GAP),
        (index / cols) as f32 * (CARD_H + CARD_GAP),
    )
}

fn swatch(kind: AssetKind) -> Color32 {
    match kind {
        AssetKind::Video { .. } => theme::VIDEO,
        AssetKind::Audio => theme::AUDIO,
        AssetKind::Image { .. } => theme::BAR,
    }
}

fn ink_on(swatch: Color32) -> Color32 {
    let [red, green, blue, _] = swatch.to_array();
    if u16::from(red) + u16::from(green) + u16::from(blue) > 520 {
        theme::GAP
    } else {
        Color32::WHITE
    }
}

/// `M:SS.t`, floored to a tenth of a second.
pub(crate) fn card_time(value: i64, timescale: u32) -> String {
    let scale = i128::from(timescale);
    let tenths = i128::from(value).max(0) * 10 / scale;
    let Ok(tenths) = i64::try_from(tenths) else {
        return "0:00.0".into();
    };
    let mins = tenths / 600;
    let secs = (tenths / 10) % 60;
    let tenth = tenths % 10;
    format!("{mins}:{secs:02}.{tenth}")
}

fn on_timeline(project: &Project, id: AssetId) -> bool {
    project.timeline().tracks().iter().any(|track| {
        track
            .segments()
            .iter()
            .any(|segment| segment.asset_id() == id)
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment, Speed, TimeRange,
    };
    use eframe::egui::pos2;

    use super::{card_origin, card_time, columns, on_timeline, swatch};
    use crate::theme;

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "Untitled")
    }

    #[test]
    fn one_column_until_a_second_card_fits() {
        assert_eq!(columns(0.0), 1);
        assert_eq!(columns(144.0), 1);
        assert_eq!(columns(295.0), 1);
        assert_eq!(columns(296.0), 2);
    }

    #[test]
    fn cards_keep_their_size_as_the_grid_grows() {
        let first = card_origin(pos2(10.0, 20.0), 0, 2);
        let second = card_origin(pos2(10.0, 20.0), 1, 2);
        let wrapped = card_origin(pos2(10.0, 20.0), 2, 2);
        assert!((first.x - 10.0).abs() < f32::EPSILON);
        assert!((second.x - 162.0).abs() < f32::EPSILON);
        assert!((second.y - 20.0).abs() < f32::EPSILON);
        assert!((wrapped.x - 10.0).abs() < f32::EPSILON);
        assert!((wrapped.y - 124.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_second_and_a_half_is_minutes_seconds_and_a_tenth() {
        let sequence = Timestamp::new(1_500_000, NonZeroU32::new(1_000_000).expect("scale"));
        assert_eq!(
            card_time(sequence.value, sequence.timescale.get()),
            "0:01.5"
        );
        let samples = Timestamp::new(72_000, NonZeroU32::new(48_000).expect("scale"));
        assert_eq!(card_time(samples.value, samples.timescale.get()), "0:01.5");
        let minute = Timestamp::new(90_000_000, NonZeroU32::new(1_000_000).expect("scale"));
        assert_eq!(card_time(minute.value, minute.timescale.get()), "1:30.0");
    }

    #[test]
    fn a_placed_asset_is_on_the_timeline_and_a_loose_one_is_not() {
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
        let loose = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        let (placed, unused) = match history
            .apply(
                &mut cut,
                &[Command::AddAsset(video), Command::AddAsset(loose)],
            )
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(placed), Outcome::Asset(unused)] => (*placed, *unused),
            other => panic!("expected two assets, got {other:?}"),
        };
        let sound = {
            let stored = cut.asset(placed).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment: sound,
                }],
            )
            .expect("place");
        assert!(on_timeline(&cut, placed));
        assert!(!on_timeline(&cut, unused));
        assert_eq!(
            swatch(cut.asset(placed).expect("video").kind()),
            theme::VIDEO
        );
        assert_eq!(
            swatch(cut.asset(unused).expect("audio").kind()),
            theme::AUDIO
        );
    }
}
