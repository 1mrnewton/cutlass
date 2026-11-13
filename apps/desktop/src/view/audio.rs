//! The audio clip's inspector: Basic, Voice changer and Speed.
//!
//! Volume, fades, speed and pitch already edit the clip. Every other
//! control is laid out where it will live and drawn dimmed with a "Later"
//! mark, so the panel can be filled in one feature at a time.

use egui::{Align, Align2, CornerRadius, FontId, Layout, RichText, Sense, Stroke, Ui, Vec2};

use crate::shell::{Inspection, Shell};
use crate::{tokens, ui};

use super::inspector::{
    BOX, CATALOG_TILE, InspectorState, LATER, ROW_GAP, ROW_HEIGHT, body, check_box, chevron,
    divider, later_badge, pages, row_label, sound_section, tabs, tile_frame, value_box,
};

/// The parts Separate audio can pull out of a clip.
const STEMS: [&str; 7] = [
    "Vocals",
    "Instruments",
    "Drums",
    "Guitar",
    "Bass",
    "Strings",
    "Others",
];

/// Voice filters: a sound put over the voice.
const FILTERS: &[&str] = &[
    "Deep",
    "High",
    "Echo",
    "Bass boost",
    "Full voice",
    "Studio",
    "Radio",
    "Phone",
    "Megaphone",
    "Hall",
    "Church",
    "Underwater",
    "Whisper",
    "Robot",
    "Glitch",
    "Low battery",
    "Left ear",
    "Right ear",
    "No bass",
    "Crisp",
    "Bleep",
];

/// Voice characters: the voice made into someone else's.
const CHARACTERS: &[&str] = &[
    "Chipmunk",
    "Giant",
    "Elf",
    "Robot",
    "Alien",
    "Monster",
    "Announcer",
    "Narrator",
    "Child",
    "Elder",
    "Villain",
    "Ghost",
    "Doll",
    "Kitten",
];

pub(super) fn audio_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let tab = state.tab;
    tabs(ui, &["Basic", "Voice changer", "Speed"], &mut state.tab);
    if tab != state.tab {
        state.page = 0;
    }
    match state.tab {
        0 => basic_tab(shell, ui, state, inspection),
        1 => voice_tab(ui, state),
        _ => super::speed::audio_speed_tab(shell, ui, state, inspection),
    }
}

fn basic_tab(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    sound_section(shell, ui, state, inspection);
    later_section(
        ui,
        state,
        "normalize",
        "Normalize loudness",
        Some("Bring the clip, or every selected clip, to one loudness."),
        None,
    );
    later_section(
        ui,
        state,
        "enhance-voice",
        "Enhance voice",
        Some("Make speech clearer and fuller."),
        Some(&mut |ui: &mut Ui| later_slider(ui, "Intensity", 75.0)),
    );
    later_section(
        ui,
        state,
        "reduce-noise",
        "Reduce noise",
        Some("Take out hum, hiss and background noise."),
        None,
    );
    later_section(
        ui,
        state,
        "separate",
        "Separate audio",
        Some("Pull the voice, or any instrument, out onto its own clip."),
        Some(&mut |ui: &mut Ui| {
            stem_grid(ui);
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add(ui::Button::new("Separate").text_size(12.0));
                    check_box(ui, true, "Keep remaining audio");
                    ui.label(RichText::new("Keep remaining audio").size(12.0));
                });
            });
            ui.add_space(ROW_GAP);
        }),
    );
    later_section(
        ui,
        state,
        "fill-channel",
        "Fill channel",
        Some("Play a sound heard in one ear in both."),
        None,
    );
}

fn voice_tab(ui: &mut Ui, state: &mut InspectorState) {
    // Speech to song, a generative voice, waits for the next version.
    pages(ui, &["Voice filters", "Voice characters"], &mut state.page);
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            check_box(ui, false, "Apply to all");
            ui.label(RichText::new("Apply to all").size(12.0));
            later_badge(ui);
        });
        ui.add_space(ROW_GAP);
        let names = if state.page == 0 { FILTERS } else { CHARACTERS };
        search_box(ui, "Search voices");
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("The voices are laid out; none changes the sound yet.")
                    .size(12.0)
                    .color(tokens::MUTED),
            );
            later_badge(ui);
        });
        ui.add_space(ROW_GAP);
        voice_tiles(ui, names);
    })
    .response
    .on_disabled_hover_text(LATER);
}

/// A section for a feature with no edit yet: a dimmed on box, its name,
/// a fold mark and a "Later" mark, a line on what it does, and, when it is
/// open, its controls dimmed. These start open; the name folds them.
pub(super) fn later_section(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    about: Option<&str>,
    controls: Option<&mut dyn FnMut(&mut Ui)>,
) {
    let shut = format!("{key} shut");
    let open = !state.open.contains(&shut);
    ui.horizontal(|ui| {
        ui.set_min_height(40.0);
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_enabled_ui(false, |ui| check_box(ui, false, label));
        let name = ui.add(
            egui::Label::new(RichText::new(label).size(13.0).color(tokens::TEXT))
                .selectable(false)
                .sense(Sense::click()),
        );
        let name = if controls.is_some() {
            name.union(chevron(ui, open))
        } else {
            name
        };
        later_badge(ui);
        if controls.is_some() && name.clicked() {
            if open {
                state.open.insert(shut.clone());
            } else {
                state.open.remove(&shut);
            }
        }
    })
    .response
    .on_hover_text(LATER);
    if let Some(about) = about {
        ui.label(RichText::new(about).size(12.0).color(tokens::MUTED));
        ui.add_space(ROW_GAP);
    }
    if open && let Some(controls) = controls {
        body(ui, |ui| {
            ui.add_enabled_ui(false, |ui| controls(ui))
                .response
                .on_disabled_hover_text(LATER);
        });
    }
    divider(ui);
}

/// A dimmed slider row with a fixed value.
pub(super) fn later_slider(ui: &mut Ui, label: &str, value: f64) {
    ui.horizontal(|ui| {
        row_label(ui, label);
        let track = (ui.available_width() - super::inspector::VALUE_WIDTH - 8.0).max(40.0);
        let mut value = value;
        ui.add(
            ui::Slider::new(&mut value, 0.0..=100.0)
                .name(label)
                .width(track),
        );
        value_box(ui, &format!("{value:.0}"));
    });
    ui.add_space(ROW_GAP);
}

/// A control with no edit yet inside a working section: its name with a
/// "Later" mark, then a dimmed slider at `value`.
pub(super) fn later_value(ui: &mut Ui, label: &str, value: f64) {
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).size(12.0));
            later_badge(ui);
        });
        later_slider(ui, "", value);
    })
    .response
    .on_disabled_hover_text(LATER);
}

/// A drop-down look: a box with its text and a down mark.
pub(super) fn choice_row(ui: &mut Ui, label: Option<&str>, text: &str) {
    ui.horizontal(|ui| {
        if let Some(label) = label {
            row_label(ui, label);
        }
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), ROW_HEIGHT + 4.0),
            Sense::hover(),
        );
        ui.painter().rect_filled(rect, CornerRadius::same(8), BOX);
        ui.painter().text(
            rect.left_center() + Vec2::new(10.0, 0.0),
            Align2::LEFT_CENTER,
            text,
            FontId::proportional(12.0),
            tokens::FAINT,
        );
        let mark = rect.right_center() - Vec2::new(14.0, 0.0);
        let stroke = Stroke::new(1.5, tokens::FAINT);
        ui.painter().line_segment(
            [mark + Vec2::new(-4.0, -2.0), mark + Vec2::new(0.0, 2.0)],
            stroke,
        );
        ui.painter().line_segment(
            [mark + Vec2::new(0.0, 2.0), mark + Vec2::new(4.0, -2.0)],
            stroke,
        );
    });
    ui.add_space(ROW_GAP);
}

pub(super) fn right_button(ui: &mut Ui, label: &str) {
    ui.horizontal(|ui| {
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(ui::Button::new(label).text_size(12.0));
        });
    });
    ui.add_space(ROW_GAP);
}

/// The parts to separate, two to a row.
fn stem_grid(ui: &mut Ui) {
    let gap = 8.0;
    let width = ((ui.available_width() - gap) / 2.0).max(40.0);
    for pair in STEMS.chunks(2) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for stem in pair {
                ui.add(
                    ui::Button::new(*stem)
                        .text_size(12.0)
                        .min_size(Vec2::new(width, 30.0)),
                );
            }
        });
        ui.add_space(gap);
    }
}

fn search_box(ui: &mut Ui, hint: &str) {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), ROW_HEIGHT + 2.0),
        Sense::hover(),
    );
    ui.painter().rect_filled(rect, CornerRadius::same(8), BOX);
    let lens = rect.left_center() + Vec2::new(16.0, -1.0);
    let stroke = Stroke::new(1.3, tokens::FAINT);
    ui.painter().circle_stroke(lens, 4.5, stroke);
    ui.painter()
        .line_segment([lens + Vec2::splat(3.2), lens + Vec2::splat(6.5)], stroke);
    ui.painter().text(
        rect.left_center() + Vec2::new(30.0, 0.0),
        Align2::LEFT_CENTER,
        hint,
        FontId::proportional(12.0),
        tokens::FAINT,
    );
    ui.add_space(ROW_GAP * 2.0);
}

/// A grid of named tiles, led by "Original", the clip as it is, worn.
fn voice_tiles(ui: &mut Ui, names: &[&str]) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(8.0);
        name_tile(ui, "Original", true);
        for name in names {
            name_tile(ui, name, false);
        }
    });
    ui.add_space(ROW_GAP);
}

pub(super) fn name_tile(ui: &mut Ui, name: &str, on: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(CATALOG_TILE), Sense::click());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, false, name));
    tile_frame(ui, rect, &response, on);
    // Names break between words only, two lines at most; the type shrinks
    // until the longest word fits on a line.
    let room = rect.width() - 6.0;
    let size = [10.5, 9.5, 8.5]
        .into_iter()
        .find(|size| {
            name.split_whitespace().all(|word| {
                ui.painter()
                    .layout_no_wrap(word.to_owned(), FontId::proportional(*size), tokens::TEXT)
                    .size()
                    .x
                    <= room
            })
        })
        .unwrap_or(8.5);
    let mut job = egui::text::LayoutJob::simple(
        name.to_owned(),
        FontId::proportional(size),
        tokens::TEXT,
        room,
    );
    job.wrap.break_anywhere = false;
    job.wrap.max_rows = 2;
    job.halign = Align::Center;
    let galley = ui.painter().layout_job(job);
    let top = rect.center() - Vec2::new(0.0, galley.size().y / 2.0);
    ui.painter()
        .with_clip_rect(rect.shrink(2.0))
        .galley(top, galley, tokens::TEXT);
    response
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_lists_have_no_repeats() {
        for list in [super::FILTERS, super::CHARACTERS] {
            let mut seen = std::collections::HashSet::new();
            for name in list {
                assert!(seen.insert(*name), "{name} twice");
            }
        }
    }
}
