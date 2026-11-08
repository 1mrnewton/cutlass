//! A video clip's Retouch page: face and body touch-ups, laid out as the
//! reference lays them out.
//!
//! None of it edits the clip yet. The chips switch what is shown, so the
//! whole page can be seen; every control is dimmed and marked "Later".

use egui::{Color32, RichText, Sense, Stroke, Ui, Vec2};

use crate::{tokens, ui};

use super::audio::{later_section, name_tile};
use super::inspector::{
    InspectorState, LATER, ROW_GAP, VALUE_WIDTH, check_box, none_tile, row_label, value_box,
};

/// What the page touches up.
const PARTS: [&str; 5] = ["Face", "Body", "Hair", "Face presets", "Body presets"];

/// Skin: each a strength from nothing.
const SKIN: [&str; 9] = [
    "Clear",
    "Smooth",
    "Whitening",
    "Clear blemishes",
    "Dewrinkle",
    "Even",
    "Plump",
    "Smile lines",
    "Dark circles",
];

/// Face shape, and whether the slider starts in the middle.
const FACE: [(&str, bool); 12] = [
    ("Small face", false),
    ("Slim", true),
    ("Width", true),
    ("Jaw", false),
    ("Cheekbones", false),
    ("Shorten", false),
    ("V shape", false),
    ("Chin length", true),
    ("Lower", true),
    ("Middle", true),
    ("Upper", true),
    ("Hairline", true),
];

const FEATURES: [&str; 4] = ["Eyes", "Nose", "Mouth", "Eyebrow"];

/// Each feature's sliders, and whether each starts in the middle.
const FEATURE_SLIDERS: [&[(&str, bool)]; 4] = [
    &[
        ("Size", false),
        ("Bright eye", false),
        ("Distance", true),
        ("Inner", false),
        ("Position", true),
        ("Slant", true),
    ],
    &[
        ("Size", false),
        ("Width", true),
        ("Bridge", true),
        ("Tip", true),
        ("Lift", true),
    ],
    &[
        ("Size", true),
        ("Width", true),
        ("Smile", false),
        ("Lips", true),
    ],
    &[
        ("Thickness", true),
        ("Position", true),
        ("Tilt", true),
        ("Distance", true),
    ],
];

const MAKEUP: [&str; 11] = [
    "Looks",
    "Eyelashes",
    "Eye shadow",
    "Eyeliner",
    "Eyebrow",
    "Lipstick",
    "Blush",
    "Contour",
    "Highlights",
    "Eye color",
    "Freckles",
];

/// Whole looks, by our own names.
const LOOKS: [&str; 14] = [
    "Natural",
    "Sun-kissed",
    "Soft pink",
    "Clean",
    "Fresh",
    "Rosy",
    "Bold lip",
    "Dewy",
    "Warm glow",
    "Cool tone",
    "Smoky",
    "Bright eyes",
    "Classic",
    "Freckled",
];

const MANUAL: [&str; 3] = ["Face", "Smooth skin", "Freckle and acne"];

/// Skin tones, light to deep.
const TONES: [Color32; 9] = [
    Color32::from_rgb(0xb8, 0x9a, 0x8e),
    Color32::from_rgb(0xb5, 0xa0, 0x88),
    Color32::from_rgb(0xa5, 0x7b, 0x55),
    Color32::from_rgb(0x8a, 0x5e, 0x3c),
    Color32::from_rgb(0x83, 0x5c, 0x45),
    Color32::from_rgb(0x7d, 0x58, 0x46),
    Color32::from_rgb(0x74, 0x4a, 0x32),
    Color32::from_rgb(0x7a, 0x55, 0x42),
    Color32::from_rgb(0x55, 0x40, 0x36),
];

/// Body shape, and whether the slider starts in the middle.
const BODY: [(&str, bool); 8] = [
    ("Slim", false),
    ("Long legs", false),
    ("Small waist", false),
    ("Shoulders", true),
    ("Small head", false),
    ("Chest", true),
    ("Hips", true),
    ("Swan neck", false),
];

const HAIR_COLORS: [&str; 10] = [
    "Black", "Brown", "Chestnut", "Auburn", "Blonde", "Ash", "Silver", "Rose", "Blue", "Green",
];

pub(super) fn retouch_page(ui: &mut Ui, state: &mut InspectorState) {
    ui.add_space(8.0);
    let part = chips(ui, state, "retouch-part", &PARTS, true);
    ui.add_space(ROW_GAP);
    match part {
        0 => face(ui, state),
        1 => {
            later_section(
                ui,
                state,
                "retouch-body",
                "Body",
                Some("Reshape the person in the clip."),
                Some(&mut |ui: &mut Ui| sliders(ui, &BODY)),
            );
        }
        2 => {
            later_section(
                ui,
                state,
                "retouch-hair",
                "Hair color",
                Some("Dye the person's hair."),
                Some(&mut |ui: &mut Ui| tiles(ui, &HAIR_COLORS)),
            );
        }
        _ => {
            later_section(
                ui,
                state,
                "retouch-presets",
                if part == 3 {
                    "Face presets"
                } else {
                    "Body presets"
                },
                Some("Touch-ups kept with Save as preset show here."),
                None,
            );
        }
    }
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.add_enabled(false, ui::Button::new("Save as preset").text_size(12.0))
            .on_disabled_hover_text(LATER);
    });
}

fn face(ui: &mut Ui, state: &mut InspectorState) {
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            row_label(ui, "Faces");
            ui.add(ui::Button::new("Single face").text_size(12.0));
        });
    })
    .response
    .on_disabled_hover_text(LATER);
    ui.add_space(ROW_GAP);
    later_section(
        ui,
        state,
        "retouch-auto",
        "Auto styles",
        Some("One-click touch-ups picked for the face."),
        None,
    );
    later_section(
        ui,
        state,
        "retouch-skin",
        "Skin",
        None,
        Some(&mut |ui: &mut Ui| {
            for name in SKIN {
                dim_slider(ui, name, false);
            }
            ui.label(RichText::new("Skin tone").size(12.0).color(tokens::MUTED));
            ui.add_space(4.0);
            skin_tones(ui);
        }),
    );
    later_section(
        ui,
        state,
        "retouch-face",
        "Face",
        None,
        Some(&mut |ui: &mut Ui| sliders(ui, &FACE)),
    );
    let feature = state.chips.get("retouch-feature").copied().unwrap_or(0);
    later_section(
        ui,
        state,
        "retouch-features",
        "Features",
        None,
        Some(&mut |ui: &mut Ui| {
            chip_row(ui, &FEATURES, feature);
            sliders(ui, FEATURE_SLIDERS[feature.min(FEATURES.len() - 1)]);
        }),
    );
    let makeup = state.chips.get("retouch-makeup").copied().unwrap_or(0);
    later_section(
        ui,
        state,
        "retouch-makeup",
        "Makeup",
        None,
        Some(&mut |ui: &mut Ui| {
            chip_row(ui, &MAKEUP, makeup);
            tiles(ui, &LOOKS);
        }),
    );
    let manual = state.chips.get("retouch-manual").copied().unwrap_or(0);
    later_section(
        ui,
        state,
        "retouch-manual",
        "Manual",
        Some("Paint a touch-up onto the face by hand."),
        Some(&mut |ui: &mut Ui| {
            chip_row(ui, &MANUAL, manual);
            ui.horizontal(|ui| {
                row_label(ui, "Brush");
                ui.add(ui::Button::icon("draw").min_size(Vec2::splat(24.0)));
            });
            ui.add_space(ROW_GAP);
            dim_slider(ui, "Size", false);
            dim_slider(ui, "Intensity", false);
            ui.horizontal(|ui| {
                row_label(ui, "Protect face");
                check_box(ui, false, "Protect face");
            });
        }),
    );
}

/// A row of chips that can be picked; the pick is kept under `key`. With
/// `live` the chips are on even while the rest is Later.
fn chips(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &'static str,
    labels: &[&str],
    live: bool,
) -> usize {
    let current = state.chips.get(key).copied().unwrap_or(0);
    let mut picked = current;
    ui.add_enabled_ui(live, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            for (index, label) in labels.iter().enumerate() {
                if ui
                    .add(ui::Choice::new(
                        index == current,
                        RichText::new(*label).size(12.0),
                    ))
                    .clicked()
                {
                    picked = index;
                }
            }
        });
    });
    state.chips.insert(key, picked);
    picked
}

/// The chips inside a Later section: drawn as they will be, the first on.
fn chip_row(ui: &mut Ui, labels: &[&str], current: usize) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        for (index, label) in labels.iter().enumerate() {
            ui.add(ui::Choice::new(
                index == current,
                RichText::new(*label).size(12.0),
            ));
        }
    });
    ui.add_space(ROW_GAP);
}

fn sliders(ui: &mut Ui, rows: &[(&str, bool)]) {
    for (name, centered) in rows {
        dim_slider(ui, name, *centered);
    }
}

/// A slider at zero: at the left end, or in the middle for a value that
/// goes both ways.
fn dim_slider(ui: &mut Ui, label: &str, centered: bool) {
    ui.horizontal(|ui| {
        row_label(ui, label);
        let track = (ui.available_width() - VALUE_WIDTH - 8.0).max(40.0);
        let mut value = 0.0;
        let range = if centered { -50.0..=50.0 } else { 0.0..=100.0 };
        ui.add(ui::Slider::new(&mut value, range).name(label).width(track));
        value_box(ui, "0");
    });
    ui.add_space(ROW_GAP);
}

/// None, then a tile for each name.
fn tiles(ui: &mut Ui, names: &[&str]) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(8.0);
        none_tile(ui, true);
        for name in names {
            name_tile(ui, name, false);
        }
    });
    ui.add_space(ROW_GAP);
}

/// No tone, then a swatch for each.
fn skin_tones(ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(8.0);
        let size = Vec2::splat(26.0);
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let stroke = Stroke::new(1.5, tokens::MUTED);
        let radius = rect.width() / 2.0 - 1.0;
        ui.painter().circle_stroke(rect.center(), radius, stroke);
        let reach = Vec2::splat(radius * std::f32::consts::FRAC_1_SQRT_2);
        ui.painter()
            .line_segment([rect.center() - reach, rect.center() + reach], stroke);
        for tone in TONES {
            let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
            ui.painter()
                .circle_filled(rect.center(), rect.width() / 2.0, tone.gamma_multiply(0.6));
        }
    });
    ui.add_space(ROW_GAP);
}
