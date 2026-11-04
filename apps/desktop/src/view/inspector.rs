//! The inspector's clip panels, laid out as the desktop prototype lays them
//! out: tabs, then a page switch, then sections that fold open, each with an
//! on switch. The selected clip's kind picks the panel.
//!
//! Every control that has an edit sends one request through the shell, so
//! each change is one undo step; a slider sends when the pointer lets go. A
//! control the prototype shows with no edit in Rust yet is drawn dimmed and
//! marked "Later". A clip on a locked row shows its values without edits.

use std::collections::{HashMap, HashSet};

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Rect, RichText, Sense, Slider, Stroke,
    TextEdit, Ui, Vec2,
};

use crate::shell::{
    Inspection, LetterCase, Panel, SILENT_DB, Shell, ShellError, TextLook, volume_db,
};
use crate::{icons, tokens};

use super::ViewState;

/// The active tab color, and an on switch.
const ACCENT: Color32 = Color32::from_rgb(0x2a, 0xd4, 0xe0);
const PAGES: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
const PAGE_ON: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
const BOX: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
const LABEL_WIDTH: f32 = 72.0;
const VALUE_WIDTH: f32 = 56.0;
const PAIR_BOX: f32 = 84.0;
const LATER: &str = "Later: Cutlass has no edit for this yet";

/// Which tab, page, and sections are open. They reset when another clip
/// is picked, as the prototype's panels do.
#[derive(Default)]
pub(super) struct InspectorState {
    clip: String,
    tab: usize,
    page: usize,
    open: HashSet<String>,
    /// Typed text in a number or word field while it has focus.
    fields: HashMap<String, String>,
    /// A slider or color being dragged; it is sent when the pointer lets go.
    live: HashMap<String, f64>,
    colors: HashMap<String, Color32>,
}

pub(super) fn clip_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    inspection: &Inspection,
    _width: f32,
) {
    let panel = shell.panel().unwrap_or(Panel::Other);
    let state = &mut view.inspector;
    if state.clip != inspection.segment_id.as_str() {
        *state = InspectorState {
            clip: inspection.segment_id.as_str().to_owned(),
            open: ["position", "sound"]
                .into_iter()
                .chain(matches!(panel, Panel::Text | Panel::Caption).then_some("blend"))
                .map(str::to_owned)
                .collect(),
            ..InspectorState::default()
        };
    }
    if inspection.locked {
        note(ui, "This clip's row is locked.");
    }
    ui.add_enabled_ui(!inspection.locked, |ui| match panel {
        Panel::Main { still } | Panel::Overlay { still } => {
            video_panel(shell, ui, state, inspection, still);
        }
        Panel::Text | Panel::Caption => {
            text_panel(shell, ui, state, inspection, panel == Panel::Caption);
        }
        Panel::Audio => audio_panel(shell, ui, state, inspection),
        Panel::Sticker => sticker_panel(shell, ui, state, inspection),
        Panel::Effect => effect_panel(shell, ui, state, "Effect", false),
        Panel::Filter => effect_panel(shell, ui, state, "Filter", true),
        Panel::Adjust => adjust_panel(ui, state),
        Panel::Other => {
            tabs(ui, &["Clip"], &mut 0);
            note(ui, &inspection.material);
        }
    });
}

// Panels.

fn video_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    still: bool,
) {
    let tab = state.tab;
    tabs(
        ui,
        &[
            "Video",
            "Audio",
            "Speed",
            "Animation",
            "Adjust",
            "AI stylize",
        ],
        &mut state.tab,
    );
    if tab != state.tab {
        state.page = 0;
    }
    match state.tab {
        0 => {
            pages(ui, &["Basic", "Cutout", "Mask", "Enhance"], &mut state.page);
            if state.page != 0 {
                later_note(ui, ["", "Cutout", "Mask", "Enhance"][state.page]);
                return;
            }
            position_size(shell, ui, state, inspection, "Rotate");
            blend(shell, ui, state, inspection, true);
            for label in [
                "Stabilize",
                "Enhance quality",
                "Noise reduction",
                "Optical flow",
                "AI remove",
                "AI expand",
                "AI remix",
                "Lip sync",
                "Relight",
                "Auto reframe",
                "Removing video flickers",
                "AI movement",
                "Camera tracking",
                "Motion blur",
            ] {
                later_row(ui, label);
            }
            canvas(shell, ui, state, inspection);
        }
        1 => {
            pages(ui, &["Basic", "Voice changer"], &mut state.page);
            if still {
                note(ui, "A photo has no sound.");
                return;
            }
            if state.page == 1 {
                later_note(ui, "Voice changer");
                return;
            }
            sound_section(shell, ui, state, inspection);
            for label in [
                "Loudness normalization",
                "Enhance voice",
                "Video translator",
                "Reduce noise",
                "Isolate voice",
                "Fill channel",
            ] {
                later_row(ui, label);
            }
        }
        2 if still => note(ui, "A photo has no speed."),
        2 => speed_row(shell, ui, state, inspection),
        3 => animation_tab(ui, state, &["In", "Out", "Combo"], VIDEO_MOTIONS),
        4 => adjust_tab(ui, state),
        _ => later_note(ui, "AI stylize"),
    }
}

fn text_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    caption: bool,
) {
    let tab = state.tab;
    tabs(
        ui,
        &["Text", "Animation", "Tracking", "Text to speech"],
        &mut state.tab,
    );
    if tab != state.tab {
        state.page = 0;
    }
    match state.tab {
        0 => {
            pages(ui, &["Basic", "Bubble", "Effects"], &mut state.page);
            match state.page {
                0 => {
                    text_basic(shell, ui, state, inspection, caption);
                    footer(ui, &["Save as preset"]);
                }
                1 => later_tiles(ui, &["None", "Round", "Box", "Thought"]),
                _ => later_tiles(ui, &["None", "Shadow", "Glow", "Outline"]),
            }
        }
        1 => animation_tab(ui, state, &["In", "Out", "Loop"], TEXT_MOTIONS),
        2 => {
            later_tiles(ui, &["Face", "Body", "Hand", "Custom"]);
            later_button(ui, "Start tracking");
        }
        _ => {
            later_note(ui, "Text to speech");
            later_tiles(
                ui,
                &[
                    "Warm", "Bright", "Even", "Clear", "Light", "Low", "Soft", "Steady", "Quick",
                    "Dry",
                ],
            );
        }
    }
}

fn audio_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    tabs(ui, &["Audio", "Speed"], &mut state.tab);
    if state.tab == 1 {
        speed_row(shell, ui, state, inspection);
        return;
    }
    body(ui, |ui| {
        sound_sliders(shell, ui, state, inspection);
        later_slider(ui, "Denoise", 0.0, "0");
    });
    later_button(ui, "Change voice");
}

fn sticker_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    tabs(ui, &["Sticker", "Animation"], &mut state.tab);
    if state.tab == 1 {
        animation_tab(
            ui,
            state,
            &["In", "Out", "Loop"],
            &[
                &["Pop", "Spin", "Fade"],
                &["Pop", "Spin", "Fade"],
                &["Pop", "Spin", "Fade"],
            ],
        );
        return;
    }
    body(ui, |ui| opacity_slider(shell, ui, state, inspection));
}

fn effect_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    title: &str,
    filter: bool,
) {
    tabs(ui, &[title], &mut 0);
    note(ui, &shell.effect_name().unwrap_or_default());
    body(ui, |ui| {
        if let Some(stored) = shell.effect_strength() {
            let amount = (stored * 100.0).round();
            if let Some(value) = slider(ui, state, "amount", "Amount", amount, 0.0..=100.0, |v| {
                format!("{v:.0}")
            }) {
                send(shell, |shell| shell.set_effect_strength(value / 100.0));
            }
        } else {
            later_slider(ui, "Amount", 0.0, "0");
        }
        if filter {
            later_slider(ui, "Temp", 0.0, "0");
        }
    });
}

fn adjust_panel(ui: &mut Ui, state: &mut InspectorState) {
    tabs(ui, &["Adjust"], &mut 0);
    pages(
        ui,
        &["Basic", "HSL", "Curves", "Color wheel", "Mask"],
        &mut state.page,
    );
    if state.page != 0 {
        later_note(ui, ["", "HSL", "Curves", "Color wheel", "Mask"][state.page]);
        return;
    }
    body(ui, |ui| {
        for label in ["Temp", "Exposure", "Contrast"] {
            later_slider(ui, label, 0.0, "0");
        }
    });
}

// Sections shared by panels.

fn position_size(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    rotate_label: &str,
) {
    let header = section(ui, state, "position", "Position & Size", None, true);
    if header.reset {
        send(shell, Shell::reset_placement);
    }
    if !header.open {
        return;
    }
    let (half_width, half_height) = shell
        .draft()
        .map(|draft| {
            (
                f64::from(draft.canvas_config.width) / 2.0,
                f64::from(draft.canvas_config.height) / 2.0,
            )
        })
        .unwrap_or((960.0, 540.0));
    body(ui, |ui| {
        let scale = (inspection.scale_x * 100.0).round();
        if let Some(value) = slider(ui, state, "scale", "Scale", scale, 1.0..=500.0, |v| {
            format!("{v:.0}%")
        }) {
            send(shell, |shell| shell.set_scale(value / 100.0));
        }
        switch_row(ui, "Uniform scale", true, false).on_disabled_hover_text(LATER);
        // The prototype counts canvas pixels from the middle, with Y down.
        let x = (inspection.transform_x * half_width).round();
        let y = (-inspection.transform_y * half_height).round();
        let (new_x, new_y) = pair_row(ui, state, "Position", ("X", x), ("Y", y));
        if let Some(value) = new_x {
            send(shell, |shell| {
                shell.set_position(value / half_width, inspection.transform_y)
            });
        }
        if let Some(value) = new_y {
            send(shell, |shell| {
                shell.set_position(inspection.transform_x, -value / half_height)
            });
        }
        let rotation = inspection.rotation;
        if let (Some(value), _) = pair_row(ui, state, rotate_label, ("°", rotation), ("", f64::NAN))
        {
            send(shell, |shell| shell.set_rotation(value));
        }
        if rotate_label == "Rotate" {
            use cutlass_editor::StageAlign;
            let aims = [
                ("alignLeft", "Align left", Some(StageAlign::Left)),
                ("alignCenter", "Align center", Some(StageAlign::Center)),
                ("alignRight", "Align right", Some(StageAlign::Right)),
                ("alignTop", "Align top", Some(StageAlign::Top)),
                ("alignMiddle", "Align middle", Some(StageAlign::Middle)),
                ("alignBottom", "Align bottom", Some(StageAlign::Bottom)),
            ];
            if let Some(align) = icon_bar(ui, &aims) {
                send(shell, |shell| shell.align_selected(align));
            }
        }
        // Mirror, side to side, is on the timeline toolbar.
        if shell.mirror_enabled() {
            let flipped = shell.flipped();
            if switch_row(ui, "Flip upside down", flipped, true).clicked() {
                send(shell, Shell::toggle_vertical_flip);
            }
        }
    });
}

fn blend(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    mode: bool,
) {
    let header = section(ui, state, "blend", "Blend", Some((true, false)), false);
    if !header.open {
        return;
    }
    body(ui, |ui| {
        if mode {
            ui.add_enabled_ui(false, |ui| {
                label_row(ui, "Mode", |ui| {
                    egui::ComboBox::from_id_salt("blend-mode")
                        .selected_text("Normal")
                        .show_ui(ui, |_| {});
                });
            })
            .response
            .on_disabled_hover_text(LATER);
        }
        opacity_slider(shell, ui, state, inspection);
    });
}

fn opacity_slider(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let opacity = (inspection.opacity * 100.0).round();
    if let Some(value) = slider(ui, state, "opacity", "Opacity", opacity, 0.0..=100.0, |v| {
        format!("{v:.0}%")
    }) {
        send(shell, |shell| shell.set_opacity(value / 100.0));
    }
}

fn canvas(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    if !shell.crop_enabled() {
        later_row(ui, "Canvas");
        return;
    }
    let on = !inspection.letterbox.is_empty();
    let header = section(ui, state, "canvas", "Canvas", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#000000" };
        send(shell, |shell| shell.set_letterbox(color));
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) =
                color_row(ui, state, "canvas-color", "Color", &inspection.letterbox)
            {
                send(shell, |shell| shell.set_letterbox(&color));
            }
            later_button(ui, "Apply to all");
        });
    }
}

fn sound_section(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let header = section(ui, state, "sound", "Basic", Some((true, false)), true);
    if header.reset {
        send(shell, Shell::reset_sound);
    }
    if header.open {
        body(ui, |ui| sound_sliders(shell, ui, state, inspection));
    }
}

fn sound_sliders(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let db = (volume_db(inspection.volume) * 10.0).round() / 10.0;
    if let Some(value) = slider(ui, state, "volume", "Volume", db, SILENT_DB..=12.0, |v| {
        format!("{v:.1}dB")
    }) {
        send(shell, |shell| shell.set_volume_db(value));
    }
    if !shell.fade_enabled() {
        return;
    }
    let seconds = |micros: i64| (micros as f64 / 100_000.0).round() / 10.0;
    let fade_in = seconds(inspection.fade_in);
    if let Some(value) = slider(ui, state, "fade-in", "Fade in", fade_in, 0.0..=10.0, |v| {
        format!("{v:.1}s")
    }) {
        send(shell, |shell| shell.set_fade_in(value));
    }
    let fade_out = seconds(inspection.fade_out);
    if let Some(value) = slider(
        ui,
        state,
        "fade-out",
        "Fade out",
        fade_out,
        0.0..=10.0,
        |v| format!("{v:.1}s"),
    ) {
        send(shell, |shell| shell.set_fade_out(value));
    }
}

fn speed_row(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    body(ui, |ui| {
        let speed = (inspection.speed * 10.0).round() / 10.0;
        if let Some(value) = slider(ui, state, "speed", "Speed", speed, 0.1..=4.0, |v| {
            format!("{v:.1}×")
        }) {
            send(shell, |shell| shell.set_speed(value));
        }
    });
}

const VIDEO_MOTIONS: &[&[&str]] = &[
    &[
        "Fade in",
        "Slide up",
        "Slide down",
        "Slide left",
        "Slide right",
        "Zoom in",
        "Pop",
        "Spin",
        "Wipe",
        "Blur in",
        "Rise",
        "Flip",
    ],
    &[
        "Fade out",
        "Slide up",
        "Slide down",
        "Slide left",
        "Slide right",
        "Zoom out",
        "Shrink",
        "Spin",
        "Wipe",
        "Blur out",
        "Fall",
        "Flip",
    ],
    &[
        "Shake", "Pulse", "Float", "Bounce", "Drift", "Flicker", "Swing", "Breathe",
    ],
];

const TEXT_MOTIONS: &[&[&str]] = &[
    &[
        "Fade", "Rise", "Drop", "Slide", "Pop", "Type on", "Blur", "Spin", "Bounce", "Unfold",
    ],
    &[
        "Fade", "Rise", "Drop", "Slide", "Shrink", "Type off", "Blur", "Spin",
    ],
    &["Pulse", "Flicker", "Float", "Shake", "Swing", "Breathe"],
];

/// The In, Out, and Combo or Loop slots, each a grid of animations that
/// starts with None. Nothing stores an animation yet.
fn animation_tab(ui: &mut Ui, state: &mut InspectorState, slots: &[&str], motions: &[&[&str]]) {
    pages(ui, slots, &mut state.page);
    later_note(ui, "Animations");
    ui.add_enabled_ui(false, |ui| {
        let mut query = String::new();
        ui.add(
            TextEdit::singleline(&mut query)
                .hint_text("Search animations")
                .desired_width(ui.available_width() - 28.0),
        );
    });
    let mut names = vec!["None"];
    names.extend_from_slice(motions.get(state.page).copied().unwrap_or_default());
    later_tiles(ui, &names);
    later_slider(ui, "Duration", 2.0, "2.0s");
}

/// The Adjust tab of a picture: the color rows, each to come.
fn adjust_tab(ui: &mut Ui, state: &mut InspectorState) {
    pages(
        ui,
        &["Basic", "HSL", "Curves", "Color wheel", "Mask"],
        &mut state.page,
    );
    if state.page != 0 {
        later_note(ui, ["", "HSL", "Curves", "Color wheel", "Mask"][state.page]);
        return;
    }
    for label in [
        "Auto adjust",
        "Color match",
        "Color correction",
        "LUT",
        "Adjust",
    ] {
        later_row(ui, label);
    }
    footer(ui, &["Apply to all", "Save as preset"]);
}

// The text panel's Basic page.

fn text_basic(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    caption: bool,
) {
    body(ui, |ui| {
        if caption && !inspection.words.text.is_empty() {
            caption_words(shell, ui, state, inspection);
        } else {
            words_box(shell, ui, state);
        }
        font_row(shell, ui, inspection);
        let size = inspection.text_size.round();
        if let Some(value) = slider(
            ui,
            state,
            "font-size",
            "Font size",
            size,
            1.0..=512.0,
            |v| format!("{v:.0}"),
        ) {
            send(shell, |shell| shell.set_text_size(value));
        }
        label_row(ui, "Style", |ui| {
            ui.add_enabled_ui(false, |ui| {
                for (icon, tip) in [
                    ("bold", "Bold"),
                    ("underline", "Underline"),
                    ("italic", "Italic"),
                ] {
                    icon_toggle(ui, icon, tip, false);
                }
            })
            .response
            .on_disabled_hover_text(LATER);
        });
        label_row(ui, "Case", |ui| {
            for (label, tip, case) in [
                ("TT", "Upper case", LetterCase::Upper),
                ("tt", "Lower case", LetterCase::Lower),
                ("Tt", "Title case", LetterCase::Title),
            ] {
                if ui
                    .add(egui::Button::new(RichText::new(label).size(12.0)).fill(BOX))
                    .on_hover_text(tip)
                    .clicked()
                {
                    send(shell, |shell| shell.set_text_case(case));
                }
            }
        });
        if let Some(color) = color_row(ui, state, "text-color", "Color", &inspection.text_color) {
            send(shell, |shell| shell.set_text_color(&color));
        }
        spacing_row(shell, ui, state, inspection);
        let (width, _) = pair_row(
            ui,
            state,
            "Line width",
            ("px", inspection.line_width.round()),
            ("", f64::NAN),
        );
        if let Some(value) = width {
            send(shell, |shell| shell.set_line_width(value.max(0.0)));
        }
        alignment_row(shell, ui, inspection);
        preset_styles(shell, ui);
    });
    position_size(shell, ui, state, inspection, "Plane rotation");
    blend(shell, ui, state, inspection, false);
    if let Some(look) = shell.text_look() {
        text_parts(shell, ui, state, &look);
    }
}

fn words_box(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let stored = shell.text_words().unwrap_or_default();
    let text = state.fields.entry("words".to_owned()).or_default();
    let response = ui.add(
        TextEdit::multiline(text)
            .desired_rows(3)
            .desired_width(f32::INFINITY)
            .text_color(tokens::TEXT),
    );
    if super::commits(ui, &response) && *text != stored {
        let content = text.clone();
        send(shell, |shell| shell.set_text(content));
    } else if !response.has_focus() {
        stored.clone_into(text);
    }
    ui.add_space(6.0);
}

/// One row per recognized word: its time and its letters. A word keeps
/// its time when its letters change.
fn caption_words(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let columns = &inspection.words;
    for (index, word) in columns.text.iter().enumerate() {
        let start = columns.start_time.get(index).copied().unwrap_or_default();
        let text = state.fields.entry(format!("word {index}")).or_default();
        let mut edited = None;
        label_row(ui, &format!("{:.2} s", start as f64 / 1000.0), |ui| {
            let response = ui.add(TextEdit::singleline(text).desired_width(f32::INFINITY));
            if super::commits(ui, &response) && text != word {
                edited = Some(text.clone());
            }
            if !response.has_focus() {
                word.clone_into(text);
            }
        });
        if let Some(value) = edited {
            send(shell, |shell| shell.set_caption_word(index, &value));
        }
    }
    ui.add_space(6.0);
}

fn font_row(shell: &mut Shell, ui: &mut Ui, inspection: &Inspection) {
    let faces = shell.text_faces();
    let current = if inspection.font.is_empty() {
        "System".to_owned()
    } else {
        face_label(&inspection.font)
    };
    let mut picked = None;
    label_row(ui, "Font", |ui| {
        egui::ComboBox::from_id_salt("text-font")
            .selected_text(current)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for face in &faces {
                    if ui
                        .selectable_label(*face == inspection.font, face_label(face))
                        .clicked()
                    {
                        picked = Some(face.clone());
                    }
                }
            });
    });
    if let Some(face) = picked {
        send(shell, |shell| shell.set_text_face(&face));
    }
}

fn face_label(face: &str) -> String {
    let name = face.rsplit(['/', '\\']).next().unwrap_or(face);
    name.rsplit_once('.')
        .map_or(name, |(stem, _)| stem)
        .to_owned()
}

fn spacing_row(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let (letters, lines) = pair_row(
        ui,
        state,
        "Character",
        ("", inspection.letter_spacing),
        ("Line", inspection.line_spacing),
    );
    if let Some(value) = letters {
        send(shell, |shell| shell.set_letter_spacing(value));
    }
    if let Some(value) = lines {
        send(shell, |shell| shell.set_line_spacing(value));
    }
}

fn alignment_row(shell: &mut Shell, ui: &mut Ui, inspection: &Inspection) {
    let mut picked = None;
    label_row(ui, "Alignment", |ui| {
        for (index, (icon, tip)) in [
            ("textLeft", "Align left"),
            ("textCenter", "Align center"),
            ("textRight", "Align right"),
        ]
        .into_iter()
        .enumerate()
        {
            let alignment = i32::try_from(index).unwrap_or_default();
            if icon_toggle(ui, icon, tip, inspection.text_align == alignment).clicked() {
                picked = Some(alignment);
            }
        }
        ui.add_enabled_ui(false, |ui| {
            for (icon, tip) in [
                ("justifyLeft", "Justify left"),
                ("justifyCenter", "Justify center"),
                ("justifyRight", "Justify right"),
            ] {
                icon_toggle(ui, icon, tip, false);
            }
        })
        .response
        .on_disabled_hover_text(LATER);
    });
    if let Some(alignment) = picked.filter(|alignment| *alignment != inspection.text_align) {
        send(shell, |shell| shell.set_text_align(alignment));
    }
}

/// The preset tiles: a letter color on a plate. Picking one sets the color.
const PRESETS: [(&str, &str, &str); 16] = [
    ("None", "", ""),
    ("Plain", "#FFFFFF", "#3A3A3A"),
    ("Yellow", "#FFE14A", "#3A3A3A"),
    ("Red", "#FF4D4D", "#3A3A3A"),
    ("Orange", "#FF9A3C", "#3A3A3A"),
    ("Blue", "#5AA7FF", "#3A3A3A"),
    ("Green", "#3DDC6A", "#1C1C1C"),
    ("Gray", "#D0D0D0", "#2A2A2A"),
    ("Ink", "#1A1A1A", "#F3F3F3"),
    ("Gold", "#F2C14E", "#2A2418"),
    ("Violet", "#FFFFFF", "#7A3CFF"),
    ("Paper", "#7A3CFF", "#FFFFFF"),
    ("Night", "#FFFFFF", "#111111"),
    ("Lime", "#D6FF4A", "#111111"),
    ("Rose", "#FF6A9A", "#2A1218"),
    ("Cyan", "#3DE0FF", "#102028"),
];

fn preset_styles(shell: &mut Shell, ui: &mut Ui) {
    ui.add_space(6.0);
    ui.label(
        RichText::new("Preset style")
            .size(12.0)
            .color(tokens::MUTED),
    );
    let mut picked = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        for (name, color, plate) in PRESETS {
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(36.0), Sense::click());
            let fill = parse_hex(plate).unwrap_or(BOX);
            ui.painter().rect_filled(rect, CornerRadius::same(6), fill);
            if color.is_empty() {
                paint_icon(ui, rect, "unavailable", tokens::MUTED);
            } else {
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Aa",
                    FontId::proportional(13.0),
                    parse_hex(color).unwrap_or(Color32::WHITE),
                );
            }
            if response.on_hover_text(name).clicked() && !color.is_empty() {
                picked = Some(color);
            }
        }
    });
    if let Some(color) = picked {
        send(shell, |shell| shell.set_text_color(color));
    }
    ui.add_space(6.0);
}

/// Stroke, Background, Glow, and Shadow. A part's switch turns it on in
/// the prototype's first color, or off.
fn text_parts(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, look: &TextLook) {
    let percent = |v: f64| format!("{v:.0}");

    let on = !look.outline_color.is_empty();
    let header = section(ui, state, "stroke", "Stroke", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#FFFFFF" };
        send(shell, |shell| {
            shell.set_text_outline(color, look.outline_width)
        });
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) = color_row(ui, state, "stroke-color", "Color", &look.outline_color)
            {
                send(shell, |shell| {
                    shell.set_text_outline(&color, look.outline_width)
                });
            }
            // Thickness 100 is the widest outline a title takes.
            let thickness = (look.outline_width / 0.25 * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "thickness",
                "Thickness",
                thickness,
                0.0..=100.0,
                percent,
            ) {
                send(shell, |shell| {
                    shell.set_text_outline(&look.outline_color, value / 100.0 * 0.25)
                });
            }
        });
    }

    let on = !look.background_color.is_empty();
    let header = section(
        ui,
        state,
        "background",
        "Background",
        Some((on, true)),
        false,
    );
    if header.toggled {
        let color = if on { "" } else { "#000000" };
        send(shell, |shell| {
            shell.set_text_background(color, look.background_alpha, look.background_round)
        });
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) = color_row(
                ui,
                state,
                "background-color",
                "Color",
                &look.background_color,
            ) {
                send(shell, |shell| {
                    shell.set_text_background(&color, look.background_alpha, look.background_round)
                });
            }
            let alpha = (look.background_alpha * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "box-opacity",
                "Opacity",
                alpha,
                0.0..=100.0,
                |v| format!("{v:.0}%"),
            ) {
                send(shell, |shell| {
                    shell.set_text_background(
                        &look.background_color,
                        value / 100.0,
                        look.background_round,
                    )
                });
            }
        });
    }

    let on = !look.glow_color.is_empty();
    let header = section(ui, state, "glow", "Glow", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#FFFFFF" };
        send(shell, |shell| {
            shell.set_text_glow(color, look.glow_strength, look.glow_range)
        });
    }
    if header.open {
        body(ui, |ui| {
            later_tiles(ui, &["Soft glow", "Halo"]);
            if let Some(color) = color_row(ui, state, "glow-color", "Color", &look.glow_color) {
                send(shell, |shell| {
                    shell.set_text_glow(&color, look.glow_strength, look.glow_range)
                });
            }
            let strength = (look.glow_strength * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "intensity",
                "Intensity",
                strength,
                0.0..=100.0,
                percent,
            ) {
                send(shell, |shell| {
                    shell.set_text_glow(&look.glow_color, value / 100.0, look.glow_range)
                });
            }
            let range = (look.glow_range * 100.0).round();
            if let Some(value) = slider(ui, state, "range", "Range", range, 0.0..=100.0, percent) {
                send(shell, |shell| {
                    shell.set_text_glow(&look.glow_color, look.glow_strength, value / 100.0)
                });
            }
            later_slider(ui, "Y-offset", 0.0, "0°");
            later_slider(ui, "X-offset", 0.0, "0°");
        });
    }

    let on = !look.shadow_color.is_empty();
    let header = section(ui, state, "shadow", "Shadow", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#000000" };
        send(shell, |shell| {
            shell.set_text_shadow(
                color,
                look.shadow_distance,
                look.shadow_angle,
                look.shadow_blur,
            )
        });
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) = color_row(ui, state, "shadow-color", "Color", &look.shadow_color) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &color,
                        look.shadow_distance,
                        look.shadow_angle,
                        look.shadow_blur,
                    )
                });
            }
            later_slider(ui, "Opacity", 90.0, "90%");
            let blur = (look.shadow_blur * 100.0).round();
            if let Some(value) = slider(ui, state, "blur", "Bluriness", blur, 0.0..=100.0, |v| {
                format!("{v:.0}%")
            }) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &look.shadow_color,
                        look.shadow_distance,
                        look.shadow_angle,
                        value / 100.0,
                    )
                });
            }
            let distance = look.shadow_distance.round();
            if let Some(value) = slider(
                ui,
                state,
                "distance",
                "Distance",
                distance,
                0.0..=100.0,
                percent,
            ) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &look.shadow_color,
                        value,
                        look.shadow_angle,
                        look.shadow_blur,
                    )
                });
            }
            let angle = look.shadow_angle.round();
            if let Some(value) = slider(ui, state, "angle", "Angle", angle, -180.0..=180.0, |v| {
                format!("{v:.0}°")
            }) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &look.shadow_color,
                        look.shadow_distance,
                        value,
                        look.shadow_blur,
                    )
                });
            }
        });
    }
}

// Controls.

/// Send one edit and show its refusal.
fn send(shell: &mut Shell, edit: impl FnOnce(&mut Shell) -> Result<(), ShellError>) {
    if let Err(error) = edit(shell) {
        shell.set_notice(error.to_string());
    }
}

/// The text tabs across the top. The active one is cyan. Tabs that do not
/// fit scroll sideways, as the prototype's do.
fn tabs(ui: &mut Ui, labels: &[&str], current: &mut usize) {
    ui.add_space(12.0);
    egui::ScrollArea::horizontal()
        .id_salt(("inspector-tabs", labels.len()))
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 16.0;
                for (index, label) in labels.iter().enumerate() {
                    let color = if index == *current {
                        ACCENT
                    } else {
                        tokens::MUTED
                    };
                    let response = ui.add(
                        egui::Label::new(RichText::new(*label).size(13.0).color(color))
                            .selectable(false)
                            .sense(Sense::click()),
                    );
                    if response.clicked() {
                        *current = index;
                    }
                }
            });
        });
    ui.add_space(10.0);
    let line = ui.cursor().min.y;
    let span = ui.max_rect().x_range();
    ui.painter()
        .hline(span, line, Stroke::new(1.0, Color32::from_white_alpha(20)));
}

/// The rounded page switch under the tabs.
fn pages(ui: &mut Ui, labels: &[&str], current: &mut usize) {
    ui.add_space(12.0);
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 30.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::same(10), PAGES);
    let count = labels.len().max(1) as f32;
    let font = FontId::proportional(12.0);
    let widths: Vec<f32> = labels
        .iter()
        .map(|label| {
            ui.painter()
                .layout_no_wrap((*label).to_owned(), font.clone(), Color32::WHITE)
                .size()
                .x
        })
        .collect();
    let spare = ((rect.width() - 6.0 - widths.iter().sum::<f32>()) / count).max(0.0);
    let mut left = rect.min.x + 3.0;
    for (index, label) in labels.iter().enumerate() {
        let cell = Rect::from_min_size(
            egui::pos2(left, rect.min.y + 3.0),
            Vec2::new(widths[index] + spare, 24.0),
        );
        left += cell.width();
        let response = ui.interact(
            cell,
            ui.make_persistent_id(("page", label, index)),
            Sense::click(),
        );
        if response.clicked() {
            *current = index;
        }
        let on = index == *current;
        if on {
            ui.painter()
                .rect_filled(cell, CornerRadius::same(8), PAGE_ON);
        }
        ui.painter().text(
            cell.center(),
            Align2::CENTER_CENTER,
            *label,
            font.clone(),
            if on { Color32::WHITE } else { tokens::MUTED },
        );
    }
    ui.add_space(4.0);
}

struct Header {
    open: bool,
    toggled: bool,
    reset: bool,
}

/// A section header: an optional on switch (`(on, editable)`), the name
/// with a chevron that folds the section, and, with `keys`, the reset and
/// keyframe buttons.
fn section(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    check: Option<(bool, bool)>,
    keys: bool,
) -> Header {
    let mut header = Header {
        open: state.open.contains(key),
        toggled: false,
        reset: false,
    };
    ui.horizontal(|ui| {
        ui.set_min_height(40.0);
        if let Some((on, editable)) = check {
            let response = ui.add_enabled_ui(editable, |ui| check_box(ui, on)).inner;
            header.toggled = response.clicked();
        }
        ui.spacing_mut().item_spacing.x = 6.0;
        let name = if check.is_some() {
            let name = section_name(ui, label);
            name.union(chevron(ui, header.open))
        } else {
            let fold = chevron(ui, header.open);
            section_name(ui, label).union(fold)
        };
        if name.clicked() {
            header.open = !header.open;
            if header.open {
                state.open.insert(key.to_owned());
            } else {
                state.open.remove(key);
            }
        }
        if keys {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_enabled_ui(false, |ui| {
                    icon_toggle(ui, "chevron", "Next keyframe", false);
                    icon_toggle(ui, "keyframe", "Add keyframe", false);
                    icon_toggle(ui, "chevronLeft", "Previous keyframe", false);
                })
                .response
                .on_disabled_hover_text(LATER);
                header.reset = icon_toggle(ui, "reset", &format!("Reset {label}"), false).clicked();
            });
        }
    });
    if header.toggled && !header.open {
        header.open = true;
        state.open.insert(key.to_owned());
    }
    divider(ui);
    header
}

fn section_name(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Label::new(RichText::new(label).size(13.0).color(Color32::WHITE))
            .selectable(false)
            .sense(Sense::click()),
    )
}

/// A painted fold mark: down when open, right when shut.
fn chevron(ui: &mut Ui, open: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::click());
    let c = rect.center();
    let points = if open {
        vec![
            c + Vec2::new(-3.5, -1.5),
            c + Vec2::new(0.0, 2.0),
            c + Vec2::new(3.5, -1.5),
        ]
    } else {
        vec![
            c + Vec2::new(-1.5, -3.5),
            c + Vec2::new(2.0, 0.0),
            c + Vec2::new(-1.5, 3.5),
        ]
    };
    ui.painter()
        .add(egui::Shape::line(points, Stroke::new(1.5, tokens::MUTED)));
    response
}

/// The faint line under a section header or a feature row.
fn divider(ui: &mut Ui) {
    let line = ui.cursor().min.y;
    let span = ui.max_rect().x_range();
    ui.painter()
        .hline(span, line, Stroke::new(1.0, Color32::from_white_alpha(14)));
}

/// A feature the prototype lists that Cutlass cannot do yet: a dimmed row
/// with a "Later" mark.
fn later_row(ui: &mut Ui, label: &str) {
    ui.horizontal(|ui| {
        ui.set_min_height(40.0);
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_enabled_ui(false, |ui| check_box(ui, false));
        ui.label(RichText::new(label).size(13.0).color(tokens::FAINT));
        chevron(ui, false);
        later_badge(ui);
    })
    .response
    .on_hover_text(LATER);
    divider(ui);
}

/// A small "Later" mark, painted at text height so it never stretches a row.
fn later_badge(ui: &mut Ui) {
    let galley =
        ui.painter()
            .layout_no_wrap("Later".to_owned(), FontId::proportional(10.0), ACCENT);
    let size = galley.size() + Vec2::new(10.0, 4.0);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(6), BOX);
    ui.painter()
        .galley(rect.center() - galley.size() / 2.0, galley, ACCENT);
}

fn later_note(ui: &mut Ui, what: &str) {
    note(
        ui,
        &format!("{what} comes later. Cutlass has no edit for it yet."),
    );
}

fn note(ui: &mut Ui, text: &str) {
    ui.add_space(8.0);
    ui.label(RichText::new(text).size(12.0).color(tokens::MUTED));
    ui.add_space(4.0);
}

fn later_button(ui: &mut Ui, label: &str) {
    ui.add_space(6.0);
    ui.add_enabled(
        false,
        egui::Button::new(RichText::new(label).size(12.0)).fill(BOX),
    )
    .on_disabled_hover_text(LATER);
}

fn footer(ui: &mut Ui, labels: &[&str]) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        for label in labels {
            ui.add_enabled(
                false,
                egui::Button::new(RichText::new(*label).size(12.0)).fill(BOX),
            )
            .on_disabled_hover_text(LATER);
        }
    });
}

fn later_tiles(ui: &mut Ui, names: &[&str]) {
    ui.add_space(8.0);
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            for name in names {
                ui.add_sized(
                    [64.0, 48.0],
                    egui::Button::new(RichText::new(*name).size(11.0)).fill(BOX),
                );
            }
        });
    })
    .response
    .on_disabled_hover_text(LATER);
}

/// The indented body under a section.
fn body(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    ui.add_space(4.0);
    ui.vertical(add);
    ui.add_space(4.0);
}

fn row_label(ui: &mut Ui, label: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(LABEL_WIDTH, 26.0), Sense::hover());
    let color = if ui.is_enabled() {
        tokens::TEXT
    } else {
        tokens::FAINT
    };
    ui.painter().with_clip_rect(rect).text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(12.0),
        color,
    );
}

fn label_row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        row_label(ui, label);
        add(ui);
    });
    ui.add_space(4.0);
}

/// A slider row: name, track, value box. The value is sent once, when the
/// pointer lets go, so a drag is one undo step.
fn slider(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    stored: f64,
    range: std::ops::RangeInclusive<f64>,
    show: impl Fn(f64) -> String,
) -> Option<f64> {
    let mut value = state.live.get(key).copied().unwrap_or(stored);
    ui.horizontal(|ui| {
        row_label(ui, label);
        let track = (ui.available_width() - VALUE_WIDTH - 8.0).max(40.0);
        ui.spacing_mut().slider_width = track;
        let response = ui.add(Slider::new(&mut value, range).show_value(false));
        if response.changed() {
            state.live.insert(key.to_owned(), value);
        }
        value_box(ui, &show(value));
    });
    ui.add_space(4.0);
    let held = ui.input(|input| input.pointer.any_down());
    if held {
        return None;
    }
    state.live.remove(key).filter(|value| *value != stored)
}

fn later_slider(ui: &mut Ui, label: &str, value: f64, shown: &str) {
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            row_label(ui, label);
            let track = (ui.available_width() - VALUE_WIDTH - 8.0).max(40.0);
            ui.spacing_mut().slider_width = track;
            let mut value = value;
            ui.add(Slider::new(&mut value, -1000.0..=1000.0).show_value(false));
            value_box(ui, shown);
        });
    })
    .response
    .on_disabled_hover_text(LATER);
    ui.add_space(4.0);
}

fn value_box(ui: &mut Ui, text: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(VALUE_WIDTH, 26.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(8), BOX);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        FontId::proportional(12.0),
        tokens::TEXT,
    );
}

/// A name and two number boxes at the right. A box whose stored value is
/// NaN is left out. `Some` is a new finite number typed into that box.
fn pair_row(
    ui: &mut Ui,
    state: &mut InspectorState,
    label: &str,
    first: (&str, f64),
    second: (&str, f64),
) -> (Option<f64>, Option<f64>) {
    let mut out = (None, None);
    ui.horizontal(|ui| {
        row_label(ui, label);
        let width = if second.1.is_nan() {
            PAIR_BOX
        } else {
            ((ui.available_width() - ui.spacing().item_spacing.x) / 2.0).max(PAIR_BOX - 24.0)
        };
        out.0 = number_box(ui, state, &format!("{label} {}", first.0), first, width);
        if !second.1.is_nan() {
            out.1 = number_box(ui, state, &format!("{label} {}", second.0), second, width);
        }
    });
    ui.add_space(4.0);
    out
}

fn number_box(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    (unit, stored): (&str, f64),
    width: f32,
) -> Option<f64> {
    // Never show a minus sign on zero.
    let stored = stored + 0.0;
    let text = state.fields.entry(key.to_owned()).or_default();
    let mut edited = None;
    egui::Frame::new()
        .fill(BOX)
        .corner_radius(CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.set_width(width - 16.0);
            ui.horizontal(|ui| {
                if unit == "°" {
                    paint_inline_icon(ui, "rotate");
                } else if !unit.is_empty() {
                    ui.label(RichText::new(unit).size(12.0).color(tokens::MUTED));
                }
                let response = ui.add(
                    TextEdit::singleline(text)
                        .desired_width(ui.available_width())
                        .frame(egui::Frame::NONE)
                        .text_color(Color32::WHITE),
                );
                if super::commits(ui, &response) {
                    edited = text
                        .trim()
                        .parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite());
                }
                if !response.has_focus() {
                    let shown = if stored.fract() == 0.0 {
                        format!("{stored:.0}")
                    } else {
                        format!("{stored:.2}")
                    };
                    *text = shown;
                }
            });
        });
    edited.filter(|value| *value != stored)
}

/// A name and a swatch that opens a color picker. The color is sent once,
/// when the pointer lets go. An empty stored color shows white.
fn color_row(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    stored: &str,
) -> Option<String> {
    let stored_color = parse_hex(stored).unwrap_or(Color32::WHITE);
    let mut color = state.colors.get(key).copied().unwrap_or(stored_color);
    label_row(ui, label, |ui| {
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let response = egui::color_picker::color_edit_button_srgba(
                ui,
                &mut color,
                egui::color_picker::Alpha::Opaque,
            );
            if response.changed() {
                state.colors.insert(key.to_owned(), color);
            }
        });
    });
    if ui.input(|input| input.pointer.any_down()) {
        return None;
    }
    let picked = state.colors.remove(key)?;
    let hex = format!("#{:02X}{:02X}{:02X}", picked.r(), picked.g(), picked.b());
    (picked != stored_color || stored.is_empty()).then_some(hex)
}

/// `#RRGGBB`, `RRGGBB`, or `0xAARRGGBB`.
fn parse_hex(text: &str) -> Option<Color32> {
    let text = text.trim();
    let digits = text
        .strip_prefix('#')
        .or_else(|| {
            text.strip_prefix("0x")
                .map(|rest| rest.get(2..).unwrap_or(rest))
        })
        .unwrap_or(text);
    if digits.len() != 6 {
        return None;
    }
    let value = u32::from_str_radix(digits, 16).ok()?;
    let [_, r, g, b] = value.to_be_bytes();
    Some(Color32::from_rgb(r, g, b))
}

fn switch_row(ui: &mut Ui, label: &str, on: bool, editable: bool) -> egui::Response {
    let response = ui
        .horizontal(|ui| {
            ui.label(RichText::new(label).size(12.0).color(tokens::TEXT));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_enabled_ui(editable, |ui| {
                    let (rect, response) =
                        ui.allocate_exact_size(Vec2::new(32.0, 18.0), Sense::click());
                    let fill = if on { tokens::EXPORT } else { tokens::PANEL };
                    ui.painter().rect_filled(rect, CornerRadius::same(9), fill);
                    let knob = if on {
                        rect.right_center() - Vec2::new(9.0, 0.0)
                    } else {
                        rect.left_center() + Vec2::new(9.0, 0.0)
                    };
                    ui.painter().circle_filled(knob, 7.0, Color32::WHITE);
                    response
                })
                .inner
            })
            .inner
        })
        .inner;
    ui.add_space(4.0);
    response
}

fn check_box(ui: &mut Ui, on: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(16.0), Sense::click());
    let fill = if on { ACCENT } else { Color32::TRANSPARENT };
    ui.painter().rect(
        rect,
        CornerRadius::same(4),
        fill,
        Stroke::new(1.0, if on { ACCENT } else { tokens::FAINT }),
        egui::StrokeKind::Inside,
    );
    if on {
        let c = rect.center();
        ui.painter().add(egui::Shape::line(
            vec![
                c + Vec2::new(-4.0, 0.0),
                c + Vec2::new(-1.0, 3.0),
                c + Vec2::new(4.5, -3.0),
            ],
            Stroke::new(1.8, tokens::EXPORT_INK),
        ));
    }
    response
}

fn icon_toggle(ui: &mut Ui, name: &str, tip: &str, on: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::click());
    if on {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(6), PAGE_ON);
    }
    let color = if ui.is_enabled() {
        tokens::TEXT
    } else {
        tokens::FAINT
    };
    paint_icon(ui, rect, name, color);
    response.on_hover_text(tip)
}

/// A row of icon buttons on a dark bar. Returns the value of the one clicked.
fn icon_bar<T: Copy>(ui: &mut Ui, items: &[(&str, &str, Option<T>)]) -> Option<T> {
    let mut picked = None;
    egui::Frame::new()
        .fill(PAGES)
        .corner_radius(CornerRadius::same(10))
        .inner_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let gap = ((ui.available_width() - 24.0 * items.len() as f32)
                    / (items.len().max(2) - 1) as f32)
                    .max(2.0);
                ui.spacing_mut().item_spacing.x = gap;
                for (icon, tip, value) in items {
                    if icon_toggle(ui, icon, tip, false).clicked() {
                        picked = *value;
                    }
                }
            });
        });
    ui.add_space(4.0);
    picked
}

fn paint_icon(ui: &Ui, rect: Rect, name: &str, color: Color32) {
    let mark = Rect::from_center_size(rect.center(), Vec2::splat(16.0));
    if icons::paint(ui.painter(), mark, name, color).is_err() {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            "?",
            FontId::proportional(11.0),
            color,
        );
    }
}

fn paint_inline_icon(ui: &mut Ui, name: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::hover());
    paint_icon(ui, rect, name, tokens::MUTED);
}
