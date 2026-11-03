//! The inspector's clip panels. The selected clip's kind picks the panel,
//! and each control sends one edit through the shell, so each change is one
//! undo step. A clip on a locked row shows its values without edits.

use egui::{RichText, Ui};

use crate::shell::{Inspection, Panel, Shell, ShellError, TextLook};
use crate::tokens;
use crate::widgets;

use super::{ViewState, align_name, card_meta, color_field, fact, section_title};

pub(super) fn clip_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    inspection: &Inspection,
    width: f32,
) {
    let panel = shell.panel().unwrap_or(Panel::Other);
    section_title(ui, panel.title());
    let name = match panel {
        Panel::Effect | Panel::Filter | Panel::Adjust => shell.effect_name().unwrap_or_default(),
        _ => inspection.material.clone(),
    };
    fact(ui, "Name", &name, width);
    if inspection.locked {
        fact(ui, "Row", "Locked", width);
    }
    ui.add_enabled_ui(!inspection.locked, |ui| {
        timing(shell, ui, view, inspection);
        match panel {
            Panel::Main { still } | Panel::Overlay { still } => {
                placement(shell, ui, view, inspection);
                picture(shell, ui, view, inspection);
                if !still {
                    sound(shell, ui, view, inspection);
                    playback(shell, ui, view, inspection);
                }
            }
            Panel::Audio => {
                sound(shell, ui, view, inspection);
                playback(shell, ui, view, inspection);
            }
            Panel::Text => {
                words(shell, ui, view, inspection);
                text_style(shell, ui, view, inspection);
                text_look(shell, ui, view);
                placement(shell, ui, view, inspection);
            }
            Panel::Caption => {
                caption_words(shell, ui, view, inspection);
                text_style(shell, ui, view, inspection);
                text_look(shell, ui, view);
                placement(shell, ui, view, inspection);
            }
            Panel::Sticker => placement(shell, ui, view, inspection),
            Panel::Effect | Panel::Filter | Panel::Adjust => strength(shell, ui, view),
            Panel::Other => {}
        }
    });
}

fn group(ui: &mut Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(
        RichText::new(title)
            .size(tokens::TYPE_SIZE)
            .strong()
            .color(tokens::TEXT),
    );
    ui.add_space(6.0);
}

/// Send one edit and show its refusal.
fn send(shell: &mut Shell, edit: impl FnOnce(&mut Shell) -> Result<(), ShellError>) {
    if let Err(error) = edit(shell) {
        shell.set_notice(error.to_string());
    }
}

/// A number row. The typed text stays while the field has focus, and the
/// stored value returns when it does not. `Some` is a new finite parse.
fn number(ui: &mut Ui, view: &mut ViewState, label: &str, stored: f64, hover: &str) -> Option<f64> {
    let text = view.fields.entry(label.to_owned()).or_default();
    let mut edited = None;
    ui.horizontal(|ui| {
        ui.set_min_width(ui.available_width());
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(tokens::MUTED)),
        );
        let response = widgets::field(ui, text);
        if response.lost_focus() {
            edited = text
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite());
        }
        if !response.has_focus() {
            *text = format!("{stored}");
        }
        if !hover.is_empty() {
            response.on_hover_text(hover);
        }
    });
    ui.add_space(4.0);
    edited.filter(|value| *value != stored)
}

/// A color row kept in the panel's field table.
fn color(
    ui: &mut Ui,
    view: &mut ViewState,
    label: &str,
    stored: &str,
    hover: &str,
) -> Option<String> {
    let text = view.fields.entry(label.to_owned()).or_default();
    color_field(ui, label, text, stored, hover)
}

/// A button that shows `on` or `off` and flips when clicked.
fn switch(ui: &mut Ui, on: bool, labels: (&str, &str), hover: &str) -> bool {
    let label = if on { labels.0 } else { labels.1 };
    widgets::toolbar_button(ui, label)
        .on_hover_text(hover)
        .clicked()
}

fn seconds(microseconds: i64) -> f64 {
    microseconds as f64 / 1_000_000.0
}

fn timing(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    group(ui, "Timing");
    let start = seconds(inspection.start.0);
    if let Some(value) = number(ui, view, "Start", start, "Seconds on the timeline") {
        send(shell, |shell| shell.set_start_seconds(value));
    }
    let length = seconds(inspection.duration.0);
    if let Some(value) = number(ui, view, "Length", length, "Seconds. The end moves") {
        send(shell, |shell| shell.set_duration_seconds(value));
    }
    ui.label(
        RichText::new(card_meta(inspection.duration.0))
            .size(12.0)
            .color(tokens::MUTED),
    );
}

fn placement(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    if !shell.place_enabled() {
        return;
    }
    group(ui, "Transform");
    let (x, y) = (inspection.transform_x, inspection.transform_y);
    if let Some(value) = number(ui, view, "X", x, "Fractions of half the canvas") {
        send(shell, |shell| shell.set_position(value, y));
    }
    if let Some(value) = number(ui, view, "Y", y, "Up is positive") {
        send(shell, |shell| shell.set_position(x, value));
    }
    if let Some(value) = number(
        ui,
        view,
        "Scale",
        inspection.scale_x,
        "1 is the fitted picture",
    ) {
        send(shell, |shell| shell.set_scale(value));
    }
    if let Some(value) = number(ui, view, "Rotation", inspection.rotation, "Degrees") {
        send(shell, |shell| shell.set_rotation(value));
    }
    if let Some(value) = number(ui, view, "Opacity", inspection.opacity, "1 is solid") {
        send(shell, |shell| shell.set_opacity(value));
    }
}

fn picture(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    group(ui, "Picture");
    ui.horizontal_wrapped(|ui| {
        if shell.mirror_enabled()
            && switch(
                ui,
                shell.mirrored(),
                ("Mirrored", "Mirror"),
                "Left to right",
            )
        {
            send(shell, Shell::toggle_mirror);
        }
        if shell.place_enabled()
            && switch(ui, shell.flipped(), ("Flipped", "Upright"), "Top to bottom")
        {
            send(shell, Shell::toggle_vertical_flip);
        }
        if shell.rotate_enabled()
            && widgets::toolbar_button(ui, "Turn")
                .on_hover_text("A quarter turn clockwise")
                .clicked()
        {
            send(shell, Shell::turn_clockwise);
        }
        if shell.crop_enabled()
            && switch(
                ui,
                shell.cropped(),
                ("Cropped", "Crop"),
                "The middle of the picture",
            )
        {
            send(shell, Shell::toggle_crop);
        }
    });
    if shell.crop_enabled()
        && let Some(value) = color(
            ui,
            view,
            "Letterbox",
            &inspection.letterbox,
            "Around the picture. Empty uses the project color",
        )
    {
        send(shell, |shell| shell.set_letterbox(&value));
    }
}

fn strength(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let Some(stored) = shell.effect_strength() else {
        return;
    };
    group(ui, "Amount");
    let hint = "0 to 1. Kept in the draft; the preview does not scale by it yet";
    if let Some(value) = number(ui, view, "Strength", stored, hint) {
        send(shell, |shell| shell.set_effect_strength(value));
    }
}

fn sound(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    group(ui, "Sound");
    if let Some(value) = number(ui, view, "Volume", inspection.volume, "1 is as recorded") {
        send(shell, |shell| shell.set_volume(value));
    }
    if !shell.fade_enabled() {
        return;
    }
    let fade_in = seconds(inspection.fade_in);
    if let Some(value) = number(ui, view, "Fade in", fade_in, "Seconds") {
        send(shell, |shell| shell.set_fade_in(value));
    }
    let fade_out = seconds(inspection.fade_out);
    if let Some(value) = number(ui, view, "Fade out", fade_out, "Seconds") {
        send(shell, |shell| shell.set_fade_out(value));
    }
}

fn playback(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    group(ui, "Speed");
    if let Some(value) = number(ui, view, "Speed", inspection.speed, "1 is normal speed") {
        send(shell, |shell| shell.set_speed(value));
    }
    ui.horizontal_wrapped(|ui| {
        if shell.reverse_enabled()
            && switch(
                ui,
                shell.reversed(),
                ("Reversed", "Reverse"),
                "Play backward",
            )
        {
            send(shell, Shell::toggle_reverse);
        }
        if shell.freeze_enabled()
            && switch(
                ui,
                shell.frozen(),
                ("Frozen", "Freeze"),
                "Hold the frame under the playhead",
            )
        {
            send(shell, Shell::toggle_freeze);
        }
    });
}

fn words(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    let Some(text) = inspection.text.clone() else {
        return;
    };
    group(ui, "Words");
    if view.text_for != inspection.segment_id.as_str() {
        view.text_buffer = text;
        view.text_for = inspection.segment_id.as_str().to_owned();
    }
    let response = widgets::field(ui, &mut view.text_buffer);
    if response.lost_focus() {
        let content = view.text_buffer.clone();
        send(shell, |shell| shell.set_text(content));
    }
}

/// One row per recognized word: its time and its letters. A word keeps
/// its time when its letters change.
fn caption_words(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    group(ui, "Words");
    let columns = &inspection.words;
    if columns.text.is_empty() {
        words(shell, ui, view, inspection);
        return;
    }
    for (index, word) in columns.text.iter().enumerate() {
        let start = columns.start_time.get(index).copied().unwrap_or_default();
        let key = format!("word {index}");
        let text = view.fields.entry(key).or_default();
        let mut edited = None;
        ui.horizontal(|ui| {
            ui.add_sized(
                [108.0, 18.0],
                egui::Label::new(
                    RichText::new(format!("{:.2} s", start as f64 / 1000.0))
                        .size(13.0)
                        .color(tokens::MUTED),
                ),
            );
            let response = widgets::field(ui, text);
            if response.lost_focus() && text != word {
                edited = Some(text.clone());
            }
            if !response.has_focus() {
                word.clone_into(text);
            }
        });
        ui.add_space(2.0);
        if let Some(value) = edited {
            send(shell, |shell| shell.set_caption_word(index, &value));
        }
    }
}

fn text_style(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, inspection: &Inspection) {
    group(ui, "Style");
    let face_name = if inspection.font.is_empty() {
        "Face"
    } else {
        inspection.font.as_str()
    };
    ui.horizontal_wrapped(|ui| {
        if widgets::toolbar_button(ui, face_name)
            .on_hover_text("Installed face")
            .clicked()
        {
            send(shell, Shell::cycle_text_font);
        }
        if widgets::toolbar_button(ui, align_name(inspection.text_align))
            .on_hover_text("Left, center, or right")
            .clicked()
        {
            send(shell, Shell::cycle_text_align);
        }
    });
    if let Some(value) = number(ui, view, "Size", inspection.text_size, "Pixels") {
        send(shell, |shell| shell.set_text_size(value));
    }
    if let Some(value) = color(ui, view, "Color", &inspection.text_color, "Hex color") {
        send(shell, |shell| shell.set_text_color(&value));
    }
    let gap = "0 is the normal gap";
    if let Some(value) = number(ui, view, "Letters", inspection.letter_spacing, gap) {
        send(shell, |shell| shell.set_letter_spacing(value));
    }
    if let Some(value) = number(ui, view, "Lines", inspection.line_spacing, gap) {
        send(shell, |shell| shell.set_line_spacing(value));
    }
    let wrap = "Pixels. 0 keeps one line";
    if let Some(value) = number(ui, view, "Width", inspection.line_width, wrap) {
        send(shell, |shell| shell.set_line_width(value));
    }
}

/// Outline, shadow, box, and glow. Each part is off while its color is
/// empty, and its first color brings the starting amounts.
fn text_look(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let Some(look) = shell.text_look() else {
        return;
    };
    let TextLook {
        outline_color,
        outline_width,
        shadow_color,
        shadow_distance,
        shadow_blur,
        background_color,
        background_alpha,
        background_round,
        glow_color,
        glow_strength,
        glow_range,
    } = look;
    let off = "Hex color. Empty is off";

    group(ui, "Outline");
    if let Some(value) = color(ui, view, "Outline", &outline_color, off) {
        send(shell, |shell| shell.set_text_outline(&value, outline_width));
    }
    if let Some(value) = number(ui, view, "Outline width", outline_width, "Ems, up to 0.25") {
        send(shell, |shell| shell.set_text_outline(&outline_color, value));
    }

    group(ui, "Shadow");
    if let Some(value) = color(ui, view, "Shadow", &shadow_color, off) {
        send(shell, |shell| {
            shell.set_text_shadow(&value, shadow_distance, shadow_blur)
        });
    }
    let distance_hint = "Hundredths of an em, up to 100";
    if let Some(value) = number(ui, view, "Distance", shadow_distance, distance_hint) {
        send(shell, |shell| {
            shell.set_text_shadow(&shadow_color, value, shadow_blur)
        });
    }
    if let Some(value) = number(ui, view, "Blur", shadow_blur, "0 to 1") {
        send(shell, |shell| {
            shell.set_text_shadow(&shadow_color, shadow_distance, value)
        });
    }

    group(ui, "Background");
    if let Some(value) = color(ui, view, "Box", &background_color, off) {
        send(shell, |shell| {
            shell.set_text_background(&value, background_alpha, background_round)
        });
    }
    if let Some(value) = number(ui, view, "Box opacity", background_alpha, "0 to 1") {
        send(shell, |shell| {
            shell.set_text_background(&background_color, value, background_round)
        });
    }
    if let Some(value) = number(ui, view, "Corners", background_round, "0 square to 1 round") {
        send(shell, |shell| {
            shell.set_text_background(&background_color, background_alpha, value)
        });
    }

    group(ui, "Glow");
    if let Some(value) = color(ui, view, "Glow", &glow_color, off) {
        send(shell, |shell| {
            shell.set_text_glow(&value, glow_strength, glow_range)
        });
    }
    if let Some(value) = number(ui, view, "Strength", glow_strength, "0 to 1") {
        send(shell, |shell| {
            shell.set_text_glow(&glow_color, value, glow_range)
        });
    }
    if let Some(value) = number(ui, view, "Range", glow_range, "0 to 1") {
        send(shell, |shell| {
            shell.set_text_glow(&glow_color, glow_strength, value)
        });
    }
}
