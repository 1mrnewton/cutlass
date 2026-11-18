//! A video clip's Remove BG page: Auto removal, Stroke, Custom removal and
//! Chroma key, laid out as the reference lays them out.
//!
//! Auto removal and Custom removal cut the clip down to a matte video made
//! as a job; Feather edge, Clean up edge and Stroke change how the cut
//! looks without making the matte again.

use cutlass_effects::{Matting, MattingMode, Stroke, StrokeStyle};
use egui::{RichText, Ui, Vec2};

use crate::shell::{
    Inspection, KeyedSetting, Shell, auto_model_here, follow_model_here, matte_model_question,
};
use crate::{tokens, ui};

use super::audio::{later_slider, name_tile};
use super::inspector::{
    InspectorState, ROW_GAP, body, color_row, none_tile, note, row_label, section, send,
    setting_row, slider,
};

/// A key's strength when it is first switched on, or reset.
const KEY_STRENGTH: f64 = 0.4;

pub(super) fn remove_bg_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let matting = shell.matting();
    auto_removal(shell, ui, state, matting.as_ref());
    stroke(shell, ui, state, matting.as_ref());
    custom_removal(shell, ui, state, inspection, matting.as_ref());
    chroma_key(shell, ui, state, inspection);
}

/// How far the matte is, or why the clip draws uncut: shown under whichever
/// removal is on.
fn matte_state(shell: &mut Shell, ui: &mut Ui) {
    if let Some(done) = shell.matte_progress() {
        // The pace so far: on a computer with no GPU path a matte takes
        // seconds a frame, and the wait should say so.
        let left = shell.job().and_then(|job| job.left);
        let line = match left {
            Some(left) => format!(
                "Removing the background {:.0}%, about {} left",
                done * 100.0,
                super::spoken_duration(left)
            ),
            None => format!("Removing the background {:.0}%", done * 100.0),
        };
        note(ui, &line);
    } else if !shell.matte_made() {
        note(ui, "The clip shows uncut until its matte is made.");
        ui.horizontal(|ui| {
            row_label(ui, "");
            if ui
                .add(ui::Button::new("Make matte").text_size(12.0))
                .clicked()
            {
                send(shell, Shell::make_matte);
            }
        });
        ui.add_space(ROW_GAP);
    }
}

/// What a missing model is and a button that downloads it, or how far its
/// download is.
fn download_row(shell: &mut Shell, ui: &mut Ui, custom: bool) {
    if let Some(done) = shell.matte_model_progress() {
        note(ui, &format!("Downloading the model {:.0}%", done * 100.0));
        return;
    }
    note(ui, matte_model_question(custom));
    ui.horizontal(|ui| {
        row_label(ui, "");
        let free = shell.job().is_none();
        if ui
            .add_enabled(free, ui::Button::new("Download").text_size(12.0))
            .on_disabled_hover_text("Wait for the job that is running.")
            .clicked()
        {
            send(shell, |shell| shell.download_matte_model(custom));
        }
    });
    ui.add_space(ROW_GAP);
}

/// Feather edge and Clean up edge, 0 to 100 on the page.
fn edge_sliders(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, matting: &Matting) {
    let (feather, clean_up) = (matting.feather, matting.clean_up);
    if let Some(value) = slider(
        ui,
        state,
        "matting-feather",
        "Feather edge",
        (feather * 100.0).round(),
        0.0..=100.0,
        |v| format!("{v:.0}"),
    ) {
        send(shell, |shell| {
            shell.set_matting_edge(value / 100.0, clean_up)
        });
    }
    if let Some(value) = slider(
        ui,
        state,
        "matting-clean-up",
        "Clean up edge",
        (clean_up * 100.0).round(),
        0.0..=100.0,
        |v| format!("{v:.0}"),
    ) {
        send(shell, |shell| {
            shell.set_matting_edge(feather, value / 100.0)
        });
    }
}

/// Auto removal: the switch has a model find the person or object in every
/// frame and take the background off.
fn auto_removal(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    matting: Option<&Matting>,
) {
    let on = matting.is_some_and(|m| m.mode == MattingMode::Auto);
    let header = section(
        ui,
        state,
        "auto-removal",
        "Auto removal",
        Some((on, on || auto_model_here())),
        false,
    );
    if header.toggled {
        send(shell, |shell| shell.set_auto_removal(!on));
    }
    if !header.open {
        return;
    }
    body(ui, |ui| {
        if !auto_model_here() {
            download_row(shell, ui, false);
        }
        let Some(matting) = matting.filter(|_| on) else {
            note(
                ui,
                "Take the background off from behind the person or object.",
            );
            return;
        };
        matte_state(shell, ui);
        edge_sliders(shell, ui, state, matting);
    });
}

/// Stroke: an outline round what is kept, by style, with its size, gap,
/// grain and colour.
fn stroke(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, matting: Option<&Matting>) {
    let worn = matting.and_then(|m| m.stroke.clone());
    let header = section(
        ui,
        state,
        "stroke",
        "Stroke",
        Some((worn.is_some(), matting.is_some())),
        false,
    );
    if header.toggled {
        let next = match worn {
            Some(_) => None,
            None => Some(Stroke::new(StrokeStyle::Paper)),
        };
        send(shell, |shell| shell.set_matting_stroke(next));
        return;
    }
    if !header.open {
        return;
    }
    body(ui, |ui| {
        if matting.is_none() {
            note(
                ui,
                "Turn Auto removal or Custom removal on to outline what is kept.",
            );
            ui.add_enabled_ui(false, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::splat(8.0);
                    none_tile(ui, true);
                    for style in StrokeStyle::ALL {
                        name_tile(ui, style.label(), false);
                    }
                });
                ui.add_space(ROW_GAP);
                later_slider(ui, "Size", 35.0);
                later_slider(ui, "Spacing", 0.0);
                later_slider(ui, "Object texture", 0.0);
            });
            return;
        }
        let mut next: Option<Option<Stroke>> = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            if none_tile(ui, worn.is_none()).clicked() && worn.is_some() {
                next = Some(None);
            }
            for style in StrokeStyle::ALL {
                let on = worn.as_ref().is_some_and(|s| s.style == style);
                if name_tile(ui, style.label(), on).clicked() && !on {
                    // A new style keeps the size, gap and grain chosen.
                    let picked = match &worn {
                        Some(old) => Stroke {
                            style,
                            color: style.default_color().to_owned(),
                            ..old.clone()
                        },
                        None => Stroke::new(style),
                    };
                    next = Some(Some(picked));
                }
            }
        });
        ui.add_space(ROW_GAP);
        if let Some(stroke) = &worn {
            let value = |v: f64| (v * 100.0).round();
            let rows = [
                ("stroke-size", "Size", value(stroke.size)),
                ("stroke-spacing", "Spacing", value(stroke.spacing)),
                ("stroke-texture", "Object texture", value(stroke.texture)),
            ];
            for (index, (key, label, stored)) in rows.into_iter().enumerate() {
                if let Some(moved) = slider(ui, state, key, label, stored, 0.0..=100.0, |v| {
                    format!("{v:.0}")
                }) {
                    let mut changed = stroke.clone();
                    let unit = moved / 100.0;
                    match index {
                        0 => changed.size = unit,
                        1 => changed.spacing = unit,
                        _ => changed.texture = unit,
                    }
                    next = Some(Some(changed));
                }
            }
            if let Some(color) = color_row(ui, state, "stroke-color", "Color", &stroke.color) {
                next = Some(Some(Stroke {
                    color,
                    ..stroke.clone()
                }));
            }
        }
        if let Some(stroke) = next {
            send(shell, |shell| shell.set_matting_stroke(stroke));
        }
    });
}

/// Custom removal: click the object on the preview, or paint it by hand,
/// and the clip is cut down to it, followed through the clip. The switch
/// takes it off.
fn custom_removal(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    matting: Option<&Matting>,
) {
    let picking = shell.picking().is_some();
    let painting = shell.painting().is_some();
    let custom = matting.filter(|m| m.mode == MattingMode::Custom);
    let on = custom.is_some() || inspection.selected || picking || painting;
    let header = section(
        ui,
        state,
        "custom-removal",
        "Custom removal",
        Some((on, on || shell.pick_enabled())),
        false,
    );
    if header.toggled {
        if on {
            shell.stop_picking();
            shell.stop_painting();
            if inspection.selected {
                send(shell, Shell::clear_selection);
            }
            if custom.is_some() {
                send(shell, Shell::clear_custom_removal);
            }
        } else {
            send(shell, Shell::start_picking);
        }
    }
    if !header.open {
        return;
    }
    let follows = follow_model_here();
    body(ui, |ui| {
        if picking {
            note(
                ui,
                "Click the object on the preview a few times, across all of it: one click \
                 can find only the part under it. Shift-click to take a part out.",
            );
        } else if !shell.vision_model_ready() {
            note(ui, "The selection model is not on this machine yet.");
        }
        if !follows {
            download_row(shell, ui, true);
        }
        if custom.is_some() {
            pick_rows(shell, ui);
            matte_state(shell, ui);
        }
        ui.horizontal(|ui| {
            row_label(ui, "Select");
            if picking {
                if ui.add(ui::Button::new("Done").text_size(12.0)).clicked() {
                    if follows {
                        send(shell, Shell::follow_picked);
                    } else {
                        shell.stop_picking();
                    }
                }
            } else if ui
                .add_enabled(
                    shell.pick_enabled() && !painting,
                    ui::Button::new("Select object").text_size(12.0),
                )
                .on_disabled_hover_text("Put the playhead on an unlocked video clip.")
                .clicked()
            {
                send(shell, Shell::start_picking);
            }
            let clearable = inspection.selected || custom.is_some();
            if ui
                .add_enabled(clearable, ui::Button::new("Clear").text_size(12.0))
                .clicked()
            {
                if inspection.selected {
                    send(shell, Shell::clear_selection);
                }
                if custom.is_some() {
                    send(shell, Shell::clear_custom_removal);
                }
            }
        });
        ui.add_space(ROW_GAP);
        paint_by_hand(shell, ui, state, follows);
        if let Some(custom) = custom {
            edge_sliders(shell, ui, state, custom);
        }
    });
}

/// Each Custom removal pick: where it is, what is on it, a button that
/// moves the playhead to it and one that takes it out.
fn pick_rows(shell: &mut Shell, ui: &mut Ui) {
    let rows = shell.picks_listed();
    let free = shell.job().is_none() && shell.picking().is_none() && shell.painting().is_none();
    for (index, row) in rows.iter().enumerate() {
        let what = match (row.clicks, row.strokes) {
            (clicks, 0) => plural(clicks, "click"),
            (0, strokes) => plural(strokes, "stroke"),
            (clicks, strokes) => {
                format!("{}, {}", plural(clicks, "click"), plural(strokes, "stroke"))
            }
        };
        ui.horizontal(|ui| {
            row_label(ui, if index == 0 { "Picks" } else { "" });
            let at = row
                .at
                .map_or_else(|| "off the clip".to_owned(), |at| shell.timecode_at(at));
            ui.label(
                RichText::new(format!("{at} · {what}"))
                    .size(12.0)
                    .color(tokens::MUTED),
            );
            if let Some(at) = row.at
                && ui
                    .add(ui::Button::new("Go to").text_size(12.0))
                    .on_hover_text("Move the playhead to this pick.")
                    .clicked()
            {
                shell.seek(at);
            }
            if ui
                .add_enabled(free, ui::Button::new("Remove").text_size(12.0))
                .on_hover_text("Take this pick out and make the matte again.")
                .on_disabled_hover_text("Wait for the job, or finish picking or painting.")
                .clicked()
            {
                let frame = row.frame;
                send(shell, move |shell| shell.remove_pick(frame));
            }
        });
    }
    if !rows.is_empty() {
        ui.add_space(ROW_GAP);
    }
}

fn plural(count: usize, word: &str) -> String {
    if count == 1 {
        format!("1 {word}")
    } else {
        format!("{count} {word}s")
    }
}

/// Paint by hand: brushes that keep or take out, plain or edge-finding,
/// painted on the preview's frame and followed through the clip on Apply.
fn paint_by_hand(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, follows: bool) {
    ui.label(RichText::new("Paint by hand").size(12.0));
    ui.add_space(4.0);
    let now = shell
        .painting()
        .map(|p| (p.smart, p.keep, p.size, p.brushes.len()));
    let brushes = [
        ("Smart brush", true, true),
        ("Smart eraser", true, false),
        ("Brush", false, true),
        ("Eraser", false, false),
    ];
    let mut chosen = None;
    ui.add_enabled_ui(follows, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            for (name, smart, keep) in brushes {
                let on = now.is_some_and(|(s, k, _, _)| s == smart && k == keep);
                let button = ui::Button::new(name).on(on).text_size(11.0);
                if ui.add(button).clicked() {
                    chosen = Some((smart, keep));
                }
            }
        });
    })
    .response
    .on_disabled_hover_text("Painting needs the model that follows the object.");
    if let Some((smart, keep)) = chosen {
        if now.is_none() {
            send(shell, Shell::start_painting);
        }
        shell.set_brush(smart, keep);
    }
    ui.add_space(ROW_GAP);
    let Some((_, _, size, strokes)) = now else {
        return;
    };
    note(ui, "Paint over the preview.");
    if let Some(value) = slider(
        ui,
        state,
        "brush-size",
        "Size",
        (size * 100.0).round(),
        0.0..=100.0,
        |v| format!("{v:.0}"),
    ) {
        shell.set_brush_size(value / 100.0);
    }
    ui.horizontal(|ui| {
        row_label(ui, "");
        if ui
            .add_enabled(strokes > 0, ui::Button::new("Undo stroke").text_size(12.0))
            .clicked()
        {
            shell.undo_stroke();
        }
        if ui.add(ui::Button::new("Cancel").text_size(12.0)).clicked() {
            shell.stop_painting();
        }
        if ui
            .add_enabled(strokes > 0, ui::Button::new("Apply").text_size(12.0))
            .clicked()
        {
            send(shell, Shell::apply_painting);
        }
    });
    ui.add_space(ROW_GAP);
}

/// Chroma key: the switch keys green out or takes the key off; the colour,
/// intensity and shadow edit it. Feather edge and Clean up edge come later.
fn chroma_key(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    let key = inspection.chroma;
    let header = section(
        ui,
        state,
        "chroma",
        "Chroma key",
        Some((key.is_some(), true)),
        key.is_some(),
    );
    if header.toggled {
        if key.is_some() {
            send(shell, |shell| shell.set_chroma("", 0.0, 0.0));
        } else {
            state.open.insert("chroma".to_owned());
            send(shell, |shell| {
                shell.set_chroma("#00ff00", KEY_STRENGTH, 0.0)
            });
        }
    }
    let Some(key) = key else {
        // Off, the controls show dimmed at the values the switch starts
        // with, as in the reference.
        if header.open {
            body(ui, |ui| {
                ui.add_enabled_ui(false, |ui| {
                    ui.horizontal(|ui| {
                        row_label(ui, "Color picker");
                        ui.label(RichText::new("Green").size(12.0).color(tokens::MUTED));
                    });
                    ui.add_space(ROW_GAP);
                    later_slider(ui, "Intensity", KEY_STRENGTH * 100.0);
                    later_slider(ui, "Shadow", 0.0);
                    later_slider(ui, "Feather edge", 0.0);
                    later_slider(ui, "Clean up edge", 0.0);
                })
                .response
                .on_disabled_hover_text("Switch Chroma key on to key out a colour.");
            });
        }
        return;
    };
    if header.reset {
        let hex = key.hex();
        send(shell, |shell| {
            shell.put_chroma(&hex, [KEY_STRENGTH, 0.0, 0.0, 0.0])
        });
    }
    if !header.open && !state.open.contains("chroma") {
        return;
    }
    body(ui, |ui| {
        let (strength, shadow) = (f64::from(key.strength), f64::from(key.shadow));
        if let Some(color) = color_row(ui, state, "chroma-color", "Color picker", &key.hex()) {
            send(shell, |shell| shell.set_chroma(&color, strength, shadow));
        }
        ui.horizontal(|ui| {
            row_label(ui, "");
            for (name, hex) in [("Green", "#00ff00"), ("Blue", "#0000ff")] {
                if ui.add(ui::Button::new(name).text_size(12.0)).clicked() {
                    send(shell, |shell| shell.set_chroma(hex, strength, shadow));
                }
            }
        });
        ui.add_space(ROW_GAP);
        let hex = key.hex();
        let changed = setting_row(shell, ui, KeyedSetting::ChromaStrength, true, |ui| {
            slider(
                ui,
                state,
                "chroma-strength",
                "Intensity",
                (strength * 100.0).round(),
                0.0..=100.0,
                |v| format!("{v:.0}"),
            )
        });
        if let Some(value) = changed {
            send(shell, |shell| shell.set_chroma(&hex, value / 100.0, shadow));
        }
        let changed = setting_row(shell, ui, KeyedSetting::ChromaShadow, true, |ui| {
            slider(
                ui,
                state,
                "chroma-shadow",
                "Shadow",
                (shadow * 100.0).round(),
                0.0..=100.0,
                |v| format!("{v:.0}"),
            )
        });
        if let Some(value) = changed {
            send(shell, |shell| {
                shell.set_chroma(&hex, strength, value / 100.0)
            });
        }
        let (feather, clean_up) = (f64::from(key.feather), f64::from(key.clean_up));
        if let Some(value) = slider(
            ui,
            state,
            "chroma-feather",
            "Feather edge",
            (feather * 100.0).round(),
            0.0..=100.0,
            |v| format!("{v:.0}"),
        ) {
            send(shell, |shell| {
                shell.set_chroma_edge(value / 100.0, clean_up)
            });
        }
        if let Some(value) = slider(
            ui,
            state,
            "chroma-clean-up",
            "Clean up edge",
            (clean_up * 100.0).round(),
            0.0..=100.0,
            |v| format!("{v:.0}"),
        ) {
            send(shell, |shell| shell.set_chroma_edge(feather, value / 100.0));
        }
    });
}
