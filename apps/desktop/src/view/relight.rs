//! A video clip's Relight section: a tab for each group of lights and one
//! for the Custom light, each a row of tiles after None, every tile a ball
//! lit as the preset lights the clip; then the strength, and for Custom its
//! color and the angle it comes from.
//!
//! Picking a light stores it at once. The first light on a clip guesses its
//! depth as a job, whose progress shows here; the clip draws lit once it
//! lands.

use egui::{Align2, Color32, FontId, Rect, Sense, Ui, Vec2};

use cutlass_effects::{RELIGHT_PRESETS, RelightGroup, lighting, relight_preset, relight_tile};
use cutlass_model::Relight;

use crate::shell::{KeyedSetting, Shell};
use crate::{tokens, ui};

use super::inspector::{
    CATALOG_TILE, InspectorState, body, color_row, none_tile, note, section, send, setting_row,
    slider, tile_frame,
};

/// The tabs: each group, then Custom.
const TABS: [&str; 4] = ["Facial", "Ambient", "Creative", "Custom"];

/// The pixels each tile's picture is drawn at, twice its side for a sharp
/// tile.
const TILE_PIXELS: u32 = (CATALOG_TILE * 2.0) as u32;

/// What a fresh light starts at.
const STRENGTH: f64 = 1.0;

/// The tab a light is under.
fn tab_of(relight: Option<&Relight>) -> usize {
    let Some(relight) = relight else {
        return 0;
    };
    if relight.preset == Relight::CUSTOM {
        return TABS.len() - 1;
    }
    relight_preset(&relight.preset)
        .and_then(|preset| RelightGroup::ALL.iter().position(|g| *g == preset.group))
        .unwrap_or(0)
}

/// The Relight section of a video clip's Basic page.
pub(super) fn relight(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let worn = shell.relight();
    let header = section(
        ui,
        state,
        "relight",
        "Relight",
        Some((worn.is_some(), true)),
        false,
    );
    if header.toggled {
        match &worn {
            Some(_) => send(shell, |shell| shell.set_relight("", STRENGTH, "", 0.0)),
            None => {
                let first = RELIGHT_PRESETS[0].id;
                send(shell, move |shell| {
                    shell.set_relight(first, STRENGTH, "", Relight::DEFAULT_ANGLE)
                });
            }
        }
    }
    if !header.open {
        return;
    }
    let progress = shell.relight_progress();
    let tab = *state
        .chips
        .entry("relight")
        .or_insert_with(|| tab_of(worn.as_ref()));
    body(ui, |ui| {
        if let Some(done) = progress {
            note(ui, &format!("Reading depth {:.1}%", done * 100.0));
        }
        let mut picked_tab = tab;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for (index, name) in TABS.iter().enumerate() {
                let chip = ui::Button::new(*name)
                    .on(index == tab)
                    .text_size(12.0)
                    .radius(12);
                if ui.add(chip).clicked() {
                    picked_tab = index;
                }
            }
        });
        state.chips.insert("relight", picked_tab);
        ui.add_space(8.0);
        let strength = worn.as_ref().map_or(STRENGTH, |lit| lit.strength);
        let color = worn
            .as_ref()
            .map(|lit| lit.color.clone())
            .unwrap_or_default();
        let angle = worn
            .as_ref()
            .map_or(Relight::DEFAULT_ANGLE, |lit| lit.angle);
        let mut picked: Option<&'static str> = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            if none_tile(ui, worn.is_none()).clicked() && worn.is_some() {
                picked = Some("");
            }
            if picked_tab == TABS.len() - 1 {
                let custom = Relight {
                    preset: Relight::CUSTOM.into(),
                    strength: 1.0,
                    color: color.clone(),
                    angle,
                };
                let on = worn
                    .as_ref()
                    .is_some_and(|lit| lit.preset == Relight::CUSTOM);
                let key = format!("relight custom {color} {angle:.0}");
                if light_tile(ui, state, &key, "Custom", &custom, on).clicked() && !on {
                    picked = Some(Relight::CUSTOM);
                }
                return;
            }
            let group = RelightGroup::ALL[picked_tab.min(RelightGroup::ALL.len() - 1)];
            for preset in RELIGHT_PRESETS.iter().filter(|p| p.group == group) {
                let light = Relight {
                    preset: preset.id.into(),
                    strength: 1.0,
                    color: String::new(),
                    angle: Relight::DEFAULT_ANGLE,
                };
                let on = worn.as_ref().is_some_and(|lit| lit.preset == preset.id);
                let key = format!("relight {}", preset.id);
                if light_tile(ui, state, &key, preset.name, &light, on).clicked() && !on {
                    picked = Some(preset.id);
                }
            }
        });
        if let Some(preset) = picked {
            let color = if preset == Relight::CUSTOM {
                color.clone()
            } else {
                String::new()
            };
            send(shell, move |shell| {
                shell.set_relight(preset, strength, &color, angle)
            });
            return;
        }
        let Some(lit) = worn else {
            return;
        };
        ui.add_space(8.0);
        let changed = setting_row(shell, ui, KeyedSetting::RelightStrength, true, |ui| {
            slider(
                ui,
                state,
                "relight strength",
                "Strength",
                (lit.strength * 100.0).round(),
                0.0..=100.0,
                |v| format!("{v:.0}%"),
            )
        });
        if let Some(value) = changed {
            let (preset, color) = (lit.preset.clone(), lit.color.clone());
            send(shell, move |shell| {
                shell.set_relight(&preset, value / 100.0, &color, lit.angle)
            });
        }
        if lit.preset != Relight::CUSTOM {
            return;
        }
        let shown = if lit.color.is_empty() {
            let [r, g, b] =
                cutlass_effects::CUSTOM_COLOR.map(|c| (c.powf(1.0 / 2.2) * 255.0).round() as u8);
            format!("#{r:02X}{g:02X}{b:02X}")
        } else {
            lit.color.clone()
        };
        let picked = setting_row(shell, ui, KeyedSetting::RelightColor, true, |ui| {
            color_row(ui, state, "relight color", "Color", &shown)
        });
        if let Some(hex) = picked {
            let preset = lit.preset.clone();
            send(shell, move |shell| {
                shell.set_relight(&preset, lit.strength, &hex.to_ascii_lowercase(), lit.angle)
            });
        }
        let changed = setting_row(shell, ui, KeyedSetting::RelightAngle, true, |ui| {
            slider(
                ui,
                state,
                "relight angle",
                "Angle",
                lit.angle.round(),
                0.0..=360.0,
                |v| format!("{v:.0}°"),
            )
        });
        if let Some(value) = changed {
            let (preset, color) = (lit.preset.clone(), lit.color.clone());
            send(shell, move |shell| {
                shell.set_relight(&preset, lit.strength, &color, value)
            });
        }
    });
}

/// A light's tile: a ball lit by `light`, drawn once and kept under `key`,
/// its name along the bottom.
fn light_tile(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    name: &str,
    light: &Relight,
    on: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(CATALOG_TILE), Sense::click());
    let response = response.on_hover_text(name);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, name));
    if !state.tiles.contains_key(key)
        && ui.is_rect_visible(rect)
        && let Some(lit) = lighting(light, 0.5)
    {
        let rgba = relight_tile(&lit, TILE_PIXELS);
        let side = TILE_PIXELS as usize;
        let image = egui::ColorImage::from_rgba_unmultiplied([side, side], &rgba);
        let texture = ui
            .ctx()
            .load_texture(key.to_owned(), image, egui::TextureOptions::LINEAR);
        state.tiles.insert(key.to_owned(), texture);
    }
    tile_frame(ui, rect, &response, on);
    let inner = rect.shrink(2.0);
    let painter = ui.painter().with_clip_rect(inner);
    if let Some(texture) = state.tiles.get(key) {
        painter.image(
            texture.id(),
            inner,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    let band = Rect::from_min_max(egui::pos2(inner.left(), inner.bottom() - 14.0), inner.max);
    painter.rect_filled(band, 0.0, Color32::from_black_alpha(140));
    painter.text(
        band.center(),
        Align2::CENTER_CENTER,
        name,
        FontId::proportional(9.0),
        tokens::TEXT,
    );
    response
}
