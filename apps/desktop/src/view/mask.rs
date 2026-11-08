//! The Mask page, laid out as the reference lays it out: the Mask switch,
//! the mask's name, a tile for each shape, Mask settings, and Track mask.
//!
//! On a video clip's Video tab the mask cuts the clip; on its Adjust tab
//! it keeps the colour work inside the shape. Text, brush and pen masks,
//! more than one mask, and tracking a mask come later, dimmed and marked.

use egui::{Align, Align2, FontId, Layout, Pos2, Rect, RichText, Sense, Shape, Stroke, Ui, Vec2};

use cutlass_effects::{Mask, MaskShape};

use crate::shell::{KeyedSetting, Shell, ShellError};
use crate::{icons, tokens, ui};

use super::audio::{choice_row, right_button};
use super::inspector::{
    InspectorState, LATER, ROW_GAP, body, keyed_row, later_badge, pair_row, section, send,
    setting_step, slider, tile_frame,
};

/// A shape tile's side, and the room under it for its name.
const TILE: f32 = 56.0;
const NAME: f32 = 18.0;

/// The shapes Cutlass draws, by the reference's names.
const SHAPES: [(MaskShape, &str); 6] = [
    (MaskShape::Linear, "Split"),
    (MaskShape::Mirror, "Filmstrip"),
    (MaskShape::Circle, "Circle"),
    (MaskShape::Rectangle, "Rectangle"),
    (MaskShape::Star, "Stars"),
    (MaskShape::Heart, "Heart"),
];

/// The shapes still to come.
const LATER_SHAPES: [&str; 3] = ["Text", "Brush", "Pen"];

type SetMask = fn(&mut Shell, Option<Mask>) -> Result<(), ShellError>;

pub(super) fn mask_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    mask: Option<Mask>,
    colour: bool,
) {
    let set: SetMask = if colour {
        Shell::set_color_mask
    } else {
        Shell::set_mask
    };
    let prefix = if colour { "color-" } else { "" };
    let key = format!("{prefix}mask");
    // The section starts open, and stays as it is folded after that.
    if state.open.insert(format!("{key} seen")) {
        state.open.insert(key.clone());
    }
    let header = section(ui, state, &key, "Mask", Some((mask.is_some(), true)), false);
    if header.toggled {
        let next = if mask.is_some() {
            None
        } else {
            Some(Mask::new(MaskShape::Rectangle))
        };
        send(shell, |shell| set(shell, next));
    }
    body(ui, |ui| {
        names(ui, mask);
        if let Some(shape) = shape_tiles(ui, mask) {
            let next = mask.map_or_else(|| Mask::new(shape), |mask| Mask { shape, ..mask });
            send(shell, |shell| set(shell, Some(next)));
        }
    });
    let Some(mask) = mask else {
        ui.label(
            RichText::new(if colour {
                "Pick a shape to keep the colour work inside it."
            } else {
                "Pick a shape to mask the clip."
            })
            .size(12.0)
            .color(tokens::MUTED),
        );
        ui.add_space(ROW_GAP);
        track_mask(ui);
        return;
    };
    if let Some(next) = settings(shell, ui, state, mask, prefix, colour) {
        send(shell, |shell| set(shell, Some(next)));
    }
    track_mask(ui);
}

/// The mask's name as a chip, and + for another mask, which comes later.
fn names(ui: &mut Ui, mask: Option<Mask>) {
    ui.horizontal(|ui| {
        if let Some(mask) = mask {
            let name = SHAPES
                .iter()
                .find(|(shape, _)| *shape == mask.shape)
                .map_or("", |(_, name)| name);
            ui.add(ui::Choice::new(
                true,
                RichText::new(format!("Mask1 {name}")).size(12.0),
            ));
        }
        ui.add_enabled_ui(false, |ui| {
            ui.add(
                ui::Button::icon("plus")
                    .min_size(Vec2::splat(24.0))
                    .tip("Add a mask"),
            );
            later_badge(ui);
        })
        .response
        .on_disabled_hover_text(LATER);
    });
    ui.add_space(ROW_GAP);
}

/// A tile for each shape, the mask's own ringed; the picked shape.
fn shape_tiles(ui: &mut Ui, mask: Option<Mask>) -> Option<MaskShape> {
    let mut picked = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);
        for (shape, name) in SHAPES {
            let on = mask.is_some_and(|mask| mask.shape == shape);
            if shape_tile(ui, name, on, true).clicked() && !on {
                picked = Some(shape);
            }
        }
        for name in LATER_SHAPES {
            shape_tile(ui, name, false, false).on_hover_text(LATER);
        }
    });
    ui.add_space(ROW_GAP);
    picked
}

/// One shape: its outline in a tile and its name under it, all of it a
/// button. A shape still to come is dimmed.
fn shape_tile(ui: &mut Ui, name: &str, on: bool, live: bool) -> egui::Response {
    let sense = if live { Sense::click() } else { Sense::hover() };
    let (whole, response) = ui.allocate_exact_size(Vec2::new(TILE, TILE + NAME), sense);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, live, name));
    let rect = Rect::from_min_size(whole.min, Vec2::splat(TILE));
    tile_frame(ui, rect, &response, on);
    let ink = if live { tokens::TEXT } else { tokens::FAINT };
    glyph(ui, rect.shrink(14.0), name, ink);
    ui.painter().text(
        Pos2::new(rect.center().x, rect.max.y + NAME / 2.0 + 1.0),
        Align2::CENTER_CENTER,
        name,
        FontId::proportional(11.0),
        if live { tokens::MUTED } else { tokens::FAINT },
    );
    response
}

/// The outline drawn on a shape's tile.
fn glyph(ui: &Ui, rect: Rect, name: &str, ink: egui::Color32) {
    let painter = ui.painter();
    let stroke = Stroke::new(1.5, ink);
    let frame = |painter: &egui::Painter| {
        painter.rect_stroke(rect, 2.0, stroke, egui::StrokeKind::Middle);
    };
    let across = |painter: &egui::Painter, t: f32| {
        let y = rect.min.y + rect.height() * t;
        painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], stroke);
    };
    match name {
        "Split" => {
            frame(painter);
            across(painter, 0.5);
        }
        "Filmstrip" => {
            frame(painter);
            across(painter, 0.25);
            across(painter, 0.75);
        }
        "Circle" => {
            frame(painter);
            painter.circle_stroke(rect.center(), rect.width() * 0.32, stroke);
        }
        "Rectangle" => {
            frame(painter);
            painter.rect_stroke(
                rect.shrink(rect.width() * 0.2),
                1.0,
                stroke,
                egui::StrokeKind::Middle,
            );
        }
        "Stars" => {
            frame(painter);
            let (centre, outer) = (rect.center(), rect.width() * 0.34);
            let points = (0..10)
                .map(|step| {
                    #[allow(clippy::cast_precision_loss)]
                    let angle =
                        std::f32::consts::FRAC_PI_2 + step as f32 * std::f32::consts::PI / 5.0;
                    let reach = if step % 2 == 0 { outer } else { outer * 0.45 };
                    Pos2::new(
                        centre.x + angle.cos() * reach,
                        centre.y - angle.sin() * reach,
                    )
                })
                .collect();
            painter.add(Shape::closed_line(points, stroke));
        }
        "Heart" => {
            frame(painter);
            let (centre, size) = (rect.center(), rect.width() * 0.024);
            let points = (0..48)
                .map(|step| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = step as f32 / 48.0 * std::f32::consts::TAU;
                    let x = 16.0 * t.sin().powi(3);
                    let y = 13.0 * t.cos()
                        - 5.0 * (2.0 * t).cos()
                        - 2.0 * (3.0 * t).cos()
                        - (4.0 * t).cos();
                    Pos2::new(centre.x + x * size, centre.y - y * size)
                })
                .collect();
            painter.add(Shape::closed_line(points, stroke));
        }
        "Text" => {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "T",
                FontId::proportional(24.0),
                ink,
            );
        }
        _ => {
            let _ = icons::paint(painter, rect, "draw", ink);
        }
    }
}

/// Mask settings: invert, reset and keyframes on the heading, then
/// position, rotation, size, feather and round corners. The changed mask.
fn settings(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    mask: Mask,
    prefix: &str,
    colour: bool,
) -> Option<Mask> {
    let key = format!("{prefix}mask-settings");
    if state.open.insert(format!("{key} seen")) {
        state.open.insert(key.clone());
    }
    let mut changed = None;
    let keys = (!colour)
        .then(|| shell.setting_keys(KeyedSetting::Mask))
        .flatten();
    let playhead = shell.playhead();
    let mut step = None;
    let mut open = state.open.contains(&key);
    keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
        ui.horizontal(|ui| {
            ui.set_min_height(36.0);
            let name = ui.add(
                egui::Label::new(
                    RichText::new("Mask settings")
                        .size(13.0)
                        .color(tokens::TEXT),
                )
                .selectable(false)
                .sense(Sense::click()),
            );
            let name = name.union(super::inspector::chevron(ui, open));
            if name.clicked() {
                open = !open;
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(
                        ui::Button::icon("reset")
                            .min_size(Vec2::splat(24.0))
                            .tip("Reset the mask"),
                    )
                    .clicked()
                {
                    changed = Some(Mask::new(mask.shape));
                }
                if ui
                    .add(ui::Button::new("Invert").text_size(12.0).on(mask.invert))
                    .on_hover_text("Keep what is outside the shape")
                    .clicked()
                {
                    changed = Some(Mask {
                        invert: !mask.invert,
                        ..mask
                    });
                }
            });
        });
    });
    if !colour {
        setting_step(shell, KeyedSetting::Mask, step);
    }
    if open {
        state.open.insert(key);
    } else {
        state.open.remove(&key);
        return changed;
    }
    let percent = |v: f32| (f64::from(v) * 100.0).round();
    body(ui, |ui| {
        let mut next = changed.unwrap_or(mask);
        let (x, y) = pair_row(
            ui,
            state,
            "Position",
            ("X", percent(mask.x)),
            ("Y", percent(mask.y)),
        );
        #[allow(clippy::cast_possible_truncation)]
        let part = |v: f64| (v / 100.0) as f32;
        if let Some(x) = x {
            next.x = part(x.clamp(-200.0, 200.0));
        }
        if let Some(y) = y {
            next.y = part(y.clamp(-200.0, 200.0));
        }
        let (turn, _) = pair_row(
            ui,
            state,
            "Rotation",
            ("°", f64::from(mask.rotation).round()),
            ("", f64::NAN),
        );
        if let Some(turn) = turn {
            #[allow(clippy::cast_possible_truncation)]
            let turn = turn.clamp(-180.0, 180.0) as f32;
            next.rotation = turn;
        }
        match mask.shape {
            MaskShape::Linear => {}
            MaskShape::Mirror => {
                let (band, _) = pair_row(
                    ui,
                    state,
                    "Size",
                    ("H", percent(mask.height)),
                    ("", f64::NAN),
                );
                if let Some(band) = band {
                    next.height = part(band.clamp(0.0, 400.0));
                }
            }
            _ => {
                let (width, height) = pair_row(
                    ui,
                    state,
                    "Size",
                    ("W", percent(mask.width)),
                    ("H", percent(mask.height)),
                );
                if let Some(width) = width {
                    next.width = part(width.clamp(0.0, 400.0));
                }
                if let Some(height) = height {
                    next.height = part(height.clamp(0.0, 400.0));
                }
            }
        }
        if let Some(value) = slider(
            ui,
            state,
            &format!("{prefix}mask-feather"),
            "Feather",
            percent(mask.feather),
            0.0..=100.0,
            |v| format!("{v:.0}"),
        ) {
            next.feather = part(value);
        }
        if mask.shape == MaskShape::Rectangle
            && let Some(value) = slider(
                ui,
                state,
                &format!("{prefix}mask-round"),
                "Round corners",
                percent(mask.round),
                0.0..=100.0,
                |v| format!("{v:.0}"),
            )
        {
            next.round = part(value);
        }
        if next != mask {
            changed = Some(next);
        }
    });
    super::inspector::divider(ui);
    changed
}

/// Track mask: which way to track and Track, both still to come.
fn track_mask(ui: &mut Ui) {
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            ui.set_min_height(36.0);
            ui.label(RichText::new("Track mask").size(13.0));
            later_badge(ui);
        });
        choice_row(ui, Some("Direction"), "Both");
        right_button(ui, "Track");
    })
    .response
    .on_disabled_hover_text(LATER);
}
