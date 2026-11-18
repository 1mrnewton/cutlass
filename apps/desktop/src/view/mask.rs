//! The Mask page, laid out as the reference lays it out: the Mask switch,
//! the clip's masks as a list with the one in hand ringed, a tile for each
//! shape, how the mask joins the ones before it, its drawing for a text,
//! brush or pen mask, Mask settings, and Track mask.
//!
//! On a video clip's Video tab the masks cut the clip; on its Adjust tab
//! they keep the colour work inside them. The mask in hand shows its
//! handles on the preview (`view.rs`, `mask_layer`). Track mask is in
//! `follow.rs`.

use egui::{Align, Align2, FontId, Layout, Pos2, Rect, RichText, Sense, Shape, Stroke, Ui, Vec2};

use cutlass_effects::{Mask, MaskArt, MaskMode, MaskShape};

use crate::shell::{KeyedSetting, MaskDraw, MaskRow, Shell};
use crate::{icons, tokens, ui};

use super::inspector::{
    InspectorState, ROW_GAP, body, keyed_row, pair_row, section, send, setting_step, slider,
    tile_frame,
};

/// A shape tile's side, and the room under it for its name.
const TILE: f32 = 56.0;
const NAME: f32 = 18.0;

/// The shapes, by the reference's names.
const SHAPES: [(MaskShape, &str); 9] = [
    (MaskShape::Linear, "Split"),
    (MaskShape::Mirror, "Filmstrip"),
    (MaskShape::Circle, "Circle"),
    (MaskShape::Rectangle, "Rectangle"),
    (MaskShape::Star, "Stars"),
    (MaskShape::Heart, "Heart"),
    (MaskShape::Text, "Text"),
    (MaskShape::Brush, "Brush"),
    (MaskShape::Pen, "Pen"),
];

/// A brush's width on the canvas, in pixels, when it is first picked up.
const BRUSH_SIZE: f64 = 60.0;
/// The words a new text mask spells.
const NEW_TEXT: &str = "Text";

/// A shape's name on the page.
fn shape_name(shape: MaskShape) -> &'static str {
    SHAPES
        .iter()
        .find(|(each, _)| *each == shape)
        .map_or("", |(_, name)| name)
}

pub(super) fn mask_page(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, colour: bool) {
    shell.note_mask_page();
    let focus = shell.mask_focus();
    if focus.color != colour {
        shell.focus_mask(colour, 0);
    }
    let rows = shell.mask_rows(colour);
    let index = shell.mask_focus().index;
    let mask = rows.get(index).map(|row| row.mask);
    let prefix = if colour { "color-" } else { "" };
    let key = format!("{prefix}mask");
    // The section starts open, and stays as it is folded after that.
    if state.open.insert(format!("{key} seen")) {
        state.open.insert(key.clone());
    }
    let header = section(
        ui,
        state,
        &key,
        "Mask",
        Some((!rows.is_empty(), true)),
        false,
    );
    if header.toggled {
        if rows.is_empty() {
            send(shell, |shell| {
                shell.add_mask(colour, Mask::new(MaskShape::Rectangle), None)
            });
        } else {
            send(shell, |shell| shell.clear_masks(colour));
        }
    }
    body(ui, |ui| {
        mask_list(shell, ui, &rows, index, colour);
        if let Some(shape) = shape_tiles(ui, mask) {
            send(shell, |shell| pick_shape(shell, colour, mask, shape));
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
        super::follow::track_mask(shell, ui, state, colour);
        return;
    };
    modes(shell, ui, mask);
    drawing(shell, ui, state, &rows[index], prefix);
    if let Some(next) = settings(shell, ui, state, mask, prefix) {
        send(shell, |shell| shell.set_focused_mask(Some(next)));
    }
    super::follow::track_mask(shell, ui, state, colour);
}

/// Change the mask in hand to `shape`, or with none yet put one on. A
/// brush or pen mask covers the clip and waits to be drawn; a text mask
/// spells a word to start with.
fn pick_shape(
    shell: &mut Shell,
    colour: bool,
    mask: Option<Mask>,
    shape: MaskShape,
) -> Result<(), crate::shell::ShellError> {
    let kept = |mut next: Mask| {
        if let Some(mask) = mask {
            next.mode = mask.mode;
            next.invert = mask.invert;
            next.feather = mask.feather;
        }
        next
    };
    let next = match shape {
        MaskShape::Text => {
            let art = shell
                .text_mask_art(&shell.mask_font(), NEW_TEXT)
                .ok_or_else(|| {
                    crate::shell::ShellError::Unavailable("no face can set the text".to_owned())
                })?;
            let MaskArt::Text { aspect, .. } = &art else {
                return Ok(());
            };
            let next = kept(shell.new_mask(MaskShape::Text, Some(*aspect)));
            shell.set_mask_draw(None);
            return shell.set_focused_drawn(next, art);
        }
        MaskShape::Brush => {
            shell.set_mask_draw(Some(MaskDraw::Brush {
                size: BRUSH_SIZE,
                erase: false,
            }));
            kept(Mask::new(shape))
        }
        MaskShape::Pen => {
            shell.set_mask_draw(Some(MaskDraw::Pen(cutlass_editor::PenPath::new())));
            kept(Mask::new(shape))
        }
        _ => {
            shell.set_mask_draw(None);
            match mask {
                Some(mask) if !mask.shape.is_drawn() => Mask { shape, ..mask },
                _ => kept(Mask::new(shape)),
            }
        }
    };
    match mask {
        Some(_) => shell.set_focused_mask(Some(next)),
        None => shell.add_mask(colour, next, None),
    }
}

/// The clip's masks as chips, the one in hand ringed, with + for another,
/// and for the one in hand, up, down and delete.
fn mask_list(shell: &mut Shell, ui: &mut Ui, rows: &[MaskRow], index: usize, colour: bool) {
    ui.horizontal_wrapped(|ui| {
        for (at, row) in rows.iter().enumerate() {
            let name = format!("Mask{} {}", at + 1, shape_name(row.mask.shape));
            if ui
                .add(ui::Choice::new(at == index, RichText::new(name).size(12.0)))
                .clicked()
            {
                shell.focus_mask(colour, at);
                shell.set_mask_draw(None);
            }
        }
        let full = rows.len() >= cutlass_session::MAX_MASKS;
        let add = ui
            .add_enabled(
                !full,
                ui::Button::icon("plus")
                    .min_size(Vec2::splat(24.0))
                    .tip("Add a mask"),
            )
            .on_disabled_hover_text("A clip takes eight masks at most");
        if add.clicked() {
            send(shell, |shell| {
                shell.add_mask(colour, Mask::new(MaskShape::Rectangle), None)
            });
        }
    });
    if !rows.is_empty() {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(index > 0, ui::Button::new("Move up").text_size(12.0))
                .on_hover_text("Draw this mask earlier")
                .clicked()
            {
                send(shell, |shell| shell.move_focused_mask(true));
            }
            if ui
                .add_enabled(
                    index + 1 < rows.len(),
                    ui::Button::new("Move down").text_size(12.0),
                )
                .on_hover_text("Draw this mask later")
                .clicked()
            {
                send(shell, |shell| shell.move_focused_mask(false));
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(ui::Button::icon("delete"))
                    .on_hover_text("Delete this mask")
                    .clicked()
                {
                    send(shell, |shell| shell.set_focused_mask(None));
                }
            });
        });
    }
    ui.add_space(ROW_GAP);
}

/// How the mask in hand joins the masks before it.
fn modes(shell: &mut Shell, ui: &mut Ui, mask: Mask) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Combine").size(12.0).color(tokens::MUTED));
        for mode in MaskMode::ALL {
            let tip = match mode {
                MaskMode::Add => "Keep what this mask covers too",
                MaskMode::Subtract => "Take what this mask covers away",
                MaskMode::Intersect => "Keep only where this mask and those before it meet",
            };
            if ui
                .add(ui::Choice::new(
                    mask.mode == mode,
                    RichText::new(mode.label()).size(12.0),
                ))
                .on_hover_text(tip)
                .clicked()
                && mask.mode != mode
            {
                send(shell, |shell| shell.set_mask_mode(mode));
            }
        }
    });
    ui.add_space(ROW_GAP);
}

/// A text mask's words, a brush mask's brush, or a pen mask's outline.
fn drawing(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    row: &MaskRow,
    prefix: &str,
) {
    match (row.mask.shape, &row.art) {
        (MaskShape::Text, art) => {
            let spelled = match art {
                Some(MaskArt::Text { text, .. }) => text.clone(),
                _ => String::new(),
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new("Text").size(12.0).color(tokens::MUTED));
                let key = format!("{prefix}mask text");
                let text = state.fields.entry(key).or_default();
                let response = ui.add(
                    egui::TextEdit::singleline(text)
                        .desired_width(f32::INFINITY)
                        .char_limit(cutlass_effects::MAX_MASK_TEXT),
                );
                if super::commits(ui, &response) && *text != spelled && !text.trim().is_empty() {
                    let words = text.clone();
                    send(shell, |shell| shell.set_mask_text(&words));
                } else if !response.has_focus() {
                    spelled.clone_into(text);
                }
            });
        }
        (MaskShape::Brush, art) => {
            let drawing = shell.mask_draw().cloned();
            let (size, erase) = match &drawing {
                Some(MaskDraw::Brush { size, erase }) => (*size, *erase),
                _ => (BRUSH_SIZE, false),
            };
            ui.horizontal(|ui| {
                let painting = matches!(drawing, Some(MaskDraw::Brush { .. }));
                for (rubbing, label, tip) in [
                    (false, "Paint", "Paint on the preview to keep"),
                    (true, "Erase", "Rub out on the preview"),
                ] {
                    if ui
                        .add(ui::Choice::new(
                            painting && erase == rubbing,
                            RichText::new(label).size(12.0),
                        ))
                        .on_hover_text(tip)
                        .clicked()
                    {
                        shell.set_mask_draw(Some(MaskDraw::Brush {
                            size,
                            erase: rubbing,
                        }));
                    }
                }
                if painting
                    && ui
                        .add(ui::Button::new("Done").text_size(12.0))
                        .on_hover_text("Stop painting and show the handles")
                        .clicked()
                {
                    shell.set_mask_draw(None);
                }
                let strokes = match art {
                    Some(MaskArt::Brush { strokes }) => strokes.len(),
                    _ => 0,
                };
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add_enabled(strokes > 0, ui::Button::new("Clear").text_size(12.0))
                        .on_hover_text("Take every stroke off")
                        .clicked()
                    {
                        send(shell, |shell| {
                            shell.set_mask_art(MaskArt::Brush {
                                strokes: Vec::new(),
                            })
                        });
                    }
                });
            });
            if let Some(next) = slider(
                ui,
                state,
                &format!("{prefix}mask-brush"),
                "Brush size",
                size,
                4.0..=400.0,
                |v| format!("{v:.0}"),
            ) {
                shell.set_mask_draw(Some(MaskDraw::Brush { size: next, erase }));
            }
        }
        (MaskShape::Pen, art) => {
            let drawing = matches!(shell.mask_draw(), Some(MaskDraw::Pen(_)));
            ui.horizontal(|ui| {
                let label = if drawing {
                    "Stop drawing"
                } else if art.is_some() {
                    "Draw again"
                } else {
                    "Draw outline"
                };
                if ui
                    .add(ui::Button::new(label).text_size(12.0).on(drawing))
                    .on_hover_text("Click points on the preview; click the first to close")
                    .clicked()
                {
                    shell.set_mask_draw(
                        (!drawing).then(|| MaskDraw::Pen(cutlass_editor::PenPath::new())),
                    );
                }
            });
        }
        _ => return,
    }
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

/// Mask settings: invert and reset on the heading, then position,
/// rotation, size, feather and round corners, each with its own keyframe
/// buttons. The changed mask.
fn settings(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    mask: Mask,
    prefix: &str,
) -> Option<Mask> {
    let key = format!("{prefix}mask-settings");
    if state.open.insert(format!("{key} seen")) {
        state.open.insert(key.clone());
    }
    let mut changed = None;
    // The keys of the mask in hand, under its own names.
    let keys_of = |setting| shell.setting_keys(setting);
    let place_keys = keys_of(KeyedSetting::MaskPosition);
    let turn_keys = keys_of(KeyedSetting::MaskRotation);
    let size_keys = keys_of(KeyedSetting::MaskSize);
    let feather_keys = keys_of(KeyedSetting::MaskFeather);
    let round_keys = keys_of(KeyedSetting::MaskRound);
    let playhead = shell.playhead();
    let mut steps = [None; 5];
    let mut open = state.open.contains(&key);
    {
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
                    changed = Some(Mask {
                        mode: mask.mode,
                        ..Mask::new(mask.shape)
                    });
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
        let (x, y) = keyed_row(ui, place_keys.as_ref(), playhead, &mut steps[0], |ui| {
            pair_row(
                ui,
                state,
                "Position",
                ("X", percent(mask.x)),
                ("Y", percent(mask.y)),
            )
        });
        #[allow(clippy::cast_possible_truncation)]
        let part = |v: f64| (v / 100.0) as f32;
        if let Some(x) = x {
            next.x = part(x.clamp(-200.0, 200.0));
        }
        if let Some(y) = y {
            next.y = part(y.clamp(-200.0, 200.0));
        }
        let (turn, _) = keyed_row(ui, turn_keys.as_ref(), playhead, &mut steps[1], |ui| {
            pair_row(
                ui,
                state,
                "Rotation",
                ("°", f64::from(mask.rotation).round()),
                ("", f64::NAN),
            )
        });
        if let Some(turn) = turn {
            #[allow(clippy::cast_possible_truncation)]
            let turn = turn.clamp(-180.0, 180.0) as f32;
            next.rotation = turn;
        }
        match mask.shape {
            MaskShape::Linear => {}
            MaskShape::Mirror => {
                let (band, _) = keyed_row(ui, size_keys.as_ref(), playhead, &mut steps[2], |ui| {
                    pair_row(
                        ui,
                        state,
                        "Size",
                        ("H", percent(mask.height)),
                        ("", f64::NAN),
                    )
                });
                if let Some(band) = band {
                    next.height = part(band.clamp(0.0, 400.0));
                }
            }
            _ => {
                let (width, height) =
                    keyed_row(ui, size_keys.as_ref(), playhead, &mut steps[2], |ui| {
                        pair_row(
                            ui,
                            state,
                            "Size",
                            ("W", percent(mask.width)),
                            ("H", percent(mask.height)),
                        )
                    });
                if let Some(width) = width {
                    next.width = part(width.clamp(0.0, 400.0));
                }
                if let Some(height) = height {
                    next.height = part(height.clamp(0.0, 400.0));
                }
            }
        }
        if let Some(value) = keyed_row(ui, feather_keys.as_ref(), playhead, &mut steps[3], |ui| {
            slider(
                ui,
                state,
                &format!("{prefix}mask-feather"),
                "Feather",
                percent(mask.feather),
                0.0..=100.0,
                |v| format!("{v:.0}"),
            )
        }) {
            next.feather = part(value);
        }
        if mask.shape == MaskShape::Rectangle
            && let Some(value) = keyed_row(ui, round_keys.as_ref(), playhead, &mut steps[4], |ui| {
                slider(
                    ui,
                    state,
                    &format!("{prefix}mask-round"),
                    "Round corners",
                    percent(mask.round),
                    0.0..=100.0,
                    |v| format!("{v:.0}"),
                )
            })
        {
            next.round = part(value);
        }
        if next != mask {
            changed = Some(next);
        }
    });
    let parts = [
        KeyedSetting::MaskPosition,
        KeyedSetting::MaskRotation,
        KeyedSetting::MaskSize,
        KeyedSetting::MaskFeather,
        KeyedSetting::MaskRound,
    ];
    for (part, step) in parts.into_iter().zip(steps) {
        setting_step(shell, part, step);
    }
    super::inspector::divider(ui);
    changed
}

/// How near, in screen points, a press has to be to a handle to take it.
const HANDLE_REACH: f32 = 9.0;

/// The mask in hand on the preview: its outline and handles, a drag on
/// one moving, sizing, turning or softening it as one undo step; or, while
/// a pen or brush mask is drawn, the points or strokes going down.
pub(super) fn mask_layer(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut super::ViewState,
    frame: Rect,
    placed: Rect,
    size: Vec2,
) {
    let (Some(mask), Some(clip)) = (shell.focused_mask(), shell.mask_frame()) else {
        view.mask_drag = None;
        return;
    };
    let clip = cutlass_editor::ClipFrame(clip);
    let area = placed.intersect(frame);
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    let to_canvas = |pos: Pos2| super::canvas_pixel(placed, size, pos);
    let to_screen = |point: (f64, f64)| super::screen_point(placed, size, point);
    // Screen points to canvas pixels.
    let per_point = f64::from(size.x / placed.width().max(1.0));
    match shell.mask_draw().cloned() {
        Some(MaskDraw::Pen(_)) => {
            pen_points(shell, ui, &painter, area, per_point, &to_canvas, &to_screen);
            return;
        }
        Some(MaskDraw::Brush { size: width, erase }) => {
            brush(
                shell, ui, view, &painter, area, width, erase, &to_canvas, &to_screen, per_point,
            );
            return;
        }
        None => {}
    }
    let Some(handles) = cutlass_editor::mask_handles(&mask, &clip) else {
        return;
    };
    let response = ui.interact(
        area,
        ui.scope_id().with("preview-mask"),
        Sense::click_and_drag(),
    );
    let reach = f64::from(HANDLE_REACH) * per_point;
    let hover = response
        .hover_pos()
        .map(to_canvas)
        .and_then(|(x, y)| cutlass_editor::mask_pick(&mask, &clip, x, y, reach));
    if response.drag_started()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (x, y) = to_canvas(pos);
        let focus = shell.mask_focus();
        view.mask_drag = cutlass_editor::mask_pick(&mask, &clip, x, y, reach).and_then(|handle| {
            let draft = shell.draft()?;
            let segment = shell.selected()?.clone();
            cutlass_editor::MaskGesture::begin(
                draft,
                &segment,
                (focus.index, focus.color),
                clip,
                shell.playhead(),
                handle,
                (x, y),
            )
        });
    }
    if let Some(gesture) = &view.mask_drag
        && response.dragged()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (x, y) = to_canvas(pos);
        let next = gesture.step(x, y);
        if next != mask {
            let requests = gesture.requests(&next, &cutlass_model::MaterialId::new("mask-drag"));
            send(shell, |shell| shell.apply_mask_requests(&requests));
        }
    }
    if response.drag_stopped() {
        view.mask_drag = None;
    }
    let cursor = match hover {
        Some(cutlass_editor::MaskHandle::Move) => egui::CursorIcon::Move,
        Some(cutlass_editor::MaskHandle::Turn) => egui::CursorIcon::Alias,
        Some(_) => egui::CursorIcon::Crosshair,
        None => egui::CursorIcon::Default,
    };
    response.on_hover_cursor(cursor);
    // The outline: the box, or a line's line.
    let line = Stroke::new(1.5, tokens::EXPORT);
    let thin = Stroke::new(1.0, egui::Color32::from_white_alpha(200));
    let corners: Vec<Pos2> = handles.corners.iter().map(|&c| to_screen(c)).collect();
    match handles.line {
        Some([a, b]) => {
            painter.line_segment([to_screen(a), to_screen(b)], line);
            painter.add(Shape::closed_line(corners.clone(), thin));
        }
        None => {
            painter.add(Shape::closed_line(corners.clone(), line));
        }
    }
    let middle = to_screen(handles.middle);
    let turn = to_screen(handles.turn);
    let feather = to_screen(handles.feather);
    let top = Pos2::new(
        f32::midpoint(corners[0].x, corners[1].x),
        f32::midpoint(corners[0].y, corners[1].y),
    );
    painter.line_segment([top, turn], thin);
    painter.circle(turn, 5.0, egui::Color32::WHITE, line);
    painter.circle_filled(middle, 3.5, tokens::EXPORT);
    let diamond = |at: Pos2, r: f32| {
        vec![
            Pos2::new(at.x, at.y - r),
            Pos2::new(at.x + r, at.y),
            Pos2::new(at.x, at.y + r),
            Pos2::new(at.x - r, at.y),
        ]
    };
    painter.add(Shape::convex_polygon(
        diamond(feather, 5.0),
        egui::Color32::WHITE,
        line,
    ));
    for corner in &corners {
        painter.rect(
            Rect::from_center_size(*corner, Vec2::splat(8.0)),
            1.0,
            egui::Color32::WHITE,
            line,
            egui::StrokeKind::Middle,
        );
    }
}

/// A pen mask's outline going down: each press a point, a drag bending
/// the curve, a press on the first point closing it into the mask.
fn pen_points(
    shell: &mut Shell,
    ui: &mut Ui,
    painter: &egui::Painter,
    area: Rect,
    per_point: f64,
    to_canvas: &dyn Fn(Pos2) -> (f64, f64),
    to_screen: &dyn Fn((f64, f64)) -> Pos2,
) {
    let response = ui
        .interact(
            area,
            ui.scope_id().with("preview-mask-pen"),
            Sense::click_and_drag(),
        )
        .on_hover_cursor(egui::CursorIcon::Crosshair);
    let reach = 8.0 * per_point;
    let at = |pos: Pos2| {
        let (x, y) = to_canvas(pos);
        [x, y]
    };
    let (pressed, down, released, pointer) = ui.input(|input| {
        (
            input.pointer.primary_pressed(),
            input.pointer.primary_down(),
            input.pointer.primary_released(),
            input.pointer.interact_pos(),
        )
    });
    let mut closed = false;
    if let Some(MaskDraw::Pen(pen)) = shell.mask_draw_mut() {
        if pressed
            && response.hovered()
            && let Some(pos) = pointer
        {
            closed = pen.press(at(pos), reach) == Some(cutlass_editor::PenPress::Closed);
        } else if down
            && pen.is_pulling()
            && let Some(pos) = pointer
        {
            pen.drag(at(pos));
        }
        if released && pen.is_pulling() {
            pen.release();
            closed = pen.is_closed();
        }
    }
    if closed && !down {
        send(shell, Shell::finish_mask_pen);
        return;
    }
    let Some(MaskDraw::Pen(pen)) = shell.mask_draw() else {
        return;
    };
    hint(
        painter,
        area,
        if pen.points().is_empty() {
            "Click to put a point down, drag to curve it"
        } else {
            "Click the first point to close the mask"
        },
    );
    let screen = |p: [f64; 2]| to_screen((p[0], p[1]));
    let stroke = Stroke::new(1.5, tokens::EXPORT);
    for cubic in pen.cubics() {
        painter.add(Shape::CubicBezier(
            egui::epaint::CubicBezierShape::from_points_stroke(
                cubic.map(screen),
                false,
                egui::Color32::TRANSPARENT,
                stroke,
            ),
        ));
    }
    for (index, point) in pen.points().iter().enumerate() {
        let fill = if index == 0 {
            tokens::EXPORT
        } else {
            egui::Color32::WHITE
        };
        painter.rect(
            Rect::from_center_size(screen(point.at), Vec2::splat(8.0)),
            1.0,
            fill,
            stroke,
            egui::StrokeKind::Middle,
        );
    }
}

/// A brush mask being painted: a drag lays a stroke down, kept when the
/// pointer lets go.
#[allow(clippy::too_many_arguments)]
fn brush(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut super::ViewState,
    painter: &egui::Painter,
    area: Rect,
    width: f64,
    erase: bool,
    to_canvas: &dyn Fn(Pos2) -> (f64, f64),
    to_screen: &dyn Fn((f64, f64)) -> Pos2,
    per_point: f64,
) {
    let response = ui
        .interact(
            area,
            ui.scope_id().with("preview-mask-brush"),
            Sense::drag(),
        )
        .on_hover_cursor(egui::CursorIcon::Crosshair);
    if response.drag_started() {
        view.brush_trace.clear();
    }
    if (response.drag_started() || response.dragged())
        && let Some(pos) = response.interact_pointer_pos()
    {
        view.brush_trace.push(to_canvas(pos));
    }
    if response.drag_stopped() {
        let trace = std::mem::take(&mut view.brush_trace);
        send(shell, |shell| shell.paint_mask_stroke(&trace));
    }
    hint(
        painter,
        area,
        if erase {
            "Drag to rub out"
        } else {
            "Drag to paint what the mask keeps"
        },
    );
    #[allow(clippy::cast_possible_truncation)]
    let radius = (width / per_point / 2.0) as f32;
    let colour = if erase {
        egui::Color32::from_rgba_unmultiplied(255, 90, 90, 120)
    } else {
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 110)
    };
    let points: Vec<Pos2> = view.brush_trace.iter().map(|&p| to_screen(p)).collect();
    if points.len() > 1 {
        painter.add(Shape::line(
            points.clone(),
            Stroke::new(radius * 2.0, colour),
        ));
    }
    for point in points.first().into_iter().chain(points.last()) {
        painter.circle_filled(*point, radius, colour);
    }
    if let Some(pos) = response.hover_pos() {
        painter.circle_stroke(pos, radius, Stroke::new(1.0, egui::Color32::WHITE));
    }
}

/// A short line of help at the top of the preview.
fn hint(painter: &egui::Painter, area: Rect, text: &str) {
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        FontId::proportional(12.0),
        egui::Color32::WHITE,
    );
    let chip = Rect::from_center_size(
        Pos2::new(area.center().x, area.top() + 18.0),
        galley.size() + Vec2::new(16.0, 8.0),
    );
    painter.rect_filled(chip, 6.0, egui::Color32::from_black_alpha(170));
    painter.galley(
        chip.center() - galley.size() / 2.0,
        galley,
        egui::Color32::WHITE,
    );
}
