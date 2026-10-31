//! The column of panels. egui draws the chrome. The preview is a texture of
//! the evaluated frame. This pass does not composite.

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Pos2, Rect, RichText, Sense, Ui, Vec2,
};

use crate::shell::Shell;
use crate::tokens;
use crate::widgets::{self, HeaderFlags};

pub struct ViewState {
    preview: Option<egui::TextureHandle>,
    export_open: bool,
    export_side: i32,
    export_codec: cutlass_decode::CodecId,
    replace_prompt: bool,
    export_path: String,
    text_buffer: String,
    text_for: String,
    volume_buffer: String,
    speed_buffer: String,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            preview: None,
            export_open: false,
            export_side: 1080,
            export_codec: cutlass_decode::CodecId::Png,
            replace_prompt: false,
            export_path: String::new(),
            text_buffer: String::new(),
            text_for: String::new(),
            volume_buffer: String::new(),
            speed_buffer: String::new(),
        }
    }
}

impl ViewState {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn show(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    style(ui);
    shell.chrome = crate::shell::Chrome::default();
    let full = ui.max_rect();
    ui.painter()
        .rect_filled(full, CornerRadius::ZERO, tokens::GAP);
    let sizes = sizes(shell, full.size());
    project_bar(shell, ui, view, full, sizes.head);
    let row = Rect::from_min_size(
        Pos2::new(full.min.x, full.min.y + sizes.head + tokens::SPLIT),
        Vec2::new(full.width(), sizes.row),
    );
    regions(shell, ui, view, row, sizes);
    let timeline = Rect::from_min_size(
        Pos2::new(full.min.x, row.max.y + tokens::SPLIT),
        Vec2::new(full.width(), sizes.timeline),
    );
    timeline_panel(shell, ui, timeline);
    if view.export_open {
        export_dialog(shell, ui, view);
    }
    if !shell.notice().is_empty() {
        ui.painter().text(
            full.left_bottom() + Vec2::new(12.0, -8.0),
            Align2::LEFT_BOTTOM,
            shell.notice(),
            FontId::proportional(tokens::TYPE_SIZE),
            tokens::MUTED,
        );
    }
    shortcuts(shell, ui);
}

#[derive(Clone, Copy)]
struct Sizes {
    head: f32,
    row: f32,
    timeline: f32,
    library: f32,
    inspector: f32,
    preview: f32,
}

fn sizes(shell: &Shell, window: Vec2) -> Sizes {
    let head = tokens::HEAD_HEIGHT;
    let timeline = shell.timeline_height().clamp(
        tokens::TIMELINE_MIN,
        (window.y - head - tokens::SPLIT * 2.0 - tokens::PREVIEW_MIN_HEIGHT)
            .max(tokens::TIMELINE_MIN),
    );
    let row = (window.y - head - tokens::SPLIT * 2.0 - timeline).max(0.0);
    let library = shell.library_width();
    let inspector = shell.inspector_width();
    let preview = (window.x - library - inspector - tokens::SPLIT * 2.0).max(0.0);
    Sizes {
        head,
        row,
        timeline,
        library,
        inspector,
        preview,
    }
}

fn style(ui: &mut Ui) {
    ui.style_mut().interaction.selectable_labels = false;
    ui.style_mut().visuals.override_text_color = Some(tokens::TEXT);
    ui.style_mut().text_styles.insert(
        egui::TextStyle::Body,
        FontId::proportional(tokens::TYPE_SIZE),
    );
    ui.style_mut().text_styles.insert(
        egui::TextStyle::Monospace,
        FontId::monospace(tokens::TIMECODE_SIZE),
    );
    ui.ctx().global_style_mut(|style| {
        style.interaction.selectable_labels = false;
        style.text_styles.insert(
            egui::TextStyle::Body,
            FontId::proportional(tokens::TYPE_SIZE),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            FontId::monospace(tokens::TIMECODE_SIZE),
        );
    });
}

fn project_bar(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, full: Rect, height: f32) {
    let bar = Rect::from_min_size(full.min, Vec2::new(full.width(), height));
    ui.painter()
        .rect_filled(bar, CornerRadius::ZERO, tokens::PROJECT_BAR);
    shell.chrome.project_bar = true;
    window_buttons(shell, ui, bar);
    menus(shell, ui, bar);
    ui.painter().text(
        bar.center(),
        Align2::CENTER_CENTER,
        shell.project_name(),
        FontId::proportional(tokens::TYPE_SIZE),
        tokens::MUTED,
    );
    let response = ui
        .scope_builder(
            egui::UiBuilder::new().max_rect(Rect::from_min_size(
                bar.right_top() - Vec2::new(96.0, 0.0),
                Vec2::new(88.0, bar.height()),
            )),
            |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    widgets::export_button(ui)
                })
                .inner
            },
        )
        .inner;
    if response.clicked() {
        view.export_open = true;
    }
    shell.chrome.export = true;
}

fn window_buttons(shell: &mut Shell, ui: &mut Ui, bar: Rect) {
    let colors = [tokens::CLOSE, tokens::MINIMIZE, tokens::MAXIMIZE];
    for (index, color) in colors.into_iter().enumerate() {
        let center = bar.left_center() + Vec2::new(18.0 + index as f32 * 20.0, 0.0);
        ui.painter()
            .circle_filled(center, tokens::WINDOW_BUTTON * 0.5, color);
        let response = ui.allocate_rect(
            Rect::from_center_size(center, Vec2::splat(tokens::WINDOW_BUTTON)),
            Sense::click(),
        );
        if response.clicked() && index == 0 {
            shell.chrome.close_requested = true;
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if response.clicked() && index == 1 {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Minimized(true));
        }
        if response.clicked() && index == 2 {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }
    }
}

fn menus(shell: &mut Shell, ui: &mut Ui, bar: Rect) {
    let menu = Rect::from_min_size(
        bar.min + Vec2::new(78.0, 0.0),
        Vec2::new(160.0, bar.height()),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(menu), |ui| {
        ui.horizontal_centered(|ui| {
            ui.menu_button("File", |ui| file_menu(shell, ui));
            ui.menu_button("Edit", |ui| edit_menu(shell, ui));
        });
    });
}

fn file_menu(shell: &mut Shell, ui: &mut Ui) {
    if ui.button("Open…").clicked() {
        ui.close();
        if let Some(path) = rfd::FileDialog::new().pick_folder()
            && let Err(error) = shell.open_folder(&path)
        {
            shell.set_notice(error.to_string());
        }
    }
    if ui.button("Save").clicked() {
        ui.close();
        if let Err(error) = shell.save() {
            shell.set_notice(error.to_string());
        }
    }
    if ui.button("Import…").clicked() {
        ui.close();
        if let Some(path) = rfd::FileDialog::new().pick_file()
            && let Err(error) = shell.import_file(&path)
        {
            shell.set_notice(error.to_string());
        }
    }
}

fn edit_menu(shell: &mut Shell, ui: &mut Ui) {
    if ui.button("Undo").clicked() {
        ui.close();
        if let Err(error) = shell.undo() {
            shell.set_notice(error.to_string());
        }
    }
    if ui.button("Redo").clicked() {
        ui.close();
        if let Err(error) = shell.redo() {
            shell.set_notice(error.to_string());
        }
    }
}

fn regions(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, row: Rect, sizes: Sizes) {
    let library = Rect::from_min_size(row.min, Vec2::new(sizes.library, row.height()));
    panel(ui, library);
    library_panel(shell, ui, library);
    shell.chrome.library = true;
    let gap = Rect::from_min_size(
        Pos2::new(library.max.x, row.min.y),
        Vec2::new(tokens::SPLIT, row.height()),
    );
    drag_gap(ui, gap, |delta| shell.drag_library(delta.x, row.width()));
    let preview = Rect::from_min_size(
        Pos2::new(gap.max.x, row.min.y),
        Vec2::new(sizes.preview, row.height()),
    );
    panel(ui, preview);
    preview_panel(shell, ui, view, preview);
    shell.chrome.preview = true;
    let gap = Rect::from_min_size(
        Pos2::new(preview.max.x, row.min.y),
        Vec2::new(tokens::SPLIT, row.height()),
    );
    drag_gap(ui, gap, |delta| shell.drag_inspector(delta.x, row.width()));
    let inspector = Rect::from_min_size(
        Pos2::new(gap.max.x, row.min.y),
        Vec2::new(sizes.inspector, row.height()),
    );
    panel(ui, inspector);
    inspector_panel(shell, ui, view, inspector);
    shell.chrome.inspector = true;
}

fn library_panel(shell: &Shell, ui: &mut Ui, rect: Rect) {
    let rail = Rect::from_min_size(rect.min, Vec2::new(tokens::RAIL_WIDTH, rect.height()));
    ui.scope_builder(egui::UiBuilder::new().max_rect(rail.shrink(4.0)), |ui| {
        ui.vertical(|ui| {
            for (name, tip) in [
                ("media", "Media"),
                ("audio", "Audio"),
                ("text", "Text"),
                ("stickers", "Stickers"),
                ("effects", "Effects"),
                ("transitions", "Transitions"),
                ("filters", "Filters"),
            ] {
                widgets::icon_button(ui, name, tip);
            }
        });
    });
    let hairline = Rect::from_min_size(
        Pos2::new(rail.max.x, rect.min.y),
        Vec2::new(tokens::HAIRLINE, rect.height()),
    );
    ui.painter()
        .rect_filled(hairline, CornerRadius::ZERO, tokens::GAP);
    let assets = Rect::from_min_max(Pos2::new(hairline.max.x + 8.0, rect.min.y + 8.0), rect.max);
    ui.scope_builder(egui::UiBuilder::new().max_rect(assets), |ui| {
        if let Some(draft) = shell.draft() {
            for video in &draft.materials.videos {
                card(ui, &file_label(&video.path));
            }
            for audio in &draft.materials.audios {
                card(ui, &file_label(&audio.path));
            }
        }
    });
}

fn card(ui: &mut Ui, name: &str) {
    let response = ui.add(
        egui::Button::new(
            RichText::new(name)
                .size(tokens::TYPE_SIZE)
                .color(tokens::TEXT),
        )
        .fill(tokens::ICON_BUTTON),
    );
    response.on_hover_text(name);
}

fn preview_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let transport = Rect::from_min_max(
        Pos2::new(rect.min.x, rect.max.y - tokens::TRANSPORT_HEIGHT),
        rect.max,
    );
    let picture = Rect::from_min_max(rect.min, Pos2::new(rect.max.x, transport.min.y));
    match shell.preview_bytes() {
        Ok(frame) => {
            if let Some(image) = raster(&frame) {
                let texture = if let Some(texture) = &mut view.preview {
                    texture.set(image, egui::TextureOptions::LINEAR);
                    texture
                } else {
                    view.preview.insert(ui.ctx().load_texture(
                        "preview",
                        image,
                        egui::TextureOptions::LINEAR,
                    ))
                };
                let fitted = fit_texture(picture, texture.size_vec2());
                ui.painter().image(
                    texture.id(),
                    fitted,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }
        Err(error) => {
            if shell.draft().is_some() {
                ui.painter().text(
                    picture.center(),
                    Align2::CENTER_CENTER,
                    error.to_string(),
                    FontId::proportional(tokens::TYPE_SIZE),
                    tokens::MUTED,
                );
            }
        }
    }
    transport_bar(shell, ui, transport);
}

fn transport_bar(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, tokens::PANEL);
    ui.painter().text(
        rect.left_center() + Vec2::new(12.0, 0.0),
        Align2::LEFT_CENTER,
        shell.timecode(),
        FontId::monospace(tokens::TIMECODE_SIZE),
        tokens::TEXT,
    );
    let play = if shell.playing() { "pause" } else { "play" };
    let tip = if shell.playing() { "Pause" } else { "Play" };
    let center = Rect::from_center_size(rect.center(), Vec2::splat(tokens::ICON_BUTTON_SIZE));
    ui.scope_builder(egui::UiBuilder::new().max_rect(center), |ui| {
        if widgets::icon_button(ui, play, tip).clicked() {
            if shell.playing() {
                shell.pause();
            } else {
                shell.play();
            }
        }
    });
    let still = Rect::from_center_size(
        rect.right_center() - Vec2::new(24.0, 0.0),
        Vec2::splat(tokens::ICON_BUTTON_SIZE),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(still), |ui| {
        if widgets::icon_button(ui, "camera", "Still").clicked() {
            shell.unavailable("still");
        }
    });
    shell.chrome.transport = true;
}

fn inspector_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect.shrink(12.0)), |ui| {
        ui.label(
            RichText::new("Inspector")
                .size(tokens::TYPE_SIZE)
                .color(tokens::MUTED),
        );
        let Some(inspection) = shell.inspection() else {
            ui.label(RichText::new("Nothing selected").color(tokens::FAINT));
            return;
        };
        ui.label(format!(
            "{}  {}",
            shell.timecode_at(inspection.start),
            format_duration(inspection.duration.0)
        ));
        ui.label(format!("Speed {}", inspection.speed));
        ui.label(format!("Volume {}", inspection.volume));
        ui.label(format!(
            "Scale {:.2}, {:.2}   rotate {:.1}   move {:.1}, {:.1}",
            inspection.scale_x,
            inspection.scale_y,
            inspection.rotation,
            inspection.transform_x,
            inspection.transform_y
        ));
        ui.label(inspection.material.clone());
        if widgets::toolbar_button(ui, "Transform").clicked()
            && let Err(error) = shell.edit_transform()
        {
            shell.set_notice(error.to_string());
        }
        if let Some(text) = inspection.text.clone() {
            if view.text_for != inspection.segment_id.as_str() {
                view.text_buffer = text;
                view.text_for = inspection.segment_id.as_str().to_owned();
            }
            let response = widgets::field(ui, &mut view.text_buffer);
            if response.lost_focus()
                && !inspection.locked
                && let Err(error) = shell.set_text(view.text_buffer.clone())
            {
                shell.set_notice(error.to_string());
            }
            if widgets::toolbar_button(ui, "Style").clicked()
                && let Err(error) = shell.edit_style()
            {
                shell.set_notice(error.to_string());
            }
        }
        view.volume_buffer = format!("{}", inspection.volume);
        if widgets::field(ui, &mut view.volume_buffer).lost_focus()
            && let Ok(volume) = view.volume_buffer.parse()
            && let Err(error) = shell.set_volume(volume)
        {
            shell.set_notice(error.to_string());
        }
        view.speed_buffer = format!("{}", inspection.speed);
        if widgets::field(ui, &mut view.speed_buffer).lost_focus()
            && let Ok(speed) = view.speed_buffer.parse()
            && let Err(error) = shell.set_speed(speed)
        {
            shell.set_notice(error.to_string());
        }
    });
}

fn timeline_panel(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    panel(ui, rect);
    shell.chrome.timeline = true;
    let toolbar = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::TOOLBAR_HEIGHT));
    ui.scope_builder(egui::UiBuilder::new().max_rect(toolbar.shrink(4.0)), |ui| {
        ui.horizontal(|ui| toolbar_icons(shell, ui));
    });
    let hairline = Rect::from_min_size(
        Pos2::new(rect.min.x, toolbar.max.y),
        Vec2::new(rect.width(), tokens::HAIRLINE),
    );
    ui.painter()
        .rect_filled(hairline, CornerRadius::ZERO, tokens::GAP);
    let body = Rect::from_min_max(Pos2::new(rect.min.x, hairline.max.y), rect.max);
    ruler_and_lanes(shell, ui, body);
    let handle = Rect::from_min_size(
        Pos2::new(rect.min.x, rect.min.y - tokens::SPLIT),
        Vec2::new(rect.width(), tokens::SPLIT),
    );
    let window_height = ui.ctx().content_rect().height();
    drag_gap(ui, handle, |delta| {
        shell.drag_timeline(delta.y, window_height)
    });
}

fn toolbar_icons(shell: &mut Shell, ui: &mut Ui) {
    for (name, tip) in TOOLS {
        if widgets::icon_button(ui, name, tip).clicked() {
            tool_clicked(shell, name);
        }
    }
}

fn tool_clicked(shell: &mut Shell, name: &str) {
    let result = match name {
        "undo" => shell.undo(),
        "redo" => shell.redo(),
        "zoomIn" => shell.zoom_in(),
        "zoomOut" => shell.zoom_out(),
        "zoomFit" => shell.zoom_fit(),
        "delete" => shell.remove_selected(),
        "split" => shell.split_selected(),
        _ => {
            shell.unavailable(name);
            Ok(())
        }
    };
    if let Err(error) = result {
        shell.set_notice(error.to_string());
    }
}

const TOOLS: &[(&str, &str)] = &[
    ("undo", "Undo"),
    ("redo", "Redo"),
    ("select", "Select"),
    ("split", "Split"),
    ("delete", "Delete"),
    ("freeze", "Freeze"),
    ("reverse", "Reverse"),
    ("mirror", "Mirror"),
    ("rotate", "Rotate"),
    ("crop", "Crop"),
    ("transcribe", "Transcribe"),
    ("splitScenes", "Split scenes"),
    ("snap", "Snap"),
    ("magnet", "Magnet"),
    ("linkage", "Linkage"),
    ("marker", "Marker"),
    ("beat", "Beat"),
    ("zoomIn", "Zoom in"),
    ("zoomOut", "Zoom out"),
    ("zoomFit", "Zoom to fit"),
    ("guides", "Guides"),
    ("lock", "Lock"),
];

fn ruler_and_lanes(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    let cover = tokens::COVER_WIDTH;
    let ruler = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::RULER_HEIGHT));
    let response = ui.allocate_rect(ruler, Sense::click());
    if response.clicked() {
        let x = f64::from(
            response
                .interact_pointer_pos()
                .map(|pos| pos.x)
                .unwrap_or(ruler.min.x)
                - ruler.min.x
                - cover,
        );
        if let Err(error) = shell.seek_ruler(x.max(0.0)) {
            shell.set_notice(error.to_string());
        }
    }
    ui.painter().text(
        ruler.left_center() + Vec2::new(8.0, 0.0),
        Align2::LEFT_CENTER,
        shell.timecode(),
        FontId::monospace(tokens::TIMECODE_SIZE),
        tokens::MUTED,
    );
    let mut y = ruler.max.y;
    for lane in shell.lanes() {
        let row = Rect::from_min_size(
            Pos2::new(rect.min.x, y),
            Vec2::new(rect.width(), tokens::LANE_HEIGHT as f32),
        );
        let header = Rect::from_min_size(row.min, Vec2::new(cover, row.height()));
        ui.scope_builder(egui::UiBuilder::new().max_rect(header.shrink(2.0)), |ui| {
            let press = widgets::lane_header(
                ui,
                HeaderFlags {
                    mute: lane.mute,
                    hide: lane.hide,
                    lock: lane.lock,
                    solo: lane.solo,
                },
            );
            let track = lane.track_id.clone();
            if press.mute
                && let Err(error) =
                    shell.toggle_attribute(track.clone(), cutlass_model::TrackAttribute::MUTE)
            {
                shell.set_notice(error.to_string());
            }
            if press.hide
                && let Err(error) =
                    shell.toggle_attribute(track.clone(), cutlass_model::TrackAttribute::HIDE)
            {
                shell.set_notice(error.to_string());
            }
            if press.lock
                && let Err(error) =
                    shell.toggle_attribute(track.clone(), cutlass_model::TrackAttribute::LOCK)
            {
                shell.set_notice(error.to_string());
            }
            if press.solo
                && let Err(error) =
                    shell.toggle_attribute(track, cutlass_model::TrackAttribute::SOLO)
            {
                shell.set_notice(error.to_string());
            }
        });
        for clip in lane.clips {
            let x = rect.min.x + cover + (clip.start.0 as f32) * shell.pixel_size() as f32;
            let width = (clip.duration.0 as f32) * shell.pixel_size() as f32;
            let clip_rect = Rect::from_min_size(
                Pos2::new(x, row.min.y + 6.0),
                Vec2::new(width.max(8.0), row.height() - 12.0),
            );
            let selected = shell.selected().is_some_and(|id| id == &clip.id);
            let stroke = if selected {
                egui::Stroke::new(tokens::SELECTED_EDGE, tokens::SELECTED)
            } else {
                egui::Stroke::NONE
            };
            ui.painter().rect(
                clip_rect,
                CornerRadius::same(4),
                tokens::ICON_BUTTON,
                stroke,
                egui::StrokeKind::Inside,
            );
            let response = ui.allocate_rect(clip_rect, Sense::click_and_drag());
            let span = format!(
                "{}  {}–{}",
                clip.name,
                shell.timecode_at(clip.start),
                shell.timecode_at(cutlass_model::Micros(
                    clip.start.0.saturating_add(clip.duration.0)
                ))
            );
            response.clone().on_hover_text(span);
            if response.clicked() {
                shell.select(Some(clip.id.clone()));
            }
            if response.drag_stopped()
                && let Some(pos) = response.interact_pointer_pos()
            {
                let x = f64::from(pos.x - rect.min.x - cover).max(0.0);
                let y = f64::from(pos.y - ruler.max.y).max(0.0);
                if let Err(error) = shell.drag_segment(clip.id.clone(), x, y) {
                    shell.set_notice(error.to_string());
                }
            }
        }
        y += tokens::LANE_HEIGHT as f32;
    }
    let playhead_x = rect.min.x + cover + shell.playhead().0 as f32 * shell.pixel_size() as f32;
    ui.painter().line_segment(
        [
            Pos2::new(playhead_x, rect.min.y),
            Pos2::new(playhead_x, rect.max.y),
        ],
        egui::Stroke::new(1.0, tokens::PLAYHEAD),
    );
    shell.set_lane_width((rect.width() - cover).max(1.0));
}

fn export_dialog(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let ((), closed) = widgets::dialog(ui, "Export", |ui| {
        ui.label(RichText::new("Long side").color(tokens::MUTED));
        for side in crate::shell::LONG_SIDES {
            if ui
                .selectable_label(view.export_side == side, side.to_string())
                .clicked()
            {
                view.export_side = side;
            }
        }
        if view.replace_prompt {
            ui.label("Replace the existing file?");
        }
        if ui.button("Export").clicked() {
            let path = if view.export_path.is_empty() {
                rfd::FileDialog::new()
                    .save_file()
                    .unwrap_or_else(|| std::path::PathBuf::from("export.png"))
            } else {
                std::path::PathBuf::from(&view.export_path)
            };
            view.export_path = path.display().to_string();
            match shell.export(crate::shell::ExportRequest {
                path,
                long_side: view.export_side,
                codec: view.export_codec,
                replace: view.replace_prompt,
            }) {
                Ok(()) => view.export_open = false,
                Err(crate::shell::ShellError::Overwrite(_)) => view.replace_prompt = true,
                Err(error) => shell.set_notice(error.to_string()),
            }
        }
    });
    if closed {
        view.export_open = false;
        view.replace_prompt = false;
    }
}

fn shortcuts(shell: &mut Shell, ui: &Ui) {
    if let Some(chord) = ui.input(pressed_chord)
        && let Err(error) = shell.run_shortcut(&chord)
        && !error.to_string().contains("not available")
    {
        shell.set_notice(error.to_string());
    }
}

fn pressed_chord(input: &egui::InputState) -> Option<String> {
    let key = input
        .keys_down
        .iter()
        .find(|key| input.key_pressed(**key))?;
    let name = match key {
        egui::Key::Z => "z",
        egui::Key::Y => "y",
        _ => return None,
    };
    let mut parts = Vec::new();
    if input.modifiers.command {
        parts.push("primary");
    }
    if input.modifiers.shift {
        parts.push("shift");
    }
    parts.push(name);
    Some(parts.join("+"))
}

fn panel(ui: &mut Ui, rect: Rect) {
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, tokens::PANEL);
}

fn drag_gap(ui: &mut Ui, rect: Rect, mut on_drag: impl FnMut(Vec2)) {
    let response = ui.allocate_rect(rect, Sense::drag());
    if response.dragged() {
        on_drag(response.drag_delta());
    }
}

fn raster(frame: &crate::shell::Picture) -> Option<egui::ColorImage> {
    let width = usize::try_from(frame.width).ok()?;
    let height = usize::try_from(frame.height).ok()?;
    if width == 0
        || height == 0
        || frame.rgba.len() != width.saturating_mul(height).saturating_mul(4)
    {
        return None;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [width, height],
        &frame.rgba,
    ))
}

fn fit_texture(bounds: Rect, size: Vec2) -> Rect {
    if size.x <= 0.0 || size.y <= 0.0 {
        return bounds;
    }
    let scale = (bounds.width() / size.x).min(bounds.height() / size.y);
    Rect::from_center_size(bounds.center(), size * scale)
}

fn file_label(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned())
}

fn format_duration(micros: i64) -> String {
    format!("{micros} µs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_settings::Settings;
    use egui::Color32;

    #[test]
    fn one_frame_paints_the_shell_and_does_not_ask_to_close() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view),
        );
        assert!(shell.chrome.project_bar);
        assert!(shell.chrome.library);
        assert!(shell.chrome.preview);
        assert!(shell.chrome.inspector);
        assert!(shell.chrome.timeline);
        assert!(shell.chrome.transport);
        assert!(shell.chrome.export);
        assert!(!shell.chrome.close_requested);
        assert!(!ctx.global_style().interaction.selectable_labels);
        output.textures_delta.clear();
    }

    #[test]
    fn the_preview_widget_paints_the_evaluated_bytes() {
        let temp = std::env::temp_dir().join(format!("cutlass-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "color".into(),
                is_color_clip: true,
                solid_color: "0xFF336699".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "segment",
            "color",
            cutlass_model::TimeRange::new(0, 1_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let picture = shell.preview_bytes().unwrap();
        let image = raster(&picture).expect("image");
        assert_eq!(image.size, [2, 1]);
        assert_eq!(
            image.pixels[0],
            Color32::from_rgba_unmultiplied(0x33, 0x66, 0x99, 0xFF)
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn phones_do_not_link_egui() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let ffi = std::fs::read_to_string(root.join("crates/ffi/Cargo.toml")).unwrap();
        assert!(!ffi.contains("egui"));
        assert!(!ffi.contains("eframe"));
        for entry in std::fs::read_dir(root.join("apps")).unwrap() {
            let path = entry.unwrap().path();
            if path.ends_with("desktop") {
                continue;
            }
            let cargo = path.join("Cargo.toml");
            if cargo.is_file() {
                let text = std::fs::read_to_string(&cargo).unwrap();
                assert!(!text.contains("egui"), "{}", cargo.display());
                assert!(!text.contains("eframe"), "{}", cargo.display());
            }
        }
    }
}
