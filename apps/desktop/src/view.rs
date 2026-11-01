//! The column of panels. egui draws the chrome. The preview is a texture of
//! the evaluated frame. This pass does not composite.
//!
//! With no project open, the window lists the saved projects and can create one.

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Pos2, Rect, RichText, ScrollArea, Sense,
    Ui, Vec2,
};

use cutlass_model::{FrameRate, Micros, SegmentId};
use cutlass_session::TrimEdge;

use crate::dialogs::{Dialogs, Finished, Picker};
use crate::icons;
use crate::shell::{ExportRequest, ListedProject, MovePreview, Shell, ShellError};
use crate::tokens;
use crate::widgets;

const CARD_W: f32 = 144.0;
const CARD_H: f32 = 96.0;
const CARD_GAP: f32 = 8.0;
const TRIM_EDGE: f32 = 6.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shelf {
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

    fn label(self) -> &'static str {
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

    fn hint(self) -> &'static str {
        match self {
            Self::Media => "Search media",
            Self::Audio => "Search audio",
            Self::Text => "Search text",
            Self::Stickers => "Search stickers",
            Self::Effects => "Search effects",
            Self::Transitions => "Search transitions",
            Self::Filters => "Search filters",
            Self::Adjustments => "Search adjustments",
            Self::Templates => "Search templates",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::Audio => "audio",
            Self::Text => "text",
            Self::Stickers => "stickers",
            Self::Effects => "effects",
            Self::Transitions => "transitions",
            Self::Filters => "filters",
            Self::Adjustments => "adjustments",
            Self::Templates => "templates",
        }
    }

    fn empty(self) -> &'static str {
        match self {
            Self::Media => "No media yet. Use File → Import.",
            Self::Audio => "No audio yet. Use File → Import.",
            Self::Text => "No text in this project.",
            Self::Stickers => "No stickers in this project.",
            Self::Effects => "No effects in this project.",
            Self::Transitions => "No transitions in this project.",
            Self::Filters => "No filters in this project.",
            Self::Adjustments => "No adjustments in this project.",
            Self::Templates => "No templates in this project.",
        }
    }
}

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
    dialogs: Dialogs,
    library_query: String,
    library_shelf: Shelf,
    projects_root: Option<std::path::PathBuf>,
    listed: Vec<ListedProject>,
    listed_stamp: String,
    marquee_press: Option<Pos2>,
    marquee: Option<Rect>,
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
            dialogs: Dialogs::new(),
            library_query: String::new(),
            library_shelf: Shelf::Media,
            projects_root: None,
            listed: Vec::new(),
            listed_stamp: String::new(),
            marquee_press: None,
            marquee: None,
        }
    }
}

impl ViewState {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn show(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, parent: Option<&eframe::Frame>) {
    if let Some(finished) = view.dialogs.poll(ui.ctx()) {
        apply_dialog(shell, view, finished);
    }
    style(ui);
    shell.chrome = crate::shell::Chrome::default();
    let full = ui.max_rect();
    ui.painter()
        .rect_filled(full, CornerRadius::ZERO, tokens::GAP);
    if shell.draft().is_none() {
        projects_home(shell, ui, view, full, parent);
    } else {
        view.listed.clear();
        view.listed_stamp.clear();
        let sizes = sizes(shell, full.size());
        project_bar(shell, ui, view, full, sizes.head, parent);
        let row = Rect::from_min_size(
            Pos2::new(full.min.x, full.min.y + sizes.head),
            Vec2::new(full.width(), sizes.row),
        );
        regions(shell, ui, view, row, sizes);
        let timeline = Rect::from_min_size(
            Pos2::new(full.min.x, row.max.y + tokens::SPLIT),
            Vec2::new(full.width(), sizes.timeline),
        );
        timeline_panel(shell, ui, view, timeline);
        paint_library_ghost(shell, ui);
        if view.export_open {
            export_dialog(shell, ui, view, parent);
        }
        shortcuts(shell, ui);
        if shell.playing() {
            ui.ctx().request_repaint();
        }
    }
    if let Some(finished) = view.dialogs.poll(ui.ctx()) {
        apply_dialog(shell, view, finished);
    }
}

fn projects_root(view: &ViewState) -> std::path::PathBuf {
    view.projects_root
        .clone()
        .unwrap_or_else(Shell::projects_directory)
}

fn projects_home(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    full: Rect,
    parent: Option<&eframe::Frame>,
) {
    let bar = Rect::from_min_size(full.min, Vec2::new(full.width(), tokens::HEAD_HEIGHT));
    ui.painter()
        .rect_filled(bar, CornerRadius::ZERO, tokens::PROJECT_BAR);
    shell.chrome.project_bar = true;
    shell.chrome.projects = true;
    #[cfg(not(target_os = "macos"))]
    window_buttons(shell, ui, bar);
    let menu = Rect::from_min_size(
        bar.min + Vec2::new(78.0, 0.0),
        Vec2::new(80.0, bar.height()),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(menu), |ui| {
        ui.horizontal_centered(|ui| {
            ui.menu_button(RichText::new("File").color(tokens::FAINT), |ui| {
                file_menu(shell, ui, view, parent);
            });
        });
    });
    ui.painter().text(
        bar.center(),
        Align2::CENTER_CENTER,
        "Projects",
        FontId::proportional(14.0),
        tokens::TEXT,
    );
    let body = Rect::from_min_max(
        Pos2::new(full.min.x + 28.0, bar.bottom() + 28.0),
        Pos2::new(full.max.x - 28.0, full.max.y - 28.0),
    );
    let root = projects_root(view);
    let projects = listed_projects(view);
    let mut body_ui = ui.new_child(egui::UiBuilder::new().max_rect(body));
    ScrollArea::vertical()
        .id_salt("projects")
        .auto_shrink([false, false])
        .show(&mut body_ui, |ui| {
            if new_project_button(ui) {
                open_created(shell, &root);
            }
            ui.add_space(20.0);
            if projects.is_empty() {
                ui.label(
                    RichText::new("No projects yet.")
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::MUTED),
                );
            }
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(16.0, 16.0);
                for project in &projects {
                    if project_card(ui, project)
                        && let Err(error) = shell.open_folder(&project.path)
                    {
                        shell.set_notice(error.to_string());
                    }
                }
            });
        });
    if !shell.notice().is_empty() {
        paint_notice(ui, bar.center().x, bar.bottom() + 12.0, shell.notice());
    }
}

fn listed_projects(view: &mut ViewState) -> Vec<ListedProject> {
    let root = projects_root(view);
    let stamp = format!("{}|{}", root.display(), directory_stamp(&root));
    if view.listed_stamp != stamp {
        view.listed = Shell::list_projects(&root);
        view.listed_stamp = stamp;
    }
    view.listed.clone()
}

fn directory_stamp(root: &std::path::Path) -> String {
    let Ok(entries) = std::fs::read_dir(root) else {
        return String::new();
    };
    let mut names = Vec::new();
    for entry in entries.flatten() {
        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        names.push(format!(
            "{}:{modified}",
            entry.file_name().to_string_lossy()
        ));
    }
    names.sort();
    names.join("\n")
}

fn open_created(shell: &mut Shell, root: &std::path::Path) {
    match Shell::create_project(root) {
        Ok(path) => {
            if let Err(error) = shell.open_folder(&path) {
                shell.set_notice(error.to_string());
            }
        }
        Err(error) => shell.set_notice(error.to_string()),
    }
}

fn new_project_button(ui: &mut Ui) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(148.0, 36.0), Sense::click());
    ui.painter().rect_filled(
        rect,
        CornerRadius::same(tokens::EXPORT_RADIUS),
        tokens::EXPORT,
    );
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        "New project",
        FontId::proportional(14.0),
        tokens::EXPORT_INK,
    );
    response.clicked()
}

fn project_card(ui: &mut Ui, project: &ListedProject) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(220.0, 88.0), Sense::click());
    let fill = if response.hovered() {
        tokens::ACTIVE
    } else {
        tokens::PROJECT_BAR
    };
    ui.painter().rect_filled(rect, CornerRadius::same(12), fill);
    ui.painter().text(
        rect.min + Vec2::new(16.0, 18.0),
        Align2::LEFT_TOP,
        &project.name,
        FontId::proportional(15.0),
        tokens::TEXT,
    );
    let meta = if project.duration.0 > 0 {
        format!(
            "{}  {}",
            card_meta(project.duration.0),
            modified_label(project.modified)
        )
    } else {
        modified_label(project.modified)
    };
    ui.painter().text(
        rect.min + Vec2::new(16.0, 46.0),
        Align2::LEFT_TOP,
        meta,
        FontId::proportional(12.0),
        tokens::MUTED,
    );
    response.clicked()
}

fn modified_label(time: std::time::SystemTime) -> String {
    let Ok(elapsed) = time.duration_since(std::time::UNIX_EPOCH) else {
        return String::new();
    };
    utc_date(elapsed.as_secs())
}

fn utc_date(seconds: u64) -> String {
    let days = i64::try_from(seconds / 86_400).unwrap_or(i64::MAX);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = u64::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = i64::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    if month <= 2 {
        year += 1;
    }
    format!("{year:04}-{month:02}-{day:02}")
}

fn return_to_projects(shell: &mut Shell, view: &mut ViewState) {
    if let Err(error) = shell.close_project() {
        shell.set_notice(error.to_string());
        return;
    }
    view.preview = None;
    view.export_open = false;
    view.library_query.clear();
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
        (window.y - head - tokens::SPLIT - tokens::PREVIEW_MIN_HEIGHT).max(tokens::TIMELINE_MIN),
    );
    let row = (window.y - head - tokens::SPLIT - timeline).max(0.0);
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
    ui.ctx().set_theme(egui::ThemePreference::Dark);
    ui.style_mut().interaction.selectable_labels = false;
    ui.ctx().global_style_mut(install_style);
    ui.ctx().style_mut_of(egui::Theme::Dark, install_style);
}

fn install_style(style: &mut egui::Style) {
    style.interaction.selectable_labels = false;
    style.visuals.override_text_color = Some(tokens::TEXT);
    style.visuals.panel_fill = tokens::PANEL;
    style.visuals.window_fill = tokens::PANEL;
    style.visuals.extreme_bg_color = tokens::GAP;
    style.visuals.faint_bg_color = tokens::PROJECT_BAR;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
    style.text_styles.insert(
        egui::TextStyle::Body,
        FontId::proportional(tokens::TYPE_SIZE),
    );
    style
        .text_styles
        .insert(egui::TextStyle::Button, FontId::proportional(14.0));
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        FontId::monospace(tokens::TIMECODE_SIZE),
    );
}

fn project_bar(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    full: Rect,
    height: f32,
    parent: Option<&eframe::Frame>,
) {
    let bar = Rect::from_min_size(full.min, Vec2::new(full.width(), height));
    ui.painter()
        .rect_filled(bar, CornerRadius::ZERO, tokens::PROJECT_BAR);
    shell.chrome.project_bar = true;
    #[cfg(not(target_os = "macos"))]
    window_buttons(shell, ui, bar);
    menus(shell, ui, view, bar, parent);
    ui.painter().text(
        bar.center(),
        Align2::CENTER_CENTER,
        shell.project_name(),
        FontId::proportional(14.0),
        tokens::TEXT,
    );
    let export = Rect::from_min_size(
        bar.right_center() - Vec2::new(12.0 + 72.0, 13.0),
        Vec2::new(72.0, 26.0),
    );
    let response = ui
        .scope_builder(egui::UiBuilder::new().max_rect(export), |ui| {
            widgets::export_button(ui)
        })
        .inner;
    if response.clicked() {
        view.export_open = true;
    }
    shell.chrome.export = true;
}

#[cfg(not(target_os = "macos"))]
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

fn menus(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    bar: Rect,
    parent: Option<&eframe::Frame>,
) {
    let menu = Rect::from_min_size(
        bar.min + Vec2::new(78.0, 0.0),
        Vec2::new(240.0, bar.height()),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(menu), |ui| {
        ui.horizontal_centered(|ui| {
            if ui
                .add(egui::Button::new(RichText::new("Projects").color(tokens::FAINT)).frame(false))
                .clicked()
            {
                return_to_projects(shell, view);
            }
            ui.menu_button(RichText::new("File").color(tokens::FAINT), |ui| {
                file_menu(shell, ui, view, parent);
            });
            ui.menu_button(RichText::new("Edit").color(tokens::FAINT), |ui| {
                edit_menu(shell, ui);
            });
        });
    });
}

fn file_menu(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, parent: Option<&eframe::Frame>) {
    if ui.button("Open…").clicked() {
        ui.close();
        view.dialogs.open_project(parent);
    }
    if shell.draft().is_none() {
        return;
    }
    if ui.button("Save").clicked() {
        ui.close();
        if let Err(error) = shell.save() {
            shell.set_notice(error.to_string());
        }
    }
    if ui.button("Import…").clicked() {
        ui.close();
        view.dialogs.import(parent);
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
    library_panel(shell, ui, view, library);
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

fn library_panel(shell: &Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let rail = Rect::from_min_size(rect.min, Vec2::new(tokens::RAIL_WIDTH, rect.height()));
    shelf_rail(ui, view, rail);
    let hairline = Rect::from_min_size(
        Pos2::new(rail.max.x, rect.min.y),
        Vec2::new(tokens::HAIRLINE, rect.height()),
    );
    ui.painter()
        .rect_filled(hairline, CornerRadius::ZERO, tokens::GAP);
    let assets = Rect::from_min_max(hairline.right_top(), rect.max).shrink(12.0);
    asset_column(shell, ui, view, assets);
}

fn shelf_rail(ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let inner = Rect::from_min_max(
        Pos2::new(rect.left() + 8.0, rect.top()),
        Pos2::new(rect.right() - 8.0, rect.bottom()),
    );
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(inner));
    ScrollArea::vertical()
        .id_salt("library-rail")
        .show(&mut column, |ui| {
            ui.add_space(8.0);
            ui.spacing_mut().item_spacing.y = 2.0;
            for shelf in Shelf::ALL {
                if shelf_button(ui, shelf, view.library_shelf == shelf).clicked() {
                    view.library_shelf = shelf;
                }
            }
        });
}

fn shelf_button(ui: &mut Ui, shelf: Shelf, active: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 44.0), Sense::click());
    if active {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(10), tokens::ACTIVE);
    }
    let color = if active { tokens::TEXT } else { tokens::FAINT };
    let mark = Rect::from_center_size(rect.center(), Vec2::splat(20.0));
    if let Err(error) = icons::paint(ui.painter(), mark, shelf.icon(), color) {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            error.to_string(),
            FontId::proportional(9.0),
            tokens::FAINT,
        );
    }
    response.on_hover_text(shelf.label())
}

fn asset_column(shell: &Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    column.add(
        egui::TextEdit::singleline(&mut view.library_query)
            .hint_text(view.library_shelf.hint())
            .desired_width(f32::INFINITY)
            .text_color(tokens::TEXT)
            .background_color(tokens::PROJECT_BAR)
            .margin(egui::Margin::symmetric(8, 6)),
    );
    column.add_space(12.0);
    let grid = column.available_rect_before_wrap();
    let mut grid_ui = column.new_child(egui::UiBuilder::new().max_rect(grid));
    let shelf = view.library_shelf;
    let query = view.library_query.clone();
    ScrollArea::vertical()
        .id_salt("library-cards")
        .auto_shrink([false, false])
        .show(&mut grid_ui, |ui| {
            let cards = shell
                .draft()
                .map(|draft| bin_cards(draft, shelf, &query))
                .unwrap_or_default();
            if cards.is_empty() {
                let message = if !query.trim().is_empty() {
                    "No matches."
                } else if shell.draft().is_some() {
                    shelf.empty()
                } else {
                    "Open a project, then import a file."
                };
                ui.add_space(8.0);
                ui.label(
                    RichText::new(message)
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::MUTED),
                );
            } else {
                card_grid(ui, &cards);
            }
        });
}

fn card_grid(ui: &mut Ui, cards: &[BinCard]) {
    let width = ui.available_width().max(CARD_W);
    ui.set_min_width(width);
    let cols = ((width + CARD_GAP) / (CARD_W + CARD_GAP)).floor().max(1.0) as usize;
    let cols = cols.max(1);
    let rows = cards.len().div_ceil(cols);
    let height = rows as f32 * (CARD_H + CARD_GAP);
    let (area, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    for (index, card) in cards.iter().enumerate() {
        let origin = area.min
            + Vec2::new(
                (index % cols) as f32 * (CARD_W + CARD_GAP),
                (index / cols) as f32 * (CARD_H + CARD_GAP),
            );
        let rect = Rect::from_min_size(origin, Vec2::new(CARD_W, CARD_H));
        paint_bin_card(ui, rect, card);
        let response = ui.interact(rect, ui.scope_id().with(&card.id), Sense::drag());
        response.dnd_set_drag_payload(LibraryDrop {
            material_id: card.id.clone(),
        });
        response.on_hover_text(&card.name);
    }
}

fn paint_bin_card(ui: &Ui, rect: Rect, card: &BinCard) {
    let painter = ui.painter().with_clip_rect(rect);
    paint_card_face(&painter, rect, card);
}

fn paint_card_face(painter: &egui::Painter, rect: Rect, card: &BinCard) {
    let swatch = swatch_color(&card.name);
    painter.rect_filled(rect, CornerRadius::same(12), swatch);
    painter.text(
        rect.left_bottom() + Vec2::new(8.0, -22.0),
        Align2::LEFT_BOTTOM,
        &card.name,
        FontId::proportional(12.0),
        Color32::WHITE,
    );
    painter.text(
        rect.left_bottom() + Vec2::new(8.0, -8.0),
        Align2::LEFT_BOTTOM,
        &card.meta,
        FontId::proportional(11.0),
        Color32::from_white_alpha(220),
    );
    if card.on_timeline {
        painter.text(
            rect.left_top() + Vec2::new(8.0, 8.0),
            Align2::LEFT_TOP,
            "On timeline",
            FontId::proportional(10.0),
            Color32::WHITE,
        );
    }
}

struct BinCard {
    id: String,
    name: String,
    meta: String,
    on_timeline: bool,
}

#[derive(Clone, Debug)]
struct LibraryDrop {
    material_id: String,
}

fn bin_cards(draft: &cutlass_model::Draft, shelf: Shelf, query: &str) -> Vec<BinCard> {
    let used: std::collections::HashSet<&str> = draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .map(|segment| segment.material_id.as_str())
        .collect();
    let query = query.trim().to_lowercase();
    let mut cards = Vec::new();
    match shelf {
        Shelf::Media => {
            for video in &draft.materials.videos {
                push_card(
                    &mut cards,
                    &used,
                    &query,
                    video.id.as_str(),
                    &video.path,
                    video.duration.0,
                );
            }
        }
        Shelf::Audio => {
            for audio in &draft.materials.audios {
                push_card(
                    &mut cards,
                    &used,
                    &query,
                    audio.id.as_str(),
                    &audio.path,
                    audio.duration.0,
                );
            }
        }
        Shelf::Transitions => {
            for offer in crate::shell::TRANSITION_OFFERS {
                if !query.is_empty() && !offer.name.to_lowercase().contains(&query) {
                    continue;
                }
                cards.push(BinCard {
                    id: offer.card_id(),
                    name: offer.name.to_owned(),
                    meta: card_meta(offer.duration),
                    on_timeline: false,
                });
            }
        }
        Shelf::Text
        | Shelf::Stickers
        | Shelf::Effects
        | Shelf::Filters
        | Shelf::Adjustments
        | Shelf::Templates => {}
    }
    cards
}

fn push_card(
    cards: &mut Vec<BinCard>,
    used: &std::collections::HashSet<&str>,
    query: &str,
    id: &str,
    path: &str,
    duration: i64,
) {
    let name = file_label(path);
    if !query.is_empty() && !name.to_lowercase().contains(query) {
        return;
    }
    cards.push(BinCard {
        id: id.to_owned(),
        name,
        meta: card_meta(duration),
        on_timeline: used.contains(id),
    });
}

fn card_meta(micros: i64) -> String {
    if micros <= 0 {
        return "0.0s".to_owned();
    }
    let millis = micros / 1_000;
    let minutes = millis / 60_000;
    let seconds = millis % 60_000;
    let whole = seconds / 1_000;
    let tenth = (seconds % 1_000) / 100;
    if minutes > 0 {
        format!("{minutes}:{whole:02}.{tenth}")
    } else {
        format!("{whole}.{tenth}s")
    }
}

fn swatch_color(name: &str) -> Color32 {
    let mut hash = 2_166_136_261_u32;
    for byte in name.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    Color32::from_rgb(channel(hash, 0), channel(hash, 6), channel(hash, 12))
}

fn channel(hash: u32, shift: u32) -> u8 {
    u8::try_from(48 + ((hash >> shift) & 0x3f)).unwrap_or(48)
}

fn preview_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let transport = Rect::from_min_max(
        Pos2::new(rect.min.x, rect.max.y - tokens::TRANSPORT_HEIGHT),
        rect.max,
    );
    let frame = Rect::from_min_max(
        rect.min + Vec2::new(16.0, 16.0),
        Pos2::new(rect.max.x - 16.0, transport.min.y - 8.0),
    );
    ui.painter()
        .rect_filled(frame, CornerRadius::same(12), tokens::GAP);
    match shell.preview_bytes() {
        Ok(picture) => {
            if let Some(image) = raster(&picture) {
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
                let fitted = fit_texture(frame, texture.size_vec2());
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
                    frame.center(),
                    Align2::CENTER_CENTER,
                    error.to_string(),
                    FontId::proportional(tokens::TYPE_SIZE),
                    tokens::MUTED,
                );
            }
        }
    }
    ui.painter().text(
        frame.left_top() + Vec2::new(12.0, 12.0),
        Align2::LEFT_TOP,
        shell.timecode(),
        FontId::monospace(11.0),
        tokens::TEXT,
    );
    transport_bar(shell, ui, transport);
}

fn transport_bar(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    let live = shell.draft().is_some();
    let ink = if live { tokens::TEXT } else { tokens::MUTED };
    ui.painter().text(
        Pos2::new(rect.left() + 16.0, rect.center().y),
        Align2::LEFT_CENTER,
        format!("{}  /  {}", shell.timecode(), duration_label(shell)),
        FontId::monospace(tokens::TIMECODE_SIZE),
        ink,
    );
    let play = Rect::from_center_size(rect.center(), Vec2::splat(36.0));
    let play_response = ui.interact(play, ui.scope_id().with("play"), Sense::click());
    if play_response.clicked() {
        if shell.playing() {
            shell.pause();
        } else {
            shell.play();
        }
    }
    ui.painter()
        .circle_filled(play.center(), 18.0, tokens::TEXT);
    let mark = Rect::from_center_size(play.center(), Vec2::splat(16.0));
    let glyph = if shell.playing() { "pause" } else { "play" };
    let _ = icons::paint(
        ui.painter(),
        mark,
        glyph,
        Color32::from_rgb(0x12, 0x13, 0x0f),
    );
    play_response.on_hover_text(if shell.playing() { "Pause" } else { "Play" });

    let still = Rect::from_center_size(
        Pos2::new(rect.right() - 28.0, rect.center().y),
        Vec2::splat(32.0),
    );
    let still_response = ui.interact(still, ui.scope_id().with("still"), Sense::click());
    if still_response.clicked() {
        shell.unavailable("still");
    }
    ui.painter()
        .circle_filled(still.center(), 16.0, tokens::ICON_BUTTON);
    let camera = Rect::from_center_size(still.center(), Vec2::splat(16.0));
    let _ = icons::paint(ui.painter(), camera, "camera", tokens::MUTED);
    still_response.on_hover_text("Export still frame");
    shell.chrome.transport = true;
}

fn duration_label(shell: &Shell) -> String {
    let end = shell
        .draft()
        .map(|draft| draft.duration)
        .unwrap_or(Micros(0));
    shell.timecode_at(end)
}

fn inspector_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let padded = rect.shrink2(Vec2::new(16.0, 0.0));
    let mut details = ui.new_child(egui::UiBuilder::new().max_rect(padded));
    details.set_clip_rect(padded);
    let content_width = padded.width();
    ScrollArea::vertical()
        .id_salt("inspector")
        .show(&mut details, |ui| {
            ui.set_width(content_width);
            ui.add_space(14.0);
            if let Some(inspection) = shell.inspection() {
                section_title(ui, "Clip");
                fact(ui, "Name", &inspection.material, content_width);
                fact(
                    ui,
                    "Start",
                    &shell.timecode_at(inspection.start),
                    content_width,
                );
                fact(
                    ui,
                    "Duration",
                    &card_meta(inspection.duration.0),
                    content_width,
                );
                ui.add_space(12.0);
                clip_fields(shell, ui, view, &inspection);
            } else {
                section_title(ui, "Details");
                project_facts(shell, ui, content_width);
            }
        });
}

fn section_title(ui: &mut Ui, title: &str) {
    ui.label(
        RichText::new(title)
            .size(15.0)
            .strong()
            .color(tokens::EXPORT),
    );
    ui.add_space(14.0);
}

fn project_facts(shell: &Shell, ui: &mut Ui, width: f32) {
    fact(ui, "Name", &shell.project_name(), width);
    let saved = shell
        .project_folder()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "No project".to_owned());
    fact(ui, "Saved", &saved, width);
    let Some(draft) = shell.draft() else {
        return;
    };
    ui.add_space(8.0);
    let span = ui.available_rect_before_wrap().x_range();
    let y = ui.cursor().min.y;
    ui.painter().hline(
        span,
        y,
        egui::Stroke::new(1.0, Color32::from_white_alpha(20)),
    );
    ui.add_space(16.0);
    fact(ui, "Ratio", &ratio_label(&draft.canvas_config.ratio), width);
    fact(
        ui,
        "Resolution",
        &format!(
            "{}×{}",
            draft.canvas_config.width, draft.canvas_config.height
        ),
        width,
    );
    fact(ui, "Frame rate", &format!("{:.2}fps", draft.fps), width);
}

fn fact(ui: &mut Ui, label: &str, value: &str, row_width: f32) {
    let value_width = (row_width - 120.0).max(40.0);
    ui.horizontal(|ui| {
        ui.set_max_width(row_width);
        ui.spacing_mut().item_spacing.x = 12.0;
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(tokens::MUTED)),
        );
        let mut job = egui::text::LayoutJob::simple(
            value.to_owned(),
            FontId::proportional(13.0),
            tokens::TEXT,
            value_width,
        );
        job.wrap = egui::text::TextWrapping::truncate_at_width(value_width);
        let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
        let (rect, response) = ui.allocate_exact_size(Vec2::new(value_width, 18.0), Sense::hover());
        ui.painter().galley(
            rect.left_center() + Vec2::new(0.0, -galley.size().y / 2.0),
            galley,
            tokens::TEXT,
        );
        response.on_hover_text(value);
    });
    ui.add_space(4.0);
}

fn ratio_label(ratio: &str) -> String {
    if ratio.eq_ignore_ascii_case("original") {
        "Original".to_owned()
    } else {
        ratio.to_owned()
    }
}

fn clip_fields(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    inspection: &crate::shell::Inspection,
) {
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
}

fn timeline_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    panel(ui, rect);
    shell.chrome.timeline = true;
    let toolbar = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::TOOLBAR_HEIGHT));
    paint_toolbar(shell, ui, toolbar);
    if !shell.notice().is_empty() {
        paint_notice(
            ui,
            toolbar.center().x,
            toolbar.bottom() + tokens::RULER_HEIGHT + 8.0,
            shell.notice(),
        );
    }
    ui.painter().hline(
        toolbar.x_range(),
        toolbar.bottom(),
        egui::Stroke::new(1.0, tokens::GAP),
    );
    let body = Rect::from_min_max(toolbar.left_bottom(), rect.max);
    ruler_and_lanes(shell, ui, view, body);
    paint_clip_menu(shell, ui);
    let handle = Rect::from_min_size(
        Pos2::new(rect.min.x, rect.min.y - tokens::SPLIT),
        Vec2::new(rect.width(), tokens::SPLIT),
    );
    let window_height = ui.ctx().content_rect().height();
    drag_gap(ui, handle, |delta| {
        shell.drag_timeline(delta.y, window_height)
    });
}

fn paint_toolbar(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    let mut bar = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect.shrink2(Vec2::new(10.0, 4.0)))
            .layout(Layout::left_to_right(Align::Center)),
    );
    ScrollArea::horizontal()
        .id_salt("timeline-toolbar")
        .auto_shrink([false, true])
        .show(&mut bar, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                for item in TOOLS {
                    match item {
                        None => toolbar_separator(ui),
                        Some((name, tip)) => {
                            let enabled = tool_enabled(shell, name);
                            let active = match *name {
                                "split" => shell.blade(),
                                "marker" => shell.marking(),
                                "beat" => shell.beating(),
                                "guides" => shell.guides_shown(),
                                "lock" => shell.guides_locked(),
                                "linkage" => shell.linkage_enabled(),
                                "magnet" => shell.magnet_enabled(),
                                "snap" => shell.snap_enabled(),
                                "freeze" => shell.frozen(),
                                "reverse" => shell.reversed(),
                                "mirror" => shell.mirrored(),
                                _ => false,
                            };
                            let tip = match *name {
                                "linkage" if shell.linkage_enabled() => "Linkage on",
                                "linkage" => "Linkage off",
                                "magnet" if shell.magnet_enabled() => "Magnet on",
                                "magnet" => "Magnet off",
                                "snap" if shell.snap_enabled() => "Snap on",
                                "snap" => "Snap off",
                                "freeze" if shell.frozen() => "Frozen",
                                "freeze" => "Freeze",
                                "reverse" if shell.reversed() => "Reversed",
                                "reverse" => "Reverse",
                                "mirror" if shell.mirrored() => "Mirrored",
                                "mirror" => "Mirror",
                                _ => tip,
                            };
                            if toolbar_button(ui, name, tip, enabled, active).clicked() {
                                tool_clicked(shell, name);
                            }
                        }
                    }
                }
            });
        });
}

fn toolbar_separator(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(13.0, 28.0), Sense::hover());
    ui.painter().vline(
        rect.center().x,
        (rect.center().y - 8.0)..=(rect.center().y + 8.0),
        egui::Stroke::new(1.0, tokens::GAP),
    );
}

fn toolbar_button(
    ui: &mut Ui,
    name: &str,
    tip: &str,
    enabled: bool,
    active: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(28.0, 28.0), Sense::click());
    if active {
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(tokens::ICON_RADIUS),
            tokens::ICON_BUTTON,
        );
    }
    let color = if enabled || active {
        tokens::MUTED
    } else {
        tokens::FAINT
    };
    let mark = Rect::from_center_size(rect.center(), Vec2::splat(16.0));
    if let Err(error) = icons::paint(ui.painter(), mark, name, color) {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            error.to_string(),
            FontId::proportional(8.0),
            tokens::FAINT,
        );
    }
    response.on_hover_text(tip)
}

fn tool_enabled(shell: &Shell, name: &str) -> bool {
    match name {
        "undo" | "redo" | "select" | "delete" | "zoomIn" | "zoomOut" | "zoomFit" | "guides"
        | "lock" | "linkage" | "magnet" | "snap" => true,
        "split" | "marker" | "beat" => shell.has_clip(),
        "freeze" => shell.freeze_enabled(),
        "reverse" => shell.reverse_enabled(),
        "mirror" => shell.mirror_enabled(),
        _ => false,
    }
}

fn tool_clicked(shell: &mut Shell, name: &str) {
    if name == "select" {
        shell.clear_blade();
        shell.clear_marker();
        shell.clear_beat();
        return;
    }
    if !tool_enabled(shell, name) {
        return;
    }
    let result = match name {
        "undo" => shell.undo(),
        "redo" => shell.redo(),
        "zoomIn" => shell.zoom_in(),
        "zoomOut" => shell.zoom_out(),
        "zoomFit" => shell.zoom_fit(),
        "delete" => shell.delete_target(),
        "split" => {
            shell.toggle_blade();
            Ok(())
        }
        "marker" => {
            shell.toggle_marker();
            Ok(())
        }
        "beat" => {
            shell.toggle_beat();
            Ok(())
        }
        "guides" => {
            shell.toggle_guides();
            Ok(())
        }
        "lock" => {
            shell.toggle_guide_lock();
            Ok(())
        }
        "linkage" => shell.toggle_linkage(),
        "magnet" => shell.toggle_magnet(),
        "snap" => {
            shell.toggle_snap();
            Ok(())
        }
        "freeze" => shell.toggle_freeze(),
        "reverse" => shell.toggle_reverse(),
        "mirror" => shell.toggle_mirror(),
        _ => Ok(()),
    };
    if let Err(error) = result {
        shell.set_notice(error.to_string());
    }
}

const TOOLS: &[Option<(&str, &str)>] = &[
    Some(("undo", "Undo")),
    Some(("redo", "Redo")),
    None,
    Some(("select", "Select")),
    Some(("split", "Split")),
    Some(("delete", "Delete")),
    None,
    Some(("freeze", "Freeze")),
    Some(("reverse", "Reverse")),
    Some(("mirror", "Mirror")),
    Some(("rotate", "Rotate")),
    Some(("crop", "Crop")),
    Some(("transcribe", "Transcribe")),
    Some(("splitScenes", "Split scenes")),
    None,
    Some(("marker", "Marker")),
    Some(("beat", "Beat")),
    None,
    Some(("snap", "Snap")),
    Some(("magnet", "Magnet")),
    Some(("linkage", "Linkage")),
    Some(("guides", "Guides")),
    Some(("lock", "Lock")),
    None,
    Some(("zoomFit", "Zoom to fit")),
    Some(("zoomOut", "Zoom out")),
    Some(("zoomIn", "Zoom in")),
];

fn paint_notice(ui: &Ui, center_x: f32, top: f32, notice: &str) {
    let galley = ui.ctx().fonts_mut(|fonts| {
        fonts.layout_no_wrap(
            notice.to_owned(),
            FontId::proportional(12.0),
            Color32::from_rgb(0x14, 0x14, 0x14),
        )
    });
    let size = galley.size() + Vec2::new(24.0, 12.0);
    let rect = Rect::from_min_size(Pos2::new(center_x - size.x / 2.0, top), size);
    ui.painter()
        .rect_filled(rect, CornerRadius::same(99), Color32::WHITE);
    ui.painter().galley(
        rect.left_center() + Vec2::new(12.0, -galley.size().y / 2.0),
        galley,
        Color32::from_rgb(0x14, 0x14, 0x14),
    );
}

fn ruler_and_lanes(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let ruler = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler.left_bottom(), rect.max);
    let lanes_left = tracks.left() + tokens::TRACK_HEAD + tokens::COVER_WIDTH;
    let lane_area = Rect::from_min_max(Pos2::new(lanes_left, tracks.top()), tracks.max);
    shell.prepare_lane(lane_area.width().max(1.0));
    let (rate, drop_frame) = ruler_rate(shell);
    paint_ticks(
        ui,
        ruler,
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        rate,
        drop_frame,
    );
    let snap_at = shell.snap_line();
    let rows = shell.lanes();
    let flags: Vec<bool> = rows.iter().map(|lane| lane.main).collect();
    let boxes = lane_rects(flags, lane_area);
    let marquee = ui.interact(
        lane_area,
        ui.scope_id().with("marquee"),
        Sense::click_and_drag(),
    );
    if rows.is_empty() {
        paint_empty_main(ui, lane_area);
    }
    for (index, lane) in rows.iter().enumerate() {
        let Some(row) = boxes.get(index).copied() else {
            continue;
        };
        paint_lane(shell, ui, lane, row, lanes_left, index);
    }
    track_marquee(shell, ui, view, &marquee, lanes_left, &rows, &boxes);
    let playhead_x = lane_x(
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        shell.playhead().0,
    );
    if let Some(time) = snap_at {
        let x = lane_x(
            lanes_left,
            shell.pixel_size(),
            shell.view_origin().0,
            time.0,
        );
        if (lanes_left..=tracks.right()).contains(&x) {
            ui.painter().vline(
                x,
                ruler.top()..=tracks.bottom(),
                egui::Stroke::new(1.0, tokens::SELECTED),
            );
        }
    }
    if (lanes_left..=tracks.right()).contains(&playhead_x) {
        let mark = Rect::from_center_size(
            Pos2::new(playhead_x, ruler.bottom() - 6.0),
            Vec2::new(11.0, 12.0),
        );
        let response = ui.interact(mark, ui.scope_id().with("playhead"), Sense::drag());
        if response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let x = f64::from(pointer.x - lanes_left).max(0.0);
            if let Err(error) = shell.seek_ruler(x) {
                shell.set_notice(error.to_string());
            }
        }
        ui.painter()
            .rect_filled(mark, CornerRadius::same(2), tokens::PLAYHEAD);
        ui.painter().vline(
            playhead_x,
            tracks.y_range(),
            egui::Stroke::new(1.0, tokens::PLAYHEAD),
        );
        response.on_hover_text("Playhead");
    }
    paint_guides(shell, ui, lanes_left, tracks);
    paint_horizontal_guides(shell, ui, tracks);
    let ruler_hit = ui.interact(
        lane_ruler(ruler, lanes_left),
        ui.scope_id().with("ruler"),
        Sense::click_and_drag(),
    );
    let below_ruler = ruler.bottom() + 8.0;
    if ruler_hit.dragged()
        && let Some(pointer) = ruler_hit.interact_pointer_pos()
    {
        let x = f64::from(pointer.x - lanes_left).max(0.0);
        if pointer.y > below_ruler {
            if let Some(time) = shell.lane_time_at(x) {
                shell.preview_guide(time);
            }
        } else {
            shell.clear_guide_preview();
            if let Err(error) = shell.seek_ruler(x) {
                shell.set_notice(error.to_string());
            }
        }
    }
    if ruler_hit.drag_stopped()
        && let Some(time) = shell.take_guide_preview()
        && let Err(error) = shell.add_guide(time)
    {
        shell.set_notice(error.to_string());
    }
    let marker_clicked = paint_ruler_markers(shell, ui, lanes_left, ruler);
    if ruler_hit.clicked()
        && let Some(pointer) = ruler_hit.interact_pointer_pos()
    {
        let x = f64::from(pointer.x - lanes_left).max(0.0);
        if shell.marking() && !marker_clicked {
            if let Err(error) = shell.drop_marker(None, x, 0.0) {
                shell.set_notice(error.to_string());
            }
        } else if !marker_clicked && let Err(error) = shell.seek_ruler(x) {
            shell.set_notice(error.to_string());
        }
    }
    let corner = Rect::from_min_max(
        Pos2::new(tracks.left(), ruler.top()),
        Pos2::new((lanes_left - 1.0).max(tracks.left()), ruler.bottom()),
    );
    let corner_hit = ui.interact(
        corner,
        ui.scope_id().with("guide-corner"),
        Sense::click_and_drag(),
    );
    if corner_hit.dragged()
        && let Some(pointer) = corner_hit.interact_pointer_pos()
    {
        if pointer.y > below_ruler {
            if let Some(row) = sheet_row(tracks, pointer.y) {
                shell.preview_horizontal(row);
            }
        } else {
            shell.clear_horizontal_preview();
        }
    }
    if corner_hit.drag_stopped()
        && let Some(row) = shell.take_horizontal_preview()
        && let Err(error) = shell.add_horizontal(row)
    {
        shell.set_notice(error.to_string());
    }
    release_library_drop(shell, ui, tracks, lanes_left, &boxes);
    pan_timeline(shell, ui, lane_area);
    zoom_timeline(shell, ui, lane_area);
}

fn secondary_click(ui: &Ui) -> Option<Pos2> {
    ui.input(|input| {
        if input.pointer.button_clicked(egui::PointerButton::Secondary) {
            input.pointer.interact_pos()
        } else {
            None
        }
    })
}

fn paint_clip_menu(shell: &mut Shell, ui: &mut Ui) {
    let Some(menu) = shell.clip_menu() else {
        return;
    };
    if let Some(pos) = secondary_click(ui)
        && (pos.x, pos.y) != (menu.x, menu.y)
    {
        shell.close_clip_menu();
        return;
    }
    let width = 132.0;
    let row_height = 28.0;
    let mut labels = vec!["Copy", "Cut", "Delete", "Paste"];
    if shell.mirror_enabled() {
        labels.insert(0, "Mirror");
    }
    if shell.reverse_enabled() {
        labels.insert(0, "Reverse");
    }
    if shell.freeze_enabled() {
        labels.insert(0, "Freeze");
    }
    let rect = Rect::from_min_size(
        Pos2::new(menu.x, menu.y),
        Vec2::new(width, row_height * labels.len() as f32),
    );
    ui.painter()
        .rect_filled(rect, CornerRadius::same(8), tokens::PANEL);
    ui.painter().rect_stroke(
        rect,
        CornerRadius::same(8),
        egui::Stroke::new(1.0, tokens::LANE_LINE),
        egui::StrokeKind::Inside,
    );
    let mut acted = false;
    for (index, label) in labels.iter().enumerate() {
        let item = Rect::from_min_size(
            Pos2::new(rect.left(), rect.top() + index as f32 * row_height),
            Vec2::new(width, row_height),
        );
        let response = ui.interact(
            item,
            ui.scope_id().with(("clip-menu", *label)),
            Sense::click(),
        );
        if response.hovered() {
            ui.painter()
                .rect_filled(item, CornerRadius::same(6), tokens::ICON_BUTTON);
        }
        ui.painter().text(
            item.left_center() + Vec2::new(12.0, 0.0),
            Align2::LEFT_CENTER,
            *label,
            FontId::proportional(12.0),
            tokens::TEXT,
        );
        if response.clicked() {
            let result = match *label {
                "Copy" => shell.copy_selected(),
                "Cut" => shell.cut_selected(),
                "Paste" => shell.paste(),
                "Freeze" => shell.toggle_freeze(),
                "Reverse" => shell.toggle_reverse(),
                "Mirror" => shell.toggle_mirror(),
                _ => shell.remove_selected(),
            };
            if let Err(error) = result {
                shell.set_notice(error.to_string());
            }
            acted = true;
        }
    }
    let outside = ui.input(|input| {
        input.pointer.primary_clicked()
            && input
                .pointer
                .interact_pos()
                .is_none_or(|pos| !rect.contains(pos))
    });
    if acted || outside {
        shell.close_clip_menu();
    }
}

fn scroll_with_drag(shell: &mut Shell, ui: &Ui, x_in_view: f64) {
    if shell.drag_scroll(x_in_view) {
        ui.ctx().request_repaint();
    }
}

/// A wheel or a sideways swipe over the lanes moves the visible span.
fn pan_timeline(shell: &mut Shell, ui: &Ui, lane_area: Rect) {
    let (delta, command, pos) = ui.input(|input| {
        (
            input.smooth_scroll_delta,
            input.modifiers.command || input.modifiers.ctrl,
            input.pointer.hover_pos(),
        )
    });
    if command || delta == Vec2::ZERO {
        return;
    }
    let Some(pos) = pos else {
        return;
    };
    if !lane_area.contains(pos) {
        return;
    }
    let pixels = if delta.x.abs() >= delta.y.abs() {
        delta.x
    } else {
        delta.y
    };
    shell.pan_pixels(-f64::from(pixels));
}

/// A pinch, or a wheel with the zoom modifier, scales the lanes around the pointer.
fn zoom_timeline(shell: &mut Shell, ui: &Ui, lane_area: Rect) {
    let (factor, pos) = ui.input(|input| (input.zoom_delta(), input.pointer.hover_pos()));
    if (factor - 1.0).abs() < f32::EPSILON {
        return;
    }
    let Some(pos) = pos else {
        return;
    };
    if !lane_area.contains(pos) {
        return;
    }
    let x = f64::from(pos.x - lane_area.left()).max(0.0);
    if let Err(error) = shell.zoom_at(f64::from(factor), x) {
        shell.set_notice(error.to_string());
    }
}

fn release_library_drop(
    shell: &mut Shell,
    ui: &mut Ui,
    tracks: Rect,
    lanes_left: f32,
    boxes: &[Rect],
) {
    let Some(material_id) =
        egui::DragAndDrop::payload::<LibraryDrop>(ui.ctx()).map(|drop| drop.material_id.clone())
    else {
        return;
    };
    let Some(pos) = ui.input(|input| input.pointer.interact_pos()) else {
        return;
    };
    if !tracks.contains(pos) {
        return;
    }
    if let Some(index) = lane_under(boxes, pos)
        && let Some(row) = boxes.get(index)
    {
        ui.painter().rect_stroke(
            *row,
            CornerRadius::same(4),
            egui::Stroke::new(tokens::STROKE, tokens::EXPORT),
            egui::StrokeKind::Inside,
        );
    }
    if !ui.input(|input| input.pointer.any_released()) {
        return;
    }
    let _ = egui::DragAndDrop::take_payload::<LibraryDrop>(ui.ctx());
    let Some(index) = lane_under(boxes, pos) else {
        shell.set_notice("the drag missed the lanes".to_owned());
        return;
    };
    let x = f64::from((pos.x - lanes_left).max(0.0));
    let row = u32::try_from(index).unwrap_or(0);
    let y = f64::from(row) * tokens::LANE_HEIGHT + tokens::LANE_HEIGHT / 2.0;
    let landed = if material_id.starts_with("transition:") {
        shell.drop_transition(&material_id, x, y)
    } else {
        shell.drop_medium(&material_id, x, y)
    };
    if let Err(error) = landed {
        shell.set_notice(error.to_string());
    }
}

fn lane_under(boxes: &[Rect], pos: Pos2) -> Option<usize> {
    if let Some(index) = boxes.iter().position(|rect| rect.contains(pos)) {
        return Some(index);
    }
    boxes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (left.center().y - pos.y)
                .abs()
                .total_cmp(&(right.center().y - pos.y).abs())
        })
        .map(|(index, _)| index)
}

fn paint_library_ghost(shell: &Shell, ui: &Ui) {
    let Some(payload) = egui::DragAndDrop::payload::<LibraryDrop>(ui.ctx()) else {
        return;
    };
    let Some(pos) = ui.input(|input| input.pointer.interact_pos()) else {
        return;
    };
    let Some(card) = ghost_card(shell, &payload.material_id) else {
        return;
    };
    let rect = Rect::from_center_size(pos, Vec2::new(CARD_W, CARD_H));
    let layer = egui::LayerId::new(egui::Order::Tooltip, ui.scope_id().with("library-ghost"));
    paint_card_face(&ui.ctx().layer_painter(layer), rect, &card);
}

fn ghost_card(shell: &Shell, material_id: &str) -> Option<BinCard> {
    if let Some(offer) = crate::shell::TRANSITION_OFFERS
        .iter()
        .find(|offer| offer.card_id() == material_id)
    {
        return Some(BinCard {
            id: offer.card_id(),
            name: offer.name.to_owned(),
            meta: card_meta(offer.duration),
            on_timeline: false,
        });
    }
    let draft = shell.draft()?;
    let used: std::collections::HashSet<&str> = draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .map(|segment| segment.material_id.as_str())
        .collect();
    if let Some(video) = draft
        .materials
        .videos
        .iter()
        .find(|item| item.id.as_str() == material_id)
    {
        return Some(BinCard {
            id: video.id.as_str().to_owned(),
            name: file_label(&video.path),
            meta: card_meta(video.duration.0),
            on_timeline: used.contains(video.id.as_str()),
        });
    }
    let audio = draft
        .materials
        .audios
        .iter()
        .find(|item| item.id.as_str() == material_id)?;
    Some(BinCard {
        id: audio.id.as_str().to_owned(),
        name: file_label(&audio.path),
        meta: card_meta(audio.duration.0),
        on_timeline: used.contains(audio.id.as_str()),
    })
}

fn paint_guides(shell: &mut Shell, ui: &mut Ui, lanes_left: f32, tracks: Rect) {
    let pixel_size = shell.pixel_size();
    let origin = shell.view_origin().0;
    let selected = shell.selected_guide().map(|time| time.0);
    let times: Vec<i64> = shell.guides().iter().map(|time| time.0).collect();
    for time in times {
        let x = lane_x(lanes_left, pixel_size, origin, time);
        if x < lanes_left - 4.0 || x > tracks.right() + 4.0 {
            continue;
        }
        let hit = Rect::from_center_size(
            Pos2::new(x, tracks.center().y),
            Vec2::new(8.0, tracks.height()),
        );
        let response = ui.interact(hit, ui.scope_id().with(("guide", time)), Sense::click());
        if response.clicked() {
            shell.select_guide(Micros(time));
        }
        let color = if selected == Some(time) {
            tokens::SELECTED
        } else {
            tokens::EXPORT
        };
        ui.painter()
            .vline(x, tracks.y_range(), egui::Stroke::new(1.0, color));
    }
    if let Some(preview) = shell.guide_preview() {
        let x = lane_x(lanes_left, pixel_size, origin, preview.0);
        if (lanes_left..=tracks.right()).contains(&x) {
            ui.painter().vline(
                x,
                tracks.y_range(),
                egui::Stroke::new(1.0, tokens::EXPORT.gamma_multiply(0.55)),
            );
        }
    }
}

fn paint_horizontal_guides(shell: &mut Shell, ui: &mut Ui, tracks: Rect) {
    let selected = shell.selected_horizontal();
    let rows: Vec<i32> = shell.horizontal_guides().to_vec();
    let height = tracks.height().max(1.0);
    for row in rows {
        let y = tracks.top() + row as f32 / 1000.0 * height;
        let hit = Rect::from_center_size(
            Pos2::new(tracks.center().x, y),
            Vec2::new(tracks.width(), 8.0),
        );
        let response = ui.interact(hit, ui.scope_id().with(("hguide", row)), Sense::click());
        if response.clicked() {
            shell.select_horizontal(row);
        }
        let color = if selected == Some(row) {
            tokens::SELECTED
        } else {
            tokens::EXPORT
        };
        ui.painter()
            .hline(tracks.x_range(), y, egui::Stroke::new(1.0, color));
    }
    if let Some(row) = shell.horizontal_preview() {
        let y = tracks.top() + row as f32 / 1000.0 * height;
        ui.painter().hline(
            tracks.x_range(),
            y,
            egui::Stroke::new(1.0, tokens::EXPORT.gamma_multiply(0.55)),
        );
    }
}

fn sheet_row(tracks: Rect, y: f32) -> Option<i32> {
    let height = tracks.height();
    if height <= 0.0 || !y.is_finite() {
        return None;
    }
    let fraction = ((y - tracks.top()) / height).clamp(0.0, 1.0);
    Some((fraction * 1000.0).round() as i32)
}

fn lane_ruler(ruler: Rect, lanes_left: f32) -> Rect {
    Rect::from_min_max(Pos2::new(lanes_left, ruler.top()), ruler.max)
}

fn lane_x(lanes_left: f32, pixel_size: f64, origin: i64, time: i64) -> f32 {
    let delta = (time - origin) as f64 * pixel_size;
    lanes_left + delta as f32
}

fn paint_ticks(
    ui: &Ui,
    ruler: Rect,
    lanes_left: f32,
    pixel_size: f64,
    origin: i64,
    rate: FrameRate,
    drop_frame: bool,
) {
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return;
    }
    let grid = TickGrid {
        ui,
        ruler,
        lanes_left,
        pixel_size,
        origin,
        width: f64::from(ruler.right() - lanes_left),
    };
    if let Some(stride) = frame_stride(pixel_size, rate) {
        paint_frame_ticks(&grid, rate, drop_frame, stride);
        return;
    }
    let step = second_step(pixel_size);
    let origin = grid.origin as f64;
    let mut time = if origin <= 0.0 {
        0.0
    } else {
        (origin / step).floor() * step
    };
    if time < 0.0 {
        time = 0.0;
    }
    while (time - origin) * pixel_size <= grid.width {
        paint_major(&grid, time, second_label(time));
        paint_minor(&grid, time, time + step / 2.0);
        time += step;
        if step <= 0.0 {
            break;
        }
    }
}

struct TickGrid<'a> {
    ui: &'a Ui,
    ruler: Rect,
    lanes_left: f32,
    pixel_size: f64,
    origin: i64,
    width: f64,
}

/// One, two, or five frames, when that gap is wide enough to read. Otherwise the
/// ruler stays on whole seconds.
fn frame_stride(pixel_size: f64, rate: FrameRate) -> Option<i64> {
    let frame = rate.frame_start(1)?;
    let width = frame.0 as f64 * pixel_size;
    if frame.0 <= 0 || !width.is_finite() || width <= 0.0 {
        return None;
    }
    [1, 2, 5]
        .into_iter()
        .find(|stride| *stride as f64 * width >= 72.0)
}

fn paint_frame_ticks(grid: &TickGrid<'_>, rate: FrameRate, drop_frame: bool, stride: i64) {
    let mut index = rate.frame_index(Micros(grid.origin.max(0))).unwrap_or(0);
    if stride > 1 {
        index -= index.rem_euclid(stride);
    }
    let origin = grid.origin as f64;
    for _ in 0..10_000 {
        let Some(start) = rate.frame_start(index) else {
            return;
        };
        let time = start.0 as f64;
        if (time - origin) * grid.pixel_size > grid.width {
            return;
        }
        paint_major(grid, time, ruler_label(time, 1.0, rate, drop_frame));
        let next = index.saturating_add(stride);
        let Some(next_start) = rate.frame_start(next) else {
            return;
        };
        if next <= index {
            return;
        }
        let minor = (time + next_start.0 as f64) / 2.0;
        paint_minor(grid, time, minor);
        index = next;
    }
}

fn second_step(pixel_size: f64) -> f64 {
    const STEPS: [f64; 9] = [
        1_000_000.0,
        2_000_000.0,
        5_000_000.0,
        10_000_000.0,
        30_000_000.0,
        60_000_000.0,
        300_000_000.0,
        600_000_000.0,
        1_800_000_000.0,
    ];
    STEPS
        .into_iter()
        .find(|step| step * pixel_size >= 72.0)
        .unwrap_or(STEPS[STEPS.len() - 1])
}

fn paint_major(grid: &TickGrid<'_>, time: f64, label: String) {
    let x = lane_x(
        grid.lanes_left,
        grid.pixel_size,
        grid.origin,
        time.round() as i64,
    );
    if x < grid.lanes_left - 1.0 || x > grid.ruler.right() + 1.0 {
        return;
    }
    grid.ui.painter().vline(
        x,
        (grid.ruler.bottom() - 12.0)..=grid.ruler.bottom(),
        egui::Stroke::new(1.0, tokens::MUTED),
    );
    grid.ui.painter().text(
        Pos2::new(x + 4.0, grid.ruler.bottom() - 14.0),
        Align2::LEFT_BOTTOM,
        label,
        FontId::proportional(10.0),
        tokens::FAINT,
    );
}

fn paint_minor(grid: &TickGrid<'_>, major: f64, minor: f64) {
    let x = lane_x(
        grid.lanes_left,
        grid.pixel_size,
        grid.origin,
        major.round() as i64,
    );
    let minor_x = lane_x(
        grid.lanes_left,
        grid.pixel_size,
        grid.origin,
        minor.round() as i64,
    );
    let origin = grid.origin as f64;
    if (minor - origin) * grid.pixel_size <= grid.width && minor_x - x >= 8.0 {
        grid.ui.painter().vline(
            minor_x,
            (grid.ruler.bottom() - 6.0)..=grid.ruler.bottom(),
            egui::Stroke::new(1.0, tokens::MUTED),
        );
    }
}

fn ruler_rate(shell: &Shell) -> (FrameRate, bool) {
    match shell.draft() {
        Some(draft) => (
            draft.frame_rate().unwrap_or(FrameRate::FPS_30),
            draft.is_drop_frame_timecode,
        ),
        None => (FrameRate::FPS_30, false),
    }
}

/// A step shorter than one second names the frame. A longer step stays on the second.
fn ruler_label(micros: f64, step: f64, rate: FrameRate, drop_frame: bool) -> String {
    if step < 1_000_000.0
        && let Some(label) = sub_second_label(micros, rate, drop_frame)
    {
        return label;
    }
    second_label(micros)
}

fn sub_second_label(micros: f64, rate: FrameRate, drop_frame: bool) -> Option<String> {
    if !micros.is_finite() || micros < 0.0 {
        return None;
    }
    let rounded = micros.round();
    if rounded > i64::MAX as f64 {
        return None;
    }
    let label = rate.label(Micros(rounded as i64), drop_frame)?;
    let separator = if label.drop_frame { ';' } else { ':' };
    if label.hours > 0 {
        Some(format!(
            "{:02}:{:02}:{:02}{separator}{:02}",
            label.hours, label.minutes, label.seconds, label.frames
        ))
    } else {
        Some(format!(
            "{:02}:{:02}{separator}{:02}",
            label.minutes, label.seconds, label.frames
        ))
    }
}

fn second_label(micros: f64) -> String {
    let total = (micros / 1_000_000.0).floor();
    if !total.is_finite() || total < 0.0 {
        return String::new();
    }
    let total = total as i64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn paint_empty_main(ui: &Ui, area: Rect) {
    let height = 64.0;
    let lane = Rect::from_min_size(
        Pos2::new(area.left(), area.center().y - height / 2.0),
        Vec2::new(area.width(), height),
    );
    ui.painter()
        .rect_filled(lane, CornerRadius::ZERO, tokens::LANE);
    ui.painter().rect_stroke(
        lane,
        CornerRadius::ZERO,
        egui::Stroke::new(1.0, tokens::LANE_LINE),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        lane.left_center() + Vec2::new(14.0, 0.0),
        Align2::LEFT_CENTER,
        "No clips",
        FontId::proportional(12.0),
        tokens::MUTED,
    );
    paint_lane_head(ui, lane, "media", "Main");
}

fn paint_lane(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    row: Rect,
    lanes_left: f32,
    index: usize,
) {
    let fill = if lane.main {
        tokens::LANE
    } else {
        tokens::PANEL
    };
    ui.painter().rect_filled(row, CornerRadius::ZERO, fill);
    if lane.main && lane.clips.is_empty() {
        ui.painter().rect_stroke(
            row,
            CornerRadius::ZERO,
            egui::Stroke::new(1.0, tokens::LANE_LINE),
            egui::StrokeKind::Inside,
        );
        ui.painter().text(
            row.left_center() + Vec2::new(14.0, 0.0),
            Align2::LEFT_CENTER,
            "No clips",
            FontId::proportional(12.0),
            tokens::MUTED,
        );
    }
    let name = lane_name(&lane.kind, lane.main);
    paint_lane_head(ui, row, lane_icon(&lane.kind), &name);
    let lock = Rect::from_center_size(
        Pos2::new(row.left() - tokens::COVER_WIDTH - 18.0, row.center().y),
        Vec2::splat(24.0),
    );
    let lock_response = ui.interact(lock, ui.scope_id().with(("lock", index)), Sense::click());
    if lock_response.clicked()
        && let Err(error) =
            shell.toggle_attribute(lane.track_id.clone(), cutlass_model::TrackAttribute::LOCK)
    {
        shell.set_notice(error.to_string());
    }
    let _ = icons::paint(
        ui.painter(),
        Rect::from_center_size(lock.center(), Vec2::splat(14.0)),
        "lock",
        if lane.lock {
            tokens::TEXT
        } else {
            tokens::FAINT
        },
    );
    lock_response.on_hover_text(if lane.lock { "Locked" } else { "Lock" });
    lane_switches(shell, ui, lane, row, index);
    let followers: Vec<_> = shell
        .move_previews()
        .into_iter()
        .filter(|preview| preview.track_id == lane.track_id)
        .collect();
    for clip in &lane.clips {
        paint_clip(shell, ui, lane, clip, row, lanes_left, index);
    }
    paint_transitions(shell, ui, lane, row, lanes_left);
    for preview in &followers {
        paint_move_follower(shell, ui, lane, preview, row, lanes_left);
    }
}

fn paint_lane_head(ui: &Ui, row: Rect, icon: &str, label: &str) {
    let center = Pos2::new(
        row.left() - tokens::TRACK_HEAD - tokens::COVER_WIDTH + 22.0,
        row.center().y,
    );
    let _ = icons::paint(
        ui.painter(),
        Rect::from_center_size(center, Vec2::splat(16.0)),
        icon,
        tokens::MUTED,
    );
    ui.painter().text(
        Pos2::new(center.x + 14.0, center.y),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(12.0),
        tokens::MUTED,
    );
}

fn lane_switches(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    row: Rect,
    index: usize,
) {
    let origin = Pos2::new(row.left() - tokens::COVER_WIDTH + 4.0, row.center().y - 8.0);
    for (offset, label, on, bit) in [
        (0.0, "M", lane.mute, cutlass_model::TrackAttribute::MUTE),
        (18.0, "H", lane.hide, cutlass_model::TrackAttribute::HIDE),
        (36.0, "S", lane.solo, cutlass_model::TrackAttribute::SOLO),
    ] {
        let rect = Rect::from_min_size(origin + Vec2::new(offset, 0.0), Vec2::splat(16.0));
        let response = ui.interact(rect, ui.scope_id().with((label, index)), Sense::click());
        if response.clicked()
            && let Err(error) = shell.toggle_attribute(lane.track_id.clone(), bit)
        {
            shell.set_notice(error.to_string());
        }
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(10.0),
            if on { tokens::TEXT } else { tokens::FAINT },
        );
        response.on_hover_text(match label {
            "M" => "Mute",
            "H" => "Hide",
            _ => "Solo",
        });
    }
}

fn track_marquee(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    response: &egui::Response,
    lanes_left: f32,
    rows: &[crate::shell::LaneView],
    boxes: &[Rect],
) {
    if response.drag_started()
        && let Some(pos) = ui.input(|input| input.pointer.press_origin())
    {
        view.marquee_press = Some(pos);
        view.marquee = None;
    }
    let pos = pointer_pos(ui, response);
    let moved = view
        .marquee_press
        .is_some_and(|press| pos.is_some_and(|pos| (pos - press).length_sq() >= 16.0));
    if moved
        && response.dragged()
        && let (Some(press), Some(pos)) = (view.marquee_press, pos)
    {
        view.marquee = Some(Rect::from_two_pos(press, pos));
        scroll_with_drag(shell, ui, f64::from(pos.x - lanes_left).max(0.0));
    }
    let shift = ui.input(|input| input.modifiers.shift);
    if response.drag_stopped() {
        if moved && let Some(rect) = view.marquee {
            shell.select_box(clips_touching(shell, rows, boxes, lanes_left, rect), shift);
        } else if !shift {
            shell.select(None);
        }
        view.marquee_press = None;
        view.marquee = None;
    } else if response.clicked() && !shift {
        shell.select(None);
    }
    if let Some(rect) = view.marquee {
        ui.painter().rect_filled(
            rect,
            CornerRadius::ZERO,
            tokens::SELECTED.gamma_multiply(0.25),
        );
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            egui::Stroke::new(1.0, tokens::SELECTED),
            egui::StrokeKind::Outside,
        );
    }
}

fn clips_touching(
    shell: &Shell,
    rows: &[crate::shell::LaneView],
    boxes: &[Rect],
    lanes_left: f32,
    rect: Rect,
) -> Vec<SegmentId> {
    let mut ids = Vec::new();
    for (lane, row) in rows.iter().zip(boxes.iter().copied()) {
        for clip in &lane.clips {
            let (start, duration) = preview_bounds(shell, clip);
            let bounds = clip_bounds(shell, lane.main, start, duration, row, lanes_left);
            if bounds.intersects(rect) {
                ids.push(clip.id.clone());
            }
        }
    }
    ids
}

fn clip_bounds(
    shell: &Shell,
    main: bool,
    start: Micros,
    duration: Micros,
    row: Rect,
    lanes_left: f32,
) -> Rect {
    let (inset, height) = if main { (8.0, 48.0) } else { (6.0, 28.0) };
    let x = lane_x(
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        start.0,
    );
    let width = (duration.0 as f32 * shell.pixel_size() as f32).max(8.0);
    Rect::from_min_size(Pos2::new(x, row.top() + inset), Vec2::new(width, height))
}

fn paint_transitions(
    shell: &Shell,
    ui: &Ui,
    lane: &crate::shell::LaneView,
    row: Rect,
    lanes_left: f32,
) {
    let Some(draft) = shell.draft() else {
        return;
    };
    let Some(track) = draft.tracks.iter().find(|track| track.id == lane.track_id) else {
        return;
    };
    for segment in &track.segments {
        let Some(transition) = segment.extra_material_refs.iter().find_map(|id| {
            draft
                .materials
                .transitions
                .iter()
                .find(|item| &item.id == id)
        }) else {
            continue;
        };
        let Some((start, end)) = cutlass_model::blend_window(
            segment.target.end(),
            transition.duration,
            transition.is_overlap,
        ) else {
            continue;
        };
        let x0 = lane_x(
            lanes_left,
            shell.pixel_size(),
            shell.view_origin().0,
            start.0,
        );
        let x1 = lane_x(lanes_left, shell.pixel_size(), shell.view_origin().0, end.0);
        if x1 <= x0 {
            continue;
        }
        let mark = Rect::from_min_max(
            Pos2::new(x0, row.bottom() - 6.0),
            Pos2::new(x1, row.bottom() - 2.0),
        );
        ui.painter()
            .rect_filled(mark.intersect(row), CornerRadius::ZERO, tokens::EXPORT);
    }
}

fn paint_clip(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    clip: &crate::shell::ClipSpan,
    row: Rect,
    lanes_left: f32,
    index: usize,
) {
    let (start, duration) = preview_bounds(shell, clip);
    let clip_rect = clip_bounds(shell, lane.main, start, duration, row, lanes_left);
    let visible = clip_rect.intersect(row);
    if visible.width() <= 0.0 || visible.height() <= 0.0 {
        return;
    }
    if let Some(pos) = secondary_click(ui)
        && clip_rect.contains(pos)
    {
        if !shell.is_selected(&clip.id) {
            shell.select(Some(clip.id.clone()));
        }
        shell.open_clip_menu(pos.x, pos.y);
    }
    let selected = shell.is_selected(&clip.id);
    let ghost = shell.moving(&clip.id);
    let painter = ui.painter().with_clip_rect(row);
    let fill = if ghost {
        lane_color(&lane.kind).gamma_multiply(0.35)
    } else {
        lane_color(&lane.kind)
    };
    painter.rect_filled(clip_rect, CornerRadius::same(7), fill);
    if selected {
        painter.rect_stroke(
            clip_rect,
            CornerRadius::same(7),
            egui::Stroke::new(1.5, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    let painter = ui
        .painter()
        .with_clip_rect(clip_rect.shrink(6.0).intersect(row));
    painter.text(
        clip_rect.left_center() + Vec2::new(8.0, 0.0),
        Align2::LEFT_CENTER,
        &clip.name,
        FontId::proportional(11.0),
        if ghost {
            Color32::WHITE.gamma_multiply(0.45)
        } else {
            Color32::WHITE
        },
    );
    let span = format!(
        "{} · {}–{}",
        clip.name,
        shell.timecode_at(start),
        shell.timecode_at(Micros(start.0.saturating_add(duration.0)))
    );
    let (left_edge, right_edge) = edge_strips(visible);
    let body = Rect::from_min_max(
        Pos2::new(left_edge.right(), visible.top()),
        Pos2::new(right_edge.left(), visible.bottom()),
    );
    if body.width() > 0.0 && shell.blade() {
        let response = ui.interact(
            body,
            ui.scope_id().with(("clip", index, clip.id.as_str())),
            Sense::click(),
        );
        response.clone().on_hover_text(span.clone());
        if response.hovered() {
            ui.set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        if response.clicked()
            && let Some(pos) = pointer_pos(ui, &response)
        {
            let x = f64::from(pos.x - lanes_left).max(0.0);
            if let Err(error) = shell.split_pointer(clip.id.clone(), x, editor_lane_y(index)) {
                shell.set_notice(error.to_string());
            }
        }
    } else if body.width() > 0.0 && shell.marking() {
        let response = ui.interact(
            body,
            ui.scope_id().with(("clip", index, clip.id.as_str())),
            Sense::click(),
        );
        response.clone().on_hover_text(span.clone());
        if response.hovered() {
            ui.set_cursor_icon(egui::CursorIcon::VerticalText);
        }
        if response.clicked()
            && let Some(pos) = pointer_pos(ui, &response)
        {
            let x = f64::from(pos.x - lanes_left).max(0.0);
            if let Err(error) = shell.drop_marker(Some(clip.id.clone()), x, editor_lane_y(index)) {
                shell.set_notice(error.to_string());
            }
        }
    } else if body.width() > 0.0 && shell.beating() {
        let response = ui.interact(
            body,
            ui.scope_id().with(("clip", index, clip.id.as_str())),
            Sense::click(),
        );
        response.clone().on_hover_text(span.clone());
        if response.hovered() {
            ui.set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if response.clicked()
            && let Some(pos) = pointer_pos(ui, &response)
        {
            let x = f64::from(pos.x - lanes_left).max(0.0);
            if let Err(error) = shell.drop_beat(clip.id.clone(), x, editor_lane_y(index)) {
                shell.set_notice(error.to_string());
            }
        }
    } else if body.width() > 0.0 {
        let response = ui.interact(
            body,
            ui.scope_id().with(("clip", index, clip.id.as_str())),
            Sense::drag(),
        );
        response.clone().on_hover_text(span.clone());
        if response.drag_started()
            && let Some(pos) = pointer_pos(ui, &response)
        {
            shell.begin_clip_drag(clip.id.clone(), None, (pos.x, pos.y));
            if !ui.input(|input| input.modifiers.shift) && !shell.is_selected(&clip.id) {
                shell.select(Some(clip.id.clone()));
            }
        }
        let pos = pointer_pos(ui, &response);
        let moved = pos.is_some_and(|pos| shell.clip_drag_moved(pos.x, pos.y));
        if moved
            && response.dragged()
            && shell.clip_edge(&clip.id).is_none()
            && let Some(pos) = pos
        {
            let x = f64::from(pos.x - lanes_left).max(0.0);
            scroll_with_drag(shell, ui, x);
            shell.preview_move(clip.id.clone(), x, editor_lane_y(index));
        }
        if response.drag_stopped() && !moved {
            if ui.input(|input| input.modifiers.shift) {
                shell.toggle_selected(clip.id.clone());
            } else {
                shell.select(Some(clip.id.clone()));
            }
            shell.cancel_clip_drag();
        } else {
            finish_clip_pointer(shell, ui, &response, &clip.id, lanes_left, index, None);
        }
    }
    drag_clip_edge(
        shell,
        ui,
        clip,
        left_edge,
        TrimEdge::Start,
        lanes_left,
        index,
    );
    drag_clip_edge(
        shell,
        ui,
        clip,
        right_edge,
        TrimEdge::End,
        lanes_left,
        index,
    );
    paint_clip_markers(shell, ui, clip, lanes_left, row);
    paint_clip_beats(shell, ui, clip, lanes_left, row);
}

fn drag_clip_edge(
    shell: &mut Shell,
    ui: &mut Ui,
    clip: &crate::shell::ClipSpan,
    rect: Rect,
    edge: TrimEdge,
    lanes_left: f32,
    index: usize,
) {
    if rect.width() < 1.0 || rect.height() <= 0.0 {
        return;
    }
    let response = ui.interact(
        rect,
        ui.scope_id()
            .with(("clip-edge", index, clip.id.as_str(), edge_name(edge))),
        Sense::drag(),
    );
    response.clone().on_hover_text(&clip.name);
    if response.hovered() || response.dragged() {
        ui.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    if response.drag_started()
        && let Some(pos) = pointer_pos(ui, &response)
    {
        shell.begin_clip_drag(clip.id.clone(), Some(edge), (pos.x, pos.y));
    }
    if response.dragged()
        && let Some(pos) = pointer_pos(ui, &response)
    {
        let x = f64::from(pos.x - lanes_left).max(0.0);
        scroll_with_drag(shell, ui, x);
        shell.preview_edge(clip.id.clone(), edge, x);
    }
    finish_clip_pointer(
        shell,
        ui,
        &response,
        &clip.id,
        lanes_left,
        index,
        Some(edge),
    );
}

fn paint_ruler_markers(shell: &mut Shell, ui: &mut Ui, lanes_left: f32, ruler: Rect) -> bool {
    let marks: Vec<_> = shell
        .markers()
        .into_iter()
        .filter(|mark| mark.segment_id.is_none())
        .collect();
    let mut clicked = false;
    for mark in marks {
        let x = lane_x(
            lanes_left,
            shell.pixel_size(),
            shell.view_origin().0,
            mark.time.0,
        );
        let flag = marker_flag(x, ruler.bottom() - 16.0);
        if !ruler.intersects(flag) {
            continue;
        }
        if paint_marker_flag(shell, ui, &mark, flag, "ruler") {
            clicked = true;
        }
    }
    clicked
}

fn paint_clip_markers(
    shell: &mut Shell,
    ui: &mut Ui,
    clip: &crate::shell::ClipSpan,
    lanes_left: f32,
    row: Rect,
) {
    let marks: Vec<_> = shell
        .markers()
        .into_iter()
        .filter(|mark| mark.segment_id.as_ref() == Some(&clip.id))
        .collect();
    for mark in marks {
        let x = lane_x(
            lanes_left,
            shell.pixel_size(),
            shell.view_origin().0,
            mark.time.0,
        );
        let flag = marker_flag(x, row.top() + 4.0);
        if !row.intersects(flag) {
            continue;
        }
        let _ = paint_marker_flag(shell, ui, &mark, flag, clip.id.as_str());
    }
}

fn marker_flag(x: f32, top: f32) -> Rect {
    Rect::from_min_size(Pos2::new(x - 4.0, top), Vec2::new(8.0, 12.0))
}

fn paint_marker_flag(
    shell: &mut Shell,
    ui: &mut Ui,
    mark: &crate::shell::MarkerView,
    flag: Rect,
    salt: &str,
) -> bool {
    let response = ui.interact(
        flag,
        ui.scope_id().with(("marker", salt, mark.stored_start)),
        Sense::click(),
    );
    let selected = shell.marker_selected(&mark.segment_id, mark.stored_start);
    ui.painter()
        .rect_filled(flag, CornerRadius::same(1), marker_color(&mark.color));
    if selected {
        ui.painter().rect_stroke(
            flag,
            CornerRadius::same(1),
            egui::Stroke::new(1.0, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    response.clone().on_hover_text(marker_tip(&mark.title));
    if response.clicked() {
        shell.select_marker(mark.segment_id.clone(), mark.stored_start);
        return true;
    }
    false
}

fn paint_clip_beats(
    shell: &mut Shell,
    ui: &mut Ui,
    clip: &crate::shell::ClipSpan,
    lanes_left: f32,
    row: Rect,
) {
    let beats: Vec<_> = shell
        .beats()
        .into_iter()
        .filter(|beat| beat.segment_id == clip.id)
        .collect();
    for beat in beats {
        let x = lane_x(
            lanes_left,
            shell.pixel_size(),
            shell.view_origin().0,
            beat.time.0,
        );
        let hit = Rect::from_center_size(Pos2::new(x, row.center().y), Vec2::new(12.0, 16.0));
        if !row.intersects(hit) {
            continue;
        }
        let response = ui.interact(
            hit,
            ui.scope_id().with(("beat", clip.id.as_str(), beat.source)),
            Sense::click(),
        );
        let selected = shell.beat_selected(&clip.id, beat.source);
        let center = Pos2::new(x, row.center().y);
        ui.painter()
            .circle_filled(center, 4.0, Color32::from_rgb(0xF4, 0xFB, 0xF8));
        if selected {
            ui.painter()
                .circle_stroke(center, 6.0, egui::Stroke::new(1.5, tokens::SELECTED));
        }
        response.clone().on_hover_text("Beat");
        if response.clicked() {
            shell.select_beat(clip.id.clone(), beat.source);
        }
    }
}

fn marker_tip(title: &str) -> &str {
    if title.is_empty() { "Marker" } else { title }
}

fn marker_color(hex: &str) -> Color32 {
    let bytes = hex.trim().trim_start_matches('#');
    if bytes.len() != 6 {
        return Color32::from_rgb(0xE8, 0xB9, 0x31);
    }
    let Ok(value) = u32::from_str_radix(bytes, 16) else {
        return Color32::from_rgb(0xE8, 0xB9, 0x31);
    };
    let red = u8::try_from((value >> 16) & 0xff).unwrap_or(0);
    let green = u8::try_from((value >> 8) & 0xff).unwrap_or(0);
    let blue = u8::try_from(value & 0xff).unwrap_or(0);
    Color32::from_rgb(red, green, blue)
}

fn finish_clip_pointer(
    shell: &mut Shell,
    ui: &Ui,
    response: &egui::Response,
    id: &SegmentId,
    lanes_left: f32,
    index: usize,
    edge: Option<TrimEdge>,
) {
    if !response.drag_stopped() {
        return;
    }
    let Some(pos) = pointer_pos(ui, response) else {
        shell.cancel_clip_drag();
        return;
    };
    let x = f64::from(pos.x - lanes_left).max(0.0);
    let y = editor_lane_y(index);
    if edge.is_some() && shell.clip_edge(id).is_none() {
        shell.begin_clip_drag(id.clone(), edge, (pos.x, pos.y));
    }
    if let Err(error) = shell.finish_clip_drag(id.clone(), x, y) {
        shell.set_notice(error.to_string());
    }
}

fn editor_lane_y(index: usize) -> f64 {
    f64::from(u32::try_from(index).unwrap_or(0)) * tokens::LANE_HEIGHT + tokens::LANE_HEIGHT / 2.0
}

fn paint_move_follower(
    shell: &Shell,
    ui: &Ui,
    lane: &crate::shell::LaneView,
    preview: &MovePreview,
    row: Rect,
    lanes_left: f32,
) {
    let (inset, height) = if lane.main { (8.0, 48.0) } else { (6.0, 28.0) };
    let x = lane_x(
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        preview.start.0,
    );
    let width = (preview.duration.0 as f32 * shell.pixel_size() as f32).max(8.0);
    let clip_rect = Rect::from_min_size(Pos2::new(x, row.top() + inset), Vec2::new(width, height));
    let visible = clip_rect.intersect(row);
    if visible.width() <= 0.0 || visible.height() <= 0.0 {
        return;
    }
    let painter = ui.painter().with_clip_rect(row);
    painter.rect_filled(clip_rect, CornerRadius::same(7), lane_color(&lane.kind));
    if shell.is_selected(&preview.id) {
        painter.rect_stroke(
            clip_rect,
            CornerRadius::same(7),
            egui::Stroke::new(1.5, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    let painter = ui
        .painter()
        .with_clip_rect(clip_rect.shrink(6.0).intersect(row));
    painter.text(
        clip_rect.left_center() + Vec2::new(8.0, 0.0),
        Align2::LEFT_CENTER,
        &preview.name,
        FontId::proportional(11.0),
        Color32::WHITE,
    );
}

fn pointer_pos(ui: &Ui, response: &egui::Response) -> Option<Pos2> {
    response
        .interact_pointer_pos()
        .or_else(|| ui.input(|input| input.pointer.interact_pos()))
}

fn edge_name(edge: TrimEdge) -> &'static str {
    match edge {
        TrimEdge::Start => "start",
        TrimEdge::End => "end",
    }
}

fn preview_bounds(shell: &Shell, clip: &crate::shell::ClipSpan) -> (Micros, Micros) {
    let Some((edge, time)) = shell.preview_span(&clip.id) else {
        return (clip.start, clip.duration);
    };
    let end = clip.start.0.saturating_add(clip.duration.0);
    match edge {
        TrimEdge::Start => {
            let start = time.0.clamp(0, end.saturating_sub(1));
            (Micros(start), Micros(end.saturating_sub(start)))
        }
        TrimEdge::End => {
            let duration = time.0.saturating_sub(clip.start.0).max(1);
            (clip.start, Micros(duration))
        }
    }
}

fn edge_strips(visible: Rect) -> (Rect, Rect) {
    let edge = TRIM_EDGE.min(visible.width() * 0.25).max(0.0);
    let left = Rect::from_min_max(
        visible.left_top(),
        Pos2::new(
            (visible.left() + edge).min(visible.right()),
            visible.bottom(),
        ),
    );
    let right = Rect::from_min_max(
        Pos2::new((visible.right() - edge).max(visible.left()), visible.top()),
        visible.right_bottom(),
    );
    (left, right)
}

fn lane_rects(main_flags: Vec<bool>, area: Rect) -> Vec<Rect> {
    let Some(main) = main_flags.iter().position(|main| *main) else {
        return main_flags
            .iter()
            .enumerate()
            .map(|(index, _)| {
                Rect::from_min_size(
                    Pos2::new(area.left(), area.top() + index as f32 * 48.0),
                    Vec2::new(area.width(), 40.0),
                )
            })
            .collect();
    };
    let height = |index: usize| if main_flags[index] { 64.0 } else { 40.0 };
    let mut tops = vec![0.0; main_flags.len()];
    tops[main] = area.center().y - height(main) / 2.0;
    let mut y = tops[main];
    for index in (0..main).rev() {
        y -= 8.0 + height(index);
        tops[index] = y;
    }
    y = tops[main] + height(main);
    let mut index = main + 1;
    while index < main_flags.len() {
        y += 8.0;
        tops[index] = y;
        y += height(index);
        index += 1;
    }
    tops.into_iter()
        .enumerate()
        .map(|(index, top)| {
            Rect::from_min_size(
                Pos2::new(area.left(), top),
                Vec2::new(area.width().max(0.0), height(index)),
            )
        })
        .collect()
}

fn lane_name(kind: &cutlass_model::TrackType, main: bool) -> String {
    if main {
        return "Main".to_owned();
    }
    match kind {
        cutlass_model::TrackType::Audio => "Audio",
        cutlass_model::TrackType::Text
        | cutlass_model::TrackType::TextToVideoText
        | cutlass_model::TrackType::TextToVideoTextAudio => "Text",
        cutlass_model::TrackType::Sticker => "Sticker",
        cutlass_model::TrackType::Effect | cutlass_model::TrackType::PluginEffect => "Effect",
        cutlass_model::TrackType::Filter => "Filter",
        cutlass_model::TrackType::Adjust => "Adjust",
        cutlass_model::TrackType::Other(name) => return name.clone(),
        _ => "Video",
    }
    .to_owned()
}

fn lane_icon(kind: &cutlass_model::TrackType) -> &'static str {
    match kind {
        cutlass_model::TrackType::Audio => "audio",
        cutlass_model::TrackType::Text
        | cutlass_model::TrackType::TextToVideoText
        | cutlass_model::TrackType::TextToVideoTextAudio => "text",
        cutlass_model::TrackType::Sticker => "stickers",
        cutlass_model::TrackType::Effect | cutlass_model::TrackType::PluginEffect => "effects",
        cutlass_model::TrackType::Filter => "filters",
        cutlass_model::TrackType::Adjust => "adjustments",
        _ => "media",
    }
}

fn lane_color(kind: &cutlass_model::TrackType) -> Color32 {
    match kind {
        cutlass_model::TrackType::Audio => Color32::from_rgb(0x4e, 0x8f, 0x7c),
        cutlass_model::TrackType::Text
        | cutlass_model::TrackType::TextToVideoText
        | cutlass_model::TrackType::TextToVideoTextAudio => Color32::from_rgb(0x9a, 0x86, 0xb0),
        cutlass_model::TrackType::Sticker => Color32::from_rgb(0xc4, 0xa1, 0x5a),
        cutlass_model::TrackType::Effect | cutlass_model::TrackType::PluginEffect => {
            Color32::from_rgb(0xc4, 0x7a, 0x5a)
        }
        cutlass_model::TrackType::Filter => Color32::from_rgb(0x6a, 0x8f, 0x72),
        cutlass_model::TrackType::Adjust => Color32::from_rgb(0x8a, 0x7e, 0x9a),
        _ => Color32::from_rgb(0x5b, 0x8a, 0xa3),
    }
}

fn export_dialog(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    parent: Option<&eframe::Frame>,
) {
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
            if view.export_path.is_empty() {
                view.dialogs.save_export(parent);
            } else {
                match write_export(shell, view) {
                    Ok(()) => view.export_open = false,
                    Err(ShellError::Overwrite(_)) => view.replace_prompt = true,
                    Err(error) => shell.set_notice(error.to_string()),
                }
            }
        }
    });
    if closed {
        view.export_open = false;
        view.replace_prompt = false;
    }
}

fn apply_dialog(shell: &mut Shell, view: &mut ViewState, finished: Finished) {
    let Some(path) = finished.path else {
        return;
    };
    let result = match finished.picker {
        Picker::OpenProject => shell.open_folder(&path),
        Picker::Import => import_chosen(shell, view, &path),
        Picker::SaveExport => {
            view.export_path = path.display().to_string();
            write_export(shell, view)
        }
    };
    match result {
        Ok(()) if finished.picker == Picker::SaveExport => view.export_open = false,
        Ok(()) => {}
        Err(ShellError::Overwrite(_)) if finished.picker == Picker::SaveExport => {
            view.replace_prompt = true;
        }
        Err(error) => shell.set_notice(error.to_string()),
    }
}

fn import_chosen(
    shell: &mut Shell,
    view: &mut ViewState,
    path: &std::path::Path,
) -> Result<(), ShellError> {
    let (videos, audios) = shell.draft().map_or((0, 0), |draft| {
        (draft.materials.videos.len(), draft.materials.audios.len())
    });
    shell.import_file(path)?;
    let added_video = shell
        .draft()
        .is_some_and(|draft| draft.materials.videos.len() > videos);
    let added_audio = shell
        .draft()
        .is_some_and(|draft| draft.materials.audios.len() > audios);
    if added_video {
        view.library_shelf = Shelf::Media;
    } else if added_audio {
        view.library_shelf = Shelf::Audio;
    }
    view.library_query.clear();
    let label = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_owned());
    shell.set_notice(format!("Imported {label}"));
    Ok(())
}

fn write_export(shell: &mut Shell, view: &ViewState) -> Result<(), ShellError> {
    shell.export(ExportRequest {
        path: std::path::PathBuf::from(&view.export_path),
        long_side: view.export_side,
        codec: view.export_codec,
        replace: view.replace_prompt,
    })
}

fn shortcuts(shell: &mut Shell, ui: &Ui) {
    if ui.ctx().text_edit_focused() {
        return;
    }
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
        egui::Key::Space => "space",
        egui::Key::ArrowLeft => "left",
        egui::Key::ArrowRight => "right",
        egui::Key::Delete | egui::Key::Backspace => "delete",
        egui::Key::S => "s",
        egui::Key::C => "c",
        egui::Key::X => "x",
        egui::Key::V => "v",
        _ => return None,
    };
    if matches!(name, "space" | "left" | "right" | "delete" | "s")
        && (input.modifiers.command || input.modifiers.alt || input.modifiers.shift)
    {
        return None;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_settings::Settings;
    use egui::Color32;

    #[test]
    fn the_first_frame_lists_projects_and_does_not_ask_to_close() {
        let root = std::env::temp_dir().join(format!("cutlass-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        view.projects_root = Some(root.clone());
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        assert!(shell.chrome.project_bar);
        assert!(shell.chrome.projects);
        assert!(!shell.chrome.library);
        assert!(!shell.chrome.timeline);
        assert!(!shell.chrome.export);
        assert!(!shell.chrome.close_requested);
        let painted = painted_text(&output);
        assert!(painted.contains("New project"), "{painted}");
        assert!(painted.contains("No projects yet."), "{painted}");
        assert!(painted.lines().any(|line| line == "Projects"), "{painted}");
        assert!(!ctx.global_style().interaction.selectable_labels);
        output.textures_delta.clear();
        Shell::create_project(&root).unwrap();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let painted = painted_text(&output);
        output.textures_delta.clear();
        assert!(painted.lines().any(|line| line == "Project"), "{painted}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_open_project_paints_the_editor() {
        let temp = std::env::temp_dir().join(format!("cutlass-editor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        assert!(shell.chrome.project_bar);
        assert!(!shell.chrome.projects);
        assert!(shell.chrome.library);
        assert!(shell.chrome.preview);
        assert!(shell.chrome.inspector);
        assert!(shell.chrome.timeline);
        assert!(shell.chrome.transport);
        assert!(shell.chrome.export);
        let painted = painted_text(&output);
        output.textures_delta.clear();
        assert!(painted.contains("Details"), "{painted}");
        assert!(painted.contains("Main"), "{painted}");
        assert!(painted.contains("Untitled"), "{painted}");
        assert!(
            painted.contains("No media yet. Use File → Import."),
            "{painted}"
        );
        assert!(painted.contains("Projects"), "{painted}");
        assert!(painted.contains("00:00:01"), "{painted}");
        assert!(painted.contains("No clips"), "{painted}");
        assert!(!painted.contains("Import a file"), "{painted}");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_notice_is_painted_on_a_bar() {
        let root = std::env::temp_dir().join(format!("cutlass-notice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.set_notice("no project is open");
        let mut view = ViewState::new();
        view.projects_root = Some(root.clone());
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let painted = painted_text(&output);
        assert!(painted.contains("no project is open"), "{painted}");
        output.textures_delta.clear();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_sub_second_tick_names_the_frame() {
        let step = 1_000_000.0 / 30.0;
        let rate = FrameRate::FPS_30;
        let frame = rate.frame_start(1).expect("frame").0 as f64;
        assert_eq!(ruler_label(0.0, step, rate, false), "00:00:00");
        assert_eq!(ruler_label(frame, step, rate, false), "00:00:01");
        assert_eq!(ruler_label(1_000_000.0, step, rate, false), "00:01:00");
        assert_eq!(
            ruler_label(3_600_000_000.0, step, rate, false),
            "01:00:00:00"
        );
        assert_eq!(frame_stride(0.006, rate), Some(1));
        assert_eq!(frame_stride(72.0 / 1_000_000.0, rate), None);
        assert_eq!(ruler_label(1_000_000.0, 1_000_000.0, rate, false), "00:01");
        assert_eq!(
            ruler_label(3_661_000_000.0, 1_000_000.0, rate, false),
            "01:01:01"
        );
        assert_eq!(ruler_label(-1.0, step, rate, false), "");
        assert_eq!(
            ruler_label(0.0, step, FrameRate::FPS_29_97, true),
            "00:00;00"
        );
    }

    #[test]
    fn a_utc_date_is_the_civil_day() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(86_400), "1970-01-02");
        assert_eq!(utc_date(946_684_800), "2000-01-01");
    }

    #[test]
    fn a_video_on_the_main_lane_is_a_media_card() {
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut video = cutlass_model::MaterialVideo::new("mat");
        video.path = "Resources/clip.mp4".into();
        video.duration = Micros(2_500_000);
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "seg",
            "mat",
            cutlass_model::TimeRange::new(0, 2_500_000),
        ));
        let cards = bin_cards(&draft, Shelf::Media, "");
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].name, "clip.mp4");
        assert_eq!(cards[0].meta, "2.5s");
        assert!(cards[0].on_timeline);
        assert!(bin_cards(&draft, Shelf::Audio, "").is_empty());
        assert!(bin_cards(&draft, Shelf::Media, "nope").is_empty());
        let blends = bin_cards(&draft, Shelf::Transitions, "");
        assert_eq!(
            blends
                .iter()
                .map(|card| card.name.as_str())
                .collect::<Vec<_>>(),
            ["Dissolve", "Dip"]
        );
        assert_eq!(blends[0].meta, "0.5s");
        assert!(!blends[0].on_timeline);
        assert_eq!(bin_cards(&draft, Shelf::Transitions, "dip").len(), 1);
    }

    #[test]
    fn a_completed_import_dialog_keeps_the_file_in_the_library() {
        let temp = std::env::temp_dir().join(format!("cutlass-import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let picture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        view.dialogs.finish_for_test(Picker::Import, Some(picture));
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let painted = painted_text(&output);
        output.textures_delta.clear();
        assert!(shell.lanes().iter().all(|lane| lane.clips.is_empty()));
        assert!(temp.join("Resources").join("pattern.png").is_file());
        assert_eq!(shell.draft().unwrap().materials.videos.len(), 1);
        assert!(painted.contains("pattern.png"), "{painted}");
        assert!(painted.contains("Imported pattern.png"), "{painted}");
        assert!(!painted.contains("On timeline"), "{painted}");
        let card = text_pos(&output, "pattern.png").expect("library card");
        assert!((80.0..tokens::LIBRARY_WIDTH).contains(&card.x), "{card:?}");
        assert!((40.0..500.0).contains(&card.y), "{card:?}");
        let notice = text_pos(&output, "Imported pattern.png").expect("notice");
        let tick = text_pos(&output, "00:00:00").expect("frame tick");
        assert!(
            notice.y > tick.y,
            "notice {notice:?} overlaps tick {tick:?}"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn releasing_a_library_card_on_the_lane_places_the_clip() {
        let temp = std::env::temp_dir().join(format!("cutlass-drop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let picture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        view.dialogs.finish_for_test(Picker::Import, Some(picture));
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        assert!(shell.lanes().iter().all(|lane| lane.clips.is_empty()));
        let card = text_pos(&output, "pattern.png").expect("library card");
        let lane = text_pos(&output, "No clips").expect("lane");
        output.textures_delta.clear();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(card), press(card, true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(lane)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(
            shell.lanes().iter().all(|lane| lane.clips.is_empty()),
            "the drag has not been released"
        );
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(lane), press(lane, false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert_eq!(shell.lanes().len(), 1);
        assert_eq!(shell.lanes()[0].clips.len(), 1);
        assert!(shell.lanes()[0].main);
        assert_eq!(shell.draft().unwrap().tracks.len(), 1);
        let source = shell.draft().unwrap().tracks[0].segments[0]
            .source_range
            .clone();
        let target = shell.draft().unwrap().tracks[0].segments[0].target.clone();
        assert_eq!(source.map(|range| range.duration), Some(target.duration));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let painted = painted_text(&output);
        output.textures_delta.clear();
        assert!(painted.contains("On timeline"), "{painted}");
        assert!(!painted.contains("No clips"), "{painted}");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_long_saved_path_stays_on_one_line() {
        let temp = std::env::temp_dir().join(format!("cutlass-path-{}", "p".repeat(80)));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let lines = galleys(&output);
        output.textures_delta.clear();
        let full = temp.display().to_string();
        assert!(lines.iter().any(|line| line.contains('…')), "{lines:?}");
        assert!(lines.iter().all(|line| line != &full), "{lines:?}");
        assert!(lines.iter().all(|line| !line.contains('\n')));
        let _ = std::fs::remove_dir_all(&temp);
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

    fn painted_text(output: &egui::FullOutput) -> String {
        let mut text = String::new();
        for clipped in &output.shapes {
            append_shape(&clipped.shape, &mut text);
        }
        text
    }

    fn galleys(output: &egui::FullOutput) -> Vec<String> {
        let mut lines = Vec::new();
        for clipped in &output.shapes {
            collect_galleys(&clipped.shape, &mut lines);
        }
        lines
    }

    fn collect_galleys(shape: &egui::Shape, lines: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(shape) => {
                let shown = shape
                    .galley
                    .rows
                    .iter()
                    .map(|row| row.row.text())
                    .collect::<Vec<_>>()
                    .join("\n");
                lines.push(shown);
            }
            egui::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_galleys(shape, lines);
                }
            }
            _ => {}
        }
    }

    fn text_pos(output: &egui::FullOutput, wanted: &str) -> Option<Pos2> {
        for clipped in &output.shapes {
            if let Some(pos) = shape_pos(&clipped.shape, wanted) {
                return Some(pos);
            }
        }
        None
    }

    fn shape_pos(shape: &egui::Shape, wanted: &str) -> Option<Pos2> {
        match shape {
            egui::Shape::Text(shape) if shape.galley.text() == wanted => Some(shape.pos),
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| shape_pos(shape, wanted)),
            _ => None,
        }
    }

    fn append_shape(shape: &egui::Shape, text: &mut String) {
        match shape {
            egui::Shape::Text(shape) => {
                text.push_str(shape.galley.text());
                text.push('\n');
            }
            egui::Shape::Vec(shapes) => {
                for shape in shapes {
                    append_shape(shape, text);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn pressing_play_keeps_the_window_awake_and_space_pauses() {
        let temp = std::env::temp_dir().join(format!("cutlass-play-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
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
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let play = Pos2::new(674.0, 348.0);
        let pointer = |pressed: bool| egui::Event::PointerButton {
            pos: play,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::PointerMoved(play),
                    pointer(true),
                    pointer(false),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.playing(), "the play control did not start");
        assert!(ctx.has_requested_repaint());
        let mut last = Some(0.0);
        crate::window::advance_if_playing(&mut shell, &mut last, 0.05);
        assert_eq!(shell.playhead(), Micros(50_000));
        assert!(shell.playing());
        let key = |pressed: bool| egui::Event::Key {
            key: egui::Key::Space,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![key(true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(!shell.playing());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn dragging_the_ruler_moves_the_playhead() {
        let temp = std::env::temp_dir().join(format!("cutlass-scrub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "segment",
            "color",
            cutlass_model::TimeRange::new(0, 1_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let start = text_pos(&output, "00:00:00")
            .or_else(|| text_pos(&output, "00:00"))
            .expect("ruler");
        output.textures_delta.clear();
        let end = start + Vec2::new(120.0, 0.0);
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(start), press(start, true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(end)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(
            shell.playhead().0 > 0,
            "the ruler drag did not seek, playhead {}",
            shell.playhead().0
        );
        assert!(shell.guides().is_empty());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_drag_down_from_the_ruler_leaves_a_guide() {
        let (mut shell, temp) = open_span("guide", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let start = text_pos(&output, "00:00:00")
            .or_else(|| text_pos(&output, "00:00"))
            .expect("ruler");
        output.textures_delta.clear();
        drag(
            &ctx,
            start,
            start + Vec2::new(40.0, 90.0),
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.guides().len(), 1);
        assert!(shell.selected_guide().is_some());
        assert_eq!(shell.playhead(), Micros(0));
        let key = |pressed: bool| egui::Event::Key {
            key: egui::Key::Delete,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![key(true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.guides().is_empty());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_drag_down_from_the_names_leaves_a_horizontal_guide() {
        let (mut shell, temp) = open_span("row", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let names = text_pos(&output, "Main").expect("track names");
        let ruler = text_pos(&output, "00:00:00")
            .or_else(|| text_pos(&output, "00:00"))
            .expect("ruler");
        output.textures_delta.clear();
        let corner = Pos2::new(names.x, ruler.y);
        drag(
            &ctx,
            corner,
            corner + Vec2::new(0.0, 90.0),
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.horizontal_guides().len(), 1);
        assert!(shell.selected_horizontal().is_some());
        assert!(shell.guides().is_empty());
        let key = |pressed: bool| egui::Event::Key {
            key: egui::Key::Delete,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![key(true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.horizontal_guides().is_empty());
        let _ = std::fs::remove_dir_all(&temp);
    }

    fn drag(ctx: &egui::Context, from: Pos2, to: Pos2, shell: &mut Shell, view: &mut ViewState) {
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(from), press(from, true)],
                ..Default::default()
            },
            |ui| show(shell, ui, view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(to)],
                ..Default::default()
            },
            |ui| show(shell, ui, view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(to), press(to, false)],
                ..Default::default()
            },
            |ui| show(shell, ui, view, None),
        );
        output.textures_delta.clear();
    }

    fn open_span(name: &str, duration: i64) -> (Shell, std::path::PathBuf) {
        let temp =
            std::env::temp_dir().join(format!("cutlass-trim-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "segment",
            "color",
            cutlass_model::TimeRange::new(0, duration),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        (shell, temp)
    }

    #[test]
    fn dragging_a_clip_edge_trims_it_and_the_body_moves_it() {
        let (mut shell, temp) = open_span("edge", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let width = (2_000_000_f32 * shell.pixel_size() as f32).max(8.0);
        let edge = Pos2::new(text.x - 8.0 + width - 3.0, text.y);
        drag(
            &ctx,
            edge,
            edge - Vec2::new(120.0, 0.0),
            &mut shell,
            &mut view,
        );
        let segment = &shell.draft().unwrap().tracks[0].segments[0];
        assert_eq!(segment.target.start, Micros(0));
        assert!(
            segment.target.duration.0 < 2_000_000,
            "the edge drag did not trim, duration {}",
            segment.target.duration.0
        );
        assert!(segment.target.duration.0 > 0);

        let (mut shell, temp_move) = open_span("body", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let moved = text + Vec2::new(80.0, 0.0);
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(text), press(text, true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(text), press(text, false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.move_preview().is_none());
        assert_eq!(
            shell.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
        assert!(shell.selected().is_some());
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(text), press(text, true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(moved)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let preview = shell.move_preview().expect("the clip follows the pointer");
        assert!(preview.start.0 > 0, "the preview stayed at the start");
        assert_eq!(preview.duration, Micros(2_000_000));
        assert_eq!(
            shell.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(moved), press(moved, false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.move_preview().is_none());
        let segment = &shell.draft().unwrap().tracks[0].segments[0];
        assert!(
            segment.target.start.0 > 0,
            "the body drag did not move the clip"
        );
        assert_eq!(segment.target.duration.0, 2_000_000);
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::remove_dir_all(&temp_move);
    }

    #[test]
    fn the_blade_cuts_the_clip_under_the_pointer() {
        let (mut shell, temp) = open_span("blade", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        shell.toggle_blade();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let width = (2_000_000_f32 * shell.pixel_size() as f32).max(8.0);
        let center = Pos2::new(text.x - 8.0 + width / 2.0, text.y);
        let press = |pressed: bool| egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(center), press(true), press(false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let segments = &shell.draft().unwrap().tracks[0].segments;
        assert_eq!(
            segments.len(),
            2,
            "the blade click did not cut the clip ({})",
            shell.notice()
        );
        assert!(segments[0].target.duration.0 > 0);
        assert!(segments[1].target.duration.0 > 0);
        assert_eq!(
            segments[0].target.duration.0 + segments[1].target.duration.0,
            2_000_000
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_marker_drops_on_the_clip_under_the_pointer() {
        let (mut shell, temp) = open_span("marker", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        shell.toggle_marker();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let width = (2_000_000_f32 * shell.pixel_size() as f32).max(8.0);
        let center = Pos2::new(text.x - 8.0 + width / 2.0, text.y);
        let press = |pressed: bool| egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(center), press(true), press(false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert_eq!(
            shell.draft().unwrap().tracks[0].segments.len(),
            1,
            "the marker click changed the clip ({})",
            shell.notice()
        );
        let marks = shell.markers();
        assert_eq!(marks.len(), 1, "notice {}", shell.notice());
        assert_eq!(
            marks[0].segment_id.as_ref().map(|id| id.as_str()),
            Some(id.as_str())
        );
        assert!(marks[0].stored_start > 0);
        assert!(marks[0].stored_start < 2_000_000);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_beat_drops_on_the_audio_clip_under_the_pointer() {
        let temp = std::env::temp_dir().join(format!("cutlass-trim-{}-beat", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "picture",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        let mut sound = cutlass_model::Track::main("sound");
        sound.kind = cutlass_model::TrackType::Audio;
        sound.segments.push(cutlass_model::Segment::new(
            "tone",
            "tone",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        draft.tracks.push(sound);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        shell.toggle_beat();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let text = text_pos(&output, "tone").expect("audio clip");
        output.textures_delta.clear();
        let width = (2_000_000_f32 * shell.pixel_size() as f32).max(8.0);
        let center = Pos2::new(text.x - 8.0 + width / 2.0, text.y);
        let press = |pressed: bool| egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(center), press(true), press(false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let beats = shell.beats();
        assert_eq!(beats.len(), 1, "notice {}", shell.notice());
        assert_eq!(beats[0].segment_id.as_str(), "tone");
        assert!(beats[0].source > 0);
        assert!(beats[0].source < 2_000_000);
        assert_eq!(beats[0].time.0, beats[0].source);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_click_on_empty_lane_clears_the_selection_and_the_wheel_scrolls() {
        let (mut shell, temp) = open_span("scroll", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(text), press(text, true)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(text), press(text, false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.selected().is_some());
        let width = (2_000_000_f32 * shell.pixel_size() as f32).max(8.0);
        let empty = Pos2::new(text.x - 8.0 + width + 24.0, text.y);
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::PointerMoved(empty),
                    press(empty, true),
                    press(empty, false),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.selected().is_none());
        shell.zoom_in().unwrap();
        let before = shell.view_origin();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::PointerMoved(text),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: Vec2::new(0.0, -120.0),
                        phase: egui::TouchPhase::Move,
                        source: egui::MouseWheelSource::Wheel,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(
            shell.view_origin().0 > before.0,
            "the wheel did not scroll, origin {}",
            shell.view_origin().0
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_drag_across_empty_space_selects_the_clips_it_touches() {
        let temp = std::env::temp_dir().join(format!("cutlass-trim-{}-box", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "one",
            "color",
            cutlass_model::TimeRange::new(0, 500_000),
        ));
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "two",
            "color",
            cutlass_model::TimeRange::new(1_000_000, 500_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let names: Vec<_> = shell.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.as_str().to_owned())
            .collect();
        let one = text_pos(&output, &names[0]).expect("first clip");
        let two = text_pos(&output, &names[1]).expect("second clip");
        output.textures_delta.clear();
        let two_width = (500_000_f32 * shell.pixel_size() as f32).max(8.0);
        let start = Pos2::new(two.x - 8.0 + two_width + 36.0, two.y);
        drag(&ctx, start, one, &mut shell, &mut view);
        assert_eq!(shell.selection().len(), 2, "notice {}", shell.notice());
        let modifiers = egui::Modifiers {
            shift: true,
            ..Default::default()
        };
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers,
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::ModifiersChanged(modifiers),
                    egui::Event::PointerMoved(one),
                    press(one, true),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::ModifiersChanged(modifiers),
                    egui::Event::PointerMoved(one),
                    press(one, false),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert_eq!(shell.selection().len(), 1);
        assert!(!shell.is_selected(&SegmentId::new(&names[0])));
        let ids: Vec<_> = shell.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect();
        shell.select_box(ids, false);
        let click = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::ModifiersChanged(egui::Modifiers::default()),
                    egui::Event::PointerMoved(one),
                    click(one, true),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert_eq!(shell.selection().len(), 2);
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::ModifiersChanged(egui::Modifiers::default()),
                    egui::Event::PointerMoved(one),
                    click(one, false),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert_eq!(shell.selection().len(), 1);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_right_click_copies_the_clip() {
        let (mut shell, temp) = open_span("menu", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let click =
            |pos: Pos2, button: egui::PointerButton, pressed: bool| egui::Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::PointerMoved(text),
                    click(text, egui::PointerButton::Secondary, true),
                    click(text, egui::PointerButton::Secondary, false),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.clip_menu().is_some());
        let copy = Pos2::new(text.x + 24.0, text.y + 10.0);
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![
                    egui::Event::PointerMoved(copy),
                    click(copy, egui::PointerButton::Primary, true),
                    click(copy, egui::PointerButton::Primary, false),
                ],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(shell.has_clipboard(), "notice {}", shell.notice());
        assert!(shell.clip_menu().is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_pinch_over_the_lanes_zooms_them() {
        let (mut shell, temp) = open_span("pinch", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let id = shell.draft().unwrap().tracks[0].segments[0]
            .id
            .as_str()
            .to_owned();
        let text = text_pos(&output, &id).expect("clip");
        output.textures_delta.clear();
        let before = shell.pixel_size();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(text), egui::Event::Zoom(2.0)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(
            shell.pixel_size() > before,
            "notice {} size {}",
            shell.notice(),
            shell.pixel_size()
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
