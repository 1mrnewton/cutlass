//! The column of panels. egui draws the chrome. The preview is a texture of
//! the evaluated frame. This pass does not composite.
//!
//! With no project open, the window lists the saved projects and can create one.

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Pos2, Rect, RichText, ScrollArea, Sense,
    Ui, Vec2,
};

use cutlass_model::{FrameRate, Micros};

use crate::dialogs::{Dialogs, Finished, Picker};
use crate::icons;
use crate::shell::{ExportRequest, ListedProject, Shell, ShellError};
use crate::tokens;
use crate::widgets;

const CARD_W: f32 = 144.0;
const CARD_H: f32 = 96.0;
const CARD_GAP: f32 = 8.0;

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
        timeline_panel(shell, ui, timeline);
        paint_library_ghost(shell, ui);
        if view.export_open {
            export_dialog(shell, ui, view, parent);
        }
        shortcuts(shell, ui);
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
        Shelf::Text
        | Shelf::Stickers
        | Shelf::Effects
        | Shelf::Transitions
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

fn timeline_panel(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
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
                            if toolbar_button(ui, name, tip).clicked() {
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

fn toolbar_button(ui: &mut Ui, name: &str, tip: &str) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(28.0, 28.0), Sense::click());
    let color = if tool_enabled(name) {
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

fn tool_enabled(name: &str) -> bool {
    matches!(
        name,
        "undo" | "redo" | "select" | "split" | "delete" | "zoomIn" | "zoomOut" | "zoomFit"
    )
}

fn tool_clicked(shell: &mut Shell, name: &str) {
    if !tool_enabled(name) || name == "select" {
        return;
    }
    let result = match name {
        "undo" => shell.undo(),
        "redo" => shell.redo(),
        "zoomIn" => shell.zoom_in(),
        "zoomOut" => shell.zoom_out(),
        "zoomFit" => shell.zoom_fit(),
        "delete" => shell.remove_selected(),
        "split" => shell.split_selected(),
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

fn ruler_and_lanes(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    let ruler = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler.left_bottom(), rect.max);
    let lanes_left = tracks.left() + tokens::TRACK_HEAD + tokens::COVER_WIDTH;
    let lane_area = Rect::from_min_max(Pos2::new(lanes_left, tracks.top()), tracks.max);
    let (rate, drop_frame) = ruler_rate(shell);
    paint_ticks(ui, ruler, lanes_left, shell.pixel_size(), rate, drop_frame);
    let rows = shell.lanes();
    let flags: Vec<bool> = rows.iter().map(|lane| lane.main).collect();
    let boxes = lane_rects(flags, lane_area);
    if rows.is_empty() {
        paint_empty_main(ui, lane_area);
    }
    for (index, lane) in rows.iter().enumerate() {
        let Some(row) = boxes.get(index).copied() else {
            continue;
        };
        paint_lane(shell, ui, lane, row, lanes_left, index);
    }
    let playhead_x = lanes_left + shell.playhead().0 as f32 * shell.pixel_size() as f32;
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
    let ruler_hit = ui.interact(
        lane_ruler(ruler, lanes_left),
        ui.scope_id().with("ruler"),
        Sense::click(),
    );
    if ruler_hit.clicked()
        && let Some(pointer) = ruler_hit.interact_pointer_pos()
    {
        let x = f64::from(pointer.x - lanes_left).max(0.0);
        if let Err(error) = shell.seek_ruler(x) {
            shell.set_notice(error.to_string());
        }
    }
    release_library_drop(shell, ui, tracks, lanes_left, &boxes);
    shell.set_lane_width(lane_area.width().max(1.0));
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
    if let Err(error) = shell.drop_medium(&material_id, x, y) {
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

fn lane_ruler(ruler: Rect, lanes_left: f32) -> Rect {
    Rect::from_min_max(Pos2::new(lanes_left, ruler.top()), ruler.max)
}

fn paint_ticks(
    ui: &Ui,
    ruler: Rect,
    lanes_left: f32,
    pixel_size: f64,
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
        width: f64::from(ruler.right() - lanes_left),
    };
    if let Some(stride) = frame_stride(pixel_size, rate) {
        paint_frame_ticks(&grid, rate, drop_frame, stride);
        return;
    }
    let step = second_step(pixel_size);
    let mut time = 0.0;
    while time * pixel_size <= grid.width {
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
    let mut index = 0_i64;
    for _ in 0..10_000 {
        let Some(start) = rate.frame_start(index) else {
            return;
        };
        let time = start.0 as f64;
        if time * grid.pixel_size > grid.width {
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
    let x = grid.lanes_left + (time * grid.pixel_size) as f32;
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
    let x = grid.lanes_left + (major * grid.pixel_size) as f32;
    let minor_x = grid.lanes_left + (minor * grid.pixel_size) as f32;
    if minor * grid.pixel_size <= grid.width && minor_x - x >= 8.0 {
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
    for clip in &lane.clips {
        paint_clip(shell, ui, lane, clip, row, lanes_left, index);
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

fn paint_clip(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    clip: &crate::shell::ClipSpan,
    row: Rect,
    lanes_left: f32,
    index: usize,
) {
    let (inset, height) = if lane.main { (8.0, 48.0) } else { (6.0, 28.0) };
    let x = lanes_left + clip.start.0 as f32 * shell.pixel_size() as f32;
    let width = (clip.duration.0 as f32 * shell.pixel_size() as f32).max(8.0);
    let clip_rect = Rect::from_min_size(Pos2::new(x, row.top() + inset), Vec2::new(width, height));
    let selected = shell.selected().is_some_and(|id| id == &clip.id);
    ui.painter()
        .rect_filled(clip_rect, CornerRadius::same(7), lane_color(&lane.kind));
    if selected {
        ui.painter().rect_stroke(
            clip_rect,
            CornerRadius::same(7),
            egui::Stroke::new(1.5, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    let painter = ui.painter().with_clip_rect(clip_rect.shrink(6.0));
    painter.text(
        clip_rect.left_center() + Vec2::new(8.0, 0.0),
        Align2::LEFT_CENTER,
        &clip.name,
        FontId::proportional(11.0),
        Color32::WHITE,
    );
    let response = ui.interact(
        clip_rect,
        ui.scope_id().with(("clip", index, clip.id.as_str())),
        Sense::click_and_drag(),
    );
    let span = format!(
        "{} · {}–{}",
        clip.name,
        shell.timecode_at(clip.start),
        shell.timecode_at(Micros(clip.start.0.saturating_add(clip.duration.0)))
    );
    response.clone().on_hover_text(span);
    if response.clicked() {
        shell.select(Some(clip.id.clone()));
    }
    if response.drag_stopped()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let x = f64::from(pos.x - lanes_left).max(0.0);
        let y = f64::from(u32::try_from(index).unwrap_or(0)) * tokens::LANE_HEIGHT
            + tokens::LANE_HEIGHT / 2.0;
        if let Err(error) = shell.drag_segment(clip.id.clone(), x, y) {
            shell.set_notice(error.to_string());
        }
    }
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
