//! The column of panels. egui draws the chrome. The preview is a texture of
//! the evaluated frame. This pass does not composite.
//!
//! With no project open, the window lists the saved projects and can create one.

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Pos2, Rect, RichText, ScrollArea, Sense,
    Ui, Vec2,
};

use cutlass_editor::ruler_ticks;
use cutlass_model::{FrameRate, Micros, SegmentId};
use cutlass_session::TrimEdge;

use crate::dialogs::{Dialogs, Finished, Picker};
use crate::icons;
use crate::shell::{ExportRequest, ListedProject, MovePreview, Shell, ShellError};
use crate::tokens;
use crate::widgets;

const CARD_W: f32 = 144.0;
const CARD_H: f32 = 96.0;
/// The name row under a card's picture.
const CARD_PLUS_RADIUS: f32 = 11.0;
const CARD_PLUS_INSET: f32 = 6.0;
const CARD_LABEL: f32 = 24.0;
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

/// The open library menu: where it opens, and whether it is for the
/// picked cards or for the space between them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LibraryMenu {
    at: Pos2,
    on_card: bool,
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
    text_size_buffer: String,
    text_color_buffer: String,
    letter_buffer: String,
    line_buffer: String,
    width_buffer: String,
    x_buffer: String,
    y_buffer: String,
    scale_buffer: String,
    opacity_buffer: String,
    letterbox_buffer: String,
    volume_buffer: String,
    fade_in_buffer: String,
    fade_out_buffer: String,
    speed_buffer: String,
    background_buffer: String,
    name_buffer: String,
    dialogs: Dialogs,
    library_query: String,
    library_shelf: Shelf,
    /// The library cards picked, by card id, and the card a Shift-click
    /// range starts from.
    library_picked: Vec<String>,
    library_anchor: Option<String>,
    library_menu: Option<LibraryMenu>,
    projects_root: Option<std::path::PathBuf>,
    listed: Vec<ListedProject>,
    listed_stamp: String,
    /// When the projects folder was last looked at.
    listed_checked: Option<std::time::Instant>,
    marquee_press: Option<Pos2>,
    marquee: Option<Rect>,
    /// How far the lanes are moved down from where they center on the main
    /// lane, in points; negative shows lanes further down.
    lane_scroll: f32,
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
            text_size_buffer: String::new(),
            text_color_buffer: String::new(),
            letter_buffer: String::new(),
            line_buffer: String::new(),
            width_buffer: String::new(),
            x_buffer: String::new(),
            y_buffer: String::new(),
            scale_buffer: String::new(),
            opacity_buffer: String::new(),
            letterbox_buffer: String::new(),
            volume_buffer: String::new(),
            fade_in_buffer: String::new(),
            fade_out_buffer: String::new(),
            speed_buffer: String::new(),
            background_buffer: String::new(),
            name_buffer: String::new(),
            dialogs: Dialogs::new(),
            library_query: String::new(),
            library_shelf: Shelf::Media,
            library_picked: Vec::new(),
            library_anchor: None,
            library_menu: None,
            projects_root: None,
            listed: Vec::new(),
            listed_stamp: String::new(),
            listed_checked: None,
            marquee_press: None,
            marquee: None,
            lane_scroll: 0.0,
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
    if shell.poll_job() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));
    }
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
        job_strip(shell, ui, full);
        shortcuts(shell, ui);
        if shell.playing() || shell.preview_pending() {
            ui.ctx().request_repaint();
        }
    }
    if let Some(finished) = view.dialogs.poll(ui.ctx()) {
        apply_dialog(shell, view, finished);
    }
}

/// The running job above the foot of the window: what it is, where it has
/// got to, and Cancel.
fn job_strip(shell: &mut Shell, ui: &Ui, full: Rect) {
    const WIDTH: f32 = 340.0;
    let Some(job) = shell.job() else {
        return;
    };
    let top_left = Pos2::new(full.center().x - WIDTH / 2.0, full.bottom() - 72.0);
    egui::Area::new(ui.scope_id().with("job"))
        .order(egui::Order::Foreground)
        .fixed_pos(top_left)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(WIDTH);
                ui.horizontal(|ui| {
                    let line = if job.stage.is_empty() {
                        job.label.clone()
                    } else {
                        format!("{}: {}", job.label, job.stage)
                    };
                    ui.label(line);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Cancel").clicked() {
                            shell.cancel_job();
                        }
                    });
                });
                let bar = egui::ProgressBar::new(job.fraction.unwrap_or(0.0));
                ui.add(match job.fraction {
                    Some(_) => bar.show_percentage(),
                    None => bar.animate(true),
                });
            });
        });
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
    paint_notice(shell, ui, bar.center().x, bar.bottom() + 12.0);
}

/// How often the projects home looks at the folder for changes.
const PROJECTS_CHECK: std::time::Duration = std::time::Duration::from_secs(1);

fn listed_projects(view: &mut ViewState) -> Vec<ListedProject> {
    let due = view
        .listed_checked
        .is_none_or(|checked| checked.elapsed() >= PROJECTS_CHECK);
    if due || view.listed_stamp.is_empty() {
        view.listed_checked = Some(std::time::Instant::now());
        let root = projects_root(view);
        let stamp = format!("{}|{}", root.display(), directory_stamp(&root));
        if view.listed_stamp != stamp {
            view.listed = Shell::list_projects(&root);
            view.listed_stamp = stamp;
        }
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
    let (undo, redo) = (shell.can_undo(), shell.can_redo());
    if ui.add_enabled(undo, egui::Button::new("Undo")).clicked() {
        ui.close();
        if let Err(error) = shell.undo() {
            shell.set_notice(error.to_string());
        }
    }
    if ui.add_enabled(redo, egui::Button::new("Redo")).clicked() {
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

fn library_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
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

fn asset_column(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
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
    let shelf = view.library_shelf;
    let bin = matches!(shelf, Shelf::Media | Shelf::Audio);
    // The space between and below the cards. Cards sit above it, so a
    // click here missed every card.
    let back = column.interact(grid, column.scope_id().with("library-back"), Sense::click());
    if back.clicked() {
        view.library_picked.clear();
        view.library_anchor = None;
    }
    if bin
        && back.secondary_clicked()
        && let Some(at) = back.interact_pointer_pos()
    {
        view.library_menu = Some(LibraryMenu { at, on_card: false });
    }
    let mut grid_ui = column.new_child(egui::UiBuilder::new().max_rect(grid));
    let query = view.library_query.clone();
    ScrollArea::vertical()
        .id_salt("library-cards")
        .auto_shrink([false, false])
        .show(&mut grid_ui, |ui| {
            let mut cards = shell
                .draft()
                .map(|draft| bin_cards(draft, shelf, &query))
                .unwrap_or_default();
            fill_looks(shell, ui.ctx(), &mut cards);
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
            } else if let Some(id) = card_grid(ui, view, &cards, bin)
                && let Err(error) = shell.add_medium(&id)
            {
                shell.set_notice(error.to_string());
            }
        });
    library_menu(shell, &column, view);
}

/// The library menu, while open. No row is built yet, so a pick says so.
fn library_menu(shell: &mut Shell, ui: &Ui, view: &mut ViewState) {
    let Some(menu) = view.library_menu else {
        return;
    };
    if let Some(pos) = secondary_click(ui)
        && pos != menu.at
    {
        view.library_menu = None;
        return;
    }
    let entries = if menu.on_card {
        crate::menu::LIBRARY_CARD
    } else {
        crate::menu::LIBRARY_EMPTY
    };
    let shown = crate::menu::show(
        ui.ctx(),
        ui.scope_id().with("library-menu"),
        menu.at,
        entries,
        &|_| true,
    );
    if let Some(item) = shown.picked {
        view.library_menu = None;
        shell.set_notice(format!("{} is not built yet", item.label));
    } else if shown.dismissed {
        view.library_menu = None;
    }
}

/// Pick `id` the way a click does: alone, toggled with Command, or the run
/// from the anchor with Shift.
fn pick_card(view: &mut ViewState, cards: &[BinCard], id: &str, modifiers: egui::Modifiers) {
    let toggle = modifiers.command || modifiers.ctrl;
    let index = |wanted: &str| cards.iter().position(|card| card.id == wanted);
    if modifiers.shift
        && let Some(from) = view.library_anchor.as_deref().and_then(index)
        && let Some(to) = index(id)
    {
        let run = cards[from.min(to)..=from.max(to)]
            .iter()
            .map(|card| card.id.clone());
        if toggle {
            for card in run {
                if !view.library_picked.contains(&card) {
                    view.library_picked.push(card);
                }
            }
        } else {
            view.library_picked = run.collect();
        }
        return;
    }
    if toggle {
        if let Some(at) = view.library_picked.iter().position(|card| card == id) {
            view.library_picked.remove(at);
        } else {
            view.library_picked.push(id.to_owned());
        }
    } else {
        view.library_picked = vec![id.to_owned()];
    }
    view.library_anchor = Some(id.to_owned());
}

/// Lay out the cards. Returns the card whose plus was pressed.
fn card_grid(ui: &mut Ui, view: &mut ViewState, cards: &[BinCard], bin: bool) -> Option<String> {
    let mut added = None;
    let width = ui.available_width().max(CARD_W);
    ui.set_min_width(width);
    let cols = ((width + CARD_GAP) / (CARD_W + CARD_GAP)).floor().max(1.0) as usize;
    let cols = cols.max(1);
    let rows = cards.len().div_ceil(cols);
    let pitch = CARD_H + CARD_LABEL + CARD_GAP;
    let height = rows as f32 * pitch;
    let (area, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    for (index, card) in cards.iter().enumerate() {
        let origin = area.min
            + Vec2::new(
                (index % cols) as f32 * (CARD_W + CARD_GAP),
                (index / cols) as f32 * pitch,
            );
        let rect = Rect::from_min_size(origin, Vec2::new(CARD_W, CARD_H + CARD_LABEL));
        paint_bin_card(ui, rect, card, view.library_picked.contains(&card.id));
        // A click picks the card. A drag only starts once the pointer moves.
        let response = ui.interact(rect, ui.scope_id().with(&card.id), Sense::click_and_drag());
        response.dnd_set_drag_payload(LibraryDrop {
            material_id: card.id.clone(),
        });
        if response.clicked() {
            let modifiers = ui.input(|input| input.modifiers);
            pick_card(view, cards, &card.id, modifiers);
        }
        if response.secondary_clicked() {
            if !view.library_picked.contains(&card.id) {
                view.library_picked = vec![card.id.clone()];
                view.library_anchor = Some(card.id.clone());
            }
            if bin && let Some(at) = response.interact_pointer_pos() {
                view.library_menu = Some(LibraryMenu { at, on_card: true });
            }
        }
        response.on_hover_text(&card.name);
        if bin && ui.rect_contains_pointer(rect) && card_plus(ui, rect, &card.id) {
            added = Some(card.id.clone());
        }
    }
    added
}

/// The round plus at the bottom right of a hovered card's picture. True when pressed.
fn card_plus(ui: &mut Ui, card: Rect, id: &str) -> bool {
    let center = Pos2::new(
        card.right() - CARD_PLUS_INSET - CARD_PLUS_RADIUS,
        card.top() + CARD_H - CARD_PLUS_INSET - CARD_PLUS_RADIUS,
    );
    let rect = Rect::from_center_size(center, Vec2::splat(CARD_PLUS_RADIUS * 2.0));
    let response = ui
        .interact(rect, ui.scope_id().with(("card-plus", id)), Sense::click())
        .on_hover_text("Add to timeline");
    let (fill, ink) = if response.hovered() {
        (Color32::WHITE, tokens::PANEL)
    } else {
        (Color32::from_black_alpha(184), Color32::WHITE)
    };
    let painter = ui.painter();
    painter.circle_filled(center, CARD_PLUS_RADIUS, fill);
    let arm = 4.5;
    let stroke = egui::Stroke::new(1.5, ink);
    painter.line_segment([center - Vec2::X * arm, center + Vec2::X * arm], stroke);
    painter.line_segment([center - Vec2::Y * arm, center + Vec2::Y * arm], stroke);
    response.clicked()
}

/// A card: its picture, then its name under it.
fn paint_bin_card(ui: &Ui, rect: Rect, card: &BinCard, picked: bool) {
    let picture = Rect::from_min_size(rect.min, Vec2::new(rect.width(), CARD_H));
    let painter = ui.painter().with_clip_rect(rect);
    paint_card_face(&painter, picture, card);
    if picked {
        painter.rect_stroke(
            picture,
            CornerRadius::same(CLIP_RADIUS),
            egui::Stroke::new(tokens::STROKE, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    let font = FontId::proportional(12.0);
    let name = middle_ellipsis(ui.ctx(), &card.name, &font, rect.width() - 4.0);
    painter.text(
        Pos2::new(rect.left() + 2.0, picture.bottom() + 6.0),
        Align2::LEFT_TOP,
        name,
        font,
        tokens::MUTED,
    );
}

/// `text` cut in the middle with an ellipsis until it fits `width`, the way
/// a file name keeps its start and its extension.
fn middle_ellipsis(ctx: &egui::Context, text: &str, font: &FontId, width: f32) -> String {
    let fits = |text: &str| {
        ctx.fonts_mut(|fonts| {
            fonts
                .layout_no_wrap(text.to_owned(), font.clone(), tokens::TEXT)
                .size()
                .x
        }) <= width
    };
    if fits(text) {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    let cut = |keep: usize| -> String {
        let head = keep.div_ceil(2);
        let tail = keep / 2;
        chars[..head]
            .iter()
            .chain(std::iter::once(&'…'))
            .chain(&chars[chars.len() - tail..])
            .collect()
    };
    // The most letters that fit, found by halving rather than one letter
    // at a time: a cut only grows wider as it keeps more.
    let (mut fitting, mut over) = (0, chars.len());
    while over - fitting > 1 {
        let keep = fitting + (over - fitting) / 2;
        if fits(&cut(keep)) {
            fitting = keep;
        } else {
            over = keep;
        }
    }
    if fitting == 0 {
        return "…".to_owned();
    }
    cut(fitting)
}

/// Read each card's picture or sound from the shell, asking for what is
/// missing.
fn fill_looks(shell: &mut Shell, ctx: &egui::Context, cards: &mut [BinCard]) {
    with_film_textures(ctx, |textures| {
        for card in cards.iter_mut() {
            match card.look {
                CardLook::Picture(_) => {
                    if let Some((key, Some(thumb))) = shell.poster(&card.id) {
                        let texture = film_texture(ctx, textures, key, &thumb);
                        card.look = CardLook::Picture(Some((texture, [thumb.width, thumb.height])));
                    }
                }
                CardLook::Sound(_) => {
                    if let Some(levels) = shell.material_wave(&card.id, CARD_W as usize) {
                        card.look = CardLook::Sound(levels);
                    }
                }
                CardLook::Swatch => {}
            }
        }
    });
}

/// The card's picture: the file's first frame cut to fill, its sound, or a
/// color, with "Added" at the top left once on the timeline and the length
/// at the top right.
fn paint_card_face(painter: &egui::Painter, rect: Rect, card: &BinCard) {
    let radius = CornerRadius::same(CLIP_RADIUS);
    match &card.look {
        CardLook::Swatch => {
            painter.rect_filled(rect, radius, swatch_color(&card.name));
        }
        CardLook::Picture(None) => {
            painter.rect_filled(rect, radius, tokens::ICON_BUTTON);
        }
        CardLook::Picture(Some((texture, [width, height]))) => {
            let uv = cover_uv(rect.size(), *width as f32, *height as f32);
            painter.add(
                egui::epaint::RectShape::filled(rect, radius, Color32::WHITE)
                    .with_texture(*texture, uv),
            );
        }
        CardLook::Sound(levels) => {
            painter.rect_filled(rect, radius, tokens::CLIP_AUDIO);
            let wave = crate::shell::ClipWave {
                first: 0,
                levels: levels.to_vec(),
                gains: Vec::new(),
                under_picture: false,
                muted: false,
            };
            paint_wave(painter, &wave, rect.left(), rect, rect, WaveColors::AUDIO);
        }
    }
    let small = FontId::proportional(11.0);
    if card.on_timeline {
        let galley = painter.layout_no_wrap("Added".to_owned(), small.clone(), Color32::WHITE);
        let chip = Rect::from_min_size(
            rect.left_top() + Vec2::new(6.0, 6.0),
            galley.size() + Vec2::new(10.0, 4.0),
        );
        painter.rect_filled(chip, CornerRadius::same(4), Color32::from_black_alpha(170));
        painter.galley(chip.min + Vec2::new(5.0, 2.0), galley, Color32::WHITE);
    }
    if !card.meta.is_empty() {
        let corner = rect.right_top() + Vec2::new(-8.0, 8.0);
        painter.text(
            corner + Vec2::new(0.0, 1.0),
            Align2::RIGHT_TOP,
            &card.meta,
            small.clone(),
            Color32::from_black_alpha(160),
        );
        painter.text(corner, Align2::RIGHT_TOP, &card.meta, small, Color32::WHITE);
    }
}

/// The part of a `width` by `height` picture that fills `size` without
/// stretching: the middle, cut at the sides or the top and bottom.
fn cover_uv(size: Vec2, width: f32, height: f32) -> Rect {
    if width <= 0.0 || height <= 0.0 || size.x <= 0.0 || size.y <= 0.0 {
        return Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    }
    let shown = (size.x / size.y) / (width / height);
    if shown < 1.0 {
        let cut = (1.0 - shown) / 2.0;
        Rect::from_min_max(Pos2::new(cut, 0.0), Pos2::new(1.0 - cut, 1.0))
    } else {
        let cut = (1.0 - 1.0 / shown) / 2.0;
        Rect::from_min_max(Pos2::new(0.0, cut), Pos2::new(1.0, 1.0 - cut))
    }
}

struct BinCard {
    id: String,
    name: String,
    meta: String,
    on_timeline: bool,
    look: CardLook,
}

/// What fills a library card's picture.
#[derive(Clone, Debug, PartialEq)]
enum CardLook {
    /// A color from the name, for a card with no file.
    Swatch,
    /// A file's first frame, and its size in pixels, once read.
    Picture(Option<(egui::TextureId, [u32; 2])>),
    /// A sound file's levels, one per point across the card, once read.
    Sound(std::sync::Arc<[f32]>),
}

#[derive(Clone, Debug)]
struct LibraryDrop {
    material_id: String,
}

fn bin_cards(draft: &cutlass_model::Draft, shelf: Shelf, query: &str) -> Vec<BinCard> {
    // A clip that changed its own copy of a video still shows that file,
    // so the library lists each file once and marks it by file.
    let paths: std::collections::HashMap<&str, &str> = draft
        .materials
        .videos
        .iter()
        .map(|video| (video.id.as_str(), video.path.as_str()))
        .chain(
            draft
                .materials
                .audios
                .iter()
                .map(|audio| (audio.id.as_str(), audio.path.as_str())),
        )
        .collect();
    let used: std::collections::HashSet<&str> = draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .map(|segment| segment.material_id.as_str())
        .map(|id| {
            paths
                .get(id)
                .copied()
                .filter(|path| !path.is_empty())
                .unwrap_or(id)
        })
        .collect();
    let mut seen = std::collections::HashSet::new();
    let mut fresh = |path: &str| path.is_empty() || seen.insert(path.to_owned());
    let query = query.trim().to_lowercase();
    let mut cards = Vec::new();
    match shelf {
        Shelf::Media => {
            for video in &draft.materials.videos {
                if !fresh(&video.path) {
                    continue;
                }
                push_card(
                    &mut cards,
                    &used,
                    &query,
                    video.id.as_str(),
                    &video.path,
                    video.duration.0,
                    CardLook::Picture(None),
                );
            }
        }
        Shelf::Audio => {
            for audio in &draft.materials.audios {
                if !fresh(&audio.path) {
                    continue;
                }
                push_card(
                    &mut cards,
                    &used,
                    &query,
                    audio.id.as_str(),
                    &audio.path,
                    audio.duration.0,
                    CardLook::Sound(std::sync::Arc::from([])),
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
                    look: CardLook::Swatch,
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
    look: CardLook,
) {
    let name = file_label(path);
    if !query.is_empty() && !name.to_lowercase().contains(query) {
        return;
    }
    cards.push(BinCard {
        id: id.to_owned(),
        name,
        meta: card_clock(duration),
        on_timeline: used.contains(id) || (!path.is_empty() && used.contains(path)),
        look,
    });
}

/// A media card's length: minutes and seconds, with hours when it has them.
fn card_clock(micros: i64) -> String {
    let seconds = micros.max(0) / 1_000_000;
    let (hours, minutes, seconds) = (seconds / 3_600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
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
    let shown = match shell.preview_texture() {
        Ok(Some((id, [width, height]))) => {
            let fitted = fit_texture(frame, Vec2::new(width as f32, height as f32));
            paint_canvas(ui, id, fitted, shell.preview_black());
            Ok(())
        }
        Ok(None) => Err(None),
        Err(error) => Err(Some(error)),
    };
    let fallback = match shown {
        Ok(()) => None,
        Err(Some(error)) => Some(Err(error)),
        Err(None) => Some(shell.preview_bytes()),
    };
    match fallback {
        None => {}
        Some(Ok(picture)) => {
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
                paint_canvas(ui, texture.id(), fitted, shell.preview_black());
            }
        }
        Some(Err(error)) if shell.draft().is_some() => {
            ui.painter().text(
                frame.center(),
                Align2::CENTER_CENTER,
                error.to_string(),
                FontId::proportional(tokens::TYPE_SIZE),
                tokens::MUTED,
            );
        }
        Some(Err(_)) => {}
    }
    transport_bar(shell, ui, transport);
}

/// The preview canvas at `rect`, over black when `black` is on. Only the
/// canvas gets that background; the area around it stays unfilled.
fn paint_canvas(ui: &Ui, texture: egui::TextureId, rect: Rect, black: bool) {
    if black {
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);
    }
    ui.painter().image(
        texture,
        rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
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
    background_toggle(shell, ui, still.translate(Vec2::new(-40.0, 0.0)));
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

/// The button that shows the preview canvas over black or clear. Its mark
/// is a filled square for black and a two-by-two checker for clear.
fn background_toggle(shell: &mut Shell, ui: &Ui, button: Rect) {
    let response = ui.interact(button, ui.scope_id().with("background"), Sense::click());
    if response.clicked() {
        shell.toggle_preview_background();
    }
    ui.painter()
        .circle_filled(button.center(), 16.0, tokens::ICON_BUTTON);
    let mark = Rect::from_center_size(button.center(), Vec2::splat(12.0));
    if shell.preview_black() {
        ui.painter()
            .rect_filled(mark, CornerRadius::same(2), Color32::BLACK);
        ui.painter().rect_stroke(
            mark,
            CornerRadius::same(2),
            egui::Stroke::new(1.0, tokens::MUTED),
            egui::StrokeKind::Inside,
        );
    } else {
        let half = mark.width() / 2.0;
        for (column, row) in [(0.0, 0.0), (1.0, 1.0)] {
            let cell = Rect::from_min_size(
                mark.min + Vec2::new(column * half, row * half),
                Vec2::splat(half),
            );
            ui.painter()
                .rect_filled(cell, CornerRadius::ZERO, tokens::MUTED);
        }
        ui.painter().rect_stroke(
            mark,
            CornerRadius::ZERO,
            egui::Stroke::new(1.0, tokens::MUTED),
            egui::StrokeKind::Inside,
        );
    }
    response.on_hover_text(if shell.preview_black() {
        "Show the canvas clear"
    } else {
        "Show the canvas on black"
    });
}

fn duration_label(shell: &Shell) -> String {
    let end = shell
        .draft()
        .map(|draft| draft.duration)
        .unwrap_or(Micros(0));
    shell.timecode_at(end)
}

fn inspector_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let inspected = shell
        .selected()
        .map(|id| id.as_str().to_owned())
        .unwrap_or_default();
    ui.ctx()
        .data_mut(|data| data.insert_temp(egui::Id::unique(INSPECTED), inspected));
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
                project_facts(shell, ui, view, content_width);
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

fn project_facts(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, width: f32) {
    if let Some(stored) = shell.draft().map(|draft| draft.name.clone()) {
        if let Some(name) = name_field(ui, &mut view.name_buffer, &stored)
            && let Err(error) = shell.set_project_name(&name)
        {
            shell.set_notice(error.to_string());
        }
    } else {
        fact(ui, "Name", &shell.project_name(), width);
    }
    let saved = shell
        .project_folder()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "No project".to_owned());
    fact(ui, "Saved", &saved, width);
    let Some((ratio, canvas_width, canvas_height, background)) = shell.draft().map(|draft| {
        (
            ratio_label(&draft.canvas_config.ratio),
            draft.canvas_config.width,
            draft.canvas_config.height,
            draft.canvas_config.background.clone().unwrap_or_default(),
        )
    }) else {
        return;
    };
    let space = shell.color_space_label();
    let color_space = widgets::toolbar_button(ui, &space).on_hover_text("Project color space");
    if color_space.clicked()
        && let Err(error) = shell.cycle_color_space()
    {
        shell.set_notice(error.to_string());
    }
    ui.add_space(8.0);
    let span = ui.available_rect_before_wrap().x_range();
    let y = ui.cursor().min.y;
    ui.painter().hline(
        span,
        y,
        egui::Stroke::new(1.0, Color32::from_white_alpha(20)),
    );
    ui.add_space(16.0);
    let shape = widgets::toolbar_button(ui, &ratio).on_hover_text("Canvas shape");
    if shape.clicked()
        && let Err(error) = shell.cycle_canvas()
    {
        shell.set_notice(error.to_string());
    }
    fact(
        ui,
        "Resolution",
        &format!("{canvas_width}×{canvas_height}"),
        width,
    );
    let rate_label = shell.frame_rate_label();
    let drop_shown = shell.drop_frame_available();
    let drop_label = if shell.drop_frame_enabled() {
        "Drop frame"
    } else {
        "Non-drop"
    };
    let rate = widgets::toolbar_button(ui, &rate_label).on_hover_text("Frames per second");
    if rate.clicked()
        && let Err(error) = shell.cycle_frame_rate()
    {
        shell.set_notice(error.to_string());
    }
    if drop_shown {
        let drop = widgets::toolbar_button(ui, drop_label).on_hover_text("Timecode labels");
        if drop.clicked()
            && let Err(error) = shell.toggle_drop_frame()
        {
            shell.set_notice(error.to_string());
        }
    }
    if let Some(color) = color_field(
        ui,
        "Background",
        &mut view.background_buffer,
        &background,
        "Hex color. Empty is clear",
    ) && let Err(error) = shell.set_canvas_background(&color)
    {
        shell.set_notice(error.to_string());
    }
}

/// Where the inspector keeps the id of the clip it is showing this frame.
const INSPECTED: &str = "inspector-clip";

/// Whether a field's typing lands: it lost focus on the clip it was typed
/// for, and not because Escape was pressed. A click on another clip moves
/// the selection before the field reports losing focus, so the clip is
/// remembered while the field has focus.
fn commits(ui: &Ui, response: &egui::Response) -> bool {
    let clip: String = ui
        .ctx()
        .data(|data| data.get_temp(egui::Id::unique(INSPECTED)))
        .unwrap_or_default();
    let owner = response.id.with("clip");
    if response.has_focus() {
        ui.ctx().data_mut(|data| data.insert_temp(owner, clip));
        return false;
    }
    if !response.lost_focus() {
        return false;
    }
    let typed_for: Option<String> = ui.ctx().data_mut(|data| data.remove_temp(owner));
    let escaped = ui.input(|input| input.key_pressed(egui::Key::Escape));
    !escaped && typed_for.is_none_or(|typed_for| typed_for == clip)
}

fn name_field(ui: &mut Ui, text: &mut String, stored: &str) -> Option<String> {
    let mut edited = None;
    ui.horizontal(|ui| {
        ui.set_min_width(ui.available_width());
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new("Name").size(13.0).color(tokens::MUTED)),
        );
        let response = widgets::field(ui, text).on_hover_text("Empty shows Untitled");
        if commits(ui, &response) {
            edited = Some(text.clone());
        }
        if !response.has_focus() {
            *text = stored.to_owned();
        }
    });
    ui.add_space(4.0);
    edited
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
    if shell.place_enabled() {
        if let Some(x) = number_field(ui, "X", &mut view.x_buffer, inspection.transform_x)
            && !inspection.locked
            && let Err(error) = shell.set_position(x, inspection.transform_y)
        {
            shell.set_notice(error.to_string());
        }
        if let Some(y) = number_field(ui, "Y", &mut view.y_buffer, inspection.transform_y)
            && !inspection.locked
            && let Err(error) = shell.set_position(inspection.transform_x, y)
        {
            shell.set_notice(error.to_string());
        }
        if let Some(scale) = number_field(ui, "Scale", &mut view.scale_buffer, inspection.scale_x)
            && !inspection.locked
            && let Err(error) = shell.set_scale(scale)
        {
            shell.set_notice(error.to_string());
        }
        let label = if shell.flipped() {
            "Flipped"
        } else {
            "Upright"
        };
        let flip = widgets::toolbar_button(ui, label).on_hover_text("Top to bottom");
        if flip.clicked()
            && !inspection.locked
            && let Err(error) = shell.toggle_vertical_flip()
        {
            shell.set_notice(error.to_string());
        }
        if let Some(opacity) =
            number_field(ui, "Opacity", &mut view.opacity_buffer, inspection.opacity)
            && !inspection.locked
            && let Err(error) = shell.set_opacity(opacity)
        {
            shell.set_notice(error.to_string());
        }
    }
    if shell.crop_enabled()
        && let Some(color) = color_field(
            ui,
            "Letterbox",
            &mut view.letterbox_buffer,
            &inspection.letterbox,
            "Around the picture. Empty uses the project color",
        )
        && !inspection.locked
        && let Err(error) = shell.set_letterbox(&color)
    {
        shell.set_notice(error.to_string());
    }
    if let Some(text) = inspection.text.clone() {
        if view.text_for != inspection.segment_id.as_str() {
            view.text_buffer = text;
            view.text_for = inspection.segment_id.as_str().to_owned();
        }
        let response = widgets::field(ui, &mut view.text_buffer);
        if commits(ui, &response)
            && !inspection.locked
            && let Err(error) = shell.set_text(view.text_buffer.clone())
        {
            shell.set_notice(error.to_string());
        }
        let face_name = if inspection.font.is_empty() {
            "Face"
        } else {
            inspection.font.as_str()
        };
        let face = widgets::toolbar_button(ui, face_name).on_hover_text("Installed face");
        if face.clicked()
            && !inspection.locked
            && let Err(error) = shell.cycle_text_font()
        {
            shell.set_notice(error.to_string());
        }
        if let Some(size) =
            number_field(ui, "Size", &mut view.text_size_buffer, inspection.text_size)
            && !inspection.locked
            && let Err(error) = shell.set_text_size(size)
        {
            shell.set_notice(error.to_string());
        }
        if let Some(color) = color_field(
            ui,
            "Color",
            &mut view.text_color_buffer,
            &inspection.text_color,
            "Hex color",
        ) && !inspection.locked
            && let Err(error) = shell.set_text_color(&color)
        {
            shell.set_notice(error.to_string());
        }
        let align = widgets::toolbar_button(ui, align_name(inspection.text_align))
            .on_hover_text("Left, center, or right");
        if align.clicked()
            && !inspection.locked
            && let Err(error) = shell.cycle_text_align()
        {
            shell.set_notice(error.to_string());
        }
        if let Some(spacing) = number_field(
            ui,
            "Letters",
            &mut view.letter_buffer,
            inspection.letter_spacing,
        ) && !inspection.locked
            && let Err(error) = shell.set_letter_spacing(spacing)
        {
            shell.set_notice(error.to_string());
        }
        if let Some(spacing) =
            number_field(ui, "Lines", &mut view.line_buffer, inspection.line_spacing)
            && !inspection.locked
            && let Err(error) = shell.set_line_spacing(spacing)
        {
            shell.set_notice(error.to_string());
        }
        if let Some(width) =
            number_field(ui, "Width", &mut view.width_buffer, inspection.line_width)
            && !inspection.locked
            && let Err(error) = shell.set_line_width(width)
        {
            shell.set_notice(error.to_string());
        }
    }
    if let Some(volume) = number_field(ui, "Volume", &mut view.volume_buffer, inspection.volume)
        && !inspection.locked
        && let Err(error) = shell.set_volume(volume)
    {
        shell.set_notice(error.to_string());
    }
    if shell.fade_enabled() {
        let fade_in = fade_seconds(inspection.fade_in);
        let fade_out = fade_seconds(inspection.fade_out);
        if let Some(seconds) = number_field(ui, "Fade in", &mut view.fade_in_buffer, fade_in)
            && !inspection.locked
            && let Err(error) = shell.set_fade_in(seconds)
        {
            shell.set_notice(error.to_string());
        }
        if let Some(seconds) = number_field(ui, "Fade out", &mut view.fade_out_buffer, fade_out)
            && !inspection.locked
            && let Err(error) = shell.set_fade_out(seconds)
        {
            shell.set_notice(error.to_string());
        }
    }
    if let Some(speed) = number_field(ui, "Speed", &mut view.speed_buffer, inspection.speed)
        && !inspection.locked
        && let Err(error) = shell.set_speed(speed)
    {
        shell.set_notice(error.to_string());
    }
}

/// A number row. The typed text stays while the field has focus, and the
/// stored value returns when it does not. `Some` is a new finite parse.
fn number_field(ui: &mut Ui, label: &str, text: &mut String, stored: f64) -> Option<f64> {
    let mut edited = None;
    ui.horizontal(|ui| {
        ui.set_min_width(ui.available_width());
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(tokens::MUTED)),
        );
        let response = widgets::field(ui, text);
        if commits(ui, &response) {
            edited = text.parse().ok();
        }
        if !response.has_focus() {
            *text = format!("{stored}");
        }
        if label == "Y" {
            response.on_hover_text("Up is positive");
        } else if label == "Scale" {
            response.on_hover_text("1 is the fitted picture");
        } else if label == "Opacity" {
            response.on_hover_text("1 is solid");
        } else if label == "Size" {
            response.on_hover_text("Pixels");
        } else if label == "Letters" || label == "Lines" {
            response.on_hover_text("0 is the normal gap");
        } else if label == "Width" {
            response.on_hover_text("Pixels. 0 keeps one line");
        } else if label.starts_with("Fade") {
            response.on_hover_text("Seconds");
        }
    });
    ui.add_space(4.0);
    edited.filter(|value| *value != stored)
}

fn fade_seconds(microseconds: i64) -> f64 {
    microseconds as f64 / 1_000_000.0
}

fn color_field(
    ui: &mut Ui,
    label: &str,
    text: &mut String,
    stored: &str,
    hover: &str,
) -> Option<String> {
    let mut edited = None;
    ui.horizontal(|ui| {
        ui.set_min_width(ui.available_width());
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(tokens::MUTED)),
        );
        let response = widgets::field(ui, text);
        if commits(ui, &response) {
            edited = Some(text.clone());
        }
        if !response.has_focus() {
            *text = stored.to_owned();
        }
        response.on_hover_text(hover);
    });
    ui.add_space(4.0);
    edited.filter(|value| value != stored)
}

fn align_name(alignment: i32) -> &'static str {
    match alignment {
        1 => "Center",
        2 => "Right",
        _ => "Left",
    }
}

fn timeline_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    panel(ui, rect);
    shell.chrome.timeline = true;
    let toolbar = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::TOOLBAR_HEIGHT));
    paint_toolbar(shell, ui, toolbar);
    paint_notice(
        shell,
        ui,
        toolbar.center().x,
        toolbar.bottom() + tokens::RULER_HEIGHT + 8.0,
    );
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
                                "rotate" => shell.rotated(),
                                "crop" => shell.cropped(),
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
                                "rotate" if shell.rotated() => "Rotated",
                                "rotate" => "Rotate",
                                "crop" if shell.cropped() => "Cropped",
                                "crop" => "Crop",
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
        "undo" => shell.can_undo(),
        "redo" => shell.can_redo(),
        "select" | "delete" | "zoomIn" | "zoomOut" | "zoomFit" | "guides" | "lock" | "linkage"
        | "magnet" | "snap" => true,
        "split" | "marker" | "beat" => shell.has_clip(),
        "freeze" => shell.freeze_enabled(),
        "reverse" => shell.reverse_enabled(),
        "mirror" => shell.mirror_enabled(),
        "rotate" => shell.rotate_enabled(),
        "crop" => shell.crop_enabled(),
        "transcribe" => shell.transcribe_enabled(),
        "splitScenes" => shell.split_scenes_enabled(),
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
        "rotate" => shell.turn_clockwise(),
        "crop" => shell.toggle_crop(),
        "transcribe" => shell.transcribe(),
        "splitScenes" => shell.split_scenes(),
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

/// The notice as a toast that fades in, holds, then fades out and goes.
fn paint_notice(shell: &mut Shell, ui: &Ui, center_x: f32, top: f32) {
    shell.expire_notice();
    if shell.notice().is_empty() {
        return;
    }
    let age = shell.notice_age();
    let alpha = notice_alpha(age);
    let left = crate::shell::NOTICE_SHOWN.saturating_sub(age);
    if alpha < 1.0 {
        ui.ctx().request_repaint();
    } else {
        ui.ctx()
            .request_repaint_after(left.saturating_sub(crate::shell::NOTICE_FADE));
    }
    let ink = Color32::from_rgb(0x14, 0x14, 0x14).gamma_multiply(alpha);
    let galley = ui.ctx().fonts_mut(|fonts| {
        fonts.layout_no_wrap(shell.notice().to_owned(), FontId::proportional(12.0), ink)
    });
    let size = galley.size() + Vec2::new(24.0, 12.0);
    let rect = Rect::from_min_size(Pos2::new(center_x - size.x / 2.0, top), size);
    ui.painter().rect_filled(
        rect,
        CornerRadius::same(99),
        Color32::WHITE.gamma_multiply(alpha),
    );
    ui.painter().galley(
        rect.left_center() + Vec2::new(12.0, -galley.size().y / 2.0),
        galley,
        ink,
    );
}

/// How opaque a notice is `age` after it was set: a quick fade in, a hold,
/// and [`crate::shell::NOTICE_FADE`] out at the end.
fn notice_alpha(age: std::time::Duration) -> f32 {
    const FADE_IN: f32 = 0.04;
    let age = age.as_secs_f32();
    let shown = crate::shell::NOTICE_SHOWN.as_secs_f32();
    let fade = crate::shell::NOTICE_FADE.as_secs_f32();
    let rising = if age < FADE_IN { age / FADE_IN } else { 1.0 };
    let falling = ((shown - age) / fade).clamp(0.0, 1.0);
    rising.min(falling)
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
    let kinds: Vec<(bool, bool)> = rows
        .iter()
        .map(|lane| (lane.main, is_audio(lane)))
        .collect();
    let laid = lane_rects(kinds, lane_area);
    let (low, high) = lane_scroll_range(&laid, lane_area);
    let scrolls = low < high;
    if scrolls {
        view.lane_scroll += lane_wheel(ui, tracks);
    }
    view.lane_scroll = view.lane_scroll.clamp(low, high);
    let boxes: Vec<Rect> = laid
        .iter()
        .map(|rect| rect.translate(Vec2::new(0.0, view.lane_scroll)))
        .collect();
    let marquee = ui.interact(
        lane_area,
        ui.scope_id().with("marquee"),
        Sense::click_and_drag(),
    );
    if rows.is_empty() {
        paint_empty_main(ui, lane_area);
    }
    // Lanes scrolled past the ruler or the panel's foot are cut there.
    let clip = ui.clip_rect();
    ui.set_clip_rect(clip.intersect(tracks));
    for (index, lane) in rows.iter().enumerate() {
        let Some(row) = boxes.get(index).copied() else {
            continue;
        };
        paint_lane(shell, ui, lane, row, lanes_left, index);
    }
    ui.set_clip_rect(clip);
    shell.send_timeline_asks();
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
    pan_timeline(shell, ui, lane_area, scrolls);
    zoom_timeline(shell, ui, lane_area);
}

/// How far the lanes may move: none while they fit, else far enough to
/// bring the top lane or the bottom lane in with a little room.
fn lane_scroll_range(boxes: &[Rect], area: Rect) -> (f32, f32) {
    const ROOM: f32 = 8.0;
    let (Some(top), Some(bottom)) = (
        boxes.iter().map(|rect| rect.top()).reduce(f32::min),
        boxes.iter().map(|rect| rect.bottom()).reduce(f32::max),
    ) else {
        return (0.0, 0.0);
    };
    let low = (area.bottom() - ROOM - bottom).min(0.0);
    let high = (area.top() + ROOM - top).max(0.0);
    (low, high)
}

/// This frame's wheel, whether the zoom modifier is held, and the pointer.
/// egui already turns Shift and an up-and-down wheel sideways.
fn wheel(ui: &Ui) -> (Vec2, bool, Option<Pos2>) {
    ui.input(|input| {
        (
            input.smooth_scroll_delta,
            input.modifiers.command || input.modifiers.ctrl,
            input.pointer.hover_pos(),
        )
    })
}

/// The up-and-down wheel over the track area, in points to move the lanes.
fn lane_wheel(ui: &Ui, tracks: Rect) -> f32 {
    let (delta, command, pos) = wheel(ui);
    let over = pos.is_some_and(|pos| tracks.contains(pos) && !over_menu(ui, pos));
    if command || !over || delta.y.abs() <= delta.x.abs() {
        return 0.0;
    }
    delta.y
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
    let Some(kind) = shell.selected_track_kind() else {
        shell.close_clip_menu();
        return;
    };
    let enabled = |id: &str| match id {
        "edit-crop" => shell.crop_enabled(),
        "split-scenes" => shell.split_scenes_enabled(),
        "transcript" => shell.transcribe_enabled(),
        "extract-audio" => shell.extract_audio_enabled(),
        "flip" => shell.mirror_enabled(),
        _ => true,
    };
    let shown = crate::menu::show(
        ui.ctx(),
        ui.scope_id().with("clip-menu"),
        Pos2::new(menu.x, menu.y),
        crate::menu::clip_entries(&kind),
        &enabled,
    );
    if let Some(item) = shown.picked {
        shell.close_clip_menu();
        let result = match item.id {
            _ if !item.built => {
                shell.set_notice(format!("{} is not built yet", item.label));
                Ok(())
            }
            "copy" => shell.copy_selected(),
            "cut" => shell.cut_selected(),
            "delete" => shell.remove_selected(),
            "edit-crop" => shell.toggle_crop(),
            "split-scenes" => shell.split_scenes(),
            "transcript" => shell.transcribe(),
            "extract-audio" => shell.extract_audio(),
            "flip" => shell.toggle_mirror(),
            _ => Ok(()),
        };
        if let Err(error) = result {
            shell.set_notice(error.to_string());
        }
    } else if shown.dismissed {
        shell.close_clip_menu();
    }
}

fn scroll_with_drag(shell: &mut Shell, ui: &Ui, x_in_view: f64) {
    if shell.drag_scroll(x_in_view) {
        ui.ctx().request_repaint();
    }
}

/// A sideways swipe, or Shift and the wheel, over the lanes moves the
/// visible span. While the lanes are taller than the panel the plain wheel
/// scrolls them instead; while they fit it moves the span too.
fn pan_timeline(shell: &mut Shell, ui: &Ui, lane_area: Rect, lanes_scroll: bool) {
    let (delta, command, pos) = wheel(ui);
    if command || delta == Vec2::ZERO {
        return;
    }
    let Some(pos) = pos else {
        return;
    };
    if !lane_area.contains(pos) || over_menu(ui, pos) {
        return;
    }
    let pixels = if delta.x.abs() >= delta.y.abs() {
        delta.x
    } else if lanes_scroll {
        return;
    } else {
        delta.y
    };
    shell.pan_pixels(-f64::from(pixels));
}

/// A right-click menu is open under `pos`, so a wheel there scrolls the menu.
fn over_menu(ui: &Ui, pos: Pos2) -> bool {
    ui.ctx()
        .layer_id_at(pos)
        .is_some_and(|layer| layer.order == egui::Order::Foreground)
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
    if !lane_area.contains(pos) || over_menu(ui, pos) {
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

fn paint_library_ghost(shell: &mut Shell, ui: &Ui) {
    let Some(payload) = egui::DragAndDrop::payload::<LibraryDrop>(ui.ctx()) else {
        return;
    };
    let Some(pos) = ui.input(|input| input.pointer.interact_pos()) else {
        return;
    };
    let Some(mut card) = ghost_card(shell, &payload.material_id) else {
        return;
    };
    fill_looks(shell, ui.ctx(), std::slice::from_mut(&mut card));
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
            look: CardLook::Swatch,
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
            meta: card_clock(video.duration.0),
            on_timeline: used.contains(video.id.as_str()),
            look: CardLook::Picture(None),
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
        meta: card_clock(audio.duration.0),
        on_timeline: used.contains(audio.id.as_str()),
        look: CardLook::Sound(std::sync::Arc::from([])),
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

/// The ruler's ticks and labels across the lanes. Past the end of the cut,
/// the ticks keep going.
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
    let width = f64::from(ruler.right() - lanes_left).max(0.0);
    let span = (width / pixel_size).ceil();
    if !span.is_finite() || span > i64::MAX as f64 {
        return;
    }
    let from = Micros(origin.max(0));
    let to = Micros(origin.saturating_add(span as i64));
    let painter = ui.painter().with_clip_rect(Rect::from_min_max(
        Pos2::new(lanes_left, ruler.top()),
        ruler.max,
    ));
    for tick in ruler_ticks(rate, drop_frame, pixel_size, from, to) {
        let x = lane_x(lanes_left, pixel_size, origin, tick.time.0);
        let height = if tick.major {
            tokens::MAJOR_TICK
        } else {
            tokens::MAJOR_TICK / 2.0
        };
        painter.vline(
            x,
            (ruler.bottom() - height)..=ruler.bottom(),
            egui::Stroke::new(1.0, tokens::RULER_TICK),
        );
        if let Some(label) = tick.label {
            painter.text(
                Pos2::new(x + 4.0, ruler.bottom() - tokens::MAJOR_TICK - 2.0),
                Align2::LEFT_BOTTOM,
                label,
                FontId::monospace(11.0),
                tokens::MUTED,
            );
        }
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

fn paint_empty_main(ui: &Ui, area: Rect) {
    let height = row_height(true, false);
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
            let bounds = clip_bounds(
                shell,
                lane.main,
                is_audio(lane),
                start,
                duration,
                row,
                lanes_left,
            );
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
    audio: bool,
    start: Micros,
    duration: Micros,
    row: Rect,
    lanes_left: f32,
) -> Rect {
    let height = clip_height(main, audio);
    let x = lane_x(
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        start.0,
    );
    let width = (duration.0 as f32 * shell.pixel_size() as f32).max(8.0);
    Rect::from_min_size(
        Pos2::new(x, row.top() + clip_inset(main)),
        Vec2::new(width, height),
    )
}

/// A clip's height: the main track is tallest, audio holds its waveform
/// under a header, the rest are a single row.
fn clip_height(main: bool, audio: bool) -> f32 {
    if main {
        tokens::MAIN_CLIP_HEIGHT
    } else if audio {
        tokens::AUDIO_CLIP_HEIGHT
    } else {
        tokens::CLIP_HEIGHT
    }
}

fn is_audio(lane: &crate::shell::LaneView) -> bool {
    lane.kind == cutlass_model::TrackType::Audio
}

fn clip_inset(main: bool) -> f32 {
    if main {
        tokens::MAIN_CLIP_INSET
    } else {
        tokens::CLIP_INSET
    }
}

/// A lane row: its clips and the gap above and below them.
fn row_height(main: bool, audio: bool) -> f32 {
    clip_height(main, audio) + 2.0 * clip_inset(main)
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

/// Thumbnail textures by the picture they show, each with the frame it
/// was last drawn in. egui's memory holds a shared handle, so reaching the
/// map copies nothing.
#[derive(Default)]
struct FilmTextures {
    shown: std::collections::HashMap<cutlass_engine::ThumbKey, (egui::TextureHandle, u64)>,
    frame: u64,
}

type SharedFilm = std::sync::Arc<std::sync::Mutex<FilmTextures>>;

/// Textures kept before the least recently drawn are let go.
const FILM_TEXTURES: usize = 1500;

/// Run `f` on the thumbnail textures, then let go of the least recently
/// drawn quarter when there are too many.
fn with_film_textures<R>(ctx: &egui::Context, f: impl FnOnce(&mut FilmTextures) -> R) -> R {
    let store = egui::Id::NULL.with("film textures");
    let shared = ctx.data_mut(|data| data.get_temp_mut_or_default::<SharedFilm>(store).clone());
    let mut textures = shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    textures.frame = ctx.cumulative_pass_nr();
    let result = f(&mut textures);
    evict(&mut textures, FILM_TEXTURES);
    result
}

/// Keep at most `limit` textures: past it, drop the least recently drawn
/// until a quarter of the room is free again.
fn evict(textures: &mut FilmTextures, limit: usize) {
    if textures.shown.len() <= limit {
        return;
    }
    let keep = limit - limit / 4;
    let mut ages: Vec<u64> = textures.shown.values().map(|(_, used)| *used).collect();
    ages.sort_unstable_by(|a, b| b.cmp(a));
    let cutoff = ages[keep - 1];
    textures.shown.retain(|_, (_, used)| *used >= cutoff);
}

/// The texture of `thumb`, loaded the first time it is shown.
fn film_texture(
    ctx: &egui::Context,
    textures: &mut FilmTextures,
    key: cutlass_engine::ThumbKey,
    thumb: &cutlass_engine::Thumb,
) -> egui::TextureId {
    let frame = textures.frame;
    let entry = textures.shown.entry(key).or_insert_with(|| {
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [thumb.width as usize, thumb.height as usize],
            &thumb.rgba,
        );
        (
            ctx.load_texture("film", image, egui::TextureOptions::LINEAR),
            frame,
        )
    });
    entry.1 = frame;
    entry.0.id()
}

/// The clip's filmstrip in the `film` rows of `clip_rect`, over the tiles
/// that meet `visible`. A tile without its picture yet keeps the clip color.
fn paint_film(
    shell: &mut Shell,
    ui: &Ui,
    id: &SegmentId,
    clip_rect: Rect,
    film: Rect,
    visible: Rect,
) {
    let span =
        f64::from(visible.left() - clip_rect.left())..f64::from(visible.right() - clip_rect.left());
    let cells = shell.film(
        id,
        f64::from(clip_rect.width()),
        f64::from(film.height()),
        span,
    );
    if cells.is_empty() {
        return;
    }
    with_film_textures(ui.ctx(), |textures| {
        // Inside the clip's edge, rounded where a picture meets a corner.
        let inner = clip_rect.shrink(1.0);
        let radius = CLIP_RADIUS - 1;
        let painter = ui.painter().with_clip_rect(visible);
        for cell in cells {
            let Some(thumb) = cell.thumb else {
                continue;
            };
            let texture = film_texture(ui.ctx(), textures, cell.key, &thumb);
            let full = film.height() * thumb.width as f32 / thumb.height as f32;
            let tile = Rect::from_min_size(
                Pos2::new(clip_rect.left() + cell.tile.left as f32, film.top()),
                Vec2::new(cell.tile.width as f32, film.height()),
            );
            let drawn = tile.intersect(inner).intersect(film);
            if drawn.width() <= 0.0 || drawn.height() <= 0.0 {
                continue;
            }
            let uv = Rect::from_min_max(
                Pos2::new(
                    (drawn.left() - tile.left()) / full,
                    (drawn.top() - film.top()) / film.height(),
                ),
                Pos2::new(
                    ((drawn.right() - tile.left()) / full).min(1.0),
                    (drawn.bottom() - film.top()) / film.height(),
                ),
            );
            let at = |a: f32, b: f32| (a - b).abs() < 0.5;
            let corner = |side: bool, end: bool| if side && end { radius } else { 0 };
            let (left, right) = (
                at(drawn.left(), inner.left()),
                at(drawn.right(), inner.right()),
            );
            let (top, bottom) = (
                at(drawn.top(), inner.top()),
                at(drawn.bottom(), inner.bottom()),
            );
            let corners = CornerRadius {
                nw: corner(left, top),
                ne: corner(right, top),
                sw: corner(left, bottom),
                se: corner(right, bottom),
            };
            painter.add(
                egui::epaint::RectShape::filled(drawn, corners, Color32::from_gray(210))
                    .with_texture(texture, uv),
            );
        }
    });
}

/// The corner radius of a clip.
const CLIP_RADIUS: u8 = 7;

/// The lowest a bar at `x` reaches in `band`, which shares the clip's
/// sides and bottom: the bottom, raised inside the rounded corners.
fn corner_floor(band: Rect, x: f32) -> f32 {
    let radius = f32::from(CLIP_RADIUS);
    let inset = (x - band.left()).min(band.right() - x);
    if inset >= radius {
        return band.bottom();
    }
    let across = radius - inset.max(0.0);
    band.bottom() - (radius - (radius * radius - across * across).max(0.0).sqrt())
}

/// Waveform bars are one point wide, one every two points.
const BAR_PITCH: f32 = 2.0;
/// The quietest level a bar shows, in decibels below full scale.
const WAVE_FLOOR_DB: f32 = 48.0;
/// A bar this high in the band wears an orange cap.
const WAVE_LOUD: f32 = 0.9;
/// A level this close to full scale clips, and its bar wears a red tip.
const WAVE_CLIPPED: f32 = 0.999;

/// How much of the band a level fills: decibels from the floor up to full
/// scale, so speech sits high in the band.
fn wave_height(level: f32) -> f32 {
    if level <= 0.0 || !level.is_finite() {
        return 0.0;
    }
    ((20.0 * level.log10() + WAVE_FLOOR_DB) / WAVE_FLOOR_DB).clamp(0.0, 1.0)
}

/// A waveform's bars and volume line, and the clip fill it sits on.
#[derive(Clone, Copy)]
struct WaveColors {
    bar: Color32,
    line: Color32,
    fill: Color32,
}

impl WaveColors {
    const VIDEO: Self = Self {
        bar: tokens::CLIP_VIDEO_BAR,
        line: tokens::CLIP_VIDEO_LINE,
        fill: tokens::CLIP_VIDEO,
    };
    const AUDIO: Self = Self {
        bar: tokens::CLIP_AUDIO_BAR,
        line: tokens::CLIP_AUDIO_LINE,
        fill: tokens::CLIP_AUDIO,
    };
}

/// `from` moved `t` of the way to `to`, channel by channel.
fn lerp(from: Color32, to: Color32, t: f32) -> Color32 {
    let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        mix(from.r(), to.r()),
        mix(from.g(), to.g()),
        mix(from.b(), to.b()),
        mix(from.a(), to.a()),
    )
}

/// Where the volume line sits in the band for a gain: 100% at 45%,
/// +20 dB, the most a clip takes, at the top, and quieter gains down the
/// same decibel scale as the bars.
fn volume_height(gain: f32) -> f32 {
    const UNITY: f32 = 0.45;
    if gain <= 0.0 || !gain.is_finite() {
        return 0.0;
    }
    let db = 20.0 * gain.log10();
    if db >= 0.0 {
        UNITY + (1.0 - UNITY) * (db / 20.0).min(1.0)
    } else {
        UNITY * (1.0 + db / WAVE_FLOOR_DB).max(0.0)
    }
}

/// The waveform as bars rising from the bottom of `band`, each the loudest
/// of the pixel columns under it, crossed by the volume line. Columns count from `clip_left`.
fn paint_wave(
    painter: &egui::Painter,
    wave: &crate::shell::ClipWave,
    clip_left: f32,
    band: Rect,
    visible: Rect,
    colors: WaveColors,
) {
    let WaveColors { bar, line, fill } = colors;
    // A muted track's bars fade almost into the clip, without their caps.
    let bar = if wave.muted {
        lerp(fill, bar, 0.2)
    } else {
        bar
    };
    let painter = painter.with_clip_rect(band.intersect(visible));
    let floor = band.bottom() - 1.0;
    let reach = (band.height() - 2.0).max(1.0);
    let mut mesh = egui::Mesh::default();
    let mut index = 0;
    while index < wave.levels.len() {
        let column = wave.first as usize + index;
        let slot = column / BAR_PITCH as usize;
        let end = ((slot + 1) * BAR_PITCH as usize - wave.first as usize).min(wave.levels.len());
        let level = wave.levels[index..end].iter().copied().fold(0.0, f32::max);
        index = end;
        let height = wave_height(level) * reach;
        if height <= 0.0 {
            continue;
        }
        let left = clip_left + slot as f32 * BAR_PITCH;
        let top = floor - height;
        let bottom = floor.min(corner_floor(band, left + 0.5));
        if top >= bottom {
            continue;
        }
        mesh.add_colored_rect(
            Rect::from_min_max(Pos2::new(left, top), Pos2::new(left + 1.0, bottom)),
            bar,
        );
        let cap = |from: f32, color: Color32, mesh: &mut egui::Mesh| {
            mesh.add_colored_rect(
                Rect::from_min_max(Pos2::new(left, from), Pos2::new(left + 1.0, from + 1.0)),
                color,
            );
        };
        if wave.muted {
            continue;
        }
        if level >= WAVE_CLIPPED {
            cap(top, tokens::WAVE_CLIPPED, &mut mesh);
            cap(top + 1.0, tokens::WAVE_LOUD, &mut mesh);
        } else if wave_height(level) >= WAVE_LOUD {
            cap(top, tokens::WAVE_LOUD, &mut mesh);
        }
    }
    // The volume line, a device pixel thick, at each column's gain.
    let thick = 1.0 / painter.pixels_per_point();
    for (index, gain) in wave.gains.iter().enumerate() {
        let left = clip_left + (wave.first as usize + index) as f32;
        let y = (floor - volume_height(*gain) * reach).round();
        if y + thick > corner_floor(band, left + 0.5) {
            continue;
        }
        mesh.add_colored_rect(
            Rect::from_min_max(Pos2::new(left, y), Pos2::new(left + 1.0, y + thick)),
            line,
        );
    }
    painter.add(mesh);
}

/// A clip's header row, when it has one, its filmstrip, and its sound band.
struct ClipParts {
    header: Option<Rect>,
    film: Option<Rect>,
    band: Option<Rect>,
}

/// Main-track and audio clips carry a header row. Under it a main-track
/// clip with sound gives 30% to the band and the rest to the
/// filmstrip; an audio clip is all band. Other clips are all filmstrip.
fn clip_parts(clip_rect: Rect, main: bool, audio: bool, sound: bool) -> ClipParts {
    if !main && !audio {
        return ClipParts {
            header: None,
            film: Some(clip_rect),
            band: None,
        };
    }
    let split = clip_rect.top() + tokens::CLIP_HEADER.min(clip_rect.height());
    let header = Some(Rect::from_min_max(
        clip_rect.min,
        Pos2::new(clip_rect.right(), split),
    ));
    let body = Rect::from_min_max(Pos2::new(clip_rect.left(), split), clip_rect.max);
    if audio {
        return ClipParts {
            header,
            film: None,
            band: Some(body),
        };
    }
    if !sound {
        return ClipParts {
            header,
            film: Some(body),
            band: None,
        };
    }
    let top = body.bottom() - body.height() * tokens::WAVE_RATIO;
    ClipParts {
        header,
        film: Some(Rect::from_min_max(body.min, Pos2::new(body.right(), top))),
        band: Some(Rect::from_min_max(Pos2::new(body.left(), top), body.max)),
    }
}

/// The header's chips, left to right from its left edge: the clip's name,
/// then its length when given.
fn paint_header(ui: &Ui, header: Rect, clip: Rect, chips: &[String], fill: Color32, faded: bool) {
    let painter = ui
        .painter()
        .with_clip_rect(clip.shrink(1.0).intersect(header));
    let text = if faded {
        tokens::CLIP_TEXT.gamma_multiply(0.45)
    } else {
        tokens::CLIP_TEXT
    };
    let mut left = header.left() + 4.0;
    for chip in chips {
        let galley = painter.layout_no_wrap(chip.clone(), FontId::proportional(11.0), text);
        let rect = Rect::from_min_size(
            Pos2::new(left, header.top() + 1.5),
            Vec2::new(galley.size().x + 8.0, header.height() - 3.0),
        );
        painter.rect_filled(rect, CornerRadius::same(3), fill);
        painter.galley(
            Pos2::new(rect.left() + 4.0, rect.center().y - galley.size().y / 2.0),
            galley,
            text,
        );
        left = rect.right() + 4.0;
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
    let clip_rect = clip_bounds(
        shell,
        lane.main,
        is_audio(lane),
        start,
        duration,
        row,
        lanes_left,
    );
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
    painter.rect_filled(clip_rect, CornerRadius::same(CLIP_RADIUS), fill);
    let audio = is_audio(lane);
    let span =
        f64::from(visible.left() - clip_rect.left())..f64::from(visible.right() - clip_rect.left());
    let wave = if ghost {
        None
    } else {
        shell.wave(&clip.id, f64::from(clip_rect.width()), span)
    };
    let parts = clip_parts(clip_rect, lane.main, audio, wave.is_some());
    if !ghost {
        if let Some(film) = parts.film {
            paint_film(shell, ui, &clip.id, clip_rect, film, visible);
        }
        if let (Some(band), Some(wave)) = (parts.band, wave.as_ref()) {
            let colors = if audio {
                WaveColors::AUDIO
            } else {
                WaveColors::VIDEO
            };
            paint_wave(ui.painter(), wave, clip_rect.left(), band, visible, colors);
        }
    }
    if selected {
        painter.rect_stroke(
            clip_rect,
            CornerRadius::same(CLIP_RADIUS),
            egui::Stroke::new(1.5, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    if let Some(header) = parts.header {
        let mut chips = vec![clip.name.clone()];
        if lane.main {
            chips.push(shell.timecode_at(duration));
        }
        let chip = if audio {
            tokens::CLIP_AUDIO_CHIP
        } else {
            tokens::CLIP_VIDEO_CHIP
        };
        paint_header(ui, header, clip_rect.intersect(row), &chips, chip, ghost);
    } else {
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
    }
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
    let clip_rect = clip_bounds(
        shell,
        lane.main,
        is_audio(lane),
        preview.start,
        preview.duration,
        row,
        lanes_left,
    );
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

/// Each lane's row, the main lane centered. `kinds` is each lane's main and
/// audio flags.
fn lane_rects(kinds: Vec<(bool, bool)>, area: Rect) -> Vec<Rect> {
    let main_flags: Vec<bool> = kinds.iter().map(|(main, _)| *main).collect();
    let height = |index: usize| row_height(kinds[index].0, kinds[index].1);
    let Some(main) = main_flags.iter().position(|main| *main) else {
        let mut top = area.top();
        return (0..kinds.len())
            .map(|index| {
                let rect = Rect::from_min_size(
                    Pos2::new(area.left(), top),
                    Vec2::new(area.width(), height(index)),
                );
                top += height(index) + 8.0;
                rect
            })
            .collect();
    };
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
        cutlass_model::TrackType::Audio => tokens::CLIP_AUDIO,
        cutlass_model::TrackType::Text
        | cutlass_model::TrackType::TextToVideoText
        | cutlass_model::TrackType::TextToVideoTextAudio => Color32::from_rgb(0x9a, 0x86, 0xb0),
        cutlass_model::TrackType::Sticker => Color32::from_rgb(0xc4, 0xa1, 0x5a),
        cutlass_model::TrackType::Effect | cutlass_model::TrackType::PluginEffect => {
            Color32::from_rgb(0xc4, 0x7a, 0x5a)
        }
        cutlass_model::TrackType::Filter => Color32::from_rgb(0x6a, 0x8f, 0x72),
        cutlass_model::TrackType::Adjust => Color32::from_rgb(0x8a, 0x7e, 0x9a),
        _ => tokens::CLIP_VIDEO,
    }
}

fn export_dialog(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    parent: Option<&eframe::Frame>,
) {
    let ((), closed) = widgets::dialog(ui, "Export", |ui| {
        ui.label(RichText::new("Short side").color(tokens::MUTED));
        for side in crate::shell::SHORT_SIDES {
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
        short_side: view.export_side,
        codec: view.export_codec,
        replace: view.replace_prompt,
        ..ExportRequest::default()
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
        egui::Key::A => "a",
        egui::Key::Q => "q",
        egui::Key::W => "w",
        _ => return None,
    };
    let modifiers = input.modifiers;
    // Control apart from the primary key, as on a Mac.
    let control = modifiers.ctrl && !modifiers.command;
    let plain = !(modifiers.command || modifiers.alt || modifiers.shift || control);
    let blocked = match name {
        "space" | "left" | "right" | "delete" | "q" | "w" => !plain,
        // S splits, and Shift-Command-S extracts the audio.
        "s" => !plain && !(modifiers.command && modifiers.shift && !modifiers.alt),
        _ => false,
    };
    if blocked {
        return None;
    }
    let mut parts = Vec::new();
    if control {
        parts.push("ctrl");
    }
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
    fn film_textures_let_go_of_the_least_recently_drawn() {
        let ctx = egui::Context::default();
        let thumb = cutlass_engine::Thumb {
            width: 1,
            height: 1,
            rgba: vec![0; 4],
        };
        let key = |n: i64| cutlass_engine::ThumbKey {
            path: "clip.mov".into(),
            time: Micros(n),
        };
        let mut textures = FilmTextures::default();
        for n in 0..8 {
            textures.frame = n as u64;
            film_texture(&ctx, &mut textures, key(n), &thumb);
        }
        // Drawing the first again makes it recent.
        textures.frame = 9;
        film_texture(&ctx, &mut textures, key(0), &thumb);
        evict(&mut textures, 4);
        let mut kept: Vec<i64> = textures.shown.keys().map(|key| key.time.0).collect();
        kept.sort_unstable();
        assert_eq!(kept, [0, 6, 7]);
    }

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
        // The home looks at the folder once a second; skip the wait.
        view.listed_checked = None;
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
        let lines = galleys(&output);
        output.textures_delta.clear();
        // The timecode shows on the transport only, not over the preview.
        assert!(!lines.iter().any(|line| line == "00:00:00:00"), "{lines:?}");
        assert!(painted.contains("00:00:00:00  /  "), "{painted}");
        assert!(painted.contains("Details"), "{painted}");
        assert!(painted.contains("Main"), "{painted}");
        assert!(painted.contains("Untitled"), "{painted}");
        assert!(
            painted.contains("No media yet. Use File → Import."),
            "{painted}"
        );
        assert!(painted.contains("Projects"), "{painted}");
        assert!(painted.contains("00:10"), "{painted}");
        assert!(painted.contains("00:20"), "{painted}");
        assert!(painted.contains("No clips"), "{painted}");
        assert!(!painted.contains("Import a file"), "{painted}");
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Rects filled black larger than a button's mark: the canvas background.
    fn black_rects(output: &egui::FullOutput) -> Vec<Rect> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect)
                    if rect.fill == Color32::BLACK && rect.rect.width() > 32.0 =>
                {
                    Some(rect.rect)
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn only_the_preview_canvas_turns_black_and_the_toggle_clears_it() {
        let temp = std::env::temp_dir().join(format!("cutlass-black-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 9;
        draft.duration = Micros(1_000_000);
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "solid".into(),
                is_color_clip: true,
                solid_color: "0xFFFF0000".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "solid",
            "solid",
            cutlass_model::TimeRange::new(0, 1_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            // The round transport buttons, left to right: the toggle, the still.
            let mut buttons: Vec<Pos2> = output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::Circle(circle)
                        if circle.radius == 16.0 && circle.fill == tokens::ICON_BUTTON =>
                    {
                        Some(circle.center)
                    }
                    _ => None,
                })
                .collect();
            buttons.sort_by(|a, b| a.x.total_cmp(&b.x));
            (black_rects(&output), buttons)
        };
        let (black, buttons) = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(black.len(), 1, "{black:?}");
        let ratio = black[0].width() / black[0].height();
        assert!((ratio - 16.0 / 9.0).abs() < 0.01, "{:?}", black[0]);
        assert_eq!(buttons.len(), 2, "{buttons:?}");
        let toggle = buttons[0];
        let press = |pressed: bool| egui::Event::PointerButton {
            pos: toggle,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(toggle), press(true)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(false)]);
        assert!(!shell.preview_black());
        // The click lands after that frame's canvas was painted; the next is clear.
        let (black, _) = frame(&mut shell, &mut view, Vec::new());
        assert!(black.is_empty(), "{black:?}");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_video_clip_paints_its_filmstrip_once_the_thumbnails_land() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.mp4");
        if !movie.is_file() {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        }
        let temp = std::env::temp_dir().join(format!("cutlass-film-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = cutlass_model::TimeRange::new(0, 6_000_000);
        draft.tracks[0].segments[0].source_range =
            Some(cutlass_model::TimeRange::new(0, 6_000_000));
        draft.duration = Micros(6_000_000);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        shell.start_thumbnails(move || {
            let _ = sender.send(());
        });
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let tiles = |shell: &mut Shell, view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            // Textured tiles in the lanes, below the preview.
            output
                .shapes
                .iter()
                .filter(|clipped| match &clipped.shape {
                    egui::Shape::Rect(rect) => {
                        rect.fill_texture_id() != egui::TextureId::default()
                            && rect.rect.center().y > 450.0
                    }
                    _ => false,
                })
                .count()
        };
        // The first frame asks for the tiles; none are stored yet.
        assert_eq!(tiles(&mut shell, &mut view), 0);
        receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a thumbnail");
        while receiver
            .recv_timeout(std::time::Duration::from_millis(500))
            .is_ok()
        {}
        assert!(tiles(&mut shell, &mut view) >= 3);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn an_audio_clip_paints_its_waveform_once_the_peaks_arrive() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let sound = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !sound.is_file() {
            eprintln!("skipped: {} is missing", sound.display());
            return;
        }
        let temp = std::env::temp_dir().join(format!("cutlass-wave-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.duration = Micros(20_000_000);
        draft.materials.audios.push(cutlass_model::MaterialAudio {
            id: cutlass_model::MaterialId::new("song"),
            path: sound.display().to_string(),
            duration: Micros(20_000_000),
            ..cutlass_model::MaterialAudio::default()
        });
        let mut segment = cutlass_model::Segment::new(
            "song",
            "song",
            cutlass_model::TimeRange::new(0, 20_000_000),
        );
        segment.source_range = Some(cutlass_model::TimeRange::new(0, 20_000_000));
        let mut track = cutlass_model::Track::main("audio");
        track.kind = cutlass_model::TrackType::Audio;
        track.segments.push(segment);
        draft.tracks.push(track);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        shell.start_waveforms(move || {
            let _ = sender.send(());
        });
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let wave = tokens::CLIP_AUDIO_BAR;
        let columns = |shell: &mut Shell, view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::Mesh(mesh)
                        if mesh.vertices.first().is_some_and(|v| v.color == wave) =>
                    {
                        Some(mesh.vertices.len() / 4)
                    }
                    _ => None,
                })
                .sum::<usize>()
        };
        assert_eq!(columns(&mut shell, &mut view), 0);
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("peaks");
        while receiver
            .recv_timeout(std::time::Duration::from_millis(500))
            .is_ok()
        {}
        // One bar every two points.
        assert!(columns(&mut shell, &mut view) > 50);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_video_clip_paints_its_sound_in_a_band_along_the_bottom() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.mp4");
        if !movie.is_file() {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        }
        let temp = std::env::temp_dir().join(format!("cutlass-band-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = cutlass_model::TimeRange::new(0, 20_000_000);
        draft.tracks[0].segments[0].source_range =
            Some(cutlass_model::TimeRange::new(0, 20_000_000));
        draft.duration = Micros(20_000_000);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        shell.start_waveforms(move || {
            let _ = sender.send(());
        });
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let wave = tokens::CLIP_VIDEO_BAR;
        let frame = |shell: &mut Shell, view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output.shapes
        };
        let _ = frame(&mut shell, &mut view);
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("peaks");
        while receiver
            .recv_timeout(std::time::Duration::from_millis(500))
            .is_ok()
        {}
        let shapes = frame(&mut shell, &mut view);
        let clip = shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) if rect.fill == tokens::CLIP_VIDEO => Some(rect.rect),
                _ => None,
            })
            .expect("the clip");
        // The band is the bottom 30% of the body under the header.
        let body = clip.height() - tokens::CLIP_HEADER;
        let band_top = clip.bottom() - body * tokens::WAVE_RATIO;
        let bars: Vec<Pos2> = shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Mesh(mesh)
                    if mesh.vertices.first().is_some_and(|v| v.color == wave) =>
                {
                    Some(
                        mesh.vertices
                            .iter()
                            .filter(|v| v.color != tokens::CLIP_VIDEO_LINE)
                            .map(|v| v.pos)
                            .collect::<Vec<_>>(),
                    )
                }
                _ => None,
            })
            .flatten()
            .collect();
        assert!(bars.len() / 4 > 100, "{} bars", bars.len() / 4);
        assert!(
            bars.iter()
                .all(|pos| pos.y >= band_top && pos.y <= clip.bottom())
        );
        let floor = bars.iter().map(|pos| pos.y).fold(f32::MIN, f32::max);
        assert_eq!(floor, clip.bottom() - 1.0);
        // Bars are a point wide, one every two points from the clip's edge.
        for bar in bars.chunks(4) {
            let left = bar.iter().map(|pos| pos.x).fold(f32::MAX, f32::min);
            let right = bar.iter().map(|pos| pos.x).fold(f32::MIN, f32::max);
            assert_eq!(right - left, 1.0);
            assert_eq!((left - clip.left()) % 2.0, 0.0);
        }
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_main_track_clip_heads_itself_with_name_and_length_chips() {
        let temp = std::env::temp_dir().join(format!("cutlass-chips-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
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
        draft.duration = Micros(2_500_000);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
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
        output.textures_delta.clear();
        let rects = |fill: Color32| -> Vec<Rect> {
            output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::Rect(rect) if rect.fill == fill => Some(rect.rect),
                    _ => None,
                })
                .collect()
        };
        let clip = rects(tokens::CLIP_VIDEO);
        assert_eq!(clip.len(), 1, "{clip:?}");
        let clip = clip[0];
        assert_eq!(clip.height(), tokens::MAIN_CLIP_HEIGHT);
        let chips = rects(tokens::CLIP_VIDEO_CHIP);
        assert_eq!(chips.len(), 2, "{chips:?}");
        for chip in &chips {
            assert_eq!(chip.top(), clip.top() + 1.5);
            assert_eq!(chip.bottom(), clip.top() + tokens::CLIP_HEADER - 1.5);
        }
        assert_eq!(chips[0].left(), clip.left() + 4.0);
        assert_eq!(chips[1].left(), chips[0].right() + 4.0);
        let lines = galleys(&output);
        assert!(lines.iter().any(|line| line == "clip.mp4"), "{lines:?}");
        assert!(lines.iter().any(|line| line == "00:00:02:15"), "{lines:?}");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_bar_climbs_in_decibels_from_the_floor_to_full_scale() {
        assert_eq!(wave_height(0.0), 0.0);
        assert_eq!(wave_height(1.0), 1.0);
        assert_eq!(wave_height(2.0), 1.0);
        // -6 dB of a 48 dB range.
        assert!((wave_height(0.5) - 0.875).abs() < 0.001);
        assert_eq!(wave_height(0.001), 0.0);
        assert_eq!(wave_height(f32::NAN), 0.0);
    }

    #[test]
    fn bars_in_a_bottom_corner_stop_at_the_rounded_edge() {
        let band = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 20.0));
        assert_eq!(corner_floor(band, 50.0), 20.0);
        assert_eq!(corner_floor(band, 7.0), 20.0);
        assert_eq!(corner_floor(band, 0.0), 13.0);
        assert_eq!(corner_floor(band, 100.0), 13.0);
        assert!((corner_floor(band, 3.5) - (20.0 - (7.0 - 36.75_f32.sqrt()))).abs() < 1e-4);
        let ctx = egui::Context::default();
        let wave = crate::shell::ClipWave {
            first: 0,
            levels: vec![1.0; 100],
            gains: Vec::new(),
            under_picture: false,
            muted: true,
        };
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            paint_wave(ui.painter(), &wave, 0.0, band, band, WaveColors::AUDIO);
        });
        output.textures_delta.clear();
        let mut bottoms = Vec::new();
        for clipped in &output.shapes {
            if let egui::Shape::Mesh(mesh) = &clipped.shape {
                for quad in mesh.vertices.chunks(4) {
                    let bounds = Rect::from_points(&quad.iter().map(|v| v.pos).collect::<Vec<_>>());
                    bottoms.push((bounds.left(), bounds.bottom()));
                }
            }
        }
        assert!(!bottoms.is_empty());
        for (left, bottom) in bottoms {
            assert!(
                bottom <= corner_floor(band, left + 0.5) + 1e-4,
                "bar at {left} reaches {bottom}"
            );
        }
    }

    #[test]
    fn loud_bars_wear_an_orange_cap_and_clipped_bars_a_red_tip() {
        let ctx = egui::Context::default();
        let band = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(8.0, 20.0));
        let wave = crate::shell::ClipWave {
            first: 0,
            levels: vec![0.1, 0.1, 0.8, 0.8, 1.0, 1.0],
            gains: vec![1.0; 6],
            under_picture: true,
            muted: false,
        };
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            paint_wave(ui.painter(), &wave, 0.0, band, band, WaveColors::VIDEO);
        });
        output.textures_delta.clear();
        let colors: Vec<(f32, Color32)> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Mesh(mesh) => Some(mesh.vertices.clone()),
                _ => None,
            })
            .flatten()
            .step_by(4)
            .map(|vertex| (vertex.pos.x, vertex.color))
            .filter(|(_, color)| *color != tokens::CLIP_VIDEO_LINE)
            .collect();
        let at = |x: f32| -> Vec<Color32> {
            colors
                .iter()
                .filter(|(left, _)| *left == x)
                .map(|(_, color)| *color)
                .collect()
        };
        assert_eq!(at(0.0), [tokens::CLIP_VIDEO_BAR]);
        assert_eq!(at(2.0), [tokens::CLIP_VIDEO_BAR, tokens::WAVE_LOUD]);
        assert_eq!(
            at(4.0),
            [
                tokens::CLIP_VIDEO_BAR,
                tokens::WAVE_CLIPPED,
                tokens::WAVE_LOUD
            ]
        );
    }

    #[test]
    fn the_volume_line_sits_at_unity_and_climbs_with_the_gain() {
        assert_eq!(volume_height(0.0), 0.0);
        assert_eq!(volume_height(1.0), 0.45);
        assert_eq!(volume_height(10.0), 1.0);
        assert_eq!(volume_height(100.0), 1.0);
        assert!(volume_height(0.5) < 0.45 && volume_height(0.5) > 0.0);
        let ctx = egui::Context::default();
        let band = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(4.0, 22.0));
        let wave = crate::shell::ClipWave {
            first: 0,
            levels: vec![0.0; 4],
            gains: vec![1.0, 1.0, 10.0, 10.0],
            under_picture: true,
            muted: false,
        };
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            paint_wave(ui.painter(), &wave, 0.0, band, band, WaveColors::VIDEO);
        });
        output.textures_delta.clear();
        let tops: Vec<f32> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Mesh(mesh) => Some(mesh.vertices.clone()),
                _ => None,
            })
            .flatten()
            .filter(|vertex| vertex.color == tokens::CLIP_VIDEO_LINE)
            .step_by(4)
            .map(|vertex| vertex.pos.y)
            .collect();
        // Floor 21, reach 20: unity 9 points up, +20 dB at the top.
        assert_eq!(tops, [12.0, 12.0, 1.0, 1.0]);
    }

    #[test]
    fn a_muted_track_fades_its_bars_into_the_clip_and_drops_the_caps() {
        let ctx = egui::Context::default();
        let band = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(4.0, 20.0));
        let wave = crate::shell::ClipWave {
            first: 0,
            levels: vec![1.0; 4],
            gains: Vec::new(),
            under_picture: true,
            muted: true,
        };
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            paint_wave(ui.painter(), &wave, 0.0, band, band, WaveColors::VIDEO);
        });
        output.textures_delta.clear();
        let colors: Vec<Color32> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Mesh(mesh) => Some(mesh.vertices.clone()),
                _ => None,
            })
            .flatten()
            .map(|vertex| vertex.color)
            .collect();
        let faded = lerp(tokens::CLIP_VIDEO, tokens::CLIP_VIDEO_BAR, 0.2);
        assert_eq!(colors.len(), 8);
        assert!(colors.iter().all(|color| *color == faded));
    }

    #[test]
    fn lanes_take_their_heights_around_the_centered_main_track() {
        let area = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 400.0));
        let rows = lane_rects(vec![(false, false), (true, false), (false, true)], area);
        let heights: Vec<f32> = rows.iter().map(Rect::height).collect();
        assert_eq!(heights, [40.0, 90.0, 58.0]);
        assert_eq!(rows[1].center().y, 200.0);
        assert_eq!(rows[2].top() - rows[1].bottom(), 8.0);
    }

    #[test]
    fn a_notice_fades_in_holds_and_fades_out() {
        let at = |ms| notice_alpha(std::time::Duration::from_millis(ms));
        assert_eq!(at(0), 0.0);
        assert_eq!(at(100), 1.0);
        assert_eq!(at(1_900), 1.0);
        assert!((at(2_080) - 0.5).abs() < 0.01, "{}", at(2_080));
        assert_eq!(at(2_160), 0.0);
        assert_eq!(at(5_000), 0.0);
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
    fn a_utc_date_is_the_civil_day() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(86_400), "1970-01-02");
        assert_eq!(utc_date(946_684_800), "2000-01-01");
    }

    #[test]
    fn a_card_shows_its_length_as_a_clock() {
        assert_eq!(card_clock(0), "00:00");
        assert_eq!(card_clock(93_400_000), "01:33");
        assert_eq!(card_clock(1_011_000_000), "16:51");
        assert_eq!(card_clock(3_725_000_000), "1:02:05");
    }

    #[test]
    fn a_poster_fills_its_card_cut_at_the_sides_or_the_ends() {
        let card = Vec2::new(144.0, 96.0);
        let wide = cover_uv(card, 1920.0, 1080.0);
        assert!((wide.min.x - (1.0 - 1.5 / (16.0 / 9.0)) / 2.0).abs() < 1e-5);
        assert_eq!((wide.min.y, wide.max.y), (0.0, 1.0));
        let tall = cover_uv(card, 1080.0, 1920.0);
        assert_eq!((tall.min.x, tall.max.x), (0.0, 1.0));
        assert!(tall.min.y > 0.0 && (tall.min.y + tall.max.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_long_name_keeps_its_start_and_its_extension() {
        let ctx = egui::Context::default();
        let mut cut = String::new();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            cut = middle_ellipsis(
                ui.ctx(),
                "GOOGLE IS BACK with Gemini 3 (flash).mp4",
                &FontId::proportional(12.0),
                120.0,
            );
        });
        output.textures_delta.clear();
        assert!(cut.starts_with("GOOGLE"), "{cut}");
        assert!(cut.ends_with(".mp4"), "{cut}");
        assert!(cut.contains('…'), "{cut}");
        assert_eq!(
            middle_ellipsis(&ctx, "short.mp4", &FontId::proportional(12.0), 120.0),
            "short.mp4"
        );
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
        assert_eq!(cards[0].meta, "00:02");
        assert_eq!(cards[0].look, CardLook::Picture(None));
        assert!(cards[0].on_timeline);
        // A clip's own copy of that video is still the one file.
        let mut copy = draft.materials.videos[0].clone();
        copy.id = "seg:mat".into();
        draft.materials.videos.push(copy);
        draft.tracks[0].segments[0].material_id = "seg:mat".into();
        let cards = bin_cards(&draft, Shelf::Media, "");
        assert_eq!(cards.len(), 1);
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
        let tick = text_pos(&output, "00:00").expect("first tick");
        assert!(
            notice.y > tick.y,
            "notice {notice:?} overlaps tick {tick:?}"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    fn library_cards(cards: &[&str]) -> Vec<BinCard> {
        cards
            .iter()
            .map(|id| BinCard {
                id: (*id).to_owned(),
                name: (*id).to_owned(),
                meta: String::new(),
                on_timeline: false,
                look: CardLook::Swatch,
            })
            .collect()
    }

    #[test]
    fn a_click_picks_one_card_command_toggles_and_shift_takes_the_run() {
        let cards = library_cards(&["a", "b", "c", "d"]);
        let mut view = ViewState::new();
        let plain = egui::Modifiers::default();
        pick_card(&mut view, &cards, "b", plain);
        assert_eq!(view.library_picked, ["b"]);
        pick_card(&mut view, &cards, "d", egui::Modifiers::SHIFT);
        assert_eq!(view.library_picked, ["b", "c", "d"]);
        pick_card(&mut view, &cards, "c", egui::Modifiers::COMMAND);
        assert_eq!(view.library_picked, ["b", "d"]);
        pick_card(&mut view, &cards, "a", egui::Modifiers::COMMAND);
        assert_eq!(view.library_picked, ["b", "d", "a"]);
        pick_card(&mut view, &cards, "c", plain);
        assert_eq!(view.library_picked, ["c"]);
    }

    #[test]
    fn a_click_on_a_library_card_picks_it_and_a_right_click_opens_its_menu() {
        let temp = std::env::temp_dir().join(format!("cutlass-pick-{}", std::process::id()));
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
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        let card = text_pos(&output, "pattern.png").expect("library card");
        let click = |pos: Pos2, button: egui::PointerButton| {
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), event(true), event(false)]
        };
        frame(
            &mut shell,
            &mut view,
            click(card, egui::PointerButton::Primary),
        );
        frame(&mut shell, &mut view, Vec::new());
        assert_eq!(view.library_picked.len(), 1);
        assert!(!egui::DragAndDrop::has_any_payload(&ctx));
        assert!(shell.lanes().iter().all(|lane| lane.clips.is_empty()));
        let picked = view.library_picked.clone();
        let output = frame(
            &mut shell,
            &mut view,
            click(card, egui::PointerButton::Secondary),
        );
        assert_eq!(view.library_picked, picked);
        assert!(view.library_menu.is_some_and(|menu| menu.on_card));
        let row = text_pos(&output, "Delete").expect("the Delete row") + Vec2::new(4.0, 6.0);
        frame(
            &mut shell,
            &mut view,
            click(row, egui::PointerButton::Primary),
        );
        assert!(view.library_menu.is_none());
        assert_eq!(shell.notice(), "Delete is not built yet");
        assert!(shell.draft().unwrap().materials.videos.len() == 1);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_plus_on_a_hovered_card_adds_the_clip_to_the_main_lane() {
        let temp = std::env::temp_dir().join(format!("cutlass-plus-{}", std::process::id()));
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
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        let name = text_pos(&output, "pattern.png").expect("library card");
        let plus = Pos2::new(
            name.x - 2.0 + CARD_W - CARD_PLUS_INSET - CARD_PLUS_RADIUS,
            name.y - 6.0 - CARD_PLUS_INSET - CARD_PLUS_RADIUS,
        );
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(plus)]);
        let press = |pressed| egui::Event::PointerButton {
            pos: plus,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(&mut shell, &mut view, vec![press(true), press(false)]);
        let lanes = shell.lanes();
        let main = lanes.iter().find(|lane| lane.main).expect("main lane");
        assert_eq!(main.clips.len(), 1);
        assert!(view.library_picked.is_empty());
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
        assert!(painted.contains("Added"), "{painted}");
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
    fn s_splits_and_shift_command_s_extracts_but_other_s_chords_do_nothing() {
        let chord = |modifiers: egui::Modifiers| {
            let ctx = egui::Context::default();
            let mut chord = None;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    events: vec![
                        egui::Event::ModifiersChanged(modifiers),
                        egui::Event::Key {
                            key: egui::Key::S,
                            physical_key: None,
                            pressed: true,
                            repeat: false,
                            modifiers,
                        },
                    ],
                    ..Default::default()
                },
                |ui| chord = ui.input(pressed_chord),
            );
            output.textures_delta.clear();
            chord
        };
        assert_eq!(chord(egui::Modifiers::default()).as_deref(), Some("s"));
        assert_eq!(
            chord(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT).as_deref(),
            Some("primary+shift+s")
        );
        assert_eq!(chord(egui::Modifiers::COMMAND), None);
        assert_eq!(chord(egui::Modifiers::SHIFT), None);
    }

    #[test]
    fn control_y_q_w_and_command_a_build_their_chords() {
        let chord = |key: egui::Key, modifiers: egui::Modifiers| {
            let ctx = egui::Context::default();
            let mut chord = None;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    events: vec![
                        egui::Event::ModifiersChanged(modifiers),
                        egui::Event::Key {
                            key,
                            physical_key: None,
                            pressed: true,
                            repeat: false,
                            modifiers,
                        },
                    ],
                    ..Default::default()
                },
                |ui| chord = ui.input(pressed_chord),
            );
            output.textures_delta.clear();
            chord
        };
        let control = egui::Modifiers {
            ctrl: true,
            ..Default::default()
        };
        assert_eq!(chord(egui::Key::Y, control).as_deref(), Some("ctrl+y"));
        assert_eq!(
            chord(egui::Key::A, egui::Modifiers::COMMAND).as_deref(),
            Some("primary+a")
        );
        assert_eq!(
            chord(egui::Key::Q, egui::Modifiers::default()).as_deref(),
            Some("q")
        );
        assert_eq!(
            chord(egui::Key::W, egui::Modifiers::default()).as_deref(),
            Some("w")
        );
        assert_eq!(chord(egui::Key::Q, egui::Modifiers::COMMAND), None);
        assert_eq!(chord(egui::Key::Space, control), None);
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
    fn a_running_job_shows_its_stage_and_cancel_stops_it() {
        let (mut shell, temp) = open_span("job", 2_000_000);
        shell
            .start_job("Transcribe", move |progress| {
                progress.stage("Listening");
                while !progress.cancelled() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Box::new(|_: &mut Shell| Ok(()))
            })
            .unwrap();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while shell.job().is_some_and(|job| job.stage.is_empty())
            && std::time::Instant::now() < until
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        // A new area is measured on its first frame and drawn on the next.
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(painted_text(&output).contains("Transcribe: Listening"));
        let cancel = text_pos(&output, "Cancel").expect("a Cancel button");
        let press = |pressed: bool| egui::Event::PointerButton {
            pos: cancel,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        for events in [
            vec![egui::Event::PointerMoved(cancel), press(true)],
            vec![press(false)],
        ] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
        }
        assert_eq!(
            shell.job().map(|job| job.stage),
            Some("Stopping".to_owned()),
            "Cancel did not stop the job"
        );
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while shell.poll_job() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(shell.job().is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_wheel_scrolls_lanes_that_overflow_and_shift_wheel_pans_time() {
        let temp = std::env::temp_dir().join(format!("cutlass-trim-{}-tall", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "segment",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        for index in 0..20 {
            let mut track = cutlass_model::Track::main(format!("sound-{index}"));
            track.kind = cutlass_model::TrackType::Audio;
            track.segments.push(cutlass_model::Segment::new(
                format!("tone-{index}"),
                "tone",
                cutlass_model::TimeRange::new(0, 1_000_000),
            ));
            draft.tracks.push(track);
        }
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
        let over = text_pos(&output, "segment").expect("main clip");
        output.textures_delta.clear();
        shell.zoom_in().unwrap();
        shell.zoom_in().unwrap();
        let wheel = |delta: Vec2, shift: bool| egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta,
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
            modifiers: egui::Modifiers {
                shift,
                ..egui::Modifiers::default()
            },
        };
        let mut spin = |delta: Vec2, shift: bool| {
            for events in [
                vec![egui::Event::PointerMoved(over), wheel(delta, shift)],
                Vec::new(),
            ] {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: screen,
                        events,
                        ..Default::default()
                    },
                    |ui| show(&mut shell, ui, &mut view, None),
                );
                output.textures_delta.clear();
            }
            (shell.view_origin().0, view.lane_scroll)
        };
        let (before, _) = spin(Vec2::ZERO, false);
        let (origin, scroll) = spin(Vec2::new(0.0, -120.0), false);
        assert_eq!(origin, before, "the wheel moved time, not the lanes");
        assert!(scroll < 0.0, "the lanes did not scroll: {scroll}");
        let (origin, after) = spin(Vec2::new(0.0, -120.0), true);
        assert!(origin > before, "shift and the wheel did not pan time");
        assert_eq!(after, scroll, "shift and the wheel moved the lanes");
        // However far the wheel goes, the last lane stops inside the panel.
        for _ in 0..40 {
            spin(Vec2::new(0.0, -120.0), false);
        }
        let (_, deepest) = spin(Vec2::new(0.0, -120.0), false);
        let (_, still) = spin(Vec2::new(0.0, -120.0), false);
        assert_eq!(deepest, still);
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
        assert!(shell.clip_menu().is_some());
        // A clip low in the window opens its menu moved up to fit.
        let copy = text_pos(&output, "Copy").expect("the Copy row") + Vec2::new(4.0, 6.0);
        output.textures_delta.clear();
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
