//! The column of panels. egui draws the chrome. The preview is a texture of
//! the evaluated frame. This pass does not composite.
//!
//! With no project open, the window lists the saved projects and can create one.

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Pos2, Rect, RichText, Sense, Ui, Vec2,
};

use cutlass_editor::ruler_ticks;
use cutlass_model::{FrameRate, Micros, SegmentId, TrackId};
use cutlass_session::TrimEdge;

use crate::dialogs::{Dialogs, Finished, Picker};
use crate::icons;
use crate::shell::{ExportRequest, ListedProject, MovePreview, Shell, ShellError};
use crate::tokens;
use crate::ui;
use crate::widgets;

mod assistant;
mod audio;
mod cover;
mod follow;
mod grade;
mod inspector;
mod mask;
mod motion_blur;
mod reframe;
mod relight;
mod remove_bg;
mod retouch;
mod shape;
mod speed;
mod studio;

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
            Self::Text => "No title matches the search.",
            Self::Stickers => "No stickers.",
            Self::Effects => "No effect matches the search.",
            Self::Transitions => "No transitions in this project.",
            Self::Filters => "No filters in this project.",
            Self::Adjustments => "No adjustments in this project.",
            Self::Templates => "No templates in this project.",
        }
    }
}

/// The open library menu: where it opens, and whether it is for the
/// picked cards or for the space between them.
#[derive(Clone, Debug, PartialEq)]
struct LibraryMenu {
    at: Pos2,
    target: LibraryTarget,
}

/// What a library menu is for.
#[derive(Clone, Debug, PartialEq)]
enum LibraryTarget {
    /// The space between the cards.
    Empty,
    /// The picked file cards.
    Files,
    /// One folder.
    Folder(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SettingsTab {
    General,
    Assistant,
    Shortcuts,
}

/// Settings that turn on something that sends data off the machine, or
/// adds skills, held until the user allows it.
struct Approval {
    settings: Box<cutlass_settings::Settings>,
    actions: Vec<cutlass_agent::ApprovalAction>,
}

pub struct ViewState {
    /// The system's menu bar is up, so the window draws no menus of its own.
    pub system_menus: bool,
    preview: Option<egui::TextureHandle>,
    /// The clip menu asked for a file to replace the clip's media with.
    wants_replace: bool,
    /// The clip menu asked where the clip's missing file is now.
    wants_link: bool,
    export_open: bool,
    /// The project dialog's unsaved choices while it is open.
    project_dialog: Option<cutlass_editor::ProjectSettings>,
    /// The cover dialog while it is open.
    cover: Option<cover::CoverDialog>,
    /// "Do not ask again" is ticked in the replace-transition question.
    never_ask_transition: bool,
    /// The project dialog shows its Performance tab, not Details.
    project_performance: bool,
    /// The project dialog's proxy switch, once it has been flipped.
    project_proxies: Option<bool>,
    /// The project dialog's proxy resolution, once another is picked.
    project_proxy_side: Option<u32>,
    /// The third-party notices, read when the Licenses window opens.
    licenses: Option<cutlass_licenses::Notices>,
    settings_open: bool,
    settings_tab: SettingsTab,
    /// The action whose new keys the settings window waits for.
    recording: Option<&'static str>,
    /// Each server's arguments as typed, by row, while the window is open.
    server_args: std::collections::HashMap<usize, String>,
    /// A settings change waiting for the user to allow what it turns on.
    approval: Option<Approval>,
    /// The Linkage settings dialog is open, with the kinds as ticked so far.
    linkage_kinds: Option<cutlass_model::KindSwitches>,
    /// The dialog's choices; its path is `export_path`.
    export: ExportRequest,
    /// Write a MOV rather than an MP4.
    export_mov: bool,
    /// A typed bitrate in Mbps, used instead of the quality step.
    export_mbps: Option<String>,
    replace_prompt: bool,
    export_path: String,
    /// The camera button was pressed; ask where to save the still.
    ask_still: bool,
    /// Screen pixels per canvas pixel in the preview, or `None` to fit.
    preview_zoom: Option<f64>,
    /// How far a zoomed canvas is dragged from the middle of the preview.
    preview_pan: Vec2,
    /// The scale the canvas was last drawn at, fitted or not.
    preview_scale: f64,
    /// Where the canvas was last drawn.
    preview_canvas: Option<Rect>,
    /// The part of the preview the canvas was last drawn inside.
    preview_frame: Rect,
    /// Where a tracking box drag started on the canvas, in canvas pixels.
    track_drag: Option<(f64, f64)>,
    /// A move, scale or turn of a clip on the preview in flight.
    stage_drag: Option<StageDrag>,
    /// The selected clips' boxes as each recent revision of the draft
    /// placed them, newest last, so the box drawn matches the picture on
    /// the canvas and never runs ahead of it.
    stage_seen: Vec<(u64, Vec<cutlass_editor::StageBox>)>,
    /// The action-safe and title-safe frames are drawn over the canvas.
    preview_guides: bool,
    /// The preview fills the screen.
    preview_full: bool,
    /// Heights dragged for track rows, by track. Only this window keeps
    /// them; the project does not save them.
    lane_heights: std::collections::HashMap<TrackId, f32>,
    /// The track head's right-click menu, where it opened.
    track_menu: Option<TrackMenu>,
    /// A track being named, and what is typed so far.
    renaming: Option<(TrackId, String)>,
    /// The clip panel's open tab, page, and sections, and what is typed.
    inspector: inspector::InspectorState,
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
    /// The library folder shown; empty is the top.
    library_folder: String,
    library_sort: cutlass_editor::LibrarySort,
    /// The media shelf shows free stock instead of the project's files.
    library_stock: bool,
    /// Which part of the sticker shelf shows: Yours, Stickers, Shapes or
    /// GIPHY.
    sticker_group: usize,
    /// Which part of the effects shelf shows: Video effects or Body
    /// effects.
    effect_group: usize,
    /// The video effects category shown; empty is All.
    effect_category: String,
    /// The transitions category the shelf shows; empty shows them all.
    transition_category: String,
    /// The filters category the shelf shows; empty shows them all.
    filter_category: String,
    /// A folder being named, and what is typed so far.
    library_renaming: Option<(String, String)>,
    projects_root: Option<std::path::PathBuf>,
    /// The app itself asked to quit (to restart after an update), so a
    /// close is not turned back into a return to the projects.
    quitting: bool,
    listed: Vec<ListedProject>,
    listed_stamp: String,
    /// When the projects folder was last looked at.
    listed_checked: Option<std::time::Instant>,
    marquee_press: Option<Pos2>,
    marquee: Option<Rect>,
    /// How far the lanes above main are scrolled toward the top one, in
    /// points. Zero shows the lanes next to main.
    lanes_up: f32,
    /// How far the whole stack, main with it, is scrolled up to bring the
    /// lanes under main into view, in points.
    lanes_down: f32,
    /// The assistant's conversation, and whether its panel is open.
    pub assistant: crate::assistant::Assistant,
    /// The effect studio screen.
    pub studio: studio::StudioView,
}

/// Where a track head's menu opened, and for which track.
#[derive(Clone, Debug, PartialEq)]
struct TrackMenu {
    track_id: TrackId,
    /// The main track stays; it cannot be deleted.
    main: bool,
    lock: bool,
    at: Pos2,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            system_menus: false,
            preview: None,
            wants_replace: false,
            wants_link: false,
            export_open: false,
            project_dialog: None,
            cover: None,
            never_ask_transition: false,
            project_performance: false,
            project_proxies: None,
            project_proxy_side: None,
            licenses: None,
            settings_open: false,
            settings_tab: SettingsTab::General,
            recording: None,
            server_args: std::collections::HashMap::new(),
            approval: None,
            linkage_kinds: None,
            export: ExportRequest::default(),
            export_mov: false,
            export_mbps: None,
            replace_prompt: false,
            export_path: String::new(),
            ask_still: false,
            preview_zoom: None,
            preview_pan: Vec2::ZERO,
            preview_scale: 1.0,
            preview_canvas: None,
            preview_frame: Rect::NOTHING,
            track_drag: None,
            stage_drag: None,
            stage_seen: Vec::new(),
            preview_guides: false,
            preview_full: false,
            lane_heights: std::collections::HashMap::new(),
            track_menu: None,
            renaming: None,
            inspector: inspector::InspectorState::default(),
            background_buffer: String::new(),
            name_buffer: String::new(),
            dialogs: Dialogs::new(),
            library_query: String::new(),
            library_shelf: Shelf::Media,
            library_picked: Vec::new(),
            library_anchor: None,
            library_menu: None,
            library_folder: String::new(),
            library_sort: cutlass_editor::LibrarySort::Name,
            library_stock: false,
            sticker_group: STICKERS,
            effect_group: VIDEO_EFFECTS,
            effect_category: String::new(),
            transition_category: String::new(),
            filter_category: String::new(),
            library_renaming: None,
            projects_root: None,
            quitting: false,
            listed: Vec::new(),
            listed_stamp: String::new(),
            listed_checked: None,
            marquee_press: None,
            marquee: None,
            lanes_up: 0.0,
            lanes_down: 0.0,
            assistant: crate::assistant::Assistant::default(),
            studio: studio::StudioView::default(),
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
    if std::mem::take(&mut view.inspector.wants_lut) {
        view.dialogs.lut(parent);
    }
    // Replace clip and Link to media are asked for from a menu, whose own
    // window is closing: the dialog waits until that window is gone.
    if view.wants_replace || view.wants_link {
        if crate::menu::settled(ui.ctx()) {
            if std::mem::take(&mut view.wants_replace) {
                view.dialogs.replace(parent);
            }
            if std::mem::take(&mut view.wants_link) {
                view.dialogs.link(parent);
            }
        } else {
            ui.ctx().request_repaint();
        }
    }
    style(ui);
    shell.hold_edits(ui.input(|input| input.pointer.any_down()));
    shell.chrome = crate::shell::Chrome::default();
    if shell.poll_job() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));
    }
    if shell.poll_proxies() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));
    }
    if shell.poll_update_check() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(200));
    }
    if shell.serve_forwarded() {
        ui.ctx().request_repaint();
    }
    // The studio folder is looked at whichever screen is up, so a recipe
    // saved or edited by hand shows on the shelves at once.
    if shell.poll_studio() {
        ui.ctx().request_repaint();
    }
    // With a recipe in the making the folder is looked at every second;
    // otherwise only when something else draws a frame.
    if view.studio.open || shell.studio_in_use() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_secs(1));
    }
    if std::mem::take(&mut view.studio.wants_clip) {
        view.dialogs.studio_clip(parent);
    }
    show_imported(shell, view);
    if view.assistant.poll(shell) {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(50));
    }
    if std::mem::take(&mut view.ask_still) {
        let name = shell.project_name();
        view.dialogs.save_still(parent, &name);
    }
    let full = ui.max_rect();
    ui::Surface::gap().paint(ui.painter(), full);
    if view.preview_full && shell.draft().is_none() {
        set_full_screen(ui, view, false);
    }
    if view.studio.open {
        studio::show(shell, ui, &mut view.studio, full);
        #[cfg(not(target_os = "macos"))]
        window_buttons(
            shell,
            ui,
            Rect::from_min_size(full.min, Vec2::new(full.width(), tokens::HEAD_HEIGHT)),
        );
    } else if view.preview_full {
        if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            set_full_screen(ui, view, false);
        }
        panel(ui, full);
        preview_panel(shell, ui, view, full);
        job_strip(shell, ui, full);
        shortcuts(shell, ui, view);
        if shell.playing() || shell.preview_pending() {
            ui.ctx().request_repaint();
        }
    } else if shell.draft().is_none() {
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
        } else if !shell.library_export().is_empty() {
            // The library's files were for that one dialog; the next
            // export is of the timeline again.
            shell.clear_library_export();
        }
        if shell.asked_transition().is_some() {
            transition_prompt(shell, ui, view);
        }
        if view.project_dialog.is_some() {
            project_dialog(shell, ui, view);
        }
        if cover::take_open_request(ui.ctx()) && view.cover.is_none() {
            view.cover = Some(cover::CoverDialog::open(shell));
        }
        if view.cover.is_some() {
            let mut pick = false;
            cover::dialog(shell, ui, &mut view.cover, &mut pick);
            if pick {
                view.dialogs.cover(parent);
            }
        }
        job_strip(shell, ui, full);
        shortcuts(shell, ui, view);
        if shell.playing() || shell.preview_pending() || !shell.meter_resting() {
            ui.ctx().request_repaint();
        }
    }
    if let Some(finished) = view.dialogs.poll(ui.ctx()) {
        apply_dialog(shell, view, finished);
    }
    if view.licenses.is_some() {
        licenses_window(ui, view);
    }
    settings_keys(shell, ui, view);
    if view.settings_open {
        settings_window(shell, ui, view, parent);
    }
    if view.approval.is_some() {
        approval_dialog(shell, ui, view);
    }
    if view.linkage_kinds.is_some() {
        linkage_dialog(shell, ui, view);
    }
    if shell.wants_speech_model() {
        speech_model_dialog(shell, ui);
    }
    if shell.update_offer().is_some() {
        update_dialog(shell, ui);
    }
    restart_for_update(shell, ui, view);
    dropped_files(shell, ui);
}

/// Files dragged over the window are named on a scrim; dropped, a project
/// folder opens and files import into the open project.
fn dropped_files(shell: &mut Shell, ui: &Ui) {
    let (hovered, dropped) = ui.input(|input| {
        (
            input.raw.hovered_files.len(),
            input
                .raw
                .dropped_files
                .iter()
                .map(|file| file.path().to_path_buf())
                .filter(|path| !path.as_os_str().is_empty())
                .collect::<Vec<_>>(),
        )
    });
    if hovered > 0 {
        let rect = ui.ctx().content_rect();
        let painter = ui.ctx().layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::unique("drop-scrim"),
        ));
        ui::Surface::scrim().paint(&painter, rect);
        let words = if shell.project_folder().is_some() {
            "Drop to import"
        } else {
            "Drop a project folder to open it"
        };
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            words,
            FontId::proportional(18.0),
            tokens::TEXT,
        );
    }
    if !dropped.is_empty()
        && let Err(error) = shell.drop_files(&dropped)
    {
        shell.notice_error(&error);
    }
}

/// The running job above the foot of the window: what it is, where it has
/// got to, and Cancel.
fn job_strip(shell: &mut Shell, ui: &Ui, full: Rect) {
    const WIDTH: f32 = 340.0;
    let Some(job) = shell.job() else {
        exported_strip(shell, ui, full, WIDTH);
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
                        if ui.add(ui::Button::new("Cancel")).clicked() {
                            shell.cancel_job();
                        }
                    });
                });
                let bar = egui::ProgressBar::new(job.fraction.unwrap_or(0.0));
                ui.add(match (job.fraction, job.left) {
                    (Some(_), Some(left)) => bar.text(format!(
                        "{:.0}%, about {} left",
                        job.fraction.unwrap_or(0.0) * 100.0,
                        spoken_duration(left)
                    )),
                    (Some(_), None) => bar.show_percentage(),
                    (None, _) => bar.animate(true),
                });
            });
        });
}

/// A wait as people say it: seconds under two minutes, then minutes.
fn spoken_duration(left: std::time::Duration) -> String {
    let seconds = left.as_secs().max(1);
    if seconds < 120 {
        format!("{seconds} s")
    } else {
        format!("{} min", seconds.div_ceil(60))
    }
}

/// The movie the last export wrote, with a way to its folder.
fn exported_strip(shell: &mut Shell, ui: &Ui, full: Rect, width: f32) {
    let Some(path) = shell.exported().map(std::path::Path::to_path_buf) else {
        return;
    };
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let top_left = Pos2::new(full.center().x - width / 2.0, full.bottom() - 72.0);
    egui::Area::new(ui.scope_id().with("exported"))
        .order(egui::Order::Foreground)
        .fixed_pos(top_left)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(width);
                ui.horizontal(|ui| {
                    ui.label(format!("Exported {name}"));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.add(ui::Button::new("Close")).clicked() {
                            shell.forget_export();
                        }
                        if ui
                            .add(ui::Button::new(crate::shell::REVEAL_LABEL))
                            .clicked()
                        {
                            if let Err(error) = crate::shell::reveal(&path) {
                                shell.notice_error(&error);
                            }
                            shell.forget_export();
                        }
                    });
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
    ui::Surface::bar().paint(ui.painter(), bar);
    shell.chrome.project_bar = true;
    shell.chrome.projects = true;
    #[cfg(not(target_os = "macos"))]
    window_buttons(shell, ui, bar);
    let menu = Rect::from_min_size(
        bar.min + Vec2::new(78.0, 0.0),
        Vec2::new(120.0, bar.height()),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(menu), |ui| {
        ui.horizontal_centered(|ui| {
            ui.menu_button(RichText::new("File").color(tokens::FAINT), |ui| {
                file_menu(shell, ui, view, parent);
            });
            ui.menu_button(RichText::new("Help").color(tokens::FAINT), |ui| {
                help_menu(shell, ui, view, parent);
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
    ui::Scroll::vertical("projects")
        .auto_shrink([false, false])
        .show(&mut body_ui, |ui| {
            if new_project_button(ui) {
                open_created(shell, &root);
            }
            ui.add_space(20.0);
            let recent = shell.recent_projects(&root);
            if !recent.is_empty() {
                ui.label(
                    RichText::new("Recent, elsewhere")
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::MUTED),
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::new(16.0, 16.0);
                    for project in &recent {
                        if project_card(ui, shell, project)
                            && let Err(error) = shell.open_folder(&project.path)
                        {
                            shell.notice_error(&error);
                        }
                    }
                });
                ui.add_space(20.0);
            }
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
                    if project_card(ui, shell, project)
                        && let Err(error) = shell.open_folder(&project.path)
                    {
                        shell.notice_error(&error);
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
                shell.notice_error(&error);
            }
        }
        Err(error) => shell.notice_error(&error),
    }
}

fn new_project_button(ui: &mut Ui) -> bool {
    ui.add(
        ui::Button::primary("New project")
            .text_size(14.0)
            .min_size(Vec2::new(148.0, 36.0)),
    )
    .clicked()
}

/// A project at home: its first frame over its name, length and date. A
/// project with an empty timeline shows a quiet field there instead.
fn project_card(ui: &mut Ui, shell: &mut Shell, project: &ListedProject) -> bool {
    const WIDTH: f32 = 220.0;
    const PICTURE: f32 = 124.0;
    let (rect, response) = ui.allocate_exact_size(Vec2::new(WIDTH, PICTURE + 60.0), Sense::click());
    let fill = if response.hovered() {
        tokens::ACTIVE
    } else {
        tokens::PROJECT_BAR
    };
    ui.painter().rect_filled(rect, CornerRadius::same(12), fill);
    let picture = Rect::from_min_size(rect.min, Vec2::new(WIDTH, PICTURE));
    let top = CornerRadius {
        nw: 12,
        ne: 12,
        sw: 0,
        se: 0,
    };
    match cover::card_picture(shell, ui.ctx(), &project.path) {
        Some(texture) => {
            let [w, h] = texture.size();
            ui.painter().add(
                egui::epaint::RectShape::filled(picture, top, Color32::WHITE)
                    .with_texture(texture.id(), cover_uv(picture.size(), w as f32, h as f32)),
            );
        }
        None => {
            ui.painter().rect_filled(picture, top, tokens::LANE);
            let icon = Rect::from_center_size(picture.center(), Vec2::splat(22.0));
            let _ = icons::paint(ui.painter(), icon, "media", tokens::FAINT);
        }
    }
    let text = picture.left_bottom() + Vec2::new(14.0, 12.0);
    ui.painter().text(
        text,
        Align2::LEFT_TOP,
        &project.name,
        FontId::proportional(14.0),
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
        text + Vec2::new(0.0, 22.0),
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

/// The window's close button with a project open saves it and goes back
/// to the projects, as leaving the editor does, instead of quitting. At the
/// projects the button closes the app. Quit from the menu or the Dock ends
/// the app without asking the window, so it is never held here.
pub fn close_to_projects(shell: &mut Shell, view: &mut ViewState, ctx: &egui::Context) {
    if view.quitting
        || shell.draft().is_none()
        || !ctx.input(|input| input.viewport().close_requested())
    {
        return;
    }
    ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
    cutlass_logger::info!("window closed with a project open: back to the projects");
    return_to_projects(shell, view);
}

fn return_to_projects(shell: &mut Shell, view: &mut ViewState) {
    if let Err(error) = shell.close_project() {
        shell.notice_error(&error);
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

/// The side panels the settings can hide, by their id in the settings file.
const PANEL_LIBRARY: &str = "library";
const PANEL_INSPECTOR: &str = "inspector";

fn sizes(shell: &Shell, window: Vec2) -> Sizes {
    let head = tokens::HEAD_HEIGHT;
    let timeline = shell.timeline_height().clamp(
        tokens::TIMELINE_MIN,
        (window.y - head - tokens::SPLIT - tokens::PREVIEW_MIN_HEIGHT).max(tokens::TIMELINE_MIN),
    );
    let row = (window.y - head - tokens::SPLIT - timeline).max(0.0);
    let side = |id: &str, width: f32| {
        if shell.settings().panel_open(id) {
            width
        } else {
            0.0
        }
    };
    let library = side(PANEL_LIBRARY, shell.library_width());
    let inspector = side(PANEL_INSPECTOR, shell.inspector_width());
    let gaps = [library, inspector]
        .iter()
        .filter(|width| **width > 0.0)
        .count() as f32;
    let preview = (window.x - library - inspector - tokens::SPLIT * gaps).max(0.0);
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
    // egui draws dropdowns, tooltips and the project bar's menus in its
    // popup frame, which reads these: the same panel as a right-click menu.
    style.visuals.window_fill = tokens::MENU;
    style.visuals.window_stroke = egui::Stroke::NONE;
    style.visuals.menu_corner_radius = CornerRadius::same(tokens::MENU_RADIUS);
    style.visuals.popup_shadow = tokens::MENU_SHADOW;
    style.spacing.menu_margin = egui::Margin::same(tokens::MENU_PAD);
    style.visuals.extreme_bg_color = tokens::GAP;
    style.visuals.faint_bg_color = tokens::PROJECT_BAR;
    // Every text field's caret and selection, ui::Field's included.
    style.visuals.selection.bg_fill = tokens::TEXT_SELECTION;
    style.visuals.selection.stroke = egui::Stroke::new(1.0, tokens::TEXT);
    style.visuals.text_cursor.stroke = egui::Stroke::new(1.5, tokens::TEXT);
    // egui's own bars (popups, menus) keep one width too: no grow on hover.
    style.spacing.scroll.bar_width = tokens::SCROLL_BAR;
    style.spacing.scroll.floating_width = tokens::SCROLL_BAR;
    // No outline on any widget, in any state: buttons and fields show by
    // their fill alone. egui's own buttons match ui::Button's secondary kind:
    // the same fills and the same corners.
    let radius = CornerRadius::same(tokens::ICON_RADIUS);
    let widgets = &mut style.visuals.widgets;
    for state in [&mut widgets.noninteractive, &mut widgets.open] {
        state.bg_stroke = egui::Stroke::NONE;
    }
    for (state, fill) in [
        (&mut widgets.inactive, tokens::RAISED),
        (&mut widgets.hovered, tokens::ACTIVE),
        (&mut widgets.active, tokens::PRESSED),
    ] {
        state.weak_bg_fill = fill;
        state.bg_stroke = egui::Stroke::NONE;
        state.corner_radius = radius;
    }
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
    ui::Surface::bar().paint(ui.painter(), bar);
    // A line along its foot, end to end, so the inspector's header, the
    // same grey, does not run into it.
    ui.painter().hline(
        bar.x_range(),
        bar.bottom() - 0.5,
        egui::Stroke::new(1.0, inspector::TAB_LINE),
    );
    shell.chrome.project_bar = true;
    #[cfg(not(target_os = "macos"))]
    window_buttons(shell, ui, bar);
    // On macOS the menus live in the system's menu bar; the window's own
    // are the fallback when that could not be built, and elsewhere.
    if !view.system_menus {
        menus(shell, ui, view, bar, parent);
    }
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
    let toggle = Rect::from_min_size(
        export.left_center() - Vec2::new(12.0 + 84.0, 13.0),
        Vec2::new(84.0, 26.0),
    );
    let open = view.assistant.open;
    let response = ui
        .scope_builder(egui::UiBuilder::new().max_rect(toggle), |ui| {
            ui.add(
                ui::Button::ghost("Assistant")
                    .text_size(12.0)
                    .on(open)
                    .min_size(toggle.size()),
            )
        })
        .inner
        .on_hover_text("Ask Cutlass to edit (⌘J)");
    if response.clicked() {
        view.assistant.open = !open;
    }
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
        Vec2::new(400.0, bar.height()),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(menu), |ui| {
        ui.horizontal_centered(|ui| {
            if ui.add(ui::Button::ghost("Projects")).clicked() {
                return_to_projects(shell, view);
            }
            ui.menu_button(RichText::new("File").color(tokens::FAINT), |ui| {
                file_menu(shell, ui, view, parent);
            });
            ui.menu_button(RichText::new("Edit").color(tokens::FAINT), |ui| {
                edit_menu(shell, ui);
                menu_items(shell, ui, view, parent, "Edit", &["undo", "redo"]);
            });
            for title in ["Clip", "Playback"] {
                ui.menu_button(RichText::new(title).color(tokens::FAINT), |ui| {
                    menu_items(shell, ui, view, parent, title, &[]);
                });
            }
            ui.menu_button(RichText::new("Help").color(tokens::FAINT), |ui| {
                help_menu(shell, ui, view, parent);
            });
        });
    });
}

fn file_menu(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, parent: Option<&eframe::Frame>) {
    if ui.button("Open…").clicked() {
        ui.close();
        view.dialogs.open_project(parent);
    }
    if menu_row(ui, "Settings…", shell.settings().keys_for("settings"), true).clicked() {
        ui.close();
        view.settings_open = true;
    }
    if ui.button("Effect studio").clicked() {
        ui.close();
        view.studio.open = true;
    }
    if shell.draft().is_none() {
        return;
    }
    if ui.button("Save").clicked() {
        ui.close();
        if let Err(error) = shell.save() {
            shell.notice_error(&error);
        }
    }
    if ui.button("Import…").clicked() {
        ui.close();
        view.dialogs.import(parent);
    }
}

/// One row of the window's own menus: its label, then the keys bound to it
/// drawn as keys at the right, as the clip menu draws them. Empty `chord`
/// draws no keys.
fn menu_row(ui: &mut Ui, label: &str, chord: &str, enabled: bool) -> egui::Response {
    const GAP: f32 = 24.0;
    let shortcut = if chord.is_empty() {
        String::new()
    } else {
        crate::menu::shortcut_for(chord)
    };
    let keys = if shortcut.is_empty() {
        0.0
    } else {
        GAP + crate::menu::keys_width(ui.ctx(), &shortcut)
    };
    let padding = ui.spacing().button_padding.x;
    let text = ui
        .painter()
        .layout_no_wrap(
            label.to_owned(),
            egui::TextStyle::Button.resolve(ui.style()),
            tokens::TEXT,
        )
        .size()
        .x;
    let button = egui::Button::new(label).min_size(Vec2::new(text + 2.0 * padding + keys, 0.0));
    let response = ui.add_enabled(enabled, button);
    if !shortcut.is_empty() {
        let color = if enabled {
            tokens::MENU_KEY
        } else {
            tokens::MENU_KEY_DISABLED
        };
        crate::menu::paint_keys(
            ui.ctx(),
            ui.painter(),
            Pos2::new(response.rect.right() - padding, response.rect.center().y),
            &shortcut,
            color,
        );
    }
    response
}

/// The items of the menu bar's `title` menu, but those in `skip`, each
/// running its command. A separator stays between groups.
fn menu_items(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    parent: Option<&eframe::Frame>,
    title: &str,
    skip: &[&str],
) {
    let Some(menu) = crate::menubar::MENUS
        .iter()
        .find(|menu| menu.title == title)
    else {
        return;
    };
    let mut chosen = None;
    let mut drawn = false;
    for entry in menu.items {
        match entry {
            Some(item) if skip.contains(&item.id) => {}
            Some(item) => {
                drawn = true;
                let chord = shell.settings().keys_for(item.id).to_owned();
                if menu_row(ui, item.label, &chord, true).clicked() {
                    ui.close();
                    chosen = Some(item.id);
                }
            }
            None if drawn => {
                ui.separator();
                drawn = false;
            }
            None => {}
        }
    }
    if let Some(id) = chosen {
        run_menu(shell, view, parent, id);
    }
}

/// Run the menu bar item `id`: a window command, or a shortcut action.
pub fn run_menu(shell: &mut Shell, view: &mut ViewState, parent: Option<&eframe::Frame>, id: &str) {
    use crate::menubar::{
        CHECK_UPDATES, EXPORT, EXPORT_LOG, IMPORT, LICENSES, OPEN, PROJECTS, SAVE, SETTINGS, STUDIO,
    };

    let open = shell.draft().is_some();
    let done = match id {
        OPEN => {
            view.dialogs.open_project(parent);
            Ok(())
        }
        SETTINGS => {
            view.settings_open = true;
            Ok(())
        }
        STUDIO => {
            view.studio.open = true;
            Ok(())
        }
        LICENSES => {
            view.licenses = Some(cutlass_licenses::notices());
            Ok(())
        }
        EXPORT_LOG => {
            view.dialogs.save_log(parent);
            Ok(())
        }
        CHECK_UPDATES => {
            shell.check_for_updates();
            Ok(())
        }
        SAVE | IMPORT | EXPORT | PROJECTS if !open => Err(ShellError::NoProject),
        SAVE => shell.save(),
        IMPORT => {
            view.dialogs.import(parent);
            Ok(())
        }
        EXPORT => {
            view.export_open = true;
            Ok(())
        }
        PROJECTS => {
            return_to_projects(shell, view);
            Ok(())
        }
        action => shell.run_action(action),
    };
    if let Err(error) = done {
        shell.notice_error(&error);
    }
}

fn help_menu(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, parent: Option<&eframe::Frame>) {
    if ui.button("Check for updates…").clicked() {
        ui.close();
        shell.check_for_updates();
    }
    if ui.button("Licenses").clicked() {
        ui.close();
        view.licenses = Some(cutlass_licenses::notices());
    }
    if ui.button("Export log…").clicked() {
        ui.close();
        view.dialogs.save_log(parent);
    }
}

/// Cutlass's license and every crate it is built with. A crate opens to
/// show its license files.
fn licenses_window(ui: &mut Ui, view: &mut ViewState) {
    let Some(notices) = view.licenses.as_ref() else {
        return;
    };
    let ((), closed) = widgets::dialog(ui, "Licenses", |ui| {
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.set_max_width(372.0);
                ui.label(
                    RichText::new(format!(
                        "Cutlass is licensed {}. It is built with these {} crates.",
                        cutlass_licenses::CUTLASS_LICENSE,
                        notices.crates.len()
                    ))
                    .color(tokens::MUTED),
                );
                ui.add_space(8.0);
                sticker_credits(ui);
                ui.add_space(8.0);
                for krate in &notices.crates {
                    egui::CollapsingHeader::new(format!("{} {}", krate.name, krate.version))
                        .id_salt(("license", &krate.name, &krate.version))
                        .show(ui, |ui| {
                            ui.label(RichText::new(&krate.license).color(tokens::MUTED));
                            if !krate.repository.is_empty() {
                                ui.label(RichText::new(&krate.repository).color(tokens::FAINT));
                            }
                            for text in notices.texts_of(krate) {
                                ui.add_space(4.0);
                                ui.label(RichText::new(text).monospace().size(10.0));
                            }
                        });
                }
            });
        });
    });
    if closed || ui.input(|input| input.key_pressed(egui::Key::Escape)) {
        view.licenses = None;
    }
}

/// The open sticker sets the app ships: each set, who made it, its license
/// and its home, and the license text under it.
fn sticker_credits(ui: &mut Ui) {
    ui.label(RichText::new("Stickers").strong());
    for credit in cutlass_stickers::CREDITS {
        ui.label(format!(
            "{} by {}, licensed {}.",
            credit.set, credit.author, credit.license
        ));
        ui.label(RichText::new(credit.link).color(tokens::FAINT));
        egui::CollapsingHeader::new(format!("{} license", credit.set))
            .id_salt(("sticker license", credit.set))
            .show(ui, |ui| {
                ui.label(RichText::new(credit.text).monospace().size(10.0));
            });
    }
}

/// The settings keys on every screen: the keys bound to Settings open the
/// window, and while a shortcut is being changed the next chord is its new
/// keys. Escape stops the change.
fn settings_keys(shell: &mut Shell, ui: &Ui, view: &mut ViewState) {
    if ui.ctx().text_edit_focused() {
        return;
    }
    if let Some(action) = view.recording {
        if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            view.recording = None;
            return;
        }
        let Some(chord) = ui.input(pressed_chord) else {
            return;
        };
        view.recording = None;
        if let Some(displaced) = shell.set_shortcut(action, &chord) {
            shell.set_notice(format!(
                "{} took {} from {}",
                cutlass_settings::action_label(action),
                cutlass_settings::keys_label(&chord, MAC),
                cutlass_settings::action_label(&displaced)
            ));
        }
        return;
    }
    if !view.settings_open
        && view.licenses.is_none()
        && let Some(chord) = ui.input(pressed_chord)
        && shell.settings().action_for(&chord) == Some("settings")
    {
        view.settings_open = true;
    }
}

const MAC: bool = cfg!(target_os = "macos");

/// Playback, the side panels, the platform codecs and the shortcuts. A
/// change saves at once.
fn settings_window(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    parent: Option<&eframe::Frame>,
) {
    let mut settings = shell.settings().clone();
    let mut add_folder = false;
    let mut reset = false;
    let mut record = None;
    let mut clear = None;
    let ((), closed) = widgets::dialog(ui, "Settings", |ui| {
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.set_max_width(372.0);
                ui::Segmented::show(ui, |ui| {
                    for (tab, name) in [
                        (SettingsTab::General, "General"),
                        (SettingsTab::Assistant, "Assistant"),
                        (SettingsTab::Shortcuts, "Shortcuts"),
                    ] {
                        if ui
                            .add(ui::Choice::new(view.settings_tab == tab, name))
                            .clicked()
                        {
                            view.settings_tab = tab;
                        }
                    }
                });
                match view.settings_tab {
                    SettingsTab::General => {
                        settings_heading(ui, "Playback");
                        ui.add(ui::Check::new(&mut settings.playback.loop_playback, "Loop at the end"));
                        ui.add(ui::Check::new(
                            &mut settings.playback.scrub_audio,
                            "Play sound while the playhead is dragged",
                        ));
                        // One row: off, or the resolution a new project's
                        // proxies are drawn at.
                        choices_row(ui, "Proxies", |ui| {
                            let playback = &mut settings.playback;
                            let hint = "A new project makes a smaller copy of each clip larger than this for the viewer. Each project keeps its own choice in its settings.";
                            if ui
                                .add(ui::Choice::new(!playback.proxies, "Off"))
                                .on_hover_text(hint)
                                .clicked()
                            {
                                playback.proxies = false;
                            }
                            for value in cutlass_export::PROXY_SIDES {
                                let chosen = playback.proxies
                                    && cutlass_export::proxy_side(playback.proxy_side) == value;
                                if ui
                                    .add(ui::Choice::new(chosen, format!("{value}p")))
                                    .on_hover_text(hint)
                                    .clicked()
                                {
                                    playback.proxies = true;
                                    playback.proxy_side = value;
                                }
                            }
                        });
                        choices_row(ui, "Preview", |ui| {
                            let quality = &mut settings.playback.preview_quality;
                            for (value, name, _) in PREVIEW_QUALITIES {
                                if ui.add(ui::Choice::new(*quality == value, name)).clicked() {
                                    *quality = value;
                                }
                            }
                        });
                        choices_row(ui, "A seek", |ui| {
                            let seek = &mut settings.playback.seek_during_playback;
                            for (value, name) in [
                                (
                                    cutlass_settings::SeekDuringPlayback::KeepPlaying,
                                    "keeps playing",
                                ),
                                (cutlass_settings::SeekDuringPlayback::Pause, "pauses"),
                            ] {
                                if ui.add(ui::Choice::new(*seek == value, name)).clicked() {
                                    *seek = value;
                                }
                            }
                        });
                        settings_heading(ui, "Panels");
                        for (id, name) in
                            [(PANEL_LIBRARY, "Library"), (PANEL_INSPECTOR, "Inspector")]
                        {
                            let mut open = settings.panel_open(id);
                            if ui.add(ui::Check::new(&mut open, name)).changed() {
                                settings.set_panel_open(id, open);
                            }
                        }
                        settings_heading(ui, "Import");
                        let mut copy = !settings.import.link;
                        if ui
                            .add(ui::Check::new(&mut copy, "Copy imported files into the project"))
                            .changed()
                        {
                            settings.import.link = !copy;
                        }
                        ui.label(
                            RichText::new("Off, the project points at each file where it is.")
                                .color(tokens::FAINT),
                        );
                        settings_heading(ui, "Hardware codecs");
                        ui.add(ui::Check::new(
                            &mut settings.hardware.decode,
                            "Decode with the system codecs",
                        ));
                        ui.add(ui::Check::new(
                            &mut settings.hardware.encode,
                            "Encode with the system codecs",
                        ));
                        ui.label(
                            RichText::new("These take effect when Cutlass opens again.")
                                .color(tokens::FAINT),
                        );
                        settings_heading(ui, "Updates");
                        ui.add(ui::Check::new(&mut settings.updates.check, "Check for updates"));
                        ui.label(
                            RichText::new(format!(
                                "Cutlass {} looks for a new release once a day. Off, it never asks.",
                                crate::shell::RUNNING_VERSION
                            ))
                            .color(tokens::FAINT),
                        );
                    }
                    SettingsTab::Assistant => {
                        assistant_settings(
                            ui,
                            &mut settings,
                            &mut view.server_args,
                            &mut add_folder,
                        );
                    }
                    SettingsTab::Shortcuts => {
                        ui.add_space(8.0);
                        reset = ui.add(ui::Button::new("Reset shortcuts")).clicked();
                        ui.add_space(4.0);
                        for (action, label) in cutlass_settings::ACTIONS {
                            ui.horizontal(|ui| {
                                let (rect, _) =
                                    ui.allocate_exact_size(Vec2::new(200.0, 20.0), Sense::hover());
                                ui.painter().text(
                                    rect.left_center(),
                                    Align2::LEFT_CENTER,
                                    *label,
                                    FontId::proportional(tokens::TYPE_SIZE),
                                    tokens::MUTED,
                                );
                                let keys = settings.keys_for(action);
                                let shown = if view.recording == Some(*action) {
                                    "Press keys…".to_owned()
                                } else if keys.is_empty() {
                                    "None".to_owned()
                                } else {
                                    cutlass_settings::keys_label(keys, MAC)
                                };
                                if ui
                                    .add(ui::Button::new(shown).min_size(Vec2::new(110.0, 20.0)))
                                    .on_hover_text("Click, then press the new keys")
                                    .clicked()
                                {
                                    record = Some(*action);
                                }
                                if !keys.is_empty() && ui.add(ui::Button::new("Clear").text_size(12.0)).clicked() {
                                    clear = Some(*action);
                                }
                            });
                        }
                    }
                }
            });
        });
    });
    if let Some(action) = clear {
        settings.bind(action, "");
    }
    if reset {
        settings.reset_shortcuts();
        view.recording = None;
    }
    if record.is_some() {
        view.recording = record;
    }
    if add_folder {
        view.dialogs.skill_folder(parent);
    }
    propose_settings(shell, view, settings);
    let escape = view.recording.is_none() && ui.input(|input| input.key_pressed(egui::Key::Escape));
    if (closed || escape) && view.approval.is_none() {
        view.settings_open = false;
        view.recording = None;
        view.server_args.clear();
    }
}

/// Write `settings`, or hold them for the approval dialog when they turn on
/// the cloud model, the online sandbox or a remote server, or add skills.
fn propose_settings(shell: &mut Shell, view: &mut ViewState, settings: cutlass_settings::Settings) {
    if view.approval.is_some() {
        return;
    }
    let actions = cutlass_agent::pending_approvals(shell.settings(), &settings);
    if actions.is_empty() {
        shell.update_settings(|current| *current = settings);
    } else {
        view.approval = Some(Approval {
            settings: Box::new(settings),
            actions,
        });
    }
}

fn approval_dialog(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let Some(approval) = view.approval.as_ref() else {
        return;
    };
    let mut answer = None;
    let ((), closed) = widgets::dialog(ui, "Allow this?", |ui| {
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.set_max_width(372.0);
                for action in &approval.actions {
                    ui.label(action.prompt());
                    ui.add_space(6.0);
                }
                ui.horizontal(|ui| {
                    if ui.add(ui::Button::new("Allow")).clicked() {
                        answer = Some(true);
                    }
                    if ui.add(ui::Button::new("Don't allow")).clicked() {
                        answer = Some(false);
                    }
                });
            });
        });
    });
    let refused = closed || ui.input(|input| input.key_pressed(egui::Key::Escape));
    match answer {
        Some(true) => {
            if let Some(approval) = view.approval.take() {
                shell.update_settings(|current| *current = *approval.settings);
            }
        }
        Some(false) => view.approval = None,
        None if refused => view.approval = None,
        None => {}
    }
}

/// A newer release is out: offer to download it and restart.
fn update_dialog(shell: &mut Shell, ui: &mut Ui) {
    let Some(offer) = shell.update_offer() else {
        return;
    };
    let title = format!("Cutlass {} is available", offer.version);
    let notes: Vec<String> = offer.notes_head(6).into_iter().map(str::to_owned).collect();
    let mut answer = None;
    let ((), closed) = widgets::dialog(ui, &title, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.set_max_width(372.0);
                ui.label(format!(
                    "You have Cutlass {}. The update downloads, replaces this copy and opens again.",
                    crate::shell::RUNNING_VERSION
                ));
                for line in &notes {
                    ui.label(RichText::new(line).color(tokens::MUTED));
                }
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.add(ui::Button::primary("Update and restart")).clicked() {
                        answer = Some(true);
                    }
                    if ui.add(ui::Button::new("Later")).clicked() {
                        answer = Some(false);
                    }
                });
            });
        });
    });
    let later = closed || ui.input(|input| input.key_pressed(egui::Key::Escape));
    match answer {
        Some(true) => {
            if let Err(error) = shell.install_update() {
                shell.notice_error(&error);
            }
        }
        Some(false) => shell.decline_update(),
        None if later => shell.decline_update(),
        None => {}
    }
}

/// An update is in place: start the new version once this one quits, and
/// close. The project is saved on the way out.
fn restart_for_update(shell: &mut Shell, ui: &Ui, view: &mut ViewState) {
    let Some(relaunch) = shell.take_restart() else {
        return;
    };
    match relaunch.spawn() {
        Ok(()) => {
            cutlass_logger::info!("update installed; restarting");
            view.quitting = true;
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        Err(error) => shell.notice_error(&format!(
            "The update is installed. Open Cutlass again to use it ({error})."
        )),
    }
}

/// A transcript or hearing found no speech model: offer to download it.
fn speech_model_dialog(shell: &mut Shell, ui: &mut Ui) {
    let mut answer = None;
    let ((), closed) = widgets::dialog(ui, "Download the speech model?", |ui| {
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.set_max_width(372.0);
                ui.label(crate::shell::SPEECH_MODEL_QUESTION);
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.add(ui::Button::primary("Download")).clicked() {
                        answer = Some(true);
                    }
                    if ui.add(ui::Button::new("Not now")).clicked() {
                        answer = Some(false);
                    }
                });
            });
        });
    });
    let declined = closed || ui.input(|input| input.key_pressed(egui::Key::Escape));
    match answer {
        Some(true) => {
            if let Err(error) = shell.download_speech_model() {
                shell.notice_error(&error);
            }
        }
        Some(false) => shell.decline_speech_model(),
        None if declined => shell.decline_speech_model(),
        None => {}
    }
}

/// The assistant's model and sandbox switches, its step cap and memory, its
/// MCP servers and its skill folders.
fn assistant_settings(
    ui: &mut Ui,
    settings: &mut cutlass_settings::Settings,
    server_args: &mut std::collections::HashMap<usize, String>,
    add_folder: &mut bool,
) {
    use cutlass_settings::McpServer;
    settings_heading(ui, "Model");
    ui.add(ui::Check::new(&mut settings.model.local, "Local model"));
    if settings.model.local {
        choice_row(ui, "Address", |ui| {
            ui::Field::new(&mut settings.model.local_url)
                .hint(cutlass_settings::DEFAULT_LOCAL_URL)
                .width(300.0)
                .show(ui);
        });
        choice_row(ui, "Model", |ui| {
            ui::Field::new(&mut settings.model.local_name)
                .hint(cutlass_settings::DEFAULT_LOCAL_NAME)
                .width(180.0)
                .show(ui);
        });
        ui.label(
            RichText::new("A chat server on this computer, such as Ollama or llama.cpp.")
                .size(11.0)
                .color(tokens::FAINT),
        );
    }
    ui.add(ui::Check::new(
        &mut settings.model.cloud,
        "Cloud model (sends messages off this computer)",
    ));
    ui.add(ui::Check::new(
        &mut settings.sandbox.local,
        "Run code on this computer",
    ));
    ui.add(ui::Check::new(
        &mut settings.sandbox.online,
        "Run code online",
    ));
    ui.add(ui::Check::new(
        &mut settings.memory,
        "Remember across conversations",
    ));
    choice_row(ui, "Steps a turn", |ui| {
        ui.add(egui::DragValue::new(&mut settings.step_cap).range(1..=1000));
    });
    settings_heading(ui, "MCP servers");
    let mut remove = None;
    for (index, server) in settings.mcp_servers.iter_mut().enumerate() {
        ui.push_id(("server", index), |ui| {
            ui.horizontal(|ui| {
                ui.add(ui::Check::new(&mut server.enabled, ""));
                ui::Field::new(&mut server.name)
                    .hint("Name")
                    .width(120.0)
                    .show(ui);
                ui::Segmented::show(ui, |ui| {
                    for (kind, name) in [(McpServer::STDIO, "Program"), (McpServer::REMOTE, "URL")]
                    {
                        if ui.add(ui::Choice::new(server.kind == kind, name)).clicked() {
                            kind.clone_into(&mut server.kind);
                        }
                    }
                });
                if ui.add(ui::Button::new("Remove").text_size(12.0)).clicked() {
                    remove = Some(index);
                }
            });
            if server.kind == McpServer::REMOTE {
                ui::Field::new(&mut server.url)
                    .hint("https://…")
                    .width(340.0)
                    .show(ui);
            } else {
                ui.horizontal(|ui| {
                    ui::Field::new(&mut server.command)
                        .hint("Program")
                        .width(140.0)
                        .show(ui);
                    let args = server_args
                        .entry(index)
                        .or_insert_with(|| server.args.join(" "));
                    if ui::Field::new(args)
                        .hint("Arguments")
                        .width(190.0)
                        .show(ui)
                        .response
                        .changed()
                    {
                        server.args = args.split_whitespace().map(str::to_owned).collect();
                    }
                });
            }
            ui.add_space(4.0);
        });
    }
    if let Some(index) = remove {
        settings.mcp_servers.remove(index);
        server_args.clear();
    }
    if ui.add(ui::Button::new("Add server")).clicked() {
        settings.mcp_servers.push(McpServer::default());
    }
    settings_heading(ui, "Skill folders");
    let mut remove = None;
    for (index, folder) in settings.skill_folders.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&folder.path).color(tokens::MUTED));
            if ui.add(ui::Button::new("Remove").text_size(12.0)).clicked() {
                remove = Some(index);
            }
        });
    }
    if let Some(index) = remove {
        settings.skill_folders.remove(index);
    }
    *add_folder = ui.add(ui::Button::new("Add folder…")).clicked();
}

fn settings_heading(ui: &mut Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(text).color(tokens::TEXT).strong());
    ui.add_space(2.0);
}

fn edit_menu(shell: &mut Shell, ui: &mut Ui) {
    let (undo, redo) = (shell.can_undo(), shell.can_redo());
    let keys = |action| shell.settings().keys_for(action).to_owned();
    let (undo_keys, redo_keys) = (keys("undo"), keys("redo"));
    if menu_row(ui, "Undo", &undo_keys, undo).clicked() {
        ui.close();
        if let Err(error) = shell.undo() {
            shell.notice_error(&error);
        }
    }
    if menu_row(ui, "Redo", &redo_keys, redo).clicked() {
        ui.close();
        if let Err(error) = shell.redo() {
            shell.notice_error(&error);
        }
    }
}

/// The library, the preview and the inspector. A side panel the settings
/// hide takes no room, and the preview grows into it.
fn regions(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, row: Rect, sizes: Sizes) {
    let mut left = row.min.x;
    if sizes.library > 0.0 {
        let library = Rect::from_min_size(row.min, Vec2::new(sizes.library, row.height()));
        panel(ui, library);
        library_panel(shell, ui, view, library);
        shell.chrome.library = true;
        let gap = Rect::from_min_size(
            Pos2::new(library.max.x, row.min.y),
            Vec2::new(tokens::SPLIT, row.height()),
        );
        drag_gap(ui, gap, |delta| shell.drag_library(delta.x, row.width()));
        left = gap.max.x;
    }
    let preview = Rect::from_min_size(
        Pos2::new(left, row.min.y),
        Vec2::new(sizes.preview, row.height()),
    );
    panel(ui, preview);
    preview_panel(shell, ui, view, preview);
    shell.chrome.preview = true;
    if sizes.inspector <= 0.0 {
        return;
    }
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
    if view.assistant.open {
        assistant::panel(shell, ui, &mut view.assistant, inspector);
    } else {
        inspector_panel(shell, ui, view, inspector);
    }
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
    // The whole rect scrolls, so the bar sits on the panel edge; the
    // buttons keep 6 points in from each side.
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    ui::Scroll::vertical("library-rail")
        .content_margin(egui::Margin::symmetric(6, 0))
        .show(&mut column, |ui| {
            ui.add_space(8.0);
            ui.spacing_mut().item_spacing.y = 4.0;
            for shelf in Shelf::ALL {
                if shelf_button(ui, shelf, view.library_shelf == shelf).clicked() {
                    view.library_shelf = shelf;
                }
            }
        });
}

/// An icon with its name under it; the open shelf sits on a tile.
fn shelf_button(ui: &mut Ui, shelf: Shelf, active: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 52.0), Sense::click());
    if active {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(10), tokens::ACTIVE);
    }
    let color = if active {
        tokens::TEXT
    } else if response.hovered() {
        tokens::MUTED
    } else {
        tokens::FAINT
    };
    let mark = Rect::from_center_size(rect.center() - Vec2::new(0.0, 8.0), Vec2::splat(20.0));
    ui.painter().text(
        Pos2::new(rect.center().x, rect.bottom() - 6.0),
        Align2::CENTER_BOTTOM,
        shelf.label(),
        FontId::proportional(10.5),
        color,
    );
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
    let shelf = view.library_shelf;
    let stock = shelf == Shelf::Media && view.library_stock;
    ui::Field::new(&mut view.library_query)
        .hint(if stock {
            "Search free stock"
        } else {
            shelf.hint()
        })
        .icon("search")
        .fill(tokens::PROJECT_BAR)
        .text_size(tokens::TYPE_SIZE + 1.0)
        .height(36.0)
        .show(&mut column);
    if shelf == Shelf::Media {
        column.add_space(8.0);
        source_switch(&mut column, view);
    }
    if shelf == Shelf::Stickers {
        column.add_space(8.0);
        if let Some(group) = segmented(
            &mut column,
            "sticker-group",
            &STICKER_GROUPS,
            view.sticker_group,
        ) {
            view.sticker_group = group;
        }
    }
    if shelf == Shelf::Effects {
        column.add_space(8.0);
        if let Some(group) = segmented(
            &mut column,
            "effect-group",
            &EFFECT_GROUPS,
            view.effect_group,
        ) {
            view.effect_group = group;
        }
        if view.effect_group == VIDEO_EFFECTS {
            column.add_space(8.0);
            category_chips(
                &mut column,
                "effect-categories",
                &cutlass_editor::effect_categories(),
                &mut view.effect_category,
            );
        }
    }
    if shelf == Shelf::Transitions {
        column.add_space(8.0);
        category_chips(
            &mut column,
            "transition-categories",
            &cutlass_editor::transition_categories(),
            &mut view.transition_category,
        );
    }
    if shelf == Shelf::Filters {
        column.add_space(8.0);
        category_chips(
            &mut column,
            "filter-categories",
            &cutlass_editor::filter_categories(),
            &mut view.filter_category,
        );
    }
    let bin = matches!(shelf, Shelf::Media | Shelf::Audio) && !stock;
    if bin {
        // A folder that is gone, as after another project opened, is the top.
        if cutlass_editor::folder_trail(shell.library_store(), &view.library_folder).is_empty() {
            view.library_folder.clear();
        }
        if !view.library_folder.is_empty() {
            column.add_space(8.0);
            path_bar(shell, &mut column, view);
        }
    }
    column.add_space(12.0);
    let grid = column.available_rect_before_wrap();
    // The space between and below the cards. Cards sit above it, so a
    // click here missed every card.
    let back = column.interact(grid, column.scope_id().with("library-back"), Sense::click());
    if back.clicked() {
        view.library_picked.clear();
        view.library_anchor = None;
        // The pen's card is lit while the pen is up, so it lets go too.
        shell.stop_pen();
    }
    if bin
        && back.secondary_clicked()
        && let Some(at) = back.interact_pointer_pos()
    {
        view.library_menu = Some(LibraryMenu {
            at,
            target: LibraryTarget::Empty,
        });
    }
    // The column is inset 12 from the panel. The cards' scroll view reaches
    // back out to the panel's edge so its bar sits there, and pads the cards
    // by the same 12.
    let scroll = grid.with_max_x(grid.max.x + 12.0);
    let mut grid_ui = column.new_child(egui::UiBuilder::new().max_rect(scroll));
    let query = view.library_query.clone();
    ui::Scroll::vertical("library-cards")
        .auto_shrink([false, false])
        .content_margin(egui::Margin {
            right: 12,
            ..egui::Margin::ZERO
        })
        .show(&mut grid_ui, |ui| {
            if stock {
                ui.add_space(8.0);
                ui.label(
                    RichText::new("Free stock needs a source. None is set up yet.")
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::MUTED),
                );
                return;
            }
            if shelf == Shelf::Effects && view.effect_group == BODY_EFFECTS {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Effects that follow a body will show here.")
                            .size(tokens::TYPE_SIZE)
                            .color(tokens::MUTED),
                    );
                    inspector::later_badge(ui);
                });
                return;
            }
            let group = view.sticker_group;
            if shelf == Shelf::Stickers && (group == YOURS || group == GIPHY) {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(if group == YOURS {
                            "Stickers you make or import will show here."
                        } else {
                            "GIPHY search will show here."
                        })
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::MUTED),
                    );
                    inspector::later_badge(ui);
                });
                return;
            }
            let mut cards = if bin {
                folder_cards(shell.library_store(), &view.library_folder, &query)
            } else {
                Vec::new()
            };
            cards.extend(shell.draft().map_or_else(Vec::new, |draft| {
                let place = BinPlace {
                    store: shell.library_store(),
                    folder: &view.library_folder,
                    sort: view.library_sort,
                };
                bin_cards(draft, &place, shelf, &query)
            }));
            if shelf == Shelf::Stickers && group == SHAPES {
                cards = shape_cards(&query);
            }
            if shelf == Shelf::Effects && !view.effect_category.is_empty() {
                let category = view.effect_category.as_str();
                cards.retain(|card| {
                    cutlass_editor::effect_of_card(&card.id)
                        .and_then(|id| cutlass_effects::recipe(&id))
                        .is_some_and(|recipe| recipe.category == category)
                });
            }
            if shelf == Shelf::Transitions && !view.transition_category.is_empty() {
                let category = view.transition_category.as_str();
                cards.retain(|card| {
                    cutlass_editor::transition_of_card(&card.id)
                        .is_some_and(|offer| offer.category == category)
                });
            }
            if shelf == Shelf::Filters && !view.filter_category.is_empty() {
                let category = view.filter_category.as_str();
                cards.retain(|card| {
                    cutlass_editor::filter_of_card(&card.id)
                        .is_some_and(|look| look.category == category)
                });
            }
            fill_looks(shell, ui.ctx(), &mut cards);
            if cards.is_empty() {
                let message = if !query.trim().is_empty() {
                    "No matches."
                } else if !view.library_folder.is_empty() {
                    "This folder is empty."
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
            } else if let Some(id) = card_grid(shell, ui, view, &cards, bin) {
                let added = if id.starts_with(cutlass_editor::TITLE_CARD_PREFIX) {
                    shell.add_title(&id)
                } else if id.starts_with(cutlass_editor::EFFECT_CARD_PREFIX) {
                    shell.add_effect(&id)
                } else if id == cutlass_editor::ADJUSTMENT_CARD {
                    shell.add_adjustment()
                } else if id.starts_with(cutlass_editor::FILTER_CARD_PREFIX) {
                    shell.add_filter(&id)
                } else if id.starts_with(crate::shell::STICKER_CARD_PREFIX) {
                    shell.add_sticker(&id)
                } else if id == cutlass_editor::PEN_CARD {
                    if shell.pen().is_some() {
                        shell.stop_pen();
                        Ok(())
                    } else {
                        shell.start_pen()
                    }
                } else if id.starts_with(cutlass_editor::SHAPE_CARD_PREFIX) {
                    shell.add_shape(&id)
                } else if id.starts_with(cutlass_editor::TEMPLATE_CARD_PREFIX) {
                    shell.add_template(&id)
                } else {
                    shell.add_medium(&id)
                };
                if let Err(error) = added {
                    shell.notice_error(&error);
                }
            }
        });
    library_menu(shell, &column, view);
}

/// Project or Free stock, over the media shelf.
fn source_switch(ui: &mut Ui, view: &mut ViewState) {
    let current = usize::from(view.library_stock);
    if let Some(picked) = segmented(ui, "library-source", &["Project", "Free stock"], current) {
        view.library_stock = picked == 1;
        view.library_picked.clear();
        view.library_anchor = None;
    }
}

/// The parts of the sticker shelf, in switch order.
const STICKER_GROUPS: [&str; 4] = ["Yours", "Stickers", "Shapes", "GIPHY"];
const YOURS: usize = 0;
const STICKERS: usize = 1;
const SHAPES: usize = 2;
const GIPHY: usize = 3;

/// A shelf's categories as a row of chips, All first, that scrolls
/// sideways when there are more than fit; the mouse wheel scrolls it too.
/// `picked` is the category shown, empty for All.
fn category_chips(ui: &mut Ui, id: &str, categories: &[&str], picked: &mut String) {
    let names = std::iter::once("All").chain(categories.iter().copied());
    let row = ui.available_rect_before_wrap();
    let hovered =
        ui.rect_contains_pointer(Rect::from_min_size(row.min, Vec2::new(row.width(), 28.0)));
    if hovered {
        // A wheel turned up or down moves the row sideways.
        ui.input_mut(|input| {
            let delta = input.smooth_scroll_delta;
            if delta.x == 0.0 && delta.y != 0.0 {
                input.smooth_scroll_delta = Vec2::new(delta.y, 0.0);
            }
        });
    }
    ui::Scroll::horizontal(id).without_bar().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for name in names {
                let value = if name == "All" { "" } else { name };
                let chip = ui::Button::new(name)
                    .on(*picked == value)
                    .text_size(12.0)
                    .radius(12);
                if ui.add(chip).clicked() {
                    value.clone_into(picked);
                }
            }
        });
    });
}

/// The parts of the effects shelf, in switch order.
const EFFECT_GROUPS: [&str; 2] = ["Video effects", "Body effects"];
const VIDEO_EFFECTS: usize = 0;
const BODY_EFFECTS: usize = 1;

/// `labels` as one segmented control across the column, `current` on.
/// Returns the label clicked when it is not the current one.
fn segmented(ui: &mut Ui, id: &str, labels: &[&str], current: usize) -> Option<usize> {
    let (track, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 34.0), Sense::hover());
    ui.painter()
        .rect_filled(track, CornerRadius::same(10), tokens::PROJECT_BAR);
    let part = (track.width() - 6.0) / labels.len().max(1) as f32;
    let mut picked = None;
    for (index, label) in labels.iter().enumerate() {
        let segment = Rect::from_min_size(
            track.min + Vec2::new(3.0 + part * index as f32, 3.0),
            Vec2::new(part, track.height() - 6.0),
        );
        let response = ui.interact(segment, ui.scope_id().with((id, index)), Sense::click());
        let on = index == current;
        if on {
            ui.painter()
                .rect_filled(segment, CornerRadius::same(8), tokens::ACTIVE);
        }
        let color = if on || response.hovered() {
            tokens::TEXT
        } else {
            tokens::FAINT
        };
        ui.painter().text(
            segment.center(),
            Align2::CENTER_CENTER,
            *label,
            FontId::proportional(tokens::TYPE_SIZE),
            color,
        );
        if response.clicked() && !on {
            picked = Some(index);
        }
    }
    picked
}

/// The shape shelf's cards whose names hold `query`, then the pen.
fn shape_cards(query: &str) -> Vec<BinCard> {
    let mut cards: Vec<BinCard> = cutlass_editor::shape_cards()
        .into_iter()
        .filter(|(_, kind)| query.is_empty() || kind.name().to_lowercase().contains(query))
        .map(|(id, kind)| BinCard {
            id,
            name: kind.name().to_owned(),
            meta: String::new(),
            on_timeline: false,
            look: CardLook::Shape(kind),
        })
        .collect();
    let pen = cutlass_model::SHAPE_PEN;
    if query.is_empty() || pen.to_lowercase().contains(query) {
        cards.push(BinCard {
            id: cutlass_editor::PEN_CARD.to_owned(),
            name: pen.to_owned(),
            meta: String::new(),
            on_timeline: false,
            look: CardLook::Pen,
        });
    }
    cards
}

/// A shape's outline, white on the card, as the library shows it.
fn paint_shape_mark(painter: &egui::Painter, rect: Rect, kind: cutlass_model::ShapeKind) {
    use cutlass_model::ShapeKind;
    let stroke = egui::Stroke::new(2.0, tokens::TEXT);
    let box_side = rect.width().min(rect.height()) * 0.5;
    let mark = Rect::from_center_size(rect.center(), Vec2::splat(box_side));
    if kind == ShapeKind::Circle {
        painter.circle_stroke(mark.center(), box_side / 2.0, stroke);
        return;
    }
    let preset = cutlass_model::MaterialShape::preset(Default::default(), kind, 2.0, 2.0);
    let points: Vec<Pos2> = preset
        .points()
        .iter()
        .map(|point| {
            mark.center() + Vec2::new(point.at[0] as f32, -point.at[1] as f32) * (box_side / 2.0)
        })
        .collect();
    if kind.is_line() {
        painter.line_segment([points[0], points[1]], stroke);
        if kind == ShapeKind::Arrow {
            let tip = points[1];
            let head = box_side * 0.18;
            painter.add(egui::Shape::convex_polygon(
                vec![
                    tip,
                    tip + Vec2::new(-head * 1.6, -head),
                    tip + Vec2::new(-head * 1.6, head),
                ],
                tokens::TEXT,
                egui::Stroke::NONE,
            ));
        }
        return;
    }
    painter.add(egui::Shape::closed_line(points, stroke));
}

/// Inside a folder: "All", then each folder down to the one shown. A click
/// goes there; a card dropped on one moves into it. At the top it is not
/// shown.
fn path_bar(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let trail: Vec<(String, String)> = std::iter::once((String::new(), "All".to_owned()))
        .chain(
            cutlass_editor::folder_trail(shell.library_store(), &view.library_folder)
                .into_iter()
                .map(|folder| (folder.id.clone(), folder.display_name.clone())),
        )
        .collect();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (index, (id, name)) in trail.iter().enumerate() {
            if index > 0 {
                ui.label(
                    RichText::new("/")
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::FAINT),
                );
            }
            let current = *id == view.library_folder;
            let response = ui.add(
                egui::Label::new(
                    RichText::new(name)
                        .size(tokens::TYPE_SIZE)
                        .color(if current { tokens::TEXT } else { tokens::MUTED }),
                )
                .sense(Sense::click()),
            );
            drop_into_folder(shell, &response, id);
            if response.clicked() {
                view.library_folder.clone_from(id);
                view.library_picked.clear();
                view.library_anchor = None;
            }
        }
    });
}

/// A card dragged from the library and let go over `response` moves into
/// the folder `folder`, or to the top when it is empty.
fn drop_into_folder(shell: &mut Shell, response: &egui::Response, folder: &str) {
    if let Some(drop) = response.dnd_release_payload::<LibraryDrop>()
        && drop.material_id != folder
        && let Err(error) = shell.move_library_item(&drop.material_id, folder)
    {
        shell.notice_error(&error);
    }
}

/// The library menu for the empty space, the picked files, or a folder.
fn library_menu(shell: &mut Shell, ui: &Ui, view: &mut ViewState) {
    let Some(menu) = view.library_menu.clone() else {
        return;
    };
    if let Some(pos) = secondary_click(ui)
        && pos != menu.at
    {
        view.library_menu = None;
        return;
    }
    // Make proxy while a picked movie has none, Cancel proxy once each has
    // one or is getting one, and a greyed row when no movie is picked.
    let proxy = match menu.target {
        LibraryTarget::Files => shell.proxy_offer(&view.library_picked),
        LibraryTarget::Empty | LibraryTarget::Folder(_) => None,
    };
    let entries = match menu.target {
        LibraryTarget::Empty => crate::menu::LIBRARY_EMPTY,
        LibraryTarget::Files if proxy == Some(false) => crate::menu::LIBRARY_CARD_PROXIED,
        LibraryTarget::Files => crate::menu::LIBRARY_CARD,
        LibraryTarget::Folder(_) => crate::menu::LIBRARY_FOLDER,
    };
    let off = match menu.target {
        LibraryTarget::Empty => &[][..],
        LibraryTarget::Files => crate::menu::LIBRARY_CARD_OFF,
        LibraryTarget::Folder(_) => crate::menu::LIBRARY_FOLDER_OFF,
    };
    let shown = crate::menu::show(
        ui.ctx(),
        ui.scope_id().with("library-menu"),
        menu.at,
        entries,
        &|id| !off.contains(&id) && (id != "make-proxy" || proxy.is_some()),
    );
    if let Some(item) = shown.picked {
        view.library_menu = None;
        library_action(shell, view, &menu.target, item);
    } else if shown.dismissed {
        view.library_menu = None;
    }
}

/// Run a library menu row.
fn library_action(
    shell: &mut Shell,
    view: &mut ViewState,
    target: &LibraryTarget,
    item: crate::menu::Item<'_>,
) {
    let picked = view.library_picked.clone();
    let result = match (item.id, target) {
        ("new-folder", _) => shell
            .new_library_folder(&view.library_folder, &[])
            .map(|id| start_folder_name(shell, view, id)),
        ("new-folder-with", LibraryTarget::Files) => shell
            .new_library_folder(&view.library_folder, &picked)
            .map(|id| start_folder_name(shell, view, id)),
        ("sort-name", _) => {
            view.library_sort = cutlass_editor::LibrarySort::Name;
            shell.set_notice("Sorted by name");
            Ok(())
        }
        ("sort-duration", _) => {
            view.library_sort = cutlass_editor::LibrarySort::Duration;
            shell.set_notice("Sorted by duration");
            Ok(())
        }
        ("open-file", LibraryTarget::Files) => match picked.first() {
            Some(id) => shell.reveal_library_file(id),
            None => Ok(()),
        },
        ("make-proxy", LibraryTarget::Files) => shell.make_proxy(&picked),
        ("cancel-proxy", LibraryTarget::Files) => shell.cancel_proxy(&picked),
        ("delete", LibraryTarget::Files) => {
            let files: Vec<String> = picked
                .iter()
                .filter(|id| {
                    !shell
                        .library_store()
                        .folders
                        .iter()
                        .any(|folder| &folder.id == *id)
                })
                .cloned()
                .collect();
            view.library_picked.retain(|id| !files.contains(id));
            shell.hide_library_files(&files)
        }
        ("delete", LibraryTarget::Folder(id)) => shell.delete_library_folder(id).map(|parent| {
            view.library_picked.retain(|picked| picked != id);
            let inside = cutlass_editor::folder_trail(shell.library_store(), &view.library_folder);
            if inside.is_empty() && !view.library_folder.is_empty() {
                view.library_folder = parent.unwrap_or_default();
            }
        }),
        ("rename", LibraryTarget::Folder(id)) => {
            start_folder_name(shell, view, id.clone());
            Ok(())
        }
        ("export", LibraryTarget::Files | LibraryTarget::Folder(_)) => {
            let asked = match target {
                LibraryTarget::Folder(id) => shell.export_library(std::slice::from_ref(id)),
                _ => shell.export_library(&picked),
            };
            asked.map(|()| {
                view.export.span = crate::shell::ExportSpan::Library;
                view.export_open = true;
            })
        }
        _ => {
            shell.set_notice(format!("{} is not built yet", item.label));
            Ok(())
        }
    };
    if let Err(error) = result {
        shell.notice_error(&error);
    }
}

/// Pick a folder and put its name in a field to type over.
fn start_folder_name(shell: &Shell, view: &mut ViewState, id: String) {
    let name = shell
        .library_store()
        .folders
        .iter()
        .find(|folder| folder.id == id)
        .map(|folder| folder.display_name.clone())
        .unwrap_or_default();
    view.library_picked = vec![id.clone()];
    view.library_anchor = Some(id.clone());
    view.library_renaming = Some((id, name));
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
fn card_grid(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    cards: &[BinCard],
    bin: bool,
) -> Option<String> {
    let mut added = None;
    let width = ui.available_width().max(CARD_W);
    ui.set_min_width(width);
    let cols = ((width + CARD_GAP) / (CARD_W + CARD_GAP)).floor().max(1.0) as usize;
    let cols = cols.max(1);
    let rows = cards.len().div_ceil(cols);
    let pitch = CARD_H + CARD_LABEL + CARD_GAP;
    let height = rows as f32 * pitch;
    let (area, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    let mut hovered_effect = None;
    for (index, card) in cards.iter().enumerate() {
        let origin = area.min
            + Vec2::new(
                (index % cols) as f32 * (CARD_W + CARD_GAP),
                (index / cols) as f32 * pitch,
            );
        let rect = Rect::from_min_size(origin, Vec2::new(CARD_W, CARD_H + CARD_LABEL));
        let folder = card.look == CardLook::Folder;
        let pen = card.look == CardLook::Pen;
        let lit = view.library_picked.contains(&card.id) || (pen && shell.pen().is_some());
        // A file whose proxy is on the way shows a running clock by its
        // name, and one that plays its proxy a Proxy badge.
        let proxy = if bin {
            shell.proxy_state(&card.id)
        } else {
            crate::shell::ProxyState::None
        };
        let making = match proxy {
            crate::shell::ProxyState::Making(done) => Some(done),
            _ => None,
        };
        paint_bin_card(ui, rect, card, lit, proxy);
        if folder && folder_name_field(shell, ui, view, card, rect) {
            continue;
        }
        // A click picks the card. A drag only starts once the pointer moves.
        let sense = if folder || pen {
            Sense::click()
        } else {
            Sense::click_and_drag()
        };
        let response = ui.interact(rect, ui.scope_id().with(&card.id), sense);
        if response.hovered() && plays_on_hover(&card.id) {
            hovered_effect = Some(card.id.as_str());
        }
        if pen {
            // The pen is a tool, not a clip: a click picks it up or puts
            // it down.
            if response.clicked() {
                // Only the pen is lit while it is up.
                view.library_picked.clear();
                view.library_anchor = None;
                added = Some(card.id.clone());
            }
            response.on_hover_text("Draw a shape on the preview");
            continue;
        }
        if folder {
            drop_into_folder(shell, &response, &card.id);
        } else {
            response.dnd_set_drag_payload(LibraryDrop {
                material_id: card.id.clone(),
            });
        }
        if response.clicked() {
            let modifiers = ui.input(|input| input.modifiers);
            if folder && !(modifiers.shift || modifiers.command || modifiers.ctrl) {
                // A plain click on a folder goes into it.
                view.library_folder.clone_from(&card.id);
                view.library_picked = vec![card.id.clone()];
                view.library_anchor = Some(card.id.clone());
            } else {
                pick_card(view, cards, &card.id, modifiers);
                // Picking another card puts the pen down.
                shell.stop_pen();
            }
        }
        if response.secondary_clicked() {
            if !view.library_picked.contains(&card.id) {
                view.library_picked = vec![card.id.clone()];
                view.library_anchor = Some(card.id.clone());
            }
            if bin && let Some(at) = response.interact_pointer_pos() {
                let target = if folder {
                    LibraryTarget::Folder(card.id.clone())
                } else {
                    LibraryTarget::Files
                };
                view.library_menu = Some(LibraryMenu { at, target });
            }
        }
        match making {
            Some(done) => {
                response.on_hover_text(format!("{}\n{}", card.name, proxy_progress(done)))
            }
            None => response.on_hover_text(&card.name),
        };
        if folder {
            continue;
        }
        if bin && ui.rect_contains_pointer(rect) && card_plus(ui, rect, &card.id) {
            added = Some(card.id.clone());
        }
    }
    note_effect_hover(ui.ctx(), hovered_effect);
    added
}

/// While `card` is the folder being named, a field over its name. Enter or
/// a click away keeps the name; Escape keeps the old one. True while the
/// field is up.
fn folder_name_field(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    card: &BinCard,
    rect: Rect,
) -> bool {
    let Some((id, typed)) = view.library_renaming.as_mut() else {
        return false;
    };
    if *id != card.id {
        return false;
    }
    let label = Rect::from_min_size(
        Pos2::new(rect.left(), rect.top() + CARD_H + 2.0),
        Vec2::new(rect.width(), CARD_LABEL - 2.0),
    );
    let field = ui::Field::new(&mut *typed)
        .text_size(12.0)
        .focus()
        .show_in(ui, label);
    if field.committed || field.cancelled {
        let cancel = field.cancelled;
        let typed = typed.clone();
        let id = id.clone();
        view.library_renaming = None;
        if !cancel && let Err(error) = shell.rename_library_folder(&id, &typed) {
            shell.notice_error(&error);
        }
    }
    true
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

/// What a card says of the proxy being made for its file, `done` of it
/// drawn once the drawing has begun.
fn proxy_progress(done: Option<f32>) -> String {
    match done {
        Some(done) => format!("Making a proxy: {:.0}%", done.clamp(0.0, 1.0) * 100.0),
        None => "Waiting to make a proxy".to_owned(),
    }
}

/// The width a card's name gives up to the proxy clock before it.
const PROXY_CLOCK_ROOM: f32 = 16.0;

/// A small clock with its hands going round, `seconds` into the app's run:
/// the long hand turns once in two seconds and the short one a twelfth as
/// fast.
fn paint_proxy_clock(painter: &egui::Painter, center: Pos2, seconds: f64) {
    const RADIUS: f32 = 5.5;
    let stroke = egui::Stroke::new(1.2, tokens::MUTED);
    painter.circle_stroke(center, RADIUS, stroke);
    let turn = (seconds / 2.0).fract() as f32 * std::f32::consts::TAU;
    for (angle, reach) in [(turn, RADIUS - 1.8), (turn / 12.0, RADIUS - 3.0)] {
        let hand = Vec2::new(angle.sin(), -angle.cos()) * reach;
        painter.line_segment([center, center + hand], stroke);
    }
}

/// The badge before the name of a file that plays its proxy. Returns the
/// width the name gives up to it.
fn paint_proxy_badge(painter: &egui::Painter, at: Pos2) -> f32 {
    let galley = painter.layout_no_wrap(
        "Proxy".to_owned(),
        FontId::proportional(11.0),
        tokens::PANEL,
    );
    let chip = Rect::from_min_size(at, galley.size() + Vec2::new(8.0, 2.0));
    painter.rect_filled(chip, CornerRadius::same(3), tokens::MUTED);
    painter.galley(chip.min + Vec2::new(4.0, 1.0), galley, tokens::PANEL);
    chip.width() + 5.0
}

/// A card: its picture, then its name under it, after a running clock
/// while a proxy of its file is being made and a badge once it plays one.
fn paint_bin_card(
    ui: &Ui,
    rect: Rect,
    card: &BinCard,
    picked: bool,
    proxy: crate::shell::ProxyState,
) {
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
    let room = match proxy {
        crate::shell::ProxyState::None => 0.0,
        crate::shell::ProxyState::Making(_) => {
            let seconds = ui.input(|input| input.time);
            let center = Pos2::new(rect.left() + 8.0, picture.bottom() + 14.0);
            paint_proxy_clock(&painter, center, seconds);
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(50));
            PROXY_CLOCK_ROOM
        }
        crate::shell::ProxyState::Made => paint_proxy_badge(
            &painter,
            Pos2::new(rect.left() + 2.0, picture.bottom() + 6.0),
        ),
    };
    let name = middle_ellipsis(ui.ctx(), &card.name, &font, rect.width() - 4.0 - room);
    painter.text(
        Pos2::new(rect.left() + 2.0 + room, picture.bottom() + 6.0),
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
/// What tells one still of a card from the next: nothing for a card the
/// app ships, and for one from the studio folder the number that moves
/// each time the studio's recipes change.
fn studio_card_age(card_id: &str) -> i64 {
    let studio = card_id
        .strip_prefix(cutlass_editor::EFFECT_CARD_PREFIX)
        .is_some_and(cutlass_effects::is_studio_recipe)
        || card_id
            .strip_prefix(cutlass_editor::TRANSITION_CARD_PREFIX)
            .is_some_and(|id| {
                cutlass_effects::is_studio_transition(&format!(
                    "{}{id}",
                    cutlass_effects::BUILT_IN_PREFIX
                ))
            });
    if studio {
        i64::try_from(cutlass_effects::studio_generation()).unwrap_or(i64::MAX)
    } else {
        0
    }
}

fn fill_looks(shell: &mut Shell, ctx: &egui::Context, cards: &mut [BinCard]) {
    let hover = ctx.data(|data| data.get_temp::<(String, f64)>(effect_hover_id()));
    let now = ctx.input(|input| input.time);
    let mut stills = EFFECT_STILLS_A_FRAME;
    with_film_textures(ctx, |textures| {
        for card in cards.iter_mut() {
            match card.look {
                CardLook::Picture(_) if plays_on_hover(&card.id) => {
                    let transition = card.id.starts_with(cutlass_editor::TRANSITION_CARD_PREFIX);
                    // The card under the pointer plays.
                    if let Some((_, since)) = hover.as_ref().filter(|(id, _)| *id == card.id) {
                        let elapsed = Micros(((now - since).max(0.0) * 1_000_000.0) as i64);
                        let frame = if transition {
                            shell.transition_frame(&card.id, elapsed)
                        } else {
                            shell.effect_frame(&card.id, elapsed)
                        };
                        if let Some(frame) = frame {
                            card.look = CardLook::Picture(Some(frame));
                            let step = 1.0 / f64::from(cutlass_effects::PREVIEW_RATE);
                            ctx.request_repaint_after_secs(step as f32);
                            continue;
                        }
                    }
                    // A studio card's still is kept for the recipe it was
                    // drawn from: a saved change draws it again.
                    let key = cutlass_engine::ThumbKey {
                        path: card.id.clone().into(),
                        time: Micros(studio_card_age(&card.id)),
                    };
                    if let Some(entry) = textures.shown.get_mut(&key) {
                        entry.1 = textures.frame;
                        let size = entry.0.size();
                        card.look = CardLook::Picture(Some((
                            entry.0.id(),
                            [size[0] as u32, size[1] as u32],
                        )));
                        continue;
                    }
                    if stills == 0 {
                        ctx.request_repaint();
                        continue;
                    }
                    stills -= 1;
                    let still = if transition {
                        shell.transition_still(&card.id)
                    } else {
                        shell.effect_still(&card.id)
                    };
                    if let Some((rgba, [width, height])) = still {
                        let thumb = cutlass_engine::Thumb {
                            width,
                            height,
                            rgba,
                        };
                        let texture = film_texture(ctx, textures, key, &thumb);
                        card.look = CardLook::Picture(Some((texture, [width, height])));
                    }
                }
                CardLook::Picture(_) if card.id.starts_with(cutlass_editor::FILTER_CARD_PREFIX) => {
                    let Some(look) = cutlass_editor::filter_of_card(&card.id) else {
                        continue;
                    };
                    let key = cutlass_engine::ThumbKey {
                        path: card.id.clone().into(),
                        time: Micros(0),
                    };
                    // Twice the card's size, for a sharp picture on a dense screen.
                    let (width, height) = (192, 128);
                    let texture = if let Some(entry) = textures.shown.get_mut(&key) {
                        entry.1 = textures.frame;
                        entry.0.id()
                    } else {
                        if stills == 0 {
                            ctx.request_repaint();
                            continue;
                        }
                        stills -= 1;
                        let thumb = cutlass_engine::Thumb {
                            width,
                            height,
                            rgba: cutlass_effects::look_poster(look, width, height),
                        };
                        film_texture(ctx, textures, key, &thumb)
                    };
                    card.look = CardLook::Picture(Some((texture, [width, height])));
                }
                CardLook::Picture(_) if card.id.starts_with(crate::shell::STICKER_CARD_PREFIX) => {
                    let Some(sticker) = card
                        .id
                        .strip_prefix(crate::shell::STICKER_CARD_PREFIX)
                        .and_then(cutlass_stickers::sticker)
                    else {
                        continue;
                    };
                    let key = cutlass_engine::ThumbKey {
                        path: card.id.clone().into(),
                        time: Micros(0),
                    };
                    let (width, height) = (96, 64);
                    let texture = if let Some(entry) = textures.shown.get_mut(&key) {
                        entry.1 = textures.frame;
                        entry.0.id()
                    } else {
                        let Ok(rgba) = sticker_poster(sticker, width, height) else {
                            continue;
                        };
                        let thumb = cutlass_engine::Thumb {
                            width,
                            height,
                            rgba,
                        };
                        film_texture(ctx, textures, key, &thumb)
                    };
                    card.look = CardLook::Picture(Some((texture, [width, height])));
                }
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
                CardLook::Swatch
                | CardLook::Folder
                | CardLook::Title(_)
                | CardLook::Shape(_)
                | CardLook::Pen => {}
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
        CardLook::Title(color) => {
            painter.rect_filled(rect, radius, tokens::ICON_BUTTON);
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Aa",
                FontId::proportional(28.0),
                *color,
            );
        }
        CardLook::Shape(kind) => {
            painter.rect_filled(rect, radius, tokens::ICON_BUTTON);
            paint_shape_mark(painter, rect, *kind);
        }
        CardLook::Pen => {
            painter.rect_filled(rect, radius, tokens::ICON_BUTTON);
            let _ = icons::paint(
                painter,
                Rect::from_center_size(rect.center(), Vec2::splat(40.0)),
                "draw",
                tokens::TEXT,
            );
        }
        CardLook::Folder => {
            painter.rect_filled(rect, radius, tokens::ICON_BUTTON);
            let _ = icons::paint(
                painter,
                Rect::from_center_size(rect.center(), Vec2::splat(40.0)),
                "folder",
                tokens::MUTED,
            );
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
    /// A library folder.
    Folder,
    /// A title style: "Aa" in its color.
    Title(Color32),
    /// A shape from the shape shelf, drawn as its outline.
    Shape(cutlass_model::ShapeKind),
    /// The pen, which draws a shape on the preview.
    Pen,
}

/// Where the library is looking: its folders, the one shown, and the order.
struct BinPlace<'a> {
    store: &'a cutlass_model::VirtualStore,
    folder: &'a str,
    sort: cutlass_editor::LibrarySort,
}

/// The folder cards in `folder`, by name.
fn folder_cards(store: &cutlass_model::VirtualStore, folder: &str, query: &str) -> Vec<BinCard> {
    cutlass_editor::library_folders(store, folder, query)
        .into_iter()
        .map(|folder| BinCard {
            id: folder.id.clone(),
            name: folder.display_name.clone(),
            meta: String::new(),
            on_timeline: false,
            look: CardLook::Folder,
        })
        .collect()
}

#[derive(Clone, Debug)]
struct LibraryDrop {
    material_id: String,
}

fn bin_cards(
    draft: &cutlass_model::Draft,
    place: &BinPlace<'_>,
    shelf: Shelf,
    query: &str,
) -> Vec<BinCard> {
    let query = query.trim().to_lowercase();
    let mut cards = Vec::new();
    let files = |shelf| {
        cutlass_editor::library_files(draft, place.store, shelf, place.folder, place.sort, &query)
    };
    match shelf {
        Shelf::Media => {
            for file in files(cutlass_editor::LibraryShelf::Media) {
                cards.push(file_card(file, CardLook::Picture(None)));
            }
        }
        Shelf::Audio => {
            for file in files(cutlass_editor::LibraryShelf::Audio) {
                cards.push(file_card(file, CardLook::Sound(std::sync::Arc::from([]))));
            }
        }
        Shelf::Transitions => {
            for offer in crate::shell::transition_offers() {
                if !query.is_empty() && !offer.name.to_lowercase().contains(&query) {
                    continue;
                }
                cards.push(BinCard {
                    id: offer.card_id(),
                    name: offer.name.to_owned(),
                    meta: card_meta(offer.duration),
                    on_timeline: false,
                    look: CardLook::Picture(None),
                });
            }
        }
        Shelf::Text => {
            for style in cutlass_editor::TITLE_STYLES {
                if !query.is_empty() && !style.name.to_lowercase().contains(&query) {
                    continue;
                }
                cards.push(title_card(style));
            }
        }
        Shelf::Effects => {
            for effect in cutlass_editor::effect_cards() {
                if !query.is_empty() && !effect.name.to_lowercase().contains(&query) {
                    continue;
                }
                cards.push(effect_card(effect));
            }
        }
        Shelf::Adjustments => {
            if query.is_empty() || "adjustment".contains(&query) {
                cards.push(BinCard {
                    id: cutlass_editor::ADJUSTMENT_CARD.to_owned(),
                    name: "Adjustment".to_owned(),
                    meta: card_meta(cutlass_editor::EFFECT_LENGTH.0),
                    on_timeline: false,
                    look: CardLook::Swatch,
                });
            }
        }
        Shelf::Filters => {
            for (id, name) in cutlass_editor::filter_cards() {
                if !query.is_empty() && !name.to_lowercase().contains(&query) {
                    continue;
                }
                cards.push(BinCard {
                    id,
                    name: name.to_owned(),
                    meta: card_meta(cutlass_editor::EFFECT_LENGTH.0),
                    on_timeline: false,
                    look: CardLook::Picture(None),
                });
            }
        }
        Shelf::Stickers => {
            for sticker in cutlass_stickers::search(&query) {
                cards.push(BinCard {
                    id: format!("{}{}", crate::shell::STICKER_CARD_PREFIX, sticker.id),
                    name: sticker.name.to_owned(),
                    meta: card_meta(crate::shell::sticker_length(sticker).0),
                    on_timeline: false,
                    look: CardLook::Picture(None),
                });
            }
        }
        Shelf::Templates => {
            for template in cutlass_editor::TITLE_TEMPLATES {
                if !query.is_empty() && !template.name.to_lowercase().contains(&query) {
                    continue;
                }
                cards.push(template_card(template));
            }
        }
    }
    cards
}

fn template_card(template: &cutlass_editor::TitleTemplate) -> BinCard {
    BinCard {
        id: template.card_id(),
        name: template.name.to_owned(),
        meta: template.meta.to_owned(),
        on_timeline: false,
        look: CardLook::Title(Color32::from_hex(template.color()).unwrap_or(Color32::WHITE)),
    }
}

fn effect_card(effect: cutlass_editor::EffectCard) -> BinCard {
    BinCard {
        id: effect.id,
        name: effect.name,
        // The card is a preview of the look; its length on the timeline
        // is not part of it.
        meta: String::new(),
        on_timeline: false,
        look: CardLook::Picture(None),
    }
}

/// Whether a card plays while the pointer rests on it: an effect or a
/// transition.
fn plays_on_hover(card_id: &str) -> bool {
    card_id.starts_with(cutlass_editor::EFFECT_CARD_PREFIX)
        || card_id.starts_with(cutlass_editor::TRANSITION_CARD_PREFIX)
}

/// Where the effect card under the pointer is kept between frames: its id
/// and the time the pointer came onto it, in seconds.
fn effect_hover_id() -> egui::Id {
    egui::Id::NULL.with("effect card hover")
}

/// Note the effect card under the pointer, keeping the time it came onto
/// it while it stays there.
fn note_effect_hover(ctx: &egui::Context, hovered: Option<&str>) {
    let now = ctx.input(|input| input.time);
    ctx.data_mut(|data| {
        let kept = data.get_temp::<(String, f64)>(effect_hover_id());
        match hovered {
            Some(id) if kept.as_ref().is_some_and(|(kept, _)| kept == id) => {}
            Some(id) => {
                data.insert_temp(effect_hover_id(), (id.to_owned(), now));
            }
            None => {
                data.remove::<(String, f64)>(effect_hover_id());
            }
        }
    });
}

/// Effect stills drawn in one frame at most; the rest follow on the next.
const EFFECT_STILLS_A_FRAME: usize = 12;

/// A sticker on a clear card `width` by `height`, as tall as the card and
/// in the middle, so the card's cover crop keeps all of it.
fn sticker_poster(
    sticker: &cutlass_stickers::Sticker,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let side = height.min(width);
    let drawn = sticker.rgba(side)?;
    let mut rgba = vec![0_u8; (width * height * 4) as usize];
    let left = (width - side) / 2;
    let top = (height - side) / 2;
    for row in 0..side {
        let from = (row * side * 4) as usize;
        let to = (((top + row) * width + left) * 4) as usize;
        let span = (side * 4) as usize;
        rgba[to..to + span].copy_from_slice(&drawn[from..from + span]);
    }
    Ok(rgba)
}

fn title_card(style: &cutlass_editor::TitleStyle) -> BinCard {
    BinCard {
        id: style.card_id(),
        name: style.name.to_owned(),
        meta: style.meta.to_owned(),
        on_timeline: false,
        look: CardLook::Title(Color32::from_hex(style.color).unwrap_or(Color32::WHITE)),
    }
}

fn file_card(file: cutlass_editor::LibraryFile, look: CardLook) -> BinCard {
    BinCard {
        id: file.id,
        name: file.name,
        meta: card_clock(file.duration.0),
        on_timeline: file.on_timeline,
        look,
    }
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
    let canvas = shell.draft().map(|draft| {
        Vec2::new(
            draft.canvas_config.width.max(1) as f32,
            draft.canvas_config.height.max(1) as f32,
        )
    });
    if view.preview_zoom.is_some() {
        let pan = ui.interact(frame, ui.scope_id().with("preview-pan"), Sense::drag());
        view.preview_pan += pan.drag_delta();
    } else {
        view.preview_pan = Vec2::ZERO;
    }
    preview_wheel(ui, view, frame);
    // The preview drag lands before the canvas is drawn, on the canvas
    // where it was last drawn, so this frame's picture already has it.
    let staging = shell.picking().is_none() && shell.tracking().is_none() && shell.pen().is_none();
    let hover = match (view.preview_canvas, canvas) {
        (Some(placed), Some(size)) if staging => stage_input(shell, ui, view, frame, placed, size),
        _ => {
            view.stage_drag = None;
            None
        }
    };
    view.preview_canvas = None;
    view.preview_frame = frame;
    let mut place = |size: Vec2| {
        let size = canvas.unwrap_or(size);
        let (rect, scale) = placed_canvas(frame, size, view.preview_zoom, &mut view.preview_pan);
        view.preview_scale = scale;
        view.preview_canvas = Some(rect);
        rect
    };
    let shown = match shell.preview_texture() {
        Ok(Some((id, [width, height]))) => {
            let placed = place(Vec2::new(width as f32, height as f32));
            paint_canvas(ui, frame, id, placed, shell.preview_black());
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
    shell.note_preview(match &fallback {
        Some(Err(error)) if shell.draft().is_some() => Some(error),
        _ => None,
    });
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
                let placed = place(texture.size_vec2());
                paint_canvas(ui, frame, texture.id(), placed, shell.preview_black());
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
    if view.preview_guides
        && let Some(canvas) = view.preview_canvas
    {
        paint_safe_areas(ui, frame, canvas);
    }
    if let (Some(placed), Some(size)) = (view.preview_canvas, canvas) {
        if staging {
            stage_paint(shell, ui, view, frame, placed, size, hover);
        }
        pick_layer(shell, ui, frame, placed, size);
        track_layer(shell, ui, view, frame, placed, size);
        pen_layer(shell, ui, frame, placed, size);
    }
    transport_bar(shell, ui, view, transport);
}

/// How near, in screen points, a press has to be to the first point to
/// close the path.
const PEN_CLOSE_REACH: f32 = 8.0;

/// The pen on the preview: each press puts a point down, a drag before
/// the release bends the curve through it, and a press on the first point
/// closes the path and places the shape.
fn pen_layer(shell: &mut Shell, ui: &mut Ui, frame: Rect, placed: Rect, size: Vec2) {
    if shell.pen().is_none() {
        return;
    }
    let area = placed.intersect(frame);
    let response = ui
        .interact(
            area,
            ui.scope_id().with("preview-pen"),
            Sense::click_and_drag(),
        )
        .on_hover_cursor(egui::CursorIcon::Crosshair);
    let to_canvas = |pos: Pos2| {
        let (x, y) = canvas_pixel(placed, size, pos);
        [x, y]
    };
    // Screen points to canvas pixels, for the reach to close.
    let per_point = f64::from(size.x / placed.width().max(1.0));
    let reach = f64::from(PEN_CLOSE_REACH) * per_point;
    let (pressed, down, released, pointer) = ui.input(|input| {
        (
            input.pointer.primary_pressed(),
            input.pointer.primary_down(),
            input.pointer.primary_released(),
            input.pointer.interact_pos(),
        )
    });
    let mut closed = false;
    if let Some(pen) = shell.pen_mut() {
        if pressed
            && response.hovered()
            && let Some(pos) = pointer
        {
            closed = pen.press(to_canvas(pos), reach) == Some(cutlass_editor::PenPress::Closed);
        } else if down
            && pen.is_pulling()
            && let Some(pos) = pointer
        {
            pen.drag(to_canvas(pos));
        }
        if released && pen.is_pulling() {
            pen.release();
            closed = pen.is_closed();
        }
    }
    if closed
        && !down
        && let Err(error) = shell.finish_pen()
    {
        shell.notice_error(&error);
    }
    let Some(pen) = shell.pen() else {
        return;
    };
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    let hint = if pen.points().is_empty() {
        "Click to put a point down, drag to curve it"
    } else {
        "Click the first point to close, Enter to finish, Esc to stop"
    };
    let galley =
        painter.layout_no_wrap(hint.to_owned(), FontId::proportional(12.0), Color32::WHITE);
    let chip = Rect::from_center_size(
        Pos2::new(frame.center().x, frame.top() + 18.0),
        galley.size() + Vec2::new(16.0, 8.0),
    );
    painter.rect_filled(chip, 6.0, Color32::from_black_alpha(170));
    painter.galley(chip.center() - galley.size() / 2.0, galley, Color32::WHITE);
    let screen = |p: [f64; 2]| screen_point(placed, size, (p[0], p[1]));
    let line = egui::Stroke::new(1.5, tokens::EXPORT);
    let curve = |cubic: [[f64; 2]; 4], stroke: egui::Stroke| {
        egui::Shape::CubicBezier(egui::epaint::CubicBezierShape::from_points_stroke(
            cubic.map(screen),
            false,
            Color32::TRANSPARENT,
            stroke,
        ))
    };
    for cubic in pen.cubics() {
        painter.add(curve(cubic, line));
    }
    let hover = response.hover_pos().map(to_canvas);
    if let Some(next) = hover.and_then(|at| pen.next_cubic(at)) {
        painter.add(curve(
            next,
            egui::Stroke::new(1.0, tokens::EXPORT.gamma_multiply(0.6)),
        ));
    }
    let closing = hover.is_some_and(|at| pen.closes_at(at, reach));
    for (index, point) in pen.points().iter().enumerate() {
        let at = screen(point.at);
        if point.handle_out != [0.0, 0.0] {
            let tips = [point.handle_in, point.handle_out]
                .map(|d| screen([point.at[0] + d[0], point.at[1] + d[1]]));
            painter.line_segment([tips[0], tips[1]], egui::Stroke::new(1.0, Color32::WHITE));
            for tip in tips {
                painter.circle_filled(tip, 3.0, Color32::WHITE);
            }
        }
        let first = index == 0;
        let radius = if first && closing { 6.0 } else { 4.0 };
        let fill = if first {
            tokens::EXPORT
        } else {
            Color32::WHITE
        };
        painter.rect(
            Rect::from_center_size(at, Vec2::splat(radius * 2.0)),
            1.0,
            fill,
            egui::Stroke::new(1.0, tokens::EXPORT),
            egui::StrokeKind::Middle,
        );
    }
}

/// Enter or Escape places the path drawn so far (Escape with nothing drawn
/// puts the pen down), and Backspace or Delete takes back the last point.
/// Returns whether a key was the pen's.
fn pen_keys(shell: &mut Shell, ui: &Ui) -> bool {
    let (enter, escape, back) = ui.input(|input| {
        (
            input.key_pressed(egui::Key::Enter),
            input.key_pressed(egui::Key::Escape),
            input.key_pressed(egui::Key::Backspace) || input.key_pressed(egui::Key::Delete),
        )
    });
    let drawn = shell.pen().is_some_and(|pen| !pen.points().is_empty());
    if back {
        if let Some(pen) = shell.pen_mut() {
            pen.undo();
        }
        return true;
    }
    if escape && !drawn {
        shell.stop_pen();
        return true;
    }
    if (enter || escape)
        && let Err(error) = shell.finish_pen()
    {
        shell.notice_error(&error);
    }
    enter || escape
}

/// What a drag on the preview grabbed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageGrip {
    /// The clip's body: the drag moves it.
    Move,
    /// A corner: the drag scales it about its centre.
    Scale,
    /// The knob below its bottom edge: the drag turns it about its centre.
    Turn,
    /// Nothing under the press: a zoomed canvas pans.
    Pan,
}

/// A preview drag: what it grabbed, and where it was pressed and the
/// clip's centre then, in canvas pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StageDrag {
    grip: StageGrip,
    start: (f64, f64),
    center: (f64, f64),
}

/// How near, in screen points, a press has to be to a handle to grab it.
const STAGE_HANDLE_REACH: f32 = 9.0;
/// How far past the bottom edge, in screen points, the turn knob sits.
const STAGE_TURN_OFFSET: f32 = 26.0;

/// The selected clip's box on screen: its four corners after the turn, top
/// left first, the middle of its bottom edge, and the turn knob below that.
/// When the knob would fall outside `frame`, as on a clip that fills the
/// canvas, it sits just inside the bottom edge instead.
fn stage_handles(
    stage: &cutlass_editor::StageBox,
    placed: Rect,
    size: Vec2,
    frame: Rect,
) -> ([Pos2; 4], Pos2, Pos2) {
    let corners = stage
        .corners()
        .map(|corner| screen_point(placed, size, corner));
    let bottom = corners[3] + (corners[2] - corners[3]) / 2.0;
    let center = screen_point(placed, size, (stage.center_x, stage.center_y));
    // Down, in the box's own turned frame.
    let down = (bottom - center).normalized();
    let down = if down.is_finite() && down != Vec2::ZERO {
        down
    } else {
        let (sin, cos) = (stage.turn as f32).to_radians().sin_cos();
        Vec2::new(-sin, cos)
    };
    let outside = bottom + down * STAGE_TURN_OFFSET;
    let knob = if frame.shrink(STAGE_HANDLE_REACH).contains(outside) {
        outside
    } else {
        bottom - down * STAGE_TURN_OFFSET
    };
    (corners, bottom, knob)
}

/// What a press at screen point `at` grabs on the selected clip's box.
fn stage_grip(
    stage: Option<&cutlass_editor::StageBox>,
    placed: Rect,
    size: Vec2,
    frame: Rect,
    at: Pos2,
) -> Option<StageGrip> {
    let stage = stage?;
    let (corners, _, knob) = stage_handles(stage, placed, size, frame);
    if at.distance(knob) <= STAGE_HANDLE_REACH {
        return Some(StageGrip::Turn);
    }
    if corners
        .iter()
        .any(|corner| at.distance(*corner) <= STAGE_HANDLE_REACH)
    {
        return Some(StageGrip::Scale);
    }
    let (x, y) = canvas_pixel(placed, size, at);
    stage.contains(x, y).then_some(StageGrip::Move)
}

/// The angle, in degrees clockwise on screen, of `point` seen from `center`.
fn screen_angle(center: (f64, f64), point: (f64, f64)) -> f64 {
    (point.1 - center.1).atan2(point.0 - center.0).to_degrees()
}

/// Pictures and titles on the preview: a click selects the clip under the
/// pointer and a drag moves it. The selected clip's corners scale it and
/// the knob below it turns it. Each drag is one undo step.
///
/// This runs before the canvas is drawn, so the picture drawn this frame
/// already has the edit and [`stage_paint`]'s box sits on it. Returns where
/// the pointer hovers when nothing is being dragged.
fn stage_input(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    frame: Rect,
    placed: Rect,
    size: Vec2,
) -> Option<Pos2> {
    let area = placed
        .intersect(frame)
        .expand(STAGE_TURN_OFFSET + STAGE_HANDLE_REACH);
    let response = ui.interact(
        area.intersect(frame),
        ui.scope_id().with("preview-stage"),
        Sense::click_and_drag(),
    );
    let selected = shell
        .selected()
        .cloned()
        .and_then(|id| shell.stage_box(&id));
    let at_canvas = |pos: Pos2| canvas_pixel(placed, size, pos);
    if response.drag_started() {
        let press = ui
            .input(|input| input.pointer.press_origin())
            .or(response.interact_pointer_pos());
        view.stage_drag = press.and_then(|press| {
            let start = at_canvas(press);
            let grip = stage_grip(selected.as_ref(), placed, size, frame, press);
            // A handle of the selected clip wins; then the topmost clip under
            // the press, which may sit over the selected one.
            let target = match (grip, selected.as_ref()) {
                (Some(grip @ (StageGrip::Scale | StageGrip::Turn)), Some(stage)) => {
                    Some((stage.segment_id.clone(), grip))
                }
                (grip, stage) => shell
                    .stage_pick(start.0, start.1)
                    .or_else(|| grip.and(stage.map(|stage| stage.segment_id.clone())))
                    .map(|id| (id, StageGrip::Move)),
            };
            let Some((id, grip)) = target else {
                return Some(StageDrag {
                    grip: StageGrip::Pan,
                    start,
                    center: start,
                });
            };
            if shell.playing() {
                shell.pause();
            }
            match shell.stage_begin(&id) {
                Ok(()) => {
                    let stage = shell.stage_box(&id)?;
                    Some(StageDrag {
                        grip,
                        start,
                        center: (stage.center_x, stage.center_y),
                    })
                }
                Err(error) => {
                    shell.select(Some(id));
                    shell.notice_error(&error);
                    None
                }
            }
        });
    }
    if let Some(drag) = view.stage_drag
        && response.dragged()
    {
        if drag.grip == StageGrip::Pan {
            if view.preview_zoom.is_some() {
                view.preview_pan += response.drag_delta();
            }
        } else if let Some(pos) = response.interact_pointer_pos() {
            let now = at_canvas(pos);
            let (shift, pinch, turn) = match drag.grip {
                StageGrip::Move => ((now.0 - drag.start.0, now.1 - drag.start.1), 1.0, 0.0),
                StageGrip::Scale => {
                    let from = (drag.start.0 - drag.center.0).hypot(drag.start.1 - drag.center.1);
                    let to = (now.0 - drag.center.0).hypot(now.1 - drag.center.1);
                    let pinch = if from > f64::EPSILON { to / from } else { 1.0 };
                    ((0.0, 0.0), pinch, 0.0)
                }
                StageGrip::Turn => {
                    let turn =
                        screen_angle(drag.center, now) - screen_angle(drag.center, drag.start);
                    ((0.0, 0.0), 1.0, turn)
                }
                StageGrip::Pan => ((0.0, 0.0), 1.0, 0.0),
            };
            if let Err(error) = shell.stage_move(shift.0, shift.1, pinch, turn) {
                shell.notice_error(&error);
                shell.stage_end();
                view.stage_drag = None;
            }
        }
    }
    if response.drag_stopped() || (view.stage_drag.is_some() && !response.dragged()) {
        shell.stage_end();
        view.stage_drag = None;
    }
    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (x, y) = at_canvas(pos);
        let on_handle = matches!(
            stage_grip(selected.as_ref(), placed, size, frame, pos),
            Some(StageGrip::Scale | StageGrip::Turn)
        );
        let picked = shell.stage_pick(x, y);
        let shift = ui.input(|input| input.modifiers.shift || input.modifiers.command);
        match picked {
            Some(id) if shift => shell.toggle_selected(id),
            // A click on the selected clip's own handles keeps it.
            _ if on_handle && !shift => {}
            Some(id) => shell.select(Some(id)),
            None if !shift => shell.select(None),
            None => {}
        }
    }
    response.hover_pos().filter(|_| view.stage_drag.is_none())
}

/// How many recent revisions' boxes the preview keeps to match a picture
/// drawn from one of them.
const STAGE_SEEN: usize = 8;

/// The selected clips' boxes as the picture on the canvas shows them: from
/// the revision it was drawn from, when that is one of the last few. A clip
/// selected since then, or a picture from a render, takes the box the draft
/// gives now.
fn shown_stage_boxes(shell: &Shell, view: &mut ViewState) -> Vec<cutlass_editor::StageBox> {
    let mut ids: Vec<SegmentId> = shell.selection().to_vec();
    if let Some(moving) = shell.stage_moving()
        && !ids.contains(moving)
    {
        ids.push(moving.clone());
    }
    let now: Vec<cutlass_editor::StageBox> =
        ids.iter().filter_map(|id| shell.stage_box(id)).collect();
    if let Some(revision) = shell.revision() {
        view.stage_seen.retain(|(seen, _)| *seen != revision);
        view.stage_seen.push((revision, now.clone()));
        let extra = view.stage_seen.len().saturating_sub(STAGE_SEEN);
        view.stage_seen.drain(..extra);
    }
    let shown = shell
        .shown_revision()
        .and_then(|shown| view.stage_seen.iter().find(|(seen, _)| *seen == shown))
        .map(|(_, boxes)| boxes.as_slice())
        .unwrap_or_default();
    now.into_iter()
        .map(|stage| {
            shown
                .iter()
                .find(|old| old.segment_id == stage.segment_id)
                .cloned()
                .unwrap_or(stage)
        })
        .collect()
}

/// Over the drawn canvas: an outline on the clip under the pointer, the
/// selected clip's box with its corner handles and turn knob, plain
/// outlines on the other selected clips, and the centre lines a drag
/// snapped to.
fn stage_paint(
    shell: &Shell,
    ui: &Ui,
    view: &mut ViewState,
    frame: Rect,
    placed: Rect,
    size: Vec2,
    hover: Option<Pos2>,
) {
    let boxes = shown_stage_boxes(shell, view);
    let at_canvas = |pos: Pos2| canvas_pixel(placed, size, pos);
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    let selected_id = shell.stage_moving().or(shell.selected()).cloned();
    let shown = |id: &SegmentId| boxes.iter().find(|stage| stage.segment_id == *id);
    let selected = selected_id.as_ref().and_then(shown).cloned();
    if let Some(pos) = hover {
        let grip = stage_grip(selected.as_ref(), placed, size, frame, pos);
        let (x, y) = at_canvas(pos);
        let under = match grip {
            Some(StageGrip::Scale | StageGrip::Turn) => None,
            _ => shell.stage_pick(x, y),
        };
        let cursor = match grip {
            Some(StageGrip::Turn) => Some(egui::CursorIcon::Alias),
            Some(StageGrip::Scale) => Some(egui::CursorIcon::Crosshair),
            Some(StageGrip::Move) => Some(egui::CursorIcon::Move),
            Some(StageGrip::Pan) | None => under.is_some().then_some(egui::CursorIcon::Move),
        };
        if let Some(cursor) = cursor {
            ui.ctx().set_cursor_icon(cursor);
        }
        if let Some(stage) = under
            .filter(|id| Some(id) != selected_id.as_ref())
            .and_then(|id| shell.stage_box(&id))
        {
            let (corners, _, _) = stage_handles(&stage, placed, size, frame);
            painter.add(egui::Shape::closed_line(
                corners.to_vec(),
                egui::Stroke::new(1.0, Color32::from_white_alpha(150)),
            ));
        }
    } else if let Some(drag) = view.stage_drag {
        let cursor = match drag.grip {
            StageGrip::Move => egui::CursorIcon::Grabbing,
            StageGrip::Scale => egui::CursorIcon::Crosshair,
            StageGrip::Turn => egui::CursorIcon::Alias,
            StageGrip::Pan => egui::CursorIcon::Grabbing,
        };
        ui.ctx().set_cursor_icon(cursor);
    }
    // Every other selected clip gets a plain outline.
    for id in shell.selection() {
        if Some(id) == selected_id.as_ref() {
            continue;
        }
        if let Some(stage) = shown(id) {
            let (corners, _, _) = stage_handles(stage, placed, size, frame);
            painter.add(egui::Shape::closed_line(
                corners.to_vec(),
                egui::Stroke::new(1.0, tokens::SELECTED),
            ));
        }
    }
    if let Some(guides) = shell.stage_guides() {
        let stroke = egui::Stroke::new(1.0, tokens::EXPORT);
        if guides.vertical {
            let x = placed.center().x;
            painter.line_segment(
                [Pos2::new(x, placed.min.y), Pos2::new(x, placed.max.y)],
                stroke,
            );
        }
        if guides.horizontal {
            let y = placed.center().y;
            painter.line_segment(
                [Pos2::new(placed.min.x, y), Pos2::new(placed.max.x, y)],
                stroke,
            );
        }
    }
    let Some(stage) = selected else {
        return;
    };
    let (corners, bottom, knob) = stage_handles(&stage, placed, size, frame);
    let stroke = egui::Stroke::new(1.5, tokens::SELECTED);
    painter.add(egui::Shape::closed_line(corners.to_vec(), stroke));
    painter.line_segment([bottom, knob], stroke);
    let rest = egui::Stroke::new(1.0, Color32::from_black_alpha(160));
    for corner in corners {
        painter.rect(
            Rect::from_center_size(corner, Vec2::splat(8.0)),
            1.0,
            tokens::SELECTED,
            rest,
            egui::StrokeKind::Middle,
        );
    }
    painter.circle(knob, 5.5, tokens::SELECTED, rest);
}

/// While an object is being picked: a click on the canvas adds a point
/// (shift takes a part out), and each point so far is marked.
fn pick_layer(shell: &mut Shell, ui: &mut Ui, frame: Rect, placed: Rect, size: Vec2) {
    if shell.picking().is_none() {
        return;
    }
    let area = placed.intersect(frame);
    let response = ui
        .interact(area, ui.scope_id().with("preview-pick"), Sense::click())
        .on_hover_cursor(egui::CursorIcon::Crosshair);
    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (x, y) = canvas_pixel(placed, size, pos);
        let keep = !ui.input(|input| input.modifiers.shift);
        if let Err(error) = shell.pick_at(x, y, keep) {
            shell.notice_error(&error);
        }
    }
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    for click in shell
        .picking()
        .map(|p| p.clicks.as_slice())
        .unwrap_or_default()
    {
        let at = screen_point(placed, size, click.canvas);
        let colour = if click.keep {
            tokens::EXPORT
        } else {
            tokens::CLOSE
        };
        painter.circle(at, 5.0, colour, egui::Stroke::new(1.5, Color32::WHITE));
    }
}

/// The selected clip's tracked path: the target's middle frame by frame
/// and its box at the playhead. While a track is set up, a drag on the
/// canvas draws its box.
fn track_layer(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    frame: Rect,
    placed: Rect,
    size: Vec2,
) {
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    let screen = |p: (f64, f64)| screen_point(placed, size, p);
    let outline = |corners: [(f64, f64); 4], stroke: egui::Stroke| {
        let mut points: Vec<Pos2> = corners.iter().map(|&p| screen(p)).collect();
        if let Some(&first) = points.first() {
            points.push(first);
        }
        painter.add(egui::Shape::line(points, stroke));
    };
    if let Some(path) = shell.tracked_path() {
        let middles: Vec<Pos2> = path
            .frames
            .iter()
            .map(|(_, c)| {
                let sum = c.iter().fold((0.0, 0.0), |s, p| (s.0 + p.0, s.1 + p.1));
                screen((sum.0 / 4.0, sum.1 / 4.0))
            })
            .collect();
        painter.add(egui::Shape::line(
            middles,
            egui::Stroke::new(1.5, tokens::EXPORT.gamma_multiply(0.6)),
        ));
        if let Some(corners) = path.at(shell.playhead()) {
            outline(corners, egui::Stroke::new(1.5, tokens::EXPORT));
        }
    }
    let Some(tracking) = shell.tracking().cloned() else {
        view.track_drag = None;
        return;
    };
    let area = placed.intersect(frame);
    let response = ui
        .interact(
            area,
            ui.scope_id().with("preview-track"),
            Sense::click_and_drag(),
        )
        .on_hover_cursor(egui::CursorIcon::Crosshair);
    let pointer = response
        .interact_pointer_pos()
        .map(|pos| canvas_pixel(placed, size, pos));
    // A click finds the object under it and boxes it; a drag draws the box.
    if response.clicked()
        && shell.track_subject_enabled()
        && let Some((x, y)) = pointer
        && let Err(error) = shell.pick_track_subject(x, y)
    {
        shell.notice_error(&error);
    }
    if response.drag_started() {
        // From where the press began: a drag is told from a click only
        // once the pointer has moved.
        view.track_drag = ui
            .input(|input| input.pointer.press_origin())
            .map(|pos| canvas_pixel(placed, size, pos))
            .or(pointer);
    }
    let dragged = view.track_drag.zip(pointer);
    if response.drag_stopped() {
        if let Some((start, end)) = dragged
            && let Err(error) = shell.set_track_box(start, end)
        {
            shell.notice_error(&error);
        }
        view.track_drag = None;
    }
    let shown = match (view.track_drag.zip(pointer), tracking.canvas_box) {
        (Some((a, b)), _) | (None, Some((a, b))) => Some((a, b)),
        (None, None) => None,
    };
    if let Some((a, b)) = shown {
        outline(
            [(a.0, a.1), (b.0, a.1), (b.0, b.1), (a.0, b.1)],
            egui::Stroke::new(2.0, Color32::WHITE),
        );
    }
}

/// Where screen point `pos` falls on a canvas of `size` pixels drawn at
/// `placed`.
fn canvas_pixel(placed: Rect, size: Vec2, pos: Pos2) -> (f64, f64) {
    let at = (pos - placed.min) / placed.size() * size;
    (f64::from(at.x), f64::from(at.y))
}

/// Where canvas pixel `point` is on screen, for a canvas of `size` drawn at
/// `placed`.
fn screen_point(placed: Rect, size: Vec2, point: (f64, f64)) -> Pos2 {
    let at = Vec2::new(point.0 as f32, point.1 as f32) / size * placed.size();
    placed.min + at
}

/// The action-safe and title-safe frames over the canvas at `canvas`,
/// with a mark at its middle.
fn paint_safe_areas(ui: &Ui, frame: Rect, canvas: Rect) {
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    let stroke = egui::Stroke::new(1.0, Color32::from_white_alpha(140));
    for share in [cutlass_editor::ACTION_SAFE, cutlass_editor::TITLE_SAFE] {
        let (left, top, width, height) =
            cutlass_editor::safe_box(f64::from(canvas.width()), f64::from(canvas.height()), share);
        let inner = Rect::from_min_size(
            canvas.min + Vec2::new(left as f32, top as f32),
            Vec2::new(width as f32, height as f32),
        );
        painter.rect_stroke(inner, CornerRadius::ZERO, stroke, egui::StrokeKind::Middle);
    }
    let middle = canvas.center();
    painter.hline((middle.x - 8.0)..=(middle.x + 8.0), middle.y, stroke);
    painter.vline(middle.x, (middle.y - 8.0)..=(middle.y + 8.0), stroke);
}

/// Fill the screen with the preview, or go back to the editor.
fn set_full_screen(ui: &Ui, view: &mut ViewState, full: bool) {
    view.preview_full = full;
    ui.ctx()
        .send_viewport_cmd(egui::ViewportCommand::Fullscreen(full));
}

/// A pinch, or the wheel with Cmd (Ctrl off a Mac), zooms the preview
/// about the pointer; a plain wheel or two-finger scroll pans a zoomed
/// canvas. egui has already turned mouse wheel lines into points, and a
/// Cmd wheel into a zoom; on a Mac a Ctrl wheel is still a scroll, so it
/// is turned into a zoom here at egui's own rate.
fn preview_wheel(ui: &Ui, view: &mut ViewState, frame: Rect) {
    let (pinch, scroll, modifier, pos) = ui.input(|input| {
        (
            input.zoom_delta(),
            input.smooth_scroll_delta,
            input.modifiers.command || input.modifiers.ctrl,
            input.pointer.hover_pos(),
        )
    });
    let speed = ui
        .ctx()
        .options(|options| options.input_options.scroll_zoom_speed);
    let Some(pos) = pos.filter(|pos| frame.contains(*pos) && !over_menu(ui, *pos)) else {
        return;
    };
    let mut factor = f64::from(pinch);
    if modifier {
        factor *= f64::from(speed * (scroll.x + scroll.y)).exp();
    } else if view.preview_zoom.is_some() && scroll != Vec2::ZERO {
        view.preview_pan += scroll;
    }
    let anchor = pos - frame.center();
    let pan = view.preview_pan;
    if let Some((zoom, (x, y))) = cutlass_editor::preview_zoom_at(
        view.preview_scale,
        factor,
        (f64::from(anchor.x), f64::from(anchor.y)),
        (f64::from(pan.x), f64::from(pan.y)),
    ) {
        view.preview_zoom = Some(zoom);
        view.preview_pan = Vec2::new(x as f32, y as f32);
        // The canvas is placed later this frame at this scale.
        view.preview_scale = zoom;
    }
}

/// Where a canvas `size` canvas pixels sits in `frame`: fitted, or at
/// `zoom` screen pixels per canvas pixel moved by `pan`. A pan is held to
/// what keeps the canvas over the middle. Answers the rect and its scale.
fn placed_canvas(frame: Rect, size: Vec2, zoom: Option<f64>, pan: &mut Vec2) -> (Rect, f64) {
    let Some(zoom) = zoom else {
        let fitted = fit_texture(frame, size);
        let scale = if size.x > 0.0 {
            f64::from(fitted.width() / size.x)
        } else {
            1.0
        };
        return (fitted, scale);
    };
    let shown = size * zoom as f32;
    let (x, y) = cutlass_editor::clamp_preview_pan(
        (f64::from(pan.x), f64::from(pan.y)),
        (f64::from(shown.x), f64::from(shown.y)),
        (f64::from(frame.width()), f64::from(frame.height())),
    );
    *pan = Vec2::new(x as f32, y as f32);
    (Rect::from_center_size(frame.center() + *pan, shown), zoom)
}

/// The preview canvas at `rect`, clipped to `frame`, over black when
/// `black` is on. Only the canvas gets that background; the area around it
/// stays unfilled.
fn paint_canvas(ui: &Ui, frame: Rect, texture: egui::TextureId, rect: Rect, black: bool) {
    let painter = ui.painter().with_clip_rect(frame.intersect(ui.clip_rect()));
    if black {
        painter.rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);
    }
    painter.image(
        texture,
        rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}

/// The timecode, play button, zoom, and preview buttons.
fn transport_bar(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let live = shell.draft().is_some();
    let ink = if live { tokens::TEXT } else { tokens::MUTED };
    // A narrow preview drops what it has no room for: the guides button
    // first, then the zoom buttons, and the timecode keeps the playhead
    // alone.
    let room = transport_room(rect.width(), live);
    let font = FontId::monospace(tokens::TIMECODE_SIZE);
    let both = format!("{}  /  {}", shell.timecode(), duration_label(shell));
    let fits = ui
        .painter()
        .layout_no_wrap(both.clone(), font.clone(), ink)
        .size()
        .x
        + TRANSPORT_EDGE
        + TRANSPORT_GAP
        <= rect.width() / 2.0 - PLAY_RADIUS;
    let timecode = if fits { both } else { shell.timecode() };
    ui.painter().text(
        Pos2::new(rect.left() + TRANSPORT_EDGE, rect.center().y),
        Align2::LEFT_CENTER,
        timecode,
        font,
        ink,
    );
    let play = Rect::from_center_size(rect.center(), Vec2::splat(PLAY_RADIUS * 2.0));
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
    if room.zoom {
        zoom_buttons(
            ui,
            view,
            Pos2::new(play.right() + ZOOM_GAP, rect.center().y),
        );
    }

    let still = Rect::from_center_size(
        Pos2::new(rect.right() - 28.0, rect.center().y),
        Vec2::splat(32.0),
    );
    background_toggle(shell, ui, still.translate(Vec2::new(-40.0, 0.0)));
    let mut next = -80.0;
    if room.guides {
        let guides = still.translate(Vec2::new(next, 0.0));
        next -= 40.0;
        let tip = if view.preview_guides {
            "Hide safe-area guides"
        } else {
            "Show safe-area guides"
        };
        if round_button(ui, guides, "guides", tip, view.preview_guides).clicked() {
            view.preview_guides = !view.preview_guides;
        }
        let mark = Rect::from_center_size(guides.center(), Vec2::new(16.0, 11.0));
        let stroke = egui::Stroke::new(1.2, tokens::MUTED);
        for inset in [0.0, 3.0] {
            ui.painter().rect_stroke(
                mark.shrink(inset),
                CornerRadius::ZERO,
                stroke,
                egui::StrokeKind::Middle,
            );
        }
    }
    if live {
        let screen = still.translate(Vec2::new(next, 0.0));
        next -= 40.0;
        if room.quality {
            quality_button(shell, ui, still.translate(Vec2::new(next, 0.0)));
        }
        let full = view.preview_full;
        let tip = if full {
            "Leave full screen"
        } else {
            "Full screen"
        };
        if round_button(ui, screen, "full-screen", tip, full).clicked() {
            set_full_screen(ui, view, !full);
        }
        paint_corners(
            ui,
            Rect::from_center_size(screen.center(), Vec2::splat(13.0)),
            full,
        );
    }
    let still_response = ui.interact(still, ui.scope_id().with("still"), Sense::click());
    if still_response.clicked() && live {
        view.ask_still = true;
    }
    ui.painter()
        .circle_filled(still.center(), 16.0, tokens::ICON_BUTTON);
    let camera = Rect::from_center_size(still.center(), Vec2::splat(16.0));
    let _ = icons::paint(ui.painter(), camera, "camera", tokens::MUTED);
    still_response.on_hover_text("Export still frame");
    shell.chrome.transport = true;
}

/// The transport's side gap, the gap kept between its parts, and the play
/// button's radius.
const TRANSPORT_EDGE: f32 = 16.0;
const TRANSPORT_GAP: f32 = 8.0;
const PLAY_RADIUS: f32 = 18.0;
/// The zoom buttons' gap from play and their width, and how far one round
/// button at the right reaches in from the edge, with each further one
/// 40 px more.
const ZOOM_GAP: f32 = 12.0;
const ZOOM_WIDTH: f32 = ZOOM_GAP + 22.0 + 2.0 + 44.0 + 2.0 + 22.0;
const ROUND_REACH: f32 = 44.0;

/// Which optional transport parts fit a preview `width` wide.
#[derive(Debug, PartialEq, Eq)]
struct TransportRoom {
    zoom: bool,
    quality: bool,
    guides: bool,
}

fn transport_room(width: f32, live: bool) -> TransportRoom {
    let half = width / 2.0 - PLAY_RADIUS - TRANSPORT_GAP;
    // Still and background always; full screen, quality and guides when
    // live.
    let reach = |buttons: f32| ROUND_REACH + 40.0 * (buttons - 1.0);
    // Zoom has no other way in, so it outlasts the others; quality is also
    // in Settings, and goes before guides do.
    let zoom = live && half >= reach(3.0) + ZOOM_WIDTH;
    let zoomed = if zoom { ZOOM_WIDTH } else { 0.0 };
    let guides = live && half >= reach(4.0) + zoomed;
    let quality = live && half >= reach(if guides { 5.0 } else { 4.0 }) + zoomed;
    TransportRoom {
        zoom,
        quality,
        guides,
    }
}

/// The preview qualities, with their names and what each does.
const PREVIEW_QUALITIES: [(cutlass_settings::PreviewQuality, &str, &str); 4] = [
    (
        cutlass_settings::PreviewQuality::Full,
        "Full",
        "The canvas's own size",
    ),
    (
        cutlass_settings::PreviewQuality::High,
        "High",
        "Half size: smooth playback",
    ),
    (
        cutlass_settings::PreviewQuality::Medium,
        "Medium",
        "A third: smoother playback",
    ),
    (
        cutlass_settings::PreviewQuality::Low,
        "Low",
        "A quarter: smoothest playback",
    ),
];

/// The preview quality button: its short name on a round button, and a
/// menu of the four qualities. Export is drawn at full quality whatever
/// is picked.
fn quality_button(shell: &mut Shell, ui: &mut Ui, rect: Rect) {
    let current = shell.settings().playback.preview_quality;
    let short = match current {
        cutlass_settings::PreviewQuality::Full => "Full",
        cutlass_settings::PreviewQuality::High => "High",
        cutlass_settings::PreviewQuality::Medium => "Med",
        cutlass_settings::PreviewQuality::Low => "Low",
    };
    let id = egui::Id::unique("preview-quality-menu");
    let open = egui::Popup::is_id_open(ui.ctx(), id);
    let response = round_button(ui, rect, "preview-quality", "Preview quality", open);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "Preview quality"));
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        short,
        FontId::proportional(10.5),
        tokens::MUTED,
    );
    let mut picked = None;
    egui::Popup::menu(&response)
        .id(id)
        .width(260.0)
        .gap(6.0)
        .frame(
            egui::Frame::new()
                .fill(tokens::MENU)
                .corner_radius(CornerRadius::same(tokens::EXPORT_RADIUS))
                .inner_margin(egui::Margin::same(4)),
        )
        .show(|ui| {
            ui.set_width(252.0);
            for (value, name, detail) in PREVIEW_QUALITIES {
                let label = format!("{name} quality");
                let (row, row_response) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 44.0), Sense::click());
                row_response.widget_info(|| {
                    egui::WidgetInfo::selected(
                        egui::Role::RadioButton,
                        true,
                        value == current,
                        &label,
                    )
                });
                if row_response.hovered() {
                    ui.painter()
                        .rect_filled(row, CornerRadius::same(6), tokens::MENU_HOVER);
                }
                if value == current {
                    let c = Pos2::new(row.left() + 16.0, row.center().y - 8.0);
                    ui.painter().add(egui::Shape::line(
                        vec![
                            c + Vec2::new(-5.0, 0.0),
                            c + Vec2::new(-1.5, 3.5),
                            c + Vec2::new(5.0, -4.0),
                        ],
                        egui::Stroke::new(2.0, tokens::EXPORT),
                    ));
                }
                ui.painter().text(
                    Pos2::new(row.left() + 32.0, row.center().y - 8.0),
                    Align2::LEFT_CENTER,
                    &label,
                    FontId::proportional(tokens::TYPE_SIZE),
                    tokens::TEXT,
                );
                ui.painter().text(
                    Pos2::new(row.left() + 32.0, row.center().y + 9.0),
                    Align2::LEFT_CENTER,
                    detail,
                    FontId::proportional(11.5),
                    tokens::MUTED,
                );
                if row_response.clicked() {
                    picked = Some(value);
                    ui.close();
                }
            }
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(10.0);
                ui.label(
                    RichText::new("Export is always full quality.")
                        .size(11.5)
                        .color(tokens::FAINT),
                );
            });
            ui.add_space(4.0);
        });
    if let Some(value) = picked {
        shell.update_settings(|settings| settings.playback.preview_quality = value);
    }
}

/// A round transport button at `rect`, lit when `on`.
fn round_button(ui: &Ui, rect: Rect, name: &str, tip: &str, on: bool) -> egui::Response {
    let response = ui.interact(rect, ui.scope_id().with(name), Sense::click());
    let fill = if on {
        tokens::ACTIVE
    } else {
        tokens::ICON_BUTTON
    };
    ui.painter().circle_filled(rect.center(), 16.0, fill);
    response.on_hover_text(tip)
}

/// Four corner brackets in `mark`: pointing out to fill the screen, in to
/// leave it.
fn paint_corners(ui: &Ui, mark: Rect, inward: bool) {
    let stroke = egui::Stroke::new(1.4, tokens::MUTED);
    let arm = 4.0;
    for corner in [
        mark.left_top(),
        mark.right_top(),
        mark.left_bottom(),
        mark.right_bottom(),
    ] {
        let toward = (mark.center() - corner).normalized();
        let (x, y) = (toward.x.signum(), toward.y.signum());
        let tip = if inward {
            corner + Vec2::new(x, y) * arm
        } else {
            corner
        };
        let (sx, sy) = if inward { (-x, -y) } else { (x, y) };
        ui.painter()
            .line_segment([tip, tip + Vec2::new(sx * arm, 0.0)], stroke);
        ui.painter()
            .line_segment([tip, tip + Vec2::new(0.0, sy * arm)], stroke);
    }
}

/// − , the zoom, + from `left`: a step smaller, back to fit, a step bigger.
fn zoom_buttons(ui: &Ui, view: &mut ViewState, left: Pos2) {
    let label = match view.preview_zoom {
        None => "Fit".to_owned(),
        Some(zoom) => format!("{:.0}%", zoom * 100.0),
    };
    let buttons = [
        ("−", 22.0, "Zoom out"),
        (label.as_str(), 44.0, "Fit the canvas"),
        ("+", 22.0, "Zoom in"),
    ];
    let mut x = left.x;
    for (index, (text, width, tip)) in buttons.into_iter().enumerate() {
        let button = Rect::from_min_size(Pos2::new(x, left.y - 12.0), Vec2::new(width, 24.0));
        x += width + 2.0;
        let response = ui.interact(button, ui.scope_id().with(("zoom", index)), Sense::click());
        let ink = if response.hovered() {
            tokens::TEXT
        } else {
            tokens::MUTED
        };
        ui.painter().text(
            button.center(),
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(if index == 1 { 12.0 } else { 16.0 }),
            ink,
        );
        if response.on_hover_text(tip).clicked() {
            match index {
                1 => view.preview_zoom = None,
                _ => {
                    let up = index == 2;
                    if let Some(step) = cutlass_editor::preview_zoom_step(view.preview_scale, up) {
                        view.preview_zoom = Some(step);
                    }
                }
            }
        }
    }
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
    let transition = shell.selected_transition();
    let project = shell.inspection().is_none() && transition.is_none();
    // The project's details keep Modify in a band at the foot.
    let body = if project && shell.draft().is_some() {
        let (body, foot) = rect.split_top_bottom_at_y(rect.bottom() - PROJECT_FOOT);
        ui.painter()
            .rect_filled(foot, CornerRadius::ZERO, tokens::PROJECT_BAR);
        let button = Rect::from_min_size(
            foot.right_center() - Vec2::new(16.0 + 76.0, 15.0),
            Vec2::new(76.0, 30.0),
        );
        let mut foot_ui = ui.new_child(
            egui::UiBuilder::new()
                .id_salt("project-foot")
                .max_rect(foot),
        );
        let response = foot_ui.place(
            button,
            egui::Button::new(
                RichText::new("Modify")
                    .size(tokens::TYPE_SIZE + 1.0)
                    .color(tokens::TEXT),
            )
            .fill(tokens::MENU_HOVER)
            .corner_radius(CornerRadius::same(tokens::EXPORT_RADIUS)),
        );
        if response
            .on_hover_text("Change the project settings")
            .clicked()
        {
            view.project_dialog = shell.project_settings();
        }
        body
    } else {
        rect
    };
    let content_width = body.width() - 32.0;
    // A clip's or a transition's tabs sit in a header band that stays put
    // while the body under it scrolls.
    let inspection = shell.inspection();
    let body = if transition.is_some() || inspection.is_some() {
        let band = ui.painter().add(egui::Shape::Noop);
        let inner = Rect::from_min_max(
            body.min + Vec2::new(16.0, 0.0),
            body.max - Vec2::new(16.0, 0.0),
        );
        let mut header = ui.new_child(
            egui::UiBuilder::new()
                .id_salt("inspector-header")
                .max_rect(inner),
        );
        header.set_clip_rect(body);
        header.add_space(6.0);
        if transition.is_some() {
            inspector::tabs(&mut header, &["Transition"], &mut 0);
        } else if let Some(inspection) = &inspection {
            inspector::clip_tabs(shell, &mut header, view, inspection);
        }
        let bottom = header.min_rect().bottom().clamp(body.top(), body.bottom());
        let header_rect = Rect::from_min_max(body.min, egui::pos2(body.right(), bottom));
        ui.painter().set(
            band,
            egui::Shape::rect_filled(header_rect, CornerRadius::ZERO, tokens::PROJECT_BAR),
        );
        ui.painter().hline(
            header_rect.x_range(),
            bottom,
            egui::Stroke::new(1.0, inspector::TAB_LINE),
        );
        Rect::from_min_max(egui::pos2(body.left(), bottom + 1.0), body.max)
    } else {
        body
    };
    // The body scrolls, so the bar sits on the panel edge; the content
    // keeps 16 points in from each side.
    let mut details = ui.new_child(egui::UiBuilder::new().max_rect(body));
    details.set_clip_rect(body);
    ui::Scroll::vertical("inspector")
        .content_margin(egui::Margin::symmetric(16, 0))
        .show(&mut details, |ui| {
            ui.set_width(content_width);
            if let Some(bar) = &transition {
                ui.add_space(6.0);
                inspector::transition_panel(shell, ui, view, bar);
            } else if let Some(inspection) = &inspection {
                ui.add_space(6.0);
                inspector::clip_panel(shell, ui, view, inspection);
            } else {
                ui.add_space(14.0);
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

/// The band at the foot of the project's details that holds Modify.
const PROJECT_FOOT: f32 = 56.0;

/// The project's settings as a list; Modify opens the dialog that changes
/// them. Name and background stay editable here.
fn project_facts(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, width: f32) {
    let Some(settings) = shell.project_settings() else {
        fact(ui, "Name", &shell.project_name(), width);
        return;
    };
    let stored = settings.name.clone();
    if let Some(name) = name_field(ui, &mut view.name_buffer, &stored)
        && let Err(error) = shell.set_project_name(&name)
    {
        shell.notice_error(&error);
    }
    let saved = shell
        .project_folder()
        .map_or_else(|| "Not saved".to_owned(), short_folder);
    fact(ui, "Saved", &saved, width);
    fact(
        ui,
        "Color space",
        &cutlass_editor::color_space_name(settings.color_space),
        width,
    );
    fact(ui, "Free layer", turned(settings.free_layer), width);
    ui.add_space(6.0);
    let span = ui.available_rect_before_wrap().x_range();
    let y = ui.cursor().min.y;
    ui.painter()
        .hline(span, y, egui::Stroke::new(1.0, tokens::LANE_LINE));
    ui.add_space(12.0);
    let ratio = cutlass_editor::PROJECT_RATIOS
        .iter()
        .find(|(ratio, _)| *ratio == settings.ratio)
        .map_or_else(
            || ratio_label(&settings.ratio),
            |(_, name)| (*name).to_owned(),
        );
    fact(ui, "Ratio", &ratio, width);
    let resolution = cutlass_editor::PROJECT_RESOLUTIONS
        .iter()
        .find(|(side, _)| *side == settings.short_side)
        .map_or("Adapted", |(_, name)| name);
    let canvas = shell
        .draft()
        .map(|draft| (draft.canvas_config.width, draft.canvas_config.height));
    let resolution = match canvas {
        Some((w, h)) => format!("{resolution} · {w}×{h}"),
        None => resolution.to_owned(),
    };
    fact(ui, "Resolution", &resolution, width);
    fact(
        ui,
        "Frame rate",
        &cutlass_editor::rate_label(settings.rate),
        width,
    );
    if shell.drop_frame_available() {
        fact(ui, "Drop frame", turned(shell.drop_frame_enabled()), width);
    }
    let background = shell
        .draft()
        .and_then(|draft| draft.canvas_config.background.clone())
        .unwrap_or_default();
    if let Some(color) = color_field(ui, "Background", &mut view.background_buffer, &background)
        && let Err(error) = shell.set_canvas_background(&color)
    {
        shell.notice_error(&error);
    }
}

fn turned(on: bool) -> &'static str {
    if on { "Turned on" } else { "Turned off" }
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

/// The label column of the project's details.
const FACT_LABEL: f32 = 120.0;
/// One row of the project's details.
const FACT_ROW: f32 = 30.0;
const FACT_SIZE: f32 = 14.0;

fn fact_label(ui: &mut Ui, label: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(FACT_LABEL, FACT_ROW), Sense::hover());
    ui.painter().text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(FACT_SIZE),
        tokens::MUTED,
    );
}

/// A detail typed in place: plain text until it is clicked.
fn fact_field(
    ui: &mut Ui,
    label: &str,
    text: &mut String,
    stored: &str,
    hint: &str,
) -> Option<String> {
    let mut edited = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 12.0;
        fact_label(ui, label);
        let response = ui.add(
            egui::TextEdit::singleline(text)
                .frame(egui::Frame::NONE)
                .margin(egui::Margin::symmetric(0, 7))
                .hint_text(RichText::new(hint).color(tokens::FAINT))
                .text_color(tokens::TEXT)
                .font(FontId::proportional(FACT_SIZE))
                .desired_width(ui.available_width()),
        );
        if commits(ui, &response) {
            edited = Some(text.clone());
        }
        if !response.has_focus() {
            *text = stored.to_owned();
        }
    });
    edited
}

fn name_field(ui: &mut Ui, text: &mut String, stored: &str) -> Option<String> {
    fact_field(ui, "Name", text, stored, "Untitled")
}

fn fact(ui: &mut Ui, label: &str, value: &str, row_width: f32) {
    let value_width = (row_width - FACT_LABEL - 12.0).max(40.0);
    ui.horizontal(|ui| {
        ui.set_max_width(row_width);
        ui.spacing_mut().item_spacing.x = 12.0;
        fact_label(ui, label);
        let mut job = egui::text::LayoutJob::simple(
            value.to_owned(),
            FontId::proportional(FACT_SIZE),
            tokens::TEXT,
            value_width,
        );
        job.wrap = egui::text::TextWrapping::truncate_at_width(value_width);
        let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(value_width, FACT_ROW), Sense::hover());
        ui.painter().galley(
            rect.left_center() + Vec2::new(0.0, -galley.size().y / 2.0),
            galley,
            tokens::TEXT,
        );
        response.on_hover_text(value);
    });
}

fn ratio_label(ratio: &str) -> String {
    if ratio.eq_ignore_ascii_case("original") {
        "Original".to_owned()
    } else {
        ratio.to_owned()
    }
}

/// A hex color typed in place; empty is clear.
fn color_field(ui: &mut Ui, label: &str, text: &mut String, stored: &str) -> Option<String> {
    fact_field(ui, label, text, stored, "Clear").filter(|value| value != stored)
}

fn timeline_panel(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    panel(ui, rect);
    shell.chrome.timeline = true;
    let toolbar = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::TOOLBAR_HEIGHT));
    paint_toolbar(shell, ui, view, toolbar);
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
    let meter = Rect::from_min_max(
        Pos2::new(body.right() - tokens::METER_WIDTH, body.top()),
        body.max,
    );
    let body = Rect::from_min_max(body.min, Pos2::new(meter.left(), body.bottom()));
    ruler_and_lanes(shell, ui, view, body);
    paint_meter(ui, meter, shell.meter());
    paint_clip_menu(shell, ui, view);
    paint_gap_menu(shell, ui);
    paint_track_menu(shell, ui, view);
    let handle = Rect::from_min_size(
        Pos2::new(rect.min.x, rect.min.y - tokens::SPLIT),
        Vec2::new(rect.width(), tokens::SPLIT),
    );
    let window_height = ui.ctx().content_rect().height();
    drag_gap(ui, handle, |delta| {
        shell.drag_timeline(delta.y, window_height)
    });
}

fn paint_toolbar(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    let inner = rect.shrink2(Vec2::new(10.0, 4.0));
    // Snap through zoom keep to the right edge, apart from the edit tools,
    // which scroll sideways in the room left of them.
    let right = Rect::from_min_max(
        Pos2::new(inner.right() - tools_width(TOOLS_RIGHT), inner.top()),
        inner.max,
    );
    let left = Rect::from_min_max(
        inner.min,
        Pos2::new((right.left() - TOOLS_GAP).max(inner.left()), inner.bottom()),
    );
    let mut bar = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(left)
            .layout(Layout::left_to_right(Align::Center)),
    );
    ui::Scroll::horizontal("timeline-toolbar")
        .auto_shrink([false, true])
        .show(&mut bar, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = TOOL_SPACING;
                if shell.compound_depth() > 0 {
                    if ui
                        .add(ui::Button::new("‹ Back").text_size(12.0))
                        .on_hover_text("Back to the timeline the compound clip is on")
                        .clicked()
                        && let Err(error) = shell.close_compound()
                    {
                        shell.notice_error(&error);
                    }
                    ui.label(RichText::new(COMPOUND_TITLE).size(12.0).color(tokens::TEXT));
                    toolbar_separator(ui);
                }
                paint_tool_picker(shell, ui);
                toolbar_separator(ui);
                let picked = if shell.selection().is_empty() {
                    None
                } else {
                    shell.selected_track_kind()
                };
                let tools = left_tools(picked.as_ref());
                paint_tools(shell, ui, view, &tools);
            });
        });
    let mut pinned = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(right)
            .layout(Layout::left_to_right(Align::Center)),
    );
    pinned.spacing_mut().item_spacing.x = TOOL_SPACING;
    paint_tools(shell, &mut pinned, view, TOOLS_RIGHT);
}

/// The space between two toolbar items, and between the two groups.
const TOOL_SPACING: f32 = 4.0;
const TOOLS_GAP: f32 = 16.0;
/// A tool button, a separator, and the linkage chevron, as drawn.
const TOOL_BUTTON: f32 = 28.0;
const TOOL_SEPARATOR: f32 = 13.0;
const LINKAGE_CHEVRON: f32 = 16.0;

/// The width `items` take in a row: each button, separator and the
/// linkage chevron, with the spacing between them.
fn tools_width(items: &[Option<(&str, &str)>]) -> f32 {
    let mut widths: Vec<f32> = Vec::new();
    for item in items {
        match item {
            None => widths.push(TOOL_SEPARATOR),
            Some((name, _)) => {
                widths.push(TOOL_BUTTON);
                if *name == "linkage" {
                    widths.push(LINKAGE_CHEVRON);
                }
            }
        }
    }
    let gaps = widths.len().saturating_sub(1) as f32 * TOOL_SPACING;
    widths.iter().sum::<f32>() + gaps
}

fn paint_tools(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    items: &[Option<(&str, &str)>],
) {
    for item in items {
        match item {
            None => toolbar_separator(ui),
            Some((name, tip)) => {
                let enabled = tool_enabled(shell, name);
                let active = match *name {
                    "split" => shell.blade(),
                    "beat" => shell.beating(),
                    "linkage" => shell.linkage_enabled(),
                    "magnet" => shell.magnet_enabled(),
                    "snap" => shell.snap_enabled(),
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
                if *name == "linkage" {
                    linkage_menu(shell, ui, view);
                }
            }
        }
    }
}

fn toolbar_separator(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(TOOL_SEPARATOR, 28.0), Sense::hover());
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
    ui.add_enabled(
        enabled || active,
        ui::Button::icon(name)
            .on(active)
            .min_size(Vec2::splat(TOOL_BUTTON))
            .tip(tip),
    )
}

fn tool_enabled(shell: &Shell, name: &str) -> bool {
    match name {
        "undo" => shell.can_undo(),
        "redo" => shell.can_redo(),
        "select" | "delete" | "zoomIn" | "zoomOut" | "zoomFit" | "linkage" | "magnet" | "snap" => {
            true
        }
        "split" | "marker" | "beat" | "splitAt" | "trimLeft" | "trimRight" => shell.has_clip(),
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
        "splitAt" => shell.split_playhead(),
        "trimLeft" => shell.delete_side(TrimEdge::Start),
        "trimRight" => shell.delete_side(TrimEdge::End),
        "split" => {
            shell.toggle_blade();
            Ok(())
        }
        "marker" => shell.mark_playhead(),
        "beat" => {
            shell.toggle_beat();
            Ok(())
        }
        "linkage" => shell.toggle_linkage(),
        "magnet" => shell.toggle_magnet(),
        "snap" => {
            shell.toggle_snap();
            Ok(())
        }
        "freeze" => shell.freeze_frame(),
        "reverse" => shell.toggle_reverse(),
        "mirror" => shell.toggle_mirror(),
        "rotate" => shell.turn_clockwise(),
        "crop" => shell.toggle_crop(),
        "transcribe" => shell.transcribe(),
        "splitScenes" => shell.split_scenes(),
        _ => Ok(()),
    };
    if let Err(error) = result {
        shell.notice_error(&error);
    }
}

/// The arrow beside Linkage: a menu that opens the Linkage settings dialog.
/// The tool picker: the tool in use, Select or Split, and a chevron that
/// opens the two with their keys.
fn paint_tool_picker(shell: &mut Shell, ui: &mut Ui) {
    let blade = shell.blade();
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(TOOL_BUTTON + LINKAGE_CHEVRON, TOOL_BUTTON),
        Sense::click(),
    );
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "Tools"));
    let response = response.on_hover_text(if blade { "Split tool" } else { "Select tool" });
    let id = egui::Id::unique("tool-picker");
    let open = egui::Popup::is_id_open(ui.ctx(), id);
    if response.hovered() || open {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(6), tokens::MENU_HOVER);
    }
    let icon = Rect::from_center_size(
        Pos2::new(rect.left() + TOOL_BUTTON / 2.0, rect.center().y),
        Vec2::splat(16.0),
    );
    let _ = icons::paint(
        ui.painter(),
        icon,
        if blade { "blade" } else { "select" },
        tokens::TEXT,
    );
    let c = Pos2::new(rect.right() - LINKAGE_CHEVRON / 2.0, rect.center().y);
    let (tip, wing) = if open { (-2.0, 2.0) } else { (2.0, -2.0) };
    ui.painter().add(egui::Shape::line(
        vec![
            c + Vec2::new(-3.5, wing),
            c + Vec2::new(0.0, tip),
            c + Vec2::new(3.5, wing),
        ],
        egui::Stroke::new(1.5, tokens::MENU_CHEVRON),
    ));
    egui::Popup::menu(&response)
        .id(id)
        .width(200.0)
        .gap(4.0)
        .frame(
            egui::Frame::new()
                .fill(tokens::MENU)
                .corner_radius(CornerRadius::same(tokens::EXPORT_RADIUS))
                .inner_margin(egui::Margin::same(4)),
        )
        .show(|ui| {
            ui.set_width(192.0);
            for (pick_blade, icon, label, action) in [
                (false, "select", "Select", "select_tool"),
                (true, "blade", "Split", "blade_tool"),
            ] {
                let (row, row_response) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 32.0), Sense::click());
                row_response
                    .widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, label));
                if row_response.hovered() || blade == pick_blade {
                    ui.painter()
                        .rect_filled(row, CornerRadius::same(6), tokens::MENU_HOVER);
                }
                let _ = icons::paint(
                    ui.painter(),
                    Rect::from_center_size(
                        row.left_center() + Vec2::new(18.0, 0.0),
                        Vec2::splat(16.0),
                    ),
                    icon,
                    tokens::TEXT,
                );
                ui.painter().text(
                    row.left_center() + Vec2::new(36.0, 0.0),
                    Align2::LEFT_CENTER,
                    label,
                    FontId::proportional(tokens::TYPE_SIZE),
                    tokens::TEXT,
                );
                let keys = shell.settings().keys_for(action);
                if !keys.is_empty() {
                    crate::menu::paint_keys(
                        ui.ctx(),
                        ui.painter(),
                        row.right_center() - Vec2::new(10.0, 0.0),
                        &crate::menu::shortcut_for(keys),
                        tokens::MENU_KEY,
                    );
                }
                if row_response.clicked() {
                    shell.pick_tool(pick_blade);
                    ui.close();
                }
            }
        });
}

fn linkage_menu(shell: &Shell, ui: &mut Ui, view: &mut ViewState) {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(LINKAGE_CHEVRON, 28.0), Sense::click());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "Linkage options"));
    let response = response.on_hover_text("Linkage settings");
    let open = egui::Popup::is_id_open(ui.ctx(), egui::Id::unique("linkage-menu"));
    if response.hovered() || open {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(6), tokens::MENU_HOVER);
    }
    let c = rect.center();
    let (tip, wing) = if open { (-2.0, 2.0) } else { (2.0, -2.0) };
    ui.painter().add(egui::Shape::line(
        vec![
            c + Vec2::new(-3.5, wing),
            c + Vec2::new(0.0, tip),
            c + Vec2::new(3.5, wing),
        ],
        egui::Stroke::new(1.5, tokens::MENU_CHEVRON),
    ));
    egui::Popup::menu(&response)
        .id(egui::Id::unique("linkage-menu"))
        .width(180.0)
        .gap(4.0)
        .frame(
            egui::Frame::new()
                .fill(tokens::MENU)
                .corner_radius(CornerRadius::same(tokens::EXPORT_RADIUS))
                .inner_margin(egui::Margin::same(4)),
        )
        .show(|ui| {
            ui.set_width(172.0);
            let (row, row_response) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::click());
            row_response.widget_info(|| {
                egui::WidgetInfo::labeled(egui::Role::Button, true, "Linkage settings")
            });
            if row_response.hovered() {
                ui.painter()
                    .rect_filled(row, CornerRadius::same(6), tokens::MENU_HOVER);
            }
            ui.painter().text(
                row.left_center() + Vec2::new(10.0, 0.0),
                Align2::LEFT_CENTER,
                "Linkage settings",
                FontId::proportional(tokens::TYPE_SIZE),
                tokens::TEXT,
            );
            if row_response.clicked() {
                view.linkage_kinds = shell.linkage_kinds();
                ui.close();
            }
        });
}

/// One kind's switch in the linkage settings.
type KindField = fn(&mut cutlass_model::KindSwitches) -> &mut bool;

/// The kinds of clip that move or go with their main-track clip, as boxes
/// to tick, in the order the timeline's lanes name them.
const LINKAGE_KINDS: [(&str, KindField); 7] = [
    ("Text", |kinds| &mut kinds.text),
    ("Effects", |kinds| &mut kinds.effect),
    ("Stickers", |kinds| &mut kinds.sticker),
    ("Filters", |kinds| &mut kinds.filter),
    ("Adjust", |kinds| &mut kinds.adjust),
    ("Overlay", |kinds| &mut kinds.overlay),
    ("Audio", |kinds| &mut kinds.audio),
];

/// Choose which kinds follow their main-track clip. Done keeps the choice;
/// Cancel, Escape and the close button leave it as it was.
fn linkage_dialog(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let Some(mut kinds) = view.linkage_kinds else {
        return;
    };
    let mut answer = None;
    let ((), closed) = widgets::dialog(ui, "Linkage settings", |ui| {
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.set_max_width(372.0);
                ui.label(
                    "With linkage on, the ticked kinds move, split and are deleted with \
                     the main-track clip they sit on. Audio includes sound effects.",
                );
                ui.add_space(8.0);
                egui::Grid::new("linkage-kinds")
                    .num_columns(2)
                    .spacing(Vec2::new(24.0, 10.0))
                    .show(ui, |ui| {
                        for (index, (label, field)) in LINKAGE_KINDS.iter().enumerate() {
                            ui.add(ui::Check::new(field(&mut kinds), *label));
                            if index % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let mut all = kinds == cutlass_model::KindSwitches::ALL;
                    if ui.add(ui::Check::new(&mut all, "Select all")).changed() {
                        for (_, field) in &LINKAGE_KINDS {
                            *field(&mut kinds) = all;
                        }
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.add(ui::Button::primary("Done")).clicked() {
                            answer = Some(true);
                        }
                        if ui.add(ui::Button::new("Cancel")).clicked() {
                            answer = Some(false);
                        }
                    });
                });
            });
        });
    });
    view.linkage_kinds = Some(kinds);
    let cancelled = closed || ui.input(|input| input.key_pressed(egui::Key::Escape));
    match answer {
        Some(true) => {
            view.linkage_kinds = None;
            if let Err(error) = shell.set_linkage_kinds(kinds) {
                shell.notice_error(&error);
            }
        }
        Some(false) => view.linkage_kinds = None,
        None if cancelled => view.linkage_kinds = None,
        None => {}
    }
}

/// What the toolbar says while a compound clip is open.
const COMPOUND_TITLE: &str = "Compound clip";

/// The tools at the toolbar's left, after the tool picker, whether or not
/// a clip is picked.
const TOOLS_LEFT: &[Option<(&str, &str)>] = &[
    Some(("undo", "Undo")),
    Some(("redo", "Redo")),
    None,
    Some(("splitAt", "Split at the playhead")),
    Some(("trimLeft", "Delete left of the playhead")),
    Some(("trimRight", "Delete right of the playhead")),
    Some(("delete", "Delete")),
    None,
    Some(("marker", "Add marker at the playhead")),
];

/// The tools left of the pinned group: [`TOOLS_LEFT`], and when a clip is
/// picked, a separator and the tools for its lane's kind: [`TOOLS_AUDIO`]
/// for a sound clip, [`TOOLS_CLIP`] for the rest.
fn left_tools(
    picked: Option<&cutlass_model::TrackType>,
) -> Vec<Option<(&'static str, &'static str)>> {
    let mut tools = TOOLS_LEFT.to_vec();
    if let Some(kind) = picked {
        tools.push(None);
        tools.extend_from_slice(match kind {
            cutlass_model::TrackType::Audio | cutlass_model::TrackType::TextToVideoTextAudio => {
                TOOLS_AUDIO
            }
            _ => TOOLS_CLIP,
        });
    }
    tools
}

/// The tools a picked audio clip adds. The last three are drawn dimmed
/// until they are built.
const TOOLS_AUDIO: &[Option<(&str, &str)>] = &[
    Some(("transcribe", "Transcribe")),
    Some(("beat", "Beat")),
    Some(("normalize", "Normalize loudness (later)")),
    Some(("enhanceVoice", "Enhance voice (later)")),
    Some(("reduceNoise", "Reduce noise (later)")),
];

/// The tools a picked clip adds after them.
const TOOLS_CLIP: &[Option<(&str, &str)>] = &[
    Some(("freeze", "Freeze frame")),
    Some(("reverse", "Reverse")),
    Some(("mirror", "Mirror")),
    Some(("rotate", "Rotate")),
    Some(("crop", "Crop")),
    Some(("transcribe", "Transcribe")),
    Some(("splitScenes", "Split scenes")),
];

/// Snap through zoom, kept at the toolbar's right edge.
const TOOLS_RIGHT: &[Option<(&str, &str)>] = &[
    Some(("snap", "Snap")),
    Some(("magnet", "Magnet")),
    Some(("linkage", "Linkage")),
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
    let painter = notice_painter(ui.ctx());
    painter.rect_filled(
        rect,
        CornerRadius::same(99),
        Color32::WHITE.gamma_multiply(alpha),
    );
    painter.galley(
        rect.left_center() + Vec2::new(12.0, -galley.size().y / 2.0),
        galley,
        ink,
    );
}

/// The toast's own layer, above every panel, ruler, menu and drag: the
/// tooltip order, moved to the top of it each frame, with the whole window as
/// its clip so nothing painted later covers it.
fn notice_painter(ctx: &egui::Context) -> egui::Painter {
    let layer = notice_layer();
    ctx.move_to_top(layer);
    ctx.layer_painter(layer).with_clip_rect(ctx.content_rect())
}

fn notice_layer() -> egui::LayerId {
    egui::LayerId::new(egui::Order::Tooltip, egui::Id::unique("notice-toast"))
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

/// Where this frame's key diamonds keep their hit rects.
fn key_hits_id() -> egui::Id {
    egui::Id::unique("clip-key-hits")
}

fn ruler_and_lanes(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState, rect: Rect) {
    ui.data_mut(|data| data.remove::<Vec<Rect>>(key_hits_id()));
    let ruler = Rect::from_min_size(rect.min, Vec2::new(rect.width(), tokens::RULER_HEIGHT));
    let tracks = Rect::from_min_max(ruler.left_bottom(), rect.max);
    let lanes_left = tracks.left() + tokens::TRACK_HEAD + tokens::COVER_WIDTH;
    let lane_area = Rect::from_min_max(Pos2::new(lanes_left, tracks.top()), tracks.max);
    shell.prepare_lane(lane_area.width().max(1.0));
    let (rate, drop_frame) = ruler_rate(shell);
    paint_marks(ui, shell, ruler, lanes_left);
    paint_ticks(
        ui,
        ruler,
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        rate,
        drop_frame,
    );
    // Escape calls off a clip drag: the clip goes back and the release does nothing.
    if shell.clip_dragging() && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
        shell.escape_clip_drag();
    }
    let snap_at = shell.snap_line();
    let rows = shell.lanes();
    let kinds: Vec<(bool, f32)> = rows
        .iter()
        .map(|lane| (lane.main, lane_height(view, lane)))
        .collect();
    // A clip that just landed by a paste or from the library has its lane
    // scrolled into view.
    if let Some(landed) = shell.take_reveal()
        && let Some(index) = rows
            .iter()
            .position(|lane| lane.clips.iter().any(|clip| clip.id == landed))
    {
        (view.lanes_up, view.lanes_down) =
            reveal_row(&kinds, lane_area, index, view.lanes_up, view.lanes_down);
        ui.ctx().request_repaint();
    }
    // Main stays put. The lanes above it and the lanes under it each scroll
    // in their own band, and one wheel runs through the two in turn.
    let still = lane_layout(&kinds, lane_area, view.lanes_up, view.lanes_down);
    let scrolls = still.up_max > 0.0 || still.down_max > 0.0;
    if scrolls {
        let (up, down) = chain_scroll(
            &still,
            view.lanes_up,
            view.lanes_down,
            lane_wheel(ui, tracks),
        );
        view.lanes_up = up;
        view.lanes_down = down;
        if shell.clip_dragging()
            && let Some(pos) = ui.input(|input| input.pointer.interact_pos())
        {
            // Near the panel's top or bottom the drag scrolls as the wheel
            // does; just above main it brings the lanes above main back down.
            let edge = drag_scroll_step(lane_area, pos.y);
            if edge != 0.0 {
                let (up, down) = chain_scroll(&still, view.lanes_up, view.lanes_down, edge);
                view.lanes_up = up;
                view.lanes_down = down;
            } else if pos.y <= still.upper.bottom() {
                view.lanes_up += drag_scroll_step(still.upper, pos.y).min(0.0);
            }
        }
    }
    view.lanes_up = view.lanes_up.clamp(0.0, still.up_max);
    view.lanes_down = view.lanes_down.clamp(0.0, still.down_max);
    let laid = lane_layout(&kinds, lane_area, view.lanes_up, view.lanes_down);
    // Rows drawn whole, cut to their band; the parts that show are what the
    // pointer finds.
    let drawn = laid.rows.clone();
    let boxes = laid.shown();
    let main_index = rows.iter().position(|lane| lane.main);
    ui.data_mut(|data| data.insert_temp(lane_sheet_id(), ui.clip_rect().intersect(tracks)));
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
        let (Some(row), Some(band)) = (drawn.get(index).copied(), laid.bands.get(index)) else {
            continue;
        };
        // A band cuts its rows top and bottom only: the head on the left
        // is part of the row.
        let cut = clip
            .intersect(tracks)
            .intersect(Rect::from_x_y_ranges(tracks.x_range(), band.y_range()));
        let folded = laid.peek && main_index.is_some_and(|main| index > main);
        if cut.height() <= 0.0 || folded {
            continue;
        }
        ui.set_clip_rect(cut);
        paint_lane(shell, ui, lane, row, lanes_left, index, &boxes);
        lane_head(shell, ui, view, lane, row, index);
        resize_lane(
            ui,
            view,
            lane,
            row,
            index,
            rows.iter().position(|lane| lane.main),
        );
    }
    ui.set_clip_rect(clip.intersect(tracks));
    if let Some(main) = main_index
        && let Some(row) = drawn.get(main).copied()
    {
        // One line across the heads and one across the lanes, the bare
        // cover column between them, where the lanes above main end and
        // main's row begins.
        if main > 0 {
            let y = row.top() - LANE_GAP / 2.0;
            let heads = head_rect(row).x_range();
            for span in [heads, egui::Rangef::new(lanes_left, tracks.right())] {
                ui.painter()
                    .hline(span, y, egui::Stroke::new(1.0, tokens::LANE_LINE));
            }
        }
        if laid.peek {
            let y = row.bottom() - tokens::MAIN_CLIP_INSET / 2.0;
            paint_folded_lanes(shell, ui, &rows[main + 1..], y, lanes_left);
        }
    }
    paint_new_lane(shell, ui, &boxes, lanes_left);
    ui.set_clip_rect(clip);
    // The heads and the cover column are parted by one line, from the
    // ruler's top to the panel's foot.
    ui.painter().vline(
        tracks.left() + tokens::TRACK_HEAD,
        rect.y_range(),
        egui::Stroke::new(1.0, tokens::GAP),
    );
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
    // The ruler first, so the playhead's head, on top of it, takes its own
    // hover and drag.
    let ruler_hit = ui.interact(
        lane_ruler(ruler, lanes_left),
        ui.scope_id().with("ruler"),
        Sense::click_and_drag(),
    );
    let mut scrubbing = false;
    if (lanes_left..=tracks.right()).contains(&playhead_x) {
        let mark = Rect::from_min_size(
            Pos2::new(playhead_x - 5.5, ruler.top() + 2.0),
            Vec2::new(11.0, 14.0),
        );
        let response = ui.interact(mark, ui.scope_id().with("playhead"), Sense::drag());
        // The line drags too, over the lanes and the clips on them.
        let line = Rect::from_min_max(
            Pos2::new(playhead_x - PLAYHEAD_GRAB, mark.bottom()),
            Pos2::new(playhead_x + PLAYHEAD_GRAB, tracks.bottom()),
        );
        // A key diamond under the pointer keeps its own click and drag; the
        // line yields there unless it is already being dragged.
        let body_id = ui.scope_id().with("playhead-line");
        let keys: Vec<Rect> = ui
            .data(|data| data.get_temp(key_hits_id()))
            .unwrap_or_default();
        let on_key = ui
            .input(|input| input.pointer.hover_pos())
            .is_some_and(|pos| keys.iter().any(|key| key.contains(pos)));
        let body = if on_key && !ui.ctx().is_being_dragged(body_id) {
            None
        } else {
            Some(ui.interact(line, body_id, Sense::drag()))
        };
        for grab in std::iter::once(&response).chain(body.as_ref()) {
            if grab.hovered() || grab.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            scrubbing |= grab.dragged();
            if grab.dragged()
                && let Some(pointer) = grab.interact_pointer_pos()
            {
                let x = f64::from(pointer.x - lanes_left).max(0.0);
                if let Err(error) = shell.seek_ruler(x) {
                    shell.notice_error(&error);
                }
            }
        }
        ui.painter()
            .rect_filled(mark, CornerRadius::same(3), tokens::PLAYHEAD);
        ui.painter().vline(
            playhead_x,
            mark.bottom()..=tracks.bottom(),
            egui::Stroke::new(1.0, tokens::PLAYHEAD),
        );
        response.on_hover_text("Playhead");
    }
    shell.set_scrubbing(scrubbing || ruler_hit.dragged());
    if ruler_hit.dragged()
        && let Some(pointer) = ruler_hit.interact_pointer_pos()
    {
        let x = f64::from(pointer.x - lanes_left).max(0.0);
        if let Err(error) = shell.seek_ruler(x) {
            shell.notice_error(&error);
        }
    }
    paint_rendered_spans(shell, ui, lanes_left, ruler);
    let marker_clicked = paint_ruler_markers(shell, ui, lanes_left, ruler);
    if ruler_hit.clicked()
        && let Some(pointer) = ruler_hit.interact_pointer_pos()
    {
        let x = f64::from(pointer.x - lanes_left).max(0.0);
        if !marker_clicked && let Err(error) = shell.seek_ruler(x) {
            shell.notice_error(&error);
        }
    }
    release_library_drop(shell, ui, tracks, lanes_left, &boxes);
    pan_timeline(shell, ui, lane_area, scrolls);
    zoom_timeline(shell, ui, lane_area);
}

/// Where this frame's track sheet is kept, for painters inside one band.
fn lane_sheet_id() -> egui::Id {
    egui::Id::unique("lane-sheet")
}

/// The lanes laid out around a main lane that stays put.
#[derive(Clone, Debug, PartialEq)]
struct LaneLayout {
    /// Each lane's whole row, scrolled.
    rows: Vec<Rect>,
    /// The band each row shows in: the one above main, main's own row, or
    /// the one under main.
    bands: Vec<Rect>,
    /// The band above main and the band under it.
    upper: Rect,
    lower: Rect,
    /// How far each band may scroll.
    up_max: f32,
    down_max: f32,
    /// Main has sunk to the sheet's foot and the lanes under it are folded
    /// to a strip, drawn as one thin line per clip.
    peek: bool,
}

/// The strip left under main once it has sunk to the sheet's foot. The
/// folded lanes' line is drawn in main's own row, under its clip.
const PEEK: f32 = 4.0;
/// How far either side of the playhead line a drag picks it up.
const PLAYHEAD_GRAB: f32 = 4.0;

impl LaneLayout {
    /// The part of each row that shows in its band. A row scrolled out of
    /// sight keeps its order as a flat rect on its band's edge.
    fn shown(&self) -> Vec<Rect> {
        self.rows
            .iter()
            .zip(&self.bands)
            .map(|(row, band)| {
                // Folded lanes are only a line: nothing there to click or drop on.
                let folded = self.peek && *band == self.lower;
                let cut = row.intersect(*band);
                if cut.height() > 0.0 && !folded {
                    cut
                } else {
                    // A band scrolled out of the sheet may be empty.
                    let y = row
                        .center()
                        .y
                        .max(band.top())
                        .min(band.bottom().max(band.top()));
                    Rect::from_x_y_ranges(row.x_range(), y..=y)
                }
            })
            .collect()
    }
}

/// The scroll, as `(up, down)`, that shows row `index` whole in its band,
/// starting from the scroll the lanes have and moving them no further than
/// that takes. A row above main is brought down from the top or back out
/// from under main; a row under main is brought up from the foot, and
/// unfolded first when the lanes above main had folded it away. A row
/// taller than its band shows as much as fits.
fn reveal_row(kinds: &[(bool, f32)], area: Rect, index: usize, up: f32, down: f32) -> (f32, f32) {
    let main = kinds.iter().position(|(main, _)| *main);
    let above = main.is_some_and(|main| index < main);
    let (mut up, mut down) = (up, down);
    // Each pass settles one thing; a few reach any row.
    for _ in 0..24 {
        let laid = lane_layout(kinds, area, up, down);
        up = up.clamp(0.0, laid.up_max);
        down = down.clamp(0.0, laid.down_max);
        let (Some(row), Some(band)) = (laid.rows.get(index), laid.bands.get(index)) else {
            break;
        };
        if laid.peek && main.is_some_and(|main| index > main) {
            up = 0.0;
            continue;
        }
        // Main's band is only what shows of it, so main is held to the
        // sheet itself.
        let band = if main == Some(index) { &area } else { band };
        // A band too short for the row is given room first. Above main the
        // stack is let back down, then main is sunk; under main, main is
        // lifted back, then the stack is carried up.
        let short = row.height() - band.height();
        if short > 0.5 {
            let before = (up, down);
            if above {
                if down > 0.0 {
                    down = (down - short).max(0.0);
                } else {
                    up = (up + short).min(laid.up_max);
                }
            } else if up > 0.0 {
                // Main had sunk; lifting it gives the lanes under it room.
                up = (up - short).max(0.0);
            } else {
                down = (down + short).min(laid.down_max);
            }
            if (up, down) == before {
                break;
            }
            continue;
        }
        let lack = band.top() - row.top();
        let over = row.bottom() - band.bottom();
        let step = if lack > 0.5 {
            lack
        } else if over > 0.5 {
            -over
        } else {
            break;
        };
        // A row above main comes down as `up` grows; every other row
        // comes down as `down` shrinks.
        let before = (up, down);
        if above {
            let wanted = up + step;
            up = wanted.clamp(0.0, laid.up_max);
            // What the lanes above main cannot give, the whole stack does.
            down = (down - (wanted - up)).clamp(0.0, laid.down_max);
        } else {
            let wanted = down - step;
            down = wanted.clamp(0.0, laid.down_max);
            // What the stack cannot give, lifting main back does.
            up = (up - (wanted - down).max(0.0)).max(0.0);
        }
        if (up, down) == before {
            break;
        }
    }
    (up, down)
}

/// Lay the lanes out for rows given as (main, height).
///
/// While everything fits with a lane of room past the top and the bottom,
/// main sits in the middle, as far as that room allows, and nothing
/// scrolls. Once it does not fit, main rests just above the lanes under it,
/// which show whole while they take no more than half the space; the lanes
/// above main fill the rest from main up. `up` scrolls only the lanes above
/// main toward the top one, main staying put; `down` scrolls the whole
/// stack up, main with it, toward the bottom lane. Each is clamped to what
/// it can reach.
fn lane_layout(kinds: &[(bool, f32)], area: Rect, up: f32, down: f32) -> LaneLayout {
    const GAP: f32 = LANE_GAP;
    let room = lane_room();
    let row = |top: f32, height: f32| {
        Rect::from_min_size(
            Pos2::new(area.left(), top),
            Vec2::new(area.width().max(0.0), height),
        )
    };
    let Some(main) = kinds.iter().position(|(main, _)| *main) else {
        // No main lane: the rows stack from the top and scroll as one band.
        let total: f32 = kinds.iter().map(|(_, height)| height + GAP).sum();
        let down_max = (total + room - area.height()).max(0.0);
        let down = down.clamp(0.0, down_max);
        let mut top = area.top() - down;
        let rows = kinds
            .iter()
            .map(|(_, height)| {
                let rect = row(top, *height);
                top += height + GAP;
                rect
            })
            .collect::<Vec<_>>();
        return LaneLayout {
            bands: vec![area; rows.len()],
            rows,
            upper: Rect::from_x_y_ranges(area.x_range(), area.top()..=area.top()),
            lower: area,
            up_max: 0.0,
            down_max,
            peek: false,
        };
    };
    let main_height = kinds[main].1;
    let above: f32 = kinds[..main].iter().map(|(_, height)| height + GAP).sum();
    let under: f32 = kinds[main + 1..]
        .iter()
        .map(|(_, height)| height + GAP)
        .sum();
    let want_up = above + room;
    let want_down = under + room;
    let lowest = area.top() + want_up;
    let highest = area.bottom() - main_height - want_down;
    let main_top = if lowest <= highest {
        (area.center().y - main_height / 2.0).clamp(lowest, highest)
    } else {
        let spare = (area.height() - main_height).max(0.0);
        let down_band = want_down.min((spare - want_up).max(spare / 2.0)).max(0.0);
        area.top() + (spare - down_band)
    };
    let main_bottom = main_top + main_height;
    let up_max = (want_up - (main_top - area.top())).max(0.0);
    let down_max = (want_down - (area.bottom() - main_bottom)).max(0.0);
    let up = up.clamp(0.0, up_max);
    let down = down.clamp(0.0, down_max);
    // Scrolling the lanes above main first carries main down with them to
    // the sheet's foot, folding the lanes under it to a strip; past that,
    // main stays and they slide under it.
    let sink = up.min((area.bottom() - PEEK - main_bottom).max(0.0));
    let rest_top = main_top;
    // Scrolling down carries main, and everything around it, up.
    let main_top = main_top + sink - down;
    let main_bottom = main_bottom + sink - down;
    let upper = Rect::from_x_y_ranges(
        area.x_range(),
        area.top()..=(main_top - GAP / 2.0).max(area.top()),
    );
    let lower = Rect::from_x_y_ranges(
        area.x_range(),
        (main_bottom + GAP / 2.0).clamp(area.top(), area.bottom())..=area.bottom(),
    );
    let mut rows = vec![Rect::NOTHING; kinds.len()];
    let mut bands = vec![upper; kinds.len()];
    rows[main] = row(main_top, main_height);
    bands[main] = rows[main].intersect(area);
    let mut bottom = rest_top - down - GAP + up;
    for index in (0..main).rev() {
        let height = kinds[index].1;
        rows[index] = row(bottom - height, height);
        bottom -= height + GAP;
    }
    let mut top = main_bottom + GAP;
    for index in main + 1..kinds.len() {
        let height = kinds[index].1;
        rows[index] = row(top, height);
        bands[index] = lower;
        top += height + GAP;
    }
    let peek = main + 1 < kinds.len() && sink > 0.0 && lower.height() < kinds[main + 1].1;
    LaneLayout {
        rows,
        bands,
        upper,
        lower,
        up_max,
        down_max,
        peek,
    }
}

fn folded_color(kind: &cutlass_model::TrackType) -> Color32 {
    if matches!(kind, cutlass_model::TrackType::Audio) {
        tokens::CLIP_AUDIO_BAR
    } else {
        lane_color(kind)
    }
}

/// The lanes under main folded to a strip: one thin line, in the lane's
/// colour, along each clip's time, so it shows there is more below.
fn paint_folded_lanes(
    shell: &Shell,
    ui: &Ui,
    lanes: &[crate::shell::LaneView],
    y: f32,
    lanes_left: f32,
) {
    for lane in lanes {
        for clip in &lane.clips {
            let left = lane_x(
                lanes_left,
                shell.pixel_size(),
                shell.view_origin().0,
                clip.start.0,
            );
            let width = (clip.duration.0 as f32 * shell.pixel_size() as f32).max(2.0);
            ui.painter().line_segment(
                [Pos2::new(left, y), Pos2::new(left + width, y)],
                egui::Stroke::new(2.0, folded_color(&lane.kind)),
            );
        }
    }
}

/// Run one wheel step through the two scrolls. Scrolling down (`delta`
/// below zero) first brings the lanes above main back down to it, then
/// carries the whole stack up toward the lanes under main; scrolling up
/// first brings the stack back, then scrolls the lanes above main.
fn chain_scroll(layout: &LaneLayout, up: f32, down: f32, delta: f32) -> (f32, f32) {
    let up = up.clamp(0.0, layout.up_max);
    let down = down.clamp(0.0, layout.down_max);
    if delta > 0.0 {
        let back = delta.min(down);
        (up + delta - back, down - back)
    } else {
        let step = -delta;
        let back = step.min(up);
        (up - back, down + step - back)
    }
}

/// Rows sit this far apart: a hairline of the panel shows between them.
const LANE_GAP: f32 = 1.0;

/// The space kept past the top lane and the bottom one: a regular lane and
/// the gap beside it.
fn lane_room() -> f32 {
    row_height(RowSize::Thin) + LANE_GAP
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
/// How far to scroll a band while a clip drag is held near its top (above
/// zero) or bottom (below zero), so a lane out of view can be reached.
/// Faster the closer the pointer is to the edge; zero outside the band.
fn drag_scroll_step(tracks: Rect, y: f32) -> f32 {
    const ZONE: f32 = 32.0;
    const FASTEST: f32 = 12.0;
    let from_top = y - tracks.top();
    let from_bottom = tracks.bottom() - y;
    if from_top < ZONE {
        FASTEST * (1.0 - from_top.max(0.0) / ZONE)
    } else if from_bottom < ZONE {
        -FASTEST * (1.0 - from_bottom.max(0.0) / ZONE)
    } else {
        0.0
    }
}

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

/// What the id of an Edit effects submenu row starts with; the effect's
/// place in the clip's list follows.
const EDIT_EFFECT_ROW: &str = "edit-effects:";

fn paint_clip_menu(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
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
    // Two or more clips picked: the menu of rows that act on all of them.
    let entries = if shell.selection().len() > 1 {
        let mix = shell.selection_mix();
        crate::menu::multi_entries(mix.sound, mix.compound)
    } else {
        crate::menu::clip_entries_for(
            &kind,
            shell.recover_audio_enabled(),
            shell.compound_selected(),
        )
    };
    // Edit effects picks the clip's one effect, or opens a row for each
    // when it has several. The rows are made for this frame.
    let effects = shell.clip_effects();
    let effect_ids: Vec<String> = (0..effects.len())
        .map(|index| format!("{EDIT_EFFECT_ROW}{index}"))
        .collect();
    let effect_rows: Vec<crate::menu::Entry<'_>> = effects
        .iter()
        .zip(&effect_ids)
        .map(|(effect, id)| crate::menu::named(id, &effect.name))
        .collect();
    let with_effects;
    let entries: &[crate::menu::Entry<'_>] = if effect_rows.len() > 1 {
        with_effects = crate::menu::with_children(entries, "edit-effects", &effect_rows);
        &with_effects
    } else {
        entries
    };
    let enabled = |id: &str| match id {
        "edit-effects" => !effects.is_empty(),
        "sync" => shell.sync_enabled(),
        "edit-freeze" => shell.freeze_enabled(),
        "edit-reverse" => shell.reverse_enabled(),
        "edit-mirror" => shell.mirror_enabled(),
        "edit-rotate" => shell.rotate_enabled(),
        "edit-crop" => shell.crop_enabled(),
        "split-scenes" => shell.split_scenes_enabled(),
        "transcript" => shell.transcribe_enabled(),
        "extract-audio" => shell.extract_audio_enabled(),
        "flip" => shell.mirror_enabled(),
        "open-file" => shell.selected_has_file(),
        "open-compound" => shell.compound_selected(),
        "link-media" => shell.link_media_enabled(),
        "group" => shell.group_enabled(),
        "ungroup" => shell.ungroup_enabled(),
        "copy-attributes" => shell.copy_attributes_enabled(),
        "paste-attributes" => shell.paste_attributes_enabled(),
        // Only a video clip plays on a speed curve.
        "variable-speed" => kind == cutlass_model::TrackType::Video,
        _ => true,
    };
    let shown = crate::menu::show(
        ui.ctx(),
        ui.scope_id().with("clip-menu"),
        Pos2::new(menu.x, menu.y),
        entries,
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
            "edit-freeze" => shell.freeze_frame(),
            "edit-reverse" => shell.toggle_reverse(),
            "edit-mirror" => shell.toggle_mirror(),
            "edit-rotate" => shell.turn_clockwise(),
            "edit-crop" => shell.toggle_crop(),
            "split-scenes" => shell.split_scenes(),
            "transcript" => shell.transcribe(),
            "extract-audio" => shell.extract_audio(),
            "recover-audio" => shell.recover_audio(),
            "flip" => shell.toggle_mirror(),
            "deactivate" => shell.toggle_active(),
            "open-file" => shell.reveal_selected_file(),
            "replace" => {
                view.wants_replace = true;
                Ok(())
            }
            "compound" => shell.create_compound(),
            "render-preview" => shell.render_selected(false),
            "render-full" => shell.render_selected(true),
            "arrange-forward" => shell.arrange(cutlass_session::Arrange::Forward),
            "arrange-backward" => shell.arrange(cutlass_session::Arrange::Backward),
            "arrange-front" => shell.arrange(cutlass_session::Arrange::Front),
            "arrange-back" => shell.arrange(cutlass_session::Arrange::Back),
            "variable-speed" => {
                shell.toggle_speed_curves();
                Ok(())
            }
            "open-compound" => shell.open_compound(),
            "sync" => shell.start_sync(),
            "edit-effects" => match effects.first() {
                Some(effect) => shell.edit_clip_effect(&effect.segment_id),
                None => Ok(()),
            },
            id if id.starts_with(EDIT_EFFECT_ROW) => {
                let effect = id[EDIT_EFFECT_ROW.len()..]
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| effects.get(index));
                match effect {
                    Some(effect) => shell.edit_clip_effect(&effect.segment_id),
                    None => Ok(()),
                }
            }
            "link-media" => {
                view.wants_link = true;
                Ok(())
            }
            "group" => shell.group_selected(),
            "ungroup" => shell.ungroup_selected(),
            "copy-attributes" => shell.copy_attributes(),
            "paste-attributes" => shell.paste_attributes(),
            "range-in" => shell.mark_clip_in(),
            "range-out" => shell.mark_clip_out(),
            "beat" => {
                if !shell.beating() {
                    shell.toggle_beat();
                }
                Ok(())
            }
            "export" => {
                view.export.span = crate::shell::ExportSpan::Selected;
                view.export_open = true;
                Ok(())
            }
            id => {
                if let Some(focus) = inspector_focus(&kind, id) {
                    view.inspector.focus(focus);
                }
                Ok(())
            }
        };
        if let Err(error) = result {
            shell.notice_error(&error);
        }
    } else if shown.dismissed {
        shell.close_clip_menu();
    }
}

/// The inspector tab, page, and section a clip menu row opens.
fn inspector_focus(kind: &cutlass_model::TrackType, id: &str) -> Option<inspector::Focus> {
    let at = |tab, section| {
        Some(inspector::Focus {
            tab,
            page: 0,
            section,
        })
    };
    match (kind, id) {
        (cutlass_model::TrackType::Video, "edit-speed") => at(2, None),
        (cutlass_model::TrackType::Video, "edit-animation") => at(3, None),
        (cutlass_model::TrackType::Video, "edit-opacity") => at(0, Some("blend")),
        (_, "volume" | "fade") => at(0, Some("sound")),
        (_, "speed") => at(1, None),
        (_, "style") => at(0, Some("stroke")),
        (_, "edit-text" | "edit-effect" | "edit-filter" | "edit-adjust") => at(0, None),
        _ => None,
    }
}

/// The Close gap menu over a gap on a row.
fn paint_gap_menu(shell: &mut Shell, ui: &mut Ui) {
    let Some(menu) = shell.gap_menu().cloned() else {
        return;
    };
    if let Some(pos) = secondary_click(ui)
        && (pos.x, pos.y) != (menu.x, menu.y)
    {
        shell.close_gap_menu();
        return;
    }
    let shown = crate::menu::show(
        ui.ctx(),
        ui.scope_id().with("gap-menu"),
        Pos2::new(menu.x, menu.y),
        crate::menu::GAP_ROW,
        &|_| true,
    );
    if shown.picked.is_some() {
        shell.close_gap_menu();
        if let Err(error) = shell.close_gap(&menu.track_id, menu.time) {
            shell.notice_error(&error);
        }
    } else if shown.dismissed {
        shell.close_gap_menu();
    }
}

/// The level meter: a bar for each channel from the floor at the bottom to
/// full scale at the top, its held peak as a line, and a red cap while the
/// channel has clipped.
fn paint_meter(ui: &Ui, rect: Rect, channels: [cutlass_playback::ChannelReading; 2]) {
    let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
    ui::Surface::panel().paint(&painter, rect);
    painter.vline(
        rect.left(),
        rect.y_range(),
        egui::Stroke::new(1.0, tokens::GAP),
    );
    let span = Rect::from_min_max(
        Pos2::new(rect.left() + 16.0, rect.top() + tokens::RULER_HEIGHT),
        Pos2::new(rect.right() - 4.0, rect.bottom() - 8.0),
    );
    if span.height() <= 0.0 || span.width() <= 0.0 {
        return;
    }
    let y_at = |db: f32| span.bottom() - cutlass_playback::meter_fraction(db) * span.height();
    for db in [0.0, -12.0, -24.0, -48.0] {
        painter.text(
            Pos2::new(span.left() - 2.0, y_at(db)),
            Align2::RIGHT_CENTER,
            format!("{}", -db as i32),
            FontId::proportional(8.0),
            tokens::FAINT,
        );
    }
    let bar_width = (span.width() - 2.0) / 2.0;
    for (index, channel) in channels.iter().enumerate() {
        let left = span.left() + index as f32 * (bar_width + 2.0);
        let well = Rect::from_min_max(
            Pos2::new(left, span.top()),
            Pos2::new(left + bar_width, span.bottom()),
        );
        painter.rect_filled(well, CornerRadius::ZERO, tokens::LANE);
        // The bar in three bands, each up to where the level reaches.
        for (from, to, color) in [
            (cutlass_playback::METER_FLOOR_DB, -12.0, tokens::METER_LOW),
            (-12.0, -3.0, tokens::METER_LOUD),
            (-3.0, 0.0, tokens::METER_HOT),
        ] {
            if channel.level <= from {
                continue;
            }
            let top = y_at(channel.level.min(to));
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(well.left(), top),
                    Pos2::new(well.right(), y_at(from)),
                ),
                CornerRadius::ZERO,
                color,
            );
        }
        if channel.hold > cutlass_playback::METER_FLOOR_DB {
            painter.hline(
                well.x_range(),
                y_at(channel.hold),
                egui::Stroke::new(1.0, tokens::TEXT),
            );
        }
        if channel.clipped {
            painter.rect_filled(
                Rect::from_min_size(
                    Pos2::new(well.left(), span.top() - 5.0),
                    Vec2::new(bar_width, 3.0),
                ),
                CornerRadius::ZERO,
                tokens::METER_HOT,
            );
        }
    }
}

fn paint_track_menu(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let Some(menu) = view.track_menu.clone() else {
        return;
    };
    if let Some(pos) = secondary_click(ui)
        && pos != menu.at
    {
        view.track_menu = None;
        return;
    }
    let removable = !menu.main && !menu.lock;
    let shown = crate::menu::show(
        ui.ctx(),
        ui.scope_id().with("track-menu"),
        menu.at,
        crate::menu::TRACK_HEAD,
        &|id| id != "delete-track" || removable,
    );
    if let Some(item) = shown.picked {
        view.track_menu = None;
        track_action(shell, view, &menu.track_id, item.id);
    } else if shown.dismissed {
        view.track_menu = None;
    }
}

/// Run a row of the track head's menu.
fn track_action(shell: &mut Shell, view: &mut ViewState, track_id: &TrackId, action: &str) {
    let kind = match action {
        "add-video" => Some(cutlass_model::TrackType::Video),
        "add-audio" => Some(cutlass_model::TrackType::Audio),
        "add-text" => Some(cutlass_model::TrackType::Text),
        _ => None,
    };
    let result = match (action, kind) {
        (_, Some(kind)) => shell.add_track(kind).map(|_| ()),
        ("rename-track", None) => {
            let lanes = shell.lanes();
            let typed = lanes
                .iter()
                .find(|lane| &lane.track_id == track_id)
                .map(|lane| {
                    lane.name
                        .clone()
                        .unwrap_or_else(|| lane_name(&lane.kind, lane.main))
                })
                .unwrap_or_default();
            view.renaming = Some((track_id.clone(), typed));
            Ok(())
        }
        ("delete-track", None) => {
            view.lane_heights.remove(track_id);
            shell.remove_track(track_id)
        }
        ("solo-track", None) => {
            shell.toggle_attribute(track_id.clone(), cutlass_model::TrackAttribute::SOLO)
        }
        _ => Ok(()),
    };
    if let Err(error) = result {
        shell.notice_error(&error);
    }
}

/// The track head: a right-click opens its menu, a double-click on the
/// name names it, and while it is being named a field stands in for it.
fn lane_head(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    lane: &crate::shell::LaneView,
    row: Rect,
    index: usize,
) {
    let head = head_rect(row);
    let label = Rect::from_min_max(
        Pos2::new(head.left() + HEAD_FIRST + 14.0, row.center().y - 11.0),
        Pos2::new(head.right() - 6.0, row.center().y + 11.0),
    );
    if let Some((track_id, typed)) = view.renaming.as_mut()
        && track_id == &lane.track_id
    {
        let field = ui::Field::new(&mut *typed)
            .text_size(12.0)
            .focus()
            .show_in(ui, label);
        if field.committed || field.cancelled {
            let cancel = field.cancelled;
            let typed = typed.clone();
            view.renaming = None;
            if !cancel && let Err(error) = shell.rename_track(&lane.track_id, &typed) {
                shell.notice_error(&error);
            }
        }
        return;
    }
    let name = lane
        .name
        .clone()
        .unwrap_or_else(|| lane_name(&lane.kind, lane.main));
    let kind = paint_lane_head(ui, row, lane_icon(&lane.kind));
    let response = ui
        .interact(
            kind.expand(3.0),
            ui.scope_id().with(("head", index)),
            Sense::click(),
        )
        .on_hover_text(format!("{name} · double-click to rename"));
    if response.double_clicked() {
        track_action(shell, view, &lane.track_id, "rename-track");
    }
    lane_switches(shell, ui, view, lane, row, index);
    if let Some(pos) = secondary_click(ui)
        && head.contains(pos)
    {
        shell.close_gap_menu();
        view.track_menu = Some(TrackMenu {
            track_id: lane.track_id.clone(),
            main: lane.main,
            lock: lane.lock,
            at: pos,
        });
    }
}

/// The part of a row left of its switches, where its icon and name are.
fn head_rect(row: Rect) -> Rect {
    Rect::from_min_max(
        Pos2::new(
            row.left() - tokens::TRACK_HEAD - tokens::COVER_WIDTH,
            row.top(),
        ),
        Pos2::new(row.left() - tokens::COVER_WIDTH, row.bottom()),
    )
}

/// A row's height: the one dragged for it, or its kind's own.
fn lane_height(view: &ViewState, lane: &crate::shell::LaneView) -> f32 {
    view.lane_heights
        .get(&lane.track_id)
        .copied()
        .unwrap_or_else(|| row_height(lane_size(lane)))
}

/// Drag a row's outer edge to make it taller or shorter: the top edge for
/// rows above the main track, the bottom edge for the main track and the
/// rows under it. A double-click on the edge gives back the usual height.
fn resize_lane(
    ui: &mut Ui,
    view: &mut ViewState,
    lane: &crate::shell::LaneView,
    row: Rect,
    index: usize,
    main: Option<usize>,
) {
    let above = main.is_some_and(|main| index < main);
    let edge_y = if above { row.top() } else { row.bottom() };
    let edge = Rect::from_min_max(
        Pos2::new(
            row.left() - tokens::TRACK_HEAD - tokens::COVER_WIDTH,
            edge_y - 3.0,
        ),
        Pos2::new(row.right(), edge_y + 3.0),
    );
    let response = ui
        .interact(
            edge,
            ui.scope_id().with(("lane-edge", index)),
            Sense::click_and_drag(),
        )
        .on_hover_cursor(egui::CursorIcon::ResizeVertical);
    if response.double_clicked() {
        view.lane_heights.remove(&lane.track_id);
        return;
    }
    let delta = response.drag_delta().y;
    if delta != 0.0 {
        let height = lane_height(view, lane);
        let grown = if above {
            height - delta
        } else {
            height + delta
        };
        view.lane_heights.insert(
            lane.track_id.clone(),
            grown.clamp(tokens::MIN_LANE_HEIGHT, tokens::MAX_LANE_HEIGHT),
        );
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
        shell.notice_error(&error);
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
    let x = f64::from((pos.x - lanes_left).max(0.0));
    let y = lane_under(boxes, pos).map(|index| {
        let row = u32::try_from(index).unwrap_or(0);
        f64::from(row) * tokens::LANE_HEIGHT + tokens::LANE_HEIGHT / 2.0
    });
    // In a gap, or past the top or bottom lane, a card may open its own lane.
    let gap = match lane_spot(boxes, pos) {
        Some(LaneSpot::Between(row)) => Some(row),
        _ => None,
    };
    // Mark where the card really goes, which may be another lane or a new one.
    if let Some(landing) = y.and_then(|y| shell.library_landing(&material_id, x, y, gap)) {
        paint_library_landing(shell, ui, &landing, boxes, lanes_left);
    }
    if !ui.input(|input| input.pointer.any_released()) {
        return;
    }
    let _ = egui::DragAndDrop::take_payload::<LibraryDrop>(ui.ctx());
    let Some(y) = y else {
        shell.notice_error(&"the drag missed the lanes");
        return;
    };
    let landed = shell.drop_library(&material_id, x, y, gap);
    if let Err(error) = landed {
        shell.notice_error(&error);
    }
}

/// Outline the lane a library card lands on and the span it takes there, or
/// draw the line where a new lane opens for it.
fn paint_library_landing(
    shell: &Shell,
    ui: &Ui,
    landing: &crate::shell::LibraryLanding,
    boxes: &[Rect],
    lanes_left: f32,
) {
    use crate::shell::LibraryLanding;
    match landing {
        LibraryLanding::Row {
            track_id,
            start,
            duration,
        } => {
            let rows = shell.lanes();
            let Some((lane, row)) = rows
                .iter()
                .zip(boxes)
                .find(|(lane, _)| &lane.track_id == track_id)
            else {
                return;
            };
            ui.painter().rect_stroke(
                *row,
                CornerRadius::same(4),
                egui::Stroke::new(tokens::STROKE, tokens::EXPORT),
                egui::StrokeKind::Inside,
            );
            if duration.0 > 0 {
                let spot = clip_bounds(shell, lane_size(lane), *start, *duration, *row, lanes_left);
                ui.painter().with_clip_rect(*row).rect_stroke(
                    spot,
                    CornerRadius::same(7),
                    egui::Stroke::new(2.0, tokens::SELECTED),
                    egui::StrokeKind::Inside,
                );
            }
        }
        LibraryLanding::NewLane { row, .. } => {
            if let Some(y) = gap_y(boxes, *row) {
                let right = boxes.first().map_or(lanes_left, Rect::right);
                ui.painter().hline(
                    lanes_left..=right,
                    y,
                    egui::Stroke::new(2.0, tokens::SELECTED),
                );
            }
        }
    }
}

/// The middle of the gap above view row `row`, or just past the first or
/// last row.
fn gap_y(boxes: &[Rect], row: usize) -> Option<f32> {
    match (
        row.checked_sub(1).and_then(|above| boxes.get(above)),
        boxes.get(row),
    ) {
        (Some(above), Some(below)) => Some((above.bottom() + below.top()) / 2.0),
        (None, Some(below)) => Some(below.top() - 4.0),
        (Some(above), None) => Some(above.bottom() + 4.0),
        (None, None) => None,
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
    if let Some(style) = cutlass_editor::title_style(material_id) {
        return Some(title_card(style));
    }
    if let Some(effect) = cutlass_editor::effect_cards()
        .into_iter()
        .find(|effect| effect.id == material_id)
    {
        return Some(effect_card(effect));
    }
    if let Some(offer) = crate::shell::transition_offers()
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

fn lane_ruler(ruler: Rect, lanes_left: f32) -> Rect {
    Rect::from_min_max(Pos2::new(lanes_left, ruler.top()), ruler.max)
}

fn lane_x(lanes_left: f32, pixel_size: f64, origin: i64, time: i64) -> f32 {
    let delta = (time - origin) as f64 * pixel_size;
    lanes_left + delta as f32
}

/// The I and O marks on the ruler: a band over the marked span, with an
/// edge at each mark set.
fn paint_marks(ui: &Ui, shell: &Shell, ruler: Rect, lanes_left: f32) {
    let Some((start, end)) = shell.marked_span() else {
        return;
    };
    let (pixel_size, origin) = (shell.pixel_size(), shell.view_origin().0);
    let painter = ui.painter().with_clip_rect(Rect::from_min_max(
        Pos2::new(lanes_left, ruler.top()),
        ruler.max,
    ));
    let left = lane_x(lanes_left, pixel_size, origin, start.0);
    let right = lane_x(lanes_left, pixel_size, origin, end.0);
    let band = Rect::from_min_max(
        Pos2::new(left, ruler.top()),
        Pos2::new(right, ruler.bottom()),
    );
    painter.rect_filled(band, CornerRadius::ZERO, tokens::MARKED);
    let marks = shell.marks();
    for (mark, x) in [(marks.mark_in, left), (marks.mark_out, right)] {
        if mark.is_some() {
            painter.vline(
                x,
                ruler.y_range(),
                egui::Stroke::new(tokens::STROKE, tokens::EXPORT),
            );
        }
    }
}

/// The widest ruler label, a drop-frame timecode, with its gap.
const RULER_LABEL_LEAD: f32 = 80.0;

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
    // Start a label's width before the view, so a label whose tick has
    // just scrolled off the left shows its tail instead of vanishing.
    let lead = (f64::from(RULER_LABEL_LEAD) / pixel_size).ceil();
    let lead = if lead.is_finite() && lead < i64::MAX as f64 {
        lead as i64
    } else {
        0
    };
    let from = Micros(origin.saturating_sub(lead).max(0));
    let to = Micros(origin.saturating_add(span as i64));
    let painter = ui.painter().with_clip_rect(Rect::from_min_max(
        Pos2::new(lanes_left, ruler.top()),
        ruler.max,
    ));
    for tick in ruler_ticks(rate, drop_frame, pixel_size, from, to) {
        let x = lane_x(lanes_left, pixel_size, origin, tick.time.0);
        // Ticks hang from the top, and a label sits beside its major tick.
        let top = ruler.top() + 3.0;
        let height = if tick.major {
            tokens::MAJOR_TICK
        } else {
            tokens::MAJOR_TICK * 0.4
        };
        painter.vline(
            x,
            top..=top + height,
            egui::Stroke::new(1.0, tokens::RULER_TICK),
        );
        if let Some(label) = tick.label {
            painter.text(
                Pos2::new(x + 4.0, top + tokens::MAJOR_TICK / 2.0),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(11.0),
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
    let height = row_height(RowSize::Main);
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
    paint_lane_head(ui, lane, "media");
}

fn paint_lane(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    row: Rect,
    lanes_left: f32,
    index: usize,
    boxes: &[Rect],
) {
    // The head and the row take the lane fill; the cover column between
    // them is left bare, so the panel shows there, and the hairline between
    // rows is the panel showing through the gap.
    ui.painter()
        .rect_filled(head_rect(row), CornerRadius::ZERO, tokens::LANE);
    ui.painter()
        .rect_filled(row, CornerRadius::ZERO, tokens::LANE);
    if lane.main {
        paint_cover(shell, ui, row);
    }
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
    let followers: Vec<_> = shell
        .move_previews()
        .into_iter()
        .filter(|preview| preview.track_id == lane.track_id)
        .collect();
    for clip in &lane.clips {
        let at = RowAt { index, row, boxes };
        paint_clip(shell, ui, lane, clip, lanes_left, at);
    }
    // A right-click on the row but off its clips may be over a gap.
    if let Some(pos) = secondary_click(ui)
        && row.contains(pos)
        && pos.x >= lanes_left
        && shell
            .clip_menu()
            .is_none_or(|menu| (menu.x, menu.y) != (pos.x, pos.y))
        && let Some(time) = shell.lane_time_at(f64::from(pos.x - lanes_left))
    {
        shell.open_gap_menu(&lane.track_id, time, pos.x, pos.y);
    }
    paint_transitions(shell, ui, lane, row, lanes_left, index);
    for preview in &followers {
        paint_move_follower(shell, ui, lane, preview, row, lanes_left);
    }
}

/// The Cover box in the bare column beside the main lane.
fn paint_cover(shell: &Shell, ui: &Ui, row: Rect) {
    let column = Rect::from_x_y_ranges(
        egui::Rangef::new(head_rect(row).right(), row.left()),
        row.y_range(),
    );
    cover::paint_box(shell, ui, column);
}

/// The lane's kind as an icon at the head's left. Its name is not written
/// out; [`lane_head`] shows it on hover.
fn paint_lane_head(ui: &Ui, row: Rect, icon: &str) -> Rect {
    let center = Pos2::new(head_rect(row).left() + HEAD_FIRST, row.center().y);
    let rect = Rect::from_center_size(center, Vec2::splat(16.0));
    let _ = icons::paint(ui.painter(), rect, icon, tokens::MUTED);
    rect
}

/// Where the head's first icon sits from its left, and the step to each
/// button after it.
const HEAD_FIRST: f32 = 20.0;
const HEAD_STEP: f32 = 25.0;

/// The head's buttons, left to right: lock, show or hide (not on a sound
/// lane), mute (only on a lane that makes sound), and more, which opens the
/// track menu.
/// The head's slot for `button`, counted after the kind icon: each button
/// keeps its column on every lane, and a lane without one leaves its slot
/// empty, so the icons line up down the heads.
fn head_slot(button: &str) -> f32 {
    match button {
        "lock" => 1.0,
        "hide" => 2.0,
        "mute" => 3.0,
        _ => 4.0,
    }
}

fn head_buttons(lane: &crate::shell::LaneView) -> Vec<&'static str> {
    let sound = lane.main
        || matches!(
            lane.kind,
            cutlass_model::TrackType::Video
                | cutlass_model::TrackType::Mixed
                | cutlass_model::TrackType::Audio
        );
    let mut buttons = vec!["lock"];
    if !is_audio(lane) {
        buttons.push("hide");
    }
    if sound {
        buttons.push("mute");
    }
    // The track menu's more is on video and audio lanes only, main too; a
    // right-click on any head still opens the menu.
    if lane.main
        || matches!(
            lane.kind,
            cutlass_model::TrackType::Video | cutlass_model::TrackType::Audio
        )
    {
        buttons.push("more");
    }
    buttons
}

fn lane_switches(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    lane: &crate::shell::LaneView,
    row: Rect,
    index: usize,
) {
    let head = head_rect(row);
    for button in head_buttons(lane) {
        let step = head_slot(button);
        let center = Pos2::new(head.left() + HEAD_FIRST + HEAD_STEP * step, row.center().y);
        let rect = Rect::from_center_size(center, Vec2::splat(22.0));
        let response = ui.interact(rect, ui.scope_id().with((button, index)), Sense::click());
        let (icon, color, tip) = match button {
            "lock" if lane.lock => ("lock", tokens::TEXT, "Unlock"),
            "lock" => ("unlock", tokens::TEXT, "Lock"),
            "hide" if lane.hide => ("viewOff", tokens::MUTED, "Show"),
            "hide" => ("view", tokens::TEXT, "Hide"),
            "mute" if lane.mute => ("volumeOff", tokens::MUTED, "Unmute"),
            "mute" if lane.mute_by_solo => ("volumeOff", tokens::FAINT, "Muted by solo"),
            "mute" => ("volume", tokens::TEXT, "Mute"),
            _ => ("more", tokens::TEXT, "More"),
        };
        let color = if response.hovered() {
            tokens::TEXT
        } else {
            color
        };
        let _ = icons::paint(
            ui.painter(),
            Rect::from_center_size(center, Vec2::splat(15.0)),
            icon,
            color,
        );
        if response.clicked() {
            let bit = match button {
                "lock" => Some(cutlass_model::TrackAttribute::LOCK),
                "hide" => Some(cutlass_model::TrackAttribute::HIDE),
                "mute" => Some(cutlass_model::TrackAttribute::MUTE),
                _ => None,
            };
            match bit {
                Some(bit) => {
                    if let Err(error) = shell.toggle_attribute(lane.track_id.clone(), bit) {
                        shell.notice_error(&error);
                    }
                }
                None => {
                    shell.close_gap_menu();
                    view.track_menu = Some(TrackMenu {
                        track_id: lane.track_id.clone(),
                        main: lane.main,
                        lock: lane.lock,
                        at: rect.left_bottom(),
                    });
                }
            }
        }
        response.on_hover_text(tip);
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
    if response.drag_started_by(egui::PointerButton::Primary)
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
        && response.dragged_by(egui::PointerButton::Primary)
        && let (Some(press), Some(pos)) = (view.marquee_press, pos)
    {
        view.marquee = Some(Rect::from_two_pos(press, pos));
        scroll_with_drag(shell, ui, f64::from(pos.x - lanes_left).max(0.0));
    }
    let shift = ui.input(|input| input.modifiers.shift);
    if response.drag_stopped_by(egui::PointerButton::Primary) {
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
            let bounds = clip_bounds(shell, lane_size(lane), start, duration, row, lanes_left);
            if bounds.intersects(rect) {
                ids.push(clip.id.clone());
            }
        }
    }
    ids
}

fn clip_bounds(
    shell: &Shell,
    size: RowSize,
    start: Micros,
    duration: Micros,
    row: Rect,
    lanes_left: f32,
) -> Rect {
    // A row dragged taller or shorter takes its clips with it.
    let inset = clip_inset(size);
    let height = if (row.height() - row_height(size)).abs() < f32::EPSILON {
        clip_height(size)
    } else {
        (row.height() - 2.0 * inset).max(tokens::MIN_CLIP_HEIGHT)
    };
    let x = lane_x(
        lanes_left,
        shell.pixel_size(),
        shell.view_origin().0,
        start.0,
    );
    let width = (duration.0 as f32 * shell.pixel_size() as f32).max(8.0);
    Rect::from_min_size(Pos2::new(x, row.top() + inset), Vec2::new(width, height))
}

/// A clip's height: the main track is tallest, audio holds its waveform
/// under a header, the rest are a single row.
fn clip_height(size: RowSize) -> f32 {
    match size {
        RowSize::Main => tokens::MAIN_CLIP_HEIGHT,
        RowSize::Picture => tokens::PICTURE_CLIP_HEIGHT,
        RowSize::Audio => tokens::AUDIO_CLIP_HEIGHT,
        RowSize::Thin => tokens::CLIP_HEIGHT,
    }
}

/// How tall a lane's row is: main, an overlay picture lane, an audio lane,
/// or a thin one (text, stickers, effects and the like).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowSize {
    Main,
    Picture,
    Audio,
    Thin,
}

fn lane_size(lane: &crate::shell::LaneView) -> RowSize {
    if lane.main {
        RowSize::Main
    } else if tall_lane(lane) {
        RowSize::Picture
    } else if is_audio(lane) {
        RowSize::Audio
    } else {
        RowSize::Thin
    }
}

fn is_audio(lane: &crate::shell::LaneView) -> bool {
    lane.kind == cutlass_model::TrackType::Audio
}

/// Main and overlay video rows share the tall picture height, so a clip
/// keeps its size when it moves between them. Text, sticker and effect rows
/// stay thin.
fn tall_lane(lane: &crate::shell::LaneView) -> bool {
    lane.main
        || matches!(
            lane.kind,
            cutlass_model::TrackType::Video | cutlass_model::TrackType::Mixed
        )
}

fn clip_inset(size: RowSize) -> f32 {
    match size {
        RowSize::Main => tokens::MAIN_CLIP_INSET,
        RowSize::Picture => tokens::PICTURE_CLIP_INSET,
        RowSize::Audio | RowSize::Thin => tokens::CLIP_INSET,
    }
}

/// A lane row: its clips and the gap above and below them.
fn row_height(size: RowSize) -> f32 {
    clip_height(size) + 2.0 * clip_inset(size)
}

/// The narrowest a transition's bar is drawn, so one on a wide view of
/// the timeline can still be picked.
const TRANSITION_MIN_WIDTH: f32 = 22.0;

/// The width of the strip at each side of a transition's bar that drags it
/// longer or shorter.
const TRANSITION_EDGE: f32 = 6.0;

/// How far a transition's bar stands above and below the clips it joins.
const TRANSITION_RISE: f32 = 2.0;

/// The pale fill of a transition's bar: frosted over the clips while it
/// rests, lighter under the pointer, and clearer once picked, when its
/// outline carries it.
const TRANSITION_FILL: Color32 = Color32::from_rgba_premultiplied(92, 92, 92, 92);
const TRANSITION_FILL_HOVERED: Color32 = Color32::from_rgba_premultiplied(120, 120, 120, 120);
const TRANSITION_FILL_PICKED: Color32 = Color32::from_rgba_premultiplied(56, 56, 56, 56);

/// The mark in the middle of a transition's bar: two triangles that meet
/// point to point at `center`, one clip handing over to the next.
fn paint_transition_mark(painter: &egui::Painter, center: Pos2) {
    const HALF: f32 = 5.0;
    const REACH: f32 = 6.0;
    for side in [-1.0, 1.0] {
        painter.add(egui::Shape::convex_polygon(
            vec![
                center + Vec2::new(side * REACH, -HALF),
                center + Vec2::new(side * REACH, HALF),
                center,
            ],
            Color32::WHITE,
            egui::Stroke::NONE,
        ));
    }
}

/// The grip inside each end of a picked transition's bar: a short upright
/// line that says the end can be dragged.
fn paint_transition_grips(painter: &egui::Painter, rect: Rect) {
    for x in [rect.left() + 6.0, rect.right() - 6.0] {
        painter.line_segment(
            [
                Pos2::new(x, rect.center().y - 6.0),
                Pos2::new(x, rect.center().y + 6.0),
            ],
            egui::Stroke::new(2.0, Color32::from_white_alpha(230)),
        );
    }
}

/// Where a transition's bar is drawn on its `row`: over the blend's
/// window, a little taller than the clips, and no narrower than
/// [`TRANSITION_MIN_WIDTH`] about its middle.
fn transition_rect(
    shell: &Shell,
    lane: &crate::shell::LaneView,
    bar: &cutlass_editor::TransitionBar,
    row: Rect,
    lanes_left: f32,
) -> Rect {
    let span = Micros(bar.end.0.saturating_sub(bar.start.0));
    let rect = clip_bounds(shell, lane_size(lane), bar.start, span, row, lanes_left);
    let width = (span.0 as f32 * shell.pixel_size() as f32).max(TRANSITION_MIN_WIDTH);
    Rect::from_center_size(
        Pos2::new(
            rect.left() + span.0 as f32 * shell.pixel_size() as f32 / 2.0,
            rect.center().y,
        ),
        Vec2::new(width, rect.height() + 2.0 * TRANSITION_RISE),
    )
}

/// The transitions of `lane`, each a frosted bar over its cut with a mark
/// in its middle. A click picks one, which then wears a white outline and
/// a grip inside each end; Delete takes it off, and an end dragged makes
/// it longer or shorter between its limits.
fn paint_transitions(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    row: Rect,
    lanes_left: f32,
    index: usize,
) {
    let bars = shell.transitions();
    if bars.is_empty() {
        return;
    }
    let picked = shell.selected_transition().map(|bar| bar.segment_id);
    for bar in bars.iter().filter(|bar| bar.track_id == lane.track_id) {
        let rect = transition_rect(shell, lane, bar, row, lanes_left);
        let visible = rect.intersect(row);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            continue;
        }
        let selected = picked.as_ref() == Some(&bar.segment_id);
        let painter = ui.painter().with_clip_rect(row);
        let radius = CornerRadius::same(CLIP_RADIUS);
        let hovered = ui.rect_contains_pointer(visible);
        let fill = match (selected, hovered) {
            (true, _) => TRANSITION_FILL_PICKED,
            (false, true) => TRANSITION_FILL_HOVERED,
            (false, false) => TRANSITION_FILL,
        };
        painter.rect_filled(rect, radius, fill);
        if selected {
            painter.rect_stroke(
                rect,
                radius,
                egui::Stroke::new(2.0, Color32::WHITE),
                egui::StrokeKind::Inside,
            );
        }
        // The mark needs its own width, and the grips room beside it.
        if rect.width() >= 18.0 {
            paint_transition_mark(&painter, rect.center());
        }
        if selected && rect.width() >= 40.0 {
            paint_transition_grips(&painter, rect);
        }
        let id = bar.segment_id.clone();
        let salt = ("transition", index, id.as_str().to_owned());
        // The body takes the press whole, so the clips under the bar are
        // neither picked nor dragged through it.
        let body = Rect::from_min_max(
            Pos2::new(visible.left() + TRANSITION_EDGE, visible.top()),
            Pos2::new(visible.right() - TRANSITION_EDGE, visible.bottom()),
        );
        if body.width() > 0.0 {
            let response = ui
                .interact(body, ui.scope_id().with(&salt), Sense::click_and_drag())
                .on_hover_text(format!(
                    "{} · {:.1}s",
                    bar.name,
                    bar.duration.0 as f64 / 1_000_000.0
                ));
            if response.clicked() || response.drag_started() {
                shell.select_transition(Some(id.clone()));
            }
        }
        for (edge, x) in [
            (cutlass_editor::BarEdge::Start, visible.left()),
            (
                cutlass_editor::BarEdge::End,
                visible.right() - TRANSITION_EDGE,
            ),
        ] {
            let strip = Rect::from_min_size(
                Pos2::new(x, visible.top()),
                Vec2::new(TRANSITION_EDGE, visible.height()),
            )
            .expand2(Vec2::new(2.0, 0.0));
            let response = ui.interact(
                strip,
                ui.scope_id()
                    .with((&salt, edge == cutlass_editor::BarEdge::Start)),
                Sense::drag(),
            );
            if response.hovered() || response.dragged() {
                ui.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            if response.drag_started() {
                shell.select_transition(Some(id.clone()));
            }
            if response.dragged()
                && let Some(pos) = pointer_pos(ui, &response)
                && let Err(error) =
                    shell.drag_transition_edge(&id, edge, f64::from(pos.x - lanes_left))
            {
                shell.notice_error(&error);
            }
        }
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
    // A short overlay clip keeps a third for its name, so the pictures and
    // the sound still show under it.
    let header_height = tokens::CLIP_HEADER.min((clip_rect.height() / 3.0).max(13.0));
    let split = clip_rect.top() + header_height.min(clip_rect.height());
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

/// Where a curve clip runs fast and slow, drawn over its bar in timeline
/// time: 1× is the faint middle line, 10× the top and 0.1× the bottom.
fn paint_speed_curve(ui: &Ui, area: Rect, visible: Rect, curve: &[(f64, f64)]) {
    if area.height() < 6.0 || area.width() < 4.0 {
        return;
    }
    let points: Vec<cutlass_model::SpeedPoint> = curve
        .iter()
        .map(|&(x, y)| cutlass_model::SpeedPoint::new(x, y))
        .collect();
    // Timeline time is the sum of dx / speed over the source.
    const STEPS: usize = 96;
    let mut along = vec![(0.0, cutlass_model::curve_speed_at(&points, 0.0))];
    let mut time = 0.0;
    for step in 1..=STEPS {
        #[allow(clippy::cast_precision_loss)]
        let (x0, x1) = ((step - 1) as f64 / STEPS as f64, step as f64 / STEPS as f64);
        let speed = cutlass_model::curve_speed_at(&points, (x0 + x1) / 2.0).max(0.01);
        time += (x1 - x0) / speed;
        along.push((time, cutlass_model::curve_speed_at(&points, x1)));
    }
    let total = time.max(f64::EPSILON);
    #[allow(clippy::cast_possible_truncation)]
    let at = |(time, speed): (f64, f64)| {
        let up = (speed.clamp(0.1, 10.0).log10() + 1.0) / 2.0;
        Pos2::new(
            area.left() + (time / total) as f32 * area.width(),
            area.bottom() - up as f32 * area.height(),
        )
    };
    let painter = ui.painter().with_clip_rect(visible.intersect(area));
    let middle = area.center().y;
    painter.hline(
        area.x_range(),
        middle,
        egui::Stroke::new(1.0, Color32::WHITE.gamma_multiply(0.25)),
    );
    let line: Vec<Pos2> = along.into_iter().map(at).collect();
    painter.add(egui::Shape::line(
        line,
        egui::Stroke::new(1.5, Color32::from_rgb(0xff, 0xc8, 0x3a)),
    ));
}

/// A clip's lane: its index, its whole row (part of it may be scrolled out
/// of its band), and the part of every row that shows, which a body drag
/// reads for the row under the pointer.
#[derive(Clone, Copy)]
struct RowAt<'a> {
    index: usize,
    row: Rect,
    boxes: &'a [Rect],
}

fn paint_clip(
    shell: &mut Shell,
    ui: &mut Ui,
    lane: &crate::shell::LaneView,
    clip: &crate::shell::ClipSpan,
    lanes_left: f32,
    at: RowAt<'_>,
) {
    // The whole row, so a clip keeps its height while its row is part hidden.
    let RowAt { index, row, boxes } = at;
    let (start, duration) = preview_bounds(shell, clip);
    let clip_rect = clip_bounds(shell, lane_size(lane), start, duration, row, lanes_left);
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
    let parts = clip_parts(clip_rect, tall_lane(lane), audio, wave.is_some());
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
    if !clip.active {
        // A deactivated clip stays in its place, dimmed.
        painter.rect_filled(
            clip_rect,
            CornerRadius::same(CLIP_RADIUS),
            Color32::from_black_alpha(150),
        );
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
    if shell.speed_curves_shown()
        && let Some(curve) = &clip.speed_curve
    {
        let below = parts
            .header
            .map_or(clip_rect.top(), |header| header.bottom());
        let area = Rect::from_min_max(
            Pos2::new(clip_rect.left(), below + 2.0),
            Pos2::new(clip_rect.right(), clip_rect.bottom() - 2.0),
        );
        paint_speed_curve(ui, area, visible, curve);
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
                shell.notice_error(&error);
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
                shell.notice_error(&error);
            }
        }
    } else if body.width() > 0.0 {
        let response = ui.interact(
            body,
            ui.scope_id().with(("clip", index, clip.id.as_str())),
            Sense::drag(),
        );
        response.clone().on_hover_text(span.clone());
        // A double-click on a compound clip opens it.
        if response.hovered()
            && ui.input(|input| {
                input
                    .pointer
                    .button_double_clicked(egui::PointerButton::Primary)
            })
        {
            shell.select(Some(clip.id.clone()));
            if shell.compound_selected() {
                if let Err(error) = shell.open_compound() {
                    shell.notice_error(&error);
                }
                return;
            }
        }
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(pos) = pointer_pos(ui, &response)
        {
            let grab = (pos.x - clip_rect.left(), pos.y - clip_rect.top());
            shell.begin_clip_drag(clip.id.clone(), None, (pos.x, pos.y), grab);
            if !ui.input(|input| input.modifiers.shift) && !shell.is_selected(&clip.id) {
                shell.select(Some(clip.id.clone()));
            }
        }
        let pos = pointer_pos(ui, &response);
        let moved = pos.is_some_and(|pos| shell.clip_drag_moved(pos.x, pos.y));
        if moved
            && response.dragged_by(egui::PointerButton::Primary)
            && shell.clip_edge(&clip.id).is_none()
            && let Some(pos) = pos
        {
            scroll_with_drag(shell, ui, f64::from(pos.x - lanes_left).max(0.0));
            // The clip keeps the spot it was taken by, so it does not jump to the pointer.
            let (grab_x, grab_y) = shell.clip_grab(&clip.id);
            let x = f64::from(pos.x - grab_x - lanes_left).max(0.0);
            match lane_spot(boxes, pos).unwrap_or(LaneSpot::On(index)) {
                LaneSpot::On(row) => shell.preview_move(clip.id.clone(), x, editor_lane_y(row)),
                LaneSpot::Between(row) => shell.preview_move_between(clip.id.clone(), x, row),
            }
            let floating =
                Rect::from_min_size(Pos2::new(pos.x - grab_x, pos.y - grab_y), clip_rect.size());
            if !shell.drag_escaped(&clip.id) {
                paint_floating_clip(shell, ui, lane, clip, floating, lanes_left);
            }
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) && !moved {
            if ui.input(|input| input.modifiers.shift) {
                shell.toggle_selected(clip.id.clone());
            } else {
                shell.select(Some(clip.id.clone()));
            }
            shell.cancel_clip_drag();
        } else {
            let spot = pos
                .and_then(|pos| lane_spot(boxes, pos))
                .unwrap_or(LaneSpot::On(index));
            finish_clip_pointer(shell, ui, &response, &clip.id, lanes_left, spot, None);
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
    if shell.is_selected(&clip.id) {
        paint_clip_keys(shell, ui, clip, clip_rect, row);
    }
}

/// How far a key's diamond reaches from its middle, and the picked one's.
const KEY_REACH: f32 = 5.0;
const KEY_REACH_PICKED: f32 = 6.5;

/// The selected clip's keys as diamonds along its foot. A click puts the
/// playhead on the key and picks it, so Delete takes the key and not the
/// clip; a drag moves it, landing on the nearest frame.
fn paint_clip_keys(
    shell: &mut Shell,
    ui: &mut Ui,
    clip: &crate::shell::ClipSpan,
    clip_rect: Rect,
    row: Rect,
) {
    let painter = ui.painter().with_clip_rect(clip_rect.intersect(row));
    let pixel_size = shell.pixel_size() as f32;
    let y = clip_rect.bottom() - 8.0;
    let picked: Vec<Micros> = shell
        .selected_clip_keys()
        .into_iter()
        .filter(|(id, _)| *id == clip.id)
        .map(|(_, time)| time)
        .collect();
    for time in shell.clip_key_times(&clip.id) {
        let x = clip_rect.left() + (time.0 - clip.start.0) as f32 * pixel_size;
        let hit = Rect::from_center_size(Pos2::new(x, y), Vec2::new(12.0, 14.0));
        if !row.intersects(hit) {
            continue;
        }
        let response = ui.interact(
            hit,
            ui.scope_id().with(("key", clip.id.as_str(), time.0)),
            Sense::click_and_drag(),
        );
        ui.data_mut(|data| {
            data.get_temp_mut_or_default::<Vec<Rect>>(key_hits_id())
                .push(hit);
        });
        let dragged_to = response
            .dragged_by(egui::PointerButton::Primary)
            .then(|| ui.input(|input| input.pointer.interact_pos()))
            .flatten();
        let at = dragged_to.map_or(x, |pos| pos.x.clamp(clip_rect.left(), clip_rect.right()));
        let on = shell.playhead() == time;
        let color = if response.hovered() || response.dragged_by(egui::PointerButton::Primary) || on
        {
            tokens::EXPORT
        } else {
            Color32::WHITE
        };
        let c = Pos2::new(at, y);
        // The picked key is a size larger, filled, with a white edge.
        let chosen = picked.contains(&time);
        let (reach, fill, edge) = if chosen {
            (KEY_REACH_PICKED, tokens::EXPORT, Color32::WHITE)
        } else {
            (KEY_REACH, color, Color32::BLACK)
        };
        painter.add(egui::Shape::convex_polygon(
            vec![
                c + Vec2::new(0.0, -reach),
                c + Vec2::new(reach, 0.0),
                c + Vec2::new(0.0, reach),
                c + Vec2::new(-reach, 0.0),
            ],
            fill,
            egui::Stroke::new(if chosen { 1.5 } else { 1.0 }, edge),
        ));
        // A drag that starts on a key not yet picked picks it alone; one
        // that starts on a picked key takes all the picked ones along.
        if response.drag_started_by(egui::PointerButton::Primary) && !chosen {
            shell.select_clip_key(&clip.id, time);
        }
        if response.drag_stopped_by(egui::PointerButton::Primary)
            && let Some(pos) = ui.input(|input| input.pointer.interact_pos())
        {
            let x = pos.x.clamp(clip_rect.left(), clip_rect.right());
            let to = clip.start.0
                + (f64::from(x - clip_rect.left()) / shell.pixel_size()).round() as i64;
            if let Err(error) = shell.move_clip_key(&clip.id, time, Micros(to)) {
                shell.notice_error(&error);
            }
        } else if response.clicked() {
            // With Shift or Command held the key joins or leaves the
            // picked ones, and the playhead stays.
            let more = ui.input(|input| input.modifiers.shift || input.modifiers.command);
            if more {
                shell.toggle_clip_key(&clip.id, time);
            } else {
                shell.seek(time);
                shell.select_clip_key(&clip.id, time);
            }
        }
        response.on_hover_text(
            "Keyframe: click to pick it and go to it, Shift-click to pick more, drag to move, Delete to remove",
        );
    }
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
    if response.hovered() || response.dragged_by(egui::PointerButton::Primary) {
        ui.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    if response.drag_started_by(egui::PointerButton::Primary)
        && let Some(pos) = pointer_pos(ui, &response)
    {
        let time = match edge {
            TrimEdge::Start => clip.start.0,
            TrimEdge::End => clip.start.0.saturating_add(clip.duration.0),
        };
        let edge_x = lane_x(lanes_left, shell.pixel_size(), shell.view_origin().0, time);
        shell.begin_clip_drag(
            clip.id.clone(),
            Some(edge),
            (pos.x, pos.y),
            (pos.x - edge_x, 0.0),
        );
    }
    if response.dragged_by(egui::PointerButton::Primary)
        && let Some(pos) = pointer_pos(ui, &response)
    {
        scroll_with_drag(shell, ui, f64::from(pos.x - lanes_left).max(0.0));
        let (grab_x, _) = shell.clip_grab(&clip.id);
        let x = f64::from(pos.x - grab_x - lanes_left).max(0.0);
        shell.preview_edge(clip.id.clone(), edge, x);
    }
    finish_clip_pointer(
        shell,
        ui,
        &response,
        &clip.id,
        lanes_left,
        LaneSpot::On(index),
        Some(edge),
    );
}

/// A green line along the ruler's foot over each stretch a render plays.
fn paint_rendered_spans(shell: &Shell, ui: &Ui, lanes_left: f32, ruler: Rect) {
    for (start, end) in shell.rendered_spans() {
        let x = |time: Micros| {
            lane_x(
                lanes_left,
                shell.pixel_size(),
                shell.view_origin().0,
                time.0,
            )
        };
        let line = Rect::from_min_max(
            Pos2::new(x(start).max(lanes_left), ruler.bottom() - 3.0),
            Pos2::new(x(end), ruler.bottom() - 1.0),
        );
        if line.width() > 0.0 && ruler.intersects(line) {
            ui.painter().with_clip_rect(ruler).rect_filled(
                line,
                0.0,
                Color32::from_rgb(0x3D, 0xDC, 0x6A),
            );
        }
    }
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
        // At the ruler's foot, under the labels.
        let flag = marker_flag(x, ruler.bottom() - 12.0);
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
    spot: LaneSpot,
    edge: Option<TrimEdge>,
) {
    if !response.drag_stopped_by(egui::PointerButton::Primary) {
        return;
    }
    let Some(pos) = pointer_pos(ui, response) else {
        shell.cancel_clip_drag();
        return;
    };
    let (grab_x, _) = shell.clip_grab(id);
    let x = f64::from(pos.x - grab_x - lanes_left).max(0.0);
    if edge.is_some() && shell.clip_edge(id).is_none() && !shell.drag_escaped(id) {
        shell.begin_clip_drag(id.clone(), edge, (pos.x, pos.y), (0.0, 0.0));
    }
    let landed = match spot {
        LaneSpot::On(row) => shell.finish_clip_drag(id.clone(), x, editor_lane_y(row)),
        LaneSpot::Between(row) => shell.finish_clip_drag_between(id.clone(), x, row),
    };
    if let Err(error) = landed {
        shell.notice_error(&error);
    }
    // The lanes above were painted from the draft before the drop, so this
    // frame would show the clip back where it was for a moment. Paint it again.
    ui.ctx().request_discard("a clip was dropped");
}

/// Where a body drag is over: a row, or the gap above row `n` (the row count
/// is under the last).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneSpot {
    On(usize),
    Between(usize),
}

/// How far into a row, from its top or bottom, a drop still opens a lane.
const LANE_EDGE: f32 = 6.0;

/// A row's middle is that row. Its top and bottom edges, the gaps between
/// rows, and the space above the first or under the last are between rows.
fn lane_spot(boxes: &[Rect], pos: Pos2) -> Option<LaneSpot> {
    let first = boxes.first()?;
    let last = boxes.last()?;
    if pos.y < first.top() + LANE_EDGE {
        return Some(LaneSpot::Between(0));
    }
    if pos.y > last.bottom() - LANE_EDGE {
        return Some(LaneSpot::Between(boxes.len()));
    }
    for (index, row) in boxes.iter().enumerate() {
        if pos.y < row.top() {
            return Some(LaneSpot::Between(index));
        }
        if pos.y <= row.bottom() {
            return Some(if pos.y < row.top() + LANE_EDGE {
                LaneSpot::Between(index)
            } else if pos.y > row.bottom() - LANE_EDGE {
                LaneSpot::Between(index + 1)
            } else {
                LaneSpot::On(index)
            });
        }
    }
    Some(LaneSpot::Between(boxes.len()))
}

/// The lane a held drag would open: a line in the gap where it opens, and
/// the dragged clip on that line.
fn paint_new_lane(shell: &Shell, ui: &Ui, boxes: &[Rect], lanes_left: f32) {
    let Some((row, _)) = shell.new_lane_preview() else {
        return;
    };
    let Some(y) = gap_y(boxes, row) else {
        return;
    };
    let Some(right) = boxes.first().map(Rect::right) else {
        return;
    };
    ui.painter().hline(
        lanes_left..=right,
        y,
        egui::Stroke::new(2.0, tokens::SELECTED),
    );
}

/// The dragged clip at its own size, under the pointer, above everything.
fn paint_floating_clip(
    shell: &Shell,
    ui: &Ui,
    lane: &crate::shell::LaneView,
    clip: &crate::shell::ClipSpan,
    rect: Rect,
    lanes_left: f32,
) {
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::unique("floating-clip"),
    ));
    // The whole track sheet, not this lane's band: the clip can be carried
    // into the band above main, onto main, or under it.
    let sheet = ui
        .data(|data| data.get_temp::<Rect>(lane_sheet_id()))
        .unwrap_or_else(|| ui.clip_rect());
    let painter = painter.with_clip_rect(Rect::from_x_y_ranges(
        lanes_left..=sheet.right(),
        sheet.y_range(),
    ));
    painter.rect_filled(
        rect,
        CornerRadius::same(7),
        lane_color(&lane.kind).gamma_multiply(0.9),
    );
    let stroke = if shell.is_selected(&clip.id) {
        tokens::SELECTED
    } else {
        Color32::WHITE.gamma_multiply(0.6)
    };
    painter.rect_stroke(
        rect,
        CornerRadius::same(7),
        egui::Stroke::new(1.5, stroke),
        egui::StrokeKind::Inside,
    );
    painter
        .with_clip_rect(rect.shrink(6.0).intersect(painter.clip_rect()))
        .text(
            rect.left_top() + Vec2::new(8.0, 6.0),
            Align2::LEFT_TOP,
            &clip.name,
            FontId::proportional(11.0),
            Color32::WHITE,
        );
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
        lane_size(lane),
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
    // The dragged clip floats under the pointer; its landing spot is an outline.
    if shell
        .move_preview()
        .is_some_and(|lead| lead.id == preview.id)
    {
        painter.rect_filled(
            clip_rect,
            CornerRadius::same(7),
            lane_color(&lane.kind).gamma_multiply(0.25),
        );
        painter.rect_stroke(
            clip_rect,
            CornerRadius::same(7),
            egui::Stroke::new(1.5, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
        return;
    }
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
    // A held trim draws every clip it moves where the edit will put it.
    if let Some(span) = shell.trim_preview(&clip.id) {
        return span;
    }
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

/// Rows for lanes given as (main, height), none scrolled: main in the
/// middle of `area` while the lanes fit around it.
#[cfg(test)]
fn lane_rects(kinds: Vec<(bool, f32)>, area: Rect) -> Vec<Rect> {
    lane_layout(&kinds, area, 0.0, 0.0).rows
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
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.vertical(|ui| {
                export_choices(shell, ui, view);
                ui.add_space(8.0);
                if view.replace_prompt {
                    ui.label("Replace the existing file?");
                }
                let busy = shell.job().is_some();
                if ui
                    .add_enabled(!busy, ui::Button::primary("Export"))
                    .clicked()
                {
                    if view.export_path.is_empty() {
                        let name = shell
                            .draft()
                            .map_or_else(String::new, |draft| draft.name.clone());
                        view.dialogs
                            .save_export(parent, &name, export_extension(view));
                    } else {
                        match write_export(shell, view) {
                            Ok(()) => view.export_open = false,
                            Err(ShellError::Overwrite(_)) => view.replace_prompt = true,
                            Err(error) => shell.notice_error(&error),
                        }
                    }
                }
            });
        });
    });
    if closed {
        view.export_open = false;
        view.replace_prompt = false;
    }
}

/// The Video, Audio, GIF and Caption sections of the export dialog.
fn export_choices(shell: &Shell, ui: &mut Ui, view: &mut ViewState) {
    use crate::shell::{ExportQuality, ExportSpan, GIF_RATES, GifExport, SoundFormat};
    use cutlass_decode::CodecId;
    use cutlass_export::{GIF_SHORT_SIDES, ProresProfile, SidecarKind};
    let export = &mut view.export;
    choices_row(ui, "Range", |ui| {
        let picked = !shell.selection().is_empty();
        let marked = shell.marked_span().is_some();
        let files = shell.library_export().len();
        let offered = |span| match span {
            ExportSpan::Whole => true,
            ExportSpan::Selected => picked,
            ExportSpan::Marked => marked,
            ExportSpan::Library => files > 0,
        };
        if !offered(export.span) {
            export.span = ExportSpan::Whole;
        }
        // The library's own row is there only when its menu asked.
        let library = match files {
            0 => None,
            1 => Some((ExportSpan::Library, "Library file")),
            _ => Some((ExportSpan::Library, "Library files")),
        };
        for (span, name) in [
            (ExportSpan::Whole, "Whole timeline"),
            (ExportSpan::Marked, "In to out"),
            (ExportSpan::Selected, "Selected clips"),
        ]
        .into_iter()
        .chain(library)
        {
            let label = ui::Choice::new(export.span == span, name);
            if ui.add_enabled(offered(span), label).clicked() {
                export.span = span;
            }
        }
    });
    ui.add(ui::Check::new(&mut export.picture, "Video"));
    if export.picture {
        choices_row(ui, "Format", |ui| {
            if ui.add(ui::Choice::new(!view.export_mov, "MP4")).clicked() {
                view.export_mov = false;
            }
            if ui.add(ui::Choice::new(view.export_mov, "MOV")).clicked() {
                view.export_mov = true;
            }
        });
        // ProRes and Motion JPEG are written in a MOV, AV1, VP8 and MPEG-2
        // in an MP4. MPEG-4 in either.
        let fits = |codec: CodecId| match codec {
            CodecId::Prores | CodecId::Mjpeg => view.export_mov,
            CodecId::Av1 | CodecId::Vp8 | CodecId::Mpeg2 => !view.export_mov,
            _ => true,
        };
        if !fits(export.codec) {
            export.codec = CodecId::H264;
        }
        choices_row(ui, "Codec", |ui| {
            for (codec, name) in [
                (CodecId::H264, "H.264"),
                (CodecId::Hevc, "HEVC"),
                (CodecId::Av1, "AV1"),
                (CodecId::Mpeg4, "MPEG-4"),
                (CodecId::Vp8, "VP8"),
                (CodecId::Mpeg2, "MPEG-2"),
                (CodecId::Prores, "ProRes"),
                (CodecId::Mjpeg, "Motion JPEG"),
            ] {
                if !fits(codec) {
                    continue;
                }
                let label = ui::Choice::new(export.codec == codec, name);
                if ui.add_enabled(shell.encodes(codec), label).clicked() {
                    export.codec = codec;
                }
            }
        });
        if export.codec == CodecId::Prores {
            choices_row(ui, "Profile", |ui| {
                for profile in ProresProfile::ALL {
                    if ui
                        .add(ui::Choice::new(export.prores == profile, profile.label()))
                        .clicked()
                    {
                        export.prores = profile;
                    }
                }
            });
        } else {
            choice_row(ui, "Quality", |ui| {
                ui::Segmented::show(ui, |ui| {
                    for (quality, name) in [
                        (ExportQuality::Lower, "Lower"),
                        (ExportQuality::Recommended, "Recommended"),
                        (ExportQuality::Higher, "Higher"),
                    ] {
                        let on = view.export_mbps.is_none() && export.quality == quality;
                        if ui.add(ui::Choice::new(on, name)).clicked() {
                            export.quality = quality;
                            view.export_mbps = None;
                        }
                    }
                    if ui
                        .add(ui::Choice::new(view.export_mbps.is_some(), "Custom"))
                        .clicked()
                        && view.export_mbps.is_none()
                    {
                        let side = u32::try_from(export.short_side).unwrap_or(1080);
                        let rate = export.quality.bitrate(side, export.codec);
                        view.export_mbps = Some(format!("{}", rate / 1_000_000));
                    }
                });
                if let Some(text) = &mut view.export_mbps {
                    ui::Field::new(text).width(48.0).show(ui);
                    ui.label("Mbps");
                }
            });
            choices_row(ui, "Rate", |ui| {
                if ui
                    .add(ui::Choice::new(!export.constant_rate, "VBR"))
                    .clicked()
                {
                    export.constant_rate = false;
                }
                if ui
                    .add(ui::Choice::new(export.constant_rate, "CBR"))
                    .clicked()
                {
                    export.constant_rate = true;
                }
            });
        }
        // MPEG-2 Main profile stops at 1920×1152, 1080p30, and the Table
        // 6-4 frame rates, so the steps past those are greyed.
        let mpeg2 = export.codec == CodecId::Mpeg2;
        let holds = |side: i32, rate: Option<FrameRate>| {
            !mpeg2 || shell.mpeg2_refusal(side, rate, 0).is_none()
        };
        choices_row(ui, "Frame rate", |ui| {
            for (rate, name) in [
                (None, "Project"),
                (Some(FrameRate::FPS_24), "24"),
                (Some(FrameRate::FPS_25), "25"),
                (Some(FrameRate::FPS_30), "30"),
                (Some(FrameRate::FPS_50), "50"),
                (Some(FrameRate::FPS_60), "60"),
            ] {
                let label = ui::Choice::new(export.frame_rate == rate, name);
                if ui
                    .add_enabled(holds(export.short_side, rate), label)
                    .clicked()
                {
                    export.frame_rate = rate;
                }
            }
        });
        choices_row(ui, "Short side", |ui| {
            for side in crate::shell::SHORT_SIDES {
                let label = ui::Choice::new(export.short_side == side, side.to_string());
                if ui
                    .add_enabled(holds(side, export.frame_rate), label)
                    .clicked()
                {
                    export.short_side = side;
                }
            }
        });
        if mpeg2 {
            let custom = view.export_mbps.as_deref().and_then(parse_mbps);
            let refusal =
                shell.mpeg2_refusal(export.short_side, export.frame_rate, custom.unwrap_or(0));
            if let Some(why) = refusal {
                ui.label(why);
            }
        }
    }
    ui.add(ui::Check::new(&mut export.sound, "Audio"));
    // PCM goes in a MOV or a WAV, never an MP4. FLAC, MP3, Opus and Vorbis
    // are sound alone, each in a file of its own (Opus and Vorbis in an Ogg file).
    let pcm_fits = view.export_mov || !export.picture;
    let alone_fits = !export.picture;
    let alone = matches!(
        export.audio.codec,
        CodecId::Flac | CodecId::Mp3 | CodecId::Opus | CodecId::Vorbis
    );
    if (!pcm_fits && export.audio.codec == CodecId::Pcm) || (!alone_fits && alone) {
        export.audio.codec = CodecId::Aac;
    }
    if export.sound {
        let audio = &mut export.audio;
        choices_row(ui, "Codec", |ui| {
            if ui
                .add(ui::Choice::new(audio.codec == CodecId::Aac, "AAC"))
                .clicked()
            {
                audio.codec = CodecId::Aac;
            }
            let mp3 = ui::Choice::new(audio.codec == CodecId::Mp3, "MP3");
            if ui.add_enabled(alone_fits, mp3).clicked() {
                audio.codec = CodecId::Mp3;
            }
            if ui
                .add(ui::Choice::new(audio.codec == CodecId::Alac, "ALAC"))
                .clicked()
            {
                audio.codec = CodecId::Alac;
            }
            if ui
                .add(ui::Choice::new(audio.codec == CodecId::Ac3, "AC-3"))
                .clicked()
            {
                audio.codec = CodecId::Ac3;
            }
            if ui
                .add(ui::Choice::new(audio.codec == CodecId::Eac3, "E-AC-3"))
                .clicked()
            {
                audio.codec = CodecId::Eac3;
            }
            let pcm = ui::Choice::new(audio.codec == CodecId::Pcm, "PCM");
            if ui.add_enabled(pcm_fits, pcm).clicked() {
                audio.codec = CodecId::Pcm;
            }
            let flac = ui::Choice::new(audio.codec == CodecId::Flac, "FLAC");
            if ui.add_enabled(alone_fits, flac).clicked() {
                audio.codec = CodecId::Flac;
            }
            let opus = ui::Choice::new(audio.codec == CodecId::Opus, "Opus");
            if ui.add_enabled(alone_fits, opus).clicked() {
                audio.codec = CodecId::Opus;
                audio.sample_rate = 48_000;
            }
            let vorbis = ui::Choice::new(audio.codec == CodecId::Vorbis, "Vorbis");
            if ui.add_enabled(alone_fits, vorbis).clicked() {
                audio.codec = CodecId::Vorbis;
            }
        });
        choices_row(ui, "Sample rate", |ui| {
            for rate in SoundFormat::SAMPLE_RATES {
                let name = if rate == 44_100 { "44.1 kHz" } else { "48 kHz" };
                // Opus is coded at 48 kHz only.
                let fits = rate == 48_000 || audio.codec != CodecId::Opus;
                let button = ui::Choice::new(audio.sample_rate == rate, name);
                if ui.add_enabled(fits, button).clicked() {
                    audio.sample_rate = rate;
                }
            }
        });
        choices_row(ui, "Channels", |ui| {
            for (channels, name) in [(2, "Stereo"), (1, "Mono")] {
                if ui
                    .add(ui::Choice::new(audio.channels == channels, name))
                    .clicked()
                {
                    audio.channels = channels;
                }
            }
        });
        // Lossless sound has no bitrate to choose.
        if !audio.lossless() {
            choices_row(ui, "Bitrate", |ui| {
                for rate in SoundFormat::BITRATES {
                    if ui
                        .add(ui::Choice::new(
                            audio.bitrate == rate,
                            format!("{} kbps", rate / 1000),
                        ))
                        .clicked()
                    {
                        audio.bitrate = rate;
                    }
                }
            });
        }
    }
    let mut gif_on = export.gif.is_some();
    if ui.add(ui::Check::new(&mut gif_on, "GIF")).changed() {
        export.gif = gif_on.then(GifExport::default);
    }
    if let Some(gif) = &mut export.gif {
        choices_row(ui, "Format", |ui| {
            if ui.add(ui::Choice::new(!gif.webp, "GIF")).clicked() {
                gif.webp = false;
            }
            if ui.add(ui::Choice::new(gif.webp, "WebP")).clicked() {
                gif.webp = true;
            }
        });
        choices_row(ui, "Size", |ui| {
            for side in GIF_SHORT_SIDES {
                if ui
                    .add(ui::Choice::new(gif.short_side == side, side.to_string()))
                    .clicked()
                {
                    gif.short_side = side;
                }
            }
        });
        choices_row(ui, "Frame rate", |ui| {
            for rate in GIF_RATES {
                if ui
                    .add(ui::Choice::new(
                        gif.frame_rate.numerator == rate,
                        rate.to_string(),
                    ))
                    .clicked()
                {
                    gif.frame_rate = FrameRate {
                        numerator: rate,
                        denominator: 1,
                    };
                }
            }
        });
        choices_row(ui, "Loop", |ui| {
            if ui.add(ui::Choice::new(!gif.looped, "Once")).clicked() {
                gif.looped = false;
            }
            if ui.add(ui::Choice::new(gif.looped, "Loop")).clicked() {
                gif.looped = true;
            }
        });
    }
    let mut captions_on = export.captions.is_some();
    if ui
        .add(ui::Check::new(&mut captions_on, "Caption file"))
        .changed()
    {
        export.captions = captions_on.then_some(SidecarKind::Srt);
    }
    if let Some(kind) = &mut export.captions {
        choices_row(ui, "File", |ui| {
            for (choice, name) in [(SidecarKind::Srt, "SRT"), (SidecarKind::Plain, "TXT")] {
                if ui.add(ui::Choice::new(*kind == choice, name)).clicked() {
                    *kind = choice;
                }
            }
        });
    }
}

/// The extension of the export's main file.
fn export_extension(view: &ViewState) -> &'static str {
    if view.export.picture {
        if view.export_mov { "mov" } else { "mp4" }
    } else if view.export.sound {
        view.export.audio.extension()
    } else {
        view.export.gif.map_or("gif", |gif| gif.extension())
    }
}

/// The question a transition dropped on a cut that has one asks: replace
/// it or leave it, with a tick that stops the question from then on.
fn transition_prompt(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let Some((old, new)) = shell.asked_transition() else {
        return;
    };
    let never = &mut view.never_ask_transition;
    let ((), end) = widgets::sheet(ui, "Replace transition", 420.0, "Replace", |ui| {
        ui.label(
            RichText::new(format!("This cut has {old}. Replace it with {new}?"))
                .size(tokens::TYPE_SIZE + 1.0)
                .color(tokens::TEXT),
        );
        ui.add_space(12.0);
        ui.add(ui::Check::new(never, "Do not ask again"));
    });
    let answered = match end {
        widgets::SheetEnd::Open => return,
        widgets::SheetEnd::Cancel => shell.answer_transition(false, false),
        widgets::SheetEnd::Confirm => shell.answer_transition(true, view.never_ask_transition),
    };
    view.never_ask_transition = false;
    if let Err(error) = answered {
        shell.notice_error(&error);
    }
}

/// The proxy switch and the resolution proxies are drawn at.
fn project_performance(
    ui: &mut Ui,
    proxies: &mut Option<bool>,
    side: &mut Option<u32>,
    now: (bool, u32),
) {
    let mut on = proxies.unwrap_or(now.0);
    widgets::form_row(ui, "Proxy", |ui| {
        if widgets::switch(ui, &mut on, true).changed() {
            *proxies = Some(on);
        }
    });
    let current = format!("{}p", side.unwrap_or(now.1));
    let labels: Vec<String> = cutlass_export::PROXY_SIDES
        .iter()
        .map(|side| format!("{side}p"))
        .collect();
    let names: Vec<&str> = labels.iter().map(String::as_str).collect();
    widgets::form_row(ui, "Resolution", |ui| {
        if let Some(index) = widgets::select(ui, "project-proxy-side", &current, &names) {
            *side = Some(cutlass_export::PROXY_SIDES[index]);
        }
    });
    widgets::form_hint(
        ui,
        "Only clips larger than this get a smaller copy for the viewer. Export reads the original files.",
    );
}

/// Name, canvas, frame rate and color space under Details, and proxies
/// under Performance, saved together.
fn project_dialog(shell: &mut Shell, ui: &mut Ui, view: &mut ViewState) {
    let folder = shell.project_folder().map(short_folder);
    let now = (shell.project_proxies(), shell.project_proxy_side());
    let performance = &mut view.project_performance;
    let (proxies, side) = (&mut view.project_proxies, &mut view.project_proxy_side);
    let Some(settings) = view.project_dialog.as_mut() else {
        return;
    };
    let ((), end) = widgets::sheet(ui, "Project settings", 560.0, "Save", |ui| {
        ui::Segmented::show(ui, |ui| {
            for (tab, name) in [(false, "Details"), (true, "Performance")] {
                if ui.add(ui::Choice::new(*performance == tab, name)).clicked() {
                    *performance = tab;
                }
            }
        });
        ui.add_space(8.0);
        if *performance {
            project_performance(ui, proxies, side, now);
            return;
        }
        widgets::form_section(ui, "Details");
        widgets::form_row(ui, "Name", |ui| {
            widgets::text_box(ui, &mut settings.name);
        });
        if let Some(folder) = &folder {
            widgets::form_row(ui, "Save to", |ui| {
                ui.add_space(8.0);
                ui.add(
                    egui::Label::new(
                        RichText::new(folder)
                            .size(tokens::TYPE_SIZE + 1.0)
                            .color(tokens::TEXT),
                    )
                    .truncate(),
                );
            });
        }
        let spaces: Vec<_> = cutlass_editor::COLOR_SPACES
            .iter()
            .filter(|(space, _)| *space >= 0)
            .collect();
        let names: Vec<&str> = spaces.iter().map(|(_, name)| *name).collect();
        let current = cutlass_editor::color_space_name(settings.color_space);
        widgets::form_row(ui, "Color space", |ui| {
            if let Some(index) = widgets::select(ui, "project-color-space", &current, &names) {
                settings.color_space = spaces[index].0;
            }
        });
        widgets::form_row(ui, "Free layer", |ui| {
            widgets::switch(ui, &mut settings.free_layer, true);
        });
        widgets::form_hint(
            ui,
            "Picture clips stack by the order Arrange gives them, not by their tracks.",
        );

        ui.add_space(8.0);
        widgets::form_section(ui, "Timeline");
        let ratio = cutlass_editor::PROJECT_RATIOS
            .iter()
            .find(|(ratio, _)| *ratio == settings.ratio)
            .map_or(settings.ratio.as_str(), |(_, name)| name)
            .to_owned();
        let names: Vec<&str> = cutlass_editor::PROJECT_RATIOS
            .iter()
            .map(|(_, name)| *name)
            .collect();
        widgets::form_row(ui, "Ratio", |ui| {
            if let Some(index) = widgets::select(ui, "project-ratio", &ratio, &names) {
                settings.ratio = cutlass_editor::PROJECT_RATIOS[index].0.to_owned();
            }
        });
        let size = cutlass_editor::PROJECT_RESOLUTIONS
            .iter()
            .find(|(side, _)| *side == settings.short_side)
            .map_or("Adapted", |(_, name)| name);
        let names: Vec<&str> = cutlass_editor::PROJECT_RESOLUTIONS
            .iter()
            .map(|(_, name)| *name)
            .collect();
        widgets::form_row(ui, "Resolution", |ui| {
            if let Some(index) = widgets::select(ui, "project-resolution", size, &names) {
                settings.short_side = cutlass_editor::PROJECT_RESOLUTIONS[index].0;
            }
        });
        let labels: Vec<String> = cutlass_editor::PROJECT_RATES
            .iter()
            .map(|rate| cutlass_editor::rate_label(*rate))
            .collect();
        let names: Vec<&str> = labels.iter().map(String::as_str).collect();
        let current = cutlass_editor::rate_label(settings.rate);
        widgets::form_row(ui, "Frame rate", |ui| {
            if let Some(index) = widgets::select(ui, "project-rate", &current, &names) {
                settings.rate = cutlass_editor::PROJECT_RATES[index];
                settings.drop_frame = cutlass_editor::is_drop_rate(settings.rate);
            }
        });
        let drop = cutlass_editor::is_drop_rate(settings.rate);
        widgets::form_row(ui, "Drop frame", |ui| {
            let mut on = settings.drop_frame && drop;
            if widgets::switch(ui, &mut on, drop).changed() {
                settings.drop_frame = on;
            }
        });
        widgets::form_hint(
            ui,
            if drop {
                "29.97 drops 2 frame numbers a minute, 59.94 drops 4. Every tenth minute keeps them."
            } else {
                "Only 29.97 and 59.94 fps can drop frame numbers."
            },
        );
    });
    match end {
        widgets::SheetEnd::Open => {}
        widgets::SheetEnd::Cancel => {
            view.project_dialog = None;
            view.project_performance = false;
            view.project_proxies = None;
            view.project_proxy_side = None;
        }
        widgets::SheetEnd::Confirm => {
            if let Some(settings) = view.project_dialog.take() {
                if let Err(error) = shell.save_project_settings(&settings) {
                    shell.notice_error(&error);
                    view.project_dialog = Some(settings);
                } else {
                    view.name_buffer.clear();
                    view.project_performance = false;
                    // The resolution first, so a project just turned on
                    // draws at the size it was given.
                    let side = view.project_proxy_side.take();
                    let on = view.project_proxies.take();
                    let saved = side
                        .map_or(Ok(()), |side| shell.set_project_proxy_side(side))
                        .and_then(|()| on.map_or(Ok(()), |on| shell.set_project_proxies(on)));
                    if let Err(error) = saved {
                        shell.notice_error(&error);
                    }
                }
            }
        }
    }
}

/// The project's folder and the one holding it: `Projects/Amina & Jules`.
fn short_folder(path: &std::path::Path) -> String {
    let mut parts: Vec<_> = path
        .components()
        .rev()
        .take(2)
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    parts.reverse();
    parts.join("/")
}

/// A dim label and its choices on one line.
/// A choice row whose options sit in one well.
fn choices_row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    choice_row(ui, label, |ui| {
        ui::Segmented::show(ui, add);
    });
}

fn choice_row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        // As tall as a well of choices, so the name sits on its middle.
        let tall = tokens::CHOICE_HEIGHT + 2.0 * f32::from(tokens::SEGMENT_PAD);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(84.0, tall), Sense::hover());
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(12.0),
            tokens::MUTED,
        );
        add(ui);
    });
}

fn apply_dialog(shell: &mut Shell, view: &mut ViewState, finished: Finished) {
    let Some(path) = finished.path else {
        return;
    };
    let result = match finished.picker {
        Picker::OpenProject => shell.open_folder(&path),
        Picker::Import => shell.start_import(&path),
        Picker::SaveExport => {
            view.export_path = path.display().to_string();
            write_export(shell, view)
        }
        Picker::SaveStill => shell.export_still(&path),
        Picker::SkillFolder => {
            let mut settings = shell.settings().clone();
            let path = path.display().to_string();
            if !settings
                .skill_folders
                .iter()
                .any(|folder| folder.path == path)
            {
                settings.skill_folders.push(cutlass_settings::SkillFolder {
                    path,
                    ..Default::default()
                });
            }
            propose_settings(shell, view, settings);
            Ok(())
        }
        Picker::Lut => shell.load_lut(&path),
        Picker::Replace => shell.start_replace(&path),
        Picker::Link => shell.start_link(&path),
        Picker::StudioClip => {
            shell.load_studio_clip(path);
            view.studio.use_clip();
            Ok(())
        }
        Picker::Cover => {
            if let Some(dialog) = view.cover.as_mut() {
                dialog.picked(path);
            }
            Ok(())
        }
        Picker::SaveLog => Shell::export_log(&path).map(|()| {
            shell.set_notice(format!("Saved the log to {}", path.display()));
        }),
    };
    match result {
        Ok(()) if finished.picker == Picker::SaveExport => view.export_open = false,
        Ok(()) => {}
        Err(ShellError::Overwrite(_)) if finished.picker == Picker::SaveExport => {
            view.replace_prompt = true;
        }
        Err(error) => shell.notice_error(&error),
    }
}

/// Open the shelf an import that just landed filled.
fn show_imported(shell: &mut Shell, view: &mut ViewState) {
    let Some(kind) = shell.take_imported() else {
        return;
    };
    view.library_shelf = match kind {
        crate::shell::ImportedKind::Video => Shelf::Media,
        crate::shell::ImportedKind::Audio => Shelf::Audio,
    };
    view.library_query.clear();
}

fn write_export(shell: &mut Shell, view: &ViewState) -> Result<(), ShellError> {
    let bitrate = match &view.export_mbps {
        Some(text) => Some(parse_mbps(text).ok_or_else(|| {
            ShellError::Unavailable("type the bitrate in Mbps, like 16".to_owned())
        })?),
        None => None,
    };
    shell.export(ExportRequest {
        path: std::path::PathBuf::from(&view.export_path).with_extension(export_extension(view)),
        replace: view.replace_prompt,
        bitrate,
        ..view.export.clone()
    })
}

/// Bits a second from a typed Mbps figure such as `16` or `7.5`.
fn parse_mbps(text: &str) -> Option<u32> {
    let mbps: f64 = text.trim().parse().ok()?;
    if !mbps.is_finite() || mbps <= 0.0 || mbps > 400.0 {
        return None;
    }
    // At most 400 Mbps, so the product fits.
    Some((mbps * 1_000_000.0).round() as u32)
}

fn shortcuts(shell: &mut Shell, ui: &Ui, view: &mut ViewState) {
    if !view.settings_open
        && shell.draft().is_some()
        && ui.input(|input| input.modifiers.command && input.key_pressed(egui::Key::J))
    {
        view.assistant.open = !view.assistant.open;
        return;
    }
    if ui.ctx().text_edit_focused() || view.settings_open || view.licenses.is_some() {
        return;
    }
    if shell.pen().is_some() && pen_keys(shell, ui) {
        return;
    }
    if let Some(chord) = ui.input(pressed_chord)
        && let Err(error) = shell.run_shortcut(&chord)
        && !error.to_string().contains("not available")
    {
        shell.notice_error(&error);
    }
}

/// The chord a key press makes, like `primary+shift+s`, whatever it is
/// bound to. Modifier keys alone make none.
fn pressed_chord(input: &egui::InputState) -> Option<String> {
    let key = input
        .keys_down
        .iter()
        .find(|key| input.key_pressed(**key))?;
    let name = key_name(*key)?;
    let modifiers = input.modifiers;
    let mut parts = Vec::new();
    // Control apart from the primary key, as on a Mac.
    if modifiers.ctrl && !modifiers.command {
        parts.push("ctrl");
    }
    if modifiers.command {
        parts.push("primary");
    }
    if modifiers.shift {
        parts.push("shift");
    }
    if modifiers.alt {
        parts.push("alt");
    }
    parts.push(&name);
    Some(parts.join("+"))
}

/// A key's name in a chord. Escape, Tab and Enter are left to the window.
fn key_name(key: egui::Key) -> Option<String> {
    use egui::Key;
    Some(
        match key {
            Key::Space => "space",
            Key::ArrowLeft => "left",
            Key::ArrowRight => "right",
            Key::ArrowUp => "up",
            Key::ArrowDown => "down",
            Key::Delete | Key::Backspace => "delete",
            Key::Comma => "comma",
            Key::Period => "period",
            Key::Home => "home",
            Key::End => "end",
            Key::Escape | Key::Tab | Key::Enter => return None,
            other => {
                let name = other.name();
                let single = name.chars().count() == 1
                    && name.chars().all(|letter| letter.is_ascii_alphanumeric());
                let function = name.starts_with('F')
                    && name.len() > 1
                    && name[1..].chars().all(|digit| digit.is_ascii_digit());
                if !(single || function) {
                    return None;
                }
                return Some(name.to_ascii_lowercase());
            }
        }
        .to_owned(),
    )
}

fn panel(ui: &mut Ui, rect: Rect) {
    ui::Surface::panel().paint(ui.painter(), rect);
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

    #[test]
    fn clip_menu_edit_rows_open_their_inspector_tab() {
        use cutlass_model::TrackType;
        let tab = |kind: TrackType, id| inspector_focus(&kind, id).map(|focus| focus.tab);
        assert_eq!(tab(TrackType::Video, "edit-speed"), Some(2));
        assert_eq!(tab(TrackType::Video, "edit-animation"), Some(3));
        assert_eq!(
            inspector_focus(&TrackType::Video, "edit-opacity").and_then(|focus| focus.section),
            Some("blend")
        );
        assert_eq!(tab(TrackType::Audio, "speed"), Some(1));
        assert_eq!(tab(TrackType::Audio, "volume"), Some(0));
        assert_eq!(tab(TrackType::Text, "edit-text"), Some(0));
        assert_eq!(tab(TrackType::Effect, "edit-effect"), Some(0));
        assert_eq!(tab(TrackType::Video, "copy"), None);
    }
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
    fn an_open_project_lays_out_the_same_at_every_display_scale() {
        use std::collections::BTreeMap;

        let temp = std::env::temp_dir().join(format!("cutlass-scale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut title = cutlass_model::MaterialText::new("title");
        title.content = "Hello".to_owned();
        draft.materials.texts.push(title);
        let mut text = cutlass_model::Track::main("text");
        text.kind = cutlass_model::TrackType::Text;
        text.segments.push(cutlass_model::Segment::new(
            "words",
            "title",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        draft.tracks.push(text);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        // One context moves between monitors, as a window dragged across
        // a mixed-DPI desk does.
        let mut frame = |scale: f32| {
            let mut input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            };
            input.viewports.insert(
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(scale),
                    ..Default::default()
                },
            );
            let mut placed = BTreeMap::new();
            // The second pass settles sizes measured in the first.
            for _ in 0..2 {
                let mut output = ctx.run_ui(input.clone(), |ui| {
                    show(&mut shell, ui, &mut view, None);
                });
                assert!((output.pixels_per_point - scale).abs() < f32::EPSILON);
                placed.clear();
                for line in galleys(&output) {
                    if let Some(pos) = text_pos(&output, &line) {
                        placed.insert(line, pos);
                    }
                }
                output.textures_delta.clear();
            }
            placed
        };
        let one = frame(1.0);
        assert!(one.contains_key("Hello") && one.len() > 30, "{one:?}");
        for scale in [2.0, 1.5, 1.0] {
            let other = frame(scale);
            for (line, pos) in &one {
                let moved = other
                    .get(line)
                    .unwrap_or_else(|| panic!("{line:?} is missing at {scale}x"));
                assert!(
                    (moved.x - pos.x).abs() <= 1.0 && (moved.y - pos.y).abs() <= 1.0,
                    "{line:?} moved from {pos:?} to {moved:?} at {scale}x"
                );
            }
        }
        drop(shell);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_steadied_clip_shows_its_level_under_stabilize() {
        use cutlass_model::{Draft, MaterialVideo, Segment, TimeRange, VideoStable};
        let temp = std::env::temp_dir().join(format!("cutlass-steady-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            stable: Some(VideoStable {
                level: VideoStable::MOST_STABLE,
                ..VideoStable::default()
            }),
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        assert_eq!(shell.stable_level(), VideoStable::MOST_STABLE);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: Some("stabilize"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in ["Stabilize", "Level", "Most stable"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            !lines.iter().any(|line| line.starts_with("Processing")),
            "no job runs"
        );
        drop(shell);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn an_enhanced_clip_shows_its_level_under_enhance_quality() {
        use cutlass_model::{Algorithm, Draft, MaterialVideo, Segment, TimeRange, VideoAlgorithm};
        let temp =
            std::env::temp_dir().join(format!("cutlass-enhance-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            video_algorithm: Some(VideoAlgorithm {
                algorithms: vec![Algorithm {
                    kind: Algorithm::QUALITY_ENHANCE.into(),
                    level: Algorithm::UHD,
                    ..Algorithm::default()
                }],
                ..VideoAlgorithm::default()
            }),
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        assert_eq!(shell.enhance_level(), Algorithm::UHD);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: Some("enhance"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in ["Enhance quality", "Level", "UHD"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            !lines.iter().any(|line| line.starts_with("Processing")),
            "no job runs"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }
    #[test]
    fn a_denoised_clip_shows_its_level_under_reduce_image_noise() {
        use cutlass_model::{Algorithm, Draft, MaterialVideo, Segment, TimeRange, VideoAlgorithm};
        let temp =
            std::env::temp_dir().join(format!("cutlass-denoise-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            video_algorithm: Some(VideoAlgorithm {
                algorithms: vec![Algorithm {
                    kind: Algorithm::NOISE_REDUCTION.into(),
                    level: Algorithm::STRONG,
                    ..Algorithm::default()
                }],
                ..VideoAlgorithm::default()
            }),
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        assert_eq!(shell.denoise_level(), Algorithm::STRONG);
        assert_eq!(shell.enhance_level(), 0);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: Some("denoise"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in ["Reduce image noise", "Level", "Strong"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            !lines.iter().any(|line| line.starts_with("Processing")),
            "no job runs"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn an_interpolated_clip_shows_its_rate_and_apply_to_all_under_optical_flow() {
        use cutlass_model::{Algorithm, Draft, MaterialVideo, Segment, TimeRange, VideoAlgorithm};
        let temp =
            std::env::temp_dir().join(format!("cutlass-interpolate-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            video_algorithm: Some(VideoAlgorithm {
                algorithms: vec![Algorithm {
                    kind: Algorithm::COMPLEMENT_FRAME.into(),
                    level: 50,
                    ..Algorithm::default()
                }],
                ..VideoAlgorithm::default()
            }),
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        assert_eq!(shell.interpolate_level(), 50);
        assert_eq!(shell.denoise_level(), 0);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: Some("interpolate"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in ["Optical flow", "Frame rate", "50 fps", "Apply to all"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            !lines.iter().any(|line| line.starts_with("Processing")),
            "no job runs"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_relit_clip_shows_its_group_of_lights_and_a_custom_one_its_color_and_angle() {
        use cutlass_model::{Algorithm, Draft, MaterialVideo, Segment, TimeRange, VideoAlgorithm};
        let temp =
            std::env::temp_dir().join(format!("cutlass-relight-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        for (id, preset) in [("ambient", "neon_two_tone"), ("custom", "custom")] {
            draft.materials.videos.push(MaterialVideo {
                video_algorithm: Some(VideoAlgorithm {
                    algorithms: vec![Algorithm {
                        kind: Algorithm::RELIGHT.into(),
                        level: 1,
                        preset: preset.into(),
                        strength: Some(0.4),
                        color: if preset == "custom" {
                            "#3366ff".into()
                        } else {
                            String::new()
                        },
                        angle: Some(90.0),
                        ..Algorithm::default()
                    }],
                    ..VideoAlgorithm::default()
                }),
                ..MaterialVideo::new(id)
            });
        }
        draft.tracks[0].segments.push(Segment::new(
            "main",
            "ambient",
            TimeRange::new(0, 2_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "other",
            "custom",
            TimeRange::new(2_000_000, 2_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut draw = |shell: &mut Shell, clip: &str| {
            shell.select(Some(SegmentId::new(clip)));
            let mut lines = Vec::new();
            for _ in 0..2 {
                view.inspector.focus(inspector::Focus {
                    tab: 0,
                    page: 0,
                    section: Some("relight"),
                });
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            Pos2::ZERO,
                            Vec2::new(1600.0, 2400.0),
                        )),
                        ..Default::default()
                    },
                    |ui| show(shell, ui, &mut view, None),
                );
                lines = galleys(&output);
                output.textures_delta.clear();
            }
            lines
        };
        let lines = draw(&mut shell, "main");
        let ambient = cutlass_effects::RELIGHT_PRESETS
            .iter()
            .filter(|preset| preset.group == cutlass_effects::RelightGroup::Ambient)
            .map(|preset| preset.name);
        for label in [
            "Relight", "Facial", "Ambient", "Creative", "Custom", "Strength", "40%",
        ]
        .into_iter()
        .chain(ambient)
        {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            !lines.iter().any(|line| line == "Angle"),
            "a preset has no angle"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("Reading depth")),
            "no job runs"
        );
        let lines = draw(&mut shell, "other");
        for label in ["Custom", "Strength", "Color", "Angle", "90°"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_reframed_clip_shows_its_shape_settings_and_apply_under_auto_reframe() {
        use cutlass_model::{Algorithm, Draft, MaterialVideo, Segment, TimeRange, VideoAlgorithm};
        let temp =
            std::env::temp_dir().join(format!("cutlass-reframe-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            video_algorithm: Some(VideoAlgorithm {
                algorithms: vec![Algorithm {
                    kind: Algorithm::AUTO_REFRAME.into(),
                    level: 1,
                    ratio: "1:1".into(),
                    stability: Some(3),
                    speed: Some(1),
                    ..Algorithm::default()
                }],
                ..VideoAlgorithm::default()
            }),
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        assert_eq!(shell.reframe().unwrap().ratio, "1:1");
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: Some("reframe"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in [
            "Auto reframe",
            "Aspect ratio",
            "1:1",
            "Image stabilization",
            "Stable",
            "Fast",
            "Apply",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            lines.iter().any(|line| line.starts_with("Camera moving")),
            "{lines:?}"
        );
        assert!(
            !lines
                .iter()
                .any(|line| line.starts_with("Finding the subject")),
            "no job runs"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_blurred_clip_shows_its_blur_blend_direction_and_speed_under_motion_blur() {
        use cutlass_model::{Algorithm, Draft, MaterialVideo, Segment, TimeRange, VideoAlgorithm};
        let temp =
            std::env::temp_dir().join(format!("cutlass-motion-blur-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            video_algorithm: Some(VideoAlgorithm {
                algorithms: vec![Algorithm {
                    kind: Algorithm::MOTION_BLUR.into(),
                    level: 1,
                    strength: Some(0.8),
                    blend: Some(0.25),
                    direction: Some(2),
                    speed: Some(4),
                    ..Algorithm::default()
                }],
                ..VideoAlgorithm::default()
            }),
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: Some("motion-blur"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in [
            "Motion blur",
            "Blur",
            "80",
            "Blend",
            "25",
            "Direction",
            "Backward",
            "Speed",
            "4 times",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(
            !lines.iter().any(|line| line.starts_with("Processing")),
            "no job runs"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn each_kind_of_clip_paints_its_own_inspector_panel() {
        use cutlass_model::{
            Draft, MaterialAudio, MaterialEffect, MaterialText, MaterialVideo, RawMaterial,
            Segment, TimeRange, Track, TrackType,
        };
        let temp = std::env::temp_dir().join(format!("cutlass-panels-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo::new("movie"));
        draft.materials.audios.push(MaterialAudio {
            id: "song".into(),
            ..MaterialAudio::default()
        });
        let mut title = MaterialText::new("title");
        title.content = "Hello".to_owned();
        draft.materials.texts.push(title);
        let mut caption = MaterialText::new("caption");
        caption.content = "said words".to_owned();
        caption.words.text = vec!["said".to_owned(), "words".to_owned()];
        caption.words.start_time = vec![0, 1_500];
        caption.words.end_time = vec![1_000, 2_000];
        draft.materials.texts.push(caption);
        let mut filter = MaterialEffect::new("warm");
        filter.name = "Warm".to_owned();
        draft.materials.effects.push(filter);
        draft.materials.stickers.push(RawMaterial {
            id: "star".into(),
            ..RawMaterial::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        for (id, kind, material) in [
            ("sound", TrackType::Audio, "song"),
            ("words", TrackType::Text, "title"),
            ("said", TrackType::Text, "caption"),
            ("look", TrackType::Filter, "warm"),
            ("sticker", TrackType::Sticker, "star"),
        ] {
            let mut track = Track::main(id);
            track.kind = kind;
            track
                .segments
                .push(Segment::new(id, material, TimeRange::new(0, 2_000_000)));
            draft.tracks.push(track);
        }
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        for (id, shown, hidden) in [
            (
                "main",
                [
                    "Video",
                    "Audio",
                    "Speed",
                    "Animation",
                    "Adjust",
                    "Basic",
                    "Remove BG",
                    "Mask",
                    "Retouch",
                    "Position & Size",
                    "Scale",
                    "Uniform scale",
                    "Position",
                    "Rotate",
                    "Blend",
                    "Stabilize",
                    "Remove flicker",
                    "Motion blur",
                    "Canvas",
                    "Later",
                ]
                .as_slice(),
                // Blend starts shut on a video clip, as in the prototype.
                [
                    "Font size",
                    "Stroke",
                    "Volume",
                    "Mode",
                    "Opacity",
                    "Line width",
                    "Flip upside down",
                    "Tracking",
                    "AI remove",
                    "AI expand",
                    "AI remix",
                    "Lip sync",
                    "AI movement",
                    "Video translator",
                    "AI stylize",
                    "Removing video flickers",
                ]
                .as_slice(),
            ),
            (
                "sound",
                &[
                    "Basic",
                    "Voice changer",
                    "Speed",
                    "Volume",
                    "Fade in",
                    "Fade out",
                    "Normalize loudness",
                    "Enhance voice",
                    "Reduce noise",
                    "Separate audio",
                    "Vocals",
                    "Keep remaining audio",
                    "Fill channel",
                    "Later",
                ],
                &[
                    "Position & Size",
                    "Stroke",
                    "Video",
                    "Audio translator",
                    "Speech to song",
                ],
            ),
            (
                "words",
                &[
                    "Text",
                    "Animation",
                    "Tracking",
                    "Basic",
                    "Bubble",
                    "Effects",
                    "Font",
                    "Font size",
                    "Pattern",
                    "Case",
                    "Color",
                    "Character",
                    "Alignment",
                    "Line width",
                    "Preset style",
                    "Position & Size",
                    "Plane rotation",
                    "Flip upside down",
                    "Blend",
                    "Stroke",
                    "Background",
                    "Glow",
                    "Shadow",
                    "Hello",
                ],
                &["Volume", "Rotate", "Mode", "Text to speech", "AI avatars"],
            ),
            (
                "said",
                &[
                    "Captions", "Text", "Tracking", "Search", "Replace", "Re-run", "?", "1", "+",
                ],
                &["Volume", "Font size", "Text to speech", "AI avatars"],
            ),
            (
                "look",
                &["Filter", "Warm", "Amount", "Temp"],
                &["Position & Size", "Volume", "Basic"],
            ),
            (
                "sticker",
                &["Sticker", "Animation", "Position & Size", "Opacity"],
                &["Volume", "Speed"],
            ),
        ] {
            shell.select(Some(SegmentId::new(id)));
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            let lines = galleys(&output);
            output.textures_delta.clear();
            for label in shown {
                assert!(
                    lines.iter().any(|line| line == label),
                    "{id} {label}: {lines:?}"
                );
            }
            for label in hidden {
                assert!(
                    !lines.iter().any(|line| line == label),
                    "{id} {label}: {lines:?}"
                );
            }
        }
        // The video clip's Adjust tab shows its sliders with their values.
        shell.select(Some(SegmentId::new("main")));
        shell
            .set_adjust(cutlass_effects::AdjustKey::Temperature, 0.4)
            .unwrap();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 4,
                page: 0,
                section: Some("adjust"),
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
        }
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let lines = galleys(&output);
        output.textures_delta.clear();
        for label in [
            "Adjust", "Exposure", "Temp", "Tint", "Clarity", "Grain", "40",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        // A caption's Text tab: Templates before Bubble, its paragraphs, and
        // Transform in place of Position & Size.
        shell.select(Some(SegmentId::new("said")));
        let mut lines = Vec::new();
        for _ in 0..2 {
            view.inspector.focus(inspector::Focus {
                tab: 0,
                page: 0,
                section: None,
            });
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            lines = galleys(&output);
            output.textures_delta.clear();
        }
        for label in [
            "Templates",
            "Bubble",
            "Tracking",
            "The 1 paragraph",
            "Words",
            "Transform",
            "Scale",
            "Position",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        for label in ["Position & Size", "Plane rotation"] {
            assert!(
                !lines.iter().any(|line| line == label),
                "{label}: {lines:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_speed_preset_is_picked_and_a_dragged_curve_point_is_one_undo_step() {
        use cutlass_model::SpeedPreset;
        let temp = std::env::temp_dir().join(format!("cutlass-speed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft
            .materials
            .videos
            .push(cutlass_model::MaterialVideo::new("movie"));
        let mut clip = cutlass_model::Segment::new(
            "main",
            "movie",
            cutlass_model::TimeRange::new(0, 2_000_000),
        );
        clip.source_range = Some(cutlass_model::TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(clip);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 1200.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let click = |shell: &mut Shell, view: &mut ViewState, pos: Pos2| {
            let _ = frame(
                shell,
                view,
                vec![egui::Event::PointerMoved(pos), press(pos, true)],
            );
            let _ = frame(shell, view, vec![press(pos, false)]);
            frame(shell, view, Vec::new())
        };
        view.inspector.focus(inspector::Focus {
            tab: 2,
            page: 1,
            section: None,
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&output);
        for label in [
            "Normal",
            "Curve",
            "None",
            "Montage",
            "Jump cut",
            "Flash out",
            "Custom",
            "Change pitch",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        let hero = text_pos(&output, "Hero").unwrap();
        let output = click(&mut shell, &mut view, hero);
        let inspection = shell.inspection().unwrap();
        assert_eq!(inspection.speed_curve, Some(SpeedPreset::Hero.points()));
        let factor = cutlass_model::curve_speed_factor(&SpeedPreset::Hero.points()).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let length = (2_000_000.0 * factor).round() as i64;
        assert_eq!(inspection.duration, Micros(length));
        // Drag the third point, in the slow middle, up to run faster.
        let dots: Vec<Pos2> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Circle(circle)
                    if circle.radius == 4.5 && circle.fill == egui::Color32::WHITE =>
                {
                    Some(circle.center)
                }
                _ => None,
            })
            .collect();
        assert_eq!(dots.len(), 6, "{dots:?}");
        let slow = dots[2];
        let higher = slow - Vec2::new(0.0, 40.0);
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(slow), press(slow, true)],
        );
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(slow - Vec2::new(0.0, 20.0))],
        );
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(higher)],
        );
        // Nothing is sent while the point is held.
        assert_eq!(
            shell.inspection().unwrap().speed_curve,
            Some(SpeedPreset::Hero.points())
        );
        let _ = frame(&mut shell, &mut view, vec![press(higher, false)]);
        let output = frame(&mut shell, &mut view, Vec::new());
        let curve = shell.inspection().unwrap().speed_curve.unwrap();
        assert!(curve[2].y > 0.4, "{curve:?}");
        assert!(galleys(&output).iter().any(|line| line == "Custom"));
        shell.undo().unwrap();
        assert_eq!(
            shell.inspection().unwrap().speed_curve,
            Some(SpeedPreset::Hero.points())
        );
        // ⌥K draws the curve on the clip's bar, and only then.
        let curve_lines = |output: &egui::FullOutput| {
            output
                .shapes
                .iter()
                .filter(|clipped| match &clipped.shape {
                    egui::Shape::Path(path) => {
                        matches!(
                            path.stroke.color,
                            egui::epaint::ColorMode::Solid(color)
                                if color == egui::Color32::from_rgb(0xff, 0xc8, 0x3a)
                        )
                    }
                    _ => false,
                })
                .count()
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(curve_lines(&output), 0);
        shell.toggle_speed_curves();
        let output = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(curve_lines(&output), 1);
        shell.toggle_speed_curves();
        // None goes back to the constant speed and its length.
        let none = text_pos(&output, "None").unwrap();
        let _ = click(&mut shell, &mut view, none);
        let inspection = shell.inspection().unwrap();
        assert_eq!(inspection.speed_curve, None);
        assert_eq!(inspection.duration, Micros(2_000_000));
        // The pitch switch is one step too.
        shell.set_change_pitch(true).unwrap();
        assert!(shell.inspection().unwrap().change_pitch);
        shell.undo().unwrap();
        assert!(!shell.inspection().unwrap().change_pitch);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_colour_pages_paint_and_a_dragged_curve_point_is_saved() {
        use cutlass_effects::{CurveChannel, Wheel};
        let temp = std::env::temp_dir().join(format!("cutlass-grade-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft
            .materials
            .videos
            .push(cutlass_model::MaterialVideo::new("movie"));
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "main",
            "movie",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 2400.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        for (page, labels) in [
            (1, ["Red", "Hue", "Saturation", "Lightness"].as_slice()),
            (2, ["All", "Luma", "Green (G)", "Blue (B)"].as_slice()),
            (
                3,
                [
                    "Shadows",
                    "Middle grey",
                    "Highlights",
                    "Brightness",
                    "Offset",
                ]
                .as_slice(),
            ),
        ] {
            view.inspector.focus(inspector::Focus {
                tab: 4,
                page,
                section: None,
            });
            let _ = frame(&mut shell, &mut view, Vec::new());
            let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
            for label in labels {
                assert!(
                    lines.iter().any(|line| line == label),
                    "{page} {label}: {lines:?}"
                );
            }
        }
        // Back on Curves, drag the top end of the line down.
        view.inspector.focus(inspector::Focus {
            tab: 4,
            page: 2,
            section: None,
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let ends: Vec<Pos2> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Circle(circle)
                    if circle.radius == 4.5 && circle.fill == egui::Color32::WHITE =>
                {
                    Some(circle.center)
                }
                _ => None,
            })
            .collect();
        assert_eq!(ends.len(), 2, "{ends:?}");
        let top = ends
            .iter()
            .copied()
            .min_by(|a, b| a.y.total_cmp(&b.y))
            .unwrap();
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let lower = top + Vec2::new(0.0, 60.0);
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(top), press(top, true)],
        );
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(top + Vec2::new(0.0, 20.0))],
        );
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(lower)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(lower, false)]);
        let _ = frame(&mut shell, &mut view, Vec::new());
        let grade = shell.inspection().unwrap().grade;
        let curve = grade.curve(CurveChannel::Rgb);
        assert_eq!(curve.len(), 2, "{curve:?}");
        assert!(curve[1].1 < 0.9 && curve[1].0 > 0.99, "{curve:?}");
        // One drag is one undo step.
        shell.undo().unwrap();
        assert!(shell.inspection().unwrap().grade.is_identity());
        assert_eq!(shell.inspection().unwrap().grade.wheel(Wheel::Lift).x, 0.0);
        // A .cube picked in the LUT dialog lands on the clip, and the Basic
        // page shows it with its strength.
        let cube = temp.join("Teal punch.cube");
        std::fs::write(
            &cube,
            "LUT_3D_SIZE 2\n0 0 0\n0 1 0\n1 0 0\n1 1 0\n0 0 1\n0 1 1\n1 0 1\n1 1 1\n",
        )
        .unwrap();
        view.dialogs.finish_for_test(Picker::Lut, Some(cube));
        view.inspector.focus(inspector::Focus {
            tab: 4,
            page: 0,
            section: Some("lut"),
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        assert_eq!(
            shell.inspection().unwrap().lut,
            Some(("Teal punch".to_owned(), 1.0))
        );
        for label in ["LUT", "Teal punch", "Intensity", "100", "Protect skin tone"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        // The Remove BG page keys green out and shows the key's values, with
        // the parts still to build marked Later.
        shell.set_chroma("#00ff00", 0.4, 0.25).unwrap();
        view.inspector.focus(inspector::Focus {
            tab: 0,
            page: 1,
            section: Some("chroma"),
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        for label in [
            "Auto removal",
            "Stroke",
            "Paper",
            "Spacing",
            "Object texture",
            "Custom removal",
            "Select object",
            "Smart brush",
            "Apply",
            "Chroma key",
            "Color picker",
            "Green",
            "Intensity",
            "40",
            "Shadow",
            "25",
            "Feather edge",
            "Clean up edge",
            "Later",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        // The Retouch page lays out every touch-up, all marked Later.
        view.inspector.focus(inspector::Focus {
            tab: 0,
            page: 3,
            section: None,
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        for label in [
            "Face",
            "Body presets",
            "Auto styles",
            "Skin",
            "Smooth",
            "Skin tone",
            "Features",
            "Eyes",
            "Makeup",
            "Natural",
            "Manual",
            "Protect face",
            "Later",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        shell.set_chroma("", 0.0, 0.0).unwrap();
        assert!(shell.inspection().unwrap().chroma.is_none());
        assert!(shell.set_chroma("#00ff00", 2.0, 0.0).is_err());
        // The Mask page: a click on a shape masks the clip, and its
        // sliders show; None takes the mask off.
        view.inspector.focus(inspector::Focus {
            tab: 0,
            page: 2,
            section: None,
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(
            galleys(&output)
                .iter()
                .any(|line| line == "Pick a shape to mask the clip.")
        );
        let circle = text_pos(&output, "Circle").expect("circle shape") + Vec2::new(4.0, 4.0);
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(circle)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(circle, true)]);
        let _ = frame(&mut shell, &mut view, vec![press(circle, false)]);
        let mask = shell.inspection().unwrap().mask.expect("a mask");
        assert_eq!(mask.shape, cutlass_effects::MaskShape::Circle);
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        for label in [
            "Mask1 Circle",
            "Split",
            "Filmstrip",
            "Text",
            "Mask settings",
            "Position",
            "X",
            "Y",
            "Rotation",
            "Size",
            "W",
            "H",
            "Feather",
            "Invert",
            "Track mask",
            "Direction",
            "Later",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(!lines.iter().any(|line| line == "Round corners"));
        shell
            .set_mask(Some(cutlass_effects::Mask {
                shape: cutlass_effects::MaskShape::Rectangle,
                ..mask
            }))
            .unwrap();
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(galleys(&output).iter().any(|line| line == "Round corners"));
        // The Mask switch takes the mask off.
        let label = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) if text.galley.text() == "Mask" => Some(text.pos),
                _ => None,
            })
            .max_by(|a, b| a.y.total_cmp(&b.y))
            .expect("the Mask switch's name");
        let switch = label + Vec2::new(-14.0, 8.0);
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(switch)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(switch, true)]);
        let _ = frame(&mut shell, &mut view, vec![press(switch, false)]);
        assert!(shell.inspection().unwrap().mask.is_none());
        shell.undo().unwrap();
        assert_eq!(
            shell.inspection().unwrap().mask.map(|mask| mask.shape),
            Some(cutlass_effects::MaskShape::Rectangle)
        );
        // The Adjust tab's Mask page keeps the colour work inside a shape,
        // apart from the clip's own mask.
        view.inspector.focus(inspector::Focus {
            tab: 4,
            page: 4,
            section: None,
        });
        let _ = frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(
            galleys(&output)
                .iter()
                .any(|line| line == "Pick a shape to keep the colour work inside it.")
        );
        let star = text_pos(&output, "Stars").expect("star shape") + Vec2::new(4.0, 4.0);
        let _ = frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(star)]);
        let _ = frame(&mut shell, &mut view, vec![press(star, true)]);
        let _ = frame(&mut shell, &mut view, vec![press(star, false)]);
        let inspection = shell.inspection().unwrap();
        assert_eq!(
            inspection.color_mask.map(|mask| mask.shape),
            Some(cutlass_effects::MaskShape::Star)
        );
        assert_eq!(
            inspection.mask.map(|mask| mask.shape),
            Some(cutlass_effects::MaskShape::Rectangle)
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// A local model on this machine that answers each POST with the next
    /// of `replies`, then stops listening.
    fn local_model(replies: Vec<serde_json::Value>) -> (u16, std::thread::JoinHandle<usize>) {
        use std::io::{BufRead, BufReader, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut answered = 0;
            for message in replies {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let answer = serde_json::json!({ "choices": [{ "index": 0, "message": message }] })
                    .to_string();
                let mut stream = stream;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{answer}",
                    answer.len()
                )
                .unwrap();
                answered += 1;
            }
            answered
        });
        (port, server)
    }

    #[test]
    fn the_assistant_panel_sends_runs_a_step_and_shows_the_answer() {
        let temp = std::env::temp_dir().join(format!("cutlass-chat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "a",
            "missing",
            cutlass_model::TimeRange::new(0, 3_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (port, server) = local_model(vec![
            serde_json::json!({
                "role": "assistant",
                "content": "Moving the playhead.",
                "tool_calls": [{ "id": "c1", "type": "function",
                    "function": { "name": "player_seek", "arguments": "{\"input\":\"1500000\"}" } }],
            }),
            serde_json::json!({ "role": "assistant", "content": "The playhead is at 1.5 s." }),
        ]);
        let mut settings = Settings::default();
        settings.model.local_url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        settings.model.local_name = "test-model".to_owned();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, settings);
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 1000.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let click = |shell: &mut Shell, view: &mut ViewState, at: Pos2| {
            let _ = frame(shell, view, vec![egui::Event::PointerMoved(at)]);
            let _ = frame(shell, view, vec![press(at, true)]);
            frame(shell, view, vec![press(at, false)])
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        let toggle = text_pos(&output, "Assistant").expect("the toggle") + Vec2::new(4.0, 4.0);
        let _ = click(&mut shell, &mut view, toggle);
        assert!(view.assistant.open);
        let output = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&output);
        for label in ["test-model on this machine", "Send", assistant::HINT] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        // Type in the box and send.
        let hint = text_pos(&output, assistant::HINT).expect("the box") + Vec2::new(4.0, 4.0);
        let _ = click(&mut shell, &mut view, hint);
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::Text("go to 1.5 seconds".to_owned())],
        );
        let output = frame(&mut shell, &mut view, Vec::new());
        let send = text_pos(&output, "Send").expect("send") + Vec2::new(4.0, 4.0);
        let _ = click(&mut shell, &mut view, send);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view.assistant.busy() {
            assert!(
                std::time::Instant::now() < deadline,
                "the turn did not finish"
            );
            let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
            if view.assistant.busy() {
                assert!(
                    lines
                        .iter()
                        .any(|line| line == "Stop" || line == "Working…"),
                    "{lines:?}"
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(server.join().unwrap(), 2);
        assert_eq!(shell.playhead(), Micros(1_500_000));
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        for label in [
            "go to 1.5 seconds",
            "Moving the playhead.",
            "✓ player.seek 1500000",
            "The playhead is at 1.5 s.",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert!(view.assistant.draft.is_empty());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_menu_bar_item_runs_its_command() {
        use cutlass_model::{Segment, SegmentId, TimeRange};

        let temp = std::env::temp_dir().join(format!("cutlass-menubar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "color".into(),
                is_color_clip: true,
                solid_color: "0xFF336699".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0]
            .segments
            .push(Segment::new("one", "color", TimeRange::new(0, 2_000_000)));
        draft.duration = Micros(2_000_000);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        // With no project, a project command says so and nothing runs.
        run_menu(&mut shell, &mut view, None, crate::menubar::EXPORT);
        assert!(!view.export_open);
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("one")));
        shell.seek(Micros(1_000_000));
        run_menu(&mut shell, &mut view, None, "split");
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 2);
        run_menu(&mut shell, &mut view, None, "undo");
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 1);
        run_menu(&mut shell, &mut view, None, crate::menubar::EXPORT);
        assert!(view.export_open);
        run_menu(&mut shell, &mut view, None, crate::menubar::SETTINGS);
        assert!(view.settings_open);
        run_menu(&mut shell, &mut view, None, crate::menubar::LICENSES);
        assert!(view.licenses.is_some());
        // Every item runs without a panic.
        for item in crate::menubar::items() {
            if [
                crate::menubar::OPEN,
                crate::menubar::IMPORT,
                crate::menubar::EXPORT_LOG,
            ]
            .contains(&item.id)
            {
                continue; // These open the system's file dialog.
            }
            run_menu(&mut shell, &mut view, None, item.id);
        }
        drop(shell);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_effect_studio_opens_from_the_menu_and_draws_what_is_being_made() {
        use crate::studio::{CurveKind, Slot, Stack};
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        assert!(!view.studio.open);
        run_menu(&mut shell, &mut view, None, crate::menubar::STUDIO);
        assert!(view.studio.open);
        let ctx = egui::Context::default();
        let frame = |shell: &mut Shell, view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0))),
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            let painted = painted_text(&output);
            output.textures_delta.clear();
            painted
        };
        // Nothing open: the list, and what to do next. The editor's own
        // screens are not drawn under it.
        let painted = frame(&mut shell, &mut view);
        for text in [
            "Effect studio",
            "New effect",
            "Shake",
            "Open an effect or a transition",
        ] {
            assert!(painted.contains(text), "{text} in {painted}");
        }
        assert!(!shell.chrome.project_bar && !shell.chrome.projects);

        // An effect: its passes, the picked pass's values, its controls
        // and its card. Without a GPU the preview says so.
        shell.studio_mut().new_effect();
        let slot = Slot::pass(Stack::Effect, 0, "radius");
        shell.studio_mut().set_kind(&slot, CurveKind::Keys);
        shell.studio_mut().pick_slot(Some(slot.clone()));
        shell.studio_mut().pick_key(Some(1));
        shell.studio_mut().add_control();
        let painted = frame(&mut shell, &mut view);
        for text in [
            "Passes · 1 of 8",
            "1  Blur",
            "Radius",
            "2 keys",
            "Controls · 1 of 6",
            "Radius of pass 1",
            "On the shelf",
            "Card scene",
            "The preview needs the GPU.",
            "Unsaved changes",
            "Repeat",
        ] {
            assert!(painted.contains(text), "{text} in {painted}");
        }
        // The other ways a value moves draw their own numbers.
        for kind in [CurveKind::Wave, CurveKind::Random, CurveKind::Fixed] {
            shell.studio_mut().set_kind(&slot, kind);
            frame(&mut shell, &mut view);
        }

        // A transition: two stacks and the swap, and no controls.
        shell.studio_mut().new_transition();
        let painted = frame(&mut shell, &mut view);
        for text in [
            "Outgoing picture · 0 of 8",
            "Incoming picture · 0 of 8",
            "The swap",
            "Progress",
            "Lasts",
        ] {
            assert!(painted.contains(text), "{text} in {painted}");
        }
        assert!(!painted.contains("Controls ·"), "{painted}");

        // A document that cannot be saved says why.
        shell.studio_mut().set_id("Not An Id");
        let painted = frame(&mut shell, &mut view);
        assert!(painted.contains("id or name is not usable"), "{painted}");
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
        // A lane head shows icons, not its name.
        assert!(!lines.iter().any(|line| line == "Main"), "{lines:?}");
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
    fn a_click_on_the_preview_lands_on_the_canvas_pixel_under_it() {
        let placed = Rect::from_min_size(Pos2::new(100.0, 50.0), Vec2::new(480.0, 270.0));
        let size = Vec2::new(1920.0, 1080.0);
        assert_eq!(
            canvas_pixel(placed, size, Pos2::new(100.0, 50.0)),
            (0.0, 0.0)
        );
        assert_eq!(
            canvas_pixel(placed, size, Pos2::new(340.0, 185.0)),
            (960.0, 540.0)
        );
        assert_eq!(screen_point(placed, size, (1920.0, 1080.0)), placed.max);
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
        // Wide enough that the transport shows all four round buttons.
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 800.0)));
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
            // The round transport buttons, left to right: full screen, the
            // guides, the toggle, the still.
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
        // Quality, full screen, guides, background, still.
        assert_eq!(buttons.len(), 5, "{buttons:?}");
        let toggle = buttons[3];
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
        let rows = lane_rects(
            vec![
                (false, row_height(RowSize::Thin)),
                (true, row_height(RowSize::Main)),
                (false, row_height(RowSize::Audio)),
            ],
            area,
        );
        let heights: Vec<f32> = rows.iter().map(Rect::height).collect();
        assert_eq!(heights, [40.0, 76.0, 58.0]);
        assert_eq!(rows[1].center().y, 200.0);
        assert_eq!(rows[2].top() - rows[1].bottom(), LANE_GAP);
    }

    #[test]
    fn a_dragged_row_height_moves_the_rows_around_it_and_its_clips_grow() {
        let area = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 400.0));
        let rows = lane_rects(vec![(false, 100.0), (true, 90.0), (false, 30.0)], area);
        assert_eq!(rows[0].height(), 100.0);
        assert_eq!(rows[1].top() - rows[0].bottom(), LANE_GAP);
        assert_eq!(rows[2].height(), 30.0);
        let tall = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 100.0));
        let usual = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, row_height(RowSize::Thin)));
        let (decoders, encoders) = crate::register_codecs();
        let shell = Shell::new(decoders, encoders, Settings::default());
        let grown = clip_bounds(&shell, RowSize::Thin, Micros(0), Micros(1), tall, 0.0);
        assert_eq!(grown.height(), 100.0 - 2.0 * tokens::CLIP_INSET);
        let kept = clip_bounds(&shell, RowSize::Thin, Micros(0), Micros(1), usual, 0.0);
        assert_eq!(kept.height(), tokens::CLIP_HEIGHT);
    }

    #[test]
    fn the_meter_fills_its_bands_up_to_the_level_and_caps_a_clip() {
        let paint = |channels| {
            let ctx = egui::Context::default();
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(40.0, 300.0));
                paint_meter(ui, rect, channels);
            });
            output.textures_delta.clear();
            let fills: Vec<(Color32, Rect)> = output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::Rect(rect) => Some((rect.fill, rect.rect)),
                    _ => None,
                })
                .collect();
            fills
        };
        let quiet = cutlass_playback::ChannelReading {
            level: -30.0,
            hold: -20.0,
            clipped: false,
        };
        let fills = paint([quiet, quiet]);
        assert_eq!(
            fills
                .iter()
                .filter(|(color, _)| *color == tokens::METER_LOW)
                .count(),
            2
        );
        assert!(!fills.iter().any(|(color, _)| *color == tokens::METER_LOUD));
        assert!(!fills.iter().any(|(color, _)| *color == tokens::METER_HOT));
        let hot = cutlass_playback::ChannelReading {
            level: 0.0,
            hold: 0.0,
            clipped: true,
        };
        let fills = paint([hot, quiet]);
        // The left bar runs through all three bands and wears the clip cap.
        assert_eq!(
            fills
                .iter()
                .filter(|(color, _)| *color == tokens::METER_HOT)
                .count(),
            2
        );
        assert_eq!(
            fills
                .iter()
                .filter(|(color, _)| *color == tokens::METER_LOUD)
                .count(),
            1
        );
        let rest = cutlass_playback::LevelMeter::new().reading();
        let fills = paint(rest);
        assert!(!fills.iter().any(|(color, _)| *color == tokens::METER_LOW));
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

    /// The cards at the top of the library, by name.
    fn top_cards(draft: &cutlass_model::Draft, shelf: Shelf, query: &str) -> Vec<BinCard> {
        let store = cutlass_model::VirtualStore::default();
        let place = BinPlace {
            store: &store,
            folder: "",
            sort: cutlass_editor::LibrarySort::Name,
        };
        bin_cards(draft, &place, shelf, query)
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
        let cards = top_cards(&draft, Shelf::Media, "");
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
        let cards = top_cards(&draft, Shelf::Media, "");
        assert_eq!(cards.len(), 1);
        assert!(cards[0].on_timeline);
        assert!(top_cards(&draft, Shelf::Audio, "").is_empty());
        assert!(top_cards(&draft, Shelf::Media, "nope").is_empty());
        let blends = top_cards(&draft, Shelf::Transitions, "");
        assert_eq!(blends.len(), 202);
        assert_eq!(
            blends[..3]
                .iter()
                .map(|card| card.name.as_str())
                .collect::<Vec<_>>(),
            ["Dissolve", "Dip to Black", "Dip to White"]
        );
        assert_eq!(blends[0].meta, "0.5s");
        assert!(!blends[0].on_timeline);
        // The dips, and Blur Dip.
        assert_eq!(top_cards(&draft, Shelf::Transitions, "dip").len(), 3);
        // The pushes, and Elastic Push.
        assert_eq!(top_cards(&draft, Shelf::Transitions, "push").len(), 5);
        let layers = top_cards(&draft, Shelf::Adjustments, "");
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].id, cutlass_editor::ADJUSTMENT_CARD);
        assert_eq!(layers[0].name, "Adjustment");
        assert!(top_cards(&draft, Shelf::Adjustments, "adj").len() == 1);
        assert!(top_cards(&draft, Shelf::Adjustments, "nope").is_empty());
        let filters = top_cards(&draft, Shelf::Filters, "");
        assert_eq!(filters.len(), cutlass_effects::looks().len());
        assert_eq!(filters[0].name, cutlass_effects::looks()[0].name);
        assert_eq!(filters[0].look, CardLook::Picture(None));
        // Noir and Neo Noir.
        assert_eq!(top_cards(&draft, Shelf::Filters, "noir").len(), 2);
    }

    #[test]
    fn a_double_click_opens_a_compound_clip_and_back_leaves_it() {
        let temp = std::env::temp_dir().join(format!("cutlass-compound-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut video = cutlass_model::MaterialVideo::new("movie");
        video.kind = "photo".to_owned();
        video.path = "gone.png".to_owned();
        draft.materials.videos.push(video);
        for (id, start) in [("a", 0), ("b", 3_000_000)] {
            draft.tracks[0].segments.push(cutlass_model::Segment::new(
                id,
                "movie",
                cutlass_model::TimeRange::new(start, 3_000_000),
            ));
        }
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("a")));
        shell.toggle_selected(SegmentId::new("b"));
        shell.create_compound().unwrap();
        shell.select(None);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 1000.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let _ = frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(text_pos(&output, "‹ Back").is_none());
        let clip = text_pos(&output, COMPOUND_TITLE).expect("the compound clip on its row")
            + Vec2::new(4.0, 4.0);
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(clip), press(clip, true)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(clip, false)]);
        let _ = frame(&mut shell, &mut view, vec![press(clip, true)]);
        let _ = frame(&mut shell, &mut view, vec![press(clip, false)]);
        assert_eq!(shell.compound_depth(), 1);
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 2);
        let output = frame(&mut shell, &mut view, Vec::new());
        let back = text_pos(&output, "‹ Back").expect("the Back button");
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(back), press(back, true)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(back, false)]);
        assert_eq!(shell.compound_depth(), 0);
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 1);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_replace_dialog_swaps_the_clips_media_and_keeps_its_edits_in_one_step() {
        let temp = std::env::temp_dir().join(format!("cutlass-replace-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut old = cutlass_model::MaterialVideo::new("old");
        old.kind = "photo".to_owned();
        old.path = "gone.png".to_owned();
        draft.materials.videos.push(old);
        let mut clip =
            cutlass_model::Segment::new("main", "old", cutlass_model::TimeRange::new(0, 3_000_000));
        clip.source_range = Some(cutlass_model::TimeRange::new(0, 3_000_000));
        clip.clip.alpha = 0.5;
        draft.tracks[0].segments.push(clip);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let before = shell.draft().unwrap().clone();
        let mut view = ViewState::new();
        let picture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.jpg");
        view.dialogs.finish_for_test(Picker::Replace, Some(picture));
        land_import(&mut shell, &mut view);
        let draft = shell.draft().unwrap();
        let segment = &draft.tracks[0].segments[0];
        let material = draft
            .materials
            .videos
            .iter()
            .find(|video| video.id == segment.material_id)
            .unwrap();
        assert!(material.path.ends_with("pattern.jpg"), "{}", material.path);
        assert_eq!(segment.target, cutlass_model::TimeRange::new(0, 3_000_000));
        assert_eq!(segment.clip.alpha, 0.5);
        assert!(temp.join("Resources").join("pattern.jpg").is_file());
        shell.undo().unwrap();
        assert_eq!(shell.draft().unwrap(), &before);
        let _ = std::fs::remove_dir_all(&temp);
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
        land_import(&mut shell, &mut view);
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

    fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }

    #[test]
    fn the_settings_window_opens_on_its_keys_and_rebinds_a_shortcut() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let run = |shell: &mut Shell, view: &mut ViewState, keys: Vec<egui::Event>| {
            let mut events = Vec::new();
            for event in keys {
                if let egui::Event::Key { modifiers, .. } = &event {
                    events.push(egui::Event::ModifiersChanged(*modifiers));
                }
                events.push(event);
            }
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        run(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Comma, egui::Modifiers::COMMAND)],
        );
        assert!(view.settings_open);
        let painted = painted_text(&run(&mut shell, &mut view, Vec::new()));
        for text in [
            "Settings",
            "Playback",
            "Panels",
            "Hardware codecs",
            "Shortcuts",
        ] {
            assert!(painted.contains(text), "{text} in {painted}");
        }
        view.settings_tab = SettingsTab::Shortcuts;
        let painted = painted_text(&run(&mut shell, &mut view, Vec::new()));
        assert!(painted.contains("Play or pause"), "{painted}");
        assert!(painted.contains("Reset shortcuts"), "{painted}");
        view.recording = Some("split");
        run(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Space, egui::Modifiers::NONE)],
        );
        assert_eq!(view.recording, None);
        assert_eq!(shell.settings().keys_for("split"), "space");
        assert_eq!(shell.settings().keys_for("play"), "");
        assert!(
            shell.notice().contains("took Space from Play or pause"),
            "{}",
            shell.notice()
        );
        view.recording = Some("play");
        run(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Escape, egui::Modifiers::NONE)],
        );
        assert_eq!(view.recording, None);
        assert!(view.settings_open);
        assert_eq!(shell.settings().keys_for("play"), "");
        run(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Escape, egui::Modifiers::NONE)],
        );
        assert!(!view.settings_open);
    }

    #[test]
    fn turning_on_what_leaves_the_machine_waits_for_the_users_allow() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        view.settings_open = true;
        view.settings_tab = SettingsTab::Assistant;
        let ctx = egui::Context::default();
        let run = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let click = |pos: Pos2| {
            let pos = pos + Vec2::new(4.0, 6.0);
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), event(true), event(false)]
        };
        let painted = painted_text(&run(&mut shell, &mut view, Vec::new()));
        for text in [
            "Model",
            "Address",
            "http://127.0.0.1:11434/v1/chat/completions",
            "qwen2.5:7b",
            "Cloud model",
            "MCP servers",
            "Skill folders",
        ] {
            assert!(painted.contains(text), "{text} in {painted}");
        }
        view.dialogs.finish_for_test(
            Picker::SkillFolder,
            Some(std::path::PathBuf::from("/skills")),
        );
        let output = run(&mut shell, &mut view, Vec::new());
        assert!(view.approval.is_some());
        let output = if painted_text(&output).contains("Add the skills in /skills") {
            output
        } else {
            run(&mut shell, &mut view, Vec::new())
        };
        assert!(painted_text(&output).contains("Add the skills in /skills"));
        let refuse = text_pos(&output, "Don't allow").expect("the refuse button");
        run(&mut shell, &mut view, click(refuse));
        assert!(view.approval.is_none());
        assert!(shell.settings().skill_folders.is_empty());

        let mut cloud = shell.settings().clone();
        cloud.model.cloud = true;
        cloud.memory = false;
        propose_settings(&mut shell, &mut view, cloud);
        assert!(!shell.settings().model.cloud);
        let output = run(&mut shell, &mut view, Vec::new());
        assert!(painted_text(&output).contains("Turn on the cloud model"));
        let allow = text_pos(&output, "Allow").expect("the allow button");
        run(&mut shell, &mut view, click(allow));
        assert!(view.approval.is_none());
        assert!(shell.settings().model.cloud);
        assert!(!shell.settings().memory);
        // A change that sends nothing away is written at once.
        let mut off = shell.settings().clone();
        off.model.cloud = false;
        propose_settings(&mut shell, &mut view, off);
        assert!(view.approval.is_none());
        assert!(!shell.settings().model.cloud);
    }

    #[test]
    fn a_hidden_side_panel_gives_its_room_to_the_preview() {
        let temp = std::env::temp_dir().join(format!("cutlass-hide-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let window = Vec2::new(1280.0, 800.0);
        let shown = sizes(&shell, window);
        shell.update_settings(|settings| {
            settings.set_panel_open(PANEL_LIBRARY, false);
            settings.set_panel_open(PANEL_INSPECTOR, false);
        });
        let hidden = sizes(&shell, window);
        assert_eq!(hidden.library, 0.0);
        assert_eq!(hidden.inspector, 0.0);
        assert_eq!(hidden.preview, window.x);
        assert!(hidden.preview > shown.preview);
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, window)),
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        assert!(!shell.chrome.library);
        assert!(!shell.chrome.inspector);
        assert!(shell.chrome.preview);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_licenses_window_lists_the_crates_and_the_log_exports() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        view.licenses = Some(cutlass_licenses::notices());
        let ctx = egui::Context::default();
        let run = |shell: &mut Shell, view: &mut ViewState, events| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let painted = painted_text(&run(&mut shell, &mut view, Vec::new()));
        assert!(
            painted.contains("Cutlass is licensed MIT OR Apache-2.0"),
            "{painted}"
        );
        assert!(painted.contains("accesskit "), "{painted}");
        // The open sticker sets are credited, CC BY ones by name and link.
        assert!(painted.contains("Stickers"), "{painted}");
        assert!(
            painted.contains("Noto Emoji by Google, licensed Apache-2.0."),
            "{painted}"
        );
        assert!(
            painted.contains("Noto Animated Emoji by Google, licensed CC-BY-4.0."),
            "{painted}"
        );
        assert!(
            painted.contains("https://googlefonts.github.io/noto-emoji-animation/"),
            "{painted}"
        );
        run(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Escape, egui::Modifiers::NONE)],
        );
        assert!(view.licenses.is_none());

        let temp = std::env::temp_dir().join(format!("cutlass-log-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let out = temp.join("log.txt");
        view.dialogs
            .finish_for_test(Picker::SaveLog, Some(out.clone()));
        run(&mut shell, &mut view, Vec::new());
        assert!(out.is_file());
        assert!(
            shell.notice().starts_with("Saved the log"),
            "{}",
            shell.notice()
        );
        view.dialogs
            .finish_for_test(Picker::SaveLog, Some(temp.clone()));
        run(&mut shell, &mut view, Vec::new());
        assert_eq!(shell.notice(), "the log path is a folder");
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
        land_import(&mut shell, &mut view);
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
        assert!(
            view.library_menu
                .as_ref()
                .is_some_and(|menu| menu.target == LibraryTarget::Files)
        );
        let row = text_pos(&output, "Delete").expect("the Delete row") + Vec2::new(4.0, 6.0);
        frame(
            &mut shell,
            &mut view,
            click(row, egui::PointerButton::Primary),
        );
        assert!(view.library_menu.is_none());
        // Delete takes the card out of the library; the file stays in the draft.
        assert_eq!(shell.notice(), "Deleted pattern.png");
        assert!(shell.draft().unwrap().materials.videos.len() == 1);
        assert!(
            shell
                .library_files(
                    cutlass_editor::LibraryShelf::Media,
                    "",
                    cutlass_editor::LibrarySort::Name,
                    ""
                )
                .is_empty()
        );
        assert!(shell.unsaved());
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Every circle `output` paints: its middle and its radius.
    fn circles(output: &egui::FullOutput) -> Vec<(Pos2, f32)> {
        fn collect(shape: &egui::Shape, found: &mut Vec<(Pos2, f32)>) {
            match shape {
                egui::Shape::Circle(circle) => found.push((circle.center, circle.radius)),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, found);
                    }
                }
                _ => {}
            }
        }
        let mut found = Vec::new();
        for clipped in &output.shapes {
            collect(&clipped.shape, &mut found);
        }
        found
    }

    #[test]
    fn a_library_movie_makes_and_cancels_its_proxy_from_its_menu_under_a_clock() {
        let temp = std::env::temp_dir().join(format!("cutlass-proxy-menu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let file = temp.join("clip.mp4");
        std::fs::write(&file, b"held, so never read").unwrap();
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut movie = cutlass_model::MaterialVideo::new("movie");
        movie.kind = "video".to_owned();
        movie.path = file.to_string_lossy().into_owned();
        movie.width = 1920;
        movie.height = 1080;
        movie.duration = Micros(1_000_000);
        draft.materials.videos.push(movie);
        let project = temp.join("project");
        cutlass_store::Folder::new(draft).save(&project).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.set_proxies_folder(temp.join("proxies"));
        shell.hold_proxies();
        shell.open_folder(&project).unwrap();
        // The project makes none by itself, so the menu is what asks.
        shell.set_project_proxies(false).unwrap();
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
            output
        };
        let click = |pos: Pos2, button: egui::PointerButton| {
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), event(true), event(false)]
        };
        let clock = |output: &egui::FullOutput| {
            circles(output)
                .iter()
                .any(|(_, radius)| (radius - 5.5).abs() < 0.01)
        };
        let making = |shell: &Shell| {
            matches!(
                shell.proxy_state("movie"),
                crate::shell::ProxyState::Making(_)
            )
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        let name = text_pos(&output, "clip.mp4").expect("library card");
        // Nothing is on the way, so there is no clock.
        assert!(!making(&shell));
        assert!(!clock(&output));
        let card = name + Vec2::new(20.0, -30.0);

        // The menu makes one, and the clock runs by the name.
        frame(
            &mut shell,
            &mut view,
            click(card, egui::PointerButton::Secondary),
        );
        let output = frame(&mut shell, &mut view, Vec::new());
        let text = painted_text(&output);
        assert!(!text.contains("Cancel proxy"), "{text}");
        let row = text_pos(&output, "Make proxy").expect("the Make proxy row");
        frame(
            &mut shell,
            &mut view,
            click(row + Vec2::new(4.0, 6.0), egui::PointerButton::Primary),
        );
        assert_eq!(shell.notice(), "Making 1 proxy");
        assert!(making(&shell));
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(clock(&output), "no clock by the name");
        let moved = text_pos(&output, "clip.mp4").expect("library card");
        assert!(moved.x > name.x, "the name makes room for the clock");

        // While it is on the way the menu cancels it, and the clock goes.
        frame(
            &mut shell,
            &mut view,
            click(card, egui::PointerButton::Secondary),
        );
        let output = frame(&mut shell, &mut view, Vec::new());
        let text = painted_text(&output);
        assert!(!text.contains("Make proxy"), "{text}");
        let row = text_pos(&output, "Cancel proxy").expect("the Cancel proxy row");
        frame(
            &mut shell,
            &mut view,
            click(row + Vec2::new(4.0, 6.0), egui::PointerButton::Primary),
        );
        assert_eq!(shell.notice(), "Cancelled 1 proxy");
        assert!(!making(&shell));
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(!clock(&output));
        assert_eq!(text_pos(&output, "clip.mp4"), Some(name));
        assert!(text_pos(&output, "Proxy").is_none());

        // Once its proxy is in the folder, the name wears a Proxy badge
        // in place of the clock.
        let made = cutlass_export::proxy_name(&file, shell.project_proxy_side()).unwrap();
        std::fs::create_dir_all(temp.join("proxies")).unwrap();
        std::fs::write(temp.join("proxies").join(made), b"a proxy").unwrap();
        shell.make_proxy(&["movie".to_owned()]).unwrap();
        assert_eq!(shell.notice(), "The proxy is ready");
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(!clock(&output));
        let badge = text_pos(&output, "Proxy").expect("a Proxy badge");
        let moved = text_pos(&output, "clip.mp4").expect("library card");
        assert!(badge.x < moved.x && moved.x > name.x, "{badge:?} {moved:?}");
        assert!(
            (badge.y - moved.y).abs() < 3.0,
            "the badge is on the name's line"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_project_dialogs_performance_tab_sets_the_proxy_switch_and_resolution() {
        let (mut shell, temp) = open_span("proxies-switch", 1_000_000);
        assert!(shell.project_proxies());
        assert_eq!(shell.project_proxy_side(), 720);
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
            output
        };
        let click = |pos: Pos2| {
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            [
                vec![egui::Event::PointerMoved(pos), event(true)],
                vec![event(false)],
            ]
        };
        let press = |shell: &mut Shell, view: &mut ViewState, label: &str| {
            let output = frame(shell, view, Vec::new());
            let at = last_text_pos(&output, label).unwrap_or_else(|| panic!("no {label}"));
            for events in click(at) {
                frame(shell, view, events);
            }
        };
        // The switch's knob is the 9-point circle on the Proxy row.
        let flip = |shell: &mut Shell, view: &mut ViewState| {
            let output = frame(shell, view, Vec::new());
            let label = last_text_pos(&output, "Proxy").expect("the Proxy row");
            let knob = circles(&output)
                .into_iter()
                .find(|(center, radius)| {
                    (radius - 9.0).abs() < 0.01 && (center.y - label.y).abs() < 14.0
                })
                .expect("the Proxy switch")
                .0;
            for events in click(knob) {
                frame(shell, view, events);
            }
        };
        let pick = |shell: &mut Shell, view: &mut ViewState, name: &str| {
            let output = frame(shell, view, Vec::new());
            let picker =
                last_text_pos(&output, "Resolution").expect("Resolution") + Vec2::new(240.0, 0.0);
            for events in click(picker) {
                frame(shell, view, events);
            }
            press(shell, view, name);
        };

        // A line only the Details tab has; the window behind the dialog
        // names the frame rate and the rest too.
        const DETAILS: &str = "Picture clips stack by the order";
        // Details is shown first; proxies are under Performance.
        view.project_dialog = shell.project_settings();
        let text = painted_text(&frame(&mut shell, &mut view, Vec::new()));
        assert!(text.contains(DETAILS) && !text.contains("Proxy"), "{text}");
        press(&mut shell, &mut view, "Performance");
        let text = painted_text(&frame(&mut shell, &mut view, Vec::new()));
        assert!(!text.contains(DETAILS), "{text}");
        assert!(text.contains("720p"), "{text}");
        assert!(text.contains("Only clips larger than this"), "{text}");

        // Changed and cancelled, the project is as it was.
        flip(&mut shell, &mut view);
        pick(&mut shell, &mut view, "540p");
        assert_eq!(view.project_proxies, Some(false));
        assert_eq!(view.project_proxy_side, Some(540));
        press(&mut shell, &mut view, "Cancel");
        assert!(view.project_dialog.is_none());
        assert!(shell.project_proxies());
        assert_eq!(shell.project_proxy_side(), 720);
        assert!(!shell.unsaved());

        // Changed and saved, both are kept, and that is not an undo step.
        view.project_dialog = shell.project_settings();
        let text = painted_text(&frame(&mut shell, &mut view, Vec::new()));
        assert!(text.contains(DETAILS), "the dialog opens on Details again");
        press(&mut shell, &mut view, "Performance");
        flip(&mut shell, &mut view);
        pick(&mut shell, &mut view, "540p");
        press(&mut shell, &mut view, "Save");
        assert!(view.project_dialog.is_none(), "{}", shell.notice());
        assert!(!shell.project_proxies());
        assert_eq!(shell.project_proxy_side(), 540);
        assert!(shell.unsaved());
        assert!(shell.undo().is_err(), "the choices are not an edit");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn settings_turns_new_projects_proxies_off_or_picks_their_resolution() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        view.settings_open = true;
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
        let press = |shell: &mut Shell, view: &mut ViewState, label: &str| {
            let output = frame(shell, view, Vec::new());
            let pos = last_text_pos(&output, label).unwrap_or_else(|| panic!("no {label}"));
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            frame(
                shell,
                view,
                vec![egui::Event::PointerMoved(pos), event(true)],
            );
            frame(shell, view, vec![event(false)]);
            frame(shell, view, Vec::new());
        };
        let chosen = |shell: &Shell| {
            let playback = &shell.settings().playback;
            (playback.proxies, playback.proxy_side)
        };
        assert_eq!(chosen(&shell), (true, 720));
        press(&mut shell, &mut view, "540p");
        assert_eq!(chosen(&shell), (true, 540));
        press(&mut shell, &mut view, "Off");
        assert_eq!(chosen(&shell), (false, 540), "the resolution is kept");
        press(&mut shell, &mut view, "720p");
        assert_eq!(chosen(&shell), (true, 720));
    }

    /// The pale bars `output` paints for transitions: each filled rectangle
    /// in one of the bar's fills.
    fn transition_bars_painted(output: &egui::FullOutput) -> Vec<Rect> {
        fn collect(shape: &egui::Shape, found: &mut Vec<Rect>) {
            match shape {
                egui::Shape::Rect(rect)
                    if [
                        TRANSITION_FILL,
                        TRANSITION_FILL_HOVERED,
                        TRANSITION_FILL_PICKED,
                    ]
                    .contains(&rect.fill) =>
                {
                    found.push(rect.rect);
                }
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, found);
                    }
                }
                _ => {}
            }
        }
        let mut found = Vec::new();
        for clipped in &output.shapes {
            collect(&clipped.shape, &mut found);
        }
        found
    }

    #[test]
    fn a_transition_is_a_bar_over_its_cut_that_is_picked_resized_and_deleted() {
        let temp =
            std::env::temp_dir().join(format!("cutlass-transition-bar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        for (id, start) in [("a", 0), ("b", 4_000_000)] {
            draft.tracks[0].segments.push(cutlass_model::Segment::new(
                id,
                "color",
                cutlass_model::TimeRange::new(start, 4_000_000),
            ));
        }
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
            output
        };
        let button = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        // No transition, no bar.
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(transition_bars_painted(&output).is_empty());

        // One second of Dissolve on the cut at four seconds.
        let card = crate::shell::transition_offers()[0].card_id();
        shell.pixel_size_for_test(0.000_05);
        shell
            .drop_transition(
                &card,
                4_000_000.0 * shell.pixel_size(),
                tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap();
        shell.select(None);
        let a = SegmentId::new("a");
        shell.set_transition_length(&a, Micros(1_000_000)).unwrap();
        let output = frame(&mut shell, &mut view, Vec::new());
        let bars = transition_bars_painted(&output);
        assert_eq!(bars.len(), 1, "{bars:?}");
        let bar = bars[0];
        // A second at this zoom is 50 points, half each side of the cut.
        assert!((bar.width() - 50.0).abs() < 1.0, "{bar:?}");
        assert!(shell.selected_transition().is_none());

        // A click on the bar picks the transition, not the clip under it.
        frame(
            &mut shell,
            &mut view,
            vec![
                egui::Event::PointerMoved(bar.center()),
                button(bar.center(), true),
            ],
        );
        frame(&mut shell, &mut view, vec![button(bar.center(), false)]);
        assert_eq!(
            shell.selected_transition().map(|bar| bar.segment_id),
            Some(a.clone())
        );
        assert!(shell.selection().is_empty());
        let output = frame(&mut shell, &mut view, Vec::new());
        let text = painted_text(&output);
        for label in ["Transition", "Dissolve", "Duration", "Apply to all"] {
            assert!(text.contains(label), "no {label} in the panel: {text}");
        }

        // Its right edge dragged 25 points out makes it a second longer,
        // as it grows both ways, in one undo step.
        let edge = Pos2::new(bar.right() - 2.0, bar.center().y);
        let out = edge + Vec2::new(25.0, 0.0);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(edge), button(edge, true)],
        );
        for step in 1..=5 {
            let at = edge + Vec2::new(5.0 * step as f32, 0.0);
            frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(at)]);
        }
        frame(&mut shell, &mut view, vec![button(out, false)]);
        let length = shell.selected_transition().unwrap().duration.0;
        assert!((length - 2_000_000).abs() < 100_000, "{length}");
        shell.undo().unwrap();
        assert_eq!(shell.selected_transition().unwrap().duration.0, 1_000_000);
        assert_eq!(
            shell.draft().unwrap().tracks[0].segments[1].target.start,
            Micros(4_000_000),
            "no clip moved"
        );

        // Delete takes the picked transition off and leaves both clips.
        frame(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Delete, egui::Modifiers::NONE)],
        );
        assert!(shell.transitions().is_empty(), "{}", shell.notice());
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 2);
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(transition_bars_painted(&output).is_empty());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_transition_dropped_on_another_asks_and_can_stop_asking() {
        let temp =
            std::env::temp_dir().join(format!("cutlass-transition-ask-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        for (id, start) in [("a", 0), ("b", 4_000_000)] {
            draft.tracks[0].segments.push(cutlass_model::Segment::new(
                id,
                "color",
                cutlass_model::TimeRange::new(start, 4_000_000),
            ));
        }
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.pixel_size_for_test(0.000_05);
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
            output
        };
        let press = |shell: &mut Shell, view: &mut ViewState, label: &str| {
            let output = frame(shell, view, Vec::new());
            let pos = last_text_pos(&output, label).unwrap_or_else(|| panic!("no {label}"));
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            frame(
                shell,
                view,
                vec![egui::Event::PointerMoved(pos), event(true)],
            );
            frame(shell, view, vec![event(false)]);
            frame(shell, view, Vec::new());
        };
        let offers = crate::shell::transition_offers();
        let (x, y) = (4_000_000.0 * shell.pixel_size(), tokens::LANE_HEIGHT / 2.0);
        shell.drop_transition(&offers[0].card_id(), x, y).unwrap();
        let text = painted_text(&frame(&mut shell, &mut view, Vec::new()));
        assert!(!text.contains("Replace transition"), "{text}");

        // A second one opens the question; Cancel leaves the first.
        shell.drop_transition(&offers[1].card_id(), x, y).unwrap();
        let text = painted_text(&frame(&mut shell, &mut view, Vec::new()));
        assert!(
            text.contains("This cut has Dissolve. Replace it with Dip to Black?"),
            "{text}"
        );
        assert!(text.contains("Do not ask again"), "{text}");
        press(&mut shell, &mut view, "Cancel");
        assert!(shell.asked_transition().is_none());
        assert_eq!(shell.transitions()[0].name, "Dissolve");

        // Ticked and replaced, the next one lands without the question.
        shell.drop_transition(&offers[1].card_id(), x, y).unwrap();
        press(&mut shell, &mut view, "Do not ask again");
        assert!(view.never_ask_transition);
        press(&mut shell, &mut view, "Replace");
        assert_eq!(shell.transitions()[0].name, "Dip to Black");
        assert!(shell.settings().confirm.replace_transition_unasked);
        assert!(
            !view.never_ask_transition,
            "the tick is cleared for next time"
        );
        shell.drop_transition(&offers[0].card_id(), x, y).unwrap();
        assert_eq!(shell.transitions()[0].name, "Dissolve");
        let text = painted_text(&frame(&mut shell, &mut view, Vec::new()));
        assert!(!text.contains("Replace transition"), "{text}");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_new_folder_is_named_opened_and_left_by_the_path_bar() {
        let temp = std::env::temp_dir().join(format!("cutlass-folders-{}", std::process::id()));
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
        land_import(&mut shell, &mut view);
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
        let click = |pos: Pos2, button: egui::PointerButton| {
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), event(true), event(false)]
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        // At the top there is no path bar.
        assert!(text_pos(&output, "All").is_none());
        assert!(text_pos(&output, "Free stock").is_some());
        let card = text_pos(&output, "pattern.png").expect("library card");
        // Right-click below the cards for the empty-space menu.
        let empty = card + Vec2::new(0.0, 60.0);
        let output = frame(
            &mut shell,
            &mut view,
            click(empty, egui::PointerButton::Secondary),
        );
        assert_eq!(
            view.library_menu.as_ref().map(|menu| menu.target.clone()),
            Some(LibraryTarget::Empty)
        );
        let row = text_pos(&output, "New folder").expect("New folder row") + Vec2::new(4.0, 6.0);
        frame(
            &mut shell,
            &mut view,
            click(row, egui::PointerButton::Primary),
        );
        let (id, typed) = view.library_renaming.clone().expect("a name field");
        assert_eq!(typed, "Untitled folder");
        view.library_renaming = Some((id.clone(), "Shots".to_owned()));
        frame(&mut shell, &mut view, Vec::new());
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(view.library_renaming.is_none());
        let folder = text_pos(&output, "Shots").expect("the named folder");
        assert!(shell.unsaved());
        frame(
            &mut shell,
            &mut view,
            click(folder, egui::PointerButton::Primary),
        );
        let output = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(view.library_folder, id);
        assert!(text_pos(&output, "pattern.png").is_none());
        let all = text_pos(&output, "All").expect("the path bar");
        frame(
            &mut shell,
            &mut view,
            click(all, egui::PointerButton::Primary),
        );
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(view.library_folder.is_empty());
        assert!(text_pos(&output, "pattern.png").is_some());
        assert!(text_pos(&output, "All").is_none());
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
        land_import(&mut shell, &mut view);
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
    fn the_shape_inspector_lays_out_its_sections_and_switches_the_stroke() {
        let temp = std::env::temp_dir().join(format!("cutlass-shape-panel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.add_shape("shape:Rectangle").unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Stickers;
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 2000.0)));
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
        let _ = frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&output);
        for label in [
            "Vector image",
            "Animation",
            "Tracking",
            "Transform",
            "Size",
            "320",
            "Fill",
            "Stroke",
            "Style",
            "Weight",
            "Corner radius",
            "Each corner",
            "Shadow",
            "Distance",
            "Angle",
        ] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        assert_eq!(shell.shape().unwrap().border_width, 0.0);
        // The Stroke box, left of its name, switches the stroke on.
        let name = text_pos(&output, "Stroke").expect("the Stroke section");
        let check = name + Vec2::new(-14.0, 8.0);
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let _ = frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(check)],
        );
        let _ = frame(&mut shell, &mut view, vec![press(check, true)]);
        let _ = frame(&mut shell, &mut view, vec![press(check, false)]);
        assert_eq!(
            shell.shape().unwrap().border_width,
            cutlass_model::SHAPE_STROKE_WIDTH
        );
        // A line has no fill or corners, and its stroke cannot go off.
        shell.add_shape("shape:Line").unwrap();
        let _ = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        assert!(!lines.iter().any(|line| line == "Fill"));
        assert!(!lines.iter().any(|line| line == "Corner radius"));
        assert!(lines.iter().any(|line| line == "Stroke"));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_filters_shelf_shows_one_category_at_a_time() {
        let temp = std::env::temp_dir().join(format!("cutlass-fl-cats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Filters;
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
        assert!(text_pos(&output, "Golden").is_some());
        let chip = text_pos(&output, "Food").expect("the Food chip") + Vec2::new(4.0, 4.0);
        let press = |pressed| egui::Event::PointerButton {
            pos: chip,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(chip)]);
        frame(&mut shell, &mut view, vec![press(true), press(false)]);
        assert_eq!(view.filter_category, "Food");
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(text_pos(&output, "Tasty").is_some());
        assert!(text_pos(&output, "Golden").is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_transitions_shelf_shows_one_category_at_a_time() {
        let temp = std::env::temp_dir().join(format!("cutlass-tr-cats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Transitions;
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
        assert!(text_pos(&output, "All").is_some());
        assert!(text_pos(&output, "Dissolve").is_some());
        let chip = text_pos(&output, "Wipe").expect("the Wipe chip") + Vec2::new(4.0, 4.0);
        let press = |pressed| egui::Event::PointerButton {
            pos: chip,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(chip)]);
        frame(&mut shell, &mut view, vec![press(true), press(false)]);
        assert_eq!(view.transition_category, "Wipe");
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(text_pos(&output, "Wipe Left").is_some());
        assert!(text_pos(&output, "Dissolve").is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_effects_shelf_switches_between_video_and_body_effects() {
        let temp = std::env::temp_dir().join(format!("cutlass-fx-groups-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Effects;
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
        assert!(text_pos(&output, "Video effects").is_some());
        let first = cutlass_editor::effect_cards().remove(0).name;
        assert!(text_pos(&output, &first).is_some(), "{first}");
        // An effect card is a preview of the look, with no length on it.
        let length = card_meta(cutlass_editor::EFFECT_LENGTH.0);
        assert!(text_pos(&output, &length).is_none(), "{length}");
        // The category row: All, then each category with an effect.
        assert!(text_pos(&output, "All").is_some());
        let motion = text_pos(&output, "Motion").expect("the Motion chip") + Vec2::new(4.0, 4.0);
        let tap = |at: Pos2| {
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ]
        };
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(motion)],
        );
        frame(&mut shell, &mut view, tap(motion));
        assert_eq!(view.effect_category, "Motion");
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(text_pos(&output, "Zoom Pulse").is_some());
        assert!(text_pos(&output, "Glitch").is_none_or(|at| at.y < motion.y + 20.0));
        assert!(text_pos(&output, "Film Grain").is_none());
        // The wheel moves the row sideways.
        let before = text_pos(&output, "Motion").unwrap();
        for _ in 0..4 {
            frame(
                &mut shell,
                &mut view,
                vec![
                    egui::Event::PointerMoved(before),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: Vec2::new(0.0, -60.0),
                        modifiers: egui::Modifiers::default(),
                        phase: egui::TouchPhase::Move,
                        source: egui::MouseWheelSource::Wheel,
                    },
                ],
            );
        }
        let output = frame(&mut shell, &mut view, Vec::new());
        let after = text_pos(&output, "Motion").unwrap();
        assert!(after.x < before.x, "{before:?} then {after:?}");
        assert_eq!(after.y, before.y);
        let output = frame(&mut shell, &mut view, Vec::new());
        let body = text_pos(&output, "Body effects").unwrap() + Vec2::new(4.0, 4.0);
        let press = |pressed| egui::Event::PointerButton {
            pos: body,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(body)]);
        frame(&mut shell, &mut view, vec![press(true), press(false)]);
        assert_eq!(view.effect_group, BODY_EFFECTS);
        let output = frame(&mut shell, &mut view, Vec::new());
        let lines = galleys(&output);
        assert!(
            lines
                .iter()
                .any(|line| line == "Effects that follow a body will show here.")
        );
        assert!(lines.iter().any(|line| line == "Later"));
        assert!(text_pos(&output, &first).is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_shapes_on_the_sticker_shelf_add_a_shape_on_a_sticker_row() {
        let temp = std::env::temp_dir().join(format!("cutlass-shapes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Stickers;
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
        let click = |shell: &mut Shell, view: &mut ViewState, at: Pos2| {
            let press = |pressed| egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            frame(shell, view, vec![egui::Event::PointerMoved(at)]);
            frame(shell, view, vec![press(true), press(false)]);
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        for label in ["Yours", "Stickers", "Shapes", "GIPHY"] {
            assert!(text_pos(&output, label).is_some(), "{label}");
        }
        // GIPHY waits on its client.
        let giphy = text_pos(&output, "GIPHY").unwrap() + Vec2::new(4.0, 4.0);
        click(&mut shell, &mut view, giphy);
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        assert!(
            lines
                .iter()
                .any(|line| line == "GIPHY search will show here.")
        );
        assert!(lines.iter().any(|line| line == "Later"));
        let output = frame(&mut shell, &mut view, Vec::new());
        let shapes = text_pos(&output, "Shapes").unwrap() + Vec2::new(4.0, 4.0);
        click(&mut shell, &mut view, shapes);
        let output = frame(&mut shell, &mut view, Vec::new());
        for kind in cutlass_model::ShapeKind::ALL {
            assert!(text_pos(&output, kind.name()).is_some(), "{}", kind.name());
        }
        // The circle card dragged onto the lanes.
        let card = text_pos(&output, "Circle").expect("the circle card") - Vec2::new(0.0, 30.0);
        let lane = text_pos(&output, "No clips").expect("lane");
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(card), press(card, true)],
        );
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(lane)]);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(lane), press(lane, false)],
        );
        let draft = shell.draft().unwrap();
        let row = draft
            .tracks
            .iter()
            .find(|track| track.kind == cutlass_model::TrackType::Sticker)
            .expect("a sticker row");
        assert_eq!(row.segments.len(), 1);
        assert_eq!(
            draft.materials.shapes[0].shape_kind(),
            Some(cutlass_model::ShapeKind::Circle)
        );
        assert_eq!(shell.panel(), Some(cutlass_editor::Panel::Shape));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_pen_draws_a_shape_on_the_preview() {
        let temp = std::env::temp_dir().join(format!("cutlass-pen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Stickers;
        view.sticker_group = SHAPES;
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 1600.0)));
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
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let click = |shell: &mut Shell, view: &mut ViewState, at: Pos2| {
            frame(shell, view, vec![egui::Event::PointerMoved(at)]);
            frame(shell, view, vec![press(at, true), press(at, false)]);
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        let card = text_pos(&output, "Pen").expect("the pen card") - Vec2::new(0.0, 30.0);
        click(&mut shell, &mut view, card);
        assert!(shell.pen().is_some(), "the card picks up the pen");
        // A click on the empty shelf puts it down, as it unpicks a card.
        let empty = card + Vec2::new(200.0, 0.0);
        click(&mut shell, &mut view, empty);
        assert!(shell.pen().is_none(), "an empty click puts the pen down");
        let rectangle = text_pos(&frame(&mut shell, &mut view, Vec::new()), "Rectangle").unwrap()
            - Vec2::new(0.0, 30.0);
        click(&mut shell, &mut view, rectangle);
        click(&mut shell, &mut view, card);
        assert!(shell.pen().is_some());
        assert!(view.library_picked.is_empty(), "only the pen is lit");
        let lines = galleys(&frame(&mut shell, &mut view, Vec::new()));
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Click to put a point down"))
        );

        let canvas = view.preview_canvas.expect("the preview is drawn");
        let at = |x: f32, y: f32| canvas.min + canvas.size() * Vec2::new(x, y);
        click(&mut shell, &mut view, at(0.25, 0.25));
        // A drag bends the curve through the second point.
        let second = at(0.5, 0.25);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(second)],
        );
        frame(&mut shell, &mut view, vec![press(second, true)]);
        let pulled = second + Vec2::new(20.0, 0.0);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(pulled)],
        );
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(pulled), press(pulled, false)],
        );
        let bent = shell.pen().unwrap().points()[1];
        assert!(bent.handle_out[0] > 0.0, "{bent:?}");
        click(&mut shell, &mut view, at(0.5, 0.5));
        assert!(shell.draft().unwrap().materials.shapes.is_empty());
        // Back on the first point closes the path and places the shape.
        click(&mut shell, &mut view, at(0.25, 0.25) + Vec2::new(2.0, 1.0));
        let draft = shell.draft().unwrap();
        let shape = &draft.materials.shapes[0];
        assert_eq!(shape.points().len(), 3);
        assert_eq!(shape.shape_type, cutlass_model::SHAPE_CONTOUR);
        assert_eq!(shape.name, cutlass_model::SHAPE_PEN);
        let row = draft
            .tracks
            .iter()
            .find(|track| track.kind == cutlass_model::TrackType::Sticker)
            .expect("a sticker row");
        // The clip sits left of and above the middle, where it was drawn.
        let clip = &row.segments[0].clip;
        assert!(clip.transform.x < 0.0 && clip.transform.y > 0.0, "{clip:?}");
        assert_eq!(shell.panel(), Some(cutlass_editor::Panel::Shape));
        assert!(shell.pen().is_some_and(|pen| pen.points().is_empty()));

        // Two points and Enter make an open line; Escape then puts the pen down.
        click(&mut shell, &mut view, at(0.6, 0.6));
        click(&mut shell, &mut view, at(0.8, 0.7));
        let key = |key| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        frame(&mut shell, &mut view, vec![key(egui::Key::Enter)]);
        let shapes = &shell.draft().unwrap().materials.shapes;
        assert_eq!(shapes.len(), 2);
        assert!(shapes[1].is_open());
        frame(&mut shell, &mut view, vec![key(egui::Key::Escape)]);
        assert!(shell.pen().is_none());
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
        land_import(&mut shell, &mut view);
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
        // Play sits in the middle of the preview, between the side panels.
        let play = Pos2::new(
            (tokens::LIBRARY_WIDTH + 1280.0 - tokens::INSPECTOR_WIDTH) / 2.0,
            348.0,
        );
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
    fn every_s_chord_is_named_with_its_modifiers() {
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
        assert_eq!(
            chord(egui::Modifiers::COMMAND).as_deref(),
            Some("primary+s")
        );
        assert_eq!(chord(egui::Modifiers::SHIFT).as_deref(), Some("shift+s"));
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
        assert_eq!(
            chord(egui::Key::Q, egui::Modifiers::COMMAND).as_deref(),
            Some("primary+q")
        );
        assert_eq!(
            chord(egui::Key::Space, control).as_deref(),
            Some("ctrl+space")
        );
        assert_eq!(
            chord(egui::Key::Comma, egui::Modifiers::COMMAND).as_deref(),
            Some("primary+comma")
        );
        assert_eq!(
            chord(egui::Key::F5, egui::Modifiers::default()).as_deref(),
            Some("f5")
        );
        assert_eq!(
            chord(egui::Key::Num3, egui::Modifiers::default()).as_deref(),
            Some("3")
        );
        assert_eq!(chord(egui::Key::Escape, egui::Modifiers::default()), None);
        for (key, name) in [
            (egui::Key::J, "j"),
            (egui::Key::K, "k"),
            (egui::Key::L, "l"),
            (egui::Key::I, "i"),
            (egui::Key::O, "o"),
        ] {
            assert_eq!(
                chord(key, egui::Modifiers::default()).as_deref(),
                Some(name)
            );
            assert_eq!(
                chord(key, egui::Modifiers::COMMAND),
                Some(format!("primary+{name}"))
            );
        }
        assert_eq!(
            chord(egui::Key::X, egui::Modifiers::ALT).as_deref(),
            Some("alt+x")
        );
        assert_eq!(
            chord(egui::Key::Delete, egui::Modifiers::SHIFT).as_deref(),
            Some("shift+delete")
        );
        assert_eq!(
            chord(egui::Key::Delete, egui::Modifiers::COMMAND).as_deref(),
            Some("primary+delete")
        );
        assert_eq!(
            chord(
                egui::Key::X,
                egui::Modifiers::ALT | egui::Modifiers::COMMAND
            )
            .as_deref(),
            Some("primary+alt+x")
        );
        assert_eq!(
            chord(egui::Key::X, egui::Modifiers::COMMAND).as_deref(),
            Some("primary+x")
        );
    }

    #[test]
    fn dragging_a_slider_previews_each_value_and_undoes_in_one_step() {
        let temp = std::env::temp_dir().join(format!("cutlass-slide-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft
            .materials
            .videos
            .push(cutlass_model::MaterialVideo::new("movie"));
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "main",
            "movie",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut frame = |shell: &mut Shell, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
            output
        };
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        let label = text_pos(&output, "Scale").expect("the scale row");
        // The track starts just past the row's name.
        let start = label + Vec2::new(110.0, 6.0);
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(
            &mut shell,
            vec![egui::Event::PointerMoved(start), press(start, true)],
        );
        let mut seen = Vec::new();
        for step in 1..=3 {
            let at = start + Vec2::new(30.0 * step as f32, 0.0);
            frame(&mut shell, vec![egui::Event::PointerMoved(at)]);
            seen.push(shell.inspection().unwrap().scale_x);
        }
        let end = start + Vec2::new(90.0, 0.0);
        frame(&mut shell, vec![press(end, false)]);
        frame(&mut shell, Vec::new());
        assert!(
            seen.windows(2).all(|pair| pair[1] > pair[0]),
            "the preview did not follow the drag: {seen:?}"
        );
        assert!(shell.inspection().unwrap().scale_x > 1.0);
        shell.undo().unwrap();
        assert_eq!(shell.inspection().unwrap().scale_x, 1.0);
        assert!(!shell.can_undo());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn dragging_a_number_box_unit_scrubs_the_value_and_undoes_in_one_step() {
        let temp = std::env::temp_dir().join(format!("cutlass-unit-scrub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft
            .materials
            .videos
            .push(cutlass_model::MaterialVideo::new("movie"));
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "main",
            "movie",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut frame = |shell: &mut Shell, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
            output
        };
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        let before = shell.inspection().unwrap().transform_x;
        // The unit's text is painted from its left middle.
        let unit = text_pos(&output, "X").expect("the X unit") + Vec2::new(3.0, 7.0);
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(
            &mut shell,
            vec![egui::Event::PointerMoved(unit), press(unit, true)],
        );
        let mut seen = Vec::new();
        for step in 1..=3 {
            let at = unit + Vec2::new(20.0 * step as f32, 0.0);
            frame(&mut shell, vec![egui::Event::PointerMoved(at)]);
            seen.push(shell.inspection().unwrap().transform_x);
        }
        frame(&mut shell, vec![press(unit + Vec2::new(60.0, 0.0), false)]);
        frame(&mut shell, Vec::new());
        assert!(
            seen.windows(2).all(|pair| pair[1] > pair[0]) && seen[0] > before,
            "the value did not follow the drag: {before} {seen:?}"
        );
        shell.undo().unwrap();
        assert_eq!(shell.inspection().unwrap().transform_x, before);
        assert!(!shell.can_undo());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_animation_tab_stores_a_preset_on_the_picture() {
        let temp = std::env::temp_dir().join(format!("cutlass-animate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut movie = cutlass_model::MaterialVideo::new("movie");
        movie.width = 640;
        movie.height = 360;
        draft.materials.videos.push(movie);
        let mut main = cutlass_model::Segment::new(
            "main",
            "movie",
            cutlass_model::TimeRange::new(0, 2_000_000),
        );
        main.source_range = Some(cutlass_model::TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(main);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut frame = |shell: &mut Shell, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
            output
        };
        let click = |pos: Pos2| {
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), press(true), press(false)]
        };
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        let tab = text_pos(&output, "Animation").expect("the Animation tab");
        frame(&mut shell, click(tab + Vec2::new(4.0, 4.0)));
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        let tile = text_pos(&output, "Slide Up").expect("the Slide Up tile");
        frame(&mut shell, click(tile + Vec2::new(4.0, 4.0)));
        frame(&mut shell, Vec::new());
        let enter = shell.animations().unwrap().enter.expect("an in animation");
        assert_eq!(enter.resource_id, "cutlass-anim-in-slide-up");
        let output = frame(&mut shell, Vec::new());
        assert!(text_pos(&output, "0.5s").is_some(), "the length row");
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// The centre of the first keyframe diamond painted in `output`.
    fn diamond_pos(output: &egui::FullOutput) -> Option<Pos2> {
        diamonds(output, 12.0).first().copied()
    }

    /// The centres of the diamonds `width` wide painted in `output`.
    fn diamonds(output: &egui::FullOutput, width: f32) -> Vec<Pos2> {
        fn find(shape: &egui::Shape, width: f32, found: &mut Vec<Pos2>) {
            match shape {
                egui::Shape::Path(path) if path.points.len() == 4 && path.closed => {
                    let [top, right, bottom, left] = [0, 1, 2, 3].map(|i| path.points[i]);
                    let diamond = (top.x - bottom.x).abs() < 0.01
                        && (left.y - right.y).abs() < 0.01
                        && (right.x - left.x - width).abs() < 0.01;
                    if diamond {
                        found.push(Pos2::new(top.x, left.y));
                    }
                }
                egui::Shape::Vec(shapes) => {
                    shapes.iter().for_each(|shape| find(shape, width, found));
                }
                _ => {}
            }
        }
        let mut found = Vec::new();
        for clipped in &output.shapes {
            find(&clipped.shape, width, &mut found);
        }
        found
    }

    #[test]
    fn the_diamond_keys_the_picture_and_the_graphs_show_between_keys() {
        let temp = std::env::temp_dir().join(format!("cutlass-keys-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut movie = cutlass_model::MaterialVideo::new("movie");
        movie.width = 640;
        movie.height = 360;
        draft.materials.videos.push(movie);
        let mut main = cutlass_model::Segment::new(
            "main",
            "movie",
            cutlass_model::TimeRange::new(0, 2_000_000),
        );
        main.source_range = Some(cutlass_model::TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(main);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut frame = |shell: &mut Shell, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
            output
        };
        let click = |pos: Pos2| {
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), press(true), press(false)]
        };
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        assert!(text_pos(&output, "Graphs").is_none());
        let diamond = diamond_pos(&output).expect("a keyframe diamond");
        frame(&mut shell, click(diamond));
        assert_eq!(shell.keys().unwrap().times, vec![Micros(0)]);
        shell.seek(Micros(2_000_000));
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        frame(
            &mut shell,
            click(diamond_pos(&output).expect("the diamond")),
        );
        assert_eq!(shell.keys().unwrap().times.len(), 2);
        shell.seek(Micros(1_000_000));
        frame(&mut shell, Vec::new());
        let output = frame(&mut shell, Vec::new());
        assert!(text_pos(&output, "Graphs").is_some(), "the Graphs row");
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_zoom_buttons_step_the_canvas_and_fit_brings_it_back() {
        let (mut shell, temp) = open_span("zoom", 2_000_000);
        // A canvas bigger than the preview, so it fits below full size.
        shell.cycle_canvas().unwrap();
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
            output
        };
        let click = |pos: Pos2| {
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), press(true), press(false)]
        };
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let fit = text_pos(&output, "Fit").expect("the zoom label");
        let fitted = view.preview_scale;
        // The label's box is 48 wide; + is the next box, 24 wide.
        let plus = fit + Vec2::new(46.0, 6.0);
        frame(&mut shell, &mut view, click(plus));
        let zoom = view.preview_zoom.expect("a zoom step");
        assert!(zoom > fitted, "{zoom} after {fitted}");
        let output = frame(&mut shell, &mut view, Vec::new());
        let label = format!("{:.0}%", zoom * 100.0);
        let shown = text_pos(&output, &label).expect("the zoom shown");
        frame(&mut shell, &mut view, click(shown + Vec2::new(4.0, 4.0)));
        assert_eq!(view.preview_zoom, None);
        assert_eq!(view.preview_pan, Vec2::ZERO);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_pinch_and_a_command_wheel_zoom_about_the_pointer_and_a_scroll_pans() {
        let (mut shell, temp) = open_span("wheel-zoom", 2_000_000);
        shell.cycle_canvas().unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut time = 0.0;
        let mut frame = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            time += 1.0 / 60.0;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events,
                    time: Some(time),
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
        };
        let wheel = |unit, delta: Vec2, source, modifiers| egui::Event::MouseWheel {
            unit,
            delta,
            phase: egui::TouchPhase::Move,
            source,
            modifiers,
        };
        // The canvas pixel under `pos` as the preview last drew it.
        let under = |view: &ViewState, pos: Pos2| {
            let canvas = view.preview_canvas.expect("a canvas");
            (pos - canvas.center()) / view.preview_scale as f32
        };
        frame(&mut shell, &mut view, Vec::new());
        frame(&mut shell, &mut view, Vec::new());
        let fitted = view.preview_scale;
        let pos = view.preview_frame.center() + Vec2::new(60.0, 40.0);
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(pos)]);
        let spot = under(&view, pos);

        // A trackpad pinch zooms in, and the spot under the pointer stays.
        frame(&mut shell, &mut view, vec![egui::Event::Zoom(3.0)]);
        frame(&mut shell, &mut view, Vec::new());
        let pinched = view.preview_zoom.expect("a pinch zooms");
        assert!(
            (pinched - fitted * 3.0).abs() < 1e-6,
            "{pinched} after {fitted}"
        );
        let moved = under(&view, pos) - spot;
        assert!(moved.length() < 0.5, "the spot moved {moved:?}");

        // Two fingers without a modifier pan the zoomed canvas, not zoom it.
        let pan = view.preview_pan;
        let touch = wheel(
            egui::MouseWheelUnit::Point,
            Vec2::new(-30.0, 20.0),
            egui::MouseWheelSource::Trackpad,
            egui::Modifiers::default(),
        );
        frame(&mut shell, &mut view, vec![touch]);
        for _ in 0..30 {
            frame(&mut shell, &mut view, Vec::new());
        }
        assert_eq!(view.preview_zoom, Some(pinched));
        assert!(
            (view.preview_pan - pan - Vec2::new(-30.0, 20.0)).length() < 1.0,
            "{:?} from {pan:?}",
            view.preview_pan
        );

        // Mouse wheel lines with Cmd (Ctrl off a Mac) zoom out, smoothed
        // over a few frames.
        let spot = under(&view, pos);
        let notch = wheel(
            egui::MouseWheelUnit::Line,
            Vec2::new(0.0, -1.0),
            egui::MouseWheelSource::Wheel,
            egui::Modifiers::COMMAND,
        );
        frame(
            &mut shell,
            &mut view,
            vec![
                egui::Event::ModifiersChanged(egui::Modifiers::COMMAND),
                notch,
            ],
        );
        for _ in 0..30 {
            frame(&mut shell, &mut view, Vec::new());
        }
        let notched = view.preview_zoom.expect("still zoomed");
        assert!(
            notched < pinched && notched > fitted,
            "{notched} after {pinched}"
        );
        let moved = under(&view, pos) - spot;
        assert!(moved.length() < 0.5, "the spot moved {moved:?}");

        // A Ctrl wheel on a Mac reaches egui as a scroll; it zooms too, and
        // never pans.
        let pan = view.preview_pan;
        let ctrl = wheel(
            egui::MouseWheelUnit::Point,
            Vec2::new(0.0, 40.0),
            egui::MouseWheelSource::Trackpad,
            egui::Modifiers::CTRL,
        );
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::ModifiersChanged(egui::Modifiers::CTRL), ctrl],
        );
        for _ in 0..30 {
            frame(&mut shell, &mut view, Vec::new());
        }
        let ctrl_zoom = view.preview_zoom.expect("still zoomed");
        assert!(ctrl_zoom > notched, "{ctrl_zoom} after {notched}");
        let moved = under(&view, pos) - spot;
        assert!(moved.length() < 0.5, "the spot moved {moved:?}");
        assert_ne!(view.preview_pan, pan + Vec2::new(0.0, 40.0));
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::ModifiersChanged(egui::Modifiers::default())],
        );

        // A pinch out stops at the smallest zoom.
        frame(&mut shell, &mut view, vec![egui::Event::Zoom(0.0001)]);
        assert_eq!(
            view.preview_zoom,
            Some(cutlass_editor::PREVIEW_ZOOM_RANGE.0)
        );

        // Off the preview, a pinch leaves it alone.
        let off = Pos2::new(4.0, 790.0);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(off), egui::Event::Zoom(2.0)],
        );
        assert_eq!(
            view.preview_zoom,
            Some(cutlass_editor::PREVIEW_ZOOM_RANGE.0)
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// The centres of the round 16-pixel transport buttons, left to right.
    fn round_buttons(output: &egui::FullOutput) -> Vec<Pos2> {
        fn find(shape: &egui::Shape, found: &mut Vec<Pos2>) {
            match shape {
                egui::Shape::Circle(circle) if circle.radius == 16.0 => found.push(circle.center),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| find(shape, found)),
                _ => {}
            }
        }
        let mut found = Vec::new();
        for clipped in &output.shapes {
            find(&clipped.shape, &mut found);
        }
        found.sort_by(|a, b| a.x.total_cmp(&b.x));
        found
    }

    #[test]
    fn the_guides_and_full_screen_buttons_toggle_and_escape_leaves() {
        let (mut shell, temp) = open_span("full", 2_000_000);
        shell.cycle_canvas().unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        // Wide enough that the transport shows all four round buttons.
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 800.0)));
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
        let click = |pos: Pos2| {
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), press(true), press(false)]
        };
        let full_screen = |output: &egui::FullOutput| {
            output.viewport_output.values().find_map(|viewport| {
                viewport.commands.iter().find_map(|command| match command {
                    egui::ViewportCommand::Fullscreen(on) => Some(*on),
                    _ => None,
                })
            })
        };
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let buttons = round_buttons(&output);
        // Quality, full screen, guides, background, still.
        assert_eq!(buttons.len(), 5, "{buttons:?}");
        let (full, guides) = (buttons[1], buttons[2]);
        let editor_width = view.preview_canvas.expect("the canvas").width();
        frame(&mut shell, &mut view, click(guides));
        assert!(view.preview_guides);
        let output = frame(&mut shell, &mut view, click(full));
        assert!(view.preview_full);
        assert_eq!(full_screen(&output), Some(true));
        // Full screen shows the preview alone, so the canvas grows.
        frame(&mut shell, &mut view, Vec::new());
        let grown = view.preview_canvas.expect("the canvas").width();
        assert!(grown > editor_width, "{grown} after {editor_width}");
        let escape = egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        let output = frame(&mut shell, &mut view, vec![escape]);
        assert!(!view.preview_full);
        assert_eq!(full_screen(&output), Some(false));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_zoomed_canvas_pans_only_as_far_as_it_overhangs() {
        let frame = Rect::from_min_size(Pos2::ZERO, Vec2::new(200.0, 100.0));
        let mut pan = Vec2::new(500.0, -500.0);
        let (rect, scale) = placed_canvas(frame, Vec2::new(100.0, 50.0), Some(4.0), &mut pan);
        assert_eq!(scale, 4.0);
        // 400 by 200 shown in 200 by 100 overhangs 100 and 50 each side.
        assert_eq!(pan, Vec2::new(100.0, -50.0));
        assert_eq!(rect.left(), 0.0);
        assert_eq!(rect.bottom(), 100.0);
        let mut pan = Vec2::new(30.0, 30.0);
        let (rect, scale) = placed_canvas(frame, Vec2::new(100.0, 50.0), None, &mut pan);
        assert_eq!(
            (rect, scale),
            (Rect::from_min_size(Pos2::ZERO, frame.size()), 2.0)
        );
    }

    #[test]
    fn the_window_draws_no_menus_while_the_system_menu_bar_is_up() {
        let (mut shell, _temp) = open_span("menus", 1_000_000);
        let ctx = egui::Context::default();
        let mut painted = |view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, view, None),
            );
            output.textures_delta.clear();
            painted_text(&output)
        };
        let mut view = ViewState::new();
        let text = painted(&mut view);
        for title in ["File", "Edit", "Clip", "Playback", "Help"] {
            assert!(
                text.lines().any(|line| line == title),
                "no {title} menu: {text}"
            );
        }
        view.system_menus = true;
        let text = painted(&mut view);
        for title in ["File", "Edit", "Clip", "Playback", "Help"] {
            assert!(
                !text.lines().any(|line| line == title),
                "a {title} menu: {text}"
            );
        }
        assert!(text.lines().any(|line| line == "Export"), "{text}");
    }

    #[test]
    fn a_label_whose_tick_just_left_the_view_still_shows_its_tail() {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 100.0))),
                ..Default::default()
            },
            |ui| {
                let ruler = Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, tokens::RULER_HEIGHT));
                // 300 px a second; the 00:01 tick sits 30 px left of the lanes.
                paint_ticks(ui, ruler, 100.0, 3e-4, 1_100_000, FrameRate::FPS_30, false);
            },
        );
        output.textures_delta.clear();
        let text = painted_text(&output);
        assert!(text.contains("00:01"), "{text}");
        assert!(text.contains("00:02"), "{text}");
    }

    /// The x of the white playhead line painted nearest `guess`.
    fn shell_playhead_x(output: &egui::FullOutput, guess: f32) -> f32 {
        fn lines(shape: &egui::Shape, found: &mut Vec<f32>) {
            match shape {
                egui::Shape::LineSegment { points, stroke }
                    if stroke.color == tokens::PLAYHEAD && points[0].x == points[1].x =>
                {
                    found.push(points[0].x);
                }
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| lines(shape, found)),
                _ => {}
            }
        }
        let mut found = Vec::new();
        for clipped in &output.shapes {
            lines(&clipped.shape, &mut found);
        }
        found
            .into_iter()
            .min_by(|a, b| (a - guess).abs().total_cmp(&(b - guess).abs()))
            .expect("a playhead line")
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
        let on_frame = |time: Micros| {
            let rate = FrameRate::FPS_30;
            rate.frame_start(rate.frame_index(time).unwrap()) == Some(time)
        };
        assert!(
            on_frame(shell.playhead()),
            "{:?} is between frames",
            shell.playhead()
        );
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events: vec![press(end, false)],
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        output.textures_delta.clear();
        // The playhead's line, down in the lanes, drags it on.
        let before = shell.playhead();
        let line = Pos2::new(shell_playhead_x(&output, end.x) + 2.0, end.y + 160.0);
        for events in [
            vec![egui::Event::PointerMoved(line), press(line, true)],
            vec![egui::Event::PointerMoved(line + Vec2::new(30.0, 0.0))],
            vec![egui::Event::PointerMoved(line + Vec2::new(90.0, 0.0))],
            vec![press(line + Vec2::new(90.0, 0.0), false)],
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
        assert!(
            shell.playhead() > before,
            "the line drag did not move the playhead from {before:?}"
        );
        assert!(
            on_frame(shell.playhead()),
            "{:?} is between frames",
            shell.playhead()
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_drag_on_the_preview_draws_the_tracking_box_in_canvas_pixels() {
        let temp = std::env::temp_dir().join(format!("cutlass-track-box-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        draft.duration = Micros(2_000_000);
        draft.materials.videos.push(cutlass_model::MaterialVideo {
            id: "m".into(),
            width: 1920,
            height: 1080,
            path: "missing.mp4".into(),
            ..cutlass_model::MaterialVideo::default()
        });
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "v",
            "m",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some("v".into()));
        shell.seek(Micros(500_000));
        shell.start_tracking().unwrap();
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
        let placed = view.preview_canvas.expect("the canvas");
        let size = Vec2::new(1920.0, 1080.0);
        let from = screen_point(placed, size, (200.0, 100.0));
        let to = screen_point(placed, size, (600.0, 500.0));
        drag(&ctx, from, to, &mut shell, &mut view);
        let ((x0, y0), (x1, y1)) = shell.tracking().unwrap().canvas_box.expect("a box");
        let px = 1920.0 / f64::from(placed.width());
        for (got, want) in [(x0, 200.0), (y0, 100.0), (x1, 600.0), (y1, 500.0)] {
            assert!((got - want).abs() <= px, "{got} vs {want}");
        }
        assert_eq!(shell.tracking().unwrap().time, Micros(500_000));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_sticker_follows_a_box_drawn_from_its_tracking_tab_and_stops() {
        let temp = std::env::temp_dir().join(format!("cutlass-follow-tab-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        draft.duration = Micros(2_000_000);
        draft.materials.videos.push(cutlass_model::MaterialVideo {
            id: "m".into(),
            width: 1920,
            height: 1080,
            path: "missing.mp4".into(),
            ..cutlass_model::MaterialVideo::default()
        });
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "v",
            "m",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        let mut stickers = cutlass_model::Track::main("stickers");
        stickers.kind = cutlass_model::TrackType::Sticker;
        stickers.segments.push(cutlass_model::Segment::new(
            "s",
            "sticker-art",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        draft.tracks.push(stickers);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some("s".into()));
        shell.seek(Micros(500_000));
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let draw = |shell: &mut Shell, view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let _ = draw(&mut shell, &mut view);
        let output = draw(&mut shell, &mut view);
        let tab = text_pos(&output, "Tracking").expect("the sticker's Tracking tab");
        click(&ctx, tab + Vec2::new(4.0, 4.0), &mut shell, &mut view);
        let output = draw(&mut shell, &mut view);
        let lines = galleys(&output);
        for label in ["Face", "Body", "Hand", "Custom"] {
            assert!(lines.iter().any(|line| line == label), "{label}: {lines:?}");
        }
        let custom = text_pos(&output, "Custom").expect("the Custom tile");
        click(&ctx, custom + Vec2::new(4.0, 4.0), &mut shell, &mut view);
        let tracking = shell.tracking().expect("a track is being set up").clone();
        assert_eq!(
            tracking.segment_id,
            SegmentId::new("v"),
            "the video under it"
        );
        assert_eq!(tracking.followers, vec![SegmentId::new("s")]);
        let output = draw(&mut shell, &mut view);
        assert!(
            galleys(&output)
                .iter()
                .any(|line| line.starts_with("Drag a box") || line.starts_with("Click the subject")),
            "{:?}",
            galleys(&output)
        );
        let placed = view.preview_canvas.expect("the canvas");
        let size = Vec2::new(1920.0, 1080.0);
        let from = screen_point(placed, size, (200.0, 100.0));
        let to = screen_point(placed, size, (600.0, 500.0));
        drag(&ctx, from, to, &mut shell, &mut view);
        assert!(shell.tracking().unwrap().canvas_box.is_some());
        assert!(shell.track_ready());
        let output = draw(&mut shell, &mut view);
        assert!(text_pos(&output, "Start tracking").is_some());
        let cancel = text_pos(&output, "Cancel").expect("Cancel");
        click(&ctx, cancel + Vec2::new(4.0, 4.0), &mut shell, &mut view);
        assert!(shell.tracking().is_none());

        // A track that has landed: the tab names what it follows, and Stop
        // following takes it off.
        let path = cutlass_model::TrackPath {
            mode: cutlass_model::TrackMode::Object,
            samples: (0..10)
                .map(|i| cutlass_model::TrackSample {
                    time: Micros(500_000 + i * 33_333),
                    corners: [(-0.1, 0.1), (0.1, 0.1), (0.1, -0.1), (-0.1, -0.1)],
                })
                .collect(),
        };
        shell
            .apply(cutlass_session::Request::Track {
                target: "v".into(),
                followers: vec!["s".into()],
                tracking_id: "t".into(),
                anchor: Micros(500_000),
                path,
                masks: Vec::new(),
            })
            .unwrap();
        let output = draw(&mut shell, &mut view);
        let lines = galleys(&output);
        assert!(
            lines.iter().any(|line| line.starts_with("Follows ")),
            "{lines:?}"
        );
        assert!(text_pos(&output, "Fix frame").is_some());
        let stop = text_pos(&output, "Stop following").expect("Stop following");
        click(&ctx, stop + Vec2::new(4.0, 4.0), &mut shell, &mut view);
        assert!(shell.following().is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// A 1920×1080 project with a full-canvas picture under a half-size
    /// overlay, opened in a shell and drawn once in a 1280×800 window.
    fn stage_window(name: &str) -> (std::path::PathBuf, Shell, ViewState, egui::Context) {
        let temp = std::env::temp_dir().join(format!("cutlass-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        draft.duration = Micros(1_000_000);
        for id in ["under", "over"] {
            draft.materials.videos.push(cutlass_model::MaterialVideo {
                id: id.into(),
                width: 1920,
                height: 1080,
                path: "missing.mp4".into(),
                ..cutlass_model::MaterialVideo::default()
            });
        }
        let range = cutlass_model::TimeRange::new(0, 1_000_000);
        let mut under = cutlass_model::Segment::new("under", "under", range.clone());
        under.source_range = Some(range.clone());
        draft.tracks[0].segments.push(under);
        let mut overlay = cutlass_model::Track::main("overlay");
        overlay.flag = cutlass_model::TrackFlag::SubVideo;
        let mut over = cutlass_model::Segment::new("over", "over", range.clone());
        over.source_range = Some(range);
        over.clip.scale = cutlass_model::Point::new(0.5, 0.5);
        over.uniform_scale.value = 0.5;
        overlay.segments.push(over);
        draft.tracks.push(overlay);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        // The first frame places the canvas; the preview's clips take
        // pointer input from the next one on.
        frame_once(&ctx, Vec::new(), &mut shell, &mut view);
        frame_once(&ctx, Vec::new(), &mut shell, &mut view);
        (temp, shell, view, ctx)
    }

    fn frame_once(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        shell: &mut Shell,
        view: &mut ViewState,
    ) {
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                events,
                ..Default::default()
            },
            |ui| show(shell, ui, view, None),
        );
        output.textures_delta.clear();
    }

    fn click(ctx: &egui::Context, at: Pos2, shell: &mut Shell, view: &mut ViewState) {
        let press = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame_once(
            ctx,
            vec![egui::Event::PointerMoved(at), press(true)],
            shell,
            view,
        );
        frame_once(ctx, vec![press(false)], shell, view);
        frame_once(ctx, Vec::new(), shell, view);
    }

    fn stage_pose(shell: &Shell, id: &str) -> (f64, f64, f64, f64) {
        let segment = shell
            .draft()
            .unwrap()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .unwrap();
        (
            segment.clip.transform.x,
            segment.clip.transform.y,
            segment.uniform_scale.value,
            segment.clip.rotation,
        )
    }

    #[test]
    fn a_click_on_the_preview_selects_the_clip_under_it_and_empty_space_clears_it() {
        let (temp, mut shell, mut view, ctx) = stage_window("stage-click");
        let placed = view.preview_canvas.expect("the canvas");
        let size = Vec2::new(1920.0, 1080.0);
        click(
            &ctx,
            screen_point(placed, size, (960.0, 540.0)),
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.selected().map(|id| id.as_str()), Some("over"));
        click(
            &ctx,
            screen_point(placed, size, (100.0, 100.0)),
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.selected().map(|id| id.as_str()), Some("under"));
        // With the full-canvas clip selected, a click on the overlay still
        // reaches the overlay on top of it.
        click(
            &ctx,
            screen_point(placed, size, (900.0, 500.0)),
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.selected().map(|id| id.as_str()), Some("over"));
        // Beside the canvas, in the dark around it, nothing is selected.
        click(
            &ctx,
            placed.left_center() - Vec2::new(4.0, 0.0),
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.selected(), None);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_preview_drag_moves_the_clip_under_the_pointer_in_one_undo_step() {
        let (temp, mut shell, mut view, ctx) = stage_window("stage-move");
        let placed = view.preview_canvas.expect("the canvas");
        let size = Vec2::new(1920.0, 1080.0);
        let from = screen_point(placed, size, (960.0, 540.0));
        let to = screen_point(placed, size, (1200.0, 300.0));
        drag(&ctx, from, to, &mut shell, &mut view);
        assert_eq!(shell.selected().map(|id| id.as_str()), Some("over"));
        let (x, y, scale, rotation) = stage_pose(&shell, "over");
        let px = 1920.0 / f64::from(placed.width());
        assert!((x * 960.0 - 240.0).abs() <= px, "{x}");
        assert!((y * 540.0 - 240.0).abs() <= px, "{y}");
        assert_eq!((scale, rotation), (0.5, 0.0));
        assert_eq!(stage_pose(&shell, "under"), (0.0, 0.0, 1.0, 0.0));
        assert!(view.stage_drag.is_none());
        shell.undo().unwrap();
        assert_eq!(stage_pose(&shell, "over"), (0.0, 0.0, 0.5, 0.0));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn dragging_a_corner_scales_and_the_knob_turns_the_selected_clip() {
        let (temp, mut shell, mut view, ctx) = stage_window("stage-grips");
        let placed = view.preview_canvas.expect("the canvas");
        let size = Vec2::new(1920.0, 1080.0);
        let over = SegmentId::new("over");
        shell.select(Some(over.clone()));
        frame_once(&ctx, Vec::new(), &mut shell, &mut view);
        // The overlay's bottom right corner is at (1440, 810); dragging it
        // out to (1680, 945) is half as far again from the centre.
        let corner = screen_point(placed, size, (1440.0, 810.0));
        let out = screen_point(placed, size, (1680.0, 945.0));
        drag(&ctx, corner, out, &mut shell, &mut view);
        let (x, y, scale, rotation) = stage_pose(&shell, "over");
        assert_eq!((x, y, rotation), (0.0, 0.0, 0.0));
        assert!((scale - 0.75).abs() < 0.01, "{scale}");
        // The knob sits below the middle of the bottom edge. Dragging it
        // round to the left of the centre is a quarter turn clockwise.
        let stage = shell.stage_box(&over).unwrap();
        let (_, _, knob) = stage_handles(&stage, placed, size, view.preview_frame);
        let center = screen_point(placed, size, (stage.center_x, stage.center_y));
        let left = center - Vec2::new(knob.y - center.y, 0.0);
        drag(&ctx, knob, left, &mut shell, &mut view);
        let (_, _, scale_after, rotation) = stage_pose(&shell, "over");
        assert_eq!(rotation, 270.0, "a quarter turn clockwise");
        assert_eq!(scale_after, scale);
        // Each drag is its own undo step.
        shell.undo().unwrap();
        assert_eq!(stage_pose(&shell, "over").3, 0.0);
        shell.undo().unwrap();
        assert_eq!(stage_pose(&shell, "over"), (0.0, 0.0, 0.5, 0.0));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_preview_box_waits_for_the_picture_drawn_from_its_draft() {
        let (temp, mut shell, mut view, _ctx) = stage_window("stage-wait");
        let over = SegmentId::new("over");
        shell.select(Some(over.clone()));
        shell.preview_bytes().unwrap();
        let before = shell.revision();
        assert_eq!(shell.shown_revision(), before);
        assert_eq!(shown_stage_boxes(&shell, &mut view)[0].center_x, 960.0);
        shell.stage_begin(&over).unwrap();
        shell.stage_move(240.0, 0.0, 1.0, 0.0).unwrap();
        assert_ne!(shell.revision(), before);
        // The canvas still shows the old picture, so the box stays on it.
        assert_eq!(shell.shown_revision(), before);
        assert_eq!(shown_stage_boxes(&shell, &mut view)[0].center_x, 960.0);
        // Once the moved picture is drawn, the box moves with it.
        shell.preview_bytes().unwrap();
        assert_eq!(shell.shown_revision(), shell.revision());
        assert_eq!(shown_stage_boxes(&shell, &mut view)[0].center_x, 1200.0);
        shell.stage_end();
        // Only the last few revisions are kept.
        for step in 0..(STAGE_SEEN as u64 + 4) {
            view.stage_seen.push((10_000 + step, Vec::new()));
            shown_stage_boxes(&shell, &mut view);
        }
        assert_eq!(view.stage_seen.len(), STAGE_SEEN);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_frame_that_moves_a_clip_draws_the_moved_picture() {
        let (temp, mut shell, mut view, ctx) = stage_window("stage-same-frame");
        let placed = view.preview_canvas.expect("the canvas");
        let size = Vec2::new(1920.0, 1080.0);
        let from = screen_point(placed, size, (960.0, 540.0));
        let to = screen_point(placed, size, (1200.0, 540.0));
        let press = egui::Event::PointerButton {
            pos: from,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        };
        frame_once(
            &ctx,
            vec![egui::Event::PointerMoved(from), press],
            &mut shell,
            &mut view,
        );
        frame_once(
            &ctx,
            vec![egui::Event::PointerMoved(to)],
            &mut shell,
            &mut view,
        );
        let before = shell.revision();
        let on = to + Vec2::new(20.0, 0.0);
        frame_once(
            &ctx,
            vec![egui::Event::PointerMoved(on)],
            &mut shell,
            &mut view,
        );
        assert!(view.stage_drag.is_some());
        // Mid-drag: the clip moved this frame, and the picture drawn this
        // frame is the moved one, not the one before it.
        assert_ne!(shell.revision(), before);
        assert_eq!(shell.shown_revision(), shell.revision());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_turned_clips_handles_follow_its_turn() {
        let stage = cutlass_editor::StageBox {
            segment_id: SegmentId::new("c"),
            center_x: 960.0,
            center_y: 540.0,
            width: 400.0,
            height: 200.0,
            turn: 90.0,
        };
        let placed = Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0));
        let size = Vec2::new(1920.0, 1080.0);
        let (corners, bottom, knob) = stage_handles(&stage, placed, size, placed);
        // A quarter turn clockwise puts the bottom edge on the left.
        assert!((bottom.x - 860.0).abs() < 1e-3 && (bottom.y - 540.0).abs() < 1e-3);
        assert!((knob.x - (860.0 - STAGE_TURN_OFFSET)).abs() < 1e-3);
        assert!((corners[0].x - 1060.0).abs() < 1e-3 && (corners[0].y - 340.0).abs() < 1e-3);
        // A clip that fills the canvas keeps its knob inside the preview.
        let full = cutlass_editor::StageBox {
            width: 1920.0,
            height: 1080.0,
            turn: 0.0,
            ..stage.clone()
        };
        let (_, bottom, inside) = stage_handles(&full, placed, size, placed);
        assert_eq!(inside, bottom - Vec2::new(0.0, STAGE_TURN_OFFSET));
        assert_eq!(
            stage_grip(Some(&stage), placed, size, placed, knob),
            Some(StageGrip::Turn)
        );
        assert_eq!(
            stage_grip(Some(&stage), placed, size, placed, corners[2]),
            Some(StageGrip::Scale)
        );
        assert_eq!(
            stage_grip(Some(&stage), placed, size, placed, Pos2::new(960.0, 540.0)),
            Some(StageGrip::Move)
        );
        assert_eq!(
            stage_grip(Some(&stage), placed, size, placed, Pos2::new(5.0, 5.0)),
            None
        );
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
    fn a_clip_key_diamond_seeks_on_a_click_and_moves_on_a_drag() {
        let temp = std::env::temp_dir().join(format!("cutlass-keydrag-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        let mut clip =
            cutlass_model::Segment::new("a", "color", cutlass_model::TimeRange::new(0, 4_000_000));
        for offset in [0, 1_000_000] {
            cutlass_model::put_key(&mut clip, "KFTypePositionX", Micros(offset), 0.0);
        }
        draft.tracks[0].segments.push(clip);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let id = SegmentId::new("a");
        shell.select(Some(id.clone()));
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
            output
        };
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let keys = diamonds(&output, 10.0);
        assert_eq!(keys.len(), 2, "{keys:?}");
        let second = keys[1];
        frame(
            &mut shell,
            &mut view,
            vec![
                egui::Event::PointerMoved(second),
                press(second, true),
                press(second, false),
            ],
        );
        assert_eq!(shell.playhead(), Micros(1_000_000));
        let step = (1_000_000.0 * shell.pixel_size()) as f32;
        let to = second + Vec2::new(step, 0.0);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(second), press(second, true)],
        );
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(
                second + Vec2::new(step / 2.0, 0.0),
            )],
        );
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(to)]);
        frame(&mut shell, &mut view, vec![press(to, false)]);
        let times = shell.clip_key_times(&id);
        assert_eq!(times.len(), 2);
        assert!(
            (times[1].0 - 2_000_000).abs() <= 34_000,
            "the key landed at {times:?}"
        );
        // The key that was clicked and dragged is the picked one: it is
        // drawn larger, and Delete takes it and leaves the clip.
        assert_eq!(shell.selected_clip_key(), Some((id.clone(), times[1])));
        let output = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(diamonds(&output, 10.0).len(), 1);
        assert_eq!(diamonds(&output, 13.0).len(), 1);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::Key {
                key: egui::Key::Delete,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert_eq!(shell.clip_key_times(&id), [Micros(0)]);
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 1);
        let output = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(diamonds(&output, 10.0).len(), 1);
        assert!(diamonds(&output, 13.0).is_empty());

        // A click picks one key; a Shift-click on another picks both and
        // leaves the playhead; Delete then takes both and not the clip.
        shell.undo().unwrap();
        let output = frame(&mut shell, &mut view, Vec::new());
        let mut keys = diamonds(&output, 10.0);
        keys.extend(diamonds(&output, 13.0));
        keys.sort_by(|a, b| a.x.total_cmp(&b.x));
        assert_eq!(keys.len(), 2, "{keys:?}");
        frame(
            &mut shell,
            &mut view,
            vec![
                egui::Event::PointerMoved(keys[0]),
                press(keys[0], true),
                press(keys[0], false),
            ],
        );
        assert_eq!(shell.playhead(), Micros(0));
        let shift = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::SHIFT,
        };
        frame(
            &mut shell,
            &mut view,
            vec![
                egui::Event::ModifiersChanged(egui::Modifiers::SHIFT),
                egui::Event::PointerMoved(keys[1]),
                shift(keys[1], true),
                shift(keys[1], false),
            ],
        );
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::ModifiersChanged(egui::Modifiers::NONE)],
        );
        assert_eq!(shell.selected_clip_keys().len(), 2);
        assert_eq!(shell.playhead(), Micros(0), "the playhead stayed");
        let output = frame(&mut shell, &mut view, Vec::new());
        assert_eq!(diamonds(&output, 13.0).len(), 2);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::Key {
                key: egui::Key::Delete,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(shell.clip_key_times(&id).is_empty());
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 1);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_right_click_on_one_of_several_picked_clips_keeps_them_all() {
        let temp = std::env::temp_dir().join(format!("cutlass-multi-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        for (id, start) in [("a", 0), ("b", 3_000_000)] {
            draft.tracks[0].segments.push(cutlass_model::Segment::new(
                id,
                "color",
                cutlass_model::TimeRange::new(start, 1_000_000),
            ));
        }
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
            output
        };
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let label = text_pos(&output, "b").expect("the second clip");
        shell.select(Some(SegmentId::new("a")));
        shell.toggle_selected(SegmentId::new("b"));
        let on_b = label + Vec2::new(4.0, 12.0);
        let press = |pressed| egui::Event::PointerButton {
            pos: on_b,
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        // Pressed in one frame and let go in the next, as a hand does.
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(on_b), press(true)],
        );
        frame(&mut shell, &mut view, vec![press(false)]);
        frame(&mut shell, &mut view, Vec::new());
        assert_eq!(shell.selection().len(), 2, "both clips stay picked");
        assert!(shell.clip_menu().is_some(), "the menu is open");
        let output = frame(&mut shell, &mut view, Vec::new());
        // The menu for several clips: it has no row for one clip alone.
        assert!(text_pos(&output, "Deactivate clip").is_some());
        assert!(text_pos(&output, "Group").is_some());
        assert!(text_pos(&output, "Split scenes").is_none());
        assert!(text_pos(&output, "Trim clip").is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn edit_effects_picks_the_clips_effect_or_lists_them_by_name() {
        let temp = std::env::temp_dir().join(format!("cutlass-fxmenu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "a",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
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
            output
        };
        let click = |pos: Pos2| {
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), press(true), press(false)]
        };
        let clip = SegmentId::new("a");
        let cards = cutlass_editor::effect_cards();
        let at = Pos2::new(300.0, 80.0);
        let open = |shell: &mut Shell| {
            shell.select(Some(SegmentId::new("a")));
            shell.open_clip_menu(at.x, at.y);
        };
        frame(&mut shell, &mut view, Vec::new());

        // No effect on the clip: the row is there, greyed, and a click on
        // it does nothing.
        open(&mut shell);
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let row = text_pos(&output, "Edit effects").expect("the row");
        frame(&mut shell, &mut view, click(row + Vec2::new(4.0, 0.0)));
        frame(&mut shell, &mut view, Vec::new());
        assert_eq!(shell.selected(), Some(&clip));
        shell.close_clip_menu();

        // One effect: the row picks it.
        shell
            .effect_onto(&cards[0].id, clip.clone(), Micros(0), Micros(2_000_000))
            .unwrap();
        open(&mut shell);
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let row = text_pos(&output, "Edit effects").expect("the row");
        assert!(text_pos(&output, "Later").is_none_or(|badge| (badge.y - row.y).abs() > 6.0));
        frame(&mut shell, &mut view, click(row + Vec2::new(4.0, 0.0)));
        frame(&mut shell, &mut view, Vec::new());
        assert!(shell.clip_menu().is_none(), "the menu closed");
        assert_eq!(
            shell.selected_track_kind(),
            Some(cutlass_model::TrackType::Effect)
        );
        assert_eq!(shell.effect_name().as_deref(), Some(cards[0].name.as_str()));

        // Two effects: the row opens their names, and a name picks its own.
        shell
            .effect_onto(&cards[1].id, clip.clone(), Micros(0), Micros(2_000_000))
            .unwrap();
        open(&mut shell);
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let row = text_pos(&output, "Edit effects").expect("the row") + Vec2::new(4.0, 0.0);
        // The name is not on the menu until the row is under the pointer.
        let second = cards[1].name.as_str();
        frame(&mut shell, &mut view, vec![egui::Event::PointerMoved(row)]);
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let name = text_pos(&output, second).expect("the second effect's name");
        assert!(text_pos(&output, cards[0].name.as_str()).is_some());
        frame(&mut shell, &mut view, click(name + Vec2::new(4.0, 0.0)));
        frame(&mut shell, &mut view, Vec::new());
        assert!(shell.clip_menu().is_none(), "the menu closed");
        assert_eq!(shell.effect_name().as_deref(), Some(second));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_right_click_in_a_gap_offers_close_gap_and_closes_it() {
        let temp = std::env::temp_dir().join(format!("cutlass-gap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        for (id, start) in [("a", 0), ("b", 3_000_000)] {
            draft.tracks[0].segments.push(cutlass_model::Segment::new(
                id,
                "color",
                cutlass_model::TimeRange::new(start, 1_000_000),
            ));
        }
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
            output
        };
        let click = |pos: Pos2, button: egui::PointerButton| {
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            vec![egui::Event::PointerMoved(pos), press(true), press(false)]
        };
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let label = text_pos(&output, "a").expect("the first clip");
        // Two seconds in is halfway along the gap.
        let gap = Pos2::new(
            label.x - 8.0 + (2_000_000.0 * shell.pixel_size()) as f32,
            label.y + 4.0,
        );
        frame(
            &mut shell,
            &mut view,
            click(gap, egui::PointerButton::Secondary),
        );
        assert!(shell.gap_menu().is_some(), "no menu over the gap");
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let row = text_pos(&output, "Close gap").expect("the Close gap row");
        frame(
            &mut shell,
            &mut view,
            click(row + Vec2::new(4.0, 6.0), egui::PointerButton::Primary),
        );
        let starts: Vec<i64> = shell.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.target.start.0)
            .collect();
        assert_eq!(starts, [0, 1_000_000]);
        assert!(shell.gap_menu().is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn dragging_a_clip_onto_another_lane_moves_it_to_that_lane() {
        let temp = std::env::temp_dir().join(format!("cutlass-cross-lane-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "low",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        let mut overlay = cutlass_model::Track::main(TrackId::new("overlay"));
        overlay.flag = cutlass_model::TrackFlag::SubVideo;
        overlay.segments.push(cutlass_model::Segment::new(
            "high",
            "color",
            cutlass_model::TimeRange::new(8_000_000, 2_000_000),
        ));
        draft.tracks.insert(0, overlay);
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
        let low = text_pos(&output, "low").expect("the main clip");
        let high = text_pos(&output, "high").expect("the overlay clip");
        output.textures_delta.clear();
        assert!(high.y < low.y, "the overlay row sits above main");

        // The overlay row's middle: the name sits near its top edge, where a
        // drop opens a lane instead.
        let to = Pos2::new(
            low.x + 20.0,
            high.y + row_height(RowSize::Picture) / 2.0 - 4.0,
        );
        drag(&ctx, low, to, &mut shell, &mut view);

        let draft = shell.draft().unwrap();
        let overlay = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "overlay")
            .unwrap();
        assert!(
            overlay
                .segments
                .iter()
                .any(|segment| segment.id.as_str() == "low"),
            "the clip stayed on its own lane"
        );
        assert!(
            draft
                .tracks
                .iter()
                .filter(|track| track.id.as_str() != "overlay")
                .all(|track| track
                    .segments
                    .iter()
                    .all(|segment| segment.id.as_str() != "low"))
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_frame_a_clip_is_dropped_shows_it_where_it_landed() {
        let temp = std::env::temp_dir().join(format!("cutlass-drop-frame-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "low",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        let mut overlay = cutlass_model::Track::main(TrackId::new("overlay"));
        overlay.flag = cutlass_model::TrackFlag::SubVideo;
        overlay.segments.push(cutlass_model::Segment::new(
            "high",
            "color",
            cutlass_model::TimeRange::new(8_000_000, 2_000_000),
        ));
        draft.tracks.insert(0, overlay);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let frame = |events: Vec<egui::Event>, shell: &mut Shell, view: &mut ViewState| {
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
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let output = frame(Vec::new(), &mut shell, &mut view);
        let low = text_pos(&output, "low").expect("the main clip");
        let high = text_pos(&output, "high").expect("the overlay clip");
        // The overlay row's middle: the name sits near its top edge, where a
        // drop opens a lane instead.
        let to = Pos2::new(
            low.x + 20.0,
            high.y + row_height(RowSize::Picture) / 2.0 - 4.0,
        );
        frame(
            vec![egui::Event::PointerMoved(low), press(low, true)],
            &mut shell,
            &mut view,
        );
        frame(vec![egui::Event::PointerMoved(to)], &mut shell, &mut view);
        let dropped = frame(
            vec![egui::Event::PointerMoved(to), press(to, false)],
            &mut shell,
            &mut view,
        );

        let shown = text_pos(&dropped, "low").expect("the dropped clip is drawn");
        assert!(
            (shown.y - high.y).abs() < 1.0,
            "the drop frame shows the clip on the overlay row, not back on main"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_row_middle_is_the_row_and_its_edges_and_gaps_are_between() {
        let boxes = [
            Rect::from_min_size(Pos2::new(0.0, 100.0), Vec2::new(500.0, 48.0)),
            Rect::from_min_size(Pos2::new(0.0, 156.0), Vec2::new(500.0, 48.0)),
        ];
        let at = |y: f32| lane_spot(&boxes, Pos2::new(50.0, y));
        assert_eq!(at(20.0), Some(LaneSpot::Between(0)));
        assert_eq!(at(103.0), Some(LaneSpot::Between(0)));
        assert_eq!(at(124.0), Some(LaneSpot::On(0)));
        assert_eq!(at(146.0), Some(LaneSpot::Between(1)));
        assert_eq!(at(152.0), Some(LaneSpot::Between(1)));
        assert_eq!(at(158.0), Some(LaneSpot::Between(1)));
        assert_eq!(at(180.0), Some(LaneSpot::On(1)));
        assert_eq!(at(202.0), Some(LaneSpot::Between(2)));
        assert_eq!(at(400.0), Some(LaneSpot::Between(2)));
        assert_eq!(lane_spot(&[], Pos2::new(50.0, 20.0)), None);
    }

    #[test]
    fn the_only_main_clip_dragged_up_becomes_an_overlay() {
        let temp =
            std::env::temp_dir().join(format!("cutlass-main-to-overlay-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = true;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "only",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
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
        let only = text_pos(&output, "only").expect("the main clip");
        output.textures_delta.clear();
        drag(
            &ctx,
            only,
            only - Vec2::new(0.0, 60.0),
            &mut shell,
            &mut view,
        );
        let tracks = &shell.draft().unwrap().tracks;
        assert_eq!(tracks.len(), 2, "no overlay lane opened");
        assert!(tracks[0].is_sub_video());
        assert_eq!(tracks[0].segments[0].id.as_str(), "only");
        assert!(tracks[1].is_main() && tracks[1].segments.is_empty());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_dragged_clip_keeps_where_it_was_taken_and_does_not_jump() {
        let (mut shell, temp) = open_span("grab", 2_000_000);
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
        let text = text_pos(&output, "segment").expect("clip");
        output.textures_delta.clear();
        // Take the clip well inside its body, then move it 80 points right.
        let press = text + Vec2::new(120.0, 0.0);
        drag(
            &ctx,
            press,
            press + Vec2::new(80.0, 0.0),
            &mut shell,
            &mut view,
        );
        let start = shell.draft().unwrap().tracks[0].segments[0].target.start.0 as f64;
        let moved = start * shell.pixel_size();
        assert!(
            (moved - 80.0).abs() < 3.0,
            "the clip moved {moved} points for an 80 point drag"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn escape_during_a_clip_drag_puts_the_clip_back() {
        let (mut shell, temp) = open_span("escape", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let escape = egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        let frame = |events: Vec<egui::Event>, shell: &mut Shell, view: &mut ViewState| {
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
        let output = frame(Vec::new(), &mut shell, &mut view);
        let from = text_pos(&output, "segment").expect("clip") + Vec2::new(60.0, 0.0);
        let to = from + Vec2::new(150.0, 0.0);
        let before = shell.draft().unwrap().tracks.clone();
        let revision = shell.revision();
        frame(
            vec![egui::Event::PointerMoved(from), press(from, true)],
            &mut shell,
            &mut view,
        );
        frame(vec![egui::Event::PointerMoved(to)], &mut shell, &mut view);
        assert!(shell.move_preview().is_some());
        frame(vec![escape], &mut shell, &mut view);
        assert!(shell.move_preview().is_none());
        frame(
            vec![egui::Event::PointerMoved(to + Vec2::new(10.0, 0.0))],
            &mut shell,
            &mut view,
        );
        assert!(shell.move_preview().is_none());
        frame(
            vec![egui::Event::PointerMoved(to), press(to, false)],
            &mut shell,
            &mut view,
        );
        assert_eq!(shell.draft().unwrap().tracks, before);
        assert_eq!(shell.revision(), revision);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn the_toast_paints_on_the_top_layer() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.set_notice("the edit was refused");
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            paint_notice(&mut shell, ui, 100.0, 40.0);
            // A floating layer raised after the toast, as a drag or menu is.
            let late = egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("late"));
            ui.ctx().move_to_top(late);
            ui.ctx().layer_painter(late).rect_filled(
                Rect::from_min_size(Pos2::ZERO, Vec2::splat(400.0)),
                0.0,
                Color32::RED,
            );
        });
        output.textures_delta.clear();
        let top = ctx.memory(|memory| memory.layer_ids().last());
        assert_eq!(top, Some(notice_layer()));
    }

    /// Ten overlay rows, main, and audio rows under it.
    fn crowded(audio: usize) -> Vec<(bool, f32)> {
        let mut kinds = vec![(false, 62.0); 10];
        kinds.push((true, 90.0));
        kinds.extend(std::iter::repeat_n((false, 58.0), audio));
        kinds
    }

    #[test]
    fn a_lane_part_scrolled_out_keeps_its_clips_at_full_height() {
        let temp = std::env::temp_dir().join(format!("cutlass-crowded-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "low",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        for index in 0..14 {
            let mut track = cutlass_model::Track::main(format!("words-{index}"));
            track.kind = cutlass_model::TrackType::Text;
            track.flag = cutlass_model::TrackFlag::None;
            track.segments.push(cutlass_model::Segment::new(
                format!("line-{index}"),
                "text",
                cutlass_model::TimeRange::new(0, 1_000_000),
            ));
            draft.tracks.insert(0, track);
        }
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let words = lane_color(&cutlass_model::TrackType::Text);
        let mut heights = |view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::Rect(rect) if rect.fill == words => Some(rect.rect.height()),
                    _ => None,
                })
                .collect::<Vec<f32>>()
        };
        assert!(!heights(&mut view).is_empty());
        // Part of the lane next to main slides under main.
        view.lanes_up = 20.0;
        let scrolled = heights(&mut view);
        assert!(!scrolled.is_empty());
        assert!(
            scrolled.iter().all(|height| *height == tokens::CLIP_HEIGHT),
            "{scrolled:?}"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_picked_clip_adds_its_tools_to_the_toolbar() {
        use cutlass_model::TrackType::{Audio, Video};
        let names = |picked: Option<&cutlass_model::TrackType>| {
            left_tools(picked)
                .into_iter()
                .flatten()
                .map(|tool| tool.0)
                .collect::<Vec<_>>()
        };
        let base = [
            "undo",
            "redo",
            "splitAt",
            "trimLeft",
            "trimRight",
            "delete",
            "marker",
        ];
        assert_eq!(names(None), base);
        let video = names(Some(&Video));
        assert_eq!(video[..base.len()], base);
        assert_eq!(
            video[base.len()..],
            [
                "freeze",
                "reverse",
                "mirror",
                "rotate",
                "crop",
                "transcribe",
                "splitScenes"
            ]
        );
        let audio = names(Some(&Audio));
        assert_eq!(audio[..base.len()], base);
        assert_eq!(
            audio[base.len()..],
            [
                "transcribe",
                "beat",
                "normalize",
                "enhanceVoice",
                "reduceNoise"
            ]
        );
        for name in [
            "splitAt",
            "trimLeft",
            "trimRight",
            "blade",
            "select",
            "normalize",
            "enhanceVoice",
            "reduceNoise",
        ] {
            assert!(icons::require(name).is_ok(), "{name}");
        }
        // Not built yet: drawn dimmed.
        let (decoders, encoders) = crate::register_codecs();
        let shell = Shell::new(decoders, encoders, Settings::default());
        for name in ["normalize", "enhanceVoice", "reduceNoise"] {
            assert!(!tool_enabled(&shell, name), "{name}");
        }
    }

    #[test]
    fn the_right_tools_take_the_width_they_are_kept_for() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let mut drawn = 0.0;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| {
                let mut pinned = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(Rect::from_min_size(
                            Pos2::new(600.0, 10.0),
                            Vec2::new(600.0, 36.0),
                        ))
                        .layout(Layout::left_to_right(Align::Center)),
                );
                pinned.spacing_mut().item_spacing.x = TOOL_SPACING;
                paint_tools(&mut shell, &mut pinned, &mut view, TOOLS_RIGHT);
                drawn = pinned.min_rect().width();
            },
        );
        output.textures_delta.clear();
        assert!(
            (drawn - tools_width(TOOLS_RIGHT)).abs() < 0.5,
            "drawn {drawn}, kept {}",
            tools_width(TOOLS_RIGHT)
        );
        assert_eq!(
            TOOLS_RIGHT.first().copied().flatten().map(|tool| tool.0),
            Some("snap")
        );
        assert_eq!(
            TOOLS_RIGHT.last().copied().flatten().map(|tool| tool.0),
            Some("zoomIn")
        );
    }

    #[test]
    fn the_cover_column_between_heads_and_lanes_is_bare() {
        let temp = std::env::temp_dir().join(format!("cutlass-cover-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "low",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
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
        let at = text_pos(&output, "low").expect("the clip");
        let lane_fill_at = |point: Pos2| {
            output.shapes.iter().any(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) => {
                    rect.fill == tokens::LANE
                        && rect.rect.contains(point)
                        && clipped.clip_rect.contains(point)
                }
                _ => false,
            })
        };
        // The clip's name sits a little inside the lanes' left edge.
        let cover = Pos2::new(at.x - 12.0 - tokens::COVER_WIDTH / 2.0, at.y);
        let head = Pos2::new(at.x - 12.0 - tokens::COVER_WIDTH - 20.0, at.y);
        assert!(!lane_fill_at(cover), "the cover column takes no lane fill");
        assert!(lane_fill_at(head), "the head keeps its lane fill");
        let cover = text_pos(&output, "Cover").expect("the Cover box");
        assert!(text_pos(&output, "Later").is_none(), "the box is built");
        assert!(
            cover.x < at.x && cover.x > head.x,
            "the box is in the column"
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn every_lane_head_shows_its_icons() {
        let temp = std::env::temp_dir().join(format!("cutlass-heads-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "low",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        let mut overlay = cutlass_model::Track::main(TrackId::new("overlay"));
        overlay.flag = cutlass_model::TrackFlag::SubVideo;
        overlay.segments.push(cutlass_model::Segment::new(
            "high",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        draft.tracks.insert(0, overlay);
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
        for name in ["low", "high"] {
            let at = text_pos(&output, name).expect("the clip");
            // Something drawn, and not cut away, in the head left of the clip.
            let head = output.shapes.iter().any(|clipped| {
                let bounds = clipped.shape.visual_bounding_rect();
                bounds.right() < at.x - tokens::COVER_WIDTH
                    && bounds.left() > at.x - tokens::COVER_WIDTH - tokens::TRACK_HEAD - 20.0
                    && (bounds.center().y - at.y).abs() < 40.0
                    && clipped.clip_rect.intersect(bounds).area() > 0.0
            });
            assert!(head, "the {name} lane has no head showing");
        }
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_lane_head_has_the_buttons_its_kind_uses() {
        let lane = |kind: cutlass_model::TrackType, main: bool| crate::shell::LaneView {
            track_id: TrackId::new("lane"),
            kind,
            main,
            name: None,
            mute: false,
            hide: false,
            lock: false,
            solo: false,
            mute_by_solo: false,
            clips: Vec::new(),
        };
        use cutlass_model::TrackType::{Audio, Sticker, Text, Video};
        assert_eq!(
            head_buttons(&lane(Video, true)),
            ["lock", "hide", "mute", "more"]
        );
        assert_eq!(
            head_buttons(&lane(Video, false)),
            ["lock", "hide", "mute", "more"]
        );
        assert_eq!(head_buttons(&lane(Audio, false)), ["lock", "mute", "more"]);
        assert_eq!(head_buttons(&lane(Text, false)), ["lock", "hide"]);
        assert_eq!(
            head_buttons(&lane(cutlass_model::TrackType::Mixed, false)),
            ["lock", "hide", "mute"]
        );
        assert_eq!(head_buttons(&lane(Sticker, false)), ["lock", "hide"]);
        // A button keeps its column whichever buttons the lane has.
        let slots = |lane| {
            head_buttons(&lane)
                .into_iter()
                .map(head_slot)
                .collect::<Vec<_>>()
        };
        assert_eq!(slots(lane(Audio, false)), [1.0, 3.0, 4.0]);
        assert_eq!(slots(lane(Text, false)), [1.0, 2.0]);
        // Every icon the head draws is in the set.
        for name in [
            "lock",
            "unlock",
            "view",
            "viewOff",
            "volume",
            "volumeOff",
            "more",
        ] {
            assert!(icons::require(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn scrolling_the_lanes_above_main_sinks_main_then_slides_them_under_it() {
        let area = Rect::from_min_size(Pos2::new(0.0, 100.0), Vec2::new(800.0, 400.0));
        let kinds = crowded(1);
        let rest = lane_layout(&kinds, area, 0.0, 0.0);
        let main = rest.rows[10];
        // The audio lane shows whole under main, with room to drop below it.
        assert!(rest.rows[11].bottom() + lane_room() <= area.bottom() + 0.5);
        assert_eq!(rest.down_max, 0.0);
        // The lanes above fill up from main; the top ones are past the edge.
        assert_eq!(main.top() - rest.rows[9].bottom(), LANE_GAP);
        assert!(rest.rows[0].top() < area.top());
        assert!(rest.up_max > 0.0);

        assert!(!rest.peek);

        // A little up, main goes down with the lanes above it.
        let some = lane_layout(&kinds, area, 10.0, 0.0);
        assert_eq!(some.rows[10].top(), main.top() + 10.0);
        assert_eq!(some.rows[10].top() - some.rows[9].bottom(), LANE_GAP);

        // Scrolled all the way up, main sits at the sheet's foot over a strip,
        // the audio lane is folded to it, and the top lane shows with a lane
        // of room above it.
        let top = lane_layout(&kinds, area, f32::MAX, 0.0);
        let sunk = top.rows[10];
        assert!((sunk.bottom() - (area.bottom() - PEEK)).abs() < 0.5);
        assert_eq!(sunk.height(), main.height());
        assert!(top.peek);
        assert_eq!(top.shown()[11].height(), 0.0, "a folded lane takes no drop");
        assert!((top.rows[0].top() - (area.top() + lane_room())).abs() < 0.5);
        // The lanes next to main slid under it: what shows of them stays in
        // the band above main and never over main.
        for shown in &top.shown()[..10] {
            assert!(shown.bottom() <= sunk.top());
        }
    }

    #[test]
    fn a_lane_out_of_view_is_scrolled_in_whole_and_one_in_view_is_left_alone() {
        let area = Rect::from_min_size(Pos2::new(0.0, 100.0), Vec2::new(800.0, 400.0));
        let whole = |kinds: &[(bool, f32)], index: usize, up: f32, down: f32| {
            let laid = lane_layout(kinds, area, up, down);
            let main = kinds.iter().position(|(main, _)| *main);
            let folded = laid.peek && main.is_some_and(|main| index > main);
            let band = if main == Some(index) {
                area
            } else {
                laid.bands[index]
            };
            let row = laid.rows[index];
            !folded && row.top() >= band.top() - 0.5 && row.bottom() <= band.bottom() + 0.5
        };
        // Ten lanes above main and eight under it, and a stack with no main.
        let no_main = vec![(false, 62.0); 14];
        for kinds in [crowded(8), crowded(1), no_main] {
            let rest = lane_layout(&kinds, area, 0.0, 0.0);
            let starts = [
                (0.0, 0.0),
                (rest.up_max, 0.0),
                (0.0, rest.down_max),
                (rest.up_max / 2.0, rest.down_max / 2.0),
            ];
            for index in 0..kinds.len() {
                for (up, down) in starts {
                    let (to_up, to_down) = reveal_row(&kinds, area, index, up, down);
                    assert!(
                        whole(&kinds, index, to_up, to_down),
                        "lane {index} of {} from ({up}, {down}) at ({to_up}, {to_down})",
                        kinds.len()
                    );
                    // A lane that showed whole already keeps the scroll it had.
                    if whole(&kinds, index, up, down) {
                        assert_eq!((to_up, to_down), (up, down), "lane {index}");
                    }
                    // Revealed, asking again moves nothing more.
                    assert_eq!(
                        reveal_row(&kinds, area, index, to_up, to_down),
                        (to_up, to_down)
                    );
                }
            }
        }
        // A lane taller than the sheet shows what fits and does not loop.
        let tall = vec![(false, 900.0), (true, 90.0)];
        let (up, down) = reveal_row(&tall, area, 0, 0.0, 0.0);
        assert!(up.is_finite() && down.is_finite());
    }

    #[test]
    fn one_wheel_runs_through_the_lanes_above_main_then_the_whole_stack() {
        let area = Rect::from_min_size(Pos2::new(0.0, 100.0), Vec2::new(800.0, 400.0));
        let layout = lane_layout(&crowded(8), area, 0.0, 0.0);
        assert!(layout.up_max > 0.0 && layout.down_max > 0.0);
        // Scrolled up a little, scrolling down first brings those lanes back.
        let (up, down) = chain_scroll(&layout, 30.0, 0.0, -20.0);
        assert_eq!((up, down), (10.0, 0.0));
        // Past them, it goes on into the lanes under main.
        let (up, down) = chain_scroll(&layout, 10.0, 0.0, -50.0);
        assert_eq!((up, down), (0.0, 40.0));
        // Scrolling up undoes it in the same order.
        let (up, down) = chain_scroll(&layout, 0.0, 40.0, 60.0);
        assert_eq!((up, down), (20.0, 0.0));
        // Scrolling the lanes above main carries main down with them.
        let lifted = lane_layout(&crowded(8), area, 30.0, 0.0);
        assert_eq!(lifted.rows[10].top(), layout.rows[10].top() + 30.0);

        // Scrolled to the bottom, the whole stack went up together: main and
        // the lanes above it moved as far as the last lane did, which shows
        // with a lane of room under it.
        let bottom = lane_layout(&crowded(8), area, 0.0, f32::MAX);
        let last = bottom.rows.last().unwrap();
        assert!((last.bottom() - (area.bottom() - lane_room())).abs() < 0.5);
        let lift = layout.rows[10].top() - bottom.rows[10].top();
        assert!(
            (lift - layout.down_max).abs() < 0.5,
            "main went up with the stack"
        );
        assert!((layout.rows[9].top() - bottom.rows[9].top() - lift).abs() < 0.5);
        assert!((layout.rows[11].top() - bottom.rows[11].top() - lift).abs() < 0.5);
    }

    #[test]
    fn lanes_that_fit_with_room_to_spare_center_main_and_do_not_scroll() {
        let area = Rect::from_min_size(Pos2::new(0.0, 100.0), Vec2::new(800.0, 400.0));
        let layout = lane_layout(
            &[(false, 40.0), (true, 90.0), (false, 58.0)],
            area,
            0.0,
            0.0,
        );
        assert_eq!(layout.rows[1].center().y, area.center().y);
        assert_eq!((layout.up_max, layout.down_max), (0.0, 0.0));
    }

    #[test]
    fn a_drag_near_the_sheet_edge_scrolls_the_lanes_toward_it() {
        let tracks = Rect::from_min_max(Pos2::new(0.0, 100.0), Pos2::new(800.0, 400.0));
        assert!(drag_scroll_step(tracks, 101.0) > drag_scroll_step(tracks, 120.0));
        assert!(drag_scroll_step(tracks, 120.0) > 0.0);
        assert!(drag_scroll_step(tracks, 390.0) < 0.0);
        assert_eq!(drag_scroll_step(tracks, 250.0), 0.0);
        assert_eq!(drag_scroll_step(tracks, 60.0), 12.0);
    }

    #[test]
    fn an_overlay_video_row_is_about_half_main_and_no_thinner_than_text() {
        let lane = |kind: cutlass_model::TrackType, main: bool| crate::shell::LaneView {
            track_id: TrackId::new("t"),
            kind,
            main,
            name: None,
            mute: false,
            hide: false,
            lock: false,
            solo: false,
            mute_by_solo: false,
            clips: Vec::new(),
        };
        let view = ViewState::new();
        let main = lane_height(&view, &lane(cutlass_model::TrackType::Video, true));
        let overlay = lane_height(&view, &lane(cutlass_model::TrackType::Video, false));
        let text = lane_height(&view, &lane(cutlass_model::TrackType::Text, false));
        assert!(overlay < main);
        assert!(
            (overlay - main / 2.0).abs() < 8.0,
            "{overlay} against {main}"
        );
        assert!(text <= overlay);
        assert_eq!(main, 76.0);
    }

    #[test]
    fn dragging_a_clip_above_the_top_lane_opens_a_lane_there() {
        let temp = std::env::temp_dir().join(format!("cutlass-new-lane-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "low",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        let mut overlay = cutlass_model::Track::main(TrackId::new("overlay"));
        overlay.flag = cutlass_model::TrackFlag::SubVideo;
        overlay.segments.push(cutlass_model::Segment::new(
            "high",
            "color",
            cutlass_model::TimeRange::new(8_000_000, 2_000_000),
        ));
        draft.tracks.insert(0, overlay);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let before = shell.draft().unwrap().tracks.clone();
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let press = |pos: Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let frame = |events: Vec<egui::Event>, shell: &mut Shell, view: &mut ViewState| {
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
        let output = frame(Vec::new(), &mut shell, &mut view);
        let low = text_pos(&output, "low").expect("the main clip");
        let high = text_pos(&output, "high").expect("the overlay clip");
        let to = Pos2::new(low.x + 20.0, high.y - 40.0);

        frame(
            vec![egui::Event::PointerMoved(low), press(low, true)],
            &mut shell,
            &mut view,
        );
        frame(vec![egui::Event::PointerMoved(to)], &mut shell, &mut view);
        assert_eq!(
            shell.new_lane_preview(),
            Some((0, cutlass_model::TrackType::Video))
        );
        assert_eq!(shell.draft().unwrap().tracks, before);
        frame(
            vec![egui::Event::PointerMoved(to), press(to, false)],
            &mut shell,
            &mut view,
        );

        let tracks = &shell.draft().unwrap().tracks;
        assert_eq!(tracks.len(), 3);
        assert!(tracks[0].is_sub_video());
        assert_eq!(tracks[0].segments[0].id.as_str(), "low");
        assert_eq!(tracks[1].id.as_str(), "overlay");
        assert!(shell.new_lane_preview().is_none());
        shell.undo().unwrap();
        assert_eq!(shell.draft().unwrap().tracks, before);
        let _ = std::fs::remove_dir_all(&temp);
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
    fn the_marker_tool_marks_the_playhead_once_and_arms_nothing() {
        let (mut shell, temp) = open_span("marker", 2_000_000);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let id = shell.draft().unwrap().tracks[0].segments[0].id.clone();
        shell.select(Some(id.clone()));
        shell.seek(Micros(1_000_000));
        tool_clicked(&mut shell, "marker");
        let marks = shell.markers();
        assert_eq!(marks.len(), 1, "notice {}", shell.notice());
        assert_eq!(marks[0].segment_id.as_ref(), Some(&id));
        assert_eq!(marks[0].time, Micros(1_000_000));
        // No tool stays armed: a click on the clip afterwards drops nothing.
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: screen,
                ..Default::default()
            },
            |ui| show(&mut shell, ui, &mut view, None),
        );
        let text = text_pos(&output, id.as_str()).expect("clip");
        output.textures_delta.clear();
        let center = Pos2::new(text.x + 20.0, text.y + 12.0);
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
        assert_eq!(shell.markers().len(), 1, "the click dropped a mark");
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

    /// Draw frames until the chosen import has been copied and landed.
    fn land_import(shell: &mut Shell, view: &mut ViewState) {
        let ctx = egui::Context::default();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            if shell.job().is_none() || std::time::Instant::now() > until {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// The last place `wanted` is painted, so a dialog's button wins over
    /// the same word in the bar beneath it.
    fn last_text_pos(output: &egui::FullOutput, wanted: &str) -> Option<Pos2> {
        output
            .shapes
            .iter()
            .rev()
            .find_map(|clipped| shape_pos(&clipped.shape, wanted))
            .map(|pos| pos + Vec2::new(4.0, 4.0))
    }

    /// Wait for the cover job to land.
    fn land_cover(shell: &mut Shell) {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while shell.poll_job() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(shell.job().is_none(), "the cover job did not finish");
    }

    /// A project of one blue clip two seconds long on a 64 by 32 canvas.
    fn blue_project(name: &str) -> (Shell, std::path::PathBuf) {
        let temp =
            std::env::temp_dir().join(format!("cutlass-cover-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 64;
        draft.canvas_config.height = 32;
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
            "blue",
            "color",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        draft.duration = Micros(2_000_000);
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        (shell, temp)
    }

    #[test]
    fn the_cover_box_opens_the_dialog_and_save_makes_the_frame_the_cover() {
        let (mut shell, temp) = blue_project("frame");
        shell.seek(Micros(500_000));
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
            output
        };
        let click = |pos: Pos2| {
            let event = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            [
                vec![egui::Event::PointerMoved(pos), event(true)],
                vec![event(false)],
            ]
        };
        let press = |shell: &mut Shell, view: &mut ViewState, label: &str| {
            let output = frame(shell, view, Vec::new());
            let at = last_text_pos(&output, label).unwrap_or_else(|| panic!("no {label}"));
            for events in click(at) {
                frame(shell, view, events);
            }
        };
        // The box beside the main lane says Cover, with no Later badge.
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(text_pos(&output, "Later").is_none());
        press(&mut shell, &mut view, "Cover");
        assert!(view.cover.is_some());
        let output = frame(&mut shell, &mut view, Vec::new());
        for label in [
            "Select a cover",
            "Select from video",
            "Local",
            "Save",
            "Cancel",
        ] {
            assert!(text_pos(&output, label).is_some(), "{label}");
        }
        // Cancel leaves no cover and the playhead where it was.
        press(&mut shell, &mut view, "Cancel");
        assert!(view.cover.is_none());
        assert_eq!(shell.cover(), None);
        assert_eq!(shell.playhead(), Micros(500_000));

        // Dragging the strip's handle picks a later frame and leaves the
        // project's playhead alone.
        press(&mut shell, &mut view, "Cover");
        let output = frame(&mut shell, &mut view, Vec::new());
        let tabs = text_pos(&output, "Select from video").expect("the tabs");
        let strip_y = tabs.y + 50.0;
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let from = Pos2::new(400.0, strip_y);
        frame(
            &mut shell,
            &mut view,
            vec![egui::Event::PointerMoved(from), button(from, true)],
        );
        for x in [500.0, 700.0, 900.0] {
            frame(
                &mut shell,
                &mut view,
                vec![egui::Event::PointerMoved(Pos2::new(x, strip_y))],
            );
            assert_eq!(shell.playhead(), Micros(500_000));
        }
        let to = Pos2::new(900.0, strip_y);
        frame(&mut shell, &mut view, vec![button(to, false)]);
        assert_eq!(shell.playhead(), Micros(500_000));
        press(&mut shell, &mut view, "Save");
        assert!(view.cover.is_none());
        land_cover(&mut shell);
        assert_eq!(shell.playhead(), Micros(500_000));
        let Some(cutlass_model::Cover::Frame { time }) = shell.cover() else {
            panic!("no frame cover: {:?}", shell.cover());
        };
        assert!(time.0 > 1_000_000, "{time:?}");
        assert!(temp.join(cutlass_model::COVER_FILE).is_file());
        // The box now shows the cover, under the word.
        let output = frame(&mut shell, &mut view, Vec::new());
        assert!(text_pos(&output, "Cover").is_some());
    }

    #[test]
    fn closing_the_window_in_a_project_goes_back_to_the_projects() {
        let (mut shell, temp) = blue_project("close");
        shell.seek(Micros(700_000));
        shell
            .apply(cutlass_session::Request::SetProjectName {
                name: "Kept".to_owned(),
            })
            .unwrap();
        assert!(shell.unsaved());
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        // One pass as eframe runs it: the logic hook, then the window.
        let pass = |shell: &mut Shell, view: &mut ViewState, close: bool| {
            let mut input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            };
            if close {
                input.viewports.insert(
                    egui::ViewportId::ROOT,
                    egui::ViewportInfo {
                        events: vec![egui::ViewportEvent::Close],
                        ..Default::default()
                    },
                );
            }
            let mut output = ctx.run_ui(input, |ui| {
                close_to_projects(shell, view, ui.ctx());
                show(shell, ui, view, None);
            });
            output.textures_delta.clear();
            output.viewport_output[&egui::ViewportId::ROOT]
                .commands
                .contains(&egui::ViewportCommand::CancelClose)
        };
        pass(&mut shell, &mut view, false);
        // The close button in the editor: the app stays, the project is
        // saved and closed, and the projects show.
        assert!(pass(&mut shell, &mut view, true));
        assert!(shell.draft().is_none());
        let saved = cutlass_store::Folder::read(&temp).unwrap();
        assert_eq!(saved.draft.name, "Kept");
        pass(&mut shell, &mut view, false);
        assert!(shell.chrome.projects);
        // At the projects the button closes the app.
        assert!(!pass(&mut shell, &mut view, true));

        // When the app quits by itself, as for an update, nothing holds it.
        shell.open_folder(&temp).unwrap();
        view.quitting = true;
        assert!(!pass(&mut shell, &mut view, true));
        assert!(shell.draft().is_some());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_home_card_shows_the_first_frame_and_not_the_picked_cover() {
        let root = std::env::temp_dir().join(format!("cutlass-cards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (_, project) = blue_project("card");
        let folder = root.join("Blue");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::rename(&project, &folder).unwrap();
        // A picked cover, 10 by 10 and red, which the card does not show.
        image::RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0]))
            .save(folder.join(cutlass_model::COVER_FILE))
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let mut view = ViewState::new();
        view.projects_root = Some(root.clone());
        let ctx = egui::Context::default();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut uploaded = Vec::new();
        while !uploaded.contains(&[64, 32]) {
            assert!(std::time::Instant::now() < until, "no card picture");
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            uploaded.extend(
                output
                    .textures_delta
                    .set
                    .values()
                    .flat_map(|deltas| deltas.iter().map(|delta| delta.image.size())),
            );
            output.textures_delta.clear();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!uploaded.contains(&[10, 10]));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_local_tab_takes_a_picture_from_the_picker_and_saves_it() {
        let (mut shell, temp) = blue_project("local");
        let png = temp.join("poster.png");
        image::RgbaImage::from_pixel(40, 20, image::Rgba([0x33, 0x66, 0x99, 255]))
            .save(&png)
            .unwrap();
        let mut view = ViewState::new();
        view.cover = Some(cover::CoverDialog::open(&shell));
        let ctx = egui::Context::default();
        let screen = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));
        let frame = |shell: &mut Shell, view: &mut ViewState| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        frame(&mut shell, &mut view);
        view.dialogs
            .finish_for_test(Picker::Cover, Some(png.clone()));
        frame(&mut shell, &mut view);
        let output = frame(&mut shell, &mut view);
        assert!(text_pos(&output, "poster.png").is_some());
        assert!(text_pos(&output, "Choose picture").is_some());
        let save = last_text_pos(&output, "Save").unwrap();
        for pressed in [true, false] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: screen,
                    events: vec![
                        egui::Event::PointerMoved(save),
                        egui::Event::PointerButton {
                            pos: save,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::default(),
                        },
                    ],
                    ..Default::default()
                },
                |ui| show(&mut shell, ui, &mut view, None),
            );
            output.textures_delta.clear();
        }
        land_cover(&mut shell);
        assert_eq!(
            shell.cover(),
            Some(&cutlass_model::Cover::Image {
                path: png.display().to_string()
            })
        );
    }

    #[test]
    fn modify_opens_the_project_dialog_and_save_is_one_undo_step() {
        let (mut shell, _temp) = open_span("project", 1_000_000);
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
            output
        };
        let click = |pos: Pos2| {
            [
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                }],
            ]
        };
        shell.select(None);
        let output = frame(&mut shell, &mut view, Vec::new());
        let modify = last_text_pos(&output, "Modify").expect("a Modify button");
        for events in click(modify) {
            frame(&mut shell, &mut view, events);
        }
        let output = frame(&mut shell, &mut view, Vec::new());
        let text = painted_text(&output);
        for label in [
            "Ratio",
            "Resolution",
            "Frame rate",
            "Drop frame",
            "Color space",
            "Save",
        ] {
            assert!(text.contains(label), "the dialog has no {label}");
        }
        for (row, name) in [
            ("Ratio", "9:16"),
            ("Resolution", "720p"),
            ("Frame rate", "29.97 fps"),
            ("Color space", "HLG"),
        ] {
            let output = frame(&mut shell, &mut view, Vec::new());
            let picker = last_text_pos(&output, row).expect(row) + Vec2::new(240.0, 0.0);
            for events in click(picker) {
                frame(&mut shell, &mut view, events);
            }
            let output = frame(&mut shell, &mut view, Vec::new());
            let at = last_text_pos(&output, name).expect(name);
            for events in click(at) {
                frame(&mut shell, &mut view, events);
            }
        }
        let output = frame(&mut shell, &mut view, Vec::new());
        let save = last_text_pos(&output, "Save").expect("Save");
        for events in click(save) {
            frame(&mut shell, &mut view, events);
        }
        assert!(view.project_dialog.is_none(), "{}", shell.notice());
        let draft = shell.draft().unwrap();
        assert_eq!(
            (draft.canvas_config.width, draft.canvas_config.height),
            (720, 1280)
        );
        assert_eq!(draft.frame_rate(), Some(FrameRate::FPS_29_97));
        assert_eq!(draft.color_space, 67);
        shell.undo().unwrap();
        let draft = shell.draft().unwrap();
        assert_eq!(draft.color_space, 0);
        assert_ne!(draft.canvas_config.height, 1280);
    }

    #[test]
    fn the_export_dialog_names_its_file_by_what_it_writes_and_reads_a_typed_bitrate() {
        let mut view = ViewState::new();
        assert_eq!(export_extension(&view), "mp4");
        view.export_mov = true;
        assert_eq!(export_extension(&view), "mov");
        view.export.picture = false;
        assert_eq!(export_extension(&view), "m4a");
        view.export.audio.codec = cutlass_decode::CodecId::Alac;
        assert_eq!(export_extension(&view), "m4a");
        view.export.audio.codec = cutlass_decode::CodecId::Ac3;
        assert_eq!(export_extension(&view), "m4a");
        view.export.audio.codec = cutlass_decode::CodecId::Eac3;
        assert_eq!(export_extension(&view), "m4a");
        view.export_mov = false;
        view.export.picture = true;
        assert_eq!(export_extension(&view), "mp4");
        view.export.picture = false;
        view.export.audio.codec = cutlass_decode::CodecId::Pcm;
        assert_eq!(export_extension(&view), "wav");
        view.export.sound = false;
        assert_eq!(export_extension(&view), "gif");
        view.export.gif = Some(crate::shell::GifExport {
            webp: true,
            ..crate::shell::GifExport::default()
        });
        assert_eq!(export_extension(&view), "webp");
        assert_eq!(parse_mbps("16"), Some(16_000_000));
        assert_eq!(parse_mbps(" 7.5 "), Some(7_500_000));
        for bad in ["", "0", "-3", "fast", "401", "NaN"] {
            assert_eq!(parse_mbps(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_export_dialog_writes_a_movie_as_a_job_and_offers_its_folder() {
        let (mut shell, temp) = open_span("movie", 1_000_000);
        let destination = temp.join("cut.mp4");
        let mut view = ViewState::new();
        view.export_open = true;
        view.export.short_side = 480;
        view.export_path = destination.display().to_string();
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
        let click = |pos: Pos2| {
            vec![
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                }],
            ]
        };
        let output = frame(&mut shell, &mut view, Vec::new());
        let text = painted_text(&output);
        for label in [
            "MP4",
            "H.264",
            "Whole timeline",
            "Recommended",
            "Frame rate",
        ] {
            assert!(text.contains(label), "the dialog has no {label}");
        }
        let export = last_text_pos(&output, "Export").expect("an Export button");
        for events in click(export) {
            frame(&mut shell, &mut view, events);
        }
        assert!(
            !view.export_open,
            "the dialog stays open: {} at {export:?}",
            shell.notice()
        );
        assert_eq!(shell.job().map(|job| job.label), Some("Export".to_owned()));
        // A debug Rust encoder on a small CI runner takes well over a minute.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while shell.job().is_some() && std::time::Instant::now() < until {
            frame(&mut shell, &mut view, Vec::new());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(destination.is_file(), "{}", shell.notice());
        frame(&mut shell, &mut view, Vec::new());
        let output = frame(&mut shell, &mut view, Vec::new());
        let text = painted_text(&output);
        assert!(text.contains("Exported cut.mp4"), "{text}");
        assert!(text.contains(crate::shell::REVEAL_LABEL));
        let close = last_text_pos(&output, "Close").expect("a Close button");
        for events in click(close) {
            frame(&mut shell, &mut view, events);
        }
        assert!(shell.exported().is_none());
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
            (shell.view_origin().0, view.lanes_down)
        };
        let (before, _) = spin(Vec2::ZERO, false);
        let (origin, scroll) = spin(Vec2::new(0.0, -120.0), false);
        assert_eq!(origin, before, "the wheel moved time, not the lanes");
        assert!(scroll > 0.0, "the audio lanes did not scroll: {scroll}");
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

    #[test]
    fn an_update_offer_shows_its_notes_and_escape_puts_it_off() {
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        let name = cutlass_update::Platform::current().unwrap().asset_name();
        let json = format!(
            r#"{{"tag_name":"v99.0.0","body":"Faster export\n\n- Fixes","assets":[
                {{"name":"{name}","browser_download_url":"https://example.test/p"}},
                {{"name":"{name}.sha256","browser_download_url":"https://example.test/s"}}]}}"#
        );
        let running = cutlass_update::Version::parse("0.1.0").unwrap();
        let platform = cutlass_update::Platform::current().unwrap();
        let offer = cutlass_update::offer(&json, &running, platform)
            .unwrap()
            .unwrap();
        shell.offer_update(offer);
        let mut view = ViewState::new();
        let ctx = egui::Context::default();
        let run = |shell: &mut Shell, view: &mut ViewState, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    events,
                    ..Default::default()
                },
                |ui| show(shell, ui, view, None),
            );
            output.textures_delta.clear();
            output
        };
        let painted = painted_text(&run(&mut shell, &mut view, Vec::new()));
        assert!(painted.contains("Cutlass 99.0.0 is available"), "{painted}");
        assert!(painted.contains("Faster export") && painted.contains("- Fixes"));
        assert!(painted.contains("Update and restart") && painted.contains("Later"));
        run(
            &mut shell,
            &mut view,
            vec![key(egui::Key::Escape, egui::Modifiers::NONE)],
        );
        assert!(shell.update_offer().is_none());
        assert!(shell.job().is_none(), "nothing downloads");

        view.settings_open = true;
        let painted = painted_text(&run(&mut shell, &mut view, Vec::new()));
        assert!(painted.contains("Check for updates"), "{painted}");
    }

    /// The arrow beside Linkage opens its settings; Done keeps the ticked
    /// kinds and Cancel leaves them.
    #[test]
    fn linkage_settings_choose_the_kinds_that_follow() {
        use egui_kittest::kittest::Queryable;
        let temp = std::env::temp_dir().join(format!("cutlass-linkage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut harness = egui_kittest::Harness::builder()
            .with_size(Vec2::new(1280.0, 800.0))
            .build_ui_state(
                |ui, (shell, view): &mut (Shell, ViewState)| show(shell, ui, view, None),
                (shell, {
                    // The empty media shelf's line has a glyph the test fonts lack.
                    let mut view = ViewState::new();
                    view.library_shelf = Shelf::Filters;
                    view
                }),
            );
        harness.run_steps(2);
        harness.get_by_label("Linkage options").click();
        harness.run_steps(2);
        harness.get_by_label("Linkage settings").click();
        harness.run_steps(2);
        assert!(harness.state().1.linkage_kinds.is_some());
        // The library has an Audio tab too; the dialog is drawn last.
        harness.query_all_by_label("Audio").last().unwrap().click();
        harness.run_steps(1);
        harness.get_by_label("Cancel").click();
        harness.run_steps(2);
        assert_eq!(
            harness.state().0.linkage_kinds(),
            Some(cutlass_model::KindSwitches::ALL)
        );
        harness.state_mut().1.linkage_kinds = harness.state().0.linkage_kinds();
        harness.run_steps(2);
        // The library has an Audio tab too; the dialog is drawn last.
        harness.query_all_by_label("Audio").last().unwrap().click();
        harness.run_steps(1);
        harness.get_by_label("Done").click();
        harness.run_steps(2);
        let kinds = harness.state().0.linkage_kinds().unwrap();
        assert!(!kinds.audio);
        assert!(kinds.text && kinds.overlay);
        assert!(harness.state().1.linkage_kinds.is_none());
        drop(harness);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// The stickers shelf, the Licenses window and the project settings painted by the real
    /// renderer, as the window shows them. With `CUTLASS_RENDER_DIR` set
    /// the pictures are kept there to look at.
    #[test]
    fn the_stickers_shelf_licenses_and_project_settings_paint() {
        let temp = std::env::temp_dir().join(format!("cutlass-render-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        cutlass_store::Folder::new(cutlass_model::Draft::empty("draft"))
            .save(&temp)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Stickers;
        let mut harness = egui_kittest::Harness::builder()
            .with_size(Vec2::new(1280.0, 800.0))
            .wgpu()
            .build_ui_state(
                |ui, (shell, view): &mut (Shell, ViewState)| show(shell, ui, view, None),
                (shell, view),
            );
        // Stickers load and animate, so step frames rather than settle.
        harness.run_steps(8);
        let keep = std::env::var_os("CUTLASS_RENDER_DIR").map(std::path::PathBuf::from);
        let picture = |harness: &mut egui_kittest::Harness<'_, (Shell, ViewState)>, name: &str| {
            let image = harness.render().expect("the renderer draws the window");
            // Not one flat colour: something was drawn.
            let first = image.get_pixel(0, 0);
            assert!(
                image.pixels().any(|pixel| pixel != first),
                "{name} is blank"
            );
            if let Some(dir) = &keep {
                std::fs::create_dir_all(dir).unwrap();
                image.save(dir.join(format!("{name}.png"))).unwrap();
            }
        };
        picture(&mut harness, "stickers");
        // A Noto picture and a Noto Lottie that plays.
        for (query, name) in [("face", "stickers-noto"), ("animated", "stickers-animated")] {
            query.clone_into(&mut harness.state_mut().1.library_query);
            harness.run_steps(8);
            picture(&mut harness, name);
        }
        harness.state_mut().1.library_query.clear();
        // A search with no match: the empty shelf's line has a glyph the
        // test fonts lack.
        harness.state_mut().1.library_shelf = Shelf::Media;
        "zz".clone_into(&mut harness.state_mut().1.library_query);
        harness.run_steps(4);
        picture(&mut harness, "media");
        harness.state_mut().1.library_shelf = Shelf::Stickers;
        harness.state_mut().1.library_query.clear();
        harness.state_mut().1.licenses = Some(cutlass_licenses::notices());
        harness.run_steps(4);
        picture(&mut harness, "licenses");
        harness.state_mut().1.licenses = None;
        let settings = harness.state().0.project_settings();
        assert!(settings.is_some(), "an open project has settings");
        harness.state_mut().1.project_dialog = settings;
        harness.run_steps(4);
        picture(&mut harness, "project-settings");
        drop(harness);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// The inspector's tabs sit in a header band that stays put while the
    /// body under it scrolls, painted by the real renderer. With
    /// `CUTLASS_RENDER_DIR` set the pictures are kept there to look at.
    #[test]
    fn the_inspector_tabs_stay_put_while_its_body_scrolls() {
        use cutlass_model::{MaterialVideo, Segment, TimeRange};
        use egui_kittest::kittest::Queryable;
        let temp = std::env::temp_dir().join(format!("cutlass-render-tabs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo {
            width: 1920,
            height: 1080,
            ..MaterialVideo::new("movie")
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 2_000_000)));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        shell.select(Some(SegmentId::new("main")));
        let mut harness = egui_kittest::Harness::builder()
            .with_size(Vec2::new(1280.0, 800.0))
            .wgpu()
            .build_ui_state(
                |ui, (shell, view): &mut (Shell, ViewState)| show(shell, ui, view, None),
                (shell, ViewState::new()),
            );
        harness.run_steps(4);
        // The rightmost of each name is the inspector's.
        let at = |harness: &egui_kittest::Harness<'_, (Shell, ViewState)>, label: &str| {
            harness
                .query_all_by_label(label)
                .map(|node| node.rect())
                .max_by(|a, b| a.min.x.total_cmp(&b.min.x))
                .unwrap_or_else(|| panic!("no {label}"))
        };
        let keep = std::env::var_os("CUTLASS_RENDER_DIR").map(std::path::PathBuf::from);
        let picture = |harness: &mut egui_kittest::Harness<'_, (Shell, ViewState)>, name: &str| {
            let image = harness.render().expect("the renderer draws the window");
            if let Some(dir) = &keep {
                std::fs::create_dir_all(dir).unwrap();
                image.save(dir.join(format!("{name}.png"))).unwrap();
            }
            image
        };
        let tab = at(&harness, "Video");
        let row = at(&harness, "Position & Size");
        let before = picture(&mut harness, "inspector-top");
        // The header band, in the gutter left of the tabs, is not the
        // panel's colour under it.
        let x = (tab.min.x - 8.0) as u32;
        let header = before.get_pixel(x, tab.center().y as u32);
        let panel = before.get_pixel(x, (row.min.y - 6.0) as u32);
        assert_ne!(header, panel, "the header has no band of its own");
        harness.hover_at(row.center());
        for _ in 0..6 {
            harness.event(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: Vec2::new(0.0, -80.0),
                modifiers: egui::Modifiers::default(),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
            });
            harness.run_steps(2);
        }
        harness.run_steps(4);
        picture(&mut harness, "inspector-scrolled");
        assert_eq!(at(&harness, "Video"), tab, "the tabs moved");
        let scrolled = harness
            .query_all_by_label("Position & Size")
            .map(|node| node.rect())
            .max_by(|a, b| a.min.x.total_cmp(&b.min.x));
        assert!(
            scrolled.is_none_or(|moved| moved.min.y < row.min.y - 200.0),
            "the body did not scroll: {row:?} to {scrolled:?}"
        );
        drop(harness);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// A project with a title "AA" and a caption "AB BA" in the sample
    /// face, saved at `temp`, open.
    fn text_inspector_shell(temp: &std::path::Path) -> Shell {
        use cutlass_model::{MaterialText, Segment, TimeRange, Track, TrackType};
        let _ = std::fs::remove_dir_all(temp);
        std::fs::create_dir_all(temp).unwrap();
        let mut draft = cutlass_model::Draft::empty("draft");
        let mut title = MaterialText::new("title");
        title.content = "AA".to_owned();
        title.font_size = 160.0;
        draft.materials.texts.push(title);
        let mut caption = MaterialText::new("caption");
        caption.content = "AB BA".to_owned();
        caption.font_size = 7.0;
        caption.words.text = vec!["AB".to_owned(), "BA".to_owned()];
        caption.words.start_time = vec![0, 1_000];
        caption.words.end_time = vec![1_000, 2_000];
        draft.materials.texts.push(caption);
        for (id, material) in [("words", "title"), ("said", "caption")] {
            let mut track = Track::main(id);
            track.kind = TrackType::Text;
            // The caption sits on the main captions' row.
            if id == "said" {
                track.flag = cutlass_model::TrackFlag::TextSubtitle;
            }
            track
                .segments
                .push(Segment::new(id, material, TimeRange::new(0, 2_000_000)));
            draft.tracks.push(track);
        }
        cutlass_store::Folder::new(draft).save(temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(temp).unwrap();
        shell.use_faces(vec![(
            "sample.ttf",
            include_bytes!("../../../crates/font/fixtures/sample.ttf").to_vec(),
        )]);
        shell.keep_presets_file(temp.join("Presets").join("text.json"));
        shell
    }

    /// Nothing in the text or caption inspector draws past its content's
    /// right edge, 16 points in from the panel, where the scroll bar runs.
    #[test]
    fn the_text_inspector_keeps_its_right_margin() {
        let temp = std::env::temp_dir().join(format!("cutlass-text-margin-{}", std::process::id()));
        let mut shell = text_inspector_shell(&temp);
        let mut view = ViewState::new();
        view.library_shelf = Shelf::Stickers;
        let ctx = egui::Context::default();
        let mut over = Vec::new();
        for id in ["words", "said"] {
            shell.select(Some(SegmentId::new(id)));
            // A face gives the text a size, so its pose rows take keys, and
            // a key shows each keyed row's buttons.
            shell.set_text_face("sample.ttf").unwrap();
            shell
                .toggle_setting_key(crate::shell::KeyedSetting::TextColor)
                .unwrap();
            let mut last = Vec::new();
            for _ in 0..3 {
                // The Text tab; a caption's list comes before it.
                view.inspector.focus(inspector::Focus {
                    tab: 0,
                    page: 0,
                    section: None,
                });
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            Pos2::ZERO,
                            Vec2::new(1280.0, 2400.0),
                        )),
                        ..Default::default()
                    },
                    |ui| show(&mut shell, ui, &mut view, None),
                );
                output.textures_delta.clear();
                last.clear();
                // The inspector's clip: the whole panel, from its left
                // edge to the window's.
                let panel = output
                    .shapes
                    .iter()
                    .map(|clipped| clipped.clip_rect)
                    .filter(|clip| {
                        clip.min.x > 800.0 && clip.min.y < 100.0 && clip.height() > 400.0
                    })
                    .max_by(|a, b| a.width().total_cmp(&b.width()))
                    .unwrap();
                // Rows end 16 points in, clear of the scroll bar.
                let edge = panel.max.x - 16.0 + 0.5;
                for clipped in &output.shapes {
                    let clip = clipped.clip_rect;
                    if clip.min.x < panel.min.x - 0.5 || clip.max.y > panel.max.y + 0.5 {
                        continue;
                    }
                    let shape = clipped.shape.visual_bounding_rect();
                    if shape.is_positive() && shape.max.x > edge {
                        last.push(format!("{id}: {shape:?} past {edge}"));
                    }
                }
            }
            over.extend(last);
        }
        let _ = std::fs::remove_dir_all(&temp);
        assert!(over.is_empty(), "{over:#?}");
    }

    /// The text and caption inspector painted by the real renderer: the
    /// Basic page plain and styled, the caption's tabs, its Templates and its
    /// Bubble page.
    /// With `CUTLASS_RENDER_DIR` set the pictures are kept there to look at.
    #[test]
    fn the_text_and_caption_inspector_paint() {
        use crate::shell::KeyedSetting;
        use cutlass_model::SegmentId;
        let temp = std::env::temp_dir().join(format!("cutlass-text-render-{}", std::process::id()));
        let mut shell = text_inspector_shell(&temp);
        shell.select(Some(SegmentId::new("words")));
        let mut harness = egui_kittest::Harness::builder()
            .with_size(Vec2::new(1280.0, 1600.0))
            .wgpu()
            .build_ui_state(
                |ui, (shell, view): &mut (Shell, ViewState)| show(shell, ui, view, None),
                (shell, {
                    // The empty media shelf's hint has a glyph the test
                    // fonts lack.
                    let mut view = ViewState::new();
                    view.library_shelf = Shelf::Stickers;
                    view
                }),
            );
        harness.run_steps(4);
        let keep = std::env::var_os("CUTLASS_RENDER_DIR").map(std::path::PathBuf::from);
        let picture = |harness: &mut egui_kittest::Harness<'_, (Shell, ViewState)>, name: &str| {
            // Enough frames for a page's tile pictures.
            harness.run_steps(16);
            let image = harness.render().expect("the renderer draws the window");
            let first = image.get_pixel(0, 0);
            assert!(
                image.pixels().any(|pixel| pixel != first),
                "{name} is blank"
            );
            if let Some(dir) = &keep {
                std::fs::create_dir_all(dir).unwrap();
                image.save(dir.join(format!("{name}.png"))).unwrap();
            }
        };
        picture(&mut harness, "title-basic");
        {
            let shell = &mut harness.state_mut().0;
            shell
                .set_text_pattern(cutlass_session::TextPattern {
                    bold: true,
                    italic: true,
                    underline: true,
                })
                .unwrap();
            let pop = cutlass_editor::TEXT_PRESETS
                .iter()
                .position(|preset| preset.name == "Pop")
                .unwrap();
            shell.set_text_preset(pop).unwrap();
            shell.toggle_setting_key(KeyedSetting::TextColor).unwrap();
            shell.save_text_preset().unwrap();
        }
        picture(&mut harness, "title-styled");
        harness.state_mut().0.set_text_align(4).unwrap();
        picture(&mut harness, "title-vertical");
        harness.state_mut().0.select(Some(SegmentId::new("said")));
        harness.run_steps(2);
        // Focus counts the title's tabs; a caption's list comes first.
        harness.state_mut().1.inspector.focus(inspector::Focus {
            tab: 0,
            page: 0,
            section: None,
        });
        picture(&mut harness, "caption-basic");
        harness.state_mut().1.inspector.focus(inspector::Focus {
            tab: 0,
            page: 1,
            section: None,
        });
        picture(&mut harness, "caption-templates");
        harness.state_mut().1.inspector.focus(inspector::Focus {
            tab: 0,
            page: 2,
            section: None,
        });
        picture(&mut harness, "caption-bubble");
        harness.state_mut().1.inspector.focus(inspector::Focus {
            tab: 0,
            page: 3,
            section: None,
        });
        picture(&mut harness, "caption-effects");
        drop(harness);
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_narrow_transport_drops_guides_then_zoom() {
        let room = |width| {
            let room = transport_room(width, true);
            (room.zoom, room.guides, room.quality)
        };
        assert_eq!(room(700.0), (true, true, true));
        assert_eq!(room(620.0), (true, true, false));
        assert_eq!(room(560.0), (true, false, false));
        assert_eq!(room(450.0), (false, true, false));
        assert_eq!(room(tokens::PREVIEW_MIN_WIDTH), (false, false, false));
        // The default panels in a 1280 window leave room to zoom.
        let preview =
            1280.0 - tokens::LIBRARY_WIDTH - tokens::INSPECTOR_WIDTH - tokens::SPLIT * 2.0;
        assert!(transport_room(preview, true).zoom, "no zoom at {preview}");
        // With no project the transport has neither.
        let idle = transport_room(900.0, false);
        assert!(!idle.zoom && !idle.guides && !idle.quality);
        // The narrowest preview still keeps play clear of the three round
        // buttons it shows.
        let half = tokens::PREVIEW_MIN_WIDTH / 2.0;
        assert!(half - PLAY_RADIUS - TRANSPORT_GAP >= ROUND_REACH + 80.0);
    }
}
