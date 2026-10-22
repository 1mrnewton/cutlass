//! The library rail, the project bin, and the other shelves. `docs/library.md`.

use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Rect, ScrollArea, Sense, Stroke, UiBuilder, pos2,
    vec2,
};

use cutlass_core::Timestamp;
use cutlass_project::{
    Asset, AssetId, AssetKind, Channels, Command, Error, Project, Segment, Speed, TimeRange,
    TrackId, TrackKind,
};

use crate::icons::{self, Icon};
use crate::media_bin::{Bin, CardId, FileInfo, Sort};
use crate::theme;
use crate::timeline::file_name;
use crate::words::{self, Phrase};

const RAIL: f32 = 76.0;
const CARD_W: f32 = 144.0;
const CARD_H: f32 = 96.0;
const CARD_GAP: f32 = 8.0;

/// One shelf on the rail. Media is the project bin. Audio lists sound files.
/// The other shelves have no project items yet.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Shelf {
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

    const fn phrase(self) -> crate::words::Phrase {
        match self {
            Self::Media => crate::words::Phrase::Media,
            Self::Audio => crate::words::Phrase::Audio,
            Self::Text => crate::words::Phrase::Text,
            Self::Stickers => crate::words::Phrase::Stickers,
            Self::Effects => crate::words::Phrase::Effects,
            Self::Transitions => crate::words::Phrase::Transitions,
            Self::Filters => crate::words::Phrase::Filters,
            Self::Adjustments => crate::words::Phrase::Adjustments,
            Self::Templates => crate::words::Phrase::Templates,
        }
    }

    const fn icon(self) -> Icon {
        match self {
            Self::Media => Icon::Media,
            Self::Audio => Icon::Audio,
            Self::Text => Icon::Text,
            Self::Stickers => Icon::Stickers,
            Self::Effects => Icon::Effects,
            Self::Transitions => Icon::Transitions,
            Self::Filters => Icon::Filters,
            Self::Adjustments => Icon::Adjustments,
            Self::Templates => Icon::Templates,
        }
    }
}

/// Project is the bin. Free stock is the other side of the media shelf.
/// The choice is view state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MediaSource {
    Project,
    Stock,
}

/// Import opens the picker. Commands place a library clip.
pub struct ShelfAction {
    pub import: bool,
    pub commands: Vec<Command>,
}

pub fn show(
    ui: &mut egui::Ui,
    project: &Project,
    shelf: &mut Shelf,
    source: &mut MediaSource,
    bin: &mut Bin,
) -> ShelfAction {
    crate::tip::region(ui, words::of(ui, Phrase::RegionLibrary));
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let rail = Rect::from_min_size(full.min, vec2(RAIL, full.height()));
    let hairline = Rect::from_min_size(rail.right_top(), vec2(1.0, full.height()));
    ui.painter().rect_filled(hairline, 0.0, theme::GAP);
    paint_rail(ui, rail, shelf);
    let assets = Rect::from_min_max(hairline.right_top(), full.max).shrink(12.0);
    let (import, commands) = match *shelf {
        Shelf::Media => paint_media(ui, assets, project, bin, source),
        Shelf::Audio => (false, paint_audio(ui, assets, project, bin)),
        Shelf::Text
        | Shelf::Stickers
        | Shelf::Effects
        | Shelf::Transitions
        | Shelf::Filters
        | Shelf::Adjustments
        | Shelf::Templates => {
            paint_empty(ui, assets);
            (false, Vec::new())
        }
    };
    ShelfAction { import, commands }
}

fn paint_rail(ui: &mut egui::Ui, rect: Rect, shelf: &mut Shelf) {
    let inner = Rect::from_min_max(
        pos2(rect.left() + 8.0, rect.top()),
        pos2(rect.right() - 8.0, rect.bottom()),
    );
    let mut column = ui.new_child(UiBuilder::new().max_rect(inner));
    ScrollArea::vertical()
        .id_salt("library-rail")
        .show(&mut column, |ui| {
            ui.add_space(8.0);
            ui.spacing_mut().item_spacing.y = 2.0;
            for tool in Shelf::ALL {
                if rail_button(ui, tool, *shelf == tool).clicked() {
                    *shelf = tool;
                }
            }
        });
}

fn rail_button(ui: &mut egui::Ui, shelf: Shelf, active: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 44.0), Sense::click());
    let (fill, color) = theme::face(
        if active {
            theme::ACTIVE
        } else {
            Color32::TRANSPARENT
        },
        if active { theme::INK } else { theme::FAINT },
        theme::hot(&response),
    );
    if fill.a() > 0 {
        ui.painter().rect_filled(rect, 10.0, fill);
    }
    icons::paint(ui.painter(), rect.center(), shelf.icon(), 20.0, color);
    crate::tip::show_radio(&response, words::of(ui, shelf.phrase()), active);
    response
}

fn paint_media(
    ui: &mut egui::Ui,
    rect: Rect,
    project: &Project,
    bin: &mut Bin,
    source: &mut MediaSource,
) -> (bool, Vec<Command>) {
    let switch = Rect::from_min_size(rect.min, vec2(rect.width(), f32::min(32.0, rect.height())));
    paint_source(ui, switch, source);
    let top = switch.bottom() + 8.0;
    if top >= rect.bottom() {
        return (false, Vec::new());
    }
    let rest = Rect::from_min_max(pos2(rect.left(), top), rect.max);
    match *source {
        MediaSource::Project => paint_bin(ui, rest, project, bin),
        MediaSource::Stock => {
            paint_empty(ui, rest);
            (false, Vec::new())
        }
    }
}

fn paint_source(ui: &mut egui::Ui, rect: Rect, source: &mut MediaSource) {
    ui.painter().rect_filled(rect, 10.0, theme::BAR);
    let inner = rect.shrink(3.0);
    let width = ((inner.width() - 4.0) / 2.0).max(0.0);
    let project = Rect::from_min_size(inner.min, vec2(width, inner.height()));
    let stock = Rect::from_min_size(
        pos2(inner.right() - width, inner.top()),
        vec2(width, inner.height()),
    );
    if source_button(
        ui,
        project,
        "media-project",
        words::of(ui, Phrase::ProjectMedia),
        *source == MediaSource::Project,
    )
    .clicked()
    {
        *source = MediaSource::Project;
    }
    if source_button(
        ui,
        stock,
        "media-stock",
        words::of(ui, Phrase::FreeStock),
        *source == MediaSource::Stock,
    )
    .clicked()
    {
        *source = MediaSource::Stock;
    }
}

fn source_button(
    ui: &mut egui::Ui,
    rect: Rect,
    salt: &str,
    label: &str,
    active: bool,
) -> egui::Response {
    let response = ui.interact(rect, ui.scope_id().with(salt), Sense::click());
    let (fill, color) = theme::face(
        if active {
            theme::ACTIVE
        } else {
            Color32::TRANSPARENT
        },
        if active { theme::INK } else { theme::MUTED },
        theme::hot(&response),
    );
    if fill.a() > 0 {
        ui.painter().rect_filled(rect, 8.0, fill);
    }
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::new(12.0, FontFamily::Proportional),
        color,
    );
    crate::tip::show_radio(&response, label, active);
    response
}

fn paint_bin(
    ui: &mut egui::Ui,
    rect: Rect,
    project: &Project,
    bin: &mut Bin,
) -> (bool, Vec<Command>) {
    let row = f32::min(32.0, rect.height());
    let label = words::of(ui, Phrase::Import);
    let width = words::control_width(
        ui,
        label,
        FontId::new(12.0, FontFamily::Proportional),
        20.0,
        72.0,
    );
    let button = Rect::from_min_size(pos2(rect.right() - width, rect.top()), vec2(width, row));
    let path = Rect::from_min_max(
        rect.min,
        pos2((button.left() - 8.0).max(rect.left()), rect.top() + row),
    );
    if path.width() > 0.0 {
        paint_path(ui, path, bin);
    }
    let clicked = import_button(ui, button, label);
    let top = path.bottom() + 8.0;
    let commands = if top < rect.bottom() {
        let grid = Rect::from_min_max(pos2(rect.left(), top), rect.max);
        paint_grid(ui, grid, project, bin)
    } else {
        Vec::new()
    };
    paint_ghost(ui, project);
    (clicked, commands)
}

fn import_button(ui: &mut egui::Ui, rect: Rect, label: &str) -> bool {
    theme::with_face(ui, theme::BAR, theme::INK, |ui| {
        ui.put(
            rect,
            egui::Button::new(egui::RichText::new(label).size(12.0)).corner_radius(8),
        )
    })
    .clicked()
}

fn paint_path(ui: &mut egui::Ui, rect: Rect, bin: &mut Bin) {
    let trail = bin.trail();
    let open = bin.open();
    let mut row = ui.new_child(UiBuilder::new().max_rect(rect));
    ScrollArea::horizontal()
        .id_salt("library-path")
        .auto_shrink([false, true])
        .show(&mut row, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.add_space(2.0);
                path_crumb(ui, bin, open, None, words::of(ui, Phrase::All));
                for id in trail {
                    ui.label(egui::RichText::new("/").size(12.0).color(theme::FAINT));
                    let name = bin.folder_name(id).unwrap_or("").to_owned();
                    path_crumb(ui, bin, open, Some(id), &name);
                }
            });
        });
}

fn path_crumb(
    ui: &mut egui::Ui,
    bin: &mut Bin,
    open: Option<u64>,
    place: Option<u64>,
    label: &str,
) {
    let current = open == place;
    let fill = if current {
        theme::ACTIVE
    } else {
        Color32::TRANSPARENT
    };
    let ink = if current { theme::INK } else { theme::MUTED };
    let response = theme::with_face(ui, fill, ink, |ui| {
        ui.add(egui::Button::new(egui::RichText::new(label).size(12.0)))
    });
    if response.clicked() {
        bin.open_place(place);
    }
    let hovering = response
        .dnd_hover_payload::<AssetId>()
        .is_some_and(|drag| bin.place_of(*drag) != place);
    if hovering {
        ui.painter().rect_stroke(
            response.rect,
            6.0,
            Stroke::new(1.5, theme::EXPORT),
            egui::StrokeKind::Inside,
        );
    }
    if let Some(drag) = response.dnd_release_payload::<AssetId>()
        && bin.place_of(*drag) != place
    {
        bin.move_file(*drag, place);
    }
}

fn paint_audio(ui: &mut egui::Ui, rect: Rect, project: &Project, bin: &mut Bin) -> Vec<Command> {
    let order = {
        let infos = audio_infos(project);
        bin.flat(&infos)
    };
    paint_cards(ui, rect, project, bin, &order, "library-audio", false)
}

fn paint_empty(ui: &mut egui::Ui, rect: Rect) {
    paint_empty_word(ui, rect);
}

fn paint_grid(ui: &mut egui::Ui, rect: Rect, project: &Project, bin: &mut Bin) -> Vec<Command> {
    let order = {
        let infos = file_infos(project);
        bin.visible(&infos)
    };
    paint_cards(ui, rect, project, bin, &order, "library-cards", true)
}

fn paint_cards(
    ui: &mut egui::Ui,
    rect: Rect,
    project: &Project,
    bin: &mut Bin,
    order: &[CardId],
    salt: &'static str,
    media: bool,
) -> Vec<Command> {
    let mut commands = Vec::new();
    let mut column = ui.new_child(UiBuilder::new().max_rect(rect));
    ScrollArea::vertical()
        .id_salt(salt)
        .auto_shrink([false, false])
        .show(&mut column, |ui| {
            let width = ui.available_width();
            let cols = columns(width);
            let rows = order.len().div_ceil(cols).max(1);
            let height = (rows as f32 * (CARD_H + CARD_GAP)).max(ui.available_height());
            let (area, background) =
                ui.allocate_exact_size(vec2(width, height), Sense::click_and_drag());
            let mut picked = false;
            let mut rects = Vec::with_capacity(order.len());
            for (index, id) in order.iter().copied().enumerate() {
                let card =
                    Rect::from_min_size(card_origin(area.min, index, cols), vec2(CARD_W, CARD_H));
                rects.push((id, card));
                match id {
                    CardId::Folder(folder) => {
                        handle_folder(ui, bin, order, card, folder, &mut picked);
                    }
                    CardId::File(asset) => {
                        let (response, command, plus_hot) =
                            handle_file(ui, project, bin, order, card, asset, &mut picked);
                        if let Some(command) = command {
                            commands.push(command);
                        }
                        if media && !plus_hot {
                            let modifiers = ui.input(|input| input.modifiers);
                            if !modifiers.shift && !modifiers.command && !modifiers.ctrl {
                                response.dnd_set_drag_payload(asset);
                            }
                        }
                    }
                }
            }
            let boxed = handle_box(ui, bin, &background, &rects);
            if background.clicked() && !picked && !boxed {
                bin.clear();
            }
            if !media && order.is_empty() {
                paint_empty_word(ui, area);
            }
            crate::menu::context_menu(&background, |ui| empty_menu(ui, bin, media));
        });
    commands
}

fn paint_empty_word(ui: &mut egui::Ui, rect: Rect) {
    let label = words::of(ui, Phrase::Empty);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::new(13.0, FontFamily::Proportional),
        theme::MUTED,
    );
    crate::tip::read(
        ui,
        Rect::from_center_size(rect.center(), vec2(96.0, 24.0)),
        ui.scope_id().with("shelf-empty"),
        label,
    );
}

fn file_infos(project: &Project) -> Vec<FileInfo<'_>> {
    listed(project, |_| true)
}

fn audio_infos(project: &Project) -> Vec<FileInfo<'_>> {
    listed(project, |asset| matches!(asset.kind(), AssetKind::Audio))
}

fn listed(project: &Project, keep: impl Fn(&Asset) -> bool) -> Vec<FileInfo<'_>> {
    project
        .assets()
        .iter()
        .filter(|asset| keep(asset))
        .map(|asset| FileInfo {
            id: asset.id(),
            name: file_name(asset.source()),
            duration_us: asset
                .duration()
                .and_then(|time| crate::media_bin::duration_us(time.value, time.timescale.get())),
        })
        .collect()
}

fn handle_box(
    ui: &mut egui::Ui,
    bin: &mut Bin,
    background: &egui::Response,
    cards: &[(CardId, Rect)],
) -> bool {
    let primary = ui.input(|input| {
        input.pointer.primary_down() || input.pointer.button_released(egui::PointerButton::Primary)
    });
    if background.drag_started()
        && primary
        && let Some(pos) = background.interact_pointer_pos()
    {
        bin.begin_box(pos, ui.input(|input| input.modifiers.shift));
    }
    let pointer = background
        .interact_pointer_pos()
        .or_else(|| ui.input(|input| input.pointer.latest_pos()));
    if let Some(pos) = pointer
        && let Some(rect) = bin.box_rect(pos)
    {
        let hits = crate::media_bin::cards_touching(cards, rect);
        bin.apply_box(&hits);
        ui.painter()
            .rect_filled(rect, 0.0, Color32::from_white_alpha(28));
        ui.painter().rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, Color32::WHITE),
            egui::StrokeKind::Outside,
        );
    }
    if background.drag_stopped() {
        bin.finish_box()
    } else {
        false
    }
}

fn handle_folder(
    ui: &mut egui::Ui,
    bin: &mut Bin,
    order: &[CardId],
    rect: Rect,
    folder: u64,
    picked: &mut bool,
) {
    let name = bin.folder_name(folder).unwrap_or("").to_owned();
    let id = CardId::Folder(folder);
    let response = ui.interact(rect, ui.scope_id().with(id), Sense::click());
    if response.clicked() {
        *picked = true;
        let modifiers = ui.input(|input| input.modifiers);
        let toggle = modifiers.command || modifiers.ctrl;
        if modifiers.shift || toggle {
            bin.choose(order, id, modifiers.shift, toggle);
        } else if !bin.is_renaming(folder) {
            bin.open_folder(folder);
        }
    }
    if response.secondary_clicked() && !bin.is_selected(id) {
        bin.select_only(id);
    }
    let drop_here = response
        .dnd_hover_payload::<AssetId>()
        .is_some_and(|drag| bin.place_of(*drag) != Some(folder));
    if let Some(drag) = response.dnd_release_payload::<AssetId>()
        && bin.place_of(*drag) != Some(folder)
    {
        bin.move_file(*drag, Some(folder));
    }
    crate::menu::context_menu(&response, |ui| folder_menu(ui, bin, folder));
    crate::tip::show_selected(&response, &name, bin.is_selected(id));
    paint_folder(
        ui,
        rect,
        &name,
        bin.is_selected(id),
        drop_here,
        bin.is_renaming(folder),
    );
    if bin.is_renaming(folder) {
        let field = Rect::from_min_max(
            pos2(rect.left() + 8.0, rect.bottom() - 28.0),
            pos2(rect.right() - 8.0, rect.bottom() - 6.0),
        );
        rename_field(ui, bin, folder, field);
    }
}

fn handle_file(
    ui: &mut egui::Ui,
    project: &Project,
    bin: &mut Bin,
    order: &[CardId],
    rect: Rect,
    asset_id: AssetId,
    picked: &mut bool,
) -> (egui::Response, Option<Command>, bool) {
    let Some(asset) = project.asset(asset_id) else {
        return (
            ui.interact(
                rect,
                ui.scope_id().with(CardId::File(asset_id)),
                Sense::hover(),
            ),
            None,
            false,
        );
    };
    let name = file_name(asset.source()).to_owned();
    let placed = on_timeline(project, asset_id);
    let id = CardId::File(asset_id);
    let response = ui.interact(rect, ui.scope_id().with(id), Sense::click_and_drag());
    let plus = offer_plus(ui, project, rect, asset, asset_id);
    let plus_clicked = plus.response.as_ref().is_some_and(|plus| plus.clicked());
    let plus_hot = plus.response.as_ref().is_some_and(|plus| plus.hovered());
    if plus_clicked {
        *picked = true;
    }
    if response.clicked() && !plus_clicked {
        *picked = true;
        let modifiers = ui.input(|input| input.modifiers);
        bin.choose(
            order,
            id,
            modifiers.shift,
            modifiers.command || modifiers.ctrl,
        );
    }
    if response.secondary_clicked() && !bin.is_selected(id) {
        bin.select_only(id);
    }
    let source = asset.source().to_owned();
    crate::menu::context_menu(&response, |ui| file_menu(ui, bin, asset_id, &source));
    crate::tip::show_selected(&response, &name, bin.is_selected(id));
    paint_picture(
        ui.painter(),
        rect,
        asset,
        placed,
        words::of(ui, Phrase::OnTimeline),
    );
    if bin.is_selected(id) {
        selected_edge(ui, rect);
    }
    if let Some(plus) = plus.response {
        paint_plus(ui, plus.rect, theme::hot(&plus));
        crate::tip::show(&plus, words::of(ui, Phrase::AddToTimeline));
    }
    (response, plus.command, plus_hot)
}

struct PlusHit {
    response: Option<egui::Response>,
    command: Option<Command>,
}

fn offer_plus(
    ui: &mut egui::Ui,
    project: &Project,
    rect: Rect,
    asset: &Asset,
    asset_id: AssetId,
) -> PlusHit {
    if !pointer_inside(ui, rect) || dragging(ui) || !can_place(asset) {
        return PlusHit {
            response: None,
            command: None,
        };
    }
    let response = ui.interact(
        plus_rect(rect),
        ui.scope_id().with(("card-plus", asset_id)),
        Sense::click(),
    );
    let command = if response.clicked() {
        place_at_end(project, asset_id).ok()
    } else {
        None
    };
    PlusHit {
        response: Some(response),
        command,
    }
}

fn pointer_inside(ui: &egui::Ui, rect: Rect) -> bool {
    ui.input(|input| input.pointer.hover_pos())
        .is_some_and(|pos| rect.contains(pos))
}

fn dragging(ui: &egui::Ui) -> bool {
    ui.ctx().dragged_id().is_some()
}

fn can_place(asset: &Asset) -> bool {
    asset.duration().is_some() && matches!(asset.kind(), AssetKind::Video { .. } | AssetKind::Audio)
}

const PLUS: f32 = 22.0;

fn plus_rect(card: Rect) -> Rect {
    Rect::from_min_size(
        pos2(card.right() - 8.0 - PLUS, card.top() + 8.0),
        vec2(PLUS, PLUS),
    )
}

fn paint_plus(ui: &egui::Ui, rect: Rect, hot: bool) {
    let (fill, ink) = theme::face(Color32::from_black_alpha(170), Color32::WHITE, hot);
    ui.painter().rect_filled(rect, PLUS / 2.0, fill);
    icons::paint(ui.painter(), rect.center(), Icon::Plus, 14.0, ink);
}

fn rename_field(ui: &mut egui::Ui, bin: &mut Bin, id: u64, rect: Rect) {
    let response = {
        let text = bin.rename_text_mut();
        ui.put(
            rect,
            egui::TextEdit::singleline(text)
                .id(ui.scope_id().with(("folder-rename", id)))
                .font(FontId::new(12.0, FontFamily::Proportional))
                .text_color(Color32::WHITE)
                .background_color(theme::ACTIVE),
        )
    };
    if bin.focus_rename(id) {
        response.request_focus();
    }
    let escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
    if escape && (response.has_focus() || response.lost_focus()) {
        bin.cancel_rename();
    } else if response.lost_focus() {
        bin.commit_rename();
    }
}

fn empty_menu(ui: &mut egui::Ui, bin: &mut Bin, folders: bool) {
    if folders && crate::menu::row(ui, words::of(ui, Phrase::NewFolder), true) {
        bin.add_folder(words::of(ui, Phrase::UntitledFolder));
    }
    sort_menu(ui, bin);
}

fn folder_menu(ui: &mut egui::Ui, bin: &mut Bin, id: u64) {
    if crate::menu::row(ui, words::of(ui, Phrase::Delete), true) {
        bin.delete_folder(id);
    }
    crate::menu::row(ui, words::of(ui, Phrase::OpenLocation), false);
    crate::menu::row(ui, words::of(ui, Phrase::NewFolderWith), false);
    if crate::menu::row(ui, words::of(ui, Phrase::Rename), true) {
        bin.begin_rename(id);
    }
    crate::menu::row(ui, words::of(ui, Phrase::LinkMedia), false);
    if crate::menu::row(ui, words::of(ui, Phrase::ExportClips), true) {
        crate::timeline::show_note(ui, words::of(ui, Phrase::ExportWritesNothing));
    }
}

fn file_menu(ui: &mut egui::Ui, bin: &mut Bin, id: AssetId, source: &str) {
    if crate::menu::row(ui, words::of(ui, Phrase::OpenLocation), true) {
        crate::timeline::show_note(ui, words::of(ui, Phrase::DoesNotOpenFile));
    }
    if crate::menu::row(ui, words::of(ui, Phrase::Delete), true) {
        bin.hide_file(id);
    }
    if crate::menu::row(ui, words::of(ui, Phrase::NewFolderWith), true) {
        bin.add_folder_with(id, words::of(ui, Phrase::UntitledFolder));
    }
    if source_missing(source) {
        if crate::menu::row(ui, words::of(ui, Phrase::LinkMedia), true) {
            ask_relink(ui, id);
        }
    } else {
        crate::menu::row(ui, words::of(ui, Phrase::LinkMedia), false);
    }
    if crate::menu::row_badge(
        ui,
        words::of(ui, Phrase::SplitScenes),
        words::of(ui, Phrase::Free),
        true,
    ) {
        crate::timeline::show_note(ui, words::of(ui, Phrase::SplitScenes));
    }
    if crate::menu::row(ui, words::of(ui, Phrase::ExportClips), true) {
        crate::timeline::show_note(ui, words::of(ui, Phrase::ExportWritesNothing));
    }
}

fn sort_menu(ui: &mut egui::Ui, bin: &mut Bin) {
    crate::menu::submenu(ui, words::of(ui, Phrase::Sort), |ui| {
        if crate::menu::row(ui, words::of(ui, Phrase::Name), true) {
            bin.set_sort(Sort::Name);
            crate::timeline::show_note(ui, words::of(ui, Phrase::SortedByName));
        }
        if crate::menu::row(ui, words::of(ui, Phrase::Duration), true) {
            bin.set_sort(Sort::Duration);
            crate::timeline::show_note(ui, words::of(ui, Phrase::SortedByDuration));
        }
    });
}

/// A path that is not a file. A folder and a content URI count as missing.
fn source_missing(source: &str) -> bool {
    !std::path::Path::new(source).is_file()
}

fn relink_id() -> egui::Id {
    egui::Id::unique("library-relink")
}

fn ask_relink(ui: &egui::Ui, id: AssetId) {
    ui.data_mut(|data| data.insert_temp(relink_id(), id));
}

/// The file menu asked to relink this asset. The ask is consumed.
pub(crate) fn take_relink(ctx: &egui::Context) -> Option<AssetId> {
    let id = relink_id();
    let value = ctx.data(|data| data.get_temp::<AssetId>(id));
    if value.is_some() {
        ctx.data_mut(|data| data.remove::<AssetId>(id));
    }
    value
}

fn paint_folder(
    ui: &egui::Ui,
    rect: Rect,
    name: &str,
    selected: bool,
    drop_here: bool,
    renaming: bool,
) {
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 12.0, theme::BAR);
    icons::paint(
        &painter,
        pos2(rect.center().x, rect.top() + 36.0),
        Icon::Folder,
        28.0,
        theme::INK,
    );
    if !renaming {
        painter.text(
            rect.left_bottom() + vec2(8.0, -8.0),
            Align2::LEFT_BOTTOM,
            name,
            FontId::new(12.0, FontFamily::Proportional),
            theme::INK,
        );
    }
    if selected {
        selected_edge(ui, rect);
    }
    if drop_here {
        ui.painter().rect_stroke(
            rect,
            12.0,
            Stroke::new(1.5, theme::EXPORT),
            egui::StrokeKind::Outside,
        );
    }
}

fn paint_picture(
    painter: &egui::Painter,
    rect: Rect,
    asset: &Asset,
    placed: bool,
    on_timeline: &str,
) {
    let swatch = swatch(asset.kind());
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 12.0, swatch);
    let ink = ink_on(swatch);
    painter.text(
        rect.left_bottom()
            + vec2(
                8.0,
                if asset.duration().is_some() {
                    -22.0
                } else {
                    -8.0
                },
            ),
        Align2::LEFT_BOTTOM,
        file_name(asset.source()),
        FontId::new(12.0, FontFamily::Proportional),
        ink,
    );
    if let Some(duration) = asset.duration() {
        painter.text(
            rect.left_bottom() + vec2(8.0, -8.0),
            Align2::LEFT_BOTTOM,
            card_time(duration.value, duration.timescale.get()),
            FontId::new(11.0, FontFamily::Proportional),
            Color32::from_white_alpha(220),
        );
    }
    if placed {
        painter.text(
            rect.left_top() + vec2(8.0, 8.0),
            Align2::LEFT_TOP,
            on_timeline,
            FontId::new(10.0, FontFamily::Proportional),
            ink,
        );
    }
}

fn selected_edge(ui: &egui::Ui, rect: Rect) {
    ui.painter().rect_stroke(
        rect,
        12.0,
        Stroke::new(1.5, Color32::WHITE),
        egui::StrokeKind::Inside,
    );
}

fn paint_ghost(ui: &egui::Ui, project: &Project) {
    let Some(id) = egui::DragAndDrop::payload::<AssetId>(ui.ctx()) else {
        return;
    };
    let Some(asset) = project.asset(*id) else {
        return;
    };
    let Some(pos) = ui.input(|input| input.pointer.hover_pos()) else {
        return;
    };
    let rect = Rect::from_center_size(pos, vec2(CARD_W, CARD_H));
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::unique("bin-ghost"),
    ));
    paint_picture(
        &painter,
        rect,
        asset,
        on_timeline(project, *id),
        words::of(ui, Phrase::OnTimeline),
    );
}

/// How many 144px cards fit. One column does not stretch the card.
fn columns(width: f32) -> usize {
    if width <= 0.0 {
        return 1;
    }
    let count = ((width + CARD_GAP) / (CARD_W + CARD_GAP)).floor();
    if count < 1.0 { 1 } else { count as usize }
}

fn card_origin(area: egui::Pos2, index: usize, cols: usize) -> egui::Pos2 {
    let cols = cols.max(1);
    area + vec2(
        (index % cols) as f32 * (CARD_W + CARD_GAP),
        (index / cols) as f32 * (CARD_H + CARD_GAP),
    )
}

fn swatch(kind: AssetKind) -> Color32 {
    match kind {
        AssetKind::Video { .. } => theme::VIDEO,
        AssetKind::Audio => theme::AUDIO,
        AssetKind::Image { .. } => theme::BAR,
    }
}

fn ink_on(swatch: Color32) -> Color32 {
    let [red, green, blue, _] = swatch.to_array();
    if u16::from(red) + u16::from(green) + u16::from(blue) > 520 {
        theme::GAP
    } else {
        Color32::WHITE
    }
}

/// `M:SS.t`, floored to a tenth of a second.
pub(crate) fn card_time(value: i64, timescale: u32) -> String {
    let scale = i128::from(timescale);
    let tenths = i128::from(value).max(0) * 10 / scale;
    let Ok(tenths) = i64::try_from(tenths) else {
        return "0:00.0".into();
    };
    let mins = tenths / 600;
    let secs = (tenths / 10) % 60;
    let tenth = tenths % 10;
    format!("{mins}:{secs:02}.{tenth}")
}

fn on_timeline(project: &Project, id: AssetId) -> bool {
    project.timeline().tracks().iter().any(|track| {
        track
            .segments()
            .iter()
            .any(|segment| segment.asset_id() == Some(id))
    })
}

/// One clip covering the whole asset, starting after the last clip on its lane.
fn place_at_end(project: &Project, id: AssetId) -> Result<Command, Error> {
    let Some(asset) = project.asset(id) else {
        return Err(Error::UnknownAsset(id));
    };
    let channels = match asset.kind() {
        AssetKind::Video {
            has_audio: true, ..
        } => Channels::Both,
        AssetKind::Video { .. } => Channels::Picture,
        AssetKind::Audio => Channels::Sound,
        AssetKind::Image { .. } => return Err(Error::ImageSource),
    };
    let Some(duration) = asset.duration() else {
        return Err(Error::ZeroDuration);
    };
    let source = TimeRange::new(Timestamp::new(0, duration.timescale), duration)?;
    let track = match channels {
        Channels::Picture | Channels::Both => Some(project.timeline().main_track().id()),
        Channels::Sound => sound_track(project),
    };
    let start = track.map(|id| lane_end(project, id)).unwrap_or(0);
    let segment = Segment::media(asset, source, start, channels, Speed::ONE)?;
    Ok(Command::Place { track, segment })
}

fn sound_track(project: &Project) -> Option<TrackId> {
    project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.kind() == TrackKind::Audio)
        .map(|track| track.id())
}

fn lane_end(project: &Project, id: TrackId) -> i64 {
    project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.id() == id)
        .and_then(|track| {
            track
                .segments()
                .iter()
                .map(|segment| segment.timeline().end().value)
                .max()
        })
        .unwrap_or(0)
        .max(0)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, Error, History, Outcome, Project, ProjectId, Segment, Speed,
        TimeRange, TrackKind,
    };
    use eframe::egui::pos2;

    use super::{card_origin, card_time, columns, on_timeline, swatch};
    use crate::media_bin::CardId;
    use crate::theme;

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "Untitled")
    }

    fn node_center(tree: &eframe::egui::accesskit::TreeUpdate, label: &str) -> eframe::egui::Pos2 {
        let (_, node) = tree
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some(label))
            .unwrap_or_else(|| panic!("missing {label}"));
        let bounds = node
            .bounds()
            .unwrap_or_else(|| panic!("{label} has no bounds"));
        eframe::egui::pos2(
            ((bounds.x0 + bounds.x1) / 2.0) as f32,
            ((bounds.y0 + bounds.y1) / 2.0) as f32,
        )
    }

    #[test]
    fn one_column_until_a_second_card_fits() {
        assert_eq!(columns(0.0), 1);
        assert_eq!(columns(144.0), 1);
        assert_eq!(columns(295.0), 1);
        assert_eq!(columns(296.0), 2);
    }

    #[test]
    fn cards_keep_their_size_as_the_grid_grows() {
        let first = card_origin(pos2(10.0, 20.0), 0, 2);
        let second = card_origin(pos2(10.0, 20.0), 1, 2);
        let wrapped = card_origin(pos2(10.0, 20.0), 2, 2);
        assert!((first.x - 10.0).abs() < f32::EPSILON);
        assert!((second.x - 162.0).abs() < f32::EPSILON);
        assert!((second.y - 20.0).abs() < f32::EPSILON);
        assert!((wrapped.x - 10.0).abs() < f32::EPSILON);
        assert!((wrapped.y - 124.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_second_and_a_half_is_minutes_seconds_and_a_tenth() {
        let sequence = Timestamp::new(1_500_000, NonZeroU32::new(1_000_000).expect("scale"));
        assert_eq!(
            card_time(sequence.value, sequence.timescale.get()),
            "0:01.5"
        );
        let samples = Timestamp::new(72_000, NonZeroU32::new(48_000).expect("scale"));
        assert_eq!(card_time(samples.value, samples.timescale.get()), "0:01.5");
        let minute = Timestamp::new(90_000_000, NonZeroU32::new(1_000_000).expect("scale"));
        assert_eq!(card_time(minute.value, minute.timescale.get()), "1:30.0");
    }

    #[test]
    fn a_placed_asset_is_on_the_timeline_and_a_loose_one_is_not() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let loose = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        let (placed, unused) = match history
            .apply(
                &mut cut,
                &[Command::AddAsset(video), Command::AddAsset(loose)],
            )
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(placed), Outcome::Asset(unused)] => (*placed, *unused),
            other => panic!("expected two assets, got {other:?}"),
        };
        let sound = {
            let stored = cut.asset(placed).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment: sound,
                }],
            )
            .expect("place");
        assert!(on_timeline(&cut, placed));
        assert!(!on_timeline(&cut, unused));
        assert_eq!(
            swatch(cut.asset(placed).expect("video").kind()),
            theme::VIDEO
        );
        assert_eq!(
            swatch(cut.asset(unused).expect("audio").kind()),
            theme::AUDIO
        );
    }

    #[test]
    fn the_plus_places_a_clip_at_the_end_of_its_lane() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let quiet = Asset::video(
            "file:///cuts/quiet.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            false,
        )
        .expect("quiet");
        let sound = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        let still = Asset::image("file:///cuts/still.png", 320, 180).expect("image");
        let (video_id, quiet_id, audio_id, image_id) = match history
            .apply(
                &mut cut,
                &[
                    Command::AddAsset(video),
                    Command::AddAsset(quiet),
                    Command::AddAsset(sound),
                    Command::AddAsset(still),
                ],
            )
            .expect("add")
            .as_slice()
        {
            [
                Outcome::Asset(video),
                Outcome::Asset(quiet),
                Outcome::Asset(audio),
                Outcome::Asset(image),
            ] => (*video, *quiet, *audio, *image),
            other => panic!("expected four assets, got {other:?}"),
        };
        let main = cut.timeline().main_track().id();
        let first = super::place_at_end(&cut, video_id).expect("first");
        match &first {
            Command::Place {
                track: Some(id),
                segment,
            } => {
                assert_eq!(*id, main);
                assert_eq!(segment.channels(), Channels::Both);
                assert_eq!(segment.asset_id(), Some(video_id));
                assert_eq!(segment.timeline().start().value, 0);
                let source = segment.source().expect("source");
                assert_eq!(source.start().value, 0);
                assert_eq!(source.duration().value, 30_000);
            }
            other => panic!("expected the main track, got {other:?}"),
        }
        history.apply(&mut cut, &[first]).expect("place");
        let again = super::place_at_end(&cut, video_id).expect("again");
        match &again {
            Command::Place { segment, .. } => {
                assert_eq!(segment.timeline().start().value, 1_000_000);
            }
            other => panic!("expected a second place, got {other:?}"),
        }
        history.apply(&mut cut, &[again]).expect("second");
        assert_eq!(cut.timeline().main_track().segments().len(), 2);
        let quiet_place = super::place_at_end(&cut, quiet_id).expect("quiet");
        match &quiet_place {
            Command::Place {
                track: Some(id),
                segment,
            } => {
                assert_eq!(*id, main);
                assert_eq!(segment.channels(), Channels::Picture);
                assert_eq!(segment.timeline().start().value, 2_000_000);
            }
            other => panic!("expected a silent picture, got {other:?}"),
        }
        history
            .apply(
                &mut cut,
                &[Command::SetLocked {
                    track: main,
                    on: true,
                }],
            )
            .expect("lock");
        let blocked = super::place_at_end(&cut, video_id).expect("locked command");
        assert!(matches!(
            history.apply(&mut cut, &[blocked]),
            Err(Error::Locked)
        ));
        assert_eq!(cut.timeline().main_track().segments().len(), 2);
        let opening = super::place_at_end(&cut, audio_id).expect("open audio");
        match &opening {
            Command::Place {
                track: None,
                segment,
            } => {
                assert_eq!(segment.channels(), Channels::Sound);
                assert_eq!(segment.timeline().start().value, 0);
            }
            other => panic!("expected a new audio lane, got {other:?}"),
        }
        history.apply(&mut cut, &[opening]).expect("open");
        let appended = super::place_at_end(&cut, audio_id).expect("append");
        match &appended {
            Command::Place {
                track: Some(id),
                segment,
            } => {
                assert_eq!(segment.timeline().start().value, 1_000_000);
                assert_eq!(segment.channels(), Channels::Sound);
                let lane = cut
                    .timeline()
                    .tracks()
                    .iter()
                    .find(|track| track.id() == *id)
                    .expect("audio lane");
                assert_eq!(lane.kind(), TrackKind::Audio);
            }
            other => panic!("expected the audio lane, got {other:?}"),
        }
        assert!(matches!(
            super::place_at_end(&cut, image_id),
            Err(Error::ImageSource)
        ));
    }

    #[test]
    fn hovering_a_card_shows_a_plus_that_places_it() {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let still = Asset::image("file:///cuts/still.png", 320, 180).expect("image");
        let (video_id, image_id) = match history
            .apply(
                &mut cut,
                &[Command::AddAsset(video), Command::AddAsset(still)],
            )
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(video), Outcome::Asset(image)] => (*video, *image),
            other => panic!("expected two assets, got {other:?}"),
        };
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let mut bin = crate::media_bin::Bin::new();
        let mut shelf = super::Shelf::Media;
        let mut source = super::MediaSource::Project;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 500.0),
        );
        let commands = {
            let mut time = 0.0;
            let mut pass = |events: Vec<eframe::egui::Event>| {
                time += 0.05;
                let mut frame = Vec::new();
                let mut output = ctx.run_ui(
                    eframe::egui::RawInput {
                        screen_rect: Some(screen),
                        time: Some(time),
                        events,
                        ..Default::default()
                    },
                    |ui| {
                        let action = super::show(ui, &cut, &mut shelf, &mut source, &mut bin);
                        frame = action.commands;
                    },
                );
                let tree = output.platform_output.accesskit_update.take();
                output.textures_delta.clear();
                (tree, frame)
            };
            let (tree, _) = pass(vec![]);
            let tree = tree.expect("the tree is built");
            let still_at = node_center(&tree, "still.png");
            let video_at = node_center(&tree, "v.mp4");
            let (over_still, _) = pass(vec![eframe::egui::Event::PointerMoved(still_at)]);
            let over_still = over_still.expect("still");
            assert!(
                over_still
                    .nodes
                    .iter()
                    .all(|(_, node)| node.label() != Some("Add to timeline"))
            );
            let (over_video, _) = pass(vec![eframe::egui::Event::PointerMoved(video_at)]);
            let over_video = over_video.expect("video");
            let plus_at = node_center(&over_video, "Add to timeline");
            pass(vec![
                eframe::egui::Event::PointerMoved(plus_at),
                eframe::egui::Event::PointerButton {
                    pos: plus_at,
                    button: eframe::egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: eframe::egui::Modifiers::default(),
                },
            ]);
            let (_, commands) = pass(vec![eframe::egui::Event::PointerButton {
                pos: plus_at,
                button: eframe::egui::PointerButton::Primary,
                pressed: false,
                modifiers: eframe::egui::Modifiers::default(),
            }]);
            let expected = super::place_at_end(&cut, video_id).expect("place");
            assert_eq!(commands, vec![expected]);
            commands
        };
        assert!(!bin.is_selected(crate::media_bin::CardId::File(video_id)));
        assert!(!bin.is_selected(crate::media_bin::CardId::File(image_id)));
        history
            .apply(&mut cut, &commands)
            .expect("the plus places the clip");
        assert!(on_timeline(&cut, video_id));
        assert!(!on_timeline(&cut, image_id));
    }

    #[test]
    fn a_right_click_on_empty_space_opens_a_folder() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let mut bin = crate::media_bin::Bin::new();
        let mut shelf = super::Shelf::Media;
        let mut source = super::MediaSource::Project;
        let cut = project();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 500.0),
        );
        let spot = eframe::egui::pos2(160.0, 180.0);
        let mut time = 0.0;
        let mut pass = |events: Vec<eframe::egui::Event>| {
            time += 0.05;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, &cut, &mut shelf, &mut source, &mut bin);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            tree
        };
        pass(vec![]);
        pass(vec![
            eframe::egui::Event::PointerMoved(spot),
            eframe::egui::Event::PointerButton {
                pos: spot,
                button: eframe::egui::PointerButton::Secondary,
                pressed: true,
                modifiers: eframe::egui::Modifiers::default(),
            },
        ]);
        pass(vec![eframe::egui::Event::PointerButton {
            pos: spot,
            button: eframe::egui::PointerButton::Secondary,
            pressed: false,
            modifiers: eframe::egui::Modifiers::default(),
        }]);
        let tree = pass(vec![]).expect("the tree is built");
        let row = tree
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some("New folder"))
            .expect("new folder row");
        let bounds = row.1.bounds().expect("row bounds");
        let width = bounds.x1 - bounds.x0;
        let height = bounds.y1 - bounds.y0;
        assert!(
            (width - 236.0).abs() < 1.0,
            "a menu row is the plate's inner width, got {width}"
        );
        assert!(
            (height - 32.0).abs() < 1.0,
            "a menu row is 32px, got {height}"
        );
        let at = eframe::egui::pos2(
            ((bounds.x0 + bounds.x1) / 2.0) as f32,
            ((bounds.y0 + bounds.y1) / 2.0) as f32,
        );
        pass(vec![
            eframe::egui::Event::PointerMoved(at),
            eframe::egui::Event::PointerButton {
                pos: at,
                button: eframe::egui::PointerButton::Primary,
                pressed: true,
                modifiers: eframe::egui::Modifiers::default(),
            },
        ]);
        pass(vec![eframe::egui::Event::PointerButton {
            pos: at,
            button: eframe::egui::PointerButton::Primary,
            pressed: false,
            modifiers: eframe::egui::Modifiers::default(),
        }]);
        assert!(matches!(bin.visible(&[]).as_slice(), [CardId::Folder(_)]));
    }

    #[test]
    fn the_audio_shelf_lists_sound_and_a_text_shelf_says_empty() {
        let mut history = History::new();
        let mut cut = project();
        let picture = Asset::video(
            "file:///cuts/picture.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let sound = Asset::audio(
            "file:///cuts/boat.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        history
            .apply(
                &mut cut,
                &[Command::AddAsset(picture), Command::AddAsset(sound)],
            )
            .expect("add");
        let listed = super::audio_infos(&cut);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "boat.wav");

        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let mut bin = crate::media_bin::Bin::new();
        let mut source = super::MediaSource::Project;
        let mut shelf = super::Shelf::Audio;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 500.0),
        );
        let mut time = 0.0;
        let mut pass = |shelf: &mut super::Shelf| {
            time += 0.05;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, &cut, shelf, &mut source, &mut bin);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            tree.expect("the tree is built")
        };
        let audio = pass(&mut shelf);
        assert!(audio.nodes.iter().any(|(_, node)| named(node, "boat.wav")));
        assert!(
            audio
                .nodes
                .iter()
                .all(|(_, node)| !named(node, "picture.mp4"))
        );
        assert!(audio.nodes.iter().all(|(_, node)| !named(node, "Empty")));
        shelf = super::Shelf::Text;
        let text = pass(&mut shelf);
        assert!(text.nodes.iter().any(|(_, node)| named(node, "Empty")));
        assert!(text.nodes.iter().all(|(_, node)| !named(node, "boat.wav")));
    }

    #[test]
    fn the_other_shelves_say_empty_and_hide_the_files() {
        let mut history = History::new();
        let mut cut = project();
        let picture = Asset::video(
            "file:///cuts/picture.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let sound = Asset::audio(
            "file:///cuts/boat.wav",
            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("scale")),
        )
        .expect("audio");
        history
            .apply(
                &mut cut,
                &[Command::AddAsset(picture), Command::AddAsset(sound)],
            )
            .expect("add");

        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let mut bin = crate::media_bin::Bin::new();
        let mut source = super::MediaSource::Project;
        let mut shelf = super::Shelf::Media;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 500.0),
        );
        let mut time = 0.0;
        let mut pass = |shelf: &mut super::Shelf, events: Vec<eframe::egui::Event>| {
            time += 0.05;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, &cut, shelf, &mut source, &mut bin);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            tree.expect("the tree is built")
        };
        let media = pass(&mut shelf, vec![]);
        assert!(
            media
                .nodes
                .iter()
                .any(|(_, node)| named(node, "picture.mp4"))
        );
        let shelves = [
            ("Text", super::Shelf::Text),
            ("Stickers", super::Shelf::Stickers),
            ("Effects", super::Shelf::Effects),
            ("Transitions", super::Shelf::Transitions),
            ("Filters", super::Shelf::Filters),
            ("Adjustments", super::Shelf::Adjustments),
            ("Templates", super::Shelf::Templates),
        ];
        for (label, kind) in shelves {
            let tree = pass(&mut shelf, vec![]);
            let at = center(&tree, label);
            pass(
                &mut shelf,
                vec![
                    eframe::egui::Event::PointerMoved(at),
                    press(at, eframe::egui::PointerButton::Primary, true),
                ],
            );
            let shown = pass(
                &mut shelf,
                vec![press(at, eframe::egui::PointerButton::Primary, false)],
            );
            assert!(shelf == kind);
            assert!(shown.nodes.iter().any(|(_, node)| named(node, "Empty")));
            assert!(
                shown
                    .nodes
                    .iter()
                    .all(|(_, node)| !named(node, "picture.mp4"))
            );
            assert!(shown.nodes.iter().all(|(_, node)| !named(node, "boat.wav")));
            assert!(shown.nodes.iter().all(|(_, node)| !named(node, "Import")));
            assert!(shown.nodes.iter().all(|(_, node)| !named(node, "All")));
            assert!(
                shown
                    .nodes
                    .iter()
                    .all(|(_, node)| !named(node, "New folder"))
            );
        }
        let empty = pass(&mut shelf, vec![]);
        let at = center(&empty, "Empty");
        pass(
            &mut shelf,
            vec![
                eframe::egui::Event::PointerMoved(at),
                press(at, eframe::egui::PointerButton::Secondary, true),
            ],
        );
        let menu = pass(
            &mut shelf,
            vec![press(at, eframe::egui::PointerButton::Secondary, false)],
        );
        assert!(
            menu.nodes
                .iter()
                .all(|(_, node)| !named(node, "New folder"))
        );
        assert!(
            menu.nodes
                .iter()
                .all(|(_, node)| !named(node, "Sorted by name"))
        );
        crate::words::install(&ctx, crate::words::Language::French);
        let french = pass(&mut shelf, vec![]);
        assert!(french.nodes.iter().any(|(_, node)| named(node, "Vide")));
        assert!(french.nodes.iter().any(|(_, node)| named(node, "Modèles")));
        assert!(french.nodes.iter().all(|(_, node)| !named(node, "Empty")));
    }

    fn named(node: &eframe::egui::accesskit::Node, text: &str) -> bool {
        node.label() == Some(text) || node.value() == Some(text)
    }

    fn center(tree: &eframe::egui::accesskit::TreeUpdate, text: &str) -> eframe::egui::Pos2 {
        let node = tree
            .nodes
            .iter()
            .find(|(_, node)| named(node, text))
            .unwrap_or_else(|| panic!("missing {text}"));
        let bounds = node
            .1
            .bounds()
            .unwrap_or_else(|| panic!("no bounds {text}"));
        eframe::egui::pos2(
            ((bounds.x0 + bounds.x1) / 2.0) as f32,
            ((bounds.y0 + bounds.y1) / 2.0) as f32,
        )
    }

    fn press(
        at: eframe::egui::Pos2,
        button: eframe::egui::PointerButton,
        pressed: bool,
    ) -> eframe::egui::Event {
        eframe::egui::Event::PointerButton {
            pos: at,
            button,
            pressed,
            modifiers: eframe::egui::Modifiers::default(),
        }
    }

    #[test]
    fn free_stock_replaces_the_bin_and_lists_no_shots() {
        let mut history = History::new();
        let mut cut = project();
        let picture = Asset::video(
            "file:///cuts/picture.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        history
            .apply(&mut cut, &[Command::AddAsset(picture)])
            .expect("add");

        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let mut bin = crate::media_bin::Bin::new();
        let mut shelf = super::Shelf::Media;
        let mut source = super::MediaSource::Project;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 500.0),
        );
        let mut time = 0.0;
        let mut pass = |events: Vec<eframe::egui::Event>| {
            time += 0.05;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, &cut, &mut shelf, &mut source, &mut bin);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            (tree.expect("the tree is built"), source)
        };
        fn click(
            pass: &mut impl FnMut(
                Vec<eframe::egui::Event>,
            )
                -> (eframe::egui::accesskit::TreeUpdate, super::MediaSource),
            label: &str,
            button: eframe::egui::PointerButton,
        ) -> (eframe::egui::accesskit::TreeUpdate, super::MediaSource) {
            let (tree, _) = pass(vec![]);
            let at = center(&tree, label);
            pass(vec![
                eframe::egui::Event::PointerMoved(at),
                press(at, button, true),
            ]);
            pass(vec![press(at, button, false)])
        }

        let (tree, _) = pass(vec![]);
        assert!(
            tree.nodes
                .iter()
                .any(|(_, node)| named(node, "picture.mp4"))
        );
        assert!(tree.nodes.iter().any(|(_, node)| named(node, "All")));
        assert!(tree.nodes.iter().any(|(_, node)| named(node, "Import")));
        let (stock, shown) = click(
            &mut pass,
            "Free stock",
            eframe::egui::PointerButton::Primary,
        );
        assert!(shown == super::MediaSource::Stock);
        assert!(stock.nodes.iter().any(|(_, node)| named(node, "Empty")));
        assert!(
            stock
                .nodes
                .iter()
                .all(|(_, node)| !named(node, "picture.mp4"))
        );
        assert!(stock.nodes.iter().all(|(_, node)| !named(node, "All")));
        assert!(stock.nodes.iter().all(|(_, node)| !named(node, "Import")));
        assert!(
            stock
                .nodes
                .iter()
                .all(|(_, node)| !named(node, "New folder"))
        );
        let (menu, shown) = click(&mut pass, "Empty", eframe::egui::PointerButton::Secondary);
        assert!(
            menu.nodes
                .iter()
                .all(|(_, node)| !named(node, "New folder"))
        );
        assert!(
            menu.nodes
                .iter()
                .all(|(_, node)| !named(node, "Sorted by name"))
        );
        assert!(shown == super::MediaSource::Stock);
        let (back, shown) = click(&mut pass, "Project", eframe::egui::PointerButton::Primary);
        assert!(shown == super::MediaSource::Project);
        assert!(
            back.nodes
                .iter()
                .any(|(_, node)| named(node, "picture.mp4"))
        );
        assert!(back.nodes.iter().any(|(_, node)| named(node, "Import")));
        assert!(back.nodes.iter().all(|(_, node)| !named(node, "Empty")));

        crate::words::install(&ctx, crate::words::Language::French);
        let (french, shown) = click(
            &mut pass,
            "Stock libre",
            eframe::egui::PointerButton::Primary,
        );
        assert!(shown == super::MediaSource::Stock);
        assert!(french.nodes.iter().any(|(_, node)| named(node, "Vide")));
        assert!(
            french
                .nodes
                .iter()
                .any(|(_, node)| named(node, "Stock libre"))
        );
        assert!(
            french
                .nodes
                .iter()
                .all(|(_, node)| !named(node, "picture.mp4"))
        );
        assert!(french.nodes.iter().all(|(_, node)| !named(node, "Empty")));
        assert!(
            french
                .nodes
                .iter()
                .all(|(_, node)| !named(node, "Importer"))
        );
    }

    #[test]
    fn a_missing_file_can_ask_to_relink() {
        assert!(super::source_missing("content://media/1"));
        assert!(super::source_missing("/no/such/cutlass-relink.mp4"));
        let present = std::env::temp_dir().join("cutlass-relink-present.mp4");
        std::fs::write(&present, b"x").expect("temp file");
        assert!(!super::source_missing(
            present.to_str().expect("utf-8 path")
        ));
        std::fs::remove_file(&present).expect("remove temp");

        let mut history = History::new();
        let mut cut = project();
        let missing = Asset::video(
            "/no/such/cutlass-relink.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            16,
            16,
            false,
        )
        .expect("video");
        let id = match history
            .apply(&mut cut, &[Command::AddAsset(missing)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let mut bin = crate::media_bin::Bin::new();
        let mut shelf = super::Shelf::Media;
        let mut source = super::MediaSource::Project;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(400.0, 500.0),
        );
        let mut time = 0.0;
        let mut pass = |events: Vec<eframe::egui::Event>| {
            time += 0.05;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    super::show(ui, &cut, &mut shelf, &mut source, &mut bin);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            tree.expect("the tree is built")
        };
        let tree = pass(vec![]);
        let card = tree
            .nodes
            .iter()
            .find(|(_, node)| named(node, "cutlass-relink.mp4"))
            .expect("card");
        let bounds = card.1.bounds().expect("card bounds");
        let at = eframe::egui::pos2(
            ((bounds.x0 + bounds.x1) / 2.0) as f32,
            ((bounds.y0 + bounds.y1) / 2.0) as f32,
        );
        pass(vec![
            eframe::egui::Event::PointerMoved(at),
            eframe::egui::Event::PointerButton {
                pos: at,
                button: eframe::egui::PointerButton::Secondary,
                pressed: true,
                modifiers: eframe::egui::Modifiers::default(),
            },
        ]);
        pass(vec![eframe::egui::Event::PointerButton {
            pos: at,
            button: eframe::egui::PointerButton::Secondary,
            pressed: false,
            modifiers: eframe::egui::Modifiers::default(),
        }]);
        let menu = pass(vec![]);
        let row = menu
            .nodes
            .iter()
            .find(|(_, node)| named(node, "Link to media"))
            .expect("link row");
        assert!(!row.1.is_disabled());
        let bounds = row.1.bounds().expect("row bounds");
        let at = eframe::egui::pos2(
            ((bounds.x0 + bounds.x1) / 2.0) as f32,
            ((bounds.y0 + bounds.y1) / 2.0) as f32,
        );
        pass(vec![
            eframe::egui::Event::PointerMoved(at),
            eframe::egui::Event::PointerButton {
                pos: at,
                button: eframe::egui::PointerButton::Primary,
                pressed: true,
                modifiers: eframe::egui::Modifiers::default(),
            },
        ]);
        pass(vec![eframe::egui::Event::PointerButton {
            pos: at,
            button: eframe::egui::PointerButton::Primary,
            pressed: false,
            modifiers: eframe::egui::Modifiers::default(),
        }]);
        assert_eq!(super::take_relink(&ctx), Some(id));
    }
}
