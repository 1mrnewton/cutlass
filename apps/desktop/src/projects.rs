//! New, open, save, and the recent list.
//!
//! [`cutlass_project::Project`] turns a cut into version
//! [`cutlass_project::FILE_VERSION`] text. This module reads and writes the
//! path. The list of recent paths is not a project field. A changed cut is
//! also copied to a sidecar. That copy is not the user's file. On launch the
//! newest copy from a process that is no longer running is offered.

use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align2, Button, FontFamily, FontId, Key, Modifiers, Rect, RichText, Sense, UiBuilder,
    pos2, vec2,
};

use cutlass_project::{Project, ProjectId};

use crate::theme;
use crate::words::{self, Phrase};

const RECENT_LIMIT: usize = 10;
const RECENT_KEY: &str = "recent-projects";

/// A choice from the File menu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ask {
    New,
    Open,
    Recent(PathBuf),
    Save,
    SaveAs,
}

/// New, Open, or a recent path. Each one replaces the cut that is open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Replace {
    New,
    Open,
    Recent(PathBuf),
}

/// The unsaved-changes dialog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuardChoice {
    Save,
    Discard,
    Cancel,
}

/// A project and the text that counts as saved.
pub(crate) struct Loaded {
    pub project: Project,
    pub checkpoint: String,
}

/// Why a path did not become a project.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadError {
    /// The path is not a file. A recent entry with this error is dropped.
    Missing,
    /// The path is a file, and this build cannot read it. The open cut stays.
    Unread,
}

/// An empty Untitled cut, and the text that marks it saved.
pub(crate) fn blank() -> Loaded {
    let id = ProjectId::from_bytes(uuid::Uuid::new_v4().into_bytes());
    let project = Project::new(id, "Untitled");
    let checkpoint = project.to_json().expect("a new project writes version 1");
    Loaded {
        project,
        checkpoint,
    }
}

/// The cut differs from the last text that was saved, opened, or newly made.
pub(crate) fn changed(project: &Project, checkpoint: &str) -> bool {
    match project.to_json() {
        Ok(text) => text != checkpoint,
        Err(_) => true,
    }
}

/// New, Open, and a recent path ask before they replace a changed cut.
pub(crate) fn needs_guard(changed: bool, ask: &Ask) -> bool {
    changed && matches!(ask, Ask::New | Ask::Open | Ask::Recent(_))
}

/// Reads a `.cutlass` file. A failure loads nothing.
pub(crate) fn read(path: &Path) -> Result<Loaded, ReadError> {
    if !path.is_file() {
        return Err(ReadError::Missing);
    }
    let bytes = std::fs::read(path).map_err(|_| ReadError::Unread)?;
    let text = String::from_utf8(bytes).map_err(|_| ReadError::Unread)?;
    let project = Project::from_json(&text).map_err(|_| ReadError::Unread)?;
    let checkpoint = project.to_json().map_err(|_| ReadError::Unread)?;
    Ok(Loaded {
        project,
        checkpoint,
    })
}

/// Writes version 1. The previous file stays if this fails.
///
/// The returned text is the checkpoint. On Windows a crash between moving the
/// old file aside and moving the new one in can leave only the aside copy.
/// That copy is not offered. Crash recovery offers the sidecar.
pub(crate) fn write(path: &Path, project: &Project) -> Result<String, ()> {
    let text = project.to_json().map_err(|_| ())?;
    write_text(path, &text)?;
    Ok(text)
}

/// The copy differs from the user's file, and from the last copy.
pub(crate) fn draft_needed(checkpoint: &str, current: &str, written: Option<&str>) -> bool {
    current != checkpoint && written != Some(current)
}

/// This process's autosave directory, under the app's storage.
pub(crate) fn draft_dir() -> Option<PathBuf> {
    eframe::storage_dir("cutlass").map(|dir| dir.join("autosave"))
}

/// Copies a changed cut. `slot` is this process. The user's path is not written.
///
/// The note is written after the copy. Another slot in `dir` is left alone.
pub(crate) fn write_draft(
    dir: &Path,
    slot: &str,
    text: &str,
    name: &str,
    source: Option<&Path>,
) -> Result<(), ()> {
    if !slot_ok(slot) {
        return Err(());
    }
    std::fs::create_dir_all(dir).map_err(|_| ())?;
    write_text(&dir.join(format!("{slot}.cutlass")), text)?;
    write_text(
        &dir.join(format!("{slot}.meta")),
        &encode_note(name, source),
    )?;
    Ok(())
}

/// Removes one slot. Another slot in `dir` stays.
pub(crate) fn remove_draft(dir: &Path, slot: &str) {
    if !slot_ok(slot) {
        return;
    }
    let _ = std::fs::remove_file(dir.join(format!("{slot}.cutlass")));
    let _ = std::fs::remove_file(dir.join(format!("{slot}.meta")));
}

/// A copy left by a process that is no longer running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Recovery {
    pub slot: String,
    pub name: String,
    pub source: Option<PathBuf>,
    pub cutlass: PathBuf,
}

/// Restore keeps the cut. Discard removes that copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecoveryChoice {
    Restore,
    Discard,
}

/// The newest readable slot that is not `ours` and is not still running.
///
/// A copy that matches the user's file is removed and not offered. A copy
/// this build cannot read is left in place. A missing note still offers the
/// cut, with no source path.
pub(crate) fn find_recovery(
    dir: &Path,
    ours: u32,
    running: impl Fn(u32) -> bool,
) -> Option<Recovery> {
    let mut best: Option<(std::time::SystemTime, Recovery)> = None;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(slot) = slot_of(&path) else {
            continue;
        };
        let Ok(pid) = slot.parse::<u32>() else {
            continue;
        };
        if pid == 0 || pid == ours || running(pid) {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) else {
            continue;
        };
        let Ok(loaded) = read(&path) else {
            continue;
        };
        let note = std::fs::read_to_string(dir.join(format!("{slot}.meta")))
            .ok()
            .and_then(|text| decode_note(&text));
        let source = note.as_ref().and_then(|note| note.source.clone());
        if source.as_ref().is_some_and(|source| {
            read(source)
                .ok()
                .is_some_and(|saved| saved.checkpoint == loaded.checkpoint)
        }) {
            remove_draft(dir, &slot);
            continue;
        }
        let name = note
            .as_ref()
            .map(|note| note.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| loaded.project.name().to_owned());
        let recovery = Recovery {
            slot,
            name,
            source,
            cutlass: path,
        };
        if best.as_ref().is_none_or(|(time, _)| modified >= *time) {
            best = Some((modified, recovery));
        }
    }
    best.map(|(_, recovery)| recovery)
}

/// The text of the user's file, or empty when that file cannot be read.
pub(crate) fn saved_checkpoint(source: Option<&Path>) -> String {
    source
        .and_then(|path| read(path).ok())
        .map(|loaded| loaded.checkpoint)
        .unwrap_or_default()
}

fn slot_of(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let slot = name.strip_suffix(".cutlass")?;
    slot_ok(slot).then(|| slot.to_owned())
}

/// Puts `path` first and keeps at most ten.
pub(crate) fn remember(recent: &mut Vec<PathBuf>, path: &Path) {
    recent.retain(|item| item != path);
    recent.insert(0, path.to_path_buf());
    recent.truncate(RECENT_LIMIT);
}

/// Drops `path` from the list.
pub(crate) fn forget(recent: &mut Vec<PathBuf>, path: &Path) {
    recent.retain(|item| item != path);
}

/// The list stored with the app, or an empty one.
pub(crate) fn load_recent(storage: Option<&dyn eframe::Storage>) -> Vec<PathBuf> {
    let Some(storage) = storage else {
        return Vec::new();
    };
    storage
        .get_string(RECENT_KEY)
        .map(|text| decode_recent(&text))
        .unwrap_or_default()
}

/// Writes the list. This is not the project file.
pub(crate) fn store_recent(storage: &mut dyn eframe::Storage, recent: &[PathBuf]) {
    storage.set_string(RECENT_KEY, encode_recent(recent));
}

/// One `.cutlass` file. Cancel returns nothing. The label stays untranslated.
pub(crate) fn pick_open() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Cutlass", &["cutlass"])
        .pick_file()
}

/// A path to write. Cancel returns nothing. A missing suffix is added.
pub(crate) fn pick_save(name: &str) -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Cutlass", &["cutlass"])
        .set_file_name(file_title(name))
        .save_file()
        .map(ensure_suffix)
}

/// File on the project bar. The menu lists New project, Open, Save, Save as, and recent paths.
pub(crate) fn file_button(ui: &mut egui::Ui, rect: Rect, recent: &[PathBuf]) -> Option<Ask> {
    let label = words::of(ui, Phrase::File);
    let mut child = ui.new_child(UiBuilder::new().id_salt("file-menu").max_rect(rect));
    let response = theme::with_face(&mut child, theme::BAR, theme::INK, |ui| {
        ui.add_sized(rect.size(), Button::new(RichText::new(label)))
    });
    let mut ask = None;
    let _open = egui::Popup::menu(&response).show(|ui| {
        theme::tune_menu(ui);
        if ui.button(words::of(ui, Phrase::NewProject)).clicked() {
            ask = Some(Ask::New);
            ui.close();
        }
        if ask.is_none() && ui.button(words::of(ui, Phrase::OpenProject)).clicked() {
            ask = Some(Ask::Open);
            ui.close();
        }
        if ask.is_none() && ui.button(words::of(ui, Phrase::Save)).clicked() {
            ask = Some(Ask::Save);
            ui.close();
        }
        if ask.is_none() && ui.button(words::of(ui, Phrase::SaveAs)).clicked() {
            ask = Some(Ask::SaveAs);
            ui.close();
        }
        ui.separator();
        ui.label(
            RichText::new(words::of(ui, Phrase::RecentProjects))
                .color(theme::MUTED)
                .size(12.0),
        );
        if recent.is_empty() {
            ui.add_enabled(false, Button::new(words::of(ui, Phrase::NoRecent)));
        }
        for path in recent {
            if ask.is_some() {
                break;
            }
            if ui.button(row_label(path, recent)).clicked() {
                ask = Some(Ask::Recent(path.clone()));
                ui.close();
            }
        }
    });
    ask
}

/// Asks before a changed cut is replaced. Escape and a click outside cancel.
pub(crate) fn show_guard(ui: &egui::Ui) -> Option<GuardChoice> {
    let mut choice = None;
    let screen = ui.ctx().content_rect();
    let question = words::of(ui, Phrase::SaveChanges);
    let save = words::of(ui, Phrase::Save);
    let discard = words::of(ui, Phrase::DontSave);
    let cancel = words::of(ui, Phrase::Cancel);
    let font = FontId::new(13.0, FontFamily::Proportional);
    let save_w = words::control_width(ui, save, font.clone(), 20.0, 72.0);
    let discard_w = words::control_width(ui, discard, font.clone(), 20.0, 72.0);
    let cancel_w = words::control_width(ui, cancel, font.clone(), 20.0, 64.0);
    let buttons = save_w + discard_w + cancel_w + 16.0;
    let text_w = ui
        .painter()
        .layout_no_wrap(question.to_owned(), font.clone(), theme::INK)
        .size()
        .x;
    let width = (buttons + 48.0)
        .max(text_w + 48.0)
        .max(360.0)
        .min(screen.width() - 48.0);
    let dialog = Rect::from_center_size(screen.center(), vec2(width, 148.0));
    egui::Area::new(egui::Id::unique("save-changes"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .movable(false)
        .constrain(false)
        .fade_in(false)
        .accessible_name(question)
        .role(egui::accesskit::Role::Dialog)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(screen);
            ui.set_min_size(screen.size());
            ui.painter()
                .rect_filled(screen, 0.0, egui::Color32::from_black_alpha(140));
            let scrim = ui.interact(screen, ui.scope_id().with("scrim"), Sense::click());
            let _plate = ui.interact(dialog, ui.scope_id().with("plate"), Sense::click());
            if scrim.clicked() {
                choice = Some(GuardChoice::Cancel);
            }
            ui.painter().rect_filled(dialog, 12.0, theme::BAR);
            let text_at = pos2(dialog.center().x, dialog.top() + 48.0);
            ui.painter()
                .text(text_at, Align2::CENTER_CENTER, question, font, theme::INK);
            let text_rect = Rect::from_center_size(text_at, vec2(width - 32.0, 24.0));
            crate::tip::read(ui, text_rect, ui.scope_id().with("question"), question);
            let y = dialog.bottom() - 42.0;
            let mut x = dialog.center().x - buttons / 2.0;
            choice = button_at(ui, pos2(x, y), cancel_w, cancel, theme::BAR, theme::INK)
                .map(|()| GuardChoice::Cancel)
                .or(choice);
            x += cancel_w + 8.0;
            choice = button_at(ui, pos2(x, y), discard_w, discard, theme::BAR, theme::INK)
                .map(|()| GuardChoice::Discard)
                .or(choice);
            x += discard_w + 8.0;
            choice = button_at(
                ui,
                pos2(x, y),
                save_w,
                save,
                theme::EXPORT,
                theme::EXPORT_INK,
            )
            .map(|()| GuardChoice::Save)
            .or(choice);
        });
    if choice.is_none() && ui.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Escape)) {
        choice = Some(GuardChoice::Cancel);
    }
    choice
}

/// Restore keeps the cut. Don't save removes that copy. A click outside
/// does not choose, and neither does Escape.
pub(crate) fn show_recovery(ui: &mut egui::Ui, name: &str) -> Option<RecoveryChoice> {
    let mut choice = None;
    let screen = ui.ctx().content_rect();
    let question = words::of(ui, Phrase::RestoreCut);
    let restore = words::of(ui, Phrase::Restore);
    let discard = words::of(ui, Phrase::DontSave);
    let font = FontId::new(13.0, FontFamily::Proportional);
    let restore_w = words::control_width(ui, restore, font.clone(), 20.0, 72.0);
    let discard_w = words::control_width(ui, discard, font.clone(), 20.0, 72.0);
    let buttons = restore_w + discard_w + 8.0;
    let question_w = ui
        .painter()
        .layout_no_wrap(question.to_owned(), font.clone(), theme::INK)
        .size()
        .x;
    let name_w = ui
        .painter()
        .layout_no_wrap(name.to_owned(), font.clone(), theme::MUTED)
        .size()
        .x;
    let width = (buttons + 48.0)
        .max(question_w + 48.0)
        .max(name_w + 48.0)
        .max(360.0)
        .min(screen.width() - 48.0);
    let dialog = Rect::from_center_size(screen.center(), vec2(width, 176.0));
    egui::Area::new(egui::Id::unique("restore-cut"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .movable(false)
        .constrain(false)
        .fade_in(false)
        .accessible_name(question)
        .role(egui::accesskit::Role::Dialog)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(screen);
            ui.set_min_size(screen.size());
            ui.painter()
                .rect_filled(screen, 0.0, egui::Color32::from_black_alpha(140));
            let _scrim = ui.interact(screen, ui.scope_id().with("scrim"), Sense::click());
            let _plate = ui.interact(dialog, ui.scope_id().with("plate"), Sense::click());
            ui.painter().rect_filled(dialog, 12.0, theme::BAR);
            let question_at = pos2(dialog.center().x, dialog.top() + 44.0);
            ui.painter().text(
                question_at,
                Align2::CENTER_CENTER,
                question,
                font.clone(),
                theme::INK,
            );
            crate::tip::read(
                ui,
                Rect::from_center_size(question_at, vec2(width - 32.0, 24.0)),
                ui.scope_id().with("question"),
                question,
            );
            let name_at = pos2(dialog.center().x, dialog.top() + 76.0);
            ui.painter()
                .text(name_at, Align2::CENTER_CENTER, name, font, theme::MUTED);
            crate::tip::read(
                ui,
                Rect::from_center_size(name_at, vec2(width - 32.0, 24.0)),
                ui.scope_id().with("name"),
                name,
            );
            let y = dialog.bottom() - 42.0;
            let mut x = dialog.center().x - buttons / 2.0;
            choice = button_at(ui, pos2(x, y), discard_w, discard, theme::BAR, theme::INK)
                .map(|()| RecoveryChoice::Discard)
                .or(choice);
            x += discard_w + 8.0;
            choice = button_at(
                ui,
                pos2(x, y),
                restore_w,
                restore,
                theme::EXPORT,
                theme::EXPORT_INK,
            )
            .map(|()| RecoveryChoice::Restore)
            .or(choice);
        });
    choice
}

fn button_at(
    ui: &mut egui::Ui,
    at: egui::Pos2,
    width: f32,
    label: &str,
    fill: egui::Color32,
    ink: egui::Color32,
) -> Option<()> {
    let rect = Rect::from_min_size(at, vec2(width, 26.0));
    let clicked = theme::with_face(ui, fill, ink, |ui| {
        ui.put(rect, Button::new(RichText::new(label)))
    })
    .clicked();
    clicked.then_some(())
}

fn encode_recent(recent: &[PathBuf]) -> String {
    let mut out = String::new();
    for path in recent.iter().take(RECENT_LIMIT) {
        let Some(text) = path.to_str() else {
            continue;
        };
        if text.is_empty() || text.contains('\n') {
            continue;
        }
        out.push_str(text);
        out.push('\n');
    }
    out
}

fn decode_recent(text: &str) -> Vec<PathBuf> {
    let mut recent = Vec::new();
    for line in text.lines() {
        if recent.len() == RECENT_LIMIT {
            break;
        }
        if line.is_empty() {
            continue;
        }
        let path = PathBuf::from(line);
        if recent.iter().any(|item| item == &path) {
            continue;
        }
        recent.push(path);
    }
    recent
}

fn row_label(path: &Path, recent: &[PathBuf]) -> String {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return path.to_string_lossy().into_owned();
    };
    let duplicate = recent
        .iter()
        .filter(|other| other.file_name() == path.file_name())
        .count()
        > 1;
    if !duplicate {
        return name.to_owned();
    }
    let Some(parent) = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .filter(|parent| !parent.is_empty())
    else {
        return name.to_owned();
    };
    format!("{parent}/{name}")
}

fn file_title(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|glyph| match glyph {
            '/' | '\\' | '\n' | '\r' => '-',
            _ => glyph,
        })
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return "Untitled.cutlass".to_owned();
    }
    let bytes = cleaned.as_bytes();
    let has_suffix = bytes.len() >= 8 && bytes[bytes.len() - 8..].eq_ignore_ascii_case(b".cutlass");
    if has_suffix {
        cleaned.to_owned()
    } else {
        format!("{cleaned}.cutlass")
    }
}

fn ensure_suffix(path: PathBuf) -> PathBuf {
    let cutlass = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("cutlass"));
    if cutlass {
        return path;
    }
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(".cutlass");
    path.with_file_name(name)
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project.cutlass");
    let file = format!(".{name}.{suffix}");
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(file),
        _ => PathBuf::from(file),
    }
}

fn write_text(path: &Path, text: &str) -> Result<(), ()> {
    let tmp = sibling(path, "tmp");
    if std::fs::write(&tmp, text).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return Err(());
    }
    if replace_file(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return Err(());
    }
    Ok(())
}

fn slot_ok(slot: &str) -> bool {
    !slot.is_empty() && slot.len() <= 20 && slot.bytes().all(|byte| byte.is_ascii_digit())
}

fn encode_note(name: &str, source: Option<&Path>) -> String {
    let source = source
        .and_then(|path| path.to_str())
        .map(escape_note)
        .unwrap_or_default();
    format!("1\n{}\n{source}\n", escape_note(name))
}

fn escape_note(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for glyph in text.chars() {
        match glyph {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(glyph),
        }
    }
    out
}

struct Note {
    name: String,
    source: Option<PathBuf>,
}

fn decode_note(text: &str) -> Option<Note> {
    let mut lines = text.lines();
    if lines.next()? != "1" {
        return None;
    }
    let name = unescape_note(lines.next()?)?;
    let source = lines.next()?;
    if lines.next().is_some() {
        return None;
    }
    let source = if source.is_empty() {
        None
    } else {
        Some(PathBuf::from(unescape_note(source)?))
    };
    Some(Note { name, source })
}

fn unescape_note(text: &str) -> Option<String> {
    let mut out = String::new();
    let mut glyphs = text.chars();
    while let Some(glyph) = glyphs.next() {
        if glyph != '\\' {
            out.push(glyph);
            continue;
        }
        match glyphs.next()? {
            '\\' => out.push('\\'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            _ => return None,
        }
    }
    Some(out)
}

/// `true` when `pid` is a process that is still running.
pub(crate) fn process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unix_running(pid)
    }
    #[cfg(windows)]
    {
        windows_running(pid)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

#[cfg(unix)]
#[expect(
    unsafe_code,
    reason = "signal 0 asks whether a process is still running"
)]
fn unix_running(pid: u32) -> bool {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 does not deliver a signal. It only reports whether
    // `pid` exists.
    let result = unsafe { kill(pid, 0) };
    result == 0 || std::io::Error::last_os_error().kind() == std::io::ErrorKind::PermissionDenied
}

#[cfg(windows)]
#[expect(
    unsafe_code,
    reason = "a limited process handle asks whether that process is still running"
)]
fn windows_running(pid: u32) -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut core::ffi::c_void;
        fn CloseHandle(handle: *mut core::ffi::c_void) -> i32;
        fn GetLastError() -> u32;
    }
    if pid == 0 {
        return false;
    }
    const QUERY: u32 = 0x1000;
    const ACCESS_DENIED: u32 = 5;
    // SAFETY: query access does not change the process. A null handle means
    // the call failed, and the error distinguishes a live process from a dead one.
    let handle = unsafe { OpenProcess(QUERY, 0, pid) };
    if !handle.is_null() {
        unsafe { CloseHandle(handle) };
        return true;
    }
    unsafe { GetLastError() == ACCESS_DENIED }
}

fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    if !to.exists() {
        return std::fs::rename(from, to);
    }
    let backup = sibling(to, "bak");
    std::fs::rename(to, &backup)?;
    if let Err(error) = std::fs::rename(from, to) {
        let _ = std::fs::rename(&backup, to);
        return Err(error);
    }
    let _ = std::fs::remove_file(&backup);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use cutlass_project::{Command, History};
    use eframe::egui::accesskit::TreeUpdate;

    use eframe::Storage;

    use crate::words;

    use super::{
        Ask, GuardChoice, PathBuf, ReadError, RecoveryChoice, blank, changed, decode_note,
        decode_recent, draft_needed, encode_recent, ensure_suffix, file_button, file_title,
        find_recovery, forget, load_recent, needs_guard, process_running, read, remember,
        remove_draft, row_label, saved_checkpoint, show_guard, show_recovery, store_recent, write,
        write_draft,
    };

    #[test]
    fn a_saved_file_reads_back_and_a_bad_file_loads_nothing() {
        let dir = TempDir::new();
        let loaded = blank();
        let path = dir.0.join("one.cutlass");
        let text = write(&path, &loaded.project).expect("write");
        assert_eq!(std::fs::read_to_string(&path).expect("file"), text);
        assert!(!dir.0.join(".one.cutlass.tmp").exists());
        let again = read(&path).expect("read");
        assert_eq!(again.project.id(), loaded.project.id());
        assert_eq!(again.project.name(), "Untitled");
        assert_eq!(again.checkpoint, text);
        assert!(!changed(&again.project, &again.checkpoint));

        let junk = dir.0.join("junk.cutlass");
        std::fs::write(&junk, "{\"version\":2}\n").expect("junk");
        assert!(matches!(read(&junk), Err(ReadError::Unread)));
        std::fs::write(&junk, "{").expect("truncated");
        assert!(matches!(read(&junk), Err(ReadError::Unread)));
        assert!(matches!(
            read(&dir.0.join("missing.cutlass")),
            Err(ReadError::Missing)
        ));
        assert!(matches!(read(&dir.0), Err(ReadError::Missing)));
        assert!(write(&dir.0.join("no").join("such.cutlass"), &loaded.project).is_err());
        assert_eq!(std::fs::read_to_string(&path).expect("untouched"), text);
    }

    #[test]
    fn a_changed_cut_is_copied_beside_the_app_and_not_into_its_file() {
        let loaded = blank();
        let checkpoint = loaded.checkpoint.clone();
        assert!(!draft_needed(&checkpoint, &checkpoint, None));
        let mut project = loaded.project;
        let mut history = History::new();
        history
            .apply(&mut project, &[Command::Rename("Cut\nA\\B".to_owned())])
            .expect("rename");
        let current = project.to_json().expect("json");
        assert!(draft_needed(&checkpoint, &current, None));
        assert!(!draft_needed(&checkpoint, &current, Some(&current)));

        let dir = TempDir::new();
        let source = dir.0.join("film.cutlass");
        write_draft(&dir.0.join("autosave"), "7", &checkpoint, "Old", None).expect("old slot");
        write_draft(
            &dir.0.join("autosave"),
            "42",
            &current,
            project.name(),
            Some(&source),
        )
        .expect("this slot");
        assert!(!source.exists());
        let copy = read(&dir.0.join("autosave").join("42.cutlass")).expect("copy");
        assert_eq!(copy.project.name(), "Cut\nA\\B");
        let text = std::fs::read_to_string(dir.0.join("autosave").join("42.meta")).expect("note");
        let note = decode_note(&text).expect("note fields");
        assert_eq!(note.name, "Cut\nA\\B");
        assert_eq!(note.source.as_deref(), Some(source.as_path()));
        assert!(write_draft(&dir.0, "nope", &current, "Cut", None).is_err());
        remove_draft(&dir.0.join("autosave"), "42");
        assert!(!dir.0.join("autosave").join("42.cutlass").exists());
        assert!(!dir.0.join("autosave").join("42.meta").exists());
        assert!(dir.0.join("autosave").join("7.cutlass").is_file());
        assert!(dir.0.join("autosave").join("7.meta").is_file());
    }

    #[test]
    fn a_rename_is_unsaved_until_it_is_undone() {
        let loaded = blank();
        let checkpoint = loaded.checkpoint.clone();
        let mut project = loaded.project;
        let mut history = History::new();
        assert!(!changed(&project, &checkpoint));
        history
            .apply(&mut project, &[Command::Rename("Cut".to_owned())])
            .expect("rename");
        assert!(changed(&project, &checkpoint));
        assert_eq!(project.name(), "Cut");
        history.undo(&mut project).expect("undo");
        assert!(!changed(&project, &checkpoint));
        assert!(!needs_guard(false, &Ask::New));
        assert!(needs_guard(true, &Ask::Open));
        assert!(needs_guard(true, &Ask::Recent(PathBuf::from("a.cutlass"))));
        assert!(!needs_guard(true, &Ask::Save));
        assert!(!needs_guard(true, &Ask::SaveAs));
    }

    #[test]
    fn recent_paths_stay_newest_first_and_skip_a_broken_line() {
        let mut recent = Vec::new();
        let first = PathBuf::from("/cuts/a.cutlass");
        let second = PathBuf::from("/cuts/b.cutlass");
        remember(&mut recent, &first);
        remember(&mut recent, &second);
        assert_eq!(recent, vec![second.clone(), first.clone()]);
        remember(&mut recent, &first);
        assert_eq!(recent.first(), Some(&first));
        forget(&mut recent, &second);
        assert_eq!(recent, vec![first.clone()]);

        for index in 0..12 {
            remember(
                &mut recent,
                &PathBuf::from(format!("/cuts/{index}.cutlass")),
            );
        }
        assert_eq!(recent.len(), 10);
        assert_eq!(
            recent.first().map(PathBuf::as_path),
            Some(PathBuf::from("/cuts/11.cutlass").as_path())
        );

        let stored = encode_recent(&[
            PathBuf::from("ok.cutlass"),
            PathBuf::from("bad\nname.cutlass"),
            PathBuf::from("ok.cutlass"),
        ]);
        assert_eq!(decode_recent(&stored), vec![PathBuf::from("ok.cutlass")]);
        assert_eq!(
            decode_recent("\na.cutlass\n\na.cutlass\nb.cutlass\n"),
            vec![PathBuf::from("a.cutlass"), PathBuf::from("b.cutlass")]
        );
    }

    #[test]
    fn the_app_storage_round_trips_the_list() {
        let mut storage = Mem { recent: None };
        assert!(load_recent(None).is_empty());
        assert!(load_recent(Some(&storage)).is_empty());
        let recent = vec![PathBuf::from("/cuts/a.cutlass")];
        store_recent(&mut storage, &recent);
        assert_eq!(load_recent(Some(&storage)), recent);
        assert!(storage.get_string("egui").is_none());
    }

    #[test]
    fn a_save_name_keeps_one_cutlass_suffix() {
        assert_eq!(file_title("Untitled"), "Untitled.cutlass");
        assert_eq!(file_title("  "), "Untitled.cutlass");
        assert_eq!(file_title("a/b"), "a-b.cutlass");
        assert_eq!(file_title("Cut.CUTLASS"), "Cut.CUTLASS");
        assert_eq!(
            ensure_suffix(PathBuf::from("a.mp4")),
            PathBuf::from("a.mp4.cutlass")
        );
        assert_eq!(
            ensure_suffix(PathBuf::from("a.CUTLASS")),
            PathBuf::from("a.CUTLASS")
        );
        assert_eq!(
            ensure_suffix(PathBuf::from("a")),
            PathBuf::from("a.cutlass")
        );
    }

    #[test]
    fn two_files_with_one_name_show_the_folder() {
        let one = PathBuf::from("/films/Cut.cutlass");
        let two = PathBuf::from("/other/Cut.cutlass");
        let alone = PathBuf::from("/films/Alone.cutlass");
        let recent = vec![one.clone(), two.clone(), alone.clone()];
        assert_eq!(row_label(&one, &recent), "films/Cut.cutlass");
        assert_eq!(row_label(&two, &recent), "other/Cut.cutlass");
        assert_eq!(row_label(&alone, &recent), "Alone.cutlass");
    }

    #[test]
    fn the_file_menu_starts_a_new_cut_and_opens_a_recent_path() {
        let alone = PathBuf::from("/films/Alone.cutlass");
        assert_eq!(
            click_menu(&[], words::Language::English, "File", "New project"),
            Some(Ask::New)
        );
        assert_eq!(
            click_menu(
                std::slice::from_ref(&alone),
                words::Language::English,
                "File",
                "Alone.cutlass"
            ),
            Some(Ask::Recent(alone))
        );
        assert_eq!(
            click_menu(&[], words::Language::French, "Fichier", "Nouveau projet"),
            Some(Ask::New)
        );
    }

    #[test]
    fn an_empty_recent_list_is_shown_and_cannot_be_chosen() {
        let tree = open_menu(&[], words::Language::English, "File");
        let row = tree
            .nodes
            .iter()
            .find(|(_, node)| named(node, "No recent projects"))
            .expect("empty recent row");
        assert!(row.1.is_disabled());
        assert!(tree.nodes.iter().any(|(_, node)| named(node, "Open")));
        assert!(tree.nodes.iter().any(|(_, node)| named(node, "Save as")));
        assert!(tree.nodes.iter().any(|(_, node)| named(node, "Recent")));
    }

    #[test]
    fn the_unsaved_dialog_can_discard_or_cancel() {
        assert_eq!(
            click_guard(words::Language::English, "Don't save"),
            Some(GuardChoice::Discard)
        );
        assert_eq!(
            click_guard(words::Language::French, "Ne pas enregistrer"),
            Some(GuardChoice::Discard)
        );
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::theme::install(&ctx);
        words::install(&ctx, words::Language::English);
        let mut time = 0.0;
        let mut choice = None;
        pass_guard(&ctx, &mut time, &mut choice, vec![]);
        let cancel = pass_guard(
            &ctx,
            &mut time,
            &mut choice,
            vec![eframe::egui::Event::Key {
                key: eframe::egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: eframe::egui::Modifiers::default(),
            }],
        );
        assert!(cancel.nodes.iter().any(|(_, node)| named(node, "Save")));
        assert_eq!(choice, Some(GuardChoice::Cancel));
    }

    fn click_menu(
        recent: &[PathBuf],
        language: words::Language,
        button: &str,
        item: &str,
    ) -> Option<Ask> {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::theme::install(&ctx);
        words::install(&ctx, language);
        let ask = RefCell::new(None);
        let mut time = 0.0;
        let tree = open_menu_with(&ctx, &mut time, recent, &ask, button);
        let at = center(&tree, item);
        pass(
            &ctx,
            &mut time,
            recent,
            &ask,
            vec![eframe::egui::Event::PointerMoved(at), press(at, true)],
        );
        pass(&ctx, &mut time, recent, &ask, vec![press(at, false)]);
        ask.borrow().clone()
    }

    fn open_menu(recent: &[PathBuf], language: words::Language, button: &str) -> TreeUpdate {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::theme::install(&ctx);
        words::install(&ctx, language);
        let ask = RefCell::new(None);
        let mut time = 0.0;
        open_menu_with(&ctx, &mut time, recent, &ask, button)
    }

    fn open_menu_with(
        ctx: &eframe::egui::Context,
        time: &mut f64,
        recent: &[PathBuf],
        ask: &RefCell<Option<Ask>>,
        button: &str,
    ) -> TreeUpdate {
        pass(ctx, time, recent, ask, vec![]);
        let tree = pass(ctx, time, recent, ask, vec![]);
        let at = center(&tree, button);
        pass(
            ctx,
            time,
            recent,
            ask,
            vec![eframe::egui::Event::PointerMoved(at), press(at, true)],
        );
        pass(ctx, time, recent, ask, vec![press(at, false)]);
        pass(ctx, time, recent, ask, vec![])
    }

    fn pass(
        ctx: &eframe::egui::Context,
        time: &mut f64,
        recent: &[PathBuf],
        ask: &RefCell<Option<Ask>>,
        events: Vec<eframe::egui::Event>,
    ) -> TreeUpdate {
        *time += 0.05;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(1280.0, 800.0),
        );
        let file = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(78.0, 6.0),
            eframe::egui::vec2(72.0, 26.0),
        );
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(screen),
                time: Some(*time),
                events,
                ..Default::default()
            },
            |ui| {
                if let Some(next) = file_button(ui, file, recent) {
                    *ask.borrow_mut() = Some(next);
                }
            },
        );
        let tree = output.platform_output.accesskit_update.take();
        output.textures_delta.clear();
        tree.expect("the tree is built")
    }

    fn click_guard(language: words::Language, item: &str) -> Option<GuardChoice> {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::theme::install(&ctx);
        words::install(&ctx, language);
        let mut time = 0.0;
        let mut choice = None;
        pass_guard(&ctx, &mut time, &mut choice, vec![]);
        let tree = pass_guard(&ctx, &mut time, &mut choice, vec![]);
        assert!(
            tree.nodes
                .iter()
                .any(|(_, node)| named(node, "Save") || named(node, "Enregistrer"))
        );
        let at = center(&tree, item);
        pass_guard(
            &ctx,
            &mut time,
            &mut choice,
            vec![eframe::egui::Event::PointerMoved(at), press(at, true)],
        );
        pass_guard(&ctx, &mut time, &mut choice, vec![press(at, false)]);
        choice
    }

    fn pass_guard(
        ctx: &eframe::egui::Context,
        time: &mut f64,
        choice: &mut Option<GuardChoice>,
        events: Vec<eframe::egui::Event>,
    ) -> TreeUpdate {
        *time += 0.05;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(1280.0, 800.0),
        );
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(screen),
                time: Some(*time),
                events,
                ..Default::default()
            },
            |ui| {
                if let Some(next) = show_guard(ui) {
                    *choice = Some(next);
                }
            },
        );
        let tree = output.platform_output.accesskit_update.take();
        output.textures_delta.clear();
        tree.expect("the tree is built")
    }

    fn press(at: eframe::egui::Pos2, pressed: bool) -> eframe::egui::Event {
        eframe::egui::Event::PointerButton {
            pos: at,
            button: eframe::egui::PointerButton::Primary,
            pressed,
            modifiers: eframe::egui::Modifiers::default(),
        }
    }

    fn center(tree: &TreeUpdate, text: &str) -> eframe::egui::Pos2 {
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

    fn named(node: &eframe::egui::accesskit::Node, text: &str) -> bool {
        node.label() == Some(text) || node.value() == Some(text)
    }

    #[test]
    fn launch_offers_the_newest_copy_from_a_dead_process() {
        let dir = TempDir::new();
        let autosave = dir.0.join("autosave");
        let mut project = blank().project;
        let mut history = History::new();
        history
            .apply(&mut project, &[Command::Rename("Cut A".to_owned())])
            .expect("rename");
        let renamed = project.to_json().expect("json");

        write_draft(&autosave, "13", &renamed, "Ignored", None).expect("bare");
        std::fs::remove_file(autosave.join("13.meta")).expect("drop note");
        stamp(&autosave.join("13.cutlass"), 10);

        write_draft(&autosave, "9", &renamed, "Older", None).expect("older");
        stamp(&autosave.join("9.cutlass"), 20);

        let other = dir.0.join("other.cutlass");
        std::fs::write(&other, "nope").expect("other");
        write_draft(&autosave, "11", &renamed, "Cut A", Some(&other)).expect("newer");
        stamp(&autosave.join("11.cutlass"), 30);

        write_draft(&autosave, "42", &renamed, "Ours", None).expect("ours");
        stamp(&autosave.join("42.cutlass"), 40);

        write_draft(&autosave, "7", &renamed, "Live", None).expect("live");
        stamp(&autosave.join("7.cutlass"), 50);

        std::fs::write(autosave.join("99.cutlass"), "nope").expect("junk");
        stamp(&autosave.join("99.cutlass"), 60);

        let saved = dir.0.join("saved.cutlass");
        let same = write(&saved, &blank().project).expect("saved");
        write_draft(&autosave, "15", &same, "Blank", Some(&saved)).expect("same");
        stamp(&autosave.join("15.cutlass"), 70);

        let running = |pid| pid == 7;
        let found = find_recovery(&autosave, 42, running).expect("offer");
        assert_eq!(found.slot, "11");
        assert_eq!(found.name, "Cut A");
        assert_eq!(found.source.as_deref(), Some(other.as_path()));
        assert!(!autosave.join("15.cutlass").exists());
        assert!(!autosave.join("15.meta").exists());
        assert!(autosave.join("42.cutlass").is_file());
        assert!(autosave.join("7.cutlass").is_file());
        assert!(autosave.join("99.cutlass").is_file());

        remove_draft(&autosave, "11");
        remove_draft(&autosave, "9");
        let bare = find_recovery(&autosave, 42, running).expect("bare offer");
        assert_eq!(bare.slot, "13");
        assert_eq!(bare.name, "Cut A");
        assert!(bare.source.is_none());
    }

    #[test]
    fn a_restored_cut_stays_dirty_against_the_users_file() {
        let dir = TempDir::new();
        let path = dir.0.join("film.cutlass");
        let text = write(&path, &blank().project).expect("write");
        assert_eq!(saved_checkpoint(Some(&path)), text);
        assert_eq!(saved_checkpoint(Some(&dir.0.join("missing.cutlass"))), "");
        assert_eq!(saved_checkpoint(None), "");
    }

    #[test]
    fn this_process_is_running_and_an_impossible_pid_is_not() {
        assert!(process_running(std::process::id()));
        assert!(!process_running(u32::MAX));
        assert!(!process_running(2_000_000_000));
    }

    #[test]
    fn the_restore_dialog_names_the_cut_and_waits_for_a_choice() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::theme::install(&ctx);
        words::install(&ctx, words::Language::English);
        let mut time = 0.0;
        let mut choice = None;
        pass_recovery(&ctx, &mut time, "Cut A", &mut choice, vec![]);
        let tree = pass_recovery(&ctx, &mut time, "Cut A", &mut choice, vec![]);
        assert!(
            tree.nodes
                .iter()
                .any(|(_, node)| named(node, "Restore this cut?"))
        );
        assert!(tree.nodes.iter().any(|(_, node)| named(node, "Cut A")));
        pass_recovery(
            &ctx,
            &mut time,
            "Cut A",
            &mut choice,
            vec![eframe::egui::Event::Key {
                key: eframe::egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: eframe::egui::Modifiers::NONE,
            }],
        );
        assert!(choice.is_none());
        assert_eq!(click_recovery("Restore"), Some(RecoveryChoice::Restore));
        assert_eq!(click_recovery("Don't save"), Some(RecoveryChoice::Discard));
    }

    fn stamp(path: &std::path::Path, secs: u64) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("stamp");
        file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .expect("mtime");
    }

    fn click_recovery(item: &str) -> Option<RecoveryChoice> {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::theme::install(&ctx);
        words::install(&ctx, words::Language::English);
        let mut time = 0.0;
        let mut choice = None;
        pass_recovery(&ctx, &mut time, "Cut A", &mut choice, vec![]);
        let tree = pass_recovery(&ctx, &mut time, "Cut A", &mut choice, vec![]);
        let at = center(&tree, item);
        pass_recovery(
            &ctx,
            &mut time,
            "Cut A",
            &mut choice,
            vec![eframe::egui::Event::PointerMoved(at), press(at, true)],
        );
        pass_recovery(
            &ctx,
            &mut time,
            "Cut A",
            &mut choice,
            vec![press(at, false)],
        );
        choice
    }

    fn pass_recovery(
        ctx: &eframe::egui::Context,
        time: &mut f64,
        name: &str,
        choice: &mut Option<RecoveryChoice>,
        events: Vec<eframe::egui::Event>,
    ) -> TreeUpdate {
        *time += 0.05;
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(1280.0, 800.0),
        );
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(screen),
                time: Some(*time),
                events,
                ..Default::default()
            },
            |ui| {
                if let Some(next) = show_recovery(ui, name) {
                    *choice = Some(next);
                }
            },
        );
        let tree = output.platform_output.accesskit_update.take();
        output.textures_delta.clear();
        tree.expect("the tree is built")
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("cutlass-project-{}", uuid::Uuid::new_v4().simple()));
            std::fs::create_dir_all(&path).expect("temp dir");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct Mem {
        recent: Option<String>,
    }

    impl eframe::Storage for Mem {
        fn get_string(&self, key: &str) -> Option<String> {
            if key == super::RECENT_KEY {
                self.recent.clone()
            } else {
                None
            }
        }

        fn set_string(&mut self, key: &str, value: String) {
            if key == super::RECENT_KEY {
                self.recent = Some(value);
            }
        }

        fn remove_string(&mut self, _key: &str) {}

        fn flush(&mut self) {}
    }
}
