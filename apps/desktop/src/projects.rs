//! New, open, save, and the recent list.
//!
//! [`cutlass_project::Project`] turns a cut into version
//! [`cutlass_project::FILE_VERSION`] text. This module reads and writes the
//! path. The list of recent paths is not a project field. A changed cut is
//! also copied to a sidecar. That copy is not the user's file.

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
/// That gap is the later crash-recovery work, not this write.
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

/// Removes this process's copy. Another slot in `dir` stays.
pub(crate) fn remove_draft(dir: &Path, slot: &str) {
    if !slot_ok(slot) {
        return;
    }
    let _ = std::fs::remove_file(dir.join(format!("{slot}.cutlass")));
    let _ = std::fs::remove_file(dir.join(format!("{slot}.meta")));
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
    let response = child.add_sized(
        rect.size(),
        Button::new(RichText::new(label).color(theme::INK)).fill(theme::BAR),
    );
    let mut ask = None;
    let _open = egui::Popup::menu(&response).show(|ui| {
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

fn button_at(
    ui: &mut egui::Ui,
    at: egui::Pos2,
    width: f32,
    label: &str,
    fill: egui::Color32,
    ink: egui::Color32,
) -> Option<()> {
    let rect = Rect::from_min_size(at, vec2(width, 26.0));
    let clicked = ui
        .put(
            rect,
            Button::new(RichText::new(label).color(ink)).fill(fill),
        )
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
        Ask, GuardChoice, PathBuf, ReadError, blank, changed, decode_recent, draft_needed,
        encode_recent, ensure_suffix, file_button, file_title, forget, load_recent, needs_guard,
        read, remember, remove_draft, row_label, show_guard, store_recent, write, write_draft,
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
        let note = std::fs::read_to_string(dir.0.join("autosave").join("42.meta")).expect("note");
        let (name, path) = note_fields(&note).expect("note fields");
        assert_eq!(name, "Cut\nA\\B");
        assert_eq!(path.as_deref(), Some(source.to_str().expect("path")));
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

    fn note_fields(text: &str) -> Option<(String, Option<String>)> {
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
            Some(unescape_note(source)?)
        };
        Some((name, source))
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
