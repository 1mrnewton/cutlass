//! Folders, sort, and selection for the media bin. `docs/library.md`.
//!
//! A folder, which folder a file sits in, a hidden file, and the sort are view
//! state. They are not project fields.

use eframe::egui::{Pos2, Rect};

use cutlass_project::AssetId;

/// A card in the open folder. Folders draw before files.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CardId {
    Folder(u64),
    File(AssetId),
}

/// One order for every folder in the bin. Name is the start.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sort {
    Name,
    Duration,
}

/// A project file the bin can show. The name is borrowed from the asset.
pub(crate) struct FileInfo<'a> {
    pub id: AssetId,
    pub name: &'a str,
    pub duration_us: Option<i64>,
}

#[derive(Clone, Debug)]
struct Folder {
    id: u64,
    name: String,
    parent: Option<u64>,
}

/// The open folder, the selection, and where each file sits.
pub(crate) struct Bin {
    folders: Vec<Folder>,
    next: u64,
    places: Vec<(AssetId, u64)>,
    hidden: Vec<AssetId>,
    open: Option<u64>,
    sort: Sort,
    selected: Vec<CardId>,
    anchor: Option<CardId>,
    rename: Option<u64>,
    rename_text: String,
    focus_rename: bool,
    box_origin: Option<Pos2>,
    box_base: Vec<CardId>,
    box_shift: bool,
    box_live: bool,
}

/// A drag across empty space becomes a box once the pointer has moved this far.
const BOX_DRAG: f32 = 4.0;

impl Bin {
    pub(crate) fn new() -> Self {
        Self {
            folders: Vec::new(),
            next: 1,
            places: Vec::new(),
            hidden: Vec::new(),
            open: None,
            sort: Sort::Name,
            selected: Vec::new(),
            anchor: None,
            rename: None,
            rename_text: String::new(),
            focus_rename: false,
            box_origin: None,
            box_base: Vec::new(),
            box_shift: false,
            box_live: false,
        }
    }

    pub(crate) fn open(&self) -> Option<u64> {
        self.open
    }

    pub(crate) fn is_selected(&self, id: CardId) -> bool {
        self.selected.contains(&id)
    }

    pub(crate) fn is_renaming(&self, id: u64) -> bool {
        self.rename == Some(id)
    }

    pub(crate) fn rename_text_mut(&mut self) -> &mut String {
        &mut self.rename_text
    }

    pub(crate) fn folder_name(&self, id: u64) -> Option<&str> {
        self.folders
            .iter()
            .find(|folder| folder.id == id)
            .map(|folder| folder.name.as_str())
    }

    /// Root, then each folder down to the open one.
    pub(crate) fn trail(&self) -> Vec<u64> {
        let mut chain = Vec::new();
        let mut cursor = self.open;
        let mut guard = 0;
        while let Some(id) = cursor {
            if guard > self.folders.len() {
                break;
            }
            guard += 1;
            chain.push(id);
            cursor = self
                .folders
                .iter()
                .find(|folder| folder.id == id)
                .and_then(|folder| folder.parent);
        }
        chain.reverse();
        chain
    }

    pub(crate) fn place_of(&self, id: AssetId) -> Option<u64> {
        self.places
            .iter()
            .find(|(asset, _)| *asset == id)
            .map(|(_, folder)| *folder)
    }

    /// Folders in the open folder, by name, then its files in the bin's sort.
    pub(crate) fn visible(&self, files: &[FileInfo<'_>]) -> Vec<CardId> {
        let mut folders: Vec<&Folder> = self
            .folders
            .iter()
            .filter(|folder| folder.parent == self.open)
            .collect();
        folders.sort_by(|left, right| cmp_name(&left.name, &right.name));
        let mut rows: Vec<&FileInfo<'_>> = files
            .iter()
            .filter(|file| !self.hidden.contains(&file.id) && self.place_of(file.id) == self.open)
            .collect();
        match self.sort {
            Sort::Name => rows.sort_by(|left, right| cmp_name(left.name, right.name)),
            Sort::Duration => rows.sort_by(|left, right| cmp_duration(left, right)),
        }
        let mut cards = Vec::with_capacity(folders.len() + rows.len());
        cards.extend(folders.iter().map(|folder| CardId::Folder(folder.id)));
        cards.extend(rows.iter().map(|file| CardId::File(file.id)));
        cards
    }

    /// Files with no folders. A file inside a folder still shows. A hidden
    /// file stays out. The bin's sort still applies.
    pub(crate) fn flat(&self, files: &[FileInfo<'_>]) -> Vec<CardId> {
        let mut rows: Vec<&FileInfo<'_>> = files
            .iter()
            .filter(|file| !self.hidden.contains(&file.id))
            .collect();
        match self.sort {
            Sort::Name => rows.sort_by(|left, right| cmp_name(left.name, right.name)),
            Sort::Duration => rows.sort_by(|left, right| cmp_duration(left, right)),
        }
        rows.into_iter().map(|file| CardId::File(file.id)).collect()
    }

    /// Shift extends from the last plain click. Command toggles. Both together
    /// add that range to the selection.
    pub(crate) fn choose(&mut self, order: &[CardId], id: CardId, shift: bool, toggle: bool) {
        if shift
            && let Some(anchor) = self.anchor
            && let Some(from) = order.iter().position(|card| *card == anchor)
            && let Some(to) = order.iter().position(|card| *card == id)
        {
            let (start, end) = if from <= to { (from, to) } else { (to, from) };
            let range = &order[start..=end];
            if toggle {
                for card in range {
                    if !self.selected.contains(card) {
                        self.selected.push(*card);
                    }
                }
            } else {
                self.selected = range.to_vec();
            }
            return;
        }
        if toggle {
            if let Some(index) = self.selected.iter().position(|card| *card == id) {
                self.selected.remove(index);
            } else {
                self.selected.push(id);
            }
            self.anchor = Some(id);
            return;
        }
        self.selected = vec![id];
        self.anchor = Some(id);
    }

    pub(crate) fn select_only(&mut self, id: CardId) {
        self.selected = vec![id];
        self.anchor = Some(id);
    }

    pub(crate) fn clear(&mut self) {
        self.selected.clear();
        self.anchor = None;
    }

    pub(crate) fn open_place(&mut self, place: Option<u64>) {
        if place.is_some_and(|id| self.folder_name(id).is_none()) {
            return;
        }
        self.commit_rename();
        self.open = place;
    }

    /// A plain click opens the folder and selects that card.
    pub(crate) fn open_folder(&mut self, id: u64) {
        self.open_place(Some(id));
        let card = CardId::Folder(id);
        self.selected = vec![card];
        self.anchor = Some(card);
    }

    pub(crate) fn set_sort(&mut self, sort: Sort) {
        self.sort = sort;
    }

    /// The new folder is in the open folder, selected, and ready to rename.
    pub(crate) fn add_folder(&mut self, base: &str) -> u64 {
        self.commit_rename();
        let id = self.next;
        self.next = self.next.saturating_add(1);
        let name = unused_name(&self.folders, base);
        self.folders.push(Folder {
            id,
            name: name.clone(),
            parent: self.open,
        });
        let card = CardId::Folder(id);
        self.selected = vec![card];
        self.anchor = Some(card);
        self.rename = Some(id);
        self.rename_text = name;
        self.focus_rename = true;
        id
    }

    /// A folder in the open folder, holding this one file.
    pub(crate) fn add_folder_with(&mut self, file: AssetId, base: &str) -> u64 {
        let id = self.add_folder(base);
        self.move_file(file, Some(id));
        id
    }

    pub(crate) fn begin_rename(&mut self, id: u64) {
        if self.rename == Some(id) {
            self.focus_rename = true;
            return;
        }
        self.commit_rename();
        let Some(name) = self
            .folders
            .iter()
            .find(|folder| folder.id == id)
            .map(|folder| folder.name.clone())
        else {
            return;
        };
        self.rename = Some(id);
        self.rename_text = name;
        self.focus_rename = true;
    }

    /// `true` once, on the folder that should take the keyboard.
    pub(crate) fn focus_rename(&mut self, id: u64) -> bool {
        if self.focus_rename && self.rename == Some(id) {
            self.focus_rename = false;
            true
        } else {
            false
        }
    }

    pub(crate) fn commit_rename(&mut self) {
        let Some(id) = self.rename else {
            return;
        };
        let trimmed = self.rename_text.trim();
        if !trimmed.is_empty()
            && let Some(folder) = self.folders.iter_mut().find(|folder| folder.id == id)
        {
            folder.name = trimmed.to_owned();
        }
        self.rename = None;
        self.focus_rename = false;
    }

    pub(crate) fn cancel_rename(&mut self) {
        self.rename = None;
        self.focus_rename = false;
    }

    /// The folder and the folders inside it go away. Their files move up.
    pub(crate) fn delete_folder(&mut self, id: u64) {
        let Some(up) = self
            .folders
            .iter()
            .find(|folder| folder.id == id)
            .map(|folder| folder.parent)
        else {
            return;
        };
        let gone = ids_under(&self.folders, id);
        let mut kept = Vec::with_capacity(self.places.len());
        for (asset, place) in self.places.drain(..) {
            if gone.contains(&place) {
                if let Some(parent) = up {
                    kept.push((asset, parent));
                }
            } else {
                kept.push((asset, place));
            }
        }
        self.places = kept;
        self.folders.retain(|folder| !gone.contains(&folder.id));
        if self.open.is_some_and(|open| gone.contains(&open)) {
            self.open = up;
        }
        self.selected.retain(|card| match card {
            CardId::Folder(folder) => !gone.contains(folder),
            CardId::File(_) => true,
        });
        if self
            .anchor
            .is_some_and(|card| matches!(card, CardId::Folder(folder) if gone.contains(&folder)))
        {
            self.anchor = None;
        }
        if self.rename.is_some_and(|folder| gone.contains(&folder)) {
            self.rename = None;
            self.focus_rename = false;
        }
    }

    /// Hide the card. The asset stays in the project.
    pub(crate) fn hide_file(&mut self, id: AssetId) {
        if !self.hidden.contains(&id) {
            self.hidden.push(id);
        }
        self.selected.retain(|card| *card != CardId::File(id));
        if self.anchor == Some(CardId::File(id)) {
            self.anchor = None;
        }
    }

    pub(crate) fn move_file(&mut self, id: AssetId, parent: Option<u64>) {
        if parent.is_some_and(|folder| self.folder_name(folder).is_none()) {
            return;
        }
        if self.place_of(id) == parent {
            return;
        }
        self.places.retain(|(asset, _)| *asset != id);
        if let Some(folder) = parent {
            self.places.push((id, folder));
        }
    }

    pub(crate) fn begin_box(&mut self, origin: Pos2, shift: bool) {
        self.box_origin = Some(origin);
        self.box_shift = shift;
        self.box_base = if shift {
            self.selected.clone()
        } else {
            Vec::new()
        };
        self.box_live = false;
    }

    /// `None` until the pointer has moved [`BOX_DRAG`]. After that, the box stays.
    pub(crate) fn box_rect(&mut self, now: Pos2) -> Option<Rect> {
        let from = self.box_origin?;
        if !self.box_live && from.distance(now) < BOX_DRAG {
            return None;
        }
        self.box_live = true;
        Some(Rect::from_two_pos(from, now))
    }

    pub(crate) fn apply_box(&mut self, hits: &[CardId]) {
        let mut next = if self.box_shift {
            self.box_base.clone()
        } else {
            Vec::new()
        };
        for id in hits {
            if !next.contains(id) {
                next.push(*id);
            }
        }
        self.anchor = next.last().copied().or(self.anchor);
        self.selected = next;
        self.box_live = true;
    }

    /// A box that never moved is a click on empty space.
    ///
    /// Returns whether the pointer drew a box.
    pub(crate) fn finish_box(&mut self) -> bool {
        if self.box_origin.is_none() {
            return false;
        }
        let moved = self.box_live;
        self.box_origin = None;
        self.box_base.clear();
        self.box_live = false;
        if !moved {
            self.clear();
        }
        moved
    }
}

/// Sequence microseconds. A still has no duration, so the caller passes `None`.
pub(crate) fn duration_us(value: i64, timescale: u32) -> Option<i64> {
    let scale = i128::from(timescale);
    if scale == 0 {
        return None;
    }
    let micros = i128::from(value).max(0) * 1_000_000 / scale;
    i64::try_from(micros).ok()
}

/// Cards the box touches, including a shared edge.
pub(crate) fn cards_touching(cards: &[(CardId, Rect)], area: Rect) -> Vec<CardId> {
    cards
        .iter()
        .filter(|(_, rect)| overlaps(*rect, area))
        .map(|(id, _)| *id)
        .collect()
}

fn overlaps(card: Rect, area: Rect) -> bool {
    !(card.right() < area.left()
        || card.left() > area.right()
        || card.bottom() < area.top()
        || card.top() > area.bottom())
}

fn cmp_name(left: &str, right: &str) -> std::cmp::Ordering {
    left.to_lowercase()
        .cmp(&right.to_lowercase())
        .then_with(|| left.cmp(right))
}

fn cmp_duration(left: &FileInfo<'_>, right: &FileInfo<'_>) -> std::cmp::Ordering {
    match (left.duration_us, right.duration_us) {
        (Some(earlier), Some(later)) => earlier
            .cmp(&later)
            .then_with(|| cmp_name(left.name, right.name)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => cmp_name(left.name, right.name),
    }
}

fn unused_name(folders: &[Folder], base: &str) -> String {
    if folders.iter().all(|folder| folder.name != base) {
        return base.to_owned();
    }
    for index in 2..10_000 {
        let candidate = format!("{base} {index}");
        if folders.iter().all(|folder| folder.name != candidate) {
            return candidate;
        }
    }
    format!("{base} {}", folders.len() + 2)
}

fn ids_under(folders: &[Folder], root: u64) -> Vec<u64> {
    let mut gone = vec![root];
    let mut index = 0;
    while index < gone.len() {
        let parent = gone[index];
        index += 1;
        for folder in folders {
            if folder.parent == Some(parent) && !gone.contains(&folder.id) {
                gone.push(folder.id);
            }
        }
    }
    gone
}

#[cfg(test)]
mod tests {
    use eframe::egui::{Rect, pos2, vec2};

    use super::{Bin, CardId, FileInfo, Sort, cards_touching, duration_us};
    use cutlass_project::AssetId;

    fn file(id: u64) -> AssetId {
        AssetId::new(id).expect("id")
    }

    fn info(id: u64, name: &'static str, duration_us: Option<i64>) -> FileInfo<'static> {
        FileInfo {
            id: file(id),
            name,
            duration_us,
        }
    }

    fn card(id: u64) -> CardId {
        CardId::File(file(id))
    }

    fn rename(bin: &mut Bin, id: u64, name: &str) {
        bin.begin_rename(id);
        *bin.rename_text_mut() = name.to_owned();
        bin.commit_rename();
    }

    #[test]
    fn folders_lead_and_files_follow_the_sort() {
        let mut bin = Bin::new();
        let bee = bin.add_folder("Untitled folder");
        rename(&mut bin, bee, "b");
        let ay = bin.add_folder("Untitled folder");
        rename(&mut bin, ay, "A");
        let files = [
            info(1, "b", Some(2_000_000)),
            info(2, "a", Some(1_000_000)),
            info(3, "m", None),
            info(6, "c", Some(500_000)),
        ];
        assert_eq!(
            bin.visible(&files),
            vec![
                CardId::Folder(ay),
                CardId::Folder(bee),
                card(2),
                card(1),
                card(6),
                card(3),
            ]
        );
        bin.set_sort(Sort::Duration);
        assert_eq!(
            bin.visible(&files),
            vec![
                CardId::Folder(ay),
                CardId::Folder(bee),
                card(6),
                card(2),
                card(1),
                card(3),
            ]
        );
        let mut ties = Bin::new();
        ties.set_sort(Sort::Duration);
        let tied = [info(4, "Zed", Some(5)), info(5, "ada", Some(5))];
        assert_eq!(ties.visible(&tied), vec![card(5), card(4)]);
    }

    #[test]
    fn a_file_in_another_folder_or_a_hidden_file_stays_out() {
        let mut bin = Bin::new();
        let folder = bin.add_folder("Untitled folder");
        bin.commit_rename();
        bin.move_file(file(1), Some(folder));
        bin.hide_file(file(2));
        let files = [
            info(1, "inside", Some(500_000)),
            info(2, "gone", Some(1)),
            info(3, "here", Some(2_000_000)),
        ];
        assert_eq!(bin.visible(&files), vec![CardId::Folder(folder), card(3)]);
        bin.open_folder(folder);
        assert_eq!(bin.visible(&files), vec![card(1)]);
        assert_eq!(bin.trail(), vec![folder]);
        assert_eq!(bin.flat(&files), vec![card(3), card(1)]);
        bin.set_sort(Sort::Duration);
        assert_eq!(bin.flat(&files), vec![card(1), card(3)]);
    }

    #[test]
    fn shift_extends_command_toggles_and_both_add_the_range() {
        let mut bin = Bin::new();
        let order = [card(1), card(2), card(3)];
        bin.choose(&order, card(1), false, false);
        bin.choose(&order, card(3), true, false);
        assert_eq!(&bin.selected, &[card(1), card(2), card(3)]);
        bin.choose(&order, card(2), true, false);
        assert_eq!(&bin.selected, &[card(1), card(2)]);
        bin.choose(&order, card(2), false, true);
        assert_eq!(&bin.selected, &[card(1)]);
        bin.choose(&order, card(2), false, true);
        assert_eq!(&bin.selected, &[card(1), card(2)]);
        bin.choose(&order, card(3), true, true);
        assert_eq!(&bin.selected, &[card(1), card(2), card(3)]);
    }

    #[test]
    fn shift_without_an_anchor_in_view_selects_the_card() {
        let mut bin = Bin::new();
        bin.choose(&[card(1)], card(1), true, false);
        assert_eq!(&bin.selected, &[card(1)]);
    }

    #[test]
    fn a_short_press_clears_and_a_box_keeps_what_it_touches() {
        let mut bin = Bin::new();
        bin.choose(&[card(1)], card(1), false, false);
        bin.begin_box(pos2(0.0, 0.0), false);
        assert!(bin.box_rect(pos2(3.9, 0.0)).is_none());
        assert!(!bin.finish_box());
        assert!(bin.selected.is_empty());

        bin.choose(&[card(1), card(2)], card(1), false, false);
        bin.begin_box(pos2(0.0, 0.0), true);
        let rect = bin.box_rect(pos2(4.0, 0.0)).expect("box");
        assert!(bin.box_rect(pos2(1.0, 0.0)).is_some());
        let cards = [
            (
                card(2),
                Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)),
            ),
            (
                card(3),
                Rect::from_min_size(pos2(20.0, 0.0), vec2(10.0, 10.0)),
            ),
        ];
        bin.apply_box(&cards_touching(&cards, rect));
        assert!(bin.finish_box());
        assert_eq!(&bin.selected, &[card(1), card(2)]);

        let edge = Rect::from_min_size(pos2(10.0, 0.0), vec2(4.0, 4.0));
        let gap = Rect::from_min_size(pos2(10.1, 0.0), vec2(4.0, 4.0));
        assert_eq!(cards_touching(&cards, edge), vec![card(2)]);
        assert!(cards_touching(&cards, gap).is_empty());
    }

    #[test]
    fn deleting_a_folder_lifts_its_files_and_names_count_across_the_bin() {
        let mut bin = Bin::new();
        let parent = bin.add_folder("Untitled folder");
        bin.open_folder(parent);
        let child = bin.add_folder("Untitled folder");
        assert_eq!(bin.folder_name(child), Some("Untitled folder 2"));
        bin.move_file(file(1), Some(child));
        bin.move_file(file(2), Some(parent));
        bin.open_folder(child);
        bin.delete_folder(parent);
        assert!(bin.folder_name(parent).is_none());
        assert!(bin.folder_name(child).is_none());
        assert_eq!(bin.open(), None);
        assert_eq!(bin.place_of(file(1)), None);
        assert_eq!(bin.place_of(file(2)), None);
        let again = bin.add_folder("Untitled folder");
        assert_eq!(bin.folder_name(again), Some("Untitled folder"));
    }

    #[test]
    fn a_folder_for_one_file_holds_that_file_and_an_empty_name_is_kept() {
        let mut bin = Bin::new();
        let folder = bin.add_folder_with(file(7), "Untitled folder");
        bin.open_folder(folder);
        assert_eq!(bin.visible(&[info(7, "clip", Some(1))]), vec![card(7)]);
        bin.begin_rename(folder);
        bin.rename_text_mut().clear();
        bin.commit_rename();
        assert_eq!(bin.folder_name(folder), Some("Untitled folder"));
        bin.begin_rename(folder);
        bin.rename_text_mut().clone_from(&"Cuts".to_owned());
        bin.cancel_rename();
        assert_eq!(bin.folder_name(folder), Some("Untitled folder"));
        bin.move_file(file(7), Some(folder));
        assert_eq!(bin.place_of(file(7)), Some(folder));
        bin.move_file(file(7), None);
        assert_eq!(bin.place_of(file(7)), None);
    }

    #[test]
    fn duration_uses_the_sequence_clock() {
        assert_eq!(duration_us(1_500, 1_000), Some(1_500_000));
        assert_eq!(duration_us(30_000, 30_000), Some(1_000_000));
        assert_eq!(duration_us(1, 0), None);
    }
}
