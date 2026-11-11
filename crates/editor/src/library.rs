//! The project library: its files, its folders, and their order.
//!
//! Files come from the draft's picture and sound pools, one card per file
//! path. Folders, where each card and folder sits, and the cards taken out
//! of the library live in the project's [`VirtualStore`]. Every screen lists
//! the library through here, so the phone and the desktop show the same
//! folders in the same order.

use std::cmp::Ordering;
use std::collections::HashSet;

use cutlass_model::{BinFolder, BinLink, BinPath, Draft, Micros, VirtualStore};

/// The top of the library: the parent of everything not in a folder.
pub const LIBRARY_TOP: &str = "";
/// The name a new folder takes, with a number after it once it is used.
pub const NEW_FOLDER_NAME: &str = "Untitled folder";

/// Which pool a library lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryShelf {
    /// Pictures: videos and photos.
    Media,
    Audio,
}

/// The order of the files in a folder. Folders always go by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LibrarySort {
    #[default]
    Name,
    /// Shortest first.
    Duration,
}

/// One file card.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryFile {
    /// The first material with this file. Folders and hiding use it.
    pub id: String,
    pub path: String,
    /// The file's name, or the id when it has no path.
    pub name: String,
    pub duration: Micros,
    /// A clip on the timeline uses this file.
    pub on_timeline: bool,
}

/// The files of `shelf` in `folder`, matching `query`, in `sort` order.
///
/// A file that several materials share is one card. A hidden card stays
/// out. With a query the search reaches into every folder, as the files
/// then show wherever they sit.
#[must_use]
pub fn library_files(
    draft: &Draft,
    store: &VirtualStore,
    shelf: LibraryShelf,
    folder: &str,
    sort: LibrarySort,
    query: &str,
) -> Vec<LibraryFile> {
    let pool: Vec<(&str, &str, Micros)> = match shelf {
        LibraryShelf::Media => draft
            .materials
            .videos
            .iter()
            .map(|video| (video.id.as_str(), video.path.as_str(), video.duration))
            .collect(),
        LibraryShelf::Audio => draft
            .materials
            .audios
            .iter()
            .map(|audio| (audio.id.as_str(), audio.path.as_str(), audio.duration))
            .collect(),
    };
    let used = used_files(draft);
    let needle = query.trim().to_lowercase();
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for (id, path, duration) in pool {
        if !path.is_empty() && !seen.insert(path) {
            continue;
        }
        if is_hidden(store, id, path) {
            continue;
        }
        if needle.is_empty() && parent_of(store, id) != folder {
            continue;
        }
        let name = file_name(path, id);
        if !needle.is_empty() && !name.to_lowercase().contains(&needle) {
            continue;
        }
        files.push(LibraryFile {
            id: id.to_owned(),
            path: path.to_owned(),
            on_timeline: used.contains(id) || (!path.is_empty() && used.contains(path)),
            name,
            duration,
        });
    }
    files.sort_by(|a, b| match sort {
        LibrarySort::Name => by_name(&a.name, &b.name),
        LibrarySort::Duration => a.duration.cmp(&b.duration),
    });
    files
}

/// The folders directly in `parent`, by name. With a query, every folder
/// whose name has it.
#[must_use]
pub fn library_folders<'a>(
    store: &'a VirtualStore,
    parent: &str,
    query: &str,
) -> Vec<&'a BinFolder> {
    let needle = query.trim().to_lowercase();
    let mut folders: Vec<&BinFolder> = store
        .folders
        .iter()
        .filter(|folder| !folder.id.is_empty())
        .filter(|folder| {
            if needle.is_empty() {
                parent_of(store, &folder.id) == parent
            } else {
                folder.display_name.to_lowercase().contains(&needle)
            }
        })
        .collect();
    folders.sort_by(|a, b| by_name(&a.display_name, &b.display_name));
    folders
}

/// The folders from the top down to `folder`, for the path bar. Empty at
/// the top or for a folder that is gone.
#[must_use]
pub fn folder_trail<'a>(store: &'a VirtualStore, folder: &str) -> Vec<&'a BinFolder> {
    let mut trail = Vec::new();
    let mut at = folder;
    while let Some(found) = find_folder(store, at) {
        if trail.iter().any(|seen: &&BinFolder| seen.id == found.id) {
            break;
        }
        trail.push(found);
        at = parent_of(store, &found.id);
    }
    trail.reverse();
    trail
}

/// The folder `id` sits in, or [`LIBRARY_TOP`].
#[must_use]
pub fn parent_of<'a>(store: &'a VirtualStore, id: &str) -> &'a str {
    store
        .links
        .iter()
        .find(|link| link.child_id == id)
        .map_or(LIBRARY_TOP, |link| link.parent_id.as_str())
}

/// The files in the picked cards and folders, as material ids: a card is
/// its own file, a folder is every file in it and in the folders inside
/// it, pictures before sounds and each by name. A file picked twice is
/// listed once.
#[must_use]
pub fn picked_files(draft: &Draft, store: &VirtualStore, picked: &[String]) -> Vec<String> {
    fn inside(
        draft: &Draft,
        store: &VirtualStore,
        folder: &str,
        depth: usize,
        out: &mut Vec<String>,
    ) {
        for shelf in [LibraryShelf::Media, LibraryShelf::Audio] {
            out.extend(
                library_files(draft, store, shelf, folder, LibrarySort::Name, "")
                    .into_iter()
                    .map(|file| file.id),
            );
        }
        // A folder linked into itself must not be walked for ever.
        if depth >= store.folders.len() {
            return;
        }
        for child in library_folders(store, folder, "") {
            inside(draft, store, &child.id, depth + 1, out);
        }
    }
    let mut files = Vec::new();
    for id in picked {
        if find_folder(store, id).is_some() {
            inside(draft, store, id, 0, &mut files);
        } else if crate::material_span(draft, id).is_some() {
            files.push(id.clone());
        }
    }
    let mut seen = HashSet::new();
    files.retain(|id| seen.insert(id.clone()));
    files
}

/// A cut of the library files `files` alone, for exporting them: the
/// project with its timeline replaced by the pictures end to end on the
/// main track and the sounds end to end on a sound track, both from the
/// start. `None` when none of the files has a length to play.
#[must_use]
pub fn library_cut(draft: &Draft, files: &[String]) -> Option<Draft> {
    use cutlass_model::{Segment, TimeRange, Track, TrackType};
    let mut pictures = Track::main("library-pictures");
    let mut sounds = Track::main("library-sounds");
    sounds.kind = TrackType::Audio;
    for (index, id) in files.iter().enumerate() {
        let Some((kind, length)) = crate::material_span(draft, id) else {
            continue;
        };
        if length.0 <= 0 {
            continue;
        }
        let track = if kind == TrackType::Audio {
            &mut sounds
        } else {
            &mut pictures
        };
        let start = track.segments.last().map_or(0, |last: &Segment| {
            last.target.start.0 + last.target.duration.0
        });
        let mut segment = Segment::new(
            format!("library-clip-{index}"),
            id.as_str(),
            TimeRange::new(start, length.0),
        );
        segment.source_range = Some(TimeRange::new(0, length.0));
        track.segments.push(segment);
    }
    let end = |track: &Track| {
        track
            .segments
            .last()
            .map_or(0, |last| last.target.start.0 + last.target.duration.0)
    };
    let length = end(&pictures).max(end(&sounds));
    if length <= 0 {
        return None;
    }
    let mut cut = draft.clone();
    cut.tracks = vec![pictures];
    if !sounds.segments.is_empty() {
        cut.tracks.push(sounds);
    }
    cut.relationships.clear();
    cut.duration = Micros(length);
    Some(cut)
}

/// "Untitled folder", or the first "Untitled folder N" from 2 not taken.
#[must_use]
pub fn next_folder_name(store: &VirtualStore) -> String {
    let taken: HashSet<&str> = store
        .folders
        .iter()
        .map(|folder| folder.display_name.as_str())
        .collect();
    if !taken.contains(NEW_FOLDER_NAME) {
        return NEW_FOLDER_NAME.to_owned();
    }
    (2_u32..)
        .map(|index| format!("{NEW_FOLDER_NAME} {index}"))
        .find(|name| !taken.contains(name.as_str()))
        .unwrap_or_else(|| NEW_FOLDER_NAME.to_owned())
}

/// Make the folder `id` in `parent`, with the next free name, and move
/// `with` into it when given. `now` is seconds since 1970. Returns the
/// folder's name.
pub fn add_folder(
    store: &mut VirtualStore,
    id: &str,
    parent: &str,
    with: &[&str],
    now: i64,
) -> String {
    let name = next_folder_name(store);
    store.folders.push(BinFolder::new(id, name.clone(), now));
    place(store, id, parent);
    for child in with {
        place(store, child, id);
    }
    name
}

/// Name a folder. A blank name keeps the old one. Returns whether it
/// changed.
pub fn rename_folder(store: &mut VirtualStore, id: &str, name: &str) -> bool {
    let name = name.trim();
    let Some(folder) = store.folders.iter_mut().find(|folder| folder.id == id) else {
        return false;
    };
    if name.is_empty() || folder.display_name == name {
        return false;
    }
    folder.display_name = name.to_owned();
    true
}

/// Delete a folder and the folders inside it. The files in them move up to
/// the deleted folder's parent; nothing leaves the project. Returns that
/// parent, or `None` when there was no such folder.
pub fn delete_folder(store: &mut VirtualStore, id: &str) -> Option<String> {
    find_folder(store, id)?;
    let parent = parent_of(store, id).to_owned();
    let gone = folders_under(store, id);
    store
        .folders
        .retain(|folder| !gone.contains(folder.id.as_str()));
    store
        .links
        .retain(|link| !gone.contains(link.child_id.as_str()));
    for link in &mut store.links {
        if gone.contains(link.parent_id.as_str()) {
            link.parent_id.clone_from(&parent);
        }
    }
    Some(parent)
}

/// Move a file or folder into `parent`. A folder never goes into itself or
/// a folder inside it, and a parent that is not a folder is the top.
/// Returns whether it moved.
pub fn place(store: &mut VirtualStore, child: &str, parent: &str) -> bool {
    let parent = if find_folder(store, parent).is_some() {
        parent
    } else {
        LIBRARY_TOP
    };
    if find_folder(store, child).is_some() && folders_under(store, child).contains(parent) {
        return false;
    }
    if parent_of(store, child) == parent
        && (parent != LIBRARY_TOP || store.links.iter().any(|link| link.child_id == child))
    {
        return false;
    }
    store.links.retain(|link| link.child_id != child);
    if parent != LIBRARY_TOP {
        store.links.push(BinLink {
            child_id: child.to_owned(),
            parent_id: parent.to_owned(),
            ..BinLink::default()
        });
    }
    true
}

/// Take a file card out of the library. Its clips stay on the timeline and
/// its materials stay in the draft.
pub fn hide_file(store: &mut VirtualStore, id: &str, path: &str) {
    store.links.retain(|link| link.child_id != id);
    if let Some(row) = store.paths.iter_mut().find(|row| row.id == id) {
        row.is_hidden = true;
        row.path = path.to_owned();
        return;
    }
    store.paths.push(BinPath {
        id: id.to_owned(),
        path: path.to_owned(),
        is_hidden: true,
        ..BinPath::default()
    });
}

fn is_hidden(store: &VirtualStore, id: &str, path: &str) -> bool {
    store
        .paths
        .iter()
        .any(|row| row.is_hidden && (row.id == id || (!path.is_empty() && row.path == path)))
}

fn find_folder<'a>(store: &'a VirtualStore, id: &str) -> Option<&'a BinFolder> {
    if id.is_empty() {
        return None;
    }
    store.folders.iter().find(|folder| folder.id == id)
}

/// `id` and every folder inside it, however deep.
fn folders_under(store: &VirtualStore, id: &str) -> HashSet<String> {
    let mut gone = HashSet::from([id.to_owned()]);
    let mut grew = true;
    while grew {
        grew = false;
        for folder in &store.folders {
            if !gone.contains(&folder.id) && gone.contains(parent_of(store, &folder.id)) {
                gone.insert(folder.id.clone());
                grew = true;
            }
        }
    }
    gone
}

/// Files and materials a clip uses, by material id and by file path.
fn used_files(draft: &Draft) -> HashSet<&str> {
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
    draft
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
        .collect()
}

fn file_name(path: &str, id: &str) -> String {
    if path.is_empty() {
        return id.to_owned();
    }
    std::path::Path::new(path).file_name().map_or_else(
        || path.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Names in reading order: case set aside, then as written.
fn by_name(a: &str, b: &str) -> Ordering {
    a.to_lowercase()
        .cmp(&b.to_lowercase())
        .then_with(|| a.cmp(b))
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialAudio, MaterialId, MaterialVideo, Segment, TimeRange};

    use super::*;

    fn video(id: &str, path: &str, seconds: i64) -> MaterialVideo {
        MaterialVideo {
            id: MaterialId::new(id),
            path: path.to_owned(),
            duration: Micros(seconds * 1_000_000),
            ..MaterialVideo::default()
        }
    }

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.materials.videos = vec![
            video("b", "/m/beach.mov", 9),
            video("a", "/m/Alps.mov", 3),
            video("a-copy", "/m/Alps.mov", 3),
            video("c", "/m/city.mov", 5),
        ];
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("s"),
            path: "/m/song.m4a".to_owned(),
            ..MaterialAudio::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("clip", "a-copy", TimeRange::new(0, 1_000_000)));
        draft
    }

    fn names(files: &[LibraryFile]) -> Vec<&str> {
        files.iter().map(|file| file.name.as_str()).collect()
    }

    #[test]
    fn one_card_per_file_sorted_by_name_or_by_length() {
        let draft = draft();
        let store = VirtualStore::default();
        let list = |sort| library_files(&draft, &store, LibraryShelf::Media, LIBRARY_TOP, sort, "");
        let by_name = list(LibrarySort::Name);
        assert_eq!(names(&by_name), ["Alps.mov", "beach.mov", "city.mov"]);
        // A clip of the second material with that file marks the card.
        assert!(by_name[0].on_timeline);
        assert_eq!(by_name[0].id, "a");
        assert!(!by_name[1].on_timeline);
        assert_eq!(
            names(&list(LibrarySort::Duration)),
            ["Alps.mov", "city.mov", "beach.mov"]
        );
        let sounds = library_files(
            &draft,
            &store,
            LibraryShelf::Audio,
            "",
            LibrarySort::Name,
            "",
        );
        assert_eq!(names(&sounds), ["song.m4a"]);
    }

    #[test]
    fn picked_cards_and_folders_become_one_cut_end_to_end() {
        let mut draft = draft();
        draft.materials.audios[0].duration = Micros(4_000_000);
        let mut store = VirtualStore::default();
        add_folder(&mut store, "f1", LIBRARY_TOP, &["b", "s"], 1);
        add_folder(&mut store, "f2", "f1", &["c"], 1);
        let ids = |picked: &[&str]| {
            let picked: Vec<String> = picked.iter().map(|id| (*id).to_owned()).collect();
            picked_files(&draft, &store, &picked)
        };
        // A card is itself; a folder is all it holds, all the way down;
        // a file picked twice counts once; an unknown id is nothing.
        assert_eq!(ids(&["a"]), ["a"]);
        assert_eq!(ids(&["f1"]), ["b", "s", "c"]);
        assert_eq!(ids(&["c", "f1", "gone"]), ["c", "b", "s"]);
        assert!(ids(&[]).is_empty());

        let files = ids(&["a", "f1"]);
        let cut = library_cut(&draft, &files).unwrap();
        let spans = |track: &cutlass_model::Track| -> Vec<(String, i64, i64)> {
            track
                .segments
                .iter()
                .map(|segment| {
                    (
                        segment.material_id.as_str().to_owned(),
                        segment.target.start.0,
                        segment.target.duration.0,
                    )
                })
                .collect()
        };
        assert_eq!(cut.tracks.len(), 2);
        assert!(cut.tracks[0].is_main());
        assert_eq!(
            spans(&cut.tracks[0]),
            [
                ("a".to_owned(), 0, 3_000_000),
                ("b".to_owned(), 3_000_000, 9_000_000),
                ("c".to_owned(), 12_000_000, 5_000_000),
            ]
        );
        assert_eq!(cut.tracks[1].kind, cutlass_model::TrackType::Audio);
        assert_eq!(spans(&cut.tracks[1]), [("s".to_owned(), 0, 4_000_000)]);
        assert_eq!(cut.duration, Micros(17_000_000));
        // Every clip plays its whole file, and the timeline's own clip is
        // not in the cut.
        for segment in cut.tracks.iter().flat_map(|track| &track.segments) {
            assert_eq!(
                segment.source_range,
                Some(TimeRange::new(0, segment.target.duration.0))
            );
            assert_ne!(segment.id.as_str(), "clip");
        }
        assert_eq!(cut.materials, draft.materials);
        assert_eq!(cut.canvas_config, draft.canvas_config);
        // The project itself is as it was.
        assert_eq!(draft.tracks[0].segments.len(), 1);

        // Pictures alone make no sound track; nothing to play makes no cut.
        assert_eq!(
            library_cut(&draft, &["a".to_owned()]).unwrap().tracks.len(),
            1
        );
        draft.materials.audios[0].duration = Micros(0);
        assert!(library_cut(&draft, &["s".to_owned(), "gone".to_owned()]).is_none());
        assert!(library_cut(&draft, &[]).is_none());
    }

    #[test]
    fn a_folder_linked_into_itself_is_walked_once() {
        let draft = draft();
        let mut store = VirtualStore::default();
        add_folder(&mut store, "f1", LIBRARY_TOP, &["b"], 1);
        add_folder(&mut store, "f2", "f1", &["c"], 1);
        // A hostile store: the outer folder is also inside the inner one.
        store.links.push(BinLink {
            child_id: "f1".to_owned(),
            parent_id: "f2".to_owned(),
            ..BinLink::default()
        });
        let files = picked_files(&draft, &store, &["f2".to_owned()]);
        assert!(files.len() <= 2, "{files:?}");
        assert!(files.contains(&"c".to_owned()));
    }

    #[test]
    fn folders_hold_files_and_folders_and_the_trail_leads_down() {
        let draft = draft();
        let mut store = VirtualStore::default();
        assert_eq!(
            add_folder(&mut store, "f1", LIBRARY_TOP, &["b"], 1),
            "Untitled folder"
        );
        assert_eq!(
            add_folder(&mut store, "f2", "f1", &[], 1),
            "Untitled folder 2"
        );
        assert!(rename_folder(&mut store, "f2", " Takes "));
        assert!(!rename_folder(&mut store, "f2", "  "));
        let top = library_files(
            &draft,
            &store,
            LibraryShelf::Media,
            "",
            LibrarySort::Name,
            "",
        );
        assert_eq!(names(&top), ["Alps.mov", "city.mov"]);
        let inside = library_files(
            &draft,
            &store,
            LibraryShelf::Media,
            "f1",
            LibrarySort::Name,
            "",
        );
        assert_eq!(names(&inside), ["beach.mov"]);
        let folder_names = |parent| {
            library_folders(&store, parent, "")
                .iter()
                .map(|folder| folder.display_name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(folder_names(""), ["Untitled folder"]);
        assert_eq!(folder_names("f1"), ["Takes"]);
        let trail: Vec<_> = folder_trail(&store, "f2")
            .iter()
            .map(|f| f.id.as_str())
            .collect();
        assert_eq!(trail, ["f1", "f2"]);
        // A search looks in every folder.
        let found = library_files(
            &draft,
            &store,
            LibraryShelf::Media,
            "",
            LibrarySort::Name,
            "BEA",
        );
        assert_eq!(names(&found), ["beach.mov"]);
        assert_eq!(library_folders(&store, "", "tak").len(), 1);
    }

    #[test]
    fn a_folder_cannot_go_inside_itself_and_a_move_to_the_top_drops_the_link() {
        let mut store = VirtualStore::default();
        add_folder(&mut store, "f1", "", &[], 1);
        add_folder(&mut store, "f2", "f1", &[], 1);
        assert!(!place(&mut store, "f1", "f2"));
        assert!(!place(&mut store, "f1", "f1"));
        assert!(place(&mut store, "clip", "f2"));
        assert!(!place(&mut store, "clip", "f2"));
        assert!(place(&mut store, "clip", "missing"));
        assert_eq!(parent_of(&store, "clip"), LIBRARY_TOP);
        assert!(store.links.iter().all(|link| link.child_id != "clip"));
    }

    #[test]
    fn deleting_a_folder_takes_its_folders_and_lifts_its_files() {
        let draft = draft();
        let mut store = VirtualStore::default();
        add_folder(&mut store, "f1", "", &[], 1);
        add_folder(&mut store, "f2", "f1", &["b"], 1);
        add_folder(&mut store, "f3", "f1", &["c"], 1);
        assert!(place(&mut store, "f3", ""));
        assert_eq!(delete_folder(&mut store, "f1").as_deref(), Some(""));
        assert_eq!(delete_folder(&mut store, "f1"), None);
        let ids: Vec<_> = store.folders.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(ids, ["f3"]);
        let top = library_files(
            &draft,
            &store,
            LibraryShelf::Media,
            "",
            LibrarySort::Name,
            "",
        );
        assert_eq!(names(&top), ["Alps.mov", "beach.mov"]);
    }

    #[test]
    fn a_hidden_card_leaves_the_library_but_not_the_draft() {
        let draft = draft();
        let mut store = VirtualStore::default();
        add_folder(&mut store, "f1", "", &["c"], 1);
        hide_file(&mut store, "c", "/m/city.mov");
        hide_file(&mut store, "c", "/m/city.mov");
        assert_eq!(store.paths.len(), 1);
        assert!(store.links.iter().all(|link| link.child_id != "c"));
        let inside = library_files(
            &draft,
            &store,
            LibraryShelf::Media,
            "f1",
            LibrarySort::Name,
            "",
        );
        assert!(inside.is_empty());
        let all = library_files(
            &draft,
            &store,
            LibraryShelf::Media,
            "",
            LibrarySort::Name,
            "mov",
        );
        assert_eq!(names(&all), ["Alps.mov", "beach.mov"]);
        assert_eq!(draft.materials.videos.len(), 4);
    }

    #[test]
    fn the_top_folder_row_is_not_listed_and_a_trail_with_a_loop_ends() {
        let mut store = VirtualStore::default();
        store.folders.push(BinFolder::default());
        assert!(library_folders(&store, "", "").is_empty());
        store.folders.push(BinFolder::new("x", "X", 0));
        store.folders.push(BinFolder::new("y", "Y", 0));
        store.links.push(BinLink {
            child_id: "x".to_owned(),
            parent_id: "y".to_owned(),
            ..BinLink::default()
        });
        store.links.push(BinLink {
            child_id: "y".to_owned(),
            parent_id: "x".to_owned(),
            ..BinLink::default()
        });
        assert_eq!(folder_trail(&store, "x").len(), 2);
    }
}
