//! The project folder.
//!
//! The loaded cut is `Timelines/<main id>/draft_info.json`. The top
//! `draft_info.json` is a byte copy of that file, written on save. A folder
//! with no `project.json` is one timeline: open copies the top file into
//! `Timelines/<id>/` and writes `project.json`. The top file replaces that
//! copy only when it is newer and `last_modified_platform.os` is outside
//! `windows`, `mac`, and `macOS`. Any other difference is reported and the
//! timeline folder is what loads. A path that resolves outside the folder
//! is refused.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use cutlass_model::Draft;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::StoreError;
use crate::backup::copy_backup;
use crate::json::{parse_draft, parse_value, write_draft, write_value};
use crate::repair::{RepairNote, repair};
use crate::upgrade::upgrade;

const DRAFT_FILE: &str = "draft_info.json";
const PROJECT_FILE: &str = "project.json";
const LAYOUT_FILE: &str = "timeline_layout.json";
const META_FILE: &str = "draft_meta_info.json";
const TIMELINES_DIR: &str = "Timelines";
const ATTACHMENT_DIR: &str = "common_attachment";
const ID_MAP_FILE: &str = "attachment_id_mapping.json";
const RESOURCES_DIR: &str = "Resources";
const BIZ_CONFIG_FILE: &str = "draft_biz_config.json";
const DESKTOP_OS: &[&str] = &["windows", "mac", "macOS"];

fn default_next_index() -> i64 {
    1000
}

/// A folder that has been opened: the files, the upgraded cut, and the report.
#[derive(Clone, Debug, PartialEq)]
pub struct Opened {
    pub folder: Folder,
    pub report: Vec<RepairNote>,
}

/// Parse the folder, upgrade every loaded draft, then report on the main timeline.
pub fn open(path: &Path) -> Result<Opened, StoreError> {
    let mut folder = Folder::read(path)?;
    let root = canonicalize_root(path)?;
    let mut report = Vec::new();
    if root.join(PROJECT_FILE).is_file() {
        reconcile_main_copy(&root, &mut folder, &mut report)?;
    } else {
        folder.draft = upgrade(&folder.draft)?;
        promote_single_timeline(&root, &mut folder)?;
    }
    report.extend(skip_id_mismatches(&mut folder)?);
    load_main_timeline(&mut folder)?;
    upgrade_loaded(&mut folder)?;
    report.extend(repair(&folder.draft));
    Ok(Opened { folder, report })
}

/// The project folder in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Folder {
    /// The outward cut, `draft_info.json`.
    pub draft: Draft,
    pub project: ProjectIndex,
    pub layout: TimelineLayout,
    pub meta: DraftMeta,
    pub id_map: IdMap,
    /// Unknown keys beside `id_mapping` in the id-map file.
    pub id_map_rest: Map<String, Value>,
    /// Other JSON files in `common_attachment`, by file name.
    pub sidecars: BTreeMap<String, Value>,
    /// `draft_biz_config.json`, when the folder has one. Linkage is not applied.
    pub biz_config: Option<Value>,
    /// One draft per timeline folder. The main id is also [`Self::draft`].
    pub timelines: BTreeMap<String, Draft>,
}

/// `project.json`: the main timeline id and the timeline list.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ProjectIndex {
    #[serde(default)]
    pub main_timeline_id: String,
    #[serde(default)]
    pub timelines: Vec<TimelineEntry>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One timeline entry. A non-main row marked deleted loses its folder on save.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct TimelineEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub is_marked_delete: bool,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// `timeline_layout.json`: which timelines are docked.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct TimelineLayout {
    #[serde(default, rename = "layoutOrientation")]
    pub layout_orientation: i64,
    #[serde(default, rename = "dockItems")]
    pub dock_items: Vec<DockItem>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One dock in the layout file.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct DockItem {
    #[serde(default, rename = "dockIndex")]
    pub dock_index: i64,
    #[serde(default)]
    pub ratio: String,
    #[serde(default, rename = "timelineIds")]
    pub timeline_ids: Vec<String>,
    #[serde(default, rename = "timelineNames")]
    pub timeline_names: Vec<String>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// `draft_meta_info.json`: metadata and the imported-material bins.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct DraftMeta {
    #[serde(default)]
    pub draft_id: String,
    #[serde(default)]
    pub draft_materials: Vec<MaterialBin>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One imported-material bin. Item fields stay as JSON.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialBin {
    #[serde(default, rename = "type")]
    pub kind: i64,
    #[serde(default)]
    pub value: Vec<Value>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The id map. A new map starts at `next_index` 1000. A stored index is kept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdMap {
    #[serde(default)]
    pub mapping: Vec<IdPair>,
    #[serde(default = "default_next_index")]
    pub next_index: i64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for IdMap {
    fn default() -> Self {
        Self {
            mapping: Vec::new(),
            next_index: default_next_index(),
            rest: Map::new(),
        }
    }
}

/// One short id paired with the draft's own id.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct IdPair {
    #[serde(default)]
    pub short_id: String,
    #[serde(default)]
    pub uuid: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct IdMapFile {
    id_mapping: IdMap,
    #[serde(flatten, default)]
    rest: Map<String, Value>,
}

impl Folder {
    /// A new project. The outward cut and its timeline folder start as the same draft.
    #[must_use]
    pub fn new(draft: Draft) -> Self {
        let id = draft.id.to_string();
        let mut timelines = BTreeMap::new();
        if component_ok(&id) {
            timelines.insert(id.clone(), draft.clone());
        }
        Self {
            project: ProjectIndex {
                main_timeline_id: id.clone(),
                timelines: vec![TimelineEntry {
                    id: id.clone(),
                    name: draft.name.clone(),
                    is_marked_delete: false,
                    rest: Map::new(),
                }],
                rest: Map::new(),
            },
            layout: TimelineLayout {
                layout_orientation: 0,
                dock_items: vec![DockItem {
                    dock_index: 0,
                    ratio: draft.canvas_config.ratio.clone(),
                    timeline_ids: vec![id.clone()],
                    timeline_names: vec![draft.name.clone()],
                    rest: Map::new(),
                }],
                rest: Map::new(),
            },
            meta: DraftMeta {
                draft_id: id,
                draft_materials: Vec::new(),
                rest: Map::new(),
            },
            id_map: IdMap::default(),
            id_map_rest: Map::new(),
            sidecars: BTreeMap::new(),
            biz_config: None,
            timelines,
            draft,
        }
    }

    /// Read the folder. Missing sidecars stay empty. `draft_info.json` is required.
    pub fn read(path: &Path) -> Result<Self, StoreError> {
        let root = canonicalize_root(path)?;
        let draft_path = root.join(DRAFT_FILE);
        let draft = parse_draft(&read_required(&draft_path)?, &draft_path)?;
        let project = read_project(&root, &draft)?;
        let layout = read_struct::<TimelineLayout>(&root, LAYOUT_FILE)?;
        let meta = read_struct::<DraftMeta>(&root, META_FILE)?;
        let (id_map, id_map_rest) = read_id_map(&root)?;
        let sidecars = read_sidecars(&root)?;
        let biz_config = read_biz_config(&root)?;
        let timelines = read_timelines(&root)?;
        Ok(Self {
            draft,
            project,
            layout,
            meta,
            id_map,
            id_map_rest,
            sidecars,
            biz_config,
            timelines,
        })
    }

    /// Write the folder. The top draft file is a byte copy of the main timeline.
    pub fn save(&self, path: &Path) -> Result<(), StoreError> {
        validate(self)?;
        fs::create_dir_all(path).map_err(|error| io_error(path, error))?;
        let root = canonicalize_root(path)?;
        for entry in &self.project.timelines {
            if entry.is_marked_delete {
                remove_timeline(&root, &entry.id)?;
            }
        }
        let main_id = &self.project.main_timeline_id;
        let bytes = write_draft(&self.draft)?;
        let main_path = root.join(TIMELINES_DIR).join(main_id).join(DRAFT_FILE);
        write_bytes(&root, &main_path, &bytes)?;
        let top = root.join(DRAFT_FILE);
        write_bytes(&root, &top, &bytes)?;
        copy_backup(&top)?;
        let project_path = root.join(PROJECT_FILE);
        write_struct(&root, &project_path, &self.project)?;
        copy_backup(&project_path)?;
        write_struct(&root, &root.join(LAYOUT_FILE), &self.layout)?;
        write_struct(&root, &root.join(META_FILE), &self.meta)?;
        write_id_map(&root, self)?;
        write_sidecars(&root, self)?;
        if let Some(biz_config) = &self.biz_config {
            let path = root.join(BIZ_CONFIG_FILE);
            let bytes = write_value(biz_config, &path)?;
            write_bytes(&root, &path, &bytes)?;
        }
        for (id, timeline) in &self.timelines {
            if id == main_id || marked_deleted(self, id) {
                continue;
            }
            let draft_path = root.join(TIMELINES_DIR).join(id).join(DRAFT_FILE);
            write_bytes(&root, &draft_path, &write_draft(timeline)?)?;
        }
        let resources = root.join(RESOURCES_DIR);
        stay_inside(&root, &resources)?;
        fs::create_dir_all(&resources).map_err(|error| io_error(&resources, error))?;
        Ok(())
    }
}

fn validate(folder: &Folder) -> Result<(), StoreError> {
    require_component(&folder.project.main_timeline_id)?;
    if folder
        .project
        .timelines
        .iter()
        .any(|entry| entry.is_marked_delete && entry.id == folder.project.main_timeline_id)
    {
        return Err(StoreError::MainTimelineMarkedDeleted {
            id: folder.project.main_timeline_id.clone(),
        });
    }
    for entry in &folder.project.timelines {
        require_component(&entry.id)?;
    }
    for id in folder.timelines.keys() {
        require_component(id)?;
    }
    for name in folder.sidecars.keys() {
        if !component_ok(name) || !name.ends_with(".json") || name == ID_MAP_FILE {
            return Err(StoreError::PathEscapes { name: name.clone() });
        }
    }
    Ok(())
}

fn read_project(root: &Path, draft: &Draft) -> Result<ProjectIndex, StoreError> {
    let path = root.join(PROJECT_FILE);
    let Some(bytes) = read_optional(&path)? else {
        return Ok(ProjectIndex {
            main_timeline_id: draft.id.to_string(),
            timelines: vec![TimelineEntry {
                id: draft.id.to_string(),
                name: draft.name.clone(),
                ..TimelineEntry::default()
            }],
            rest: Map::new(),
        });
    };
    let project: ProjectIndex =
        serde_json::from_slice(&bytes).map_err(|error| StoreError::Json {
            path,
            message: error.to_string(),
        })?;
    if !project.main_timeline_id.is_empty() {
        require_component(&project.main_timeline_id)?;
    }
    for entry in &project.timelines {
        require_component(&entry.id)?;
    }
    Ok(project)
}

fn read_struct<T>(root: &Path, name: &str) -> Result<T, StoreError>
where
    T: for<'de> Deserialize<'de> + Default,
{
    let path = root.join(name);
    let Some(bytes) = read_optional(&path)? else {
        return Ok(T::default());
    };
    serde_json::from_slice(&bytes).map_err(|error| StoreError::Json {
        path,
        message: error.to_string(),
    })
}

fn read_id_map(root: &Path) -> Result<(IdMap, Map<String, Value>), StoreError> {
    let path = root.join(ATTACHMENT_DIR).join(ID_MAP_FILE);
    let Some(bytes) = read_optional(&path)? else {
        return Ok((IdMap::default(), Map::new()));
    };
    let file: IdMapFile = serde_json::from_slice(&bytes).map_err(|error| StoreError::Json {
        path,
        message: error.to_string(),
    })?;
    Ok((file.id_mapping, file.rest))
}

fn read_sidecars(root: &Path) -> Result<BTreeMap<String, Value>, StoreError> {
    let dir = root.join(ATTACHMENT_DIR);
    if !dir.exists() {
        return Ok(BTreeMap::new());
    }
    stay_inside(root, &dir)?;
    let mut sidecars = BTreeMap::new();
    for entry in fs::read_dir(&dir).map_err(|error| io_error(&dir, error))? {
        let entry = entry.map_err(|error| io_error(&dir, error))?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Err(StoreError::PathEscapes {
                name: entry.path().display().to_string(),
            });
        };
        if name == ID_MAP_FILE || !name.ends_with(".json") || !component_ok(&name) {
            if name == ID_MAP_FILE {
                continue;
            }
            if !component_ok(&name) {
                return Err(StoreError::PathEscapes { name });
            }
            continue;
        }
        let path = entry.path();
        stay_inside(root, &path)?;
        let value = parse_value(
            &fs::read(&path).map_err(|error| io_error(&path, error))?,
            &path,
        )?;
        sidecars.insert(name, value);
    }
    Ok(sidecars)
}

fn read_timelines(root: &Path) -> Result<BTreeMap<String, Draft>, StoreError> {
    let dir = root.join(TIMELINES_DIR);
    if !dir.exists() {
        return Ok(BTreeMap::new());
    }
    stay_inside(root, &dir)?;
    let mut timelines = BTreeMap::new();
    for entry in fs::read_dir(&dir).map_err(|error| io_error(&dir, error))? {
        let entry = entry.map_err(|error| io_error(&dir, error))?;
        let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
            return Err(StoreError::PathEscapes {
                name: entry.path().display().to_string(),
            });
        };
        require_component(&id)?;
        let draft_path = entry.path().join(DRAFT_FILE);
        if !draft_path.exists() {
            continue;
        }
        stay_inside(root, &draft_path)?;
        let draft = parse_draft(
            &fs::read(&draft_path).map_err(|error| io_error(&draft_path, error))?,
            &draft_path,
        )?;
        timelines.insert(id, draft);
    }
    Ok(timelines)
}

fn promote_single_timeline(root: &Path, folder: &mut Folder) -> Result<(), StoreError> {
    let id = folder.draft.id.to_string();
    require_component(&id)?;
    let bytes = read_required(&root.join(DRAFT_FILE))?;
    let timeline_path = root.join(TIMELINES_DIR).join(&id).join(DRAFT_FILE);
    write_bytes(root, &timeline_path, &bytes)?;
    folder.project = ProjectIndex {
        main_timeline_id: id.clone(),
        timelines: vec![TimelineEntry {
            id: id.clone(),
            name: folder.draft.name.clone(),
            is_marked_delete: false,
            rest: Map::new(),
        }],
        rest: Map::new(),
    };
    write_struct(root, &root.join(PROJECT_FILE), &folder.project)?;
    folder.timelines.insert(id, folder.draft.clone());
    Ok(())
}

fn reconcile_main_copy(
    root: &Path,
    folder: &mut Folder,
    report: &mut Vec<RepairNote>,
) -> Result<(), StoreError> {
    let main_id = folder.project.main_timeline_id.clone();
    if main_id.is_empty() {
        return Ok(());
    }
    require_component(&main_id)?;
    let top_path = root.join(DRAFT_FILE);
    let timeline_path = root.join(TIMELINES_DIR).join(&main_id).join(DRAFT_FILE);
    let top_bytes = read_required(&top_path)?;
    let Some(timeline_bytes) = read_optional(&timeline_path)? else {
        return Err(StoreError::MissingDraft {
            path: timeline_path,
        });
    };
    if top_bytes == timeline_bytes {
        return Ok(());
    }
    let newer = modified(&top_path)? > modified(&timeline_path)?;
    if newer && outside_desktop(platform_os(&folder.draft)) {
        let parsed = parse_draft(&top_bytes, &top_path)?;
        if parsed.id.as_str() != main_id {
            return Err(StoreError::TimelineIdMismatch {
                directory: main_id,
                draft_id: parsed.id.to_string(),
            });
        }
        let upgraded = upgrade(&parsed)?;
        write_bytes(root, &timeline_path, &top_bytes)?;
        folder.timelines.insert(main_id, upgraded);
    } else {
        report.push(RepairNote::OutwardCopyDiffers);
    }
    Ok(())
}

fn skip_id_mismatches(folder: &mut Folder) -> Result<Vec<RepairNote>, StoreError> {
    let main_id = folder.project.main_timeline_id.clone();
    let mut notes = Vec::new();
    let mut kept = BTreeMap::new();
    for (id, timeline) in std::mem::take(&mut folder.timelines) {
        if timeline.id.as_str() == id {
            kept.insert(id, timeline);
            continue;
        }
        if id == main_id {
            return Err(StoreError::TimelineIdMismatch {
                directory: id,
                draft_id: timeline.id.to_string(),
            });
        }
        notes.push(RepairNote::TimelineIdMismatch {
            directory: id,
            draft_id: timeline.id.to_string(),
        });
    }
    folder.timelines = kept;
    Ok(notes)
}

fn load_main_timeline(folder: &mut Folder) -> Result<(), StoreError> {
    let main_id = &folder.project.main_timeline_id;
    if main_id.is_empty() {
        return Ok(());
    }
    let Some(timeline) = folder.timelines.get(main_id).cloned() else {
        return Err(StoreError::MissingDraft {
            path: PathBuf::from(TIMELINES_DIR).join(main_id).join(DRAFT_FILE),
        });
    };
    folder.draft = timeline;
    Ok(())
}

fn upgrade_loaded(folder: &mut Folder) -> Result<(), StoreError> {
    let main_id = folder.project.main_timeline_id.clone();
    folder.draft = upgrade(&folder.draft)?;
    let timelines = std::mem::take(&mut folder.timelines);
    let mut upgraded = BTreeMap::new();
    for (id, timeline) in timelines {
        if id == main_id {
            upgraded.insert(id, folder.draft.clone());
        } else {
            upgraded.insert(id, upgrade(&timeline)?);
        }
    }
    folder.timelines = upgraded;
    Ok(())
}

fn platform_os(draft: &Draft) -> Option<&str> {
    draft
        .rest
        .get("last_modified_platform")
        .and_then(|value| value.get("os"))
        .and_then(Value::as_str)
}

fn outside_desktop(os: Option<&str>) -> bool {
    !os.is_some_and(|os| DESKTOP_OS.contains(&os))
}

fn modified(path: &Path) -> Result<SystemTime, StoreError> {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .map_err(|error| io_error(path, error))
}

fn marked_deleted(folder: &Folder, id: &str) -> bool {
    folder
        .project
        .timelines
        .iter()
        .any(|entry| entry.id == id && entry.is_marked_delete)
}

fn remove_timeline(root: &Path, id: &str) -> Result<(), StoreError> {
    require_component(id)?;
    let dir = root.join(TIMELINES_DIR).join(id);
    if dir.symlink_metadata().is_err() {
        return Ok(());
    }
    stay_inside(root, &dir)?;
    fs::remove_dir_all(&dir).map_err(|error| io_error(&dir, error))
}

fn read_biz_config(root: &Path) -> Result<Option<Value>, StoreError> {
    let path = root.join(BIZ_CONFIG_FILE);
    let Some(bytes) = read_optional(&path)? else {
        return Ok(None);
    };
    Ok(Some(parse_value(&bytes, &path)?))
}

fn write_struct<T>(root: &Path, path: &Path, value: &T) -> Result<(), StoreError>
where
    T: Serialize,
{
    let json = serde_json::to_value(value).map_err(|error| StoreError::Json {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let bytes = write_value(&json, path)?;
    write_bytes(root, path, &bytes)
}

fn write_id_map(root: &Path, folder: &Folder) -> Result<(), StoreError> {
    let path = root.join(ATTACHMENT_DIR).join(ID_MAP_FILE);
    let file = IdMapFile {
        id_mapping: folder.id_map.clone(),
        rest: folder.id_map_rest.clone(),
    };
    write_struct(root, &path, &file)
}

fn write_sidecars(root: &Path, folder: &Folder) -> Result<(), StoreError> {
    for (name, value) in &folder.sidecars {
        let path = root.join(ATTACHMENT_DIR).join(name);
        let bytes = write_value(value, &path)?;
        write_bytes(root, &path, &bytes)?;
    }
    Ok(())
}

fn write_bytes(root: &Path, path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    stay_inside(root, path)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
    }
    fs::write(path, bytes).map_err(|error| io_error(path, error))
}

/// A single path component. `.`, `..`, a separator, or a null byte cannot be joined.
fn component_ok(name: &str) -> bool {
    if name.is_empty() || name == "." || name == ".." {
        return false;
    }
    !name.contains('/') && !name.contains('\\') && !name.contains('\0')
}

fn require_component(name: &str) -> Result<(), StoreError> {
    if component_ok(name) {
        Ok(())
    } else {
        Err(StoreError::PathEscapes {
            name: name.to_owned(),
        })
    }
}

fn canonicalize_root(path: &Path) -> Result<PathBuf, StoreError> {
    match path.canonicalize() {
        Ok(root) => Ok(root),
        Err(error) if error.kind() == ErrorKind::NotFound => Err(StoreError::MissingDraft {
            path: path.join(DRAFT_FILE),
        }),
        Err(error) => Err(io_error(path, error)),
    }
}

/// Refuse a path whose existing ancestor, or whose own target, sits outside `root`.
/// `root` is already canonical. A missing tail is allowed only when every existing
/// ancestor is still inside the folder and the lexical path does not climb out.
fn stay_inside(root: &Path, path: &Path) -> Result<(), StoreError> {
    if path
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(StoreError::PathEscapes {
            name: path.display().to_string(),
        });
    }
    let mut ancestor = path.to_path_buf();
    while !ancestor.exists() {
        let Some(parent) = ancestor.parent() else {
            return Err(StoreError::PathEscapes {
                name: path.display().to_string(),
            });
        };
        if parent == ancestor {
            return Err(StoreError::PathEscapes {
                name: path.display().to_string(),
            });
        }
        ancestor = parent.to_path_buf();
    }
    let ancestor = ancestor
        .canonicalize()
        .map_err(|error| io_error(&ancestor, error))?;
    if ancestor.starts_with(root) {
        Ok(())
    } else {
        Err(StoreError::PathEscapes {
            name: path.display().to_string(),
        })
    }
}

fn read_required(path: &Path) -> Result<Vec<u8>, StoreError> {
    match fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(error) if error.kind() == ErrorKind::NotFound => Err(StoreError::MissingDraft {
            path: path.to_path_buf(),
        }),
        Err(error) => Err(io_error(path, error)),
    }
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, StoreError> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_error(path, error)),
    }
}

fn io_error(path: &Path, error: std::io::Error) -> StoreError {
    StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_model::{Draft, Segment, TimeRange, Track};
    use serde_json::Value;

    use super::{Folder, open};
    use crate::{RepairNote, StoreError};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("cutlass-store-folder-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn a_new_folder_has_the_project_files_and_a_draft_backup() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let folder = Folder::new(Draft::empty("D"));
        folder.save(&root).unwrap();
        assert!(root.join("draft_info.json").is_file());
        assert!(root.join("project.json").is_file());
        assert!(root.join("Timelines/D/draft_info.json").is_file());
        assert!(root.join("timeline_layout.json").is_file());
        assert!(root.join("draft_meta_info.json").is_file());
        assert!(
            root.join("common_attachment/attachment_id_mapping.json")
                .is_file()
        );
        assert!(root.join("Resources").is_dir());
        let draft = fs::read(root.join("draft_info.json")).unwrap();
        assert_eq!(fs::read(root.join("draft_info.json.bak")).unwrap(), draft);
        assert_eq!(
            fs::read(root.join("Timelines/D/draft_info.json")).unwrap(),
            draft
        );
        assert!(!root.join("Timelines/D/draft_info.json.bak").exists());
        let project = fs::read(root.join("project.json")).unwrap();
        assert_eq!(fs::read(root.join("project.json.bak")).unwrap(), project);
        let id_map: Value = serde_json::from_slice(
            &fs::read(root.join("common_attachment/attachment_id_mapping.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(id_map["id_mapping"]["next_index"], 1000);
    }

    #[test]
    fn an_unknown_nested_field_survives_the_folder() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut draft = Draft::empty("D");
        draft
            .canvas_config
            .rest
            .insert("future_canvas".to_owned(), Value::from(1));
        let mut main = Track::main("main");
        let mut segment = Segment::new("s", "m", TimeRange::new(0, 1));
        segment
            .target
            .rest
            .insert("future_span".to_owned(), Value::from(true));
        segment
            .clip
            .rest
            .insert("future_clip".to_owned(), Value::from(2));
        main.segments.push(segment);
        draft.tracks = vec![main];
        Folder::new(draft).save(&root).unwrap();
        let opened = open(&root).unwrap();
        assert!(opened.report.is_empty());
        assert_eq!(opened.folder.draft.canvas_config.rest["future_canvas"], 1);
        assert_eq!(
            opened.folder.draft.tracks[0].segments[0].target.rest["future_span"],
            true
        );
        assert_eq!(
            opened.folder.draft.tracks[0].segments[0].clip.rest["future_clip"],
            2
        );
        opened.folder.save(&root).unwrap();
        let value: Value =
            serde_json::from_slice(&fs::read(root.join("draft_info.json")).unwrap()).unwrap();
        assert_eq!(value["canvas_config"]["future_canvas"], 1);
        assert_eq!(
            value["tracks"][0]["segments"][0]["target_timerange"]["future_span"],
            true
        );
        assert_eq!(value["tracks"][0]["segments"][0]["clip"]["future_clip"], 2);
    }

    #[test]
    fn open_is_parse_then_upgrade_then_a_report() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut draft = Draft::empty("D");
        draft.new_version = "75.0.0".to_owned();
        let mut main = Track::main("main");
        main.segments = vec![
            Segment::new("a", "m", TimeRange::new(0, 100)),
            Segment::new("b", "m", TimeRange::new(50, 50)),
        ];
        draft.tracks = vec![main];
        Folder::new(draft).save(&root).unwrap();
        let bytes = fs::read(root.join("draft_info.json")).unwrap();
        let opened = open(&root).unwrap();
        assert_eq!(opened.folder.draft.new_version, "75.0.0");
        assert_eq!(opened.folder.draft.version, 360_000);
        assert_eq!(
            opened.report,
            vec![RepairNote::RangeConflict {
                track_id: "main".into(),
                segment_id: "b".into(),
            }]
        );
        assert_eq!(opened.folder.draft.tracks[0].segments.len(), 2);
        assert_eq!(fs::read(root.join("draft_info.json")).unwrap(), bytes);
    }

    #[test]
    fn a_newer_protocol_is_refused_and_the_file_stays() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut draft = Draft::empty("D");
        draft.new_version = "192.0.0".to_owned();
        Folder::new(draft).save(&root).unwrap();
        let bytes = fs::read(root.join("draft_info.json")).unwrap();
        let error = open(&root).unwrap_err();
        assert!(matches!(
            error,
            StoreError::NewerProtocol { ref found, ref draft_id }
                if found == "192.0.0" && draft_id == "D"
        ));
        assert_eq!(fs::read(root.join("draft_info.json")).unwrap(), bytes);
    }

    #[test]
    fn a_missing_or_truncated_draft_fails_closed() {
        let temp = Temp::new();
        let missing = open(&temp.path.join("absent")).unwrap_err();
        assert!(matches!(missing, StoreError::MissingDraft { .. }));
        let root = temp.path.join("project");
        fs::create_dir_all(&root).unwrap();
        let error = open(&root).unwrap_err();
        assert!(matches!(error, StoreError::MissingDraft { .. }));
        fs::write(root.join("draft_info.json"), b"{").unwrap();
        let error = open(&root).unwrap_err();
        assert!(matches!(error, StoreError::Json { .. }));
        Folder::new(Draft::empty("D")).save(&root).unwrap();
        fs::write(root.join("project.json"), b"{").unwrap();
        let error = open(&root).unwrap_err();
        assert!(matches!(
            error,
            StoreError::Json { ref path, .. } if path.ends_with("project.json")
        ));
    }

    #[test]
    fn a_timeline_id_that_climbs_out_of_the_folder_is_refused() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder = Folder::new(Draft::empty("D"));
        folder.project.timelines.push(super::TimelineEntry {
            id: "../outside".to_owned(),
            ..super::TimelineEntry::default()
        });
        let error = folder.save(&root).unwrap_err();
        assert!(matches!(
            error,
            StoreError::PathEscapes { ref name } if name == "../outside"
        ));
        assert!(!temp.path.join("outside").exists());
        assert!(!root.join("outside").exists());
    }

    #[test]
    fn a_timeline_symlink_outside_the_folder_is_refused() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        Folder::new(Draft::empty("D")).save(&root).unwrap();
        let outside = temp.path.join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("draft_info.json"), b"{").unwrap();
        fs::remove_dir_all(root.join("Timelines")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("Timelines")).unwrap();
        let error = open(&root).unwrap_err();
        assert!(matches!(error, StoreError::PathEscapes { .. }));
    }

    #[test]
    fn a_stored_id_index_and_an_unknown_project_field_survive() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder = Folder::new(Draft::empty("D"));
        folder.id_map.next_index = 1002;
        folder
            .project
            .rest
            .insert("future_project".to_owned(), Value::from(true));
        folder.save(&root).unwrap();
        let opened = open(&root).unwrap();
        assert_eq!(opened.folder.id_map.next_index, 1002);
        assert_eq!(opened.folder.project.rest["future_project"], true);
        assert_eq!(opened.folder.project.timelines[0].name, "");
    }

    #[test]
    fn a_folder_without_project_json_becomes_one_timeline() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        fs::create_dir_all(&root).unwrap();
        let mut draft = Draft::empty("D");
        draft.name = "only".to_owned();
        fs::write(
            root.join("draft_info.json"),
            crate::write_draft(&draft).unwrap(),
        )
        .unwrap();
        let opened = open(&root).unwrap();
        let top = fs::read(root.join("draft_info.json")).unwrap();
        assert_eq!(
            fs::read(root.join("Timelines/D/draft_info.json")).unwrap(),
            top
        );
        let project: Value =
            serde_json::from_slice(&fs::read(root.join("project.json")).unwrap()).unwrap();
        assert_eq!(project["main_timeline_id"], "D");
        assert_eq!(project["timelines"][0]["id"], "D");
        assert_eq!(project["timelines"][0]["name"], "only");
        assert_eq!(project["timelines"][0]["is_marked_delete"], false);
        assert_eq!(opened.folder.draft.name, "only");
    }

    #[test]
    fn a_draft_id_that_climbs_out_is_not_promoted() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        fs::create_dir_all(&root).unwrap();
        let draft = Draft::empty("../outside");
        fs::write(
            root.join("draft_info.json"),
            crate::write_draft(&draft).unwrap(),
        )
        .unwrap();
        let error = open(&root).unwrap_err();
        assert!(matches!(
            error,
            StoreError::PathEscapes { ref name } if name == "../outside"
        ));
        assert!(!temp.path.join("outside").exists());
        assert!(!root.join("outside").exists());
        assert!(!root.join("project.json").exists());
        assert!(!root.join("Timelines").exists());
    }

    #[test]
    fn a_newer_outside_desktop_file_replaces_the_main_timeline() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        Folder::new(Draft::empty("D")).save(&root).unwrap();
        let mut phone = Draft::empty("D");
        phone.name = "phone".to_owned();
        phone.rest.insert(
            "last_modified_platform".to_owned(),
            serde_json::json!({"os": "android"}),
        );
        let phone_bytes = crate::write_draft(&phone).unwrap();
        fs::write(root.join("draft_info.json"), &phone_bytes).unwrap();
        set_mtime(&root.join("Timelines/D/draft_info.json"), 1_700_000_000);
        set_mtime(&root.join("draft_info.json"), 1_700_000_100);
        let opened = open(&root).unwrap();
        assert_eq!(opened.folder.draft.name, "phone");
        assert!(!opened.report.contains(&RepairNote::OutwardCopyDiffers));
        assert_eq!(
            fs::read(root.join("Timelines/D/draft_info.json")).unwrap(),
            phone_bytes
        );
    }

    #[test]
    fn a_desktop_difference_loads_the_timeline_folder() {
        for os in ["mac", "windows", "macOS"] {
            let temp = Temp::new();
            let root = temp.path.join("project");
            let mut folder_draft = Draft::empty("D");
            folder_draft.name = "folder".to_owned();
            Folder::new(folder_draft).save(&root).unwrap();
            let timeline_bytes = fs::read(root.join("Timelines/D/draft_info.json")).unwrap();
            let mut top = Draft::empty("D");
            top.name = "top-copy".to_owned();
            top.rest.insert(
                "last_modified_platform".to_owned(),
                serde_json::json!({"os": os}),
            );
            let top_bytes = crate::write_draft(&top).unwrap();
            fs::write(root.join("draft_info.json"), &top_bytes).unwrap();
            set_mtime(&root.join("Timelines/D/draft_info.json"), 1_700_000_000);
            set_mtime(&root.join("draft_info.json"), 1_700_000_100);
            let opened = open(&root).unwrap();
            assert!(opened.report.contains(&RepairNote::OutwardCopyDiffers));
            assert_eq!(opened.folder.draft.name, "folder");
            assert_eq!(
                fs::read(root.join("Timelines/D/draft_info.json")).unwrap(),
                timeline_bytes
            );
            assert_eq!(fs::read(root.join("draft_info.json")).unwrap(), top_bytes);
        }
    }

    #[test]
    fn an_older_outside_desktop_file_does_not_replace_the_timeline() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder_draft = Draft::empty("D");
        folder_draft.name = "folder".to_owned();
        Folder::new(folder_draft).save(&root).unwrap();
        let timeline_bytes = fs::read(root.join("Timelines/D/draft_info.json")).unwrap();
        let mut phone = Draft::empty("D");
        phone.name = "phone".to_owned();
        phone.rest.insert(
            "last_modified_platform".to_owned(),
            serde_json::json!({"os": "android"}),
        );
        let phone_bytes = crate::write_draft(&phone).unwrap();
        fs::write(root.join("draft_info.json"), &phone_bytes).unwrap();
        set_mtime(&root.join("draft_info.json"), 1_700_000_000);
        set_mtime(&root.join("Timelines/D/draft_info.json"), 1_700_000_100);
        let opened = open(&root).unwrap();
        assert!(opened.report.contains(&RepairNote::OutwardCopyDiffers));
        assert_eq!(opened.folder.draft.name, "folder");
        assert_eq!(
            fs::read(root.join("Timelines/D/draft_info.json")).unwrap(),
            timeline_bytes
        );
        assert_eq!(fs::read(root.join("draft_info.json")).unwrap(), phone_bytes);
    }

    #[test]
    fn a_timeline_whose_id_disagrees_with_its_folder_is_skipped() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        Folder::new(Draft::empty("D")).save(&root).unwrap();
        fs::create_dir_all(root.join("Timelines/b")).unwrap();
        let mut other = Draft::empty("not-b");
        other.new_version = "192.0.0".to_owned();
        fs::write(
            root.join("Timelines/b/draft_info.json"),
            crate::write_draft(&other).unwrap(),
        )
        .unwrap();
        let opened = open(&root).unwrap();
        assert!(opened.report.contains(&RepairNote::TimelineIdMismatch {
            directory: "b".to_owned(),
            draft_id: "not-b".to_owned(),
        }));
        assert!(!opened.folder.timelines.contains_key("b"));
        assert!(root.join("Timelines/b/draft_info.json").is_file());
        assert_eq!(opened.folder.draft.id.as_str(), "D");
    }

    #[test]
    fn a_main_timeline_whose_id_disagrees_is_refused() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut top = Draft::empty("D");
        top.rest.insert(
            "last_modified_platform".to_owned(),
            serde_json::json!({"os": "mac"}),
        );
        Folder::new(top).save(&root).unwrap();
        let other = crate::write_draft(&Draft::empty("other")).unwrap();
        fs::write(root.join("Timelines/D/draft_info.json"), &other).unwrap();
        set_mtime(&root.join("Timelines/D/draft_info.json"), 1_700_000_000);
        set_mtime(&root.join("draft_info.json"), 1_700_000_100);
        let error = open(&root).unwrap_err();
        assert!(matches!(
            error,
            StoreError::TimelineIdMismatch { ref directory, ref draft_id }
                if directory == "D" && draft_id == "other"
        ));
        assert_eq!(
            fs::read(root.join("Timelines/D/draft_info.json")).unwrap(),
            other
        );
    }

    #[test]
    fn a_deleted_timeline_folder_is_removed_and_the_main_timeline_cannot_be() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder = Folder::new(Draft::empty("D"));
        folder.timelines.insert("B".to_owned(), Draft::empty("B"));
        folder.project.timelines.push(super::TimelineEntry {
            id: "B".to_owned(),
            name: "b".to_owned(),
            is_marked_delete: false,
            rest: serde_json::Map::new(),
        });
        folder.save(&root).unwrap();
        assert!(root.join("Timelines/B/draft_info.json").is_file());
        folder.project.timelines[1].is_marked_delete = true;
        folder.save(&root).unwrap();
        assert!(!root.join("Timelines/B").exists());
        assert!(root.join("Timelines/D/draft_info.json").is_file());
        folder.project.timelines[0].is_marked_delete = true;
        let error = folder.save(&root).unwrap_err();
        assert!(matches!(
            error,
            StoreError::MainTimelineMarkedDeleted { ref id } if id == "D"
        ));
        assert!(root.join("Timelines/D/draft_info.json").is_file());
    }

    #[test]
    fn a_deleted_mark_does_not_remove_a_folder_outside_the_project() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder = Folder::new(Draft::empty("D"));
        folder.save(&root).unwrap();
        let outside = temp.path.join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("keep.txt"), b"keep").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("Timelines/evil")).unwrap();
        folder.project.timelines.push(super::TimelineEntry {
            id: "evil".to_owned(),
            is_marked_delete: true,
            ..super::TimelineEntry::default()
        });
        let error = folder.save(&root).unwrap_err();
        assert!(matches!(error, StoreError::PathEscapes { .. }));
        assert_eq!(fs::read(outside.join("keep.txt")).unwrap(), b"keep");
    }

    #[test]
    fn draft_biz_config_survives_the_folder() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder = Folder::new(Draft::empty("D"));
        folder.biz_config = Some(serde_json::json!({
            "timeline_settings": { "D": { "linkage_enabled": true } }
        }));
        folder.save(&root).unwrap();
        let opened = open(&root).unwrap();
        assert_eq!(
            opened.folder.biz_config.as_ref().unwrap()["timeline_settings"]["D"]["linkage_enabled"],
            true
        );
        let other = temp.path.join("copy");
        opened.folder.save(&other).unwrap();
        assert_eq!(
            fs::read(other.join("draft_biz_config.json")).unwrap(),
            b"{\"timeline_settings\":{\"D\":{\"linkage_enabled\":true}}}\n"
        );
    }

    #[test]
    fn a_resource_file_and_an_extra_sidecar_stay_put() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let mut folder = Folder::new(Draft::empty("D"));
        folder
            .sidecars
            .insert("extra.json".to_owned(), serde_json::json!({"kept": true}));
        folder.save(&root).unwrap();
        fs::write(root.join("Resources").join("clip.bin"), b"media").unwrap();
        let opened = open(&root).unwrap();
        assert_eq!(opened.folder.sidecars["extra.json"]["kept"], true);
        opened.folder.save(&root).unwrap();
        assert_eq!(
            fs::read(root.join("Resources").join("clip.bin")).unwrap(),
            b"media"
        );
        assert_eq!(
            fs::read(root.join("common_attachment/extra.json")).unwrap(),
            b"{\"kept\":true}\n"
        );
    }

    fn set_mtime(path: &std::path::Path, secs: u64) {
        let file = fs::File::options().write(true).open(path).unwrap();
        file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .unwrap();
    }
}
