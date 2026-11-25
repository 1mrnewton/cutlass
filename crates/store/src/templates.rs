//! Project templates: a finished project with some clips marked as slots,
//! saved so it can be used again with other media.
//!
//! A template is a project folder plus `cutlass_template.json`, which
//! names it and files it under a category. The slots, editable texts and
//! music are the marks on the draft's clips, so nothing else has to agree
//! with the timeline. The sample media stays in the folder, so a template
//! previews exactly as it was made. Using one copies it into a new
//! project; the template itself is never changed.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_model::{Micros, template_slots};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::projects::{copy_tree, unix_millis};
use crate::{Folder, StoreError};

/// The file that makes a project folder a template.
pub const TEMPLATE_FILE: &str = "cutlass_template.json";

/// Bumped only when the template file's meaning changes.
pub const TEMPLATE_FORMAT_VERSION: u32 = 1;

/// What `cutlass_template.json` holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemplateInfo {
    pub format_version: u32,
    pub name: String,
    /// Where the gallery files it, such as "Travel"; empty when unfiled.
    #[serde(default)]
    pub category: String,
    /// Keys this build does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One template in the gallery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedTemplate {
    pub path: PathBuf,
    pub name: String,
    pub category: String,
    pub duration: Micros,
    /// How many photos or videos it takes.
    pub slots: usize,
    pub width: i32,
    pub height: i32,
    /// When the template was saved: its file's time.
    pub modified: SystemTime,
}

/// Save the project at `project` as a template under `root`, named `name`
/// and filed under `category`, and return the template's folder.
///
/// A project with no slots is refused: there would be nothing to fill. A
/// linked file outside the project is copied into the template's
/// `Resources`, so the template carries every file it plays. On any
/// failure nothing is left under `root`.
pub fn save_as_template(
    project: &Path,
    root: &Path,
    name: &str,
    category: &str,
) -> Result<PathBuf, StoreError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(StoreError::EmptyName);
    }
    let folder = Folder::read(project)?;
    if template_slots(&folder.draft).is_empty() {
        return Err(StoreError::NoSlots);
    }
    fs::create_dir_all(root).map_err(|error| io(root, &error))?;
    let target = free_folder(root, name)?;
    let made = (|| {
        copy_tree(project, &target)?;
        let mut copied = Folder::read(&target)?;
        carry_linked_media(&mut copied, &target)?;
        name.clone_into(&mut copied.draft.name);
        copied.draft.update_time = unix_millis();
        copied.save(&target)?;
        write_info(
            &target,
            &TemplateInfo {
                format_version: TEMPLATE_FORMAT_VERSION,
                name: name.to_owned(),
                category: category.trim().to_owned(),
                rest: Map::new(),
            },
        )
    })();
    if let Err(error) = made {
        let _ = fs::remove_dir_all(&target);
        return Err(error);
    }
    Ok(target)
}

/// The gallery card for the template at `path`, or `None` when the folder
/// holds no template, or its template file is newer than this build.
#[must_use]
pub fn listed_template(path: &Path) -> Option<ListedTemplate> {
    let info = read_info(path).ok()?;
    let folder = Folder::read(path).ok()?;
    let modified = fs::metadata(path.join(TEMPLATE_FILE))
        .and_then(|metadata| metadata.modified())
        .unwrap_or(UNIX_EPOCH);
    Some(ListedTemplate {
        path: path.to_path_buf(),
        name: info.name,
        category: info.category,
        duration: folder.draft.duration,
        slots: template_slots(&folder.draft).len(),
        width: folder.draft.canvas_config.width,
        height: folder.draft.canvas_config.height,
        modified,
    })
}

/// The templates under `root`, newest first. A folder that is not a
/// template is skipped, and a missing root lists nothing.
#[must_use]
pub fn list_templates(root: &Path) -> Vec<ListedTemplate> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut templates: Vec<ListedTemplate> = entries
        .flatten()
        .filter_map(|entry| listed_template(&entry.path()))
        .collect();
    templates.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| left.name.cmp(&right.name))
    });
    templates
}

/// Start a new project under `projects` from the template at `template`,
/// and return its folder. The project is a copy, named after the
/// template, with its marks kept so its slots can be filled; the template
/// file stays behind.
pub fn use_template(template: &Path, projects: &Path) -> Result<PathBuf, StoreError> {
    let info = read_info(template)?;
    fs::create_dir_all(projects).map_err(|error| io(projects, &error))?;
    let target = free_folder(projects, &info.name)?;
    let made = (|| {
        copy_tree(template, &target)?;
        let info_file = target.join(TEMPLATE_FILE);
        fs::remove_file(&info_file).map_err(|error| io(&info_file, &error))?;
        let mut copied = Folder::read(&target)?;
        copied.draft.name.clone_from(&info.name);
        let now = unix_millis();
        copied.draft.create_time = now;
        copied.draft.update_time = now;
        copied.save(&target)
    })();
    if let Err(error) = made {
        let _ = fs::remove_dir_all(&target);
        return Err(error);
    }
    Ok(target)
}

/// Delete the template at `path`. A folder that is not a template is
/// refused, so nothing else is removed.
pub fn delete_template(path: &Path) -> Result<(), StoreError> {
    read_info(path)?;
    fs::remove_dir_all(path).map_err(|error| io(path, &error))
}

fn read_info(path: &Path) -> Result<TemplateInfo, StoreError> {
    let file = path.join(TEMPLATE_FILE);
    let bytes = fs::read(&file).map_err(|error| io(&file, &error))?;
    let info: TemplateInfo = serde_json::from_slice(&bytes).map_err(|error| StoreError::Json {
        path: file.clone(),
        message: error.to_string(),
    })?;
    if info.format_version > TEMPLATE_FORMAT_VERSION {
        return Err(StoreError::Json {
            path: file,
            message: format!(
                "template format {} is newer than this build",
                info.format_version
            ),
        });
    }
    Ok(info)
}

fn write_info(path: &Path, info: &TemplateInfo) -> Result<(), StoreError> {
    let file = path.join(TEMPLATE_FILE);
    let bytes = serde_json::to_vec_pretty(info).map_err(|error| StoreError::Json {
        path: file.clone(),
        message: error.to_string(),
    })?;
    crate::atomic::write_atomic(&file, &bytes).map_err(|error| io(&file, &error))
}

/// Copy every video or sound the draft plays from outside the folder into
/// its `Resources`, and point the material at the copy.
fn carry_linked_media(folder: &mut Folder, target: &Path) -> Result<(), StoreError> {
    let resources = target.join(crate::folder::RESOURCES_DIR);
    let paths = folder
        .draft
        .materials
        .videos
        .iter_mut()
        .map(|video| &mut video.path)
        .chain(
            folder
                .draft
                .materials
                .audios
                .iter_mut()
                .map(|audio| &mut audio.path),
        );
    for path in paths {
        let source = PathBuf::from(path.as_str());
        if path.is_empty() || !source.is_absolute() {
            continue;
        }
        if !source.is_file() {
            return Err(StoreError::MissingMedia { path: source });
        }
        let Some(file_name) = source.file_name() else {
            return Err(StoreError::MissingMedia { path: source });
        };
        fs::create_dir_all(&resources).map_err(|error| io(&resources, &error))?;
        let copy = free_file(&resources, &file_name.to_string_lossy());
        fs::copy(&source, &copy).map_err(|error| io(&copy, &error))?;
        let stored = copy
            .file_name()
            .map(|name| {
                format!(
                    "{}/{}",
                    crate::folder::RESOURCES_DIR,
                    name.to_string_lossy()
                )
            })
            .ok_or_else(|| StoreError::MissingMedia { path: copy.clone() })?;
        *path = stored;
    }
    Ok(())
}

/// The first free folder under `root` for `name`: the name made safe for
/// a folder, then "… 2", "… 3", ….
fn free_folder(root: &Path, name: &str) -> Result<PathBuf, StoreError> {
    let stem: String = name
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control() {
                '-'
            } else {
                c
            }
        })
        .collect();
    let stem = stem.trim().trim_start_matches('.').trim();
    let stem = if stem.is_empty() { "Template" } else { stem };
    (1..100_000)
        .map(|index| {
            if index == 1 {
                root.join(stem)
            } else {
                root.join(format!("{stem} {index}"))
            }
        })
        .find(|path| !path.exists())
        .ok_or_else(|| StoreError::Io {
            path: root.to_path_buf(),
            message: "no free name for the folder".to_owned(),
        })
}

fn free_file(folder: &Path, name: &str) -> PathBuf {
    let path = Path::new(name);
    let stem = path.file_stem().map_or_else(
        || "media".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    let extension = path
        .extension()
        .map(|extension| format!(".{}", extension.to_string_lossy()))
        .unwrap_or_default();
    let mut candidate = folder.join(name);
    let mut index = 2;
    while candidate.exists() {
        candidate = folder.join(format!("{stem} {index}{extension}"));
        index += 1;
    }
    candidate
}

fn io(path: &Path, error: &std::io::Error) -> StoreError {
    StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_model::{
        Draft, MaterialVideo, MetaType, Micros, Segment, SlotTakes, TemplateMark, TimeRange,
    };

    use super::{
        TEMPLATE_FILE, delete_template, list_templates, listed_template, save_as_template,
        use_template,
    };
    use crate::{Folder, StoreError};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cutlass-store-templates-{}-{n}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    /// A project with one video clip in `Resources`, marked as a slot when
    /// `slot` is set, and one clip playing `linked` from outside when given.
    fn project(root: &Path, slot: bool, linked: Option<&Path>) -> PathBuf {
        let path = root.join("Trip");
        fs::create_dir_all(path.join("Resources")).unwrap();
        fs::write(path.join("Resources").join("beach.mp4"), b"sample").unwrap();
        let mut draft = Draft::empty("trip");
        draft.name = "Trip".to_owned();
        draft.duration = Micros(4_000_000);
        let mut video = MaterialVideo::new("v");
        video.kind = MetaType::Video.as_str().to_owned();
        video.path = "Resources/beach.mp4".to_owned();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("a", "v", TimeRange::new(0, 2_000_000));
        if slot {
            segment.template_mark = Some(TemplateMark::Slot {
                takes: SlotTakes::Either,
            });
        }
        draft.tracks[0].segments.push(segment);
        if let Some(linked) = linked {
            let mut outside = MaterialVideo::new("w");
            outside.kind = MetaType::Video.as_str().to_owned();
            outside.path = linked.to_string_lossy().into_owned();
            draft.materials.videos.push(outside);
            draft.tracks[0].segments.push(Segment::new(
                "b",
                "w",
                TimeRange::new(2_000_000, 2_000_000),
            ));
        }
        Folder::new(draft).save(&path).unwrap();
        path
    }

    #[test]
    fn a_project_saves_as_a_template_with_its_media_and_lists() {
        let temp = Temp::new();
        let project = project(&temp.path, true, None);
        let root = temp.path.join("Templates");
        let template = save_as_template(&project, &root, "  Beach day ", "Travel").unwrap();
        assert_eq!(template, root.join("Beach day"));
        assert!(template.join(TEMPLATE_FILE).is_file());
        assert_eq!(
            fs::read(template.join("Resources").join("beach.mp4")).unwrap(),
            b"sample"
        );
        let listed = list_templates(&root);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Beach day");
        assert_eq!(listed[0].category, "Travel");
        assert_eq!(listed[0].slots, 1);
        assert_eq!(listed[0].duration, Micros(4_000_000));
        // The project itself is left as it was.
        assert_eq!(Folder::read(&project).unwrap().draft.name, "Trip");
        assert!(!project.join(TEMPLATE_FILE).exists());
    }

    #[test]
    fn a_project_without_slots_or_a_name_is_refused_and_leaves_nothing() {
        let temp = Temp::new();
        let root = temp.path.join("Templates");
        let plain = project(&temp.path, false, None);
        assert!(matches!(
            save_as_template(&plain, &root, "Plain", ""),
            Err(StoreError::NoSlots)
        ));
        assert!(matches!(
            save_as_template(&plain, &root, "   ", ""),
            Err(StoreError::EmptyName)
        ));
        assert!(list_templates(&root).is_empty());
        assert!(fs::read_dir(&root).map_or(true, |mut entries| entries.next().is_none()));
    }

    #[test]
    fn a_linked_file_is_carried_into_the_template_and_a_missing_one_is_refused() {
        let temp = Temp::new();
        let outside = temp.path.join("elsewhere.mp4");
        fs::write(&outside, b"linked").unwrap();
        let project = project(&temp.path, true, Some(&outside));
        let root = temp.path.join("Templates");
        let template = save_as_template(&project, &root, "Linked", "").unwrap();
        let draft = Folder::read(&template).unwrap().draft;
        let linked = draft
            .materials
            .videos
            .iter()
            .find(|video| video.id.as_str() == "w")
            .unwrap();
        assert_eq!(linked.path, "Resources/elsewhere.mp4");
        assert_eq!(fs::read(template.join(&linked.path)).unwrap(), b"linked");

        fs::remove_file(&outside).unwrap();
        let refused = save_as_template(&project, &root, "Broken", "");
        assert!(matches!(refused, Err(StoreError::MissingMedia { path }) if path == outside));
        assert!(!root.join("Broken").exists());
    }

    #[test]
    fn a_name_that_names_a_path_stays_inside_the_root() {
        let temp = Temp::new();
        let project = project(&temp.path, true, None);
        let root = temp.path.join("Templates");
        let template = save_as_template(&project, &root, "../../escape/me", "").unwrap();
        assert_eq!(template.parent(), Some(root.as_path()));
        let again = save_as_template(&project, &root, "../../escape/me", "").unwrap();
        assert_ne!(again, template);
        assert_eq!(again.parent(), Some(root.as_path()));
    }

    #[test]
    fn using_a_template_makes_a_new_project_and_leaves_the_template() {
        let temp = Temp::new();
        let project = project(&temp.path, true, None);
        let root = temp.path.join("Templates");
        let template = save_as_template(&project, &root, "Beach day", "").unwrap();
        let projects = temp.path.join("Projects");
        let first = use_template(&template, &projects).unwrap();
        let second = use_template(&template, &projects).unwrap();
        assert_ne!(first, second);
        for made in [&first, &second] {
            assert!(!made.join(TEMPLATE_FILE).exists());
            let draft = Folder::read(made).unwrap().draft;
            assert_eq!(draft.name, "Beach day");
            assert!(draft.tracks[0].segments[0].template_mark.is_some());
            assert!(made.join("Resources").join("beach.mp4").is_file());
        }
        assert!(listed_template(&template).is_some());
    }

    #[test]
    fn a_folder_that_is_not_a_template_is_not_listed_used_or_deleted() {
        let temp = Temp::new();
        let plain = project(&temp.path, true, None);
        assert!(listed_template(&plain).is_none());
        assert!(use_template(&plain, &temp.path.join("Projects")).is_err());
        assert!(delete_template(&plain).is_err());
        assert!(plain.is_dir());
        assert!(list_templates(&temp.path.join("missing")).is_empty());
    }

    #[test]
    fn a_newer_or_broken_template_file_is_skipped() {
        let temp = Temp::new();
        let project = project(&temp.path, true, None);
        let root = temp.path.join("Templates");
        let template = save_as_template(&project, &root, "T", "").unwrap();
        fs::write(
            template.join(TEMPLATE_FILE),
            br#"{"format_version": 99, "name": "T"}"#,
        )
        .unwrap();
        assert!(listed_template(&template).is_none());
        fs::write(template.join(TEMPLATE_FILE), b"{not json").unwrap();
        assert!(list_templates(&root).is_empty());
    }

    #[test]
    fn deleting_a_template_removes_its_folder() {
        let temp = Temp::new();
        let project = project(&temp.path, true, None);
        let root = temp.path.join("Templates");
        let template = save_as_template(&project, &root, "T", "").unwrap();
        delete_template(&template).unwrap();
        assert!(!template.exists());
        assert!(project.is_dir());
    }
}
