//! What a project says about proxies: whether it makes them by itself, and
//! the files the user chose one way or the other.
//!
//! It is kept under `proxies` in `draft_biz_config.json`, beside the
//! timeline settings. The proxy files are not in the project.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::Folder;

const PROXIES: &str = "proxies";
const ENABLED: &str = "enabled";
const FILES: &str = "files";

/// A project's proxy choices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectProxies {
    /// Whether the project makes proxies by itself. `None` when the
    /// project has never said, so the app's own setting decides.
    pub enabled: Option<bool>,
    /// The files chosen by hand, by the path their material stores: true
    /// has a proxy whatever the project makes by itself, false has none.
    pub files: BTreeMap<String, bool>,
}

impl ProjectProxies {
    /// Whether the file stored at `path` plays a proxy, when the project
    /// would make one by itself only if `automatic`.
    #[must_use]
    pub fn wants(&self, path: &str, automatic: bool) -> bool {
        self.files.get(path).copied().unwrap_or(automatic)
    }
}

impl Folder {
    /// The project's proxy choices. Anything that is not the shape
    /// [`Self::set_proxies`] writes reads as never said.
    #[must_use]
    pub fn proxies(&self) -> ProjectProxies {
        let Some(stored) = self
            .biz_config
            .as_ref()
            .and_then(|biz| biz.get(PROXIES))
            .and_then(Value::as_object)
        else {
            return ProjectProxies::default();
        };
        ProjectProxies {
            enabled: stored.get(ENABLED).and_then(Value::as_bool),
            files: stored
                .get(FILES)
                .and_then(Value::as_object)
                .map(|files| {
                    files
                        .iter()
                        .filter_map(|(path, on)| Some((path.clone(), on.as_bool()?)))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// Record the project's proxy choices. Other keys in the file stay.
    pub fn set_proxies(&mut self, proxies: &ProjectProxies) {
        let Some(biz) = self
            .biz_config
            .get_or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
        else {
            return;
        };
        let entry = biz
            .entry(PROXIES)
            .or_insert_with(|| Value::Object(Map::new()));
        if !entry.is_object() {
            *entry = Value::Object(Map::new());
        }
        let Some(stored) = entry.as_object_mut() else {
            return;
        };
        match proxies.enabled {
            Some(on) => stored.insert(ENABLED.to_owned(), Value::Bool(on)),
            None => stored.remove(ENABLED),
        };
        let files = proxies
            .files
            .iter()
            .map(|(path, on)| (path.clone(), Value::Bool(*on)))
            .collect();
        stored.insert(FILES.to_owned(), Value::Object(files));
    }
}

#[cfg(test)]
mod tests {
    use super::ProjectProxies;
    use crate::{Draft, Folder};

    #[test]
    fn a_project_that_never_said_reads_as_empty() {
        let mut folder = Folder::new(Draft::empty("D"));
        assert_eq!(folder.proxies(), ProjectProxies::default());
        for hostile in [
            serde_json::json!([1, 2]),
            serde_json::json!({ "proxies": "on" }),
            serde_json::json!({ "proxies": { "enabled": "yes", "files": [true] } }),
            serde_json::json!({ "proxies": { "files": { "a.mp4": 1, "b.mp4": null } } }),
        ] {
            folder.biz_config = Some(hostile);
            assert_eq!(folder.proxies(), ProjectProxies::default());
        }
    }

    #[test]
    fn a_file_chosen_by_hand_wins_over_what_the_project_makes() {
        let mut proxies = ProjectProxies::default();
        proxies.files.insert("on.mp4".to_owned(), true);
        proxies.files.insert("off.mp4".to_owned(), false);
        for automatic in [false, true] {
            assert!(proxies.wants("on.mp4", automatic));
            assert!(!proxies.wants("off.mp4", automatic));
            assert_eq!(proxies.wants("other.mp4", automatic), automatic);
        }
    }

    #[test]
    fn the_choices_come_back_from_disk_and_keep_the_other_keys() {
        let temp = std::env::temp_dir().join(format!("cutlass-proxies-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut folder = Folder::new(Draft::empty("D"));
        folder.biz_config = Some(serde_json::json!({
            "kept": 1,
            "proxies": { "later": "kept too", "enabled": false }
        }));
        let mut proxies = ProjectProxies {
            enabled: Some(true),
            ..ProjectProxies::default()
        };
        proxies.files.insert("Resources/a.mp4".to_owned(), false);
        proxies.files.insert("/movies/b.mov".to_owned(), true);
        folder.set_proxies(&proxies);
        folder.save(&temp).unwrap();

        let mut opened = crate::open(&temp).unwrap().folder;
        assert_eq!(opened.proxies(), proxies);
        let biz = opened.biz_config.clone().unwrap();
        assert_eq!(biz["kept"], 1);
        assert_eq!(biz["proxies"]["later"], "kept too");

        // Never said again, and a file no longer chosen, leave no trace.
        opened.set_proxies(&ProjectProxies::default());
        assert_eq!(opened.proxies(), ProjectProxies::default());
        assert!(
            opened.biz_config.unwrap()["proxies"]
                .get("enabled")
                .is_none()
        );
        std::fs::remove_dir_all(&temp).unwrap();
    }

    #[test]
    fn a_config_that_is_not_an_object_is_left_alone() {
        let mut folder = Folder::new(Draft::empty("D"));
        folder.biz_config = Some(serde_json::json!([1, 2]));
        folder.set_proxies(&ProjectProxies {
            enabled: Some(true),
            ..ProjectProxies::default()
        });
        assert_eq!(folder.biz_config, Some(serde_json::json!([1, 2])));
    }
}
