//! The menu bar: every command the window has, in menus. On macOS it is
//! the system's own menu bar; the window's menus run the same commands.
//!
//! An item's id is either a shortcut action from the settings, run by
//! [`crate::shell::Shell::run_action`], or one of the window commands
//! below, which open a dialog or a page.

/// One item: its id and the name the menu shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub id: &'static str,
    pub label: &'static str,
}

/// One menu: its title and items, `None` for a separator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Menu {
    pub title: &'static str,
    pub items: &'static [Option<Item>],
}

const fn item(id: &'static str, label: &'static str) -> Option<Item> {
    Some(Item { id, label })
}

/// Commands the window runs itself, apart from the shortcut actions.
pub const OPEN: &str = "open";
pub const SAVE: &str = "save";
pub const IMPORT: &str = "import";
pub const EXPORT: &str = "export";
pub const PROJECTS: &str = "projects";
pub const SETTINGS: &str = "settings";
pub const LICENSES: &str = "licenses";
pub const EXPORT_LOG: &str = "export_log";

/// The menus, in order. The app menu of macOS (About, Hide, Quit) comes
/// before them there.
pub const MENUS: &[Menu] = &[
    Menu {
        title: "File",
        items: &[
            item(OPEN, "Open…"),
            item(PROJECTS, "Projects"),
            None,
            item(SAVE, "Save"),
            item(IMPORT, "Import…"),
            item(EXPORT, "Export…"),
            None,
            item(SETTINGS, "Settings…"),
        ],
    },
    Menu {
        title: "Edit",
        items: &[
            item("undo", "Undo"),
            item("redo", "Redo"),
            None,
            item("cut", "Cut"),
            item("copy", "Copy"),
            item("paste", "Paste"),
            item("delete", "Delete"),
            item("ripple_delete", "Ripple delete"),
            item("select_all", "Select all"),
        ],
    },
    Menu {
        title: "Clip",
        items: &[
            item("split", "Split at the playhead"),
            item("delete_left", "Delete left of the playhead"),
            item("delete_right", "Delete right of the playhead"),
            None,
            item("extract_audio", "Extract audio"),
            item("deactivate", "Turn the clip off or on"),
            item("compound", "Create compound clip"),
            item("speed_curves", "Show variable speed animation"),
        ],
    },
    Menu {
        title: "Playback",
        items: &[
            item("play", "Play or pause"),
            item("frame_back", "Back one frame"),
            item("frame_forward", "Forward one frame"),
            None,
            item("shuttle_back", "Shuttle back"),
            item("shuttle_stop", "Shuttle stop"),
            item("shuttle_forward", "Shuttle forward"),
            None,
            item("mark_in", "Mark in"),
            item("mark_out", "Mark out"),
            item("clear_marks", "Clear marks"),
        ],
    },
    Menu {
        title: "Help",
        items: &[item(LICENSES, "Licenses"), item(EXPORT_LOG, "Export log…")],
    },
];

/// Every item, menu by menu.
pub fn items() -> impl Iterator<Item = &'static Item> {
    MENUS.iter().flat_map(|menu| menu.items.iter().flatten())
}

/// Whether `id` is a command the window runs itself.
#[must_use]
pub fn is_window_command(id: &str) -> bool {
    [
        OPEN, SAVE, IMPORT, EXPORT, PROJECTS, SETTINGS, LICENSES, EXPORT_LOG,
    ]
    .contains(&id)
}

/// The system menu bar on macOS, built from [`MENUS`]. A chosen item is
/// waiting in [`Native::take`].
#[cfg(target_os = "macos")]
pub struct Native {
    _menu: muda::Menu,
    chosen: std::sync::mpsc::Receiver<String>,
}

#[cfg(target_os = "macos")]
impl Native {
    /// Build the menu bar and put it up. `wake` asks for a frame when an
    /// item is chosen.
    pub fn install(wake: impl Fn() + Send + Sync + 'static) -> Option<Self> {
        use muda::{AboutMetadata, Menu as Bar, MenuItem, PredefinedMenuItem, Submenu};

        let bar = Bar::new();
        let app = Submenu::new("Cutlass", true);
        let about = AboutMetadata {
            name: Some("Cutlass".to_owned()),
            version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            ..AboutMetadata::default()
        };
        app.append_items(&[
            &PredefinedMenuItem::about(None, Some(about)),
            &PredefinedMenuItem::separator(),
            &MenuItem::with_id(SETTINGS, "Settings…", true, None),
            &PredefinedMenuItem::separator(),
            &PredefinedMenuItem::services(None),
            &PredefinedMenuItem::separator(),
            &PredefinedMenuItem::hide(None),
            &PredefinedMenuItem::hide_others(None),
            &PredefinedMenuItem::show_all(None),
            &PredefinedMenuItem::separator(),
            &PredefinedMenuItem::quit(None),
        ])
        .ok()?;
        bar.append(&app).ok()?;
        for menu in MENUS {
            let submenu = Submenu::new(menu.title, true);
            for entry in menu.items {
                let added = match entry {
                    Some(item) => {
                        submenu.append(&MenuItem::with_id(item.id, item.label, true, None))
                    }
                    None => submenu.append(&PredefinedMenuItem::separator()),
                };
                added.ok()?;
            }
            bar.append(&submenu).ok()?;
        }
        let window = Submenu::new("Window", true);
        window
            .append_items(&[
                &PredefinedMenuItem::minimize(None),
                &PredefinedMenuItem::fullscreen(None),
            ])
            .ok()?;
        bar.append(&window).ok()?;
        window.set_as_windows_menu_for_nsapp();
        bar.init_for_nsapp();
        let (sender, chosen) = std::sync::mpsc::channel();
        muda::MenuEvent::set_event_handler(Some(move |event: muda::MenuEvent| {
            let _ = sender.send(event.id().0.clone());
            wake();
        }));
        Some(Self { _menu: bar, chosen })
    }

    /// The items chosen since the last call, oldest first.
    pub fn take(&self) -> Vec<String> {
        self.chosen.try_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_is_in_a_menu_once() {
        let ids: Vec<&str> = items().map(|item| item.id).collect();
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "no item twice");
        for (action, _) in cutlass_settings::ACTIONS {
            // The second redo key is the same command as Redo.
            if *action == "redo_y" {
                continue;
            }
            assert!(ids.contains(action), "{action} has no menu item");
        }
        for id in &ids {
            assert!(
                is_window_command(id)
                    || cutlass_settings::ACTIONS
                        .iter()
                        .any(|(action, _)| action == id),
                "{id} runs nothing"
            );
        }
    }
}
