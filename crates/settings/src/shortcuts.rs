//! Key bindings.
//!
//! A shortcut change is not a draft undo step. The binding is an action
//! name and the keys that run it. A field this record does not name stays
//! on the binding.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Settings;

/// Every action a shortcut can run, with its name in the settings window,
/// in the order the window lists them.
pub const ACTIONS: &[(&str, &str)] = &[
    ("play", "Play or pause"),
    ("frame_back", "Back one frame"),
    ("frame_forward", "Forward one frame"),
    ("shuttle_back", "Shuttle back"),
    ("shuttle_stop", "Shuttle stop"),
    ("shuttle_forward", "Shuttle forward"),
    ("mark_in", "Mark in"),
    ("mark_out", "Mark out"),
    ("clear_marks", "Clear marks"),
    ("undo", "Undo"),
    ("redo", "Redo"),
    ("redo_y", "Redo (second key)"),
    ("split", "Split at the playhead"),
    ("delete", "Delete"),
    ("ripple_delete", "Ripple delete"),
    ("delete_left", "Delete left of the playhead"),
    ("delete_right", "Delete right of the playhead"),
    ("copy", "Copy"),
    ("cut", "Cut"),
    ("paste", "Paste"),
    ("select_all", "Select all"),
    ("extract_audio", "Extract audio"),
    ("deactivate", "Turn a clip off or on"),
    ("settings", "Settings"),
];

/// The keys an action has before the user changes them. Keys are names
/// joined by `+`: `ctrl`, `primary` (Command on a Mac, Control elsewhere),
/// `shift`, `alt`, then the key.
#[must_use]
pub fn default_keys(action: &str) -> Option<&'static str> {
    Some(match action {
        "play" => "space",
        "frame_back" => "left",
        "frame_forward" => "right",
        "shuttle_back" => "j",
        "shuttle_stop" => "k",
        "shuttle_forward" => "l",
        "mark_in" => "i",
        "mark_out" => "o",
        "clear_marks" => "alt+x",
        "undo" => "primary+z",
        "redo" => "primary+shift+z",
        // Control-Y is the second redo. Away from a Mac, Control is the
        // primary key.
        "redo_y" if cfg!(target_os = "macos") => "ctrl+y",
        "redo_y" => "primary+y",
        "split" => "s",
        "delete" => "delete",
        "ripple_delete" => "shift+delete",
        "delete_left" => "q",
        "delete_right" => "w",
        "copy" => "primary+c",
        "cut" => "primary+x",
        "paste" => "primary+v",
        "select_all" => "primary+a",
        "extract_audio" => "primary+shift+s",
        "deactivate" => "v",
        "settings" => "primary+comma",
        _ => return None,
    })
}

/// The name of an action in the settings window, or the action itself.
#[must_use]
pub fn action_label(action: &str) -> &str {
    ACTIONS
        .iter()
        .find(|(name, _)| *name == action)
        .map_or(action, |(_, label)| label)
}

/// `keys` as a person reads them: `⇧⌘S` on a Mac, `Ctrl+Shift+S`
/// elsewhere. Empty keys read as an empty string.
#[must_use]
pub fn keys_label(keys: &str, mac: bool) -> String {
    let mut out = String::new();
    let parts: Vec<&str> = keys.split('+').filter(|part| !part.is_empty()).collect();
    let Some((key, modifiers)) = parts.split_last() else {
        return out;
    };
    let order: &[&str] = if mac {
        &["ctrl", "alt", "shift", "primary"]
    } else {
        &["primary", "ctrl", "alt", "shift"]
    };
    for name in order {
        if !modifiers.contains(name) {
            continue;
        }
        let shown = match (*name, mac) {
            ("ctrl", true) => "⌃",
            ("alt", true) => "⌥",
            ("shift", true) => "⇧",
            ("primary", true) => "⌘",
            ("ctrl" | "primary", false) => "Ctrl+",
            ("alt", false) => "Alt+",
            ("shift", false) => "Shift+",
            _ => "",
        };
        out.push_str(shown);
    }
    let key = match *key {
        "space" => "Space".to_owned(),
        "left" => if mac { "←" } else { "Left" }.to_owned(),
        "right" => if mac { "→" } else { "Right" }.to_owned(),
        "up" => if mac { "↑" } else { "Up" }.to_owned(),
        "down" => if mac { "↓" } else { "Down" }.to_owned(),
        "delete" => if mac { "⌫" } else { "Delete" }.to_owned(),
        "comma" => ",".to_owned(),
        "period" => ".".to_owned(),
        other => other.to_uppercase(),
    };
    out.push_str(&key);
    out
}

impl Settings {
    /// Give every action without a binding its default keys. A binding the
    /// user cleared stays cleared, and a default someone else already holds
    /// is left off.
    pub fn ensure_default_shortcuts(&mut self) {
        for (action, _) in ACTIONS {
            if self.shortcuts.iter().any(|shortcut| shortcut.action == *action) {
                continue;
            }
            let keys = default_keys(action).unwrap_or_default();
            let taken = self.shortcuts.iter().any(|shortcut| shortcut.keys == keys);
            self.shortcuts.push(Shortcut {
                action: (*action).to_owned(),
                keys: if taken { String::new() } else { keys.to_owned() },
                ..Shortcut::default()
            });
        }
    }

    /// The action `keys` run, if any.
    #[must_use]
    pub fn action_for(&self, keys: &str) -> Option<&str> {
        if keys.is_empty() {
            return None;
        }
        self.shortcuts
            .iter()
            .find(|shortcut| shortcut.keys == keys)
            .map(|shortcut| shortcut.action.as_str())
    }

    /// The keys that run `action`, empty when none.
    #[must_use]
    pub fn keys_for(&self, action: &str) -> &str {
        self.shortcuts
            .iter()
            .find(|shortcut| shortcut.action == action)
            .map_or("", |shortcut| shortcut.keys.as_str())
    }

    /// Bind `keys` to `action`. Another action that held the same keys
    /// loses them, and its name comes back. Empty keys clear the binding.
    pub fn bind(&mut self, action: &str, keys: &str) -> Option<String> {
        let mut displaced = None;
        if !keys.is_empty() {
            for shortcut in &mut self.shortcuts {
                if shortcut.keys == keys && shortcut.action != action {
                    shortcut.keys.clear();
                    displaced = Some(shortcut.action.clone());
                }
            }
        }
        match self
            .shortcuts
            .iter_mut()
            .find(|shortcut| shortcut.action == action)
        {
            Some(shortcut) => keys.clone_into(&mut shortcut.keys),
            None => self.shortcuts.push(Shortcut {
                action: action.to_owned(),
                keys: keys.to_owned(),
                ..Shortcut::default()
            }),
        }
        displaced
    }

    /// Put every known action back on its default keys. Bindings for
    /// actions this build does not know are kept.
    pub fn reset_shortcuts(&mut self) {
        self.shortcuts
            .retain(|shortcut| !ACTIONS.iter().any(|(action, _)| *action == shortcut.action));
        self.ensure_default_shortcuts();
    }
}

/// One action and the keys that run it.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Shortcut {
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub keys: String,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{ACTIONS, Shortcut, action_label, default_keys, keys_label};
    use crate::Settings;

    #[test]
    fn every_action_has_a_label_and_distinct_default_keys() {
        let mut seen = Vec::new();
        for (action, label) in ACTIONS {
            assert!(!label.is_empty());
            assert_eq!(action_label(action), *label);
            let keys = default_keys(action).unwrap();
            assert!(!seen.contains(&keys), "{keys} twice");
            seen.push(keys);
        }
        assert_eq!(action_label("future"), "future");
    }

    #[test]
    fn defaults_fill_the_gaps_and_keep_what_the_user_set() {
        let mut settings = Settings::default();
        settings.bind("undo", "primary+y");
        settings.bind("split", "");
        settings.ensure_default_shortcuts();
        assert_eq!(settings.keys_for("undo"), "primary+y");
        assert_eq!(settings.keys_for("split"), "");
        assert_eq!(settings.keys_for("play"), "space");
        assert_eq!(settings.action_for("primary+z"), None);
        assert_eq!(settings.action_for(""), None);
        assert_eq!(settings.shortcuts.len(), ACTIONS.len());
        // Off a Mac, the second redo's default is what undo now holds.
        if !cfg!(target_os = "macos") {
            assert_eq!(settings.keys_for("redo_y"), "");
        }
    }

    #[test]
    fn binding_taken_keys_moves_them_and_names_the_loser() {
        let mut settings = Settings::default();
        settings.ensure_default_shortcuts();
        assert_eq!(settings.bind("split", "space").as_deref(), Some("play"));
        assert_eq!(settings.action_for("space"), Some("split"));
        assert_eq!(settings.keys_for("play"), "");
        assert_eq!(settings.bind("split", "space"), None);
        settings.bind("future", "f");
        settings.reset_shortcuts();
        assert_eq!(settings.keys_for("play"), "space");
        assert_eq!(settings.keys_for("split"), "s");
        assert_eq!(settings.keys_for("future"), "f");
    }

    #[test]
    fn keys_read_as_each_system_writes_them() {
        assert_eq!(keys_label("primary+shift+s", true), "⇧⌘S");
        assert_eq!(keys_label("primary+shift+s", false), "Ctrl+Shift+S");
        assert_eq!(keys_label("alt+x", true), "⌥X");
        assert_eq!(keys_label("ctrl+y", true), "⌃Y");
        assert_eq!(keys_label("space", false), "Space");
        assert_eq!(keys_label("shift+delete", true), "⇧⌫");
        assert_eq!(keys_label("primary+comma", true), "⌘,");
        assert_eq!(keys_label("", true), "");
    }

    #[test]
    fn a_binding_keeps_an_unnamed_field() {
        let shortcut: Shortcut =
            serde_json::from_str(r#"{"action":"split","keys":"s","future_key":true}"#).unwrap();
        assert_eq!(shortcut.action, "split");
        assert_eq!(shortcut.keys, "s");
        assert_eq!(shortcut.rest["future_key"], true);
        let value = serde_json::to_value(&shortcut).unwrap();
        assert_eq!(value["action"], "split");
        assert_eq!(value["keys"], "s");
        assert_eq!(value["future_key"], true);
    }

    #[test]
    fn a_missing_binding_is_empty() {
        let shortcut: Shortcut = serde_json::from_str("{}").unwrap();
        assert_eq!(shortcut, Shortcut::default());
        assert!(shortcut.action.is_empty());
        assert!(shortcut.keys.is_empty());
        assert!(shortcut.rest.is_empty());
    }
}
