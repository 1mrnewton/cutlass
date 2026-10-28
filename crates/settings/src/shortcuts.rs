//! Key bindings.
//!
//! A shortcut change is not a draft undo step. The binding is an action
//! name and the keys that run it. A field this record does not name stays
//! on the binding.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
    use super::Shortcut;

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
