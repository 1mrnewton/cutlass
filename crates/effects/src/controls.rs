//! The values a recipe effect's controls are set to, kept on its material.
//!
//! They live in the material's `adjust_params` as a list of
//! `{"name", "default_value", "value"}`, the shape the reference keeps an
//! effect's settings in. A control with no entry is at its default.

use std::collections::BTreeMap;

use cutlass_model::MaterialVideoEffect;
use serde_json::{Value, json};

/// The control values stored on `effect`, by control id.
#[must_use]
pub fn stored_controls(effect: &MaterialVideoEffect) -> BTreeMap<String, f32> {
    let Some(Value::Array(entries)) = &effect.adjust_params else {
        return BTreeMap::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name")?.as_str()?;
            let value = entry.get("value")?.as_f64()?;
            Some((name.to_owned(), value as f32))
        })
        .collect()
}

/// Store `value` for the control `id` on `effect`, with its `default`
/// alongside, replacing what it had.
pub fn store_control(effect: &mut MaterialVideoEffect, id: &str, value: f32, default: f32) {
    let mut entries = match effect.adjust_params.take() {
        Some(Value::Array(entries)) => entries,
        _ => Vec::new(),
    };
    entries.retain(|entry| entry.get("name").and_then(Value::as_str) != Some(id));
    entries.push(json!({
        "name": id,
        "default_value": f64::from(default),
        "value": f64::from(value),
    }));
    effect.adjust_params = Some(Value::Array(entries));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_value_reads_back_and_replaces_the_last() {
        let mut effect = MaterialVideoEffect::default();
        assert!(stored_controls(&effect).is_empty());
        store_control(&mut effect, "speed", 1.5, 1.0);
        store_control(&mut effect, "intensity", 0.5, 1.0);
        store_control(&mut effect, "speed", 2.0, 1.0);
        let stored = stored_controls(&effect);
        assert_eq!(stored.len(), 2);
        assert_eq!(stored.get("speed"), Some(&2.0));
        assert_eq!(stored.get("intensity"), Some(&0.5));
        // Something else in the field is not read as controls.
        effect.adjust_params = Some(json!({"size": 2}));
        assert!(stored_controls(&effect).is_empty());
    }
}
