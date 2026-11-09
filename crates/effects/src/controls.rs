//! The values a recipe effect's controls are set to, kept on its material.
//!
//! They live in the material's `adjust_params` as a list of
//! `{"name", "default_value", "value"}`, the shape the reference keeps an
//! effect's settings in. A control with no entry is at its default.

use std::collections::BTreeMap;

use cutlass_model::{KeyframeGraph, MaterialVideoEffect, Micros, Segment};

use crate::recipe::Recipe;
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

/// `stored` with each control of `recipe` that `segment` keys set as keyed
/// at timeline `time`. A speed control is never keyed. Keys that do not
/// read leave the stored value.
#[must_use]
pub fn keyed_controls(
    recipe: &Recipe,
    mut stored: BTreeMap<String, f32>,
    segment: &Segment,
    graphs: &[KeyframeGraph],
    time: Micros,
) -> BTreeMap<String, f32> {
    let offset = time.0.saturating_sub(segment.target.start.0);
    for control in recipe.controls.iter().filter(|control| control.keyable()) {
        if let Some(Some(value)) = cutlass_model::sample(
            &segment.common_keyframes,
            graphs,
            &[control.keyframe.as_str()],
            offset,
        ) {
            #[allow(clippy::cast_possible_truncation)]
            stored.insert(control.id.clone(), control.clamp(value as f32));
        }
    }
    stored
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
    fn a_keyed_control_moves_along_the_clip_and_speed_is_never_keyed() {
        let recipe = crate::bundled_recipe("recipe/8-bit").unwrap();
        let size = recipe.controls.iter().find(|c| c.id == "size").unwrap();
        assert!(size.keyable());
        assert_eq!(size.keyframe, "KFTypeEffectControl_size");
        let mut segment = Segment::new("s", "m", cutlass_model::TimeRange::new(1_000, 2_000));
        cutlass_model::put_key(&mut segment, &size.keyframe, Micros(0), 0.5);
        cutlass_model::put_key(&mut segment, &size.keyframe, Micros(2_000), 2.5);
        cutlass_model::put_key(&mut segment, &size.keyframe, Micros(4_000), 9.0);
        let at = |time: i64| keyed_controls(recipe, BTreeMap::new(), &segment, &[], Micros(time))["size"];
        assert_eq!(at(1_000), 0.5);
        // Past the range is held to the control's own.
        assert!((at(2_000) - 1.5).abs() < 1e-6);
        assert_eq!(at(5_000), size.max);
        // No key keeps what is stored.
        let plain = Segment::new("p", "m", cutlass_model::TimeRange::new(0, 1));
        let stored = BTreeMap::from([("size".to_owned(), 2.0)]);
        assert_eq!(
            keyed_controls(recipe, stored.clone(), &plain, &[], Micros(0)),
            stored
        );
        for recipe in crate::recipe::bundled() {
            for control in &recipe.controls {
                assert_eq!(control.keyable(), !control.speed, "{}", recipe.id);
                assert!(
                    cutlass_model::is_keyable(&control.keyframe),
                    "{}",
                    recipe.id
                );
            }
        }
    }

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
