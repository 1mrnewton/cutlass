use super::*;

#[test]
fn catalog_ids_are_unique() {
    fn assert_unique(ids: Vec<&str>, what: &str) {
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate {what} id");
    }
    assert_unique(mask_catalog().iter().map(|s| s.kind.id()).collect(), "mask");
    assert_unique(filter_catalog().iter().map(|s| s.id).collect(), "filter");
    assert_unique(
        animation_catalog().iter().map(|s| s.id).collect(),
        "animation",
    );
    assert_unique(
        text_effect_catalog().iter().map(|s| s.id).collect(),
        "text effect",
    );
}

#[test]
fn enum_ids_match_their_serde_names() {
    for spec in mask_catalog() {
        let json = serde_json::to_value(spec.kind).unwrap();
        assert_eq!(json, serde_json::json!(spec.kind.id()));
    }
    for level in StabilizeLevel::ALL {
        let json = serde_json::to_value(level).unwrap();
        assert_eq!(json, serde_json::json!(level.id()));
    }
    for role in AudioRole::ALL {
        let json = serde_json::to_value(role).unwrap();
        assert_eq!(json, serde_json::json!(role.id()));
    }
}

#[test]
fn defaults_are_elided_from_the_wire() {
    let mask = Mask::new(MaskKind::Circle);
    assert_eq!(
        serde_json::to_value(mask).unwrap(),
        serde_json::json!({"kind": "circle"})
    );

    let chroma = ChromaKey {
        rgb: [0, 255, 0],
        strength: 0.0.into(),
        shadow: 0.0.into(),
    };
    assert_eq!(
        serde_json::to_value(chroma).unwrap(),
        serde_json::json!({"rgb": [0, 255, 0]})
    );

    let filter = Filter::new("vivid");
    assert_eq!(
        serde_json::to_value(filter).unwrap(),
        serde_json::json!({"id": "vivid"})
    );
}

#[test]
fn validation_rejects_out_of_range_values() {
    let mut mask = Mask::new(MaskKind::Linear);
    mask.feather = 1.5.into();
    assert!(mask.validate().is_err());

    let chroma = ChromaKey {
        rgb: [0, 255, 0],
        strength: (-0.1).into(),
        shadow: 0.0.into(),
    };
    assert!(chroma.validate().is_err());

    assert!(Filter::new("nope").validate().is_err());
    let mut filter = Filter::new("vivid");
    filter.intensity = 2.0.into();
    assert!(filter.validate().is_err());

    let adjust = ColorAdjustments {
        brightness: (-1.5).into(),
        ..Default::default()
    };
    assert!(adjust.validate().is_err());
    assert!(ColorAdjustments::default().is_neutral());
}

#[test]
fn animation_catalog_slots_and_text_flags() {
    assert_eq!(animation_spec("fade_in").unwrap().slot, AnimationSlot::In);
    assert_eq!(animation_spec("drop").unwrap().slot, AnimationSlot::Out);
    assert_eq!(animation_spec("pulse").unwrap().slot, AnimationSlot::Combo);
    assert!(animation_spec("typewriter").unwrap().text_only);
    assert!(animation_spec("missing").is_none());
    // Whole-layer presets expose speed/intensity; per-char also stagger.
    let fade = animation_spec("fade_in").unwrap();
    assert!(fade.knobs.speed && fade.knobs.intensity && !fade.knobs.stagger);
    let wave = animation_spec("wave").unwrap();
    assert!(wave.knobs.speed && wave.knobs.intensity && wave.knobs.stagger);
}

#[test]
fn animation_ref_defaults_and_normalization() {
    let spec = animation_spec("wave").unwrap();
    let a = AnimationRef::new("wave");
    assert_eq!(a.speed, ANIMATION_PARAM_DEFAULT);
    let ok = a.clone().normalized_for(spec).unwrap();
    assert_eq!(ok.intensity, 1.0);

    let mut hot = AnimationRef::new("wave");
    hot.intensity = 1.5;
    hot.stagger = 0.5;
    let hot = hot.normalized_for(spec).unwrap();
    assert_eq!(hot.intensity, 1.5);
    assert_eq!(hot.stagger, 0.5);

    let mut bad = AnimationRef::new("wave");
    bad.speed = 99.0;
    assert!(bad.normalized_for(spec).is_err());

    // Unsupported knobs snap back to default.
    let fade = animation_spec("fade_in").unwrap();
    let mut staggered = AnimationRef::new("fade_in");
    staggered.stagger = 1.5;
    let cleaned = staggered.normalized_for(fade).unwrap();
    assert_eq!(cleaned.stagger, ANIMATION_PARAM_DEFAULT);
}

#[test]
fn animation_ref_serde_omits_default_knobs() {
    let a = AnimationRef::new("pulse");
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::json!({"id": "pulse"})
    );
    let mut tuned = AnimationRef::new("pulse");
    tuned.speed = 2.0;
    assert_eq!(
        serde_json::to_value(&tuned).unwrap(),
        serde_json::json!({"id": "pulse", "speed": 2.0})
    );
    // Old id-only JSON still loads with defaults.
    let loaded: AnimationRef = serde_json::from_value(serde_json::json!({"id": "pulse"})).unwrap();
    assert_eq!(loaded.speed, ANIMATION_PARAM_DEFAULT);
    assert_eq!(loaded.intensity, ANIMATION_PARAM_DEFAULT);
}

#[test]
fn text_effect_presets_resolve() {
    let neon = text_effect_spec("neon").unwrap();
    assert!(neon.stroke.is_some() && neon.shadow.is_some());
    assert!(text_effect_spec("nope").is_none());
}
