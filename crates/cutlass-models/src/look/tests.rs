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
    for mode in BlendMode::ALL {
        let json = serde_json::to_value(mode).unwrap();
        assert_eq!(json, serde_json::json!(mode.id()));
    }
}

#[test]
fn blend_mode_id_label_roundtrip() {
    for mode in BlendMode::ALL {
        assert_eq!(BlendMode::from_id(mode.id()), Some(*mode));
        assert!(!mode.label().is_empty());
    }
    assert!(BlendMode::from_id("nope").is_none());
    assert!(BlendMode::Normal.is_normal());
    assert!(!BlendMode::Multiply.is_normal());
}

#[test]
fn motion_blur_defaults_and_validation() {
    let defaults = MotionBlur::default();
    assert!(defaults.is_default());
    assert!(defaults.validate().is_ok());

    assert!(
        MotionBlur {
            enabled: true,
            ..MotionBlur::default()
        }
        .validate()
        .is_ok()
    );
    assert!(
        MotionBlur {
            enabled: false,
            shutter_deg: 0.0,
            samples: 2,
        }
        .validate()
        .is_ok()
    );
    assert!(
        MotionBlur {
            enabled: true,
            shutter_deg: 720.0,
            samples: 32,
        }
        .validate()
        .is_ok()
    );

    assert!(
        MotionBlur {
            enabled: true,
            shutter_deg: -1.0,
            samples: 8,
        }
        .validate()
        .is_err()
    );
    assert!(
        MotionBlur {
            enabled: true,
            shutter_deg: 721.0,
            samples: 8,
        }
        .validate()
        .is_err()
    );
    assert!(
        MotionBlur {
            enabled: true,
            shutter_deg: f32::NAN,
            samples: 8,
        }
        .validate()
        .is_err()
    );
    assert!(
        MotionBlur {
            enabled: true,
            shutter_deg: 180.0,
            samples: 1,
        }
        .validate()
        .is_err()
    );
    assert!(
        MotionBlur {
            enabled: true,
            shutter_deg: 180.0,
            samples: 33,
        }
        .validate()
        .is_err()
    );
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
fn legacy_mask_json_deserializes_default_geometry() {
    let mask: Mask = serde_json::from_value(serde_json::json!({"kind": "circle"})).unwrap();
    assert_eq!(mask.kind, MaskKind::Circle);
    assert_eq!(mask.center, Param::Constant([0.0, 0.0]));
    assert_eq!(mask.size, Param::Constant([1.0, 1.0]));
    assert_eq!(mask.rotation, Param::Constant(0.0));
    assert_eq!(mask.roundness, Param::Constant(0.0));
    assert_eq!(mask.feather, Param::Constant(0.0));
    assert!(!mask.invert);
}

#[test]
fn default_geometry_mask_omits_new_keys_and_keyframed_center_roundtrips() {
    let mask = Mask::new(MaskKind::Rectangle);
    let value = serde_json::to_value(&mask).unwrap();
    let obj = value.as_object().unwrap();
    assert!(!obj.contains_key("center"));
    assert!(!obj.contains_key("size"));
    assert!(!obj.contains_key("rotation"));
    assert!(!obj.contains_key("roundness"));

    let mut keyed = Mask::new(MaskKind::Circle);
    keyed.center = Param::Keyframed {
        keyframes: vec![
            crate::param::Keyframe {
                tick: 0,
                value: [0.0, 0.0],
                easing: crate::param::Easing::Linear,
                tangents: None,
            },
            crate::param::Keyframe {
                tick: 10,
                value: [0.25, -0.1],
                easing: crate::param::Easing::EaseIn,
                tangents: None,
            },
        ],
    };
    let json = serde_json::to_string(&keyed).unwrap();
    assert!(json.contains("\"center\""));
    assert!(json.contains("\"kf\""));
    let loaded: Mask = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.center, keyed.center);
}

#[test]
fn validation_rejects_out_of_range_values() {
    let mut mask = Mask::new(MaskKind::Linear);
    mask.feather = 1.5.into();
    assert!(mask.validate().is_err());

    let mut size_zero = Mask::new(MaskKind::Circle);
    size_zero.size = Param::Constant([0.0, 1.0]);
    assert!(size_zero.validate().is_err());

    let mut roundness = Mask::new(MaskKind::Rectangle);
    roundness.roundness = 1.5.into();
    assert!(roundness.validate().is_err());

    let mut center = Mask::new(MaskKind::Circle);
    center.center = Param::Constant([20.0, 0.0]);
    assert!(center.validate().is_err());

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

    // Signed adjust sliders accept negatives; one-directional ones do not.
    assert!(
        ColorAdjustments {
            tint: (-0.5).into(),
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    assert!(
        ColorAdjustments {
            sharpness: (-0.5).into(),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        ColorAdjustments {
            vignette: 1.5.into(),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn color_adjustments_serde_skips_zeros_and_loads_legacy_json() {
    let neutral = ColorAdjustments::default();
    assert_eq!(
        serde_json::to_value(&neutral).unwrap(),
        serde_json::json!({})
    );

    let tuned = ColorAdjustments {
        tint: 0.25.into(),
        hue: (-0.5).into(),
        sharpness: 0.75.into(),
        ..Default::default()
    };
    let value = serde_json::to_value(&tuned).unwrap();
    let obj = value.as_object().unwrap();
    assert_eq!(obj.get("tint"), Some(&serde_json::json!(0.25)));
    assert_eq!(obj.get("hue"), Some(&serde_json::json!(-0.5)));
    assert_eq!(obj.get("sharpness"), Some(&serde_json::json!(0.75)));
    assert!(!obj.contains_key("brightness"));
    assert!(!obj.contains_key("vignette"));

    let roundtrip: ColorAdjustments = serde_json::from_value(value).unwrap();
    assert_eq!(roundtrip, tuned);

    // Pre-slider JSON without the new keys deserializes to neutral zeros.
    let legacy: ColorAdjustments = serde_json::from_value(serde_json::json!({
        "brightness": 0.1,
        "temperature": -0.2
    }))
    .unwrap();
    assert_eq!(legacy.brightness, 0.1.into());
    assert_eq!(legacy.temperature, (-0.2).into());
    assert_eq!(legacy.tint, 0.0.into());
    assert_eq!(legacy.hue, 0.0.into());
    assert_eq!(legacy.highlights, 0.0.into());
    assert_eq!(legacy.shadows, 0.0.into());
    assert_eq!(legacy.sharpness, 0.0.into());
    assert_eq!(legacy.vignette, 0.0.into());
    assert!(!legacy.is_neutral());
}

#[test]
fn layer_styles_defaults_and_empty() {
    assert!(LayerStyles::default().is_empty());
    let shadow = LayerShadow::default();
    assert_eq!(shadow.rgba, Param::Constant([0, 0, 0, 128]));
    assert_eq!(shadow.offset, Param::Constant([4.0, 4.0]));
    assert_eq!(shadow.blur, Param::Constant(8.0));
    assert!(LayerShadow::default().validate().is_ok());
    assert!(LayerGlow::default().validate().is_ok());
    assert!(LayerOutline::default().validate().is_ok());
    assert!(LayerBackground::default().validate().is_ok());
}

#[test]
fn layer_styles_validation_rejects_out_of_range_values() {
    let shadow = LayerShadow {
        blur: (-1.0).into(),
        ..Default::default()
    };
    assert!(shadow.validate().is_err());

    let glow = LayerGlow {
        intensity: 5.0.into(),
        ..Default::default()
    };
    assert!(glow.validate().is_err());

    let bad_offset = LayerShadow {
        offset: Param::Constant([f32::NAN, 4.0]),
        ..Default::default()
    };
    assert!(bad_offset.validate().is_err());

    let styles = LayerStyles {
        shadow: Some(shadow),
        ..Default::default()
    };
    assert!(styles.validate().is_err());
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
