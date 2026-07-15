use cutlass_models::{
    ClipId, ClipParam, Easing, Generator, ModelError, ParamValue, Project, Rational, RationalTime,
    TimeRange, TrackKind,
};

const R24: Rational = Rational::FPS_24;

fn project_with_effect_chain() -> (Project, ClipId) {
    let mut project = Project::new("effect order", R24);
    let track = project.add_track(TrackKind::Sticker, "Overlays");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [32, 64, 128, 255],
            },
            TimeRange::at_rate(0, 48, R24),
        )
        .unwrap();

    project.add_effect(clip, "gaussian_blur").unwrap();
    project.add_effect(clip, "glitch").unwrap();
    project.add_effect(clip, "vignette").unwrap();
    project.set_effect_param(clip, 0, 0, 12.0).unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Effect {
                effect: 1,
                param: 0,
            },
            RationalTime::new(0, R24),
            ParamValue::Scalar(0.2),
            Easing::EaseIn,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Effect {
                effect: 1,
                param: 0,
            },
            RationalTime::new(24, R24),
            ParamValue::Scalar(0.8),
            Easing::EaseOut,
        )
        .unwrap();
    project.set_effect_param(clip, 2, 0, 0.75).unwrap();

    (project, clip)
}

#[test]
fn move_effect_forward_and_backward_preserves_instances_exactly() {
    let (mut project, clip) = project_with_effect_chain();
    let before = project.clip(clip).unwrap().effects.clone();

    project.move_effect(clip, 0, 2).unwrap();
    assert_eq!(
        project.clip(clip).unwrap().effects,
        vec![before[1].clone(), before[2].clone(), before[0].clone()]
    );

    project.move_effect(clip, 2, 0).unwrap();
    assert_eq!(project.clip(clip).unwrap().effects, before);
}

#[test]
fn move_effect_rejects_invalid_indices_atomically_and_accepts_same_index() {
    let (mut project, clip) = project_with_effect_chain();
    let before = project.clip(clip).unwrap().effects.clone();
    let unknown = ClipId::from_raw(u64::MAX);

    assert_eq!(
        project.move_effect(unknown, 0, 1),
        Err(ModelError::UnknownClip(unknown))
    );
    assert!(matches!(
        project.move_effect(clip, 3, 0),
        Err(ModelError::InvalidParam(message))
            if message.contains("from index 3") && message.contains("chain length 3")
    ));
    assert_eq!(project.clip(clip).unwrap().effects, before);

    assert!(matches!(
        project.move_effect(clip, 0, 3),
        Err(ModelError::InvalidParam(message))
            if message.contains("to index 3") && message.contains("chain length 3")
    ));
    assert_eq!(project.clip(clip).unwrap().effects, before);

    project.move_effect(clip, 1, 1).unwrap();
    assert_eq!(project.clip(clip).unwrap().effects, before);
}
