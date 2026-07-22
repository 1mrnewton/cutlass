//! Resolve persisted clip look fields into compositor [`ColorGrade`] values.

use cutlass_compositor::ColorGrade;
use cutlass_models::{ColorAdjustments, Filter};

/// Hand-tuned [`ColorGrade`] recipe for one filter-catalog id.
///
/// Returns `None` for unknown ids so unknown presets behave as identity while
/// manual adjustments still apply.
pub(crate) fn preset_recipe(id: &str) -> Option<ColorGrade> {
    Some(match id {
        "vivid" => ColorGrade {
            saturation: 0.45,
            contrast: 0.15,
            ..ColorGrade::IDENTITY
        },
        "warm" => ColorGrade {
            temperature: 0.4,
            brightness: 0.05,
            ..ColorGrade::IDENTITY
        },
        "cool" => ColorGrade {
            temperature: -0.4,
            ..ColorGrade::IDENTITY
        },
        "mono" => ColorGrade {
            saturation: -1.0,
            contrast: 0.1,
            ..ColorGrade::IDENTITY
        },
        "fade" => ColorGrade {
            contrast: -0.3,
            brightness: 0.1,
            saturation: -0.2,
            ..ColorGrade::IDENTITY
        },
        "chrome" => ColorGrade {
            contrast: 0.25,
            saturation: 0.1,
            brightness: 0.05,
            ..ColorGrade::IDENTITY
        },
        "noir" => ColorGrade {
            saturation: -1.0,
            contrast: 0.35,
            brightness: -0.05,
            ..ColorGrade::IDENTITY
        },
        "sunset" => ColorGrade {
            temperature: 0.5,
            tint: 0.1,
            saturation: 0.15,
            ..ColorGrade::IDENTITY
        },
        "forest" => ColorGrade {
            tint: -0.25,
            saturation: 0.2,
            brightness: 0.03,
            ..ColorGrade::IDENTITY
        },
        "berry" => ColorGrade {
            tint: 0.35,
            temperature: -0.1,
            saturation: 0.25,
            ..ColorGrade::IDENTITY
        },
        _ => return None,
    })
}

/// Fold a clip's filter preset and manual adjustments into one compositor grade.
#[cfg(test)]
pub(crate) fn effective_grade(filter: Option<&Filter>, adjust: &ColorAdjustments) -> ColorGrade {
    effective_grade_at(filter, adjust, 0)
}

/// Fold a clip's filter preset and manual adjustments at a clip-local tick.
pub(crate) fn effective_grade_at(
    filter: Option<&Filter>,
    adjust: &ColorAdjustments,
    tick: i64,
) -> ColorGrade {
    let intensity = filter
        .map(|f| clamp_unit(f.intensity.sample(tick)))
        .unwrap_or(0.0);
    let preset = filter
        .and_then(|f| preset_recipe(&f.id))
        .unwrap_or(ColorGrade::IDENTITY);

    let mut grade = scale_grade(preset, intensity);
    grade.exposure += clamp_adjust(adjust.exposure.sample(tick));
    grade.brightness += clamp_adjust(adjust.brightness.sample(tick));
    grade.contrast += clamp_adjust(adjust.contrast.sample(tick));
    grade.saturation += clamp_adjust(adjust.saturation.sample(tick));
    grade.temperature += clamp_adjust(adjust.temperature.sample(tick));
    clamp_grade(grade)
}

/// Resolve a grade for compositor submission. `None` is the identity fast path.
#[cfg(test)]
pub(crate) fn resolve_color_grade(
    filter: Option<&Filter>,
    adjust: &ColorAdjustments,
) -> Option<ColorGrade> {
    let grade = effective_grade(filter, adjust);
    (!grade.is_identity()).then_some(grade)
}

/// Resolve a grade at a clip-local tick.
pub(crate) fn resolve_color_grade_at(
    filter: Option<&Filter>,
    adjust: &ColorAdjustments,
    tick: i64,
) -> Option<ColorGrade> {
    let grade = effective_grade_at(filter, adjust, tick);
    (!grade.is_identity()).then_some(grade)
}

fn scale_grade(grade: ColorGrade, intensity: f32) -> ColorGrade {
    ColorGrade {
        exposure: grade.exposure * intensity,
        brightness: grade.brightness * intensity,
        contrast: grade.contrast * intensity,
        saturation: grade.saturation * intensity,
        temperature: grade.temperature * intensity,
        tint: grade.tint * intensity,
    }
}

fn clamp_unit(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(0.0, 1.0)
}

fn clamp_adjust(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(-1.0, 1.0)
}

fn clamp_grade(mut grade: ColorGrade) -> ColorGrade {
    grade.exposure = clamp_adjust(grade.exposure);
    grade.brightness = clamp_adjust(grade.brightness);
    grade.contrast = clamp_adjust(grade.contrast);
    grade.saturation = clamp_adjust(grade.saturation);
    grade.temperature = clamp_adjust(grade.temperature);
    grade.tint = clamp_adjust(grade.tint);
    grade
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::filter_catalog;

    #[test]
    fn neutral_look_resolves_to_none() {
        assert!(resolve_color_grade(None, &ColorAdjustments::default()).is_none());
    }

    #[test]
    fn identity_when_no_filter_and_neutral_adjust() {
        let grade = effective_grade(None, &ColorAdjustments::default());
        assert_eq!(grade, ColorGrade::IDENTITY);
        assert!(grade.is_identity());
    }

    #[test]
    fn intensity_zero_with_neutral_adjust_yields_identity() {
        let filter = Filter {
            id: "mono".into(),
            intensity: 0.0.into(),
        };
        let grade = effective_grade(Some(&filter), &ColorAdjustments::default());
        assert_eq!(grade, ColorGrade::IDENTITY);
        assert!(resolve_color_grade(Some(&filter), &ColorAdjustments::default()).is_none());
    }

    #[test]
    fn catalog_ids_all_have_recipes_and_only_known_ids() {
        for spec in filter_catalog() {
            assert!(
                preset_recipe(spec.id).is_some(),
                "missing recipe for catalog id '{}'",
                spec.id
            );
        }
        for id in [
            "vivid", "warm", "cool", "mono", "fade", "chrome", "noir", "sunset", "forest", "berry",
        ] {
            assert!(preset_recipe(id).is_some());
        }
        assert!(preset_recipe("unknown").is_none());
        assert!(preset_recipe("").is_none());
    }

    #[test]
    fn clamps_out_of_range_and_non_finite_inputs() {
        let filter = Filter {
            id: "vivid".into(),
            intensity: f32::NAN.into(),
        };
        let adjust = ColorAdjustments {
            saturation: f32::INFINITY.into(),
            brightness: 2.0.into(),
            contrast: (-3.0).into(),
            exposure: f32::NAN.into(),
            temperature: 1.5.into(),
        };
        let grade = effective_grade(Some(&filter), &adjust);
        assert_eq!(grade.brightness, 1.0);
        assert_eq!(grade.contrast, -1.0);
        assert_eq!(grade.temperature, 1.0);
        assert_eq!(grade.saturation, 0.0);
        assert_eq!(grade.exposure, 0.0);

        let filter = Filter {
            id: "mono".into(),
            intensity: 1.0.into(),
        };
        let adjust = ColorAdjustments {
            saturation: (-1.5).into(),
            ..ColorAdjustments::default()
        };
        let grade = effective_grade(Some(&filter), &adjust);
        assert_eq!(grade.saturation, -1.0);
        assert_eq!(grade.contrast, 0.1);
    }

    #[test]
    fn preset_scaled_by_intensity_plus_adjustments() {
        let filter = Filter {
            id: "warm".into(),
            intensity: 0.5.into(),
        };
        let adjust = ColorAdjustments {
            brightness: 0.1.into(),
            ..ColorAdjustments::default()
        };
        let grade = effective_grade(Some(&filter), &adjust);
        assert!((grade.temperature - 0.2).abs() < 1e-6);
        assert!((grade.brightness - 0.125).abs() < 1e-6);
    }

    #[test]
    fn resolve_color_grade_returns_some_for_non_identity() {
        let grade = resolve_color_grade(
            Some(&Filter {
                id: "mono".into(),
                intensity: 0.5.into(),
            }),
            &ColorAdjustments::default(),
        )
        .unwrap();
        assert!((grade.saturation - (-0.5)).abs() < 1e-5);
    }
}
