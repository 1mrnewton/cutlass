//! Resolve persisted clip look fields into compositor [`ColorGrade`] values.

use cutlass_compositor::ColorGrade;
use cutlass_models::{ColorAdjustments, Filter};

/// Combine a clip's manual adjustments and optional filter preset into a
/// resolved grade. Returns `None` when the result is identity (no GPU work).
pub fn resolve_color_grade(
    filter: Option<&Filter>,
    adjust: &ColorAdjustments,
) -> Option<ColorGrade> {
    let mut grade = ColorGrade {
        brightness: adjust.brightness,
        contrast: adjust.contrast,
        saturation: adjust.saturation,
        exposure: adjust.exposure,
        temperature: adjust.temperature,
    };
    if let Some(f) = filter {
        let preset = filter_preset(&f.id);
        let t = f.intensity;
        grade.brightness += preset.brightness * t;
        grade.contrast += preset.contrast * t;
        grade.saturation += preset.saturation * t;
        grade.exposure += preset.exposure * t;
        grade.temperature += preset.temperature * t;
    }
    clamp_grade(&mut grade);
    if grade.is_identity() {
        None
    } else {
        Some(grade)
    }
}

fn clamp_grade(grade: &mut ColorGrade) {
    let c = |v: f32| v.clamp(-1.0, 1.0);
    grade.brightness = c(grade.brightness);
    grade.contrast = c(grade.contrast);
    grade.saturation = c(grade.saturation);
    grade.exposure = c(grade.exposure);
    grade.temperature = c(grade.temperature);
}

/// Catalog filter presets as predefined grade strengths (CapCut filters).
fn filter_preset(id: &str) -> ColorGrade {
    match id {
        "vivid" => ColorGrade {
            saturation: 0.35,
            contrast: 0.2,
            exposure: 0.05,
            ..ColorGrade::default()
        },
        "warm" => ColorGrade {
            temperature: 0.4,
            saturation: 0.1,
            ..ColorGrade::default()
        },
        "cool" => ColorGrade {
            temperature: -0.4,
            saturation: 0.1,
            ..ColorGrade::default()
        },
        "mono" => ColorGrade {
            saturation: -1.0,
            ..ColorGrade::default()
        },
        "fade" => ColorGrade {
            contrast: -0.25,
            brightness: -0.1,
            saturation: -0.35,
            ..ColorGrade::default()
        },
        "chrome" => ColorGrade {
            contrast: 0.25,
            saturation: -0.15,
            brightness: 0.1,
            ..ColorGrade::default()
        },
        "noir" => ColorGrade {
            saturation: -0.85,
            contrast: 0.45,
            brightness: -0.15,
            ..ColorGrade::default()
        },
        "sunset" => ColorGrade {
            temperature: 0.55,
            saturation: 0.25,
            exposure: 0.1,
            ..ColorGrade::default()
        },
        "forest" => ColorGrade {
            temperature: -0.15,
            saturation: 0.2,
            brightness: -0.08,
            ..ColorGrade::default()
        },
        "berry" => ColorGrade {
            saturation: 0.3,
            contrast: 0.12,
            temperature: 0.2,
            ..ColorGrade::default()
        },
        _ => ColorGrade::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::Filter;

    #[test]
    fn neutral_look_resolves_to_none() {
        assert!(resolve_color_grade(None, &ColorAdjustments::default()).is_none());
    }

    #[test]
    fn manual_adjustments_resolve() {
        let grade = resolve_color_grade(
            None,
            &ColorAdjustments {
                exposure: 0.5,
                ..ColorAdjustments::default()
            },
        )
        .unwrap();
        assert_eq!(grade.exposure, 0.5);
        assert!(!grade.is_identity());
    }

    #[test]
    fn filter_preset_blends_at_intensity() {
        let grade = resolve_color_grade(
            Some(&Filter {
                id: "mono".into(),
                intensity: 0.5,
            }),
            &ColorAdjustments::default(),
        )
        .unwrap();
        assert!((grade.saturation - (-0.5)).abs() < 1e-5);
    }
}
