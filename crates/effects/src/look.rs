//! The bundled filters: twenty looks, each a set of adjust sliders.
//!
//! A filter clip sits on a filter row and runs over every picture below
//! it. Its material is an effect of type `filter` whose `effect_id` is
//! `look/<id>` and whose `value` is the strength, from 0 to 1, that scales
//! every slider of the look. Sliders the clip sets itself, such as its own
//! temperature, are added on top.

use cutlass_model::{Draft, MaterialRef, Segment};

use crate::adjust::{Adjust, AdjustKey, segment_adjust};

/// The `effect_id` prefix of a bundled filter.
pub const LOOK_PREFIX: &str = "look/";
/// The material `type` of a filter clip.
pub const FILTER: &str = "filter";

/// One bundled filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// The id after [`LOOK_PREFIX`], such as `warm`.
    pub id: &'static str,
    pub name: &'static str,
    /// The sliders at full strength.
    pub values: &'static [(AdjustKey, f32)],
}

impl Look {
    /// The `effect_id` stored on the filter material.
    #[must_use]
    pub fn effect_id(&self) -> String {
        format!("{LOOK_PREFIX}{}", self.id)
    }

    /// The sliders at `strength`, from 0 to 1.
    #[must_use]
    pub fn at(&self, strength: f32) -> Adjust {
        let strength = if strength.is_finite() {
            strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut adjust = Adjust::default();
        for (key, value) in self.values {
            adjust.set(*key, value * strength);
        }
        adjust
    }
}

const fn look(id: &'static str, name: &'static str, values: &'static [(AdjustKey, f32)]) -> Look {
    Look { id, name, values }
}

use AdjustKey as K;

/// The twenty filters, in shelf order.
pub const LOOKS: [Look; 20] = [
    look(
        "warm",
        "Warm",
        &[(K::Temperature, 0.35), (K::Saturation, 0.1)],
    ),
    look("cool", "Cool", &[(K::Temperature, -0.35), (K::Tint, 0.05)]),
    look(
        "vivid",
        "Vivid",
        &[(K::Saturation, 0.45), (K::Contrast, 0.15)],
    ),
    look("muted", "Muted", &[(K::Saturation, -0.4), (K::Fade, 0.2)]),
    look("mono", "Mono", &[(K::Saturation, -1.0), (K::Contrast, 0.1)]),
    look(
        "noir",
        "Noir",
        &[
            (K::Saturation, -1.0),
            (K::Contrast, 0.45),
            (K::Blacks, -0.2),
            (K::Vignette, 0.4),
        ],
    ),
    look(
        "sepia",
        "Sepia",
        &[(K::Saturation, -0.8), (K::Temperature, 0.6), (K::Fade, 0.1)],
    ),
    look(
        "faded",
        "Faded",
        &[(K::Fade, 0.5), (K::Contrast, -0.15), (K::Saturation, -0.15)],
    ),
    look(
        "film",
        "Film",
        &[
            (K::Fade, 0.25),
            (K::Grain, 0.3),
            (K::Contrast, 0.1),
            (K::Temperature, 0.1),
        ],
    ),
    look(
        "vintage",
        "Vintage",
        &[
            (K::Temperature, 0.3),
            (K::Fade, 0.35),
            (K::Saturation, -0.25),
            (K::Vignette, 0.3),
        ],
    ),
    look(
        "golden",
        "Golden",
        &[
            (K::Temperature, 0.45),
            (K::Highlights, 0.15),
            (K::Saturation, 0.15),
        ],
    ),
    look(
        "cinema",
        "Cinema",
        &[
            (K::Temperature, 0.15),
            (K::Contrast, 0.2),
            (K::Saturation, 0.2),
            (K::Shadows, -0.1),
            (K::Vignette, 0.2),
        ],
    ),
    look(
        "bright",
        "Bright",
        &[(K::Exposure, 0.2), (K::Shadows, 0.2), (K::Contrast, -0.05)],
    ),
    look(
        "moody",
        "Moody",
        &[
            (K::Exposure, -0.15),
            (K::Contrast, 0.25),
            (K::Saturation, -0.2),
            (K::Vignette, 0.35),
        ],
    ),
    look(
        "dream",
        "Dream",
        &[
            (K::Brilliance, 0.3),
            (K::Fade, 0.25),
            (K::Clarity, -0.4),
            (K::Saturation, 0.1),
        ],
    ),
    look(
        "crisp",
        "Crisp",
        &[(K::Clarity, 0.4), (K::Sharpen, 0.4), (K::Contrast, 0.1)],
    ),
    look(
        "pastel",
        "Pastel",
        &[
            (K::Fade, 0.3),
            (K::Saturation, -0.2),
            (K::Brightness, 0.1),
            (K::Contrast, -0.2),
        ],
    ),
    look(
        "night",
        "Night",
        &[
            (K::Temperature, -0.4),
            (K::Exposure, -0.2),
            (K::Saturation, -0.2),
            (K::Contrast, 0.15),
        ],
    ),
    look(
        "rose",
        "Rose",
        &[(K::Tint, 0.25), (K::Temperature, 0.1), (K::Fade, 0.1)],
    ),
    look(
        "fresh",
        "Fresh",
        &[(K::Tint, -0.2), (K::Saturation, 0.2), (K::Brightness, 0.05)],
    ),
];

/// The bundled filter an `effect_id` such as `look/warm` names.
#[must_use]
pub fn bundled_look(effect_id: &str) -> Option<&'static Look> {
    let id = effect_id.strip_prefix(LOOK_PREFIX)?;
    LOOKS.iter().find(|look| look.id == id)
}

/// What a filter clip does: its look at its strength, with the clip's own
/// sliders added. `None` when the clip is not a bundled filter.
#[must_use]
pub fn segment_look(draft: &Draft, segment: &Segment) -> Option<Adjust> {
    let Some(MaterialRef::Effect(effect)) = draft.materials.get(segment.material_id.as_str())
    else {
        return None;
    };
    let look = bundled_look(&effect.effect_id)?;
    #[allow(clippy::cast_possible_truncation)]
    let strength = effect.value.unwrap_or(1.0) as f32;
    let mut adjust = look.at(strength);
    let own = segment_adjust(draft, segment);
    for key in AdjustKey::ALL {
        adjust.set(key, adjust.get(key) + own.get(key));
    }
    Some(adjust)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialEffect, MaterialId, Segment, TimeRange};

    use super::{FILTER, LOOKS, bundled_look, segment_look};
    use crate::adjust::AdjustKey;

    #[test]
    fn twenty_named_looks_each_change_something() {
        assert_eq!(LOOKS.len(), 20);
        for (index, look) in LOOKS.iter().enumerate() {
            assert!(!look.at(1.0).is_identity(), "{}", look.id);
            assert!(look.at(0.0).is_identity(), "{}", look.id);
            assert_eq!(bundled_look(&look.effect_id()), Some(look));
            assert!(
                LOOKS[index + 1..].iter().all(|other| other.id != look.id),
                "{}",
                look.id
            );
        }
        assert!(bundled_look("warm").is_none());
        assert!(bundled_look("look/none").is_none());
    }

    #[test]
    fn strength_scales_the_look_and_the_clip_adds_its_own() {
        let mut draft = Draft::empty("D");
        draft.materials.effects.push(MaterialEffect {
            kind: FILTER.to_owned(),
            effect_id: "look/warm".to_owned(),
            value: Some(0.5),
            ..MaterialEffect::new(MaterialId::new("f"))
        });
        draft.materials.effects.push(MaterialEffect {
            kind: "temperature".to_owned(),
            value: Some(-0.1),
            ..MaterialEffect::new(MaterialId::new("t"))
        });
        let mut segment = Segment::new("s", "f", TimeRange::new(0, 1));
        let adjust = segment_look(&draft, &segment).unwrap();
        assert!((adjust.get(AdjustKey::Temperature) - 0.175).abs() < 1e-6);
        segment.extra_material_refs.push(MaterialId::new("t"));
        let adjust = segment_look(&draft, &segment).unwrap();
        assert!((adjust.get(AdjustKey::Temperature) - 0.075).abs() < 1e-6);
        assert!((adjust.get(AdjustKey::Saturation) - 0.05).abs() < 1e-6);
        let other = Segment::new("o", "missing", TimeRange::new(0, 1));
        assert!(segment_look(&draft, &other).is_none());
    }
}
