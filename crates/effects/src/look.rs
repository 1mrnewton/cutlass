//! The bundled filters: looks read from `filters/*.json`, each a set of
//! adjust sliders with curves, colour wheels and HSL on top.
//!
//! A filter clip sits on a filter row and runs over every picture below
//! it. Its material is an effect of type `filter` whose `effect_id` is
//! `look/<id>` and whose `value` is the strength, from 0 to 1, that scales
//! the whole look: every slider, how far each curve bends from the
//! straight line, each wheel and each HSL colour. Sliders the clip sets
//! itself, such as its own temperature, are added on top, and its own
//! curves, wheels and HSL run after the look's.
//!
//! A filter file:
//!
//! ```json
//! {
//!   "version": 1,
//!   "id": "teal-orange",
//!   "name": "Teal & Orange",
//!   "category": "Cinema",
//!   "adjust": {"contrast": 0.15},
//!   "curves": {"rgb": [[0, 0.03], [0.25, 0.2], [0.75, 0.8], [1, 0.97]]},
//!   "wheels": {"lift": {"hue": 190, "amount": 0.4}, "gain": {"hue": 30, "amount": 0.3}},
//!   "hsl": {"orange": [0, 0.2, 0.1]}
//! }
//! ```
//!
//! Adjust names are the slider names in lower case, `temperature` for
//! Temp. Curves are `rgb`, `red`, `green` and `blue`, points from 0 to 1.
//! Wheels are `lift`, `gamma` and `gain`, each a hue in degrees (0 red,
//! 120 green, 240 blue), how far toward it from 0 to 1, and `luma` from
//! -1 to 1. HSL is `[hue, saturation, lightness]` from -1 to 1 for each of
//! `red`, `orange`, `yellow`, `green`, `cyan`, `blue`, `purple` and
//! `magenta`.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use cutlass_model::{Draft, MaterialRef, Segment};
use serde::Deserialize;

use crate::adjust::{Adjust, AdjustKey};
use crate::grade::{Colour, Grade, HSL_COLORS, WheelSetting, segment_colour, tidy_points};

/// The `effect_id` prefix of a bundled filter.
pub const LOOK_PREFIX: &str = "look/";
/// The material `type` of a filter clip.
pub const FILTER: &str = "filter";

/// The filters shelf's categories, in shelf order.
pub const FILTER_CATEGORIES: [&str; 10] = [
    "Portrait",
    "Landscape",
    "Food",
    "Film",
    "Vintage",
    "Mono",
    "Cinema",
    "Moody",
    "Bright",
    "Creative",
];

/// One bundled filter.
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    /// The id after [`LOOK_PREFIX`], such as `warm`.
    pub id: String,
    pub name: String,
    /// One of [`FILTER_CATEGORIES`].
    pub category: String,
    /// The preview scene its card shows, by name.
    pub scene: String,
    /// The sliders at full strength.
    pub values: Vec<(AdjustKey, f32)>,
    /// The curves, wheels and HSL at full strength.
    pub grade: Grade,
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
        let strength = held(strength);
        let mut adjust = Adjust::default();
        for (key, value) in &self.values {
            adjust.set(*key, value * strength);
        }
        adjust
    }

    /// The curves, wheels and HSL at `strength`, from 0 to 1: each curve
    /// that part of the way from the straight line.
    #[must_use]
    pub fn grade_at(&self, strength: f32) -> Grade {
        let strength = held(strength);
        let mut grade = self.grade.clone();
        for points in &mut grade.curves {
            for (x, y) in points.iter_mut() {
                *y = *x + (*y - *x) * strength;
            }
        }
        for wheel in &mut grade.wheels {
            *wheel = WheelSetting {
                x: wheel.x * strength,
                y: wheel.y * strength,
                luma: wheel.luma * strength,
            };
        }
        for colour in &mut grade.hsl {
            *colour = colour.map(|value| value * strength);
        }
        grade
    }
}

fn held(strength: f32) -> f32 {
    if strength.is_finite() {
        strength.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// A wheel as a filter file writes it.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct WheelFile {
    #[serde(default)]
    hue: f32,
    #[serde(default)]
    amount: f32,
    #[serde(default)]
    luma: f32,
}

/// A filter file.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LookFile {
    version: u32,
    id: String,
    name: String,
    category: String,
    #[serde(default)]
    preview: Option<String>,
    #[serde(default)]
    adjust: BTreeMap<String, f32>,
    #[serde(default)]
    curves: BTreeMap<String, Vec<[f32; 2]>>,
    #[serde(default)]
    wheels: BTreeMap<String, WheelFile>,
    #[serde(default)]
    hsl: BTreeMap<String, [f32; 3]>,
}

/// A slider's name in a filter file.
fn adjust_name(key: AdjustKey) -> String {
    match key {
        AdjustKey::Temperature => "temperature".to_owned(),
        other => other.label().to_lowercase(),
    }
}

/// The scene a category's cards show when a filter names none.
fn category_scene(category: &str) -> &'static str {
    match category {
        "Landscape" => "lake",
        "Food" => "cafe",
        "Film" => "beach",
        "Vintage" => "road",
        "Mono" => "city",
        "Cinema" => "city",
        "Moody" => "road",
        "Bright" => "flowers",
        "Creative" => "skater",
        _ => "portrait",
    }
}

/// Read one filter file, or say what is wrong with it.
///
/// # Errors
///
/// A message naming the first thing that is not a filter.
pub fn read_look(text: &str) -> Result<Look, String> {
    let file: LookFile = serde_json::from_str(text).map_err(|error| error.to_string())?;
    if file.version != 1 {
        return Err(format!("version {}", file.version));
    }
    if file.id.is_empty()
        || !file
            .id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(format!("id {:?}", file.id));
    }
    if !FILTER_CATEGORIES.contains(&file.category.as_str()) {
        return Err(format!("category {:?}", file.category));
    }
    let mut values = Vec::new();
    for (name, value) in &file.adjust {
        let key = AdjustKey::ALL
            .into_iter()
            .find(|key| adjust_name(*key) == *name)
            .ok_or_else(|| format!("adjust {name:?}"))?;
        if !(key.min()..=1.0).contains(value) {
            return Err(format!("{name} {value}"));
        }
        values.push((key, *value));
    }
    let mut grade = Grade::default();
    for (name, points) in &file.curves {
        let channel = crate::grade::CurveChannel::ALL
            .into_iter()
            .find(|channel| channel.key() == name)
            .ok_or_else(|| format!("curve {name:?}"))?;
        if points.len() < 2
            || points
                .iter()
                .flatten()
                .any(|value| !(0.0..=1.0).contains(value))
        {
            return Err(format!("curve {name}"));
        }
        let pairs: Vec<(f32, f32)> = points.iter().map(|[x, y]| (*x, *y)).collect();
        grade.curves[channel_index(channel)] = tidy_points(&pairs);
    }
    for (name, wheel) in &file.wheels {
        let index = crate::grade::Wheel::ALL
            .iter()
            .position(|known| known.key() == name)
            .ok_or_else(|| format!("wheel {name:?}"))?;
        if !(0.0..=1.0).contains(&wheel.amount) || !(-1.0..=1.0).contains(&wheel.luma) {
            return Err(format!("wheel {name}"));
        }
        let turn = wheel.hue.to_radians();
        grade.wheels[index] = WheelSetting {
            x: turn.cos() * wheel.amount,
            y: turn.sin() * wheel.amount,
            luma: wheel.luma,
        };
    }
    for (name, setting) in &file.hsl {
        let index = HSL_COLORS
            .iter()
            .position(|colour| colour.to_lowercase() == *name)
            .ok_or_else(|| format!("hsl {name:?}"))?;
        if setting.iter().any(|value| !(-1.0..=1.0).contains(value)) {
            return Err(format!("hsl {name}"));
        }
        grade.hsl[index] = *setting;
    }
    let scene = file
        .preview
        .unwrap_or_else(|| category_scene(&file.category).to_owned());
    if crate::preview::preview_scenes()
        .iter()
        .all(|known| *known != scene)
    {
        return Err(format!("preview {scene:?}"));
    }
    let look = Look {
        id: file.id,
        name: file.name,
        category: file.category,
        scene,
        values,
        grade,
    };
    if look.at(1.0).is_identity() && look.grade.is_identity() {
        return Err("it changes nothing".to_owned());
    }
    Ok(look)
}

fn channel_index(channel: crate::grade::CurveChannel) -> usize {
    crate::grade::CurveChannel::ALL
        .iter()
        .position(|known| *known == channel)
        .unwrap_or(0)
}

/// The filter files, in shelf order: by category, then as listed.
const BUNDLED_FILES: [&str; 20] = [
    include_str!("../filters/golden.json"),
    include_str!("../filters/rose.json"),
    include_str!("../filters/vivid.json"),
    include_str!("../filters/crisp.json"),
    include_str!("../filters/fresh.json"),
    include_str!("../filters/film.json"),
    include_str!("../filters/sepia.json"),
    include_str!("../filters/faded.json"),
    include_str!("../filters/vintage.json"),
    include_str!("../filters/mono.json"),
    include_str!("../filters/noir.json"),
    include_str!("../filters/cinema.json"),
    include_str!("../filters/night.json"),
    include_str!("../filters/cool.json"),
    include_str!("../filters/muted.json"),
    include_str!("../filters/moody.json"),
    include_str!("../filters/warm.json"),
    include_str!("../filters/bright.json"),
    include_str!("../filters/pastel.json"),
    include_str!("../filters/dream.json"),
];

/// Every bundled filter, in shelf order. A file that does not read is left
/// out; a test reads them all.
#[must_use]
pub fn looks() -> &'static [Look] {
    static LOOKS: OnceLock<Vec<Look>> = OnceLock::new();
    LOOKS.get_or_init(|| {
        BUNDLED_FILES
            .iter()
            .filter_map(|text| read_look(text).ok())
            .collect()
    })
}

/// The bundled filter an `effect_id` such as `look/warm` names.
#[must_use]
pub fn bundled_look(effect_id: &str) -> Option<&'static Look> {
    let id = effect_id.strip_prefix(LOOK_PREFIX)?;
    looks().iter().find(|look| look.id == id)
}

/// The poster for `look`: its scene with the look over it, as opaque RGBA,
/// shrunk to `width` × `height` (cropped to that shape from the middle).
#[must_use]
pub fn look_poster(look: &Look, width: u32, height: u32) -> Vec<u8> {
    let mut rgba = crate::preview::preview_scene(&look.scene)
        .map(|scene| shrink(scene, width, height))
        .unwrap_or_else(|| vec![128; width as usize * height as usize * 4]);
    Colour {
        look: look.grade_at(1.0),
        adjust: look.at(1.0),
        ..Colour::default()
    }
    .apply(&mut rgba, width, height, 7);
    rgba
}

/// `scene` cropped to the shape of `width` × `height` and averaged down to
/// it.
fn shrink(scene: &crate::preview::PreviewScene, width: u32, height: u32) -> Vec<u8> {
    let (sw, sh) = (f64::from(scene.width), f64::from(scene.height));
    let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let scale = (sw / w).min(sh / h);
    let (left, top) = ((sw - w * scale) / 2.0, (sh - h * scale) / 2.0);
    let mut out = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let span = |start: f64, at: u32, limit: u32| {
                let from = (start + f64::from(at) * scale).floor() as u32;
                let to = ((start + f64::from(at + 1) * scale).ceil() as u32).clamp(from + 1, limit);
                from.min(limit - 1)..to
            };
            let (xs, ys) = (span(left, x, scene.width), span(top, y, scene.height));
            let mut sum = [0u32; 4];
            let mut count = 0;
            for sy in ys {
                for sx in xs.clone() {
                    let at = ((sy * scene.width + sx) * 4) as usize;
                    for (total, byte) in sum.iter_mut().zip(&scene.rgba[at..at + 4]) {
                        *total += u32::from(*byte);
                    }
                    count += 1;
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            out.extend(sum.map(|total| (total / count.max(1)) as u8));
        }
    }
    out
}

/// What a filter clip does: its look at its strength, with the clip's own
/// colour work after it and its own sliders added. `None` when the clip is
/// not a bundled filter.
#[must_use]
pub fn segment_look(draft: &Draft, segment: &Segment) -> Option<Colour> {
    let Some(MaterialRef::Effect(effect)) = draft.materials.get(segment.material_id.as_str())
    else {
        return None;
    };
    let look = bundled_look(&effect.effect_id)?;
    #[allow(clippy::cast_possible_truncation)]
    let strength = effect.value.unwrap_or(1.0) as f32;
    let mut colour = segment_colour(draft, segment);
    let mut adjust = look.at(strength);
    for key in AdjustKey::ALL {
        adjust.set(key, adjust.get(key) + colour.adjust.get(key));
    }
    colour.adjust = adjust;
    colour.look = look.grade_at(strength);
    Some(colour)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialEffect, MaterialId, Segment, TimeRange};

    use super::{BUNDLED_FILES, FILTER, bundled_look, look_poster, looks, read_look, segment_look};
    use crate::adjust::AdjustKey;

    #[test]
    fn every_bundled_filter_reads_and_changes_something() {
        for text in BUNDLED_FILES {
            if let Err(error) = read_look(text) {
                panic!("{error} in {text}");
            }
        }
        assert_eq!(looks().len(), BUNDLED_FILES.len());
        for (index, look) in looks().iter().enumerate() {
            assert!(look.at(0.0).is_identity(), "{}", look.id);
            assert!(look.grade_at(0.0).is_identity(), "{}", look.id);
            assert_eq!(bundled_look(&look.effect_id()), Some(look));
            assert!(
                looks()[index + 1..].iter().all(|other| other.id != look.id),
                "{}",
                look.id
            );
        }
        // In shelf order: each category's filters together.
        let mut seen: Vec<&str> = Vec::new();
        for look in looks() {
            if seen.last() != Some(&look.category.as_str()) {
                assert!(!seen.contains(&look.category.as_str()), "{}", look.id);
                seen.push(&look.category);
            }
        }
        assert!(bundled_look("warm").is_none());
        assert!(bundled_look("look/none").is_none());
    }

    #[test]
    fn a_broken_filter_is_refused() {
        let good = r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "adjust": {"saturation": -1}}"#;
        assert!(read_look(good).is_ok());
        for broken in [
            r#"{"version": 2, "id": "a", "name": "A", "category": "Mono", "adjust": {"saturation": -1}}"#,
            r#"{"version": 1, "id": "A", "name": "A", "category": "Mono", "adjust": {"saturation": -1}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Nope", "adjust": {"saturation": -1}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "adjust": {"loud": 1}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "adjust": {"fade": -0.5}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "curves": {"rgb": [[0, 2], [1, 1]]}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "wheels": {"lift": {"amount": 2}}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "hsl": {"pink": [0, 1, 0]}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono", "preview": "moon", "adjust": {"saturation": -1}}"#,
            r#"{"version": 1, "id": "a", "name": "A", "category": "Mono"}"#,
        ] {
            assert!(read_look(broken).is_err(), "{broken}");
        }
    }

    #[test]
    fn strength_bends_a_curve_part_of_the_way() {
        let look = read_look(
            r#"{"version": 1, "id": "a", "name": "A", "category": "Film", "curves": {"rgb": [[0, 0.2], [1, 0.8]]}, "wheels": {"gain": {"hue": 30, "amount": 0.5}}, "hsl": {"blue": [0, -1, 0]}}"#,
        )
        .unwrap();
        let half = look.grade_at(0.5);
        assert_eq!(half.curves[0], vec![(0.0, 0.1), (1.0, 0.9)]);
        let full = look.grade_at(1.0);
        assert!((half.wheels[2].x - full.wheels[2].x / 2.0).abs() < 1e-6);
        assert!((half.hsl[5][1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn each_poster_is_its_scene_under_its_own_look() {
        let posters: Vec<Vec<u8>> = looks()
            .iter()
            .map(|look| look_poster(look, 24, 16))
            .collect();
        for (index, poster) in posters.iter().enumerate() {
            assert_eq!(poster.len(), 24 * 16 * 4);
            assert!(poster.chunks(4).all(|pixel| pixel[3] == 255));
            assert!(
                posters[index + 1..].iter().all(|other| other != poster),
                "{}",
                looks()[index].id
            );
        }
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
        let colour = segment_look(&draft, &segment).unwrap();
        assert!((colour.adjust.get(AdjustKey::Temperature) - 0.175).abs() < 1e-6);
        segment.extra_material_refs.push(MaterialId::new("t"));
        let colour = segment_look(&draft, &segment).unwrap();
        assert!((colour.adjust.get(AdjustKey::Temperature) - 0.075).abs() < 1e-6);
        assert!((colour.adjust.get(AdjustKey::Saturation) - 0.05).abs() < 1e-6);
        let other = Segment::new("o", "missing", TimeRange::new(0, 1));
        assert!(segment_look(&draft, &other).is_none());
    }
}
