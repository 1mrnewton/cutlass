//! A clip's colour grade: one HSL colour, one curve, or one wheel at a time.
//!
//! Each part is a material the clip lists among its extra materials (see
//! `cutlass_effects::grade`). A part set back to nothing loses its
//! material. A material another clip shares is left to that clip, and this
//! clip gets its own copy, named by the request.

use cutlass_effects::{
    COLOR_CURVES, COLOR_WHEELS, CurveChannel, HSL, Wheel, WheelSetting, points_json, tidy_points,
    wheel_json,
};
use cutlass_model::{
    Draft, MaterialHsl, MaterialId, MaterialRef, RawMaterial, SegmentId, TrackType,
};
use serde_json::Value;

use super::{EditError, ensure_unlocked};

/// Set HSL colour `color` (0 red to 7 magenta) on `segment_id`: `hue`,
/// `saturation` and `lightness` from -1 to 1. All zero takes it off.
pub fn set_hsl(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    color: u8,
    values: [f64; 3],
) -> Result<(), EditError> {
    if color > 7
        || values
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1.0)
    {
        return Err(EditError::EffectStrength);
    }
    let (track_index, segment_index) = colour_target(draft, segment_id)?;
    #[allow(clippy::cast_possible_truncation)]
    let [hue, saturation, lightness] = values.map(|value| (value * 100.0).round() as i32);
    let current = extra_where(
        draft,
        track_index,
        segment_index,
        |material| matches!(material, MaterialRef::Hsl(hsl) if hsl.hsl_color_type == i32::from(color)),
    );
    let off = hue == 0 && saturation == 0 && lightness == 0;
    if let Some(id) = current {
        if !shared(draft, &id) {
            if off {
                unlist(draft, track_index, segment_index, &id);
                draft.materials.hsl.retain(|hsl| hsl.id != id);
            } else if let Some(hsl) = draft.materials.hsl.iter_mut().find(|hsl| hsl.id == id) {
                hsl.hue = hue;
                hsl.saturation = saturation;
                hsl.lightness = lightness;
            }
            return Ok(());
        }
        unlist(draft, track_index, segment_index, &id);
    }
    if off {
        return Ok(());
    }
    fresh(draft, material_id)?;
    draft.materials.hsl.push(MaterialHsl {
        id: material_id.clone(),
        kind: HSL.to_owned(),
        hsl_color_type: i32::from(color),
        hue,
        saturation,
        lightness,
        visible: true,
        ..MaterialHsl::default()
    });
    list(draft, track_index, segment_index, material_id);
    Ok(())
}

/// Set curve `channel` (`rgb`, `red`, `green` or `blue`) on `segment_id`
/// to `points`, each `[x, y]` from 0 to 1. No points, or the diagonal,
/// is the straight line.
pub fn set_curve(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    channel: &str,
    points: &[[f64; 2]],
) -> Result<(), EditError> {
    let channel = CurveChannel::from_key(channel).ok_or(EditError::UnknownEffect)?;
    if points.len() > cutlass_effects::MAX_CURVE_POINTS
        || points
            .iter()
            .flatten()
            .any(|value| !(0.0..=1.0).contains(value))
    {
        return Err(EditError::EffectStrength);
    }
    #[allow(clippy::cast_possible_truncation)]
    let points: Vec<(f32, f32)> = points.iter().map(|[x, y]| (*x as f32, *y as f32)).collect();
    let mut probe = cutlass_effects::Grade::default();
    probe.curves[0] = tidy_points(&points);
    let value = if probe.is_identity() {
        Value::Array(Vec::new())
    } else {
        points_json(&points)
    };
    set_raw(
        draft,
        segment_id,
        material_id,
        COLOR_CURVES,
        channel.key(),
        value,
    )
}

/// Set colour wheel `wheel` (`lift`, `gamma` or `gain`) on `segment_id`:
/// the puck at `x`, `y` in the unit circle, the ring at `luma`.
pub fn set_wheel(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    wheel: &str,
    setting: [f64; 3],
) -> Result<(), EditError> {
    let wheel = Wheel::from_key(wheel).ok_or(EditError::UnknownEffect)?;
    let [x, y, luma] = setting;
    if setting.iter().any(|value| !value.is_finite())
        || x * x + y * y > 1.0 + 1e-6
        || luma.abs() > 1.0
    {
        return Err(EditError::EffectStrength);
    }
    #[allow(clippy::cast_possible_truncation)]
    let setting = WheelSetting {
        x: x as f32,
        y: y as f32,
        luma: luma as f32,
    };
    let value = if setting.held().is_zero() {
        Value::Array(Vec::new())
    } else {
        wheel_json(setting)
    };
    set_raw(
        draft,
        segment_id,
        material_id,
        COLOR_WHEELS,
        wheel.key(),
        value,
    )
}

/// Store `value` under `field` on the clip's `kind` record. A record with
/// every field empty is taken off.
fn set_raw(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    kind: &str,
    field: &str,
    value: Value,
) -> Result<(), EditError> {
    let (track_index, segment_index) = colour_target(draft, segment_id)?;
    let current = extra_where(
        draft,
        track_index,
        segment_index,
        |material| matches!(material, MaterialRef::Other(raw) if raw.kind == kind),
    );
    let mut fields = current
        .as_ref()
        .and_then(|id| pool(draft, kind).iter().find(|raw| &raw.id == id))
        .map(|raw| raw.fields.clone())
        .unwrap_or_default();
    fields.insert(field.to_owned(), value);
    let empty = fields
        .values()
        .all(|value| value.as_array().is_some_and(Vec::is_empty));
    if let Some(id) = current {
        if !shared(draft, &id) {
            if empty {
                unlist(draft, track_index, segment_index, &id);
                pool_mut(draft, kind).retain(|raw| raw.id != id);
            } else if let Some(raw) = pool_mut(draft, kind).iter_mut().find(|raw| raw.id == id) {
                raw.fields = fields;
            }
            return Ok(());
        }
        unlist(draft, track_index, segment_index, &id);
    }
    if empty {
        return Ok(());
    }
    fresh(draft, material_id)?;
    pool_mut(draft, kind).push(RawMaterial {
        id: material_id.clone(),
        kind: kind.to_owned(),
        fields,
    });
    list(draft, track_index, segment_index, material_id);
    Ok(())
}

fn pool<'a>(draft: &'a Draft, kind: &str) -> &'a [RawMaterial] {
    if kind == COLOR_CURVES {
        &draft.materials.color_curves
    } else {
        &draft.materials.primary_color_wheels
    }
}

fn pool_mut<'a>(draft: &'a mut Draft, kind: &str) -> &'a mut Vec<RawMaterial> {
    if kind == COLOR_CURVES {
        &mut draft.materials.color_curves
    } else {
        &mut draft.materials.primary_color_wheels
    }
}

/// The clip, when it takes a grade: a picture, an adjustment layer, or a
/// filter, on an unlocked row.
fn colour_target(draft: &Draft, segment_id: &SegmentId) -> Result<(usize, usize), EditError> {
    let (track_index, segment_index) = super::speed::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if matches!(track.kind, TrackType::Audio | TrackType::Effect) {
        return Err(EditError::EffectTarget);
    }
    Ok((track_index, segment_index))
}

fn extra_where(
    draft: &Draft,
    track_index: usize,
    segment_index: usize,
    test: impl Fn(MaterialRef<'_>) -> bool,
) -> Option<MaterialId> {
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|id| draft.materials.get(id.as_str()).is_some_and(&test))
        .cloned()
}

fn shared(draft: &Draft, id: &MaterialId) -> bool {
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .filter(|segment| segment.extra_material_refs.contains(id))
        .count()
        > 1
}

fn fresh(draft: &Draft, material_id: &MaterialId) -> Result<(), EditError> {
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    Ok(())
}

fn list(draft: &mut Draft, track_index: usize, segment_index: usize, id: &MaterialId) {
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(id.clone());
}

fn unlist(draft: &mut Draft, track_index: usize, segment_index: usize, id: &MaterialId) {
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .retain(|other| other != id);
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{CurveChannel, Wheel, segment_grade};
    use cutlass_model::{Draft, MaterialId, Segment, SegmentId, TimeRange};

    use super::{set_curve, set_hsl, set_wheel};
    use crate::edit::EditError;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        draft
    }

    #[test]
    fn hsl_colours_are_set_changed_and_taken_off() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_hsl(&mut draft, &a, &MaterialId::new("h1"), 2, [0.5, 0.0, -0.25]).unwrap();
        set_hsl(&mut draft, &a, &MaterialId::new("h2"), 5, [0.0, 1.0, 0.0]).unwrap();
        set_hsl(&mut draft, &a, &MaterialId::new("h3"), 2, [0.1, 0.0, 0.0]).unwrap();
        let grade = segment_grade(&draft, &draft.tracks[0].segments[0]);
        assert_eq!(grade.hsl[2], [0.1, 0.0, 0.0]);
        assert_eq!(grade.hsl[5], [0.0, 1.0, 0.0]);
        assert_eq!(draft.materials.hsl.len(), 2);
        set_hsl(&mut draft, &a, &MaterialId::new("h4"), 2, [0.0; 3]).unwrap();
        assert_eq!(draft.materials.hsl.len(), 1);
        assert!(matches!(
            set_hsl(&mut draft, &a, &MaterialId::new("x"), 8, [0.1, 0.0, 0.0]),
            Err(EditError::EffectStrength)
        ));
        assert!(matches!(
            set_hsl(&mut draft, &a, &MaterialId::new("x"), 1, [1.5, 0.0, 0.0]),
            Err(EditError::EffectStrength)
        ));
    }

    #[test]
    fn a_curve_is_stored_per_channel_and_a_straight_one_goes() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        let bend = [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]];
        set_curve(&mut draft, &a, &MaterialId::new("c"), "red", &bend).unwrap();
        set_curve(
            &mut draft,
            &a,
            &MaterialId::new("c2"),
            "rgb",
            &[[0.0, 0.1], [1.0, 0.9]],
        )
        .unwrap();
        let grade = segment_grade(&draft, &draft.tracks[0].segments[0]);
        assert_eq!(grade.curve(CurveChannel::Red).len(), 3);
        assert_eq!(grade.curve(CurveChannel::Rgb), &[(0.0, 0.1), (1.0, 0.9)]);
        assert_eq!(draft.materials.color_curves.len(), 1);
        set_curve(&mut draft, &a, &MaterialId::new("c3"), "red", &[]).unwrap();
        set_curve(
            &mut draft,
            &a,
            &MaterialId::new("c3"),
            "rgb",
            &[[0.0, 0.0], [1.0, 1.0]],
        )
        .unwrap();
        assert!(draft.materials.color_curves.is_empty());
        assert!(draft.tracks[0].segments[0].extra_material_refs.is_empty());
        assert!(matches!(
            set_curve(&mut draft, &a, &MaterialId::new("x"), "alpha", &bend),
            Err(EditError::UnknownEffect)
        ));
        assert!(matches!(
            set_curve(&mut draft, &a, &MaterialId::new("x"), "red", &[[0.0, 2.0]]),
            Err(EditError::EffectStrength)
        ));
    }

    #[test]
    fn a_wheel_is_stored_and_a_shared_one_is_copied() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_wheel(
            &mut draft,
            &a,
            &MaterialId::new("w"),
            "gain",
            [0.6, 0.0, 0.2],
        )
        .unwrap();
        let mut b = Segment::new("b", "m", TimeRange::new(1_000_000, 1_000_000));
        b.extra_material_refs.push(MaterialId::new("w"));
        draft.tracks[0].segments.push(b);
        set_wheel(
            &mut draft,
            &a,
            &MaterialId::new("w2"),
            "lift",
            [0.0, -0.5, 0.0],
        )
        .unwrap();
        let mine = segment_grade(&draft, &draft.tracks[0].segments[0]);
        let theirs = segment_grade(&draft, &draft.tracks[0].segments[1]);
        assert!((mine.wheel(Wheel::Lift).y + 0.5).abs() < 1e-6);
        assert!((mine.wheel(Wheel::Gain).x - 0.6).abs() < 1e-6);
        assert_eq!(theirs.wheel(Wheel::Lift).y, 0.0);
        assert!(matches!(
            set_wheel(
                &mut draft,
                &a,
                &MaterialId::new("x"),
                "gain",
                [1.0, 1.0, 0.0]
            ),
            Err(EditError::EffectStrength)
        ));
    }
}
