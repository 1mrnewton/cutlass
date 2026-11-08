//! Selections: an object cut out of a clip by its pixels, as click-to-select
//! returns it.
//!
//! A clip's selection is a `selection` record in the draft's `common_mask`
//! pool that the clip lists among its extra materials, beside any shape
//! mask. It holds the mask at the size of the frame it was made from, run
//! length coded row by row: counts of pixels that alternate between not
//! selected and selected, starting with not selected (so the first count
//! may be 0). `frame` is the source time of that frame. The selection
//! covers the whole clip in the clip's own frame, so it moves, turns and
//! scales with the clip. Following the object from frame to frame is a
//! later record.

use cutlass_model::{Draft, MaterialRef, Micros, Segment};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// The material `type` of a selection.
pub const SELECTION: &str = "selection";

/// The longest side a selection may have. An 8K frame is 7680.
pub const MAX_SELECTION_SIDE: u32 = 16_384;

/// One selection: which pixels of one frame are the object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub width: u32,
    pub height: u32,
    /// Alternating counts of unselected and selected pixels, row by row.
    pub runs: Vec<u32>,
    /// The source time of the frame the selection was made from.
    pub frame: Micros,
    /// Keep everything but the object.
    pub invert: bool,
}

impl Selection {
    /// Codes `alpha`, one byte per pixel with nonzero selected. `None` when
    /// a side is zero or too long, or `alpha` is not `width * height` bytes.
    #[must_use]
    pub fn from_alpha(width: u32, height: u32, alpha: &[u8], frame: Micros) -> Option<Self> {
        if !sides_ok(width, height) || pixels(width, height) != Some(alpha.len()) {
            return None;
        }
        let mut runs = Vec::new();
        let mut selected = false;
        let mut count = 0_u32;
        for &a in alpha {
            if (a != 0) == selected {
                count += 1;
            } else {
                runs.push(count);
                selected = !selected;
                count = 1;
            }
        }
        runs.push(count);
        Some(Self {
            width,
            height,
            runs,
            frame,
            invert: false,
        })
    }

    /// Sides in range, and the counts add up to exactly every pixel.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let Some(total) = pixels(self.width, self.height) else {
            return false;
        };
        if !sides_ok(self.width, self.height) || self.runs.len() > total + 1 {
            return false;
        }
        let mut sum = 0_usize;
        for (index, run) in self.runs.iter().enumerate() {
            // Only the first count may be empty: a later empty run is padding.
            if *run == 0 && index > 0 {
                return false;
            }
            match sum.checked_add(*run as usize) {
                Some(next) if next <= total => sum = next,
                _ => return false,
            }
        }
        sum == total
    }

    /// One byte per pixel, 255 selected and 0 not, before `invert`. `None`
    /// for a selection that is not [`Self::is_valid`].
    #[must_use]
    pub fn alpha(&self) -> Option<Vec<u8>> {
        if !self.is_valid() {
            return None;
        }
        let mut alpha = Vec::with_capacity(pixels(self.width, self.height)?);
        let mut value = 0_u8;
        for run in &self.runs {
            alpha.resize(alpha.len() + *run as usize, value);
            value = !value;
        }
        Some(alpha)
    }

    /// How many pixels the selection keeps, before `invert`.
    #[must_use]
    pub fn selected(&self) -> u64 {
        self.runs
            .iter()
            .skip(1)
            .step_by(2)
            .map(|run| u64::from(*run))
            .sum()
    }

    /// The selection as the record's fields.
    #[must_use]
    pub fn to_fields(&self) -> Map<String, Value> {
        let mut fields = Map::new();
        fields.insert("width".to_owned(), json!(self.width));
        fields.insert("height".to_owned(), json!(self.height));
        fields.insert("runs".to_owned(), json!(self.runs));
        fields.insert("frame".to_owned(), json!(self.frame.0));
        fields.insert("invert".to_owned(), json!(self.invert));
        fields
    }

    /// A selection read from a record's fields, when every field is whole
    /// and the counts cover the frame exactly.
    #[must_use]
    pub fn from_fields(fields: &Map<String, Value>) -> Option<Self> {
        let side = |name: &str| u32::try_from(fields.get(name)?.as_u64()?).ok();
        let (width, height) = (side("width")?, side("height")?);
        let total = pixels(width, height)?;
        let listed = fields.get("runs")?.as_array()?;
        if !sides_ok(width, height) || listed.len() > total + 1 {
            return None;
        }
        let runs = listed
            .iter()
            .map(|run| u32::try_from(run.as_u64()?).ok())
            .collect::<Option<Vec<u32>>>()?;
        let selection = Self {
            width,
            height,
            runs,
            frame: Micros(
                fields
                    .get("frame")
                    .and_then(Value::as_i64)
                    .unwrap_or_default(),
            ),
            invert: fields
                .get("invert")
                .and_then(Value::as_bool)
                .unwrap_or_default(),
        };
        selection.is_valid().then_some(selection)
    }

    /// Multiply premultiplied RGBA by the selection. `to_clip` turns a
    /// canvas pixel centre into the clip point it shows (-1 to 1 across the
    /// clip, y up). The edge is sampled bilinearly, so it is not stepped.
    pub fn apply(
        &self,
        rgba: &mut [u8],
        width: u32,
        to_clip: impl Fn(f32, f32) -> Option<(f32, f32)>,
    ) {
        let canvas_width = width as usize;
        let Some(alpha) = self.alpha() else {
            return;
        };
        if canvas_width == 0 {
            return;
        }
        let lookup = Lookup {
            alpha: &alpha,
            width: self.width as usize,
            height: self.height as usize,
        };
        for (index, pixel) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if pixel[3] == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let (x, y) = (
                (index % canvas_width) as f32 + 0.5,
                (index / canvas_width) as f32 + 0.5,
            );
            let inside = to_clip(x, y).map_or(0.0, |(u, v)| lookup.coverage(u, v));
            let keep = if self.invert { 1.0 - inside } else { inside };
            if keep >= 1.0 {
                continue;
            }
            for channel in pixel.iter_mut() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let value = (f32::from(*channel) * keep.clamp(0.0, 1.0)).round() as u8;
                *channel = value;
            }
        }
    }
}

/// A decoded selection, read at clip points.
struct Lookup<'a> {
    alpha: &'a [u8],
    width: usize,
    height: usize,
}

impl Lookup<'_> {
    /// From 0 outside the object to 1 inside, at clip point `u`, `v`. A
    /// point off the clip is outside.
    fn coverage(&self, u: f32, v: f32) -> f32 {
        if !(-1.0..=1.0).contains(&u) || !(-1.0..=1.0).contains(&v) {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let (w, h) = (self.width as f32, self.height as f32);
        // Pixel centres sit at half steps.
        let x = ((u + 1.0) * 0.5 * w - 0.5).clamp(0.0, w - 1.0);
        let y = ((1.0 - v) * 0.5 * h - 0.5).clamp(0.0, h - 1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        #[allow(clippy::cast_precision_loss)]
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |px: usize, py: usize| {
            self.alpha
                .get(py * self.width + px)
                .map_or(0.0, |a| f32::from(*a) / 255.0)
        };
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
        top + (bottom - top) * fy
    }
}

fn sides_ok(width: u32, height: u32) -> bool {
    (1..=MAX_SELECTION_SIDE).contains(&width) && (1..=MAX_SELECTION_SIDE).contains(&height)
}

fn pixels(width: u32, height: u32) -> Option<usize> {
    (width as usize).checked_mul(height as usize)
}

/// Whether `segment` lists a selection, without decoding it.
#[must_use]
pub fn segment_has_selection(draft: &Draft, segment: &Segment) -> bool {
    segment.extra_material_refs.iter().any(|id| {
        matches!(
            draft.materials.get(id.as_str()),
            Some(MaterialRef::Other(raw)) if raw.kind == SELECTION
        )
    })
}

/// The selection `segment` lists, when it has a whole one.
#[must_use]
pub fn segment_selection(draft: &Draft, segment: &Segment) -> Option<Selection> {
    segment.extra_material_refs.iter().find_map(|id| {
        let Some(MaterialRef::Other(raw)) = draft.materials.get(id.as_str()) else {
            return None;
        };
        (raw.kind == SELECTION).then(|| Selection::from_fields(&raw.fields))?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 by 2 frame whose middle two columns are the object.
    fn middle() -> Selection {
        let alpha = [0, 255, 255, 0, 0, 255, 255, 0];
        Selection::from_alpha(4, 2, &alpha, Micros(40_000)).unwrap()
    }

    #[test]
    fn runs_alternate_from_unselected_and_cover_every_pixel() {
        let selection = middle();
        assert_eq!(selection.runs, vec![1, 2, 2, 2, 1]);
        assert!(selection.is_valid());
        assert_eq!(selection.selected(), 4);
        assert_eq!(
            selection.alpha().unwrap(),
            vec![0, 255, 255, 0, 0, 255, 255, 0]
        );
    }

    #[test]
    fn a_frame_that_starts_selected_starts_with_an_empty_run() {
        let selection = Selection::from_alpha(2, 1, &[9, 0], Micros(0)).unwrap();
        assert_eq!(selection.runs, vec![0, 1, 1]);
        assert_eq!(selection.alpha().unwrap(), vec![255, 0]);
        let all = Selection::from_alpha(2, 2, &[1; 4], Micros(0)).unwrap();
        assert_eq!(all.runs, vec![0, 4]);
        let none = Selection::from_alpha(2, 2, &[0; 4], Micros(0)).unwrap();
        assert_eq!((none.runs.clone(), none.selected()), (vec![4], 0));
    }

    #[test]
    fn a_bad_size_is_refused() {
        assert_eq!(Selection::from_alpha(0, 1, &[], Micros(0)), None);
        assert_eq!(Selection::from_alpha(2, 2, &[0; 3], Micros(0)), None);
        assert_eq!(
            Selection::from_alpha(MAX_SELECTION_SIDE + 1, 1, &[0; 16_385], Micros(0)),
            None
        );
    }

    #[test]
    fn the_fields_round_trip() {
        let selection = Selection {
            invert: true,
            ..middle()
        };
        assert_eq!(
            Selection::from_fields(&selection.to_fields()),
            Some(selection)
        );
    }

    #[test]
    fn hostile_fields_are_refused() {
        let good = middle().to_fields();
        let with = |name: &str, value: Value| {
            let mut fields = good.clone();
            fields.insert(name.to_owned(), value);
            Selection::from_fields(&fields)
        };
        // Counts that fall short of, or run past, the frame.
        assert_eq!(with("runs", json!([1, 2, 2, 2])), None);
        assert_eq!(with("runs", json!([1, 2, 2, 2, 2])), None);
        // A count that overflows when added up.
        assert_eq!(with("runs", json!([u32::MAX, u32::MAX, 2])), None);
        // A negative, fractional or too-large count.
        assert_eq!(with("runs", json!([-1, 9])), None);
        assert_eq!(with("runs", json!([1.5, 6.5])), None);
        assert_eq!(with("runs", json!([u64::from(u32::MAX) + 1])), None);
        // An empty run after the first.
        assert_eq!(with("runs", json!([1, 0, 7])), None);
        // More counts than pixels.
        assert_eq!(with("runs", json!(vec![1; 10])), None);
        // Sides that are zero, too long, or not numbers.
        assert_eq!(with("width", json!(0)), None);
        assert_eq!(with("width", json!(MAX_SELECTION_SIDE + 1)), None);
        assert_eq!(with("height", json!("2")), None);
        assert_eq!(with("runs", json!("1,2")), None);
    }

    #[test]
    fn applying_keeps_the_object_and_clears_the_rest() {
        // The canvas shows the clip edge to edge: canvas pixels map to clip
        // points one to one across 4 by 2.
        let selection = middle();
        let to_clip = |x: f32, y: f32| Some((x / 2.0 - 1.0, 1.0 - y));
        let mut rgba = vec![200_u8; 4 * 2 * 4];
        selection.apply(&mut rgba, 4, to_clip);
        let alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
        assert_eq!(alpha, vec![0, 200, 200, 0, 0, 200, 200, 0]);

        let inverted = Selection {
            invert: true,
            ..middle()
        };
        let mut rgba = vec![200_u8; 4 * 2 * 4];
        inverted.apply(&mut rgba, 4, to_clip);
        let alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
        assert_eq!(alpha, vec![200, 0, 0, 200, 200, 0, 0, 200]);
    }

    #[test]
    fn a_canvas_point_off_the_clip_is_not_kept() {
        let selection = Selection::from_alpha(2, 2, &[1; 4], Micros(0)).unwrap();
        let mut rgba = vec![255_u8; 4];
        selection.apply(&mut rgba, 1, |_, _| Some((1.5, 0.0)));
        assert_eq!(rgba, vec![0; 4]);
        let mut rgba = vec![255_u8; 4];
        selection.apply(&mut rgba, 1, |_, _| None);
        assert_eq!(rgba, vec![0; 4]);
    }

    #[test]
    fn the_edge_is_soft_between_pixels() {
        let selection = Selection::from_alpha(2, 1, &[255, 0], Micros(0)).unwrap();
        let lookup_alpha = selection.alpha().unwrap();
        let lookup = Lookup {
            alpha: &lookup_alpha,
            width: 2,
            height: 1,
        };
        // Halfway between the two pixel centres.
        assert!((lookup.coverage(0.0, 0.0) - 0.5).abs() < 1e-6);
        assert_eq!(lookup.coverage(-0.5, 0.0), 1.0);
        assert_eq!(lookup.coverage(0.5, 0.0), 0.0);
    }

    #[test]
    fn a_1080p_object_codes_to_a_few_runs_per_row() {
        // An ellipse: two edges a row, so about two counts a row.
        let (w, h) = (1920_u32, 1080_u32);
        let alpha: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (f64::from(i % w) - 960.0, f64::from(i / w) - 540.0);
                u8::from((x / 600.0).powi(2) + (y / 400.0).powi(2) < 1.0) * 255
            })
            .collect();
        let selection = Selection::from_alpha(w, h, &alpha, Micros(0)).unwrap();
        assert!(
            selection.runs.len() <= 2 * h as usize + 1,
            "{}",
            selection.runs.len()
        );
        assert_eq!(
            selection.alpha().unwrap(),
            alpha
                .iter()
                .map(|a| if *a == 0 { 0 } else { 255 })
                .collect::<Vec<_>>()
        );
    }
}
