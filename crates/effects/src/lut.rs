//! Look-up tables: a `.cube` file read, kept in the draft, and run over
//! a picture.
//!
//! A clip's LUT is an effect material of type `lut` it lists among its
//! extra materials. `value` is the strength, from 0 to 1, and
//! `adjust_params` holds the table: `{"size": n, "rgb": "<hex>"}`, with the
//! table resampled to [`KEPT_SIZE`] points a side and red changing
//! fastest, one byte a channel. The table travels with the project, so a
//! clip looks the same on every screen and in an export, with no file to
//! find.

use std::fmt;

use cutlass_model::{Draft, MaterialRef, Segment};
use serde_json::{Value, json};

/// The material `type` of a clip's LUT.
pub const LUT: &str = "lut";
/// The points a side a kept table has.
pub const KEPT_SIZE: usize = 17;
/// The largest `.cube` table read, points a side.
pub const MAX_CUBE_SIZE: usize = 65;

/// A 3D table from colour to colour, red changing fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    size: usize,
    table: Vec<[f32; 3]>,
}

/// Why a `.cube` file was not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LutError {
    /// The file has no `LUT_3D_SIZE`, or a 1D table only.
    NoTable,
    /// The size is below 2 or above [`MAX_CUBE_SIZE`].
    Size(usize),
    /// A line that is not three numbers, by line number.
    Line(usize),
    /// The number of rows is not the size cubed.
    Rows { expected: usize, found: usize },
}

impl fmt::Display for LutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTable => write!(f, "the file has no 3D table"),
            Self::Size(size) => write!(
                f,
                "a table {size} points a side is not read; 2 to {MAX_CUBE_SIZE} are"
            ),
            Self::Line(line) => write!(f, "line {line} is not three numbers"),
            Self::Rows { expected, found } => {
                write!(f, "the table has {found} rows, not {expected}")
            }
        }
    }
}

impl std::error::Error for LutError {}

impl Lut {
    /// Read a `.cube` file's text. Values are held to 0 to 1 after the
    /// file's domain is taken out.
    pub fn parse_cube(text: &str) -> Result<Self, LutError> {
        let mut size = None;
        let mut low = [0.0f32; 3];
        let mut high = [1.0f32; 3];
        let mut rows = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split_whitespace();
            let Some(first) = words.next() else {
                continue;
            };
            let numbers = |words: std::str::SplitWhitespace<'_>| -> Option<[f32; 3]> {
                let values: Vec<f32> = words.map(str::parse).collect::<Result<_, _>>().ok()?;
                (values.len() == 3 && values.iter().all(|value| value.is_finite()))
                    .then(|| [values[0], values[1], values[2]])
            };
            match first {
                "TITLE" | "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => {}
                "LUT_1D_SIZE" => return Err(LutError::NoTable),
                "LUT_3D_SIZE" => {
                    let value: usize = words
                        .next()
                        .and_then(|word| word.parse().ok())
                        .ok_or(LutError::Line(index + 1))?;
                    if !(2..=MAX_CUBE_SIZE).contains(&value) {
                        return Err(LutError::Size(value));
                    }
                    size = Some(value);
                }
                "DOMAIN_MIN" => low = numbers(words).ok_or(LutError::Line(index + 1))?,
                "DOMAIN_MAX" => high = numbers(words).ok_or(LutError::Line(index + 1))?,
                _ => {
                    let row = numbers(line.split_whitespace()).ok_or(LutError::Line(index + 1))?;
                    rows.push(row);
                }
            }
        }
        let size = size.ok_or(LutError::NoTable)?;
        let expected = size * size * size;
        if rows.len() != expected {
            return Err(LutError::Rows {
                expected,
                found: rows.len(),
            });
        }
        let table = rows
            .into_iter()
            .map(|row| {
                std::array::from_fn(|channel| {
                    let span = (high[channel] - low[channel]).max(1e-6);
                    ((row[channel] - low[channel]) / span).clamp(0.0, 1.0)
                })
            })
            .collect();
        Ok(Self { size, table })
    }

    /// The table that leaves every colour as it is, `size` a side.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let size = size.clamp(2, MAX_CUBE_SIZE);
        #[allow(clippy::cast_precision_loss)]
        let step = |index: usize| index as f32 / (size - 1) as f32;
        let mut table = Vec::with_capacity(size * size * size);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    table.push([step(r), step(g), step(b)]);
                }
            }
        }
        Self { size, table }
    }

    /// Points a side.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The colour the table gives `colour`, between its points.
    #[must_use]
    pub fn sample(&self, colour: [f32; 3]) -> [f32; 3] {
        #[allow(clippy::cast_precision_loss)]
        let top = (self.size - 1) as f32;
        let mut base = [0usize; 3];
        let mut part = [0f32; 3];
        for channel in 0..3 {
            let position = colour[channel].clamp(0.0, 1.0) * top;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let low = (position.floor() as usize).min(self.size - 2);
            #[allow(clippy::cast_precision_loss)]
            let low_f = low as f32;
            base[channel] = low;
            part[channel] = position - low_f;
        }
        let at = |r: usize, g: usize, b: usize| self.table[(b * self.size + g) * self.size + r];
        let mut out = [0f32; 3];
        for corner in 0..8 {
            let (dr, dg, db) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let weight = (if dr == 1 { part[0] } else { 1.0 - part[0] })
                * (if dg == 1 { part[1] } else { 1.0 - part[1] })
                * (if db == 1 { part[2] } else { 1.0 - part[2] });
            if weight == 0.0 {
                continue;
            }
            let value = at(base[0] + dr, base[1] + dg, base[2] + db);
            for channel in 0..3 {
                out[channel] += value[channel] * weight;
            }
        }
        out
    }

    /// The same table with `size` points a side.
    #[must_use]
    pub fn resampled(&self, size: usize) -> Self {
        let mut grid = Self::identity(size);
        for value in &mut grid.table {
            *value = self.sample(*value);
        }
        grid
    }

    /// The table as stored on the material: [`KEPT_SIZE`] a side.
    #[must_use]
    pub fn to_params(&self) -> Value {
        let kept = if self.size == KEPT_SIZE {
            self.clone()
        } else {
            self.resampled(KEPT_SIZE)
        };
        let mut hex = String::with_capacity(kept.table.len() * 6);
        for value in kept.table.iter().flatten() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            hex.push_str(&format!("{byte:02x}"));
        }
        json!({ "size": KEPT_SIZE, "rgb": hex })
    }

    /// The table stored on a material, when it is whole.
    #[must_use]
    pub fn from_params(params: &Value) -> Option<Self> {
        let size = usize::try_from(params.get("size")?.as_u64()?).ok()?;
        if !(2..=MAX_CUBE_SIZE).contains(&size) {
            return None;
        }
        let hex = params.get("rgb")?.as_str()?.as_bytes();
        if hex.len() != size * size * size * 6 {
            return None;
        }
        let digit = |byte: u8| char::from(byte).to_digit(16);
        let mut bytes = Vec::with_capacity(hex.len() / 2);
        for pair in hex.as_chunks::<2>().0 {
            let value = digit(pair[0])? * 16 + digit(pair[1])?;
            #[allow(clippy::cast_precision_loss)]
            bytes.push(value as f32 / 255.0);
        }
        let table = bytes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|rgb| [rgb[0], rgb[1], rgb[2]])
            .collect();
        Some(Self { size, table })
    }
}

/// A clip's LUT and its strength, when it has one.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipLut {
    pub name: String,
    pub lut: Lut,
    pub strength: f32,
}

impl ClipLut {
    /// Run the table over premultiplied RGBA, mixed in by the strength.
    pub fn apply(&self, rgba: &mut [u8]) {
        let strength = self.strength.clamp(0.0, 1.0);
        if strength == 0.0 {
            return;
        }
        for pixel in rgba.as_chunks_mut::<4>().0 {
            let alpha = f32::from(pixel[3]) / 255.0;
            if alpha <= 0.0 {
                continue;
            }
            let colour = [0, 1, 2].map(|index| (f32::from(pixel[index]) / 255.0 / alpha).min(1.0));
            let mapped = self.lut.sample(colour);
            for index in 0..3 {
                let value = colour[index] + (mapped[index] - colour[index]) * strength;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte = (value.clamp(0.0, 1.0) * alpha * 255.0).round() as u8;
                pixel[index] = byte;
            }
        }
    }
}

/// The LUT `segment` lists among its extra materials.
#[must_use]
pub fn segment_lut(draft: &Draft, segment: &Segment) -> Option<ClipLut> {
    segment.extra_material_refs.iter().find_map(|id| {
        let Some(MaterialRef::Effect(effect)) = draft.materials.get(id.as_str()) else {
            return None;
        };
        if effect.kind != LUT {
            return None;
        }
        let lut = Lut::from_params(effect.adjust_params.as_ref()?)?;
        #[allow(clippy::cast_possible_truncation)]
        let strength = effect.value.unwrap_or(1.0) as f32;
        Some(ClipLut {
            name: effect.name.clone(),
            lut,
            strength,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{ClipLut, KEPT_SIZE, Lut, LutError};

    fn invert_cube(size: usize) -> String {
        let mut text = format!("TITLE \"invert\"\n# a comment\nLUT_3D_SIZE {size}\n\n");
        #[allow(clippy::cast_precision_loss)]
        let step = |index: usize| index as f32 / (size - 1) as f32;
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    text.push_str(&format!(
                        "{} {} {}\n",
                        1.0 - step(r),
                        1.0 - step(g),
                        1.0 - step(b)
                    ));
                }
            }
        }
        text
    }

    #[test]
    fn a_cube_file_is_read_and_sampled_between_its_points() {
        let lut = Lut::parse_cube(&invert_cube(5)).unwrap();
        assert_eq!(lut.size(), 5);
        let out = lut.sample([0.1, 0.5, 0.9]);
        for (value, want) in out.iter().zip([0.9, 0.5, 0.1]) {
            assert!((value - want).abs() < 1e-5, "{out:?}");
        }
        let same = Lut::identity(9).sample([0.3, 0.6, 0.2]);
        assert!((same[0] - 0.3).abs() < 1e-5 && (same[2] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn a_domain_is_taken_out_and_bad_files_are_refused() {
        let text = "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n\
            0 0 0\n2 0 0\n0 2 0\n2 2 0\n0 0 2\n2 0 2\n0 2 2\n2 2 2\n";
        let lut = Lut::parse_cube(text).unwrap();
        let out = lut.sample([1.0, 0.0, 0.5]);
        assert!((out[0] - 1.0).abs() < 1e-6 && (out[2] - 0.5).abs() < 1e-6);
        assert_eq!(Lut::parse_cube("TITLE x\n"), Err(LutError::NoTable));
        assert_eq!(Lut::parse_cube("LUT_1D_SIZE 4\n"), Err(LutError::NoTable));
        assert_eq!(Lut::parse_cube("LUT_3D_SIZE 1\n"), Err(LutError::Size(1)));
        assert_eq!(
            Lut::parse_cube("LUT_3D_SIZE 2\n0 0 0\n"),
            Err(LutError::Rows {
                expected: 8,
                found: 1
            })
        );
        assert_eq!(
            Lut::parse_cube("LUT_3D_SIZE 2\n0 zero 0\n"),
            Err(LutError::Line(2))
        );
    }

    #[test]
    fn the_kept_table_survives_the_draft_and_mixes_by_strength() {
        let lut = Lut::parse_cube(&invert_cube(33)).unwrap();
        let params = lut.to_params();
        let kept = Lut::from_params(&params).unwrap();
        assert_eq!(kept.size(), KEPT_SIZE);
        let out = kept.sample([0.25, 0.5, 1.0]);
        assert!((out[0] - 0.75).abs() < 0.01, "{out:?}");
        assert!(Lut::from_params(&serde_json::json!({"size": 17, "rgb": "00"})).is_none());

        let mut half = ClipLut {
            name: "invert".to_owned(),
            lut: kept,
            strength: 0.5,
        };
        let mut pixel = vec![255, 0, 0, 255];
        half.apply(&mut pixel);
        assert_eq!(&pixel[..3], &[128, 128, 128]);
        half.strength = 1.0;
        let mut pixel = vec![100, 0, 0, 128];
        half.apply(&mut pixel);
        assert_eq!(pixel[3], 128);
        assert!(
            (27..=29).contains(&pixel[0]) && pixel[1] == 128,
            "{pixel:?}"
        );
    }
}
