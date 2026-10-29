//! `.cube` and `.3dl` RGB lattices.
//!
//! A cube is the Adobe 3D LUT: `LUT_3D_SIZE` then red-fastest samples in
//! `0..=1`. A 3dl is the Autodesk lattice: one strictly increasing mesh line,
//! then blue-fastest integer samples. The two parsers do not accept each
//! other's text. A file that ends before `size³` samples is [`crate::LutIssue::Truncated`].
//!
//! Output integers at or below the mesh end are divided by that end. A sample
//! above the mesh end is 12-bit and is divided by 4095. Sampling is trilinear.

use crate::convert;
use crate::{Error, LutIssue, Working};

const MAX_TEXT: usize = 8 * 1024 * 1024;
const MAX_SIZE: u32 = 64;

/// A parsed `.cube` lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct Cube {
    lattice: Lattice,
}

/// A parsed `.3dl` lattice. This is not a [`Cube`].
#[derive(Clone, Debug, PartialEq)]
pub struct ThreeDl {
    lattice: Lattice,
}

/// Either LUT. Applying one remaps RGB and leaves alpha alone.
#[derive(Clone, Debug, PartialEq)]
pub enum Lut {
    Cube(Cube),
    ThreeDl(ThreeDl),
}

#[derive(Clone, Debug, PartialEq)]
struct Lattice {
    size: u32,
    domain_min: [f64; 3],
    domain_max: [f64; 3],
    samples: Vec<[f64; 3]>,
}

impl Cube {
    /// Reads a `.cube` file. A `.3dl` file has no `LUT_3D_SIZE` and is refused.
    pub fn parse(text: &str) -> Result<Self, Error> {
        bounded(text)?;
        let mut size = None;
        let mut domain_min = [0.0, 0.0, 0.0];
        let mut domain_max = [1.0, 1.0, 1.0];
        let mut samples = Vec::new();
        for line in data_lines(text) {
            if line.starts_with("TITLE") {
                if !samples.is_empty() {
                    return Err(Error::Lut(LutIssue::Value));
                }
                continue;
            }
            if line.starts_with("LUT_1D_SIZE") {
                return Err(Error::Lut(LutIssue::OneDimensional));
            }
            if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
                if size.is_some() || !samples.is_empty() {
                    return Err(Error::Lut(LutIssue::Header));
                }
                size = Some(parse_size(rest)?);
                continue;
            }
            if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
                if !samples.is_empty() {
                    return Err(Error::Lut(LutIssue::Value));
                }
                domain_min = parse_triplet(rest)?;
                continue;
            }
            if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
                if !samples.is_empty() {
                    return Err(Error::Lut(LutIssue::Value));
                }
                domain_max = parse_triplet(rest)?;
                continue;
            }
            let Some(n) = size else {
                return Err(Error::Lut(LutIssue::Header));
            };
            samples.push(parse_triplet(line)?);
            if samples.len() > lattice_len(n)? {
                return Err(Error::Lut(LutIssue::Value));
            }
        }
        let Some(n) = size else {
            return Err(Error::Lut(LutIssue::Header));
        };
        if samples.len() < lattice_len(n)? {
            return Err(Error::Lut(LutIssue::Truncated));
        }
        for channel in 0..3 {
            if domain_min[channel] >= domain_max[channel] {
                return Err(Error::Lut(LutIssue::Value));
            }
        }
        Ok(Self {
            lattice: Lattice {
                size: n,
                domain_min,
                domain_max,
                samples,
            },
        })
    }

    #[must_use]
    pub fn map(&self, rgb: [u8; 3]) -> [u8; 3] {
        self.lattice.map_bytes(rgb)
    }
}

impl ThreeDl {
    /// Reads a `.3dl` file. A `.cube` file has no integer mesh and is refused.
    pub fn parse(text: &str) -> Result<Self, Error> {
        bounded(text)?;
        let mut lines = data_lines(text);
        let Some(mesh_line) = lines.next() else {
            return Err(Error::Lut(LutIssue::Header));
        };
        let mesh = parse_mesh(mesh_line)?;
        let size = u32::try_from(mesh.len()).map_err(|_| Error::Lut(LutIssue::Size))?;
        if !(2..=MAX_SIZE).contains(&size) {
            return Err(Error::Lut(LutIssue::Size));
        }
        let need = lattice_len(size)?;
        let mut raw = Vec::with_capacity(need);
        let mut peak = 0_i64;
        for line in lines {
            if raw.len() == need {
                return Err(Error::Lut(LutIssue::Value));
            }
            let sample = parse_ints(line, 3)?;
            if sample.len() != 3 {
                return Err(Error::Lut(LutIssue::Value));
            }
            for value in &sample {
                peak = peak.max(*value);
            }
            raw.push(sample);
        }
        if raw.len() < need {
            return Err(Error::Lut(LutIssue::Truncated));
        }
        let Some(mesh_end) = mesh.last().copied() else {
            return Err(Error::Lut(LutIssue::Mesh));
        };
        if peak > 4095 {
            return Err(Error::Lut(LutIssue::Value));
        }
        let scale = if peak > mesh_end {
            4095.0
        } else if mesh_end == 0 {
            return Err(Error::Lut(LutIssue::Mesh));
        } else {
            let end = u32::try_from(mesh_end).map_err(|_| Error::Lut(LutIssue::Value))?;
            f64::from(end)
        };
        let mut samples = vec![[0.0; 3]; need];
        let mut cursor = raw.into_iter();
        for red in 0..size {
            for green in 0..size {
                for blue in 0..size {
                    let Some(sample) = cursor.next() else {
                        return Err(Error::Lut(LutIssue::Truncated));
                    };
                    let index = lattice_index(size, red, green, blue)?;
                    let slot = samples
                        .get_mut(index)
                        .ok_or(Error::Lut(LutIssue::Truncated))?;
                    *slot = [
                        f64::from(
                            u32::try_from(sample[0]).map_err(|_| Error::Lut(LutIssue::Value))?,
                        ) / scale,
                        f64::from(
                            u32::try_from(sample[1]).map_err(|_| Error::Lut(LutIssue::Value))?,
                        ) / scale,
                        f64::from(
                            u32::try_from(sample[2]).map_err(|_| Error::Lut(LutIssue::Value))?,
                        ) / scale,
                    ];
                }
            }
        }
        Ok(Self {
            lattice: Lattice {
                size,
                domain_min: [0.0, 0.0, 0.0],
                domain_max: [1.0, 1.0, 1.0],
                samples,
            },
        })
    }

    #[must_use]
    pub fn map(&self, rgb: [u8; 3]) -> [u8; 3] {
        self.lattice.map_bytes(rgb)
    }
}

impl Lut {
    #[must_use]
    pub fn map(&self, rgb: [u8; 3]) -> [u8; 3] {
        match self {
            Self::Cube(cube) => cube.map(rgb),
            Self::ThreeDl(table) => table.map(rgb),
        }
    }
}

pub(crate) fn apply(working: &mut Working, lut: &Lut) -> Result<(), Error> {
    let width = usize::try_from(working.width).map_err(|_| Error::Limit)?;
    let height = usize::try_from(working.height).map_err(|_| Error::Limit)?;
    let need = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(4))
        .ok_or(Error::Limit)?;
    if working.rgba.len() != need {
        return Err(Error::Length);
    }
    let (pixels, _) = working.rgba.as_chunks_mut::<4>();
    for pixel in pixels {
        let [red, green, blue, alpha] = *pixel;
        let mapped = lut.map([red, green, blue]);
        *pixel = [mapped[0], mapped[1], mapped[2], alpha];
    }
    Ok(())
}

impl Lattice {
    fn map_bytes(&self, rgb: [u8; 3]) -> [u8; 3] {
        let unit = [
            f64::from(rgb[0]) / 255.0,
            f64::from(rgb[1]) / 255.0,
            f64::from(rgb[2]) / 255.0,
        ];
        convert::quantize(self.sample(unit))
    }

    fn sample(&self, rgb: [f64; 3]) -> [f64; 3] {
        let red = normalize(rgb[0], self.domain_min[0], self.domain_max[0]);
        let green = normalize(rgb[1], self.domain_min[1], self.domain_max[1]);
        let blue = normalize(rgb[2], self.domain_min[2], self.domain_max[2]);
        let levels = self.size.saturating_sub(1);
        let (r0, rt) = axis(red, levels);
        let (g0, gt) = axis(green, levels);
        let (b0, bt) = axis(blue, levels);
        let r1 = (r0 + 1).min(levels);
        let g1 = (g0 + 1).min(levels);
        let b1 = (b0 + 1).min(levels);
        let corner = |r, g, b| self.at(r, g, b);
        let c00 = lerp3(corner(r0, g0, b0), corner(r1, g0, b0), rt);
        let c01 = lerp3(corner(r0, g0, b1), corner(r1, g0, b1), rt);
        let c10 = lerp3(corner(r0, g1, b0), corner(r1, g1, b0), rt);
        let c11 = lerp3(corner(r0, g1, b1), corner(r1, g1, b1), rt);
        let c0 = lerp3(c00, c10, gt);
        let c1 = lerp3(c01, c11, gt);
        lerp3(c0, c1, bt)
    }

    fn at(&self, red: u32, green: u32, blue: u32) -> [f64; 3] {
        let Ok(index) = lattice_index(self.size, red, green, blue) else {
            return [0.0, 0.0, 0.0];
        };
        self.samples.get(index).copied().unwrap_or([0.0, 0.0, 0.0])
    }
}

fn bounded(text: &str) -> Result<(), Error> {
    if text.len() > MAX_TEXT {
        return Err(Error::Lut(LutIssue::Limit));
    }
    Ok(())
}

fn data_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter_map(|line| {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') {
            None
        } else {
            Some(line)
        }
    })
}

fn parse_size(rest: &str) -> Result<u32, Error> {
    let mut parts = rest.split_whitespace();
    let Some(token) = parts.next() else {
        return Err(Error::Lut(LutIssue::Header));
    };
    if parts.next().is_some() {
        return Err(Error::Lut(LutIssue::Header));
    }
    let size: u32 = token.parse().map_err(|_| Error::Lut(LutIssue::Size))?;
    if !(2..=MAX_SIZE).contains(&size) {
        return Err(Error::Lut(LutIssue::Size));
    }
    Ok(size)
}

fn parse_triplet(text: &str) -> Result<[f64; 3], Error> {
    let mut parts = text.split_whitespace();
    let mut out = [0.0; 3];
    for slot in &mut out {
        let Some(token) = parts.next() else {
            return Err(Error::Lut(LutIssue::Value));
        };
        let value: f64 = token.parse().map_err(|_| Error::Lut(LutIssue::Value))?;
        if !value.is_finite() {
            return Err(Error::Lut(LutIssue::Value));
        }
        *slot = value;
    }
    if parts.next().is_some() {
        return Err(Error::Lut(LutIssue::Value));
    }
    Ok(out)
}

fn parse_mesh(line: &str) -> Result<Vec<i64>, Error> {
    let cap = usize::try_from(MAX_SIZE).map_err(|_| Error::Lut(LutIssue::Size))?;
    let values = parse_ints(line, cap.saturating_add(1))?;
    if values.len() < 2 {
        return Err(Error::Lut(LutIssue::Header));
    }
    let max = usize::try_from(MAX_SIZE).map_err(|_| Error::Lut(LutIssue::Size))?;
    if values.len() > max {
        return Err(Error::Lut(LutIssue::Size));
    }
    let mut previous = None;
    for value in &values {
        if *value < 0 {
            return Err(Error::Lut(LutIssue::Mesh));
        }
        if let Some(earlier) = previous
            && *value <= earlier
        {
            return Err(Error::Lut(LutIssue::Mesh));
        }
        previous = Some(*value);
    }
    Ok(values)
}

fn parse_ints(line: &str, cap: usize) -> Result<Vec<i64>, Error> {
    let mut values = Vec::new();
    for token in line.split_whitespace() {
        if values.len() > cap {
            return Err(Error::Lut(LutIssue::Value));
        }
        let value: i64 = token.parse().map_err(|_| Error::Lut(LutIssue::Value))?;
        if value < 0 {
            return Err(Error::Lut(LutIssue::Value));
        }
        values.push(value);
    }
    if values.is_empty() {
        return Err(Error::Lut(LutIssue::Value));
    }
    Ok(values)
}

fn lattice_len(size: u32) -> Result<usize, Error> {
    let size = usize::try_from(size).map_err(|_| Error::Lut(LutIssue::Size))?;
    size.checked_mul(size)
        .and_then(|square| square.checked_mul(size))
        .ok_or(Error::Lut(LutIssue::Size))
}

fn lattice_index(size: u32, red: u32, green: u32, blue: u32) -> Result<usize, Error> {
    let size = usize::try_from(size).map_err(|_| Error::Lut(LutIssue::Size))?;
    let red = usize::try_from(red).map_err(|_| Error::Lut(LutIssue::Size))?;
    let green = usize::try_from(green).map_err(|_| Error::Lut(LutIssue::Size))?;
    let blue = usize::try_from(blue).map_err(|_| Error::Lut(LutIssue::Size))?;
    blue.checked_mul(size)
        .and_then(|value| value.checked_add(green))
        .and_then(|value| value.checked_mul(size))
        .and_then(|value| value.checked_add(red))
        .ok_or(Error::Lut(LutIssue::Size))
}

fn normalize(value: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        return 0.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn axis(value: f64, levels: u32) -> (u32, f64) {
    if levels == 0 {
        return (0, 0.0);
    }
    let scaled = value * f64::from(levels);
    let base = scaled.floor();
    let index = if base <= 0.0 {
        0
    } else {
        u32::try_from(base as u64).unwrap_or(levels)
    };
    let index = index.min(levels);
    if index == levels {
        (levels, 0.0)
    } else {
        (index, (scaled - f64::from(index)).clamp(0.0, 1.0))
    }
}

fn lerp3(from: [f64; 3], to: [f64; 3], t: f64) -> [f64; 3] {
    [
        from[0] + (to[0] - from[0]) * t,
        from[1] + (to[1] - from[1]) * t,
        from[2] + (to[2] - from[2]) * t,
    ]
}
