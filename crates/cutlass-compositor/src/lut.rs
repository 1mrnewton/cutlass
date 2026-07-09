//! Adobe/Resolve `.cube` 3D lookup tables.
//!
//! A [`CubeLut`] is the CPU-side parse of a `.cube` file: an `N×N×N` grid of
//! RGB output triples, red-fastest (the format's mandated order). The
//! compositor uploads it as an RGBA8 3D texture and applies it with trilinear
//! sampling in a fullscreen pass (see `shaders/lut.wgsl`); intensity blends
//! the graded result over the original.
//!
//! 8-bit texels are deliberate: the canvas itself is `Rgba8Unorm`, so LUT
//! quantization adds at most one LSB to an already 8-bit pipeline while
//! keeping uploads small (a 33³ LUT is ~140 KB) and the texture filterable on
//! every backend without `FLOAT32_FILTERABLE`.

use thiserror::Error;

/// Largest accepted `LUT_3D_SIZE`. 65³ is already beyond every LUT we ship
/// (33 is the industry norm); the cap bounds a hostile file to ~3 MB of f32s.
pub const MAX_CUBE_SIZE: u32 = 65;

/// A `.cube` file failed to parse.
#[derive(Debug, Error)]
pub enum LutError {
    #[error("missing LUT_3D_SIZE (1D LUTs are not supported)")]
    MissingSize,
    #[error("invalid LUT_3D_SIZE {0} (expected 2..={MAX_CUBE_SIZE})")]
    InvalidSize(u32),
    #[error("line {line}: {message}")]
    Malformed { line: usize, message: String },
    #[error("expected {expected} data rows, found {found}")]
    WrongRowCount { expected: usize, found: usize },
}

/// A parsed 3D LUT: `size³` RGB triples, red index fastest.
#[derive(Debug, Clone, PartialEq)]
pub struct CubeLut {
    size: u32,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    /// `size³ * 3` floats, `[r, g, b]` per grid point, red-fastest.
    data: Vec<f32>,
}

impl CubeLut {
    /// Parse `.cube` text (comments, `TITLE`, `DOMAIN_MIN`/`MAX` honored).
    pub fn parse(text: &str) -> Result<Self, LutError> {
        let mut size: Option<u32> = None;
        let mut domain_min = [0.0f32; 3];
        let mut domain_max = [1.0f32; 3];
        let mut data: Vec<f32> = Vec::new();

        for (idx, raw) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tokens = line.split_whitespace();
            let head = tokens.next().expect("non-empty line has a token");
            match head {
                "TITLE" => {}
                "LUT_3D_SIZE" => {
                    let n: u32 = tokens
                        .next()
                        .and_then(|t| t.parse().ok())
                        .ok_or(LutError::Malformed {
                            line: line_no,
                            message: "unreadable LUT_3D_SIZE".into(),
                        })?;
                    if !(2..=MAX_CUBE_SIZE).contains(&n) {
                        return Err(LutError::InvalidSize(n));
                    }
                    size = Some(n);
                    data.reserve_exact((n as usize).pow(3) * 3);
                }
                "LUT_1D_SIZE" => return Err(LutError::MissingSize),
                "DOMAIN_MIN" | "DOMAIN_MAX" => {
                    let mut v = [0.0f32; 3];
                    for slot in &mut v {
                        *slot = tokens
                            .next()
                            .and_then(|t| t.parse().ok())
                            .filter(|f: &f32| f.is_finite())
                            .ok_or(LutError::Malformed {
                                line: line_no,
                                message: format!("unreadable {head}"),
                            })?;
                    }
                    if head == "DOMAIN_MIN" {
                        domain_min = v;
                    } else {
                        domain_max = v;
                    }
                }
                _ => {
                    // A data row: three floats.
                    let parse = |t: Option<&str>| {
                        t.and_then(|t| t.parse::<f32>().ok())
                            .filter(|f| f.is_finite())
                            .ok_or(LutError::Malformed {
                                line: line_no,
                                message: "expected three numbers".into(),
                            })
                    };
                    data.push(parse(Some(head))?);
                    data.push(parse(tokens.next())?);
                    data.push(parse(tokens.next())?);
                    if tokens.next().is_some() {
                        return Err(LutError::Malformed {
                            line: line_no,
                            message: "expected exactly three numbers".into(),
                        });
                    }
                }
            }
        }

        let size = size.ok_or(LutError::MissingSize)?;
        for (i, (min, max)) in domain_min.iter().zip(&domain_max).enumerate() {
            if max <= min {
                return Err(LutError::Malformed {
                    line: 0,
                    message: format!("domain axis {i} is empty ({min} >= {max})"),
                });
            }
        }
        let expected = (size as usize).pow(3);
        if data.len() != expected * 3 {
            return Err(LutError::WrongRowCount {
                expected,
                found: data.len() / 3,
            });
        }
        Ok(Self {
            size,
            domain_min,
            domain_max,
            data,
        })
    }

    /// Build a LUT by evaluating `f(r, g, b) -> [r, g, b]` on an `size³` grid
    /// over the unit domain (the starter-pack baker path).
    pub fn from_fn(size: u32, mut f: impl FnMut([f32; 3]) -> [f32; 3]) -> Self {
        assert!((2..=MAX_CUBE_SIZE).contains(&size), "cube size {size}");
        let n = size as usize;
        let step = 1.0 / (size - 1) as f32;
        let mut data = Vec::with_capacity(n.pow(3) * 3);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let out = f([r as f32 * step, g as f32 * step, b as f32 * step]);
                    data.extend_from_slice(&out);
                }
            }
        }
        Self {
            size,
            domain_min: [0.0; 3],
            domain_max: [1.0; 3],
            data,
        }
    }

    /// Grid resolution per axis.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Input value mapped to texture coordinate 0 per axis.
    pub fn domain_min(&self) -> [f32; 3] {
        self.domain_min
    }

    /// Input value mapped to texture coordinate 1 per axis.
    pub fn domain_max(&self) -> [f32; 3] {
        self.domain_max
    }

    /// Tightly packed RGBA8 texels for a 3D texture upload (alpha = 255),
    /// red-fastest, i.e. exactly WGPU's x-fastest 3D layout.
    pub fn rgba8_texels(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() / 3 * 4);
        for rgb in self.data.chunks_exact(3) {
            for c in rgb {
                out.push((c.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            out.push(255);
        }
        out
    }

    /// CPU trilinear sample (drift tests and the baker's round-trip checks).
    pub fn sample(&self, rgb: [f32; 3]) -> [f32; 3] {
        let n = self.size as usize;
        let mut coord = [0.0f32; 3];
        for i in 0..3 {
            let t = (rgb[i] - self.domain_min[i]) / (self.domain_max[i] - self.domain_min[i]);
            coord[i] = t.clamp(0.0, 1.0) * (self.size - 1) as f32;
        }
        let base = coord.map(|c| (c.floor() as usize).min(n - 2));
        let frac = [
            coord[0] - base[0] as f32,
            coord[1] - base[1] as f32,
            coord[2] - base[2] as f32,
        ];
        let at = |r: usize, g: usize, b: usize| {
            let idx = ((b * n + g) * n + r) * 3;
            [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
        };
        let mut out = [0.0f32; 3];
        for corner in 0..8usize {
            let (dr, dg, db) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let w = (if dr == 1 { frac[0] } else { 1.0 - frac[0] })
                * (if dg == 1 { frac[1] } else { 1.0 - frac[1] })
                * (if db == 1 { frac[2] } else { 1.0 - frac[2] });
            let v = at(base[0] + dr, base[1] + dg, base[2] + db);
            for i in 0..3 {
                out[i] += w * v[i];
            }
        }
        out
    }

    /// Serialize as `.cube` text (the starter-pack baker's output format).
    pub fn to_cube_string(&self, title: &str) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        if !title.is_empty() {
            let _ = writeln!(out, "TITLE \"{title}\"");
        }
        let _ = writeln!(out, "LUT_3D_SIZE {}", self.size);
        for rgb in self.data.chunks_exact(3) {
            let _ = writeln!(out, "{:.6} {:.6} {:.6}", rgb[0], rgb[1], rgb[2]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(size: u32) -> CubeLut {
        CubeLut::from_fn(size, |rgb| rgb)
    }

    #[test]
    fn parses_a_minimal_cube_file() {
        let text = "# comment\nTITLE \"tiny\"\nLUT_3D_SIZE 2\n\
                    0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let lut = CubeLut::parse(text).unwrap();
        assert_eq!(lut.size(), 2);
        // Red-fastest: entry 1 is r=1.
        assert_eq!(lut.sample([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
        assert_eq!(lut.sample([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn identity_roundtrips_through_cube_text() {
        let lut = identity(5);
        let reparsed = CubeLut::parse(&lut.to_cube_string("id")).unwrap();
        for probe in [[0.0, 0.5, 1.0], [0.25, 0.75, 0.1], [1.0, 1.0, 1.0]] {
            let got = reparsed.sample(probe);
            for i in 0..3 {
                assert!((got[i] - probe[i]).abs() < 1e-4, "{probe:?} -> {got:?}");
            }
        }
    }

    #[test]
    fn trilinear_sample_interpolates_between_grid_points() {
        // 2³ LUT that doubles green: g_out = min(2*g, 1) sampled at corners
        // {0,1} is g_out = g at 0, 1 at 1 — halfway in, expect 0.5..1 blend.
        let lut = CubeLut::from_fn(2, |[r, g, b]| [r, (2.0 * g).min(1.0), b]);
        let got = lut.sample([0.0, 0.5, 0.0]);
        assert!((got[1] - 0.5).abs() < 1e-5, "{got:?}");
    }

    #[test]
    fn rejects_wrong_row_counts_and_bad_sizes() {
        assert!(matches!(
            CubeLut::parse("LUT_3D_SIZE 2\n0 0 0\n"),
            Err(LutError::WrongRowCount { .. })
        ));
        assert!(matches!(
            CubeLut::parse("LUT_3D_SIZE 1\n0 0 0\n"),
            Err(LutError::InvalidSize(1))
        ));
        assert!(matches!(
            CubeLut::parse("0 0 0\n"),
            Err(LutError::MissingSize)
        ));
        assert!(matches!(
            CubeLut::parse("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n"),
            Err(LutError::MissingSize)
        ));
    }

    #[test]
    fn rejects_malformed_rows_and_domains() {
        assert!(matches!(
            CubeLut::parse("LUT_3D_SIZE 2\n0 0\n"),
            Err(LutError::Malformed { .. })
        ));
        assert!(matches!(
            CubeLut::parse("LUT_3D_SIZE 2\n0 0 0 0\n"),
            Err(LutError::Malformed { .. })
        ));
        assert!(matches!(
            CubeLut::parse("DOMAIN_MIN 0 0 0\nDOMAIN_MAX 0 0 0\nLUT_3D_SIZE 2\n"),
            Err(LutError::Malformed { .. })
        ));
    }

    #[test]
    fn rgba8_texels_quantize_and_pad_alpha() {
        let lut = identity(2);
        let texels = lut.rgba8_texels();
        assert_eq!(texels.len(), 8 * 4);
        assert_eq!(&texels[0..4], &[0, 0, 0, 255]);
        assert_eq!(&texels[4..8], &[255, 0, 0, 255]); // r-fastest
        assert_eq!(&texels[28..32], &[255, 255, 255, 255]);
    }
}
