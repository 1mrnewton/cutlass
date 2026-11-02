//! Still gamma, chromaticity, and an RGB matrix profile.
//!
//! The working still is a power curve of `45455/100000` and BT.709 primaries
//! with D65. That is the usual PNG `gAMA`, so a file that already names it
//! is copied. An `sRGB` intent names that same working still, and its gamma
//! and chromaticity are not applied again. A readable RGB profile replaces
//! both. A profile this crate cannot read falls back to the gamma and the
//! chromaticity. Video transfer curves stay on the Rec.709, HLG, and PQ path.

use crate::Error;

/// PNG gamma whose power is the working still curve.
const REFERENCE_GAMMA_TIMES: u32 = 45_455;

/// White, red, green, and blue chromaticity, each x then y, times 100000.
const BT709_CHROMATICITY: [u32; 8] = [
    31_270, 32_900, 64_000, 33_000, 30_000, 60_000, 15_000, 6_000,
];

const D65: [f64; 3] = [0.95047, 1.0, 1.08883];

const BRADFORD: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

const BRADFORD_INVERSE: [[f64; 3]; 3] = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
];

const XYZ_TO_BT709: [[f64; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

const MAX_PROFILE: usize = 1 << 20;
const MAX_TAGS: u32 = 32;
const MAX_CURVE: usize = 4_096;

/// Tags that can move a still into the working curve.
#[derive(Clone, Copy, Debug)]
pub struct StillTags<'a> {
    /// Gamma times 100000. Absent when the file did not name one.
    pub gamma_times_100000: Option<u32>,
    /// White point, then red, green, and blue, each x then y, times 100000.
    pub chromaticity: Option<[u32; 8]>,
    /// Rendering intent 0 through 3. Any other value is not an sRGB intent.
    pub srgb_intent: Option<u8>,
    /// Profile bytes. Empty when the file did not name one.
    pub icc_profile: &'a [u8],
}

/// A prepared map from straight 8-bit RGB into the working still.
#[derive(Debug)]
pub struct StillMap {
    kind: Kind,
}

#[derive(Debug)]
enum Kind {
    Identity,
    Curve(Box<[u8; 256]>),
    Space(Box<Space>),
}

#[derive(Debug)]
struct Space {
    decode: [[f64; 256]; 3],
    to_xyz: [[f64; 3]; 3],
    adapt: [[f64; 3]; 3],
}

enum Trc {
    Gamma(f64),
    Table(Vec<f64>),
    Param(Param),
}

struct Param {
    kind: u16,
    g: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl StillMap {
    /// The map for `tags`.
    ///
    /// A gamma of zero, with no sRGB intent and no readable profile, is
    /// [`Error::Gamma`]. An unreadable profile is ignored.
    pub fn prepare(tags: &StillTags<'_>) -> Result<Self, Error> {
        if let Some(space) = profile_space(tags.icc_profile) {
            return Ok(Self {
                kind: Kind::Space(Box::new(space)),
            });
        }
        if tags.srgb_intent.is_some_and(|intent| intent <= 3) {
            return Ok(Self {
                kind: Kind::Identity,
            });
        }
        let gamma_times = match tags.gamma_times_100000 {
            Some(0) => return Err(Error::Gamma),
            Some(value) => value,
            None => REFERENCE_GAMMA_TIMES,
        };
        let gamma = f64::from(gamma_times) / 100_000.0;
        if let Some(space) = chromaticity_space(tags.chromaticity, gamma) {
            return Ok(Self {
                kind: Kind::Space(Box::new(space)),
            });
        }
        if gamma_times == REFERENCE_GAMMA_TIMES {
            return Ok(Self {
                kind: Kind::Identity,
            });
        }
        Ok(Self {
            kind: Kind::Curve(Box::new(curve_lut(gamma))),
        })
    }

    /// True when every sample is copied.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(self.kind, Kind::Identity)
    }

    /// One straight pixel. Alpha is not part of the map.
    #[must_use]
    pub fn map_rgb(&self, rgb: [u8; 3]) -> [u8; 3] {
        match &self.kind {
            Kind::Identity => rgb,
            Kind::Curve(lut) => [
                lut[usize::from(rgb[0])],
                lut[usize::from(rgb[1])],
                lut[usize::from(rgb[2])],
            ],
            Kind::Space(space) => {
                let linear = [
                    space.decode[0][usize::from(rgb[0])],
                    space.decode[1][usize::from(rgb[1])],
                    space.decode[2][usize::from(rgb[2])],
                ];
                let xyz = multiply(space.adapt, multiply(space.to_xyz, linear));
                let working = multiply(XYZ_TO_BT709, xyz);
                [
                    encode_linear(working[0]),
                    encode_linear(working[1]),
                    encode_linear(working[2]),
                ]
            }
        }
    }
}

fn reference_gamma() -> f64 {
    f64::from(REFERENCE_GAMMA_TIMES) / 100_000.0
}

fn curve_lut(file_gamma: f64) -> [u8; 256] {
    let exponent = reference_gamma() / file_gamma;
    let mut lut = [0_u8; 256];
    for (index, slot) in lut.iter_mut().enumerate() {
        let encoded = index as f64 / 255.0;
        *slot = unit_byte(encoded.powf(exponent));
    }
    lut
}

fn encode_linear(linear: f64) -> u8 {
    if !linear.is_finite() || linear <= 0.0 {
        return 0;
    }
    unit_byte(linear.min(1.0).powf(reference_gamma()))
}

fn unit_byte(unit: f64) -> u8 {
    if !unit.is_finite() || unit <= 0.0 {
        return 0;
    }
    if unit >= 1.0 {
        return 255;
    }
    let rounded = (unit * 255.0).round();
    if rounded <= 0.0 {
        0
    } else if rounded >= 255.0 {
        255
    } else {
        let whole = rounded as u16;
        u8::try_from(whole).unwrap_or(255)
    }
}

fn chromaticity_space(chromaticity: Option<[u32; 8]>, gamma: f64) -> Option<Space> {
    let values = chromaticity?;
    if values == BT709_CHROMATICITY || !gamma.is_finite() || gamma <= 0.0 {
        return None;
    }
    let white = xy(values[0], values[1])?;
    let red = xy(values[2], values[3])?;
    let green = xy(values[4], values[5])?;
    let blue = xy(values[6], values[7])?;
    let to_xyz = rgb_to_xyz(red, green, blue, white)?;
    let trc = Trc::Gamma(1.0 / gamma);
    space_from(to_xyz, xy_to_xyz(white)?, [&trc, &trc, &trc])
}

fn profile_space(profile: &[u8]) -> Option<Space> {
    if profile.is_empty() {
        return None;
    }
    let parsed = parse_profile(profile)?;
    space_from(
        parsed.to_xyz,
        parsed.illuminant,
        [&parsed.trc[0], &parsed.trc[1], &parsed.trc[2]],
    )
}

struct ParsedProfile {
    to_xyz: [[f64; 3]; 3],
    illuminant: [f64; 3],
    trc: [Trc; 3],
}

fn space_from(to_xyz: [[f64; 3]; 3], illuminant: [f64; 3], trc: [&Trc; 3]) -> Option<Space> {
    let adapt = adaptation(illuminant)?;
    let decode = [
        decode_table(trc[0])?,
        decode_table(trc[1])?,
        decode_table(trc[2])?,
    ];
    Some(Space {
        decode,
        to_xyz,
        adapt,
    })
}

fn decode_table(trc: &Trc) -> Option<[f64; 256]> {
    let mut table = [0.0; 256];
    for (index, slot) in table.iter_mut().enumerate() {
        let encoded = index as f64 / 255.0;
        let linear = trc.eval(encoded);
        if !linear.is_finite() {
            return None;
        }
        *slot = linear;
    }
    Some(table)
}

fn xy(x_times: u32, y_times: u32) -> Option<(f64, f64)> {
    if y_times == 0 || x_times.saturating_add(y_times) > 100_000 {
        return None;
    }
    Some((
        f64::from(x_times) / 100_000.0,
        f64::from(y_times) / 100_000.0,
    ))
}

fn xy_to_xyz((x, y): (f64, f64)) -> Option<[f64; 3]> {
    if y == 0.0 {
        return None;
    }
    Some([x / y, 1.0, (1.0 - x - y) / y])
}

fn rgb_to_xyz(
    red: (f64, f64),
    green: (f64, f64),
    blue: (f64, f64),
    white: (f64, f64),
) -> Option<[[f64; 3]; 3]> {
    let columns = [xy_to_xyz(red)?, xy_to_xyz(green)?, xy_to_xyz(blue)?];
    let unscaled = [
        [columns[0][0], columns[1][0], columns[2][0]],
        [columns[0][1], columns[1][1], columns[2][1]],
        [columns[0][2], columns[1][2], columns[2][2]],
    ];
    let scale = multiply(inverse(unscaled)?, xy_to_xyz(white)?);
    Some([
        [
            columns[0][0] * scale[0],
            columns[1][0] * scale[1],
            columns[2][0] * scale[2],
        ],
        [
            columns[0][1] * scale[0],
            columns[1][1] * scale[1],
            columns[2][1] * scale[2],
        ],
        [
            columns[0][2] * scale[0],
            columns[1][2] * scale[1],
            columns[2][2] * scale[2],
        ],
    ])
}

fn adaptation(illuminant: [f64; 3]) -> Option<[[f64; 3]; 3]> {
    let source = multiply(BRADFORD, illuminant);
    let destination = multiply(BRADFORD, D65);
    if source.iter().any(|cone| !cone.is_finite() || *cone == 0.0) {
        return None;
    }
    let scale = [
        [destination[0] / source[0], 0.0, 0.0],
        [0.0, destination[1] / source[1], 0.0],
        [0.0, 0.0, destination[2] / source[2]],
    ];
    Some(multiply_matrix(
        BRADFORD_INVERSE,
        multiply_matrix(scale, BRADFORD),
    ))
}

fn multiply(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2],
        matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2],
        matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2],
    ]
}

fn multiply_matrix(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            out[row][column] = left[row][0] * right[0][column]
                + left[row][1] * right[1][column]
                + left[row][2] * right[2][column];
        }
    }
    out
}

fn inverse(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let [a, b, c] = matrix[0];
    let [d, e, f] = matrix[1];
    let [g, h, i] = matrix[2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !det.is_finite() || det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (e * i - f * h) * inv,
            (c * h - b * i) * inv,
            (b * f - c * e) * inv,
        ],
        [
            (f * g - d * i) * inv,
            (a * i - c * g) * inv,
            (c * d - a * f) * inv,
        ],
        [
            (d * h - e * g) * inv,
            (b * g - a * h) * inv,
            (a * e - b * d) * inv,
        ],
    ])
}

fn parse_profile(bytes: &[u8]) -> Option<ParsedProfile> {
    if bytes.len() < 132 || bytes.len() > MAX_PROFILE {
        return None;
    }
    let declared = usize::try_from(be_u32(bytes, 0)?).ok()?;
    if declared < 132 || declared > bytes.len() {
        return None;
    }
    let bytes = &bytes[..declared];
    if &bytes[16..20] != b"RGB " || &bytes[20..24] != b"XYZ " || &bytes[36..40] != b"acsp" {
        return None;
    }
    let illuminant = [
        fixed(be_i32(bytes, 68)?),
        fixed(be_i32(bytes, 72)?),
        fixed(be_i32(bytes, 76)?),
    ];
    if illuminant
        .iter()
        .any(|value| !value.is_finite() || *value == 0.0)
    {
        return None;
    }
    let count = be_u32(bytes, 128)?;
    if count == 0 || count > MAX_TAGS {
        return None;
    }
    let table = 132_usize.checked_add(usize::try_from(count).ok()?.checked_mul(12)?)?;
    if table > bytes.len() {
        return None;
    }
    let mut red_xyz = None;
    let mut green_xyz = None;
    let mut blue_xyz = None;
    let mut red_trc = None;
    let mut green_trc = None;
    let mut blue_trc = None;
    for index in 0..count {
        let at = 132 + usize::try_from(index).ok()? * 12;
        let signature = be_u32(bytes, at)?;
        let offset = usize::try_from(be_u32(bytes, at + 4)?).ok()?;
        let size = usize::try_from(be_u32(bytes, at + 8)?).ok()?;
        match &signature.to_be_bytes() {
            b"rXYZ" => red_xyz = read_xyz(bytes, offset, size),
            b"gXYZ" => green_xyz = read_xyz(bytes, offset, size),
            b"bXYZ" => blue_xyz = read_xyz(bytes, offset, size),
            b"rTRC" => red_trc = read_trc(bytes, offset, size),
            b"gTRC" => green_trc = read_trc(bytes, offset, size),
            b"bTRC" => blue_trc = read_trc(bytes, offset, size),
            _ => {}
        }
    }
    let red_xyz = red_xyz?;
    let green_xyz = green_xyz?;
    let blue_xyz = blue_xyz?;
    Some(ParsedProfile {
        to_xyz: [
            [red_xyz[0], green_xyz[0], blue_xyz[0]],
            [red_xyz[1], green_xyz[1], blue_xyz[1]],
            [red_xyz[2], green_xyz[2], blue_xyz[2]],
        ],
        illuminant,
        trc: [red_trc?, green_trc?, blue_trc?],
    })
}

fn read_xyz(bytes: &[u8], offset: usize, size: usize) -> Option<[f64; 3]> {
    let end = offset.checked_add(size)?;
    if size < 20 || end > bytes.len() || &bytes[offset..offset + 4] != b"XYZ " {
        return None;
    }
    Some([
        fixed(be_i32(bytes, offset + 8)?),
        fixed(be_i32(bytes, offset + 12)?),
        fixed(be_i32(bytes, offset + 16)?),
    ])
}

fn read_trc(bytes: &[u8], offset: usize, size: usize) -> Option<Trc> {
    let end = offset.checked_add(size)?;
    if size < 12 || end > bytes.len() {
        return None;
    }
    let tag = &bytes[offset..end];
    match &tag[0..4] {
        b"curv" => read_curve(tag),
        b"para" => read_parametric(tag),
        _ => None,
    }
}

fn read_curve(tag: &[u8]) -> Option<Trc> {
    let count = usize::try_from(be_u32(tag, 8)?).ok()?;
    if count == 0 {
        return Some(Trc::Gamma(1.0));
    }
    if count == 1 {
        if tag.len() < 14 {
            return None;
        }
        let gamma = f64::from(be_u16(tag, 12)?) / 256.0;
        if gamma <= 0.0 || !gamma.is_finite() {
            return None;
        }
        return Some(Trc::Gamma(gamma));
    }
    if count > MAX_CURVE || tag.len() < 12 + count * 2 {
        return None;
    }
    let mut samples = Vec::with_capacity(count);
    for index in 0..count {
        samples.push(f64::from(be_u16(tag, 12 + index * 2)?) / 65_535.0);
    }
    Some(Trc::Table(samples))
}

fn read_parametric(tag: &[u8]) -> Option<Trc> {
    if tag.len() < 12 {
        return None;
    }
    let kind = be_u16(tag, 8)?;
    let count = match kind {
        0 => 1,
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 7,
        _ => return None,
    };
    if tag.len() < 12 + count * 4 {
        return None;
    }
    let mut params = [0.0; 7];
    for (index, slot) in params.iter_mut().enumerate().take(count) {
        *slot = fixed(be_i32(tag, 12 + index * 4)?);
        if !slot.is_finite() {
            return None;
        }
    }
    Some(Trc::Param(Param {
        kind,
        g: params[0],
        a: params[1],
        b: params[2],
        c: params[3],
        d: params[4],
        e: params[5],
        f: params[6],
    }))
}

impl Trc {
    fn eval(&self, encoded: f64) -> f64 {
        let encoded = encoded.clamp(0.0, 1.0);
        match self {
            Self::Gamma(gamma) => power(encoded, *gamma),
            Self::Table(samples) => sample_table(samples, encoded),
            Self::Param(param) => param.eval(encoded),
        }
    }
}

impl Param {
    fn eval(&self, x: f64) -> f64 {
        match self.kind {
            0 => power(x, self.g),
            1 => {
                if self.a == 0.0 {
                    return 0.0;
                }
                if x >= -self.b / self.a {
                    power(self.a * x + self.b, self.g)
                } else {
                    0.0
                }
            }
            2 => {
                if self.a == 0.0 {
                    return self.c;
                }
                if x >= -self.b / self.a {
                    power(self.a * x + self.b, self.g) + self.c
                } else {
                    self.c
                }
            }
            3 => {
                if x >= self.d {
                    power(self.a * x + self.b, self.g)
                } else {
                    self.c * x
                }
            }
            4 => {
                if x >= self.d {
                    power(self.a * x + self.b, self.g) + self.e
                } else {
                    self.c * x + self.f
                }
            }
            _ => f64::NAN,
        }
    }
}

fn power(base: f64, exponent: f64) -> f64 {
    if !base.is_finite() || !exponent.is_finite() {
        return f64::NAN;
    }
    if base <= 0.0 {
        return if exponent == 0.0 { 1.0 } else { 0.0 };
    }
    base.powf(exponent)
}

fn sample_table(samples: &[f64], encoded: f64) -> f64 {
    let Some(last) = samples.len().checked_sub(1) else {
        return f64::NAN;
    };
    if last == 0 {
        return samples[0];
    }
    let position = encoded.clamp(0.0, 1.0) * last as f64;
    let index = position.floor() as usize;
    if index >= last {
        return samples[last];
    }
    let fraction = position - index as f64;
    samples[index] * (1.0 - fraction) + samples[index + 1] * fraction
}

fn fixed(value: i32) -> f64 {
    f64::from(value) / 65_536.0
}

fn be_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let chunk = bytes.get(at..at + 4)?;
    let chunk: [u8; 4] = chunk.try_into().ok()?;
    Some(u32::from_be_bytes(chunk))
}

fn be_i32(bytes: &[u8], at: usize) -> Option<i32> {
    Some(be_u32(bytes, at)? as i32)
}

fn be_u16(bytes: &[u8], at: usize) -> Option<u16> {
    let chunk = bytes.get(at..at + 2)?;
    let chunk: [u8; 2] = chunk.try_into().ok()?;
    Some(u16::from_be_bytes(chunk))
}
