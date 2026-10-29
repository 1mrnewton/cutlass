//! Rec.709, HLG, and PQ.
//!
//! Rec.709 is the camera OETF in ITU-R BT.709 and its inverse. The break is
//! `0.018` linear and `0.081` encoded. HLG is the BT.2100 OETF
//! (`a = 0.17883277`, `b = 0.28466892`, `c = 0.55991073`) and its inverse.
//! The HLG OOTF, which applies a system gamma from the display peak, is not
//! part of this transfer. PQ is SMPTE ST 2084: the signal maps to cd/m² with
//! a 10 000 nit peak. A relative linear value of 1 is `nits` cd/m², or 100
//! when the segment peak is 0.
//!
//! Primaries are the CIE xy values in BT.709, BT.2020, and SMPTE 170M, with
//! D65 at `(0.3127, 0.3290)`. Adaptation runs on linear RGB.

use cutlass_codec::access::{ColorPrimaries, Transfer};

use crate::{Error, Picture, Request};

/// A source or destination curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Curve {
    #[default]
    Unspecified,
    Rec709,
    /// SMPTE 170M. The OETF matches Rec.709.
    Rec601,
    Hlg,
    Pq,
}

impl Curve {
    #[must_use]
    pub fn from_codec(value: Transfer) -> Self {
        match value {
            Transfer::Unspecified => Self::Unspecified,
            Transfer::Bt709 | Transfer::Bt2020 => Self::Rec709,
            Transfer::Bt601 => Self::Rec601,
        }
    }

    fn family(self) -> Family {
        match self {
            Self::Unspecified | Self::Rec709 | Self::Rec601 => Family::Rec709,
            Self::Hlg => Family::Hlg,
            Self::Pq => Family::Pq,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Rec709,
    Hlg,
    Pq,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Primaries {
    Bt709,
    Bt2020,
    Smpte170,
}

pub(crate) struct Mapper {
    source: Family,
    dest: Family,
    adapt: Option<[[f64; 3]; 3]>,
    nits: f64,
}

impl Mapper {
    pub(crate) fn same_signal(&self) -> bool {
        self.adapt.is_none() && self.source == self.dest
    }

    pub(crate) fn map(&self, encoded: [f64; 3]) -> [f64; 3] {
        if self.same_signal() {
            return encoded;
        }
        let linear = decode(encoded, self.source, self.nits);
        let linear = match self.adapt {
            Some(matrix) => clamp3(mul(matrix, linear)),
            None => linear,
        };
        encode(linear, self.dest, self.nits)
    }
}

/// The pixel mapping, and the unforced path when intensity blends toward it.
///
/// Modes 0 and 1 follow the picture tags into the draft working space.
/// Modes 2 through 11 are the segment HDR mapping. The draft integer is not
/// an input to that mapping.
pub(crate) fn mappers(
    picture: &Picture<'_>,
    request: &Request,
) -> Result<(Mapper, Option<Mapper>), Error> {
    check_segment_numbers(request)?;
    let nits = reference_nits(request.segment.nits);
    let tags = tag_path(picture, request.draft)?;
    if request.segment.mode == 0 || request.segment.mode == 1 {
        return Ok((Mapper::build(tags, nits)?, None));
    }
    let forced = Mapper::build(forced_path(request.segment.mode)?, nits)?;
    if request.segment.intensity == 1.0 {
        return Ok((forced, None));
    }
    Ok((forced, Some(Mapper::build(tags, nits)?)))
}

struct Path {
    source: Curve,
    source_primaries: Primaries,
    dest: Curve,
    dest_primaries: Primaries,
}

fn check_segment_numbers(request: &Request) -> Result<(), Error> {
    let intensity = request.segment.intensity;
    if !intensity.is_finite() || !(0.0..=1.0).contains(&intensity) {
        return Err(Error::Intensity);
    }
    let nits = request.segment.nits;
    if !nits.is_finite() || nits < 0.0 {
        return Err(Error::Nits);
    }
    Ok(())
}

fn reference_nits(nits: f64) -> f64 {
    if nits == 0.0 { 100.0 } else { nits }
}

fn tag_path(picture: &Picture<'_>, draft: i32) -> Result<Path, Error> {
    let (dest, dest_primaries) = draft_destination(draft)?;
    Ok(Path {
        source: picture.curve,
        source_primaries: picture_primaries(picture),
        dest,
        dest_primaries,
    })
}

fn picture_primaries(picture: &Picture<'_>) -> Primaries {
    match picture.primaries {
        ColorPrimaries::Bt709 => Primaries::Bt709,
        ColorPrimaries::Bt2020 => Primaries::Bt2020,
        ColorPrimaries::Bt601 => Primaries::Smpte170,
        ColorPrimaries::Unspecified => match picture.matrix {
            crate::Matrix::Bt601 => Primaries::Smpte170,
            crate::Matrix::Bt2020 => Primaries::Bt2020,
            crate::Matrix::Bt709 => Primaries::Bt709,
        },
    }
}

fn draft_destination(code: i32) -> Result<(Curve, Primaries), Error> {
    match code {
        -1 | 0 => Ok((Curve::Rec709, Primaries::Bt709)),
        2 => Ok((Curve::Rec709, Primaries::Bt2020)),
        67 => Ok((Curve::Hlg, Primaries::Bt2020)),
        2084 => Ok((Curve::Pq, Primaries::Bt2020)),
        other => Err(Error::Draft(other)),
    }
}

fn forced_path(mode: i32) -> Result<Path, Error> {
    let path = match mode {
        0 | 1 => return Err(Error::Segment(mode)),
        2 => (
            Curve::Rec709,
            Primaries::Bt709,
            Curve::Hlg,
            Primaries::Bt2020,
        ),
        3 => (
            Curve::Rec709,
            Primaries::Bt709,
            Curve::Pq,
            Primaries::Bt2020,
        ),
        4 => (
            Curve::Hlg,
            Primaries::Bt2020,
            Curve::Rec709,
            Primaries::Bt709,
        ),
        5 => (Curve::Hlg, Primaries::Bt2020, Curve::Pq, Primaries::Bt2020),
        6 => (
            Curve::Pq,
            Primaries::Bt2020,
            Curve::Rec709,
            Primaries::Bt709,
        ),
        7 => (Curve::Pq, Primaries::Bt2020, Curve::Hlg, Primaries::Bt2020),
        8 => (
            Curve::Rec709,
            Primaries::Bt2020,
            Curve::Hlg,
            Primaries::Bt2020,
        ),
        9 => (
            Curve::Rec709,
            Primaries::Bt2020,
            Curve::Pq,
            Primaries::Bt2020,
        ),
        10 => (
            Curve::Hlg,
            Primaries::Bt2020,
            Curve::Rec709,
            Primaries::Bt2020,
        ),
        11 => (
            Curve::Pq,
            Primaries::Bt2020,
            Curve::Rec709,
            Primaries::Bt2020,
        ),
        other => return Err(Error::Segment(other)),
    };
    Ok(Path {
        source: path.0,
        source_primaries: path.1,
        dest: path.2,
        dest_primaries: path.3,
    })
}

impl Mapper {
    fn build(path: Path, nits: f64) -> Result<Self, Error> {
        let adapt = adaptation(path.source_primaries, path.dest_primaries)?;
        Ok(Self {
            source: path.source.family(),
            dest: path.dest.family(),
            adapt,
            nits,
        })
    }
}

fn adaptation(source: Primaries, dest: Primaries) -> Result<Option<[[f64; 3]; 3]>, Error> {
    if source == dest {
        return Ok(None);
    }
    let to_dest = inverse(rgb_to_xyz(dest)?)?;
    Ok(Some(mul_matrix(to_dest, rgb_to_xyz(source)?)))
}

fn decode(encoded: [f64; 3], family: Family, nits: f64) -> [f64; 3] {
    match family {
        Family::Rec709 => encoded.map(rec709_eotf),
        Family::Hlg => encoded.map(hlg_inverse_oetf),
        Family::Pq => encoded.map(|signal| clamp01(pq_eotf(signal) / nits)),
    }
}

fn encode(linear: [f64; 3], family: Family, nits: f64) -> [f64; 3] {
    match family {
        Family::Rec709 => linear.map(rec709_oetf),
        Family::Hlg => linear.map(hlg_oetf),
        Family::Pq => linear.map(|value| pq_oetf(clamp01(value) * nits)),
    }
}

fn rec709_oetf(linear: f64) -> f64 {
    let linear = clamp01(linear);
    if linear < 0.018 {
        4.5 * linear
    } else {
        1.099 * linear.powf(0.45) - 0.099
    }
}

fn rec709_eotf(encoded: f64) -> f64 {
    let encoded = clamp01(encoded);
    if encoded < 0.081 {
        encoded / 4.5
    } else {
        ((encoded + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

fn hlg_oetf(scene: f64) -> f64 {
    let scene = clamp01(scene);
    if scene <= 1.0 / 12.0 {
        (3.0 * scene).sqrt()
    } else {
        const A: f64 = 0.17883277;
        const B: f64 = 0.28466892;
        const C: f64 = 0.55991073;
        A * (12.0 * scene - B).ln() + C
    }
}

fn hlg_inverse_oetf(signal: f64) -> f64 {
    let signal = clamp01(signal);
    if signal <= 0.5 {
        signal * signal / 3.0
    } else {
        const A: f64 = 0.17883277;
        const B: f64 = 0.28466892;
        const C: f64 = 0.55991073;
        (((signal - C) / A).exp() + B) / 12.0
    }
}

fn pq_oetf(nits: f64) -> f64 {
    if nits <= 0.0 {
        return 0.0;
    }
    let (c1, c2, c3, m1, m2) = pq_constants();
    let y = (nits / 10_000.0).clamp(0.0, 1.0);
    let ym = y.powf(m1);
    ((c1 + c2 * ym) / (1.0 + c3 * ym)).powf(m2)
}

fn pq_eotf(signal: f64) -> f64 {
    let (c1, c2, c3, m1, m2) = pq_constants();
    let raised = clamp01(signal).powf(1.0 / m2);
    if raised <= c1 {
        return 0.0;
    }
    10_000.0 * ((raised - c1) / (c2 - c3 * raised)).powf(1.0 / m1)
}

fn pq_constants() -> (f64, f64, f64, f64, f64) {
    (
        3424.0 / 4096.0,
        2413.0 / 128.0,
        2392.0 / 128.0,
        2610.0 / 16384.0,
        2523.0 / 32.0,
    )
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn clamp3(value: [f64; 3]) -> [f64; 3] {
    value.map(clamp01)
}

fn rgb_to_xyz(primaries: Primaries) -> Result<[[f64; 3]; 3], Error> {
    let (xy, white) = match primaries {
        Primaries::Bt709 => ([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], (0.3127, 0.3290)),
        Primaries::Bt2020 => (
            [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
            (0.3127, 0.3290),
        ),
        Primaries::Smpte170 => (
            [(0.630, 0.340), (0.310, 0.595), (0.155, 0.070)],
            (0.3127, 0.3290),
        ),
    };
    let columns = [
        xy_to_xyz(xy[0].0, xy[0].1)?,
        xy_to_xyz(xy[1].0, xy[1].1)?,
        xy_to_xyz(xy[2].0, xy[2].1)?,
    ];
    let mut matrix = [
        [columns[0][0], columns[1][0], columns[2][0]],
        [columns[0][1], columns[1][1], columns[2][1]],
        [columns[0][2], columns[1][2], columns[2][2]],
    ];
    let scale = solve(matrix, xy_to_xyz(white.0, white.1)?)?;
    for (column, factor) in scale.into_iter().enumerate() {
        for row in &mut matrix {
            row[column] *= factor;
        }
    }
    Ok(matrix)
}

fn xy_to_xyz(x: f64, y: f64) -> Result<[f64; 3], Error> {
    if y == 0.0 {
        return Err(Error::Range);
    }
    Ok([x / y, 1.0, (1.0 - x - y) / y])
}

fn solve(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Result<[f64; 3], Error> {
    let mut row = [
        [matrix[0][0], matrix[0][1], matrix[0][2], rhs[0]],
        [matrix[1][0], matrix[1][1], matrix[1][2], rhs[1]],
        [matrix[2][0], matrix[2][1], matrix[2][2], rhs[2]],
    ];
    for column in 0..3 {
        let mut pivot = column;
        for candidate in column + 1..3 {
            if row[candidate][column].abs() > row[pivot][column].abs() {
                pivot = candidate;
            }
        }
        if row[pivot][column].abs() < 1.0e-12 {
            return Err(Error::Range);
        }
        row.swap(column, pivot);
        let divisor = row[column][column];
        for value in &mut row[column][column..] {
            *value /= divisor;
        }
        for other in 0..3 {
            if other == column {
                continue;
            }
            let factor = row[other][column];
            let pivot_row = row[column];
            for (slot, pivot) in row[other]
                .iter_mut()
                .skip(column)
                .zip(pivot_row.iter().skip(column))
            {
                *slot -= factor * pivot;
            }
        }
    }
    Ok([row[0][3], row[1][3], row[2][3]])
}

fn inverse(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], Error> {
    let x = solve(matrix, [1.0, 0.0, 0.0])?;
    let y = solve(matrix, [0.0, 1.0, 0.0])?;
    let z = solve(matrix, [0.0, 0.0, 1.0])?;
    Ok([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]])
}

fn mul_matrix(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

fn mul(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2],
        matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2],
        matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2],
    ]
}
