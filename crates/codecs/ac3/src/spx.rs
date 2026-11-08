//! Spectral extension (A/52:2018 Annex E clause 3.6), and the 12-bin
//! sub-band banding it shares with channel coupling.

use crate::tables::SPX_ATTEN;

/// Most bands of either banding: 18 coupling sub-bands, 17 for SPX.
pub(crate) const MAX_BANDS: usize = 18;

/// Bands formed from 12-bin sub-bands `begin..end`: a set
/// `structure[sbnd]` merges sub-band `sbnd` into the band before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Bands {
    pub count: usize,
    pub sizes: [usize; MAX_BANDS],
}

impl Bands {
    pub(crate) fn new(structure: &[bool], begin: usize, end: usize) -> Self {
        let mut bands = Self::default();
        if begin >= end {
            return bands;
        }
        bands.count = 1;
        bands.sizes[0] = 12;
        for sbnd in begin + 1..end {
            if structure.get(sbnd).copied().unwrap_or(false) {
                bands.sizes[bands.count - 1] += 12;
            } else if bands.count < MAX_BANDS {
                bands.sizes[bands.count] = 12;
                bands.count += 1;
            }
        }
        bands
    }

    pub(crate) fn sizes(&self) -> &[usize] {
        &self.sizes[..self.count]
    }
}

/// First bin of SPX sub-band `sbnd` (Table E3.13).
pub(crate) fn band_bin(sbnd: usize) -> usize {
    25 + 12 * sbnd
}

/// `spx_begin_subbnd` from `spxbegf` and `spx_end_subbnd` from `spxendf`.
pub(crate) fn begin_subband(spxbegf: usize) -> usize {
    if spxbegf < 6 {
        spxbegf + 2
    } else {
        spxbegf * 2 - 3
    }
}

pub(crate) fn end_subband(spxendf: usize) -> usize {
    if spxendf < 3 {
        spxendf + 5
    } else {
        spxendf * 2 + 3
    }
}

/// The frame-level SPX ranges in force for a block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Range {
    /// `spxstrtf`: the first copied sub-band.
    pub start: usize,
    pub begin: usize,
    pub end: usize,
    pub bands: Bands,
}

impl Range {
    pub(crate) fn copy_start(&self) -> usize {
        band_bin(self.start)
    }

    /// The first synthesized bin; channels in SPX code mantissas below it.
    pub(crate) fn begin_bin(&self) -> usize {
        band_bin(self.begin)
    }

    pub(crate) fn end_bin(&self) -> usize {
        band_bin(self.end)
    }
}

/// One channel's SPX coordinates and blend factors per band.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Coords {
    pub coord: [f32; MAX_BANDS],
    pub noise: [f32; MAX_BANDS],
    pub signal: [f32; MAX_BANDS],
}

impl Coords {
    /// Blend factors from `spxblnd` (clause 3.6.4.2.1).
    pub(crate) fn blend(&mut self, range: &Range, spxblnd: u32) {
        let offset = spxblnd as f32 / 32.0;
        let mut bin = range.begin_bin();
        let end = range.end_bin() as f32;
        for (band, &size) in range.bands.sizes().iter().enumerate() {
            let ratio = ((bin as f32 + 0.5 * size as f32) / end - offset).clamp(0.0, 1.0);
            self.noise[band] = ratio.sqrt();
            self.signal[band] = (1.0 - ratio).sqrt();
            bin += size;
        }
    }

    /// `spxco` from `spxcoexp`, `spxcomant` and `mstrspxco`
    /// (clause 3.6.3), with the final factor of 32 of clause 3.6.4.3.
    pub(crate) fn set(&mut self, band: usize, exp: u32, mant: u32, master: u32) {
        let temp = if exp == 15 {
            mant as f32 / 4.0
        } else {
            (mant + 4) as f32 / 8.0
        };
        self.coord[band] = temp * 32.0 * (-((exp + 3 * master) as f32)).exp2();
    }
}

/// Synthesize one channel's coefficients from `range.begin_bin()` to
/// `range.end_bin()` (clause 3.6.4). `noise` returns zero-mean,
/// unit-variance samples.
pub(crate) fn synthesize(
    tc: &mut [f32; 256],
    range: &Range,
    coords: &Coords,
    atten: Option<u8>,
    mut noise: impl FnMut() -> f32,
) {
    let copy_start = range.copy_start();
    let copy_end = range.begin_bin();
    let sizes = range.bands.sizes();
    let mut wrap = [false; MAX_BANDS];
    let mut copy = copy_start;
    let mut insert = copy_end;
    for (band, &size) in sizes.iter().enumerate() {
        if copy + size > copy_end {
            copy = copy_start;
            wrap[band] = true;
        }
        for _ in 0..size {
            if copy == copy_end {
                copy = copy_start;
            }
            if let (Some(&value), true) = (tc.get(copy), insert < tc.len()) {
                tc[insert] = value;
            }
            insert += 1;
            copy += 1;
        }
    }

    let mut rms = [0.0f32; MAX_BANDS];
    let mut bin = copy_end;
    for (band, &size) in sizes.iter().enumerate() {
        let energy: f32 = tc[bin..bin + size].iter().map(|v| v * v).sum();
        rms[band] = (energy / size as f32).sqrt();
        bin += size;
    }

    if let Some(code) = atten {
        let gains = SPX_ATTEN[usize::from(code & 31)];
        let filter = [gains[0], gains[1], gains[2], gains[1], gains[0]];
        let mut border = copy_end;
        for (band, &size) in sizes.iter().enumerate() {
            if band == 0 || wrap[band] {
                for (k, gain) in filter.iter().enumerate() {
                    if let Some(value) = tc.get_mut(border + k - 2) {
                        *value *= *gain as f32;
                    }
                }
            }
            border += size;
        }
    }

    let mut bin = copy_end;
    for (band, &size) in sizes.iter().enumerate() {
        let nscale = rms[band] * coords.noise[band];
        let sscale = coords.signal[band];
        let coord = coords.coord[band];
        for value in &mut tc[bin..bin + size] {
            *value = (*value * sscale + noise() * nscale) * coord;
        }
        bin += size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{DEFAULT_CPL_BANDS, DEFAULT_SPX_BANDS};

    #[test]
    fn subbands_follow_table_e3_13() {
        assert_eq!(begin_subband(0), 2);
        assert_eq!(begin_subband(5), 7);
        assert_eq!(begin_subband(6), 9);
        assert_eq!(begin_subband(7), 11);
        assert_eq!(end_subband(0), 5);
        assert_eq!(end_subband(2), 7);
        assert_eq!(end_subband(3), 9);
        assert_eq!(end_subband(7), 17);
        assert_eq!(band_bin(17), 229);
    }

    #[test]
    fn default_bandings_merge_upper_subbands() {
        let cpl = Bands::new(&DEFAULT_CPL_BANDS, 0, 18);
        assert_eq!(cpl.sizes(), &[12, 12, 12, 12, 12, 12, 12, 24, 36, 72]);
        let spx = Bands::new(&DEFAULT_SPX_BANDS, 2, 17);
        assert_eq!(spx.sizes().iter().sum::<usize>(), 15 * 12);
        assert_eq!(spx.sizes()[0], 12);
        assert_eq!(Bands::new(&DEFAULT_SPX_BANDS, 4, 4).count, 0);
    }

    fn range(start: usize, begin: usize, end: usize) -> Range {
        Range {
            start,
            begin,
            end,
            bands: Bands::new(&[false; 17], begin, end),
        }
    }

    #[test]
    fn translation_wraps_back_to_the_copy_start() {
        // Copy 37..61 into 61..109: two bands fit, the third wraps.
        let range = range(1, 3, 7);
        let mut tc = [0.0f32; 256];
        for (bin, value) in tc.iter_mut().enumerate().take(61) {
            *value = bin as f32;
        }
        let mut coords = Coords::default();
        for band in 0..4 {
            coords.signal[band] = 1.0;
            coords.coord[band] = 1.0;
        }
        synthesize(&mut tc, &range, &coords, None, || 0.0);
        assert_eq!(tc[61], 37.0);
        assert_eq!(tc[84], 60.0);
        assert_eq!(tc[85], 37.0);
        assert_eq!(tc[108], 60.0);
        assert_eq!(tc[109], 0.0);
    }

    #[test]
    fn noise_follows_band_energy_and_blend() {
        let range = range(0, 2, 3);
        let mut tc = [0.0f32; 256];
        for value in &mut tc[25..37] {
            *value = 0.5;
        }
        let mut coords = Coords::default();
        coords.noise[0] = 1.0;
        coords.coord[0] = 2.0;
        synthesize(&mut tc, &range, &coords, None, || 1.0);
        for &value in &tc[49..61] {
            assert!((value - 1.0).abs() < 1e-6, "{value}");
        }
    }

    #[test]
    fn notch_attenuates_the_border() {
        let range = range(0, 2, 3);
        let mut tc = [1.0f32; 256];
        let mut coords = Coords::default();
        coords.signal[0] = 1.0;
        coords.coord[0] = 1.0;
        synthesize(&mut tc, &range, &coords, Some(0), || 0.0);
        let gains = SPX_ATTEN[0];
        assert!((tc[47] - gains[0] as f32).abs() < 1e-6);
        assert!((tc[49] - gains[2] as f32).abs() < 1e-6);
        assert!((tc[51] - gains[0] as f32).abs() < 1e-6);
        assert_eq!(tc[52], 1.0);
    }

    #[test]
    fn coordinates_scale_by_exponent_and_master() {
        let mut coords = Coords::default();
        coords.set(0, 0, 0, 0);
        assert!((coords.coord[0] - 16.0).abs() < 1e-6);
        coords.set(1, 15, 2, 0);
        assert!((coords.coord[1] - 16.0 * (-15f32).exp2()).abs() < 1e-9);
        coords.set(2, 1, 3, 1);
        assert!((coords.coord[2] - 7.0 / 8.0 * 32.0 / 16.0).abs() < 1e-6);
    }
}
