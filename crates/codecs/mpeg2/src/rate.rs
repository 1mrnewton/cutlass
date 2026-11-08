//! Rate control: Test Model 5 bit allocation with a VBV buffer model.
//!
//! Each GOP gets the bits the rate allows. Pictures split that budget by
//! the complexity (bits times quantiser) their type showed last time.
//! Inside a picture, a virtual buffer turns the bits spent so far into a
//! quantiser. TM5's third step, scaling that quantiser by how busy each
//! macroblock is, is left out: it costs about 1 dB of PSNR at equal rate.
//! The VBV model (Annex C) caps every picture so the decoder buffer never
//! underflows, and at a constant rate pads pictures so it never overflows.

use crate::headers::{PICTURE_I, PICTURE_P};

/// Bits per second, buffer size, and whether the rate is constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RateConfig {
    /// Average target.
    pub(crate) bitrate: f64,
    /// Rate the decoder buffer fills at; the header's `bit_rate`.
    pub(crate) peak: f64,
    pub(crate) vbv_bits: f64,
    pub(crate) frame_rate: f64,
    pub(crate) constant: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Rate {
    config: RateConfig,
    /// Complexity of I, P, and B pictures.
    complexity: [f64; 3],
    /// Virtual buffer fullness for I, P, and B pictures.
    virtual_buffer: [f64; 3],
    /// Bits left for the current GOP.
    remaining: f64,
    /// P and B pictures left in the current GOP.
    left: [u32; 2],
    /// Decoder buffer occupancy just before the next picture is removed.
    vbv: f64,
}

/// Per-picture state while its macroblocks are coded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PictureRate {
    kind: usize,
    target: f64,
    start: f64,
    reaction: f64,
    macroblocks: f64,
    /// Added to every quantiser code, after a VBV retry.
    pub(crate) offset: i32,
    quant_sum: f64,
    quant_count: f64,
}

fn kind(picture_type: u8) -> usize {
    match picture_type {
        PICTURE_I => 0,
        PICTURE_P => 1,
        _ => 2,
    }
}

const K_P: f64 = 1.0;
const K_B: f64 = 1.4;

impl Rate {
    pub(crate) fn new(config: RateConfig) -> Self {
        let reaction = 2.0 * config.bitrate / config.frame_rate;
        let d0 = 10.0 * reaction / 31.0;
        Self {
            config,
            complexity: [
                160.0 * config.bitrate / 115.0,
                60.0 * config.bitrate / 115.0,
                42.0 * config.bitrate / 115.0,
            ],
            virtual_buffer: [d0, K_P * d0, K_B * d0],
            remaining: 0.0,
            left: [0, 0],
            vbv: config.vbv_bits,
        }
    }

    /// Opens a GOP of `p` P pictures and `b` B pictures after its I picture.
    pub(crate) fn start_gop(&mut self, p: u32, b: u32) {
        let frames = f64::from(1 + p + b);
        self.remaining += self.config.bitrate * frames / self.config.frame_rate;
        self.left = [p, b];
    }

    /// Target bits for the next picture of `picture_type`.
    fn target(&self, kind: usize) -> f64 {
        let [xi, xp, xb] = self.complexity;
        let (np, nb) = (f64::from(self.left[0]), f64::from(self.left[1]));
        let r = self.remaining.max(0.0);
        let floor = self.config.bitrate / (8.0 * self.config.frame_rate);
        // A picture past its GOP's count still shares with itself.
        let t = match kind {
            0 => r / (1.0 + np * xp / (xi * K_P) + nb * xb / (xi * K_B)),
            1 => r / (np.max(1.0) + nb * K_P * xb / (K_B * xp)),
            _ => r / (nb.max(1.0) + np * K_B * xp / (K_P * xb)),
        };
        // Never ask for more than the decoder buffer holds.
        t.max(floor).min(0.9 * self.vbv)
    }

    /// Starts a picture of `macroblocks` macroblocks.
    pub(crate) fn begin(&self, picture_type: u8, macroblocks: usize) -> PictureRate {
        let kind = kind(picture_type);
        PictureRate {
            kind,
            target: self.target(kind),
            start: self.virtual_buffer[kind],
            reaction: 2.0 * self.config.bitrate / self.config.frame_rate,
            macroblocks: macroblocks.max(1) as f64,
            offset: 0,
            quant_sum: 0.0,
            quant_count: 0.0,
        }
    }

    /// Largest picture the decoder buffer can take now.
    pub(crate) fn max_bits(&self) -> f64 {
        self.vbv
    }

    /// Smallest picture that keeps a constant-rate buffer from overflowing.
    pub(crate) fn min_bits(&self) -> f64 {
        if self.config.constant {
            (self.vbv + self.config.bitrate / self.config.frame_rate - self.config.vbv_bits)
                .max(0.0)
        } else {
            0.0
        }
    }

    /// `vbv_delay` for the next picture, in 90 kHz ticks.
    pub(crate) fn vbv_delay(&self) -> u16 {
        if !self.config.constant {
            return 0xffff;
        }
        let ticks = (90_000.0 * self.vbv / self.config.bitrate).round();
        ticks.clamp(0.0, 65_534.0) as u16
    }

    /// Records a coded picture of `bits` bits.
    pub(crate) fn end(&mut self, picture: &PictureRate, bits: f64) {
        let kind = picture.kind;
        let average_q = if picture.quant_count > 0.0 {
            picture.quant_sum / picture.quant_count
        } else {
            16.0
        };
        self.complexity[kind] = (bits * average_q).max(1.0);
        self.virtual_buffer[kind] = (picture.start + bits - picture.target).max(0.0);
        self.remaining -= bits;
        match kind {
            1 => self.left[0] = self.left[0].saturating_sub(1),
            2 => self.left[1] = self.left[1].saturating_sub(1),
            _ => {}
        }
        let fill = if self.config.constant {
            self.config.bitrate
        } else {
            self.config.peak
        } / self.config.frame_rate;
        self.vbv = (self.vbv - bits + fill).min(self.config.vbv_bits);
    }

    /// Back to the state after `new`, keeping the configuration.
    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.config);
    }
}

impl PictureRate {
    /// Quantiser code (1 to 31) for macroblock `index`, `spent` bits into
    /// the picture.
    pub(crate) fn quant(&mut self, index: usize, spent: f64) -> u32 {
        let fullness = self.start + spent - self.target * index as f64 / self.macroblocks;
        let base = (fullness * 31.0 / self.reaction).max(1.0);
        let code = (base.round() as i32 + self.offset).clamp(1, 31);
        self.quant_sum += f64::from(code);
        self.quant_count += 1.0;
        code as u32
    }

    /// The quantiser the first macroblock would get, for the search weight.
    pub(crate) fn base_quant(&self) -> u32 {
        let base = (self.start * 31.0 / self.reaction).round() as i32 + self.offset;
        base.clamp(1, 31) as u32
    }

    #[cfg(test)]
    pub(crate) fn target(&self) -> f64 {
        self.target
    }

    /// Clears the per-macroblock tallies before a retry.
    pub(crate) fn restart(&mut self) {
        self.quant_sum = 0.0;
        self.quant_count = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::PICTURE_B;

    fn config(constant: bool) -> RateConfig {
        RateConfig {
            bitrate: 4_000_000.0,
            peak: 6_000_000.0,
            vbv_bits: 1_835_008.0,
            frame_rate: 25.0,
            constant,
        }
    }

    #[test]
    fn intra_pictures_get_more_bits_than_b_pictures() {
        let mut rate = Rate::new(config(false));
        rate.start_gop(3, 8);
        let i = rate.begin(PICTURE_I, 100).target();
        let b = rate.begin(PICTURE_B, 100).target();
        assert!(i > 2.0 * b, "{i} vs {b}");
    }

    #[test]
    fn a_full_buffer_raises_the_quantiser() {
        let rate = Rate::new(config(false));
        let mut picture = rate.begin(PICTURE_P, 100);
        let early = picture.quant(10, 0.0);
        let late = picture.quant(10, 400_000.0);
        assert!(late > early, "{early} {late}");
        // Spending under the target lowers it again.
        assert!(picture.quant(99, 0.0) < early);
        picture.offset = 31;
        assert_eq!(picture.quant(99, 0.0), 31);
    }

    #[test]
    fn pictures_past_the_gop_count_keep_a_bounded_target() {
        let mut rate = Rate::new(config(false));
        rate.start_gop(0, 0);
        let i = rate.begin(PICTURE_I, 100);
        rate.end(&i, 100_000.0);
        // No P picture was planned; one more still gets a share, not the
        // whole decoder buffer.
        let p = rate.begin(PICTURE_P, 100).target();
        assert!(p.is_finite() && p <= 160_000.0 * 1.01, "{p}");
    }

    #[test]
    fn the_buffer_drains_and_refills() {
        let mut rate = Rate::new(config(true));
        rate.start_gop(0, 0);
        let full = rate.max_bits();
        let picture = rate.begin(PICTURE_I, 100);
        rate.end(&picture, 1_000_000.0);
        assert!((rate.max_bits() - (full - 1_000_000.0 + 160_000.0)).abs() < 1.0);
        // Constant rate: a tiny picture would overflow, so stuffing is asked for.
        assert!(rate.min_bits() < rate.max_bits());
        let mut rate = Rate::new(config(true));
        assert!(rate.min_bits() > 0.0);
        rate.start_gop(0, 0);
        assert!(rate.vbv_delay() > 0 && rate.vbv_delay() < 0xffff);
        assert_eq!(Rate::new(config(false)).vbv_delay(), 0xffff);
    }
}
