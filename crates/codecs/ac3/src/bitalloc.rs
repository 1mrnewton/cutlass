//! The parametric bit allocation of A/52 section 7.2.2, with the Annex E
//! high-efficiency pointers (clause 3.4.3.1). Encoder and decoder run this
//! same integer routine, so their allocations agree bit for bit.

use cutlass_decode::CodecError;

use crate::tables::{
    BAPTAB, BNDTAB, DBPBTAB, FASTDEC, FASTGAIN, FLOORTAB, HEBAPTAB, HTH, LATAB, MASKTAB, SLOWDEC,
    SLOWGAIN,
};

/// Critical bands.
pub(crate) const BANDS: usize = 50;

/// Most delta bit allocation segments: `deltnseg` is 3 bits.
pub(crate) const MAX_SEGMENTS: usize = 8;

/// The five parameter codes `baie` sends, as table values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Params {
    pub sdecay: i32,
    pub fdecay: i32,
    pub sgain: i32,
    pub dbknee: i32,
    pub floor: i32,
}

impl Params {
    pub(crate) fn from_codes(
        sdcycod: u32,
        fdcycod: u32,
        sgaincod: u32,
        dbpbcod: u32,
        floorcod: u32,
    ) -> Self {
        Self {
            sdecay: SLOWDEC[(sdcycod & 3) as usize],
            fdecay: FASTDEC[(fdcycod & 3) as usize],
            sgain: SLOWGAIN[(sgaincod & 3) as usize],
            dbknee: DBPBTAB[(dbpbcod & 3) as usize],
            floor: FLOORTAB[(floorcod & 7) as usize],
        }
    }
}

/// One delta bit allocation segment (`deltoffst`, `deltlen`, `deltba`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Segment {
    pub offset: u8,
    pub len: u8,
    pub ba: u8,
}

/// Delta bit allocation for one channel: the segments in force.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Delta {
    pub segments: [Segment; MAX_SEGMENTS],
    pub count: usize,
}

impl Delta {
    pub(crate) fn segments(&self) -> &[Segment] {
        &self.segments[..self.count.min(MAX_SEGMENTS)]
    }
}

/// What one channel's allocation needs besides its exponents.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Channel<'a> {
    pub start: usize,
    pub end: usize,
    pub fscod: u8,
    pub params: Params,
    pub fgaincod: u32,
    /// `((csnroffst - 15) << 4) + fsnroffst`, before the shift by 2.
    pub snr: i32,
    /// `cplfleak` and `cplsleak` for the coupling channel.
    pub leak: Option<(u32, u32)>,
    pub delta: Option<&'a [Segment]>,
    /// Use `hebaptab` (an AHT channel) instead of `baptab`.
    pub high_efficiency: bool,
}

/// The masking curve per band, after the hearing threshold and the delta
/// bit allocation, and the PSD per bin. The encoder reuses both while it
/// searches SNR offsets.
#[derive(Clone, Debug)]
pub(crate) struct Mask {
    pub psd: [i32; 256],
    pub mask: [i32; BANDS],
    start: usize,
    end: usize,
}

fn logadd(a: i32, b: i32) -> i32 {
    let c = a - b;
    let address = ((c.abs() >> 1) as usize).min(255);
    if c >= 0 {
        a + i32::from(LATAB[address])
    } else {
        b + i32::from(LATAB[address])
    }
}

fn calc_lowcomp(a: i32, b0: i32, b1: i32, bin: usize) -> i32 {
    if bin < 7 {
        if b0 + 256 == b1 {
            384
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else if bin < 20 {
        if b0 + 256 == b1 {
            320
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else {
        (a - 128).max(0)
    }
}

fn band_end(band: usize) -> usize {
    usize::from(BNDTAB[band + 1])
}

/// Sections 7.2.2.2 to 7.2.2.6 for one channel.
pub(crate) fn mask(exps: &[u8; 256], ch: &Channel<'_>) -> Result<Mask, CodecError> {
    let (start, end) = (ch.start, ch.end);
    let mut out = Mask {
        psd: [0; 256],
        mask: [0; BANDS],
        start,
        end,
    };
    if start >= end || end > 253 {
        return Ok(out);
    }
    for (psd, exp) in out.psd[start..end].iter_mut().zip(&exps[start..end]) {
        *psd = 3072 - (i32::from(*exp) << 7);
    }
    let mut bndpsd = [0i32; BANDS];
    let mut j = start;
    let mut k = usize::from(MASKTAB[start]);
    loop {
        let lastbin = band_end(k).min(end);
        bndpsd[k] = out.psd[j];
        j += 1;
        while j < lastbin {
            bndpsd[k] = logadd(bndpsd[k], out.psd[j]);
            j += 1;
        }
        k += 1;
        if end <= lastbin {
            break;
        }
    }
    let bndstrt = usize::from(MASKTAB[start]);
    let bndend = usize::from(MASKTAB[end - 1]) + 1;
    let fgain = FASTGAIN[(ch.fgaincod & 7) as usize];
    let Params {
        sdecay,
        fdecay,
        sgain,
        dbknee,
        ..
    } = ch.params;
    let mut excite = [0i32; BANDS];
    let mut begin;
    let mut fastleak = 0;
    let mut slowleak = 0;
    let lfe_tail = |bin: usize| bndend == 7 && bin == 6;
    if bndstrt == 0 {
        let mut lowcomp = calc_lowcomp(0, bndpsd[0], bndpsd[1], 0);
        excite[0] = bndpsd[0] - fgain - lowcomp;
        lowcomp = calc_lowcomp(lowcomp, bndpsd[1], bndpsd[2], 1);
        excite[1] = bndpsd[1] - fgain - lowcomp;
        begin = 7;
        for bin in 2..7 {
            if !lfe_tail(bin) {
                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
            }
            fastleak = bndpsd[bin] - fgain;
            slowleak = bndpsd[bin] - sgain;
            excite[bin] = fastleak - lowcomp;
            if !lfe_tail(bin) && bndpsd[bin] <= bndpsd[bin + 1] {
                begin = bin + 1;
                break;
            }
        }
        for bin in begin..bndend.min(22) {
            if !lfe_tail(bin) {
                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
            }
            fastleak -= fdecay;
            fastleak = fastleak.max(bndpsd[bin] - fgain);
            slowleak -= sdecay;
            slowleak = slowleak.max(bndpsd[bin] - sgain);
            excite[bin] = (fastleak - lowcomp).max(slowleak);
        }
        begin = 22;
    } else {
        begin = bndstrt;
        if let Some((fast, slow)) = ch.leak {
            fastleak = ((fast as i32) << 8) + 768;
            slowleak = ((slow as i32) << 8) + 768;
        }
    }
    for bin in begin..bndend {
        fastleak -= fdecay;
        fastleak = fastleak.max(bndpsd[bin] - fgain);
        slowleak -= sdecay;
        slowleak = slowleak.max(bndpsd[bin] - sgain);
        excite[bin] = fastleak.max(slowleak);
    }
    let hth = &HTH[usize::from(ch.fscod.min(2))];
    for bin in bndstrt..bndend {
        if bndpsd[bin] < dbknee {
            excite[bin] += (dbknee - bndpsd[bin]) >> 2;
        }
        out.mask[bin] = excite[bin].max(i32::from(hth[bin]));
    }
    if let Some(segments) = ch.delta {
        let mut band = 0usize;
        for segment in segments {
            band += usize::from(segment.offset);
            let delta = if segment.ba >= 4 {
                (i32::from(segment.ba) - 3) << 7
            } else {
                (i32::from(segment.ba) - 4) << 7
            };
            let len = usize::from(segment.len);
            if band + len > BANDS {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "delta bit allocation band",
                });
            }
            for value in &mut out.mask[band..band + len] {
                *value += delta;
            }
            band += len;
        }
    }
    Ok(out)
}

/// Section 7.2.2.7: pointers for one SNR offset. `snr` is the unshifted
/// `((csnroffst - 15) << 4) + fsnroffst`.
pub(crate) fn pointers(mask: &Mask, ch: &Channel<'_>, snr: i32, bap: &mut [u8; 256]) {
    let (start, end) = (mask.start, mask.end);
    if start >= end {
        return;
    }
    let snroffset = snr << 2;
    let floor = ch.params.floor;
    let table: &[u8; 64] = if ch.high_efficiency {
        &HEBAPTAB
    } else {
        &BAPTAB
    };
    let mut i = start;
    let mut j = usize::from(MASKTAB[start]);
    loop {
        let lastbin = band_end(j).min(end);
        let mut m = mask.mask[j] - snroffset - floor;
        if m < 0 {
            m = 0;
        }
        m &= 0x1fe0;
        m += floor;
        while i < lastbin {
            let address = ((mask.psd[i] - m) >> 5).clamp(0, 63) as usize;
            bap[i] = table[address];
            i += 1;
        }
        j += 1;
        if end <= lastbin {
            break;
        }
    }
}

/// The whole routine for one channel.
pub(crate) fn allocate(
    exps: &[u8; 256],
    ch: &Channel<'_>,
    bap: &mut [u8; 256],
) -> Result<(), CodecError> {
    let mask = mask(exps, ch)?;
    pointers(&mask, ch, ch.snr, bap);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> Params {
        Params::from_codes(2, 1, 1, 2, 4)
    }

    fn channel(end: usize, snr: i32) -> Channel<'static> {
        Channel {
            start: 0,
            end,
            fscod: 0,
            params: default_params(),
            fgaincod: 4,
            snr,
            leak: None,
            delta: None,
            high_efficiency: false,
        }
    }

    #[test]
    fn a_louder_offset_never_takes_bits_away() {
        let mut exps = [24u8; 256];
        for (bin, value) in exps.iter_mut().enumerate().take(253) {
            *value = ((bin / 9) % 12) as u8 + 2;
        }
        let mut previous = [0u8; 256];
        for csnr in [0, 10, 15, 25, 40, 63] {
            let mut bap = [0u8; 256];
            allocate(&exps, &channel(253, ((csnr - 15) << 4) + 8), &mut bap).unwrap();
            for bin in 0..253 {
                assert!(bap[bin] >= previous[bin], "csnr {csnr} bin {bin}");
                assert!(bap[bin] <= 15);
            }
            previous = bap;
        }
        assert!(previous.iter().take(253).all(|&value| value > 0));
    }

    #[test]
    fn a_silent_spectrum_gets_no_bits_at_a_low_offset() {
        let exps = [24u8; 256];
        let mut bap = [9u8; 256];
        allocate(&exps, &channel(253, (5 - 15) << 4), &mut bap).unwrap();
        assert!(bap.iter().take(253).all(|&value| value == 0));
        assert_eq!(bap[253], 9, "bins past the end stay untouched");
    }

    #[test]
    fn a_delta_segment_moves_the_mask_by_six_db_steps() {
        let mut exps = [8u8; 256];
        exps[..7].fill(4);
        let base = channel(253, 0);
        let plain = mask(&exps, &base).unwrap();
        let segments = [Segment {
            offset: 10,
            len: 3,
            ba: 6,
        }];
        let shifted = mask(
            &exps,
            &Channel {
                delta: Some(&segments),
                ..base
            },
        )
        .unwrap();
        for band in 0..BANDS {
            let expected = if (10..13).contains(&band) { 3 << 7 } else { 0 };
            assert_eq!(
                shifted.mask[band] - plain.mask[band],
                expected,
                "band {band}"
            );
        }
        let past = [Segment {
            offset: 49,
            len: 2,
            ba: 0,
        }];
        assert!(
            mask(
                &exps,
                &Channel {
                    delta: Some(&past),
                    ..base
                }
            )
            .is_err()
        );
    }

    #[test]
    fn the_lfe_channel_stops_at_band_seven() {
        let mut exps = [24u8; 256];
        exps[..7].copy_from_slice(&[3, 4, 5, 5, 6, 7, 7]);
        let mut bap = [0u8; 256];
        let lfe = Channel {
            end: 7,
            ..channel(7, (30 - 15) << 4)
        };
        allocate(&exps, &lfe, &mut bap).unwrap();
        assert!(bap[..7].iter().all(|&value| value > 0));
        assert!(bap[7..].iter().all(|&value| value == 0));
    }

    #[test]
    fn high_efficiency_pointers_reach_past_fifteen() {
        let exps = [0u8; 256];
        let mut bap = [0u8; 256];
        let ch = Channel {
            high_efficiency: true,
            ..channel(253, (63 - 15) << 4)
        };
        allocate(&exps, &ch, &mut bap).unwrap();
        assert!(bap[..253].iter().any(|&value| value > 15));
        assert!(bap[..253].iter().all(|&value| value <= 19));
    }
}
