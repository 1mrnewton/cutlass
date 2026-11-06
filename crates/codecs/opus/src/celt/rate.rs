//! Bit allocation (RFC 6716 section 4.3.3, `rate.c`).
//!
//! All values are in 1/8 bit (`BITRES`). The allocation is a mandatory part
//! of the bitstream: the decoder must reproduce the encoder's split of the
//! budget exactly, so every operation here is integer and follows the
//! reference order.

use super::coder::Coder;
use super::tables::{
    BAND_ALLOCATION, CACHE_BITS, CACHE_CAPS, CACHE_INDEX, EBANDS, FINE_OFFSET, LOG2_FRAC_TABLE,
    LOGN, MAX_FINE_BITS, NB_ALLOC_VECTORS, NB_EBANDS,
};

pub(crate) const BITRES: i32 = 3;
const ALLOC_STEPS: i32 = 6;
const LOG_MAX_PSEUDO: usize = 6;

/// Pulses for pseudo-pulse index `i` (`get_pulses`).
#[inline]
pub(crate) fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// The pulse cache row for `band` at `lm` (which may be -1 inside a split).
fn cache(band: usize, lm: i32) -> Option<&'static [u8]> {
    let row = usize::try_from(lm + 1).ok()?;
    let index = *CACHE_INDEX.get(row * NB_EBANDS + band)?;
    let start = usize::try_from(index).ok()?;
    CACHE_BITS.get(start..)
}

/// The largest bit cost in the cache for `band` at `lm` plus nothing else:
/// `cache[cache[0]]`, or `None` without an entry.
pub(crate) fn cache_max_bits(band: usize, lm: i32) -> Option<i32> {
    let c = cache(band, lm)?;
    let last = usize::from(*c.first()?);
    c.get(last).map(|&b| i32::from(b))
}

/// The pseudo-pulse count whose cost is nearest `bits` (`bits2pulses`).
pub(crate) fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let Some(c) = cache(band, lm) else {
        return 0;
    };
    let at = |i: usize| i32::from(c.get(i).copied().unwrap_or(u8::MAX));
    let mut lo = 0usize;
    let mut hi = usize::from(c.first().copied().unwrap_or(0));
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if at(mid) >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 { -1 } else { at(lo) };
    if bits - lo_bits <= at(hi) - bits {
        lo as i32
    } else {
        hi as i32
    }
}

/// The bits pseudo-pulse count `pulses` costs (`pulses2bits`).
pub(crate) fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses <= 0 {
        return 0;
    }
    cache(band, lm)
        .and_then(|c| c.get(pulses as usize))
        .map_or(0, |&b| i32::from(b) + 1)
}

/// Per-band maximum allocation (`init_caps`).
pub(crate) fn init_caps(lm: usize, channels: usize) -> [i32; NB_EBANDS] {
    let mut cap = [0i32; NB_EBANDS];
    for (i, slot) in cap.iter_mut().enumerate() {
        let n = ((EBANDS[i + 1] - EBANDS[i]) << lm) as i32;
        let row = NB_EBANDS * (2 * lm + channels - 1);
        let c = i32::from(CACHE_CAPS.get(row + i).copied().unwrap_or(0));
        *slot = ((c + 64) * channels as i32 * n) >> 2;
    }
    cap
}

/// The result of [`compute_allocation`].
#[derive(Clone, Debug, Default)]
pub(crate) struct Allocation {
    /// PVQ bits per band.
    pub pulses: [i32; NB_EBANDS],
    /// Fine energy bits per band and channel.
    pub fine_quant: [i32; NB_EBANDS],
    pub fine_priority: [i32; NB_EBANDS],
    pub coded_bands: usize,
    pub balance: i32,
    pub intensity: usize,
    pub dual_stereo: bool,
}

/// Inputs to [`compute_allocation`] besides the coder.
pub(crate) struct AllocRequest<'r> {
    pub start: usize,
    pub end: usize,
    pub offsets: &'r [i32; NB_EBANDS],
    pub cap: &'r [i32; NB_EBANDS],
    pub alloc_trim: i32,
    /// The encoder's intensity choice; ignored when decoding.
    pub intensity: usize,
    /// The encoder's dual-stereo choice; ignored when decoding.
    pub dual_stereo: bool,
    pub total: i32,
    pub channels: usize,
    pub lm: usize,
    /// Coded bands of the previous frame, for the encoder's skip hysteresis.
    pub prev: usize,
}

/// Splits `total` 1/8 bits between PVQ and fine energy in each band, and
/// codes the skip, intensity, and dual-stereo decisions
/// (`compute_allocation`).
pub(crate) fn compute_allocation(req: &AllocRequest<'_>, coder: &mut Coder<'_, '_>) -> Allocation {
    let (start, end) = (req.start, req.end.min(NB_EBANDS));
    let c = req.channels as i32;
    let lm = req.lm as i32;
    let mut total = req.total.max(0);
    let len = NB_EBANDS;
    let mut skip_start = start;
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE.get(end - start).copied().unwrap_or(0);
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }
    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];
    for j in start..end {
        let width = (EBANDS[j + 1] - EBANDS[j]) as i32;
        thresh[j] = (c << BITRES).max(((3 * width) << lm << BITRES) >> 4);
        trim_offset[j] = (c
            * width
            * (req.alloc_trim - 5 - lm)
            * (end as i32 - j as i32 - 1)
            * (1 << (lm + BITRES)))
            >> 6;
        if width << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }
    let alloc = |row: usize, j: usize| -> i32 {
        let n = (EBANDS[j + 1] - EBANDS[j]) as i32;
        let a = i32::from(BAND_ALLOCATION.get(row * len + j).copied().unwrap_or(0));
        (c * n * a) << lm >> 2
    };
    let mut lo = 1i32;
    let mut hi = NB_ALLOC_VECTORS as i32 - 1;
    while lo <= hi {
        let mut done = false;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let mut bitsj = alloc(mid as usize, j);
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += req.offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(req.cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    hi = lo;
    lo -= 1;
    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    for j in start..end {
        let mut bits1j = alloc(lo as usize, j);
        let mut bits2j = if hi >= NB_ALLOC_VECTORS as i32 {
            req.cap[j]
        } else {
            alloc(hi as usize, j)
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j]).max(0);
        }
        if lo > 0 {
            bits1j += req.offsets[j];
        }
        bits2j += req.offsets[j];
        if req.offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }
    let interp = Interp {
        start,
        end,
        skip_start,
        bits1: &bits1,
        bits2: &bits2,
        thresh: &thresh,
        cap: req.cap,
        total,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        channels: c,
        lm,
        prev: req.prev,
    };
    interp_bits2pulses(&interp, req.intensity, req.dual_stereo, coder)
}

struct Interp<'r> {
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &'r [i32; NB_EBANDS],
    bits2: &'r [i32; NB_EBANDS],
    thresh: &'r [i32; NB_EBANDS],
    cap: &'r [i32; NB_EBANDS],
    total: i32,
    skip_rsv: i32,
    intensity_rsv: i32,
    dual_stereo_rsv: i32,
    channels: i32,
    lm: i32,
    prev: usize,
}

fn interp_bits2pulses(
    p: &Interp<'_>,
    enc_intensity: usize,
    enc_dual_stereo: bool,
    coder: &mut Coder<'_, '_>,
) -> Allocation {
    let (start, end) = (p.start, p.end);
    let c = p.channels;
    let lm = p.lm;
    let mut total = p.total;
    let mut intensity_rsv = p.intensity_rsv;
    let mut dual_stereo_rsv = p.dual_stereo_rsv;
    let alloc_floor = c << BITRES;
    let stereo = i32::from(c > 1);
    let log_m = lm << BITRES;
    let mut out = Allocation::default();
    let bits = &mut out.pulses;
    let ebits = &mut out.fine_quant;
    let fine_priority = &mut out.fine_priority;

    let mut lo = 0i32;
    let mut hi = 1i32 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = p.bits1[j] + ((mid * p.bits2[j]) >> ALLOC_STEPS);
            if tmp >= p.thresh[j] || done {
                done = true;
                psum += tmp.min(p.cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = p.bits1[j] + ((lo * p.bits2[j]) >> ALLOC_STEPS);
        if tmp < p.thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(p.cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    let width_from = |from: usize, to: usize| (EBANDS[to] - EBANDS[from]) as i32;
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        if j <= p.skip_start {
            total += p.skip_rsv;
            break;
        }
        let mut left = total - psum;
        let span = width_from(start, coded_bands);
        let percoeff = left / span;
        left -= span * percoeff;
        let rem = (left - width_from(start, j)).max(0);
        let band_width = width_from(j, coded_bands);
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= p.thresh[j].max(alloc_floor + (1 << BITRES)) {
            // The encoder keeps the band (codes a 1) unless it is thin
            // enough to fold, with hysteresis against the previous frame.
            let keep = if coder.encoding() {
                let depth = match (coded_bands > 17, j < p.prev) {
                    (false, _) => 0,
                    (true, true) => 7,
                    (true, false) => 9,
                };
                coded_bands <= start + 2 || band_bits > ((depth * band_width) << lm << BITRES) >> 4
            } else {
                false
            };
            if coder.bit_logp(keep, 1) {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE.get(j - start).copied().unwrap_or(0);
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }

    if intensity_rsv > 0 {
        let ft = (coded_bands + 1 - start) as u32;
        let want = enc_intensity.min(coded_bands).saturating_sub(start) as u32;
        out.intensity = start + coder.uint(want, ft) as usize;
    } else {
        out.intensity = 0;
    }
    if out.intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    out.dual_stereo = if dual_stereo_rsv > 0 {
        coder.bit_logp(enc_dual_stereo, 1)
    } else {
        false
    };

    let mut left = total - psum;
    let span = width_from(start, coded_bands);
    let percoeff = left / span;
    left -= span * percoeff;
    for (j, b) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        *b += percoeff * width_from(j, j + 1);
    }
    for (j, b) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        let tmp = left.min(width_from(j, j + 1));
        *b += tmp;
        left -= tmp;
    }

    let mut balance = 0i32;
    let mut j = start;
    while j < coded_bands {
        let n0 = width_from(j, j + 1);
        let n = n0 << lm;
        bits[j] += balance;
        let excess;
        if n > 1 {
            excess = (bits[j] - p.cap[j]).max(0);
            bits[j] -= excess;
            let den = c * n + i32::from(c == 2 && n > 2 && !out.dual_stereo && j < out.intensity);
            let nclogn = den * (LOGN[j] + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += (den << BITRES) >> 2;
            }
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclogn >> 3;
            }
            ebits[j] = ((bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES)).max(0);
            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            excess = (bits[j] - (c << BITRES)).max(0);
            bits[j] -= excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }
        let mut excess = excess;
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = i32::from(extra_bits >= excess - balance);
            excess -= extra_bits;
        }
        balance = excess;
        j += 1;
    }
    out.balance = balance;
    while j < end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = i32::from(ebits[j] < 1);
        j += 1;
    }
    out.coded_bands = coded_bands;
    out
}

#[cfg(test)]
mod tests {
    use super::{
        AllocRequest, BITRES, bits2pulses, compute_allocation, get_pulses, init_caps, pulses2bits,
    };
    use crate::celt::coder::Coder;
    use crate::celt::tables::{EBANDS, NB_EBANDS};
    use crate::range::RangeEncoder;

    #[test]
    fn pulse_cache_is_monotonic_and_inverts() {
        for lm in -1..=3i32 {
            for band in 0..NB_EBANDS {
                let mut last = 0;
                let max_q = super::cache(band, lm).map_or(0, |c| i32::from(c[0]));
                // A one-bin band costs its sign bit whatever the pulse count.
                if max_q > 1 && pulses2bits(band, lm, 1) == pulses2bits(band, lm, max_q) {
                    continue;
                }
                for q in 1..=max_q {
                    let b = pulses2bits(band, lm, q);
                    if b == 0 {
                        break;
                    }
                    // Costs never fall; neighbouring sizes may tie after
                    // rounding, and then either one is the inverse.
                    assert!(b >= last, "band {band} lm {lm} q {q}");
                    last = b;
                    assert_eq!(pulses2bits(band, lm, bits2pulses(band, lm, b)), b);
                }
            }
        }
        assert_eq!(get_pulses(7), 7);
        assert_eq!(get_pulses(8), 8);
        assert_eq!(get_pulses(15), 15);
        assert_eq!(get_pulses(16), 16);
        assert_eq!(get_pulses(39), 120);
    }

    #[test]
    fn allocation_never_exceeds_the_budget() {
        let mut seed = 99u32;
        for _ in 0..3000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let lm = (seed >> 3) as usize % 4;
            let channels = 1 + (seed >> 7) as usize % 2;
            let start = if (seed >> 9).is_multiple_of(4) { 17 } else { 0 };
            let end = [13usize, 17, 19, 21][(seed >> 11) as usize % 4].max(start + 1);
            let total = ((seed >> 14) % (1275 * 64)) as i32;
            let trim = ((seed >> 20) % 11) as i32;
            let cap = init_caps(lm, channels);
            let mut offsets = [0i32; NB_EBANDS];
            if seed & 1 == 0 {
                offsets[(seed >> 22) as usize % end.max(1)] = 8 << BITRES;
            }
            let mut enc = RangeEncoder::new(1275);
            let mut coder = Coder::Enc(&mut enc);
            let req = AllocRequest {
                start,
                end,
                offsets: &offsets,
                cap: &cap,
                alloc_trim: trim,
                intensity: end,
                dual_stereo: false,
                total,
                channels,
                lm,
                prev: end,
            };
            let a = compute_allocation(&req, &mut coder);
            let used: i32 = (start..end)
                .map(|j| a.pulses[j] + ((channels as i32 * a.fine_quant[j]) << BITRES))
                .sum();
            assert!(used <= total.max(0), "used {used} of {total}");
            assert!(a.coded_bands > start && a.coded_bands <= end);
            for (j, &c) in cap.iter().enumerate().take(end).skip(start) {
                assert!(a.pulses[j] >= 0 && a.fine_quant[j] >= 0 && a.fine_quant[j] <= 8);
                assert!(a.pulses[j] <= c, "cap");
            }
            let _ = EBANDS;
        }
    }
}
