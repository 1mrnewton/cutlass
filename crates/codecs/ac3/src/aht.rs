//! Adaptive hybrid transform mantissas (A/52:2018 Annex E clause 3.4): the
//! vector and gain adaptive quantizers, and the six-point inverse DCT that
//! spreads one frame of AHT mantissas over its six blocks.

use std::f32::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::tables::vq::{
    VQ_HEBAP1, VQ_HEBAP2, VQ_HEBAP3, VQ_HEBAP4, VQ_HEBAP5, VQ_HEBAP6, VQ_HEBAP7,
};
use crate::tables::{GAQ_A1, GAQ_A2, GAQ_A4, GAQ_B2, GAQ_B4, HEBAP_BITS};

/// One bin's six mantissas, one per block, before the exponent.
pub(crate) type Premantissas = [[f32; 6]; 256];

/// The six mantissas of VQ index `index` at `hebap` 1 to 7.
fn vq(hebap: u8, index: usize) -> Option<&'static [i16; 6]> {
    match hebap {
        1 => VQ_HEBAP1.get(index),
        2 => VQ_HEBAP2.get(index),
        3 => VQ_HEBAP3.get(index),
        4 => VQ_HEBAP4.get(index),
        5 => VQ_HEBAP5.get(index),
        6 => VQ_HEBAP6.get(index),
        7 => VQ_HEBAP7.get(index),
        _ => None,
    }
}

fn fraction(value: u16) -> f32 {
    f32::from(value as i16) / 32768.0
}

/// Read one channel's AHT mantissas (`chgaqmod`, the gains, and the
/// mantissas of `hebap` above zero) for bins `start..end`, then invert the
/// DCT. Bins with `hebap` 0 come back as zeros and `zero[bin]` set, so the
/// caller can dither them block by block.
pub(crate) fn read(
    r: &mut BitReader<'_>,
    hebap: &[u8; 256],
    start: usize,
    end: usize,
    out: &mut Premantissas,
    zero: &mut [bool; 256],
) -> Result<u32, CodecError> {
    let gaqmod = r.read(2, "gaqmod")?;
    let end_bap = if gaqmod < 2 { 12 } else { 17 };
    let gained = |bap: u8| gaqmod != 0 && (8..end_bap).contains(&bap);

    // log2 of each gain, in ascending bin order of the gained bins.
    let mut gains = [0u8; 256];
    let mut count = 0;
    match gaqmod {
        1 | 2 => {
            for &bap in &hebap[start..end] {
                if gained(bap) {
                    gains[count] = (r.read(1, "gaqgain")? as u8) << (gaqmod - 1);
                    count += 1;
                }
            }
        }
        3 => {
            let wanted = hebap[start..end].iter().filter(|&&b| gained(b)).count();
            while count < wanted {
                let at = r.pos() as u64;
                let code = r.read(5, "gaqgain")?;
                if code > 26 {
                    return Err(CodecError::Malformed {
                        at_bit: at,
                        what: "gaqgain group",
                    });
                }
                for m in [code / 9, (code % 9) / 3, code % 3] {
                    if count < wanted {
                        gains[count] = m as u8;
                        count += 1;
                    }
                }
            }
        }
        _ => {}
    }

    let mut next_gain = 0;
    for bin in start..end {
        let bap = hebap[bin];
        let slot = &mut out[bin];
        zero[bin] = bap == 0;
        if bap == 0 {
            *slot = [0.0; 6];
            continue;
        }
        let bits = HEBAP_BITS[usize::from(bap.min(19))];
        if bap < 8 {
            let at = r.pos() as u64;
            let index = r.read(bits, "vq index")? as usize;
            let Some(vector) = vq(bap, index) else {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "vq index",
                });
            };
            for (value, &q) in slot.iter_mut().zip(vector) {
                *value = f32::from(q) / 32768.0;
            }
        } else {
            let log_gain = if gained(bap) {
                let g = gains[next_gain];
                next_gain += 1;
                g
            } else {
                0
            };
            read_gaq(r, bap, bits, log_gain, slot)?;
        }
        idct6(slot);
    }
    Ok(gaqmod)
}

/// Six GAQ mantissas of one bin (Table E3.5 and E3.6).
fn read_gaq(
    r: &mut BitReader<'_>,
    hebap: u8,
    bits: u32,
    log_gain: u8,
    slot: &mut [f32; 6],
) -> Result<(), CodecError> {
    let row = usize::from(hebap - 8);
    for value in slot.iter_mut() {
        if log_gain == 0 {
            let q = r.signed(bits, "gaq mantissa")?;
            let x = q as f32 / (1u32 << (bits - 1)) as f32;
            *value = x + fraction(GAQ_A1[row] as u16) * x;
            continue;
        }
        let small = bits - u32::from(log_gain);
        let q = r.signed(small, "gaq mantissa")?;
        if q != -(1 << (small - 1)) {
            *value = q as f32 / (1u32 << (bits - 1)) as f32;
            continue;
        }
        let large = bits - (2 - u32::from(log_gain));
        let q = r.signed(large, "gaq large mantissa")?;
        let x = q as f32 / (1u32 << (large - 1)) as f32;
        let (a, b) = if log_gain == 1 {
            (GAQ_A2[row], if x >= 0.0 { 0x4000 } else { GAQ_B2[row] })
        } else {
            (GAQ_A4[row], if x >= 0.0 { 0x2000 } else { GAQ_B4[row] })
        };
        *value = x + fraction(a) * x + fraction(b);
    }
    Ok(())
}

/// `C(m) = sqrt(2) * sum_j R(j) X(j) cos(pi (2m + 1) j / 12)`, with
/// `R(0) = 1/sqrt(2)` and `R(j) = 1` above it (clause 3.4.5).
pub(crate) fn idct6(x: &mut [f32; 6]) {
    let mut out = [0.0f32; 6];
    for (m, value) in out.iter_mut().enumerate() {
        let mut sum = x[0] * FRAC_1_SQRT_2;
        for (j, &xj) in x.iter().enumerate().skip(1) {
            sum += xj * (PI * ((2 * m + 1) * j) as f32 / 12.0).cos();
        }
        *value = SQRT_2 * sum;
    }
    *x = out;
}

/// The forward DCT: the transpose of `idct6`, scaled so `idct6(dct6(x)) == x`.
pub(crate) fn dct6(c: &mut [f32; 6]) {
    let mut out = [0.0f32; 6];
    for (j, value) in out.iter_mut().enumerate() {
        let r = if j == 0 { FRAC_1_SQRT_2 } else { 1.0 };
        let mut sum = 0.0;
        for (m, &cm) in c.iter().enumerate() {
            sum += cm * (PI * ((2 * m + 1) * j) as f32 / 12.0).cos();
        }
        *value = SQRT_2 * r * sum / 6.0;
    }
    *c = out;
}

/// Bits of one channel's AHT mantissas, including the gain words when a
/// gain spends fewer bits than a fixed mantissa.
pub(crate) fn payload_bits(bap: &[u8], normalized: &[[f32; 6]], start: usize, end: usize) -> usize {
    chosen(bap, normalized, start, end).1
}

/// Write one channel's AHT mantissas from its six normalized blocks.
/// `normalized[bin][block]` is the mantissa before the inverse DCT.
/// `gaqmod` 1 or 2 is written when that gain is shorter than `gaqmod` 0
/// and no coefficient is more than one step further from the original.
pub(crate) fn write(
    w: &mut crate::bits::BitWriter,
    bap: &[u8],
    normalized: &[[f32; 6]],
    start: usize,
    end: usize,
) {
    let end = end.min(bap.len()).min(normalized.len());
    let (mode, _) = chosen(bap, normalized, start, end);
    w.put(2, u32::from(mode));
    if mode != 0 {
        for bin in start..end {
            let hebap = bap[bin];
            if (8..end_bap(mode)).contains(&hebap) {
                let choice = gain_choice(mode, hebap, &freq(&normalized[bin]));
                w.put(1, u32::from(choice.high));
            }
        }
    }
    for bin in start..end {
        let hebap = bap[bin];
        if hebap == 0 {
            continue;
        }
        let coeffs = freq(&normalized[bin]);
        let width = HEBAP_BITS[usize::from(hebap.min(19))];
        if hebap < 8 {
            w.put(width, closest_vq(hebap, &coeffs));
            continue;
        }
        if mode != 0 && (8..end_bap(mode)).contains(&hebap) {
            let choice = gain_choice(mode, hebap, &coeffs);
            if choice.high {
                for piece in choice.pieces {
                    write_piece(w, width, mode, piece);
                }
                continue;
            }
        }
        for &coeff in &coeffs {
            w.put(width, gaq(hebap, coeff) as u32);
        }
    }
}

/// `gaqmod` and its length. Mode 0 wins a tie, so a frame with nothing to
/// gain stays bit-identical to a fixed mantissa.
fn chosen(bap: &[u8], normalized: &[[f32; 6]], start: usize, end: usize) -> (u8, usize) {
    let end = end.min(bap.len()).min(normalized.len());
    let mut best_mode = 0u8;
    let mut best = channel_bits(0, bap, normalized, start, end);
    if !bap[start..end]
        .iter()
        .any(|&hebap| (8..17).contains(&hebap))
    {
        return (0, best);
    }
    for mode in [1u8, 2] {
        let bits = channel_bits(mode, bap, normalized, start, end);
        if bits < best {
            best = bits;
            best_mode = mode;
        }
    }
    (best_mode, best)
}

fn channel_bits(mode: u8, bap: &[u8], normalized: &[[f32; 6]], start: usize, end: usize) -> usize {
    let mut bits = 2usize;
    for bin in start..end {
        let hebap = bap[bin];
        if hebap == 0 {
            continue;
        }
        let width = HEBAP_BITS[usize::from(hebap.min(19))];
        if hebap < 8 {
            bits += width as usize;
            continue;
        }
        if mode == 0 || !(8..end_bap(mode)).contains(&hebap) {
            bits += 6 * width as usize;
            continue;
        }
        let choice = gain_choice(mode, hebap, &freq(&normalized[bin]));
        bits += 1 + choice.mant_bits(width, mode);
    }
    bits
}

fn end_bap(mode: u8) -> u8 {
    if mode < 2 { 12 } else { 17 }
}

fn freq(row: &[f32; 6]) -> [f32; 6] {
    let mut coeffs = *row;
    dct6(&mut coeffs);
    coeffs
}

#[derive(Clone, Copy)]
enum Piece {
    Full,
    Small(i32),
    Large(i32),
}

struct GainChoice {
    high: bool,
    pieces: [Piece; 6],
}

impl GainChoice {
    fn mant_bits(&self, width: u32, log_gain: u8) -> usize {
        if !self.high {
            return 6 * width as usize;
        }
        self.pieces
            .iter()
            .map(|piece| piece_bits(*piece, width, log_gain))
            .sum()
    }
}

fn piece_bits(piece: Piece, width: u32, log_gain: u8) -> usize {
    let small = width - u32::from(log_gain);
    let large = width - (2 - u32::from(log_gain));
    match piece {
        Piece::Full => width as usize,
        Piece::Small(_) => small as usize,
        Piece::Large(_) => (small + large) as usize,
    }
}

/// Gain 2 or 4 for one bin when every coefficient stays within one step of
/// the fixed quantizer and the six codes are shorter than six fixed mantissas.
fn gain_choice(log_gain: u8, hebap: u8, coeffs: &[f32; 6]) -> GainChoice {
    let width = HEBAP_BITS[usize::from(hebap.min(19))];
    let mut pieces = [Piece::Full; 6];
    let mut high_bits = 0usize;
    let mut ok = true;
    for (piece, &value) in pieces.iter_mut().zip(coeffs) {
        match quantize_high(hebap, log_gain, width, value) {
            Some((next, bits)) => {
                *piece = next;
                high_bits += bits;
            }
            None => {
                ok = false;
                break;
            }
        }
    }
    let high = ok && high_bits < 6 * width as usize;
    if !high {
        pieces = [Piece::Full; 6];
    }
    GainChoice { high, pieces }
}

fn quantize_high(hebap: u8, log_gain: u8, width: u32, value: f32) -> Option<(Piece, usize)> {
    let (_, hat0) = gk1(hebap, value);
    let err0 = (hat0 - value).abs();
    let step = 1.0 / (1u32 << (width - 1)) as f32;
    let small_bits = width - u32::from(log_gain);
    let mut best: Option<(Piece, usize)> = None;
    if let Some(code) = small_code(width, log_gain, value) {
        let hat = code as f32 / (1u32 << (width - 1)) as f32;
        if (hat - value).abs() <= err0 + step {
            best = Some((Piece::Small(code), small_bits as usize));
        }
    }
    if let Some((code, hat)) = large_code(hebap, log_gain, value) {
        let bits = piece_bits(Piece::Large(code), width, log_gain);
        if (hat - value).abs() <= err0 + step {
            let replace = best.is_none_or(|(_, so_far)| bits < so_far);
            if replace {
                best = Some((Piece::Large(code), bits));
            }
        }
    }
    best
}

fn small_code(width: u32, log_gain: u8, value: f32) -> Option<i32> {
    if !value.is_finite() {
        return None;
    }
    let small = width - u32::from(log_gain);
    let half = (1u32 << (width - 1)) as f32;
    let rounded = (value * half).round();
    if !rounded.is_finite() {
        return None;
    }
    let code = rounded as i32;
    let limit = 1i32 << (small - 1);
    (code > -limit && code < limit).then_some(code)
}

fn large_code(hebap: u8, log_gain: u8, value: f32) -> Option<(i32, f32)> {
    if !(8..17).contains(&hebap) || !value.is_finite() {
        return None;
    }
    let width = HEBAP_BITS[usize::from(hebap)];
    let large_bits = width - (2 - u32::from(log_gain));
    if large_bits == 0 || large_bits >= 31 {
        return None;
    }
    let half = 1i32 << (large_bits - 1);
    let mut best: Option<(i32, f32, f32)> = None;
    let consider = |code: i32, best: &mut Option<(i32, f32, f32)>| {
        if !(-half..half).contains(&code) {
            return;
        }
        let hat = recon_large(hebap, log_gain, code);
        let err = (hat - value).abs();
        let replace = best.is_none_or(|(_, _, so_far)| err < so_far);
        if replace {
            *best = Some((code, hat, err));
        }
    };
    if large_bits <= 8 {
        for code in -half..half {
            consider(code, &mut best);
        }
    } else {
        for positive in [true, false] {
            let (scale, offset) = large_remap(hebap, log_gain, positive);
            if scale == 0.0 {
                continue;
            }
            let guess = (((value - offset) / scale) * half as f32).round() as i32;
            for code in guess - 2..=guess + 2 {
                if (code >= 0) != positive {
                    continue;
                }
                consider(code, &mut best);
            }
        }
    }
    best.map(|(code, hat, _)| (code, hat))
}

fn recon_large(hebap: u8, log_gain: u8, code: i32) -> f32 {
    let width = HEBAP_BITS[usize::from(hebap)];
    let large_bits = width - (2 - u32::from(log_gain));
    let x = code as f32 / (1u32 << (large_bits - 1)) as f32;
    let (scale, offset) = large_remap(hebap, log_gain, x >= 0.0);
    x * scale + offset
}

fn large_remap(hebap: u8, log_gain: u8, positive: bool) -> (f32, f32) {
    let row = usize::from(hebap - 8);
    let (a, b) = if log_gain == 1 {
        let offset = if positive { 0x4000 } else { GAQ_B2[row] };
        (GAQ_A2[row], offset)
    } else {
        let offset = if positive { 0x2000 } else { GAQ_B4[row] };
        (GAQ_A4[row], offset)
    };
    (1.0 + fraction(a), fraction(b))
}

fn write_piece(w: &mut crate::bits::BitWriter, width: u32, log_gain: u8, piece: Piece) {
    match piece {
        Piece::Full => {}
        Piece::Small(code) => w.put(width - u32::from(log_gain), code as u32),
        Piece::Large(code) => {
            let small = width - u32::from(log_gain);
            let tag = -1i32 << (small - 1);
            w.put(small, tag as u32);
            let large = width - (2 - u32::from(log_gain));
            w.put(large, code as u32);
        }
    }
}

/// Fixed-gain code and the value the decoder rebuilds from it.
fn gk1(hebap: u8, value: f32) -> (i32, f32) {
    let code = gaq(hebap, value);
    let bits = HEBAP_BITS[usize::from(hebap.min(19))];
    let x = code as f32 / (1u32 << (bits - 1)) as f32;
    let scale = f32::from(GAQ_A1[usize::from(hebap.saturating_sub(8).min(11))]) / 32768.0;
    (code, x + scale * x)
}

fn closest_vq(hebap: u8, target: &[f32; 6]) -> u32 {
    let table: &[[i16; 6]] = match hebap {
        1 => &VQ_HEBAP1,
        2 => &VQ_HEBAP2,
        3 => &VQ_HEBAP3,
        4 => &VQ_HEBAP4,
        5 => &VQ_HEBAP5,
        6 => &VQ_HEBAP6,
        _ => &VQ_HEBAP7,
    };
    let mut best = 0u32;
    let mut error = f32::MAX;
    for (index, vector) in table.iter().enumerate() {
        let mut distance = 0.0f32;
        for (&wanted, &code) in target.iter().zip(vector) {
            let delta = wanted - f32::from(code) / 32768.0;
            distance += delta * delta;
        }
        if distance < error {
            error = distance;
            best = index as u32;
        }
    }
    best
}

fn gaq(hebap: u8, value: f32) -> i32 {
    let bits = HEBAP_BITS[usize::from(hebap.min(19))];
    let row = usize::from(hebap.saturating_sub(8).min(11));
    let scale = f32::from(GAQ_A1[row]) / 32768.0;
    let half = (1i32 << (bits - 1)) as f32;
    let x = value / (1.0 + scale);
    (x * half).round().clamp(-half, half - 1.0) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn dct_pair_reconstructs() {
        let original = [0.3, -0.2, 0.9, -0.7, 0.05, 0.4];
        let mut x = original;
        dct6(&mut x);
        idct6(&mut x);
        for (a, b) in x.iter().zip(original) {
            assert!((a - b).abs() < 1e-5, "{a} {b}");
        }
    }

    #[test]
    fn idct_of_dc_is_flat() {
        let mut x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        idct6(&mut x);
        for value in x {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    fn hebap_row(values: &[(usize, u8)]) -> [u8; 256] {
        let mut hebap = [0u8; 256];
        for &(bin, bap) in values {
            hebap[bin] = bap;
        }
        hebap
    }

    #[test]
    fn vq_reads_one_index_for_six_blocks() {
        let mut w = BitWriter::with_capacity(8);
        w.put(2, 0); // gaqmod 0
        w.put(2, 3); // hebap 1, index 3
        let bytes = w.finish();
        let hebap = hebap_row(&[(0, 1)]);
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        read(
            &mut BitReader::new(&bytes),
            &hebap,
            0,
            2,
            &mut out,
            &mut zero,
        )
        .unwrap();
        let mut expected = VQ_HEBAP1[3].map(|q| f32::from(q) / 32768.0);
        idct6(&mut expected);
        assert_eq!(out[0], expected);
        assert!(zero[1] && !zero[0]);
    }

    #[test]
    fn gaq_large_mantissa_follows_the_tag() {
        // hebap 8 is 3 bits; gain 2 makes small mantissas 2 bits and the
        // tag "10"; the large mantissa is then 2 bits.
        let mut w = BitWriter::with_capacity(8);
        w.put(2, 1); // gaqmod 1
        w.put(1, 1); // Gk = 2
        w.signed(2, 1); // small: 1/4
        w.signed(2, -2); // tag
        w.signed(2, 1); // large x = 0.5
        for _ in 0..4 {
            w.signed(2, 0);
        }
        let bytes = w.finish();
        let hebap = hebap_row(&[(0, 8)]);
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        let mut r = BitReader::new(&bytes);
        read(&mut r, &hebap, 0, 1, &mut out, &mut zero).unwrap();
        assert_eq!(r.pos(), 2 + 1 + 2 + 2 + 2 + 8);
        let mut pre = out[0];
        dct6(&mut pre);
        assert!((pre[0] - 0.25).abs() < 1e-5);
        let large = 0.5 + fraction(0xd555) * 0.5 + 0.5;
        assert!((pre[1] - large).abs() < 1e-5, "{}", pre[1]);
    }

    #[test]
    fn gaq_group_code_above_26_is_malformed() {
        let mut w = BitWriter::with_capacity(8);
        w.put(2, 3);
        w.put(5, 27);
        w.put(16, 0);
        let bytes = w.finish();
        let hebap = hebap_row(&[(0, 9)]);
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        let error = read(
            &mut BitReader::new(&bytes),
            &hebap,
            0,
            1,
            &mut out,
            &mut zero,
        );
        assert!(matches!(error, Err(CodecError::Malformed { .. })));
    }

    #[test]
    fn a_constant_mantissa_uses_a_shorter_gain() {
        let mut normalized = [[0.0f32; 6]; 256];
        normalized[0] = [0.45; 6];
        let mut bap = [0u8; 256];
        bap[0] = 8;
        let bits = payload_bits(&bap, &normalized, 0, 1);
        let fixed = 2 + 6 * 3;
        assert!(bits < fixed, "gain did not save bits: {bits}");
        let mut w = BitWriter::with_capacity(32);
        write(&mut w, &bap, &normalized, 0, 1);
        assert_eq!(w.bits(), bits);
        let bytes = w.finish();
        let mode = bytes[0] >> 6;
        assert!(mode == 1 || mode == 2, "gaqmod {mode}");
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        let mut r = BitReader::new(&bytes);
        read(&mut r, &bap, 0, 1, &mut out, &mut zero).unwrap();
        assert_eq!(r.pos(), bits);
        for (got, want) in out[0].iter().zip(normalized[0]) {
            assert!((got - want).abs() < 0.2, "{got} {want}");
        }
    }

    #[test]
    fn a_full_scale_bin_keeps_the_fixed_mantissa() {
        // Every DCT coefficient is outside the small-mantissa range, so a
        // gain would only add the gain bit and the large-mantissa tags.
        let mut spectrum = [0.8, -0.7, 0.65, -0.6, 0.55, -0.5];
        idct6(&mut spectrum);
        let mut normalized = [[0.0f32; 6]; 256];
        normalized[0] = spectrum;
        let mut bap = [0u8; 256];
        bap[0] = 8;
        let bits = payload_bits(&bap, &normalized, 0, 1);
        assert_eq!(bits, 2 + 6 * 3);
        let mut w = BitWriter::with_capacity(32);
        write(&mut w, &bap, &normalized, 0, 1);
        assert_eq!(w.bits(), bits);
        let bytes = w.finish();
        assert_eq!(bytes[0] >> 6, 0);
    }
}
