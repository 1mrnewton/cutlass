//! The excitation pulses of one SILK frame: rate level, pulse counts per
//! 16-sample block, the shell coder split tree, LSBs, and signs
//! (RFC 6716 section 4.2.7.8; reference `decode_pulses.c`,
//! `shell_coder.c`, `code_signs.c`).

use super::tables::{
    LSB_ICDF, PULSES_PER_BLOCK_ICDF, RATE_LEVELS_ICDF, SHELL_CODE_TABLE_OFFSETS, SHELL_CODE_TABLE0,
    SHELL_CODE_TABLE1, SHELL_CODE_TABLE2, SHELL_CODE_TABLE3, SIGN_ICDF,
};
use crate::range::RangeDecoder;

/// Samples per shell block.
const SHELL_BLOCK: usize = 16;
/// `MAX_PULSES`: the largest pulse count coded directly for a block.
const MAX_PULSES: usize = 16;
/// Longest frame: 20 ms at 16 kHz.
pub(crate) const MAX_FRAME_LENGTH: usize = 320;
/// Shell blocks in the longest frame.
const MAX_BLOCKS: usize = MAX_FRAME_LENGTH / SHELL_BLOCK;
/// Most LSB levels a block can carry: after 10 the table cannot escape.
const MAX_LSHIFTS: usize = 10;

/// `decode_split`: splits `p` pulses between two halves.
fn decode_split(rd: &mut RangeDecoder<'_>, p: i32, table: &[u8]) -> (i32, i32) {
    if p <= 0 {
        return (0, 0);
    }
    let offset = SHELL_CODE_TABLE_OFFSETS
        .get(p as usize)
        .map_or(usize::MAX, |&o| usize::from(o));
    let left = rd.dec_icdf(table.get(offset..).unwrap_or(&[0]), 8) as i32;
    (left, p - left)
}

/// `silk_shell_decoder`: the 16 pulse magnitudes of one block holding
/// `pulses4` pulses, by recursive binary splits.
fn shell_decoder(out: &mut [i32], rd: &mut RangeDecoder<'_>, pulses4: i32) {
    let mut p3 = [0i32; 2];
    let mut p2 = [0i32; 4];
    let mut p1 = [0i32; 8];
    (p3[0], p3[1]) = decode_split(rd, pulses4, &SHELL_CODE_TABLE3);
    // Depth first, left to right, as the reference orders the symbols.
    for i3 in 0..2 {
        (p2[2 * i3], p2[2 * i3 + 1]) = decode_split(rd, p3[i3], &SHELL_CODE_TABLE2);
        for i2 in 2 * i3..2 * i3 + 2 {
            (p1[2 * i2], p1[2 * i2 + 1]) = decode_split(rd, p2[i2], &SHELL_CODE_TABLE1);
            for i1 in 2 * i2..2 * i2 + 2 {
                let (a, b) = decode_split(rd, p1[i1], &SHELL_CODE_TABLE0);
                out[2 * i1] = a;
                out[2 * i1 + 1] = b;
            }
        }
    }
}

/// `silk_decode_pulses`: the signed pulse signal of one frame into
/// `pulses[..frame_length]` (rounded up to whole 16-sample blocks).
pub(crate) fn decode_pulses(
    rd: &mut RangeDecoder<'_>,
    pulses: &mut [i32; MAX_FRAME_LENGTH],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) {
    let rate_icdf = &RATE_LEVELS_ICDF[usize::from(signal_type >= 2)];
    let rate_level = rd.dec_icdf(rate_icdf, 8);

    // 10 ms at 12 kHz is 120 samples, which takes a partial eighth block.
    let blocks = frame_length.div_ceil(SHELL_BLOCK).min(MAX_BLOCKS);
    let mut sum_pulses = [0i32; MAX_BLOCKS];
    let mut n_lshifts = [0usize; MAX_BLOCKS];
    let level_icdf = PULSES_PER_BLOCK_ICDF
        .get(rate_level)
        .map_or(&[0u8][..], |t| &t[..]);
    let last_level = &PULSES_PER_BLOCK_ICDF[PULSES_PER_BLOCK_ICDF.len() - 1];
    for i in 0..blocks {
        sum_pulses[i] = rd.dec_icdf(level_icdf, 8) as i32;
        // MAX_PULSES + 1 escapes to one more LSB level; after 10 levels the
        // table is shifted so the escape cannot be coded.
        while sum_pulses[i] == MAX_PULSES as i32 + 1 {
            n_lshifts[i] += 1;
            let table = &last_level[usize::from(n_lshifts[i] == MAX_LSHIFTS)..];
            sum_pulses[i] = rd.dec_icdf(table, 8) as i32;
        }
    }

    for i in 0..blocks {
        let block = &mut pulses[i * SHELL_BLOCK..(i + 1) * SHELL_BLOCK];
        if sum_pulses[i] > 0 {
            shell_decoder(block, rd, sum_pulses[i]);
        } else {
            block.fill(0);
        }
    }

    for i in 0..blocks {
        let shifts = n_lshifts[i];
        if shifts > 0 {
            let block = &mut pulses[i * SHELL_BLOCK..(i + 1) * SHELL_BLOCK];
            for q in block.iter_mut() {
                let mut abs_q = *q;
                for _ in 0..shifts {
                    abs_q = (abs_q << 1) + rd.dec_icdf(&LSB_ICDF, 8) as i32;
                }
                *q = abs_q;
            }
            // Mark the block as non-empty for sign decoding.
            sum_pulses[i] |= (shifts as i32) << 5;
        }
    }

    // Signs, with a probability chosen by signal type, offset, and count.
    let sign_base = 7 * (quant_offset_type + 2 * signal_type).clamp(0, 5) as usize;
    let sign_icdf = &SIGN_ICDF[sign_base..sign_base + 7];
    let sign_blocks = ((frame_length + SHELL_BLOCK / 2) >> 4).min(blocks);
    for i in 0..sign_blocks {
        let p = sum_pulses[i];
        if p > 0 {
            let icdf = [sign_icdf[((p & 0x1f) as usize).min(6)], 0];
            for q in pulses[i * SHELL_BLOCK..(i + 1) * SHELL_BLOCK].iter_mut() {
                if *q > 0 && rd.dec_icdf(&icdf, 8) == 0 {
                    *q = -*q;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_FRAME_LENGTH, decode_pulses, shell_decoder};
    use crate::range::{RangeDecoder, RangeEncoder};

    /// Shell-encodes one block the way the reference encoder does, so the
    /// decoder can be checked on known magnitudes.
    fn shell_encode(enc: &mut RangeEncoder, pulses0: &[i32; 16]) {
        use super::super::tables::{
            SHELL_CODE_TABLE_OFFSETS, SHELL_CODE_TABLE0, SHELL_CODE_TABLE1, SHELL_CODE_TABLE2,
            SHELL_CODE_TABLE3,
        };
        let combine = |v: &[i32]| v.chunks(2).map(|c| c[0] + c[1]).collect::<Vec<_>>();
        let p1 = combine(pulses0);
        let p2 = combine(&p1);
        let p3 = combine(&p2);
        let p4 = combine(&p3);
        let mut split = |child: i32, parent: i32, table: &[u8]| {
            if parent > 0 {
                let off = usize::from(SHELL_CODE_TABLE_OFFSETS[parent as usize]);
                enc.enc_icdf(child as usize, &table[off..], 8);
            }
        };
        split(p3[0], p4[0], &SHELL_CODE_TABLE3);
        for i3 in 0..2 {
            split(p2[2 * i3], p3[i3], &SHELL_CODE_TABLE2);
            for i2 in 2 * i3..2 * i3 + 2 {
                split(p1[2 * i2], p2[i2], &SHELL_CODE_TABLE1);
                for i1 in 2 * i2..2 * i2 + 2 {
                    split(pulses0[2 * i1], p1[i1], &SHELL_CODE_TABLE0);
                }
            }
        }
    }

    #[test]
    fn shell_decoder_inverts_the_reference_split_order() {
        let blocks: [[i32; 16]; 3] = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [3, 0, 1, 0, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 0, 5],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ];
        let mut enc = RangeEncoder::new(256);
        for b in &blocks {
            shell_encode(&mut enc, b);
        }
        enc.done();
        let bytes = enc.into_bytes();
        let mut rd = RangeDecoder::new(&bytes);
        for b in &blocks {
            let mut out = [0i32; 16];
            shell_decoder(&mut out, &mut rd, b.iter().sum());
            assert_eq!(&out, b);
        }
    }

    #[test]
    fn hostile_bytes_give_bounded_pulses() {
        let mut seed = 0x1234_5678u32;
        for len in [0usize, 1, 3, 40, 200] {
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u8
                })
                .collect();
            for (frame_length, signal_type) in [(80, 0), (120, 1), (160, 2), (240, 2), (320, 1)] {
                let mut rd = RangeDecoder::new(&bytes);
                let mut pulses = [0i32; MAX_FRAME_LENGTH];
                decode_pulses(&mut rd, &mut pulses, signal_type, 1, frame_length);
                // At most 16 pulses with 10 LSB levels per sample.
                assert!(pulses.iter().all(|p| p.abs() < (17 << 10)));
            }
        }
        // All-0xff input escapes to every LSB level.
        let ff = [0xffu8; 64];
        let mut rd = RangeDecoder::new(&ff);
        let mut pulses = [0i32; MAX_FRAME_LENGTH];
        decode_pulses(&mut rd, &mut pulses, 2, 0, 320);
    }
}
