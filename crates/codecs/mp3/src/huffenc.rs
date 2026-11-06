//! Huffman coding of a quantized spectrum: codeword tables by symbol, bit
//! counts for table selection, region division, and the writers.
//!
//! The tables are the decoder's (ISO/IEC 11172-3 Table B.7), read back by
//! symbol, so the encoder cannot emit a code the decoder does not read.

use std::sync::OnceLock;

use crate::bits::BitWriter;
use crate::huffman::tables;
use crate::tables::{TABLE_SELECT, long_starts};

/// Codeword lengths and codes by `x * 16 + y`, for code tables 0..=24.
pub(crate) struct Enc {
    len: Vec<[u8; 256]>,
    code: Vec<[u32; 256]>,
    quad_a: [(u8, u32); 16],
}

pub(crate) fn enc() -> &'static Enc {
    static ENC: OnceLock<Enc> = OnceLock::new();
    ENC.get_or_init(|| {
        let all = tables();
        let mut len = vec![[0u8; 256]; 25];
        let mut code = vec![[0u32; 256]; 25];
        for index in 0u8..25 {
            if let Some(table) = all.big(index) {
                for (sym, (l, c)) in table.by_symbol().into_iter().enumerate() {
                    len[usize::from(index)][sym] = l;
                    code[usize::from(index)][sym] = c;
                }
            }
        }
        let by_symbol = all.quad_a.by_symbol();
        let quad_a = std::array::from_fn(|sym| by_symbol[sym]);
        Enc { len, code, quad_a }
    })
}

/// The largest value a non-escape table codes, by code table.
const TABLE_MAX: [u32; 16] = [0, 1, 2, 2, 0, 3, 3, 5, 5, 5, 7, 7, 7, 15, 0, 15];

/// The largest quantized magnitude any table codes: 15 plus 13 linbits.
pub(crate) const MAX_VALUE: u32 = 15 + 8191;

/// The table groups a region whose largest value is `max` (1..=15) tries.
fn candidates(max: u32) -> &'static [u8] {
    match max {
        1 => &[1, 2, 3],
        2 => &[2, 3],
        3 => &[5, 6],
        4 | 5 => &[7, 8, 9],
        6 | 7 => &[10, 11, 12],
        _ => &[13, 15],
    }
}

/// Bits of the pairs in `from..to` with `table_select`, signs and linbits included.
pub(crate) fn count_pairs(ix: &[i32; 576], from: usize, to: usize, select: u8) -> u32 {
    let (table, linbits) = TABLE_SELECT[usize::from(select & 31)];
    if table == 0 || from >= to {
        return 0;
    }
    let lens = &enc().len[usize::from(table)];
    let linbits = u32::from(linbits);
    let mut bits = 0u32;
    let mut line = from;
    while line < to {
        let x = ix[line].unsigned_abs();
        let y = ix[line + 1].unsigned_abs();
        let (cx, cy) = if linbits > 0 {
            let mut extra = 0;
            let cx = if x >= 15 {
                extra += linbits;
                15
            } else {
                x
            };
            let cy = if y >= 15 {
                extra += linbits;
                15
            } else {
                y
            };
            bits += extra;
            (cx, cy)
        } else {
            (x, y)
        };
        bits += u32::from(lens[(cx * 16 + cy) as usize]);
        bits += u32::from(x != 0) + u32::from(y != 0);
        line += 2;
    }
    bits
}

/// The cheapest `table_select` for `from..to`, and its bits.
pub(crate) fn choose_table(ix: &[i32; 576], from: usize, to: usize) -> (u8, u32) {
    if from >= to {
        return (0, 0);
    }
    let max = ix[from..to]
        .iter()
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    if max == 0 {
        return (0, 0);
    }
    if max <= 15 {
        let mut best = (0u8, u32::MAX);
        for &table in candidates(max) {
            if TABLE_MAX[usize::from(table)] < max {
                continue;
            }
            let bits = count_pairs(ix, from, to, table);
            if bits < best.1 {
                best = (table, bits);
            }
        }
        return best;
    }
    let mut best = (0u8, u32::MAX);
    for family in [16u8, 24] {
        let Some(select) = (family..family + 8).find(|&select| {
            let linbits = u32::from(TABLE_SELECT[usize::from(select)].1);
            max - 15 < (1 << linbits)
        }) else {
            continue;
        };
        let bits = count_pairs(ix, from, to, select);
        if bits < best.1 {
            best = (select, bits);
        }
    }
    best
}

/// Where the big-value region ends and where the count1 region ends.
///
/// Above `count1_end` every line is zero; between the two ends every line
/// is -1, 0 or 1, in whole quadruples.
pub(crate) fn extents(ix: &[i32; 576]) -> (usize, usize) {
    let mut end = 576;
    while end >= 2 && ix[end - 1] == 0 && ix[end - 2] == 0 {
        end -= 2;
    }
    let count1_end = end;
    let mut big = end;
    while big >= 4 && ix[big - 4..big].iter().all(|v| v.unsigned_abs() <= 1) {
        big -= 4;
    }
    (big, count1_end)
}

/// Bits of the count1 region and whether table B is the cheaper.
pub(crate) fn count1(ix: &[i32; 576], from: usize, to: usize) -> (bool, u32) {
    let quad = &enc().quad_a;
    let mut a = 0u32;
    let mut b = 0u32;
    let mut line = from;
    while line < to {
        let sym = quad_symbol(&ix[line..line + 4]);
        let signs = sym.count_ones();
        a += u32::from(quad[usize::from(sym)].0) + signs;
        b += 4 + signs;
        line += 4;
    }
    if b < a { (true, b) } else { (false, a) }
}

fn quad_symbol(values: &[i32]) -> u8 {
    let mut sym = 0u8;
    for (slot, value) in values.iter().enumerate() {
        if *value != 0 {
            sym |= 8 >> slot;
        }
    }
    sym
}

/// Write the pairs in `from..to` with `table_select`.
pub(crate) fn write_pairs(
    writer: &mut BitWriter,
    ix: &[i32; 576],
    from: usize,
    to: usize,
    select: u8,
) {
    let (table, linbits) = TABLE_SELECT[usize::from(select & 31)];
    if table == 0 || table == 255 {
        return;
    }
    let e = enc();
    let lens = &e.len[usize::from(table)];
    let codes = &e.code[usize::from(table)];
    let linbits = u32::from(linbits);
    let mut line = from;
    while line < to {
        let x = ix[line];
        let y = ix[line + 1];
        let (ax, ay) = (x.unsigned_abs(), y.unsigned_abs());
        let cx = if linbits > 0 { ax.min(15) } else { ax };
        let cy = if linbits > 0 { ay.min(15) } else { ay };
        let sym = (cx * 16 + cy) as usize;
        writer.put(codes[sym], u32::from(lens[sym]));
        if linbits > 0 && ax >= 15 {
            writer.put(ax - 15, linbits);
        }
        if x != 0 {
            writer.put(u32::from(x < 0), 1);
        }
        if linbits > 0 && ay >= 15 {
            writer.put(ay - 15, linbits);
        }
        if y != 0 {
            writer.put(u32::from(y < 0), 1);
        }
        line += 2;
    }
}

/// Write the count1 quadruples in `from..to`.
pub(crate) fn write_count1(
    writer: &mut BitWriter,
    ix: &[i32; 576],
    from: usize,
    to: usize,
    table_b: bool,
) {
    let quad = &enc().quad_a;
    let mut line = from;
    while line < to {
        let values = &ix[line..line + 4];
        let sym = quad_symbol(values);
        if table_b {
            writer.put(15 - u32::from(sym), 4);
        } else {
            let (len, code) = quad[usize::from(sym)];
            writer.put(code, u32::from(len));
        }
        for value in values {
            if *value != 0 {
                writer.put(u32::from(*value < 0), 1);
            }
        }
        line += 4;
    }
}

/// A long-block region division: `region0_count`, `region1_count`, the
/// three tables, and the bits of the big-value region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Division {
    pub region0_count: u8,
    pub region1_count: u8,
    pub table_select: [u8; 3],
    pub bits: u32,
}

/// Region boundaries in lines for a long block division.
pub(crate) fn long_bounds(
    rate_table: usize,
    region0: u8,
    region1: u8,
    big_end: usize,
) -> (usize, usize) {
    let starts = long_starts(rate_table);
    let a = usize::from(region0) + 1;
    let b = (usize::from(region0) + usize::from(region1) + 2).min(22);
    (
        usize::from(starts[a]).min(big_end),
        usize::from(starts[b]).min(big_end),
    )
}

/// A quick division for rate control: about a third of the bands in
/// region 0 and a third in region 1.
pub(crate) fn quick_division(ix: &[i32; 576], rate_table: usize, big_end: usize) -> Division {
    let starts = long_starts(rate_table);
    let bands = (1..=22)
        .find(|&band| usize::from(starts[band]) >= big_end)
        .unwrap_or(22);
    let region0 = (bands / 3).clamp(1, 16) - 1;
    let region1 = ((bands - (region0 + 1)) / 2).clamp(1, 8) - 1;
    let region0 = region0 as u8;
    let region1 = region1 as u8;
    divide(ix, rate_table, big_end, region0, region1)
}

fn divide(
    ix: &[i32; 576],
    rate_table: usize,
    big_end: usize,
    region0: u8,
    region1: u8,
) -> Division {
    let (r0, r1) = long_bounds(rate_table, region0, region1, big_end);
    let (t0, b0) = choose_table(ix, 0, r0);
    let (t1, b1) = choose_table(ix, r0, r1);
    let (t2, b2) = choose_table(ix, r1, big_end);
    Division {
        region0_count: region0,
        region1_count: region1,
        table_select: [t0, t1, t2],
        bits: b0 + b1 + b2,
    }
}

/// The cheapest of every long-block division of the big-value region.
pub(crate) fn best_division(ix: &[i32; 576], rate_table: usize, big_end: usize) -> Division {
    let starts = long_starts(rate_table);
    // Cheapest table and its bits for each run of whole bands, computed once.
    let mut boundaries: Vec<usize> = starts
        .iter()
        .map(|s| usize::from(*s).min(big_end))
        .collect();
    boundaries.dedup();
    let mut best = quick_division(ix, rate_table, big_end);
    let mut cache = std::collections::HashMap::new();
    let mut cost = |from: usize, to: usize| -> (u8, u32) {
        *cache
            .entry((from, to))
            .or_insert_with(|| choose_table(ix, from, to))
    };
    for region0 in 0u8..16 {
        let a = usize::from(region0) + 1;
        if a > 22 {
            break;
        }
        let r0 = usize::from(starts[a]).min(big_end);
        let (t0, b0) = cost(0, r0);
        if b0 >= best.bits {
            continue;
        }
        for region1 in 0u8..8 {
            let b = (usize::from(region0) + usize::from(region1) + 2).min(22);
            let r1 = usize::from(starts[b]).min(big_end);
            let (t1, b1) = cost(r0, r1);
            if b0 + b1 >= best.bits {
                continue;
            }
            let (t2, b2) = cost(r1, big_end);
            let bits = b0 + b1 + b2;
            if bits < best.bits {
                best = Division {
                    region0_count: region0,
                    region1_count: region1,
                    table_select: [t0, t1, t2],
                    bits,
                };
            }
            if r1 >= big_end {
                break;
            }
        }
        if r0 >= big_end {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{
        best_division, choose_table, count_pairs, count1, extents, write_count1, write_pairs,
    };
    use crate::bits::{BitReader, BitWriter};
    use crate::header::{FrameHeader, Mode, Version};
    use crate::layer3::read_spectrum;
    use crate::sideinfo::GranuleInfo;

    fn header() -> FrameHeader {
        FrameHeader {
            version: Version::Mpeg1,
            layer: 3,
            protected: false,
            bitrate_index: 9,
            bitrate_kbps: 128,
            rate_index: 0,
            sample_rate: 44_100,
            padding: false,
            private: false,
            mode: Mode::Mono,
            mode_extension: 0,
            copyright: false,
            original: false,
            emphasis: 0,
        }
    }

    /// A spectrum with large low lines, an escape value, a count1 tail.
    fn spectrum(seed: u32) -> [i32; 576] {
        let mut ix = [0i32; 576];
        let mut state = seed;
        for (line, slot) in ix.iter_mut().enumerate().take(500) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) as i32;
            let scale = if line < 20 {
                400
            } else if line < 100 {
                20
            } else if line < 300 {
                4
            } else {
                2
            };
            let value = r % scale;
            *slot = if r & 1 == 1 { -value } else { value };
        }
        for slot in ix.iter_mut().skip(300).take(200) {
            *slot = (*slot).clamp(-1, 1);
        }
        ix[3] = 8206;
        ix
    }

    #[test]
    fn what_the_writer_codes_the_decoder_reads_and_the_counts_agree() {
        for seed in 0..8 {
            let ix = spectrum(seed);
            let (big_end, count1_end) = extents(&ix);
            assert_eq!(big_end % 2, 0);
            assert_eq!((count1_end - big_end) % 4, 0);
            let division = best_division(&ix, 0, big_end);
            let (table_b, quad_bits) = count1(&ix, big_end, count1_end);
            let mut writer = BitWriter::new();
            let (r0, r1) =
                super::long_bounds(0, division.region0_count, division.region1_count, big_end);
            let bounds = [(0, r0), (r0, r1), (r1, big_end)];
            for (region, (from, to)) in bounds.into_iter().enumerate() {
                write_pairs(&mut writer, &ix, from, to, division.table_select[region]);
            }
            assert_eq!(writer.bits() as u32, division.bits);
            write_count1(&mut writer, &ix, big_end, count1_end, table_b);
            let bits = writer.bits();
            assert_eq!(bits as u32, division.bits + quad_bits);
            let bytes = writer.finish();
            let info = GranuleInfo {
                part2_3_length: bits as u16,
                big_values: (big_end / 2) as u16,
                table_select: division.table_select,
                region0_count: division.region0_count,
                region1_count: division.region1_count,
                count1_table_b: table_b,
                ..GranuleInfo::default()
            };
            let mut reader = BitReader::window(&bytes, 0, bits);
            let mut decoded = [0i32; 576];
            read_spectrum(&mut reader, &header(), &info, &mut decoded).expect("decode");
            assert_eq!(decoded, ix, "seed {seed}");
        }
    }

    #[test]
    fn the_chosen_table_is_the_cheapest_legal_one() {
        let ix = spectrum(3);
        let (select, bits) = choose_table(&ix, 100, 300);
        for other in 1u8..32 {
            if other == 4 || other == 14 {
                continue;
            }
            let (table, linbits) = crate::tables::TABLE_SELECT[usize::from(other)];
            let max = ix[100..300]
                .iter()
                .map(|v| v.unsigned_abs())
                .max()
                .unwrap_or(0);
            let limit = if table >= 16 {
                15 + (1u32 << linbits) - 1
            } else {
                super::TABLE_MAX[usize::from(table)]
            };
            if limit >= max {
                assert!(
                    count_pairs(&ix, 100, 300, other) >= bits,
                    "{other} vs {select}"
                );
            }
        }
    }
}
