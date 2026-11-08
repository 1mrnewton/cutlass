//! Palette colors, the neighbor cache, and the color-index context.
//!
//! Default probabilities are the specification's initial CDFs. A row is the
//! cumulative values, a terminal 32768, and an adaptation counter.

const HASH_MUL: [i32; 3] = [1, 2, 2];
const COLOR_CTX: [i8; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];

pub fn ceil_log2(value: u32) -> u32 {
    if value < 2 {
        return 0;
    }
    let mut bits = 1u32;
    let mut power = 2u32;
    while power < value {
        bits += 1;
        power <<= 1;
    }
    bits
}

/// Merge the above and left palettes. Both inputs are sorted.
pub fn merge_cache(above: &[u16], left: &[u16]) -> (usize, [u16; 16]) {
    let mut cache = [0u16; 16];
    let mut above_idx = 0;
    let mut left_idx = 0;
    let mut n = 0usize;
    while above_idx < above.len() && left_idx < left.len() && n < cache.len() {
        let above_c = above[above_idx];
        let left_c = left[left_idx];
        if left_c < above_c {
            if n == 0 || left_c != cache[n - 1] {
                cache[n] = left_c;
                n += 1;
            }
            left_idx += 1;
        } else {
            if n == 0 || above_c != cache[n - 1] {
                cache[n] = above_c;
                n += 1;
            }
            above_idx += 1;
            if left_c == above_c {
                left_idx += 1;
            }
        }
    }
    while above_idx < above.len() && n < cache.len() {
        let val = above[above_idx];
        above_idx += 1;
        if n == 0 || val != cache[n - 1] {
            cache[n] = val;
            n += 1;
        }
    }
    while left_idx < left.len() && n < cache.len() {
        let val = left[left_idx];
        left_idx += 1;
        if n == 0 || val != cache[n - 1] {
            cache[n] = val;
            n += 1;
        }
    }
    (n, cache)
}

/// `get_palette_color_context`. The context indexes a color-map CDF.
/// The order maps the decoded symbol back to a palette index.
pub fn color_context(
    map: &[u8],
    stride: usize,
    row: usize,
    col: usize,
    n: usize,
) -> Option<(usize, [u8; 8])> {
    let mut scores = [0i32; 8];
    let mut order = [0u8, 1, 2, 3, 4, 5, 6, 7];
    let at = |r: usize, c: usize| map.get(r * stride + c).copied().unwrap_or(0) as usize;
    if col > 0 {
        let neighbor = at(row, col - 1);
        if neighbor < 8 {
            scores[neighbor] += 2;
        }
    }
    if row > 0 && col > 0 {
        let neighbor = at(row - 1, col - 1);
        if neighbor < 8 {
            scores[neighbor] += 1;
        }
    }
    if row > 0 {
        let neighbor = at(row - 1, col);
        if neighbor < 8 {
            scores[neighbor] += 2;
        }
    }
    let limit = n.min(8);
    for i in 0..3 {
        let mut max_score = scores[i];
        let mut max_idx = i;
        let mut j = i + 1;
        while j < limit {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
            j += 1;
        }
        if max_idx != i {
            let max_order = order[max_idx];
            let mut k = max_idx;
            while k > i {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
                k -= 1;
            }
            scores[i] = max_score;
            order[i] = max_order;
        }
    }
    let mut hash = 0i32;
    for i in 0..3 {
        hash += scores[i] * HASH_MUL[i];
    }
    let hash = usize::try_from(hash).ok()?;
    let ctx = i32::from(*COLOR_CTX.get(hash)?);
    if ctx < 0 {
        return None;
    }
    usize::try_from(ctx).ok().map(|ctx| (ctx, order))
}

fn pack(rows: &[&[u16]]) -> Vec<u16> {
    let mut out = Vec::new();
    for row in rows {
        out.extend_from_slice(row);
        out.push(32_768);
        out.push(0);
    }
    out
}

fn bools(probs: &[u16]) -> Vec<u16> {
    let rows: Vec<[u16; 1]> = probs.iter().map(|prob| [*prob]).collect();
    let borrowed: Vec<&[u16]> = rows.iter().map(|row| row.as_slice()).collect();
    pack(&borrowed)
}

pub fn palette_y_cdf() -> Vec<u16> {
    bools(&[
        31_676, 3_419, 1_261, 31_912, 2_859, 980, 31_823, 3_400, 781, 32_030, 3_561, 904, 32_309,
        7_337, 1_462, 32_265, 4_015, 1_521, 32_450, 7_946, 129,
    ])
}

pub fn palette_uv_cdf() -> Vec<u16> {
    bools(&[32_461, 21_488])
}

pub fn palette_size_y_cdf() -> Vec<u16> {
    pack(&[
        &[7_952, 13_000, 18_149, 21_478, 25_527, 29_241],
        &[7_139, 11_421, 16_195, 19_544, 23_666, 28_073],
        &[7_788, 12_741, 17_325, 20_500, 24_315, 28_530],
        &[8_271, 14_064, 18_246, 21_564, 25_071, 28_533],
        &[12_725, 19_180, 21_863, 24_839, 27_535, 30_120],
        &[9_711, 14_888, 16_923, 21_052, 25_661, 27_875],
        &[14_940, 20_797, 21_678, 24_186, 27_033, 28_999],
    ])
}

pub fn palette_size_uv_cdf() -> Vec<u16> {
    pack(&[
        &[8_713, 19_979, 27_128, 29_609, 31_331, 32_272],
        &[5_839, 15_573, 23_581, 26_947, 29_848, 31_700],
        &[4_426, 11_260, 17_999, 21_483, 25_863, 29_430],
        &[3_228, 9_464, 14_993, 18_089, 22_523, 27_420],
        &[3_768, 8_886, 13_091, 17_852, 22_495, 27_207],
        &[2_464, 8_451, 12_861, 21_632, 25_525, 28_555],
        &[1_269, 5_435, 10_433, 18_963, 21_700, 25_865],
    ])
}

pub fn intrabc_cdf() -> Vec<u16> {
    bools(&[30_531])
}

pub fn color_map_y() -> [Vec<u16>; 7] {
    [
        pack(&[&[28_710], &[16_384], &[10_553], &[27_036], &[31_603]]),
        pack(&[
            &[27_877, 30_490],
            &[11_532, 25_697],
            &[6_544, 30_234],
            &[23_018, 28_072],
            &[31_915, 32_385],
        ]),
        pack(&[
            &[25_572, 28_046, 30_045],
            &[9_478, 21_590, 27_256],
            &[7_248, 26_837, 29_824],
            &[19_167, 24_486, 28_349],
            &[31_400, 31_825, 32_250],
        ]),
        pack(&[
            &[24_779, 26_955, 28_576, 30_282],
            &[8_669, 20_364, 24_073, 28_093],
            &[4_255, 27_565, 29_377, 31_067],
            &[19_864, 23_674, 26_716, 29_530],
            &[31_646, 31_893, 32_147, 32_426],
        ]),
        pack(&[
            &[23_132, 25_407, 26_970, 28_435, 30_073],
            &[7_443, 17_242, 20_717, 24_762, 27_982],
            &[6_300, 24_862, 26_944, 28_784, 30_671],
            &[18_916, 22_895, 25_267, 27_435, 29_652],
            &[31_270, 31_550, 31_808, 32_059, 32_353],
        ]),
        pack(&[
            &[23_105, 25_199, 26_464, 27_684, 28_931, 30_318],
            &[6_950, 15_447, 18_952, 22_681, 25_567, 28_563],
            &[7_560, 23_474, 25_490, 27_203, 28_921, 30_708],
            &[18_544, 22_373, 24_457, 26_195, 28_119, 30_045],
            &[31_198, 31_451, 31_670, 31_882, 32_123, 32_391],
        ]),
        pack(&[
            &[21_689, 23_883, 25_163, 26_352, 27_506, 28_827, 30_195],
            &[6_892, 15_385, 17_840, 21_606, 24_287, 26_753, 29_204],
            &[5_651, 23_182, 25_042, 26_518, 27_982, 29_392, 30_900],
            &[19_349, 22_578, 24_418, 25_994, 27_524, 29_031, 30_448],
            &[31_028, 31_270, 31_504, 31_705, 31_927, 32_153, 32_392],
        ]),
    ]
}

pub fn color_map_uv() -> [Vec<u16>; 7] {
    [
        pack(&[&[29_089], &[16_384], &[8_713], &[29_257], &[31_610]]),
        pack(&[
            &[25_257, 29_145],
            &[12_287, 27_293],
            &[7_033, 27_960],
            &[20_145, 25_405],
            &[30_608, 31_639],
        ]),
        pack(&[
            &[24_210, 27_175, 29_903],
            &[9_888, 22_386, 27_214],
            &[5_901, 26_053, 29_293],
            &[18_318, 22_152, 28_333],
            &[30_459, 31_136, 31_926],
        ]),
        pack(&[
            &[22_980, 25_479, 27_781, 29_986],
            &[8_413, 21_408, 24_859, 28_874],
            &[2_257, 29_449, 30_594, 31_598],
            &[19_189, 21_202, 25_915, 28_620],
            &[31_844, 32_044, 32_281, 32_518],
        ]),
        pack(&[
            &[22_217, 24_567, 26_637, 28_683, 30_548],
            &[7_307, 16_406, 19_636, 24_632, 28_424],
            &[4_441, 25_064, 26_879, 28_942, 30_919],
            &[17_210, 20_528, 23_319, 26_750, 29_582],
            &[30_674, 30_953, 31_396, 31_735, 32_207],
        ]),
        pack(&[
            &[21_239, 23_168, 25_044, 26_962, 28_705, 30_506],
            &[6_545, 15_012, 18_004, 21_817, 25_503, 28_701],
            &[3_448, 26_295, 27_437, 28_704, 30_126, 31_442],
            &[15_889, 18_323, 21_704, 24_698, 26_976, 29_690],
            &[30_988, 31_204, 31_479, 31_734, 31_983, 32_325],
        ]),
        pack(&[
            &[21_442, 23_288, 24_758, 26_246, 27_649, 28_980, 30_563],
            &[5_863, 14_933, 17_552, 20_668, 23_683, 26_411, 29_273],
            &[3_415, 25_810, 26_877, 27_990, 29_223, 30_394, 31_618],
            &[17_965, 20_084, 22_232, 23_974, 26_274, 28_402, 30_390],
            &[31_190, 31_329, 31_516, 31_679, 31_825, 32_026, 32_322],
        ]),
    ]
}

#[cfg(test)]
mod tests {
    use super::{color_context, merge_cache};

    #[test]
    fn palette_cache_merges_sorted_neighbors() {
        let (n, cache) = merge_cache(&[1, 4, 9], &[4, 7, 12]);
        assert_eq!(n, 5);
        assert_eq!(&cache[..n], &[1, 4, 7, 9, 12]);
    }

    #[test]
    fn a_left_neighbor_uses_the_first_color_context() {
        let map = [0u8, 0];
        let (ctx, order) = color_context(&map, 2, 0, 1, 2).expect("context");
        assert_eq!(ctx, 0);
        assert_eq!(order[0], 0);
    }
}
