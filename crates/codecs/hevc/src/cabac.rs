//! CABAC decoding engine and context initialization (clause 9.3).
//!
//! The offset register keeps seven bits of look-ahead below the nine bits
//! of clause 9.3.4.3, so a decision compares against `range << 7`. Reading
//! past the end of the substream feeds zeros and counts the overrun; the
//! slice decoder turns an overrun into [`CodecError::Truncated`].

pub const SAO_MERGE: usize = 0;
pub const SAO_TYPE: usize = SAO_MERGE + 1;
pub const SPLIT_CU: usize = SAO_TYPE + 1;
pub const TRANSQUANT_BYPASS: usize = SPLIT_CU + 3;
pub const SKIP: usize = TRANSQUANT_BYPASS + 1;
pub const CU_QP_DELTA: usize = SKIP + 3;
pub const PRED_MODE: usize = CU_QP_DELTA + 2;
pub const PART_MODE: usize = PRED_MODE + 1;
pub const PREV_INTRA: usize = PART_MODE + 4;
pub const CHROMA_PRED: usize = PREV_INTRA + 1;
pub const MERGE_FLAG: usize = CHROMA_PRED + 1;
pub const MERGE_IDX: usize = MERGE_FLAG + 1;
pub const INTER_PRED: usize = MERGE_IDX + 1;
pub const REF_IDX: usize = INTER_PRED + 5;
pub const MVD_G0: usize = REF_IDX + 2;
pub const MVD_G1: usize = MVD_G0 + 1;
pub const MVP_FLAG: usize = MVD_G1 + 1;
pub const RQT_ROOT_CBF: usize = MVP_FLAG + 1;
pub const SPLIT_TRANSFORM: usize = RQT_ROOT_CBF + 1;
pub const CBF_LUMA: usize = SPLIT_TRANSFORM + 3;
pub const CBF_CHROMA: usize = CBF_LUMA + 2;
pub const TRANSFORM_SKIP: usize = CBF_CHROMA + 5;
pub const LAST_X: usize = TRANSFORM_SKIP + 2;
pub const LAST_Y: usize = LAST_X + 18;
pub const CODED_SUB_BLOCK: usize = LAST_Y + 18;
pub const SIG_COEFF: usize = CODED_SUB_BLOCK + 4;
pub const GREATER1: usize = SIG_COEFF + 42;
pub const GREATER2: usize = GREATER1 + 24;
pub const CONTEXTS: usize = GREATER2 + 6;

/// `initValue` of every context, by `initType` (Tables 9-5 to 9-37).
static INIT: [[u8; CONTEXTS]; 3] = [
    [
        153, // sao_merge
        200, // sao_type_idx
        139, 141, 157, // split_cu_flag
        154, // cu_transquant_bypass_flag
        154, 154, 154, // cu_skip_flag
        154, 154, // cu_qp_delta_abs
        154, // pred_mode_flag
        184, 154, 154, 154, // part_mode
        184, // prev_intra_luma_pred_flag
        63,  // intra_chroma_pred_mode
        154, // merge_flag
        154, // merge_idx
        154, 154, 154, 154, 154, // inter_pred_idc
        154, 154, // ref_idx
        154, // abs_mvd_greater0_flag
        154, // abs_mvd_greater1_flag
        154, // mvp_lx_flag
        154, // rqt_root_cbf
        153, 138, 138, // split_transform_flag
        111, 141, // cbf_luma
        94, 138, 182, 154, 154, // cbf_cb, cbf_cr
        139, 139, // transform_skip_flag
        110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123,
        63, // last_sig_coeff_x_prefix
        110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123,
        63, // last_sig_coeff_y_prefix
        91, 171, 134, 141, // coded_sub_block_flag
        111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141,
        179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
        136, 139, 111, 136, 139, 111, // sig_coeff_flag
        140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166,
        182, 140, 227, 122, 197, // coeff_abs_level_greater1_flag
        138, 153, 136, 167, 152, 152, // coeff_abs_level_greater2_flag
    ],
    [
        153, // sao_merge
        185, // sao_type_idx
        107, 139, 126, // split_cu_flag
        154, // cu_transquant_bypass_flag
        197, 185, 201, // cu_skip_flag
        154, 154, // cu_qp_delta_abs
        149, // pred_mode_flag
        154, 139, 154, 154, // part_mode
        154, // prev_intra_luma_pred_flag
        152, // intra_chroma_pred_mode
        110, // merge_flag
        122, // merge_idx
        95, 79, 63, 31, 31, // inter_pred_idc
        153, 153, // ref_idx
        140, // abs_mvd_greater0_flag
        198, // abs_mvd_greater1_flag
        168, // mvp_lx_flag
        79,  // rqt_root_cbf
        124, 138, 94, // split_transform_flag
        153, 111, // cbf_luma
        149, 107, 167, 154, 154, // cbf_cb, cbf_cr
        139, 139, // transform_skip_flag
        125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123,
        108, // last_sig_coeff_x_prefix
        125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123,
        108, // last_sig_coeff_y_prefix
        121, 140, 61, 154, // coded_sub_block_flag
        155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167,
        151, 183, 140, 151, 183, 140, // sig_coeff_flag
        154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194,
        166, 167, 154, 167, 137, 182, // coeff_abs_level_greater1_flag
        107, 167, 91, 122, 107, 167, // coeff_abs_level_greater2_flag
    ],
    [
        153, // sao_merge
        160, // sao_type_idx
        107, 139, 126, // split_cu_flag
        154, // cu_transquant_bypass_flag
        197, 185, 201, // cu_skip_flag
        154, 154, // cu_qp_delta_abs
        134, // pred_mode_flag
        154, 139, 154, 154, // part_mode
        183, // prev_intra_luma_pred_flag
        152, // intra_chroma_pred_mode
        154, // merge_flag
        137, // merge_idx
        95, 79, 63, 31, 31, // inter_pred_idc
        153, 153, // ref_idx
        169, // abs_mvd_greater0_flag
        198, // abs_mvd_greater1_flag
        168, // mvp_lx_flag
        79,  // rqt_root_cbf
        224, 167, 122, // split_transform_flag
        153, 111, // cbf_luma
        149, 92, 167, 154, 154, // cbf_cb, cbf_cr
        139, 139, // transform_skip_flag
        125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123,
        93, // last_sig_coeff_x_prefix
        125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123,
        93, // last_sig_coeff_y_prefix
        121, 140, 61, 154, // coded_sub_block_flag
        170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167,
        151, 183, 140, 151, 183, 140, // sig_coeff_flag
        154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208,
        166, 167, 154, 152, 167, 182, // coeff_abs_level_greater1_flag
        107, 167, 91, 107, 107, 167, // coeff_abs_level_greater2_flag
    ],
];

static RANGE_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

static TRANS_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// Context variables: `pStateIdx << 1 | valMps`.
pub type Contexts = [u8; CONTEXTS];

/// Initialized contexts for `init_type` and the slice QP (clause 9.3.2.2).
pub fn init_contexts(init_type: usize, qp: i32) -> Contexts {
    let mut out = [0u8; CONTEXTS];
    let qp = qp.clamp(0, 51);
    for (ctx, &value) in out.iter_mut().zip(INIT[init_type.min(2)].iter()) {
        let slope = i32::from(value >> 4) * 5 - 45;
        let offset = (i32::from(value & 15) << 3) - 16;
        let pre = (((slope * qp) >> 4) + offset).clamp(1, 126);
        *ctx = if pre <= 63 {
            ((63 - pre) as u8) << 1
        } else {
            (((pre - 64) as u8) << 1) | 1
        };
    }
    out
}

pub struct Cabac<'a> {
    data: &'a [u8],
    at: usize,
    range: u32,
    value: u32,
    bits_needed: i32,
    /// Bytes fed past the end of `data`.
    overrun: usize,
    pub ctx: Contexts,
}

impl<'a> Cabac<'a> {
    pub fn new(data: &'a [u8], ctx: Contexts) -> Self {
        let mut cabac = Self {
            data,
            at: 0,
            range: 510,
            value: 0,
            bits_needed: 8,
            overrun: 0,
            ctx,
        };
        cabac.start();
        cabac
    }

    fn start(&mut self) {
        self.range = 510;
        self.bits_needed = 8;
        self.value = 0;
        self.value = u32::from(self.next_byte()) << 8;
        self.bits_needed -= 8;
        self.value |= u32::from(self.next_byte());
        self.bits_needed -= 8;
    }

    /// Start the engine again at `data` (an entry point or after PCM).
    pub fn restart(&mut self, data: &'a [u8]) {
        self.data = data;
        self.at = 0;
        self.start();
    }

    #[inline(always)]
    fn next_byte(&mut self) -> u8 {
        if let Some(&byte) = self.data.get(self.at) {
            self.at += 1;
            byte
        } else {
            self.overrun += 1;
            0
        }
    }

    /// More than the two bytes of look-ahead the engine may hold were read
    /// past the end.
    pub fn overrun(&self) -> bool {
        self.overrun > 2
    }

    #[inline(always)]
    pub fn decide(&mut self, index: usize) -> bool {
        let model = self.ctx[index];
        let state = usize::from(model >> 1);
        let mps = model & 1;
        let lps = u32::from(RANGE_LPS[state][((self.range >> 6) & 3) as usize]);
        self.range -= lps;
        let scaled = self.range << 7;
        if self.value < scaled {
            self.ctx[index] = (((state as u8) + u8::from(state < 62)) << 1) | mps;
            if scaled < (256 << 7) {
                self.range = scaled >> 6;
                self.value <<= 1;
                self.bits_needed += 1;
                if self.bits_needed == 0 {
                    self.bits_needed = -8;
                    self.value |= u32::from(self.next_byte());
                }
            }
            mps == 1
        } else {
            self.value -= scaled;
            let shift = RENORM[(lps >> 3) as usize];
            self.value <<= shift;
            self.range = lps << shift;
            let new_mps = if state == 0 { 1 - mps } else { mps };
            self.ctx[index] = (TRANS_LPS[state] << 1) | new_mps;
            self.bits_needed += i32::from(shift);
            if self.bits_needed >= 0 {
                self.value |= u32::from(self.next_byte()) << self.bits_needed;
                self.bits_needed -= 8;
            }
            mps == 0
        }
    }

    #[inline(always)]
    pub fn bypass(&mut self) -> bool {
        self.value <<= 1;
        self.bits_needed += 1;
        if self.bits_needed >= 0 {
            self.bits_needed = -8;
            self.value |= u32::from(self.next_byte());
        }
        let scaled = self.range << 7;
        if self.value >= scaled {
            self.value -= scaled;
            true
        } else {
            false
        }
    }

    /// `n` bypass bins, most significant first.
    #[inline(always)]
    pub fn bypass_bits(&mut self, n: u32) -> u32 {
        let mut value = 0;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.bypass());
        }
        value
    }

    pub fn terminate(&mut self) -> bool {
        self.range -= 2;
        let scaled = self.range << 7;
        if self.value >= scaled {
            true
        } else {
            if scaled < (256 << 7) {
                self.range = scaled >> 6;
                self.value <<= 1;
                self.bits_needed += 1;
                if self.bits_needed == 0 {
                    self.bits_needed = -8;
                    self.value |= u32::from(self.next_byte());
                }
            }
            false
        }
    }

    /// The bytes after a terminate bin of 1. The stop bit and the alignment
    /// zeros after it sit in the last byte the engine read, so the next
    /// aligned data starts at the next byte.
    pub fn remaining(&self) -> &'a [u8] {
        self.data.get(self.at..).unwrap_or(&[])
    }
}

static RENORM: [u8; 32] = [
    6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[cfg(test)]
mod tests {
    use super::{CONTEXTS, Cabac, GREATER2, init_contexts};

    #[test]
    fn context_layout_has_one_entry_per_initial_value() {
        assert_eq!(GREATER2 + 6, CONTEXTS);
        assert_eq!(CONTEXTS, 155);
    }

    #[test]
    fn initialization_follows_equation_9_6() {
        // initValue 154 is m = 0, n = 64: state 0, MPS 1 at any QP.
        let ctx = init_contexts(0, 30);
        assert_eq!(ctx[super::SKIP], 1);
        // initValue 63: m = -30, n = 104. At QP 30: ((-30 * 30) >> 4) + 104 = 47.
        // 47 <= 63, so MPS 0 and state 63 - 47 = 16.
        assert_eq!(ctx[super::CHROMA_PRED], 16 << 1);
    }

    #[test]
    fn bypass_bins_read_the_stream_bits() {
        // With range 510 and offset 0, a run of zero bytes reads zeros and
        // never overruns early.
        let data = [0u8; 8];
        let mut cabac = Cabac::new(&data, init_contexts(0, 26));
        assert_eq!(cabac.bypass_bits(16), 0);
        assert!(!cabac.overrun());
        for _ in 0..64 {
            cabac.bypass();
        }
        assert!(cabac.overrun());
    }
}
