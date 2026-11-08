//! Fixed tables from ATSC A/52:2018. Each one names the table it is
//! transcribed from. The bit allocation tables were extracted from the
//! specification text by script and checked against it entry by entry.

pub(crate) mod vq;

use std::f32::consts::{FRAC_1_SQRT_2, SQRT_2};

/// Slow decay, fast decay, slow gain, dB per bit, floor (Tables 7.6 to 7.10).
pub(crate) const SLOWDEC: [i32; 4] = [0x0f, 0x11, 0x13, 0x15];
pub(crate) const FASTDEC: [i32; 4] = [0x3f, 0x53, 0x67, 0x7b];
pub(crate) const SLOWGAIN: [i32; 4] = [0x540, 0x4d8, 0x478, 0x410];
pub(crate) const DBPBTAB: [i32; 4] = [0x000, 0x700, 0x900, 0xb00];
pub(crate) const FLOORTAB: [i32; 8] = [0x2f0, 0x2b0, 0x270, 0x230, 0x1f0, 0x170, 0x0f0, -2048];

/// Fast gain (Table 7.11).
pub(crate) const FASTGAIN: [i32; 8] = [0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400];

/// First bin of each of the 50 bands (Table 7.12). Band 50 is the end.
pub(crate) const BNDTAB: [u16; 51] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 31, 34, 37, 40, 43, 46, 49, 55, 61, 67, 73, 79, 85, 97, 109, 121, 133, 157, 181,
    205, 229, 253,
];

/// Band of each bin (Table 7.13).
pub(crate) const MASKTAB: [u8; 253] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35,
    35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 39,
    39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
    49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
];

/// Log-addition (Table 7.14).
pub(crate) const LATAB: [u16; 256] = [
    64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 52, 51, 50, 49, 48, 47, 47, 46, 45, 44, 44,
    43, 42, 41, 41, 40, 39, 38, 38, 37, 36, 36, 35, 35, 34, 33, 33, 32, 32, 31, 30, 30, 29, 29, 28,
    28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 21, 20, 20, 19, 19, 19, 18, 18, 18,
    17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11,
    10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6,
    5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0,
];

/// Hearing threshold by fscod and band (Table 7.15).
pub(crate) const HTH: [[u16; 50]; 3] = [
    [
        1232, 1232, 1088, 1024, 992, 960, 944, 944, 928, 928, 928, 928, 928, 912, 912, 912, 896,
        896, 880, 880, 864, 864, 848, 848, 832, 832, 816, 800, 784, 768, 752, 752, 752, 752, 768,
        784, 832, 912, 992, 1056, 1120, 1168, 1184, 1120, 1088, 1088, 1312, 2048, 2112, 2112,
    ],
    [
        1264, 1264, 1120, 1040, 992, 976, 960, 944, 944, 928, 928, 928, 928, 928, 912, 912, 912,
        896, 896, 896, 880, 880, 864, 864, 848, 848, 832, 832, 800, 784, 768, 752, 752, 752, 752,
        768, 800, 848, 912, 992, 1056, 1104, 1184, 1168, 1120, 1088, 1152, 1584, 2112, 2112,
    ],
    [
        1408, 1408, 1200, 1104, 1056, 1008, 992, 976, 960, 944, 944, 944, 928, 928, 928, 928, 928,
        928, 928, 928, 912, 912, 912, 912, 896, 896, 896, 880, 864, 848, 832, 816, 800, 784, 768,
        752, 752, 752, 768, 784, 816, 848, 960, 1040, 1136, 1184, 1120, 1088, 1104, 1248,
    ],
];

/// Bit allocation pointer by address (Table 7.16).
pub(crate) const BAPTAB: [u8; 64] = [
    0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
    10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15, 15,
];

/// Words per syncframe by `frmsizecod`, for 48, 44.1 and 32 kHz (Table 5.18).
pub(crate) const FRAME_WORDS: [[u16; 3]; 38] = [
    [64, 69, 96],
    [64, 70, 96],
    [80, 87, 120],
    [80, 88, 120],
    [96, 104, 144],
    [96, 105, 144],
    [112, 121, 168],
    [112, 122, 168],
    [128, 139, 192],
    [128, 140, 192],
    [160, 174, 240],
    [160, 175, 240],
    [192, 208, 288],
    [192, 209, 288],
    [224, 243, 336],
    [224, 244, 336],
    [256, 278, 384],
    [256, 279, 384],
    [320, 348, 480],
    [320, 349, 480],
    [384, 417, 576],
    [384, 418, 576],
    [448, 487, 672],
    [448, 488, 672],
    [512, 557, 768],
    [512, 558, 768],
    [640, 696, 960],
    [640, 697, 960],
    [768, 835, 1152],
    [768, 836, 1152],
    [896, 975, 1344],
    [896, 976, 1344],
    [1024, 1114, 1536],
    [1024, 1115, 1536],
    [1152, 1253, 1728],
    [1152, 1254, 1728],
    [1280, 1393, 1920],
    [1280, 1394, 1920],
];

/// Nominal bit rate in kbit/s by `frmsizecod / 2` (Table 5.18).
pub(crate) const NOMINAL_KBPS: [u16; 19] = [
    32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640,
];

/// Mantissa bits for `bap` 0 to 15 (Table 7.18). Grouped `bap` 1, 2 and 4
/// are zero here: their bits come per group.
pub(crate) const BAP_BITS: [u32; 16] = [0, 0, 0, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16];

/// Quantizer levels of the symmetric `bap` 1 to 5 (Table 7.18).
pub(crate) const BAP_LEVELS: [u32; 6] = [0, 3, 5, 7, 11, 15];

/// High-efficiency bit allocation pointers (Annex E Table E3.1).
pub(crate) const HEBAPTAB: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12,
    13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18,
    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
];

/// Bits per `hebap`: the VQ index for 1 to 7, the mantissa for 8 to 19
/// (Annex E Table E3.2).
pub(crate) const HEBAP_BITS: [u32; 20] = [
    0, 2, 3, 4, 5, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16,
];

/// Exponent strategy: reuse, D15, D25, D45 (Table 7.4).
pub(crate) const REUSE: u8 = 0;
pub(crate) const D15: u8 = 1;
pub(crate) const D25: u8 = 2;
pub(crate) const D45: u8 = 3;

/// Frame exponent strategy combinations (Annex E Table E2.10).
pub(crate) const FRAME_EXPSTR: [[u8; 6]; 32] = [
    [D15, REUSE, REUSE, REUSE, REUSE, REUSE],
    [D15, REUSE, REUSE, REUSE, REUSE, D45],
    [D15, REUSE, REUSE, REUSE, D25, REUSE],
    [D15, REUSE, REUSE, REUSE, D45, D45],
    [D25, REUSE, REUSE, D25, REUSE, REUSE],
    [D25, REUSE, REUSE, D25, REUSE, D45],
    [D25, REUSE, REUSE, D45, D25, REUSE],
    [D25, REUSE, REUSE, D45, D45, D45],
    [D25, REUSE, D15, REUSE, REUSE, REUSE],
    [D25, REUSE, D25, REUSE, REUSE, D45],
    [D25, REUSE, D25, REUSE, D25, REUSE],
    [D25, REUSE, D25, REUSE, D45, D45],
    [D25, REUSE, D45, D25, REUSE, REUSE],
    [D25, REUSE, D45, D25, REUSE, D45],
    [D25, REUSE, D45, D45, D25, REUSE],
    [D25, REUSE, D45, D45, D45, D45],
    [D45, D15, REUSE, REUSE, REUSE, REUSE],
    [D45, D15, REUSE, REUSE, REUSE, D45],
    [D45, D25, REUSE, REUSE, D25, REUSE],
    [D45, D25, REUSE, REUSE, D45, D45],
    [D45, D25, REUSE, D25, REUSE, REUSE],
    [D45, D25, REUSE, D25, REUSE, D45],
    [D45, D25, REUSE, D45, D25, REUSE],
    [D45, D25, REUSE, D45, D45, D45],
    [D45, D45, D15, REUSE, REUSE, REUSE],
    [D45, D45, D25, REUSE, REUSE, D45],
    [D45, D45, D25, REUSE, D25, REUSE],
    [D45, D45, D25, REUSE, D45, D45],
    [D45, D45, D45, D25, REUSE, REUSE],
    [D45, D45, D45, D25, REUSE, D45],
    [D45, D45, D45, D45, D25, REUSE],
    [D45, D45, D45, D45, D45, D45],
];

/// Default coupling band structure by sub-band (Annex E Table E2.12).
pub(crate) const DEFAULT_CPL_BANDS: [bool; 18] = [
    false, false, false, false, false, false, false, false, true, false, true, true, false, true,
    true, true, true, true,
];

/// Default spectral extension band structure by sub-band (Table E2.11).
pub(crate) const DEFAULT_SPX_BANDS: [bool; 17] = [
    false, false, false, false, false, false, false, false, true, false, true, false, true, false,
    true, false, true,
];

/// Spectral extension notch filter gains (Annex E Table E3.14).
pub(crate) const SPX_ATTEN: [[f64; 3]; 32] = [
    [0.954841604, 0.911722489, 0.870550563],
    [0.911722489, 0.831237896, 0.757858283],
    [0.870550563, 0.757858283, 0.659753955],
    [0.831237896, 0.690956440, 0.574349177],
    [0.793700526, 0.629960525, 0.500000000],
    [0.757858283, 0.574349177, 0.435275282],
    [0.723634619, 0.523647061, 0.378929142],
    [0.690956440, 0.477420802, 0.329876978],
    [0.659753955, 0.435275282, 0.287174589],
    [0.629960525, 0.396850263, 0.250000000],
    [0.601512518, 0.361817309, 0.217637641],
    [0.574349177, 0.329876978, 0.189464571],
    [0.548412490, 0.300756259, 0.164938489],
    [0.523647061, 0.274206245, 0.143587294],
    [0.500000000, 0.250000000, 0.125000000],
    [0.477420802, 0.227930622, 0.108818820],
    [0.455861244, 0.207809474, 0.094732285],
    [0.435275282, 0.189464571, 0.082469244],
    [0.415618948, 0.172739110, 0.071793647],
    [0.396850263, 0.157490131, 0.062500000],
    [0.378929142, 0.143587294, 0.054409410],
    [0.361817309, 0.130911765, 0.047366143],
    [0.345478220, 0.119355200, 0.041234622],
    [0.329876978, 0.108818820, 0.035896824],
    [0.314980262, 0.099212566, 0.031250000],
    [0.300756259, 0.090454327, 0.027204705],
    [0.287174589, 0.082469244, 0.023683071],
    [0.274206245, 0.075189065, 0.020617311],
    [0.261823531, 0.068551561, 0.017948412],
    [0.250000000, 0.062500000, 0.015625000],
    [0.238710401, 0.056982656, 0.013602353],
    [0.227930622, 0.051952369, 0.011841536],
];

/// Large-mantissa remapping `a` for a gain of 1, `hebap` 8 to 19
/// (Annex E Table E3.6), as 16-bit fractions.
pub(crate) const GAQ_A1: [i16; 12] = [
    0x1249, 0x0889, 0x0421, 0x0208, 0x0102, 0x0081, 0x0040, 0x0020, 0x0010, 0x0008, 0x0002, 0,
];

/// Remapping `a` and the negative-`x` offset `b` for gains 2 and 4,
/// `hebap` 8 to 16 (Table E3.6). Non-negative `x` takes `b` = 1/2 or 1/4.
pub(crate) const GAQ_A2: [u16; 9] = [
    0xd555, 0xc925, 0xc444, 0xc211, 0xc104, 0xc081, 0xc040, 0xc020, 0xc010,
];
pub(crate) const GAQ_B2: [u16; 9] = [
    0xeaab, 0xd249, 0xc889, 0xc421, 0xc208, 0xc102, 0xc081, 0xc040, 0xc020,
];
pub(crate) const GAQ_A4: [u16; 9] = [
    0xedb7, 0xe666, 0xe319, 0xe186, 0xe0c2, 0xe060, 0xe030, 0xe018, 0xe00c,
];
pub(crate) const GAQ_B4: [u16; 9] = [
    0xfb6e, 0xeccd, 0xe632, 0xe30c, 0xe183, 0xe0c1, 0xe060, 0xe030, 0xe018,
];

/// Center mix level by `cmixlev` (Table 5.9); the reserved code reads as -4.5 dB.
pub(crate) const CMIXLEV: [f32; 4] = [FRAC_1_SQRT_2, 0.594_603_6, 0.5, 0.594_603_6];

/// Surround mix level by `surmixlev` (Table 5.10); the reserved code reads as -6 dB.
pub(crate) const SURMIXLEV: [f32; 4] = [FRAC_1_SQRT_2, 0.5, 0.0, 0.5];

/// Annex D and Annex E extended mix levels by 3-bit code (Tables D2.3 to D2.6).
pub(crate) const EXT_MIXLEV: [f32; 8] = [
    SQRT_2,
    1.189_207_1,
    1.0,
    0.840_896_4,
    FRAC_1_SQRT_2,
    0.594_603_6,
    0.5,
    0.0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_band_maps_its_own_bins() {
        for band in 0..50 {
            let start = usize::from(BNDTAB[band]);
            let end = usize::from(BNDTAB[band + 1]);
            assert!(start < end);
            for (offset, &mapped) in MASKTAB[start..end].iter().enumerate() {
                assert_eq!(usize::from(mapped), band, "bin {}", start + offset);
            }
        }
    }

    #[test]
    fn frame_sizes_follow_the_bit_rate_formula() {
        for (code, words) in FRAME_WORDS.iter().enumerate() {
            let kbps = u32::from(NOMINAL_KBPS[code / 2]);
            assert_eq!(u32::from(words[0]), kbps * 2, "48 kHz code {code}");
            assert_eq!(u32::from(words[2]), kbps * 3, "32 kHz code {code}");
            let base = kbps * 1536 * 1000 / 44_100 / 16;
            assert_eq!(
                u32::from(words[1]),
                base + (code as u32 & 1),
                "44.1 kHz code {code}"
            );
        }
    }

    #[test]
    fn the_remap_constants_scale_gain_one_quantizers_to_full_scale() {
        for (index, &a) in GAQ_A1.iter().enumerate() {
            let bits = HEBAP_BITS[index + 8];
            let half = (1u32 << (bits - 1)) as f64;
            let largest = (half - 1.0) / half;
            let remapped = largest * (1.0 + f64::from(a) / 32768.0);
            assert!(
                (remapped - 1.0 + 1.0 / ((1u32 << bits) - 1) as f64).abs() < 2e-4,
                "hebap {}",
                index + 8
            );
        }
    }

    #[test]
    fn the_high_efficiency_pointers_never_fall() {
        assert!(HEBAPTAB.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(HEBAPTAB[63], 19);
    }
}
