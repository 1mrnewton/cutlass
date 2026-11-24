//! The kernels built for AVX2, with the BMI1, BMI2, LZCNT and POPCNT its CPUs have.

/// Marks each function as built for this tier's CPU features.
macro_rules! feat {
    ($($item:item)*) => {
        $(
            #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
            $item
        )*
    };
}

#[path = "kernels.rs"]
mod kernels;
#[path = "wide_avx2.rs"]
mod wide;

pub(crate) use kernels::{
    avg_block, avg_row, avg8, avg16, centre8, centre16, chroma_block, chroma_edge, chroma_mb,
    chroma_normal, chroma_pair, chroma_planes, chroma_strong, chroma8, dc_add, dc_add_quads, half8,
    half16, idct4_add, idct8_add, luma_block, luma_edge, luma_mb, luma_normal_lines,
    luma_normal_rows, luma_pair, luma_strong_lines, luma_strong_rows, mb_planes, mb_twin,
    pred_chroma, pred4, pred8, pred16, predict_420, prefetch, prefetch_item, raw8, raw16,
    scale_block, store, weight_block, weight_onto,
};
