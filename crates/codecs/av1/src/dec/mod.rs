//! The full-spec AV1 decoder.

pub mod bits;
pub mod cdf;
pub mod decoder;
pub mod frame;
pub mod inter;
pub mod msac;
pub mod mvpred;
pub mod obu;
pub mod postfilter;
pub mod recon;
pub mod tile;
pub mod types;

/// Tool usage counters, compiled in with the `stats` feature.
#[cfg(feature = "stats")]
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub const NAMES: [&str; 16] = [
        "intra",
        "inter",
        "compound",
        "wedge",
        "diffwtd",
        "distance",
        "interintra",
        "obmc",
        "localwarp",
        "globalwarp",
        "scaled",
        "palette",
        "intrabc",
        "filter_intra",
        "cfl",
        "lossless",
    ];
    pub static COUNTS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    pub fn hit(i: usize) {
        COUNTS[i].fetch_add(1, Ordering::Relaxed);
    }
    pub fn report() -> String {
        let mut s = String::new();
        for (i, n) in NAMES.iter().enumerate() {
            let v = COUNTS[i].load(Ordering::Relaxed);
            if v > 0 {
                s.push_str(&format!("{n}={v} "));
            }
        }
        s
    }
}
