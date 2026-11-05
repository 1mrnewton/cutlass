//! The full-spec AV1 decoder.

pub mod bits;
pub mod cdf;
pub mod coder;
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
    /// Area (in 4x4 units) coded with each block size, luma mode and transform size.
    pub static BSIZE: [AtomicU64; 22] = [const { AtomicU64::new(0) }; 22];
    pub static YMODE: [AtomicU64; 32] = [const { AtomicU64::new(0) }; 32];
    pub static TXSIZE: [AtomicU64; 19] = [const { AtomicU64::new(0) }; 19];
    /// Luma area with each transform type (coded blocks only).
    pub static TXTYPE: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    /// One line per coded frame: order hint, q index and size.
    pub static FRAMES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    pub fn frame(s: String) {
        if let Ok(mut v) = FRAMES.lock() {
            v.push(s);
        }
    }
    pub static SKIP: AtomicU64 = AtomicU64::new(0);
    pub static QIDX: AtomicU64 = AtomicU64::new(0);
    pub fn add(a: &[AtomicU64], i: usize, v: u64) {
        if let Some(c) = a.get(i) {
            c.fetch_add(v, Ordering::Relaxed);
        }
    }
    fn hist(name: &str, a: &[AtomicU64]) -> String {
        let tot: u64 = a
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum::<u64>()
            .max(1);
        let mut s = format!("\n{name}:");
        for (i, c) in a.iter().enumerate() {
            let v = c.load(Ordering::Relaxed);
            if v > 0 {
                s.push_str(&format!(" {i}:{:.1}%", v as f64 * 100.0 / tot as f64));
            }
        }
        s
    }
    pub fn report() -> String {
        let mut s = String::new();
        s.push_str(&hist("bsize", &BSIZE));
        s.push_str(&hist("ymode", &YMODE));
        s.push_str(&hist("txsize", &TXSIZE));
        s.push_str(&hist("txtype", &TXTYPE));
        if let Ok(v) = FRAMES.lock() {
            s.push_str("\nframes: ");
            s.push_str(&v.join(" "));
        }
        s.push_str(&format!(
            "\nskip_area={} last_qidx={}\n",
            SKIP.load(Ordering::Relaxed),
            QIDX.load(Ordering::Relaxed)
        ));
        for (i, n) in NAMES.iter().enumerate() {
            let v = COUNTS[i].load(Ordering::Relaxed);
            if v > 0 {
                s.push_str(&format!("{n}={v} "));
            }
        }
        s
    }
}
