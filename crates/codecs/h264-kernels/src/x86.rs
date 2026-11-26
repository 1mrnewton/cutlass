//! Which x86-64 kernels this CPU runs, found once and kept.
//!
//! The vector kernels are in `x86/kernels.rs`, one source built twice: for
//! SSE4.1 (with SSE2 and SSSE3, which every x86-64 CPU since 2008 has) in
//! [`sse41`], and for AVX2 in [`avx2`]. Most kernels work on at most
//! sixteen samples, a 128-bit register; the AVX2 build encodes the same
//! steps with three operands, and its sixteen-wide luma predictors, in
//! `x86/wide_avx2.rs`, hold a row's sixteen 16-bit sums in one 256-bit
//! register. A CPU with neither runs [`crate::scalar`].
//!
//! Each tier's functions carry `#[target_feature]`, which the `feat!`
//! wrapper in the tier's module adds to every function of the shared
//! source. [`level`] is the run-time check that makes calling them sound.

use core::sync::atomic::{AtomicU8, Ordering};

/// The kernels a CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// [`crate::scalar`].
    Scalar,
    /// [`sse41`].
    Sse41,
    /// [`avx2`].
    Avx2,
}

#[cfg(test)]
thread_local! {
    /// A tier a test asks for on its own thread, in place of the CPU's
    /// best.
    static FORCED: core::cell::Cell<Option<Level>> = const { core::cell::Cell::new(None) };
}

/// The best kernels this CPU has. Detected on the first call.
#[inline]
pub(crate) fn level() -> Level {
    #[cfg(test)]
    if let Some(forced) = FORCED.with(core::cell::Cell::get) {
        return forced;
    }
    detected()
}

#[inline]
fn detected() -> Level {
    /// 0: not checked yet, then one more than the level's place.
    static FOUND: AtomicU8 = AtomicU8::new(0);
    match FOUND.load(Ordering::Relaxed) {
        1 => Level::Scalar,
        2 => Level::Sse41,
        3 => Level::Avx2,
        _ => {
            let level = detect();
            let code = match level {
                Level::Scalar => 1,
                Level::Sse41 => 2,
                Level::Avx2 => 3,
            };
            FOUND.store(code, Ordering::Relaxed);
            level
        }
    }
}

#[cold]
fn detect() -> Level {
    let sse41 = std::is_x86_feature_detected!("sse2")
        && std::is_x86_feature_detected!("ssse3")
        && std::is_x86_feature_detected!("sse4.1");
    if !sse41 {
        return Level::Scalar;
    }
    let avx2 = std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("bmi1")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("lzcnt")
        && std::is_x86_feature_detected!("popcnt");
    if avx2 { Level::Avx2 } else { Level::Sse41 }
}

/// Runs `body` once on each vector tier this CPU has, so a test of a
/// kernel against its plain twin covers the AVX2 version and the SSE4.1
/// one on a machine with both. A CPU with neither runs it once, on the
/// plain kernels.
#[cfg(test)]
pub(crate) fn each_tier(body: fn()) {
    let best = detected();
    if best == Level::Scalar {
        return body();
    }
    for tier in [Level::Sse41, Level::Avx2] {
        if tier == Level::Sse41 || best == Level::Avx2 {
            FORCED.with(|forced| forced.set(Some(tier)));
            body();
        }
    }
    FORCED.with(|forced| forced.set(None));
}

/// One row loop of a luma predictor: `$row` is the row's samples, made
/// from `$p`, the row's first full sample.
macro_rules! x86_luma_rows {
    ($put:ident, $avg:ident, ($dst:ident, $ds:ident, $src:ident, $s:ident, $h:ident),
     |$p:ident| $row:expr) => {
        for row in 0..$h {
            let $p = $src.add(row * $s);
            $put::<$avg>($dst.add(row * $ds), $row);
        }
    };
}

/// A luma predictor for one block width at every quarter position, over
/// the row pieces named: `$full` loads a row of full samples, `$rawh` and
/// `$rawv` are the six-tap sums along a row and down a column, `$round`
/// makes half samples of sums, `$centre` makes centre samples of six rows
/// of sums, and `$put` stores a row or averages it into what is there.
/// `$v` is the type of a row of sums.
macro_rules! x86_luma {
    ($name:ident, $v:ty, $full:ident, $rawh:ident, $rawv:ident, $round:ident, $centre:ident,
     $put:ident) => {
        feat! {
        /// The block of `h` rows whose first full sample is `src`, rows
        /// `s` apart, at quarter offsets `frac`, written to `dst`, rows
        /// `ds` apart, or averaged into it when `AVG`.
        ///
        /// # Safety
        ///
        /// Every byte `luma_reach` checks for these offsets and this size
        /// must be readable from `src`, and the block's rows writable at
        /// `dst`.
        pub(crate) unsafe fn $name<const AVG: bool>(
            dst: *mut u8,
            ds: usize,
            src: *const u8,
            s: usize,
            h: usize,
            frac: (u8, u8),
        ) {
            // SAFETY: each arm reads the rows and columns its offsets
            // need and no more: two before and three after the block in a
            // direction with an offset, which the caller promises.
            unsafe {
                match (frac.0 & 3, frac.1 & 3) {
                    (0, 0) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| $full(p)),
                    (1, 0) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($full(p), $round($rawh(p)))
                    }),
                    (2, 0) => {
                        x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| $round($rawh(p)))
                    }
                    (3, 0) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($round($rawh(p)), $full(p.add(1)))
                    }),
                    (0, 1) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($full(p), $round($rawv(p, s)))
                    }),
                    (0, 2) => {
                        x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| $round($rawv(p, s)))
                    }
                    (0, 3) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($round($rawv(p, s)), $full(p.add(s)))
                    }),
                    (1, 1) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($round($rawh(p)), $round($rawv(p, s)))
                    }),
                    (3, 1) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($round($rawh(p)), $round($rawv(p.add(1), s)))
                    }),
                    (1, 3) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($round($rawv(p, s)), $round($rawh(p.add(s))))
                    }),
                    (3, 3) => x86_luma_rows!($put, AVG, (dst, ds, src, s, h), |p| {
                        _mm_avg_epu8($round($rawv(p.add(1), s)), $round($rawh(p.add(s))))
                    }),
                    (xf, yf) => {
                        // A centre sample is the column filter over the
                        // row sums of six rows: the five before a row's
                        // last tap are kept from one row to the next.
                        let first = $rawh(src.sub(2 * s));
                        let mut sums: [$v; 6] = [first; 6];
                        sums[2] = $rawh(src.sub(s));
                        sums[3] = $rawh(src);
                        sums[4] = $rawh(src.add(s));
                        sums[5] = $rawh(src.add(2 * s));
                        for row in 0..h {
                            let p = src.add(row * s);
                            sums = [sums[1], sums[2], sums[3], sums[4], sums[5], $rawh(p.add(3 * s))];
                            let centre = $centre(&sums);
                            let made = match (xf, yf) {
                                (2, 1) => _mm_avg_epu8($round(sums[2]), centre),
                                (2, 3) => _mm_avg_epu8(centre, $round(sums[3])),
                                (1, 2) => _mm_avg_epu8($round($rawv(p, s)), centre),
                                (3, 2) => _mm_avg_epu8(centre, $round($rawv(p.add(1), s))),
                                _ => centre,
                            };
                            $put::<AVG>(dst.add(row * ds), made);
                        }
                    }
                }
            }
        }
        }
    };
}

pub(crate) mod avx2;
pub(crate) mod sse41;

#[cfg(test)]
mod tests {
    use super::{Level, detected, each_tier, level};
    use core::cell::RefCell;

    thread_local! {
        static SEEN: RefCell<Vec<Level>> = const { RefCell::new(Vec::new()) };
    }

    fn note() {
        SEEN.with(|seen| seen.borrow_mut().push(level()));
    }

    #[test]
    fn each_tier_runs_every_vector_version_this_cpu_has() {
        each_tier(note);
        let seen = SEEN.with(RefCell::take);
        let want: &[Level] = match detected() {
            Level::Avx2 => &[Level::Sse41, Level::Avx2],
            Level::Sse41 => &[Level::Sse41],
            Level::Scalar => &[Level::Scalar],
        };
        assert_eq!(seen, want);
        // Afterwards the CPU's best is back.
        assert_eq!(level(), detected());
    }

    #[test]
    fn detection_gives_the_same_answer_every_time() {
        let first = detected();
        assert!((0..100).all(|_| detected() == first));
        // The best tier is the one the CPU's features allow.
        let sse41 =
            std::is_x86_feature_detected!("ssse3") && std::is_x86_feature_detected!("sse4.1");
        let avx2 = sse41
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("popcnt");
        let want = match (sse41, avx2) {
            (_, true) => Level::Avx2,
            (true, false) => Level::Sse41,
            (false, false) => Level::Scalar,
        };
        assert_eq!(first, want);
    }
}
