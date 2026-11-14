//! The 8-bit loop filter on AVX2, sixteen lines at a time in 16-bit
//! lanes: tap `k` of every line is one vector. A horizontal edge loads its
//! rows straight into those vectors; a vertical edge loads its lines and
//! transposes them, and back.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_and_si128, _mm_cmpeq_epi8, _mm_cmpgt_epi8, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_or_si128, _mm_set_epi64x, _mm_set1_epi8, _mm_setzero_si128,
    _mm_srli_epi16, _mm_storel_epi64, _mm_storeu_si128, _mm_subs_epu8, _mm_unpackhi_epi8,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_loadu_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64,
    _mm256_setzero_si256, _mm256_storeu_si256,
};

use super::{Thresh, Wd};

/// Sixteen 16-bit lanes to sixteen bytes.
#[inline]
#[target_feature(enable = "avx2")]
fn pack(v: __m256i) -> __m128i {
    let p = _mm256_packus_epi16(v, v);
    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(p))
}

/// The transpose of a 16×16 block of bytes.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(r: &[__m128i; 16]) -> [__m128i; 16] {
    let z = _mm_setzero_si128();
    let mut a = [z; 16];
    for i in 0..8 {
        a[2 * i] = _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]);
        a[2 * i + 1] = _mm_unpackhi_epi8(r[2 * i], r[2 * i + 1]);
    }
    let mut b = [z; 16];
    for j in 0..4 {
        b[4 * j] = _mm_unpacklo_epi16(a[4 * j], a[4 * j + 2]);
        b[4 * j + 1] = _mm_unpackhi_epi16(a[4 * j], a[4 * j + 2]);
        b[4 * j + 2] = _mm_unpacklo_epi16(a[4 * j + 1], a[4 * j + 3]);
        b[4 * j + 3] = _mm_unpackhi_epi16(a[4 * j + 1], a[4 * j + 3]);
    }
    let mut c = [z; 16];
    for g in 0..2 {
        for q in 0..4 {
            c[8 * g + 2 * q] = _mm_unpacklo_epi32(b[8 * g + q], b[8 * g + 4 + q]);
            c[8 * g + 2 * q + 1] = _mm_unpackhi_epi32(b[8 * g + q], b[8 * g + 4 + q]);
        }
    }
    let mut o = [z; 16];
    for q in 0..4 {
        o[4 * q] = _mm_unpacklo_epi64(c[2 * q], c[8 + 2 * q]);
        o[4 * q + 1] = _mm_unpackhi_epi64(c[2 * q], c[8 + 2 * q]);
        o[4 * q + 2] = _mm_unpacklo_epi64(c[2 * q + 1], c[8 + 2 * q + 1]);
        o[4 * q + 3] = _mm_unpackhi_epi64(c[2 * q + 1], c[8 + 2 * q + 1]);
    }
    o
}

/// Bytes `lo` in lanes 0 to 7 and `hi` in lanes 8 to 15.
#[inline]
#[target_feature(enable = "avx2")]
fn split8(lo: u8, hi: u8) -> __m128i {
    _mm_set_epi64x(i64::from_le_bytes([hi; 8]), i64::from_le_bytes([lo; 8]))
}

/// All ones in the lanes of the halves `on` picks (0 to 7, 8 to 15).
#[inline]
#[target_feature(enable = "avx2")]
fn halves8(on: [bool; 2]) -> __m128i {
    split8(if on[0] { 0xff } else { 0 }, if on[1] { 0xff } else { 0 })
}

#[inline]
#[target_feature(enable = "avx2")]
fn absd(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

/// All ones where `x <= t` (unsigned bytes).
#[inline]
#[target_feature(enable = "avx2")]
fn le(x: __m128i, t: __m128i) -> __m128i {
    _mm_cmpeq_epi8(_mm_subs_epu8(x, t), _mm_setzero_si128())
}

/// Signed bytes shifted right by `N` (3 or 1), as libvpx's SSE2 does it.
#[inline]
#[target_feature(enable = "avx2")]
fn sra8<const N: i32>(x: __m128i) -> __m128i {
    // The high bits the logical 16-bit shift lets in from the next byte
    // are masked off, and the sign bits put back.
    let neg = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
    let keep = (0xffu8 >> N) as i8;
    let sign = !keep;
    let shifted = _mm_and_si128(_mm_srli_epi16::<N>(x), _mm_set1_epi8(keep));
    _mm_or_si128(shifted, _mm_and_si128(neg, _mm_set1_epi8(sign)))
}

/// Sixteen bytes to 16-bit lanes and back.
#[inline]
#[target_feature(enable = "avx2")]
fn wide16(x: __m128i) -> __m256i {
    _mm256_cvtepu8_epi16(x)
}

/// What a filter changed: nothing, taps p2 to q2, or p6 to q6.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Changed {
    None,
    Inner,
    Outer,
}

/// Whether a half's filter reaches the 7-tap one.
#[inline(always)]
fn big(w: Option<Wd>) -> bool {
    matches!(w, Some(Wd::W8 | Wd::W16))
}

/// The filter's operations on 16 lines (one edge) a vector, and on its
/// taps widened to 16 bits.
#[allow(dead_code)]
mod x16 {
    use super::{Thresh, Wd};
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_adds_epi8, _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128,
        _mm_blendv_epi8, _mm_max_epu8, _mm_or_si128, _mm_set1_epi8, _mm_setzero_si128,
        _mm_srli_epi16, _mm_subs_epi8, _mm_testz_si128, _mm_xor_si128, _mm256_add_epi16,
        _mm256_set1_epi16, _mm256_slli_epi16, _mm256_srli_epi16, _mm256_sub_epi16,
    };

    pub type V = __m128i;
    pub type W = __m256i;

    macro_rules! two {
        ($($f:ident => $i:ident),*) => {$(
            #[inline]
            #[target_feature(enable = "avx2")]
            pub fn $f(a: V, b: V) -> V {
                $i(a, b)
            }
        )*};
    }
    two!(max_u8 => _mm_max_epu8, and => _mm_and_si128, or => _mm_or_si128,
        xor => _mm_xor_si128, andnot => _mm_andnot_si128, adds_u8 => _mm_adds_epu8,
        subs_i8 => _mm_subs_epi8, adds_i8 => _mm_adds_epi8);

    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn zero() -> V {
        _mm_setzero_si128()
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn splat(b: i8) -> V {
        _mm_set1_epi8(b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn blendv(a: V, b: V, m: V) -> V {
        _mm_blendv_epi8(a, b, m)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn srli16<const N: i32>(a: V) -> V {
        _mm_srli_epi16::<N>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn none(a: V) -> bool {
        _mm_testz_si128(a, a) != 0
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn absd(a: V, b: V) -> V {
        super::absd(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn le(x: V, t: V) -> V {
        super::le(x, t)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn sra8<const N: i32>(x: V) -> V {
        super::sra8::<N>(x)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn thresholds(t: &[Thresh; 2]) -> (V, V, V) {
        (
            super::split8(t[0].limit, t[1].limit),
            super::split8(t[0].blimit, t[1].blimit),
            super::split8(t[0].hev, t[1].hev),
        )
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn masks_some(w: &[Option<Wd>; 2]) -> V {
        super::halves8([w[0].is_some(), w[1].is_some()])
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn masks_big(w: &[Option<Wd>; 2]) -> V {
        super::halves8([super::big(w[0]), super::big(w[1])])
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn masks16(w: &[Option<Wd>; 2]) -> V {
        super::halves8([w[0] == Some(Wd::W16), w[1] == Some(Wd::W16)])
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wide(x: V) -> W {
        super::wide16(x)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn pack(w: W) -> V {
        super::pack(w)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wadd(a: W, b: W) -> W {
        _mm256_add_epi16(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wsub(a: W, b: W) -> W {
        _mm256_sub_epi16(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wset1(k: i16) -> W {
        _mm256_set1_epi16(k)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wsrli<const N: i32>(a: W) -> W {
        _mm256_srli_epi16::<N>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wslli<const N: i32>(a: W) -> W {
        _mm256_slli_epi16::<N>(a)
    }
}

/// [`x16`]'s operations on two edges a vector, one in each 128-bit half
/// (four groups of eight lines); the taps widen to pairs of vectors, each
/// half's low and high eight bytes, which keeps every half in order.
#[allow(dead_code)]
mod x32 {
    use super::{Thresh, Wd};
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi16, _mm256_adds_epi8, _mm256_adds_epu8, _mm256_and_si256,
        _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8,
        _mm256_max_epu8, _mm256_or_si256, _mm256_packus_epi16, _mm256_set_epi64x, _mm256_set1_epi8,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srli_epi16,
        _mm256_sub_epi16, _mm256_subs_epi8, _mm256_subs_epu8, _mm256_testz_si256,
        _mm256_unpackhi_epi8, _mm256_unpacklo_epi8, _mm256_xor_si256,
    };

    pub type V = __m256i;
    pub type W = (__m256i, __m256i);

    macro_rules! two {
        ($($f:ident => $i:ident),*) => {$(
            #[inline]
            #[target_feature(enable = "avx2")]
            pub fn $f(a: V, b: V) -> V {
                $i(a, b)
            }
        )*};
    }
    two!(max_u8 => _mm256_max_epu8, and => _mm256_and_si256, or => _mm256_or_si256,
        xor => _mm256_xor_si256, andnot => _mm256_andnot_si256,
        adds_u8 => _mm256_adds_epu8, subs_i8 => _mm256_subs_epi8,
        adds_i8 => _mm256_adds_epi8);

    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn zero() -> V {
        _mm256_setzero_si256()
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn splat(b: i8) -> V {
        _mm256_set1_epi8(b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn blendv(a: V, b: V, m: V) -> V {
        _mm256_blendv_epi8(a, b, m)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn srli16<const N: i32>(a: V) -> V {
        _mm256_srli_epi16::<N>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn none(a: V) -> bool {
        _mm256_testz_si256(a, a) != 0
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn absd(a: V, b: V) -> V {
        _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn le(x: V, t: V) -> V {
        _mm256_cmpeq_epi8(_mm256_subs_epu8(x, t), _mm256_setzero_si256())
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn sra8<const N: i32>(x: V) -> V {
        let neg = _mm256_cmpgt_epi8(_mm256_setzero_si256(), x);
        let keep = (0xffu8 >> N) as i8;
        let shifted = _mm256_and_si256(_mm256_srli_epi16::<N>(x), _mm256_set1_epi8(keep));
        _mm256_or_si256(shifted, _mm256_and_si256(neg, _mm256_set1_epi8(!keep)))
    }
    /// Bytes `g[i]` in the lanes of group `i` (eight lanes each).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn groups(g: [u8; 4]) -> V {
        let q = |b: u8| i64::from_le_bytes([b; 8]);
        _mm256_set_epi64x(q(g[3]), q(g[2]), q(g[1]), q(g[0]))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    fn on(b: bool) -> u8 {
        if b { 0xff } else { 0 }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn thresholds(t: &[Thresh; 4]) -> (V, V, V) {
        (
            groups([t[0].limit, t[1].limit, t[2].limit, t[3].limit]),
            groups([t[0].blimit, t[1].blimit, t[2].blimit, t[3].blimit]),
            groups([t[0].hev, t[1].hev, t[2].hev, t[3].hev]),
        )
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn masks_some(w: &[Option<Wd>; 4]) -> V {
        groups([
            on(w[0].is_some()),
            on(w[1].is_some()),
            on(w[2].is_some()),
            on(w[3].is_some()),
        ])
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn masks_big(w: &[Option<Wd>; 4]) -> V {
        let b = super::big;
        groups([on(b(w[0])), on(b(w[1])), on(b(w[2])), on(b(w[3]))])
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn masks16(w: &[Option<Wd>; 4]) -> V {
        let b = |x: Option<Wd>| x == Some(Wd::W16);
        groups([on(b(w[0])), on(b(w[1])), on(b(w[2])), on(b(w[3]))])
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wide(x: V) -> W {
        let z = _mm256_setzero_si256();
        (_mm256_unpacklo_epi8(x, z), _mm256_unpackhi_epi8(x, z))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn pack(w: W) -> V {
        _mm256_packus_epi16(w.0, w.1)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wadd(a: W, b: W) -> W {
        (_mm256_add_epi16(a.0, b.0), _mm256_add_epi16(a.1, b.1))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wsub(a: W, b: W) -> W {
        (_mm256_sub_epi16(a.0, b.0), _mm256_sub_epi16(a.1, b.1))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wset1(k: i16) -> W {
        (_mm256_set1_epi16(k), _mm256_set1_epi16(k))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wsrli<const N: i32>(a: W) -> W {
        (_mm256_srli_epi16::<N>(a.0), _mm256_srli_epi16::<N>(a.1))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn wslli<const N: i32>(a: W) -> W {
        (_mm256_slli_epi16::<N>(a.0), _mm256_slli_epi16::<N>(a.1))
    }
}

macro_rules! lanes_fn {
    ($name:ident, $m:ident, $g:literal) => {
        #[inline]
        #[target_feature(enable = "avx2")]
        fn $name(
            v: &mut [$m::V; 16],
            wds: [Option<Wd>; $g],
            wd: Wd,
            thr: &[Thresh; $g],
            outer: impl FnOnce(&mut [$m::V; 16]),
        ) -> Changed {
            let (p3, p2, p1, p0) = (v[4], v[5], v[6], v[7]);
            let (q0, q1, q2, q3) = (v[8], v[9], v[10], v[11]);
            let (limit, blimit, hevt) = $m::thresholds(thr);
            let ap = $m::absd(p1, p0);
            let aq = $m::absd(q1, q0);
            let m = $m::max_u8(
                $m::max_u8(
                    $m::max_u8($m::absd(p3, p2), $m::absd(p2, p1)),
                    $m::max_u8(ap, aq),
                ),
                $m::max_u8($m::absd(q2, q1), $m::absd(q3, q2)),
            );
            let apq0 = $m::absd(p0, q0);
            let apq1 = $m::and($m::srli16::<1>($m::absd(p1, q1)), $m::splat(0x7f));
            let e = $m::adds_u8($m::adds_u8(apq0, apq0), apq1);
            let on = $m::masks_some(&wds);
            let mask = $m::and($m::and($m::le(m, limit), $m::le(e, blimit)), on);
            if $m::none(mask) {
                return Changed::None;
            }
            let hev = $m::xor($m::le($m::max_u8(ap, aq), hevt), $m::splat(-1));
            // filter4, in signed bytes.
            let x80 = $m::splat(-128);
            let (ps1, ps0) = ($m::xor(p1, x80), $m::xor(p0, x80));
            let (qs0, qs1) = ($m::xor(q0, x80), $m::xor(q1, x80));
            let f = $m::and($m::subs_i8(ps1, qs1), hev);
            let d = $m::subs_i8(qs0, ps0);
            let f = $m::adds_i8($m::adds_i8($m::adds_i8(f, d), d), d);
            let f = $m::and(f, mask);
            let f1 = $m::sra8::<3>($m::adds_i8(f, $m::splat(4)));
            let f2 = $m::sra8::<3>($m::adds_i8(f, $m::splat(3)));
            let nq0 = $m::xor($m::subs_i8(qs0, f1), x80);
            let np0 = $m::xor($m::adds_i8(ps0, f2), x80);
            let fo = $m::andnot(hev, $m::sra8::<1>($m::adds_i8(f1, $m::splat(1))));
            let nq1 = $m::xor($m::subs_i8(qs1, fo), x80);
            let np1 = $m::xor($m::adds_i8(ps1, fo), x80);
            v[6] = np1;
            v[7] = np0;
            v[8] = nq0;
            v[9] = nq1;
            if wd == Wd::W4 {
                return Changed::Inner;
            }
            let one = $m::splat(1);
            let fm = $m::max_u8(
                $m::max_u8(
                    $m::max_u8(ap, aq),
                    $m::max_u8($m::absd(p2, p0), $m::absd(q2, q0)),
                ),
                $m::max_u8($m::absd(p3, p0), $m::absd(q3, q0)),
            );
            let w8 = $m::masks_big(&wds);
            let flat = $m::and($m::and($m::le(fm, one), mask), w8);
            if $m::none(flat) {
                return Changed::Inner;
            }
            // Where the 15-tap filter runs it replaces the 7-tap one.
            let flat2 = if wd == Wd::W16 {
                outer(v);
                let f2m = $m::max_u8(
                    $m::max_u8(
                        $m::max_u8($m::absd(v[3], p0), $m::absd(v[2], p0)),
                        $m::max_u8($m::absd(v[1], p0), $m::absd(v[0], p0)),
                    ),
                    $m::max_u8(
                        $m::max_u8($m::absd(v[12], q0), $m::absd(v[13], q0)),
                        $m::max_u8($m::absd(v[14], q0), $m::absd(v[15], q0)),
                    ),
                );
                let w16 = $m::masks16(&wds);
                $m::and($m::and($m::le(f2m, one), flat), w16)
            } else {
                $m::zero()
            };
            let flat_only = $m::andnot(flat2, flat);
            if !$m::none(flat_only) {
                let flat = flat_only;
                // The 7-tap sums in 16-bit lanes.
                let (w3, w2, w1, w0) = ($m::wide(p3), $m::wide(p2), $m::wide(p1), $m::wide(p0));
                let (x0, x1, x2, x3) = ($m::wide(q0), $m::wide(q1), $m::wide(q2), $m::wide(q3));
                let add = |a: $m::W, b: $m::W| $m::wadd(a, b);
                let r3 = |x: $m::W| $m::pack($m::wsrli::<3>(add(x, $m::wset1(4))));
                let o2 = r3(add(
                    add(add(w3, w3), add(w3, w2)),
                    add(add(w2, w1), add(w0, x0)),
                ));
                let o1 = r3(add(
                    add(add(w3, w3), add(w2, w1)),
                    add(add(w1, w0), add(x0, x1)),
                ));
                let o0 = r3(add(
                    add(add(w3, w2), add(w1, w0)),
                    add(add(w0, x0), add(x1, x2)),
                ));
                let oq0 = r3(add(
                    add(add(w2, w1), add(w0, x0)),
                    add(add(x0, x1), add(x2, x3)),
                ));
                let oq1 = r3(add(
                    add(add(w1, w0), add(x0, x1)),
                    add(add(x1, x2), add(x3, x3)),
                ));
                let oq2 = r3(add(
                    add(add(w0, x0), add(x1, x2)),
                    add(add(x2, x3), add(x3, x3)),
                ));
                v[5] = $m::blendv(p2, o2, flat);
                v[6] = $m::blendv(np1, o1, flat);
                v[7] = $m::blendv(np0, o0, flat);
                v[8] = $m::blendv(nq0, oq0, flat);
                v[9] = $m::blendv(nq1, oq1, flat);
                v[10] = $m::blendv(q2, oq2, flat);
            }
            if $m::none(flat2) {
                return Changed::Inner;
            }
            let s: [$m::W; 16] = [
                $m::wide(v[0]),
                $m::wide(v[1]),
                $m::wide(v[2]),
                $m::wide(v[3]),
                $m::wide(p3),
                $m::wide(p2),
                $m::wide(p1),
                $m::wide(p0),
                $m::wide(q0),
                $m::wide(q1),
                $m::wide(q2),
                $m::wide(q3),
                $m::wide(v[12]),
                $m::wide(v[13]),
                $m::wide(v[14]),
                $m::wide(v[15]),
            ];
            let add = |a: $m::W, b: $m::W| $m::wadd(a, b);
            // The 15-tap window as a running sum, as libvpx writes it out.
            let mut sum = add(
                add($m::wslli::<3>(s[0]), $m::wsub(s[1], s[0])),
                add(
                    add(s[1], s[2]),
                    add(add(s[3], s[4]), add(add(s[5], s[6]), add(s[7], s[8]))),
                ),
            );
            let r4 = |x: $m::W| $m::pack($m::wsrli::<4>(add(x, $m::wset1(8))));
            let mut out = [v[0]; 14];
            out[0] = r4(sum);
            for k in 1usize..14 {
                let drop_a = s[k.saturating_sub(7)];
                let drop_b = s[k];
                let add_a = s[k + 1];
                let add_b = s[(k + 8).min(15)];
                sum = $m::wsub(add(sum, add(add_a, add_b)), add(drop_a, drop_b));
                out[k] = r4(sum);
            }
            for k in 0..14 {
                v[k + 1] = $m::blendv(v[k + 1], out[k], flat2);
            }
            Changed::Outer
        }
    };
}

// `lanes` filters the taps `v` (p7 at 0, q0 at 8; sixteen lines a vector)
// in place, each half of the lines with its own filter (libvpx's SSE2
// arithmetic, which equals its C). `v` comes with p3 to q3; `outer` fills
// p7 to p4 and q4 to q7 when the 15-tap filter may run.
lanes_fn!(lanes, x16, 2);

// `lanes32` is `lanes` for two edges side by side, each half of a 256-bit
// vector one edge (the instructions keep to their 128-bit halves): four
// groups of eight lines, each with its own filter.
lanes_fn!(lanes32, x32, 4);

/// The transpose of 16 rows of 8 bytes (the low half of each): 8
/// vectors of 16, one per column.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose16x8(r: &[__m128i; 16]) -> [__m128i; 8] {
    let z = _mm_setzero_si128();
    let a: [__m128i; 8] = core::array::from_fn(|i| _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]));
    let mut b = [z; 8];
    for j in 0..4 {
        b[2 * j] = _mm_unpacklo_epi16(a[2 * j], a[2 * j + 1]);
        b[2 * j + 1] = _mm_unpackhi_epi16(a[2 * j], a[2 * j + 1]);
    }
    let mut c = [z; 8];
    for g in 0..2 {
        for q in 0..2 {
            c[4 * g + 2 * q] = _mm_unpacklo_epi32(b[4 * g + q], b[4 * g + 2 + q]);
            c[4 * g + 2 * q + 1] = _mm_unpackhi_epi32(b[4 * g + q], b[4 * g + 2 + q]);
        }
    }
    let mut o = [z; 8];
    for q in 0..2 {
        for t in 0..2 {
            o[4 * q + 2 * t] = _mm_unpacklo_epi64(c[2 * q + t], c[4 + 2 * q + t]);
            o[4 * q + 2 * t + 1] = _mm_unpackhi_epi64(c[2 * q + t], c[4 + 2 * q + t]);
        }
    }
    o
}

/// The transpose back: 8 columns of 16 bytes to 16 rows of 8 (each row's
/// 8 bytes in the low half, then the high half, of the 8 vectors: row
/// `2i` in the low half of vector `i`, row `2i + 1` in its high half).
#[inline]
#[target_feature(enable = "avx2")]
fn transpose8x16(c: &[__m128i; 8]) -> [__m128i; 8] {
    let z = _mm_setzero_si128();
    let mut p = [z; 8];
    for k in 0..4 {
        p[2 * k] = _mm_unpacklo_epi8(c[2 * k], c[2 * k + 1]);
        p[2 * k + 1] = _mm_unpackhi_epi8(c[2 * k], c[2 * k + 1]);
    }
    let mut q = [z; 8];
    for h in 0..2 {
        q[4 * h] = _mm_unpacklo_epi16(p[h], p[2 + h]);
        q[4 * h + 1] = _mm_unpackhi_epi16(p[h], p[2 + h]);
        q[4 * h + 2] = _mm_unpacklo_epi16(p[4 + h], p[6 + h]);
        q[4 * h + 3] = _mm_unpackhi_epi16(p[4 + h], p[6 + h]);
    }
    let mut o = [z; 8];
    for h in 0..2 {
        for k in 0..2 {
            o[4 * h + 2 * k] = _mm_unpacklo_epi32(q[4 * h + k], q[4 * h + 2 + k]);
            o[4 * h + 2 * k + 1] = _mm_unpackhi_epi32(q[4 * h + k], q[4 * h + 2 + k]);
        }
    }
    o
}

/// Sixteen bytes at `o` of `r`.
///
/// # Safety
/// `o + 16 <= r.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn ld16(r: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 16 <= r.len());
    // SAFETY: the caller proves the 16 bytes lie in `r`.
    unsafe { _mm_loadu_si128(r.as_ptr().add(o).cast()) }
}

/// Eight bytes at `o` of `r`, in the low half.
///
/// # Safety
/// `o + 8 <= r.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn ld8(r: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 8 <= r.len());
    // SAFETY: the caller proves the 8 bytes lie in `r`.
    unsafe { _mm_loadl_epi64(r.as_ptr().add(o).cast()) }
}

/// Stores sixteen bytes at `o` of `r`.
///
/// # Safety
/// `o + 16 <= r.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn st16(r: &mut [u8], o: usize, v: __m128i) {
    debug_assert!(o + 16 <= r.len());
    // SAFETY: the caller proves the 16 bytes lie in `r`.
    unsafe { _mm_storeu_si128(r.as_mut_ptr().add(o).cast(), v) }
}

/// Stores the low eight bytes at `o` of `r`.
///
/// # Safety
/// `o + 8 <= r.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn st8(r: &mut [u8], o: usize, v: __m128i) {
    debug_assert!(o + 8 <= r.len());
    // SAFETY: the caller proves the 8 bytes lie in `r`.
    unsafe { _mm_storel_epi64(r.as_mut_ptr().add(o).cast(), v) }
}

#[target_feature(enable = "avx2")]
// The edge comes field by field, in registers: built on the caller's
// stack, a byte at a time, its fields read back wider stalled each call.
pub(super) fn filter(
    buf: &mut [u8],
    at: usize,
    stride: usize,
    vertical: bool,
    wds: [Option<Wd>; 2],
    thr: [Thresh; 2],
) {
    let Some(wd) = super::widest(wds) else {
        return;
    };
    let z = _mm_setzero_si128();
    // The edge's sixteen lines, as one region checked once.
    let (back, span) = if vertical {
        let w = if wd == Wd::W16 { 8 } else { 4 };
        (w, 15 * stride + 2 * w)
    } else {
        let taps = if wd == Wd::W16 { 8 } else { 4 };
        (taps * stride, (2 * taps - 1) * stride + 16)
    };
    let Some(first) = at.checked_sub(back) else {
        return;
    };
    let Some(reg) = buf.get_mut(first..first + span) else {
        return;
    };
    let taps = if wd == Wd::W16 { 8 } else { 4 };
    let mut v = [z; 16];
    // SAFETY (every load and store below): `reg` holds `span` bytes, and
    // `span` was sized above so that each row's 8 or 16 bytes at
    // `i * stride` (i up to 15, or `2 * taps - 1` for a horizontal edge)
    // end inside it. Taps p3 to q3 start `taps - 4` samples (or rows) into
    // the region; p7 to q7 are there when `taps` is 8.
    #[allow(unsafe_code)]
    if vertical {
        let rows: [__m128i; 16] =
            core::array::from_fn(|i| unsafe { ld8(reg, i * stride + taps - 4) });
        let cols = transpose16x8(&rows);
        v[4..12].copy_from_slice(&cols);
    } else {
        for k in 0..8 {
            v[4 + k] = unsafe { ld16(reg, (k + taps - 4) * stride) };
        }
    }
    let changed = {
        let reg = &*reg;
        #[allow(unsafe_code)]
        lanes(&mut v, wds, wd, &thr, |v| {
            if vertical {
                let rows: [__m128i; 16] =
                    core::array::from_fn(|i| unsafe { ld16(reg, i * stride) });
                let t = transpose(&rows);
                v[..4].copy_from_slice(&t[..4]);
                v[12..].copy_from_slice(&t[12..]);
            } else {
                for k in 0..4 {
                    v[k] = unsafe { ld16(reg, k * stride) };
                    v[12 + k] = unsafe { ld16(reg, (12 + k) * stride) };
                }
            }
        })
    };
    // Store what the filter changed.
    #[allow(unsafe_code)]
    match (changed, vertical) {
        (Changed::None, _) => {}
        (Changed::Inner, true) => {
            let back: [__m128i; 8] = core::array::from_fn(|k| v[4 + k]);
            let out = transpose8x16(&back);
            let o = taps - 4;
            for (i, r) in out.iter().enumerate() {
                unsafe {
                    st8(reg, 2 * i * stride + o, *r);
                    st8(reg, (2 * i + 1) * stride + o, _mm_unpackhi_epi64(*r, *r));
                }
            }
        }
        (Changed::Outer, true) => {
            let rows = transpose(&v);
            for (i, r) in rows.iter().enumerate() {
                unsafe { st16(reg, i * stride, *r) };
            }
        }
        (c, false) => {
            let (first, last) = if c == Changed::Outer {
                (1, 15)
            } else {
                (5, 11)
            };
            // An index loop: the iterator chain stays out of line.
            #[allow(clippy::needless_range_loop)]
            for k in first..last {
                unsafe {
                    st16(reg, (k + taps - 8) * stride, v[k]);
                }
            }
        }
    }
}

/// Thirty-two bytes at `o` of `r`.
///
/// # Safety
/// `o + 32 <= r.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn ld32(r: &[u8], o: usize) -> __m256i {
    debug_assert!(o + 32 <= r.len());
    // SAFETY: the caller proves the 32 bytes lie in `r`.
    unsafe { _mm256_loadu_si256(r.as_ptr().add(o).cast()) }
}

/// Stores thirty-two bytes at `o` of `r`.
///
/// # Safety
/// `o + 32 <= r.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn st32(r: &mut [u8], o: usize, v: __m256i) {
    debug_assert!(o + 32 <= r.len());
    // SAFETY: the caller proves the 32 bytes lie in `r`.
    unsafe { _mm256_storeu_si256(r.as_mut_ptr().add(o).cast(), v) }
}

/// Two horizontal edges side by side, 32 columns from `at`: four groups
/// of eight columns, each with its own filter, as two [`filter`] calls
/// on `at` and `at + 16` would filter them.
#[target_feature(enable = "avx2")]
pub(super) fn filter32(
    buf: &mut [u8],
    at: usize,
    stride: usize,
    wds: [Option<Wd>; 4],
    thr: [Thresh; 4],
) {
    let w01 = super::widest([wds[0], wds[1]]);
    let w23 = super::widest([wds[2], wds[3]]);
    let Some(wd) = super::widest([w01, w23]) else {
        return;
    };
    let taps = if wd == Wd::W16 { 8 } else { 4 };
    let (back, span) = (taps * stride, (2 * taps - 1) * stride + 32);
    let Some(first) = at.checked_sub(back) else {
        return;
    };
    let Some(reg) = buf.get_mut(first..first + span) else {
        return;
    };
    let mut v = [_mm256_setzero_si256(); 16];
    // SAFETY (every load and store below): `reg` holds `span` bytes, so
    // each of rows `0..2 * taps` holds 32 from its start.
    #[allow(unsafe_code)]
    for k in 0..8 {
        v[4 + k] = unsafe { ld32(reg, (k + taps - 4) * stride) };
    }
    let changed = {
        let reg = &*reg;
        #[allow(unsafe_code)]
        lanes32(&mut v, wds, wd, &thr, |v| {
            for k in 0..4 {
                v[k] = unsafe { ld32(reg, k * stride) };
                v[12 + k] = unsafe { ld32(reg, (12 + k) * stride) };
            }
        })
    };
    let (first, last) = match changed {
        Changed::None => return,
        Changed::Outer => (1, 15),
        Changed::Inner => (5, 11),
    };
    // An index loop: the iterator chain stays out of line.
    #[allow(clippy::needless_range_loop, unsafe_code)]
    for k in first..last {
        unsafe { st32(reg, (k + taps - 8) * stride, v[k]) };
    }
}
