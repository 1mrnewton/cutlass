// The x86-64 version of each kernel, in 128-bit registers. This source is
// not a module of its own: `x86/sse41.rs` and `x86/avx2.rs` each include
// it, so it is built twice, as `crate::x86::sse41::kernels` and
// `crate::x86::avx2::kernels`: every function is inside `feat!`, which
// marks it as built for the tier's CPU features, so only the callers in
// this crate that have checked [`crate::x86::level`] reach them. The
// plain twins in [`crate::scalar`] are the reference, and the tests in
// [`crate::mc`], [`crate::deblock`], [`crate::residual`] and
// [`crate::intra`] compare each tier with them.
//
// # Memory
//
// As the NEON whole-block kernels do, these load and store through
// pointers, because a block is rows of a plane. Each kernel a caller can
// reach takes slices or fixed-size arrays and checks once, before any
// load, that every byte its loads and stores touch lies inside them; it
// returns `false`, or hands the block to the plain twin, when one does
// not. The pointer functions under them are `unsafe` and say what their
// caller has checked.

use core::arch::x86_64::{
    __m128i, _MM_HINT_T0, _mm_add_epi8, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16,
    _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128, _mm_avg_epu8, _mm_blendv_epi8, _mm_cmpeq_epi8,
    _mm_cmpgt_epi8, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi16,
    _mm_extract_epi32, _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16,
    _mm_max_epi8, _mm_max_epu8, _mm_min_epi8, _mm_min_epu8, _mm_movemask_epi8, _mm_mullo_epi16,
    _mm_or_si128, _mm_packs_epi16, _mm_packs_epi32, _mm_packus_epi16, _mm_prefetch, _mm_sad_epu8,
    _mm_set_epi16, _mm_set_epi32, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128,
    _mm_shuffle_epi8, _mm_slli_epi16, _mm_sra_epi16, _mm_sra_epi32, _mm_srai_epi16, _mm_srai_epi32,
    _mm_srli_epi16, _mm_srli_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi8,
    _mm_sub_epi16, _mm_sub_epi32, _mm_subs_epu8, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
};

use super::wide::luma16;
use crate::deblock::{ChromaRows, Lines, Rows};
use crate::edge::{Edge, Filter, MbEdges};
use crate::intra::{Near4, Near8};
use crate::mc_block::{BiWeight, Block, ChromaSource, Planes420, Planes420Mut, UniWeight};

/// The weights of a [`BiWeight`] as the mix takes them.
#[derive(Clone, Copy)]
struct Mix {
    /// `w0` and `w1` in each pair of 16-bit lanes.
    weights: __m128i,
    round: __m128i,
    shift: __m128i,
    offset: __m128i,
}

feat! {
    /// Sixteen bytes from `p`.
    ///
    /// # Safety
    ///
    /// Sixteen bytes from `p` must be readable.
    #[inline]
    pub(super) unsafe fn ld16(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise; the load needs no alignment.
        unsafe { _mm_loadu_si128(p.cast()) }
    }

    /// Eight bytes from `p`, in the low half.
    ///
    /// # Safety
    ///
    /// Eight bytes from `p` must be readable.
    #[inline]
    pub(super) unsafe fn ld8(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise; the load needs no alignment.
        unsafe { _mm_loadl_epi64(p.cast()) }
    }

    /// Four bytes from `p`, in the low quarter.
    ///
    /// # Safety
    ///
    /// Four bytes from `p` must be readable.
    #[inline]
    unsafe fn ld4(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise; the read needs no alignment.
        _mm_cvtsi32_si128(unsafe { p.cast::<i32>().read_unaligned() })
    }

    /// Sixteen bytes to `p`.
    ///
    /// # Safety
    ///
    /// Sixteen bytes from `p` must be writable.
    #[inline]
    pub(super) unsafe fn st16(p: *mut u8, v: __m128i) {
        // SAFETY: the caller's promise; the store needs no alignment.
        unsafe { _mm_storeu_si128(p.cast(), v) }
    }

    /// The low eight bytes of `v` to `p`.
    ///
    /// # Safety
    ///
    /// Eight bytes from `p` must be writable.
    #[inline]
    unsafe fn st8(p: *mut u8, v: __m128i) {
        // SAFETY: the caller's promise; the store needs no alignment.
        unsafe { _mm_storel_epi64(p.cast(), v) }
    }

    /// The low four bytes of `v` to `p`.
    ///
    /// # Safety
    ///
    /// Four bytes from `p` must be writable.
    #[inline]
    unsafe fn st4(p: *mut u8, v: __m128i) {
        // SAFETY: the caller's promise; the write needs no alignment.
        unsafe { p.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(v)) }
    }

    /// `value` in every byte.
    #[inline]
    fn splat(value: u8) -> __m128i {
        _mm_set1_epi8(value.cast_signed())
    }

    /// The low eight bytes of `v` as 16-bit lanes.
    #[inline]
    fn low(v: __m128i) -> __m128i {
        _mm_cvtepu8_epi16(v)
    }

    /// The high eight bytes of `v` as 16-bit lanes.
    #[inline]
    fn high(v: __m128i) -> __m128i {
        _mm_unpackhi_epi8(v, _mm_setzero_si128())
    }

    /// [`low`] of each of four vectors.
    #[inline]
    fn low4(v: [__m128i; 4]) -> [__m128i; 4] {
        [low(v[0]), low(v[1]), low(v[2]), low(v[3])]
    }

    /// [`high`] of each of four vectors.
    #[inline]
    fn high4(v: [__m128i; 4]) -> [__m128i; 4] {
        [high(v[0]), high(v[1]), high(v[2]), high(v[3])]
    }

    /// The six-tap filter's sum in each 16-bit lane: 20·(c + d) −
    /// 5·(b + e) + (a + f), as 5·(4·(c + d) − (b + e)) + (a + f). From
    /// samples it lies between −2550 and 10710.
    #[inline]
    fn six(t: [__m128i; 6]) -> __m128i {
        let outer = _mm_add_epi16(t[0], t[5]);
        let near = _mm_add_epi16(t[1], t[4]);
        let mid = _mm_add_epi16(t[2], t[3]);
        let inner = _mm_sub_epi16(_mm_slli_epi16::<2>(mid), near);
        _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16::<2>(inner), inner), outer)
    }

    /// Half samples from sums, as 16-bit lanes: (sum + 16) >> 5, to be
    /// packed with saturation.
    #[inline]
    fn round_lanes(sums: __m128i) -> __m128i {
        _mm_srai_epi16::<5>(_mm_add_epi16(sums, _mm_set1_epi16(16)))
    }

    /// Centre samples from six rows of sums, as 16-bit lanes to be packed
    /// with saturation: (a − 5b + 20c + 512) >> 10 for a, b and c the
    /// outer, near and middle pairs, which needs 20 bits, worked out in
    /// 16 as ((((a − b) >> 2) − b + c) >> 2) + c, then (… + 32) >> 6:
    /// each shift drops only bits the next would. The one sum that can
    /// pass 16 bits is added with saturation, where the sample is far
    /// outside 0 to 255 either way and the pack clips it the same.
    #[inline]
    fn centre_lanes(s: &[__m128i; 6]) -> __m128i {
        let a = _mm_add_epi16(s[0], s[5]);
        let b = _mm_add_epi16(s[1], s[4]);
        let c = _mm_add_epi16(s[2], s[3]);
        let t = _mm_srai_epi16::<2>(_mm_sub_epi16(a, b));
        let t = _mm_srai_epi16::<2>(_mm_adds_epi16(_mm_sub_epi16(t, b), c));
        _mm_srai_epi16::<6>(_mm_add_epi16(_mm_add_epi16(t, c), _mm_set1_epi16(32)))
    }

    /// Eight samples from `p` as 16-bit lanes.
    ///
    /// # Safety
    ///
    /// Eight bytes from `p` must be readable.
    #[inline]
    unsafe fn wide8(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise.
        low(unsafe { ld8(p) })
    }

    /// A row of eight full samples.
    ///
    /// # Safety
    ///
    /// Eight bytes from `p` must be readable.
    #[inline]
    unsafe fn full8(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise.
        unsafe { ld8(p) }
    }

    /// The six-tap sums along the row at `p` for eight samples.
    ///
    /// # Safety
    ///
    /// Thirteen bytes from two before `p` must be readable.
    #[inline]
    unsafe fn rawh8(p: *const u8) -> __m128i {
        // SAFETY: the last load ends 13 bytes after `p - 2`.
        unsafe {
            let p = p.sub(2);
            six([
                wide8(p),
                wide8(p.add(1)),
                wide8(p.add(2)),
                wide8(p.add(3)),
                wide8(p.add(4)),
                wide8(p.add(5)),
            ])
        }
    }

    /// The six-tap sums down the columns at `p` for eight samples.
    ///
    /// # Safety
    ///
    /// Eight bytes must be readable from `p` in each of the two rows
    /// before it, its own and the three after, `s` apart.
    #[inline]
    unsafe fn rawv8(p: *const u8, s: usize) -> __m128i {
        // SAFETY: the six rows the caller promises.
        unsafe {
            let p = p.sub(2 * s);
            six([
                wide8(p),
                wide8(p.add(s)),
                wide8(p.add(2 * s)),
                wide8(p.add(3 * s)),
                wide8(p.add(4 * s)),
                wide8(p.add(5 * s)),
            ])
        }
    }

    /// Eight half samples from their sums, in the low half.
    #[inline]
    fn round8(sums: __m128i) -> __m128i {
        let lanes = round_lanes(sums);
        _mm_packus_epi16(lanes, lanes)
    }

    /// Eight centre samples from six rows of sums, in the low half.
    #[inline]
    fn centre8v(sums: &[__m128i; 6]) -> __m128i {
        let lanes = centre_lanes(sums);
        _mm_packus_epi16(lanes, lanes)
    }

    /// The low eight bytes of `row` to `dst`, or their rounded average
    /// with what is there when `AVG`.
    ///
    /// # Safety
    ///
    /// Eight bytes from `dst` must be readable and writable.
    #[inline]
    unsafe fn put8<const AVG: bool>(dst: *mut u8, row: __m128i) {
        // SAFETY: the caller's promise.
        unsafe {
            let row = if AVG { _mm_avg_epu8(row, ld8(dst)) } else { row };
            st8(dst, row);
        }
    }

    /// [`put8`] for the low four bytes.
    ///
    /// # Safety
    ///
    /// Four bytes from `dst` must be readable and writable.
    #[inline]
    unsafe fn put4<const AVG: bool>(dst: *mut u8, row: __m128i) {
        // SAFETY: the caller's promise.
        unsafe {
            let row = if AVG { _mm_avg_epu8(row, ld4(dst)) } else { row };
            st4(dst, row);
        }
    }
}

x86_luma!(luma8, __m128i, full8, rawh8, rawv8, round8, centre8v, put8);

feat! {
    /// Where a luma block's routine reads, when that and the block's rows
    /// in `dst` all fit: the place of the block's top-left sample in its
    /// source. The rule is the NEON build's, so both take the same
    /// blocks. Each direction is read beyond the block only when it has
    /// an offset. Without a horizontal one, every source reads the `w`
    /// samples of the block's columns; with one, at most `2 × w` bytes
    /// from two samples left of each row. Without a vertical one, the
    /// sources read the block's rows; with one, two rows above them and
    /// three below.
    #[inline]
    fn luma_reach(block: &Block<'_>, frac: (u8, u8), dst: (&[u8], usize)) -> Option<usize> {
        let Block {
            src,
            stride,
            at: (x, y),
            size: (w, h),
        } = *block;
        let (dst, dst_stride) = dst;
        if (w != 8 && w != 16) || h == 0 || h > 16 || frac.0 > 3 || frac.1 > 3 {
            return None;
        }
        if dst_stride < w || (h - 1) * dst_stride + w > dst.len() {
            return None;
        }
        // Samples read left of the block and in each row from there.
        let (left, span) = if frac.0 == 0 { (0, w) } else { (2, 2 * w) };
        // Rows read above the block's first and below its last.
        let (above, below) = if frac.1 == 0 { (0, 0) } else { (2, 3) };
        if x < left || y < above || (frac.0 != 0 && stride < w + 5) {
            return None;
        }
        let last = (y + h - 1 + below)
            .checked_mul(stride)?
            .checked_add(x - left + span)?;
        if last > src.len() {
            return None;
        }
        Some(y * stride + x)
    }

    /// The luma routine for a block `w` wide, 8 or 16.
    ///
    /// # Safety
    ///
    /// As for the routine: see `luma_reach`.
    #[inline]
    unsafe fn luma_run(
        w: usize,
        avg: bool,
        dst: (*mut u8, usize),
        src: (*const u8, usize),
        h: usize,
        frac: (u8, u8),
    ) {
        let ((dst, ds), (src, s)) = (dst, src);
        // SAFETY: the caller's promise.
        unsafe {
            match (w == 16, avg) {
                (true, false) => luma16::<false>(dst, ds, src, s, h, frac),
                (true, true) => luma16::<true>(dst, ds, src, s, h, frac),
                (false, false) => luma8::<false>(dst, ds, src, s, h, frac),
                (false, true) => luma8::<true>(dst, ds, src, s, h, frac),
            }
        }
    }

    /// See [`crate::mc::luma_block`].
    pub(crate) fn luma_block(
        dst: &mut [u8],
        dst_stride: usize,
        block: Block<'_>,
        frac: (u8, u8),
    ) -> bool {
        let Some(at) = luma_reach(&block, frac, (dst, dst_stride)) else {
            return false;
        };
        let (w, h) = block.size;
        // SAFETY: `luma_reach` checked every byte the routine reads and
        // writes, from `at` in the source.
        unsafe {
            let src = block.src.as_ptr().add(at);
            luma_run(w, false, (dst.as_mut_ptr(), dst_stride), (src, block.stride), h, frac);
        }
        true
    }

    /// See [`crate::mc::luma_pair`].
    pub(crate) fn luma_pair(
        dst: &mut [u8],
        dst_stride: usize,
        first: (Block<'_>, (u8, u8)),
        second: Option<(Block<'_>, (u8, u8))>,
    ) -> bool {
        let Some(one) = luma_reach(&first.0, first.1, (dst, dst_stride)) else {
            return false;
        };
        let two = match second {
            Some((block, frac)) if block.size == first.0.size => {
                match luma_reach(&block, frac, (dst, dst_stride)) {
                    Some(at) => Some((block, at, frac)),
                    None => return false,
                }
            }
            Some(_) => return false,
            None => None,
        };
        let (w, h) = first.0.size;
        let out = (dst.as_mut_ptr(), dst_stride);
        // SAFETY: `luma_reach` checked every byte each routine reads and
        // writes.
        unsafe {
            let src = first.0.src.as_ptr().add(one);
            luma_run(w, false, out, (src, first.0.stride), h, first.1);
            if let Some((block, at, frac)) = two {
                let src = block.src.as_ptr().add(at);
                luma_run(w, true, out, (src, block.stride), h, frac);
            }
        }
        true
    }

    /// The low `W` bytes of `row`, 4 or 8, to `dst`, or averaged into it.
    ///
    /// # Safety
    ///
    /// `W` bytes from `dst` must be readable and writable.
    #[inline]
    unsafe fn put_chroma<const W: usize, const AVG: bool>(dst: *mut u8, row: __m128i) {
        // SAFETY: the caller's promise.
        unsafe {
            if W == 8 {
                put8::<AVG>(dst, row);
            } else {
                put4::<AVG>(dst, row);
            }
        }
    }

    /// Each of the eight samples from `p` beside the one after it, as
    /// pairs of bytes.
    ///
    /// # Safety
    ///
    /// Nine bytes from `p` must be readable.
    #[inline]
    unsafe fn chroma_pairs(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise.
        unsafe { _mm_unpacklo_epi8(ld8(p), ld8(p.add(1))) }
    }

    /// The four weights of clause 8.4.2.2.2 at eighth offsets `frac`, as
    /// the pair for a row and the pair for the row below it, each a
    /// sample's weight and its right neighbour's in two signed bytes.
    #[inline]
    fn chroma_taps(frac: (u8, u8)) -> (__m128i, __m128i) {
        let (xf, yf) = (i16::from(frac.0.min(7)), i16::from(frac.1.min(7)));
        let pair = |left: i16, right: i16| _mm_set1_epi16(left | (right << 8));
        (
            pair((8 - xf) * (8 - yf), xf * (8 - yf)),
            pair((8 - xf) * yf, xf * yf),
        )
    }

    /// One row of chroma samples from the pairs of its row and of the row
    /// below, in the low half.
    #[inline]
    fn chroma_row(above: __m128i, below: __m128i, taps: (__m128i, __m128i)) -> __m128i {
        let sum = _mm_add_epi16(
            _mm_maddubs_epi16(above, taps.0),
            _mm_maddubs_epi16(below, taps.1),
        );
        let lanes = _mm_srli_epi16::<6>(_mm_add_epi16(sum, _mm_set1_epi16(32)));
        _mm_packus_epi16(lanes, lanes)
    }

    /// A chroma block `W` wide, 4 or 8, of `h` rows whose first sample is
    /// `src`, rows `s` apart, at eighth offsets `frac`, written to `dst`,
    /// rows `ds` apart, or averaged into it when `AVG`.
    ///
    /// # Safety
    ///
    /// Nine bytes must be readable from the start of each of the block's
    /// rows and of the row after its last, and `W` writable in each of
    /// its rows at `dst`: what `chroma_reach` checks.
    unsafe fn chroma<const W: usize, const AVG: bool>(
        dst: *mut u8,
        ds: usize,
        src: *const u8,
        s: usize,
        h: usize,
        frac: (u8, u8),
    ) {
        // SAFETY: the caller's promise.
        unsafe {
            if frac == (0, 0) {
                for row in 0..h {
                    put_chroma::<W, AVG>(dst.add(row * ds), ld8(src.add(row * s)));
                }
                return;
            }
            let taps = chroma_taps(frac);
            let mut above = chroma_pairs(src);
            for row in 0..h {
                let below = chroma_pairs(src.add((row + 1) * s));
                put_chroma::<W, AVG>(dst.add(row * ds), chroma_row(above, below, taps));
                above = below;
            }
        }
    }

    /// The chroma routine for a block `w` wide, 4 or 8.
    ///
    /// # Safety
    ///
    /// As for [`chroma`].
    #[inline]
    unsafe fn chroma_run(
        w: usize,
        avg: bool,
        dst: (*mut u8, usize),
        src: (*const u8, usize),
        h: usize,
        frac: (u8, u8),
    ) {
        let ((dst, ds), (src, s)) = (dst, src);
        // SAFETY: the caller's promise.
        unsafe {
            match (w == 8, avg) {
                (true, false) => chroma::<8, false>(dst, ds, src, s, h, frac),
                (true, true) => chroma::<8, true>(dst, ds, src, s, h, frac),
                (false, false) => chroma::<4, false>(dst, ds, src, s, h, frac),
                (false, true) => chroma::<4, true>(dst, ds, src, s, h, frac),
            }
        }
    }

    /// Where a chroma block's routine reads, when that and the block's
    /// rows in `dst` all fit: the place of the block's top-left sample in
    /// its source. The rule is the NEON build's: nine bytes from the
    /// start of each row and of the row after the last, and an even
    /// height.
    #[inline]
    fn chroma_reach(block: &Block<'_>, frac: (u8, u8), dst: (&[u8], usize)) -> Option<usize> {
        let Block {
            src,
            stride,
            at: (x, y),
            size: (w, h),
        } = *block;
        let (dst, dst_stride) = dst;
        if (w != 4 && w != 8) || h == 0 || h % 2 != 0 || frac.0 > 7 || frac.1 > 7 {
            return None;
        }
        if dst_stride < w || (h - 1).checked_mul(dst_stride)?.checked_add(w)? > dst.len() {
            return None;
        }
        let last = (y.checked_add(h)?)
            .checked_mul(stride)?
            .checked_add(x.checked_add(9)?)?;
        if last > src.len() {
            return None;
        }
        Some(y * stride + x)
    }

    /// See [`crate::mc::chroma_block`].
    pub(crate) fn chroma_block(
        dst: &mut [u8],
        dst_stride: usize,
        block: Block<'_>,
        frac: (u8, u8),
    ) -> bool {
        let Some(at) = chroma_reach(&block, frac, (dst, dst_stride)) else {
            return false;
        };
        let (w, h) = block.size;
        // SAFETY: `chroma_reach` checked every byte the routine reads and
        // writes, from `at` in the source.
        unsafe {
            let src = block.src.as_ptr().add(at);
            chroma_run(w, false, (dst.as_mut_ptr(), dst_stride), (src, block.stride), h, frac);
        }
        true
    }

    /// See [`crate::mc::chroma_pair`].
    pub(crate) fn chroma_pair(
        dst: &mut [u8],
        dst_stride: usize,
        first: (Block<'_>, (u8, u8)),
        second: Option<(Block<'_>, (u8, u8))>,
    ) -> bool {
        let Some(one) = chroma_reach(&first.0, first.1, (dst, dst_stride)) else {
            return false;
        };
        let two = match second {
            Some((block, frac)) if block.size == first.0.size => {
                match chroma_reach(&block, frac, (dst, dst_stride)) {
                    Some(at) => Some((block, at, frac)),
                    None => return false,
                }
            }
            Some(_) => return false,
            None => None,
        };
        let (w, h) = first.0.size;
        let out = (dst.as_mut_ptr(), dst_stride);
        // SAFETY: `chroma_reach` checked every byte each routine reads
        // and writes.
        unsafe {
            let src = first.0.src.as_ptr().add(one);
            chroma_run(w, false, out, (src, first.0.stride), h, first.1);
            if let Some((block, at, frac)) = two {
                let src = block.src.as_ptr().add(at);
                chroma_run(w, true, out, (src, block.stride), h, frac);
            }
        }
        true
    }

    /// See [`crate::mc::chroma_planes`]: each plane as `chroma_pair`
    /// makes it, every block checked before any is written.
    pub(crate) fn chroma_planes(
        dst: (&mut [u8], &mut [u8]),
        dst_stride: usize,
        size: (usize, usize),
        first: ChromaSource<'_>,
        second: Option<ChromaSource<'_>>,
    ) -> bool {
        let (dst_cb, dst_cr) = dst;
        if size.1 > 8 {
            return false;
        }
        let reach = |source: &ChromaSource<'_>| {
            let cb = chroma_reach(&source.plane(true, size), source.frac, (dst_cb, dst_stride))?;
            let cr = chroma_reach(&source.plane(false, size), source.frac, (dst_cr, dst_stride))?;
            Some((cb, cr))
        };
        let Some(one) = reach(&first) else {
            return false;
        };
        let two = match second {
            Some(source) => match reach(&source) {
                Some(at) => Some((source, at)),
                None => return false,
            },
            None => None,
        };
        let (w, h) = size;
        let (out_cb, out_cr) = (dst_cb.as_mut_ptr(), dst_cr.as_mut_ptr());
        // SAFETY: `chroma_reach` checked every byte each plane's routine
        // reads from either source and writes.
        unsafe {
            let (s, frac) = (first.stride, first.frac);
            chroma_run(w, false, (out_cb, dst_stride), (first.cb.as_ptr().add(one.0), s), h, frac);
            chroma_run(w, false, (out_cr, dst_stride), (first.cr.as_ptr().add(one.1), s), h, frac);
            if let Some((source, at)) = two {
                let (s, frac) = (source.stride, source.frac);
                let (cb, cr) = (source.cb.as_ptr().add(at.0), source.cr.as_ptr().add(at.1));
                chroma_run(w, true, (out_cb, dst_stride), (cb, s), h, frac);
                chroma_run(w, true, (out_cr, dst_stride), (cr, s), h, frac);
            }
        }
        true
    }

    /// See [`crate::mc::predict_420`].
    pub(crate) fn predict_420<'r>(
        dst: &mut Planes420Mut<'_>,
        block: (usize, usize, usize, usize),
        first: (&Planes420<'r>, [i16; 2]),
        second: Option<(&Planes420<'r>, [i16; 2])>,
    ) -> (bool, bool) {
        let (x, y, w, h) = block;
        if !dst.holds((x, y), (w, h)) {
            return (false, false);
        }
        let (dst_y, dst_u, dst_v, stride) = dst.planes();
        let luma = (w == 8 || w == 16) && (1..=16).contains(&h) && {
            let from = |(planes, mv): (&Planes420<'r>, [i16; 2])| {
                let at = (
                    x as i64 + i64::from(mv[0] >> 2),
                    y as i64 + i64::from(mv[1] >> 2),
                );
                let frac = ((mv[0] & 3) as u8, (mv[1] & 3) as u8);
                let at_sample = planes.luma_from(at, (w, h), frac)?;
                Some((planes.y(), at_sample, planes.stride(), frac))
            };
            match (from(first), second.map(from)) {
                (Some(one), two @ (None | Some(Some(_)))) => {
                    let out = y * stride + x;
                    // SAFETY: `holds` put the block's rows inside `dst_y`,
                    // and `luma_from` every byte each routine reads inside
                    // its plane, from the offsets it gave.
                    unsafe {
                        let out = (dst_y.as_mut_ptr().add(out), stride);
                        let (src, at, src_stride, frac) = one;
                        luma_run(w, false, out, (src.as_ptr().add(at), src_stride), h, frac);
                        if let Some(Some((src, at, src_stride, frac))) = two {
                            luma_run(w, true, out, (src.as_ptr().add(at), src_stride), h, frac);
                        }
                    }
                    true
                }
                _ => false,
            }
        };
        let (cw, ch) = (w / 2, h / 2);
        let chroma = (cw == 4 || cw == 8) && ch != 0 && ch <= 8 && ch % 2 == 0 && {
            let (cx, cy) = (x / 2, y / 2);
            let from = |(planes, mv): (&Planes420<'r>, [i16; 2])| {
                let at = (
                    cx as i64 + i64::from(mv[0] >> 3),
                    cy as i64 + i64::from(mv[1] >> 3),
                );
                let frac = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
                let at_sample = planes.chroma_from(at, (cw, ch), frac)?;
                Some((planes.cb(), planes.cr(), at_sample, planes.stride() / 2, frac))
            };
            match (from(first), second.map(from)) {
                (Some(one), two @ (None | Some(Some(_)))) => {
                    let stride = stride / 2;
                    let out = cy * stride + cx;
                    // SAFETY: `holds` put the block's rows inside both
                    // chroma planes of `dst`, and `chroma_from` every
                    // byte each routine reads inside its sources' planes,
                    // which are as long as each other.
                    unsafe {
                        let out_u = (dst_u.as_mut_ptr().add(out), stride);
                        let out_v = (dst_v.as_mut_ptr().add(out), stride);
                        let (cb, cr, at, s, frac) = one;
                        chroma_run(cw, false, out_u, (cb.as_ptr().add(at), s), ch, frac);
                        chroma_run(cw, false, out_v, (cr.as_ptr().add(at), s), ch, frac);
                        if let Some(Some((cb, cr, at, s, frac))) = two {
                            chroma_run(cw, true, out_u, (cb.as_ptr().add(at), s), ch, frac);
                            chroma_run(cw, true, out_v, (cr.as_ptr().add(at), s), ch, frac);
                        }
                    }
                    true
                }
                _ => false,
            }
        };
        (luma, chroma)
    }

    /// See [`crate::mc::avg_row`].
    pub(crate) fn avg_row(dst: &mut [u8], a: &[u8], b: &[u8]) -> bool {
        let n = dst.len();
        if a.len() < n || b.len() < n {
            return false;
        }
        // SAFETY: all three hold `n` bytes.
        unsafe { avg_span(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n) };
        true
    }

    /// The rounded average of `n` bytes of `a` and of `b`, to `dst`.
    ///
    /// # Safety
    ///
    /// `n` bytes must be readable from `a` and `b` and writable at `dst`.
    #[inline]
    unsafe fn avg_span(dst: *mut u8, a: *const u8, b: *const u8, n: usize) {
        // SAFETY: every load and store stays below `n`.
        unsafe {
            let mut done = 0;
            while done + 16 <= n {
                st16(dst.add(done), _mm_avg_epu8(ld16(a.add(done)), ld16(b.add(done))));
                done += 16;
            }
            if done + 8 <= n {
                st8(dst.add(done), _mm_avg_epu8(ld8(a.add(done)), ld8(b.add(done))));
                done += 8;
            }
            if done + 4 <= n {
                st4(dst.add(done), _mm_avg_epu8(ld4(a.add(done)), ld4(b.add(done))));
                done += 4;
            }
            while done < n {
                let sum = u16::from(*a.add(done)) + u16::from(*b.add(done)) + 1;
                *dst.add(done) = (sum >> 1) as u8;
                done += 1;
            }
        }
    }

    /// See [`crate::mc::avg_block`].
    pub(crate) fn avg_block(
        dst: &mut [u8],
        dst_stride: usize,
        a: &[u8],
        b: &[u8],
        size: (usize, usize),
    ) -> bool {
        let (w, h) = size;
        let Some(area) = w.checked_mul(h) else {
            return false;
        };
        if w == 0 || h == 0 || a.len() < area || b.len() < area {
            return false;
        }
        let Some(end) = (h - 1).checked_mul(dst_stride).and_then(|rows| rows.checked_add(w))
        else {
            return false;
        };
        if dst.len() < end {
            return false;
        }
        for row in 0..h {
            // SAFETY: row `row` of each block ends inside its slice, as
            // checked above.
            unsafe {
                avg_span(
                    dst.as_mut_ptr().add(row * dst_stride),
                    a.as_ptr().add(row * w),
                    b.as_ptr().add(row * w),
                    w,
                );
            }
        }
        true
    }

    #[inline]
    fn mix_of(weight: BiWeight) -> Mix {
        Mix {
            weights: _mm_set1_epi32((weight.w0 & 0xffff) | (weight.w1 << 16)),
            round: _mm_set1_epi32(1 << (weight.shift - 1)),
            shift: _mm_cvtsi32_si128(weight.shift),
            offset: _mm_set1_epi32(weight.offset),
        }
    }

    /// Eight samples of `a` and of `b`, as 16-bit lanes, mixed: each
    /// `((a·w0 + b·w1 + 2^(shift−1)) >> shift) + offset` in 32 bits, as
    /// the plain twin works it out, then held to 0 to 255, in the low
    /// half.
    #[inline]
    fn mix8(a: __m128i, b: __m128i, mix: &Mix) -> __m128i {
        let part = |pairs: __m128i| {
            let sum = _mm_add_epi32(_mm_madd_epi16(pairs, mix.weights), mix.round);
            _mm_add_epi32(_mm_sra_epi32(sum, mix.shift), mix.offset)
        };
        let lanes = _mm_packs_epi32(
            part(_mm_unpacklo_epi16(a, b)),
            part(_mm_unpackhi_epi16(a, b)),
        );
        _mm_packus_epi16(lanes, lanes)
    }

    /// `n` bytes of `a` mixed with `n` of `b`, to `dst`, which may be
    /// `a`.
    ///
    /// # Safety
    ///
    /// `n` bytes must be readable from `a` and `b` and writable at `dst`.
    #[inline]
    unsafe fn mix_span(dst: *mut u8, a: *const u8, b: *const u8, n: usize, weight: BiWeight) {
        let mix = mix_of(weight);
        // SAFETY: every load and store stays below `n`.
        unsafe {
            let mut done = 0;
            while done + 8 <= n {
                let mixed = mix8(low(ld8(a.add(done))), low(ld8(b.add(done))), &mix);
                st8(dst.add(done), mixed);
                done += 8;
            }
            if done + 4 <= n {
                let mixed = mix8(low(ld4(a.add(done))), low(ld4(b.add(done))), &mix);
                st4(dst.add(done), mixed);
                done += 4;
            }
            while done < n {
                *dst.add(done) = crate::mc_block::weight_one(*a.add(done), *b.add(done), weight);
                done += 1;
            }
        }
    }

    /// See [`crate::mc::weight_block`].
    pub(crate) fn weight_block(
        dst: &mut [u8],
        dst_stride: usize,
        a: &[u8],
        b: &[u8],
        size: (usize, usize),
        weight: BiWeight,
    ) -> bool {
        let (w, h) = size;
        let Some(area) = w.checked_mul(h) else {
            return false;
        };
        if !weight.fits() || w == 0 || h == 0 || a.len() < area || b.len() < area {
            return false;
        }
        let Some(end) = (h - 1).checked_mul(dst_stride).and_then(|rows| rows.checked_add(w))
        else {
            return false;
        };
        if dst.len() < end {
            return false;
        }
        for row in 0..h {
            // SAFETY: row `row` of each block ends inside its slice, as
            // checked above.
            unsafe {
                mix_span(
                    dst.as_mut_ptr().add(row * dst_stride),
                    a.as_ptr().add(row * w),
                    b.as_ptr().add(row * w),
                    w,
                    weight,
                );
            }
        }
        true
    }

    /// See [`crate::mc::weight_onto`].
    pub(crate) fn weight_onto(
        dst: &mut [u8],
        dst_stride: usize,
        b: &[u8],
        size: (usize, usize),
        weight: BiWeight,
    ) -> bool {
        if !crate::mc_block::onto_fits(dst, dst_stride, b, size, weight) {
            return false;
        }
        let (w, h) = size;
        for row in 0..h {
            // SAFETY: `onto_fits` put row `row` of the block inside `dst`
            // and of `b` inside `b`.
            unsafe {
                let out = dst.as_mut_ptr().add(row * dst_stride);
                mix_span(out, out, b.as_ptr().add(row * w), w, weight);
            }
        }
        true
    }

    /// See [`crate::mc::scale_block`]. A sample times its weight, with the
    /// rounding term, fits 16 bits; the offset is added with saturation,
    /// which only touches sums far outside 0 to 255.
    pub(crate) fn scale_block(
        dst: &mut [u8],
        dst_stride: usize,
        size: (usize, usize),
        weight: UniWeight,
    ) -> bool {
        if !weight.fits() || !crate::mc_block::scale_fits(dst, dst_stride, size) {
            return false;
        }
        let (w, h) = size;
        let by = _mm_set1_epi16(weight.weight as i16);
        let round = _mm_set1_epi16(if weight.shift > 0 { 1 << (weight.shift - 1) } else { 0 });
        let shift = _mm_cvtsi32_si128(weight.shift);
        let offset = _mm_set1_epi16(weight.offset as i16);
        let scaled = |samples: __m128i| {
            let product = _mm_add_epi16(_mm_mullo_epi16(low(samples), by), round);
            let lanes = _mm_adds_epi16(_mm_sra_epi16(product, shift), offset);
            _mm_packus_epi16(lanes, lanes)
        };
        for row in 0..h {
            // SAFETY: `scale_fits` put row `row` of the block, a multiple
            // of four samples wide, inside `dst`.
            unsafe {
                let out = dst.as_mut_ptr().add(row * dst_stride);
                let mut done = 0;
                while done + 8 <= w {
                    st8(out.add(done), scaled(ld8(out.add(done))));
                    done += 8;
                }
                if done + 4 <= w {
                    st4(out.add(done), scaled(ld4(out.add(done))));
                }
            }
        }
        true
    }

    /// See [`crate::mc::half16`].
    pub(crate) fn half16(taps: [&[u8; 16]; 6]) -> [u8; 16] {
        let (lo, hi) = six16(taps);
        let mut out = [0u8; 16];
        // SAFETY: `out` is sixteen writable bytes.
        unsafe { st16(out.as_mut_ptr(), _mm_packus_epi16(round_lanes(lo), round_lanes(hi))) };
        out
    }

    /// The six-tap sums of sixteen samples, low eight and high eight.
    #[inline]
    fn six16(taps: [&[u8; 16]; 6]) -> (__m128i, __m128i) {
        // SAFETY: each line is sixteen readable bytes.
        let line = |tap: usize| unsafe { ld16(taps[tap].as_ptr()) };
        let lines = [line(0), line(1), line(2), line(3), line(4), line(5)];
        let [a, b, c, d, e, f] = lines;
        (
            six([low(a), low(b), low(c), low(d), low(e), low(f)]),
            six([high(a), high(b), high(c), high(d), high(e), high(f)]),
        )
    }

    /// The six-tap sums of eight samples.
    #[inline]
    fn six8(taps: [&[u8; 8]; 6]) -> __m128i {
        // SAFETY: each line is eight readable bytes.
        let line = |tap: usize| unsafe { wide8(taps[tap].as_ptr()) };
        six([line(0), line(1), line(2), line(3), line(4), line(5)])
    }

    /// See [`crate::mc::half8`].
    pub(crate) fn half8(taps: [&[u8; 8]; 6]) -> [u8; 8] {
        let mut out = [0u8; 8];
        // SAFETY: `out` is eight writable bytes.
        unsafe { st8(out.as_mut_ptr(), round8(six8(taps))) };
        out
    }

    /// See [`crate::mc::raw16`].
    pub(crate) fn raw16(taps: [&[u8; 16]; 6]) -> [i16; 16] {
        let (lo, hi) = six16(taps);
        let mut out = [0i16; 16];
        // SAFETY: `out` is sixteen 16-bit lanes, 32 writable bytes.
        unsafe {
            st16(out.as_mut_ptr().cast(), lo);
            st16(out.as_mut_ptr().add(8).cast(), hi);
        }
        out
    }

    /// See [`crate::mc::raw8`].
    pub(crate) fn raw8(taps: [&[u8; 8]; 6]) -> [i16; 8] {
        let mut out = [0i16; 8];
        // SAFETY: `out` is eight 16-bit lanes, sixteen writable bytes.
        unsafe { st16(out.as_mut_ptr().cast(), six8(taps)) };
        out
    }

    /// See [`crate::mc::centre16`].
    pub(crate) fn centre16(taps: [&[i16; 16]; 6]) -> [u8; 16] {
        // SAFETY: each line is sixteen 16-bit lanes, 32 readable bytes.
        let lo = |tap: usize| unsafe { ld16(taps[tap].as_ptr().cast()) };
        let hi = |tap: usize| unsafe { ld16(taps[tap].as_ptr().add(8).cast()) };
        let lo = [lo(0), lo(1), lo(2), lo(3), lo(4), lo(5)];
        let hi = [hi(0), hi(1), hi(2), hi(3), hi(4), hi(5)];
        let mut out = [0u8; 16];
        // SAFETY: `out` is sixteen writable bytes.
        unsafe {
            st16(
                out.as_mut_ptr(),
                _mm_packus_epi16(centre_lanes(&lo), centre_lanes(&hi)),
            );
        }
        out
    }

    /// See [`crate::mc::centre8`].
    pub(crate) fn centre8(taps: [&[i16; 8]; 6]) -> [u8; 8] {
        // SAFETY: each line is eight 16-bit lanes, sixteen readable bytes.
        let line = |tap: usize| unsafe { ld16(taps[tap].as_ptr().cast()) };
        let sums = [line(0), line(1), line(2), line(3), line(4), line(5)];
        let mut out = [0u8; 8];
        // SAFETY: `out` is eight writable bytes.
        unsafe { st8(out.as_mut_ptr(), centre8v(&sums)) };
        out
    }

    /// See [`crate::mc::avg16`].
    pub(crate) fn avg16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: each array is sixteen bytes.
        unsafe {
            st16(out.as_mut_ptr(), _mm_avg_epu8(ld16(a.as_ptr()), ld16(b.as_ptr())));
        }
        out
    }

    /// See [`crate::mc::avg8`].
    pub(crate) fn avg8(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
        let mut out = [0u8; 8];
        // SAFETY: each array is eight bytes.
        unsafe {
            st8(out.as_mut_ptr(), _mm_avg_epu8(ld8(a.as_ptr()), ld8(b.as_ptr())));
        }
        out
    }

    /// See [`crate::mc::chroma8`].
    pub(crate) fn chroma8(top: &[u8; 9], bottom: &[u8; 9], xf: u8, yf: u8) -> [u8; 8] {
        let taps = chroma_taps((xf, yf));
        let mut out = [0u8; 8];
        // SAFETY: each line is nine readable bytes and `out` eight
        // writable ones.
        unsafe {
            let row = chroma_row(chroma_pairs(top.as_ptr()), chroma_pairs(bottom.as_ptr()), taps);
            st8(out.as_mut_ptr(), row);
        }
        out
    }

    /// Asks the cache for `rows` rows of `plane`, `stride` apart from
    /// sample `at`, with `prefetcht0`, as FFmpeg's `ff_prefetch_mmxext`
    /// does. A row outside `plane` ends the request.
    #[inline]
    pub(crate) fn prefetch(plane: &[u8], at: usize, stride: usize, rows: usize) {
        let mut index = at;
        for _ in 0..rows {
            if index >= plane.len() {
                return;
            }
            let sample = plane.as_ptr().wrapping_add(index);
            // A hint that reads, writes and faults on nothing.
            _mm_prefetch::<_MM_HINT_T0>(sample.cast());
            let Some(next) = index.checked_add(stride) else {
                return;
            };
            index = next;
        }
    }

    /// Asks for each cache line of `items[at]` to be brought in. Nothing
    /// is asked for when `at` is outside `items`.
    #[inline]
    pub(crate) fn prefetch_item<T>(items: &[T], at: usize) {
        let Some(item) = items.get(at) else {
            return;
        };
        let size = size_of::<T>();
        if size == 0 {
            return;
        }
        let start = core::ptr::from_ref(item).cast::<u8>();
        let mut offset = 0;
        while offset < size + 63 {
            // The last line asked for is the one holding the last byte.
            let byte = start.wrapping_add(offset.min(size - 1));
            // A hint that reads, writes and faults on nothing.
            _mm_prefetch::<_MM_HINT_T0>(byte.cast());
            offset += 64;
        }
    }
}

feat! {
    /// |a − b| in each byte.
    #[inline]
    fn absdiff(a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
    }

    /// 0xff in each byte of `a` that is `limit` or more.
    #[inline]
    fn at_least(a: __m128i, limit: __m128i) -> __m128i {
        _mm_cmpeq_epi8(_mm_max_epu8(a, limit), a)
    }

    /// (a + b) >> 1 in each byte: the rounded average less the bit it
    /// rounded up by.
    #[inline]
    fn half_sum(a: __m128i, b: __m128i) -> __m128i {
        let odd = _mm_and_si128(_mm_xor_si128(a, b), splat(1));
        _mm_sub_epi8(_mm_avg_epu8(a, b), odd)
    }

    /// 0xff in each line the thresholds of 8.7.2.2 leave alone: one where
    /// |p0 − q0| is `alpha` or more, or |p1 − p0| or |q1 − q0| is `beta`
    /// or more.
    #[inline]
    fn unfiltered(v: [__m128i; 4], alpha: u8, beta: u8) -> __m128i {
        let [p1, p0, q0, q1] = v;
        let beta = splat(beta);
        _mm_or_si128(
            at_least(absdiff(p0, q0), splat(alpha)),
            _mm_or_si128(
                at_least(absdiff(p1, p0), beta),
                at_least(absdiff(q1, q0), beta),
            ),
        )
    }

    /// (4·(q0 − p0) + (p1 − q1) + 4) >> 3 for eight lines, in 16 bits.
    #[inline]
    fn lean(v: [__m128i; 4]) -> __m128i {
        let [p1, p0, q0, q1] = v;
        let gap = _mm_slli_epi16::<2>(_mm_sub_epi16(q0, p0));
        let sum = _mm_add_epi16(_mm_add_epi16(gap, _mm_sub_epi16(p1, q1)), _mm_set1_epi16(4));
        _mm_srai_epi16::<3>(sum)
    }

    /// The new p0 and q0 of 8.7.2.3: each moved by the step above, held
    /// to within `tc`, 0 to 127 a line, and to a sample.
    #[inline]
    fn moved(v: [__m128i; 4], tc: __m128i) -> (__m128i, __m128i) {
        let zero = _mm_setzero_si128();
        // The narrowing saturates, which the clip to ±tc then hides.
        let raw = _mm_packs_epi16(lean(low4(v)), lean(high4(v)));
        let delta = _mm_max_epi8(_mm_min_epi8(raw, tc), _mm_sub_epi8(zero, tc));
        // Adding a signed step to a sample and holding it to 0..=255: add
        // the step's positive part, take away its negative part, both
        // saturating.
        let rise = _mm_max_epi8(delta, zero);
        let fall = _mm_max_epi8(_mm_sub_epi8(zero, delta), zero);
        let [_, p0, q0, _] = v;
        (
            _mm_subs_epu8(_mm_adds_epu8(p0, rise), fall),
            _mm_adds_epu8(_mm_subs_epu8(q0, rise), fall),
        )
    }

    /// 8.7.2.3 on sixteen lines, in the samples' own 8 bits where it can
    /// be. `v` is p2, p1, p0, q0, q1, q2. Gives the new p1, p0, q0, q1,
    /// or `None` when no line passes the thresholds and nothing changes.
    /// `tc0` is a line's clipping value, at most 125; a negative one
    /// leaves its line alone.
    #[inline]
    fn normal(v: [__m128i; 6], alpha: u8, beta: u8, tc0: __m128i) -> Option<[__m128i; 4]> {
        let [p2, p1, p0, q0, q1, q2] = v;
        let zero = _mm_setzero_si128();
        let skip = _mm_or_si128(
            unfiltered([p1, p0, q0, q1], alpha, beta),
            _mm_cmpgt_epi8(zero, tc0),
        );
        if _mm_movemask_epi8(skip) == 0xffff {
            return None;
        }
        let beta = splat(beta);
        // 0xff where the side's outer sample is near enough to be moved.
        let near = |outer: __m128i, edge: __m128i| {
            _mm_cmpeq_epi8(_mm_or_si128(at_least(absdiff(outer, edge), beta), skip), zero)
        };
        let (ap, aq) = (near(p2, p0), near(q2, q0));
        // Zero on a line that is not filtered, so nothing moves there.
        let tc0 = _mm_andnot_si128(skip, tc0);
        // A true lane is 0xff, which is −1, so subtracting it adds one.
        let tc = _mm_sub_epi8(_mm_sub_epi8(tc0, ap), aq);
        // p1 + Clip3(−tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) − 2·p1) >> 1)
        // is the halved sum of p2 and the average, held within tc0 of p1.
        let average = _mm_avg_epu8(p0, q0);
        let side = |outer: __m128i, inner: __m128i, near: __m128i| {
            let held = _mm_max_epu8(
                _mm_min_epu8(half_sum(outer, average), _mm_adds_epu8(inner, tc0)),
                _mm_subs_epu8(inner, tc0),
            );
            _mm_blendv_epi8(inner, held, near)
        };
        let (new_p0, new_q0) = moved([p1, p0, q0, q1], tc);
        Some([side(p2, p1, ap), new_p0, new_q0, side(q2, q1, aq)])
    }

    /// One side of 8.7.2.4 for eight lines in 16 bits: `o3` to `o0` are
    /// the side's own samples, outermost first, and `n0`, `n1` the other
    /// side's nearest two. Gives the strongly filtered o0, o1 and o2 and
    /// the weakly filtered o0.
    #[inline]
    fn strong_lanes(o: [__m128i; 4], n0: __m128i, n1: __m128i) -> [__m128i; 4] {
        let [o3, o2, o1, o0] = o;
        let inner = _mm_add_epi16(o1, _mm_add_epi16(o0, n0));
        let two = _mm_set1_epi16(2);
        let four = _mm_set1_epi16(4);
        let both = _mm_add_epi16(o2, inner);
        // (o2 + 2·o1 + 2·o0 + 2·n0 + n1 + 4) >> 3
        let s0 = _mm_srai_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(both, inner),
            _mm_add_epi16(n1, four),
        ));
        // (o2 + o1 + o0 + n0 + 2) >> 2
        let s1 = _mm_srai_epi16::<2>(_mm_add_epi16(both, two));
        // (2·o3 + 3·o2 + o1 + o0 + n0 + 4) >> 3
        let s2 = _mm_srai_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(_mm_slli_epi16::<1>(_mm_add_epi16(o3, o2)), both),
            four,
        ));
        // (2·o1 + o0 + n1 + 2) >> 2
        let weak = _mm_srai_epi16::<2>(_mm_add_epi16(
            _mm_add_epi16(_mm_slli_epi16::<1>(o1), o0),
            _mm_add_epi16(n1, two),
        ));
        [s0, s1, s2, weak]
    }

    /// [`strong_lanes`] for sixteen lines of bytes.
    #[inline]
    fn strong_side(o: [__m128i; 4], n0: __m128i, n1: __m128i) -> [__m128i; 4] {
        let lo = strong_lanes(low4(o), low(n0), low(n1));
        let hi = strong_lanes(high4(o), high(n0), high(n1));
        [
            _mm_packus_epi16(lo[0], hi[0]),
            _mm_packus_epi16(lo[1], hi[1]),
            _mm_packus_epi16(lo[2], hi[2]),
            _mm_packus_epi16(lo[3], hi[3]),
        ]
    }

    /// 8.7.2.4 on sixteen lines. `v` is p3 to p0, then q0 to q3. Gives
    /// the new p2, p1, p0, q0, q1, q2, or `None` when no line passes the
    /// thresholds.
    #[inline]
    fn strong(v: [__m128i; 8], alpha: u8, beta: u8) -> Option<[__m128i; 6]> {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
        let skip = unfiltered([p1, p0, q0, q1], alpha, beta);
        if _mm_movemask_epi8(skip) == 0xffff {
            return None;
        }
        // Lines whose |p0 − q0| is not below (alpha >> 2) + 2 take the
        // weak filter on both sides.
        let large = at_least(absdiff(p0, q0), splat((alpha >> 2) + 2));
        let beta = splat(beta);
        let weak_p = _mm_or_si128(large, at_least(absdiff(p2, p0), beta));
        let weak_q = _mm_or_si128(large, at_least(absdiff(q2, q0), beta));
        let [p0s, p1s, p2s, p0w] = strong_side([p3, p2, p1, p0], q0, q1);
        let [q0s, q1s, q2s, q0w] = strong_side([q3, q2, q1, q0], p0, p1);
        let keep_p = _mm_or_si128(weak_p, skip);
        let keep_q = _mm_or_si128(weak_q, skip);
        Some([
            _mm_blendv_epi8(p2s, p2, keep_p),
            _mm_blendv_epi8(p1s, p1, keep_p),
            _mm_blendv_epi8(_mm_blendv_epi8(p0s, p0w, weak_p), p0, skip),
            _mm_blendv_epi8(_mm_blendv_epi8(q0s, q0w, weak_q), q0, skip),
            _mm_blendv_epi8(q1s, q1, keep_q),
            _mm_blendv_epi8(q2s, q2, keep_q),
        ])
    }

    /// The chroma filters of 8.7.2.3 and 8.7.2.4 on up to sixteen lines:
    /// `v` is p1, p0, q0, q1, and `tc0` a line's clipping value for a
    /// strength below 4, at most 125 and negative for a line left alone,
    /// or `None` for strength 4. `lines` has a bit for each line that is
    /// there. Gives the new p0 and q0, or `None` when none of those lines
    /// passes the thresholds.
    #[inline]
    fn chroma_filtered(
        v: [__m128i; 4],
        alpha: u8,
        beta: u8,
        tc0: Option<__m128i>,
        lines: i32,
    ) -> Option<(__m128i, __m128i)> {
        let [p1, p0, q0, q1] = v;
        let skip = unfiltered(v, alpha, beta);
        match tc0 {
            Some(tc0) => {
                let skip = _mm_or_si128(skip, _mm_cmpgt_epi8(_mm_setzero_si128(), tc0));
                if _mm_movemask_epi8(skip) & lines == lines {
                    return None;
                }
                let tc = _mm_andnot_si128(skip, _mm_add_epi8(tc0, splat(1)));
                Some(moved(v, tc))
            }
            None => {
                if _mm_movemask_epi8(skip) & lines == lines {
                    return None;
                }
                // (2·p1 + p0 + q1 + 2) >> 2 is the rounded average of p1
                // and the halved sum of p0 and q1.
                let new_p0 = _mm_avg_epu8(p1, half_sum(p0, q1));
                let new_q0 = _mm_avg_epu8(q1, half_sum(q0, p1));
                Some((
                    _mm_blendv_epi8(new_p0, p0, skip),
                    _mm_blendv_epi8(new_q0, q0, skip),
                ))
            }
        }
    }

    /// Each of `tc0`'s four values in four bytes running.
    #[inline]
    fn spread4(tc0: [i8; 4]) -> __m128i {
        let all = _mm_cvtsi32_si128(i32::from_le_bytes(tc0.map(i8::cast_unsigned)));
        let twice = _mm_unpacklo_epi8(all, all);
        _mm_unpacklo_epi16(twice, twice)
    }

    /// Each of `tc0`'s four values in two bytes running, in the low half
    /// and again in the high one, when there are any.
    #[inline]
    fn spread2(tc0: Option<[i8; 4]>) -> Option<__m128i> {
        let all = _mm_cvtsi32_si128(i32::from_le_bytes(tc0?.map(i8::cast_unsigned)));
        let twice = _mm_unpacklo_epi8(all, all);
        Some(_mm_unpacklo_epi64(twice, twice))
    }

    /// The thresholds as the 8-bit filters take them, when they fit: a
    /// sample difference is at most 255, and a clipping value at most 125
    /// leaves room for the two that may be added to it.
    #[inline]
    fn small(alpha: i32, beta: i32, tc0: Option<[i8; 4]>) -> Option<(u8, u8)> {
        if tc0.is_some_and(|tc0| tc0.iter().any(|&group| group > 125)) {
            return None;
        }
        Some((u8::try_from(alpha).ok()?, u8::try_from(beta).ok()?))
    }

    /// A luma edge along a row: `q0` is its first sample below the edge,
    /// in rows `down` apart.
    ///
    /// # Safety
    ///
    /// Sixteen bytes must be readable and writable in each of the four
    /// rows before `q0` and the four from it.
    #[inline]
    unsafe fn luma_rows_at(q0: *mut u8, down: usize, alpha: u8, beta: u8, tc0: Option<[i8; 4]>) {
        // SAFETY: the eight rows the caller promises.
        unsafe {
            let row = |index: usize| q0.sub(4 * down).add(index * down);
            match tc0 {
                Some(tc0) => {
                    let v = [ld16(row(1)), ld16(row(2)), ld16(row(3)), ld16(row(4)), ld16(row(5)),
                        ld16(row(6))];
                    if let Some(new) = normal(v, alpha, beta, spread4(tc0)) {
                        st16(row(2), new[0]);
                        st16(row(3), new[1]);
                        st16(row(4), new[2]);
                        st16(row(5), new[3]);
                    }
                }
                None => {
                    let v = [ld16(row(0)), ld16(row(1)), ld16(row(2)), ld16(row(3)), ld16(row(4)),
                        ld16(row(5)), ld16(row(6)), ld16(row(7))];
                    if let Some(new) = strong(v, alpha, beta) {
                        for (index, samples) in new.into_iter().enumerate() {
                            st16(row(index + 1), samples);
                        }
                    }
                }
            }
        }
    }

    /// The sixteen lines of eight bytes from `first`, rows `down` apart,
    /// turned: vector `n` holds byte `n` of every line.
    ///
    /// # Safety
    ///
    /// Eight bytes must be readable in each of the sixteen rows from
    /// `first`.
    #[inline]
    unsafe fn load_lines(first: *const u8, down: usize) -> [__m128i; 8] {
        // SAFETY: the sixteen lines the caller promises.
        let line = |index: usize| unsafe { ld8(first.add(index * down)) };
        // Pairs of lines, then fours, then eights, then all sixteen.
        let pair = |index: usize| _mm_unpacklo_epi8(line(index), line(index + 1));
        let four = |index: usize| {
            let (a, b) = (pair(index), pair(index + 2));
            (_mm_unpacklo_epi16(a, b), _mm_unpackhi_epi16(a, b))
        };
        let eight = |index: usize| {
            let ((a_lo, a_hi), (b_lo, b_hi)) = (four(index), four(index + 4));
            [
                _mm_unpacklo_epi32(a_lo, b_lo),
                _mm_unpackhi_epi32(a_lo, b_lo),
                _mm_unpacklo_epi32(a_hi, b_hi),
                _mm_unpackhi_epi32(a_hi, b_hi),
            ]
        };
        let (top, bottom) = (eight(0), eight(8));
        [
            _mm_unpacklo_epi64(top[0], bottom[0]),
            _mm_unpackhi_epi64(top[0], bottom[0]),
            _mm_unpacklo_epi64(top[1], bottom[1]),
            _mm_unpackhi_epi64(top[1], bottom[1]),
            _mm_unpacklo_epi64(top[2], bottom[2]),
            _mm_unpackhi_epi64(top[2], bottom[2]),
            _mm_unpacklo_epi64(top[3], bottom[3]),
            _mm_unpackhi_epi64(top[3], bottom[3]),
        ]
    }

    /// Four columns of sixteen lines turned back: each of the four
    /// vectors given back holds four lines of four bytes, lines 0 to 3
    /// first.
    #[inline]
    fn lines_of_four(c: [__m128i; 4]) -> [__m128i; 4] {
        let (a_lo, a_hi) = (_mm_unpacklo_epi8(c[0], c[1]), _mm_unpackhi_epi8(c[0], c[1]));
        let (b_lo, b_hi) = (_mm_unpacklo_epi8(c[2], c[3]), _mm_unpackhi_epi8(c[2], c[3]));
        [
            _mm_unpacklo_epi16(a_lo, b_lo),
            _mm_unpackhi_epi16(a_lo, b_lo),
            _mm_unpacklo_epi16(a_hi, b_hi),
            _mm_unpackhi_epi16(a_hi, b_hi),
        ]
    }

    /// The four lines of four bytes in `lines` to `first` and the three
    /// rows after it.
    ///
    /// # Safety
    ///
    /// Four bytes must be writable in each of the four rows from `first`.
    #[inline]
    unsafe fn store_four_lines(first: *mut u8, down: usize, lines: __m128i) {
        // SAFETY: the four lines the caller promises.
        unsafe {
            first.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(lines));
            first.add(down).cast::<i32>().write_unaligned(_mm_extract_epi32::<1>(lines));
            first.add(2 * down).cast::<i32>().write_unaligned(_mm_extract_epi32::<2>(lines));
            first.add(3 * down).cast::<i32>().write_unaligned(_mm_extract_epi32::<3>(lines));
        }
    }

    /// A luma edge down the picture: `q0` is its first sample right of
    /// the edge, in rows `down` apart.
    ///
    /// # Safety
    ///
    /// Eight bytes from four before `q0` must be readable and writable in
    /// each of the sixteen rows from its own.
    #[inline]
    unsafe fn luma_lines_at(q0: *mut u8, down: usize, alpha: u8, beta: u8, tc0: Option<[i8; 4]>) {
        // SAFETY: the sixteen lines the caller promises.
        unsafe {
            let first = q0.sub(4);
            let [p3, p2, p1, p0, q0, q1, q2, q3] = load_lines(first, down);
            match tc0 {
                Some(tc0) => {
                    let Some(new) = normal([p2, p1, p0, q0, q1, q2], alpha, beta, spread4(tc0))
                    else {
                        return;
                    };
                    // Only p1 to q1 change: four bytes a line.
                    let fours = lines_of_four(new);
                    for (index, lines) in fours.into_iter().enumerate() {
                        store_four_lines(first.add(2 + 4 * index * down), down, lines);
                    }
                }
                None => {
                    let Some(new) = strong([p3, p2, p1, p0, q0, q1, q2, q3], alpha, beta) else {
                        return;
                    };
                    // p2 to q2 change: p3 and q3 go back as they were.
                    let left = lines_of_four([p3, new[0], new[1], new[2]]);
                    let right = lines_of_four([new[3], new[4], new[5], q3]);
                    for (index, (left, right)) in left.into_iter().zip(right).enumerate() {
                        let at = first.add(4 * index * down);
                        store_four_lines(at, down, left);
                        store_four_lines(at.add(4), down, right);
                    }
                }
            }
        }
    }

    /// The eight lines of four bytes from `first`, rows `down` apart,
    /// turned: p1 in the low half of the first vector and p0 in its high
    /// half, then q0 and q1 the same way in the second.
    ///
    /// # Safety
    ///
    /// Four bytes must be readable in each of the eight rows from `first`.
    #[inline]
    unsafe fn load_chroma_lines(first: *const u8, down: usize) -> (__m128i, __m128i) {
        // SAFETY: the eight lines the caller promises.
        let pair = |index: usize| unsafe {
            _mm_unpacklo_epi8(ld4(first.add(index * down)), ld4(first.add((index + 1) * down)))
        };
        let top = _mm_unpacklo_epi16(pair(0), pair(2));
        let bottom = _mm_unpacklo_epi16(pair(4), pair(6));
        (_mm_unpacklo_epi32(top, bottom), _mm_unpackhi_epi32(top, bottom))
    }

    /// Eight lines' new p0 and q0, a pair of bytes a line in `pairs`, to
    /// the two bytes at `first` and in the seven rows after it.
    ///
    /// # Safety
    ///
    /// Two bytes must be writable in each of the eight rows from `first`.
    #[inline]
    unsafe fn store_chroma_pairs(first: *mut u8, down: usize, pairs: __m128i) {
        // SAFETY: the eight lines the caller promises.
        unsafe {
            let put = |index: usize, pair: i32| {
                first.add(index * down).cast::<u16>().write_unaligned(pair as u16);
            };
            put(0, _mm_extract_epi16::<0>(pairs));
            put(1, _mm_extract_epi16::<1>(pairs));
            put(2, _mm_extract_epi16::<2>(pairs));
            put(3, _mm_extract_epi16::<3>(pairs));
            put(4, _mm_extract_epi16::<4>(pairs));
            put(5, _mm_extract_epi16::<5>(pairs));
            put(6, _mm_extract_epi16::<6>(pairs));
            put(7, _mm_extract_epi16::<7>(pairs));
        }
    }

    /// A chroma edge along a row: `q0` is its first sample below the
    /// edge. `tc0` is per two samples.
    ///
    /// # Safety
    ///
    /// Eight bytes must be readable and writable in each of the two rows
    /// before `q0` and the two from it.
    #[inline]
    unsafe fn chroma_rows_at(q0: *mut u8, down: usize, alpha: u8, beta: u8, tc0: Option<[i8; 4]>) {
        // SAFETY: the four rows the caller promises.
        unsafe {
            let v = [ld8(q0.sub(2 * down)), ld8(q0.sub(down)), ld8(q0), ld8(q0.add(down))];
            if let Some((p0, new_q0)) = chroma_filtered(v, alpha, beta, spread2(tc0), 0xff) {
                st8(q0.sub(down), p0);
                st8(q0, new_q0);
            }
        }
    }

    /// A chroma edge down the picture: `q0` is its first sample right of
    /// the edge.
    ///
    /// # Safety
    ///
    /// Four bytes from two before `q0` must be readable and writable in
    /// each of the eight rows from its own.
    #[inline]
    unsafe fn chroma_lines_at(q0: *mut u8, down: usize, alpha: u8, beta: u8, tc0: Option<[i8; 4]>) {
        // SAFETY: the eight lines the caller promises.
        unsafe {
            let first = q0.sub(2);
            let (p, q) = load_chroma_lines(first, down);
            let v = [p, _mm_srli_si128::<8>(p), q, _mm_srli_si128::<8>(q)];
            if let Some((p0, q0)) = chroma_filtered(v, alpha, beta, spread2(tc0), 0xff) {
                store_chroma_pairs(first.add(1), down, _mm_unpacklo_epi8(p0, q0));
            }
        }
    }

    /// [`chroma_rows_at`] on the same edge of both chroma planes at once,
    /// Cb in the low half of each vector and Cr in the high.
    ///
    /// # Safety
    ///
    /// As for [`chroma_rows_at`], at `cb` and at `cr`.
    #[inline]
    unsafe fn chroma_rows_pair(cb: *mut u8, cr: *mut u8, down: usize, filter: &Filter) {
        // SAFETY: the four rows of each plane the caller promises.
        unsafe {
            let both = |back: usize, on: usize| {
                let (cb, cr) = (cb.sub(back * down).add(on * down), cr.sub(back * down).add(on * down));
                _mm_unpacklo_epi64(ld8(cb), ld8(cr))
            };
            let v = [both(2, 0), both(1, 0), both(0, 0), both(0, 1)];
            let tc0 = spread2(filter.tc0);
            if let Some((p0, q0)) = chroma_filtered(v, filter.alpha, filter.beta, tc0, 0xffff) {
                st8(cb.sub(down), p0);
                st8(cr.sub(down), _mm_srli_si128::<8>(p0));
                st8(cb, q0);
                st8(cr, _mm_srli_si128::<8>(q0));
            }
        }
    }

    /// [`chroma_lines_at`] on the same edge of both chroma planes at
    /// once.
    ///
    /// # Safety
    ///
    /// As for [`chroma_lines_at`], at `cb` and at `cr`.
    #[inline]
    unsafe fn chroma_lines_pair(cb: *mut u8, cr: *mut u8, down: usize, filter: &Filter) {
        // SAFETY: the eight lines of each plane the caller promises.
        unsafe {
            let (cb, cr) = (cb.sub(2), cr.sub(2));
            let (cb_p, cb_q) = load_chroma_lines(cb, down);
            let (cr_p, cr_q) = load_chroma_lines(cr, down);
            let v = [
                _mm_unpacklo_epi64(cb_p, cr_p),
                _mm_unpackhi_epi64(cb_p, cr_p),
                _mm_unpacklo_epi64(cb_q, cr_q),
                _mm_unpackhi_epi64(cb_q, cr_q),
            ];
            let tc0 = spread2(filter.tc0);
            if let Some((p0, q0)) = chroma_filtered(v, filter.alpha, filter.beta, tc0, 0xffff) {
                store_chroma_pairs(cb.add(1), down, _mm_unpacklo_epi8(p0, q0));
                store_chroma_pairs(cr.add(1), down, _mm_unpackhi_epi8(p0, q0));
            }
        }
    }

    /// See [`crate::deblock::luma_normal_rows`].
    pub(crate) fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
        let Some((alpha, beta)) = small(alpha, beta, Some(tc0)) else {
            return crate::scalar::luma_normal_rows(rows, alpha, beta, tc0);
        };
        // SAFETY: `rows` is eight rows of sixteen bytes, sixteen apart,
        // and q0 the fifth.
        unsafe { luma_rows_at(rows.as_mut_ptr().cast::<u8>().add(64), 16, alpha, beta, Some(tc0)) };
    }

    /// See [`crate::deblock::luma_strong_rows`].
    pub(crate) fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
        let Some((alpha, beta)) = small(alpha, beta, None) else {
            return crate::scalar::luma_strong_rows(rows, alpha, beta);
        };
        // SAFETY: as in `luma_normal_rows`.
        unsafe { luma_rows_at(rows.as_mut_ptr().cast::<u8>().add(64), 16, alpha, beta, None) };
    }

    /// See [`crate::deblock::luma_normal_lines`].
    pub(crate) fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
        let Some((alpha, beta)) = small(alpha, beta, Some(tc0)) else {
            return crate::scalar::luma_normal_lines(lines, alpha, beta, tc0);
        };
        // SAFETY: `lines` is sixteen lines of eight bytes, eight apart,
        // and q0 the fifth byte of each.
        unsafe { luma_lines_at(lines.as_mut_ptr().cast::<u8>().add(4), 8, alpha, beta, Some(tc0)) };
    }

    /// See [`crate::deblock::luma_strong_lines`].
    pub(crate) fn luma_strong_lines(lines: &mut Lines, alpha: i32, beta: i32) {
        let Some((alpha, beta)) = small(alpha, beta, None) else {
            return crate::scalar::luma_strong_lines(lines, alpha, beta);
        };
        // SAFETY: as in `luma_normal_lines`.
        unsafe { luma_lines_at(lines.as_mut_ptr().cast::<u8>().add(4), 8, alpha, beta, None) };
    }

    /// See [`crate::deblock::chroma_normal`].
    pub(crate) fn chroma_normal(rows: &mut ChromaRows, alpha: i32, beta: i32, tc0: [i8; 4]) {
        let Some((alpha, beta)) = small(alpha, beta, Some(tc0)) else {
            return crate::scalar::chroma_normal(rows, alpha, beta, tc0);
        };
        // SAFETY: `rows` is four rows of eight bytes, eight apart, and q0
        // the third.
        unsafe { chroma_rows_at(rows.as_mut_ptr().cast::<u8>().add(16), 8, alpha, beta, Some(tc0)) };
    }

    /// See [`crate::deblock::chroma_strong`].
    pub(crate) fn chroma_strong(rows: &mut ChromaRows, alpha: i32, beta: i32) {
        let Some((alpha, beta)) = small(alpha, beta, None) else {
            return crate::scalar::chroma_strong(rows, alpha, beta);
        };
        // SAFETY: as in `chroma_normal`.
        unsafe { chroma_rows_at(rows.as_mut_ptr().cast::<u8>().add(16), 8, alpha, beta, None) };
    }

    /// See [`crate::deblock::luma_edge`]: the plain twin's checks, then
    /// the samples go straight between the plane and registers.
    pub(crate) fn luma_edge(plane: &mut [u8], edge: Edge) -> bool {
        let Some((alpha, beta)) = small(edge.alpha, edge.beta, edge.tc0) else {
            return crate::scalar::luma_edge(plane, edge);
        };
        let Edge { at, down, vertical, tc0, .. } = edge;
        if vertical {
            let Some(first) = at.checked_sub(4) else {
                return false;
            };
            let end = 15usize
                .checked_mul(down)
                .and_then(|rows| rows.checked_add(first))
                .and_then(|start| start.checked_add(8));
            if end.is_none_or(|end| end > plane.len()) {
                return false;
            }
            // SAFETY: sixteen lines of eight bytes from `first` are in
            // the plane.
            unsafe { luma_lines_at(plane.as_mut_ptr().add(at), down, alpha, beta, tc0) };
            return true;
        }
        let Some(first) = down.checked_mul(4).and_then(|rows| at.checked_sub(rows)) else {
            return false;
        };
        let end = 7usize
            .checked_mul(down)
            .and_then(|rows| rows.checked_add(first))
            .and_then(|start| start.checked_add(16));
        if end.is_none_or(|end| end > plane.len()) {
            return false;
        }
        // SAFETY: eight rows of sixteen bytes from `first` are in the
        // plane.
        unsafe { luma_rows_at(plane.as_mut_ptr().add(at), down, alpha, beta, tc0) };
        true
    }

    /// See [`crate::deblock::chroma_edge`].
    pub(crate) fn chroma_edge(plane: &mut [u8], edge: Edge) -> bool {
        let Some((alpha, beta)) = small(edge.alpha, edge.beta, edge.tc0) else {
            return crate::scalar::chroma_edge(plane, edge);
        };
        let Edge { at, down, vertical, tc0, .. } = edge;
        if vertical {
            let Some(first) = at.checked_sub(2) else {
                return false;
            };
            let end = 7usize
                .checked_mul(down)
                .and_then(|rows| rows.checked_add(first))
                .and_then(|start| start.checked_add(4));
            if end.is_none_or(|end| end > plane.len()) {
                return false;
            }
            // SAFETY: eight lines of four bytes from `first` are in the
            // plane.
            unsafe { chroma_lines_at(plane.as_mut_ptr().add(at), down, alpha, beta, tc0) };
            return true;
        }
        let Some(first) = down.checked_mul(2).and_then(|rows| at.checked_sub(rows)) else {
            return false;
        };
        let end = 3usize
            .checked_mul(down)
            .and_then(|rows| rows.checked_add(first))
            .and_then(|start| start.checked_add(8));
        if end.is_none_or(|end| end > plane.len()) {
            return false;
        }
        // SAFETY: four rows of eight bytes from `first` are in the plane.
        unsafe { chroma_rows_at(plane.as_mut_ptr().add(at), down, alpha, beta, tc0) };
        true
    }

    /// One edge of a macroblock whose samples `mb_fits` has found in the
    /// plane: `step` samples from one edge to the next, 4 in luma and 2
    /// in chroma. A clipping value past the 8-bit filters' range sends
    /// the edge to the plain twin.
    #[inline]
    fn mb_edge(
        plane: &mut [u8],
        edges: &MbEdges,
        (vertical, index): (bool, usize),
        filter: &Filter,
        step: usize,
    ) {
        let at = edges.at + if vertical { step * index } else { step * index * edges.down };
        if !filter.tame() {
            let edge = Edge {
                at,
                down: edges.down,
                vertical,
                alpha: i32::from(filter.alpha),
                beta: i32::from(filter.beta),
                tc0: filter.tc0,
            };
            if step == 4 {
                crate::scalar::luma_edge(plane, edge);
            } else {
                crate::scalar::chroma_edge(plane, edge);
            }
            return;
        }
        let (down, alpha, beta, tc0) = (edges.down, filter.alpha, filter.beta, filter.tc0);
        // SAFETY: `mb_fits` put the macroblock in the plane, and with it
        // the samples before its first edge of each kind when that edge
        // is filtered; every other edge's samples are the macroblock's.
        unsafe {
            let q0 = plane.as_mut_ptr().add(at);
            match (step == 4, vertical) {
                (true, true) => luma_lines_at(q0, down, alpha, beta, tc0),
                (true, false) => luma_rows_at(q0, down, alpha, beta, tc0),
                (false, true) => chroma_lines_at(q0, down, alpha, beta, tc0),
                (false, false) => chroma_rows_at(q0, down, alpha, beta, tc0),
            }
        }
    }

    /// Every edge of a macroblock `mb_fits` has found in the plane, in
    /// the standard's order.
    #[inline]
    fn mb_edges(plane: &mut [u8], edges: &MbEdges, step: usize) {
        for (index, filter) in edges.vertical.iter().enumerate() {
            if let Some(filter) = filter {
                mb_edge(plane, edges, (true, index), filter, step);
            }
        }
        for (index, filter) in edges.horizontal.iter().enumerate() {
            if let Some(filter) = filter {
                mb_edge(plane, edges, (false, index), filter, step);
            }
        }
    }

    /// See [`crate::deblock::luma_mb`].
    pub(crate) fn luma_mb(plane: &mut [u8], edges: &MbEdges) -> bool {
        if !crate::edge::mb_fits(plane.len(), edges, 16, 4) {
            return false;
        }
        mb_edges(plane, edges, 4);
        true
    }

    /// See [`crate::deblock::chroma_mb`].
    pub(crate) fn chroma_mb(plane: &mut [u8], edges: &MbEdges) -> bool {
        if !crate::edge::mb_fits(plane.len(), edges, 8, 2) {
            return false;
        }
        mb_edges(plane, edges, 2);
        true
    }

    /// See [`crate::deblock::mb`].
    pub(crate) fn mb_planes(
        planes: (&mut [u8], &mut [u8], &mut [u8]),
        edges: &[MbEdges; 3],
        some: [bool; 3],
    ) -> bool {
        let (luma, cb, cr) = planes;
        let [for_luma, for_cb, for_cr] = edges;
        let [some_luma, some_cb, some_cr] = some;
        if (some_luma && !for_luma.fits(luma.len(), 16))
            || (some_cb && !for_cb.fits(cb.len(), 8))
            || (some_cr && !for_cr.fits(cr.len(), 8))
        {
            return false;
        }
        if some_luma {
            mb_edges(luma, for_luma, 4);
        }
        if some_cb {
            mb_edges(cb, for_cb, 2);
        }
        if some_cr {
            mb_edges(cr, for_cr, 2);
        }
        true
    }

    /// See [`crate::deblock::mb_twin`]: the two chroma planes are
    /// filtered side by side, sixteen lines to a register.
    pub(crate) fn mb_twin(
        planes: (&mut [u8], &mut [u8], &mut [u8]),
        luma: &MbEdges,
        chroma: &MbEdges,
        some: [bool; 2],
    ) -> bool {
        let (y, cb, cr) = planes;
        let [some_luma, some_chroma] = some;
        if (some_luma && !luma.fits(y.len(), 16))
            || (some_chroma && !(chroma.fits(cb.len(), 8) && chroma.fits(cr.len(), 8)))
        {
            return false;
        }
        if some_luma {
            mb_edges(y, luma, 4);
        }
        if !some_chroma {
            return true;
        }
        let tame = |filter: &Option<Filter>| filter.is_none_or(|filter| filter.tame());
        if !(chroma.vertical.iter().all(tame) && chroma.horizontal.iter().all(tame)) {
            mb_edges(cb, chroma, 2);
            mb_edges(cr, chroma, 2);
            return true;
        }
        let (at, down) = (chroma.at, chroma.down);
        // SAFETY: `fits` put the macroblock in both planes, and with it
        // the samples before its first edge of each kind when that edge
        // is filtered; every other edge's samples are the macroblock's.
        unsafe {
            let (cb, cr) = (cb.as_mut_ptr().add(at), cr.as_mut_ptr().add(at));
            for (index, filter) in chroma.vertical.iter().enumerate() {
                if let Some(filter) = filter {
                    chroma_lines_pair(cb.add(2 * index), cr.add(2 * index), down, filter);
                }
            }
            for (index, filter) in chroma.horizontal.iter().enumerate() {
                if let Some(filter) = filter {
                    let rows = 2 * index * down;
                    chroma_rows_pair(cb.add(rows), cr.add(rows), down, filter);
                }
            }
        }
        true
    }
}

feat! {
    /// Four vectors of four 32-bit lanes, turned: rows become columns.
    #[inline]
    fn turn4(v: [__m128i; 4]) -> [__m128i; 4] {
        let a = _mm_unpacklo_epi32(v[0], v[1]);
        let b = _mm_unpackhi_epi32(v[0], v[1]);
        let c = _mm_unpacklo_epi32(v[2], v[3]);
        let d = _mm_unpackhi_epi32(v[2], v[3]);
        [
            _mm_unpacklo_epi64(a, c),
            _mm_unpackhi_epi64(a, c),
            _mm_unpacklo_epi64(b, d),
            _mm_unpackhi_epi64(b, d),
        ]
    }

    /// One pass of the 4×4 transform, clause 8.5.12.2, on four lines at
    /// once, wrapping as the plain twin does.
    #[inline]
    fn pass4(d: [__m128i; 4]) -> [__m128i; 4] {
        let e0 = _mm_add_epi32(d[0], d[2]);
        let e1 = _mm_sub_epi32(d[0], d[2]);
        let e2 = _mm_sub_epi32(_mm_srai_epi32::<1>(d[1]), d[3]);
        let e3 = _mm_add_epi32(d[1], _mm_srai_epi32::<1>(d[3]));
        [
            _mm_add_epi32(e0, e3),
            _mm_add_epi32(e1, e2),
            _mm_sub_epi32(e1, e2),
            _mm_sub_epi32(e0, e3),
        ]
    }

    /// Eight samples, in the low half of `samples`, each plus its
    /// residual, (value + 32) >> 6 of the eight 32-bit lanes in `lo` and
    /// `hi`, held to 0 to 255, in the low half. Narrowing the residual
    /// to 16 bits with saturation and adding with saturation gives what
    /// the plain twin's saturating 32-bit sum gives once clipped.
    #[inline]
    fn add_residual(samples: __m128i, lo: __m128i, hi: __m128i) -> __m128i {
        let round = _mm_set1_epi32(32);
        let residual = _mm_packs_epi32(
            _mm_srai_epi32::<6>(_mm_add_epi32(lo, round)),
            _mm_srai_epi32::<6>(_mm_add_epi32(hi, round)),
        );
        let lanes = _mm_adds_epi16(low(samples), residual);
        _mm_packus_epi16(lanes, lanes)
    }

    /// See [`crate::residual::idct4_add`].
    pub(crate) fn idct4_add(plane: &mut [u8], at: usize, down: usize, block: &[i32; 16]) -> bool {
        let end = 3usize
            .checked_mul(down)
            .and_then(|span| span.checked_add(at))
            .and_then(|start| start.checked_add(4));
        if down < 4 || end.is_none_or(|end| end > plane.len()) {
            return false;
        }
        // SAFETY: `block` is sixteen 32-bit levels, and the four rows of
        // four samples from `at` are in the plane, as checked above.
        unsafe {
            let levels = block.as_ptr().cast::<u8>();
            let rows = [ld16(levels), ld16(levels.add(16)), ld16(levels.add(32)),
                ld16(levels.add(48))];
            // Along each row, then down each column.
            let out = pass4(turn4(pass4(turn4(rows))));
            let first = plane.as_mut_ptr().add(at);
            let top = _mm_unpacklo_epi32(ld4(first), ld4(first.add(down)));
            let bottom = _mm_unpacklo_epi32(ld4(first.add(2 * down)), ld4(first.add(3 * down)));
            let top = add_residual(top, out[0], out[1]);
            let bottom = add_residual(bottom, out[2], out[3]);
            st4(first, top);
            st4(first.add(down), _mm_srli_si128::<4>(top));
            st4(first.add(2 * down), bottom);
            st4(first.add(3 * down), _mm_srli_si128::<4>(bottom));
        }
        true
    }

    /// One pass of the 8×8 transform, clause 8.5.13.2, on four lines at
    /// once, wrapping as the plain twin's `idct8_line` does.
    #[inline]
    fn pass8(d: [__m128i; 8]) -> [__m128i; 8] {
        let half = |v: __m128i| _mm_srai_epi32::<1>(v);
        let quarter = |v: __m128i| _mm_srai_epi32::<2>(v);
        let a0 = _mm_add_epi32(d[0], d[4]);
        let a2 = _mm_sub_epi32(d[0], d[4]);
        let a4 = _mm_sub_epi32(half(d[2]), d[6]);
        let a6 = _mm_add_epi32(half(d[6]), d[2]);
        let b0 = _mm_add_epi32(a0, a6);
        let b2 = _mm_add_epi32(a2, a4);
        let b4 = _mm_sub_epi32(a2, a4);
        let b6 = _mm_sub_epi32(a0, a6);
        let a1 = _mm_sub_epi32(_mm_sub_epi32(_mm_sub_epi32(d[5], d[3]), d[7]), half(d[7]));
        let a3 = _mm_sub_epi32(_mm_sub_epi32(_mm_add_epi32(d[1], d[7]), d[3]), half(d[3]));
        let a5 = _mm_add_epi32(_mm_add_epi32(_mm_sub_epi32(d[7], d[1]), d[5]), half(d[5]));
        let a7 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(d[3], d[5]), d[1]), half(d[1]));
        let b1 = _mm_add_epi32(quarter(a7), a1);
        let b3 = _mm_add_epi32(a3, quarter(a5));
        let b5 = _mm_sub_epi32(quarter(a3), a5);
        let b7 = _mm_sub_epi32(a7, quarter(a1));
        [
            _mm_add_epi32(b0, b7),
            _mm_add_epi32(b2, b5),
            _mm_add_epi32(b4, b3),
            _mm_add_epi32(b6, b1),
            _mm_sub_epi32(b6, b1),
            _mm_sub_epi32(b4, b3),
            _mm_sub_epi32(b2, b5),
            _mm_sub_epi32(b0, b7),
        ]
    }

    /// Eight rows of eight 32-bit lanes, each row the four lanes in `lo`
    /// then the four in `hi`, turned: rows become columns.
    #[inline]
    fn turn8(lo: [__m128i; 8], hi: [__m128i; 8]) -> ([__m128i; 8], [__m128i; 8]) {
        // The four quarters turn on their own; the two off the diagonal
        // change places.
        let a = turn4([lo[0], lo[1], lo[2], lo[3]]);
        let b = turn4([hi[0], hi[1], hi[2], hi[3]]);
        let c = turn4([lo[4], lo[5], lo[6], lo[7]]);
        let d = turn4([hi[4], hi[5], hi[6], hi[7]]);
        (
            [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]],
            [c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3]],
        )
    }

    /// See [`crate::residual::idct8_add`].
    pub(crate) fn idct8_add(plane: &mut [u8], at: usize, down: usize, block: &[i32; 64]) -> bool {
        let end = 7usize
            .checked_mul(down)
            .and_then(|span| span.checked_add(at))
            .and_then(|start| start.checked_add(8));
        if down < 8 || end.is_none_or(|end| end > plane.len()) {
            return false;
        }
        // SAFETY: `block` is 64 32-bit levels, and the eight rows of
        // eight samples from `at` are in the plane, as checked above.
        unsafe {
            let levels = block.as_ptr().cast::<u8>();
            let mut lo = [_mm_setzero_si128(); 8];
            let mut hi = lo;
            for (row, (lo, hi)) in lo.iter_mut().zip(&mut hi).enumerate() {
                *lo = ld16(levels.add(32 * row));
                *hi = ld16(levels.add(32 * row + 16));
            }
            // Along each row: the columns as vectors, a pass, and back to
            // rows. Then down each column.
            let (lo, hi) = turn8(lo, hi);
            let (lo, hi) = turn8(pass8(lo), pass8(hi));
            let (lo, hi) = (pass8(lo), pass8(hi));
            let first = plane.as_mut_ptr().add(at);
            for (row, (lo, hi)) in lo.into_iter().zip(hi).enumerate() {
                let samples = first.add(row * down);
                st8(samples, add_residual(ld8(samples), lo, hi));
            }
        }
        true
    }

    /// See [`crate::residual::dc_add`].
    pub(crate) fn dc_add(plane: &mut [u8], at: usize, down: usize, dc: i32, size: usize) -> bool {
        if !(size == 4 || size == 8) {
            return false;
        }
        let Some(rest) = plane.len().checked_sub(at) else {
            return false;
        };
        if (size - 1)
            .checked_mul(down)
            .and_then(|rows| rows.checked_add(size))
            .is_none_or(|end| end > rest)
        {
            return false;
        }
        // A sample moves by at most 255 either way.
        let dc = dc.clamp(-255, 255);
        let rise = splat(dc.max(0) as u8);
        let fall = splat((-dc).max(0) as u8);
        // SAFETY: the `size` rows of `size` samples from `at` are in the
        // plane, as checked above.
        unsafe {
            let first = plane.as_mut_ptr().add(at);
            for row in 0..size {
                let samples = first.add(row * down);
                if size == 8 {
                    st8(samples, _mm_subs_epu8(_mm_adds_epu8(ld8(samples), rise), fall));
                } else {
                    st4(samples, _mm_subs_epu8(_mm_adds_epu8(ld4(samples), rise), fall));
                }
            }
        }
        true
    }

    /// See [`crate::residual::dc_add_quads`].
    pub(crate) fn dc_add_quads(plane: &mut [u8], at: usize, down: usize, dcs: [i32; 4]) -> bool {
        if down < 8
            || 7usize
                .checked_mul(down)
                .and_then(|rows| rows.checked_add(8))
                .and_then(|span| span.checked_add(at))
                .is_none_or(|end| end > plane.len())
        {
            return false;
        }
        let dcs = dcs.map(|dc| dc.clamp(-255, 255));
        // How far the samples rise and how far they fall: the left
        // quarter's value in the low four bytes and the right one's in the
        // next four.
        let four = |value: i32| (value.max(0) as u32).wrapping_mul(0x0101_0101).cast_signed();
        let pair = |left: i32, right: i32| _mm_set_epi32(0, 0, four(right), four(left));
        let moves = |left: i32, right: i32| (pair(left, right), pair(-left, -right));
        let halves = [moves(dcs[0], dcs[1]), moves(dcs[2], dcs[3])];
        // SAFETY: the eight rows of eight samples from `at` are in the
        // plane, as checked above.
        unsafe {
            let first = plane.as_mut_ptr().add(at);
            for row in 0..8 {
                let (rise, fall) = halves[row / 4];
                let samples = first.add(row * down);
                st8(samples, _mm_subs_epu8(_mm_adds_epu8(ld8(samples), rise), fall));
            }
        }
        true
    }

    /// See [`crate::residual::store`]: rows copied as the plain twin
    /// copies them.
    #[inline]
    pub(crate) fn store<const N: usize>(
        plane: &mut [u8],
        at: usize,
        down: usize,
        rows: &[[u8; N]; N],
    ) -> bool {
        crate::scalar::store(plane, at, down, rows)
    }

    /// (a + 2·b + c + 2) >> 2 in each byte: the rounded average of b and
    /// the halved sum of a and c.
    #[inline]
    fn avg3(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
        _mm_avg_epu8(b, half_sum(a, c))
    }

    /// `row` to each of the sixteen rows of a 16×16 block.
    ///
    /// # Safety
    ///
    /// Sixteen bytes must be writable in each of the sixteen rows from
    /// `block`, `down` apart.
    #[inline]
    unsafe fn fill16(block: *mut u8, down: usize, row: __m128i) {
        for y in 0..16 {
            // SAFETY: the caller's promise.
            unsafe { st16(block.add(y * down), row) };
        }
    }

    /// The rows of a plane prediction, equations 8-126 and 8-144: sample
    /// (x, y) is (`first` + b·x + c·y) >> 5 clipped, where `first` is the
    /// sum at the block's first sample. Every sum fits 16 bits: see the
    /// NEON build's note. `W` is 8 or 16, the block's width and height.
    ///
    /// # Safety
    ///
    /// `W` bytes must be writable in each of the `W` rows from `block`,
    /// `down` apart.
    #[inline]
    unsafe fn plane_rows<const W: usize>(block: *mut u8, down: usize, first: i32, b: i32, c: i32) {
        let steps = _mm_mullo_epi16(
            _mm_set1_epi16(b as i16),
            _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0),
        );
        let mut lo = _mm_add_epi16(_mm_set1_epi16(first as i16), steps);
        let mut hi = _mm_add_epi16(lo, _mm_set1_epi16((8 * b) as i16));
        let c = _mm_set1_epi16(c as i16);
        for y in 0..W {
            // SAFETY: the caller's promise.
            unsafe {
                let row = block.add(y * down);
                if W == 16 {
                    st16(row, _mm_packus_epi16(_mm_srai_epi16::<5>(lo), _mm_srai_epi16::<5>(hi)));
                } else {
                    let lanes = _mm_srai_epi16::<5>(lo);
                    st8(row, _mm_packus_epi16(lanes, lanes));
                }
            }
            lo = _mm_add_epi16(lo, c);
            hi = _mm_add_epi16(hi, c);
        }
    }

    /// See [`crate::intra::pred16`].
    pub(crate) fn pred16(
        plane: &mut [u8],
        at: usize,
        down: usize,
        mode: u8,
        near: (bool, bool),
    ) -> bool {
        let Some(start) = crate::intra::corner_at::<16>(plane.len(), at, down) else {
            return false;
        };
        let (above, left) = near;
        if mode == 3 && above && left {
            let (corner, up, side) = crate::scalar::beside::<16>(plane, start, down);
            let h = crate::scalar::gradient(&up, corner);
            let v = crate::scalar::gradient(&side, corner);
            let (b, c) = ((5 * h + 32) >> 6, (5 * v + 32) >> 6);
            let first = 16 * (up[15] + side[15]) + 16 - 7 * (b + c);
            // SAFETY: `corner_at` found the block in the plane.
            unsafe { plane_rows::<16>(plane.as_mut_ptr().add(at), down, first, b, c) };
            return true;
        }
        // SAFETY: `corner_at` found the block in the plane, with the row
        // above it from the corner on and the column to its left.
        unsafe {
            let block = plane.as_mut_ptr().add(at);
            let up = ld16(block.sub(down));
            match mode {
                0 if above => fill16(block, down, up),
                1 if left => {
                    for y in 0..16 {
                        let row = block.add(y * down);
                        st16(row, splat(*row.sub(1)));
                    }
                }
                2 if above || left => {
                    let halves = _mm_sad_epu8(up, _mm_setzero_si128());
                    let along = _mm_cvtsi128_si32(halves) + _mm_extract_epi16::<4>(halves);
                    let mut beside = 0i32;
                    for y in 0..16 {
                        beside += i32::from(*block.add(y * down).sub(1));
                    }
                    let dc = match (above, left) {
                        (true, true) => (along + beside + 16) >> 5,
                        (true, false) => (along + 8) >> 4,
                        _ => (beside + 8) >> 4,
                    };
                    fill16(block, down, splat(dc as u8));
                }
                _ => fill16(block, down, splat(128)),
            }
        }
        true
    }

    /// See [`crate::intra::pred_chroma`].
    pub(crate) fn pred_chroma(
        plane: &mut [u8],
        at: usize,
        down: usize,
        mode: u8,
        near: (bool, bool),
    ) -> bool {
        let Some(start) = crate::intra::corner_at::<8>(plane.len(), at, down) else {
            return false;
        };
        let (above, left) = near;
        let (corner, up, side) = crate::scalar::beside::<8>(plane, start, down);
        // SAFETY: `corner_at` found the block in the plane, with the row
        // above it and the column to its left.
        unsafe {
            let block = plane.as_mut_ptr().add(at);
            match mode {
                1 if left => {
                    for y in 0..8 {
                        let row = block.add(y * down);
                        st8(row, splat(*row.sub(1)));
                    }
                }
                2 if above => {
                    let row = ld8(block.sub(down));
                    for y in 0..8 {
                        st8(block.add(y * down), row);
                    }
                }
                3 if above && left => {
                    let h = crate::scalar::gradient(&up, corner);
                    let v = crate::scalar::gradient(&side, corner);
                    let (b, c) = ((17 * h + 16) >> 5, (17 * v + 16) >> 5);
                    let first = 16 * (up[7] + side[7]) + 16 - 3 * (b + c);
                    plane_rows::<8>(block, down, first, b, c);
                }
                _ => {
                    // The DC of each 4×4 quarter: from both edges beside
                    // it when the quarter touches both, else from the one
                    // edge it touches, else from the other.
                    let sum = |samples: &[i32; 8], half: usize| {
                        samples[half * 4..half * 4 + 4].iter().sum::<i32>()
                    };
                    let quarter = |x: usize, y: usize| {
                        let (along, beside) = (sum(&up, x), sum(&side, y));
                        let dc = match (above, left) {
                            (true, true) if x == y => (along + beside + 4) >> 3,
                            (true, true) if y == 0 => (along + 2) >> 2,
                            (true, true) | (false, true) => (beside + 2) >> 2,
                            (true, false) => (along + 2) >> 2,
                            (false, false) => 128,
                        };
                        dc as u64 * 0x0101_0101
                    };
                    for half in 0..2 {
                        let row = quarter(0, half) | (quarter(1, half) << 32);
                        for y in 0..4 {
                            block.add((half * 4 + y) * down).cast::<u64>().write_unaligned(row);
                        }
                    }
                }
            }
        }
        true
    }

    /// The two-sample and three-sample averages of the eighteen bytes at
    /// `edge`, sixteen of each, shuffled by `picks` into sixteen samples
    /// of a directional prediction.
    ///
    /// # Safety
    ///
    /// Eighteen bytes from `edge` must be readable.
    #[inline]
    unsafe fn picked(edge: *const u8, picks: &([u8; 16], [u8; 16])) -> __m128i {
        // SAFETY: the caller's promise; each table is sixteen bytes.
        unsafe {
            let (a, b, c) = (ld16(edge), ld16(edge.add(1)), ld16(edge.add(2)));
            _mm_or_si128(
                _mm_shuffle_epi8(_mm_avg_epu8(a, b), ld16(picks.0.as_ptr())),
                _mm_shuffle_epi8(avg3(a, b, c), ld16(picks.1.as_ptr())),
            )
        }
    }

    /// See [`crate::intra::pred8`]: the edges are gathered and filtered
    /// as the plain twin does it, and the block made from them.
    pub(crate) fn pred8(plane: &mut [u8], at: usize, down: usize, mode: u8, near: Near8) -> bool {
        let Some(start) = crate::intra::corner8_at(plane.len(), at, down) else {
            return false;
        };
        let edges = crate::scalar::edges8(plane, (start, at, down), near);
        let (above, left) = (&edges.above, &edges.left);
        let both = near.above && near.left;
        let directional = match mode {
            3 | 7 => near.above,
            4..=6 => both,
            8 => near.left,
            _ => false,
        };
        // SAFETY: `corner8_at` found the block's eight rows of eight in
        // the plane. The edge buffers hold the 18 bytes a shuffle reads.
        unsafe {
            let block = plane.as_mut_ptr().add(at);
            if directional {
                // The samples a direction reads, in a row: the column to
                // the left, bottom first, the corner, and the row above;
                // or, for horizontal-up, the column top first. The last
                // sample repeats past each end.
                let mut line = [0u8; 34];
                match mode {
                    3 | 7 => {
                        line[..16].copy_from_slice(above);
                        line[16..].fill(above[15]);
                    }
                    8 => {
                        line[..8].copy_from_slice(left);
                        line[8..].fill(left[7]);
                    }
                    _ => {
                        for (sample, &value) in line.iter_mut().zip(left.iter().rev()) {
                            *sample = value;
                        }
                        line[8] = edges.corner;
                        line[9..25].copy_from_slice(above);
                    }
                }
                let picks = &PICKS8[usize::from(mode - 3)];
                for (pair, picks) in picks.iter().enumerate() {
                    let rows = picked(line.as_ptr(), picks);
                    st8(block.add(2 * pair * down), rows);
                    st8(block.add((2 * pair + 1) * down), _mm_srli_si128::<8>(rows));
                }
                return true;
            }
            match mode {
                0 if near.above => {
                    let row = ld8(above.as_ptr());
                    for y in 0..8 {
                        st8(block.add(y * down), row);
                    }
                }
                1 if near.left => {
                    for (y, &sample) in left.iter().enumerate() {
                        st8(block.add(y * down), splat(sample));
                    }
                }
                _ => {
                    let sum = |samples: &[u8]| samples.iter().map(|&s| u32::from(s)).sum::<u32>();
                    let dc = match (near.above, near.left) {
                        (true, true) => (sum(&above[..8]) + sum(left) + 8) >> 4,
                        (true, false) => (sum(&above[..8]) + 4) >> 3,
                        (false, true) => (sum(left) + 4) >> 3,
                        (false, false) => 128,
                    };
                    let row = splat(dc as u8);
                    for y in 0..8 {
                        st8(block.add(y * down), row);
                    }
                }
            }
        }
        true
    }

    /// See [`crate::intra::pred4`]: the edges are gathered as the plain
    /// twin does it, and the block made from them.
    pub(crate) fn pred4(plane: &mut [u8], at: usize, down: usize, mode: u8, near: Near4) -> bool {
        let Some(start) = crate::intra::corner4_at(plane.len(), at, down) else {
            return false;
        };
        let Some(&top) = plane[start..].first_chunk::<9>() else {
            return false;
        };
        let side = core::array::from_fn(|y| plane[at - 1 + y * down]);
        let (above, left, corner) = crate::intra::edges4(&top, side, near);
        let directional = match mode {
            3 | 7 => near.above,
            4..=6 => near.above && near.left,
            8 => near.left,
            _ => false,
        };
        // SAFETY: `corner4_at` found the block's four rows of four in the
        // plane. The edge buffer holds the 18 bytes a shuffle reads.
        unsafe {
            let block = plane.as_mut_ptr().add(at);
            let rows = if directional {
                let mut line = [0u8; 18];
                match mode {
                    3 | 7 => {
                        line[..8].copy_from_slice(&above);
                        line[8..].fill(above[7]);
                    }
                    8 => {
                        line[..4].copy_from_slice(&left);
                        line[4..].fill(left[3]);
                    }
                    _ => {
                        line[..4].copy_from_slice(&[left[3], left[2], left[1], left[0]]);
                        line[4] = corner;
                        line[5..13].copy_from_slice(&above);
                    }
                }
                picked(line.as_ptr(), &PICKS4[usize::from(mode - 3)])
            } else {
                match mode {
                    0 if near.above => _mm_set1_epi32(i32::from_le_bytes([
                        above[0], above[1], above[2], above[3],
                    ])),
                    1 if near.left => _mm_set_epi32(
                        i32::from_le_bytes([left[3]; 4]),
                        i32::from_le_bytes([left[2]; 4]),
                        i32::from_le_bytes([left[1]; 4]),
                        i32::from_le_bytes([left[0]; 4]),
                    ),
                    _ => splat(crate::intra::dc4(&above, &left, near)),
                }
            };
            st4(block, rows);
            st4(block.add(down), _mm_srli_si128::<4>(rows));
            st4(block.add(2 * down), _mm_srli_si128::<8>(rows));
            st4(block.add(3 * down), _mm_srli_si128::<12>(rows));
        }
        true
    }
}

/// Where sample (`x`, `y`) of an `n`×`n` block comes from in a directional
/// Intra_4x4 or Intra_8x8 mode, 3 to 8: whether it is a three-sample
/// average, (a + 2b + c + 2) >> 2, rather than a two-sample one, and the
/// place of its first sample in the line the mode reads. That line is the
/// row above for modes 3 and 7, the column to the left top first for mode
/// 8, and for the rest the column bottom first, then the corner, then the
/// row above; its last sample repeats. These are equations 8-46 to 8-60
/// and their 8×8 twins with each sample's place written out.
const fn pick(n: i32, mode: u8, x: i32, y: i32) -> (bool, i32) {
    match mode {
        // Diagonal down left.
        3 => (true, x + y),
        // Vertical left.
        7 => (y % 2 == 1, x + y / 2),
        // Diagonal down right.
        4 => (true, n - 1 + x - y),
        // Vertical right.
        5 => {
            let (z, along) = (2 * x - y, x - (y >> 1));
            if z >= 0 && z % 2 == 0 {
                (false, n + along)
            } else if z >= -1 {
                (true, n - 1 + along)
            } else {
                (true, n - y + 2 * x)
            }
        }
        // Horizontal down.
        6 => {
            let (z, along) = (2 * y - x, y - (x >> 1));
            if z >= 0 && z % 2 == 0 {
                (false, n - 1 - along)
            } else if z >= -1 {
                (true, n - 1 - along)
            } else {
                (true, n - 2 + x - 2 * y)
            }
        }
        // Horizontal up.
        _ => (x % 2 == 1, y + x / 2),
    }
}

/// The byte shuffles that make sixteen samples of a directional mode,
/// those from `first` on in raster order of an `n`-wide block, from the
/// two-sample averages of its line and from the three-sample ones: 0x80
/// takes nothing from that set.
const fn picks(n: usize, mode: u8, first: usize) -> ([u8; 16], [u8; 16]) {
    let mut two = [0x80u8; 16];
    let mut three = [0x80u8; 16];
    let mut index = 0;
    while index < 16 {
        let at = first + index;
        let (is_three, from) = pick(n as i32, mode, (at % n) as i32, (at / n) as i32);
        if is_three {
            three[index] = from as u8;
        } else {
            two[index] = from as u8;
        }
        index += 1;
    }
    (two, three)
}

/// [`picks`] for a 4×4 block in each of modes 3 to 8.
const PICKS4: [([u8; 16], [u8; 16]); 6] = [
    picks(4, 3, 0),
    picks(4, 4, 0),
    picks(4, 5, 0),
    picks(4, 6, 0),
    picks(4, 7, 0),
    picks(4, 8, 0),
];

/// [`picks`] for each pair of rows of an 8×8 block in each of modes 3 to
/// 8.
const PICKS8: [[([u8; 16], [u8; 16]); 4]; 6] = {
    let mut all = [[([0u8; 16], [0u8; 16]); 4]; 6];
    let mut mode = 0;
    while mode < 6 {
        let mut pair = 0;
        while pair < 4 {
            all[mode][pair] = picks(8, mode as u8 + 3, 16 * pair);
            pair += 1;
        }
        mode += 1;
    }
    all
};
