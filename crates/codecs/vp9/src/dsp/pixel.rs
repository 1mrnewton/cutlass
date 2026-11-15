//! Samples of 8 bits (`u8`) and of 10 or 12 bits (`u16`).

pub trait Pixel: Copy + Default + PartialEq + Send + Sync + core::fmt::Debug + 'static {
    /// Whether samples are wider than 8 bits.
    const HIGH: bool;
    fn get(self) -> i32;
    /// `v` held to `0..=max`.
    fn clip(v: i32, max: i32) -> Self;
    /// `v`, which the caller has already held to the sample range.
    fn new(v: i32) -> Self;

    /// Plane `p` of a finished picture as bytes the caller's picture can
    /// share without a copy (16-bit samples as little-endian bytes, on a
    /// little-endian target), at its stride of samples times
    /// `size_of::<Self>()` bytes.
    #[inline(always)]
    fn share(
        _buf: &std::sync::Arc<cutlass_simd::vp9::frame::SharedFrame<Self>>,
        _p: usize,
    ) -> Option<std::sync::Arc<dyn cutlass_decode::SharedBytes>> {
        None
    }

    /// The samples as bytes, for 8-bit samples.
    #[inline(always)]
    fn bytes(_s: &[Self]) -> Option<&[u8]> {
        None
    }

    /// Inter prediction with vector instructions; `false` when not taken.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn mc_fast(
        _dst: &mut [Self],
        _ds: usize,
        _src: &[Self],
        _at: usize,
        _ss: usize,
        _wh: (usize, usize),
        _f: (
            Option<&cutlass_simd::vp9::mc::Filter>,
            Option<&cutlass_simd::vp9::mc::Filter>,
        ),
        _avg: bool,
        _max: i32,
        _tmp: &mut [u8],
    ) -> bool {
        false
    }

    /// The loop filter across 16 lines, two halves of 8 with their own
    /// filters and thresholds, with vector instructions; `false` when not
    /// taken.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn lf_fast(
        _buf: &mut [Self],
        _at: usize,
        _stride: usize,
        _vertical: bool,
        _wd: [Option<super::lf::Wd>; 2],
        _thr: [super::lf::Thresh; 2],
        _bd: u32,
    ) -> bool {
        false
    }

    /// Two horizontal edges side by side (32 columns from `at`), four
    /// groups of eight columns each with its own filter, with vector
    /// instructions; `false` when not taken.
    #[inline(always)]
    fn lf_fast32(
        _buf: &mut [Self],
        _at: usize,
        _stride: usize,
        _wd: [Option<super::lf::Wd>; 4],
        _thr: [super::lf::Thresh; 4],
    ) -> bool {
        false
    }
}

/// A finished 8-bit plane, shared with the caller's picture. The decoder
/// writes a buffer again only when no one else holds it, so its bytes do
/// not change while this lives.
struct SharedPlane {
    frame: std::sync::Arc<cutlass_simd::vp9::frame::SharedFrame<u8>>,
    p: usize,
}

impl cutlass_decode::SharedBytes for SharedPlane {
    fn bytes(&self) -> &[u8] {
        self.frame.plane(self.p).0
    }
}

/// A finished 16-bit plane, shared as little-endian bytes, as
/// [`SharedPlane`].
struct SharedPlane16 {
    frame: std::sync::Arc<cutlass_simd::vp9::frame::SharedFrame<u16>>,
    p: usize,
}

impl cutlass_decode::SharedBytes for SharedPlane16 {
    fn bytes(&self) -> &[u8] {
        self.frame.plane_le_bytes(self.p).map_or(&[], |(b, _)| b)
    }
}

impl Pixel for u8 {
    const HIGH: bool = false;
    #[inline(always)]
    fn share(
        buf: &std::sync::Arc<cutlass_simd::vp9::frame::SharedFrame<Self>>,
        p: usize,
    ) -> Option<std::sync::Arc<dyn cutlass_decode::SharedBytes>> {
        Some(std::sync::Arc::new(SharedPlane {
            frame: std::sync::Arc::clone(buf),
            p,
        }))
    }
    #[inline(always)]
    fn bytes(s: &[Self]) -> Option<&[u8]> {
        Some(s)
    }
    #[inline(always)]
    fn lf_fast(
        buf: &mut [Self],
        at: usize,
        stride: usize,
        vertical: bool,
        wd: [Option<super::lf::Wd>; 2],
        thr: [super::lf::Thresh; 2],
        _bd: u32,
    ) -> bool {
        use cutlass_simd::vp9::lf;
        let t = |t: super::lf::Thresh| lf::Thresh {
            blimit: t.blimit,
            limit: t.limit,
            hev: t.hev,
        };
        let w = |w: Option<super::lf::Wd>| {
            w.map(|w| match w {
                super::lf::Wd::W4 => lf::Wd::W4,
                super::lf::Wd::W8 => lf::Wd::W8,
                super::lf::Wd::W16 => lf::Wd::W16,
            })
        };
        lf::filter(lf::Edge {
            buf,
            at,
            stride,
            vertical,
            wd: [w(wd[0]), w(wd[1])],
            thr: [t(thr[0]), t(thr[1])],
        })
    }
    #[inline(always)]
    fn lf_fast32(
        buf: &mut [Self],
        at: usize,
        stride: usize,
        wd: [Option<super::lf::Wd>; 4],
        thr: [super::lf::Thresh; 4],
    ) -> bool {
        use cutlass_simd::vp9::lf;
        let t = |t: super::lf::Thresh| lf::Thresh {
            blimit: t.blimit,
            limit: t.limit,
            hev: t.hev,
        };
        let w = |w: Option<super::lf::Wd>| {
            w.map(|w| match w {
                super::lf::Wd::W4 => lf::Wd::W4,
                super::lf::Wd::W8 => lf::Wd::W8,
                super::lf::Wd::W16 => lf::Wd::W16,
            })
        };
        lf::filter32(lf::Edge32 {
            buf,
            at,
            stride,
            wd: [w(wd[0]), w(wd[1]), w(wd[2]), w(wd[3])],
            thr: [t(thr[0]), t(thr[1]), t(thr[2]), t(thr[3])],
        })
    }
    #[inline(always)]
    fn mc_fast(
        dst: &mut [Self],
        ds: usize,
        src: &[Self],
        at: usize,
        ss: usize,
        (w, h): (usize, usize),
        (fx, fy): (
            Option<&cutlass_simd::vp9::mc::Filter>,
            Option<&cutlass_simd::vp9::mc::Filter>,
        ),
        avg: bool,
        _max: i32,
        tmp: &mut [u8],
    ) -> bool {
        cutlass_simd::vp9::mc::predict(cutlass_simd::vp9::mc::Block {
            dst,
            dst_stride: ds,
            src,
            at,
            src_stride: ss,
            w,
            h,
            fx,
            fy,
            avg,
            tmp,
        })
    }
    #[inline(always)]
    fn get(self) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn clip(v: i32, _max: i32) -> Self {
        v.clamp(0, 255) as u8
    }
    #[inline(always)]
    fn new(v: i32) -> Self {
        v as u8
    }
}

impl Pixel for u16 {
    const HIGH: bool = true;
    #[inline(always)]
    fn share(
        buf: &std::sync::Arc<cutlass_simd::vp9::frame::SharedFrame<Self>>,
        p: usize,
    ) -> Option<std::sync::Arc<dyn cutlass_decode::SharedBytes>> {
        buf.plane_le_bytes(p)?;
        Some(std::sync::Arc::new(SharedPlane16 {
            frame: std::sync::Arc::clone(buf),
            p,
        }))
    }
    #[inline(always)]
    fn lf_fast(
        buf: &mut [Self],
        at: usize,
        stride: usize,
        vertical: bool,
        wd: [Option<super::lf::Wd>; 2],
        thr: [super::lf::Thresh; 2],
        bd: u32,
    ) -> bool {
        use cutlass_simd::vp9::lf16 as lf;
        let t = |t: super::lf::Thresh| lf::Thresh {
            blimit: t.blimit,
            limit: t.limit,
            hev: t.hev,
        };
        let w = |w: Option<super::lf::Wd>| {
            w.map(|w| match w {
                super::lf::Wd::W4 => lf::Wd::W4,
                super::lf::Wd::W8 => lf::Wd::W8,
                super::lf::Wd::W16 => lf::Wd::W16,
            })
        };
        lf::filter(lf::Edge {
            buf,
            at,
            stride,
            vertical,
            wd: [w(wd[0]), w(wd[1])],
            thr: [t(thr[0]), t(thr[1])],
            bd,
        })
    }
    #[inline(always)]
    fn mc_fast(
        dst: &mut [Self],
        ds: usize,
        src: &[Self],
        at: usize,
        ss: usize,
        (w, h): (usize, usize),
        (fx, fy): (
            Option<&cutlass_simd::vp9::mc::Filter>,
            Option<&cutlass_simd::vp9::mc::Filter>,
        ),
        avg: bool,
        max: i32,
        _tmp: &mut [u8],
    ) -> bool {
        let Ok(max) = u16::try_from(max) else {
            return false;
        };
        cutlass_simd::vp9::mc16::predict(cutlass_simd::vp9::mc16::Block {
            dst,
            dst_stride: ds,
            src,
            at,
            src_stride: ss,
            w,
            h,
            fx,
            fy,
            avg,
            max,
        })
    }
    #[inline(always)]
    fn get(self) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn clip(v: i32, max: i32) -> Self {
        v.clamp(0, max) as u16
    }
    #[inline(always)]
    fn new(v: i32) -> Self {
        v as u16
    }
}
