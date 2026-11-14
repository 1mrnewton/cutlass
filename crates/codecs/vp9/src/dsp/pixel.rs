//! Samples of 8 bits (`u8`) and of 10 or 12 bits (`u16`).

pub trait Pixel: Copy + Default + PartialEq + Send + Sync + core::fmt::Debug + 'static {
    /// Whether samples are wider than 8 bits.
    const HIGH: bool;
    fn get(self) -> i32;
    /// `v` held to `0..=max`.
    fn clip(v: i32, max: i32) -> Self;
    /// `v`, which the caller has already held to the sample range.
    fn new(v: i32) -> Self;

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
        _f: (Option<[i16; 8]>, Option<[i16; 8]>),
        _avg: bool,
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
    ) -> bool {
        false
    }
}

impl Pixel for u8 {
    const HIGH: bool = false;
    #[inline(always)]
    fn lf_fast(
        buf: &mut [Self],
        at: usize,
        stride: usize,
        vertical: bool,
        wd: [Option<super::lf::Wd>; 2],
        thr: [super::lf::Thresh; 2],
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
    fn mc_fast(
        dst: &mut [Self],
        ds: usize,
        src: &[Self],
        at: usize,
        ss: usize,
        (w, h): (usize, usize),
        (fx, fy): (Option<[i16; 8]>, Option<[i16; 8]>),
        avg: bool,
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
