//! Lane vectors for the transforms: `L<T, N>` holds `N` independent values
//! of `T` (`i32` for 8-bit pictures, `i64` for deeper ones, as libvpx's
//! `tran_high_t`), and every operation applies to each lane. Written as
//! plain loops over arrays, which the compiler turns into vector
//! instructions.

/// A transform constant (`cospi_N_64`, `sinpi_N_9`).
#[derive(Clone, Copy)]
pub struct K(pub i64);

pub trait Elem: Copy {
    const ZERO: Self;
    fn wa(self, o: Self) -> Self;
    fn ws(self, o: Self) -> Self;
    fn wm(self, o: Self) -> Self;
    fn sr(self, n: u32) -> Self;
    fn k(c: i64) -> Self;
    fn from_i32(v: i32) -> Self;
    fn to_i32(self) -> i32;
}

impl Elem for i32 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn wa(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline(always)]
    fn ws(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline(always)]
    fn wm(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline(always)]
    fn sr(self, n: u32) -> Self {
        self >> n
    }
    #[inline(always)]
    fn k(c: i64) -> Self {
        c as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
}

impl Elem for i64 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn wa(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline(always)]
    fn ws(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline(always)]
    fn wm(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline(always)]
    fn sr(self, n: u32) -> Self {
        self >> n
    }
    #[inline(always)]
    fn k(c: i64) -> Self {
        c
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        i64::from(v)
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
}

pub trait Lanes: Copy {
    const ZERO: Self;
    fn wrapping_add(self, o: Self) -> Self;
    fn wrapping_sub(self, o: Self) -> Self;
    fn wrapping_mul(self, k: K) -> Self;
    fn wrapping_neg(self) -> Self;
    fn round_shift(self) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L<T, const N: usize>(pub [T; N]);

impl<T: Elem, const N: usize> Lanes for L<T, N> {
    const ZERO: Self = L([T::ZERO; N]);
    #[inline(always)]
    fn wrapping_add(self, o: Self) -> Self {
        L(core::array::from_fn(|i| self.0[i].wa(o.0[i])))
    }
    #[inline(always)]
    fn wrapping_sub(self, o: Self) -> Self {
        L(core::array::from_fn(|i| self.0[i].ws(o.0[i])))
    }
    #[inline(always)]
    fn wrapping_mul(self, k: K) -> Self {
        let k = T::k(k.0);
        L(core::array::from_fn(|i| self.0[i].wm(k)))
    }
    #[inline(always)]
    fn wrapping_neg(self) -> Self {
        L(core::array::from_fn(|i| T::ZERO.ws(self.0[i])))
    }
    #[inline(always)]
    fn round_shift(self) -> Self {
        let r = T::k(8192);
        L(core::array::from_fn(|i| self.0[i].wa(r).sr(14)))
    }
}

impl K {
    #[inline(always)]
    pub fn wrapping_mul<V: Lanes>(self, v: V) -> V {
        v.wrapping_mul(self)
    }
}

/// `ROUND_POWER_OF_TWO(v, 14)`, libvpx `dct_const_round_shift`.
#[inline(always)]
pub fn rs<V: Lanes>(v: V) -> V {
    v.round_shift()
}

impl K {
    #[inline(always)]
    pub const fn wrapping_neg(self) -> K {
        K(-self.0)
    }
}
