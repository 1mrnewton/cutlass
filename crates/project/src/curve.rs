//! A speed curve that replaces one rational.
//!
//! Points run from the start of the source through the end. Between two points
//! the speed changes in a straight line. The timeline length of a piece is that
//! line integrated with the same rounding as a constant speed. `docs/project.md`
//! is the model.

use std::cmp::Ordering;
use std::num::NonZeroU32;

use cutlass_core::Timestamp;

use crate::Error;
use crate::time::{self, SEQUENCE_TIMESCALE, Speed};

/// Fixed-point scale for the natural log. `ln` is this many fractional bits.
const FRAC: u32 = 60;

/// `ln(2) * 2^60`, rounded to the nearest integer.
const LN2: i128 = 799_144_290_325_165_979;

/// One position on a speed curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeedPoint {
    at: Fraction,
    speed: Speed,
}

impl SpeedPoint {
    /// `at_numerator / at_denominator` is the position, from 0 through 1.
    pub fn new(at_numerator: u32, at_denominator: u32, speed: Speed) -> Result<Self, Error> {
        Ok(Self {
            at: Fraction::new(at_numerator, at_denominator)?,
            speed,
        })
    }

    /// Position numerator and denominator, reduced.
    pub const fn at(self) -> (u32, u32) {
        (self.at.numerator, self.at.denominator.get())
    }

    pub const fn speed(self) -> Speed {
        self.speed
    }
}

/// Speeds along one source, in place of a single rational.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeedCurve {
    points: Vec<SpeedPoint>,
}

impl SpeedCurve {
    /// Two or more points. The first is at 0, the last is at 1, and the
    /// positions strictly increase.
    pub fn new(points: Vec<SpeedPoint>) -> Result<Self, Error> {
        let (Some(first), Some(last)) = (points.first(), points.last()) else {
            return Err(Error::Curve);
        };
        if points.len() < 2 || !first.at.is_zero() || !last.at.is_one() {
            return Err(Error::Curve);
        }
        let mut previous = first.at;
        for point in points.iter().skip(1) {
            if cmp_fraction(previous, point.at) != Ordering::Less {
                return Err(Error::Curve);
            }
            previous = point.at;
        }
        Ok(Self { points })
    }

    pub fn points(&self) -> &[SpeedPoint] {
        &self.points
    }

    /// Timeline length of `source` at this curve.
    pub(crate) fn timeline_span(&self, source: Timestamp) -> Result<Timestamp, Error> {
        if source.value < 0 {
            return Err(Error::NegativeDuration);
        }
        if source.value == 0 {
            return Err(Error::ZeroDuration);
        }
        let value = if self.is_flat() {
            time::timeline_duration(source, self.points[0].speed())?.value
        } else {
            let pieces = self.pieces(source.value)?;
            prefix(&pieces, source.timescale.get(), source.value)?
        };
        if value <= 0 {
            return Err(Error::RoundsToZero);
        }
        Ok(Timestamp::new(value, SEQUENCE_TIMESCALE))
    }

    /// Source ticks covered by `timeline_us` on a source of this length.
    ///
    /// The result is the latest tick whose curve time is still at or before
    /// `timeline_us`. A tie takes the later tick.
    pub(crate) fn source_ticks(&self, timeline_us: i64, source: Timestamp) -> Result<i64, Error> {
        if timeline_us < 0 {
            return Err(Error::NegativeTime);
        }
        let total = source.value;
        if timeline_us == 0 {
            return Ok(0);
        }
        if self.is_flat() {
            return time::source_ticks(timeline_us, source.timescale, self.points[0].speed());
        }
        let pieces = self.pieces(total)?;
        let end = prefix(&pieces, source.timescale.get(), total)?;
        if end <= 0 {
            return Err(Error::RoundsToZero);
        }
        if timeline_us >= end {
            return Ok(total);
        }
        let mut lo = 0i64;
        let mut hi = total;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if prefix(&pieces, source.timescale.get(), mid)? <= timeline_us {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let floor = lo - 1;
        let ceil = floor + 1;
        if ceil > total {
            return Ok(floor.max(0));
        }
        let at_floor = prefix(&pieces, source.timescale.get(), floor)?;
        let at_ceil = prefix(&pieces, source.timescale.get(), ceil)?;
        let before = timeline_us.saturating_sub(at_floor);
        let after = at_ceil.saturating_sub(timeline_us);
        if after < before {
            Ok(ceil)
        } else if before < after {
            Ok(floor)
        } else {
            Ok(ceil)
        }
    }

    /// Source length, in ticks, whose curve time is closest to `want_us`.
    ///
    /// The search stays within `room`. A tie takes the longer source.
    pub(crate) fn ticks_for_span(
        &self,
        want_us: i64,
        scale: NonZeroU32,
        room: i64,
    ) -> Result<i64, Error> {
        if want_us < 0 {
            return Err(Error::NegativeTime);
        }
        if want_us == 0 || room <= 0 {
            return Ok(0);
        }
        if self.is_flat() {
            return time::source_ticks(want_us, scale, self.points[0].speed());
        }
        let mut lo = 1i64;
        let mut hi = room;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let span = self.timeline_span(Timestamp::new(mid, scale))?.value;
            if span < want_us {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let at = self.timeline_span(Timestamp::new(lo, scale))?.value;
        if lo == 1 {
            return Ok(1);
        }
        let prev = lo - 1;
        let before = if prev == 0 {
            0
        } else {
            self.timeline_span(Timestamp::new(prev, scale))?.value
        };
        let after_gap = at.saturating_sub(want_us);
        let before_gap = want_us.saturating_sub(before);
        if after_gap < before_gap {
            Ok(lo)
        } else if before_gap < after_gap {
            Ok(prev)
        } else {
            Ok(lo)
        }
    }

    /// Cuts the curve at `cut_ticks` of `total_ticks` and scales each piece
    /// back to run from 0 to 1.
    pub(crate) fn split(
        &self,
        cut_ticks: i64,
        total_ticks: i64,
    ) -> Result<(SpeedCurve, SpeedCurve), Error> {
        if cut_ticks <= 0 || total_ticks <= cut_ticks {
            return Err(Error::SplitOutside);
        }
        let cut = fit_fraction(u64_as_u128(cut_ticks)?, u64_as_u128(total_ticks)?)?;
        let speed = self.speed_at(cut)?;
        let mut left = Vec::new();
        let mut right = Vec::new();
        for point in &self.points {
            match cmp_fraction(point.at, cut) {
                Ordering::Less => {
                    left.push(SpeedPoint {
                        at: div_fraction(point.at, cut)?,
                        speed: point.speed,
                    });
                }
                Ordering::Equal => {}
                Ordering::Greater => {
                    let shifted = sub_fraction(point.at, cut)?;
                    let rest = sub_fraction(Fraction::ONE, cut)?;
                    right.push(SpeedPoint {
                        at: div_fraction(shifted, rest)?,
                        speed: point.speed,
                    });
                }
            }
        }
        left.push(SpeedPoint {
            at: Fraction::ONE,
            speed,
        });
        right.insert(
            0,
            SpeedPoint {
                at: Fraction::ZERO,
                speed,
            },
        );
        Ok((SpeedCurve::new(left)?, SpeedCurve::new(right)?))
    }

    fn is_flat(&self) -> bool {
        let first = self.points[0].speed;
        self.points
            .iter()
            .all(|point| same_speed(first, point.speed))
    }

    fn pieces(&self, total: i64) -> Result<Vec<Piece>, Error> {
        let mut bounds = Vec::with_capacity(self.points.len());
        let mut previous = 0i64;
        for (index, point) in self.points.iter().enumerate() {
            let mut tick = if index == 0 {
                0
            } else if index + 1 == self.points.len() {
                total
            } else {
                fraction_ticks(total, point.at)?
            };
            if tick < previous {
                tick = previous;
            }
            if tick > total {
                tick = total;
            }
            bounds.push(tick);
            previous = tick;
        }
        let mut pieces = Vec::with_capacity(self.points.len().saturating_sub(1));
        for (index, point) in self.points.iter().enumerate().skip(1) {
            let start = bounds[index - 1];
            let end = bounds[index];
            let left = self.points[index - 1].speed;
            pieces.push(Piece {
                start,
                end,
                n0: left.numerator(),
                d0: left.denominator().get(),
                n1: point.speed.numerator(),
                d1: point.speed.denominator().get(),
            });
        }
        Ok(pieces)
    }

    fn speed_at(&self, at: Fraction) -> Result<Speed, Error> {
        let mut previous = &self.points[0];
        if cmp_fraction(previous.at, at) == Ordering::Equal {
            return Ok(previous.speed);
        }
        for point in &self.points[1..] {
            match cmp_fraction(point.at, at) {
                Ordering::Equal => return Ok(point.speed),
                Ordering::Greater => return interpolate(previous, point, at),
                Ordering::Less => previous = point,
            }
        }
        Err(Error::Curve)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fraction {
    numerator: u32,
    denominator: NonZeroU32,
}

impl Fraction {
    const ZERO: Self = Self {
        numerator: 0,
        denominator: NonZeroU32::MIN,
    };
    const ONE: Self = Self {
        numerator: 1,
        denominator: NonZeroU32::MIN,
    };

    fn new(numerator: u32, denominator: u32) -> Result<Self, Error> {
        if denominator == 0 || numerator > denominator {
            return Err(Error::Curve);
        }
        fit_fraction(u128::from(numerator), u128::from(denominator))
    }

    const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    const fn is_one(self) -> bool {
        self.numerator == self.denominator.get()
    }
}

struct Piece {
    start: i64,
    end: i64,
    n0: u32,
    d0: u32,
    n1: u32,
    d1: u32,
}

fn prefix(pieces: &[Piece], timescale: u32, ticks: i64) -> Result<i64, Error> {
    if ticks <= 0 {
        return Ok(0);
    }
    let mut time = 0i64;
    for piece in pieces {
        if ticks <= piece.start {
            break;
        }
        let len = piece.end.saturating_sub(piece.start);
        if len == 0 {
            continue;
        }
        let into = if ticks >= piece.end {
            len
        } else {
            ticks.saturating_sub(piece.start)
        };
        let part = integrate(len, into, timescale, piece)?;
        time = time.checked_add(part).ok_or(Error::DurationOverflow)?;
        if ticks < piece.end {
            break;
        }
    }
    Ok(time)
}

/// Timeline microseconds to play `into` ticks of a piece `len` ticks long.
fn integrate(len: i64, into: i64, timescale: u32, piece: &Piece) -> Result<i64, Error> {
    if into <= 0 || len <= 0 {
        return Ok(0);
    }
    if same_ratio(piece.n0, piece.d0, piece.n1, piece.d1) {
        let speed = Speed::new(piece.n0, piece.d0)?;
        return Ok(
            time::timeline_duration(Timestamp::new(into, nonzero(timescale)?), speed)?.value,
        );
    }
    let len_u = u64_as_u128(len)?;
    let into_u = u64_as_u128(into)?;
    let stay = len_u.checked_sub(into_u).ok_or(Error::DurationOverflow)?;
    let n0 = u128::from(piece.n0);
    let d0 = u128::from(piece.d0);
    let n1 = u128::from(piece.n1);
    let d1 = u128::from(piece.d1);
    let left = n0
        .checked_mul(d1)
        .and_then(|value| value.checked_mul(stay))
        .ok_or(Error::DurationOverflow)?;
    let right = n1
        .checked_mul(d0)
        .and_then(|value| value.checked_mul(into_u))
        .ok_or(Error::DurationOverflow)?;
    let ratio_n = left.checked_add(right).ok_or(Error::DurationOverflow)?;
    let ratio_d = n0
        .checked_mul(d1)
        .and_then(|value| value.checked_mul(len_u))
        .ok_or(Error::DurationOverflow)?;
    let ln = ln_ratio(ratio_n, ratio_d)?;
    if ln == 0 {
        let speed = Speed::new(piece.n0, piece.d0)?;
        return Ok(
            time::timeline_duration(Timestamp::new(into, nonzero(timescale)?), speed)?.value,
        );
    }
    let delta = n1
        .checked_mul(d0)
        .and_then(|value| {
            let other = n0.checked_mul(d1)?;
            value
                .checked_sub(other)
                .or_else(|| other.checked_sub(value))
        })
        .ok_or(Error::DurationOverflow)?;
    if delta == 0 {
        return Err(Error::ZeroSpeed);
    }
    let ln_abs = ln.unsigned_abs();
    let mut numer = mul_wide(u64_as_u128(len)?, 1_000_000);
    numer = mul_wide_by(numer, ln_abs)?;
    numer = mul_wide_by(numer, d0)?;
    numer = mul_wide_by(numer, d1)?;
    let divisor = u128::from(timescale)
        .checked_mul(delta)
        .ok_or(Error::DurationOverflow)?;
    round_wide(numer, divisor)
}

fn same_speed(left: Speed, right: Speed) -> bool {
    same_ratio(
        left.numerator(),
        left.denominator().get(),
        right.numerator(),
        right.denominator().get(),
    )
}

fn same_ratio(n0: u32, d0: u32, n1: u32, d1: u32) -> bool {
    u128::from(n0) * u128::from(d1) == u128::from(n1) * u128::from(d0)
}

fn fraction_ticks(total: i64, at: Fraction) -> Result<i64, Error> {
    if at.is_zero() {
        return Ok(0);
    }
    if at.is_one() {
        return Ok(total);
    }
    time::round_half_away(
        i128::from(total) * i128::from(at.numerator),
        i128::from(at.denominator.get()),
    )
}

fn interpolate(left: &SpeedPoint, right: &SpeedPoint, at: Fraction) -> Result<Speed, Error> {
    let w0 = sub_fraction(right.at, at)?;
    let w1 = sub_fraction(at, left.at)?;
    let w = sub_fraction(right.at, left.at)?;
    let (w0n, w0d) = ratio(w0);
    let (w1n, w1d) = ratio(w1);
    let (wn, wd) = ratio(w);
    let (s0n, s0d) = speed_ratio(left.speed);
    let (s1n, s1d) = speed_ratio(right.speed);
    let (a_n, a_d) = reduced(
        s0n.checked_mul(w0n).ok_or(Error::DurationOverflow)?,
        s0d.checked_mul(w0d).ok_or(Error::DurationOverflow)?,
    )?;
    let (b_n, b_d) = reduced(
        s1n.checked_mul(w1n).ok_or(Error::DurationOverflow)?,
        s1d.checked_mul(w1d).ok_or(Error::DurationOverflow)?,
    )?;
    let sum_n = a_n
        .checked_mul(b_d)
        .and_then(|value| value.checked_add(b_n.checked_mul(a_d)?))
        .ok_or(Error::DurationOverflow)?;
    let sum_d = a_d.checked_mul(b_d).ok_or(Error::DurationOverflow)?;
    let (n, d) = reduced(
        sum_n.checked_mul(wd).ok_or(Error::DurationOverflow)?,
        sum_d.checked_mul(wn).ok_or(Error::DurationOverflow)?,
    )?;
    fit_speed(n, d)
}

fn ratio(fraction: Fraction) -> (u128, u128) {
    (
        u128::from(fraction.numerator),
        u128::from(fraction.denominator.get()),
    )
}

fn speed_ratio(speed: Speed) -> (u128, u128) {
    (
        u128::from(speed.numerator()),
        u128::from(speed.denominator().get()),
    )
}

fn cmp_fraction(left: Fraction, right: Fraction) -> Ordering {
    let left_cross = u128::from(left.numerator) * u128::from(right.denominator.get());
    let right_cross = u128::from(right.numerator) * u128::from(left.denominator.get());
    left_cross.cmp(&right_cross)
}

fn div_fraction(left: Fraction, right: Fraction) -> Result<Fraction, Error> {
    if right.numerator == 0 {
        return Err(Error::Curve);
    }
    fit_fraction(
        u128::from(left.numerator) * u128::from(right.denominator.get()),
        u128::from(left.denominator.get()) * u128::from(right.numerator),
    )
}

fn sub_fraction(left: Fraction, right: Fraction) -> Result<Fraction, Error> {
    let n = u128::from(left.numerator) * u128::from(right.denominator.get());
    let m = u128::from(right.numerator) * u128::from(left.denominator.get());
    let Some(n) = n.checked_sub(m) else {
        return Err(Error::Curve);
    };
    fit_fraction(
        n,
        u128::from(left.denominator.get()) * u128::from(right.denominator.get()),
    )
}

fn fit_fraction(numerator: u128, denominator: u128) -> Result<Fraction, Error> {
    if denominator == 0 || numerator > denominator {
        return Err(Error::Curve);
    }
    let (numerator, denominator) = reduced(numerator, denominator)?;
    let numerator = u32::try_from(numerator).map_err(|_| Error::DurationOverflow)?;
    let denominator = u32::try_from(denominator).map_err(|_| Error::DurationOverflow)?;
    let Some(denominator) = NonZeroU32::new(denominator) else {
        return Err(Error::Curve);
    };
    Ok(Fraction {
        numerator,
        denominator,
    })
}

fn fit_speed(numerator: u128, denominator: u128) -> Result<Speed, Error> {
    if numerator == 0 || denominator == 0 {
        return Err(Error::ZeroSpeed);
    }
    let (numerator, denominator) = reduced(numerator, denominator)?;
    let numerator = u32::try_from(numerator).map_err(|_| Error::DurationOverflow)?;
    let denominator = u32::try_from(denominator).map_err(|_| Error::DurationOverflow)?;
    Speed::new(numerator, denominator)
}

fn reduced(mut numerator: u128, mut denominator: u128) -> Result<(u128, u128), Error> {
    if denominator == 0 {
        return Err(Error::Curve);
    }
    let divisor = gcd(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;
    Ok((numerator, denominator))
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

fn nonzero(value: u32) -> Result<NonZeroU32, Error> {
    NonZeroU32::new(value).ok_or(Error::ZeroTimescale)
}

fn u64_as_u128(value: i64) -> Result<u128, Error> {
    u128::try_from(value).map_err(|_| Error::DurationOverflow)
}

fn ln_ratio(numer: u128, denom: u128) -> Result<i128, Error> {
    if numer == 0 || denom == 0 {
        return Err(Error::ZeroSpeed);
    }
    if numer == denom {
        return Ok(0);
    }
    let (ln_n, k_n) = ln_integer(numer)?;
    let (ln_d, k_d) = ln_integer(denom)?;
    let k = i128::from(k_n) - i128::from(k_d);
    let scaled = k.checked_mul(LN2).ok_or(Error::DurationOverflow)?;
    ln_n.checked_sub(ln_d)
        .and_then(|value| value.checked_add(scaled))
        .ok_or(Error::DurationOverflow)
}

fn ln_integer(value: u128) -> Result<(i128, u32), Error> {
    let k = value.ilog2();
    let mant = if k >= FRAC {
        i128::try_from(value >> (k - FRAC)).map_err(|_| Error::DurationOverflow)?
    } else {
        i128::try_from(value << (FRAC - k)).map_err(|_| Error::DurationOverflow)?
    };
    Ok((ln_mant(mant)?, k))
}

fn ln_mant(mant: i128) -> Result<i128, Error> {
    let one = 1i128 << FRAC;
    if mant <= one {
        return Ok(0);
    }
    let z = (mant - one)
        .checked_shl(FRAC)
        .ok_or(Error::DurationOverflow)?
        / (mant + one);
    let z2 = z.checked_mul(z).ok_or(Error::DurationOverflow)? >> FRAC;
    let mut term = z;
    let mut sum = z;
    let mut n = 1i128;
    for _ in 0..40 {
        term = term.checked_mul(z2).ok_or(Error::DurationOverflow)? >> FRAC;
        n += 2;
        if term == 0 {
            break;
        }
        sum += term / n;
    }
    sum.checked_shl(1).ok_or(Error::DurationOverflow)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Wide {
    hi: u128,
    lo: u128,
}

fn mul_wide(left: u128, right: u128) -> Wide {
    let left_lo = left as u64 as u128;
    let left_hi = left >> 64;
    let right_lo = right as u64 as u128;
    let right_hi = right >> 64;
    let low = left_lo * right_lo;
    let mid_left = left_lo * right_hi;
    let mid_right = left_hi * right_lo;
    let high = left_hi * right_hi;
    let mid =
        (low >> 64) + (mid_left & 0xffff_ffff_ffff_ffff) + (mid_right & 0xffff_ffff_ffff_ffff);
    Wide {
        lo: (low & 0xffff_ffff_ffff_ffff) | ((mid & 0xffff_ffff_ffff_ffff) << 64),
        hi: high + (mid_left >> 64) + (mid_right >> 64) + (mid >> 64),
    }
}

fn mul_wide_by(value: Wide, factor: u128) -> Result<Wide, Error> {
    let low = mul_wide(value.lo, factor);
    let high = mul_wide(value.hi, factor);
    if high.hi != 0 {
        return Err(Error::DurationOverflow);
    }
    let (hi, overflow) = low.hi.overflowing_add(high.lo);
    if overflow {
        return Err(Error::DurationOverflow);
    }
    Ok(Wide { hi, lo: low.lo })
}

fn shr_frac(value: Wide) -> (Wide, u128) {
    let dropped = value.lo & ((1u128 << FRAC) - 1);
    let lo = (value.lo >> FRAC) | (value.hi << (128 - FRAC));
    let hi = value.hi >> FRAC;
    (Wide { hi, lo }, dropped)
}

/// `numer / (divisor * 2^FRAC)`, rounded half away from zero.
fn round_wide(numer: Wide, divisor: u128) -> Result<i64, Error> {
    if divisor == 0 {
        return Err(Error::ZeroSpeed);
    }
    let (shifted, dropped) = shr_frac(numer);
    let (quotient, remainder) = div_wide(shifted, divisor)?;
    let rem = add_wide(mul_wide(remainder, 1u128 << FRAC), dropped)?;
    let half = mul_wide(divisor, 1u128 << (FRAC - 1));
    let quotient = if wide_cmp(rem, half) != Ordering::Less {
        quotient.checked_add(1).ok_or(Error::DurationOverflow)?
    } else {
        quotient
    };
    i64::try_from(quotient).map_err(|_| Error::DurationOverflow)
}

fn add_wide(value: Wide, extra: u128) -> Result<Wide, Error> {
    let (lo, carry) = value.lo.overflowing_add(extra);
    let hi = value
        .hi
        .checked_add(u128::from(carry))
        .ok_or(Error::DurationOverflow)?;
    Ok(Wide { hi, lo })
}

fn wide_cmp(left: Wide, right: Wide) -> Ordering {
    left.hi.cmp(&right.hi).then(left.lo.cmp(&right.lo))
}

fn div_wide(numer: Wide, denom: u128) -> Result<(u128, u128), Error> {
    if denom == 0 {
        return Err(Error::ZeroSpeed);
    }
    if numer.hi >= denom {
        return Err(Error::DurationOverflow);
    }
    let mut remainder = 0u128;
    let mut high = 0u128;
    let mut quotient = 0u128;
    for index in (0..256).rev() {
        let bit = if index >= 128 {
            (numer.hi >> (index - 128)) & 1
        } else {
            (numer.lo >> index) & 1
        };
        let top = remainder >> 127;
        let shifted = remainder.wrapping_shl(1) | bit;
        let subtract = top == 1 || shifted >= denom;
        if top == 1 && shifted >= denom {
            return Err(Error::DurationOverflow);
        }
        if subtract {
            remainder = if top == 1 {
                shifted.wrapping_sub(denom)
            } else {
                shifted - denom
            };
            if index >= 128 {
                high |= 1u128 << (index - 128);
            } else {
                quotient |= 1u128 << index;
            }
        } else {
            remainder = shifted;
        }
    }
    if high != 0 {
        return Err(Error::DurationOverflow);
    }
    Ok((quotient, remainder))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scale(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("scale")
    }

    fn second() -> Timestamp {
        Timestamp::new(1_000_000, scale(1_000_000))
    }

    fn ramp(start: u32, end: u32) -> SpeedCurve {
        SpeedCurve::new(vec![
            SpeedPoint::new(0, 1, Speed::new(start, 1).expect("start")).expect("start"),
            SpeedPoint::new(1, 1, Speed::new(end, 1).expect("end")).expect("end"),
        ])
        .expect("curve")
    }

    #[test]
    fn a_one_to_two_ramp_lasts_ln2_microseconds() {
        let curve = ramp(1, 2);
        assert_eq!(curve.timeline_span(second()).expect("span").value, 693_147);
        let down = ramp(2, 1);
        assert_eq!(down.timeline_span(second()).expect("span").value, 693_147);
        assert_eq!(curve.source_ticks(0, second()).expect("start"), 0);
        assert_eq!(
            curve.source_ticks(693_147, second()).expect("end"),
            1_000_000
        );
        assert_eq!(curve.source_ticks(346_573, second()).expect("mid"), 414_213);
        assert_eq!(
            curve
                .ticks_for_span(693_147, scale(1_000_000), 2_000_000)
                .expect("ticks"),
            1_000_000
        );
        assert_eq!(
            curve
                .ticks_for_span(346_574, scale(1_000_000), 2_000_000)
                .expect("half"),
            500_000
        );
    }

    #[test]
    fn a_flat_curve_matches_the_rational() {
        let curve = SpeedCurve::new(vec![
            SpeedPoint::new(0, 1, Speed::new(2, 1).expect("speed")).expect("start"),
            SpeedPoint::new(1, 2, Speed::new(4, 2).expect("speed")).expect("middle"),
            SpeedPoint::new(1, 1, Speed::new(2, 1).expect("speed")).expect("end"),
        ])
        .expect("curve");
        let source = Timestamp::new(1_000, scale(30_000));
        let rational =
            time::timeline_duration(source, Speed::new(2, 1).expect("speed")).expect("span");
        assert_eq!(curve.timeline_span(source).expect("span"), rational);
    }

    #[test]
    fn half_the_source_of_a_ramp_is_the_slower_side() {
        let curve = ramp(1, 2);
        let half = Timestamp::new(500_000, scale(1_000_000));
        assert_eq!(curve.timeline_span(half).expect("half").value, 346_574);
        let pieces = curve.pieces(1_000_000).expect("pieces");
        assert_eq!(
            prefix(&pieces, 1_000_000, 500_000).expect("prefix"),
            405_465
        );
    }

    #[test]
    fn a_split_at_the_source_middle_cuts_the_line() {
        let curve = ramp(1, 2);
        let (left, right) = curve.split(500_000, 1_000_000).expect("split");
        assert_eq!(left.points()[0].at(), (0, 1));
        assert_eq!(left.points()[0].speed(), Speed::new(1, 1).expect("speed"));
        assert_eq!(left.points()[1].at(), (1, 1));
        assert_eq!(left.points()[1].speed(), Speed::new(3, 2).expect("speed"));
        assert_eq!(right.points()[0].speed(), Speed::new(3, 2).expect("speed"));
        assert_eq!(right.points()[1].speed(), Speed::new(2, 1).expect("speed"));
        let half = Timestamp::new(500_000, scale(1_000_000));
        assert_eq!(left.timeline_span(half).expect("left").value, 405_465);
        assert_eq!(right.timeline_span(half).expect("right").value, 287_682);
    }

    #[test]
    fn a_wide_quotient_matches_the_power_of_two() {
        let (quotient, remainder) = div_wide(Wide { hi: 1, lo: 0 }, 3).expect("divide");
        assert_eq!(quotient, u128::MAX / 3);
        assert_eq!(remainder, 1);
        let product = mul_wide(u128::MAX, u128::MAX);
        assert_eq!(product.lo, 1);
        assert_eq!(product.hi, u128::MAX - 1);
    }

    #[test]
    fn a_curve_needs_both_ends() {
        let speed = Speed::new(1, 1).expect("speed");
        let point = SpeedPoint::new(0, 1, speed).expect("point");
        assert_eq!(SpeedCurve::new(vec![point]).expect_err("one"), Error::Curve);
        let late = SpeedPoint::new(2, 1, speed).expect_err("past");
        assert_eq!(late, Error::Curve);
        let backwards = SpeedCurve::new(vec![
            SpeedPoint::new(1, 1, speed).expect("end"),
            SpeedPoint::new(0, 1, speed).expect("start"),
        ]);
        assert_eq!(backwards.expect_err("order"), Error::Curve);
    }
}
