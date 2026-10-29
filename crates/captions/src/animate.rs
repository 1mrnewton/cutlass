//! Color, motion, and glyph batches for a selected range.
//!
//! Channels sample in word-clock milliseconds. Shake and pulse are functions
//! of that clock, and stagger shifts a sample by glyph index. Seeking
//! evaluates the same functions at another millisecond. A batch carries the
//! placed glyphs from the letter map. It does not build an atlas or a mesh.

// The crate root does not export this module yet, so a non-test build has no
// caller. Tests call every public function.
#![cfg_attr(not(test), allow(dead_code))]

use cutlass_text::PlacedGlyph;

use crate::CaptionError;
use crate::page::{LetterMap, WordBytes};
use crate::select::{RangeKind, Selection, TextRange};

/// Four color channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
    pub alpha: f64,
}

/// Fill, background, and the motion sampled at one millisecond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledStyle {
    pub color: Rgba,
    pub background: Rgba,
    pub position_x: f64,
    pub position_y: f64,
    pub rotation: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

/// One curve sample. `millis` is the word clock.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub millis: i64,
    pub value: f64,
}

/// A piecewise-linear curve.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub keys: Vec<Key>,
}

/// Curves for one selector.
///
/// `color` and `background` are red, green, blue, then alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleCurves {
    pub color: [Channel; 4],
    pub background: [Channel; 4],
    pub position_x: Channel,
    pub position_y: Channel,
    pub rotation: Channel,
    pub scale_x: Channel,
    pub scale_y: Channel,
}

/// Curves for unread, reading, read, and keyword text.
#[derive(Clone, Debug, PartialEq)]
pub struct RangeStyles {
    pub unread: StyleCurves,
    pub reading: StyleCurves,
    pub read: StyleCurves,
    pub keyword: StyleCurves,
}

/// Axis that receives a shake offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShakeAxis {
    PositionX,
    PositionY,
    Rotation,
}

/// A glyph the text system already placed, plus the style at this sample.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedSample {
    pub glyph: PlacedGlyph,
    pub style: SampledStyle,
}

/// Glyphs that share one selected range.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphBatch {
    pub kind: RangeKind,
    pub glyphs: Vec<PlacedSample>,
}

const OPAQUE_WHITE: Rgba = Rgba {
    red: 1.0,
    green: 1.0,
    blue: 1.0,
    alpha: 1.0,
};

const CLEAR: Rgba = Rgba {
    red: 0.0,
    green: 0.0,
    blue: 0.0,
    alpha: 0.0,
};

impl SampledStyle {
    /// White fill, transparent background, no move, no rotation, scale 1.
    pub fn identity() -> Self {
        Self {
            color: OPAQUE_WHITE,
            background: CLEAR,
            position_x: 0.0,
            position_y: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        }
    }
}

impl Channel {
    /// Sample the curve at `millis`.
    ///
    /// An empty channel returns `identity`. Keys are ordered by time without
    /// changing this channel. A later key wins when two keys share a time, and
    /// that value is the one held outside the curve and used as an endpoint.
    /// Between distinct times the sample is linear. A non-finite number is
    /// [`CaptionError::Measure`]. A timestamp difference that does not fit in
    /// `i64` is [`CaptionError::Overflow`].
    pub fn sample(&self, millis: i64, identity: f64) -> Result<f64, CaptionError> {
        if !identity.is_finite() {
            return Err(CaptionError::Measure);
        }
        if self.keys.is_empty() {
            return Ok(identity);
        }
        sample_curve(&collapse(self)?, millis)
    }
}

impl StyleCurves {
    /// Empty channels. Sampling them returns [`SampledStyle::identity`].
    pub fn identity() -> Self {
        Self {
            color: empty_channels(),
            background: empty_channels(),
            position_x: empty_channel(),
            position_y: empty_channel(),
            rotation: empty_channel(),
            scale_x: empty_channel(),
            scale_y: empty_channel(),
        }
    }

    /// Sample every curve at `millis`.
    ///
    /// Color identity is white. Background identity is clear. Position and
    /// rotation identity are 0. Scale identity is 1.
    pub fn sample(&self, millis: i64) -> Result<SampledStyle, CaptionError> {
        Ok(SampledStyle {
            color: sample_color(&self.color, millis, OPAQUE_WHITE)?,
            background: sample_color(&self.background, millis, CLEAR)?,
            position_x: self.position_x.sample(millis, 0.0)?,
            position_y: self.position_y.sample(millis, 0.0)?,
            rotation: self.rotation.sample(millis, 0.0)?,
            scale_x: self.scale_x.sample(millis, 1.0)?,
            scale_y: self.scale_y.sample(millis, 1.0)?,
        })
    }
}

impl RangeStyles {
    /// Empty curves for unread, reading, read, and keyword text.
    pub fn identity() -> Self {
        Self {
            unread: StyleCurves::identity(),
            reading: StyleCurves::identity(),
            read: StyleCurves::identity(),
            keyword: StyleCurves::identity(),
        }
    }
}

/// Map `millis` through the half-open span `[start, end)` onto `domain`.
///
/// Before the span the result is 0. At or after `end` it is `domain` when the
/// span is positive. A zero-length span is 0 at `millis <= start` and `domain`
/// after that. An inverted span is [`CaptionError::Inverted`]. A non-finite
/// domain is [`CaptionError::Measure`]. The ratio uses `i128`, so every pair
/// of `i64` timestamps has a defined difference.
pub fn local_domain(millis: i64, start: i64, end: i64, domain: f64) -> Result<f64, CaptionError> {
    if end < start {
        return Err(CaptionError::Inverted);
    }
    if !domain.is_finite() {
        return Err(CaptionError::Measure);
    }
    if millis <= start {
        return Ok(0.0);
    }
    if end == start || millis >= end {
        return Ok(domain);
    }
    let elapsed = i128::from(millis)
        .checked_sub(i128::from(start))
        .ok_or(CaptionError::Overflow)?;
    let span = i128::from(end)
        .checked_sub(i128::from(start))
        .ok_or(CaptionError::Overflow)?;
    let progress = unit_ratio(elapsed, span)?;
    finite_value(progress * domain)
}

/// Add an alternating offset on `axis`.
///
/// The phase is `millis` wrapped into `period_ms`. The first half of the
/// period adds `amplitude` and the second half subtracts it. `period_ms < 2`
/// is [`CaptionError::Period`]. A non-finite amplitude is
/// [`CaptionError::Measure`].
pub fn apply_shake(
    mut style: SampledStyle,
    axis: ShakeAxis,
    amplitude: f64,
    period_ms: i64,
    millis: i64,
) -> Result<SampledStyle, CaptionError> {
    let (phase, half) = phase_in_period(period_ms, millis)?;
    if !amplitude.is_finite() {
        return Err(CaptionError::Measure);
    }
    let offset = if phase < half { amplitude } else { -amplitude };
    match axis {
        ShakeAxis::PositionX => style.position_x = finite_value(style.position_x + offset)?,
        ShakeAxis::PositionY => style.position_y = finite_value(style.position_y + offset)?,
        ShakeAxis::Rotation => style.rotation = finite_value(style.rotation + offset)?,
    }
    Ok(style)
}

/// Multiply both scales by `1 + amplitude * triangle(millis)`.
///
/// The triangle is 0 at phase 0, 1 at `period_ms / 2`, and 0 again at the next
/// period. `period_ms < 2` is [`CaptionError::Period`]. A non-finite amplitude
/// is [`CaptionError::Measure`].
pub fn apply_pulse(
    mut style: SampledStyle,
    amplitude: f64,
    period_ms: i64,
    millis: i64,
) -> Result<SampledStyle, CaptionError> {
    let (phase, half) = phase_in_period(period_ms, millis)?;
    if !amplitude.is_finite() {
        return Err(CaptionError::Measure);
    }
    let wave = triangle(period_ms, phase, half)?;
    let factor = finite_value(1.0 + amplitude * wave)?;
    style.scale_x = finite_value(style.scale_x * factor)?;
    style.scale_y = finite_value(style.scale_y * factor)?;
    Ok(style)
}

/// Sample `channel` at `millis - index * shift_ms`.
///
/// Index 0 is the unshifted sample. A negative shift is legal and moves the
/// sample later. An index that does not fit in `i64`, or a product or
/// difference that overflows, is [`CaptionError::Overflow`].
pub fn stagger(
    channel: &Channel,
    index: usize,
    shift_ms: i64,
    millis: i64,
    identity: f64,
) -> Result<f64, CaptionError> {
    let at = millis_at_index(index, shift_ms, millis)?;
    channel.sample(at, identity)
}

/// Place the selected glyphs at `millis`.
///
/// Ranges are walked unread, reading, read, then keyword. One batch is
/// emitted for each range that contains at least one glyph. A word that is
/// not on this page, or a character span that does not fit that word, is
/// skipped. Glyph `i` in a batch is sampled at `millis - i * stagger_shift_ms`.
/// The same glyph can appear in a reading batch and a keyword batch.
pub fn batches(
    letters: &LetterMap,
    selection: &Selection,
    styles: &RangeStyles,
    millis: i64,
    stagger_shift_ms: i64,
) -> Result<Vec<GlyphBatch>, CaptionError> {
    let mut out = Vec::new();
    let groups = [
        (RangeKind::Unread, &selection.unread, &styles.unread),
        (RangeKind::Reading, &selection.reading, &styles.reading),
        (RangeKind::Read, &selection.read, &styles.read),
        (RangeKind::Keyword, &selection.keyword, &styles.keyword),
    ];
    for (kind, ranges, curves) in groups {
        for range in ranges {
            if let Some(batch) = batch_for(letters, range, kind, curves, millis, stagger_shift_ms)?
            {
                out.push(batch);
            }
        }
    }
    Ok(out)
}

fn finite_value(value: f64) -> Result<f64, CaptionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CaptionError::Measure)
    }
}

#[allow(clippy::cast_precision_loss)]
fn unit_ratio(numerator: i128, denominator: i128) -> Result<f64, CaptionError> {
    if denominator == 0 {
        return Err(CaptionError::Overflow);
    }
    finite_value((numerator as f64) / (denominator as f64))
}

fn empty_channel() -> Channel {
    Channel { keys: Vec::new() }
}

fn empty_channels() -> [Channel; 4] {
    std::array::from_fn(|_| empty_channel())
}

struct CurveKey {
    millis: i64,
    value: f64,
}

fn collapse(channel: &Channel) -> Result<Vec<CurveKey>, CaptionError> {
    let mut ranked = Vec::with_capacity(channel.keys.len());
    for (index, key) in channel.keys.iter().enumerate() {
        if !key.value.is_finite() {
            return Err(CaptionError::Measure);
        }
        ranked.push((
            index,
            CurveKey {
                millis: key.millis,
                value: key.value,
            },
        ));
    }
    ranked.sort_by_key(|item| (item.1.millis, item.0));
    let mut curve: Vec<CurveKey> = Vec::with_capacity(ranked.len());
    for (_, key) in ranked {
        if let Some(last) = curve.last_mut()
            && last.millis == key.millis
        {
            last.value = key.value;
            continue;
        }
        curve.push(key);
    }
    Ok(curve)
}

fn sample_curve(curve: &[CurveKey], millis: i64) -> Result<f64, CaptionError> {
    let (Some(first), Some(last)) = (curve.first(), curve.last()) else {
        return Err(CaptionError::Measure);
    };
    if millis <= first.millis {
        return Ok(first.value);
    }
    if millis >= last.millis {
        return Ok(last.value);
    }
    let slot = curve.partition_point(|key| key.millis <= millis);
    let Some(left) = curve.get(slot.saturating_sub(1)) else {
        return Ok(first.value);
    };
    if millis == left.millis {
        return Ok(left.value);
    }
    let Some(right) = curve.get(slot) else {
        return Ok(last.value);
    };
    lerp(left, right, millis)
}

fn lerp(left: &CurveKey, right: &CurveKey, millis: i64) -> Result<f64, CaptionError> {
    let span = right
        .millis
        .checked_sub(left.millis)
        .ok_or(CaptionError::Overflow)?;
    let elapsed = millis
        .checked_sub(left.millis)
        .ok_or(CaptionError::Overflow)?;
    if span <= 0 {
        return Ok(right.value);
    }
    let t = unit_ratio(i128::from(elapsed), i128::from(span))?;
    finite_value(left.value + (right.value - left.value) * t)
}

fn sample_color(
    channels: &[Channel; 4],
    millis: i64,
    identity: Rgba,
) -> Result<Rgba, CaptionError> {
    Ok(Rgba {
        red: channels[0].sample(millis, identity.red)?,
        green: channels[1].sample(millis, identity.green)?,
        blue: channels[2].sample(millis, identity.blue)?,
        alpha: channels[3].sample(millis, identity.alpha)?,
    })
}

/// `rem_euclid` keeps a negative clock inside the period.
fn phase_in_period(period_ms: i64, millis: i64) -> Result<(i64, i64), CaptionError> {
    if period_ms < 2 {
        return Err(CaptionError::Period);
    }
    Ok((millis.rem_euclid(period_ms), period_ms / 2))
}

fn triangle(period_ms: i64, phase: i64, half: i64) -> Result<f64, CaptionError> {
    let numerator = if phase <= half {
        phase
    } else {
        period_ms.checked_sub(phase).ok_or(CaptionError::Overflow)?
    };
    unit_ratio(i128::from(numerator), i128::from(half))
}

fn millis_at_index(index: usize, shift_ms: i64, millis: i64) -> Result<i64, CaptionError> {
    let index = i64::try_from(index).map_err(|_| CaptionError::Overflow)?;
    let shift = index.checked_mul(shift_ms).ok_or(CaptionError::Overflow)?;
    millis.checked_sub(shift).ok_or(CaptionError::Overflow)
}

fn batch_for(
    letters: &LetterMap,
    range: &TextRange,
    kind: RangeKind,
    curves: &StyleCurves,
    millis: i64,
    stagger_shift_ms: i64,
) -> Result<Option<GlyphBatch>, CaptionError> {
    let Some((start, end)) = byte_span(letters, range)? else {
        return Ok(None);
    };
    let glyphs = place_glyphs(letters, curves, start, end, millis, stagger_shift_ms)?;
    if glyphs.is_empty() {
        Ok(None)
    } else {
        Ok(Some(GlyphBatch { kind, glyphs }))
    }
}

fn word_on_page(words: &[WordBytes], index: usize) -> Option<&WordBytes> {
    words.iter().find(|word| word.index == index)
}

fn byte_span(
    letters: &LetterMap,
    range: &TextRange,
) -> Result<Option<(usize, usize)>, CaptionError> {
    let Some(word) = word_on_page(&letters.words, range.word) else {
        return Ok(None);
    };
    let Some(slice) = letters.text.get(word.byte_start..word.byte_end) else {
        return Ok(None);
    };
    let Some((rel_start, rel_end)) = char_span_bytes(slice, range.char_start, range.char_end)
    else {
        return Ok(None);
    };
    let start = word
        .byte_start
        .checked_add(rel_start)
        .ok_or(CaptionError::Overflow)?;
    let end = word
        .byte_start
        .checked_add(rel_end)
        .ok_or(CaptionError::Overflow)?;
    Ok(Some((start, end)))
}

/// Half-open character indexes inside `slice`. An index past the slice is `None`.
fn char_span_bytes(slice: &str, char_start: usize, char_end: usize) -> Option<(usize, usize)> {
    if char_end < char_start {
        return None;
    }
    let mut start_byte = None;
    let mut end_byte = None;
    let mut index = 0usize;
    for (offset, _) in slice.char_indices() {
        if index == char_start {
            start_byte = Some(offset);
        }
        if index == char_end {
            end_byte = Some(offset);
        }
        if start_byte.is_some() && end_byte.is_some() {
            break;
        }
        index = index.checked_add(1)?;
    }
    if start_byte.is_none() && index == char_start {
        start_byte = Some(slice.len());
    }
    if end_byte.is_none() && index == char_end {
        end_byte = Some(slice.len());
    }
    match (start_byte, end_byte) {
        (Some(start), Some(end)) if start <= end => Some((start, end)),
        _ => None,
    }
}

fn place_glyphs(
    letters: &LetterMap,
    curves: &StyleCurves,
    start: usize,
    end: usize,
    millis: i64,
    stagger_shift_ms: i64,
) -> Result<Vec<PlacedSample>, CaptionError> {
    let mut glyphs = Vec::new();
    for letter in &letters.letters {
        if letter.source < start || letter.source >= end {
            continue;
        }
        let at = millis_at_index(glyphs.len(), stagger_shift_ms, millis)?;
        glyphs.push(PlacedSample {
            glyph: letter.glyph.clone(),
            style: curves.sample(at)?,
        });
    }
    Ok(glyphs)
}

#[cfg(test)]
mod tests {
    use super::{
        Channel, Key, RangeStyles, Rgba, SampledStyle, ShakeAxis, StyleCurves, apply_pulse,
        apply_shake, batches, local_domain, stagger,
    };
    use crate::CaptionError;
    use crate::page::{LetterMap, MappedLetter, WordBytes};
    use crate::select::{RangeKind, Selection, TextRange};
    use cutlass_text::{Alignment, Layout, Line, PlacedGlyph};

    fn ramp() -> Channel {
        Channel {
            keys: vec![
                Key {
                    millis: 1_000,
                    value: 1_000.0,
                },
                Key {
                    millis: 0,
                    value: 0.0,
                },
            ],
        }
    }

    #[test]
    fn an_empty_channel_returns_its_identity() {
        let channel = Channel { keys: Vec::new() };
        assert_eq!(channel.sample(40, 2.5), Ok(2.5));
        assert_eq!(channel.sample(0, f64::NAN), Err(CaptionError::Measure));
    }

    #[test]
    fn a_channel_holds_outside_its_keys_and_lerps_the_middle() {
        let channel = ramp();
        assert_eq!(channel.sample(-10, 7.0), Ok(0.0));
        assert_eq!(channel.sample(0, 7.0), Ok(0.0));
        assert_eq!(channel.sample(500, 7.0), Ok(500.0));
        assert_eq!(channel.sample(1_000, 7.0), Ok(1_000.0));
        assert_eq!(channel.sample(1_500, 7.0), Ok(1_000.0));
        assert_eq!(channel.keys[0].millis, 1_000);
        assert_eq!(channel.keys[1].value, 0.0);
    }

    #[test]
    fn a_later_key_at_the_same_time_wins() {
        let channel = Channel {
            keys: vec![
                Key {
                    millis: 0,
                    value: 0.0,
                },
                Key {
                    millis: 500,
                    value: 1.0,
                },
                Key {
                    millis: 500,
                    value: 9.0,
                },
                Key {
                    millis: 1_000,
                    value: 0.0,
                },
            ],
        };
        assert_eq!(channel.sample(500, 0.0), Ok(9.0));
        assert_eq!(channel.sample(250, 0.0), Ok(4.5));
        assert_eq!(channel.sample(750, 0.0), Ok(4.5));
        assert_eq!(channel.sample(-1, 0.0), Ok(0.0));

        let tied = Channel {
            keys: vec![
                Key {
                    millis: 10,
                    value: 1.0,
                },
                Key {
                    millis: 10,
                    value: 4.0,
                },
            ],
        };
        assert_eq!(tied.sample(10, 0.0), Ok(4.0));
        assert_eq!(tied.sample(0, 0.0), Ok(4.0));
        assert_eq!(tied.sample(11, 0.0), Ok(4.0));
    }

    #[test]
    fn a_non_finite_channel_value_is_measure() {
        let missing = Channel {
            keys: vec![Key {
                millis: 0,
                value: f64::NAN,
            }],
        };
        assert_eq!(missing.sample(0, 0.0), Err(CaptionError::Measure));
        let infinite = Channel {
            keys: vec![Key {
                millis: 0,
                value: f64::INFINITY,
            }],
        };
        assert_eq!(infinite.sample(0, 1.0), Err(CaptionError::Measure));
        let wide = Channel {
            keys: vec![
                Key {
                    millis: 0,
                    value: f64::MIN,
                },
                Key {
                    millis: 2,
                    value: f64::MAX,
                },
            ],
        };
        assert_eq!(wide.sample(1, 0.0), Err(CaptionError::Measure));
        let overflow = Channel {
            keys: vec![
                Key {
                    millis: i64::MIN,
                    value: 0.0,
                },
                Key {
                    millis: i64::MAX,
                    value: 1.0,
                },
            ],
        };
        assert_eq!(overflow.sample(0, 0.0), Err(CaptionError::Overflow));
    }

    #[test]
    fn each_style_curve_samples_its_key() {
        let styles = RangeStyles::identity();
        for curves in [
            &styles.unread,
            &styles.reading,
            &styles.read,
            &styles.keyword,
        ] {
            assert_eq!(curves.sample(12).unwrap(), SampledStyle::identity());
        }

        let mut curves = StyleCurves::identity();
        curves.color = [hold(0.1), hold(0.2), hold(0.3), hold(0.4)];
        curves.background = [hold(0.5), hold(0.6), hold(0.7), hold(0.8)];
        curves.position_x = hold(3.0);
        curves.position_y = hold(4.0);
        curves.rotation = hold(5.0);
        curves.scale_x = hold(6.0);
        curves.scale_y = hold(7.0);
        let style = curves.sample(25).unwrap();
        let identity = SampledStyle::identity();
        assert_eq!(
            style.color,
            Rgba {
                red: 0.1,
                green: 0.2,
                blue: 0.3,
                alpha: 0.4,
            }
        );
        assert_eq!(
            style.background,
            Rgba {
                red: 0.5,
                green: 0.6,
                blue: 0.7,
                alpha: 0.8,
            }
        );
        assert_eq!(style.position_x, 3.0);
        assert_eq!(style.position_y, 4.0);
        assert_eq!(style.rotation, 5.0);
        assert_eq!(style.scale_x, 6.0);
        assert_eq!(style.scale_y, 7.0);
        assert_ne!(style.color, identity.color);
        assert_ne!(style.background, identity.background);
        assert_ne!(style.position_x, identity.position_x);
        assert_ne!(style.position_y, identity.position_y);
        assert_ne!(style.rotation, identity.rotation);
        assert_ne!(style.scale_x, identity.scale_x);
        assert_ne!(style.scale_y, identity.scale_y);
    }

    fn hold(value: f64) -> Channel {
        Channel {
            keys: vec![Key { millis: 0, value }],
        }
    }

    #[test]
    fn shake_changes_when_the_clock_moves() {
        let at_zero = shaken(ShakeAxis::PositionX, 0);
        let at_half = shaken(ShakeAxis::PositionX, 4);
        assert_ne!(at_zero.position_x, at_half.position_x);
        assert_eq!(at_zero.position_x, 3.0);
        assert_eq!(shaken(ShakeAxis::PositionX, 3).position_x, 3.0);
        assert_eq!(at_half.position_x, -3.0);
        assert_eq!(shaken(ShakeAxis::PositionX, 8).position_x, 3.0);
        assert_eq!(shaken(ShakeAxis::PositionX, -1).position_x, -3.0);
        assert_eq!(at_zero.position_y, 0.0);
        assert_eq!(at_zero.rotation, 0.0);
        assert_eq!(at_zero.scale_x, 1.0);

        assert_ne!(
            shaken(ShakeAxis::PositionY, 0).position_y,
            shaken(ShakeAxis::PositionY, 4).position_y
        );
        assert_eq!(shaken(ShakeAxis::PositionY, 0).position_x, 0.0);
        assert_ne!(
            shaken(ShakeAxis::Rotation, 0).rotation,
            shaken(ShakeAxis::Rotation, 4).rotation
        );
        assert_eq!(shaken(ShakeAxis::Rotation, 0).position_x, 0.0);

        let style = SampledStyle::identity();
        assert_eq!(
            apply_shake(style, ShakeAxis::PositionX, 3.0, 1, 0),
            Err(CaptionError::Period)
        );
        assert_eq!(
            apply_shake(style, ShakeAxis::PositionY, 1.0, 0, 0),
            Err(CaptionError::Period)
        );
        assert_eq!(
            apply_shake(style, ShakeAxis::Rotation, 1.0, -4, 0),
            Err(CaptionError::Period)
        );
        assert_eq!(
            apply_shake(style, ShakeAxis::PositionX, f64::NAN, 8, 0),
            Err(CaptionError::Measure)
        );
    }

    fn shaken(axis: ShakeAxis, millis: i64) -> SampledStyle {
        apply_shake(SampledStyle::identity(), axis, 3.0, 8, millis).unwrap()
    }

    #[test]
    fn pulse_rests_at_phase_zero_and_grows_at_the_half() {
        let style = SampledStyle::identity();
        let rest = apply_pulse(style, 0.5, 10, 0).unwrap();
        assert_eq!(rest.scale_x, 1.0);
        assert_eq!(rest.scale_y, 1.0);
        let peak = apply_pulse(style, 0.5, 10, 5).unwrap();
        assert!(peak.scale_x > 1.0);
        assert!(peak.scale_y > 1.0);
        assert_eq!(peak.scale_x, 1.5);
        assert_eq!(peak.scale_y, 1.5);
        assert_eq!(peak.position_x, 0.0);
        assert_eq!(apply_pulse(style, 0.5, 10, 8).unwrap().scale_x, 1.2);
        assert_eq!(apply_pulse(style, 1.0, 11, 5).unwrap().scale_x, 2.0);
        assert_eq!(apply_pulse(style, 0.5, 1, 0), Err(CaptionError::Period));
        assert_eq!(
            apply_pulse(style, f64::INFINITY, 10, 5),
            Err(CaptionError::Measure)
        );
    }

    #[test]
    fn stagger_shifts_later_glyphs_and_reports_overflow() {
        let channel = ramp();
        assert_eq!(stagger(&channel, 0, 400, 1_000, 0.0), Ok(1_000.0));
        assert_eq!(channel.sample(1_000, 0.0), Ok(1_000.0));
        assert_eq!(stagger(&channel, 1, 400, 1_000, 0.0), Ok(600.0));
        assert_eq!(stagger(&channel, 1, -250, 250, 0.0), Ok(500.0));
        assert_eq!(stagger(&channel, 0, i64::MAX, 1_000, 0.0), Ok(1_000.0));
        assert_eq!(
            stagger(&Channel { keys: Vec::new() }, 0, 5, 9, 4.0),
            Ok(4.0)
        );
        assert_eq!(
            stagger(&channel, 2, i64::MAX, 0, 0.0),
            Err(CaptionError::Overflow)
        );
        assert_eq!(
            stagger(&channel, 1, 1, i64::MIN, 0.0),
            Err(CaptionError::Overflow)
        );
        if let Some(index) = index_past_i64() {
            assert_eq!(
                stagger(&channel, index, 1, 0, 0.0),
                Err(CaptionError::Overflow)
            );
        }
    }

    fn index_past_i64() -> Option<usize> {
        let max = usize::try_from(i64::MAX).ok()?;
        let index = max.checked_add(1)?;
        i64::try_from(index).err().map(|_| index)
    }

    #[test]
    fn local_domain_scales_a_spoken_span() {
        assert_eq!(local_domain(99, 100, 300, 1.0), Ok(0.0));
        assert_eq!(local_domain(100, 100, 300, 1.0), Ok(0.0));
        assert_eq!(local_domain(200, 100, 300, 1.0), Ok(0.5));
        assert_eq!(local_domain(300, 100, 300, 1.0), Ok(1.0));
        assert_eq!(local_domain(301, 100, 300, 1.0), Ok(1.0));
        assert_eq!(local_domain(300, 100, 300, 3.0), Ok(3.0));
        assert_eq!(local_domain(200, 100, 300, 3.0), Ok(1.5));
        assert_eq!(local_domain(200, 100, 300, -2.0), Ok(-1.0));
        assert_eq!(local_domain(10, 10, 10, 3.0), Ok(0.0));
        assert_eq!(local_domain(11, 10, 10, 3.0), Ok(3.0));
        assert_eq!(local_domain(0, i64::MIN, i64::MAX, 1.0), Ok(0.5));
        assert_eq!(local_domain(0, 5, 4, 1.0), Err(CaptionError::Inverted));
        assert_eq!(local_domain(0, 0, 10, f64::NAN), Err(CaptionError::Measure));
        assert_eq!(
            local_domain(0, 0, 10, f64::INFINITY),
            Err(CaptionError::Measure)
        );
    }

    #[test]
    fn an_empty_selection_batches_nothing() {
        let styles = RangeStyles::identity();
        let page = two_words();
        let empty = picked(vec![], vec![], vec![], vec![]);
        assert!(batches(&page, &empty, &styles, 0, 0).unwrap().is_empty());

        let mut blank = two_words();
        blank.text.clear();
        blank.words.clear();
        blank.letters.clear();
        blank.layout.lines.clear();
        let reading = picked(vec![], vec![span(0, 0, 1)], vec![], vec![]);
        assert!(batches(&blank, &reading, &styles, 0, 0).unwrap().is_empty());
    }

    #[test]
    fn seeking_changes_the_reading_color() {
        let page = two_words();
        let mut styles = RangeStyles::identity();
        styles.reading.color[0] = Channel {
            keys: vec![
                Key {
                    millis: 0,
                    value: 0.0,
                },
                Key {
                    millis: 1_000,
                    value: 1.0,
                },
            ],
        };
        let selection = picked(vec![], vec![span(0, 0, 1)], vec![], vec![]);
        let early = batches(&page, &selection, &styles, 0, 0).unwrap();
        let late = batches(&page, &selection, &styles, 1_000, 0).unwrap();
        assert_eq!(early[0].glyphs[0].style.color.red, 0.0);
        assert_eq!(late[0].glyphs[0].style.color.red, 1.0);
        assert_ne!(
            early[0].glyphs[0].style.color.red,
            late[0].glyphs[0].style.color.red
        );
    }

    #[test]
    fn a_keyword_batch_is_separate_from_reading() {
        let page = two_words();
        let mut styles = RangeStyles::identity();
        styles.reading.color[0] = hold(0.2);
        styles.keyword.color[0] = hold(0.9);
        let selection = picked(vec![], vec![span(0, 0, 2)], vec![], vec![span(0, 1, 2)]);
        let found = batches(&page, &selection, &styles, 0, 0).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].kind, RangeKind::Reading);
        assert_eq!(found[1].kind, RangeKind::Keyword);
        assert_eq!(found[0].glyphs.len(), 2);
        assert_eq!(found[1].glyphs.len(), 1);
        assert_eq!(
            found[1].glyphs[0].glyph.source,
            found[0].glyphs[1].glyph.source
        );
        assert_eq!(found[1].glyphs[0].glyph.x, found[0].glyphs[1].glyph.x);
        assert_eq!(found[0].glyphs[1].style.color.red, 0.2);
        assert_eq!(found[1].glyphs[0].style.color.red, 0.9);
    }

    #[test]
    fn batch_glyphs_keep_the_placed_source() {
        let page = two_words();
        assert_eq!(page.text.len(), 5);
        assert!(page.text.is_char_boundary(3));
        assert!(page.text.is_char_boundary(4));
        let selection = picked(vec![], vec![span(0, 0, 2)], vec![], vec![]);
        let found = batches(&page, &selection, &RangeStyles::identity(), 0, 0).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].glyphs.len(), 2);
        assert_eq!(found[0].glyphs[0].glyph.x, page.letters[0].glyph.x);
        assert_eq!(
            found[0].glyphs[0].glyph.source,
            page.letters[0].glyph.source
        );
        assert_eq!(found[0].glyphs[1].glyph.x, page.letters[1].glyph.x);
        assert_eq!(
            found[0].glyphs[1].glyph.source,
            page.letters[1].glyph.source
        );
        assert_eq!(found[0].glyphs[0].glyph.x, page.layout.lines[0].glyphs[0].x);
        assert_eq!(
            found[0].glyphs[0].glyph.source,
            page.layout.lines[0].glyphs[0].source
        );
        assert_eq!(found[0].glyphs[0].glyph.character, 'é');
        assert_eq!(found[0].glyphs[1].glyph.source, 2);
    }

    #[test]
    fn batches_walk_each_range_with_its_curves() {
        let page = two_words();
        let mut styles = RangeStyles::identity();
        styles.unread.color[0] = hold(0.1);
        styles.reading.color[0] = hold(0.2);
        styles.read.color[0] = hold(0.3);
        styles.keyword.color[0] = hold(0.4);
        let selection = picked(
            vec![span(1, 0, 1)],
            vec![span(1, 0, 1)],
            vec![span(1, 0, 1)],
            vec![span(1, 0, 1)],
        );
        let found = batches(&page, &selection, &styles, 0, 0).unwrap();
        assert_eq!(found.len(), 4);
        assert_eq!(found[0].kind, RangeKind::Unread);
        assert_eq!(found[1].kind, RangeKind::Reading);
        assert_eq!(found[2].kind, RangeKind::Read);
        assert_eq!(found[3].kind, RangeKind::Keyword);
        assert_eq!(found[0].glyphs[0].style.color.red, 0.1);
        assert_eq!(found[1].glyphs[0].style.color.red, 0.2);
        assert_eq!(found[2].glyphs[0].style.color.red, 0.3);
        assert_eq!(found[3].glyphs[0].style.color.red, 0.4);
        assert_eq!(found[0].glyphs[0].glyph.character, 'y');
        assert_eq!(found[0].glyphs[0].glyph.source, 4);
        assert_eq!(found[0].glyphs[0].glyph.x, 9.0);
    }

    #[test]
    fn each_range_is_its_own_batch() {
        let page = two_words();
        let selection = picked(vec![], vec![span(0, 0, 1), span(1, 0, 1)], vec![], vec![]);
        let found = batches(&page, &selection, &RangeStyles::identity(), 0, 0).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].kind, RangeKind::Reading);
        assert_eq!(found[1].kind, RangeKind::Reading);
        assert_eq!(found[0].glyphs[0].glyph.character, 'é');
        assert_eq!(found[1].glyphs[0].glyph.character, 'y');
    }

    #[test]
    fn stagger_shifts_each_glyph_in_the_batch() {
        let page = two_words();
        let mut styles = RangeStyles::identity();
        styles.reading.color[0] = Channel {
            keys: vec![
                Key {
                    millis: 0,
                    value: 0.0,
                },
                Key {
                    millis: 1_000,
                    value: 1.0,
                },
            ],
        };
        let selection = picked(vec![], vec![span(0, 0, 2)], vec![], vec![]);
        let found = batches(&page, &selection, &styles, 1_000, 1_000).unwrap();
        assert_eq!(found[0].glyphs.len(), 2);
        assert_eq!(found[0].glyphs[0].style.color.red, 1.0);
        assert_eq!(found[0].glyphs[1].style.color.red, 0.0);
        assert_eq!(found[0].glyphs[0].glyph.x, page.letters[0].glyph.x);
        assert_eq!(found[0].glyphs[1].glyph.x, page.letters[1].glyph.x);
    }

    #[test]
    fn a_range_that_does_not_fit_the_page_is_skipped() {
        let page = two_words();
        let selection = picked(
            vec![span(9, 0, 1)],
            vec![span(0, 0, 40), span(0, 3, 1), span(1, 0, 1)],
            vec![],
            vec![],
        );
        let found = batches(&page, &selection, &RangeStyles::identity(), 0, 0).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, RangeKind::Reading);
        assert_eq!(found[0].glyphs[0].glyph.character, 'y');

        let mut broken = two_words();
        broken.words[0].byte_start = 1;
        let only_broken = picked(vec![], vec![span(0, 0, 1)], vec![], vec![]);
        assert!(
            batches(&broken, &only_broken, &RangeStyles::identity(), 0, 0)
                .unwrap()
                .is_empty()
        );
    }

    /// "éx y": é is bytes 0..2, x is 2..3, the space is 3..4, and y is 4..5.
    fn two_words() -> LetterMap {
        let text = "éx y".to_owned();
        let placed = vec![
            placed('é', 0, 1.0, 4),
            placed('x', 2, 5.0, 5),
            placed('y', 4, 9.0, 6),
        ];
        let letters = placed
            .iter()
            .cloned()
            .map(|glyph| {
                let word = if glyph.source < 3 { 0 } else { 1 };
                MappedLetter {
                    source: glyph.source,
                    character: glyph.character,
                    layout_line: 0,
                    word: Some(word),
                    glyph,
                }
            })
            .collect();
        LetterMap {
            text,
            words: vec![
                WordBytes {
                    index: 0,
                    byte_start: 0,
                    byte_end: 3,
                },
                WordBytes {
                    index: 1,
                    byte_start: 4,
                    byte_end: 5,
                },
            ],
            layout: Layout {
                lines: vec![Line {
                    glyphs: placed,
                    width: 40.0,
                    baseline: 16.0,
                }],
                font_size: 16.0,
                scale: 1.0,
                alignment: Alignment::Left,
            },
            letters,
        }
    }

    fn placed(character: char, source: usize, x: f32, glyph_id: u16) -> PlacedGlyph {
        PlacedGlyph {
            source,
            character,
            glyph_id,
            missing: false,
            x,
            advance: 8.0,
        }
    }

    fn span(word: usize, char_start: usize, char_end: usize) -> TextRange {
        TextRange {
            word,
            char_start,
            char_end,
            start: 0,
            end: 1_000,
        }
    }

    fn picked(
        unread: Vec<TextRange>,
        reading: Vec<TextRange>,
        read: Vec<TextRange>,
        keyword: Vec<TextRange>,
    ) -> Selection {
        Selection {
            unread,
            reading,
            read,
            keyword,
            original: cutlass_model::WordColumns::default(),
        }
    }
}
