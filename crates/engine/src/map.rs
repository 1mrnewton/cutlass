//! Timeline time to a source microsecond.
//!
//! This is the only module that speaks the file timescale. Segment ranges
//! stay signed microseconds. The order is freeze, then speed, then reverse.
//! A freeze holds the source timestamp and skips the curve. Reverse then
//! flips that position inside the source range.

use cutlass_codec::Timescale;
use cutlass_model::{
    Draft, MaterialRef, MaterialSpeed, Micros, Segment, SpeedMode, SpeedPoint, TimeRange, TrackType,
};

/// Why a timeline time did not become a source microsecond.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    /// The time is outside the target range, or the source position is outside the source range.
    Outside,
    /// Speed was not finite and greater than zero.
    NonPositiveSpeed,
    /// The timescale does not name a clock.
    NoClock,
    /// The product does not fit in an `i64`.
    Overflow,
    /// The segment stores no source range.
    NoSource,
    /// Curve mode has fewer than two points. The constant speed number stays in force.
    UnusedCurve,
}

/// Constant speed, or curve points `(x, y)`.
///
/// `x` is the fraction of the source duration. `y` is speed and is linear in `x`.
#[derive(Clone, Debug, PartialEq)]
pub enum MapSpeed {
    Constant(f64),
    Curve(Vec<SpeedPoint>),
}

/// The segment fields the map reads. Ranges are microseconds.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineMap {
    pub target: TimeRange,
    pub source: TimeRange,
    pub freeze: Option<Micros>,
    pub speed: MapSpeed,
    pub reverse: bool,
}

/// Ticks of `timescale` as signed microseconds, rounded to the nearest tick of that clock.
pub fn ticks_to_micros(ticks: i64, timescale: Timescale) -> Result<Micros, MapError> {
    scale(ticks, 1_000_000, i128::from(timescale.ticks_per_second())).map(Micros)
}

/// Microseconds as ticks of `timescale`, rounded to the nearest tick.
pub fn micros_to_ticks(time: Micros, timescale: Timescale) -> Result<i64, MapError> {
    scale(time.0, i128::from(timescale.ticks_per_second()), 1_000_000)
}

/// Instantaneous speed at `time`. A freeze does not consult the curve.
pub fn speed_at(map: &TimelineMap, time: Micros) -> Result<f64, MapError> {
    if map.freeze.is_some() {
        return Ok(1.0);
    }
    match &map.speed {
        MapSpeed::Constant(speed) => positive(*speed),
        MapSpeed::Curve(points) => {
            let elapsed = elapsed_in(map, time)?;
            let duration = finite_f64(map.source.duration.0)?;
            let fraction = source_fraction(points, finite_f64(elapsed)? / duration)?;
            speed_y(points, fraction)
        }
    }
}

/// Source microsecond for `time` on the timeline.
///
/// Freeze wins and skips the curve. Speed then maps the elapsed target time.
/// Reverse reflects the result inside the half-open source range.
pub fn map_timeline(map: &TimelineMap, time: Micros) -> Result<Micros, MapError> {
    let elapsed = elapsed_in(map, time)?;
    let position = if let Some(freeze) = map.freeze {
        freeze.0
    } else {
        match &map.speed {
            MapSpeed::Constant(speed) => constant_position(map, elapsed, *speed)?,
            MapSpeed::Curve(points) => curve_position(map, elapsed, points)?,
        }
    };
    let position = if map.reverse {
        flip(map.source.start.0, map.source.duration.0, position)?
    } else {
        inside(map.source.start.0, map.source.duration.0, position)?
    };
    Ok(Micros(position))
}

/// The map stored on `segment`. A curve with fewer than two points keeps the constant speed.
pub fn placement_of(draft: &Draft, segment: &Segment) -> Result<TimelineMap, MapError> {
    let Some(source) = segment.source_range.clone() else {
        return Err(MapError::NoSource);
    };
    Ok(TimelineMap {
        target: segment.target.clone(),
        source,
        freeze: freeze_timestamp(draft, segment),
        speed: speed_of(draft, segment)?,
        reverse: reverse_flag(draft, segment),
    })
}

/// File to open. A reverse flag with a path uses that path. A flag with no path uses the forward file.
pub fn media_path<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a str> {
    if let Some(MaterialRef::Video(video)) = material(draft, segment) {
        if segment.reverse && !video.reverse_path.is_empty() {
            return Some(video.reverse_path.as_str());
        }
        if !video.path.is_empty() {
            return Some(video.path.as_str());
        }
    }
    if let Some(MaterialRef::Audio(audio)) = material(draft, segment)
        && !audio.path.is_empty()
    {
        return Some(audio.path.as_str());
    }
    None
}

/// Text, a sticker, a shape, or a solid color is not a decoder input.
#[must_use]
pub fn is_graphic(draft: &Draft, segment: &Segment) -> bool {
    if draft.tracks.iter().any(|track| {
        track.segments.iter().any(|item| item.id == segment.id) && graphic_track(&track.kind)
    }) {
        return true;
    }
    matches!(
        material(draft, segment),
        Some(MaterialRef::Text(_) | MaterialRef::Shape(_) | MaterialRef::Color(_))
    )
}

fn graphic_track(kind: &TrackType) -> bool {
    matches!(
        kind,
        TrackType::Text
            | TrackType::Sticker
            | TrackType::Effect
            | TrackType::Filter
            | TrackType::Adjust
            | TrackType::TextToVideoText
    )
}

fn material<'a>(draft: &'a Draft, segment: &Segment) -> Option<MaterialRef<'a>> {
    draft.materials.get(segment.material_id.as_str())
}

fn freeze_timestamp(draft: &Draft, segment: &Segment) -> Option<Micros> {
    match material(draft, segment) {
        Some(MaterialRef::Video(video)) => video.freeze.as_ref().map(|freeze| freeze.timestamp),
        _ => None,
    }
}

fn reverse_flag(draft: &Draft, segment: &Segment) -> bool {
    if !segment.reverse {
        return false;
    }
    match material(draft, segment) {
        Some(MaterialRef::Video(video)) => video.reverse_path.is_empty(),
        _ => true,
    }
}

fn speed_of(draft: &Draft, segment: &Segment) -> Result<MapSpeed, MapError> {
    for id in &segment.extra_material_refs {
        if let Some(MaterialRef::Speed(speed)) = draft.materials.get(id.as_str()) {
            return speed_law(speed);
        }
    }
    Ok(MapSpeed::Constant(segment.speed))
}

fn speed_law(speed: &MaterialSpeed) -> Result<MapSpeed, MapError> {
    if speed.mode != SpeedMode::Curve {
        return Ok(MapSpeed::Constant(speed.speed));
    }
    let points = speed
        .curve_speed
        .as_ref()
        .map(|curve| curve.speed_points.clone())
        .unwrap_or_default();
    match prepare(&points) {
        Ok(_) => Ok(MapSpeed::Curve(points)),
        Err(MapError::UnusedCurve) => Ok(MapSpeed::Constant(speed.speed)),
        Err(error) => Err(error),
    }
}

fn elapsed_in(map: &TimelineMap, time: Micros) -> Result<i64, MapError> {
    if map.target.duration.0 <= 0 || map.source.duration.0 <= 0 {
        return Err(MapError::Outside);
    }
    let elapsed = time
        .0
        .checked_sub(map.target.start.0)
        .ok_or(MapError::Overflow)?;
    if elapsed < 0 || elapsed >= map.target.duration.0 {
        return Err(MapError::Outside);
    }
    Ok(elapsed)
}

fn positive(speed: f64) -> Result<f64, MapError> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(MapError::NonPositiveSpeed);
    }
    Ok(speed)
}

fn constant_position(map: &TimelineMap, elapsed: i64, speed: f64) -> Result<i64, MapError> {
    let speed = positive(speed)?;
    let offset = round_i64(finite_f64(elapsed)? * speed)?;
    map.source
        .start
        .0
        .checked_add(offset)
        .ok_or(MapError::Overflow)
}

fn finite_f64(value: i64) -> Result<f64, MapError> {
    if value.unsigned_abs() > (1_u64 << 53) {
        return Err(MapError::Overflow);
    }
    Ok(value as f64)
}

fn curve_position(map: &TimelineMap, elapsed: i64, points: &[SpeedPoint]) -> Result<i64, MapError> {
    let duration = finite_f64(map.source.duration.0)?;
    let fraction = source_fraction(points, finite_f64(elapsed)? / duration)?;
    let offset = round_i64(fraction * duration)?;
    map.source
        .start
        .0
        .checked_add(offset)
        .ok_or(MapError::Overflow)
}

fn inside(start: i64, duration: i64, position: i64) -> Result<i64, MapError> {
    let end = start.checked_add(duration).ok_or(MapError::Overflow)?;
    if position == end && duration > 0 {
        return Ok(end - 1);
    }
    if position < start || position >= end {
        return Err(MapError::Outside);
    }
    Ok(position)
}

/// Reflect `position` so the start and the last microsecond trade places.
fn flip(start: i64, duration: i64, position: i64) -> Result<i64, MapError> {
    let position = inside(start, duration, position)?;
    let end = start.checked_add(duration).ok_or(MapError::Overflow)?;
    let offset = position.checked_sub(start).ok_or(MapError::Overflow)?;
    end.checked_sub(1)
        .and_then(|last| last.checked_sub(offset))
        .ok_or(MapError::Overflow)
}

fn scale(value: i64, numer: i128, denom: i128) -> Result<i64, MapError> {
    if denom <= 0 {
        return Err(MapError::NoClock);
    }
    let product = i128::from(value)
        .checked_mul(numer)
        .ok_or(MapError::Overflow)?;
    let half = denom / 2;
    let rounded = if product >= 0 {
        product.checked_add(half).ok_or(MapError::Overflow)? / denom
    } else {
        let magnitude = product.checked_neg().ok_or(MapError::Overflow)?;
        let rounded = magnitude.checked_add(half).ok_or(MapError::Overflow)? / denom;
        rounded.checked_neg().ok_or(MapError::Overflow)?
    };
    i64::try_from(rounded).map_err(|_| MapError::Overflow)
}

fn round_i64(value: f64) -> Result<i64, MapError> {
    if !value.is_finite() || value.abs() >= (1_i64 << 53) as f64 {
        return Err(MapError::Overflow);
    }
    Ok(value.round() as i64)
}

fn speed_y(points: &[SpeedPoint], x: f64) -> Result<f64, MapError> {
    let prepared = prepare(points)?;
    let (first_x, first_y) = prepared[0];
    if x <= first_x {
        return positive(first_y);
    }
    let (last_x, last_y) = prepared[prepared.len() - 1];
    if x >= last_x {
        return positive(last_y);
    }
    for pair in prepared.windows(2) {
        if x >= pair[0].0 && x <= pair[1].0 {
            return positive(interpolate(pair[0].0, pair[0].1, pair[1].0, pair[1].1, x));
        }
    }
    positive(last_y)
}

struct Piece {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

fn source_fraction(points: &[SpeedPoint], need: f64) -> Result<f64, MapError> {
    if !need.is_finite() || need < 0.0 {
        return Err(MapError::Outside);
    }
    let pieces = curve_pieces(points)?;
    let mut total = 0.0;
    for piece in &pieces {
        total += piece_factor(piece)?;
    }
    if need >= total {
        return Ok(1.0);
    }
    let mut cursor = 0.0;
    for piece in &pieces {
        let span = piece_factor(piece)?;
        if need <= cursor + span {
            return invert_piece(piece, need - cursor);
        }
        cursor += span;
    }
    Ok(1.0)
}

fn curve_pieces(points: &[SpeedPoint]) -> Result<Vec<Piece>, MapError> {
    let prepared = prepare(points)?;
    let (first_x, first_y) = prepared[0];
    let (last_x, last_y) = prepared[prepared.len() - 1];
    let mut pieces = Vec::new();
    if first_x > 0.0 {
        let end = first_x.min(1.0);
        if end > 0.0 {
            pieces.push(Piece {
                x0: 0.0,
                x1: end,
                y0: first_y,
                y1: first_y,
            });
        }
    }
    for pair in prepared.windows(2) {
        let (left_x, left_y) = pair[0];
        let (right_x, right_y) = pair[1];
        let start = left_x.max(0.0);
        let end = right_x.min(1.0);
        if end <= start {
            continue;
        }
        pieces.push(Piece {
            x0: start,
            x1: end,
            y0: interpolate(left_x, left_y, right_x, right_y, start),
            y1: interpolate(left_x, left_y, right_x, right_y, end),
        });
    }
    if last_x < 1.0 {
        let start = last_x.max(0.0);
        if start < 1.0 {
            pieces.push(Piece {
                x0: start,
                x1: 1.0,
                y0: last_y,
                y1: last_y,
            });
        }
    }
    Ok(pieces)
}

fn prepare(points: &[SpeedPoint]) -> Result<Vec<(f64, f64)>, MapError> {
    let mut indexed = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if !point.x.is_finite() {
            return Err(MapError::Overflow);
        }
        if !point.y.is_finite() || point.y <= 0.0 {
            return Err(MapError::NonPositiveSpeed);
        }
        indexed.push((index, point.x, point.y));
    }
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut prepared: Vec<(f64, f64)> = Vec::new();
    for (_, x, y) in indexed {
        if let Some(last) = prepared.last_mut()
            && last.0 == x
        {
            last.1 = y;
            continue;
        }
        prepared.push((x, y));
    }
    if prepared.len() < 2 {
        return Err(MapError::UnusedCurve);
    }
    Ok(prepared)
}

fn interpolate(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    let span = x1 - x0;
    if span == 0.0 {
        return y1;
    }
    let t = (x - x0) / span;
    y0 + (y1 - y0) * t
}

fn piece_factor(piece: &Piece) -> Result<f64, MapError> {
    if piece.y0 <= 0.0 || piece.y1 <= 0.0 || !piece.y0.is_finite() || !piece.y1.is_finite() {
        return Err(MapError::NonPositiveSpeed);
    }
    let width = piece.x1 - piece.x0;
    if width <= 0.0 {
        return Ok(0.0);
    }
    let value = if (piece.y0 - piece.y1).abs() <= f64::EPSILON * piece.y0.abs().max(1.0) {
        width / piece.y0
    } else {
        width * (piece.y1 / piece.y0).ln() / (piece.y1 - piece.y0)
    };
    if !value.is_finite() || value < 0.0 {
        return Err(MapError::Overflow);
    }
    Ok(value)
}

fn invert_piece(piece: &Piece, local: f64) -> Result<f64, MapError> {
    if local <= 0.0 {
        return Ok(piece.x0);
    }
    let width = piece.x1 - piece.x0;
    if width <= 0.0 {
        return Ok(piece.x0);
    }
    let close = (piece.y0 - piece.y1).abs() <= f64::EPSILON * piece.y0.abs().max(1.0);
    let x = if close {
        piece.x0 + local * piece.y0
    } else {
        let y = piece.y0 * (local * (piece.y1 - piece.y0) / width).exp();
        let t = (y - piece.y0) / (piece.y1 - piece.y0);
        piece.x0 + t * width
    };
    if !x.is_finite() {
        return Err(MapError::Overflow);
    }
    Ok(x.clamp(piece.x0, piece.x1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_model::{
        Draft, FreezeInfo, MaterialId, MaterialSpeed, MaterialVideo, Micros, Segment, SpeedMode,
        SpeedPoint, TimeRange, curve_speed_factor, curve_target_duration,
    };
    use serde_json::Map;

    fn point(x: f64, y: f64) -> SpeedPoint {
        SpeedPoint {
            x,
            y,
            rest: Map::new(),
        }
    }

    fn mapped(
        source: i64,
        target: i64,
        speed: MapSpeed,
        reverse: bool,
        at: i64,
    ) -> Result<i64, MapError> {
        let map = TimelineMap {
            target: TimeRange::new(0, target),
            source: TimeRange::new(0, source),
            freeze: None,
            speed,
            reverse,
        };
        map_timeline(&map, Micros(at)).map(|time| time.0)
    }

    #[test]
    fn file_ticks_convert_only_through_the_timescale() {
        let scale = Timescale::new(90_000);
        assert_eq!(ticks_to_micros(90_000, scale), Ok(Micros(1_000_000)));
        assert_eq!(micros_to_ticks(Micros(1_000_000), scale), Ok(90_000));
        assert_eq!(ticks_to_micros(-90_000, scale), Ok(Micros(-1_000_000)));
        let audio = Timescale::new(48_000);
        assert_eq!(micros_to_ticks(Micros(1_000_000), audio), Ok(48_000));
        assert_eq!(ticks_to_micros(48_000, audio), Ok(Micros(1_000_000)));
        assert_eq!(ticks_to_micros(1, Timescale::new(3)), Ok(Micros(333_333)));
        assert_eq!(ticks_to_micros(2, Timescale::new(3)), Ok(Micros(666_667)));
        assert_eq!(
            ticks_to_micros(1, Timescale::new(0)),
            Err(MapError::NoClock)
        );
    }

    #[test]
    fn constant_speed_then_reverse_flips_inside_the_source_range() {
        assert_eq!(
            mapped(2_000_000, 1_000_000, MapSpeed::Constant(2.0), false, 0),
            Ok(0)
        );
        assert_eq!(
            mapped(
                2_000_000,
                1_000_000,
                MapSpeed::Constant(2.0),
                false,
                500_000
            ),
            Ok(1_000_000)
        );
        assert_eq!(
            mapped(1_000, 1_000, MapSpeed::Constant(1.0), true, 0),
            Ok(999)
        );
        assert_eq!(
            mapped(1_000, 1_000, MapSpeed::Constant(1.0), true, 999),
            Ok(0)
        );
        assert_eq!(
            mapped(1_000, 1_000, MapSpeed::Constant(1.0), true, 250),
            Ok(749)
        );
        assert_eq!(
            mapped(1_000, 1_000, MapSpeed::Constant(1.0), false, 1_000),
            Err(MapError::Outside)
        );
        assert_eq!(
            mapped(1_000, 1_000, MapSpeed::Constant(0.0), false, 0),
            Err(MapError::NonPositiveSpeed)
        );
    }

    #[test]
    fn a_freeze_holds_the_timestamp_and_skips_the_curve() {
        let map = TimelineMap {
            target: TimeRange::new(10, 5_000),
            source: TimeRange::new(0, 2_000),
            freeze: Some(Micros(100)),
            speed: MapSpeed::Curve(vec![point(0.0, 1.0), point(1.0, 8.0)]),
            reverse: false,
        };
        assert_eq!(map_timeline(&map, Micros(10)), Ok(Micros(100)));
        assert_eq!(map_timeline(&map, Micros(4_000)), Ok(Micros(100)));
        let reversed = TimelineMap {
            reverse: true,
            ..map
        };
        assert_eq!(map_timeline(&reversed, Micros(10)), Ok(Micros(1_899)));
    }

    #[test]
    fn a_curve_seek_inverts_the_integral() {
        let ramp = vec![point(0.0, 1.0), point(1.0, 2.0)];
        assert_eq!(
            curve_target_duration(Micros(1_000_000), &ramp),
            Ok(Micros(693_147))
        );
        assert_eq!(
            mapped(1_000_000, 693_147, MapSpeed::Curve(ramp.clone()), false, 0),
            Ok(0)
        );
        let need: f64 = 346_574.0 / 1_000_000.0;
        let expected = ((need.exp() - 1.0) * 1_000_000.0).round() as i64;
        assert_eq!(
            mapped(
                1_000_000,
                693_147,
                MapSpeed::Curve(ramp.clone()),
                false,
                346_574
            ),
            Ok(expected)
        );
        let end_need: f64 = 693_146.0 / 1_000_000.0;
        let end_expected = ((end_need.exp() - 1.0) * 1_000_000.0).round() as i64;
        assert_eq!(
            mapped(1_000_000, 693_147, MapSpeed::Curve(ramp), false, 693_146),
            Ok(end_expected)
        );
        assert_eq!(
            mapped(
                1_000_000,
                693_147,
                MapSpeed::Curve(vec![point(0.0, 1.0), point(1.0, 2.0)]),
                false,
                693_147
            ),
            Err(MapError::Outside)
        );

        let flat = vec![point(0.0, 2.0), point(1.0, 2.0)];
        assert_eq!(
            mapped(2_000_000, 1_000_000, MapSpeed::Curve(flat), false, 500_000),
            Ok(1_000_000)
        );
        assert_eq!(
            mapped(
                1_000_000,
                250_000,
                MapSpeed::Curve(vec![point(0.0, 1.0), point(0.0, 4.0), point(1.0, 4.0)]),
                false,
                125_000
            ),
            Ok(500_000)
        );
        assert_eq!(
            source_fraction(&[point(0.0, 0.0), point(1.0, 1.0)], 0.1),
            Err(MapError::NonPositiveSpeed)
        );
        assert_eq!(
            source_fraction(&[point(0.0, 1.0)], 0.1),
            Err(MapError::UnusedCurve)
        );
    }

    #[test]
    fn the_piece_integral_matches_the_model_including_a_held_tail() {
        let cases = [
            vec![point(0.0, 1.0), point(1.0, 1.0)],
            vec![point(0.0, 1.0), point(1.0, 2.0)],
            vec![point(0.25, 2.0), point(0.75, 2.0)],
            vec![point(-0.2, 1.0), point(1.2, 3.0)],
            vec![point(0.0, 1.0), point(0.0, 4.0), point(1.0, 4.0)],
        ];
        for points in cases {
            let mut total = 0.0;
            for piece in curve_pieces(&points).expect("pieces") {
                total += piece_factor(&piece).expect("piece");
            }
            let model = curve_speed_factor(&points).expect("model");
            assert!((total - model).abs() < 1e-12, "{total} vs {model}");
        }
    }

    fn video(path: &str, reverse_path: &str, freeze: Option<i64>) -> MaterialVideo {
        let mut video = MaterialVideo::new("v");
        video.path = path.to_owned();
        video.reverse_path = reverse_path.to_owned();
        video.freeze = freeze.map(|timestamp| FreezeInfo {
            timestamp: Micros(timestamp),
            ..FreezeInfo::default()
        });
        video
    }

    #[test]
    fn a_reverse_flag_with_no_path_flips_and_a_path_does_not() {
        let mut draft = Draft::empty("D");
        draft.materials.videos.push(video("forward", "", None));
        let mut segment = Segment::new("s", "v", TimeRange::new(0, 1_000));
        segment.source_range = Some(TimeRange::new(0, 1_000));
        segment.reverse = true;
        draft.tracks[0].segments.push(segment);
        let map = placement_of(&draft, &draft.tracks[0].segments[0]).expect("map");
        assert!(map.reverse);
        assert_eq!(
            media_path(&draft, &draft.tracks[0].segments[0]),
            Some("forward")
        );
        assert_eq!(map_timeline(&map, Micros(0)), Ok(Micros(999)));

        draft.materials.videos[0].reverse_path = "backward".to_owned();
        let map = placement_of(&draft, &draft.tracks[0].segments[0]).expect("path");
        assert!(!map.reverse);
        assert_eq!(
            media_path(&draft, &draft.tracks[0].segments[0]),
            Some("backward")
        );
        assert_eq!(map_timeline(&map, Micros(0)), Ok(Micros(0)));
    }

    #[test]
    fn fewer_than_two_curve_points_keep_the_constant_speed() {
        let mut draft = Draft::empty("D");
        draft.materials.videos.push(video("clip", "", Some(40)));
        let speed = MaterialSpeed {
            id: MaterialId::new("speed"),
            mode: SpeedMode::Curve,
            speed: 2.0,
            curve_speed: Some(cutlass_model::CurveSpeed {
                speed_points: vec![point(0.0, 4.0)],
                ..cutlass_model::CurveSpeed::default()
            }),
            ..MaterialSpeed::default()
        };
        draft.materials.speeds.push(speed);
        let mut segment = Segment::new("s", "v", TimeRange::new(0, 1_000));
        segment.source_range = Some(TimeRange::new(0, 2_000));
        segment.extra_material_refs.push(MaterialId::new("speed"));
        segment.speed = 9.0;
        draft.tracks[0].segments.push(segment);
        let map = placement_of(&draft, &draft.tracks[0].segments[0]).expect("constant");
        assert_eq!(map.freeze, Some(Micros(40)));
        assert_eq!(map_timeline(&map, Micros(10)), Ok(Micros(40)));

        draft.materials.videos[0].freeze = None;
        draft.materials.speeds[0].mode = SpeedMode::Normal;
        let map = placement_of(&draft, &draft.tracks[0].segments[0]).expect("normal");
        assert_eq!(map_timeline(&map, Micros(100)), Ok(Micros(200)));
    }

    #[test]
    fn text_is_not_a_media_path() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].kind = TrackType::Text;
        let segment = Segment::new("s", "t", TimeRange::new(0, 1));
        draft.tracks[0].segments.push(segment);
        assert!(is_graphic(&draft, &draft.tracks[0].segments[0]));
    }
}
