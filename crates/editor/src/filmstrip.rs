//! Where a clip's filmstrip tiles sit, as CapCut 9.6.0 lays them out.
//!
//! Tiles start at the clip's left edge and sit side by side, each one tile
//! wide; the last is cut at the clip's right edge. Only tiles that touch the
//! visible span are returned. A tile shows the picture at its left edge:
//! `x / pixel_size` microseconds into the clip.

use std::ops::Range;

use cutlass_model::Micros;

/// The most tiles one clip asks for at once.
pub const MAX_TILES: usize = 256;

/// One tile, in pixels from the clip's left edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmTile {
    pub left: f64,
    pub width: f64,
    /// Time into the clip, on the timeline clock, at the tile's left edge.
    pub offset: Micros,
}

/// The tiles of a clip `clip_width` pixels wide whose span meets `visible`,
/// pixels from the clip's left edge. `pixel_size` is pixels per microsecond.
#[must_use]
pub fn film_tiles(
    clip_width: f64,
    tile_width: f64,
    visible: Range<f64>,
    pixel_size: f64,
) -> Vec<FilmTile> {
    let usable = |value: f64| value.is_finite() && value > 0.0;
    if !usable(clip_width) || !usable(tile_width) || !usable(pixel_size) {
        return Vec::new();
    }
    let from = visible.start.max(0.0);
    let to = visible.end.min(clip_width);
    if !from.is_finite() || !to.is_finite() || to <= from {
        return Vec::new();
    }
    let first = (from / tile_width).floor() as usize;
    let mut tiles = Vec::new();
    for index in first.. {
        let left = index as f64 * tile_width;
        if left >= to || tiles.len() == MAX_TILES {
            break;
        }
        tiles.push(FilmTile {
            left,
            width: tile_width.min(clip_width - left),
            offset: Micros((left / pixel_size).floor() as i64),
        });
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_start_at_the_clip_and_the_last_is_cut_at_its_end() {
        let tiles = film_tiles(250.0, 100.0, 0.0..250.0, 0.0001);
        let lefts: Vec<f64> = tiles.iter().map(|tile| tile.left).collect();
        assert_eq!(lefts, [0.0, 100.0, 200.0]);
        assert_eq!(tiles[2].width, 50.0);
        assert_eq!(tiles[1].offset, Micros(1_000_000));
    }

    #[test]
    fn only_tiles_that_meet_the_visible_span_are_returned() {
        let tiles = film_tiles(1_000.0, 100.0, 250.0..420.0, 0.0001);
        let lefts: Vec<f64> = tiles.iter().map(|tile| tile.left).collect();
        assert_eq!(lefts, [200.0, 300.0, 400.0]);
        assert!(film_tiles(1_000.0, 100.0, 1_200.0..1_500.0, 0.0001).is_empty());
        assert!(film_tiles(1_000.0, 0.0, 0.0..100.0, 0.0001).is_empty());
    }

    #[test]
    fn a_very_long_clip_stops_at_the_tile_limit() {
        let tiles = film_tiles(1.0e9, 10.0, 0.0..1.0e9, 0.0001);
        assert_eq!(tiles.len(), MAX_TILES);
    }
}
