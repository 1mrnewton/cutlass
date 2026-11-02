//! A small picture cache for the timeline.
//!
//! A lookup misses immediately. It does not decode. A stored cell's long side
//! is at most [`MAX_EDGE`], and the strip keeps at most [`MAX_STRIP`] cells.

use cutlass_model::Micros;
use cutlass_picture::rgba_len;

use crate::error::EngineError;

/// Cells kept for one segment. The farthest cell is dropped past this.
pub const MAX_STRIP: usize = 16;

/// Longest edge of a stored cell, in pixels.
pub const MAX_EDGE: u32 = 160;

/// One timeline thumbnail, keyed by source time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StripCell {
    time: Micros,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl StripCell {
    #[must_use]
    pub const fn time(&self) -> Micros {
        self.time
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

pub(crate) struct Filmstrip {
    cells: Vec<StripCell>,
}

impl Filmstrip {
    pub(crate) fn new() -> Self {
        Self {
            cells: Vec::with_capacity(MAX_STRIP),
        }
    }

    /// The newest cell at or before `time`. The scan does not allocate.
    pub(crate) fn at(&self, time: i64) -> Option<&StripCell> {
        self.cells
            .iter()
            .filter(|cell| cell.time.0 <= time)
            .max_by_key(|cell| cell.time.0)
    }

    pub(crate) fn insert(&mut self, cell: StripCell) {
        if let Some(existing) = self
            .cells
            .iter_mut()
            .find(|stored| stored.time == cell.time)
        {
            *existing = cell;
            return;
        }
        if self.cells.len() == MAX_STRIP {
            self.drop_farthest(cell.time.0);
        }
        self.cells.push(cell);
    }

    fn drop_farthest(&mut self, time: i64) {
        let Some(index) = self
            .cells
            .iter()
            .enumerate()
            .max_by_key(|(_, cell)| cell.time.0.abs_diff(time))
            .map(|(index, _)| index)
        else {
            return;
        };
        self.cells.remove(index);
    }
}

/// Scale `rgba` so the long side is at most [`MAX_EDGE`].
pub(crate) fn fit(
    time: Micros,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<StripCell, EngineError> {
    if width == 0 || height == 0 {
        return Err(EngineError::Window);
    }
    let Some(expected) = rgba_len(width, height) else {
        return Err(EngineError::Hostile {
            declared: byte_len(rgba.len()),
            limit: 0,
        });
    };
    if rgba.len() != expected {
        return Err(EngineError::Hostile {
            declared: byte_len(rgba.len()),
            limit: byte_len(expected),
        });
    }
    let (width_out, height_out) = fitted(width, height)?;
    let bytes = if width_out == width && height_out == height {
        rgba.to_vec()
    } else {
        scale(width, height, rgba, width_out, height_out)?
    };
    Ok(StripCell {
        time,
        width: width_out,
        height: height_out,
        rgba: bytes,
    })
}

fn fitted(width: u32, height: u32) -> Result<(u32, u32), EngineError> {
    let long = width.max(height);
    if long <= MAX_EDGE {
        return Ok((width, height));
    }
    let (dst_w, dst_h) = if width >= height {
        let dst_h = (u64::from(height) * u64::from(MAX_EDGE) / u64::from(width)).max(1);
        let Ok(dst_h) = u32::try_from(dst_h) else {
            return Err(EngineError::Window);
        };
        (MAX_EDGE, dst_h)
    } else {
        let dst_w = (u64::from(width) * u64::from(MAX_EDGE) / u64::from(height)).max(1);
        let Ok(dst_w) = u32::try_from(dst_w) else {
            return Err(EngineError::Window);
        };
        (dst_w, MAX_EDGE)
    };
    if dst_w == 0 || dst_h == 0 || dst_w > MAX_EDGE || dst_h > MAX_EDGE {
        return Err(EngineError::Window);
    }
    Ok((dst_w, dst_h))
}

fn scale(
    width: u32,
    height: u32,
    rgba: &[u8],
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, EngineError> {
    if dst_w == 0 || dst_h == 0 || dst_w > MAX_EDGE || dst_h > MAX_EDGE {
        return Err(EngineError::Window);
    }
    let Some(dst_len) = rgba_len(dst_w, dst_h) else {
        return Err(EngineError::Hostile {
            declared: byte_len(rgba.len()),
            limit: 0,
        });
    };
    let mut out = Vec::with_capacity(dst_len);
    for y in 0..dst_h {
        let src_y = nearest(y, height, dst_h)?;
        for x in 0..dst_w {
            let src_x = nearest(x, width, dst_w)?;
            out.extend_from_slice(pixel(rgba, src_x, src_y, width)?);
        }
    }
    if out.len() != dst_len {
        return Err(EngineError::Hostile {
            declared: byte_len(out.len()),
            limit: byte_len(dst_len),
        });
    }
    Ok(out)
}

fn nearest(pos: u32, src_edge: u32, dst_edge: u32) -> Result<u32, EngineError> {
    if src_edge == 0 || dst_edge == 0 {
        return Err(EngineError::Window);
    }
    let value = u64::from(pos) * u64::from(src_edge) / u64::from(dst_edge);
    let value = u32::try_from(value).map_err(|_| EngineError::Hostile {
        declared: value,
        limit: u64::from(src_edge),
    })?;
    Ok(value.min(src_edge - 1))
}

fn pixel(rgba: &[u8], x: u32, y: u32, width: u32) -> Result<&[u8], EngineError> {
    let from = pixel_index(x, y, width)?;
    let end = from.checked_add(4).ok_or(EngineError::Hostile {
        declared: byte_len(rgba.len()),
        limit: 0,
    })?;
    rgba.get(from..end).ok_or(EngineError::Hostile {
        declared: byte_len(rgba.len()),
        limit: byte_len(end),
    })
}

fn pixel_index(x: u32, y: u32, width: u32) -> Result<usize, EngineError> {
    let x = usize::try_from(x).map_err(|_| EngineError::Hostile {
        declared: u64::from(x),
        limit: 0,
    })?;
    let y = usize::try_from(y).map_err(|_| EngineError::Hostile {
        declared: u64::from(y),
        limit: 0,
    })?;
    let width = usize::try_from(width).map_err(|_| EngineError::Hostile {
        declared: u64::from(width),
        limit: 0,
    })?;
    y.checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(4))
        .ok_or(EngineError::Hostile {
            declared: byte_len(width),
            limit: 0,
        })
}

fn byte_len(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel_cell(time: i64, red: u8) -> StripCell {
        fit(Micros(time), 1, 1, &[red, 0, 0, 255]).expect("cell")
    }

    #[test]
    fn a_miss_stores_nothing() {
        let strip = Filmstrip::new();
        assert!(strip.at(0).is_none());
        assert!(strip.cells.is_empty());
    }

    #[test]
    fn a_later_time_uses_the_newest_cell_at_or_before_it() {
        let mut strip = Filmstrip::new();
        strip.insert(pixel_cell(10, 1));
        strip.insert(pixel_cell(20, 2));
        assert!(strip.at(9).is_none());
        assert_eq!(strip.at(10).expect("ten").rgba()[0], 1);
        assert_eq!(strip.at(15).expect("between").rgba()[0], 1);
        assert_eq!(strip.at(20).expect("twenty").rgba()[0], 2);
        strip.insert(pixel_cell(10, 9));
        assert_eq!(strip.cells.len(), 2);
        assert_eq!(strip.at(10).expect("replaced").rgba()[0], 9);
    }

    #[test]
    fn a_long_edge_is_scaled_to_the_cap() {
        let small = fit(Micros(0), 1, 1, &[4, 5, 6, 7]).expect("small");
        assert_eq!(small.width(), 1);
        assert_eq!(small.rgba(), &[4, 5, 6, 7]);

        let full = vec![3_u8; usize::try_from(MAX_EDGE).expect("edge") * 4];
        let kept = fit(Micros(0), MAX_EDGE, 1, &full).expect("kept");
        assert_eq!(kept.width(), MAX_EDGE);
        assert_eq!(kept.rgba(), full.as_slice());

        let mut wide = vec![0_u8; 320 * 4];
        wide[0] = 9;
        wide[318 * 4] = 7;
        let wide = fit(Micros(4), 320, 1, &wide).expect("wide");
        assert_eq!((wide.width(), wide.height()), (MAX_EDGE, 1));
        assert_eq!(wide.time(), Micros(4));
        assert_eq!(wide.rgba()[0], 9);
        assert_eq!(wide.rgba()[159 * 4], 7);

        let mut tall = vec![0_u8; 320 * 4];
        tall[0] = 9;
        tall[318 * 4] = 7;
        let tall = fit(Micros(1), 1, 320, &tall).expect("tall");
        assert_eq!((tall.width(), tall.height()), (1, MAX_EDGE));
        assert_eq!(tall.rgba()[0], 9);
        assert_eq!(tall.rgba()[159 * 4], 7);
    }

    #[test]
    fn a_bad_picture_is_refused() {
        assert_eq!(
            fit(Micros(0), 0, 1, &[]).expect_err("empty"),
            EngineError::Window
        );
        assert_eq!(
            fit(Micros(0), 1, 1, &[]).expect_err("short"),
            EngineError::Hostile {
                declared: 0,
                limit: 4,
            }
        );
        assert_eq!(
            fit(Micros(0), 2, 2, &[0; 4]).expect_err("mismatch"),
            EngineError::Hostile {
                declared: 4,
                limit: 16,
            }
        );
        assert!(matches!(
            fit(Micros(0), u32::MAX, u32::MAX, &[]),
            Err(EngineError::Hostile { limit: 0, .. })
        ));
    }

    #[test]
    fn the_strip_drops_the_farthest_cell() {
        let mut strip = Filmstrip::new();
        for time in 0..MAX_STRIP {
            strip.insert(pixel_cell(i64::try_from(time).expect("time"), 1));
        }
        assert_eq!(strip.cells.len(), MAX_STRIP);
        strip.insert(pixel_cell(1_000, 3));
        assert_eq!(strip.cells.len(), MAX_STRIP);
        assert!(strip.at(0).is_none());
        assert_eq!(strip.at(1).expect("kept").time(), Micros(1));
        assert_eq!(strip.at(1_000).expect("new").rgba()[0], 3);
        assert!(strip.cells.len() <= MAX_STRIP);
    }
}
