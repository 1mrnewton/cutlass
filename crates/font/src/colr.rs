//! Color glyphs from `COLR` and `CPAL`: outlines filled with palette colors
//! and gradients, stacked, moved, and blended.
//!
//! This reads the tables into one tree of paints for a glyph. It draws
//! nothing; the caller fills the outlines. Version 0 is a list of layers,
//! each a glyph and a palette color. Version 1 is a graph of paint tables,
//! read here into a tree with every transform as one matrix and every
//! color looked up in palette 0. The variable forms of a paint read as
//! their default values. A graph that loops, nests too deep, or names too
//! many paints is a bad table.

use crate::{Directory, FontError, bytes_at, i16_at, u16_at, u32_at};

/// Paints nested inside one another, at most.
const MAX_DEPTH: usize = 32;
/// Paints in one glyph's tree, at most.
const MAX_PAINTS: usize = 4_096;
/// Stops on one color line, at most.
const MAX_STOPS: usize = 256;
/// The palette index that means the color of the text.
const TEXT_COLOR: u16 = 0xFFFF;

/// One color of a paint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red, green, and blue as the palette stores them, not premultiplied.
    /// `None` is the color of the text the glyph is set in.
    pub rgb: Option<[u8; 3]>,
    /// Opacity from 0 to 1: the palette's alpha times the paint's.
    pub alpha: f32,
}

/// One stop of a gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    pub offset: f32,
    pub color: Color,
}

/// How a gradient continues outside its stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extend {
    /// The nearest stop's color.
    Pad,
    /// The stops again, in the same order.
    Repeat,
    /// The stops again, back and forth.
    Reflect,
}

/// The colors along a gradient. The stops are in offset order.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorLine {
    pub extend: Extend,
    pub stops: Vec<ColorStop>,
}

/// A map of the plane: `x' = xx·x + xy·y + dx` and `y' = yx·x + yy·y + dy`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub xx: f32,
    pub yx: f32,
    pub xy: f32,
    pub yy: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Affine {
    /// The map that changes nothing.
    pub const IDENTITY: Self = Self {
        xx: 1.0,
        yx: 0.0,
        xy: 0.0,
        yy: 1.0,
        dx: 0.0,
        dy: 0.0,
    };

    /// Where the map sends a point.
    #[must_use]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.xx * x + self.xy * y + self.dx,
            self.yx * x + self.yy * y + self.dy,
        )
    }

    /// The map that applies `inner` first and then this one.
    #[must_use]
    pub fn then_after(&self, inner: &Self) -> Self {
        let (dx, dy) = self.apply(inner.dx, inner.dy);
        Self {
            xx: self.xx * inner.xx + self.xy * inner.yx,
            yx: self.yx * inner.xx + self.yy * inner.yx,
            xy: self.xx * inner.xy + self.xy * inner.yy,
            yy: self.yx * inner.xy + self.yy * inner.yy,
            dx,
            dy,
        }
    }

    /// The same linear part, turned about `(x, y)` instead of the origin.
    fn about(self, x: f32, y: f32) -> Self {
        let (moved_x, moved_y) = self.apply(x, y);
        Self {
            dx: self.dx + x - moved_x,
            dy: self.dy + y - moved_y,
            ..self
        }
    }
}

/// How a source paint is blended onto a backdrop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeMode {
    Clear,
    Src,
    Dest,
    SrcOver,
    DestOver,
    SrcIn,
    DestIn,
    SrcOut,
    DestOut,
    SrcAtop,
    DestAtop,
    Xor,
    Plus,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Multiply,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl CompositeMode {
    fn read(value: u8) -> Result<Self, FontError> {
        Ok(match value {
            0 => Self::Clear,
            1 => Self::Src,
            2 => Self::Dest,
            3 => Self::SrcOver,
            4 => Self::DestOver,
            5 => Self::SrcIn,
            6 => Self::DestIn,
            7 => Self::SrcOut,
            8 => Self::DestOut,
            9 => Self::SrcAtop,
            10 => Self::DestAtop,
            11 => Self::Xor,
            12 => Self::Plus,
            13 => Self::Screen,
            14 => Self::Overlay,
            15 => Self::Darken,
            16 => Self::Lighten,
            17 => Self::ColorDodge,
            18 => Self::ColorBurn,
            19 => Self::HardLight,
            20 => Self::SoftLight,
            21 => Self::Difference,
            22 => Self::Exclusion,
            23 => Self::Multiply,
            24 => Self::Hue,
            25 => Self::Saturation,
            26 => Self::Color,
            27 => Self::Luminosity,
            _ => return Err(FontError::Unsupported),
        })
    }
}

/// The box a color glyph is drawn inside, in font design units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// One paint of a color glyph. Points and lengths are in font design units,
/// with y up.
#[derive(Clone, Debug, PartialEq)]
pub enum Paint {
    /// Paints drawn one over another, the first at the bottom.
    Layers(Vec<Paint>),
    /// One color everywhere.
    Solid(Color),
    /// Colors along the line from `p0` to `p1`. `p2` turns the gradient:
    /// its stripes run along the line from `p0` to `p2`.
    LinearGradient {
        line: ColorLine,
        p0: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
    },
    /// Colors from one circle to another.
    RadialGradient {
        line: ColorLine,
        c0: [f32; 2],
        r0: f32,
        c1: [f32; 2],
        r1: f32,
    },
    /// Colors around a center, from `start` to `end` in degrees
    /// counter-clockwise from the positive x axis.
    SweepGradient {
        line: ColorLine,
        center: [f32; 2],
        start: f32,
        end: f32,
    },
    /// `paint`, inside the outline of `glyph_id`.
    Glyph { glyph_id: u16, paint: Box<Paint> },
    /// Another color glyph of the face, inside its own clip box.
    ColorGlyph(Box<ColorGlyph>),
    /// `paint`, with its outlines and gradients moved by `transform`.
    Transform {
        transform: Affine,
        paint: Box<Paint>,
    },
    /// `source` blended onto `backdrop`.
    Composite {
        source: Box<Paint>,
        mode: CompositeMode,
        backdrop: Box<Paint>,
    },
}

/// A glyph drawn in color.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorGlyph {
    pub paint: Paint,
    /// Nothing is drawn outside this box. A glyph without one is bounded by
    /// the outlines it fills.
    pub clip: Option<ClipBox>,
}

/// The `COLR` table of a face and the palette its colors come from.
pub(crate) struct Colr<'a> {
    table: &'a [u8],
    /// Version 0 base glyph records: where they start, and how many.
    bases: (usize, usize),
    /// Version 0 layer records.
    layers: (usize, usize),
    /// Version 1 lists: where each starts, when the table has it.
    base_list: Option<usize>,
    layer_list: Option<usize>,
    clip_list: Option<usize>,
    palette: Palette<'a>,
}

/// Palette 0 of `CPAL`: color records of blue, green, red, alpha.
struct Palette<'a> {
    records: &'a [u8],
}

impl<'a> Palette<'a> {
    fn empty() -> Self {
        Self { records: &[] }
    }

    fn parse(table: &'a [u8]) -> Result<Self, FontError> {
        let entries = usize::from(u16_at(table, 2)?);
        let palettes = usize::from(u16_at(table, 4)?);
        let records = usize::from(u16_at(table, 6)?);
        let records_at = offset32(table, 8)?;
        if palettes == 0 || entries == 0 {
            return Ok(Self::empty());
        }
        let first = usize::from(u16_at(table, 12)?);
        if first + entries > records {
            return Err(FontError::BadTable);
        }
        let all = bytes_at(table, records_at, records * 4)?;
        Ok(Self {
            records: bytes_at(all, first * 4, entries * 4)?,
        })
    }

    fn color(&self, index: u16, alpha: f32) -> Result<Color, FontError> {
        let alpha = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if index == TEXT_COLOR {
            return Ok(Color { rgb: None, alpha });
        }
        let Some(record) = self.records.as_chunks::<4>().0.get(usize::from(index)) else {
            return Err(FontError::BadTable);
        };
        let [blue, green, red, opacity] = *record;
        Ok(Color {
            rgb: Some([red, green, blue]),
            alpha: alpha * f32::from(opacity) / 255.0,
        })
    }
}

impl<'a> Colr<'a> {
    /// The face's color table, when it has one this reader accepts. A table
    /// of a later version is passed over; a torn one is an error.
    pub(crate) fn parse(directory: &Directory<'a>) -> Result<Option<Self>, FontError> {
        let table = match directory.table(b"COLR") {
            Ok([]) | Err(FontError::MissingTable) => return Ok(None),
            Ok(table) => table,
            Err(error) => return Err(error),
        };
        let version = u16_at(table, 0)?;
        if version > 1 {
            return Ok(None);
        }
        let bases = (offset32(table, 4)?, usize::from(u16_at(table, 2)?));
        let layers = (offset32(table, 8)?, usize::from(u16_at(table, 12)?));
        let _records = bytes_at(table, bases.0, bases.1 * 6)?;
        let _records = bytes_at(table, layers.0, layers.1 * 4)?;
        let list = |at: usize| -> Result<Option<usize>, FontError> {
            if version == 0 {
                return Ok(None);
            }
            let offset = offset32(table, at)?;
            if offset == 0 {
                return Ok(None);
            }
            if offset >= table.len() {
                return Err(FontError::Truncated);
            }
            Ok(Some(offset))
        };
        let palette = match directory.table(b"CPAL") {
            Ok([]) | Err(FontError::MissingTable) => Palette::empty(),
            Ok(table) => Palette::parse(table)?,
            Err(error) => return Err(error),
        };
        Ok(Some(Self {
            table,
            bases,
            layers,
            base_list: list(14)?,
            layer_list: list(18)?,
            clip_list: list(22)?,
            palette,
        }))
    }

    /// The color glyph for `glyph_id`, when the table has one.
    pub(crate) fn glyph(
        &self,
        glyph_id: u16,
        num_glyphs: u16,
    ) -> Result<Option<ColorGlyph>, FontError> {
        let mut walk = Walk {
            colr: self,
            num_glyphs,
            paints: 0,
            open: Vec::new(),
        };
        walk.color_glyph(glyph_id)
    }

    /// Where the version 1 paint of `glyph_id` starts.
    fn base_paint(&self, glyph_id: u16) -> Result<Option<usize>, FontError> {
        let Some(list) = self.base_list else {
            return Ok(None);
        };
        let count = offset32(self.table, list)?;
        let records = bytes_at(self.table, list + 4, count.checked_mul(6).ok_or(BAD)?)?;
        let Some(record) = find(records, 6, glyph_id)? else {
            return Ok(None);
        };
        let offset = offset32(records, record + 2)?;
        Ok(Some(list.checked_add(offset).ok_or(BAD)?))
    }

    /// The version 0 layers of `glyph_id`: where they start, and how many.
    fn base_layers(&self, glyph_id: u16) -> Result<Option<(usize, usize)>, FontError> {
        let records = bytes_at(self.table, self.bases.0, self.bases.1 * 6)?;
        let Some(record) = find(records, 6, glyph_id)? else {
            return Ok(None);
        };
        let first = usize::from(u16_at(records, record + 2)?);
        let count = usize::from(u16_at(records, record + 4)?);
        if first + count > self.layers.1 {
            return Err(BAD);
        }
        Ok(Some((self.layers.0 + first * 4, count)))
    }

    fn clip(&self, glyph_id: u16) -> Result<Option<ClipBox>, FontError> {
        let Some(list) = self.clip_list else {
            return Ok(None);
        };
        let count = offset32(self.table, list + 1)?;
        let records = bytes_at(self.table, list + 5, count.checked_mul(7).ok_or(BAD)?)?;
        let (mut low, mut high) = (0, count);
        while low < high {
            let mid = low + (high - low) / 2;
            let at = mid * 7;
            if glyph_id < u16_at(records, at)? {
                high = mid;
            } else if glyph_id > u16_at(records, at + 2)? {
                low = mid + 1;
            } else {
                let found = list.checked_add(u24_at(records, at + 4)?).ok_or(BAD)?;
                let values = bytes_at(self.table, found, 9)?;
                return Ok(Some(ClipBox {
                    x_min: f32::from(i16_at(values, 1)?),
                    y_min: f32::from(i16_at(values, 3)?),
                    x_max: f32::from(i16_at(values, 5)?),
                    y_max: f32::from(i16_at(values, 7)?),
                }));
            }
        }
        Ok(None)
    }
}

const BAD: FontError = FontError::BadTable;

/// The record of `glyph_id` among `size`-byte records that start with a
/// glyph id and are in glyph order.
fn find(records: &[u8], size: usize, glyph_id: u16) -> Result<Option<usize>, FontError> {
    let (mut low, mut high) = (0, records.len() / size);
    while low < high {
        let mid = low + (high - low) / 2;
        match u16_at(records, mid * size)?.cmp(&glyph_id) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => return Ok(Some(mid * size)),
        }
    }
    Ok(None)
}

/// One reading of a glyph's paints: how many it has read, and the ones it
/// is inside of, by where they start.
struct Walk<'c, 'a> {
    colr: &'c Colr<'a>,
    num_glyphs: u16,
    paints: usize,
    open: Vec<usize>,
}

impl Walk<'_, '_> {
    fn color_glyph(&mut self, glyph_id: u16) -> Result<Option<ColorGlyph>, FontError> {
        if glyph_id >= self.num_glyphs {
            return Err(FontError::UnknownGlyph);
        }
        if let Some(at) = self.colr.base_paint(glyph_id)? {
            return Ok(Some(ColorGlyph {
                paint: self.paint(at)?,
                clip: self.colr.clip(glyph_id)?,
            }));
        }
        let Some((at, count)) = self.colr.base_layers(glyph_id)? else {
            return Ok(None);
        };
        let records = bytes_at(self.colr.table, at, count * 4)?;
        let mut layers = Vec::with_capacity(count.min(MAX_PAINTS));
        for record in records.as_chunks::<4>().0 {
            self.count()?;
            let layer = u16::from_be_bytes([record[0], record[1]]);
            let index = u16::from_be_bytes([record[2], record[3]]);
            layers.push(Paint::Glyph {
                glyph_id: self.known(layer)?,
                paint: Box::new(Paint::Solid(self.colr.palette.color(index, 1.0)?)),
            });
        }
        Ok(Some(ColorGlyph {
            paint: Paint::Layers(layers),
            clip: None,
        }))
    }

    fn count(&mut self) -> Result<(), FontError> {
        self.paints += 1;
        if self.paints > MAX_PAINTS {
            return Err(BAD);
        }
        Ok(())
    }

    fn known(&self, glyph_id: u16) -> Result<u16, FontError> {
        if glyph_id >= self.num_glyphs {
            return Err(BAD);
        }
        Ok(glyph_id)
    }

    /// The paint that starts at `at`. A paint that is inside itself is a
    /// loop, and the table is bad.
    fn paint(&mut self, at: usize) -> Result<Paint, FontError> {
        if self.open.len() >= MAX_DEPTH || self.open.contains(&at) {
            return Err(BAD);
        }
        self.count()?;
        self.open.push(at);
        let paint = self.read(at);
        self.open.pop();
        paint
    }

    /// The paint a 24-bit offset at `field` of the paint at `at` points to.
    fn child(&mut self, at: usize, field: usize) -> Result<Box<Paint>, FontError> {
        let offset = u24_at(self.colr.table, at + field)?;
        Ok(Box::new(self.paint(at.checked_add(offset).ok_or(BAD)?)?))
    }

    fn read(&mut self, at: usize) -> Result<Paint, FontError> {
        let table = self.colr.table;
        let data = table.get(at..).ok_or(FontError::Truncated)?;
        let format = *data.first().ok_or(FontError::Truncated)?;
        let moved = |walk: &mut Self, transform: Affine| -> Result<Paint, FontError> {
            Ok(Paint::Transform {
                transform,
                paint: walk.child(at, 1)?,
            })
        };
        // The center of a transform about a point, as two values at `field`.
        let center = |field: usize| -> Result<(f32, f32), FontError> {
            Ok((
                f32::from(i16_at(data, field)?),
                f32::from(i16_at(data, field + 2)?),
            ))
        };
        match format {
            1 => self.layers(data),
            2 | 3 => {
                let color = self
                    .colr
                    .palette
                    .color(u16_at(data, 1)?, f2dot14(data, 3)?)?;
                Ok(Paint::Solid(color))
            }
            4 | 5 => {
                let point = |field: usize| center(field).map(|(x, y)| [x, y]);
                Ok(Paint::LinearGradient {
                    line: self.line(at, format == 5)?,
                    p0: point(4)?,
                    p1: point(8)?,
                    p2: point(12)?,
                })
            }
            6 | 7 => Ok(Paint::RadialGradient {
                line: self.line(at, format == 7)?,
                c0: center(4).map(|(x, y)| [x, y])?,
                r0: f32::from(u16_at(data, 8)?),
                c1: center(10).map(|(x, y)| [x, y])?,
                r1: f32::from(u16_at(data, 14)?),
            }),
            8 | 9 => Ok(Paint::SweepGradient {
                line: self.line(at, format == 9)?,
                center: center(4).map(|(x, y)| [x, y])?,
                // The stored angles are half turns less one.
                start: (f2dot14(data, 8)? + 1.0) * 180.0,
                end: (f2dot14(data, 10)? + 1.0) * 180.0,
            }),
            10 => Ok(Paint::Glyph {
                glyph_id: self.known(u16_at(data, 4)?)?,
                paint: self.child(at, 1)?,
            }),
            11 => {
                let glyph_id = self.known(u16_at(data, 1)?)?;
                // A glyph named here that has no paint of its own is a bad table.
                let glyph = self.color_glyph(glyph_id)?.ok_or(BAD)?;
                Ok(Paint::ColorGlyph(Box::new(glyph)))
            }
            12 | 13 => {
                let values = table
                    .get(at.checked_add(u24_at(data, 4)?).ok_or(BAD)?..)
                    .ok_or(FontError::Truncated)?;
                let transform = Affine {
                    xx: fixed(values, 0)?,
                    yx: fixed(values, 4)?,
                    xy: fixed(values, 8)?,
                    yy: fixed(values, 12)?,
                    dx: fixed(values, 16)?,
                    dy: fixed(values, 20)?,
                };
                moved(self, transform)
            }
            14 | 15 => {
                let (dx, dy) = center(4)?;
                moved(
                    self,
                    Affine {
                        dx,
                        dy,
                        ..Affine::IDENTITY
                    },
                )
            }
            16..=23 => {
                let uniform = format >= 20;
                let x = f2dot14(data, 4)?;
                let y = if uniform { x } else { f2dot14(data, 6)? };
                let scale = Affine {
                    xx: x,
                    yy: y,
                    ..Affine::IDENTITY
                };
                match format {
                    18 | 19 => {
                        let (x, y) = center(8)?;
                        moved(self, scale.about(x, y))
                    }
                    22 | 23 => {
                        let (x, y) = center(6)?;
                        moved(self, scale.about(x, y))
                    }
                    _ => moved(self, scale),
                }
            }
            24..=27 => {
                // Half turns, counter-clockwise.
                let (sin, cos) = (f2dot14(data, 4)? * std::f32::consts::PI).sin_cos();
                let turn = Affine {
                    xx: cos,
                    yx: sin,
                    xy: -sin,
                    yy: cos,
                    dx: 0.0,
                    dy: 0.0,
                };
                if format >= 26 {
                    let (x, y) = center(6)?;
                    moved(self, turn.about(x, y))
                } else {
                    moved(self, turn)
                }
            }
            28..=31 => {
                let lean = Affine {
                    yx: (f2dot14(data, 6)? * std::f32::consts::PI).tan(),
                    xy: -(f2dot14(data, 4)? * std::f32::consts::PI).tan(),
                    ..Affine::IDENTITY
                };
                if format >= 30 {
                    let (x, y) = center(8)?;
                    moved(self, lean.about(x, y))
                } else {
                    moved(self, lean)
                }
            }
            32 => Ok(Paint::Composite {
                source: self.child(at, 1)?,
                mode: CompositeMode::read(*data.get(4).ok_or(FontError::Truncated)?)?,
                backdrop: self.child(at, 5)?,
            }),
            _ => Err(FontError::Unsupported),
        }
    }

    /// Format 1: a run of the layer list.
    fn layers(&mut self, data: &[u8]) -> Result<Paint, FontError> {
        let count = usize::from(*data.get(1).ok_or(FontError::Truncated)?);
        let first = offset32(data, 2)?;
        let list = self.colr.layer_list.ok_or(BAD)?;
        let total = offset32(self.colr.table, list)?;
        if first.checked_add(count).ok_or(BAD)? > total {
            return Err(BAD);
        }
        let offsets = bytes_at(self.colr.table, list + 4 + first * 4, count * 4)?;
        let mut layers = Vec::with_capacity(count);
        for index in 0..count {
            let offset = offset32(offsets, index * 4)?;
            layers.push(self.paint(list.checked_add(offset).ok_or(BAD)?)?);
        }
        Ok(Paint::Layers(layers))
    }

    /// The color line of the gradient at `at`. A variable line's stops are
    /// four bytes longer.
    fn line(&self, at: usize, variable: bool) -> Result<ColorLine, FontError> {
        let table = self.colr.table;
        let line = table
            .get(at.checked_add(u24_at(table, at + 1)?).ok_or(BAD)?..)
            .ok_or(FontError::Truncated)?;
        let extend = match *line.first().ok_or(FontError::Truncated)? {
            1 => Extend::Repeat,
            2 => Extend::Reflect,
            _ => Extend::Pad,
        };
        let count = usize::from(u16_at(line, 1)?);
        if count > MAX_STOPS {
            return Err(BAD);
        }
        let size = if variable { 10 } else { 6 };
        let records = bytes_at(line, 3, count * size)?;
        let mut stops = Vec::with_capacity(count);
        for index in 0..count {
            let record = index * size;
            stops.push(ColorStop {
                offset: f2dot14(records, record)?,
                color: self
                    .colr
                    .palette
                    .color(u16_at(records, record + 2)?, f2dot14(records, record + 4)?)?,
            });
        }
        stops.sort_by(|left, right| left.offset.total_cmp(&right.offset));
        Ok(ColorLine { extend, stops })
    }
}

fn offset32(data: &[u8], at: usize) -> Result<usize, FontError> {
    usize::try_from(u32_at(data, at)?).map_err(|_| BAD)
}

fn u24_at(data: &[u8], at: usize) -> Result<usize, FontError> {
    let bytes = bytes_at(data, at, 3)?;
    Ok(usize::from(bytes[0]) << 16 | usize::from(bytes[1]) << 8 | usize::from(bytes[2]))
}

/// A signed value with 14 fraction bits.
fn f2dot14(data: &[u8], at: usize) -> Result<f32, FontError> {
    Ok(f32::from(i16_at(data, at)?) / 16_384.0)
}

/// A signed value with 16 fraction bits.
fn fixed(data: &[u8], at: usize) -> Result<f32, FontError> {
    let bytes = bytes_at(data, at, 4)?;
    let whole = i16::from_be_bytes([bytes[0], bytes[1]]);
    let fraction = u16::from_be_bytes([bytes[2], bytes[3]]);
    Ok(f32::from(whole) + f32::from(fraction) / 65_536.0)
}
