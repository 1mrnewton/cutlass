//! `GPOS`: where a font moves glyphs, and how far each one advances.

use crate::layout::{ClassDef, Coverage, SequenceContext, sub, sub_at};
use crate::{FontError, i16_at, u16_at};

/// How one glyph is moved, in font design units. Y grows upward.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Adjust {
    pub x: i16,
    pub y: i16,
    pub x_advance: i16,
    pub y_advance: i16,
}

/// A point on a glyph that another glyph's point is set on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Anchor {
    pub x: i16,
    pub y: i16,
}

/// One subtable of a positioning lookup.
#[derive(Clone, Copy, Debug)]
pub enum Positioning<'a> {
    /// One glyph moved.
    Single(SingleAdjust<'a>),
    /// A pair of glyphs moved: kerning.
    Pair(PairAdjust<'a>),
    /// A glyph's entry joined to the exit of the one before.
    Cursive(Cursive<'a>),
    /// A mark set on a base glyph.
    MarkToBase(MarkAttach<'a>),
    /// A mark set on one part of a ligature.
    MarkToLigature(MarkAttach<'a>),
    /// A mark set on another mark.
    MarkToMark(MarkAttach<'a>),
    /// Other lookups, where a run of glyphs matches.
    Context(SequenceContext<'a>),
}

impl<'a> Positioning<'a> {
    /// Read a subtable of lookup type `kind`, 1 to 8.
    pub fn read(kind: u16, data: &'a [u8]) -> Result<Self, FontError> {
        match kind {
            1 => Ok(Self::Single(SingleAdjust::read(data)?)),
            2 => Ok(Self::Pair(PairAdjust::read(data)?)),
            3 => Ok(Self::Cursive(Cursive::read(data)?)),
            4 => Ok(Self::MarkToBase(MarkAttach::read(data)?)),
            5 => Ok(Self::MarkToLigature(MarkAttach::read(data)?)),
            6 => Ok(Self::MarkToMark(MarkAttach::read(data)?)),
            7 => Ok(Self::Context(SequenceContext::read(data, false)?)),
            8 => Ok(Self::Context(SequenceContext::read(data, true)?)),
            _ => Err(FontError::BadTable),
        }
    }
}

/// Bytes one value record of `format` takes.
fn value_size(format: u16) -> usize {
    (format & 0xFF).count_ones() as usize * 2
}

/// The value record of `format` at `at`. Device and variation tables are
/// stepped over.
fn value(data: &[u8], mut at: usize, format: u16) -> Result<Adjust, FontError> {
    let mut adjust = Adjust::default();
    for (bit, field) in [
        (1, &mut adjust.x),
        (2, &mut adjust.y),
        (4, &mut adjust.x_advance),
        (8, &mut adjust.y_advance),
    ] {
        if format & bit != 0 {
            *field = i16_at(data, at)?;
            at += 2;
        }
    }
    Ok(adjust)
}

/// The anchor a 16-bit offset stored at `at` points to; zero is no anchor.
fn anchor(data: &[u8], at: usize) -> Result<Option<Anchor>, FontError> {
    let Some(table) = sub_at(data, at)? else {
        return Ok(None);
    };
    if !(1..=3).contains(&u16_at(table, 0)?) {
        return Err(FontError::BadTable);
    }
    Ok(Some(Anchor {
        x: i16_at(table, 2)?,
        y: i16_at(table, 4)?,
    }))
}

#[derive(Clone, Copy, Debug)]
pub struct SingleAdjust<'a> {
    data: &'a [u8],
    coverage: Coverage<'a>,
    format: u16,
    /// One value for every covered glyph.
    shared: bool,
}

impl<'a> SingleAdjust<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        let shared = match u16_at(data, 0)? {
            1 => true,
            2 => false,
            _ => return Err(FontError::BadTable),
        };
        Ok(Self {
            data,
            coverage: Coverage::at(data, 2)?,
            format: u16_at(data, 4)?,
            shared,
        })
    }

    /// How `glyph` is moved, when this subtable covers it.
    pub fn adjust(&self, glyph: u16) -> Result<Option<Adjust>, FontError> {
        let Some(index) = self.coverage.index(glyph) else {
            return Ok(None);
        };
        if self.shared {
            return value(self.data, 6, self.format).map(Some);
        }
        if index >= u16_at(self.data, 6)? {
            return Err(FontError::BadTable);
        }
        let at = 8 + usize::from(index) * value_size(self.format);
        value(self.data, at, self.format).map(Some)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PairAdjust<'a> {
    data: &'a [u8],
    coverage: Coverage<'a>,
    formats: (u16, u16),
    /// The classes of the first and second glyph, for a table of classes.
    classes: Option<(ClassDef<'a>, ClassDef<'a>)>,
}

impl<'a> PairAdjust<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        let classes = match u16_at(data, 0)? {
            1 => None,
            2 => Some((ClassDef::at(data, 8)?, ClassDef::at(data, 10)?)),
            _ => return Err(FontError::BadTable),
        };
        Ok(Self {
            data,
            coverage: Coverage::at(data, 2)?,
            formats: (u16_at(data, 4)?, u16_at(data, 6)?),
            classes,
        })
    }

    /// Whether the pair's second glyph is moved too, so the pair after it
    /// starts from the glyph that follows.
    #[must_use]
    pub fn moves_second(&self) -> bool {
        self.formats.1 != 0
    }

    /// How `first` and `second` are moved, when this subtable has the pair.
    pub fn adjust(&self, first: u16, second: u16) -> Result<Option<(Adjust, Adjust)>, FontError> {
        let Some(index) = self.coverage.index(first) else {
            return Ok(None);
        };
        let (one, two) = (value_size(self.formats.0), value_size(self.formats.1));
        let read = |data: &[u8], at: usize| {
            Ok(Some((
                value(data, at, self.formats.0)?,
                value(data, at + one, self.formats.1)?,
            )))
        };
        if let Some((first_classes, second_classes)) = &self.classes {
            let (rows, columns) = (u16_at(self.data, 12)?, u16_at(self.data, 14)?);
            let (row, column) = (first_classes.class(first), second_classes.class(second));
            if row >= rows || column >= columns {
                return Ok(None);
            }
            let record = usize::from(row) * usize::from(columns) + usize::from(column);
            return read(self.data, 16 + record * (one + two));
        }
        if index >= u16_at(self.data, 8)? {
            return Err(FontError::BadTable);
        }
        let set = sub_at(self.data, 10 + usize::from(index) * 2)?.ok_or(FontError::BadTable)?;
        let size = 2 + one + two;
        let (mut low, mut high) = (0, usize::from(u16_at(set, 0)?));
        while low < high {
            let mid = low + (high - low) / 2;
            let at = 2 + mid * size;
            let glyph = u16_at(set, at)?;
            if glyph < second {
                low = mid + 1;
            } else if glyph > second {
                high = mid;
            } else {
                return read(set, at + 2);
            }
        }
        Ok(None)
    }
}

/// Where a glyph of a joined script meets its neighbours.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Joins {
    /// Where the glyph before it arrives.
    pub entry: Option<Anchor>,
    /// Where the glyph after it leaves from.
    pub exit: Option<Anchor>,
}

#[derive(Clone, Copy, Debug)]
pub struct Cursive<'a> {
    data: &'a [u8],
    coverage: Coverage<'a>,
}

impl<'a> Cursive<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            data,
            coverage: Coverage::at(data, 2)?,
        })
    }

    /// Where `glyph` is entered and where it is left, when covered.
    pub fn anchors(&self, glyph: u16) -> Result<Option<Joins>, FontError> {
        let Some(index) = self.coverage.index(glyph) else {
            return Ok(None);
        };
        if index >= u16_at(self.data, 4)? {
            return Err(FontError::BadTable);
        }
        let at = 6 + usize::from(index) * 4;
        Ok(Some(Joins {
            entry: anchor(self.data, at)?,
            exit: anchor(self.data, at + 2)?,
        }))
    }
}

/// Marks and what they sit on: bases, ligatures, or other marks. Each mark
/// has a class, and each thing it sits on an anchor for each class.
#[derive(Clone, Copy, Debug)]
pub struct MarkAttach<'a> {
    marks: Coverage<'a>,
    bases: Coverage<'a>,
    classes: u16,
    mark_array: &'a [u8],
    base_array: &'a [u8],
}

impl<'a> MarkAttach<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            marks: Coverage::at(data, 2)?,
            bases: Coverage::at(data, 4)?,
            classes: u16_at(data, 6)?,
            mark_array: sub_at(data, 8)?.ok_or(FontError::BadTable)?,
            base_array: sub_at(data, 10)?.ok_or(FontError::BadTable)?,
        })
    }

    /// Whether `glyph` is a mark this subtable places.
    #[must_use]
    pub fn places(&self, glyph: u16) -> bool {
        self.marks.covers(glyph)
    }

    /// Whether `glyph` is something this subtable sets marks on.
    #[must_use]
    pub fn holds(&self, glyph: u16) -> bool {
        self.bases.covers(glyph)
    }

    /// The mark's class and its own anchor.
    fn mark(&self, mark: u16) -> Result<Option<(u16, Anchor)>, FontError> {
        let Some(index) = self.marks.index(mark) else {
            return Ok(None);
        };
        if index >= u16_at(self.mark_array, 0)? {
            return Err(FontError::BadTable);
        }
        let at = 2 + usize::from(index) * 4;
        let class = u16_at(self.mark_array, at)?;
        if class >= self.classes {
            return Err(FontError::BadTable);
        }
        Ok(anchor(self.mark_array, at + 2)?.map(|anchor| (class, anchor)))
    }

    /// The anchors that set `mark` on `base`, a base glyph or another
    /// mark: the mark's own, then the base's.
    pub fn anchors(&self, mark: u16, base: u16) -> Result<Option<(Anchor, Anchor)>, FontError> {
        let (Some((class, on_mark)), Some(index)) = (self.mark(mark)?, self.bases.index(base))
        else {
            return Ok(None);
        };
        if index >= u16_at(self.base_array, 0)? {
            return Err(FontError::BadTable);
        }
        let at = 2 + (usize::from(index) * usize::from(self.classes) + usize::from(class)) * 2;
        Ok(anchor(self.base_array, at)?.map(|on_base| (on_mark, on_base)))
    }

    /// How many parts `ligature` has anchors for, when this subtable
    /// sets marks on it.
    pub fn ligature_parts(&self, ligature: u16) -> Result<Option<usize>, FontError> {
        let Some(index) = self.bases.index(ligature) else {
            return Ok(None);
        };
        if index >= u16_at(self.base_array, 0)? {
            return Err(FontError::BadTable);
        }
        let attach = sub(
            self.base_array,
            usize::from(u16_at(self.base_array, 2 + usize::from(index) * 2)?),
        )?
        .ok_or(FontError::BadTable)?;
        Ok(Some(usize::from(u16_at(attach, 0)?)))
    }

    /// The anchors that set `mark` on part `component` of `ligature`.
    pub fn ligature_anchors(
        &self,
        mark: u16,
        ligature: u16,
        component: usize,
    ) -> Result<Option<(Anchor, Anchor)>, FontError> {
        let (Some((class, on_mark)), Some(index)) = (self.mark(mark)?, self.bases.index(ligature))
        else {
            return Ok(None);
        };
        if index >= u16_at(self.base_array, 0)? {
            return Err(FontError::BadTable);
        }
        let attach = sub(
            self.base_array,
            usize::from(u16_at(self.base_array, 2 + usize::from(index) * 2)?),
        )?
        .ok_or(FontError::BadTable)?;
        if component >= usize::from(u16_at(attach, 0)?) {
            return Ok(None);
        }
        let at = 2 + (component * usize::from(self.classes) + usize::from(class)) * 2;
        Ok(anchor(attach, at)?.map(|on_base| (on_mark, on_base)))
    }
}
