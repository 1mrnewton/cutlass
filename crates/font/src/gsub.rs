//! `GSUB`: the glyphs a font puts in place of others.

use crate::layout::{Coverage, ReverseChain, SequenceContext, U16s, sub, sub_at};
use crate::{FontError, u16_at};

/// One subtable of a substitution lookup.
#[derive(Clone, Copy, Debug)]
pub enum Substitution<'a> {
    /// One glyph for one.
    Single(Single<'a>),
    /// Several glyphs for one.
    Multiple(Sequences<'a>),
    /// One of several glyphs for one; the first is the one to use.
    Alternate(Sequences<'a>),
    /// One glyph for several.
    Ligature(Ligatures<'a>),
    /// Other lookups, where a run of glyphs matches.
    Context(SequenceContext<'a>),
    /// One glyph for one, told by its neighbours, walked from the end.
    Reverse(ReverseChain<'a>),
}

impl<'a> Substitution<'a> {
    /// Read a subtable of lookup type `kind`, 1 to 8 without 7.
    pub fn read(kind: u16, data: &'a [u8]) -> Result<Self, FontError> {
        match kind {
            1 => Ok(Self::Single(Single::read(data)?)),
            2 => Ok(Self::Multiple(Sequences::read(data)?)),
            3 => Ok(Self::Alternate(Sequences::read(data)?)),
            4 => Ok(Self::Ligature(Ligatures::read(data)?)),
            5 => Ok(Self::Context(SequenceContext::read(data, false)?)),
            6 => Ok(Self::Context(SequenceContext::read(data, true)?)),
            8 => Ok(Self::Reverse(ReverseChain::read(data)?)),
            _ => Err(FontError::BadTable),
        }
    }
}

/// A single substitution: a shift of the glyph id, or a list of glyphs.
#[derive(Clone, Copy, Debug)]
pub struct Single<'a> {
    coverage: Coverage<'a>,
    delta: u16,
    substitutes: Option<U16s<'a>>,
}

impl<'a> Single<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        let coverage = Coverage::at(data, 2)?;
        match u16_at(data, 0)? {
            1 => Ok(Self {
                coverage,
                delta: u16_at(data, 4)?,
                substitutes: None,
            }),
            2 => Ok(Self {
                coverage,
                delta: 0,
                substitutes: Some(U16s::read(data, 6, usize::from(u16_at(data, 4)?))?),
            }),
            _ => Err(FontError::BadTable),
        }
    }

    /// The glyph that replaces `glyph`, when this subtable covers it.
    pub fn substitute(&self, glyph: u16) -> Result<Option<u16>, FontError> {
        let Some(index) = self.coverage.index(glyph) else {
            return Ok(None);
        };
        match self.substitutes {
            None => Ok(Some(glyph.wrapping_add(self.delta))),
            Some(list) => list
                .get(usize::from(index))
                .map(Some)
                .ok_or(FontError::BadTable),
        }
    }
}

/// A list of glyphs for each covered glyph: what a multiple substitution
/// writes, or the alternates to choose from.
#[derive(Clone, Copy, Debug)]
pub struct Sequences<'a> {
    data: &'a [u8],
    coverage: Coverage<'a>,
}

impl<'a> Sequences<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            data,
            coverage: Coverage::at(data, 2)?,
        })
    }

    /// The glyphs listed for `glyph`, when this subtable covers it.
    pub fn glyphs(&self, glyph: u16) -> Result<Option<U16s<'a>>, FontError> {
        let Some(index) = self.coverage.index(glyph) else {
            return Ok(None);
        };
        if index >= u16_at(self.data, 4)? {
            return Err(FontError::BadTable);
        }
        let list = sub_at(self.data, 6 + usize::from(index) * 2)?.ok_or(FontError::BadTable)?;
        Ok(Some(U16s::read(list, 2, usize::from(u16_at(list, 0)?))?))
    }
}

/// The ligatures that start with each covered glyph.
#[derive(Clone, Copy, Debug)]
pub struct Ligatures<'a> {
    data: &'a [u8],
    coverage: Coverage<'a>,
}

/// One ligature: the glyph it makes and the glyphs after the first.
#[derive(Clone, Copy, Debug)]
pub struct Ligature<'a> {
    pub glyph: u16,
    pub rest: U16s<'a>,
}

impl<'a> Ligatures<'a> {
    fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            data,
            coverage: Coverage::at(data, 2)?,
        })
    }

    /// The ligatures that start with `first`, in the order to try them.
    pub fn starting(&self, first: u16) -> Result<Vec<Ligature<'a>>, FontError> {
        let Some(index) = self.coverage.index(first) else {
            return Ok(Vec::new());
        };
        if index >= u16_at(self.data, 4)? {
            return Err(FontError::BadTable);
        }
        let set = sub_at(self.data, 6 + usize::from(index) * 2)?.ok_or(FontError::BadTable)?;
        let count = usize::from(u16_at(set, 0)?);
        let mut found = Vec::with_capacity(count);
        for offset in U16s::read(set, 2, count)?.iter() {
            let ligature = sub(set, usize::from(offset))?.ok_or(FontError::BadTable)?;
            let parts = usize::from(u16_at(ligature, 2)?);
            found.push(Ligature {
                glyph: u16_at(ligature, 0)?,
                rest: U16s::read(
                    ligature,
                    4,
                    parts.checked_sub(1).ok_or(FontError::BadTable)?,
                )?,
            });
        }
        Ok(found)
    }
}
