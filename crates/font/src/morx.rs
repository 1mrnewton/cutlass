//! `morx`: the substitutions of a face made for Apple's layout engine.
//!
//! Where an OpenType face lists rules, these faces carry small state
//! machines: each glyph of a run has a class, and a class in a state
//! gives an action and the next state. A chain of subtables runs in
//! order, each a machine of one kind: one makes ligatures, one swaps a
//! glyph for another by what came before it, one moves glyphs, one puts
//! glyphs in, and one swaps glyphs with no machine at all. The system
//! emoji face joins flags, skin tones and families this way, and the
//! system's Arabic face picks its letter forms this way.
//!
//! This reader finds the subtables and answers for one class, one state,
//! or one glyph; the shaper runs the machines.

use crate::{FontError, bytes_at, u16_at, u32_at};

/// The most chains and subtables read from one table.
const MAX_PARTS: usize = 1_024;

/// The class of the end of the run.
pub const CLASS_END: u16 = 0;
/// The class of a glyph the machine's class table does not name.
pub const CLASS_OTHER: u16 = 1;
/// The class of a glyph a ligature took out.
pub const CLASS_DELETED: u16 = 2;
/// The glyph that stands for one a ligature took out.
pub const DELETED_GLYPH: u16 = 0xFFFF;

/// A table of one value for each of some glyphs.
#[derive(Clone, Copy, Debug)]
pub struct GlyphLookup<'a> {
    data: &'a [u8],
}

impl<'a> GlyphLookup<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if !matches!(u16_at(data, 0)?, 0 | 2 | 4 | 6 | 8 | 10) {
            return Err(FontError::BadTable);
        }
        Ok(Self { data })
    }

    /// The value for `glyph`, when the table has one.
    #[must_use]
    pub fn value(&self, glyph: u16) -> Option<u16> {
        self.sized(glyph, 2)
            .and_then(|value| u16::try_from(value).ok())
    }

    /// The value for `glyph` in a table whose values are four bytes each.
    #[must_use]
    pub fn wide(&self, glyph: u16) -> Option<u32> {
        self.sized(glyph, 4)
    }

    /// The value for `glyph` in a table of values `width` bytes each, 2
    /// or 4.
    fn sized(&self, glyph: u16, width: usize) -> Option<u32> {
        let data = self.data;
        let word = |at: usize| u16_at(data, at).ok();
        let value = |at: usize| {
            if width == 4 {
                u32_at(data, at).ok()
            } else {
                word(at).map(u32::from)
            }
        };
        match word(0)? {
            0 => value(2 + usize::from(glyph) * width),
            8 => {
                let index = glyph.checked_sub(word(2)?)?;
                (index < word(4)?).then(|| value(6 + usize::from(index) * width))?
            }
            10 => {
                // This one says how wide its own values are.
                let size = usize::from(word(2)?);
                let index = glyph.checked_sub(word(4)?)?;
                if index >= word(6)? {
                    return None;
                }
                let at = 8 + usize::from(index) * size;
                match size {
                    1 => data.get(at).map(|byte| u32::from(*byte)),
                    2 => word(at).map(u32::from),
                    4 if width == 4 => u32_at(data, at).ok(),
                    _ => None,
                }
            }
            format @ (2 | 4 | 6) => {
                let size = usize::from(word(2)?);
                if size < 4 {
                    return None;
                }
                // Sorted units: a glyph and its value, or a range of
                // glyphs with one value or a list of them.
                let (mut low, mut high) = (0, usize::from(word(4)?));
                while low < high {
                    let mid = low + (high - low) / 2;
                    let at = 12 + mid * size;
                    let last = word(at)?;
                    let first = if format == 6 { last } else { word(at + 2)? };
                    if glyph > last {
                        low = mid + 1;
                    } else if glyph < first {
                        high = mid;
                    } else {
                        return match format {
                            6 => value(at + 2),
                            2 => value(at + 4),
                            _ => {
                                let list = usize::from(word(at + 4)?);
                                value(list + usize::from(glyph - first) * width)
                            }
                        };
                    }
                }
                None
            }
            _ => None,
        }
    }
}

/// One state machine: a class for each glyph, and for each state and
/// class the entry that says what to do.
#[derive(Clone, Copy, Debug)]
pub struct StateTable<'a> {
    classes: u32,
    lookup: GlyphLookup<'a>,
    states: &'a [u8],
    entries: &'a [u8],
    entry_size: usize,
}

impl<'a> StateTable<'a> {
    /// Read the machine at the start of `body`, whose entries are
    /// `entry_size` bytes each.
    pub(crate) fn read(body: &'a [u8], entry_size: usize) -> Result<Self, FontError> {
        let offset = |at: usize| {
            let offset = usize::try_from(u32_at(body, at)?).map_err(|_| FontError::BadTable)?;
            body.get(offset..).ok_or(FontError::Truncated)
        };
        let classes = u32_at(body, 0)?;
        if !(4..=65_535).contains(&classes) {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            classes,
            lookup: GlyphLookup::read(offset(4)?)?,
            states: offset(8)?,
            entries: offset(12)?,
            entry_size,
        })
    }

    /// The class of `glyph` in this machine.
    #[must_use]
    pub fn class(&self, glyph: u16) -> u16 {
        if glyph == DELETED_GLYPH {
            return CLASS_DELETED;
        }
        self.lookup
            .value(glyph)
            .filter(|class| u32::from(*class) < self.classes)
            .unwrap_or(CLASS_OTHER)
    }

    /// The bytes of the entry for `class` in `state`: the next state, the
    /// flags, and what else its kind of machine keeps.
    pub fn entry(&self, state: u16, class: u16) -> Result<&'a [u8], FontError> {
        if u32::from(class) >= self.classes {
            return Err(FontError::BadTable);
        }
        let cell = usize::from(state)
            .checked_mul(self.classes as usize)
            .and_then(|row| row.checked_add(usize::from(class)))
            .and_then(|cell| cell.checked_mul(2))
            .ok_or(FontError::BadTable)?;
        let index = usize::from(u16_at(self.states, cell)?);
        bytes_at(self.entries, index * self.entry_size, self.entry_size)
    }
}

/// What one subtable does.
#[derive(Clone, Copy, Debug)]
pub enum Metamorphosis<'a> {
    /// Moves glyphs within a marked stretch.
    Rearrange(StateTable<'a>),
    /// Swaps a marked glyph and the current one for others.
    Contextual {
        machine: StateTable<'a>,
        /// The substitution tables, each found by a 32-bit offset.
        tables: &'a [u8],
    },
    /// Makes one glyph of several.
    Ligature {
        machine: StateTable<'a>,
        actions: &'a [u8],
        components: &'a [u8],
        ligatures: &'a [u8],
    },
    /// Swaps each glyph it names for another.
    Swap(GlyphLookup<'a>),
    /// Puts glyphs in beside a marked glyph and the current one.
    Insert {
        machine: StateTable<'a>,
        glyphs: &'a [u8],
    },
}

/// The value at place `index` of a list of 16-bit values.
pub fn word_at(list: &[u8], index: usize) -> Result<u16, FontError> {
    u16_at(list, index.checked_mul(2).ok_or(FontError::BadTable)?)
}

/// The value at place `index` of a list of 32-bit values.
pub fn long_at(list: &[u8], index: usize) -> Result<u32, FontError> {
    u32_at(list, index.checked_mul(4).ok_or(FontError::BadTable)?)
}

/// Substitution table `index` of a contextual subtable's `tables`, and
/// its value for `glyph`.
pub fn substitute(tables: &[u8], index: u16, glyph: u16) -> Result<Option<u16>, FontError> {
    let offset =
        usize::try_from(long_at(tables, usize::from(index))?).map_err(|_| FontError::BadTable)?;
    let table = tables.get(offset..).ok_or(FontError::Truncated)?;
    Ok(GlyphLookup::read(table)?.value(glyph))
}

/// One subtable of a chain.
#[derive(Clone, Copy, Debug)]
pub struct Subtable<'a> {
    /// It runs when the chain's flags share a bit with these.
    pub flags: u32,
    /// It is for text set down the page.
    pub vertical: bool,
    /// It is for text set either way.
    pub any_direction: bool,
    /// It walks the run from its end.
    pub backward: bool,
    /// Its order is that of the characters, not of the glyphs as seen.
    pub logical: bool,
    pub kind: Metamorphosis<'a>,
}

/// One chain: the flags that are on by default, and its subtables in the
/// order they run.
#[derive(Clone, Debug)]
pub struct Chain<'a> {
    pub default_flags: u32,
    pub subtables: Vec<Subtable<'a>>,
}

/// The `morx` table.
#[derive(Clone, Copy, Debug)]
pub struct Morx<'a> {
    data: &'a [u8],
    chains: usize,
}

impl<'a> Morx<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if !(2..=3).contains(&u16_at(data, 0)?) {
            return Err(FontError::Unsupported);
        }
        let chains = usize::try_from(u32_at(data, 4)?).map_err(|_| FontError::BadTable)?;
        if chains > MAX_PARTS {
            return Err(FontError::BadTable);
        }
        Ok(Self { data, chains })
    }

    /// The chains, in the order they run. A subtable of a kind this
    /// reader does not know is left out.
    pub fn chains(&self) -> Result<Vec<Chain<'a>>, FontError> {
        let mut chains = Vec::with_capacity(self.chains);
        let mut at = 8;
        for _ in 0..self.chains {
            let length =
                usize::try_from(u32_at(self.data, at + 4)?).map_err(|_| FontError::BadTable)?;
            let chain = bytes_at(self.data, at, length)?;
            let features = usize::try_from(u32_at(chain, 8)?).map_err(|_| FontError::BadTable)?;
            let count = usize::try_from(u32_at(chain, 12)?).map_err(|_| FontError::BadTable)?;
            if features > MAX_PARTS || count > MAX_PARTS || length < 16 {
                return Err(FontError::BadTable);
            }
            let mut subtables = Vec::with_capacity(count);
            let mut start = 16 + features * 12;
            for _ in 0..count {
                let size =
                    usize::try_from(u32_at(chain, start)?).map_err(|_| FontError::BadTable)?;
                if size < 12 {
                    return Err(FontError::BadTable);
                }
                let subtable = bytes_at(chain, start, size)?;
                let coverage = u32_at(subtable, 4)?;
                let body = &subtable[12..];
                let kind = match coverage & 0xFF {
                    0 => Some(Metamorphosis::Rearrange(StateTable::read(body, 4)?)),
                    1 => Some(Metamorphosis::Contextual {
                        machine: StateTable::read(body, 8)?,
                        tables: part(body, 16)?,
                    }),
                    2 => Some(Metamorphosis::Ligature {
                        machine: StateTable::read(body, 6)?,
                        actions: part(body, 16)?,
                        components: part(body, 20)?,
                        ligatures: part(body, 24)?,
                    }),
                    4 => Some(Metamorphosis::Swap(GlyphLookup::read(body)?)),
                    5 => Some(Metamorphosis::Insert {
                        machine: StateTable::read(body, 8)?,
                        glyphs: part(body, 16)?,
                    }),
                    _ => None,
                };
                if let Some(kind) = kind {
                    subtables.push(Subtable {
                        flags: u32_at(subtable, 8)?,
                        vertical: coverage & 0x8000_0000 != 0,
                        backward: coverage & 0x4000_0000 != 0,
                        any_direction: coverage & 0x2000_0000 != 0,
                        logical: coverage & 0x1000_0000 != 0,
                        kind,
                    });
                }
                start += size;
            }
            chains.push(Chain {
                default_flags: u32_at(chain, 0)?,
                subtables,
            });
            at += length;
        }
        Ok(chains)
    }
}

/// The part of `body` a 32-bit offset stored at `at` points to.
fn part(body: &[u8], at: usize) -> Result<&[u8], FontError> {
    let offset = usize::try_from(u32_at(body, at)?).map_err(|_| FontError::BadTable)?;
    body.get(offset..).ok_or(FontError::Truncated)
}
