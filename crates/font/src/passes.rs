//! Kerning as Apple's layout engine stores it: the subtables of a version 1
//! `kern` table and of a `kerx` table, each a pass over the line.
//!
//! A pass is a list of pairs (sorted pairs, or two tables of classes and
//! the array they index) or a state machine. A machine reads the glyphs
//! in order, remembers some of them, and on some steps moves the ones it
//! remembered: along the line, or across it, which is how these faces
//! lift a mark over a tall letter. Passes for vertical text and for a
//! variation are left out, as are the formats that attach by control
//! points.
//!
//! The shaper drives a machine: this reader gives it each glyph's class,
//! each step, and the values a step names. Every read is checked against
//! the subtable, so a table that lies about its sizes gives an error or no
//! value and never a read past it.

use crate::morx::{GlyphLookup, StateTable};
use crate::{FontError, bytes_at, i16_at, u16_at, u32_at};

/// The most passes one face may have. A real face has a dozen.
const MAX_PASSES: usize = 64;

/// The class of the end of the line.
pub const KERN_CLASS_END: u16 = 0;
/// The class of a glyph a machine does not name.
pub const KERN_CLASS_OTHER: u16 = 1;
/// The class of a glyph an earlier pass deleted.
pub const KERN_CLASS_DELETED: u16 = 2;
/// The glyph id a deleted glyph has.
const DELETED: u16 = 0xFFFF;

/// One subtable.
#[derive(Clone, Copy, Debug)]
pub enum KernPass<'a> {
    Pairs(KernPairs<'a>),
    Machine(KernMachine<'a>),
    Attach(KernAttach<'a>),
}

impl KernPass<'_> {
    /// Whether the pass moves glyphs across the line and not along it.
    #[must_use]
    pub fn cross_stream(&self) -> bool {
        match self {
            Self::Pairs(pairs) => pairs.cross_stream,
            Self::Machine(machine) => machine.cross_stream,
            Self::Attach(_) => false,
        }
    }

    /// Whether the pass reads the line from its end. Only `kerx` says so.
    #[must_use]
    pub fn backwards(&self) -> bool {
        match self {
            Self::Pairs(pairs) => pairs.backwards,
            Self::Machine(machine) => machine.backwards,
            Self::Attach(attach) => attach.backwards,
        }
    }
}

/// A pass that adjusts pairs of glyphs.
#[derive(Clone, Copy, Debug)]
pub struct KernPairs<'a> {
    kind: Pairs<'a>,
    /// The whole subtable, which a value may be an offset into.
    subtable: &'a [u8],
    /// How many values each pair has; more than none in a face with
    /// variations, whose values are then offsets to lists.
    tuples: u32,
    pub cross_stream: bool,
    pub backwards: bool,
}

#[derive(Clone, Copy, Debug)]
enum Pairs<'a> {
    /// Records of a left glyph, a right glyph and a value, in rising order.
    Sorted { records: &'a [u8], count: usize },
    /// `kern` format 2: each glyph's class is a byte offset, the left one
    /// to its row and the right one to its column, and the two added are
    /// where the value is in the subtable.
    Offsets {
        left: &'a [u8],
        right: &'a [u8],
        array: usize,
    },
    /// `kern` format 3: a class for each glyph on each side, an index for
    /// each pair of classes, and the values the indexes pick.
    Compact {
        values: &'a [u8],
        left: &'a [u8],
        right: &'a [u8],
        index: &'a [u8],
        left_classes: usize,
        right_classes: usize,
    },
    /// `kerx` format 2: lookups give each glyph a byte offset, the left
    /// one to its row and the right one to its column, and the two added
    /// are where the value is from the start of the array. Every face
    /// that ships with a Mac is built this way, and Core Text reads them
    /// so; HarfBuzz 14 counts values where these count bytes, and reads
    /// another cell.
    Lookups {
        left: GlyphLookup<'a>,
        right: GlyphLookup<'a>,
        array: &'a [u8],
    },
}

impl KernPairs<'_> {
    /// The adjustment for `left` followed by `right`, in font units; 0
    /// when the pass has none for them.
    #[must_use]
    pub fn get(&self, left: u16, right: u16) -> i32 {
        let value = match self.kind {
            Pairs::Sorted { records, count } => sorted(records, count, left, right),
            Pairs::Offsets {
                left: lefts,
                right: rights,
                array,
            } => {
                let at = usize::from(offset_class(lefts, left))
                    + usize::from(offset_class(rights, right));
                // An offset before the array names no value.
                if at < array {
                    None
                } else {
                    i16_at(self.subtable, at).ok()
                }
            }
            Pairs::Compact {
                values,
                left: lefts,
                right: rights,
                index,
                left_classes,
                right_classes,
            } => {
                let row = lefts
                    .get(usize::from(left))
                    .map(|class| usize::from(*class));
                let column = rights
                    .get(usize::from(right))
                    .map(|class| usize::from(*class));
                match (row, column) {
                    (Some(row), Some(column)) if row < left_classes && column < right_classes => {
                        index
                            .get(row * right_classes + column)
                            .and_then(|value| i16_at(values, usize::from(*value) * 2).ok())
                    }
                    _ => None,
                }
            }
            Pairs::Lookups {
                left: lefts,
                right: rights,
                array,
            } => {
                let at = usize::from(lefts.value(left).unwrap_or(0))
                    + usize::from(rights.value(right).unwrap_or(0));
                i16_at(array, at).ok()
            }
        };
        let Some(value) = value else {
            return 0;
        };
        if self.tuples == 0 {
            return i32::from(value);
        }
        // The value is where the pair's list is; its first entry is the
        // adjustment of the face as it is drawn.
        u16::try_from(value)
            .ok()
            .and_then(|at| i16_at(self.subtable, usize::from(at)).ok())
            .map_or(0, i32::from)
    }
}

fn sorted(records: &[u8], count: usize, left: u16, right: u16) -> Option<i16> {
    let (mut low, mut high) = (0, count);
    while low < high {
        let mid = low + (high - low) / 2;
        let at = mid * 6;
        let pair = (u16_at(records, at).ok()?, u16_at(records, at + 2).ok()?);
        match pair.cmp(&(left, right)) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => return i16_at(records, at + 4).ok(),
        }
    }
    None
}

/// The value of `glyph` in a class table of a first glyph, a count, and a
/// two-byte value for each glyph from the first; 0 outside it.
fn offset_class(table: &[u8], glyph: u16) -> u16 {
    let (Ok(first), Ok(count)) = (u16_at(table, 0), u16_at(table, 2)) else {
        return 0;
    };
    match glyph.checked_sub(first) {
        Some(index) if index < count => u16_at(table, 4 + usize::from(index) * 2).unwrap_or(0),
        _ => 0,
    }
}

/// A pass that is a state machine.
#[derive(Clone, Copy, Debug)]
pub struct KernMachine<'a> {
    kind: Machine<'a>,
    tuples: u32,
    pub cross_stream: bool,
    pub backwards: bool,
}

#[derive(Clone, Copy, Debug)]
enum Machine<'a> {
    /// The `kern` table's: one byte for each class and each cell, and
    /// states and values named by their byte offset in the table.
    Bytes {
        table: &'a [u8],
        classes: usize,
        class_table: usize,
        states: usize,
        entries: usize,
    },
    /// The `kerx` table's: classes from a lookup, cells of two bytes, and
    /// values by their index.
    Words {
        table: StateTable<'a>,
        values: &'a [u8],
    },
}

/// What a machine does on reading one glyph in one state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernStep {
    /// The state after it.
    pub next: i32,
    /// Remember this glyph.
    pub push: bool,
    /// Read this glyph again in the next state.
    pub stay: bool,
    /// Forget the glyphs remembered.
    pub reset: bool,
    /// Where the values for the remembered glyphs are, when the step
    /// moves them.
    pub values: Option<usize>,
}

impl KernMachine<'_> {
    /// The class of `glyph`; of the end of the line for `None`.
    #[must_use]
    pub fn class(&self, glyph: Option<u16>) -> u16 {
        let Some(glyph) = glyph else {
            return KERN_CLASS_END;
        };
        if glyph == DELETED {
            return KERN_CLASS_DELETED;
        }
        match self.kind {
            Machine::Bytes {
                table,
                classes,
                class_table,
                ..
            } => {
                let found = (|| {
                    let first = u16_at(table, class_table).ok()?;
                    let count = u16_at(table, class_table + 2).ok()?;
                    let index = glyph.checked_sub(first).filter(|index| *index < count)?;
                    table.get(class_table + 4 + usize::from(index)).copied()
                })();
                found
                    .map(u16::from)
                    .filter(|class| usize::from(*class) < classes)
                    .unwrap_or(KERN_CLASS_OTHER)
            }
            Machine::Words { table, .. } => table.class(glyph),
        }
    }

    /// The step for `class` in `state`. The first state is 0.
    pub fn step(&self, state: i32, class: u16) -> Result<KernStep, FontError> {
        match self.kind {
            Machine::Bytes {
                table,
                classes,
                states,
                entries,
                ..
            } => {
                if usize::from(class) >= classes {
                    return Err(FontError::BadTable);
                }
                // A state may be numbered below the first: its row is
                // before the array, and still inside the table.
                let row = i64::from(state)
                    .checked_mul(i64::try_from(classes).map_err(|_| FontError::BadTable)?)
                    .and_then(|row| {
                        row.checked_add(
                            i64::try_from(states)
                                .map_err(|_| FontError::BadTable)
                                .ok()?,
                        )
                    })
                    .and_then(|row| row.checked_add(i64::from(class)))
                    .and_then(|cell| usize::try_from(cell).ok())
                    .ok_or(FontError::BadTable)?;
                let entry = usize::from(*table.get(row).ok_or(FontError::Truncated)?);
                let at = entries + entry * 4;
                let new_state = i64::from(u16_at(table, at)?);
                let flags = u16_at(table, at + 2)?;
                // The next state is stored as the byte offset of its row.
                let classes = i64::try_from(classes).map_err(|_| FontError::BadTable)?;
                let states = i64::try_from(states).map_err(|_| FontError::BadTable)?;
                let next = i32::try_from((new_state - states) / classes)
                    .map_err(|_| FontError::BadTable)?;
                let values = usize::from(flags & 0x3FFF);
                Ok(KernStep {
                    next,
                    push: flags & 0x8000 != 0,
                    stay: flags & 0x4000 != 0,
                    reset: false,
                    values: (values != 0).then_some(values),
                })
            }
            Machine::Words { table, .. } => {
                let state = u16::try_from(state).map_err(|_| FontError::BadTable)?;
                let entry = table.entry(state, class)?;
                let flags = u16_at(entry, 2)?;
                let index = u16_at(entry, 4)?;
                Ok(KernStep {
                    next: i32::from(u16_at(entry, 0)?),
                    push: flags & 0x8000 != 0,
                    stay: flags & 0x4000 != 0,
                    reset: flags & 0x2000 != 0,
                    values: (index != 0xFFFF).then_some(usize::from(index)),
                })
            }
        }
    }

    /// The value `nth` places into the list a step named, in font units;
    /// `None` past the table.
    #[must_use]
    pub fn value(&self, values: usize, nth: usize) -> Option<i16> {
        match self.kind {
            Machine::Bytes { table, .. } => {
                i16_at(table, values.checked_add(nth.checked_mul(2)?)?).ok()
            }
            Machine::Words { values: list, .. } => {
                let each = usize::try_from(self.tuples.max(1)).ok()?;
                let at = values.checked_add(nth.checked_mul(each)?)?;
                i16_at(list, at.checked_mul(2)?).ok()
            }
        }
    }
}

/// Whether a `kern` table is a version 1 table with a subtable that is more
/// than sorted pairs along the line: such a table is applied pass by pass.
pub(crate) fn has_passes(table: &[u8]) -> bool {
    if (u16_at(table, 0), u16_at(table, 2)) != (Ok(1), Ok(0)) {
        return false;
    }
    let Ok(count) = u32_at(table, 4) else {
        return false;
    };
    let mut at = 8_usize;
    for _ in 0..count.min(MAX_PASSES as u32) {
        let (Ok(length), Ok(coverage)) = (u32_at(table, at), u16_at(table, at + 4)) else {
            return false;
        };
        if coverage & 0xA000 == 0 && coverage & 0x40FF != 0 {
            return true;
        }
        let Some(next) = usize::try_from(length)
            .ok()
            .filter(|length| *length >= 8)
            .and_then(|length| at.checked_add(length))
        else {
            return false;
        };
        at = next;
    }
    false
}

/// A pass that sets one glyph on another: a machine marks a glyph, and on
/// some later glyph names a point of each for the two to meet at. This is
/// how `kerx` places a mark on its letter.
#[derive(Clone, Copy, Debug)]
pub struct KernAttach<'a> {
    table: StateTable<'a>,
    /// The pairs of points the steps name.
    actions: &'a [u8],
    points: Points<'a>,
    pub backwards: bool,
}

/// Where a step's two points are.
#[derive(Clone, Copy, Debug)]
enum Points<'a> {
    /// Each is the number of an anchor of its glyph in the `ankr` table:
    /// a lookup from a glyph to its list, and the lists.
    Anchors {
        lookup: GlyphLookup<'a>,
        lists: &'a [u8],
    },
    /// Each is written out as an x and a y.
    Written,
}

/// What an attaching machine does on reading one glyph in one state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachStep {
    pub next: u16,
    /// Remember this glyph as the one later glyphs are set on.
    pub mark: bool,
    /// Read this glyph again in the next state.
    pub stay: bool,
    /// The pair of points that sets this glyph on the remembered one.
    pub action: Option<u16>,
}

impl KernAttach<'_> {
    /// The class of `glyph`; of the end of the line for `None`.
    #[must_use]
    pub fn class(&self, glyph: Option<u16>) -> u16 {
        glyph.map_or(KERN_CLASS_END, |glyph| self.table.class(glyph))
    }

    /// The step for `class` in `state`. The first state is 0.
    pub fn step(&self, state: u16, class: u16) -> Result<AttachStep, FontError> {
        let entry = self.table.entry(state, class)?;
        let flags = u16_at(entry, 2)?;
        let action = u16_at(entry, 4)?;
        Ok(AttachStep {
            next: u16_at(entry, 0)?,
            mark: flags & 0x8000 != 0,
            stay: flags & 0x4000 != 0,
            action: (action != 0xFFFF).then_some(action),
        })
    }

    /// How far `glyph` moves, right and up in font units, to meet the
    /// remembered glyph `marked` at the points `action` names. `None`
    /// when the table does not hold the action.
    #[must_use]
    pub fn offset(&self, action: u16, marked: u16, glyph: u16) -> Option<(i32, i32)> {
        let action = usize::from(action);
        let word = |at: usize| u16_at(self.actions, at).ok();
        match self.points {
            Points::Anchors { lookup, lists } => {
                let (on_marked, on_glyph) = (word(action * 4)?, word(action * 4 + 2)?);
                // A glyph without the anchor meets at its origin.
                let anchor = |glyph: u16, index: u16| {
                    let found = (|| {
                        let list = lists.get(usize::from(lookup.value(glyph)?)..)?;
                        let count = u32_at(list, 0).ok()?;
                        if u32::from(index) >= count {
                            return None;
                        }
                        let at = 4 + usize::from(index) * 4;
                        Some((i16_at(list, at).ok()?, i16_at(list, at + 2).ok()?))
                    })();
                    found.unwrap_or((0, 0))
                };
                let (marked, glyph) = (anchor(marked, on_marked), anchor(glyph, on_glyph));
                Some((
                    i32::from(marked.0) - i32::from(glyph.0),
                    i32::from(marked.1) - i32::from(glyph.1),
                ))
            }
            Points::Written => {
                let number = |at: usize| i16_at(self.actions, action * 8 + at).ok().map(i32::from);
                Some((number(0)? - number(4)?, number(2)? - number(6)?))
            }
        }
    }
}

/// The passes of a version 1 `kern` table, in the order they are applied.
pub(crate) fn kern_passes(table: &[u8]) -> Result<Vec<KernPass<'_>>, FontError> {
    let count = usize::try_from(u32_at(table, 4)?).map_err(|_| FontError::BadTable)?;
    let mut passes = Vec::new();
    let mut at = 8;
    for _ in 0..count.min(MAX_PASSES) {
        // A subtable that is not all there ends the list: the passes
        // before it still apply.
        let Some(subtable) = u32_at(table, at)
            .ok()
            .and_then(|length| usize::try_from(length).ok())
            .filter(|length| *length >= 8)
            .and_then(|length| bytes_at(table, at, length).ok())
        else {
            break;
        };
        at += subtable.len();
        let coverage = u16_at(subtable, 4)?;
        // Vertical text and variations are not this reader's.
        if coverage & 0xA000 != 0 {
            continue;
        }
        let cross_stream = coverage & 0x4000 != 0;
        let body = &subtable[8..];
        let pairs = |kind| {
            KernPass::Pairs(KernPairs {
                kind,
                subtable,
                tuples: 0,
                cross_stream,
                backwards: false,
            })
        };
        // A subtable that does not read is left out, and the rest still
        // apply.
        let pass = match coverage & 0xFF {
            0 => sorted_pairs(body, 2).map(pairs),
            1 => byte_machine(body).map(|kind| {
                KernPass::Machine(KernMachine {
                    kind,
                    tuples: 0,
                    cross_stream,
                    backwards: false,
                })
            }),
            2 => offset_pairs(subtable).map(pairs),
            3 => compact_pairs(body).map(pairs),
            _ => Err(FontError::Unsupported),
        };
        if let Ok(pass) = pass {
            passes.push(pass);
        }
    }
    Ok(passes)
}

/// The passes of a `kerx` table. `anchors` is the face's `ankr` table,
/// which the passes that set a mark on a letter read their points from.
pub(crate) fn kerx_passes<'a>(
    table: &'a [u8],
    anchors: Option<&'a [u8]>,
) -> Result<Vec<KernPass<'a>>, FontError> {
    if !(2..=4).contains(&u16_at(table, 0)?) {
        return Err(FontError::Unsupported);
    }
    let count = usize::try_from(u32_at(table, 4)?).map_err(|_| FontError::BadTable)?;
    let mut passes = Vec::new();
    let mut at = 8;
    for _ in 0..count.min(MAX_PASSES) {
        let Some(subtable) = u32_at(table, at)
            .ok()
            .and_then(|length| usize::try_from(length).ok())
            .filter(|length| *length >= 12)
            .and_then(|length| bytes_at(table, at, length).ok())
        else {
            break;
        };
        at += subtable.len();
        let coverage = u32_at(subtable, 4)?;
        if coverage & 0xA000_0000 != 0 {
            continue;
        }
        let tuples = u32_at(subtable, 8)?;
        let cross_stream = coverage & 0x4000_0000 != 0;
        let backwards = coverage & 0x1000_0000 != 0;
        let body = &subtable[12..];
        let pairs = |kind| {
            KernPass::Pairs(KernPairs {
                kind,
                subtable,
                tuples,
                cross_stream,
                backwards,
            })
        };
        let pass = match coverage & 0xFF {
            0 => sorted_pairs(body, 4).map(pairs),
            1 => word_machine(body).map(|kind| {
                KernPass::Machine(KernMachine {
                    kind,
                    tuples,
                    cross_stream,
                    backwards,
                })
            }),
            2 => lookup_pairs(subtable).map(pairs),
            4 => attach(body, anchors, backwards).map(KernPass::Attach),
            _ => Err(FontError::Unsupported),
        };
        if let Ok(pass) = pass {
            passes.push(pass);
        }
    }
    Ok(passes)
}

/// Sorted pairs after a count and three search fields, each `field`
/// bytes wide.
fn sorted_pairs(body: &[u8], field: usize) -> Result<Pairs<'_>, FontError> {
    let count = if field == 2 {
        usize::from(u16_at(body, 0)?)
    } else {
        usize::try_from(u32_at(body, 0)?).map_err(|_| FontError::BadTable)?
    };
    let bytes = count.checked_mul(6).ok_or(FontError::BadTable)?;
    Ok(Pairs::Sorted {
        records: bytes_at(body, field * 4, bytes)?,
        count,
    })
}

fn offset_pairs(subtable: &[u8]) -> Result<Pairs<'_>, FontError> {
    let table = |at: usize| {
        let offset = usize::from(u16_at(subtable, at)?);
        let table = subtable.get(offset..).ok_or(FontError::Truncated)?;
        let count = usize::from(u16_at(table, 2)?);
        bytes_at(table, 0, 4 + count * 2)
    };
    Ok(Pairs::Offsets {
        left: table(10)?,
        right: table(12)?,
        array: usize::from(u16_at(subtable, 14)?),
    })
}

fn compact_pairs(body: &[u8]) -> Result<Pairs<'_>, FontError> {
    let glyphs = usize::from(u16_at(body, 0)?);
    let header = bytes_at(body, 2, 4)?;
    let (values, left_classes, right_classes) = (
        usize::from(header[0]),
        usize::from(header[1]),
        usize::from(header[2]),
    );
    let mut at = 6;
    let mut take = |length: usize| {
        let bytes = bytes_at(body, at, length);
        at += length;
        bytes
    };
    Ok(Pairs::Compact {
        values: take(values * 2)?,
        left: take(glyphs)?,
        right: take(glyphs)?,
        index: take(left_classes * right_classes)?,
        left_classes,
        right_classes,
    })
}

fn lookup_pairs(subtable: &[u8]) -> Result<Pairs<'_>, FontError> {
    let at = |field: usize| {
        let offset = usize::try_from(u32_at(subtable, field)?).map_err(|_| FontError::BadTable)?;
        subtable.get(offset..).ok_or(FontError::Truncated)
    };
    Ok(Pairs::Lookups {
        left: GlyphLookup::read(at(16)?)?,
        right: GlyphLookup::read(at(20)?)?,
        array: at(24)?,
    })
}

fn byte_machine(body: &[u8]) -> Result<Machine<'_>, FontError> {
    let classes = usize::from(u16_at(body, 0)?);
    // Four classes are fixed, and a class is one byte.
    if !(4..=256).contains(&classes) {
        return Err(FontError::BadTable);
    }
    let (class_table, states, entries) = (
        usize::from(u16_at(body, 2)?),
        usize::from(u16_at(body, 4)?),
        usize::from(u16_at(body, 6)?),
    );
    // The class table's header and the first state are in the table.
    let _header = bytes_at(body, class_table, 4)?;
    let _first = bytes_at(body, states, classes)?;
    if entries >= body.len() {
        return Err(FontError::Truncated);
    }
    Ok(Machine::Bytes {
        table: body,
        classes,
        class_table,
        states,
        entries,
    })
}

fn word_machine(body: &[u8]) -> Result<Machine<'_>, FontError> {
    let table = StateTable::read(body, 6)?;
    let values = usize::try_from(u32_at(body, 16)?).map_err(|_| FontError::BadTable)?;
    Ok(Machine::Words {
        table,
        values: body.get(values..).ok_or(FontError::Truncated)?,
    })
}

fn attach<'a>(
    body: &'a [u8],
    anchors: Option<&'a [u8]>,
    backwards: bool,
) -> Result<KernAttach<'a>, FontError> {
    let table = StateTable::read(body, 6)?;
    let flags = u32_at(body, 16)?;
    let actions = usize::try_from(flags & 0x00FF_FFFF).map_err(|_| FontError::BadTable)?;
    let points = match flags >> 30 {
        1 => {
            let anchors = anchors.ok_or(FontError::MissingTable)?;
            let at = |field: usize| {
                let offset =
                    usize::try_from(u32_at(anchors, field)?).map_err(|_| FontError::BadTable)?;
                anchors.get(offset..).ok_or(FontError::Truncated)
            };
            Points::Anchors {
                lookup: GlyphLookup::read(at(4)?)?,
                lists: at(8)?,
            }
        }
        2 => Points::Written,
        // Points of the outline, by number: not read.
        _ => return Err(FontError::Unsupported),
    };
    Ok(KernAttach {
        table,
        actions: body.get(actions..).ok_or(FontError::Truncated)?,
        points,
        backwards,
    })
}

#[cfg(test)]
mod tests {
    use super::{KERN_CLASS_DELETED, KERN_CLASS_END, KERN_CLASS_OTHER, KernPass, KernStep};
    use crate::fixture::{boxed, font};
    use crate::fixture_kern::{extended, kerned};
    use crate::read;

    const A: u16 = 1;
    const B: u16 = 2;
    const ACUTE: u16 = 3;
    const C: u16 = 4;

    #[test]
    fn a_kern_table_of_machines_and_classes_is_read_pass_by_pass() {
        let bytes = kerned();
        let face = read(&bytes).expect("face");
        // Such a table is not a list of pairs.
        assert!(!face.has_kern_table());
        assert_eq!(face.kerning(A, B), 0);
        let passes = face.kern_passes();
        assert_eq!(passes.len(), 4);
        assert!(face.kerx_passes().is_empty());
        let across: Vec<bool> = passes.iter().map(KernPass::cross_stream).collect();
        assert_eq!(across, [false, true, false, false]);
        assert!(passes.iter().all(|pass| !pass.backwards()));
        let KernPass::Machine(along) = passes[0] else {
            panic!("a machine");
        };
        // A and B have classes of their own; the rest are the fixed ones.
        assert_eq!(along.class(Some(A)), 4);
        assert_eq!(along.class(Some(B)), 5);
        assert_eq!(along.class(Some(C)), KERN_CLASS_OTHER);
        assert_eq!(along.class(Some(0)), KERN_CLASS_OTHER);
        assert_eq!(along.class(Some(0xFFFF)), KERN_CLASS_DELETED);
        assert_eq!(along.class(None), KERN_CLASS_END);
        // An A is remembered and leads to the state that waits for a B.
        let seen = along.step(0, 4).expect("step");
        assert_eq!(
            seen,
            KernStep {
                next: 2,
                push: true,
                stay: false,
                reset: false,
                values: None
            }
        );
        let moved = along.step(2, 5).expect("step");
        assert_eq!((moved.next, moved.push), (0, true));
        let values = moved.values.expect("values");
        assert_eq!(along.value(values, 0), Some(-100));
        assert_eq!(along.value(values, 1), Some(-49));
        // The list is the end of the table: there is no third value.
        assert_eq!(along.value(values, 2), None);
        assert_eq!(along.value(usize::MAX, 1), None);
        // A class or a state the machine does not have is an error.
        assert!(along.step(0, 6).is_err());
        assert!(along.step(40, 4).is_err());
        assert!(along.step(-40, 4).is_err());
        // Pairs by offset and by compact classes.
        let (KernPass::Pairs(offsets), KernPass::Pairs(compact)) = (passes[2], passes[3]) else {
            panic!("pairs");
        };
        assert_eq!(offsets.get(C, A), -80);
        assert_eq!(offsets.get(A, C), 0);
        assert_eq!(offsets.get(C, C), 0);
        assert_eq!(offsets.get(900, 900), 0);
        assert_eq!(compact.get(A, C), -40);
        assert_eq!(compact.get(C, A), 0);
        assert_eq!(compact.get(A, 900), 0);
    }

    #[test]
    fn an_extended_table_is_read_with_its_anchors() {
        let bytes = extended();
        let face = read(&bytes).expect("face");
        assert!(face.kern_passes().is_empty());
        let passes = face.kerx_passes();
        assert_eq!(passes.len(), 4);
        let (KernPass::Pairs(sorted), KernPass::Pairs(classes)) = (passes[0], passes[1]) else {
            panic!("pairs");
        };
        assert_eq!(sorted.get(A, B), -60);
        assert_eq!(sorted.get(B, A), 0);
        // The class values are byte offsets into the array: the second
        // row and the second column.
        assert_eq!(classes.get(C, A), -90);
        assert_eq!(classes.get(C, B), 0);
        assert_eq!(classes.get(A, A), 0);
        let KernPass::Machine(along) = passes[2] else {
            panic!("a machine");
        };
        assert_eq!((along.class(Some(B)), along.class(Some(C))), (4, 5));
        let moved = along.step(2, 5).expect("step");
        assert_eq!(along.value(moved.values.expect("values"), 0), Some(-69));
        assert_eq!(along.step(0, 4).expect("step").values, None);
        let KernPass::Attach(attach) = passes[3] else {
            panic!("an attaching machine");
        };
        let marked = attach.step(0, attach.class(Some(A))).expect("step");
        assert!(marked.mark && marked.action.is_none());
        let set = attach
            .step(marked.next, attach.class(Some(ACUTE)))
            .expect("step");
        // The acute's anchor at (-150, 600) meets A's at (300, 700).
        assert_eq!(
            attach.offset(set.action.expect("action"), A, ACUTE),
            Some((450, 100))
        );
        // A glyph without an anchor meets at its origin.
        assert_eq!(attach.offset(0, B, ACUTE), Some((150, -600)));
        assert_eq!(attach.offset(9, A, ACUTE), None);
    }

    /// One table of a face.
    fn table_of(whole: &[u8], tag: &[u8; 4]) -> Vec<u8> {
        let count = usize::from(u16::from_be_bytes([whole[4], whole[5]]));
        let record = (0..count)
            .map(|index| 12 + index * 16)
            .find(|at| &whole[*at..*at + 4] == tag)
            .expect("table");
        let word = |at: usize| {
            usize::try_from(u32::from_be_bytes([
                whole[at],
                whole[at + 1],
                whole[at + 2],
                whole[at + 3],
            ]))
            .expect("word")
        };
        whole[word(record + 8)..word(record + 8) + word(record + 12)].to_vec()
    }

    /// A face of the fixture's glyphs with `tables`.
    fn face_with(tables: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let glyphs = [boxed(500), boxed(600), boxed(600), boxed(0), boxed(600)];
        font(&glyphs, &[('A', A), ('B', B), ('C', C)], tables)
    }

    /// Ask every pass of a face for everything it can give.
    fn exercise(bytes: &[u8]) {
        let Ok(face) = read(bytes) else {
            return;
        };
        for pass in face.kern_passes().iter().chain(face.kerx_passes()) {
            match pass {
                KernPass::Pairs(pairs) => {
                    for left in [0, A, B, C, 900, 0xFFFF] {
                        for right in [0, A, B, C, 900, 0xFFFF] {
                            let _ = pairs.get(left, right);
                        }
                    }
                }
                KernPass::Machine(machine) => {
                    for state in -3..12 {
                        for class in 0..8 {
                            if let Ok(step) = machine.step(state, class) {
                                for nth in 0..10 {
                                    let _ = machine.value(step.values.unwrap_or(0), nth);
                                }
                            }
                        }
                    }
                    let _ = machine.class(Some(A));
                }
                KernPass::Attach(attach) => {
                    for state in 0..12 {
                        for class in 0..8 {
                            if let Ok(step) = attach.step(state, class) {
                                let _ = attach.offset(step.action.unwrap_or(0), A, ACUTE);
                            }
                        }
                    }
                    let _ = attach.class(Some(A));
                }
            }
        }
    }

    #[test]
    fn a_torn_or_hostile_table_is_no_kerning_and_never_a_read_past_it() {
        let anchors = table_of(&extended(), b"ankr");
        let kern = table_of(&kerned(), b"kern");
        let kerx = table_of(&extended(), b"kerx");
        // Each table cut at every length, and with every byte turned to
        // all ones, to nothing and to a large number: the face opens or
        // does not, and its passes answer or give nothing.
        for (tag, table) in [(b"kern", &kern), (b"kerx", &kerx)] {
            let build = |table: Vec<u8>| face_with(&[(tag, table), (b"ankr", anchors.clone())]);
            exercise(&build(table.clone()));
            for cut in 0..table.len() {
                exercise(&build(table[..cut].to_vec()));
            }
            for at in 0..table.len() {
                for value in [0xFF, 0x00, 0x7F] {
                    let mut torn = table.clone();
                    torn[at] = value;
                    exercise(&build(torn));
                }
            }
        }
        // The anchors table torn the same way under a whole `kerx`.
        for cut in 0..anchors.len() {
            exercise(&face_with(&[
                (b"kerx", kerx.clone()),
                (b"ankr", anchors[..cut].to_vec()),
            ]));
        }
        // A table that says it has a great many passes gives the ones it
        // holds and stops where it ends.
        let mut many = kern.clone();
        many[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
        let bytes = face_with(&[(b"kern", many)]);
        assert_eq!(read(&bytes).expect("face").kern_passes().len(), 4);
        // Without the anchors table the pass that needs it is left out,
        // and the other three stay.
        let bytes = face_with(&[(b"kerx", kerx)]);
        assert_eq!(read(&bytes).expect("face").kerx_passes().len(), 3);
    }

    #[test]
    fn a_pair_is_found_quickly_in_every_kind_of_pass() {
        let (kern, kerx) = (kerned(), extended());
        let (kern, kerx) = (read(&kern).expect("kern"), read(&kerx).expect("kerx"));
        let started = std::time::Instant::now();
        let mut sum = 0_i64;
        for round in 0..200_000_u32 {
            let glyph = u16::try_from(round % 6).expect("glyph");
            for pass in kern.kern_passes().iter().chain(kerx.kerx_passes()) {
                if let KernPass::Pairs(pairs) = pass {
                    sum += i64::from(pairs.get(glyph, A)) + i64::from(pairs.get(A, glyph));
                }
            }
        }
        assert!(sum < 0);
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
}
