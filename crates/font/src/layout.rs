//! What `GSUB`, `GPOS` and `GDEF` share: scripts, features, lookups,
//! coverage, classes, and the rules that match a run of glyphs.
//!
//! These tables say what a shaper may do; they do not shape. Each reader
//! answers one question about one glyph or one rule, and the caller walks
//! its own glyphs. Offsets and counts are checked where they are read, and
//! an offset of zero is a table that is not there.

use crate::gpos::Positioning;
use crate::vary::Deltas;
use crate::{FontError, bytes_at, i16_at, u16_at, u32_at};

/// A four-letter tag: a script, a language, or a feature.
pub type Tag = [u8; 4];

/// The most lookups, features, or subtables one list may hold here.
const MAX_LIST: usize = 16_384;

/// `count` big-endian `u16` values.
#[derive(Clone, Copy, Debug, Default)]
pub struct U16s<'a>(&'a [u8]);

impl<'a> U16s<'a> {
    pub(crate) fn read(data: &'a [u8], at: usize, count: usize) -> Result<Self, FontError> {
        let bytes = count.checked_mul(2).ok_or(FontError::BadTable)?;
        Ok(Self(bytes_at(data, at, bytes)?))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len() / 2
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<u16> {
        let at = index.checked_mul(2)?;
        let bytes = self.0.get(at..at.checked_add(2)?)?;
        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    pub fn iter(&self) -> impl Iterator<Item = u16> + 'a {
        self.0
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes(*pair))
    }
}

/// The table at `offset` inside `data`, or `None` for an offset of zero.
pub(crate) fn sub(data: &[u8], offset: usize) -> Result<Option<&[u8]>, FontError> {
    if offset == 0 {
        return Ok(None);
    }
    data.get(offset..).map(Some).ok_or(FontError::Truncated)
}

/// The table a 16-bit offset stored at `at` points to.
pub(crate) fn sub_at(data: &[u8], at: usize) -> Result<Option<&[u8]>, FontError> {
    sub(data, usize::from(u16_at(data, at)?))
}

/// Which glyphs a subtable acts on, and each one's place in its lists.
#[derive(Clone, Copy, Debug)]
pub struct Coverage<'a> {
    ranges: bool,
    /// Glyph ids, or records of a first glyph, a last, and the first's index.
    records: &'a [u8],
}

impl<'a> Coverage<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        let format = u16_at(data, 0)?;
        let count = usize::from(u16_at(data, 2)?);
        match format {
            1 => Ok(Self {
                ranges: false,
                records: bytes_at(data, 4, count * 2)?,
            }),
            2 => Ok(Self {
                ranges: true,
                records: bytes_at(data, 4, count * 6)?,
            }),
            _ => Err(FontError::BadTable),
        }
    }

    /// The coverage table a 16-bit offset stored at `at` points to.
    pub(crate) fn at(data: &'a [u8], at: usize) -> Result<Self, FontError> {
        Self::read(sub_at(data, at)?.ok_or(FontError::BadTable)?)
    }

    /// The glyph's place in the coverage, when it is covered.
    #[must_use]
    pub fn index(&self, glyph: u16) -> Option<u16> {
        let size = if self.ranges { 6 } else { 2 };
        let read = |at: usize| {
            self.records
                .get(at..at + 2)
                .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
        };
        let (mut low, mut high) = (0, self.records.len() / size);
        while low < high {
            let mid = low + (high - low) / 2;
            let first = read(mid * size)?;
            if glyph < first {
                high = mid;
                continue;
            }
            if !self.ranges {
                if glyph == first {
                    return u16::try_from(mid).ok();
                }
                low = mid + 1;
                continue;
            }
            if glyph > read(mid * size + 2)? {
                low = mid + 1;
                continue;
            }
            return read(mid * size + 4)?.checked_add(glyph - first);
        }
        None
    }

    #[must_use]
    pub fn covers(&self, glyph: u16) -> bool {
        self.index(glyph).is_some()
    }
}

/// A class for every glyph; one the table does not name is class 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassDef<'a> {
    /// The first glyph of a format 1 table; a format 2 table has ranges.
    first: Option<u16>,
    records: &'a [u8],
}

impl<'a> ClassDef<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        match u16_at(data, 0)? {
            1 => {
                let count = usize::from(u16_at(data, 4)?);
                Ok(Self {
                    first: Some(u16_at(data, 2)?),
                    records: bytes_at(data, 6, count * 2)?,
                })
            }
            2 => {
                let count = usize::from(u16_at(data, 2)?);
                Ok(Self {
                    first: None,
                    records: bytes_at(data, 4, count * 6)?,
                })
            }
            _ => Err(FontError::BadTable),
        }
    }

    /// The class table a 16-bit offset stored at `at` points to; no table
    /// puts every glyph in class 0.
    pub(crate) fn at(data: &'a [u8], at: usize) -> Result<Self, FontError> {
        sub_at(data, at)?.map_or(Ok(Self::default()), Self::read)
    }

    #[must_use]
    pub fn class(&self, glyph: u16) -> u16 {
        let read = |at: usize| {
            self.records
                .get(at..at + 2)
                .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
        };
        if let Some(first) = self.first {
            return glyph
                .checked_sub(first)
                .and_then(|index| read(usize::from(index) * 2))
                .unwrap_or(0);
        }
        let (mut low, mut high) = (0, self.records.len() / 6);
        while low < high {
            let mid = low + (high - low) / 2;
            let (Some(start), Some(end)) = (read(mid * 6), read(mid * 6 + 2)) else {
                return 0;
            };
            if glyph < start {
                high = mid;
            } else if glyph > end {
                low = mid + 1;
            } else {
                return read(mid * 6 + 4).unwrap_or(0);
            }
        }
        0
    }
}

/// A lookup's flags: which glyphs it steps over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupFlags {
    pub right_to_left: bool,
    pub ignore_bases: bool,
    pub ignore_ligatures: bool,
    pub ignore_marks: bool,
    /// Only marks in this set of `GDEF` count; the rest are stepped over.
    pub mark_set: Option<u16>,
    /// Only marks of this attachment class count, when it is not zero.
    pub mark_class: u8,
}

/// One lookup: its kind, its flags, and its subtables, tried in order.
#[derive(Clone, Debug)]
pub struct Lookup<'a> {
    /// The lookup type, with an extension lookup read as what it extends.
    pub kind: u16,
    pub flags: LookupFlags,
    pub subtables: Vec<&'a [u8]>,
}

/// The features a script and language turn on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Language {
    /// The script that was found, of the ones asked for.
    pub script: Option<Tag>,
    /// A feature that is always applied, when the language names one.
    pub required: Option<u16>,
    /// Feature indices, in the font's order.
    pub features: Vec<u16>,
}

/// `GSUB` or `GPOS`: scripts, features and lookups.
#[derive(Clone, Copy, Debug)]
pub struct LayoutTable<'a> {
    scripts: Option<&'a [u8]>,
    features: Option<&'a [u8]>,
    lookups: Option<&'a [u8]>,
    /// The lookup type that wraps another: 7 in `GSUB`, 9 in `GPOS`.
    extension: u16,
    /// The table of a variable face that says which features change with
    /// its setting.
    variations: Option<&'a [u8]>,
    /// The features that take another's place where the face is set.
    substitutes: Option<&'a [u8]>,
    /// What changes the table's values where the face is set.
    deltas: Option<Deltas<'a>>,
}

impl<'a> LayoutTable<'a> {
    pub(crate) fn read(data: &'a [u8], extension: u16) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::Unsupported);
        }
        // From version 1.1 a table may say which features change with
        // a variable face's setting.
        let variations = if u16_at(data, 2)? >= 1 {
            match usize::try_from(u32_at(data, 10)?).map_err(|_| FontError::BadTable)? {
                0 => None,
                at => Some(data.get(at..).ok_or(FontError::Truncated)?),
            }
        } else {
            None
        };
        let mut table = Self {
            scripts: sub_at(data, 4)?,
            features: sub_at(data, 6)?,
            lookups: sub_at(data, 8)?,
            extension,
            variations,
            substitutes: None,
            deltas: None,
        };
        // A feature may change at the face's default too.
        table.vary(&[], None);
        // The lists' own counts are read now, so a torn table fails here.
        for list in [table.scripts, table.features, table.lookups]
            .into_iter()
            .flatten()
        {
            let _count = u16_at(list, 0)?;
        }
        Ok(table)
    }

    /// The language system for the first of `scripts` the font has, else
    /// its default script. `language` picks one of that script's languages;
    /// without it, or when the script lacks it, the script's default.
    pub fn language(&self, scripts: &[Tag], language: Option<Tag>) -> Result<Language, FontError> {
        let Some(list) = self.scripts else {
            return Ok(Language::default());
        };
        let find = |tag: Tag| tagged(list, tag);
        let mut found = None;
        for tag in scripts
            .iter()
            .copied()
            .chain([*b"DFLT", *b"dflt", *b"latn"])
        {
            if let Some(script) = find(tag)? {
                found = Some((tag, script));
                break;
            }
        }
        let Some((tag, script)) = found else {
            return Ok(Language::default());
        };
        let by_language = match language {
            Some(language) => tagged_from(script, 2, language)?,
            None => None,
        };
        let Some(system) =
            by_language.map_or_else(|| sub_at(script, 0), |found| Ok(Some(found)))?
        else {
            return Ok(Language {
                script: Some(tag),
                ..Language::default()
            });
        };
        let required = Some(u16_at(system, 2)?).filter(|index| *index != 0xFFFF);
        let count = usize::from(u16_at(system, 4)?);
        Ok(Language {
            script: Some(tag),
            required,
            features: U16s::read(system, 6, count)?.iter().collect(),
        })
    }

    /// The tag of feature `index`.
    pub fn feature_tag(&self, index: u16) -> Result<Tag, FontError> {
        let list = self.features.ok_or(FontError::BadTable)?;
        if index >= u16_at(list, 0)? {
            return Err(FontError::BadTable);
        }
        let bytes = bytes_at(list, 2 + usize::from(index) * 6, 4)?;
        Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    /// Set the table for a variable face at `places` along its axes:
    /// the features that change there, and the `deltas` of its values.
    pub(crate) fn vary(&mut self, places: &[i16], deltas: Option<Deltas<'a>>) {
        self.substitutes = self
            .variations
            .and_then(|variations| substitutes(variations, places).ok().flatten());
        self.deltas = deltas;
    }

    /// A positioning subtable of lookup type `kind`, 1 to 8, read with
    /// the values it has where a variable face is set: kerning, anchors
    /// and the rest move with the face. For a face that does not vary, or
    /// one at its default, this is [`Positioning::read`].
    pub fn positioning(&self, kind: u16, data: &'a [u8]) -> Result<Positioning<'a>, FontError> {
        Positioning::read_varied(kind, data, self.deltas)
    }

    /// The lookups feature `index` runs. In a variable face a feature may
    /// run other lookups where the face is set than at its default.
    pub fn feature_lookups(&self, index: u16) -> Result<U16s<'a>, FontError> {
        let list = self.features.ok_or(FontError::BadTable)?;
        if index >= u16_at(list, 0)? {
            return Err(FontError::BadTable);
        }
        if let Some(feature) = self.substitute(index)? {
            return U16s::read(feature, 4, usize::from(u16_at(feature, 2)?));
        }
        let feature = sub_at(list, 2 + usize::from(index) * 6 + 4)?.ok_or(FontError::BadTable)?;
        U16s::read(feature, 4, usize::from(u16_at(feature, 2)?))
    }

    /// The feature table that takes the place of feature `index` where
    /// the face is set, when one does.
    fn substitute(&self, index: u16) -> Result<Option<&'a [u8]>, FontError> {
        let Some(table) = self.substitutes else {
            return Ok(None);
        };
        // Records of a feature and where its stand-in is, in rising order.
        let count = usize::from(u16_at(table, 4)?);
        let (mut low, mut high) = (0, count);
        while low < high {
            let mid = low + (high - low) / 2;
            let at = 6 + mid * 6;
            let feature = u16_at(table, at)?;
            if feature < index {
                low = mid + 1;
            } else if feature > index {
                high = mid;
            } else {
                let offset =
                    usize::try_from(u32_at(table, at + 2)?).map_err(|_| FontError::BadTable)?;
                return sub(table, offset);
            }
        }
        Ok(None)
    }

    #[must_use]
    pub fn lookup_count(&self) -> u16 {
        self.lookups
            .and_then(|list| u16_at(list, 0).ok())
            .unwrap_or(0)
    }

    /// Lookup `index`, with an extension lookup's subtables followed to
    /// the tables they wrap.
    pub fn lookup(&self, index: u16) -> Result<Lookup<'a>, FontError> {
        let list = self.lookups.ok_or(FontError::BadTable)?;
        if index >= u16_at(list, 0)? {
            return Err(FontError::BadTable);
        }
        let lookup = sub_at(list, 2 + usize::from(index) * 2)?.ok_or(FontError::BadTable)?;
        let mut kind = u16_at(lookup, 0)?;
        let flag = u16_at(lookup, 2)?;
        let count = usize::from(u16_at(lookup, 4)?);
        if count > MAX_LIST {
            return Err(FontError::BadTable);
        }
        let offsets = U16s::read(lookup, 6, count)?;
        let mark_set = if flag & 0x10 == 0 {
            None
        } else {
            Some(u16_at(lookup, 6 + count * 2)?)
        };
        let mut subtables = Vec::with_capacity(count);
        let mut wrapped = None;
        for offset in offsets.iter() {
            let table = sub(lookup, usize::from(offset))?.ok_or(FontError::BadTable)?;
            if kind != self.extension {
                subtables.push(table);
                continue;
            }
            if u16_at(table, 0)? != 1 {
                return Err(FontError::BadTable);
            }
            let inner = u16_at(table, 2)?;
            if inner == self.extension || wrapped.is_some_and(|seen| seen != inner) {
                return Err(FontError::BadTable);
            }
            wrapped = Some(inner);
            let at = usize::try_from(u32_at(table, 4)?).map_err(|_| FontError::BadTable)?;
            subtables.push(sub(table, at)?.ok_or(FontError::BadTable)?);
        }
        if let Some(inner) = wrapped {
            kind = inner;
        }
        Ok(Lookup {
            kind,
            flags: LookupFlags {
                right_to_left: flag & 1 != 0,
                ignore_bases: flag & 2 != 0,
                ignore_ligatures: flag & 4 != 0,
                ignore_marks: flag & 8 != 0,
                mark_set,
                mark_class: u8::try_from(flag >> 8).unwrap_or(0),
            },
            subtables,
        })
    }
}

/// The features that take others' places at `places`, from the table
/// `variations`: those of the first record whose conditions all hold.
/// Each condition is a range of places along one axis.
fn substitutes<'a>(variations: &'a [u8], places: &[i16]) -> Result<Option<&'a [u8]>, FontError> {
    if u16_at(variations, 0)? != 1 {
        return Ok(None);
    }
    let offset = |data: &[u8], at: usize| {
        usize::try_from(u32_at(data, at)?).map_err(|_| FontError::BadTable)
    };
    let count = offset(variations, 4)?.min(MAX_LIST);
    for record in 0..count {
        let at = 8 + record * 8;
        // A record with no conditions always holds.
        let holds = match sub(variations, offset(variations, at)?)? {
            None => true,
            Some(set) => {
                let conditions = usize::from(u16_at(set, 0)?);
                let mut holds = true;
                for index in 0..conditions {
                    let condition =
                        sub(set, offset(set, 2 + index * 4)?)?.ok_or(FontError::BadTable)?;
                    // A condition of a form this reader does not know
                    // does not hold.
                    if u16_at(condition, 0)? != 1 {
                        holds = false;
                        break;
                    }
                    let place = places
                        .get(usize::from(u16_at(condition, 2)?))
                        .copied()
                        .unwrap_or(0);
                    if place < i16_at(condition, 4)? || place > i16_at(condition, 6)? {
                        holds = false;
                        break;
                    }
                }
                holds
            }
        };
        if holds {
            return sub(variations, offset(variations, at + 4)?);
        }
    }
    Ok(None)
}

/// The table a list of tagged records at the start of `list` holds for
/// `tag`: a count, then records of a tag and a 16-bit offset.
fn tagged(list: &[u8], tag: Tag) -> Result<Option<&[u8]>, FontError> {
    tagged_from(list, 0, tag)
}

fn tagged_from(list: &[u8], at: usize, tag: Tag) -> Result<Option<&[u8]>, FontError> {
    let count = usize::from(u16_at(list, at)?);
    let records = bytes_at(list, at + 2, count * 6)?;
    for record in records.as_chunks::<6>().0 {
        if record[..4] == tag {
            return sub(
                list,
                usize::from(u16::from_be_bytes([record[4], record[5]])),
            );
        }
    }
    Ok(None)
}

/// `GDEF`: what kind each glyph is, and which marks a lookup counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gdef<'a> {
    classes: ClassDef<'a>,
    has_classes: bool,
    mark_classes: ClassDef<'a>,
    mark_sets: Option<&'a [u8]>,
}

/// A glyph's kind in `GDEF`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlyphClass {
    /// The table does not class the glyph.
    None,
    Base,
    Ligature,
    Mark,
    Component,
}

impl<'a> Gdef<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::Unsupported);
        }
        let minor = u16_at(data, 2)?;
        let mark_sets = if minor >= 2 { sub_at(data, 12)? } else { None };
        if let Some(sets) = mark_sets
            && u16_at(sets, 0)? != 1
        {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            classes: ClassDef::at(data, 4)?,
            has_classes: u16_at(data, 4)? != 0,
            mark_classes: ClassDef::at(data, 10)?,
            mark_sets,
        })
    }

    /// Whether the table classes glyphs at all.
    #[must_use]
    pub fn has_classes(&self) -> bool {
        self.has_classes
    }

    #[must_use]
    pub fn class(&self, glyph: u16) -> GlyphClass {
        match self.classes.class(glyph) {
            1 => GlyphClass::Base,
            2 => GlyphClass::Ligature,
            3 => GlyphClass::Mark,
            4 => GlyphClass::Component,
            _ => GlyphClass::None,
        }
    }

    /// The mark attachment class of a glyph; 0 when it has none.
    #[must_use]
    pub fn mark_class(&self, glyph: u16) -> u16 {
        self.mark_classes.class(glyph)
    }

    /// Whether mark set `set` holds `glyph`. A set the table lacks holds
    /// nothing.
    #[must_use]
    pub fn in_mark_set(&self, set: u16, glyph: u16) -> bool {
        let found = (|| {
            let sets = self.mark_sets?;
            if set >= u16_at(sets, 2).ok()? {
                return None;
            }
            let at = usize::try_from(u32_at(sets, 4 + usize::from(set) * 4).ok()?).ok()?;
            let coverage = Coverage::read(sub(sets, at).ok()??).ok()?;
            Some(coverage.covers(glyph))
        })();
        found.unwrap_or(false)
    }
}

/// How a rule's glyphs are told: by id, by class, or by coverage.
#[derive(Clone, Copy, Debug)]
enum Matcher<'a> {
    Glyphs,
    Classes {
        backtrack: ClassDef<'a>,
        input: ClassDef<'a>,
        lookahead: ClassDef<'a>,
    },
    /// Each value is an offset to a coverage table inside this one.
    Coverages(&'a [u8]),
}

impl Matcher<'_> {
    fn matches(&self, part: Part, value: u16, glyph: u16) -> Result<bool, FontError> {
        match self {
            Self::Glyphs => Ok(value == glyph),
            Self::Classes {
                backtrack,
                input,
                lookahead,
            } => {
                let classes = match part {
                    Part::Backtrack => backtrack,
                    Part::Input => input,
                    Part::Lookahead => lookahead,
                };
                Ok(classes.class(glyph) == value)
            }
            Self::Coverages(table) => {
                let coverage =
                    Coverage::read(sub(table, usize::from(value))?.ok_or(FontError::BadTable)?)?;
                Ok(coverage.covers(glyph))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Part {
    Backtrack,
    Input,
    Lookahead,
}

/// One rule of a contextual lookup: the glyphs before, the glyphs it
/// acts on after the first, the glyphs after, and the lookups to run.
#[derive(Clone, Copy, Debug)]
pub struct SequenceRule<'a> {
    matcher: Matcher<'a>,
    backtrack: U16s<'a>,
    input: U16s<'a>,
    lookahead: U16s<'a>,
    /// Records of a place in the input and a lookup to run there.
    records: &'a [u8],
}

impl<'a> SequenceRule<'a> {
    /// How many glyphs before the first must match.
    #[must_use]
    pub fn backtrack_len(&self) -> usize {
        self.backtrack.len()
    }

    /// How many glyphs after the first the rule acts on.
    #[must_use]
    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    /// How many glyphs after the input must match.
    #[must_use]
    pub fn lookahead_len(&self) -> usize {
        self.lookahead.len()
    }

    /// Whether `glyph` may be the one `index` places before the first; 0 is
    /// the nearest.
    pub fn backtrack_matches(&self, index: usize, glyph: u16) -> Result<bool, FontError> {
        let value = self.backtrack.get(index).ok_or(FontError::BadTable)?;
        self.matcher.matches(Part::Backtrack, value, glyph)
    }

    /// Whether `glyph` may be the one `index` places after the first; 0 is
    /// the second glyph of the input.
    pub fn input_matches(&self, index: usize, glyph: u16) -> Result<bool, FontError> {
        let value = self.input.get(index).ok_or(FontError::BadTable)?;
        self.matcher.matches(Part::Input, value, glyph)
    }

    /// Whether `glyph` may be the one `index` places after the input.
    pub fn lookahead_matches(&self, index: usize, glyph: u16) -> Result<bool, FontError> {
        let value = self.lookahead.get(index).ok_or(FontError::BadTable)?;
        self.matcher.matches(Part::Lookahead, value, glyph)
    }

    /// The lookups to run, each with the place in the input it starts at.
    pub fn records(&self) -> impl Iterator<Item = (u16, u16)> + 'a {
        self.records.as_chunks::<4>().0.iter().map(|record| {
            (
                u16::from_be_bytes([record[0], record[1]]),
                u16::from_be_bytes([record[2], record[3]]),
            )
        })
    }
}

/// A contextual lookup subtable, with or without glyphs before and after.
#[derive(Clone, Copy, Debug)]
pub struct SequenceContext<'a> {
    data: &'a [u8],
    format: u16,
    chained: bool,
}

impl<'a> SequenceContext<'a> {
    pub(crate) fn read(data: &'a [u8], chained: bool) -> Result<Self, FontError> {
        let format = u16_at(data, 0)?;
        if !(1..=3).contains(&format) {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            data,
            format,
            chained,
        })
    }

    /// The rules that may start at `glyph`, in the order to try them.
    pub fn rules(&self, glyph: u16) -> Result<Vec<SequenceRule<'a>>, FontError> {
        let data = self.data;
        if self.format == 3 {
            return self.coverage_rule(glyph);
        }
        let Some(covered) = Coverage::at(data, 2)?.index(glyph) else {
            return Ok(Vec::new());
        };
        let (matcher, set, sets_at) = match (self.format, self.chained) {
            (1, _) => (Matcher::Glyphs, covered, 4),
            (_, false) => {
                let input = ClassDef::at(data, 4)?;
                let matcher = Matcher::Classes {
                    backtrack: ClassDef::default(),
                    input,
                    lookahead: ClassDef::default(),
                };
                (matcher, input.class(glyph), 6)
            }
            (_, true) => {
                let input = ClassDef::at(data, 6)?;
                let matcher = Matcher::Classes {
                    backtrack: ClassDef::at(data, 4)?,
                    input,
                    lookahead: ClassDef::at(data, 8)?,
                };
                (matcher, input.class(glyph), 10)
            }
        };
        if set >= u16_at(data, sets_at)? {
            return Ok(Vec::new());
        }
        let Some(rules) = sub_at(data, sets_at + 2 + usize::from(set) * 2)? else {
            return Ok(Vec::new());
        };
        let count = usize::from(u16_at(rules, 0)?);
        let mut found = Vec::with_capacity(count);
        for offset in U16s::read(rules, 2, count)?.iter() {
            let rule = sub(rules, usize::from(offset))?.ok_or(FontError::BadTable)?;
            found.push(self.rule(matcher, rule)?);
        }
        Ok(found)
    }

    /// One rule of format 1 or 2.
    fn rule(&self, matcher: Matcher<'a>, rule: &'a [u8]) -> Result<SequenceRule<'a>, FontError> {
        if !self.chained {
            let glyphs = usize::from(u16_at(rule, 0)?);
            let lookups = usize::from(u16_at(rule, 2)?);
            let input = U16s::read(rule, 4, glyphs.checked_sub(1).ok_or(FontError::BadTable)?)?;
            return Ok(SequenceRule {
                matcher,
                backtrack: U16s::default(),
                input,
                lookahead: U16s::default(),
                records: bytes_at(rule, 4 + input.len() * 2, lookups * 4)?,
            });
        }
        let mut at = 0;
        let backtrack = U16s::read(rule, at + 2, usize::from(u16_at(rule, at)?))?;
        at += 2 + backtrack.len() * 2;
        let glyphs = usize::from(u16_at(rule, at)?);
        let input = U16s::read(
            rule,
            at + 2,
            glyphs.checked_sub(1).ok_or(FontError::BadTable)?,
        )?;
        at += 2 + input.len() * 2;
        let lookahead = U16s::read(rule, at + 2, usize::from(u16_at(rule, at)?))?;
        at += 2 + lookahead.len() * 2;
        let lookups = usize::from(u16_at(rule, at)?);
        Ok(SequenceRule {
            matcher,
            backtrack,
            input,
            lookahead,
            records: bytes_at(rule, at + 2, lookups * 4)?,
        })
    }

    /// The one rule of a format 3 subtable, when `glyph` is in its first
    /// coverage.
    fn coverage_rule(&self, glyph: u16) -> Result<Vec<SequenceRule<'a>>, FontError> {
        let data = self.data;
        let matcher = Matcher::Coverages(data);
        let rule = if self.chained {
            let mut at = 2;
            let backtrack = U16s::read(data, at + 2, usize::from(u16_at(data, at)?))?;
            at += 2 + backtrack.len() * 2;
            let glyphs = usize::from(u16_at(data, at)?);
            let all = U16s::read(data, at + 2, glyphs)?;
            at += 2 + glyphs * 2;
            let lookahead = U16s::read(data, at + 2, usize::from(u16_at(data, at)?))?;
            at += 2 + lookahead.len() * 2;
            let lookups = usize::from(u16_at(data, at)?);
            (
                all,
                SequenceRule {
                    matcher,
                    backtrack,
                    input: U16s(all.0.get(2..).unwrap_or_default()),
                    lookahead,
                    records: bytes_at(data, at + 2, lookups * 4)?,
                },
            )
        } else {
            let glyphs = usize::from(u16_at(data, 2)?);
            let lookups = usize::from(u16_at(data, 4)?);
            let all = U16s::read(data, 6, glyphs)?;
            (
                all,
                SequenceRule {
                    matcher,
                    backtrack: U16s::default(),
                    input: U16s(all.0.get(2..).unwrap_or_default()),
                    lookahead: U16s::default(),
                    records: bytes_at(data, 6 + glyphs * 2, lookups * 4)?,
                },
            )
        };
        let (all, rule) = rule;
        let first = all.get(0).ok_or(FontError::BadTable)?;
        if matcher.matches(Part::Input, first, glyph)? {
            Ok(vec![rule])
        } else {
            Ok(Vec::new())
        }
    }
}

/// A reverse chaining substitution: one glyph replaced, told by the
/// glyphs around it, for a run walked from its end.
#[derive(Clone, Copy, Debug)]
pub struct ReverseChain<'a> {
    data: &'a [u8],
}

impl<'a> ReverseChain<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::BadTable);
        }
        Ok(Self { data })
    }

    /// The glyphs that must surround `glyph`, and what replaces it.
    pub fn rule(&self, glyph: u16) -> Result<Option<(SequenceRule<'a>, u16)>, FontError> {
        let data = self.data;
        let Some(index) = Coverage::at(data, 2)?.index(glyph) else {
            return Ok(None);
        };
        let mut at = 4;
        let backtrack = U16s::read(data, at + 2, usize::from(u16_at(data, at)?))?;
        at += 2 + backtrack.len() * 2;
        let lookahead = U16s::read(data, at + 2, usize::from(u16_at(data, at)?))?;
        at += 2 + lookahead.len() * 2;
        let substitutes = U16s::read(data, at + 2, usize::from(u16_at(data, at)?))?;
        let Some(substitute) = substitutes.get(usize::from(index)) else {
            return Err(FontError::BadTable);
        };
        Ok(Some((
            SequenceRule {
                matcher: Matcher::Coverages(data),
                backtrack,
                input: U16s::default(),
                lookahead,
                records: &[],
            },
            substitute,
        )))
    }
}
