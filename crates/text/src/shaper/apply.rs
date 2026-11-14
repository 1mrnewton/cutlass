//! Runs the font's lookups over a buffer of glyphs.
//!
//! A lookup is walked along the run; at each glyph it applies to, its
//! subtables are tried in order and the first that matches acts. A
//! substitution changes the glyphs; a positioning lookup moves them or
//! hangs one on another. The font's rules are data from a file, so every
//! walk is bounded: a run may not grow past a limit, lookups nest only so
//! deep, and the whole run has a budget of steps.

use std::rc::Rc;

use cutlass_font::{
    Anchor, Font, FontError, Gdef, GlyphClass, LayoutTable, LookupFlags, Positioning, SequenceRule,
    Substitution,
};

use super::buffer::{Attach, Buffer, Glyph};
use crate::TextError;
use crate::shape::Pair;
use crate::unicode::gc;

/// The most glyphs a contextual rule may act on.
const MAX_CONTEXT: usize = 64;
/// How deep a contextual lookup may run other lookups.
const MAX_NESTING: u8 = 6;

/// One lookup to run and the glyphs it runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LookupRef {
    pub(crate) index: u16,
    /// It applies to glyphs whose mask shares a bit with this.
    pub(crate) mask: u32,
    /// A zero width joiner between glyphs does not stop a ligature.
    pub(crate) auto_joiner: bool,
    /// A zero width non-joiner is stepped over where a rule looks around.
    pub(crate) auto_non_joiner: bool,
    /// Rules do not look past the syllable of the glyph they start at.
    pub(crate) per_syllable: bool,
}

/// A lookup read once: its flags and its subtables.
struct Parsed<'a> {
    kind: u16,
    flags: LookupFlags,
    subtables: Vec<Subtable<'a>>,
}

enum Subtable<'a> {
    Substitution(Substitution<'a>),
    Positioning(Positioning<'a>),
}

/// The lookup being applied.
#[derive(Clone, Copy)]
struct Active {
    index: u16,
    flags: LookupFlags,
    mask: u32,
    auto_joiner: bool,
    auto_non_joiner: bool,
    per_syllable: bool,
}

/// Whether a walk steps over a glyph.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Skip {
    No,
    Yes,
    /// Only when it is not the glyph being looked for.
    Maybe,
}

/// The glyphs a rule's input matched.
struct Matched {
    /// The place of each matched glyph, the first at the cursor.
    places: Vec<usize>,
    /// One past the last.
    end: usize,
    /// How many ligature parts they come to.
    parts: u8,
}

/// Applies the lookups of `GSUB` or of `GPOS`.
pub(crate) struct Engine<'f, 'a> {
    table: &'f LayoutTable<'a>,
    gdef: Option<&'f Gdef<'a>>,
    positioning: bool,
    /// The run is read right to left.
    backward: bool,
    lookups: Vec<Option<Rc<Parsed<'a>>>>,
    steps: usize,
    max_glyphs: usize,
    nesting: u8,
}

fn font_error(error: FontError) -> TextError {
    TextError::Font(error)
}

impl<'f, 'a> Engine<'f, 'a> {
    /// An engine for the face's substitutions, or for its positioning.
    /// `None` when the face has no such table. A face whose glyph kinds
    /// are not `trusted` has them guessed.
    pub(crate) fn new(
        font: &'f Font<'a>,
        positioning: bool,
        backward: bool,
        glyphs: usize,
        trusted: bool,
    ) -> Option<Self> {
        let table = if positioning {
            font.gpos()
        } else {
            font.gsub()
        }?;
        Some(Self {
            table,
            gdef: font.gdef().filter(|gdef| trusted && gdef.has_classes()),
            positioning,
            backward,
            lookups: vec![None; usize::from(table.lookup_count())],
            steps: glyphs.saturating_mul(1_024).max(16_384),
            max_glyphs: glyphs.saturating_mul(64).max(16_384),
            nesting: 0,
        })
    }

    fn parsed(&mut self, index: u16) -> Result<Rc<Parsed<'a>>, TextError> {
        let slot = self
            .lookups
            .get(usize::from(index))
            .ok_or(TextError::Font(FontError::BadTable))?;
        if let Some(parsed) = slot {
            return Ok(Rc::clone(parsed));
        }
        let lookup = self.table.lookup(index).map_err(font_error)?;
        let mut subtables = Vec::with_capacity(lookup.subtables.len());
        for data in &lookup.subtables {
            subtables.push(if self.positioning {
                Subtable::Positioning(Positioning::read(lookup.kind, data).map_err(font_error)?)
            } else {
                Subtable::Substitution(Substitution::read(lookup.kind, data).map_err(font_error)?)
            });
        }
        let parsed = Rc::new(Parsed {
            kind: lookup.kind,
            flags: lookup.flags,
            subtables,
        });
        if let Some(slot) = self.lookups.get_mut(usize::from(index)) {
            *slot = Some(Rc::clone(&parsed));
        }
        Ok(parsed)
    }

    fn step(&mut self) -> Result<(), TextError> {
        self.steps = self.steps.checked_sub(1).ok_or(TextError::Limit)?;
        Ok(())
    }

    /// The glyph's kind: what `GDEF` says, or `guess` without it.
    fn classify(&self, glyph: &mut Glyph, guess: Option<GlyphClass>) {
        if let Some(gdef) = self.gdef {
            glyph.class = gdef.class(glyph.id);
        } else if let Some(guess) = guess {
            glyph.class = guess;
        }
    }

    /// Whether the lookup's flags let it see `glyph`.
    fn sees(&self, glyph: &Glyph, flags: &LookupFlags) -> bool {
        match glyph.class {
            GlyphClass::Base if flags.ignore_bases => false,
            GlyphClass::Ligature if flags.ignore_ligatures => false,
            GlyphClass::Mark => {
                if flags.ignore_marks {
                    false
                } else if let Some(set) = flags.mark_set {
                    self.gdef
                        .is_some_and(|gdef| gdef.in_mark_set(set, glyph.id))
                } else if flags.mark_class != 0 {
                    self.gdef.is_some_and(|gdef| {
                        gdef.mark_class(glyph.id) == u16::from(flags.mark_class)
                    })
                } else {
                    true
                }
            }
            _ => true,
        }
    }

    fn skips(&self, glyph: &Glyph, active: &Active, context: bool) -> Skip {
        if !self.sees(glyph, &active.flags) {
            return Skip::Yes;
        }
        let over_non_joiner = self.positioning || (context && active.auto_non_joiner);
        let over_joiner = context || active.auto_joiner;
        if glyph.steps_over()
            && (over_non_joiner || !glyph.non_joiner)
            && (over_joiner || !glyph.joiner)
        {
            Skip::Maybe
        } else {
            Skip::No
        }
    }

    /// The next glyph from `from`, forward or back, that `wanted` accepts,
    /// stepping over the glyphs the lookup does not see. `wanted` gives
    /// `None` when any glyph will do. `syllable` holds the walk to one
    /// syllable when the lookup asks.
    #[allow(clippy::too_many_arguments)]
    fn seek(
        &self,
        buffer: &Buffer,
        from: usize,
        forward: bool,
        active: &Active,
        context: bool,
        syllable: u8,
        wanted: &mut dyn FnMut(&Glyph) -> Result<Option<bool>, FontError>,
    ) -> Result<Option<usize>, TextError> {
        let mut index = from;
        loop {
            index = if forward {
                index + 1
            } else {
                match index.checked_sub(1) {
                    Some(index) => index,
                    None => return Ok(None),
                }
            };
            let Some(glyph) = buffer.get(index) else {
                return Ok(None);
            };
            let skip = self.skips(glyph, active, context);
            if skip == Skip::Yes {
                continue;
            }
            let masked = !context && glyph.mask & active.mask == 0;
            let outside = active.per_syllable && syllable != 0 && glyph.syllable != syllable;
            let found = if masked || outside {
                Some(false)
            } else {
                wanted(glyph).map_err(font_error)?
            };
            match (found, skip) {
                (Some(true), _) | (None, Skip::No) => return Ok(Some(index)),
                (_, Skip::No) => return Ok(None),
                _ => {}
            }
        }
    }

    /// Match the glyphs after the cursor's against `count` more, each
    /// tested by `wanted` with its place in the rule.
    fn match_input(
        &self,
        buffer: &Buffer,
        active: &Active,
        count: usize,
        wanted: &dyn Fn(usize, u16) -> Result<bool, FontError>,
    ) -> Result<Option<Matched>, TextError> {
        if count + 1 > MAX_CONTEXT {
            return Ok(None);
        }
        let start = buffer.cursor();
        let Some(first) = buffer.current() else {
            return Ok(None);
        };
        let syllable = first.syllable;
        let (first_id, first_part) = (first.lig_id, first.lig_part);
        let mut parts = first.parts();
        let mut places = Vec::with_capacity(count + 1);
        places.push(start);
        // Whether the ligature the first glyph is a mark of may be
        // stepped over, found once when it is needed.
        let mut base_skips: Option<bool> = None;
        let mut at = start;
        for place in 0..count {
            let Some(next) =
                self.seek(buffer, at, true, active, false, syllable, &mut |glyph| {
                    wanted(place, glyph.id).map(Some)
                })?
            else {
                return Ok(None);
            };
            at = next;
            let Some(glyph) = buffer.get(at) else {
                return Ok(None);
            };
            let (this_id, this_part) = (glyph.lig_id, glyph.lig_part);
            if first_id != 0 && first_part != 0 {
                // The first glyph is a mark on part of a ligature: the
                // rest must be marks on that same part, unless the
                // ligature itself is one the lookup steps over.
                if first_id != this_id || first_part != this_part {
                    let skips = *base_skips.get_or_insert_with(|| {
                        let mut index = start;
                        while index > 0 {
                            let Some(before) = buffer.get(index - 1) else {
                                break;
                            };
                            if before.lig_id != first_id {
                                break;
                            }
                            index -= 1;
                            if before.lig_part == 0 {
                                return self.skips(before, active, false) == Skip::Yes;
                            }
                        }
                        false
                    });
                    if !skips {
                        return Ok(None);
                    }
                }
            } else if this_id != 0 && this_part != 0 && this_id != first_id {
                // The first glyph is on no ligature: neither may the rest
                // be, unless on the first glyph itself.
                return Ok(None);
            }
            parts = parts.saturating_add(glyph.parts());
            places.push(at);
        }
        Ok(Some(Matched {
            places,
            end: at + 1,
            parts,
        }))
    }

    /// Match the glyphs around the input: `rule`'s before the cursor and
    /// after glyph `last`.
    fn match_around(
        &self,
        buffer: &Buffer,
        active: &Active,
        rule: &SequenceRule<'a>,
        last: usize,
    ) -> Result<bool, TextError> {
        let syllable = buffer.current().map_or(0, |glyph| glyph.syllable);
        let mut at = buffer.cursor();
        for place in 0..rule.backtrack_len() {
            let Some(found) =
                self.seek(buffer, at, false, active, true, syllable, &mut |glyph| {
                    rule.backtrack_matches(place, glyph.id).map(Some)
                })?
            else {
                return Ok(false);
            };
            at = found;
        }
        let mut at = last;
        for place in 0..rule.lookahead_len() {
            let Some(found) =
                self.seek(buffer, at, true, active, true, syllable, &mut |glyph| {
                    rule.lookahead_matches(place, glyph.id).map(Some)
                })?
            else {
                return Ok(false);
            };
            at = found;
        }
        Ok(true)
    }

    /// Run `lookup` over the whole buffer.
    pub(crate) fn apply(
        &mut self,
        buffer: &mut Buffer,
        lookup: &LookupRef,
    ) -> Result<(), TextError> {
        let parsed = self.parsed(lookup.index)?;
        let active = Active {
            index: lookup.index,
            flags: parsed.flags,
            mask: lookup.mask,
            auto_joiner: lookup.auto_joiner,
            auto_non_joiner: lookup.auto_non_joiner,
            per_syllable: lookup.per_syllable,
        };
        if !self.positioning && parsed.kind == 8 {
            return self.apply_reverse(buffer, &parsed, &active);
        }
        buffer.move_to(0);
        while let Some(glyph) = buffer.current() {
            let at = buffer.cursor();
            let glyphs = buffer.len();
            let applies = glyph.mask & active.mask != 0 && self.sees(glyph, &active.flags);
            let next = if applies {
                self.step()?;
                self.apply_here(buffer, &parsed, &active)?
            } else {
                None
            };
            // A lookup that acted says where to go on from. The walk
            // always moves on, or the run gets shorter.
            let next = next
                .filter(|next| *next > at || buffer.len() < glyphs)
                .unwrap_or(at + 1);
            buffer.move_to(next);
            if buffer.len() > self.max_glyphs {
                return Err(TextError::Limit);
            }
        }
        Ok(())
    }

    /// Try the lookup's subtables at the cursor. Where one acts, the
    /// place to go on from.
    fn apply_here(
        &mut self,
        buffer: &mut Buffer,
        parsed: &Parsed<'a>,
        active: &Active,
    ) -> Result<Option<usize>, TextError> {
        for subtable in &parsed.subtables {
            let acted = match subtable {
                Subtable::Substitution(substitution) => {
                    self.substitute(buffer, substitution, active)?
                }
                Subtable::Positioning(positioning) => self.position(buffer, positioning, active)?,
            };
            if acted.is_some() {
                return Ok(acted);
            }
        }
        Ok(None)
    }

    /// A reverse chaining lookup: walked from the end, one glyph for one.
    fn apply_reverse(
        &mut self,
        buffer: &mut Buffer,
        parsed: &Parsed<'a>,
        active: &Active,
    ) -> Result<(), TextError> {
        for at in (0..buffer.len()).rev() {
            buffer.move_to(at);
            let Some(glyph) = buffer.current() else {
                continue;
            };
            if glyph.mask & active.mask == 0 || !self.sees(glyph, &active.flags) {
                continue;
            }
            self.step()?;
            let id = glyph.id;
            for subtable in &parsed.subtables {
                let Subtable::Substitution(Substitution::Reverse(reverse)) = subtable else {
                    continue;
                };
                let Some((rule, substitute)) = reverse.rule(id).map_err(font_error)? else {
                    continue;
                };
                if !self.match_around(buffer, active, &rule, at)? {
                    continue;
                }
                if let Some(glyph) = buffer.current_mut() {
                    glyph.id = substitute;
                    glyph.substituted = true;
                }
                self.reclassify_current(buffer, None);
                break;
            }
        }
        Ok(())
    }

    fn reclassify_current(&self, buffer: &mut Buffer, guess: Option<GlyphClass>) {
        if let Some(glyph) = buffer.current_mut() {
            self.classify(glyph, guess);
        }
    }

    fn substitute(
        &mut self,
        buffer: &mut Buffer,
        substitution: &Substitution<'a>,
        active: &Active,
    ) -> Result<Option<usize>, TextError> {
        let at = buffer.cursor();
        let Some(id) = buffer.current().map(|glyph| glyph.id) else {
            return Ok(None);
        };
        match substitution {
            Substitution::Single(single) => {
                let Some(substitute) = single.substitute(id).map_err(font_error)? else {
                    return Ok(None);
                };
                self.replace_current(buffer, substitute);
                Ok(Some(at + 1))
            }
            Substitution::Alternate(alternates) => {
                let Some(list) = alternates.glyphs(id).map_err(font_error)? else {
                    return Ok(None);
                };
                let Some(first) = list.get(0) else {
                    return Ok(None);
                };
                self.replace_current(buffer, first);
                Ok(Some(at + 1))
            }
            Substitution::Multiple(sequences) => {
                let Some(list) = sequences.glyphs(id).map_err(font_error)? else {
                    return Ok(None);
                };
                if list.len() == 1 {
                    if let Some(only) = list.get(0) {
                        self.replace_current(buffer, only);
                    }
                    return Ok(Some(at + 1));
                }
                let Some(original) = buffer.remove() else {
                    return Ok(None);
                };
                // A glyph written as nothing is taken out.
                let guess = (original.class == GlyphClass::Ligature).then_some(GlyphClass::Base);
                for (part, glyph_id) in list.iter().enumerate() {
                    let mut glyph = original.clone();
                    glyph.id = glyph_id;
                    glyph.substituted = true;
                    glyph.multiplied = true;
                    if glyph.lig_id == 0 {
                        glyph.lig_part = u8::try_from(part).unwrap_or(u8::MAX);
                    }
                    self.classify(&mut glyph, guess);
                    buffer.insert(glyph);
                }
                Ok(Some(buffer.cursor()))
            }
            Substitution::Ligature(ligatures) => {
                for ligature in ligatures.starting(id).map_err(font_error)? {
                    let rest = ligature.rest;
                    let wanted = |place: usize, glyph: u16| {
                        Ok(rest.get(place).is_some_and(|id| id == glyph))
                    };
                    let Some(matched) = self.match_input(buffer, active, rest.len(), &wanted)?
                    else {
                        continue;
                    };
                    if matched.places.len() == 1 {
                        self.replace_current(buffer, ligature.glyph);
                        return Ok(Some(at + 1));
                    }
                    return Ok(Some(self.ligate(buffer, &matched, ligature.glyph)));
                }
                Ok(None)
            }
            Substitution::Context(context) => {
                for rule in context.rules(id).map_err(font_error)? {
                    if let Some(end) = self.apply_rule(buffer, active, &rule)? {
                        return Ok(Some(end));
                    }
                }
                Ok(None)
            }
            Substitution::Reverse(_) => Ok(None),
        }
    }

    fn replace_current(&self, buffer: &mut Buffer, id: u16) {
        if let Some(glyph) = buffer.current_mut() {
            glyph.id = id;
            glyph.substituted = true;
        }
        self.reclassify_current(buffer, None);
    }

    /// Put `ligature` in place of the matched glyphs. Marks that sat
    /// between or after its parts stay, and remember which part they are
    /// on. Gives the place after the last part.
    fn ligate(&self, buffer: &mut Buffer, matched: &Matched, ligature: u16) -> usize {
        let class_at = |index: usize| buffer.get(index).map(|glyph| glyph.class);
        let all_marks = matched
            .places
            .iter()
            .all(|place| class_at(*place) == Some(GlyphClass::Mark));
        let first_base = matched
            .places
            .first()
            .is_some_and(|place| class_at(*place) == Some(GlyphClass::Base));
        let rest_marks = matched
            .places
            .iter()
            .skip(1)
            .all(|place| class_at(*place) == Some(GlyphClass::Mark));
        // A base with marks stays a base, and marks alone stay a mark.
        let is_ligature = !(all_marks || (first_base && rest_marks));
        let lig_id = if is_ligature { buffer.next_lig_id() } else { 0 };
        let Some(first) = buffer.current_mut() else {
            return matched.end;
        };
        let mut last_id = first.lig_id;
        let mut last_parts = first.parts();
        let mut so_far = last_parts;
        if is_ligature {
            first.lig_id = lig_id;
            first.lig_part = 0;
            first.lig_parts = matched.parts;
            if first.category == gc::NONSPACING_MARK {
                first.category = gc::OTHER_LETTER;
            }
        }
        first.id = ligature;
        first.substituted = true;
        first.ligated = true;
        first.multiplied = false;
        self.reclassify_current(buffer, is_ligature.then_some(GlyphClass::Ligature));
        if !all_marks {
            buffer.merge_clusters(buffer.cursor(), matched.end);
        }
        let on_part = |so_far: u8, last_parts: u8, part: u8| {
            let part = if part == 0 { last_parts } else { part };
            so_far - last_parts + part.min(last_parts)
        };
        buffer.move_to(buffer.cursor() + 1);
        for (removed, place) in matched.places.iter().skip(1).enumerate() {
            // Marks between two parts sit on the part before them.
            while buffer.cursor() < place - removed {
                if is_ligature && let Some(mark) = buffer.current_mut() {
                    mark.lig_part = on_part(so_far, last_parts, mark.lig_part);
                    mark.lig_id = lig_id;
                }
                buffer.move_to(buffer.cursor() + 1);
            }
            if let Some(part) = buffer.remove() {
                last_id = part.lig_id;
                last_parts = part.parts();
                so_far = so_far.saturating_add(last_parts);
            }
        }
        if !all_marks && last_id != 0 {
            // Marks after the last part were on a ligature it was.
            for index in buffer.cursor()..buffer.len() {
                let Some(mark) = buffer.get_mut(index) else {
                    break;
                };
                if mark.lig_id != last_id || mark.lig_part == 0 {
                    break;
                }
                mark.lig_part = on_part(so_far, last_parts, mark.lig_part);
                mark.lig_id = lig_id;
            }
        }
        buffer.cursor()
    }

    /// Try one contextual rule at the cursor; where it matches, run its
    /// lookups and give the place after its input.
    fn apply_rule(
        &mut self,
        buffer: &mut Buffer,
        active: &Active,
        rule: &SequenceRule<'a>,
    ) -> Result<Option<usize>, TextError> {
        self.step()?;
        let wanted = |place: usize, glyph: u16| rule.input_matches(place, glyph);
        let Some(matched) = self.match_input(buffer, active, rule.input_len(), &wanted)? else {
            return Ok(None);
        };
        if !self.match_around(buffer, active, rule, matched.end - 1)? {
            return Ok(None);
        }
        let mut places = matched.places;
        let mut end = matched.end;
        for (place, lookup) in rule.records() {
            let place = usize::from(place);
            let Some(at) = places.get(place).copied() else {
                continue;
            };
            // A rule does not run its own lookup where it started.
            if place == 0 && lookup == active.index {
                continue;
            }
            buffer.move_to(at);
            let before = buffer.len();
            if !self.nested(buffer, lookup, active)? {
                continue;
            }
            let after = buffer.len();
            if after == before {
                continue;
            }
            // The nested lookup changed how many glyphs there are: the
            // places after it move.
            let next = place + 1;
            if after > before {
                let grown = after - before;
                if places.len() + grown > MAX_CONTEXT {
                    break;
                }
                end += grown;
                for place in places.iter_mut().skip(next) {
                    *place += grown;
                }
                for step in 0..grown {
                    places.insert(next + step, at + 1 + step);
                }
            } else {
                let mut shrunk = before - after;
                // The end never moves back past the glyph just acted on.
                if end < at + shrunk {
                    shrunk = end.saturating_sub(at);
                    end = at;
                } else {
                    end -= shrunk;
                }
                shrunk = shrunk.min(places.len() - next);
                places.drain(next..next + shrunk);
                for place in places.iter_mut().skip(next) {
                    *place = place.saturating_sub(shrunk);
                }
            }
        }
        buffer.move_to(end);
        Ok(Some(end))
    }

    /// Run another lookup once, at the cursor, for a contextual rule.
    fn nested(
        &mut self,
        buffer: &mut Buffer,
        lookup: u16,
        outer: &Active,
    ) -> Result<bool, TextError> {
        if self.nesting >= MAX_NESTING {
            return Ok(false);
        }
        self.step()?;
        let parsed = self.parsed(lookup)?;
        if !self.positioning && parsed.kind == 8 {
            return Ok(false);
        }
        let active = Active {
            index: lookup,
            flags: parsed.flags,
            ..*outer
        };
        let at = buffer.cursor();
        self.nesting += 1;
        let acted = self.apply_here(buffer, &parsed, &active);
        self.nesting -= 1;
        let acted = acted?.is_some();
        // A positioning lookup leaves the cursor; a substitution may have
        // moved it past what it wrote.
        if self.positioning {
            buffer.move_to(at);
        }
        Ok(acted)
    }

    fn position(
        &mut self,
        buffer: &mut Buffer,
        positioning: &Positioning<'a>,
        active: &Active,
    ) -> Result<Option<usize>, TextError> {
        let at = buffer.cursor();
        let Some(current) = buffer.current() else {
            return Ok(None);
        };
        let id = current.id;
        match positioning {
            Positioning::Single(single) => {
                let Some(adjust) = single.adjust(id).map_err(font_error)? else {
                    return Ok(None);
                };
                if let Some(glyph) = buffer.current_mut() {
                    glyph.x_offset += i32::from(adjust.x);
                    glyph.y_offset += i32::from(adjust.y);
                    glyph.x_advance += i32::from(adjust.x_advance);
                }
                Ok(Some(at + 1))
            }
            Positioning::Pair(pairs) => {
                let syllable = current.syllable;
                let Some(second) =
                    self.seek(buffer, at, true, active, false, syllable, &mut |_| Ok(None))?
                else {
                    return Ok(None);
                };
                let Some(second_id) = buffer.get(second).map(|glyph| glyph.id) else {
                    return Ok(None);
                };
                let Some((one, two)) = pairs.adjust(id, second_id).map_err(font_error)? else {
                    return Ok(None);
                };
                if let Some(glyph) = buffer.get_mut(at) {
                    glyph.x_offset += i32::from(one.x);
                    glyph.y_offset += i32::from(one.y);
                    glyph.x_advance += i32::from(one.x_advance);
                }
                if let Some(glyph) = buffer.get_mut(second) {
                    glyph.x_offset += i32::from(two.x);
                    glyph.y_offset += i32::from(two.y);
                    glyph.x_advance += i32::from(two.x_advance);
                    // A line cut between the two takes the advances back.
                    glyph.pair = Pair {
                        back: second - at,
                        first_advance: glyph.pair.first_advance + i32::from(one.x_advance),
                        x_offset: glyph.pair.x_offset + i32::from(two.x),
                        advance: glyph.pair.advance + i32::from(two.x_advance),
                    };
                }
                Ok(Some(if pairs.moves_second() {
                    second + 1
                } else {
                    second
                }))
            }
            Positioning::Cursive(cursive) => self.join(buffer, cursive, active),
            Positioning::MarkToBase(marks) => {
                if !marks.places(id) {
                    return Ok(None);
                }
                let Some(base) = self.base_before(buffer, at, active, &|glyph| marks.holds(glyph))
                else {
                    return Ok(None);
                };
                let Some(base_id) = buffer.get(base).map(|glyph| glyph.id) else {
                    return Ok(None);
                };
                let anchors = marks.anchors(id, base_id).map_err(font_error)?;
                Ok(Self::hang(buffer, at, base, anchors))
            }
            Positioning::MarkToLigature(marks) => {
                if !marks.places(id) {
                    return Ok(None);
                }
                let Some(base) = self.base_before(buffer, at, active, &|_| false) else {
                    return Ok(None);
                };
                let (Some(mark), Some(ligature)) = (buffer.get(at), buffer.get(base)) else {
                    return Ok(None);
                };
                if !marks.holds(ligature.id) {
                    return Ok(None);
                }
                let Some(parts) = marks.ligature_parts(ligature.id).map_err(font_error)? else {
                    return Ok(None);
                };
                if parts == 0 {
                    return Ok(None);
                }
                // A mark that remembers its part of this ligature sits on
                // it; any other sits on the last.
                let part = if ligature.lig_id != 0
                    && ligature.lig_id == mark.lig_id
                    && mark.lig_part > 0
                {
                    parts.min(usize::from(mark.lig_part)) - 1
                } else {
                    parts - 1
                };
                let anchors = marks
                    .ligature_anchors(id, ligature.id, part)
                    .map_err(font_error)?;
                Ok(Self::hang(buffer, at, base, anchors))
            }
            Positioning::MarkToMark(marks) => {
                if !marks.places(id) {
                    return Ok(None);
                }
                // The glyph before, with only the lookup's choice of marks
                // deciding what is stepped over.
                let only_marks = Active {
                    flags: LookupFlags {
                        ignore_bases: false,
                        ignore_ligatures: false,
                        ignore_marks: false,
                        ..active.flags
                    },
                    ..*active
                };
                let Some(before) =
                    self.seek(buffer, at, false, &only_marks, false, 0, &mut |_| Ok(None))?
                else {
                    return Ok(None);
                };
                let (Some(mark), Some(under)) = (buffer.get(at), buffer.get(before)) else {
                    return Ok(None);
                };
                if !under.is_mark() {
                    return Ok(None);
                }
                // Both on the same base, or on the same part of the same
                // ligature, or one of them is itself made of marks.
                let together = if mark.lig_id == under.lig_id {
                    mark.lig_id == 0 || mark.lig_part == under.lig_part
                } else {
                    (mark.lig_id > 0 && mark.lig_part == 0)
                        || (under.lig_id > 0 && under.lig_part == 0)
                };
                if !together {
                    return Ok(None);
                }
                let anchors = marks.anchors(id, under.id).map_err(font_error)?;
                Ok(Self::hang(buffer, at, before, anchors))
            }
            Positioning::Context(context) => {
                for rule in context.rules(id).map_err(font_error)? {
                    if let Some(end) = self.apply_rule(buffer, active, &rule)? {
                        return Ok(Some(end));
                    }
                }
                Ok(None)
            }
        }
    }

    /// The glyph a mark at `at` sits on: the nearest before it that is
    /// not a mark. One of several glyphs written for one character is
    /// passed over unless it is the first of them or `holds` it. A joiner
    /// the lookup's feature reads for itself is not passed over: it
    /// stands between the mark and the letter, and holds no mark.
    fn base_before(
        &self,
        buffer: &Buffer,
        at: usize,
        active: &Active,
        holds: &dyn Fn(u16) -> bool,
    ) -> Option<usize> {
        let mut index = at;
        while index > 0 {
            index -= 1;
            let glyph = buffer.get(index)?;
            let stands = glyph.joiner && !active.auto_joiner;
            if glyph.is_mark() || (glyph.steps_over() && !stands) {
                continue;
            }
            let first_of_many = !glyph.multiplied
                || glyph.lig_part == 0
                || index == 0
                || buffer.get(index - 1).is_none_or(|before| {
                    before.is_mark()
                        || !before.multiplied
                        || before.lig_id != glyph.lig_id
                        || before.lig_part + 1 != glyph.lig_part
                });
            if first_of_many || holds(glyph.id) {
                return Some(index);
            }
        }
        None
    }

    /// Set the mark at `at` on the glyph at `on` by their anchors.
    fn hang(
        buffer: &mut Buffer,
        at: usize,
        on: usize,
        anchors: Option<(Anchor, Anchor)>,
    ) -> Option<usize> {
        let (mark_anchor, base_anchor) = anchors?;
        let mark = buffer.get_mut(at)?;
        mark.x_offset = i32::from(base_anchor.x) - i32::from(mark_anchor.x);
        mark.y_offset = i32::from(base_anchor.y) - i32::from(mark_anchor.y);
        mark.attach = Attach::Mark;
        mark.attach_to = i32::try_from(on).ok()? - i32::try_from(at).ok()?;
        Some(at + 1)
    }

    /// Join the glyph at the cursor to the one before it, where the first
    /// leaves and the second is entered.
    fn join(
        &self,
        buffer: &mut Buffer,
        cursive: &cutlass_font::Cursive<'a>,
        active: &Active,
    ) -> Result<Option<usize>, TextError> {
        let at = buffer.cursor();
        let Some(current) = buffer.current() else {
            return Ok(None);
        };
        let Some(entry) = cursive
            .anchors(current.id)
            .map_err(font_error)?
            .and_then(|joins| joins.entry)
        else {
            return Ok(None);
        };
        let Some(before) = self.seek(buffer, at, false, active, false, 0, &mut |_| Ok(None))?
        else {
            return Ok(None);
        };
        let Some(before_id) = buffer.get(before).map(|glyph| glyph.id) else {
            return Ok(None);
        };
        let Some(exit) = cursive
            .anchors(before_id)
            .map_err(font_error)?
            .and_then(|joins| joins.exit)
        else {
            return Ok(None);
        };
        let (exit_x, exit_y) = (i32::from(exit.x), i32::from(exit.y));
        let (entry_x, entry_y) = (i32::from(entry.x), i32::from(entry.y));
        // Along the line, the first glyph's advance ends at its exit and
        // the second starts at its entry.
        if self.backward {
            if let Some(first) = buffer.get_mut(before) {
                let shift = exit_x + first.x_offset;
                first.x_advance -= shift;
                first.x_offset -= shift;
            }
            if let Some(second) = buffer.get_mut(at) {
                second.x_advance = entry_x + second.x_offset;
            }
        } else {
            if let Some(first) = buffer.get_mut(before) {
                first.x_advance = exit_x + first.x_offset;
            }
            if let Some(second) = buffer.get_mut(at) {
                let shift = entry_x + second.x_offset;
                second.x_advance -= shift;
                second.x_offset -= shift;
            }
        }
        // Across the line, one of the two hangs on the other: the later
        // one, unless the lookup says the line is anchored at its end.
        let (child, parent, rise) = if active.flags.right_to_left {
            (before, at, entry_y - exit_y)
        } else {
            (at, before, exit_y - entry_y)
        };
        unhang_cursive(buffer, child, parent);
        let offset = i32::try_from(parent).unwrap_or(0) - i32::try_from(child).unwrap_or(0);
        if let Some(glyph) = buffer.get_mut(child) {
            glyph.attach = Attach::Cursive;
            glyph.attach_to = offset;
            glyph.y_offset = rise;
        }
        // If the parent hung on the child, it is free now.
        if let Some(glyph) = buffer.get_mut(parent)
            && glyph.attach == Attach::Cursive
            && glyph.attach_to == -offset
        {
            glyph.attach = Attach::None;
            glyph.attach_to = 0;
            glyph.y_offset = 0;
        }
        Ok(Some(at + 1))
    }
}

/// Turn around the chain of cursive joins that starts at `index`, so that
/// it can hang on `new_parent` instead.
fn unhang_cursive(buffer: &mut Buffer, index: usize, new_parent: usize) {
    let mut chain = Vec::new();
    let mut at = index;
    // Walk the chain, then turn each link around from its far end.
    while let Some(glyph) = buffer.get(at) {
        if glyph.attach != Attach::Cursive || glyph.attach_to == 0 || chain.len() > 4_096 {
            break;
        }
        let Some(next) =
            usize::try_from(i64::try_from(at).unwrap_or(0) + i64::from(glyph.attach_to)).ok()
        else {
            break;
        };
        chain.push((at, next, glyph.attach_to, glyph.y_offset));
        if next == new_parent {
            break;
        }
        at = next;
    }
    for (from, _, _, _) in &chain {
        if let Some(glyph) = buffer.get_mut(*from) {
            glyph.attach = Attach::None;
            glyph.attach_to = 0;
        }
    }
    for (_, to, offset, rise) in chain.iter().rev() {
        if *to == new_parent {
            continue;
        }
        if let Some(glyph) = buffer.get_mut(*to) {
            glyph.y_offset = -rise;
            glyph.attach = Attach::Cursive;
            glyph.attach_to = -offset;
        }
    }
}

/// After positioning: a glyph that hangs on another takes its place from
/// it. A mark's offset is measured from where its base starts, so the
/// advances between the two are taken off, or added when the run is read
/// right to left.
pub(crate) fn settle(glyphs: &mut [Glyph], backward: bool) {
    for index in 0..glyphs.len() {
        settle_one(glyphs, index, backward, 64);
    }
}

fn settle_one(glyphs: &mut [Glyph], index: usize, backward: bool, depth: u8) {
    let Some(glyph) = glyphs.get(index) else {
        return;
    };
    let (attach, offset) = (glyph.attach, glyph.attach_to);
    if attach == Attach::None || offset == 0 {
        return;
    }
    glyphs[index].attach = Attach::None;
    glyphs[index].attach_to = 0;
    let Some(parent) = i64::try_from(index)
        .ok()
        .and_then(|index| usize::try_from(index + i64::from(offset)).ok())
        .filter(|parent| *parent < glyphs.len())
    else {
        return;
    };
    if depth == 0 {
        return;
    }
    settle_one(glyphs, parent, backward, depth - 1);
    let (parent_x, parent_y) = (glyphs[parent].x_offset, glyphs[parent].y_offset);
    if attach == Attach::Cursive {
        glyphs[index].y_offset += parent_y;
        return;
    }
    glyphs[index].x_offset += parent_x;
    glyphs[index].y_offset += parent_y;
    if parent >= index {
        return;
    }
    let between: i32 = if backward {
        glyphs[parent + 1..=index]
            .iter()
            .map(|glyph| glyph.x_advance)
            .sum()
    } else {
        -glyphs[parent..index]
            .iter()
            .map(|glyph| glyph.x_advance)
            .sum::<i32>()
    };
    glyphs[index].x_offset += between;
}
