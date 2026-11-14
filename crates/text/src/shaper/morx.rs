//! Runs the state machines of a face made for Apple's layout engine.
//!
//! Such a face has no OpenType substitutions: its ligatures, its joined
//! letter forms and its emoji sequences are made by the machines in its
//! `morx` table. Each subtable is one machine. The run is walked glyph by
//! glyph; the glyph's class and the machine's state pick an entry, the
//! entry acts, and the state moves on. A machine may stay on a glyph
//! instead of moving to the next, so each walk has a budget of steps.

use cutlass_font::morx::{
    CLASS_END, DELETED_GLYPH, Metamorphosis, StateTable, long_at, substitute, word_at,
};
use cutlass_font::{Font, FontError};

use super::buffer::Glyph;
use crate::TextError;

/// An entry flag: stay on this glyph for the next step.
const STAY: u16 = 0x4000;
/// The most glyphs a ligature is made of, and a moved stretch holds.
const MAX_STRETCH: usize = 64;

fn font_error(error: FontError) -> TextError {
    TextError::Font(error)
}

fn word(entry: &[u8], at: usize) -> u16 {
    entry
        .get(at..at + 2)
        .map_or(0, |bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
}

/// A glyph a machine puts in place of another is not marked as put there:
/// a character that draws nothing still draws nothing whatever glyph its
/// machines give it, which is how these faces are read elsewhere.
///
/// Run every chain of the face's `morx` table over `glyphs`, which are in
/// the order the text is stored. `backward` is the direction the run is
/// read in.
pub(crate) fn apply(
    font: &Font<'_>,
    mut glyphs: Vec<Glyph>,
    backward: bool,
) -> Result<Vec<Glyph>, TextError> {
    let Some(table) = font.morx() else {
        return Ok(glyphs);
    };
    let most = glyphs.len().saturating_mul(8).max(1_024);
    let mut steps = glyphs.len().saturating_mul(256).max(16_384);
    for chain in table.chains().map_err(font_error)? {
        for subtable in &chain.subtables {
            if subtable.flags & chain.default_flags == 0
                || (subtable.vertical && !subtable.any_direction)
            {
                continue;
            }
            // A subtable walks the glyphs as they are seen, or as they
            // are stored, from either end.
            let reverse = if subtable.logical {
                subtable.backward
            } else {
                subtable.backward != backward
            };
            if reverse {
                glyphs.reverse();
            }
            match subtable.kind {
                Metamorphosis::Swap(lookup) => {
                    for glyph in &mut glyphs {
                        if let Some(other) = lookup.value(glyph.id) {
                            glyph.id = other;
                        }
                    }
                }
                Metamorphosis::Rearrange(machine) => {
                    rearrange(&machine, &mut glyphs, &mut steps)?;
                }
                Metamorphosis::Contextual { machine, tables } => {
                    contextual(&machine, tables, &mut glyphs, &mut steps)?;
                }
                Metamorphosis::Ligature {
                    machine,
                    actions,
                    components,
                    ligatures,
                } => {
                    let parts = Ligatures {
                        actions,
                        components,
                        ligatures,
                    };
                    ligate(&machine, &parts, &mut glyphs, &mut steps)?;
                }
                Metamorphosis::Insert {
                    machine,
                    glyphs: list,
                } => {
                    insert(&machine, list, &mut glyphs, &mut steps, most)?;
                }
            }
            if reverse {
                glyphs.reverse();
            }
        }
    }
    // A glyph a machine deleted stays, as the glyph that marks it: the
    // caller takes those out, and gives their place in the text away.
    Ok(glyphs)
}

/// Walk `glyphs` through `machine`. `act` is given the place in the run
/// and the entry, and may move the place; it gives back the place to go
/// on from.
fn walk(
    machine: &StateTable<'_>,
    glyphs: &mut Vec<Glyph>,
    steps: &mut usize,
    mut act: impl FnMut(&mut Vec<Glyph>, usize, &[u8]) -> Result<usize, TextError>,
) -> Result<(), TextError> {
    let mut state = 0_u16;
    let mut at = 0;
    loop {
        let class = glyphs
            .get(at)
            .map_or(CLASS_END, |glyph| machine.class(glyph.id));
        let entry = machine.entry(state, class).map_err(font_error)?;
        at = act(glyphs, at, entry)?;
        state = word(entry, 0);
        if at >= glyphs.len() {
            return Ok(());
        }
        *steps = steps.checked_sub(1).ok_or(TextError::Limit)?;
        // A machine that stays put too long is moved on.
        if word(entry, 2) & STAY == 0 || *steps < glyphs.len() {
            at += 1;
        }
    }
}

/// Give the glyphs from `start` up to `end` the earliest of their
/// clusters, and with them the glyphs on either side that share a cluster
/// with the first or the last of them: a mark that goes with the last
/// part of a ligature stays with the ligature.
fn merge(glyphs: &mut [Glyph], start: usize, end: usize) {
    let (mut start, mut end) = (start, end.min(glyphs.len()));
    if end.saturating_sub(start) < 2 {
        return;
    }
    let Some(first) = glyphs[start..end].iter().map(|glyph| glyph.cluster).min() else {
        return;
    };
    while start > 0 && glyphs[start - 1].cluster == glyphs[start].cluster {
        start -= 1;
    }
    while end < glyphs.len() && glyphs[end - 1].cluster == glyphs[end].cluster {
        end += 1;
    }
    for glyph in &mut glyphs[start..end] {
        glyph.cluster = first;
    }
}

/// Move glyphs at the two ends of a marked stretch to its other end.
fn rearrange(
    machine: &StateTable<'_>,
    glyphs: &mut Vec<Glyph>,
    steps: &mut usize,
) -> Result<(), TextError> {
    // For each verb: how many glyphs move from the start to the end and
    // from the end to the start. Three is two, turned around.
    const MOVES: [(usize, usize); 16] = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (3, 0),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
        (2, 3),
        (3, 3),
    ];
    let (mut start, mut end) = (0, 0);
    walk(machine, glyphs, steps, |glyphs, at, entry| {
        let flags = word(entry, 2);
        if flags & 0x8000 != 0 {
            start = at;
        }
        if flags & 0x2000 != 0 {
            end = (at + 1).min(glyphs.len());
        }
        let (from_start, from_end) = MOVES[usize::from(flags & 0x000F)];
        let (left, right) = (from_start.min(2), from_end.min(2));
        if (from_start, from_end) != (0, 0)
            && start < end
            && end <= glyphs.len()
            && end - start >= left + right
            && end - start <= MAX_STRETCH
        {
            merge(glyphs, start, (at + 1).max(end));
            let mut head: Vec<Glyph> = glyphs[start..start + left].to_vec();
            let mut tail: Vec<Glyph> = glyphs[end - right..end].to_vec();
            if from_start == 3 {
                head.reverse();
            }
            if from_end == 3 {
                tail.reverse();
            }
            let middle: Vec<Glyph> = glyphs[start + left..end - right].to_vec();
            let moved = tail.into_iter().chain(middle).chain(head);
            for (slot, glyph) in glyphs[start..end].iter_mut().zip(moved) {
                *slot = glyph;
            }
        }
        Ok(at)
    })
}

/// Swap a marked glyph and the current one for the glyphs the entry's
/// tables give.
fn contextual(
    machine: &StateTable<'_>,
    tables: &[u8],
    glyphs: &mut Vec<Glyph>,
    steps: &mut usize,
) -> Result<(), TextError> {
    let mut mark: Option<usize> = None;
    walk(machine, glyphs, steps, |glyphs, at, entry| {
        let (flags, for_mark, for_current) = (word(entry, 2), word(entry, 4), word(entry, 6));
        // At the end of the run nothing is swapped unless a mark was set.
        if at >= glyphs.len() && mark.is_none() {
            return Ok(at);
        }
        let swap = |glyphs: &mut [Glyph], index: Option<usize>, table: u16| {
            let Some(glyph) = index.and_then(|index| glyphs.get_mut(index)) else {
                return Ok(());
            };
            if table != 0xFFFF
                && let Some(other) = substitute(tables, table, glyph.id).map_err(font_error)?
            {
                glyph.id = other;
            }
            Ok::<(), TextError>(())
        };
        swap(glyphs, mark, for_mark)?;
        let current = at.min(glyphs.len().saturating_sub(1));
        swap(glyphs, Some(current), for_current)?;
        if flags & 0x8000 != 0 {
            mark = Some(at);
        }
        Ok(at)
    })
}

/// The lists a ligature subtable keeps beside its machine.
struct Ligatures<'a> {
    actions: &'a [u8],
    components: &'a [u8],
    ligatures: &'a [u8],
}

/// Make one glyph of the glyphs the machine marks as its parts. The
/// first part becomes the ligature and the rest are taken out.
fn ligate(
    machine: &StateTable<'_>,
    parts: &Ligatures<'_>,
    glyphs: &mut Vec<Glyph>,
    steps: &mut usize,
) -> Result<(), TextError> {
    const LAST: u32 = 0x8000_0000;
    const STORE: u32 = 0x4000_0000;
    let mut marked: Vec<usize> = Vec::new();
    walk(machine, glyphs, steps, |glyphs, at, entry| {
        let flags = word(entry, 2);
        if flags & 0x8000 != 0 {
            // A glyph the machine stayed on is not marked twice.
            if marked.last() == Some(&at) {
                marked.pop();
            }
            if marked.len() == MAX_STRETCH {
                marked.remove(0);
            }
            marked.push(at);
        }
        if flags & 0x2000 == 0 || marked.is_empty() || at >= glyphs.len() {
            return Ok(at);
        }
        let mut action_at = usize::from(word(entry, 4));
        let mut cursor = marked.len();
        let mut sum = 0_u32;
        loop {
            if cursor == 0 {
                marked.clear();
                break;
            }
            cursor -= 1;
            let place = marked[cursor];
            let action = long_at(parts.actions, action_at).map_err(font_error)?;
            action_at += 1;
            // The low thirty bits are a signed offset added to the glyph.
            let mut offset = action & 0x3FFF_FFFF;
            if offset & 0x2000_0000 != 0 {
                offset |= 0xC000_0000;
            }
            let Some(id) = glyphs.get(place).map(|glyph| glyph.id) else {
                break;
            };
            let component = u32::from(id).wrapping_add(offset);
            let component = usize::try_from(component).map_err(|_| TextError::Limit)?;
            sum = sum.wrapping_add(u32::from(
                word_at(parts.components, component).map_err(font_error)?,
            ));
            if action & (STORE | LAST) != 0 {
                let ligature = usize::try_from(sum).map_err(|_| TextError::Limit)?;
                let ligature = word_at(parts.ligatures, ligature).map_err(font_error)?;
                let end = marked.last().map_or(place, |last| *last) + 1;
                if let Some(glyph) = glyphs.get_mut(place) {
                    glyph.id = ligature;
                    glyph.ligated = true;
                }
                // The parts after the first are taken out of the run.
                while marked.len() - 1 > cursor {
                    if let Some(gone) = marked.pop().and_then(|gone| glyphs.get_mut(gone)) {
                        gone.id = DELETED_GLYPH;
                    }
                }
                merge(glyphs, place, end);
            }
            if action & LAST != 0 {
                break;
            }
        }
        Ok(at)
    })
}

/// Put glyphs in beside a marked glyph and beside the current one.
fn insert(
    machine: &StateTable<'_>,
    list: &[u8],
    glyphs: &mut Vec<Glyph>,
    steps: &mut usize,
    most: usize,
) -> Result<(), TextError> {
    let mut mark = 0;
    walk(machine, glyphs, steps, |glyphs, at, entry| {
        let (flags, for_current, for_mark) = (word(entry, 2), word(entry, 4), word(entry, 6));
        let mut at = at;
        // Put `count` glyphs of the list, from `first`, before or after
        // the glyph at `place`. They take that glyph's place in the text.
        let put =
            |glyphs: &mut Vec<Glyph>, place: usize, first: u16, count: usize, before: bool| {
                let Some(template) = glyphs.get(place).or(glyphs.last()).cloned() else {
                    return Ok(0);
                };
                if glyphs.len() + count > most {
                    return Err(TextError::Limit);
                }
                let from = if before || place >= glyphs.len() {
                    place.min(glyphs.len())
                } else {
                    place + 1
                };
                for step in 0..count {
                    let mut glyph = template.clone();
                    glyph.id = word_at(list, usize::from(first) + step).map_err(font_error)?;
                    glyphs.insert(from + step, glyph);
                }
                Ok(count)
            };
        let before = at;
        if for_mark != 0xFFFF {
            let count = usize::from(flags & 0x001F);
            at += put(glyphs, mark, for_mark, count, flags & 0x0400 != 0)?;
        }
        if flags & 0x8000 != 0 {
            mark = before;
        }
        if for_current != 0xFFFF {
            let count = usize::from((flags & 0x03E0) >> 5);
            let added = put(glyphs, at, for_current, count, flags & 0x0800 != 0)?;
            // Staying, the machine sees the first glyph put in; moving
            // on, it passes them all.
            if flags & STAY == 0 {
                at += added;
            }
        }
        Ok(at)
    })
}
