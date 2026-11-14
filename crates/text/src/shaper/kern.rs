//! Kerning by a `kern` or `kerx` table: pairs of glyphs, and the passes of
//! a face made for Apple's layout engine.
//!
//! Such a face kerns pass by pass. A pass of pairs moves each pair of
//! glyphs that are not marks. A pass that is a state machine reads the
//! glyphs in order, remembers some, and on some steps moves the ones it
//! remembered. A pass across the line lifts or lowers a glyph, and every
//! glyph after it with it, until a value sets a glyph back on the line:
//! this is how these faces raise a mark over a tall letter.

use cutlass_font::morx::DELETED_GLYPH;
use cutlass_font::{KernAttach, KernMachine, KernPass};

use super::buffer::{Attach, Glyph};
use crate::shape::Pair;

/// How many glyphs a machine remembers at once.
const STACK: usize = 8;

/// The value that sets a glyph back on the line.
const BACK_ON_THE_LINE: i32 = -0x8000;

/// Where a numbered point of a glyph's outline is, in font units: the
/// glyph, then the number.
pub(crate) type PointOf<'p> = &'p dyn Fn(u16, u16) -> Option<(i32, i32)>;

/// Apply `passes` in order to `glyphs`, which are in the order they are
/// seen; `backward` when they read right to left. `point` gives the
/// points a pass may set one glyph on another by.
pub(crate) fn apply(
    passes: &[KernPass<'_>],
    glyphs: &mut [Glyph],
    backward: bool,
    point: PointOf<'_>,
) {
    let mut chained = false;
    for pass in passes {
        // Before the first pass across the line, every glyph is hung on
        // the one read before it, so a lift carries along the line.
        if !chained && pass.cross_stream() {
            chained = true;
            for glyph in glyphs.iter_mut() {
                glyph.attach = Attach::Cursive;
                glyph.attach_to = if backward { 1 } else { -1 };
            }
        }
        // A pass reads the glyphs in the order they are read.
        let reverse = pass.backwards() != backward;
        if reverse {
            glyphs.reverse();
        }
        match pass {
            KernPass::Pairs(table) => {
                if !table.backwards {
                    // These tables know nothing of marks: a pair is two
                    // glyphs side by side, whatever they are. Core Text
                    // reads them so; HarfBuzz 14 steps over marks here.
                    pairs(
                        glyphs,
                        table.cross_stream,
                        !backward,
                        false,
                        |left, right| table.get(left, right),
                    );
                }
            }
            KernPass::Machine(machine) => run(machine, glyphs),
            KernPass::Attach(table) => attach(table, glyphs, reverse, point),
        }
        if reverse {
            glyphs.reverse();
        }
    }
    // A glyph an earlier machine deleted takes no room and is not moved
    // itself, though a lift still passes through it.
    for glyph in glyphs.iter_mut().filter(|glyph| glyph.id == DELETED_GLYPH) {
        glyph.x_advance = 0;
        glyph.x_offset = 0;
        glyph.y_offset = 0;
    }
    if !chained {
        return;
    }
    // Each glyph still hung on the one read before it is lifted with it,
    // from the first glyph read to the last.
    let count = glyphs.len();
    for step in 1..count {
        let (index, before) = if backward {
            (count - 1 - step, count - step)
        } else {
            (step, step - 1)
        };
        if glyphs[index].attach == Attach::Cursive {
            glyphs[index].y_offset += glyphs[before].y_offset;
        }
    }
    for glyph in glyphs
        .iter_mut()
        .filter(|glyph| glyph.attach == Attach::Cursive)
    {
        glyph.attach = Attach::None;
        glyph.attach_to = 0;
    }
}

/// Walk `glyphs` through a machine that sets glyphs on a remembered one.
/// `reversed` when the glyphs were turned around for the pass, and will
/// be turned back.
fn attach(table: &KernAttach<'_>, glyphs: &mut [Glyph], reversed: bool, point: PointOf<'_>) {
    let mut marked: Option<usize> = None;
    let mut state = 0;
    let mut index = 0;
    let mut again = glyphs.len().saturating_mul(4).saturating_add(64);
    loop {
        let class = table.class(glyphs.get(index).map(|glyph| glyph.id));
        let Ok(step) = table.step(state, class) else {
            return;
        };
        if let (Some(base), Some(action)) = (marked, step.action)
            && let Some(on) = glyphs.get(base).map(|glyph| glyph.id)
            && let Some(glyph) = glyphs.get_mut(index)
            && let Some((x, y)) = table.offset(action, on, glyph.id).or_else(|| {
                // By a numbered point of each outline: the glyph's own
                // goes where the remembered glyph's is.
                let (of_marked, of_glyph) = table.point_numbers(action)?;
                let (marked, own) = (point(on, of_marked)?, point(glyph.id, of_glyph)?);
                Some((marked.0 - own.0, marked.1 - own.1))
            })
        {
            glyph.x_offset = x;
            glyph.y_offset = y;
            glyph.attach = Attach::Mark;
            // How far back the remembered glyph is, in the order the
            // glyphs are kept in.
            let back = i32::try_from(index.abs_diff(base)).unwrap_or(0);
            glyph.attach_to = if (base < index) != reversed {
                -back
            } else {
                back
            };
        }
        if step.mark {
            marked = Some(index);
        }
        state = step.next;
        if index >= glyphs.len() {
            return;
        }
        if step.stay && again > 0 {
            again -= 1;
        } else {
            index += 1;
        }
    }
}

/// Kern each pair of glyphs by what `get` gives for them: along the line
/// the adjustment is shared between the two, and across it the second
/// glyph is moved. `over_marks` makes a pair of two glyphs with only marks
/// and glyphs that draw nothing between them, as an OpenType face's
/// `kern` table is read. `remember` keeps what was done to each pair, so a
/// line cut between the two can take it back.
pub(crate) fn pairs(
    glyphs: &mut [Glyph],
    cross_stream: bool,
    remember: bool,
    over_marks: bool,
    get: impl Fn(u16, u16) -> i32,
) {
    // A glyph a state machine deleted is never one of a pair: Core Text
    // kerns across it, where HarfBuzz 14 takes it for a glyph.
    let passed = |glyph: &Glyph| {
        glyph.id == DELETED_GLYPH || (over_marks && (glyph.is_mark() || glyph.steps_over()))
    };
    let mut first = 0;
    while first < glyphs.len() {
        if passed(&glyphs[first]) {
            first += 1;
            continue;
        }
        let Some(second) = (first + 1..glyphs.len()).find(|index| !passed(&glyphs[*index])) else {
            break;
        };
        let kerning = get(glyphs[first].id, glyphs[second].id);
        if kerning != 0 && cross_stream {
            glyphs[second].y_offset = kerning;
        } else if kerning != 0 {
            let half = kerning >> 1;
            let rest = kerning - half;
            glyphs[first].x_advance += half;
            glyphs[second].x_advance += rest;
            glyphs[second].x_offset += rest;
            if remember {
                let back = second - first;
                let before = glyphs[second].pair;
                // A later pass may kern the same pair again.
                glyphs[second].pair = if before.back == back {
                    Pair {
                        back,
                        first_advance: before.first_advance + half,
                        x_offset: before.x_offset + rest,
                        advance: before.advance + rest,
                    }
                } else {
                    Pair {
                        back,
                        first_advance: half,
                        x_offset: rest,
                        advance: rest,
                    }
                };
            }
        }
        first = second;
    }
}

/// Walk `glyphs` through `machine`. A machine whose table does not read
/// stops where it is, with what it has done.
fn run(machine: &KernMachine<'_>, glyphs: &mut [Glyph]) {
    let mut stack = [0_usize; STACK];
    let mut depth = 0;
    let mut state = 0;
    let mut index = 0;
    // A machine may read a glyph again and again; it is given this many
    // such steps for the whole run.
    let mut again = glyphs.len().saturating_mul(4).saturating_add(64);
    loop {
        let class = machine.class(glyphs.get(index).map(|glyph| glyph.id));
        let Ok(step) = machine.step(state, class) else {
            return;
        };
        if step.reset {
            depth = 0;
        }
        if step.push {
            if depth < STACK {
                stack[depth] = index;
                depth += 1;
            } else {
                depth = 0;
            }
        }
        if let Some(values) = step.values
            && depth > 0
        {
            // The table must hold a value for every glyph remembered.
            if machine.value(values, depth - 1).is_none() {
                depth = 0;
            }
            let mut nth = 0;
            let mut last = false;
            while !last && depth > 0 {
                depth -= 1;
                let at = stack[depth];
                let Some(value) = machine.value(values, nth) else {
                    break;
                };
                nth += 1;
                let Some(glyph) = glyphs.get_mut(at) else {
                    continue;
                };
                // An odd value ends the list.
                let value = i32::from(value);
                last = value & 1 != 0;
                let value = value & !1;
                if !machine.cross_stream {
                    glyph.x_advance += value;
                    glyph.x_offset += value;
                } else if value == BACK_ON_THE_LINE {
                    glyph.attach = Attach::None;
                    glyph.attach_to = 0;
                    glyph.y_offset = 0;
                } else if glyph.attach != Attach::None {
                    glyph.y_offset += value;
                }
            }
        }
        state = step.next;
        if index >= glyphs.len() {
            return;
        }
        if step.stay && again > 0 {
            again -= 1;
        } else {
            index += 1;
        }
    }
}
