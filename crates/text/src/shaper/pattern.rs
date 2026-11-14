//! Patterns over kinds of character, and cutting a run into syllables by
//! them.
//!
//! A script written by syllable says what a syllable is as a pattern: a
//! consonant, then maybe a sign below it, then any number of marks, and so
//! on. Each character has a kind, a small number, and a pattern is built
//! of kinds: one of a set, one pattern after another, a choice, an
//! optional part, a repeated part. A run is cut by trying every kind of
//! syllable where the last one ended and taking the longest, and the
//! first listed of two that are as long.

use super::buffer::Glyph;

/// A pattern over kinds of character.
pub(super) enum Pattern {
    /// One character of any of these kinds, a bit each.
    One(u64),
    Sequence(Vec<Pattern>),
    Either(Vec<Pattern>),
    Maybe(Box<Pattern>),
    Repeat(Box<Pattern>),
}

impl Pattern {
    /// Every place a match starting at `from` may end, in order.
    pub(super) fn ends(&self, kinds: &[u8], from: usize) -> Vec<usize> {
        self.ends_from(kinds, &[from])
    }

    /// Every place a match starting at any of `starts` may end, in order.
    /// `starts` are in order and each is there once. The starts are taken
    /// together, so a long stretch is walked once however many parts of a
    /// pattern may match along it.
    fn ends_from(&self, kinds: &[u8], starts: &[usize]) -> Vec<usize> {
        match self {
            Self::One(mask) => starts
                .iter()
                .filter(|start| {
                    kinds
                        .get(**start)
                        .is_some_and(|kind| *kind < 64 && (1_u64 << *kind) & mask != 0)
                })
                .map(|start| start + 1)
                .collect(),
            Self::Sequence(parts) => {
                let mut ends = starts.to_vec();
                for part in parts {
                    ends = part.ends_from(kinds, &ends);
                    if ends.is_empty() {
                        break;
                    }
                }
                ends
            }
            Self::Either(choices) => {
                let mut ends: Vec<usize> = choices
                    .iter()
                    .flat_map(|choice| choice.ends_from(kinds, starts))
                    .collect();
                ends.sort_unstable();
                ends.dedup();
                ends
            }
            Self::Maybe(inner) => union(starts, &inner.ends_from(kinds, starts)),
            Self::Repeat(inner) => {
                let mut ends: std::collections::BTreeSet<usize> = starts.iter().copied().collect();
                let mut frontier = starts.to_vec();
                // Each round matches the part once more, from the places
                // the last round first reached.
                while !frontier.is_empty() {
                    let reached = inner.ends_from(kinds, &frontier);
                    frontier = reached
                        .into_iter()
                        .filter(|end| ends.insert(*end))
                        .collect();
                }
                ends.into_iter().collect()
            }
        }
    }
}

/// Two lists in order, as one in order with each place once.
fn union(first: &[usize], second: &[usize]) -> Vec<usize> {
    let mut all = Vec::with_capacity(first.len() + second.len());
    let (mut left, mut right) = (0, 0);
    while left < first.len() || right < second.len() {
        let next = match (first.get(left), second.get(right)) {
            (Some(a), Some(b)) if a <= b => {
                left += 1;
                if a == b {
                    right += 1;
                }
                *a
            }
            (Some(a), None) => {
                left += 1;
                *a
            }
            (_, Some(b)) => {
                right += 1;
                *b
            }
            (None, None) => break,
        };
        all.push(next);
    }
    all
}

pub(super) fn seq(parts: Vec<Pattern>) -> Pattern {
    Pattern::Sequence(parts)
}

pub(super) fn either(choices: Vec<Pattern>) -> Pattern {
    Pattern::Either(choices)
}

pub(super) fn maybe(pattern: Pattern) -> Pattern {
    Pattern::Maybe(Box::new(pattern))
}

pub(super) fn repeat(pattern: Pattern) -> Pattern {
    Pattern::Repeat(Box::new(pattern))
}

/// One character of any of `kinds`.
pub(super) fn one(kinds: &[u8]) -> Pattern {
    Pattern::One(
        kinds
            .iter()
            .filter(|kind| **kind < 64)
            .fold(0, |mask, kind| mask | (1_u64 << *kind)),
    )
}

/// Cut `kinds` into syllables: where each ends and which kind of syllable
/// it is. `grammar` lists the kinds of syllable in the order of choice
/// between two that match the same stretch; a character none of them
/// starts is a syllable of kind `other` on its own.
pub(super) fn cut(kinds: &[u8], grammar: &[(u8, Pattern)], other: u8) -> Vec<(usize, u8)> {
    let mut cuts = Vec::new();
    let mut start = 0;
    while start < kinds.len() {
        // The longest match wins, and the first listed among equals.
        let mut best: Option<(usize, u8)> = None;
        for (kind, pattern) in grammar {
            if let Some(end) = pattern.ends(kinds, start).last().copied()
                && end > start
                && best.is_none_or(|(most, _)| end > most)
            {
                best = Some((end, *kind));
            }
        }
        let (end, kind) = best.unwrap_or((start + 1, other));
        cuts.push((end, kind));
        start = end;
    }
    cuts
}

/// Number each glyph's syllable from `cuts`, which are places in
/// `glyphs`: the high bits a count that changes from one syllable to the
/// next, the low bits its kind.
pub(super) fn number(glyphs: &mut [Glyph], cuts: &[(usize, u8)]) {
    let mut serial = 1_u8;
    let mut start = 0;
    for (end, kind) in cuts {
        let end = (*end).min(glyphs.len());
        for glyph in &mut glyphs[start.min(end)..end] {
            glyph.syllable = (serial << 4) | (kind & 0x0F);
        }
        serial = if serial == 15 { 1 } else { serial + 1 };
        start = end;
    }
}

/// Call `act` with each syllable's range. `act` may move glyphs within
/// the range but not change how many there are.
pub(super) fn each_syllable(glyphs: &mut [Glyph], mut act: impl FnMut(&mut [Glyph], usize, usize)) {
    let mut start = 0;
    while start < glyphs.len() {
        let syllable = glyphs[start].syllable;
        let mut end = start + 1;
        while end < glyphs.len() && glyphs[end].syllable == syllable {
            end += 1;
        }
        act(glyphs, start, end);
        start = end;
    }
}

/// Give the glyphs from `start` up to `end` one place in the text, the
/// earliest of theirs, and with them the neighbours that share a place
/// with the first or the last: glyphs that change places stand for the
/// same stretch of text.
pub(super) fn merge(glyphs: &mut [Glyph], start: usize, end: usize) {
    let end = end.min(glyphs.len());
    if end.saturating_sub(start) < 2 {
        return;
    }
    let (mut start, mut end) = (start, end);
    while end < glyphs.len() && glyphs[end].cluster == glyphs[end - 1].cluster {
        end += 1;
    }
    while start > 0 && glyphs[start - 1].cluster == glyphs[start].cluster {
        start -= 1;
    }
    if let Some(first) = glyphs[start..end].iter().map(|glyph| glyph.cluster).min() {
        for glyph in &mut glyphs[start..end] {
            glyph.cluster = first;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Pattern, cut, either, maybe, number, one, repeat, seq};
    use crate::shaper::buffer::Glyph;

    /// A letter, then any number of marks, then maybe a final sign; or a
    /// lone mark.
    fn grammar() -> Vec<(u8, Pattern)> {
        vec![
            (0, seq(vec![one(&[1]), repeat(one(&[2])), maybe(one(&[3]))])),
            (1, either(vec![one(&[2]), one(&[3])])),
        ]
    }

    #[test]
    fn a_run_is_cut_at_the_longest_match_and_the_first_listed() {
        let kinds = [1, 2, 2, 3, 2, 9, 1, 1, 3];
        assert_eq!(
            cut(&kinds, &grammar(), 7),
            [(4, 0), (5, 1), (6, 7), (7, 0), (9, 0)]
        );
        assert!(cut(&[], &grammar(), 7).is_empty());
        // Two rules that match the same stretch: the first listed.
        let same = vec![(4, one(&[1])), (5, one(&[1]))];
        assert_eq!(cut(&[1], &same, 7), [(1, 4)]);
        // A kind past the mask is never matched, and is no trouble.
        assert_eq!(cut(&[200, 64], &grammar(), 7), [(1, 7), (2, 7)]);
        assert_eq!(one(&[64, 200]).ends(&[64], 0), Vec::<usize>::new());
    }

    #[test]
    fn syllables_are_numbered_and_the_count_wraps() {
        let mut glyphs: Vec<Glyph> = (0..40).map(|at| Glyph::new('a', at)).collect();
        let cuts: Vec<(usize, u8)> = (1..=20).map(|at| (at * 2, 3)).collect();
        number(&mut glyphs, &cuts);
        assert_eq!(glyphs[0].syllable, (1 << 4) | 3);
        assert_eq!(glyphs[1].syllable, glyphs[0].syllable);
        assert_eq!(glyphs[2].syllable, (2 << 4) | 3);
        // After fifteen the count starts again at one, never at nothing.
        assert_eq!(glyphs[30].syllable, (1 << 4) | 3);
        assert!(glyphs.iter().all(|glyph| glyph.syllable >> 4 != 0));
        // A cut past the run is held to it.
        number(&mut glyphs[..3], &[(9, 1)]);
        assert_eq!(glyphs[2].syllable, (1 << 4) | 1);
    }

    #[test]
    fn a_long_run_is_cut_in_time() {
        let started = std::time::Instant::now();
        let mut kinds = vec![1_u8];
        kinds.extend(std::iter::repeat_n(2, 20_000));
        assert_eq!(cut(&kinds, &grammar(), 7).len(), 1);
        let lone = vec![2_u8; 20_000];
        assert_eq!(cut(&lone, &grammar(), 7).len(), 20_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }
}
