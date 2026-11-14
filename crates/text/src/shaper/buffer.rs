//! The glyphs of one run while it is shaped.
//!
//! Substitutions put glyphs in and take them out all along a run, so the
//! glyphs are kept in two stacks around a cursor: moving the cursor moves
//! glyphs from one to the other, and a glyph is added or removed at the
//! cursor without moving the rest.

use cutlass_font::GlyphClass;

use crate::shape::Pair;

/// How a glyph hangs on another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Attach {
    #[default]
    None,
    /// A mark set on a base, a ligature, or another mark.
    Mark,
    /// A letter joined to its neighbour in a cursive script.
    Cursive,
}

/// How wide a space is when the face has no glyph for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Space {
    /// As wide as the face's own space.
    Plain,
    /// Half of that.
    Narrow,
    /// This share of an em: one over the number.
    Em(i32),
    /// Four eighteenths of an em.
    Mathematical,
    /// As wide as a digit.
    Figure,
    /// As wide as a full stop.
    Punctuation,
}

impl Space {
    /// The width a space character falls back to.
    pub(crate) fn of(character: char) -> Option<Self> {
        Some(match character {
            ' ' | '\u{A0}' => Self::Plain,
            '\u{2000}' | '\u{2002}' => Self::Em(2),
            '\u{2001}' | '\u{2003}' | '\u{3000}' => Self::Em(1),
            '\u{2004}' => Self::Em(3),
            '\u{2005}' => Self::Em(4),
            '\u{2006}' => Self::Em(6),
            '\u{2007}' => Self::Figure,
            '\u{2008}' => Self::Punctuation,
            '\u{2009}' => Self::Em(5),
            '\u{200A}' => Self::Em(16),
            '\u{202F}' => Self::Narrow,
            '\u{205F}' => Self::Mathematical,
            _ => return None,
        })
    }
}

/// One glyph and what shaping needs to know about it.
#[derive(Clone, Debug)]
pub(crate) struct Glyph {
    pub(crate) id: u16,
    /// Byte index of the first character it draws.
    pub(crate) cluster: usize,
    /// The character it was mapped from.
    pub(crate) character: char,
    pub(crate) missing: bool,
    /// A space the face lacks, drawn with its plain space at a width of
    /// its own.
    pub(crate) space: Option<Space>,
    /// The features that apply to it, a bit each.
    pub(crate) mask: u32,
    pub(crate) class: GlyphClass,
    /// A lookup put it in place of another glyph.
    pub(crate) substituted: bool,
    /// It was made of several glyphs.
    pub(crate) ligated: bool,
    /// It is one of several glyphs made from one.
    pub(crate) multiplied: bool,
    /// The ligature it is, or is a mark on; 0 is none.
    pub(crate) lig_id: u8,
    /// Which part of its ligature a mark sits on, from 1; a ligature
    /// itself has 0.
    pub(crate) lig_part: u8,
    /// How many parts a ligature was made of.
    pub(crate) lig_parts: u8,
    /// General category, as the Unicode table numbers it.
    pub(crate) category: u8,
    /// Combining class, in the order marks are sorted.
    pub(crate) combining: u8,
    /// The side of its letter a mark goes on, as a combining class, when
    /// the face does not place it.
    pub(crate) side: u8,
    /// It draws nothing unless the font gives it a glyph a lookup uses.
    pub(crate) ignorable: bool,
    pub(crate) joiner: bool,
    pub(crate) non_joiner: bool,
    /// It goes with the character before it: a mark, the second half of
    /// an emoji sequence.
    pub(crate) continues: bool,
    /// The syllable it belongs to, for scripts shaped by syllable.
    pub(crate) syllable: u8,
    /// What the script's own shaping says of it: a joining form, or a
    /// kind of letter.
    pub(crate) kind: u8,
    /// Where in its syllable it goes.
    pub(crate) place: u8,
    pub(crate) x_advance: i32,
    pub(crate) x_offset: i32,
    pub(crate) y_offset: i32,
    pub(crate) attach: Attach,
    /// How far back, or forward when negative, the glyph it hangs on is.
    pub(crate) attach_to: i32,
    pub(crate) pair: Pair,
}

impl Glyph {
    pub(crate) fn new(character: char, cluster: usize) -> Self {
        Self {
            id: 0,
            cluster,
            character,
            missing: false,
            space: None,
            mask: 0,
            class: GlyphClass::None,
            substituted: false,
            ligated: false,
            multiplied: false,
            lig_id: 0,
            lig_part: 0,
            lig_parts: 0,
            category: 0,
            combining: 0,
            side: 0,
            ignorable: false,
            joiner: false,
            non_joiner: false,
            continues: false,
            syllable: 0,
            kind: 0,
            place: 0,
            x_advance: 0,
            x_offset: 0,
            y_offset: 0,
            attach: Attach::None,
            attach_to: 0,
            pair: Pair::default(),
        }
    }

    pub(crate) fn is_mark(&self) -> bool {
        self.class == GlyphClass::Mark
    }

    /// How many parts it counts as when a ligature is made of it.
    pub(crate) fn parts(&self) -> u8 {
        if self.class == GlyphClass::Ligature && self.lig_id != 0 && self.lig_part == 0 {
            self.lig_parts.max(1)
        } else {
            1
        }
    }

    /// A character the font draws nothing for, that a lookup may step
    /// over: no longer one once a lookup has put a glyph in its place.
    pub(crate) fn steps_over(&self) -> bool {
        self.ignorable && !self.substituted
    }
}

/// The glyphs of a run, around a cursor.
#[derive(Clone, Debug, Default)]
pub(crate) struct Buffer {
    before: Vec<Glyph>,
    /// The glyphs from the cursor on, last first.
    after: Vec<Glyph>,
    last_lig_id: u8,
}

impl Buffer {
    pub(crate) fn new(glyphs: Vec<Glyph>) -> Self {
        Self {
            before: glyphs,
            after: Vec::new(),
            last_lig_id: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.before.len() + self.after.len()
    }

    pub(crate) fn cursor(&self) -> usize {
        self.before.len()
    }

    pub(crate) fn get(&self, index: usize) -> Option<&Glyph> {
        match index.checked_sub(self.before.len()) {
            None => self.before.get(index),
            Some(past) => self
                .after
                .len()
                .checked_sub(past + 1)
                .and_then(|at| self.after.get(at)),
        }
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut Glyph> {
        match index.checked_sub(self.before.len()) {
            None => self.before.get_mut(index),
            Some(past) => self
                .after
                .len()
                .checked_sub(past + 1)
                .and_then(|at| self.after.get_mut(at)),
        }
    }

    /// Put the cursor before glyph `index`, or at the end.
    pub(crate) fn move_to(&mut self, index: usize) {
        while self.before.len() > index {
            if let Some(glyph) = self.before.pop() {
                self.after.push(glyph);
            }
        }
        while self.before.len() < index {
            match self.after.pop() {
                Some(glyph) => self.before.push(glyph),
                None => break,
            }
        }
    }

    /// The glyph at the cursor.
    pub(crate) fn current(&self) -> Option<&Glyph> {
        self.after.last()
    }

    pub(crate) fn current_mut(&mut self) -> Option<&mut Glyph> {
        self.after.last_mut()
    }

    /// Take the glyph at the cursor out.
    pub(crate) fn remove(&mut self) -> Option<Glyph> {
        self.after.pop()
    }

    /// Put `glyph` in before the cursor.
    pub(crate) fn insert(&mut self, glyph: Glyph) {
        self.before.push(glyph);
    }

    /// A number no ligature in this run has yet.
    pub(crate) fn next_lig_id(&mut self) -> u8 {
        self.last_lig_id = if self.last_lig_id == u8::MAX {
            1
        } else {
            self.last_lig_id + 1
        };
        self.last_lig_id
    }

    /// Give glyphs `start..end` one cluster, the earliest of theirs, and
    /// with them any neighbour that shared a cluster with one of them.
    pub(crate) fn merge_clusters(&mut self, start: usize, end: usize) {
        let (mut start, mut end) = (start, end.min(self.len()));
        if end.saturating_sub(start) < 2 {
            return;
        }
        let cluster_at = |buffer: &Self, index: usize| buffer.get(index).map(|glyph| glyph.cluster);
        while end < self.len() && cluster_at(self, end) == cluster_at(self, end - 1) {
            end += 1;
        }
        while start > 0 && cluster_at(self, start - 1) == cluster_at(self, start) {
            start -= 1;
        }
        let Some(first) = (start..end)
            .filter_map(|index| cluster_at(self, index))
            .min()
        else {
            return;
        };
        for index in start..end {
            if let Some(glyph) = self.get_mut(index) {
                glyph.cluster = first;
            }
        }
    }

    /// Every glyph, in order.
    pub(crate) fn into_glyphs(mut self) -> Vec<Glyph> {
        self.move_to(usize::MAX);
        self.before
    }

    /// Every glyph, in order, to read or change in place.
    pub(crate) fn glyphs_mut(&mut self) -> &mut Vec<Glyph> {
        self.move_to(usize::MAX);
        &mut self.before
    }
}

#[cfg(test)]
mod tests {
    use super::{Buffer, Glyph};

    fn buffer(count: usize) -> Buffer {
        Buffer::new(
            (0..count)
                .map(|index| {
                    let mut glyph = Glyph::new('a', index);
                    glyph.id = u16::try_from(index).expect("id");
                    glyph
                })
                .collect(),
        )
    }

    fn ids(buffer: &Buffer) -> Vec<u16> {
        (0..buffer.len())
            .map(|index| buffer.get(index).expect("glyph").id)
            .collect()
    }

    #[test]
    fn the_cursor_moves_and_glyphs_keep_their_order() {
        let mut glyphs = buffer(5);
        assert_eq!(glyphs.cursor(), 5);
        assert!(glyphs.current().is_none());
        glyphs.move_to(2);
        assert_eq!((glyphs.cursor(), glyphs.len()), (2, 5));
        assert_eq!(glyphs.current().expect("current").id, 2);
        assert_eq!(ids(&glyphs), [0, 1, 2, 3, 4]);
        glyphs.move_to(0);
        assert_eq!(ids(&glyphs), [0, 1, 2, 3, 4]);
        glyphs.move_to(99);
        assert_eq!(glyphs.cursor(), 5);
        assert!(glyphs.get(5).is_none() && glyphs.get_mut(7).is_none());
    }

    #[test]
    fn glyphs_go_in_and_out_at_the_cursor() {
        let mut glyphs = buffer(4);
        glyphs.move_to(1);
        let taken = glyphs.remove().expect("removed");
        assert_eq!(taken.id, 1);
        assert_eq!(ids(&glyphs), [0, 2, 3]);
        let mut added = Glyph::new('b', 9);
        added.id = 9;
        glyphs.insert(added);
        assert_eq!(ids(&glyphs), [0, 9, 2, 3]);
        assert_eq!(glyphs.cursor(), 2);
        glyphs.get_mut(3).expect("last").id = 7;
        assert_eq!(ids(&glyphs), [0, 9, 2, 7]);
        assert_eq!(
            glyphs
                .into_glyphs()
                .iter()
                .map(|glyph| glyph.id)
                .collect::<Vec<_>>(),
            [0, 9, 2, 7]
        );
    }

    #[test]
    fn merged_clusters_take_the_earliest_and_reach_their_neighbours() {
        let mut glyphs = buffer(6);
        glyphs.get_mut(4).expect("glyph").cluster = 3;
        glyphs.merge_clusters(1, 3);
        let clusters = |buffer: &Buffer| {
            (0..buffer.len())
                .map(|index| buffer.get(index).expect("glyph").cluster)
                .collect::<Vec<_>>()
        };
        assert_eq!(clusters(&glyphs), [0, 1, 1, 3, 3, 5]);
        // Glyph 3 shares its cluster with glyph 4, which comes along.
        glyphs.merge_clusters(2, 4);
        assert_eq!(clusters(&glyphs), [0, 1, 1, 1, 1, 5]);
        glyphs.merge_clusters(5, 6);
        glyphs.merge_clusters(4, 99);
        assert_eq!(clusters(&glyphs), [0, 1, 1, 1, 1, 1]);
        assert_eq!((glyphs.next_lig_id(), glyphs.next_lig_id()), (1, 2));
    }
}
