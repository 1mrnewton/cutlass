//! Which way each character runs: the Unicode bidirectional algorithm
//! (UAX #9).
//!
//! [`resolve`] gives every character of one paragraph an embedding level:
//! even runs left to right, odd right to left. A line is then put in the
//! order it is seen with [`line_levels`] and [`visual_order`]. The direction
//! of a paragraph is that of its first strong character, and left to right
//! when it has none.

use crate::unicode::{Bracket, bidi, closing, props};

/// The deepest embedding the algorithm allows.
const MAX_DEPTH: u8 = 125;
/// The most open brackets held while pairing.
const MAX_BRACKETS: usize = 63;

/// One paragraph's characters, each with its level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Levels {
    /// 0 for a paragraph that runs left to right, 1 for right to left.
    pub(crate) base: u8,
    /// The level of each character, in the order they are stored.
    pub(crate) levels: Vec<u8>,
    /// Each character's class as the data gives it, for [`line_levels`].
    classes: Vec<u8>,
}

fn strong(class: u8) -> bool {
    matches!(class, bidi::L | bidi::R | bidi::AL)
}

fn isolate_start(class: u8) -> bool {
    matches!(class, bidi::LRI | bidi::RLI | bidi::FSI)
}

/// The formatting characters rule X9 takes out of the text.
fn removed(class: u8) -> bool {
    matches!(
        class,
        bidi::RLE | bidi::LRE | bidi::RLO | bidi::LRO | bidi::PDF | bidi::BN
    )
}

/// Characters that take their direction from what is around them.
fn neutral(class: u8) -> bool {
    matches!(class, bidi::B | bidi::S | bidi::WS | bidi::ON | bidi::PDI) || isolate_start(class)
}

/// Rules P2 and P3 over `classes`: the level its first strong character
/// gives, skipping what isolates hold, or `None` without one.
fn first_strong(classes: &[u8]) -> Option<u8> {
    let mut depth = 0_usize;
    for class in classes {
        match *class {
            class if isolate_start(class) => depth += 1,
            bidi::PDI => depth = depth.saturating_sub(1),
            bidi::L if depth == 0 => return Some(0),
            bidi::R | bidi::AL if depth == 0 => return Some(1),
            _ => {}
        }
    }
    None
}

/// The levels of `text`, which is one paragraph: it holds no character
/// that ends a line.
pub(crate) fn resolve(text: &str) -> Levels {
    let characters: Vec<char> = text.chars().collect();
    let classes: Vec<u8> = characters
        .iter()
        .map(|character| props(*character).bidi())
        .collect();
    let base = first_strong(&classes).unwrap_or(0);
    // Text with nothing that runs right to left is all at level 0.
    let plain = base == 0
        && classes.iter().all(|class| {
            !matches!(
                *class,
                bidi::R | bidi::AL | bidi::AN | bidi::RLE | bidi::RLO | bidi::RLI | bidi::FSI
            )
        });
    if plain {
        return Levels {
            base,
            levels: vec![0; classes.len()],
            classes,
        };
    }
    let mut working = classes.clone();
    let (mut levels, matches) = explicit(base, &mut working);
    for sequence in sequences(base, &working, &levels, &matches) {
        weak(&sequence, &mut working);
        brackets(&sequence, &characters, &mut working, &classes);
        neutrals(&sequence, &mut working, &levels);
    }
    // Rules I1 and I2, and a removed character takes the level before it.
    let mut before = base;
    for (index, level) in levels.iter_mut().enumerate() {
        if removed(classes[index]) {
            *level = before;
            continue;
        }
        *level += match (*level % 2, working[index]) {
            (0, bidi::R) | (1, bidi::L | bidi::EN | bidi::AN) => 1,
            (0, bidi::AN | bidi::EN) => 2,
            _ => 0,
        };
        before = *level;
    }
    Levels {
        base,
        levels,
        classes,
    }
}

/// One entry of the directional status stack.
#[derive(Clone, Copy)]
struct Status {
    level: u8,
    /// A class every character inside is forced to, when not neutral.
    forced: Option<u8>,
    isolate: bool,
}

/// Rules X1 to X8: the level each character is embedded at. A character
/// inside an override takes the override's class in `classes`. Also the
/// character that closes each isolate, by the index of the one opening it.
fn explicit(base: u8, classes: &mut [u8]) -> (Vec<u8>, Vec<Option<usize>>) {
    let count = classes.len();
    let mut levels = vec![base; count];
    let mut matches: Vec<Option<usize>> = vec![None; count];
    let mut stack = vec![Status {
        level: base,
        forced: None,
        isolate: false,
    }];
    // The isolates that are open and counted, by index.
    let mut open: Vec<usize> = Vec::new();
    let (mut overflow_isolates, mut overflow_embeddings) = (0_usize, 0_usize);
    for index in 0..count {
        let class = classes[index];
        let Some(top) = stack.last().copied() else {
            break;
        };
        let deeper = |odd: bool| {
            let next = if odd {
                (top.level + 1) | 1
            } else {
                (top.level + 2) & !1
            };
            (next <= MAX_DEPTH).then_some(next)
        };
        match class {
            bidi::RLE | bidi::LRE | bidi::RLO | bidi::LRO => {
                levels[index] = top.level;
                let odd = matches!(class, bidi::RLE | bidi::RLO);
                match deeper(odd) {
                    Some(level) if overflow_isolates == 0 && overflow_embeddings == 0 => {
                        stack.push(Status {
                            level,
                            forced: match class {
                                bidi::RLO => Some(bidi::R),
                                bidi::LRO => Some(bidi::L),
                                _ => None,
                            },
                            isolate: false,
                        });
                    }
                    _ if overflow_isolates == 0 => overflow_embeddings += 1,
                    _ => {}
                }
            }
            bidi::RLI | bidi::LRI | bidi::FSI => {
                levels[index] = top.level;
                if let Some(forced) = top.forced {
                    classes[index] = forced;
                }
                let odd = match class {
                    bidi::RLI => true,
                    bidi::LRI => false,
                    // The first strong character up to the matching end.
                    _ => {
                        let end = closing_isolate(classes, index).unwrap_or(count);
                        first_strong(&classes[index + 1..end]) == Some(1)
                    }
                };
                match deeper(odd) {
                    Some(level) if overflow_isolates == 0 && overflow_embeddings == 0 => {
                        open.push(index);
                        stack.push(Status {
                            level,
                            forced: None,
                            isolate: true,
                        });
                    }
                    _ => overflow_isolates += 1,
                }
            }
            bidi::PDI => {
                if overflow_isolates > 0 {
                    overflow_isolates -= 1;
                } else if let Some(opener) = open.pop() {
                    overflow_embeddings = 0;
                    while stack.last().is_some_and(|status| !status.isolate) {
                        stack.pop();
                    }
                    stack.pop();
                    matches[opener] = Some(index);
                }
                let top = stack.last().copied().unwrap_or(top);
                levels[index] = top.level;
                if let Some(forced) = top.forced {
                    classes[index] = forced;
                }
            }
            bidi::PDF => {
                levels[index] = top.level;
                if overflow_isolates > 0 {
                } else if overflow_embeddings > 0 {
                    overflow_embeddings -= 1;
                } else if !top.isolate && stack.len() >= 2 {
                    stack.pop();
                }
            }
            bidi::BN => levels[index] = top.level,
            _ => {
                levels[index] = top.level;
                if let Some(forced) = top.forced {
                    classes[index] = forced;
                }
            }
        }
    }
    (levels, matches)
}

/// The end of the isolate that starts at `start`, for the classes as the
/// data gives them.
fn closing_isolate(classes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, class) in classes.iter().enumerate().skip(start + 1) {
        if isolate_start(*class) {
            depth += 1;
        } else if *class == bidi::PDI {
            if depth == 0 {
                return Some(index);
            }
            depth -= 1;
        }
    }
    None
}

/// An isolating run sequence: the characters the later rules treat as one
/// run, with the direction at each of its ends.
struct Sequence {
    /// Indices of its characters, without the ones rule X9 removes.
    members: Vec<usize>,
    /// The class at its start and at its end: `L` or `R`.
    start: u8,
    end: u8,
    level: u8,
}

/// Rule X10: level runs, joined across each isolate to the run after its
/// end.
fn sequences(base: u8, classes: &[u8], levels: &[u8], matches: &[Option<usize>]) -> Vec<Sequence> {
    let count = classes.len();
    let kept: Vec<usize> = (0..count)
        .filter(|index| !removed(classes[*index]))
        .collect();
    // Level runs over the characters that are kept: each a range of `kept`.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for place in 1..=kept.len() {
        if place == kept.len() || levels[kept[place]] != levels[kept[start]] {
            runs.push((start, place));
            start = place;
        }
    }
    // The run that starts at each kept character.
    let mut run_at = vec![usize::MAX; count];
    for (run, (first, _)) in runs.iter().enumerate() {
        run_at[kept[*first]] = run;
    }
    let mut taken = vec![false; runs.len()];
    let mut found = Vec::new();
    for first in 0..runs.len() {
        if taken[first] {
            continue;
        }
        let mut members = Vec::new();
        let mut run = first;
        let last = loop {
            taken[run] = true;
            let (from, to) = runs[run];
            members.extend_from_slice(&kept[from..to]);
            let last = kept[to - 1];
            // An isolate with an end goes on at the run its end starts.
            let next = isolate_start(classes[last])
                .then(|| matches[last])
                .flatten()
                .map(|end| run_at[end])
                .filter(|next| *next != usize::MAX && !taken[*next]);
            match next {
                Some(next) => run = next,
                None => break last,
            }
        };
        let level = levels[members[0]];
        let first_place = kept.partition_point(|index| *index < members[0]);
        let before = first_place
            .checked_sub(1)
            .map_or(base, |place| levels[kept[place]]);
        let last_place = kept.partition_point(|index| *index <= last);
        // An isolate with no end runs to the paragraph's.
        let after = if isolate_start(classes[last]) && matches[last].is_none() {
            base
        } else {
            kept.get(last_place).map_or(base, |index| levels[*index])
        };
        let side = |other: u8| {
            if level.max(other) % 2 == 1 {
                bidi::R
            } else {
                bidi::L
            }
        };
        found.push(Sequence {
            members,
            start: side(before),
            end: side(after),
            level,
        });
    }
    found
}

/// Rules W1 to W7.
fn weak(sequence: &Sequence, classes: &mut [u8]) {
    let members = &sequence.members;
    // W1 to W3, with the last strong class for W2.
    let mut previous = sequence.start;
    let mut last_strong = sequence.start;
    for index in members {
        let mut class = classes[*index];
        if class == bidi::NSM {
            class = if isolate_start(previous) || previous == bidi::PDI {
                bidi::ON
            } else {
                previous
            };
        }
        if class == bidi::EN && last_strong == bidi::AL {
            class = bidi::AN;
        }
        if strong(class) {
            last_strong = class;
        }
        previous = class;
        classes[*index] = class;
    }
    for index in members {
        if classes[*index] == bidi::AL {
            classes[*index] = bidi::R;
        }
    }
    // W4: one separator between two numbers.
    for place in 1..members.len().saturating_sub(1) {
        let (before, here, after) = (
            classes[members[place - 1]],
            classes[members[place]],
            classes[members[place + 1]],
        );
        if before == after
            && ((here == bidi::ES && before == bidi::EN)
                || (here == bidi::CS && matches!(before, bidi::EN | bidi::AN)))
        {
            classes[members[place]] = before;
        }
    }
    // W5: terminators beside a European number.
    let mut place = 0;
    while place < members.len() {
        if classes[members[place]] != bidi::ET {
            place += 1;
            continue;
        }
        let from = place;
        while place < members.len() && classes[members[place]] == bidi::ET {
            place += 1;
        }
        let before = from.checked_sub(1).map(|at| classes[members[at]]);
        let after = members.get(place).map(|index| classes[*index]);
        if before == Some(bidi::EN) || after == Some(bidi::EN) {
            for index in &members[from..place] {
                classes[*index] = bidi::EN;
            }
        }
    }
    // W6 and W7.
    let mut last_strong = sequence.start;
    for index in members {
        let class = classes[*index];
        if matches!(class, bidi::ES | bidi::ET | bidi::CS) {
            classes[*index] = bidi::ON;
        } else if class == bidi::EN && last_strong == bidi::L {
            classes[*index] = bidi::L;
        } else if matches!(class, bidi::L | bidi::R) {
            last_strong = class;
        }
    }
}

/// A bracket and its canonical equal, read as one.
fn canonical(character: char) -> char {
    match character {
        '\u{2329}' => '\u{3008}',
        '\u{232A}' => '\u{3009}',
        other => other,
    }
}

/// Rule N0: each pair of brackets takes one direction.
fn brackets(sequence: &Sequence, characters: &[char], classes: &mut [u8], original: &[u8]) {
    let members = &sequence.members;
    // Pair them: a closing bracket takes the nearest opener that fits.
    let mut open: Vec<(char, usize)> = Vec::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (place, index) in members.iter().enumerate() {
        if classes[*index] != bidi::ON {
            continue;
        }
        let character = characters[*index];
        match props(character).bracket() {
            Bracket::Open => {
                if open.len() == MAX_BRACKETS {
                    break;
                }
                if let Some(close) = closing(character) {
                    open.push((canonical(close), place));
                }
            }
            Bracket::Close => {
                let wanted = canonical(character);
                if let Some(at) = open.iter().rposition(|(close, _)| *close == wanted) {
                    pairs.push((open[at].1, place));
                    open.truncate(at);
                }
            }
            Bracket::None => {}
        }
    }
    pairs.sort_unstable();
    let embedding = if sequence.level % 2 == 1 {
        bidi::R
    } else {
        bidi::L
    };
    let directed = |class: u8| match class {
        bidi::L => Some(bidi::L),
        bidi::R | bidi::EN | bidi::AN => Some(bidi::R),
        _ => None,
    };
    for (first, last) in pairs {
        let mut same = false;
        let mut other = false;
        for index in &members[first + 1..last] {
            match directed(classes[*index]) {
                Some(class) if class == embedding => same = true,
                Some(_) => other = true,
                None => {}
            }
        }
        let class = if same {
            embedding
        } else if other {
            // The direction before the opener decides.
            let before = members[..first]
                .iter()
                .rev()
                .find_map(|index| directed(classes[*index]))
                .unwrap_or(sequence.start);
            if before == embedding {
                embedding
            } else {
                before
            }
        } else {
            continue;
        };
        for bracket in [first, last] {
            classes[members[bracket]] = class;
            // Marks written on the bracket go with it.
            for index in &members[bracket + 1..] {
                if original[*index] != bidi::NSM {
                    break;
                }
                classes[*index] = class;
            }
        }
    }
}

/// Rules N1 and N2.
fn neutrals(sequence: &Sequence, classes: &mut [u8], levels: &[u8]) {
    let members = &sequence.members;
    let directed = |class: u8| match class {
        bidi::L => bidi::L,
        _ => bidi::R,
    };
    let mut place = 0;
    while place < members.len() {
        if !neutral(classes[members[place]]) {
            place += 1;
            continue;
        }
        let from = place;
        while place < members.len() && neutral(classes[members[place]]) {
            place += 1;
        }
        let before = from
            .checked_sub(1)
            .map_or(sequence.start, |at| directed(classes[members[at]]));
        let after = members
            .get(place)
            .map_or(sequence.end, |index| directed(classes[*index]));
        for index in &members[from..place] {
            classes[*index] = if before == after {
                before
            } else if levels[*index] % 2 == 1 {
                bidi::R
            } else {
                bidi::L
            };
        }
    }
}

impl Levels {
    /// Rule L1: the levels of characters `from..to` as one line, where the
    /// spaces at its end, and those before a tab, fall back to the
    /// paragraph's level.
    pub(crate) fn line_levels(&self, from: usize, to: usize) -> Vec<u8> {
        let to = to.min(self.levels.len());
        let from = from.min(to);
        let mut levels = self.levels[from..to].to_vec();
        let mut reset = true;
        for (level, class) in levels.iter_mut().zip(&self.classes[from..to]).rev() {
            match *class {
                bidi::S | bidi::B => {
                    reset = true;
                    *level = self.base;
                }
                class if class == bidi::WS || class == bidi::PDI || isolate_start(class) => {
                    if reset {
                        *level = self.base;
                    }
                }
                class if removed(class) => {
                    if reset {
                        *level = self.base;
                    }
                }
                _ => reset = false,
            }
        }
        levels
    }
}

/// Rule L2: the order in which a line's items are seen, left to right,
/// given the level of each in the order they are stored.
pub(crate) fn visual_order(levels: &[u8]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..levels.len()).collect();
    let highest = levels.iter().copied().max().unwrap_or(0);
    let lowest_odd = levels
        .iter()
        .copied()
        .filter(|level| level % 2 == 1)
        .min()
        .unwrap_or(highest + 1);
    let mut level = highest;
    while level >= lowest_odd && level > 0 {
        let mut place = 0;
        while place < order.len() {
            if levels[order[place]] < level {
                place += 1;
                continue;
            }
            let from = place;
            while place < order.len() && levels[order[place]] >= level {
                place += 1;
            }
            order[from..place].reverse();
        }
        level -= 1;
    }
    order
}

#[cfg(test)]
mod tests {
    use super::{resolve, visual_order};

    /// The levels of `text` as one line, a digit each.
    fn levels(text: &str) -> String {
        let resolved = resolve(text);
        resolved
            .line_levels(0, resolved.levels.len())
            .iter()
            .map(|level| char::from(b'0' + level))
            .collect()
    }

    /// `text` in the order it is seen.
    fn seen(text: &str) -> String {
        let resolved = resolve(text);
        let characters: Vec<char> = text.chars().collect();
        let line = resolved.line_levels(0, characters.len());
        visual_order(&line)
            .into_iter()
            .map(|index| characters[index])
            .collect()
    }

    // Hebrew letters stand in for right-to-left text: alef, bet, gimel.
    const A: char = '\u{5D0}';
    const B: char = '\u{5D1}';
    const G: char = '\u{5D2}';

    #[test]
    fn text_in_one_direction_keeps_its_order() {
        assert_eq!(levels("abc"), "000");
        assert_eq!(levels(""), "");
        assert_eq!(resolve("abc").base, 0);
        assert_eq!(seen("abc def"), "abc def");
        let hebrew: String = [A, B, G].iter().collect();
        assert_eq!(levels(&hebrew), "111");
        assert_eq!(resolve(&hebrew).base, 1);
        assert_eq!(seen(&hebrew), [G, B, A].iter().collect::<String>());
        assert_eq!(resolve("123").base, 0);
        assert_eq!(resolve(" \u{627}").base, 1);
    }

    #[test]
    fn a_run_against_the_paragraph_is_turned_around() {
        let mixed = format!("ab {A}{B} cd");
        assert_eq!(levels(&mixed), "00011000");
        assert_eq!(seen(&mixed), format!("ab {B}{A} cd"));
        // Two words stay in their order within the run, with their space.
        let words = format!("ab {A}{B} {G}{A} cd");
        assert_eq!(levels(&words), "00011111000");
        assert_eq!(seen(&words), format!("ab {A}{G} {B}{A} cd"));
        // Latin inside a right-to-left paragraph.
        let inside = format!("{A}{B} ab {G}");
        assert_eq!(levels(&inside), "1112211");
        assert_eq!(seen(&inside), format!("{G} ab {B}{A}"));
    }

    #[test]
    fn numbers_keep_their_order_inside_right_to_left_text() {
        let priced = format!("{A}{B} 123 {G}");
        assert_eq!(levels(&priced), "11122211");
        assert_eq!(seen(&priced), format!("{G} 123 {B}{A}"));
        // A separator between digits stays with them; a sign before too.
        let decimal = format!("{A} 1.5 {B}");
        assert_eq!(levels(&decimal), "1122211");
        let money = format!("{A} $5 {B}");
        assert_eq!(levels(&money), "112211");
        // Arabic letters make European digits Arabic numbers.
        assert_eq!(levels("\u{627}1"), "12");
        assert_eq!(levels("\u{627}\u{661}"), "12");
        // After Latin, digits in a left-to-right paragraph stay level.
        assert_eq!(levels("a1"), "00");
    }

    #[test]
    fn neutrals_take_the_direction_around_them() {
        assert_eq!(levels(&format!("{A} {B}")), "111");
        assert_eq!(levels(&format!("a {A}")), "001");
        assert_eq!(levels(&format!("{A}, a")), "1112");
        assert_eq!(levels(&format!("a{A}!")), "010");
        // The spaces that end a line fall back to the paragraph.
        assert_eq!(levels(&format!("a {A} ")), "0010");
        assert_eq!(levels(&format!("{A} a ")), "1121");
    }

    #[test]
    fn brackets_pair_and_face_one_way() {
        // Brackets around right-to-left text in a left-to-right line.
        let wrapped = format!("a ({A}{B}) b");
        assert_eq!(levels(&wrapped), "00011000");
        assert_eq!(seen(&wrapped), format!("a ({B}{A}) b"));
        // In a right-to-left paragraph the pair holding Latin stays whole.
        let inside = format!("{A} (ab) {B}");
        assert_eq!(levels(&inside), "11122111");
        // A pair takes the direction before it when it holds the other.
        let after = format!("{A}{B} (a)");
        assert_eq!(levels(&after), "111121");
    }

    #[test]
    fn explicit_marks_and_isolates_set_levels() {
        // An override makes Latin run right to left.
        // The mark that ends it sits at the level of what it ends.
        assert_eq!(levels("a\u{202E}bc\u{202C}d"), "001110");
        assert_eq!(seen("\u{202E}abc\u{202C}"), "\u{202E}cba\u{202C}");
        // An isolate keeps its text out of what is around it.
        let isolated = format!("a \u{2067}{A}1\u{2069} b");
        assert_eq!(levels(&isolated), "00012000");
        // The first strong character inside decides a first-strong isolate.
        let first = format!("a\u{2068}{A}b\u{2069}");
        assert_eq!(levels(&first), "00120");
        // Embedding past the deepest level changes nothing more.
        let deep = "\u{202B}".repeat(200) + "a";
        let resolved = resolve(&deep);
        assert!(resolved.levels.iter().all(|level| *level <= 126));
    }

    #[test]
    fn the_order_seen_reverses_runs_from_the_deepest() {
        assert_eq!(visual_order(&[]), Vec::<usize>::new());
        assert_eq!(visual_order(&[0, 0, 0]), [0, 1, 2]);
        assert_eq!(visual_order(&[1, 1, 1]), [2, 1, 0]);
        assert_eq!(visual_order(&[0, 1, 1, 0]), [0, 2, 1, 3]);
        assert_eq!(visual_order(&[1, 2, 2, 1]), [3, 1, 2, 0]);
        assert_eq!(visual_order(&[2, 2]), [0, 1]);
    }

    /// Compare with another implementation's levels, one case a line: code
    /// points in hex, a semicolon, then a level for each, with `x` where
    /// it does not say, and `e` or `o` where it only says even or odd (a
    /// line all one way, whose levels it does not report). Runs when
    /// `CUTLASS_BIDI_CASES` names such a file.
    #[test]
    #[ignore = "needs a file of cases from another implementation"]
    fn another_implementation_agrees() {
        let Ok(path) = std::env::var("CUTLASS_BIDI_CASES") else {
            return;
        };
        let cases = std::fs::read_to_string(path).expect("cases");
        let mut wrong = Vec::new();
        for case in cases.lines() {
            let (points, expected) = case.split_once(';').expect("case");
            let text: String = points
                .split_whitespace()
                .map(|hex| {
                    char::from_u32(u32::from_str_radix(hex, 16).expect("hex")).expect("char")
                })
                .collect();
            let resolved = resolve(&text);
            let line = resolved.line_levels(0, resolved.levels.len());
            let expected: Vec<&str> = expected.split_whitespace().collect();
            let ours: Vec<String> = line
                .iter()
                .zip(&expected)
                .map(|(level, theirs)| match *theirs {
                    "x" => "x".to_owned(),
                    "e" | "o" if level % 2 == 0 => "e".to_owned(),
                    "e" | "o" => "o".to_owned(),
                    _ => level.to_string(),
                })
                .collect();
            if ours != expected || line.len() != expected.len() {
                wrong.push(format!(
                    "{points}: ours {} theirs {}",
                    ours.join(" "),
                    expected.join(" ")
                ));
            }
        }
        let total = cases.lines().count();
        let report: Vec<&String> = wrong.iter().take(40).collect();
        assert!(
            wrong.is_empty(),
            "{} of {total} differ: {report:#?}",
            wrong.len()
        );
    }
}
