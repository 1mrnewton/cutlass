//! Where a line may end: the Unicode line breaking algorithm (UAX #14).
//!
//! Each place between two characters is one [`Boundary`]. A line ends at an
//! allowed place when it can. When one word is wider than the line, it ends
//! at any place that is not inside a letter with its marks, an emoji
//! sequence, a flag, or a syllable. The rules are the default ones, with no
//! dictionary: Thai, Lao, Khmer and Burmese words are not told apart.
//! Those scripts write no spaces between words, so they also break where
//! their spelling shows a syllable starting, which is never inside a word
//! part that must stay whole: before a vowel written ahead of its
//! consonant in Thai and Lao, and before a consonant that opens a
//! syllable in Khmer and Burmese.

use crate::unicode::{Props, ea, gc, lb, props};

/// What may happen between two characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Boundary {
    /// Inside a letter and its marks, a sequence, or a syllable: no line
    /// ends here.
    Joined,
    /// Inside a word: a line ends here only when the word does not fit.
    Inside,
    /// A line may end here.
    Allowed,
    /// A line ends here.
    Mandatory,
}

/// The boundary before each byte of `text`: entry `n` is the place before
/// the character starting at byte `n`. Entries inside a character, and the
/// first, are [`Boundary::Joined`]. The end of the text has no entry.
pub(crate) fn boundaries(text: &str) -> Vec<Boundary> {
    let units = units(text);
    let mut found = vec![Boundary::Joined; text.len()];
    for at in 1..units.len() {
        if let Some(entry) = found.get_mut(units[at].start) {
            *entry = match between(&units, at) {
                Boundary::Inside => {
                    syllable(&units[at - 1], &units[at]).unwrap_or(Boundary::Inside)
                }
                other => other,
            };
        }
    }
    found
}

/// Whether a line ends after `character` whatever follows, aside from a
/// carriage return, which a line feed may follow.
pub(crate) fn ends_line(character: char) -> bool {
    matches!(
        props(character).line_break(),
        lb::MANDATORY_BREAK | lb::CARRIAGE_RETURN | lb::LINE_FEED | lb::NEXT_LINE
    )
}

/// A character and the marks and joiners that follow it, which break as it
/// does (rule LB9).
#[derive(Clone, Copy)]
struct Unit {
    /// Byte of the first character.
    start: usize,
    /// The class every later rule sees.
    class: u8,
    /// Its last character is a zero width joiner.
    joiner_last: bool,
    east_asian: bool,
    initial_quote: bool,
    final_quote: bool,
    /// A pictograph no character is assigned to yet.
    reserved_pictograph: bool,
    dotted_circle: bool,
    /// The nearest unit at or before this one that is not a space.
    past_spaces: Option<usize>,
    /// A number, or separators and symbols that follow one.
    number: bool,
    /// A regional indicator that is the first half of a flag.
    flag_open: bool,
    /// Its first character, and its last with its marks.
    first: char,
    last: char,
    /// A letter of a script written without spaces between words.
    unspaced: bool,
    /// Its marks are a Burmese killer and tone marks alone: with them a
    /// consonant ends the syllable before it.
    killed: bool,
    /// It has a mark that is not one of those.
    vowelled: bool,
}

fn units(text: &str) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();
    for (start, character) in text.char_indices() {
        let props = props(character);
        let class = resolved(props);
        let attaches = matches!(class, lb::COMBINING_MARK | lb::ZERO_WIDTH_JOINER);
        let joiner = class == lb::ZERO_WIDTH_JOINER;
        if attaches
            && let Some(base) = units.last_mut()
            && !matches!(
                base.class,
                lb::MANDATORY_BREAK
                    | lb::CARRIAGE_RETURN
                    | lb::LINE_FEED
                    | lb::NEXT_LINE
                    | lb::SPACE
                    | lb::ZERO_WIDTH_SPACE
            )
        {
            base.joiner_last = joiner;
            base.last = character;
            base.killed = match character {
                '\u{103A}' => !base.vowelled,
                '\u{1037}' | '\u{1038}' => base.killed,
                _ => false,
            };
            base.vowelled |= !matches!(character, '\u{103A}' | '\u{1037}' | '\u{1038}');
            continue;
        }
        let east_asian = matches!(props.east_asian(), ea::FULLWIDTH | ea::WIDE | ea::HALFWIDTH);
        // A mark with nothing to sit on breaks as a letter (rule LB10).
        let class = if attaches { lb::ALPHABETIC } else { class };
        let last = units.last();
        let past_spaces = if class == lb::SPACE {
            last.and_then(|unit| unit.past_spaces)
        } else {
            Some(units.len())
        };
        let number = match class {
            lb::NUMERIC => true,
            lb::BREAK_SYMBOLS | lb::INFIX_NUMERIC => last.is_some_and(|unit| unit.number),
            _ => false,
        };
        let flag_open = class == lb::REGIONAL_INDICATOR && !last.is_some_and(|unit| unit.flag_open);
        units.push(Unit {
            start,
            class,
            joiner_last: joiner,
            east_asian,
            initial_quote: class == lb::QUOTATION && props.category() == gc::INITIAL_PUNCTUATION,
            final_quote: class == lb::QUOTATION && props.category() == gc::FINAL_PUNCTUATION,
            reserved_pictograph: props.pictographic() && props.category() == gc::UNASSIGNED,
            dotted_circle: character == '\u{25CC}',
            past_spaces,
            number,
            flag_open,
            first: character,
            last: character,
            unspaced: props.line_break() == lb::COMPLEX_CONTEXT,
            killed: false,
            vowelled: false,
        });
    }
    units
}

/// Where a syllable starts or must stay whole between two letters of a
/// script written without spaces, when its spelling says. `None` leaves
/// the place inside a word.
fn syllable(before: &Unit, after: &Unit) -> Option<Boundary> {
    if !(before.unspaced && after.unspaced) {
        return None;
    }
    let code = |character: char| u32::from(character);
    let (ahead, behind) = (code(after.first), code(before.first));
    // Thai and Lao: a vowel written ahead of its consonant opens a
    // syllable and stays with the consonant; the vowels and signs written
    // after it close one and stay with what is before them.
    let leading = |code: u32| matches!(code, 0x0E40..=0x0E44 | 0x0EC0..=0x0EC4);
    let closing = |code: u32| {
        matches!(
            code,
            0x0E2F | 0x0E30 | 0x0E33 | 0x0E46 | 0x0EAF | 0x0EB0 | 0x0EB3 | 0x0EC6
        )
    };
    let following =
        |code: u32| matches!(code, 0x0E30..=0x0E33 | 0x0E45 | 0x0EB0..=0x0EB3) || closing(code);
    if leading(behind) || following(ahead) {
        return Some(Boundary::Joined);
    }
    if leading(ahead) || closing(behind) {
        return Some(Boundary::Allowed);
    }
    // Khmer: a consonant under the sign that stacks it stays with the one
    // above; a consonant with a vowel or a stack of its own opens a
    // syllable. A bare consonant may end the syllable before it.
    if before.last == '\u{17D2}' || before.last == '\u{1039}' {
        return Some(Boundary::Joined);
    }
    if matches!(ahead, 0x1780..=0x17B3) {
        return (after.last != after.first).then_some(Boundary::Allowed);
    }
    // Burmese: a consonant with its killer ends the syllable before it;
    // any other consonant opens one.
    if matches!(ahead, 0x1000..=0x102A | 0x103F | 0x104E | 0x1050..=0x1055) {
        return Some(if after.killed {
            Boundary::Joined
        } else {
            Boundary::Allowed
        });
    }
    None
}

/// Rule LB1: the classes the rules do not name become ones they do.
fn resolved(props: Props) -> u8 {
    match props.line_break() {
        lb::AMBIGUOUS | lb::SURROGATE | lb::UNKNOWN => lb::ALPHABETIC,
        lb::COMPLEX_CONTEXT => {
            if matches!(props.category(), gc::NONSPACING_MARK | gc::SPACING_MARK) {
                lb::COMBINING_MARK
            } else {
                lb::ALPHABETIC
            }
        }
        lb::CONDITIONAL_JAPANESE_STARTER => lb::NONSTARTER,
        class => class,
    }
}

fn letter(class: u8) -> bool {
    matches!(class, lb::ALPHABETIC | lb::HEBREW_LETTER)
}

fn hyphen(class: u8) -> bool {
    matches!(class, lb::HYPHEN | lb::UNAMBIGUOUS_HYPHEN)
}

fn hangul(class: u8) -> bool {
    matches!(class, lb::JL | lb::JV | lb::JT | lb::H2 | lb::H3)
}

/// An aksara, a dotted circle standing for one, or one that starts a
/// cluster.
fn aksara(unit: &Unit) -> bool {
    matches!(unit.class, lb::AKSARA | lb::AKSARA_START) || unit.dotted_circle
}

/// The boundary between `units[at - 1]` and `units[at]`, rules LB4 to LB31
/// in their order. `at` is at least 1.
#[allow(clippy::too_many_lines)]
fn between(units: &[Unit], at: usize) -> Boundary {
    use Boundary::{Allowed, Inside, Joined, Mandatory};
    let (Some(before), Some(after)) = (units.get(at - 1), units.get(at)) else {
        return Joined;
    };
    let (a, b) = (before.class, after.class);
    let class_at = |index: Option<usize>| index.and_then(|index| units.get(index));
    let earlier = class_at(at.checked_sub(2));
    let later = class_at(at.checked_add(1));
    let later_class = later.map(|unit| unit.class);
    // The unit before any spaces that end at `before`.
    let past_spaces = before.past_spaces.and_then(|index| units.get(index));
    // Whether the unit at `index` is a number or the separators after one.
    let number_before = |index: usize| units.get(index).is_some_and(|unit| unit.number);

    // LB4 and LB5.
    if a == lb::MANDATORY_BREAK {
        return Mandatory;
    }
    if a == lb::CARRIAGE_RETURN && b == lb::LINE_FEED {
        return Joined;
    }
    if matches!(a, lb::CARRIAGE_RETURN | lb::LINE_FEED | lb::NEXT_LINE) {
        return Mandatory;
    }
    // LB6 and LB7.
    if matches!(
        b,
        lb::MANDATORY_BREAK | lb::CARRIAGE_RETURN | lb::LINE_FEED | lb::NEXT_LINE
    ) || matches!(b, lb::SPACE | lb::ZERO_WIDTH_SPACE)
    {
        return Inside;
    }
    // LB8.
    if a == lb::ZERO_WIDTH_SPACE
        || (a == lb::SPACE && past_spaces.is_some_and(|unit| unit.class == lb::ZERO_WIDTH_SPACE))
    {
        return Allowed;
    }
    // LB8a.
    if before.joiner_last {
        return Joined;
    }
    // LB11 and LB12.
    if a == lb::WORD_JOINER || b == lb::WORD_JOINER || a == lb::GLUE {
        return Inside;
    }
    // LB12a.
    if b == lb::GLUE && !(a == lb::SPACE || a == lb::BREAK_AFTER || hyphen(a)) {
        return Inside;
    }
    // LB13.
    if matches!(
        b,
        lb::CLOSE_PUNCTUATION | lb::CLOSE_PARENTHESIS | lb::EXCLAMATION | lb::BREAK_SYMBOLS
    ) {
        return Inside;
    }
    // LB14.
    if past_spaces.is_some_and(|unit| unit.class == lb::OPEN_PUNCTUATION) {
        return Inside;
    }
    // LB15a: an opening quote after a break, a bracket or a space.
    if let Some(index) = before
        .past_spaces
        .filter(|index| units.get(*index).is_some_and(|unit| unit.initial_quote))
    {
        let opens = index
            .checked_sub(1)
            .and_then(|index| units.get(index))
            .is_none_or(|unit| {
                matches!(
                    unit.class,
                    lb::MANDATORY_BREAK
                        | lb::CARRIAGE_RETURN
                        | lb::LINE_FEED
                        | lb::NEXT_LINE
                        | lb::OPEN_PUNCTUATION
                        | lb::QUOTATION
                        | lb::GLUE
                        | lb::SPACE
                        | lb::ZERO_WIDTH_SPACE
                )
            });
        if opens {
            return Inside;
        }
    }
    // LB15b: a closing quote before a break, a bracket or a space.
    if after.final_quote
        && later_class.is_none_or(|class| {
            matches!(
                class,
                lb::SPACE
                    | lb::GLUE
                    | lb::WORD_JOINER
                    | lb::CLOSE_PUNCTUATION
                    | lb::QUOTATION
                    | lb::CLOSE_PARENTHESIS
                    | lb::EXCLAMATION
                    | lb::INFIX_NUMERIC
                    | lb::BREAK_SYMBOLS
                    | lb::MANDATORY_BREAK
                    | lb::CARRIAGE_RETURN
                    | lb::LINE_FEED
                    | lb::NEXT_LINE
                    | lb::ZERO_WIDTH_SPACE
            )
        })
    {
        return Inside;
    }
    // LB15c and LB15d.
    if b == lb::INFIX_NUMERIC {
        if a == lb::SPACE && later_class == Some(lb::NUMERIC) {
            return Allowed;
        }
        return Inside;
    }
    // LB16 and LB17.
    if past_spaces.is_some_and(|unit| {
        (matches!(unit.class, lb::CLOSE_PUNCTUATION | lb::CLOSE_PARENTHESIS) && b == lb::NONSTARTER)
            || (unit.class == lb::BREAK_BOTH && b == lb::BREAK_BOTH)
    }) {
        return Inside;
    }
    // LB18.
    if a == lb::SPACE {
        return Allowed;
    }
    // LB19.
    if (b == lb::QUOTATION && !after.initial_quote) || (a == lb::QUOTATION && !before.final_quote) {
        return Inside;
    }
    // LB19a.
    if b == lb::QUOTATION && (!before.east_asian || later.is_none_or(|unit| !unit.east_asian)) {
        return Inside;
    }
    if a == lb::QUOTATION && (!after.east_asian || earlier.is_none_or(|unit| !unit.east_asian)) {
        return Inside;
    }
    // LB20.
    if a == lb::CONTINGENT_BREAK || b == lb::CONTINGENT_BREAK {
        return Allowed;
    }
    // LB20a: a hyphen that starts a word stays with it.
    if hyphen(a)
        && letter(b)
        && earlier.is_none_or(|unit| {
            matches!(
                unit.class,
                lb::MANDATORY_BREAK
                    | lb::CARRIAGE_RETURN
                    | lb::LINE_FEED
                    | lb::NEXT_LINE
                    | lb::SPACE
                    | lb::ZERO_WIDTH_SPACE
                    | lb::CONTINGENT_BREAK
                    | lb::GLUE
            )
        })
    {
        return Inside;
    }
    // LB21.
    if matches!(b, lb::BREAK_AFTER | lb::NONSTARTER) || hyphen(b) || a == lb::BREAK_BEFORE {
        return Inside;
    }
    // LB21a and LB21b.
    if earlier.is_some_and(|unit| unit.class == lb::HEBREW_LETTER)
        && hyphen(a)
        && b != lb::HEBREW_LETTER
    {
        return Inside;
    }
    if a == lb::BREAK_SYMBOLS && b == lb::HEBREW_LETTER {
        return Inside;
    }
    // LB22.
    if b == lb::INSEPARABLE {
        return Inside;
    }
    // LB23.
    if (letter(a) && b == lb::NUMERIC) || (a == lb::NUMERIC && letter(b)) {
        return Inside;
    }
    // LB23a.
    let ideograph = |class: u8| matches!(class, lb::IDEOGRAPHIC | lb::E_BASE | lb::E_MODIFIER);
    if (a == lb::PREFIX_NUMERIC && ideograph(b)) || (ideograph(a) && b == lb::POSTFIX_NUMERIC) {
        return Inside;
    }
    // LB24.
    let affix = |class: u8| matches!(class, lb::PREFIX_NUMERIC | lb::POSTFIX_NUMERIC);
    if (affix(a) && letter(b)) || (letter(a) && affix(b)) {
        return Inside;
    }
    // LB25: a number with its signs, separators and brackets.
    if affix(b) {
        let closed = matches!(a, lb::CLOSE_PUNCTUATION | lb::CLOSE_PARENTHESIS);
        if (closed && at >= 2 && number_before(at - 2)) || number_before(at - 1) {
            return Inside;
        }
    }
    if affix(a) {
        if b == lb::NUMERIC {
            return Inside;
        }
        if b == lb::OPEN_PUNCTUATION {
            let after_bracket = class_at(at.checked_add(2)).map(|unit| unit.class);
            if later_class == Some(lb::NUMERIC)
                || (later_class == Some(lb::INFIX_NUMERIC) && after_bracket == Some(lb::NUMERIC))
            {
                return Inside;
            }
        }
    }
    if b == lb::NUMERIC && (matches!(a, lb::HYPHEN | lb::INFIX_NUMERIC) || number_before(at - 1)) {
        return Inside;
    }
    // LB26: the parts of a Korean syllable.
    if (a == lb::JL && matches!(b, lb::JL | lb::JV | lb::H2 | lb::H3))
        || (matches!(a, lb::JV | lb::H2) && matches!(b, lb::JV | lb::JT))
        || (matches!(a, lb::JT | lb::H3) && b == lb::JT)
    {
        return Joined;
    }
    // LB27.
    if (hangul(a) && b == lb::POSTFIX_NUMERIC) || (a == lb::PREFIX_NUMERIC && hangul(b)) {
        return Inside;
    }
    // LB28.
    if letter(a) && letter(b) {
        return Inside;
    }
    // LB28a: an aksara with what is written under or before it.
    if (a == lb::AKSARA_PREBASE && aksara(after))
        || (aksara(before) && matches!(b, lb::VIRAMA_FINAL | lb::VIRAMA))
        || (earlier.is_some_and(aksara)
            && a == lb::VIRAMA
            && (b == lb::AKSARA || after.dotted_circle))
        || (aksara(before) && aksara(after) && later_class == Some(lb::VIRAMA_FINAL))
    {
        return Joined;
    }
    // LB29.
    if a == lb::INFIX_NUMERIC && letter(b) {
        return Inside;
    }
    // LB30.
    let wordy = |class: u8| letter(class) || class == lb::NUMERIC;
    if (wordy(a) && b == lb::OPEN_PUNCTUATION && !after.east_asian)
        || (a == lb::CLOSE_PARENTHESIS && !before.east_asian && wordy(b))
    {
        return Inside;
    }
    // LB30a: the two halves of a flag.
    if before.flag_open && b == lb::REGIONAL_INDICATOR {
        return Joined;
    }
    // LB30b: an emoji and its skin tone.
    if b == lb::E_MODIFIER && (a == lb::E_BASE || before.reserved_pictograph) {
        return Joined;
    }
    // LB31.
    Allowed
}

#[cfg(test)]
mod tests {
    use super::{Boundary, boundaries, ends_line};

    /// `text` with `|` where a line may end, `!` where it must, and `~`
    /// where it may when a word does not fit.
    fn marked(text: &str) -> String {
        let found = boundaries(text);
        let mut out = String::new();
        for (at, character) in text.char_indices() {
            match found[at] {
                Boundary::Allowed => out.push('|'),
                Boundary::Mandatory => out.push('!'),
                Boundary::Inside if at > 0 => out.push('~'),
                Boundary::Inside | Boundary::Joined => {}
            }
            out.push(character);
        }
        out
    }

    /// `marked`, without the places inside words.
    fn lines(text: &str) -> String {
        marked(text).replace('~', "")
    }

    #[test]
    fn words_break_after_their_spaces() {
        assert_eq!(lines("the quick fox"), "the |quick |fox");
        assert_eq!(lines("two  spaces"), "two  |spaces");
        assert_eq!(lines(" lead"), " |lead");
        assert_eq!(lines("trail "), "trail ");
        assert_eq!(lines(""), "");
        assert_eq!(lines("a"), "a");
    }

    #[test]
    fn a_word_only_breaks_inside_when_it_must() {
        assert_eq!(marked("abc"), "a~b~c");
        assert_eq!(marked("a b"), "a~ |b");
        // A letter keeps its accent, and a flag its two halves.
        assert_eq!(marked("e\u{301}a"), "e\u{301}~a");
        assert_eq!(
            marked("\u{1F1EC}\u{1F1ED}\u{1F1EC}\u{1F1ED}"),
            "\u{1F1EC}\u{1F1ED}|\u{1F1EC}\u{1F1ED}"
        );
        // A family joined by zero width joiners, and a hand with its tone.
        assert_eq!(
            marked("\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}x"),
            "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}|x"
        );
        assert_eq!(marked("\u{1F44D}\u{1F3FD}"), "\u{1F44D}\u{1F3FD}");
        assert_eq!(marked("a\r\nb"), "a~\r\n!b");
    }

    #[test]
    fn line_ends_are_mandatory() {
        assert_eq!(lines("a\nb"), "a\n!b");
        assert_eq!(lines("a\r\nb"), "a\r\n!b");
        assert_eq!(lines("a\rb"), "a\r!b");
        assert_eq!(lines("a\u{2028}b"), "a\u{2028}!b");
        assert_eq!(lines("a\u{85}b"), "a\u{85}!b");
        assert_eq!(lines("a \nb"), "a \n!b");
        assert!(ends_line('\n') && ends_line('\r') && ends_line('\u{2029}'));
        assert!(!ends_line(' ') && !ends_line('a'));
    }

    #[test]
    fn punctuation_stays_with_its_word() {
        assert_eq!(lines("wait, what?"), "wait, |what?");
        assert_eq!(lines("(a b) c"), "(a |b) |c");
        assert_eq!(lines("say \"hi there\" now"), "say |\"hi |there\" |now");
        assert_eq!(
            lines("\u{201C}hi there\u{201D} now"),
            "\u{201C}hi |there\u{201D} |now"
        );
        assert_eq!(lines("a\u{2026}b"), "a\u{2026}|b");
        assert_eq!(lines("a\u{2014}b"), "a|\u{2014}|b");
        assert_eq!(lines("a\u{2014}\u{2014}b"), "a|\u{2014}\u{2014}|b");
        assert_eq!(lines("end.Next"), "end.Next");
        assert_eq!(lines("a/b"), "a/|b");
    }

    #[test]
    fn hyphens_break_after_and_glue_does_not() {
        assert_eq!(lines("well-known fact"), "well-|known |fact");
        assert_eq!(lines("a -b"), "a |-b");
        assert_eq!(lines("-b"), "-b");
        assert_eq!(lines("a\u{2010}b"), "a\u{2010}|b");
        assert_eq!(lines("no\u{A0}break here"), "no\u{A0}break |here");
        assert_eq!(lines("a\u{2060}b c"), "a\u{2060}b |c");
        assert_eq!(lines("a\u{200B}b"), "a\u{200B}|b");
        assert_eq!(lines("soft\u{AD}hyphen"), "soft\u{AD}|hyphen");
    }

    #[test]
    fn numbers_keep_their_signs_and_separators() {
        assert_eq!(lines("costs $1,234.50 now"), "costs |$1,234.50 |now");
        assert_eq!(lines("up 50% today"), "up |50% |today");
        assert_eq!(lines("-5 and (12)"), "-5 |and |(12)");
        assert_eq!(lines("3.14 x 2"), "3.14 |x |2");
        assert_eq!(lines("12:30 a"), "12:30 |a");
        assert_eq!(lines("a1b"), "a1b");
        assert_eq!(lines("no .5"), "no |.5");
    }

    #[test]
    fn ideographs_break_between_and_kana_rules_hold() {
        assert_eq!(
            lines("\u{4E2D}\u{6587}\u{5B57}"),
            "\u{4E2D}|\u{6587}|\u{5B57}"
        );
        // A closing bracket and a small kana do not start a line.
        assert_eq!(
            lines("\u{4E2D}\u{3002}\u{6587}"),
            "\u{4E2D}\u{3002}|\u{6587}"
        );
        assert_eq!(
            lines("\u{30AD}\u{30E3}\u{30C3}"),
            "\u{30AD}\u{30E3}\u{30C3}"
        );
        assert_eq!(
            lines("\u{300C}\u{4E2D}\u{300D}\u{6587}"),
            "\u{300C}\u{4E2D}\u{300D}|\u{6587}"
        );
        assert_eq!(lines("abc\u{4E2D}"), "abc|\u{4E2D}");
        assert_eq!(lines("\u{4E2D}abc"), "\u{4E2D}|abc");
    }

    #[test]
    fn korean_syllables_and_emoji_break_around_not_inside() {
        assert_eq!(
            marked("\u{1100}\u{1161}\u{11A8}"),
            "\u{1100}\u{1161}\u{11A8}"
        );
        assert_eq!(lines("\u{D55C}\u{AE00}"), "\u{D55C}|\u{AE00}");
        assert_eq!(lines("\u{1F600}\u{1F600}"), "\u{1F600}|\u{1F600}");
        assert_eq!(lines("hi\u{1F600}"), "hi|\u{1F600}");
    }

    #[test]
    fn other_scripts_break_at_spaces() {
        assert_eq!(
            lines("\u{645}\u{631}\u{62D}\u{628}\u{627} \u{628}\u{643}"),
            "\u{645}\u{631}\u{62D}\u{628}\u{627} |\u{628}\u{643}"
        );
        assert_eq!(
            lines("\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947} \u{926}\u{941}"),
            "\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947} |\u{926}\u{941}"
        );
        // After a Hebrew word and its hyphen, only another Hebrew word
        // may start a line.
        assert_eq!(
            lines("\u{5E9}\u{5DC}\u{5D5}\u{5DD}-\u{5E2}"),
            "\u{5E9}\u{5DC}\u{5D5}\u{5DD}-|\u{5E2}"
        );
        assert_eq!(
            lines("\u{5E9}\u{5DC}\u{5D5}\u{5DD}-a"),
            "\u{5E9}\u{5DC}\u{5D5}\u{5DD}-a"
        );
        // Thai with nothing in its spelling to show a syllable breaks at
        // its spaces alone.
        assert_eq!(
            lines("\u{E2A}\u{E27}\u{E31} \u{E2A}"),
            "\u{E2A}\u{E27}\u{E31} |\u{E2A}"
        );
    }

    #[test]
    fn scripts_without_spaces_break_where_a_syllable_starts() {
        // Thai: before a vowel written ahead of its consonant, and after a
        // vowel that closes a syllable. "Thai language", "will go".
        assert_eq!(
            lines("\u{E20}\u{E32}\u{E29}\u{E32}\u{E44}\u{E17}\u{E22}"),
            "\u{E20}\u{E32}\u{E29}\u{E32}|\u{E44}\u{E17}\u{E22}"
        );
        assert_eq!(
            lines("\u{E08}\u{E30}\u{E44}\u{E1B}"),
            "\u{E08}\u{E30}|\u{E44}\u{E1B}"
        );
        // The vowel ahead stays with its consonant, and the vowels after
        // with theirs, even when a word is too wide for the line.
        assert_eq!(
            marked("\u{E44}\u{E17}\u{E22}\u{E32}"),
            "\u{E44}\u{E17}~\u{E22}\u{E32}"
        );
        // After sara am, and after the repeat sign.
        assert_eq!(
            lines("\u{E17}\u{E33}\u{E07}\u{E32}\u{E19}"),
            "\u{E17}\u{E33}|\u{E07}\u{E32}\u{E19}"
        );
        assert_eq!(
            lines("\u{E14}\u{E35}\u{E46}\u{E19}\u{E30}"),
            "\u{E14}\u{E35}\u{E46}|\u{E19}\u{E30}"
        );
        // Lao the same way: "Lao language".
        assert_eq!(
            lines("\u{E9E}\u{EB2}\u{EAA}\u{EB2}\u{EA5}\u{EB2}\u{EA7}"),
            "\u{E9E}\u{EB2}\u{EAA}\u{EB2}\u{EA5}\u{EB2}\u{EA7}"
        );
        assert_eq!(
            lines("\u{E84}\u{EB3}\u{EC0}\u{EA7}\u{EBB}\u{EC9}\u{EB2}"),
            "\u{E84}\u{EB3}|\u{EC0}\u{EA7}\u{EBB}\u{EC9}\u{EB2}"
        );
        // Khmer, "Khmer language": before a consonant with a vowel or a
        // stack; the consonant under the stacking sign stays, and the
        // bare consonant that ends the word is not set on a line alone.
        assert_eq!(
            marked("\u{1797}\u{17B6}\u{179F}\u{17B6}\u{1781}\u{17D2}\u{1798}\u{17C2}\u{179A}"),
            "\u{1797}\u{17B6}|\u{179F}\u{17B6}|\u{1781}\u{17D2}\u{1798}\u{17C2}~\u{179A}"
        );
        // Burmese, "Myanmar script": a consonant with its killer ends the
        // syllable before it and stays there; the others open one.
        assert_eq!(
            marked("\u{1019}\u{103C}\u{1014}\u{103A}\u{1019}\u{102C}\u{1005}\u{102C}"),
            "\u{1019}\u{103C}\u{1014}\u{103A}|\u{1019}\u{102C}|\u{1005}\u{102C}"
        );
        // A killer after a vowel is part of the vowel: that consonant
        // opens its syllable. A stacked consonant stays under its stack.
        assert_eq!(
            lines("\u{1000}\u{1031}\u{102C}\u{103A}\u{1010}\u{1031}\u{102C}\u{103A}"),
            "\u{1000}\u{1031}\u{102C}\u{103A}|\u{1010}\u{1031}\u{102C}\u{103A}"
        );
        assert_eq!(
            marked("\u{1000}\u{1039}\u{1000}\u{1014}"),
            "\u{1000}\u{1039}\u{1000}|\u{1014}"
        );
        // Beside a letter of another script nothing changes.
        assert_eq!(marked("a\u{E44}"), "a~\u{E44}");
        // A long run is walked once.
        let started = std::time::Instant::now();
        let long = "\u{1019}\u{103C}\u{1014}\u{103A}\u{1019}\u{102C}".repeat(20_000);
        assert_eq!(
            boundaries(&long)
                .iter()
                .filter(|place| **place == Boundary::Allowed)
                .count(),
            39_999
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn long_runs_of_one_class_are_not_walked_again_for_each_place() {
        let started = std::time::Instant::now();
        for run in [" ", "\u{1F1EC}", "1,", "\u{301}", "\u{200D}"] {
            let text = format!("a{}b", run.repeat(60_000));
            assert_eq!(boundaries(&text).len(), text.len());
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    /// Compare with another implementation's answers, one case a line:
    /// code points in hex, a semicolon, then the character indices where
    /// it breaks, with `!` after a mandatory one. Runs when
    /// `CUTLASS_LINEBREAK_CASES` names such a file.
    #[test]
    #[ignore = "needs a file of cases from another implementation"]
    fn another_implementation_agrees() {
        let Ok(path) = std::env::var("CUTLASS_LINEBREAK_CASES") else {
            return;
        };
        let cases = std::fs::read_to_string(path).expect("cases");
        let mut wrong = Vec::new();
        let mut skipped = 0;
        for case in cases.lines() {
            let (points, expected) = case.split_once(';').expect("case");
            let text: String = points
                .split_whitespace()
                .map(|hex| {
                    char::from_u32(u32::from_str_radix(hex, 16).expect("hex")).expect("char")
                })
                .collect();
            // The other implementation breaks these four scripts by its
            // dictionaries, where this one goes by their spelling.
            if text.chars().any(|character| {
                matches!(
                    u32::from(character),
                    0x0E00..=0x0EFF | 0x1000..=0x109F | 0x1780..=0x17FF
                )
            }) {
                skipped += 1;
                continue;
            }
            let found = boundaries(&text);
            let ours: Vec<String> = text
                .char_indices()
                .enumerate()
                .filter_map(|(index, (at, _))| match found[at] {
                    Boundary::Allowed => Some(index.to_string()),
                    Boundary::Mandatory => Some(format!("{index}!")),
                    Boundary::Inside | Boundary::Joined => None,
                })
                .collect();
            if ours.join(" ") != expected.trim() {
                wrong.push(format!(
                    "{points}: ours [{}] theirs [{}]",
                    ours.join(" "),
                    expected.trim()
                ));
            }
        }
        let total = cases.lines().count() - skipped;
        let report: Vec<&String> = wrong.iter().take(60).collect();
        assert!(
            wrong.is_empty(),
            "{} of {total} differ: {report:#?}",
            wrong.len()
        );
    }

    #[test]
    fn every_entry_is_on_a_character() {
        let text = "word \u{1F600}e\u{301} ".repeat(4_000);
        let found = boundaries(&text);
        assert_eq!(found.len(), text.len());
        for (at, boundary) in found.iter().enumerate() {
            if !text.is_char_boundary(at) {
                assert_eq!(*boundary, Boundary::Joined);
            }
        }
        assert_eq!(found[0], Boundary::Joined);
    }
}
