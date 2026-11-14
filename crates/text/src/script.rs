//! Scripts: which way each runs, and what a font calls it.

use cutlass_font::Tag;

use crate::unicode::props;

/// A script, by its ISO 15924 code such as `Latn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Script(pub(crate) [u8; 4]);

/// Which shaping a script needs beyond the font's own lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shaping {
    /// The font's lookups alone.
    Plain,
    /// Letters take a form from the letters they join.
    Joined,
    /// Syllables are found and their parts put in the order they are
    /// written, as Devanagari and the scripts like it are.
    Syllabic,
    /// The same for Khmer, by its own rules.
    Khmer,
    /// And for the Myanmar script.
    Myanmar,
}

impl Script {
    pub(crate) const COMMON: Self = Self(*b"Zyyy");
    pub(crate) const INHERITED: Self = Self(*b"Zinh");
    pub(crate) const UNKNOWN: Self = Self(*b"Zzzz");
    pub(crate) const LATIN: Self = Self(*b"Latn");

    pub(crate) fn of(character: char) -> Self {
        Self(props(character).script())
    }

    /// A script of its own, which a run can be named after: not the
    /// characters every script shares.
    pub(crate) fn real(self) -> bool {
        !matches!(self, Self::COMMON | Self::INHERITED | Self::UNKNOWN)
    }

    /// The script is written right to left.
    pub(crate) fn right_to_left(self) -> bool {
        matches!(
            &self.0,
            b"Arab"
                | b"Hebr"
                | b"Syrc"
                | b"Thaa"
                | b"Nkoo"
                | b"Samr"
                | b"Mand"
                | b"Adlm"
                | b"Rohg"
                | b"Phnx"
                | b"Armi"
                | b"Avst"
                | b"Cprt"
                | b"Khar"
                | b"Lydi"
                | b"Nbat"
                | b"Narb"
                | b"Sarb"
                | b"Palm"
                | b"Phli"
                | b"Phlp"
                | b"Prti"
                | b"Orkh"
                | b"Hatr"
                | b"Mani"
                | b"Merc"
                | b"Mero"
                | b"Hung"
                | b"Sogd"
                | b"Sogo"
                | b"Elym"
                | b"Chrs"
                | b"Ougr"
                | b"Yezi"
                | b"Gara"
        )
    }

    pub(crate) fn shaping(self) -> Shaping {
        match &self.0 {
            b"Arab" | b"Syrc" | b"Nkoo" | b"Mong" | b"Phag" | b"Mand" | b"Mani" | b"Phlp"
            | b"Adlm" | b"Rohg" | b"Sogd" | b"Chrs" | b"Ougr" => Shaping::Joined,
            b"Deva" | b"Beng" | b"Guru" | b"Gujr" | b"Orya" | b"Taml" | b"Telu" | b"Knda"
            | b"Mlym" => Shaping::Syllabic,
            b"Khmr" => Shaping::Khmer,
            b"Mymr" => Shaping::Myanmar,
            _ => Shaping::Plain,
        }
    }

    /// The names a font may know the script by, the newest first.
    pub(crate) fn tags(self) -> Vec<Tag> {
        match &self.0 {
            b"Deva" => vec![*b"dev2", *b"deva"],
            b"Beng" => vec![*b"bng2", *b"beng"],
            b"Guru" => vec![*b"gur2", *b"guru"],
            b"Gujr" => vec![*b"gjr2", *b"gujr"],
            b"Orya" => vec![*b"ory2", *b"orya"],
            b"Taml" => vec![*b"tml2", *b"taml"],
            b"Telu" => vec![*b"tel2", *b"telu"],
            b"Knda" => vec![*b"knd2", *b"knda"],
            b"Mlym" => vec![*b"mlm2", *b"mlym"],
            b"Mymr" => vec![*b"mym2", *b"mymr"],
            b"Hira" | b"Kana" => vec![*b"kana"],
            b"Laoo" => vec![*b"lao "],
            b"Yiii" => vec![*b"yi  "],
            b"Nkoo" => vec![*b"nko "],
            b"Vaii" => vec![*b"vai "],
            b"Zyyy" | b"Zinh" | b"Zzzz" => Vec::new(),
            [first, rest @ ..] => vec![[first.to_ascii_lowercase(), rest[0], rest[1], rest[2]]],
        }
    }
}

/// The script each character of `text` is shaped as: its own, or for a
/// character every script shares, the script of the text before it, and
/// at the start of the text the first script that follows.
pub(crate) fn resolve(text: &str) -> Vec<Script> {
    let mut scripts: Vec<Script> = text.chars().map(Script::of).collect();
    let first = scripts
        .iter()
        .copied()
        .find(|script| script.real())
        .unwrap_or(Script::LATIN);
    let mut current = first;
    for script in &mut scripts {
        if script.real() {
            current = *script;
        } else {
            *script = current;
        }
    }
    scripts
}

#[cfg(test)]
mod tests {
    use super::{Script, Shaping, resolve};

    #[test]
    fn scripts_know_their_direction_shaping_and_names() {
        assert_eq!(Script::of('A'), Script::LATIN);
        assert_eq!(Script::of(' '), Script::COMMON);
        assert_eq!(Script::of('\u{301}'), Script::INHERITED);
        assert!(Script::of('\u{5D0}').right_to_left());
        assert!(Script::of('\u{628}').right_to_left());
        assert!(!Script::LATIN.right_to_left() && !Script::of('\u{915}').right_to_left());
        assert_eq!(Script::of('\u{628}').shaping(), Shaping::Joined);
        assert_eq!(Script::of('\u{915}').shaping(), Shaping::Syllabic);
        assert_eq!(Script::of('\u{5D0}').shaping(), Shaping::Plain);
        assert_eq!(Script::LATIN.tags(), [*b"latn"]);
        assert_eq!(Script::of('\u{915}').tags(), [*b"dev2", *b"deva"]);
        assert_eq!(Script::of('\u{E01}').tags(), [*b"thai"]);
        assert_eq!(Script::of('\u{E81}').tags(), [*b"lao "]);
        assert!(Script::COMMON.tags().is_empty());
    }

    #[test]
    fn shared_characters_take_the_script_around_them() {
        let tags = |text: &str| -> Vec<String> {
            resolve(text)
                .iter()
                .map(|script| String::from_utf8_lossy(&script.0).into_owned())
                .collect()
        };
        assert_eq!(tags("a b"), ["Latn", "Latn", "Latn"]);
        assert_eq!(tags("\u{628} a"), ["Arab", "Arab", "Latn"]);
        // Leading punctuation takes the first script that follows.
        assert_eq!(tags("(\u{5D0})"), ["Hebr", "Hebr", "Hebr"]);
        assert_eq!(tags("1 2"), ["Latn", "Latn", "Latn"]);
        assert_eq!(tags("e\u{301}"), ["Latn", "Latn"]);
        assert!(resolve("").is_empty());
    }
}
