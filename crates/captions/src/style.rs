//! Caption styles: how unread, spoken and read words look, and how many
//! words a page shows.
//!
//! White is the caption's own color, so a style that leaves a word white
//! keeps whatever color the caption has. A lift is a share of the font
//! size, up; a box is drawn behind the word being spoken.

use crate::{Channel, Key, PageLimits, RangeStyles, StyleCurves};

/// How one kind of word looks: its color, the box behind it, its size and
/// how far it is lifted, in ems.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WordLook {
    /// Straight RGBA, 0 to 1; white keeps the caption's color.
    pub color: [f64; 4],
    /// Straight RGBA; a clear box is none.
    pub background: [f64; 4],
    pub scale: f64,
    pub lift: f64,
}

const PLAIN: WordLook = WordLook {
    color: [1.0, 1.0, 1.0, 1.0],
    background: [0.0, 0.0, 0.0, 0.0],
    scale: 1.0,
    lift: 0.0,
};

const fn color(red: u8, green: u8, blue: u8, alpha: f64) -> [f64; 4] {
    [
        red as f64 / 255.0,
        green as f64 / 255.0,
        blue as f64 / 255.0,
        alpha,
    ]
}

const fn ink(color: [f64; 4]) -> WordLook {
    WordLook { color, ..PLAIN }
}

/// One style on the Captions panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptionStyle {
    pub key: &'static str,
    pub name: &'static str,
    pub limits: PageLimits,
    pub unread: WordLook,
    pub reading: WordLook,
    pub read: WordLook,
}

const fn limits(words_per_line: usize, lines_per_page: usize) -> PageLimits {
    PageLimits {
        words_per_line,
        lines_per_page,
    }
}

const YELLOW: [f64; 4] = color(0xFF, 0xE1, 0x4A, 1.0);

/// The ten styles, in the order the panel shows them.
pub const CAPTION_STYLES: [CaptionStyle; 10] = [
    CaptionStyle {
        key: "classic",
        name: "Classic",
        limits: limits(6, 2),
        unread: PLAIN,
        reading: PLAIN,
        read: PLAIN,
    },
    CaptionStyle {
        key: "karaoke",
        name: "Karaoke",
        limits: limits(5, 2),
        unread: PLAIN,
        reading: ink(YELLOW),
        read: ink(YELLOW),
    },
    CaptionStyle {
        key: "highlight",
        name: "Highlight",
        limits: limits(4, 2),
        unread: PLAIN,
        reading: WordLook {
            color: color(0x11, 0x11, 0x11, 1.0),
            background: YELLOW,
            ..PLAIN
        },
        read: PLAIN,
    },
    CaptionStyle {
        key: "pop",
        name: "Pop",
        limits: limits(4, 1),
        unread: PLAIN,
        reading: WordLook {
            color: YELLOW,
            scale: 1.25,
            ..PLAIN
        },
        read: PLAIN,
    },
    CaptionStyle {
        key: "word-by-word",
        name: "Word by word",
        limits: limits(4, 2),
        unread: ink(color(0xFF, 0xFF, 0xFF, 0.0)),
        reading: WordLook {
            scale: 1.15,
            ..PLAIN
        },
        read: PLAIN,
    },
    CaptionStyle {
        key: "one-word",
        name: "One word",
        limits: limits(1, 1),
        unread: PLAIN,
        reading: WordLook {
            scale: 1.2,
            ..PLAIN
        },
        read: PLAIN,
    },
    CaptionStyle {
        key: "neon",
        name: "Neon",
        limits: limits(5, 2),
        unread: ink(color(0xFF, 0xFF, 0xFF, 0.7)),
        reading: ink(color(0x3D, 0xE0, 0xFF, 1.0)),
        read: ink(color(0xFF, 0xFF, 0xFF, 0.7)),
    },
    CaptionStyle {
        key: "bold-box",
        name: "Bold box",
        limits: limits(3, 1),
        unread: PLAIN,
        reading: WordLook {
            background: color(0xFF, 0x3B, 0x6B, 1.0),
            scale: 1.1,
            ..PLAIN
        },
        read: PLAIN,
    },
    CaptionStyle {
        key: "fade",
        name: "Fade",
        limits: limits(6, 2),
        unread: PLAIN,
        reading: PLAIN,
        read: ink(color(0xFF, 0xFF, 0xFF, 0.45)),
    },
    CaptionStyle {
        key: "rise",
        name: "Rise",
        limits: limits(4, 1),
        unread: PLAIN,
        reading: WordLook {
            color: color(0x7C, 0xFF, 0x9B, 1.0),
            lift: 0.15,
            ..PLAIN
        },
        read: PLAIN,
    },
];

/// The style stored as `key`, if it is one.
#[must_use]
pub fn caption_style(key: &str) -> Option<&'static CaptionStyle> {
    CAPTION_STYLES.iter().find(|style| style.key == key)
}

impl CaptionStyle {
    /// The curves [`crate::sample_caption`] reads, for a caption drawn at
    /// `font_size` pixels: one held value each.
    #[must_use]
    pub fn styles(&self, font_size: f64) -> RangeStyles {
        let keyword = self.reading;
        RangeStyles {
            unread: curves(self.unread, font_size),
            reading: curves(self.reading, font_size),
            read: curves(self.read, font_size),
            keyword: curves(keyword, font_size),
        }
    }
}

fn held(value: f64) -> Channel {
    Channel {
        keys: vec![Key { millis: 0, value }],
    }
}

/// How long a sized word takes to grow to its size, in milliseconds.
pub const GROW_MS: i64 = 120;

/// From its own size to `scale` over [`GROW_MS`], then held.
fn grow(scale: f64) -> Channel {
    Channel {
        keys: vec![
            Key {
                millis: 0,
                value: 1.0,
            },
            Key {
                millis: GROW_MS,
                value: scale,
            },
        ],
    }
}

fn curves(look: WordLook, font_size: f64) -> StyleCurves {
    let mut curves = StyleCurves::identity();
    if look.color != PLAIN.color {
        curves.color = look.color.map(held);
    }
    if look.background[3] > 0.0 {
        curves.background = look.background.map(held);
    }
    if look.scale != 1.0 {
        // A sized word grows to its size as it starts.
        curves.scale_x = grow(look.scale);
        curves.scale_y = grow(look.scale);
        curves.from_word_start = true;
    }
    if look.lift != 0.0 {
        curves.position_y = held(-look.lift * font_size);
    }
    curves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_styles_each_with_a_key_and_paging() {
        let mut keys: Vec<_> = CAPTION_STYLES.iter().map(|style| style.key).collect();
        keys.dedup();
        assert_eq!(keys.len(), 10);
        assert!(
            CAPTION_STYLES
                .iter()
                .all(|style| style.limits.words_per_line > 0)
        );
        assert_eq!(caption_style("pop").map(|style| style.name), Some("Pop"));
        assert_eq!(caption_style("loud"), None);
        // The model knows them by the same keys, so an edit can check one.
        assert_eq!(
            CAPTION_STYLES.map(|style| style.key),
            cutlass_model::CAPTION_STYLE_KEYS
        );
    }

    #[test]
    fn a_sized_word_grows_in_from_its_start() {
        let pop = caption_style("pop").unwrap().styles(64.0);
        assert!(pop.reading.from_word_start);
        assert_eq!(pop.reading.sample(-10).unwrap().scale_x, 1.0);
        let half = pop.reading.sample(GROW_MS / 2).unwrap().scale_x;
        assert!(half > 1.0 && half < 1.25, "{half}");
        assert_eq!(pop.reading.sample(GROW_MS).unwrap().scale_x, 1.25);
        // Held looks keep the caption's clock.
        let karaoke = caption_style("karaoke").unwrap().styles(64.0);
        assert!(!karaoke.reading.from_word_start);
    }

    #[test]
    fn a_style_samples_to_its_looks_and_plain_stays_identity() {
        let pop = caption_style("pop").unwrap().styles(64.0);
        let reading = pop.reading.sample(1_234).unwrap();
        assert_eq!(reading.scale_x, 1.25);
        assert!((reading.color.green - f64::from(0xE1_u8) / 255.0).abs() < 1e-9);
        let unread = pop.unread.sample(0).unwrap();
        assert_eq!(unread, crate::SampledStyle::identity());
        let rise = caption_style("rise").unwrap().styles(100.0);
        assert!((rise.reading.sample(5).unwrap().position_y + 15.0).abs() < 1e-9);
        let highlight = caption_style("highlight").unwrap().styles(64.0);
        assert_eq!(highlight.reading.sample(0).unwrap().background.alpha, 1.0);
    }
}
