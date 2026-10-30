//! Stable package ids, still posters, and preview URLs.
//!
//! Implemented here: five blend stand-ins, three distortion and glitch
//! stand-ins with a defined CPU result (invert, contrast, and pixelate), and
//! three text-animation packages whose kinds are in, out, and loop. Pixelate's
//! block is [`PIXELATE_BLOCK`] pixels on each side. Text-animation kinds use
//! [`AnimType`]. Each row has a stable id, a still poster path, and a preview
//! URL. This module stores those rows and the pixelate block. Preview videos
//! stay a URL for storage to fetch. This module does not start a transfer and
//! does not sample a picture.
//!
//! Unsupported: shader packages, a script host, a shader dialect, and the rest
//! of a catalog of about 200 effect packages and about 30 text animations.
//! Those ids are not listed here. A checksum does not authorize running
//! package bytes. Text-animation kinds are parameters evaluated elsewhere;
//! this module does not write a draft.
//!
//! [`entry`] walks the eleven rows in [`entries`] order. That scan allocates
//! nothing.

use cutlass_model::AnimType;

/// Stable id for the source-over blend stand-in.
pub const SOURCE_OVER: &str = "blend-source-over";
/// Stable id for the multiply blend stand-in.
pub const MULTIPLY: &str = "blend-multiply";
/// Stable id for the screen blend stand-in.
pub const SCREEN: &str = "blend-screen";
/// Stable id for the add blend stand-in.
pub const ADD: &str = "blend-add";
/// Stable id for the difference blend stand-in.
pub const DIFFERENCE: &str = "blend-difference";
/// Stable id for the invert distortion stand-in.
pub const INVERT: &str = "distort-invert";
/// Stable id for the contrast distortion stand-in.
pub const CONTRAST: &str = "distort-contrast";
/// Stable id for the pixelate distortion stand-in.
pub const PIXELATE: &str = "distort-pixelate";
/// Stable id for the text-animation in package.
pub const ANIM_IN: &str = "text-anim-in";
/// Stable id for the text-animation out package.
pub const ANIM_OUT: &str = "text-anim-out";
/// Stable id for the text-animation loop package.
pub const ANIM_LOOP: &str = "text-anim-loop";

/// Pixelate stand-in block, in pixels on each side.
pub const PIXELATE_BLOCK: u32 = 2;

/// Blend stand-in stored on a catalog row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Source over.
    SourceOver,
    /// Multiply.
    Multiply,
    /// Screen.
    Screen,
    /// Add.
    Add,
    /// Difference.
    Difference,
}

/// Distortion stand-in with a defined CPU result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distort {
    /// Invert.
    Invert,
    /// Contrast.
    Contrast,
    /// Pixelate, in blocks of [`PIXELATE_BLOCK`] pixels.
    Pixelate,
}

/// Kind of one catalog row.
///
/// Text-animation rows are [`AnimType::In`], [`AnimType::Out`], and
/// [`AnimType::Loop`]. Caption, group, and none are absent from this catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageKind {
    /// A blend stand-in.
    Blend(BlendMode),
    /// A distortion stand-in.
    Distort(Distort),
    /// A text-animation package.
    TextAnim(AnimType),
}

/// One catalog row: stable id, still poster, and preview URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Stable package id.
    pub id: &'static str,
    /// Still poster path (`posters/{id}.png`).
    pub poster: &'static str,
    /// Preview video URL. Storage fetches it.
    pub preview_url: &'static str,
    /// Blend, distortion, or text-animation kind.
    pub kind: PackageKind,
}

static ENTRIES: [CatalogEntry; 11] = [
    CatalogEntry {
        id: SOURCE_OVER,
        poster: "posters/blend-source-over.png",
        preview_url: "https://assets.invalid/effects/blend-source-over.mp4",
        kind: PackageKind::Blend(BlendMode::SourceOver),
    },
    CatalogEntry {
        id: MULTIPLY,
        poster: "posters/blend-multiply.png",
        preview_url: "https://assets.invalid/effects/blend-multiply.mp4",
        kind: PackageKind::Blend(BlendMode::Multiply),
    },
    CatalogEntry {
        id: SCREEN,
        poster: "posters/blend-screen.png",
        preview_url: "https://assets.invalid/effects/blend-screen.mp4",
        kind: PackageKind::Blend(BlendMode::Screen),
    },
    CatalogEntry {
        id: ADD,
        poster: "posters/blend-add.png",
        preview_url: "https://assets.invalid/effects/blend-add.mp4",
        kind: PackageKind::Blend(BlendMode::Add),
    },
    CatalogEntry {
        id: DIFFERENCE,
        poster: "posters/blend-difference.png",
        preview_url: "https://assets.invalid/effects/blend-difference.mp4",
        kind: PackageKind::Blend(BlendMode::Difference),
    },
    CatalogEntry {
        id: INVERT,
        poster: "posters/distort-invert.png",
        preview_url: "https://assets.invalid/effects/distort-invert.mp4",
        kind: PackageKind::Distort(Distort::Invert),
    },
    CatalogEntry {
        id: CONTRAST,
        poster: "posters/distort-contrast.png",
        preview_url: "https://assets.invalid/effects/distort-contrast.mp4",
        kind: PackageKind::Distort(Distort::Contrast),
    },
    CatalogEntry {
        id: PIXELATE,
        poster: "posters/distort-pixelate.png",
        preview_url: "https://assets.invalid/effects/distort-pixelate.mp4",
        kind: PackageKind::Distort(Distort::Pixelate),
    },
    CatalogEntry {
        id: ANIM_IN,
        poster: "posters/text-anim-in.png",
        preview_url: "https://assets.invalid/effects/text-anim-in.mp4",
        kind: PackageKind::TextAnim(AnimType::In),
    },
    CatalogEntry {
        id: ANIM_OUT,
        poster: "posters/text-anim-out.png",
        preview_url: "https://assets.invalid/effects/text-anim-out.mp4",
        kind: PackageKind::TextAnim(AnimType::Out),
    },
    CatalogEntry {
        id: ANIM_LOOP,
        poster: "posters/text-anim-loop.png",
        preview_url: "https://assets.invalid/effects/text-anim-loop.mp4",
        kind: PackageKind::TextAnim(AnimType::Loop),
    },
];

/// The row for `id`, or nothing when the id is unknown.
///
/// The walk is the eleven rows from [`entries`], in that order. It allocates
/// nothing.
#[must_use]
pub fn entry(id: &str) -> Option<&'static CatalogEntry> {
    entries().iter().find(|row| row.id == id)
}

/// The eleven catalog rows, in stable order.
#[must_use]
pub fn entries() -> &'static [CatalogEntry] {
    &ENTRIES
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use cutlass_model::AnimType;

    use super::{
        ADD, ANIM_IN, ANIM_LOOP, ANIM_OUT, BlendMode, CONTRAST, DIFFERENCE, Distort, INVERT,
        MULTIPLY, PIXELATE, PIXELATE_BLOCK, PackageKind, SCREEN, SOURCE_OVER, entries, entry,
    };

    #[test]
    fn every_row_round_trips_its_poster_and_preview() {
        for row in entries() {
            let found = entry(row.id).unwrap();
            let id = found.id;
            assert_eq!(found.id, row.id);
            assert_eq!(found.poster, format!("posters/{id}.png"));
            assert_eq!(
                found.preview_url,
                format!("https://assets.invalid/effects/{id}.mp4")
            );
            assert!(!found.poster.starts_with("http"));
            assert!(found.preview_url.starts_with("https://"));
        }
    }

    #[test]
    fn entries_follow_the_eleven_ids() {
        let ids = [
            SOURCE_OVER,
            MULTIPLY,
            SCREEN,
            ADD,
            DIFFERENCE,
            INVERT,
            CONTRAST,
            PIXELATE,
            ANIM_IN,
            ANIM_OUT,
            ANIM_LOOP,
        ];
        assert_eq!(entries().len(), 11);
        assert_eq!(entries().len(), ids.len());
        for (row, id) in entries().iter().zip(ids) {
            assert_eq!(row.id, id);
        }
    }

    #[test]
    fn unknown_ids_are_missing() {
        assert_eq!(entry(""), None);
        assert_eq!(entry("shader-unknown"), None);
        assert_eq!(entry("blend-source-over "), None);
    }

    #[test]
    fn kinds_are_blend_distort_and_text_anim() {
        assert_eq!(kind(SOURCE_OVER), PackageKind::Blend(BlendMode::SourceOver));
        assert_eq!(kind(MULTIPLY), PackageKind::Blend(BlendMode::Multiply));
        assert_eq!(kind(SCREEN), PackageKind::Blend(BlendMode::Screen));
        assert_eq!(kind(ADD), PackageKind::Blend(BlendMode::Add));
        assert_eq!(kind(DIFFERENCE), PackageKind::Blend(BlendMode::Difference));
        assert_eq!(kind(INVERT), PackageKind::Distort(Distort::Invert));
        assert_eq!(kind(CONTRAST), PackageKind::Distort(Distort::Contrast));
        assert_eq!(kind(PIXELATE), PackageKind::Distort(Distort::Pixelate));
        assert_eq!(kind(ANIM_IN), PackageKind::TextAnim(AnimType::In));
        assert_eq!(kind(ANIM_OUT), PackageKind::TextAnim(AnimType::Out));
        assert_eq!(kind(ANIM_LOOP), PackageKind::TextAnim(AnimType::Loop));
    }

    #[test]
    fn entry_is_stable_for_the_same_id() {
        for row in entries() {
            let first = entry(row.id).unwrap();
            let second = entry(row.id).unwrap();
            assert!(ptr::eq(first, second));
            assert_eq!(first.id, second.id);
        }
    }

    #[test]
    fn pixelate_block_is_two() {
        assert_eq!(PIXELATE_BLOCK, 2);
    }

    fn kind(id: &str) -> PackageKind {
        entry(id).unwrap().kind
    }
}
