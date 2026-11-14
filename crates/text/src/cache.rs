//! What is kept from one title to the next, so a frame of text does not
//! start from the font file each time.
//!
//! A title that moves, and every caption, is opened again for each frame.
//! Two parts of that are the same from frame to frame: the image of each
//! glyph, and the title's layout while its words, faces and sizes do not
//! change. A [`TextCache`] keeps both. Glyph images are kept by face and
//! glyph, so two titles in one face share them; a layout is kept by
//! everything that goes into it. The cache holds a bounded amount and
//! forgets everything when it is full.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use cutlass_font::Font;
use cutlass_model::MaterialText;

use crate::TextError;
use crate::atlas::GlyphImage;
use crate::layout::Layout;
use crate::richtext::Run;

/// The most bytes of glyph images kept.
const MAX_GLYPH_BYTES: usize = 64 * 1024 * 1024;
/// The most layouts kept.
const MAX_LAYOUTS: usize = 256;

#[derive(Default)]
struct Kept {
    glyphs: HashMap<(u64, u16), Arc<GlyphImage>>,
    glyph_bytes: usize,
    layouts: HashMap<u64, Arc<Layout>>,
    /// How many glyph images were asked for and not found.
    painted: u64,
    /// How many layouts were asked for and not found.
    laid: u64,
}

/// Glyph images and layouts kept across titles and frames.
#[derive(Default)]
pub struct TextCache {
    kept: Mutex<Kept>,
}

impl std::fmt::Debug for TextCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextCache")
            .field("glyphs", &self.glyphs())
            .field("layouts", &self.layouts())
            .finish()
    }
}

impl TextCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many glyph images are kept.
    #[must_use]
    pub fn glyphs(&self) -> usize {
        self.kept.lock().map_or(0, |kept| kept.glyphs.len())
    }

    /// How many layouts are kept.
    #[must_use]
    pub fn layouts(&self) -> usize {
        self.kept.lock().map_or(0, |kept| kept.layouts.len())
    }

    /// How many glyph images had to be painted, and how many layouts had
    /// to be made, since the cache was new.
    #[must_use]
    pub fn misses(&self) -> (u64, u64) {
        self.kept
            .lock()
            .map_or((0, 0), |kept| (kept.painted, kept.laid))
    }

    /// The image of glyph `glyph` of the face with fingerprint `face`,
    /// painted by `paint` the first time it is asked for.
    pub(crate) fn glyph(
        &self,
        face: u64,
        glyph: u16,
        paint: impl FnOnce() -> Result<GlyphImage, TextError>,
    ) -> Result<Arc<GlyphImage>, TextError> {
        if let Ok(kept) = self.kept.lock()
            && let Some(image) = kept.glyphs.get(&(face, glyph))
        {
            return Ok(Arc::clone(image));
        }
        let image = Arc::new(paint()?);
        if let Ok(mut kept) = self.kept.lock() {
            let bytes = image.sdf.len() + image.color.len();
            if kept.glyph_bytes + bytes > MAX_GLYPH_BYTES {
                kept.glyphs.clear();
                kept.glyph_bytes = 0;
            }
            kept.painted += 1;
            kept.glyph_bytes += bytes;
            kept.glyphs.insert((face, glyph), Arc::clone(&image));
        }
        Ok(image)
    }

    /// The layout with key `key`, made by `lay` the first time.
    pub(crate) fn layout(
        &self,
        key: u64,
        lay: impl FnOnce() -> Result<Layout, TextError>,
    ) -> Result<Arc<Layout>, TextError> {
        if let Ok(kept) = self.kept.lock()
            && let Some(layout) = kept.layouts.get(&key)
        {
            return Ok(Arc::clone(layout));
        }
        let layout = Arc::new(lay()?);
        if let Ok(mut kept) = self.kept.lock() {
            if kept.layouts.len() >= MAX_LAYOUTS {
                kept.layouts.clear();
            }
            kept.laid += 1;
            kept.layouts.insert(key, Arc::clone(&layout));
        }
        Ok(layout)
    }
}

/// A key for the layout of `material`'s `text` in `fonts`: everything the
/// layout is made from. `faces` is the face each character is set in.
pub(crate) fn layout_key(
    fonts: &[Font<'_>],
    material: &MaterialText,
    text: &str,
    runs: &[Run],
    faces: &dyn Fn(usize) -> usize,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    for font in fonts {
        font.fingerprint().hash(&mut hasher);
    }
    text.hash(&mut hasher);
    for (at, _) in text.char_indices() {
        faces(at).hash(&mut hasher);
    }
    for run in runs {
        (run.start, run.end, run.size.to_bits()).hash(&mut hasher);
    }
    for value in [
        material.font_size,
        material.letter_spacing,
        material.line_spacing,
        material.line_max_width,
    ] {
        value.to_bits().hash(&mut hasher);
    }
    (material.alignment, material.typesetting).hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{TextCache, layout_key};
    use crate::TextError;
    use crate::atlas::{GlyphImage, GlyphKind};
    use cutlass_model::MaterialText;

    const SAMPLE: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const SHAPING: &[u8] = include_bytes!("../../font/fixtures/shaping.ttf");

    fn image(bytes: usize) -> GlyphImage {
        GlyphImage {
            width: 1,
            height: 1,
            left: 0.0,
            top: 0.0,
            unit: 1.0,
            pixels_per_em: 64.0,
            kind: GlyphKind::Distance,
            sdf: vec![0; bytes],
            color: Vec::new(),
        }
    }

    #[test]
    fn a_glyph_is_painted_once_for_its_face() {
        let cache = TextCache::new();
        let first = cache.glyph(7, 2, || Ok(image(4))).expect("painted");
        let again = cache
            .glyph(7, 2, || Err(TextError::Limit))
            .expect("kept, so not painted");
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        // Another face's glyph of the same number is its own.
        let other = cache.glyph(8, 2, || Ok(image(4))).expect("other face");
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        assert_eq!((cache.glyphs(), cache.misses().0), (2, 2));
        // A glyph that fails to paint is not kept, and is tried again.
        assert_eq!(
            cache.glyph(7, 3, || Err(TextError::Limit)).err(),
            Some(TextError::Limit)
        );
        assert!(cache.glyph(7, 3, || Ok(image(1))).is_ok());
    }

    #[test]
    fn a_full_cache_forgets_and_starts_again() {
        let cache = TextCache::new();
        let big = super::MAX_GLYPH_BYTES / 2 + 1;
        cache.glyph(1, 1, || Ok(image(big))).expect("first");
        cache.glyph(1, 2, || Ok(image(big))).expect("second");
        // The second did not fit beside the first.
        assert_eq!(cache.glyphs(), 1);
        for key in 0..=u64::try_from(super::MAX_LAYOUTS).expect("count") {
            cache
                .layout(key, || Err(TextError::Limit))
                .expect_err("not laid");
        }
        assert_eq!(cache.layouts(), 0);
    }

    #[test]
    fn a_layout_key_changes_with_what_the_layout_is_made_from() {
        let sample = cutlass_font::read(SAMPLE).expect("sample");
        let shaping = cutlass_font::read(SHAPING).expect("shaping");
        let mut text = MaterialText::new("t");
        text.content = "AB".to_owned();
        text.font_size = 64.0;
        let key = |fonts: &[cutlass_font::Font<'_>], text: &MaterialText| {
            layout_key(fonts, text, &text.content, &[], &|_| 0)
        };
        let base = key(std::slice::from_ref(&sample), &text);
        assert_eq!(base, key(std::slice::from_ref(&sample), &text));
        assert_ne!(base, key(std::slice::from_ref(&shaping), &text));
        let mut changed = text.clone();
        changed.content = "BA".to_owned();
        assert_ne!(base, key(std::slice::from_ref(&sample), &changed));
        let mut sized = text.clone();
        sized.font_size = 65.0;
        assert_ne!(base, key(std::slice::from_ref(&sample), &sized));
        let mut wrapped = text.clone();
        wrapped.line_max_width = 100.0;
        assert_ne!(base, key(std::slice::from_ref(&sample), &wrapped));
        // What only colors the letters does not change the layout.
        let mut colored = text.clone();
        colored.text_color = "#FF0000".to_owned();
        assert_eq!(base, key(std::slice::from_ref(&sample), &colored));
        // The face a character is set in does.
        assert_ne!(
            base,
            layout_key(
                std::slice::from_ref(&sample),
                &text,
                &text.content,
                &[],
                &|at| at
            )
        );
    }
}
