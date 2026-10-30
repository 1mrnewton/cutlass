//! One mesh record per glyph.
//!
//! The record is the letter a later fill, stroke, or shadow can attach to.
//! The string is not flattened into one picture, and the material id is the
//! one the caller already has.

use cutlass_font::Font;
use cutlass_model::{MaterialId, MaterialText};

use crate::TextError;
use crate::atlas::{Atlas, GlyphImage};
use crate::layout::{Layout, PlacedGlyph, layout};

/// A glyph quad in pixel space. The picture origin is the top left.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphRecord {
    pub glyph_id: u16,
    pub missing: bool,
    pub source: usize,
    pub line: u32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Glyph quads for one text material. This is not a picture.
#[derive(Clone, Debug, PartialEq)]
pub struct TextMesh {
    pub material_id: MaterialId,
    pub glyphs: Vec<GlyphRecord>,
}

/// Build a mesh from a layout whose glyphs are already in `atlas`.
pub fn build(
    material_id: MaterialId,
    layout: &Layout,
    atlas: &Atlas,
) -> Result<TextMesh, TextError> {
    let mut glyphs = Vec::new();
    for (index, line) in layout.lines.iter().enumerate() {
        let line_index = u32::try_from(index).map_err(|_| TextError::Limit)?;
        for placed in &line.glyphs {
            let Some(image) = atlas.image(placed.glyph_id) else {
                return Err(TextError::Limit);
            };
            glyphs.push(record(placed, line.baseline, layout, image, line_index)?);
        }
    }
    Ok(TextMesh {
        material_id,
        glyphs,
    })
}

/// Read a face, lay the material out, cache each glyph, and build the mesh.
pub fn build_text(bytes: &[u8], material: &MaterialText) -> Result<TextMesh, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    let laid = layout(&font, material)?;
    let atlas = cache(&font, &laid)?;
    build(material.id.clone(), &laid, &atlas)
}

fn cache(font: &Font<'_>, layout: &Layout) -> Result<Atlas, TextError> {
    let mut atlas = Atlas::new();
    for placed in layout.glyphs() {
        let _image = atlas.glyph(font, placed.glyph_id)?;
    }
    Ok(atlas)
}

fn record(
    placed: &PlacedGlyph,
    baseline: f32,
    layout: &Layout,
    image: &GlyphImage,
    line: u32,
) -> Result<GlyphRecord, TextError> {
    let (x, y, width, height) = if image.width == 0 || image.height == 0 {
        (placed.x, baseline, 0.0, 0.0)
    } else {
        if !(image.pixels_per_em.is_finite() && image.pixels_per_em > 0.0) {
            return Err(TextError::Limit);
        }
        let texel = layout.font_size / image.pixels_per_em;
        let width = pixel(image.width) * texel;
        let height = pixel(image.height) * texel;
        let x = placed.x + image.left * layout.scale;
        let y = baseline - image.top * layout.scale;
        (x, y, width, height)
    };
    if ![x, y, width, height].iter().all(|value| value.is_finite()) {
        return Err(TextError::Limit);
    }
    Ok(GlyphRecord {
        glyph_id: placed.glyph_id,
        missing: placed.missing,
        source: placed.source,
        line,
        x,
        y,
        width,
        height,
    })
}

fn pixel(value: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{build, build_text};
    use crate::TextError;
    use crate::atlas::Atlas;
    use crate::layout::layout;
    use cutlass_font::{FontError, read};
    use cutlass_model::MaterialText;

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const COLOR: &[u8] = include_bytes!("../../font/fixtures/color-strike.otf");

    fn material(content: &str) -> MaterialText {
        let mut text = MaterialText::new("text-material");
        text.content = content.to_owned();
        text.font_size = 64.0;
        text.line_max_width = 0.0;
        text
    }

    #[test]
    fn each_glyph_is_its_own_record_and_the_material_id_stays() {
        let mut text = material("AB");
        let original = text.clone();
        let mesh = build_text(TTF, &text).expect("mesh");
        assert_eq!(text, original);
        assert_eq!(mesh.material_id, original.id);
        assert_eq!(mesh.glyphs.len(), 2);
        assert_eq!(mesh.glyphs[0].glyph_id, 2);
        assert_eq!(mesh.glyphs[1].glyph_id, 3);
        assert!(mesh.glyphs[0].width > 0.0);
        assert!(mesh.glyphs[1].height > 0.0);
        let font = read(TTF).expect("ttf");
        let laid = layout(&font, &text).expect("layout");
        let mut atlas = Atlas::new();
        let image = atlas.glyph(&font, 2).expect("glyph");
        let outlined = super::pixel(image.width) * laid.font_size / image.pixels_per_em;
        assert_eq!(mesh.glyphs[0].width.to_bits(), outlined.to_bits());

        let strike = build_text(COLOR, &material("A")).expect("strike");
        assert_eq!(strike.glyphs[0].width.to_bits(), 8.0_f32.to_bits());
        assert_eq!(strike.glyphs[0].height.to_bits(), 8.0_f32.to_bits());
        assert_eq!(mesh.glyphs[0].line, 0);
        assert_eq!(mesh.glyphs[1].line, 0);

        text.text_color = "#FF0000".to_owned();
        text.alignment = 2;
        text.line_max_width = 4_000.0;
        let restyled = build_text(TTF, &text).expect("restyle");
        assert_eq!(restyled.material_id, mesh.material_id);
        assert_eq!(restyled.glyphs.len(), 2);
        assert_ne!(restyled.glyphs[0].x.to_bits(), mesh.glyphs[0].x.to_bits());
    }

    #[test]
    fn repeated_glyphs_share_an_atlas_entry_and_keep_two_records() {
        let text = material("AA");
        let font = read(TTF).expect("ttf");
        let laid = layout(&font, &text).expect("layout");
        let mut atlas = Atlas::new();
        for placed in laid.glyphs() {
            let _image = atlas.glyph(&font, placed.glyph_id).expect("glyph");
        }
        assert_eq!(atlas.paints(), 1);
        let mesh = build(text.id.clone(), &laid, &atlas).expect("mesh");
        assert_eq!(mesh.glyphs.len(), 2);
        assert_eq!(mesh.glyphs[0].glyph_id, mesh.glyphs[1].glyph_id);
        assert_eq!(mesh.material_id.as_str(), "text-material");
    }

    #[test]
    fn an_empty_string_and_a_missing_glyph_keep_the_letter_structure() {
        let empty = build_text(TTF, &material("")).expect("empty");
        assert!(empty.glyphs.is_empty());
        let missing = build_text(TTF, &material("Z")).expect("missing");
        assert_eq!(missing.glyphs.len(), 1);
        assert!(missing.glyphs[0].missing);
        assert_eq!(missing.glyphs[0].glyph_id, 0);
        assert_eq!(missing.glyphs[0].width.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn a_truncated_font_is_an_error() {
        assert!(matches!(
            build_text(&[], &material("A")),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            build_text(&[0, 1, 0, 0], &material("A")),
            Err(TextError::Font(FontError::Truncated))
        ));
    }
}
