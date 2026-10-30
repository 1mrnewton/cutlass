//! `PictureSource` for one text segment.
//!
//! A sample inside the segment reads the cached atlas through the mesh and
//! returns an RGBA picture. A sample outside the segment is a miss. The draft
//! is not written. Sampling does not paint the atlas again. A color strike
//! keeps its color. A gray or mono strike is coverage under the text color.

use cutlass_font::Font;
use cutlass_model::{MaterialText, Micros, Segment, TimeRange};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample, rgba_len};

use crate::TextError;
use crate::atlas::{Atlas, GlyphKind};
use crate::layout::layout;
use crate::mesh::{GlyphRecord, TextMesh, build};

/// A text segment that can be sampled at a timeline microsecond.
pub struct TextSource {
    range: TimeRange,
    material_id: cutlass_model::MaterialId,
    fill: [u8; 4],
    atlas: Atlas,
    mesh: TextMesh,
    origin_x: f32,
    origin_y: f32,
    width: u32,
    height: u32,
}

impl TextSource {
    /// Read one face and prepare the atlas and mesh.
    ///
    /// `face` is `None` for a single font file and `Some(index)` for a
    /// collection. A face without outlines is refused.
    pub fn open(
        bytes: &[u8],
        face: Option<u32>,
        material: &MaterialText,
        segment: &Segment,
    ) -> Result<Self, TextError> {
        if segment.target.duration.0 > 0
            && segment
                .target
                .start
                .0
                .checked_add(segment.target.duration.0)
                .is_none()
        {
            return Err(TextError::Measure);
        }
        let font = match face {
            None => cutlass_font::read(bytes).map_err(TextError::Font)?,
            Some(index) => cutlass_font::read_collection(bytes, index).map_err(TextError::Font)?,
        };
        Self::from_font(&font, material, segment)
    }

    fn from_font(
        font: &Font<'_>,
        material: &MaterialText,
        segment: &Segment,
    ) -> Result<Self, TextError> {
        let fill = fill_rgba(material)?;
        let laid = layout(font, material)?;
        let mut atlas = Atlas::new();
        for placed in laid.glyphs() {
            let _image = atlas.glyph(font, placed.glyph_id)?;
        }
        let mesh = build(material.id.clone(), &laid, &atlas)?;
        let (origin_x, origin_y, width, height) = picture_bounds(&mesh)?;
        Ok(Self {
            range: segment.target.clone(),
            material_id: material.id.clone(),
            fill,
            atlas,
            mesh,
            origin_x,
            origin_y,
            width,
            height,
        })
    }

    #[must_use]
    pub fn material_id(&self) -> &cutlass_model::MaterialId {
        &self.material_id
    }

    #[must_use]
    pub fn atlas_paints(&self) -> u32 {
        self.atlas.paints()
    }

    #[must_use]
    pub fn records(&self) -> &[GlyphRecord] {
        &self.mesh.glyphs
    }

    /// Replace the ink color. The material id and the atlas stay as they are.
    pub fn set_fill(&mut self, fill: [u8; 4]) {
        self.fill = fill;
    }

    fn raster(&self) -> Result<Picture, TextError> {
        let len = rgba_len(self.width, self.height).ok_or(TextError::Limit)?;
        let mut rgba = vec![0_u8; len];
        for record in &self.mesh.glyphs {
            self.splat(&mut rgba, record);
        }
        let bytes = CpuBytes::new(self.width, self.height, rgba).map_err(|_| TextError::Limit)?;
        Ok(Picture::Cpu(bytes))
    }

    fn splat(&self, rgba: &mut [u8], record: &GlyphRecord) {
        if record.width <= 0.0 || record.height <= 0.0 {
            return;
        }
        let Some(image) = self.atlas.image(record.glyph_id) else {
            return;
        };
        if image.width == 0 || image.height == 0 || !drawable(image) {
            return;
        }
        let left = record.x - self.origin_x;
        let top = record.y - self.origin_y;
        let Some((x0, x1)) = span(left, left + record.width, self.width) else {
            return;
        };
        let Some((y0, y1)) = span(top, top + record.height, self.height) else {
            return;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let u = (pixel(x) + 0.5 - left) / record.width;
                let v = (pixel(y) + 0.5 - top) / record.height;
                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                    continue;
                }
                let Some(ink) = ink_at(image, u, v, self.fill) else {
                    continue;
                };
                blend_at(rgba, self.width, x, y, ink);
            }
        }
    }
}

impl PictureSource for TextSource {
    fn sample(&self, time: Micros) -> Sample {
        if !covers(&self.range, time) {
            return Sample::Miss;
        }
        match self.raster() {
            Ok(picture) => Sample::Ready(picture),
            Err(_) => Sample::Miss,
        }
    }
}

fn covers(range: &TimeRange, time: Micros) -> bool {
    let duration = range.duration.0;
    if duration <= 0 {
        return false;
    }
    let Some(end) = range.start.0.checked_add(duration) else {
        return false;
    };
    time.0 >= range.start.0 && time.0 < end
}

fn picture_bounds(mesh: &TextMesh) -> Result<(f32, f32, u32, u32), TextError> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut any = false;
    for glyph in &mesh.glyphs {
        if glyph.width <= 0.0 || glyph.height <= 0.0 {
            continue;
        }
        if ![glyph.x, glyph.y, glyph.width, glyph.height]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(TextError::Limit);
        }
        any = true;
        min_x = min_x.min(glyph.x);
        min_y = min_y.min(glyph.y);
        max_x = max_x.max(glyph.x + glyph.width);
        max_y = max_y.max(glyph.y + glyph.height);
    }
    if !any {
        return Ok((0.0, 0.0, 1, 1));
    }
    let width = edge(max_x - min_x)?;
    let height = edge(max_y - min_y)?;
    if rgba_len(width, height).is_none() {
        return Err(TextError::Limit);
    }
    Ok((min_x, min_y, width, height))
}

fn edge(value: f32) -> Result<u32, TextError> {
    if !value.is_finite() || value <= 0.0 || value > 8_192.0 {
        return Err(TextError::Limit);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let edge = value.ceil() as u32;
    if edge == 0 { Ok(1) } else { Ok(edge) }
}

fn span(start: f32, end: f32, limit: u32) -> Option<(u32, u32)> {
    if !start.is_finite() || !end.is_finite() || end <= start || limit == 0 {
        return None;
    }
    let start = start.floor().clamp(0.0, f32::from(u16::MAX));
    let end = end.ceil().clamp(0.0, f32::from(u16::MAX));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let start = start as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let end = end as u32;
    let end = end.min(limit);
    let start = start.min(limit);
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

fn pixel(value: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

fn drawable(image: &crate::GlyphImage) -> bool {
    match image.kind {
        GlyphKind::Color => !image.color.is_empty(),
        GlyphKind::Distance | GlyphKind::Coverage => !image.sdf.is_empty(),
    }
}

fn ink_at(image: &crate::GlyphImage, u: f32, v: f32, fill: [u8; 4]) -> Option<[u8; 4]> {
    let (col, row) = image_texel(image, u, v)?;
    match image.kind {
        GlyphKind::Distance => {
            let sdf = image.sdf.get(texel_offset(image, col, row)?)?;
            let covered = coverage(*sdf);
            if covered == 0 {
                return None;
            }
            let alpha = u16::from(covered) * u16::from(fill[3]) / 255;
            Some([fill[0], fill[1], fill[2], u8::try_from(alpha).ok()?])
        }
        GlyphKind::Coverage => {
            let covered = *image.sdf.get(texel_offset(image, col, row)?)?;
            if covered == 0 {
                return None;
            }
            let alpha = u16::from(covered) * u16::from(fill[3]) / 255;
            Some([fill[0], fill[1], fill[2], u8::try_from(alpha).ok()?])
        }
        GlyphKind::Color => {
            let offset = texel_offset(image, col, row)?.checked_mul(4)?;
            let pixel = image.color.get(offset..offset + 4)?;
            let alpha = fill[3];
            if alpha == 0 || pixel[3] == 0 {
                return None;
            }
            Some([
                scale(pixel[0], alpha),
                scale(pixel[1], alpha),
                scale(pixel[2], alpha),
                scale(pixel[3], alpha),
            ])
        }
    }
}

fn scale(channel: u8, alpha: u8) -> u8 {
    u8::try_from(u16::from(channel) * u16::from(alpha) / 255).unwrap_or(0)
}

fn image_texel(image: &crate::GlyphImage, u: f32, v: f32) -> Option<(u32, u32)> {
    let col = (u * pixel(image.width)).floor();
    let row = (v * pixel(image.height)).floor();
    if col < 0.0 || row < 0.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let col = col as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let row = row as u32;
    if col >= image.width || row >= image.height {
        None
    } else {
        Some((col, row))
    }
}

fn texel_offset(image: &crate::GlyphImage, col: u32, row: u32) -> Option<usize> {
    let offset = row.checked_mul(image.width)?.checked_add(col)?;
    usize::try_from(offset).ok()
}

fn coverage(sdf: u8) -> u8 {
    let distance = i32::from(sdf);
    const EDGE: i32 = 128;
    const HALF: i32 = 16;
    if distance <= EDGE - HALF {
        0
    } else if distance >= EDGE + HALF {
        255
    } else {
        let mixed = (distance - (EDGE - HALF)) * 255 / (HALF * 2);
        u8::try_from(mixed).unwrap_or(0)
    }
}

fn blend_at(rgba: &mut [u8], width: u32, x: u32, y: u32, source: [u8; 4]) {
    let Some(offset) = y.checked_mul(width).and_then(|row| row.checked_add(x)) else {
        return;
    };
    let Some(offset) = usize::try_from(offset)
        .ok()
        .and_then(|index| index.checked_mul(4))
    else {
        return;
    };
    let Some(pixel) = rgba.get_mut(offset..offset + 4) else {
        return;
    };
    let inverse = u16::from(255_u8.saturating_sub(source[3]));
    for channel in 0..4 {
        let mixed = u16::from(source[channel]) + u16::from(pixel[channel]) * inverse / 255;
        pixel[channel] = u8::try_from(mixed.min(255)).unwrap_or(255);
    }
}

fn fill_rgba(material: &MaterialText) -> Result<[u8; 4], TextError> {
    let text_alpha = unit_u8(material.text_alpha)?;
    let global_alpha = unit_u8(material.global_alpha)?;
    let (rgb, extra) = parse_hex(material.text_color.trim())?;
    let mut alpha = u16::from(text_alpha) * u16::from(global_alpha) / 255;
    if let Some(channel) = extra {
        alpha = u16::from(channel) * alpha / 255;
    }
    let alpha = u8::try_from(alpha).unwrap_or(0);
    Ok([rgb[0], rgb[1], rgb[2], alpha])
}

fn unit_u8(value: f64) -> Result<u8, TextError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(TextError::Measure);
    }
    let byte = (value * 255.0).round();
    if !(0.0..=255.0).contains(&byte) {
        return Err(TextError::Measure);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(byte as u8)
}

fn parse_hex(text: &str) -> Result<([u8; 3], Option<u8>), TextError> {
    if text.is_empty() {
        return Ok(([255, 255, 255], None));
    }
    let text = text.strip_prefix('#').unwrap_or(text);
    match text.len() {
        6 => Ok((
            [hex(&text[0..2])?, hex(&text[2..4])?, hex(&text[4..6])?],
            None,
        )),
        8 => Ok((
            [hex(&text[0..2])?, hex(&text[2..4])?, hex(&text[4..6])?],
            Some(hex(&text[6..8])?),
        )),
        _ => Err(TextError::Color),
    }
}

fn hex(text: &str) -> Result<u8, TextError> {
    u8::from_str_radix(text, 16).map_err(|_| TextError::Color)
}

#[cfg(test)]
mod tests {
    use super::TextSource;
    use crate::TextError;
    use cutlass_font::FontError;
    use cutlass_model::{Draft, MaterialText, Micros, Segment, TimeRange};
    use cutlass_picture::{Picture, PictureSource, Sample};

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const TTC: &[u8] = include_bytes!("../../font/fixtures/sample.ttc");
    const NO_OUTLINE: &[u8] = include_bytes!("../../font/fixtures/no-outline.otf");
    const COLOR: &[u8] = include_bytes!("../../font/fixtures/color-strike.otf");
    const GRAY: &[u8] = include_bytes!("../../font/fixtures/gray-strike.otf");

    fn material(content: &str) -> MaterialText {
        let mut text = MaterialText::new("text-material");
        text.content = content.to_owned();
        text.font_size = 64.0;
        text.text_color = "#FFFFFF".to_owned();
        text.background_color = "#00FF00".to_owned();
        text
    }

    fn segment() -> Segment {
        Segment::new("segment", "text-material", TimeRange::new(1_000, 5_000))
    }

    fn cpu(sample: Sample) -> Vec<u8> {
        match sample {
            Sample::Ready(Picture::Cpu(bytes)) => bytes.rgba().to_vec(),
            Sample::Ready(Picture::Platform(_)) => panic!("platform buffer"),
            Sample::Miss => panic!("miss"),
        }
    }

    fn opaque_index(rgba: &[u8]) -> Option<usize> {
        rgba.chunks(4)
            .position(|pixel| pixel.len() == 4 && pixel[3] == 255)
    }

    #[test]
    fn a_sample_is_ready_inside_the_segment_and_a_miss_outside_it() {
        let text = material("A");
        let clip = segment();
        let mut draft = Draft::empty("draft");
        draft.materials.texts.push(text.clone());
        draft.tracks[0].segments.push(clip.clone());
        let before = draft.clone();
        let source = TextSource::open(TTF, None, &text, &clip).expect("open");
        let paints = source.atlas_paints();
        assert!(paints >= 1);
        assert_eq!(source.records().len(), 1);
        assert_eq!(source.material_id().as_str(), "text-material");

        let ready = cpu(source.sample(Micros(1_000)));
        let index = opaque_index(&ready).expect("ink");
        assert_eq!(&ready[index * 4..index * 4 + 4], &[255, 255, 255, 255]);
        assert!(ready.chunks(4).any(|pixel| pixel[3] == 0));
        assert!(matches!(source.sample(Micros(5_999)), Sample::Ready(_)));
        assert!(matches!(source.sample(Micros(6_000)), Sample::Miss));
        assert!(matches!(source.sample(Micros(999)), Sample::Miss));
        assert_eq!(source.atlas_paints(), paints);
        assert_eq!(draft, before);
        assert_eq!(text.content, "A");
    }

    #[test]
    fn a_fill_change_samples_the_cached_atlas() {
        let text = material("A");
        let clip = segment();
        let mut source = TextSource::open(TTF, None, &text, &clip).expect("open");
        let paints = source.atlas_paints();
        let id = source.material_id().clone();
        let index = opaque_index(&cpu(source.sample(Micros(2_000)))).expect("ink");
        source.set_fill([255, 0, 0, 255]);
        let again = cpu(source.sample(Micros(2_000)));
        assert_eq!(&again[index * 4..index * 4 + 4], &[255, 0, 0, 255]);
        assert_eq!(source.material_id(), &id);
        assert_eq!(source.records().len(), 1);
        assert_eq!(source.atlas_paints(), paints);
    }

    #[test]
    fn an_empty_string_is_a_clear_picture_and_a_zero_duration_is_a_miss() {
        let text = material("");
        let clip = segment();
        let source = TextSource::open(TTF, None, &text, &clip).expect("empty");
        assert!(source.records().is_empty());
        let ready = cpu(source.sample(Micros(1_000)));
        assert!(ready.iter().all(|byte| *byte == 0));
        assert_eq!(source.atlas_paints(), 0);

        let mut dot = segment();
        dot.target = TimeRange::new(10, 0);
        let source = TextSource::open(TTF, None, &material("A"), &dot).expect("empty span");
        assert!(matches!(source.sample(Micros(10)), Sample::Miss));
    }

    #[test]
    fn a_bitmap_strike_keeps_its_pixels_and_is_not_painted_again() {
        let clip = segment();
        let mut color = material("A");
        color.text_color = "#00FF00".to_owned();
        let mut source = TextSource::open(COLOR, None, &color, &clip).expect("color");
        let paints = source.atlas_paints();
        assert_eq!(paints, 1);
        let ready = cpu(source.sample(Micros(1_000)));
        assert!(ready.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        assert!(ready.chunks(4).any(|pixel| pixel == [0, 128, 0, 128]));
        assert!(ready.chunks(4).any(|pixel| pixel == [0, 0, 255, 255]));
        source.set_fill([0, 0, 255, 255]);
        let again = cpu(source.sample(Micros(2_000)));
        assert!(again.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        assert_eq!(source.atlas_paints(), paints);

        let mut gray = material("A");
        gray.text_color = "#FF0000".to_owned();
        let gray_source = TextSource::open(GRAY, None, &gray, &clip).expect("gray");
        assert_eq!(gray_source.atlas_paints(), 1);
        let covered = cpu(gray_source.sample(Micros(1_000)));
        assert!(covered.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        assert!(covered.chunks(4).any(|pixel| pixel == [255, 0, 0, 128]));
        assert!(covered.chunks(4).any(|pixel| pixel == [255, 0, 0, 64]));
        assert!(covered.chunks(4).any(|pixel| pixel[3] == 0));
        assert_eq!(gray_source.atlas_paints(), 1);
    }

    #[test]
    fn a_hostile_font_or_color_is_an_error() {
        let text = material("A");
        let clip = segment();
        assert!(matches!(
            TextSource::open(&[], None, &text, &clip),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            TextSource::open(&[0, 1, 0, 0], None, &text, &clip),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            TextSource::open(TTC, None, &text, &clip),
            Err(TextError::Font(FontError::Collection))
        ));
        assert!(matches!(
            TextSource::open(TTC, Some(9), &text, &clip),
            Err(TextError::Font(FontError::CollectionIndex { .. }))
        ));
        let face = TextSource::open(TTC, Some(1), &text, &clip).expect("face");
        assert!(matches!(face.sample(Micros(1_000)), Sample::Ready(_)));
        assert!(matches!(
            TextSource::open(NO_OUTLINE, None, &text, &clip),
            Err(TextError::MissingStrike)
        ));
        let mut bad = material("A");
        bad.text_color = "#gg0000".to_owned();
        assert!(matches!(
            TextSource::open(TTF, None, &bad, &clip),
            Err(TextError::Color)
        ));
        let mut huge = material("A");
        huge.font_size = 100_000.0;
        assert!(matches!(
            TextSource::open(TTF, None, &huge, &clip),
            Err(TextError::Limit)
        ));
    }
}
