//! Text segments for one evaluated frame.
//!
//! The font bytes come from [`crate::Fonts`]. This module does not open
//! `font_path`. A missing face, a refused run, a hidden track, or a segment
//! that does not cover the time contributes nothing. The text picture is
//! placed with the segment clip, including its line keyframes.
//!
//! Word columns replace `content`. Those glyphs are the caption sample.
//! The caption's style pages the words and sets each word's color, box,
//! size and lift; rotation is not applied.

use cutlass_captions::{CaptionQuery, KeywordRange, RangeStyles, Rgba, sample_caption};
use cutlass_engine::DisplayQuarter;
use std::collections::HashMap;

use cutlass_gpu::{Affine, Placement};
use cutlass_model::{
    BloomParams, Clip, Draft, LetterPose, LetterStep, MaterialEffect, MaterialRef, MaterialText,
    Micros, Segment, SegmentId, TrackAttribute, UniformScale, WordColumns, segment_bloom,
};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample};
use cutlass_text::{GlyphLook, TextSource, text_fill};

use crate::Fonts;
use crate::gpu::{GpuLayer, GpuSource};
use crate::media::{gpu_place, picture_transform, place_with};

pub(crate) struct TextLayer {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: PlacedText,
}

pub(crate) struct PlacedText {
    /// The text drawn tight around its glyphs, before placement.
    raster: CpuBytes,
    width: u32,
    height: u32,
    transform: crate::media::PictureTransform,
}

impl PictureSource for PlacedText {
    fn sample(&self, _time: Micros) -> Sample {
        place_with(
            self.width,
            self.height,
            self.raster.clone(),
            self.transform,
            DisplayQuarter::Zero,
        )
    }
}

impl PlacedText {
    /// The raster as a GPU layer, placed as [`PictureSource::sample`] places
    /// it, so the canvas-sized picture is never built.
    pub(crate) fn gpu(&self) -> Option<Vec<GpuLayer>> {
        let size = (self.raster.width(), self.raster.height());
        let placement = Placement {
            to_picture: gpu_place(self.transform, size, (self.width, self.height))?,
            picture: size,
            to_oriented: Affine::IDENTITY,
            oriented: size,
            to_texture: Affine::IDENTITY,
            opacity: f64::from(self.transform.opacity()) / 255.0,
        };
        Some(vec![GpuLayer {
            source: GpuSource::Rgba(self.raster.clone()),
            placement,
            backdrop: None,
        }])
    }
}

/// The most picture pixels a title draws to each pixel it is shown at.
const MAX_DENSITY: f64 = 8.0;
/// The largest face, in pixels, a title is drawn with to keep it sharp.
const MAX_FACE_PX: f64 = 1_024.0;

/// Text segments covering `time`, placed. A title is drawn at its scale, so
/// one scaled up keeps sharp edges; its raster is kept in `rasters` for its
/// segment with that density, so a later frame of the same revision and
/// scale does not lay it out and draw it again. Captions change with time
/// and are drawn each frame.
pub(crate) fn text_layers(
    draft: &Draft,
    fonts: &dyn Fonts,
    width: u32,
    height: u32,
    time: Micros,
    rasters: &mut HashMap<SegmentId, (f64, CpuBytes)>,
) -> Vec<TextLayer> {
    let mut layers = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible {
                continue;
            }
            if !covers(segment.target.start.0, segment.target.duration.0, time.0) {
                continue;
            }
            let Some(MaterialRef::Text(stored)) = draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            // Keyed letters take their color and style at `time`, and are
            // drawn for each frame and not kept.
            let graphs = &draft.keyframe_graph_list;
            let colored =
                cutlass_model::text_color_at(segment, graphs, time).map(|rgb| recolor(stored, rgb));
            let styled = cutlass_model::text_style_at(
                colored.as_ref().unwrap_or(stored),
                segment,
                graphs,
                time,
            );
            let keyed = (colored.is_some() || styled.is_some()).then_some(());
            let material = styled.as_ref().or(colored.as_ref()).unwrap_or(stored);
            // Text with no face draws nothing; the log says so once.
            if material.font_path.is_empty() {
                logged::<(), _>("text", segment, Err("it has no font file"));
                continue;
            }
            let Some(face) = fonts.face(&material.font_path) else {
                logged::<(), _>("text", segment, Err("its font file is not here"));
                continue;
            };
            let Some((clip, uniform)) = cutlass_model::animated_at(draft, segment, time) else {
                continue;
            };
            let Some(transform) = picture_transform(&clip, &uniform, width, height)
                .map(crate::media::PictureTransform::native)
            else {
                continue;
            };
            if transform.omitted() {
                continue;
            }
            let (raster, transform) = if words_present(material) {
                let wanted = density(&clip, &uniform, material.font_size);
                let Some((drawn, bytes)) = caption_picture(
                    Faces { fonts, face },
                    material,
                    glow(draft, segment).as_ref(),
                    segment,
                    time,
                    wanted,
                    f64::from(width),
                ) else {
                    continue;
                };
                (bytes, transform.dense(drawn))
            } else {
                let wanted = density(&clip, &uniform, material.font_size);
                let glow = glow(draft, segment);
                // Letters on the move are drawn for each frame and not kept.
                let steps = cutlass_model::letters_at(draft, segment, time);
                let kept = rasters
                    .get(&segment.id)
                    .filter(|(drawn, _)| *drawn == wanted && steps.is_empty() && keyed.is_none())
                    .cloned();
                let Some((drawn, bytes)) = kept.or_else(|| {
                    title_picture(
                        Faces { fonts, face },
                        material,
                        glow.as_ref(),
                        segment,
                        time,
                        wanted,
                        &steps,
                    )
                }) else {
                    continue;
                };
                if steps.is_empty() && keyed.is_none() {
                    rasters.insert(segment.id.clone(), (drawn, bytes.clone()));
                }
                (bytes, transform.dense(drawn))
            };
            layers.push(TextLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                placed: PlacedText {
                    raster,
                    width,
                    height,
                    transform,
                },
            });
        }
    }
    layers
}

/// `material` with its letters, and every rich range's, in `rgb`.
fn recolor(material: &MaterialText, rgb: [f64; 3]) -> MaterialText {
    let mut text = material.clone();
    let byte = |unit: f64| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let byte = (unit.clamp(0.0, 1.0) * 255.0).round() as u8;
        byte
    };
    text.text_color = format!(
        "#{:02X}{:02X}{:02X}",
        byte(rgb[0]),
        byte(rgb[1]),
        byte(rgb[2])
    );
    text.paint_rich(rgb);
    text
}

/// A title drawn with `density` pixels to each shown pixel, and that
/// density; at 1 when it cannot be drawn larger. Its letters are posed by
/// the letter animations at `time`, `steps`.
fn title_picture(
    faces: Faces<'_>,
    material: &MaterialText,
    glow: Option<&BloomParams>,
    segment: &Segment,
    time: Micros,
    density: f64,
    steps: &[LetterStep],
) -> Option<(f64, CpuBytes)> {
    let draw = |density: f64| {
        let mut larger = material.clone();
        larger.font_size *= density;
        larger.line_max_width *= density;
        let mut source = logged(
            "title",
            segment,
            TextSource::open_with_faces(faces.face, None, &larger, glow, segment, &|path| {
                faces.named(path)
            }),
        )?;
        if !steps.is_empty() {
            let poses = source
                .letters()
                .iter()
                .map(|letter| {
                    steps
                        .iter()
                        .fold(LetterPose::REST, |pose, step| pose.then(step.pose(*letter)))
                })
                .collect();
            source.set_poses(poses);
        }
        match source.sample(time) {
            Sample::Ready(Picture::Cpu(bytes)) => Some((density, bytes)),
            _ => None,
        }
    };
    if density > 1.0
        && let Some(drawn) = draw(density)
    {
        return Some(drawn);
    }
    draw(1.0)
}

/// Picture pixels per shown pixel for a title at this clip scale: the
/// larger scale, up in quarter steps so a pinch does not redraw it at every
/// move, from 1 up to [`MAX_DENSITY`], with the face kept to [`MAX_FACE_PX`].
fn density(clip: &Clip, uniform: &UniformScale, font_size: f64) -> f64 {
    let scale = if uniform.on {
        uniform.value.abs()
    } else {
        clip.scale.x.abs().max(clip.scale.y.abs())
    };
    if !scale.is_finite() || scale <= 1.0 {
        return 1.0;
    }
    let mut most = MAX_DENSITY;
    if font_size.is_finite() && font_size > 0.0 {
        most = most.min((MAX_FACE_PX / font_size).max(1.0));
    }
    ((scale * 4.0).ceil() / 4.0).min(most)
}

/// The size a title's text picture is drawn at, before placement: tight
/// around its glyphs, as [`text_layers`] draws it. Captions change with time
/// and have none; neither has a clip whose face is missing.
#[must_use]
pub fn text_size(draft: &Draft, fonts: &dyn Fonts, segment: &Segment) -> Option<(u32, u32)> {
    let Some(MaterialRef::Text(material)) = draft.materials.get(segment.material_id.as_str())
    else {
        return None;
    };
    if material.font_path.is_empty() || words_present(material) {
        return None;
    }
    title_size(draft, fonts, segment, material)
}

/// The size a title's or caption's text picture is drawn at `time`, before
/// placement, as [`text_layers`] draws it on a canvas of the draft's width.
/// A caption has the size of the page of words showing then, and none
/// between pages; a clip whose face is missing has none.
#[must_use]
pub fn text_size_at(
    draft: &Draft,
    fonts: &dyn Fonts,
    segment: &Segment,
    time: Micros,
) -> Option<(u32, u32)> {
    let Some(MaterialRef::Text(stored)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    // A keyed style sizes the letters as they are drawn at `time`.
    let styled = cutlass_model::text_style_at(stored, segment, &draft.keyframe_graph_list, time);
    let material = styled.as_ref().unwrap_or(stored);
    if material.font_path.is_empty() {
        return None;
    }
    if !words_present(material) {
        return title_size(draft, fonts, segment, material);
    }
    let face = fonts.face(&material.font_path)?;
    let canvas_width = f64::from(draft.canvas_config.width.max(1));
    let (_, bytes) = caption_picture(
        Faces { fonts, face },
        material,
        glow(draft, segment).as_ref(),
        segment,
        time,
        1.0,
        canvas_width,
    )?;
    let extent = bytes.extent();
    Some((extent.width, extent.height))
}

fn title_size(
    draft: &Draft,
    fonts: &dyn Fonts,
    segment: &Segment,
    material: &MaterialText,
) -> Option<(u32, u32)> {
    let face = fonts.face(&material.font_path)?;
    let glow = glow(draft, segment);
    let faces = Faces { fonts, face };
    let source = logged(
        "title",
        segment,
        TextSource::open_with_faces(face, None, material, glow.as_ref(), segment, &|path| {
            faces.named(path)
        }),
    )?;
    Some(source.size())
}

/// How many draw failures are logged before the rest are only counted, so
/// a title that fails on every frame does not fill the log.
const LOGGED_FAILURES: usize = 64;

/// `result`'s value, or nothing with the failure logged the first time a
/// clip hits it: a title or caption that cannot draw says why.
fn logged<T, E: std::fmt::Display>(
    what: &str,
    segment: &Segment,
    result: Result<T, E>,
) -> Option<T> {
    use std::sync::Mutex;
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            let line = format!("{what} {} did not draw: {error}", segment.id);
            if let Ok(mut seen) = SEEN.lock()
                && seen.len() < LOGGED_FAILURES
                && !seen.contains(&line)
            {
                cutlass_logger::warn!("{line}");
                seen.push(line);
            }
            None
        }
    }
}

/// A title's own face, and the shell's fonts for its ranges.
#[derive(Clone, Copy)]
struct Faces<'a> {
    fonts: &'a dyn Fonts,
    face: &'a [u8],
}

impl<'a> Faces<'a> {
    /// The face a range's font `path` names: by the whole path, else by its
    /// file name, else the first face of a collection by that name.
    fn named(&self, path: &str) -> Option<&'a [u8]> {
        let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
        self.fonts
            .face(path)
            .or_else(|| self.fonts.face(file))
            .or_else(|| self.fonts.face(&format!("{file}#0")))
    }
}

/// The glow a title's clip points at, when it reads as one.
fn glow(draft: &Draft, segment: &Segment) -> Option<BloomParams> {
    segment_bloom(&draft.materials, segment).and_then(MaterialEffect::bloom)
}

fn words_present(material: &MaterialText) -> bool {
    columns_present(&material.current_words) || columns_present(&material.words)
}

fn columns_present(columns: &WordColumns) -> bool {
    !columns.start_time.is_empty() || !columns.end_time.is_empty() || !columns.text.is_empty()
}

/// Word columns replace `content`. The caption's style, when it names one,
/// pages the words and gives the curves; otherwise every word is drawn
/// plain. The page is drawn as a title is, so the material's outline,
/// shadow, glow and box hold for it, at `density` pixels to each shown
/// pixel. An empty style keeps the material color; a sampled color that is
/// not opaque white replaces it. Position is a pixel shift, right and down;
/// scale sizes a word about its middle, and a background is a round box
/// behind the word. Rotation on the style is not applied.
/// The share of the canvas width a caption with no line width wraps at.
const CAPTION_WIDTH: f64 = 0.9;

fn caption_picture(
    faces: Faces<'_>,
    material: &MaterialText,
    glow: Option<&BloomParams>,
    segment: &Segment,
    time: Micros,
    density: f64,
    canvas_width: f64,
) -> Option<(f64, CpuBytes)> {
    let style = cutlass_captions::caption_style(&material.caption_style);
    let styles = style.map_or_else(RangeStyles::identity, |style| {
        style.styles(material.font_size)
    });
    let limits = cutlass_captions::caption_limits(&material.caption_style);
    let frame = sample_caption(&CaptionQuery {
        material,
        font: faces.face,
        timeline: time,
        placed_at: segment.target.start,
        limits,
        keywords: &caption_keywords(material),
        styles: &styles,
        stagger_shift_ms: material.caption_stagger_ms.max(0),
    });
    let frame = logged("caption", segment, frame)?;
    let looks = caption_looks(&frame, logged("caption", segment, text_fill(material))?)?;
    let hidden = GlyphLook {
        pose: LetterPose {
            alpha: 0.0,
            ..LetterPose::REST
        },
        ..GlyphLook::REST
    };
    let draw = |density: f64| {
        let mut page = material.clone();
        page.content.clone_from(&frame.letters.text);
        page.is_rich_text = false;
        // A caption with no line width wraps inside the canvas.
        if page.line_max_width <= 0.0 {
            page.line_max_width = canvas_width * CAPTION_WIDTH;
        }
        page.font_size *= density;
        page.line_max_width *= density;
        let mut source = logged(
            "caption",
            segment,
            TextSource::open_with_faces(faces.face, None, &page, glow, segment, &|path| {
                faces.named(path)
            }),
        )?;
        let each: Vec<GlyphLook> = source
            .records()
            .iter()
            .map(|record| looks.get(&record.source).copied().unwrap_or(hidden))
            .collect();
        source.set_looks(&each);
        match source.sample(time) {
            Sample::Ready(Picture::Cpu(bytes)) => Some((density, bytes)),
            _ => None,
        }
    };
    if density > 1.0
        && let Some(drawn) = draw(density)
    {
        return Some(drawn);
    }
    draw(1.0)
}

/// The caption's marked words, each whole. A mark past its words is left
/// out.
fn caption_keywords(material: &MaterialText) -> Vec<KeywordRange> {
    let columns = if columns_present(&material.current_words) {
        &material.current_words
    } else {
        &material.words
    };
    material
        .caption_keywords
        .iter()
        .filter_map(|word| {
            let text = columns.text.get(*word)?;
            Some(KeywordRange {
                word: *word,
                char_start: 0,
                char_end: text.chars().count(),
            })
        })
        .collect()
}

/// How each glyph of the caption's page looks, by its place in the page's
/// text. A glyph the style does not draw has none.
fn caption_looks(
    frame: &cutlass_captions::CaptionFrame,
    base: [u8; 4],
) -> Option<HashMap<usize, GlyphLook>> {
    let font_size = frame.letters.layout.font_size;
    if !font_size.is_finite() || font_size <= 0.0 {
        return None;
    }
    // Each word's middle, so a word sized up grows about it.
    let mut spans: HashMap<usize, (f32, f32)> = HashMap::new();
    let mut words: HashMap<usize, usize> = HashMap::new();
    for letter in &frame.letters.letters {
        if let Some(word) = letter.word {
            let span = spans.entry(word).or_insert((f32::MAX, f32::MIN));
            span.0 = span.0.min(letter.glyph.x);
            span.1 = span.1.max(letter.glyph.x);
            words.insert(letter.source, word);
        }
    }
    let mut looks = HashMap::new();
    for sample in frame.batches.iter().flat_map(|batch| &batch.glyphs) {
        let look = (|| {
            let shift_x = shift_px(sample.style.position_x)?;
            let shift_y = shift_px(sample.style.position_y)?;
            let fill = ink_color(base, sample.style.color)?;
            let back = back_color(sample.style.background)?;
            #[allow(clippy::cast_possible_truncation)]
            let scale = sample.style.scale_x as f32;
            if !scale.is_finite() || scale <= 0.0 {
                return None;
            }
            let middle = words
                .get(&sample.glyph.source)
                .and_then(|word| spans.get(word))
                .map_or(sample.glyph.x, |(left, right)| (left + right) / 2.0);
            let dx = shift_x + (sample.glyph.x - middle) * (scale - 1.0);
            Some(GlyphLook {
                pose: LetterPose {
                    alpha: 1.0,
                    dx: f64::from(dx / font_size),
                    dy: f64::from(shift_y / font_size),
                    scale: f64::from(scale),
                },
                ink: (fill != base).then_some(fill),
                back,
            })
        })();
        if let Some(look) = look {
            looks.insert(sample.glyph.source, look);
        }
    }
    Some(looks)
}

/// A box color from a sampled background: clear when it has no opacity.
fn back_color(color: Rgba) -> Option<[u8; 4]> {
    if color.alpha <= 0.0 {
        return Some([0, 0, 0, 0]);
    }
    Some([
        unit_byte(color.red)?,
        unit_byte(color.green)?,
        unit_byte(color.blue)?,
        unit_byte(color.alpha)?,
    ])
}

fn shift_px(value: f64) -> Option<f32> {
    if !value.is_finite() || value.abs() > 8_192.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    Some(value as f32)
}

fn ink_color(base: [u8; 4], color: Rgba) -> Option<[u8; 4]> {
    if is_one(color.red) && is_one(color.green) && is_one(color.blue) && is_one(color.alpha) {
        return Some(base);
    }
    Some([
        unit_byte(color.red)?,
        unit_byte(color.green)?,
        unit_byte(color.blue)?,
        unit_byte(color.alpha)?,
    ])
}

fn is_one(value: f64) -> bool {
    value.to_bits() == 1.0_f64.to_bits()
}

fn unit_byte(value: f64) -> Option<u8> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some((value * 255.0).round() as u8)
}

fn covers(start: i64, duration: i64, time: i64) -> bool {
    if duration <= 0 {
        return false;
    }
    let Some(end) = start.checked_add(duration) else {
        return false;
    };
    time >= start && time < end
}

#[cfg(test)]
mod tests {
    use cutlass_engine::ReaderFactory;
    use cutlass_model::{
        BloomParams, Clip, CommonKeyframe, CommonKeyframeGroup, Draft, MaterialDraft,
        MaterialEffect, MaterialId, MaterialText, Micros, Segment, TimeRange, TrackAttribute,
        WordColumns,
    };
    use cutlass_renderer_software::SoftwareDevice;

    use super::Fonts;
    use crate::frame::{Snapshot, frame};
    use crate::media::OpenClips;

    const FONT: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    struct OneFont;

    impl Fonts for OneFont {
        fn face(&self, path: &str) -> Option<&[u8]> {
            if path == "sample.ttf" {
                Some(FONT)
            } else {
                None
            }
        }
    }

    fn pixels(draft: &Draft, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        frame(
            &Snapshot::with_fonts(draft, &OneFont),
            Micros(time),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba
    }

    fn text_segment(content: &str, rich: bool) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 256;
        draft.canvas_config.height = 256;
        let mut text = MaterialText::new("words");
        text.content = content.to_owned();
        text.font_size = 64.0;
        text.font_path = "sample.ttf".to_owned();
        text.text_color = "#FFFFFF".to_owned();
        text.is_rich_text = rich;
        draft.materials.texts.push(text);
        let mut segment = Segment::new("line", "words", TimeRange::new(0, 1_000_000));
        segment.render_index = 1;
        draft.tracks[0].segments.push(segment);
        draft
    }

    fn ink(rgba: &[u8]) -> bool {
        rgba.chunks(4).any(|pixel| pixel == [255, 255, 255, 255])
    }

    #[test]
    fn a_text_segment_is_drawn_inside_its_span() {
        let draft = text_segment("A", false);
        let before = draft.clone();
        let shown = pixels(&draft, 0);
        assert!(ink(&shown));
        assert!(shown.chunks(4).any(|pixel| pixel == [0, 0, 0, 0]));
        assert!(!ink(&pixels(&draft, 1_000_000)));
        assert_eq!(draft, before);
    }

    /// Rows and columns that hold any drawn pixel.
    fn inked_span(rgba: &[u8], width: usize) -> (usize, usize) {
        let lit: Vec<usize> = rgba
            .chunks(4)
            .enumerate()
            .filter(|(_, pixel)| pixel[3] > 0)
            .map(|(index, _)| index)
            .collect();
        let columns = lit.iter().map(|index| index % width);
        let rows = lit.iter().map(|index| index / width);
        let span = |values: Vec<usize>| {
            values.iter().max().unwrap_or(&0) - values.iter().min().unwrap_or(&0) + 1
        };
        (span(columns.collect()), span(rows.collect()))
    }

    #[test]
    fn a_title_keeps_its_own_size_instead_of_fitting_the_canvas() {
        let small = text_segment("A", false);
        let mut large = small.clone();
        large.canvas_config.width = 512;
        large.canvas_config.height = 512;
        let (width, height) = inked_span(&pixels(&small, 0), 256);
        assert!(width < 128 && height < 128, "{width}x{height}");
        assert_eq!(inked_span(&pixels(&large, 0), 512), (width, height));
    }

    #[test]
    fn a_typewriter_title_shows_its_letters_one_by_one() {
        let mut draft = text_segment("AAAA", false);
        draft
            .materials
            .material_animations
            .push(cutlass_model::MaterialAnimations {
                id: MaterialId::new("moves"),
                kind: "sticker_animation".to_owned(),
                animations: vec![cutlass_model::StickerAnimation {
                    kind: "in".to_owned(),
                    resource_id: "cutlass-anim-in-typewriter".to_owned(),
                    start: Micros(0),
                    duration: Some(Micros(800_000)),
                    ..cutlass_model::StickerAnimation::default()
                }],
                ..cutlass_model::MaterialAnimations::default()
            });
        draft.tracks[0].segments[0]
            .extra_material_refs
            .push(MaterialId::new("moves"));
        assert!(!ink(&pixels(&draft, 0)));
        let (one, _) = inked_span(&pixels(&draft, 250_000), 256);
        let (three, _) = inked_span(&pixels(&draft, 650_000), 256);
        let (all, _) = inked_span(&pixels(&draft, 900_000), 256);
        assert!(one < three && three < all, "{one} {three} {all}");
        // At rest it is the title without an animation.
        let still = text_segment("AAAA", false);
        assert_eq!(pixels(&draft, 900_000), pixels(&still, 900_000));
    }

    /// Pixels drawn part covered, at the letters' edges.
    fn soft_edge(rgba: &[u8]) -> usize {
        rgba.chunks(4)
            .filter(|pixel| pixel[3] > 0 && pixel[3] < 255)
            .count()
    }

    #[test]
    fn a_title_scaled_up_is_drawn_at_its_shown_size() {
        let small = text_segment("A", false);
        let mut large = small.clone();
        large.tracks[0].segments[0].uniform_scale.value = 2.0;
        let (small_w, small_h) = inked_span(&pixels(&small, 0), 256);
        let (large_w, large_h) = inked_span(&pixels(&large, 0), 256);
        assert!(large_w.abs_diff(small_w * 2) <= 4, "{large_w} {small_w}");
        assert!(large_h.abs_diff(small_h * 2) <= 4, "{large_h} {small_h}");
        // Stretched, the soft edge would widen with the letters, to about
        // four times; drawn large, it only grows with their outline.
        let (before, after) = (soft_edge(&pixels(&small, 0)), soft_edge(&pixels(&large, 0)));
        assert!(after < before * 3, "{before} {after}");
    }

    #[test]
    fn a_title_reports_the_size_it_is_drawn_at() {
        let draft = text_segment("A", false);
        let segment = &draft.tracks[0].segments[0];
        let (width, height) = super::text_size(&draft, &OneFont, segment).unwrap();
        // The picture holds the ink, with the glyph cells' margin around it.
        let (inked_w, inked_h) = inked_span(&pixels(&draft, 0), 256);
        assert!(
            (inked_w..inked_w + 32).contains(&(width as usize)),
            "{width} {inked_w}"
        );
        assert!(
            (inked_h..inked_h + 32).contains(&(height as usize)),
            "{height} {inked_h}"
        );
        let mut missing = draft.clone();
        missing.materials.texts[0].font_path = "other.ttf".to_owned();
        assert_eq!(super::text_size(&missing, &OneFont, segment), None);
    }

    #[test]
    fn a_rich_range_is_set_in_the_face_its_path_names() {
        struct TwoFonts;
        impl Fonts for TwoFonts {
            fn face(&self, path: &str) -> Option<&[u8]> {
                match path {
                    "sample.ttf" => Some(FONT),
                    "Other.otf" => Some(include_bytes!("../../font/fixtures/sample.otf")),
                    _ => None,
                }
            }
        }
        let content = |path: &str| {
            format!(r#"{{"text":"AB","styles":[{{"range":[1,2],"font":{{"path":"{path}"}}}}]}}"#)
        };
        let own = text_segment(&content("/nowhere/Missing.otf"), true);
        let other = text_segment(&content("/Library/Fonts/Other.otf"), true);
        let segment = &own.tracks[0].segments[0];
        let size = |draft: &Draft| super::text_size(draft, &TwoFonts, segment).unwrap();
        // A face the table lacks keeps the title's; one it has by file name
        // sets the range in it.
        assert_eq!(size(&own), size(&text_segment("AB", false)));
        assert_ne!(size(&other), size(&own));
    }

    #[test]
    fn a_title_glows_with_the_bloom_its_clip_points_at() {
        let plain = text_segment("A", false);
        let mut glowing = plain.clone();
        let bloom = BloomParams {
            color: "#00FF00".to_owned(),
            strength: 1.0,
            range: 0.5,
            ..BloomParams::default()
        };
        glowing
            .materials
            .effects
            .push(MaterialEffect::glow("glow", &bloom));
        glowing.tracks[0].segments[0]
            .extra_material_refs
            .push(MaterialId::new("glow"));
        let green = |rgba: &[u8]| {
            rgba.chunks(4)
                .filter(|pixel| pixel[0] == 0 && pixel[1] > 0 && pixel[2] == 0)
                .count()
        };
        assert_eq!(green(&pixels(&plain, 0)), 0);
        assert!(green(&pixels(&glowing, 0)) > 50);
        let segment = &glowing.tracks[0].segments[0];
        let (plain_w, _) = super::text_size(&plain, &OneFont, segment).unwrap();
        let (glow_w, _) = super::text_size(&glowing, &OneFont, segment).unwrap();
        assert!(glow_w > plain_w, "{plain_w} {glow_w}");
    }

    #[test]
    fn a_missing_font_a_hidden_track_and_unreadable_rich_text_draw_nothing() {
        let mut missing = text_segment("A", false);
        missing.materials.texts[0].font_path = "other.ttf".to_owned();
        assert!(!ink(&pixels(&missing, 0)));

        let mut hidden = text_segment("A", false);
        hidden.tracks[0].attribute = TrackAttribute(TrackAttribute::HIDE);
        assert!(!ink(&pixels(&hidden, 0)));

        let mut invisible = text_segment("A", false);
        invisible.tracks[0].segments[0].visible = false;
        assert!(!ink(&pixels(&invisible, 0)));

        assert!(!ink(&pixels(&text_segment("A", true), 0)));
        assert!(ink(&pixels(
            &text_segment(r#"{"text":"A","styles":[]}"#, true),
            0
        )));
    }

    #[test]
    fn a_text_layer_paints_above_a_lower_index() {
        let mut draft = text_segment("A", false);
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "solid".into(),
                is_color_clip: true,
                solid_color: "0xFFFF0000".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0]
            .segments
            .push(Segment::new("solid", "solid", TimeRange::new(0, 1_000_000)));
        let shown = pixels(&draft, 0);
        assert!(ink(&shown));
        assert!(shown.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));

        draft.tracks[0].segments[0].render_index = 0;
        draft.tracks[0].segments[1].render_index = 2;
        let covered = pixels(&draft, 0);
        assert!(!ink(&covered));
        assert!(covered.chunks(4).all(|pixel| pixel == [255, 0, 0, 255]));
    }

    #[test]
    fn an_alpha_key_hides_the_text_until_the_line_reaches_it() {
        let mut draft = text_segment("A", false);
        draft.tracks[0].segments[0].common_keyframes = vec![CommonKeyframeGroup {
            property_type: "KFTypeAlpha".to_owned(),
            keyframe_list: vec![
                CommonKeyframe {
                    time_offset: Micros(0),
                    values: vec![0.0],
                    curve_type: "Line".to_owned(),
                    ..CommonKeyframe::default()
                },
                CommonKeyframe {
                    time_offset: Micros(1_000),
                    values: vec![1.0],
                    curve_type: "Line".to_owned(),
                    ..CommonKeyframe::default()
                },
            ],
            ..CommonKeyframeGroup::default()
        }];
        assert!(!ink(&pixels(&draft, 0)));
        assert!(ink(&pixels(&draft, 1_000)));
    }

    #[test]
    fn a_nested_text_segment_uses_the_same_font_bytes() {
        let child = text_segment("A", false);
        let mut parent = Draft::empty("parent");
        parent.canvas_config.width = 256;
        parent.canvas_config.height = 256;
        parent.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let mut segment = Segment::new("compound", "nest", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        parent.tracks[0].segments.push(segment);
        assert!(ink(&pixels(&parent, 0)));

        parent.tracks[0].segments[0].clip = Clip {
            alpha: 0.0,
            ..Clip::default()
        };
        assert!(!ink(&pixels(&parent, 0)));
    }

    fn words(rows: &[(&str, i64, i64)]) -> WordColumns {
        let mut columns = WordColumns::default();
        for (text, start, end) in rows {
            columns.text.push((*text).to_owned());
            columns.start_time.push(*start);
            columns.end_time.push(*end);
        }
        columns
    }

    #[test]
    fn word_columns_replace_the_text_content() {
        let mut draft = text_segment("A", false);
        draft.materials.texts[0].words = words(&[("\n", 0, 100)]);
        let before = draft.clone();
        assert!(!ink(&pixels(&draft, 0)));
        assert_eq!(draft, before);
    }

    #[test]
    fn a_caption_reports_the_size_of_the_page_showing_at_a_time() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].words = words(&[("A", 0, 400), ("AB", 500, 1_000)]);
        let segment = draft.tracks[0].segments[0].clone();
        let at = |time: i64| super::text_size_at(&draft, &OneFont, &segment, Micros(time));
        let (width, height) = at(0).expect("the first page has a size");
        // The box is the picture the preview places: the whole page, with
        // the words still to come drawn clear.
        let drawn = |time: i64| {
            let layers = super::text_layers(
                &draft,
                &OneFont,
                256,
                256,
                Micros(time),
                &mut std::collections::HashMap::new(),
            );
            let extent = layers[0].placed.raster.extent();
            (extent.width, extent.height)
        };
        assert_eq!((width, height), drawn(0));
        assert_eq!(at(700_000), Some(drawn(700_000)));
        // Past its words a caption shows nothing and has no size.
        assert_eq!(at(1_000_000), None);
        // The timeless size still has none for a caption.
        assert_eq!(super::text_size(&draft, &OneFont, &segment), None);
        // A title measures the same with or without a time.
        let title = text_segment("A", false);
        let line = &title.tracks[0].segments[0];
        assert_eq!(
            super::text_size_at(&title, &OneFont, line, Micros(0)),
            super::text_size(&title, &OneFont, line)
        );
        let mut faceless = draft.clone();
        faceless.materials.texts[0].font_path.clear();
        assert_eq!(
            super::text_size_at(&faceless, &OneFont, &segment, Micros(0)),
            None
        );
    }

    #[test]
    fn keyed_letters_change_color_between_their_keys() {
        let mut draft = text_segment("A", false);
        let segment = &mut draft.tracks[0].segments[0];
        for (offset, rgb) in [(0, [1.0, 0.0, 0.0]), (1_000_000, [0.0, 0.0, 1.0])] {
            for (name, value) in cutlass_model::TEXT_COLOR_KEYS.iter().zip(rgb) {
                cutlass_model::put_key(segment, name, Micros(offset), value);
            }
        }
        let has =
            |time: i64, rgba: [u8; 4]| pixels(&draft, time).chunks(4).any(|pixel| pixel == rgba);
        assert!(has(0, [255, 0, 0, 255]));
        assert!(!has(0, [255, 255, 255, 255]));
        assert!(has(500_000, [128, 0, 128, 255]));
        assert!(has(999_999, [0, 0, 255, 255]));
    }

    #[test]
    fn keyed_text_size_grows_the_letters_between_its_keys() {
        let mut draft = text_segment("A", false);
        let stored = draft.materials.texts[0].font_size;
        let segment = &mut draft.tracks[0].segments[0];
        let size = cutlass_model::TextStyleKey::Size.names()[0];
        cutlass_model::put_key(segment, size, Micros(0), stored / 2.0);
        cutlass_model::put_key(segment, size, Micros(1_000_000), stored);
        let lit = |time: i64| {
            pixels(&draft, time)
                .chunks(4)
                .filter(|pixel| pixel[3] > 0)
                .count()
        };
        let (small, middle, large) = (lit(0), lit(500_000), lit(999_999));
        assert!(small < middle && middle < large, "{small} {middle} {large}");
        // The drawn picture follows the key each frame, not a kept one.
        assert_eq!(lit(0), small);
        // The box the stage shows follows the keyed size too.
        let segment = &draft.tracks[0].segments[0];
        let box_at = |time: i64| super::text_size_at(&draft, &OneFont, segment, Micros(time));
        let (small, large) = (box_at(0).unwrap(), box_at(999_999).unwrap());
        assert!(
            small.0 < large.0 && small.1 < large.1,
            "{small:?} {large:?}"
        );
    }

    #[test]
    fn a_caption_word_uses_the_material_color() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].text_color = "#FF0000".to_owned();
        draft.materials.texts[0].words = words(&[("A", 0, 1_000)]);
        let before = draft.clone();
        let shown = pixels(&draft, 0);
        assert!(shown.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        assert!(
            pixels(&draft, 1_000_000)
                .chunks(4)
                .all(|pixel| pixel == [0, 0, 0, 0])
        );
        assert_eq!(draft, before);
    }

    #[test]
    fn current_words_are_the_caption_that_is_drawn() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].text_color = "#FF0000".to_owned();
        draft.materials.texts[0].words = words(&[("\n", 0, 100)]);
        draft.materials.texts[0].current_words = words(&[("A", 0, 1_000)]);
        assert!(
            pixels(&draft, 0)
                .chunks(4)
                .any(|pixel| pixel == [255, 0, 0, 255])
        );
    }

    #[test]
    fn a_caption_style_pages_colors_sizes_and_boxes_the_spoken_word() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].font_size = 32.0;
        // Milliseconds: AB, then BA, then AA, then BB.
        draft.materials.texts[0].words = words(&[
            ("AB", 0, 200),
            ("BA", 200, 400),
            ("AA", 400, 600),
            ("BB", 600, 800),
        ]);
        let plain = pixels(&draft, 100_000);
        let width = 256;
        let wide = |rgba: &[u8]| inked_span(rgba, width).0;
        let yellow = |rgba: &[u8]| {
            rgba.chunks(4)
                .filter(|pixel| {
                    pixel[0] > 200 && pixel[1] > 180 && pixel[2] < 120 && pixel[3] == 255
                })
                .count()
        };
        // One word a page shows the spoken word alone.
        draft.materials.texts[0].caption_style = "one-word".to_owned();
        let one = pixels(&draft, 100_000);
        assert!(
            wide(&one) < wide(&plain) / 2,
            "{} {}",
            wide(&one),
            wide(&plain)
        );
        // Karaoke turns the spoken and read words yellow, the rest stay.
        draft.materials.texts[0].caption_style = "karaoke".to_owned();
        let early = pixels(&draft, 100_000);
        let later = pixels(&draft, 500_000);
        assert!(yellow(&early) > 0 && yellow(&later) > yellow(&early));
        assert!(ink(&early));
        // Highlight puts a yellow box behind the spoken word.
        draft.materials.texts[0].caption_style = "highlight".to_owned();
        let boxed = pixels(&draft, 100_000);
        // A box is one unbroken run of yellow across the word; inked
        // letters alone are broken up by the gaps between their strokes.
        let run = |rgba: &[u8]| {
            rgba.chunks(width * 4)
                .map(|row| {
                    row.chunks(4)
                        .fold((0, 0), |(best, now), pixel| {
                            let on = pixel[0] > 200 && pixel[1] > 180 && pixel[2] < 120;
                            let now = if on && pixel[3] == 255 { now + 1 } else { 0 };
                            (best.max(now), now)
                        })
                        .0
                })
                .max()
                .unwrap_or(0)
        };
        assert!(
            run(&boxed) > run(&early) * 2,
            "{} {}",
            run(&boxed),
            run(&early)
        );
        // Word by word hides the words not yet spoken.
        draft.materials.texts[0].caption_style = "word-by-word".to_owned();
        let first = pixels(&draft, 100_000);
        let all = pixels(&draft, 700_000);
        assert!(wide(&first) < wide(&all));
        // Pop grows the spoken word.
        draft.materials.texts[0].caption_style = "classic".to_owned();
        let classic = pixels(&draft, 100_000);
        draft.materials.texts[0].caption_style = "pop".to_owned();
        let pop = pixels(&draft, 100_000);
        assert_ne!(pop, classic);
        // An unknown style draws the words plain.
        draft.materials.texts[0].caption_style = "loud".to_owned();
        assert_eq!(pixels(&draft, 100_000), plain);
    }

    #[test]
    fn a_caption_takes_its_material_outline_shadow_and_box_like_a_title() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].words = words(&[("AB", 0, 500), ("BA", 500, 1_000)]);
        let has = |rgba: &[u8], color: [u8; 4]| rgba.chunks(4).any(|pixel| pixel == color);
        let plain = pixels(&draft, 100_000);
        assert!(ink(&plain));
        assert!(!has(&plain, [255, 0, 0, 255]));
        let text = &mut draft.materials.texts[0];
        text.border_color = "#FF0000".to_owned();
        text.border_alpha = 1.0;
        text.border_width = 0.08;
        let outlined = pixels(&draft, 100_000);
        assert!(has(&outlined, [255, 0, 0, 255]), "a red outline");
        assert!(ink(&outlined), "white letters inside it");
        let text = &mut draft.materials.texts[0];
        text.border_color = String::new();
        text.background_style = 1;
        text.background_color = "#0000FF".to_owned();
        text.background_alpha = 1.0;
        let boxed = pixels(&draft, 100_000);
        assert!(has(&boxed, [0, 0, 255, 255]), "a blue box behind the page");
        let text = &mut draft.materials.texts[0];
        text.background_style = 0;
        text.has_shadow = true;
        text.shadow_color = "#00FF00".to_owned();
        text.shadow_alpha = 1.0;
        text.shadow_distance = 8.0;
        let shaded = pixels(&draft, 100_000);
        assert!(
            shaded
                .chunks(4)
                .any(|pixel| pixel[1] > 100 && pixel[0] < 50 && pixel[3] > 100),
            "a green shadow"
        );
    }

    #[test]
    fn a_keyword_takes_the_spoken_look_and_a_stagger_runs_across_its_letters() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].font_size = 32.0;
        draft.materials.texts[0].caption_style = "karaoke".to_owned();
        draft.materials.texts[0].words = words(&[("AB", 0, 200), ("BA", 200, 400)]);
        let yellow = |rgba: &[u8]| {
            rgba.chunks(4)
                .filter(|pixel| {
                    pixel[0] > 200 && pixel[1] > 180 && pixel[2] < 120 && pixel[3] == 255
                })
                .count()
        };
        // Before either word is spoken nothing is yellow, unless it is marked.
        let unmarked = pixels(&draft, 0);
        draft.materials.texts[0].caption_keywords = vec![1];
        let marked = pixels(&draft, 0);
        assert!(
            yellow(&marked) > yellow(&unmarked),
            "the keyword is drawn yellow"
        );
        // A mark past the words is left out.
        draft.materials.texts[0].caption_keywords = vec![9];
        assert_eq!(pixels(&draft, 0), unmarked);

        // Pop grows the spoken word; with a stagger its letters grow in turn.
        draft.materials.texts[0].caption_keywords.clear();
        draft.materials.texts[0].caption_style = "pop".to_owned();
        let together = pixels(&draft, 210_000);
        draft.materials.texts[0].caption_stagger_ms = 150;
        let staggered = pixels(&draft, 210_000);
        assert_ne!(together, staggered);
    }

    #[test]
    fn a_long_plain_caption_shows_the_page_being_spoken_inside_the_canvas() {
        let mut draft = text_segment("", false);
        draft.materials.texts[0].font_size = 24.0;
        // Thirty words, a tenth of a second each, with no style and no line
        // width: Plain.
        let rows: Vec<(String, i64, i64)> = (0..30)
            .map(|index| {
                (
                    match index / 12 {
                        0 => "AB",
                        1 => "BA",
                        _ => "AA",
                    }
                    .to_owned(),
                    index * 30,
                    index * 30 + 30,
                )
            })
            .collect();
        let rows: Vec<(&str, i64, i64)> = rows
            .iter()
            .map(|(text, start, end)| (text.as_str(), *start, *end))
            .collect();
        draft.materials.texts[0].words = words(&rows);
        assert!(draft.materials.texts[0].caption_style.is_empty());
        assert_eq!(draft.materials.texts[0].line_max_width, 0.0);
        let width = 256;
        let first = pixels(&draft, 0);
        let later = pixels(&draft, 700_000);
        assert!(ink(&first) && ink(&later));
        // The page moves on as the words are spoken.
        assert_ne!(first, later);
        // A page is two short lines, not thirty words on one.
        let (wide, tall) = inked_span(&first, width);
        assert!(wide <= width * 9 / 10, "{wide}");
        assert!(tall < 24 * 4, "{tall}");
        // Six words a line are wider than the canvas at a larger size: the
        // line wraps inside it.
        draft.materials.texts[0].font_size = 48.0;
        let big = pixels(&draft, 0);
        let (wide, _) = inked_span(&big, width);
        assert!(ink(&big));
        assert!(wide <= width * 9 / 10 + 2, "{wide}");
        assert_eq!(
            cutlass_captions::caption_limits(""),
            cutlass_captions::PLAIN_LIMITS
        );
        assert_eq!(
            cutlass_captions::caption_limits("one-word"),
            cutlass_captions::caption_style("one-word").unwrap().limits
        );
    }
}
