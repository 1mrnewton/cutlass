//! Text segments for one evaluated frame.
//!
//! The font bytes come from [`crate::Fonts`]. This module does not open
//! `font_path`. A missing face, a refused run, a hidden track, or a segment
//! that does not cover the time contributes nothing. The text picture is
//! placed with the segment clip, including its line keyframes.
//!
//! Word columns replace `content`. Those glyphs are the caption sample.
//! An identity style keeps the material color. A sampled color replaces it.
//! Style position is a pixel shift. Scale and rotation are not applied.

use cutlass_captions::{CaptionQuery, PageLimits, RangeStyles, Rgba, sample_caption};
use cutlass_engine::DisplayQuarter;
use std::collections::HashMap;

use cutlass_gpu::{Affine, Placement};
use cutlass_model::{
    BloomParams, Clip, Draft, MaterialEffect, MaterialRef, MaterialText, Micros, Segment,
    SegmentId, TrackAttribute, UniformScale, WordColumns, segment_bloom,
};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample};
use cutlass_text::{DrawnGlyph, Layout, PlacedGlyph, TextSource, paint_glyphs, text_fill};

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
            let Some(MaterialRef::Text(material)) =
                draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            if material.font_path.is_empty() {
                continue;
            }
            let Some(face) = fonts.face(&material.font_path) else {
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
                let Some(bytes) = caption_picture(face, material, segment.target.start, time)
                else {
                    continue;
                };
                (bytes, transform)
            } else {
                let wanted = density(&clip, &uniform, material.font_size);
                let glow = glow(draft, segment);
                let kept = rasters
                    .get(&segment.id)
                    .filter(|(drawn, _)| *drawn == wanted)
                    .cloned();
                let Some((drawn, bytes)) = kept.or_else(|| {
                    title_picture(
                        Faces { fonts, face },
                        material,
                        glow.as_ref(),
                        segment,
                        time,
                        wanted,
                    )
                }) else {
                    continue;
                };
                rasters.insert(segment.id.clone(), (drawn, bytes.clone()));
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

/// A title drawn with `density` pixels to each shown pixel, and that
/// density; at 1 when it cannot be drawn larger.
fn title_picture(
    faces: Faces<'_>,
    material: &MaterialText,
    glow: Option<&BloomParams>,
    segment: &Segment,
    time: Micros,
    density: f64,
) -> Option<(f64, CpuBytes)> {
    let draw = |density: f64| {
        let mut larger = material.clone();
        larger.font_size *= density;
        larger.line_max_width *= density;
        let source =
            TextSource::open_with_faces(faces.face, None, &larger, glow, segment, &|path| {
                faces.named(path)
            })
            .ok()?;
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
    let face = fonts.face(&material.font_path)?;
    let glow = glow(draft, segment);
    let faces = Faces { fonts, face };
    let source =
        TextSource::open_with_faces(face, None, material, glow.as_ref(), segment, &|path| {
            faces.named(path)
        })
        .ok()?;
    Some(source.size())
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

/// Word columns replace `content`. An empty style keeps the material color.
/// A sampled color that is not opaque white replaces it. Position is a pixel
/// shift, right and down. Scale and rotation on the style are not applied.
fn caption_picture(
    face: &[u8],
    material: &MaterialText,
    placed_at: Micros,
    time: Micros,
) -> Option<CpuBytes> {
    let styles = RangeStyles::identity();
    let frame = sample_caption(&CaptionQuery {
        material,
        font: face,
        timeline: time,
        placed_at,
        limits: PageLimits {
            words_per_line: 0,
            lines_per_page: 0,
        },
        keywords: &[],
        styles: &styles,
        stagger_shift_ms: 0,
    })
    .ok()?;
    let base = text_fill(material).ok()?;
    let mut drawn = Vec::new();
    for batch in &frame.batches {
        for sample in &batch.glyphs {
            let Some(baseline) = baseline_of(&frame.letters.layout, &sample.glyph) else {
                continue;
            };
            let Some(shift_x) = shift_px(sample.style.position_x) else {
                continue;
            };
            let Some(shift_y) = shift_px(sample.style.position_y) else {
                continue;
            };
            let Some(fill) = ink_color(base, sample.style.color) else {
                continue;
            };
            drawn.push(HeldGlyph {
                placed: sample.glyph.clone(),
                baseline,
                fill,
                shift_x,
                shift_y,
            });
        }
    }
    let borrowed: Vec<DrawnGlyph<'_>> = drawn
        .iter()
        .map(|glyph| DrawnGlyph {
            placed: &glyph.placed,
            baseline: glyph.baseline,
            fill: glyph.fill,
            shift_x: glyph.shift_x,
            shift_y: glyph.shift_y,
        })
        .collect();
    match paint_glyphs(face, frame.letters.layout.font_size, &borrowed) {
        Ok(Picture::Cpu(bytes)) => Some(bytes),
        Ok(Picture::Platform(_)) | Err(_) => None,
    }
}

struct HeldGlyph {
    placed: PlacedGlyph,
    baseline: f32,
    fill: [u8; 4],
    shift_x: f32,
    shift_y: f32,
}

fn baseline_of(layout: &Layout, glyph: &PlacedGlyph) -> Option<f32> {
    layout.lines.iter().find_map(|line| {
        let found = line.glyphs.iter().any(|item| {
            item.source == glyph.source && item.glyph_id == glyph.glyph_id && item.x == glyph.x
        });
        found.then_some(line.baseline)
    })
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
}
