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
    Draft, MaterialRef, MaterialText, Micros, SegmentId, TrackAttribute, WordColumns,
};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample};
use cutlass_text::{DrawnGlyph, Layout, PlacedGlyph, TextSource, paint_glyphs, text_fill};

use crate::Fonts;
use crate::gpu::{GpuLayer, GpuSource};
use crate::media::{gpu_place, picture_transform, place_with, segment_clip};

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

/// Text segments covering `time`, placed. A title's raster is kept in
/// `rasters` for its segment, so a later frame of the same revision does not
/// lay it out and draw it again. Captions change with time and are drawn
/// each frame.
pub(crate) fn text_layers(
    draft: &Draft,
    fonts: &dyn Fonts,
    width: u32,
    height: u32,
    time: Micros,
    rasters: &mut HashMap<SegmentId, CpuBytes>,
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
            let Some((clip, uniform)) = segment_clip(segment, time) else {
                continue;
            };
            let Some(transform) = picture_transform(&clip, &uniform, width, height) else {
                continue;
            };
            if transform.omitted() {
                continue;
            }
            let raster = if words_present(material) {
                let Some(bytes) = caption_picture(face, material, segment.target.start, time)
                else {
                    continue;
                };
                bytes
            } else if let Some(kept) = rasters.get(&segment.id) {
                kept.clone()
            } else {
                let Ok(source) = TextSource::open(face, None, material, segment) else {
                    continue;
                };
                let Sample::Ready(Picture::Cpu(bytes)) = source.sample(time) else {
                    continue;
                };
                rasters.insert(segment.id.clone(), bytes.clone());
                bytes
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
        Clip, CommonKeyframe, CommonKeyframeGroup, Draft, MaterialDraft, MaterialText, Micros,
        Segment, TimeRange, TrackAttribute, WordColumns,
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

    #[test]
    fn a_missing_font_a_hidden_track_and_rich_text_draw_nothing() {
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
