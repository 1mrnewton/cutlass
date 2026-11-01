//! Text segments for one evaluated frame.
//!
//! The font bytes come from [`crate::Fonts`]. This module does not open
//! `font_path`. A missing face, a refused run, a hidden track, or a segment
//! that does not cover the time contributes nothing. The text picture is
//! placed with the segment clip, including its line keyframes.

use cutlass_engine::DisplayQuarter;
use cutlass_model::{Draft, MaterialRef, Micros, TrackAttribute};
use cutlass_picture::{Picture, PictureSource, Sample};
use cutlass_text::TextSource;

use crate::Fonts;
use crate::media::{picture_transform, place_with, segment_clip};

pub(crate) struct TextLayer {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: PlacedText,
}

pub(crate) struct PlacedText {
    source: TextSource,
    width: u32,
    height: u32,
    transform: crate::media::PictureTransform,
}

impl PictureSource for PlacedText {
    fn sample(&self, time: Micros) -> Sample {
        let Sample::Ready(Picture::Cpu(bytes)) = self.source.sample(time) else {
            return Sample::Miss;
        };
        place_with(
            self.width,
            self.height,
            bytes,
            self.transform,
            DisplayQuarter::Zero,
        )
    }
}

pub(crate) fn text_layers(
    draft: &Draft,
    fonts: &dyn Fonts,
    width: u32,
    height: u32,
    time: Micros,
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
            let Some(bytes) = fonts.face(&material.font_path) else {
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
            let Ok(source) = TextSource::open(bytes, None, material, segment) else {
                continue;
            };
            layers.push(TextLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                placed: PlacedText {
                    source,
                    width,
                    height,
                    transform,
                },
            });
        }
    }
    layers
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
        Segment, TimeRange, TrackAttribute,
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
}
