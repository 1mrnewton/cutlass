//! A compound segment is the child draft's picture.
//!
//! Parent time is mapped with the engine placement: freeze, then speed,
//! then reverse. That microsecond is where the child is evaluated, on the
//! child's canvas. The parent then places that picture with the compound
//! clip. A cycle, a chain past [`MAX_COMPOUND_DEPTH`], or a material with
//! no inlined draft contributes no picture. A file path on the material is
//! not opened, and `precompile_combination` does not replace the draft.
//! Playback does not read `sub_draft_stack`.

use cutlass_compositor::Device;
use cutlass_engine::{DisplayQuarter, ReaderFactory, map_timeline, placement_of};
use cutlass_model::{Draft, DraftId, MaterialRef, Micros, SegmentId, TrackAttribute};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample};

use crate::frame::{EvalError, Fonts, Snapshot, compose};
use crate::media::{OpenClips, PictureTransform, picture_transform, place_with, segment_clip};

/// Compound drafts above this many ancestors are not drawn.
pub(crate) const MAX_COMPOUND_DEPTH: usize = 32;

pub(crate) struct NestedLayer {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: NestedPicture,
}

pub(crate) struct NestedPicture {
    bytes: CpuBytes,
    canvas_w: u32,
    canvas_h: u32,
    transform: PictureTransform,
}

impl PictureSource for NestedPicture {
    fn sample(&self, _time: Micros) -> Sample {
        let Sample::Ready(Picture::Cpu(bytes)) = place_with(
            self.canvas_w,
            self.canvas_h,
            self.bytes.clone(),
            self.transform,
            DisplayQuarter::Zero,
        ) else {
            return Sample::Miss;
        };
        Sample::Ready(Picture::Cpu(bytes))
    }
}

pub(crate) struct NestPass<'a, D: Device> {
    pub(crate) media: &'a ReaderFactory,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) time: Micros,
    pub(crate) fonts: &'a dyn Fonts,
    pub(crate) device: &'a mut D,
    pub(crate) clips: &'a mut OpenClips,
    pub(crate) revision: u64,
    pub(crate) ancestors: &'a [DraftId],
}

pub(crate) fn nest_layers<D: Device>(
    draft: &Draft,
    pass: &mut NestPass<'_, D>,
) -> Result<Vec<NestedLayer>, EvalError<D::Error>> {
    let mut layers = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible || segment.source_range.is_none() {
                continue;
            }
            if !covers(
                segment.target.start.0,
                segment.target.duration.0,
                pass.time.0,
            ) {
                continue;
            }
            let Some(MaterialRef::Draft(material)) =
                draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            let Some(child) = material.draft.as_deref() else {
                continue;
            };
            if skips_child(draft, child, pass.ancestors) {
                continue;
            }
            let Some((clip, uniform)) = segment_clip(segment, pass.time) else {
                continue;
            };
            let Some(transform) = picture_transform(&clip, &uniform, pass.width, pass.height)
            else {
                continue;
            };
            if transform.omitted() {
                continue;
            }
            let Ok(map) = placement_of(draft, segment) else {
                continue;
            };
            let Ok(mapped) = map_timeline(&map, pass.time) else {
                continue;
            };
            let Some(rendered) = child_frame(draft, child, &segment.id, mapped, pass)? else {
                continue;
            };
            let Ok(bytes) =
                CpuBytes::new(rendered.extent.width, rendered.extent.height, rendered.rgba)
            else {
                continue;
            };
            layers.push(NestedLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                placed: NestedPicture {
                    bytes,
                    canvas_w: pass.width,
                    canvas_h: pass.height,
                    transform,
                },
            });
        }
    }
    Ok(layers)
}

fn child_frame<D: Device>(
    draft: &Draft,
    child: &Draft,
    segment: &SegmentId,
    time: Micros,
    pass: &mut NestPass<'_, D>,
) -> Result<Option<crate::Frame>, EvalError<D::Error>> {
    let child_clips = pass.clips.child(segment);
    let chain = descend(pass.ancestors, &draft.id);
    match compose(
        &Snapshot::with_fonts(child, pass.fonts),
        time,
        pass.device,
        pass.media,
        child_clips,
        pass.revision,
        &chain,
    ) {
        Ok(frame) => Ok(Some(frame)),
        Err(EvalError::Canvas) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(crate) fn skips_child(draft: &Draft, child: &Draft, ancestors: &[DraftId]) -> bool {
    ancestors.len() >= MAX_COMPOUND_DEPTH
        || child.id == draft.id
        || ancestors.iter().any(|id| id == &child.id)
}

pub(crate) fn descend(ancestors: &[DraftId], here: &DraftId) -> Vec<DraftId> {
    let mut chain = Vec::with_capacity(ancestors.len().saturating_add(1));
    chain.extend_from_slice(ancestors);
    chain.push(here.clone());
    chain
}

fn covers(start: i64, duration: i64, time: i64) -> bool {
    let Some(end) = start.checked_add(duration) else {
        return false;
    };
    time >= start && time < end
}

#[cfg(test)]
mod tests {
    use cutlass_engine::ReaderFactory;
    use cutlass_model::{
        Draft, DraftId, MaterialColorConfig, MaterialDraft, Micros, Segment, TimeRange,
        TrackAttribute,
    };
    use cutlass_renderer_software::SoftwareDevice;

    use super::*;
    use crate::frame::{Snapshot, frame};
    use crate::media::OpenClips;

    fn pixels(draft: &Draft, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        frame(
            &Snapshot::new(draft),
            Micros(time),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba
    }

    fn painted(id: &str, spans: &[(&str, i64, i64)]) -> Draft {
        let mut draft = Draft::empty(id);
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        for (index, (color, start, duration)) in spans.iter().enumerate() {
            let material = format!("color-{index}");
            draft.materials.material_colors.push(MaterialColorConfig {
                id: material.clone().into(),
                is_color_clip: true,
                solid_color: (*color).to_owned(),
                ..MaterialColorConfig::default()
            });
            draft.tracks[0].segments.push(Segment::new(
                format!("segment-{index}"),
                material,
                TimeRange::new(*start, *duration),
            ));
        }
        draft
    }

    fn compound(id: &str, child: Draft, source: TimeRange, target: TimeRange) -> Draft {
        let mut draft = Draft::empty(id);
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        draft.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let mut segment = Segment::new("compound", "nest", target);
        segment.source_range = Some(source);
        segment.render_index = 1;
        draft.tracks[0].segments.push(segment);
        draft
    }

    fn solid(draft: &mut Draft, color: &str) {
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "solid".into(),
            is_color_clip: true,
            solid_color: color.to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("solid", "solid", TimeRange::new(0, 2_000)));
    }

    #[test]
    fn a_nested_draft_is_the_child_picture() {
        let child = painted("child", &[("0xFFFF0000", 0, 1_000)]);
        let draft = compound(
            "parent",
            child,
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        let before = draft.clone();
        let mut faded = compound(
            "parent",
            painted("child", &[("0xFFC80000", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        faded.tracks[0].segments[0].clip.alpha = 0.5;
        assert_eq!(pixels(&draft, 0), vec![255, 0, 0, 255]);
        assert_eq!(pixels(&faded, 0), vec![100, 0, 0, 128]);
        assert_eq!(draft, before);

        let offset = compound(
            "parent",
            painted(
                "child",
                &[("0xFFFF0000", 0, 1_000), ("0xFF0000FF", 1_000, 1_000)],
            ),
            TimeRange::new(1_000, 1_000),
            TimeRange::new(0, 1_000),
        );
        assert_eq!(pixels(&offset, 0), vec![0, 0, 255, 255]);

        let inner = painted("inner", &[("0xFFFF0000", 0, 1_000)]);
        let mid = compound(
            "mid",
            inner,
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        let outer = compound(
            "outer",
            mid,
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        assert_eq!(pixels(&outer, 0), vec![255, 0, 0, 255]);
    }

    #[test]
    fn speed_and_reverse_choose_the_child_time() {
        let child = painted(
            "child",
            &[("0xFFFF0000", 0, 1_000), ("0xFF0000FF", 1_000, 1_000)],
        );
        let mut draft = compound(
            "parent",
            child,
            TimeRange::new(0, 2_000),
            TimeRange::new(0, 1_000),
        );
        draft.tracks[0].segments[0].speed = 2.0;
        assert_eq!(pixels(&draft, 0), vec![255, 0, 0, 255]);
        assert_eq!(pixels(&draft, 500), vec![0, 0, 255, 255]);

        let mut reversed = compound(
            "parent",
            painted("child", &[("0xFFFF0000", 0, 500), ("0xFF0000FF", 500, 500)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        reversed.tracks[0].segments[0].reverse = true;
        assert_eq!(pixels(&reversed, 0), vec![0, 0, 255, 255]);
        assert_eq!(pixels(&reversed, 999), vec![255, 0, 0, 255]);
    }

    #[test]
    fn a_cycle_a_missing_draft_and_a_bad_child_leave_the_other_layers() {
        let cycle = compound(
            "parent",
            painted("parent", &[("0xFF00FF00", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        assert_eq!(pixels(&cycle, 0), vec![0, 0, 0, 0]);

        let mut unmapped = compound(
            "parent",
            painted("child", &[("0xFF00FF00", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        unmapped.tracks[0].segments[0].source_range = None;
        assert_eq!(pixels(&unmapped, 0), vec![0, 0, 0, 0]);

        let back = compound(
            "mid",
            painted("outer", &[("0xFF00FF00", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        let mut outer = compound(
            "outer",
            back,
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        solid(&mut outer, "0xFF0000FF");
        assert_eq!(pixels(&outer, 0), vec![0, 0, 255, 255]);

        let mut path = Draft::empty("parent");
        path.canvas_config.width = 1;
        path.canvas_config.height = 1;
        path.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft_file_path: "missing.json".into(),
            ..MaterialDraft::default()
        });
        let mut segment = Segment::new("compound", "nest", TimeRange::new(0, 1_000));
        segment.source_range = Some(TimeRange::new(0, 1_000));
        path.tracks[0].segments.push(segment);
        assert_eq!(pixels(&path, 0), vec![0, 0, 0, 0]);

        let mut hidden = compound(
            "parent",
            painted("child", &[("0xFFFF0000", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        hidden.tracks[0].attribute = TrackAttribute(TrackAttribute::HIDE);
        assert_eq!(pixels(&hidden, 0), vec![0, 0, 0, 0]);

        let mut stalled = compound(
            "parent",
            painted("child", &[("0xFFFF0000", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        stalled.tracks[0].segments[0].speed = 0.0;
        solid(&mut stalled, "0xFF0000FF");
        assert_eq!(pixels(&stalled, 0), vec![0, 0, 255, 255]);

        let mut broken = painted("child", &[("0xFFFF0000", 0, 1_000)]);
        broken.canvas_config.width = -1;
        let mut parent = compound(
            "parent",
            broken,
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        solid(&mut parent, "0xFF0000FF");
        assert_eq!(pixels(&parent, 0), vec![0, 0, 255, 255]);
    }

    #[test]
    fn a_chain_past_the_depth_limit_is_not_drawn() {
        let draft = compound(
            "parent",
            painted("child", &[("0xFFFF0000", 0, 1_000)]),
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        let deep: Vec<DraftId> = (0..MAX_COMPOUND_DEPTH)
            .map(|index| DraftId::new(format!("above-{index}")))
            .collect();
        assert!(layers(&draft, &deep).is_empty());
        let under: Vec<DraftId> = (0..MAX_COMPOUND_DEPTH - 1)
            .map(|index| DraftId::new(format!("above-{index}")))
            .collect();
        assert_eq!(layers(&draft, &under).len(), 1);
    }

    fn layers(draft: &Draft, ancestors: &[DraftId]) -> Vec<NestedLayer> {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        let fonts = crate::frame::NoFonts;
        let mut pass = NestPass {
            media: &media,
            width: 1,
            height: 1,
            time: Micros(0),
            fonts: &fonts,
            device: &mut device,
            clips: &mut clips,
            revision: 0,
            ancestors,
        };
        nest_layers(draft, &mut pass).expect("layers")
    }
}
