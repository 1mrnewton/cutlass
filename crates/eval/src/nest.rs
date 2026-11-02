//! A compound segment is the child draft's picture.
//!
//! Parent time is mapped with the engine placement: freeze, then speed,
//! then reverse. That microsecond is where the child is evaluated, on the
//! child's canvas. The parent then places that picture with the compound
//! clip. A crop quad on the compound material is sampled from that picture
//! first. An all-zero quad keeps the child picture. A cycle, a chain past
//! [`MAX_COMPOUND_DEPTH`], or a material with
//! no inlined draft contributes no picture. A file path on the material is
//! not opened, and `precompile_combination` does not replace the draft.
//! Playback does not read `sub_draft_stack`.

use cutlass_compositor::Device;
use cutlass_effects::PackageStore;
use cutlass_engine::{DisplayQuarter, ReaderFactory, map_timeline, placement_of};
use cutlass_gpu::{Affine, Placement};
use cutlass_model::{
    Crop, Draft, DraftId, MaterialRef, Micros, Segment, SegmentId, TrackAttribute,
};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample};

use crate::canvas::over_backdrop;
use crate::frame::{EvalError, Fonts, Snapshot, compose, compose_layers};
use crate::gpu::{Composed, GpuLayer, GpuSource};
use crate::media::{
    OpenClips, PictureTransform, apply_crop, backdrop_for, gpu_crop, gpu_place, picture_transform,
    place_with, segment_clip,
};

/// Compound drafts above this many ancestors are not drawn.
pub(crate) const MAX_COMPOUND_DEPTH: usize = 32;

pub(crate) struct NestedLayer {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: NestedPicture,
}

pub(crate) struct NestedPicture {
    child: Child,
    canvas_w: u32,
    canvas_h: u32,
    transform: PictureTransform,
    backdrop: crate::canvas::Backdrop,
}

/// The child frame a compound places.
enum Child {
    /// Composed on the CPU, then cropped.
    Cpu(CpuBytes),
    /// Left as layers for the GPU, with the crop still to apply.
    Gpu(Composed, Crop),
}

impl PictureSource for NestedPicture {
    fn sample(&self, _time: Micros) -> Sample {
        let Child::Cpu(bytes) = &self.child else {
            return Sample::Miss;
        };
        let Sample::Ready(Picture::Cpu(placed)) = place_with(
            self.canvas_w,
            self.canvas_h,
            bytes.clone(),
            self.transform,
            DisplayQuarter::Zero,
        ) else {
            return Sample::Miss;
        };
        let sharp = matches!(self.backdrop, crate::canvas::Backdrop::Blur(_)).then_some(bytes);
        Sample::Ready(Picture::Cpu(over_backdrop(placed, sharp, &self.backdrop)))
    }
}

impl NestedPicture {
    /// The compound as GPU layers: a color backdrop, if any, then the child
    /// frame placed like a picture. `None` for a child composed on the CPU,
    /// which the caller samples instead.
    pub(crate) fn gpu(&self) -> Option<Vec<GpuLayer>> {
        let Child::Gpu(composed, crop) = &self.child else {
            return None;
        };
        let (source, size) = match composed {
            Composed::Layers(layers) => (
                GpuSource::Nested(Box::new(layers.clone())),
                (layers.width, layers.height),
            ),
            Composed::Frame(frame) => {
                let bytes =
                    CpuBytes::new(frame.extent.width, frame.extent.height, frame.rgba.clone())
                        .ok()?;
                (
                    GpuSource::Rgba(bytes),
                    (frame.extent.width, frame.extent.height),
                )
            }
        };
        let (to_oriented, picture) = gpu_crop(*crop, 1.0, size)?;
        let placement = Placement {
            to_picture: gpu_place(self.transform, picture, (self.canvas_w, self.canvas_h))?,
            picture,
            to_oriented,
            oriented: size,
            to_texture: Affine::IDENTITY,
            opacity: f64::from(self.transform.opacity()) / 255.0,
        };
        let mut layers = Vec::new();
        match &self.backdrop {
            crate::canvas::Backdrop::None => {}
            crate::canvas::Backdrop::Color(pixel) => {
                layers.push(GpuLayer::fill(CpuBytes::new(1, 1, pixel.to_vec()).ok()?));
            }
            crate::canvas::Backdrop::Blur(_) | crate::canvas::Backdrop::Image(_) => return None,
        }
        layers.push(GpuLayer {
            source,
            placement,
            backdrop: None,
        });
        Some(layers)
    }
}

pub(crate) struct NestPass<'a, D: Device> {
    pub(crate) media: &'a ReaderFactory,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) time: Micros,
    pub(crate) fonts: &'a dyn Fonts,
    pub(crate) packages: &'a dyn PackageStore,
    pub(crate) device: &'a mut D,
    pub(crate) clips: &'a mut OpenClips,
    pub(crate) revision: u64,
    pub(crate) ancestors: &'a [DraftId],
    /// Leave each child as GPU layers instead of composing it.
    pub(crate) gpu: bool,
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
            let Some(placed) = place_child(draft, segment, pass.time, pass, pass.gpu)? else {
                continue;
            };
            layers.push(NestedLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                placed,
            });
        }
    }
    Ok(layers)
}

/// The compound picture at `time`, placed on the parent canvas.
///
/// `time` must fall inside the segment. A segment that is not a compound,
/// a skipped child, or a child that does not draw returns nothing.
pub(crate) fn picture_at<D: Device>(
    draft: &Draft,
    segment: &Segment,
    time: Micros,
    pass: &mut NestPass<'_, D>,
) -> Result<Option<CpuBytes>, EvalError<D::Error>> {
    let Some(placed) = place_child(draft, segment, time, pass, false)? else {
        return Ok(None);
    };
    match placed.sample(time) {
        Sample::Ready(Picture::Cpu(bytes)) => Ok(Some(bytes)),
        Sample::Ready(Picture::Platform(_)) | Sample::Miss => Ok(None),
    }
}

fn place_child<D: Device>(
    draft: &Draft,
    segment: &Segment,
    time: Micros,
    pass: &mut NestPass<'_, D>,
    gpu: bool,
) -> Result<Option<NestedPicture>, EvalError<D::Error>> {
    if !segment.visible || segment.source_range.is_none() {
        return Ok(None);
    }
    if !covers(segment.target.start.0, segment.target.duration.0, time.0) {
        return Ok(None);
    }
    let Some(MaterialRef::Draft(material)) = draft.materials.get(segment.material_id.as_str())
    else {
        return Ok(None);
    };
    let crop = material.crop;
    let Some(child) = material.draft.as_deref() else {
        return Ok(None);
    };
    if skips_child(draft, child, pass.ancestors) {
        return Ok(None);
    }
    let Some((clip, uniform)) = segment_clip(segment, time) else {
        return Ok(None);
    };
    let Some(transform) = picture_transform(&clip, &uniform, pass.width, pass.height) else {
        return Ok(None);
    };
    if transform.omitted() {
        return Ok(None);
    }
    let Ok(map) = placement_of(draft, segment) else {
        return Ok(None);
    };
    let Ok(mapped) = map_timeline(&map, time) else {
        return Ok(None);
    };
    let backdrop = backdrop_for(
        draft,
        segment,
        pass.media,
        pass.clips,
        pass.width,
        pass.height,
    );
    let on_gpu = gpu
        && matches!(
            backdrop,
            crate::canvas::Backdrop::None | crate::canvas::Backdrop::Color(_)
        );
    let child = if on_gpu {
        let Some(layers) = child_layers(draft, child, &segment.id, mapped, pass)? else {
            return Ok(None);
        };
        Child::Gpu(layers, crop)
    } else {
        let Some(rendered) = child_frame(draft, child, &segment.id, mapped, pass)? else {
            return Ok(None);
        };
        let Ok(bytes) = CpuBytes::new(rendered.extent.width, rendered.extent.height, rendered.rgba)
        else {
            return Ok(None);
        };
        Child::Cpu(apply_crop(bytes, crop, 1.0))
    };
    Ok(Some(NestedPicture {
        child,
        canvas_w: pass.width,
        canvas_h: pass.height,
        transform,
        backdrop,
    }))
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
        &Snapshot::with_fonts_and_packages(child, pass.fonts, pass.packages),
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

/// The child draft at `time` as GPU layers, or its CPU frame when it needs one.
fn child_layers<D: Device>(
    draft: &Draft,
    child: &Draft,
    segment: &SegmentId,
    time: Micros,
    pass: &mut NestPass<'_, D>,
) -> Result<Option<Composed>, EvalError<D::Error>> {
    let child_clips = pass.clips.child(segment);
    let chain = descend(pass.ancestors, &draft.id);
    match compose_layers(
        &Snapshot::with_fonts_and_packages(child, pass.fonts, pass.packages),
        time,
        pass.device,
        pass.media,
        child_clips,
        pass.revision,
        &chain,
    ) {
        Ok(composed) => Ok(Some(composed)),
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
        Crop, Draft, DraftId, MaterialColorConfig, MaterialDraft, Micros, Segment, TimeRange,
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
    fn a_crop_on_the_compound_keeps_the_middle() {
        let mut child = Draft::empty("child");
        child.canvas_config.width = 4;
        child.canvas_config.height = 4;
        child.materials.material_colors.push(MaterialColorConfig {
            id: "solid".into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        child.tracks[0]
            .segments
            .push(Segment::new("solid", "solid", TimeRange::new(0, 1_000)));
        let mut draft = compound(
            "parent",
            child,
            TimeRange::new(0, 1_000),
            TimeRange::new(0, 1_000),
        );
        draft.canvas_config.width = 4;
        draft.canvas_config.height = 4;
        let full = pixels(&draft, 0);
        assert_eq!(&full[0..4], &[255, 0, 0, 255]);
        draft.materials.drafts[0].crop = Crop {
            upper_left_x: 0.25,
            upper_left_y: 0.25,
            upper_right_x: 0.75,
            upper_right_y: 0.25,
            lower_left_x: 0.25,
            lower_left_y: 0.75,
            lower_right_x: 0.75,
            lower_right_y: 0.75,
        };
        let cropped = pixels(&draft, 0);
        assert_eq!(&cropped[0..4], &[0, 0, 0, 0]);
        assert_eq!(&cropped[20..24], &[255, 0, 0, 255]);
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
        let packages = cutlass_effects::MissingPackages;
        let mut pass = NestPass {
            media: &media,
            width: 1,
            height: 1,
            time: Micros(0),
            fonts: &fonts,
            packages: &packages,
            device: &mut device,
            clips: &mut clips,
            revision: 0,
            ancestors,
            gpu: false,
        };
        nest_layers(draft, &mut pass).expect("layers")
    }
}
