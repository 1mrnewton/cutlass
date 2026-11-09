//! Lottie stickers: a video material whose file is a `.json` animation.
//!
//! The clip's own time comes from the segment's playback map, as a movie's
//! does, so speed, reverse and freeze apply; the animation plays round and
//! round for as long as the clip lasts, as an animated GIF does. The frame
//! is drawn at the animation's own size (at most [`MAX_RASTER_EDGE`] on its
//! longer edge), fitted and placed with the clip like any picture. The GPU
//! path hands the frame's geometry to the renderer; the CPU path draws it
//! with `cutlass-lottie`'s rasterizer, which the GPU matches. A file that
//! does not read or parse leaves the clip off the frame.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;

use cutlass_engine::DisplayQuarter;
use cutlass_gpu::{Affine, Placement};
use cutlass_lottie::{Scene, is_lottie_path, rasterize};
use cutlass_model::{Draft, MaterialRef, Micros, SegmentId, TrackAttribute};
use cutlass_picture::{CpuBytes, PictureSource, Sample};

use crate::gpu::{GpuLayer, GpuSource};
use crate::media::{
    KeptShape, OpenClips, PictureTransform, gpu_place, picture_transform, place_with,
};

/// The longest edge a Lottie frame is drawn at.
pub(crate) const MAX_RASTER_EDGE: f64 = 2_048.0;
/// Above this many sample-edge tests the frame is drawn on the CPU, which
/// walks edges once per row instead of once per pixel.
const MAX_GPU_COST: u64 = 1 << 31;

pub(crate) struct LottieLayer {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: PlacedLottie,
}

pub(crate) struct PlacedLottie {
    scene: Arc<Scene>,
    width: u32,
    height: u32,
    transform: PictureTransform,
    raster: OnceCell<Option<CpuBytes>>,
    /// Drawn before, as a shape is: the GPU takes the picture rather than
    /// drawing the geometry again.
    kept: bool,
}

impl PlacedLottie {
    fn raster(&self) -> Option<&CpuBytes> {
        self.raster
            .get_or_init(|| {
                CpuBytes::new(self.scene.width, self.scene.height, rasterize(&self.scene)).ok()
            })
            .as_ref()
    }

    /// The frame as one vector layer, placed as [`PictureSource::sample`]
    /// places the raster. `None` when the frame costs the GPU too much.
    pub(crate) fn gpu(&self) -> Option<Vec<GpuLayer>> {
        let kept = if self.kept {
            self.raster().cloned()
        } else {
            None
        };
        if kept.is_none() && self.scene.gpu_cost() > MAX_GPU_COST {
            return None;
        }
        let size = (self.scene.width, self.scene.height);
        let placement = Placement {
            to_picture: gpu_place(self.transform, size, (self.width, self.height))?,
            picture: size,
            to_oriented: Affine::IDENTITY,
            oriented: size,
            to_texture: Affine::IDENTITY,
            opacity: f64::from(self.transform.opacity()) / 255.0,
        };
        Some(vec![GpuLayer {
            source: kept.map_or_else(
                || GpuSource::Vector(Arc::clone(&self.scene)),
                GpuSource::Rgba,
            ),
            placement,
            backdrop: None,
        }])
    }
}

impl PictureSource for PlacedLottie {
    fn sample(&self, _time: Micros) -> Sample {
        let Some(raster) = self.raster() else {
            return Sample::Miss;
        };
        place_with(
            self.width,
            self.height,
            raster.clone(),
            self.transform,
            DisplayQuarter::Zero,
        )
    }
}

/// The size a `width` by `height` animation is drawn at.
/// Shape clips at `time`, each a vector scene drawn at the clip's scale so
/// it stays sharp, placed like a picture at its own size in canvas pixels.
///
/// A shape is drawn once at the density its scale asks for and the picture
/// is kept in `kept`, shown on every frame after, and after moves and other
/// clips' edits; only a change to the shape, or a scale that crosses a
/// quarter step, draws it again.
pub(crate) fn shape_layers(
    draft: &Draft,
    width: u32,
    height: u32,
    time: Micros,
    kept: &mut HashMap<SegmentId, KeptShape>,
) -> Vec<LottieLayer> {
    let mut layers = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible {
                continue;
            }
            let start = segment.target.start.0;
            let Some(end) = start.checked_add(segment.target.duration.0) else {
                continue;
            };
            if time.0 < start || time.0 >= end {
                continue;
            }
            let Some(MaterialRef::Shape(stored)) =
                draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            // A keyed shape is drawn as it is at `time`; the kept picture
            // is compared with that, so it is redrawn only when it moves.
            let keyed =
                cutlass_model::shape_style_at(stored, segment, &draft.keyframe_graph_list, time);
            let shape = keyed.as_ref().unwrap_or(stored);
            let Some((clip, uniform)) = cutlass_model::animated_at(draft, segment, time) else {
                continue;
            };
            let Some(transform) =
                picture_transform(&clip, &uniform, width, height).map(PictureTransform::native)
            else {
                continue;
            };
            if transform.omitted() {
                continue;
            }
            let scale = if uniform.on {
                uniform.value.abs()
            } else {
                clip.scale.x.abs().max(clip.scale.y.abs())
            };
            // Up in quarter steps, so a pinch does not redraw at every move.
            let wanted = if scale.is_finite() && scale > 1.0 {
                ((scale * 4.0).ceil() / 4.0).min(MAX_SHAPE_DENSITY)
            } else {
                1.0
            };
            let known = kept
                .get(&segment.id)
                .filter(|known| known.wanted == wanted && known.shape == *shape)
                .cloned();
            let Some(shape) = known.or_else(|| {
                let (scene, drawn) = cutlass_generated::shape_scene(shape, wanted)?;
                let raster = CpuBytes::new(scene.width, scene.height, rasterize(&scene)).ok()?;
                let drawn = KeptShape {
                    shape: shape.clone(),
                    wanted,
                    drawn,
                    raster,
                };
                // Pictures of shapes since deleted go once there are many.
                if kept.len() >= MAX_KEPT_SHAPES {
                    kept.clear();
                }
                kept.insert(segment.id.clone(), drawn.clone());
                Some(drawn)
            }) else {
                continue;
            };
            let (picture_w, picture_h) = (shape.raster.width(), shape.raster.height());
            layers.push(LottieLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                placed: PlacedLottie {
                    // The size the picture is placed by; the geometry is
                    // not drawn again.
                    scene: Arc::new(Scene {
                        width: picture_w,
                        height: picture_h,
                        items: Vec::new(),
                    }),
                    width,
                    height,
                    transform: transform.dense(shape.drawn),
                    raster: OnceCell::from(Some(shape.raster)),
                    kept: true,
                },
            });
        }
    }
    layers
}

/// The most shape pictures kept at once.
const MAX_KEPT_SHAPES: usize = 64;

/// The most picture pixels a shape is drawn with per shown pixel.
const MAX_SHAPE_DENSITY: f64 = 8.0;

fn raster_size(width: f64, height: f64) -> (u32, u32) {
    let scale = (MAX_RASTER_EDGE / width.max(height)).min(1.0);
    let edge = |value: f64| (value * scale).round().clamp(1.0, MAX_RASTER_EDGE) as u32;
    (edge(width), edge(height))
}

/// Visible Lottie stickers covering `time`, each at its own frame.
pub(crate) fn lottie_layers(
    draft: &Draft,
    width: u32,
    height: u32,
    time: Micros,
    clips: &mut OpenClips,
) -> Vec<LottieLayer> {
    let mut layers = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible || segment.source_range.is_none() {
                continue;
            }
            let start = segment.target.start.0;
            let Some(end) = start.checked_add(segment.target.duration.0) else {
                continue;
            };
            if time.0 < start || time.0 >= end {
                continue;
            }
            let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            if !is_lottie_path(&video.path) {
                continue;
            }
            let Some((clip, uniform)) = cutlass_model::animated_at(draft, segment, time) else {
                continue;
            };
            let Some(transform) = picture_transform(&clip, &uniform, width, height) else {
                continue;
            };
            if transform.omitted() {
                continue;
            }
            let Some(animation) = clips.lottie(&video.path) else {
                continue;
            };
            let Some(own) = cutlass_engine::placement_of(draft, segment)
                .ok()
                .and_then(|map| cutlass_engine::map_timeline(&map, time).ok())
            else {
                continue;
            };
            let frame = animation.frame_at(own.0 as f64 / 1_000_000.0);
            let (raster_w, raster_h) = raster_size(animation.width(), animation.height());
            let Ok(scene) = animation.scene(frame, raster_w, raster_h) else {
                continue;
            };
            layers.push(LottieLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                placed: PlacedLottie {
                    scene: Arc::new(scene),
                    width,
                    height,
                    transform,
                    raster: OnceCell::new(),
                    kept: false,
                },
            });
        }
    }
    layers
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_engine::ReaderFactory;
    use cutlass_model::{MaterialVideo, Segment, TimeRange, Track, TrackType};
    use cutlass_renderer_software::SoftwareDevice;

    use super::raster_size;
    use super::*;
    use crate::{Composed, Snapshot, frame_in, layers_in};

    const SQUARE: &str = include_str!("../../lottie/tests/fixtures/square.json");

    /// A folder of its own for one test.
    fn folder(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("cutlass-eval-lottie-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    /// A 100 by 100 draft with the square sticker on a sticker row from 1 s
    /// for 3 s.
    fn sticker_draft(path: &Path, speed: f64) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 100;
        draft.canvas_config.height = 100;
        let mut video = MaterialVideo::new("sticker");
        video.kind = "sticker".to_owned();
        video.path = path.display().to_string();
        video.width = 100;
        video.height = 100;
        draft.materials.videos.push(video);
        let mut row = Track::main("stickers");
        row.kind = TrackType::Sticker;
        let mut segment = Segment::new("sticker", "sticker", TimeRange::new(1_000_000, 3_000_000));
        segment.source_range = Some(TimeRange::new(0, (3_000_000.0 * speed) as i64));
        segment.speed = speed;
        row.segments.push(segment);
        draft.tracks.push(row);
        draft
    }

    fn red_at(rgba: &[u8], x: usize, y: usize) -> bool {
        let at = (y * 100 + x) * 4;
        rgba[at..at + 4] == [255, 0, 0, 255]
    }

    fn cpu(draft: &Draft, time: i64, clips: &mut OpenClips, revision: u64) -> Vec<u8> {
        cpu_in(draft, time, clips, revision, None)
    }

    fn cpu_in(
        draft: &Draft,
        time: i64,
        clips: &mut OpenClips,
        revision: u64,
        root: Option<&Path>,
    ) -> Vec<u8> {
        let media = ReaderFactory::new();
        let mut device = SoftwareDevice::new();
        frame_in(
            &Snapshot::new(draft),
            Micros(time),
            &mut device,
            &media,
            clips,
            revision,
            root,
        )
        .unwrap()
        .rgba
    }

    #[test]
    fn a_shape_draws_at_its_own_size_where_its_clip_puts_it() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 100;
        draft.canvas_config.height = 100;
        let mut square = cutlass_model::MaterialShape::preset(
            "sq".into(),
            cutlass_model::ShapeKind::Rectangle,
            20.0,
            20.0,
        );
        square.color = "#ff0000".to_owned();
        draft.materials.shapes.push(square);
        let mut row = Track::main("stickers");
        row.kind = TrackType::Sticker;
        row.segments
            .push(Segment::new("shape", "sq", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(row);
        let mut clips = OpenClips::new();
        // A 20 pixel square in the middle of the 100 pixel canvas.
        let picture = cpu(&draft, 0, &mut clips, 0);
        assert!(red_at(&picture, 50, 50));
        assert!(red_at(&picture, 41, 59));
        assert!(!red_at(&picture, 38, 50));
        assert!(!red_at(&picture, 50, 62));
        // Doubled by the clip's scale, and still sharp at its edge.
        draft.tracks[1].segments[0].uniform_scale.value = 2.0;
        let picture = cpu(&draft, 0, &mut clips, 1);
        assert!(red_at(&picture, 31, 50));
        assert!(red_at(&picture, 69, 69));
        assert!(!red_at(&picture, 28, 50));
        // After the clip: nothing.
        assert!(
            cpu(&draft, 1_000_000, &mut clips, 2)
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn a_shape_is_drawn_once_and_again_only_when_it_changes() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 100;
        draft.canvas_config.height = 100;
        draft
            .materials
            .shapes
            .push(cutlass_model::MaterialShape::preset(
                "sq".into(),
                cutlass_model::ShapeKind::Rectangle,
                20.0,
                20.0,
            ));
        let mut row = Track::main("stickers");
        row.kind = TrackType::Sticker;
        row.segments
            .push(Segment::new("shape", "sq", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(row);
        let mut kept = HashMap::new();
        let shown = |draft: &Draft, kept: &mut HashMap<SegmentId, KeptShape>, time: i64| {
            let layers = shape_layers(draft, 100, 100, Micros(time), kept);
            let placed = &layers[0].placed;
            assert!(placed.kept);
            placed.raster().unwrap().clone()
        };
        let first = shown(&draft, &mut kept, 0);
        assert_eq!(kept.len(), 1);
        // Mark the kept picture: a later frame and a move show the mark,
        // so neither drew the shape again.
        let id = SegmentId::new("shape");
        let marked =
            CpuBytes::new(first.width(), first.height(), vec![7; first.rgba().len()]).unwrap();
        kept.get_mut(&id).unwrap().raster = marked.clone();
        assert_eq!(shown(&draft, &mut kept, 500_000), marked);
        draft.tracks[1].segments[0].clip.transform.x = 0.5;
        assert_eq!(shown(&draft, &mut kept, 0), marked);
        // A new fill draws it again.
        draft.materials.shapes[0].color = "#ff0000".to_owned();
        assert_ne!(shown(&draft, &mut kept, 0), marked);
        // So does a scale past the next quarter step, at the denser size.
        kept.get_mut(&id).unwrap().raster = marked.clone();
        draft.tracks[1].segments[0].uniform_scale.value = 2.0;
        let dense = shown(&draft, &mut kept, 0);
        assert!(dense.width() > first.width());
    }

    #[test]
    fn a_keyed_shape_setting_draws_as_keyed_at_each_frame() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 100;
        draft.canvas_config.height = 100;
        draft
            .materials
            .shapes
            .push(cutlass_model::MaterialShape::preset(
                "sq".into(),
                cutlass_model::ShapeKind::Rectangle,
                20.0,
                20.0,
            ));
        let mut row = Track::main("stickers");
        row.kind = TrackType::Sticker;
        let mut segment = Segment::new("shape", "sq", TimeRange::new(0, 1_000_000));
        let width = cutlass_model::ShapeStyleKey::Size.names()[0];
        cutlass_model::put_key(&mut segment, width, Micros(0), 20.0);
        cutlass_model::put_key(&mut segment, width, Micros(1_000_000), 60.0);
        row.segments.push(segment);
        draft.tracks.push(row);
        let mut kept = HashMap::new();
        let mut drawn = |time: i64| {
            let layers = shape_layers(&draft, 100, 100, Micros(time), &mut kept);
            layers[0].placed.raster().unwrap().clone()
        };
        let start = drawn(0);
        let middle = drawn(500_000);
        // Wider halfway along, and the height is left as it was.
        assert!(middle.width() > start.width());
        assert_eq!(middle.height(), start.height());
        // The same frame again is the kept picture.
        assert_eq!(drawn(500_000), middle);
    }

    #[test]
    fn a_lottie_sticker_plays_from_its_clip_start_and_loops() {
        let folder = folder("plays");
        let path = folder.join("square.json");
        std::fs::write(&path, SQUARE).unwrap();
        let draft = sticker_draft(&path, 1.0);
        let mut clips = OpenClips::new();
        // Before the clip: nothing.
        assert!(
            cpu(&draft, 500_000, &mut clips, 0)
                .iter()
                .all(|byte| *byte == 0)
        );
        // The clip's first frame: the square at x 20 to 40.
        let first = cpu(&draft, 1_000_000, &mut clips, 0);
        assert!(red_at(&first, 30, 50));
        assert!(!red_at(&first, 50, 50));
        // Ten frames in (a third of a second): x 40 to 60.
        let later = cpu(&draft, 1_333_334, &mut clips, 0);
        assert!(red_at(&later, 50, 50));
        assert!(!red_at(&later, 30, 50));
        // One play is a second; the second play starts over.
        let again = cpu(&draft, 2_000_000, &mut clips, 0);
        assert_eq!(again, first);
        // After the clip: nothing.
        assert!(
            cpu(&draft, 4_000_000, &mut clips, 0)
                .iter()
                .all(|byte| *byte == 0)
        );
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn the_clip_speed_runs_the_animation_faster() {
        let folder = folder("speed");
        let path = folder.join("square.json");
        std::fs::write(&path, SQUARE).unwrap();
        let draft = sticker_draft(&path, 2.0);
        let mut clips = OpenClips::new();
        // A sixth of a second at twice the speed is ten frames in.
        let later = cpu(&draft, 1_166_667, &mut clips, 0);
        assert!(red_at(&later, 50, 50));
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn the_gpu_path_hands_over_the_geometry_placed_like_the_raster() {
        let folder = folder("gpu");
        let path = folder.join("square.json");
        std::fs::write(&path, SQUARE).unwrap();
        let draft = sticker_draft(&path, 1.0);
        let media = ReaderFactory::new();
        let mut device = SoftwareDevice::new();
        let mut clips = OpenClips::new();
        let composed = layers_in(
            &Snapshot::new(&draft),
            Micros(1_000_000),
            &mut device,
            &media,
            &mut clips,
            0,
            None,
        )
        .unwrap();
        let Composed::Layers(layers) = composed else {
            panic!("composed on the CPU");
        };
        let vectors: Vec<_> = layers
            .layers
            .iter()
            .filter_map(|layer| match &layer.source {
                GpuSource::Vector(scene) => Some((scene, layer.placement)),
                _ => None,
            })
            .collect();
        assert_eq!(vectors.len(), 1);
        let (scene, placement) = vectors[0];
        assert_eq!((scene.width, scene.height), (100, 100));
        assert_eq!(placement.picture, (100, 100));
        assert_eq!(placement.to_picture, Affine::IDENTITY);
        assert!((placement.opacity - 1.0).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn a_broken_file_draws_nothing_and_is_read_once_a_revision() {
        let folder = folder("broken");
        let path = folder.join("square.json");
        std::fs::write(&path, SQUARE).unwrap();
        let draft = sticker_draft(&path, 1.0);
        let mut clips = OpenClips::new();
        let first = cpu(&draft, 1_000_000, &mut clips, 0);
        assert!(red_at(&first, 30, 50));
        // The same revision keeps the animation it read.
        std::fs::write(&path, b"{ not json").unwrap();
        assert_eq!(cpu(&draft, 1_000_000, &mut clips, 0), first);
        // A new revision reads the file again, and a broken file is left out.
        assert!(
            cpu(&draft, 1_000_000, &mut clips, 1)
                .iter()
                .all(|byte| *byte == 0)
        );
        // A missing file and a path that leaves the project are left out too.
        let missing = sticker_draft(&folder.join("gone.json"), 1.0);
        assert!(
            cpu(&missing, 1_000_000, &mut clips, 2)
                .iter()
                .all(|byte| *byte == 0)
        );
        let inner = folder.join("project");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(&path, SQUARE).unwrap();
        let outside = sticker_draft(Path::new("../square.json"), 1.0);
        let drawn = cpu_in(&outside, 1_000_000, &mut clips, 3, Some(&inner));
        assert!(drawn.iter().all(|byte| *byte == 0));
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn a_relative_path_is_read_from_the_project_folder() {
        let folder = folder("relative");
        std::fs::write(folder.join("square.json"), SQUARE).unwrap();
        let draft = sticker_draft(Path::new("square.json"), 1.0);
        let mut clips = OpenClips::new();
        let drawn = cpu_in(&draft, 1_000_000, &mut clips, 0, Some(&folder));
        assert!(red_at(&drawn, 30, 50));
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn a_large_animation_is_drawn_at_most_the_raster_edge() {
        assert_eq!(raster_size(512.0, 256.0), (512, 256));
        assert_eq!(raster_size(8_192.0, 4_096.0), (2_048, 1_024));
        assert_eq!(raster_size(0.2, 0.2), (1, 1));
    }
}
