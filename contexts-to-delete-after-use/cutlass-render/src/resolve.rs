//! The pure timeline → [`Scene`] resolver.
//!
//! Given a [`Project`] and a timeline instant, [`resolve`] walks the visual
//! track stack bottom-to-top, finds the clip active on each lane, and turns it
//! into a placed [`SceneLayer`]: canvas geometry, transform, crop/mirror, and a
//! classified pixel source. It decodes nothing and touches no GPU, so the
//! geometry is deterministic and unit-testable on any platform.
//!
//! ## Coverage (v1)
//!
//! - **Media**: video sources (aspect-fit into the canvas, then scaled).
//! - **Generators**: text, solid fills, and rectangles.
//! - **Deferred**: still images (no still decoder yet), ellipses, stickers,
//!   effects, filters, and adjustment layers are skipped (they produce no
//!   layer) rather than rendered wrong.

use cutlass_core::{RationalTime, resample};
use cutlass_models::{
    ClipSource, Generator, MediaKind, Project, Shape, TextAlignH, TextStyle as ModelTextStyle,
};
use cutlass_text::{FontFamily, TextAlign, TextStyle};

use crate::scene::{LayerSource, Scene, SceneLayer, SizeSpec};

/// Vertical reference height that a generator's reference-pixel sizes (text
/// `size`, shape `width`/`height`) are authored against. Matches the model's
/// `canvas_height / 1080` convention.
const REFERENCE_HEIGHT: f32 = 1080.0;

/// Fallback canvas size when `Auto` aspect can't find any video media.
const DEFAULT_CANVAS: (u32, u32) = (1920, 1080);

/// Resolve `project` at timeline instant `t` into a [`Scene`].
///
/// `t` is interpreted at the timeline frame rate (it is resampled to it first),
/// so callers may pass a tick at any rate.
pub fn resolve(project: &Project, t: RationalTime) -> Result<Scene, cutlass_models::ModelError> {
    let timeline = project.timeline();
    let rate = timeline.frame_rate;
    let t = resample(t, rate);

    let (width, height) = canvas_size(project);
    let bg = timeline.canvas().background;
    let mut scene = Scene::empty(width, height, [bg[0], bg[1], bg[2], 255]);

    let cw = width as f32;
    let ch = height as f32;

    for track in timeline.tracks_ordered() {
        if !track.kind.is_visual() || !track.enabled {
            continue;
        }
        let Some(clip) = track.clip_at(t)? else {
            continue;
        };
        if let Some(layer) = resolve_clip(project, clip, t, cw, ch)? {
            scene.layers.push(layer);
        }
    }

    Ok(scene)
}

/// Canvas pixel size for `project`: fixed presets resolve to a 1080-baseline
/// box on the longer side; `Auto` follows the largest video media (falling back
/// to 1920×1080 when there is none).
pub fn canvas_size(project: &Project) -> (u32, u32) {
    match project.timeline().canvas().aspect.ratio() {
        Some((rw, rh)) => ratio_to_pixels(rw, rh),
        None => auto_canvas_size(project),
    }
}

/// Largest visible dimension box for a `w:h` ratio, even-rounded for encoders.
fn ratio_to_pixels(rw: u32, rh: u32) -> (u32, u32) {
    const BASE: f32 = REFERENCE_HEIGHT;
    let (rw, rh) = (rw as f32, rh as f32);
    let (w, h) = if rw >= rh {
        ((BASE * rw / rh).round(), BASE)
    } else {
        (BASE, (BASE * rh / rw).round())
    };
    (even(w as u32), even(h as u32))
}

/// The largest video media used anywhere on the timeline, or the default.
fn auto_canvas_size(project: &Project) -> (u32, u32) {
    let mut best: Option<(u32, u32)> = None;
    for track in project.timeline().tracks_ordered() {
        if !track.kind.is_visual() {
            continue;
        }
        for clip in track.clips() {
            let Some(id) = clip.media() else { continue };
            let Some(media) = project.media(id) else {
                continue;
            };
            if media.kind() != MediaKind::Video {
                continue;
            }
            let area = u64::from(media.width) * u64::from(media.height);
            if best.is_none_or(|(bw, bh)| area > u64::from(bw) * u64::from(bh)) {
                best = Some((media.width, media.height));
            }
        }
    }
    best.map_or(DEFAULT_CANVAS, |(w, h)| (even(w), even(h)))
}

/// Round `v` down to the nearest even value (≥ 2): H.264 needs even dimensions.
fn even(v: u32) -> u32 {
    (v & !1).max(2)
}

fn resolve_clip(
    project: &Project,
    clip: &cutlass_models::Clip,
    t: RationalTime,
    cw: f32,
    ch: f32,
) -> Result<Option<SceneLayer>, cutlass_models::ModelError> {
    // Clip-relative tick at the timeline rate (both `t` and the clip start are
    // expressed at it), which is what animated transforms key against.
    let local_tick = t.value - clip.timeline.start.value;
    let xf = clip.transform.sample(local_tick);

    // `position` is the anchor's offset from the canvas center, as a fraction of
    // canvas width/height. With the default center anchor (the common case) this
    // is the placed quad's center; non-center anchors are a follow-up.
    let center = [cw * (0.5 + xf.position[0]), ch * (0.5 + xf.position[1])];
    let rotation = xf.rotation.to_radians();
    let opacity = xf.opacity.clamp(0.0, 1.0);
    let uv = crop_flip_uv(clip);

    match &clip.content {
        ClipSource::Media { media, .. } => {
            let Some(src) = project.media(*media) else {
                return Ok(None);
            };
            if src.kind() != MediaKind::Video {
                return Ok(None);
            }
            let Some(source_time) = clip.source_time_at(t)? else {
                return Ok(None);
            };
            let fit = fit_scale(src.width as f32, src.height as f32, cw, ch);
            let size = SizeSpec::Fixed([
                src.width as f32 * fit * xf.scale,
                src.height as f32 * fit * xf.scale,
            ]);
            Ok(Some(SceneLayer {
                source: LayerSource::Media {
                    media: *media,
                    source_time,
                },
                center,
                size,
                rotation,
                opacity,
                uv,
            }))
        }
        ClipSource::Generated(generator) => Ok(resolve_generator(
            generator, center, rotation, opacity, uv, cw, ch, xf.scale,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_generator(
    generator: &Generator,
    center: [f32; 2],
    rotation: f32,
    opacity: f32,
    uv: [f32; 4],
    cw: f32,
    ch: f32,
    scale: f32,
) -> Option<SceneLayer> {
    let ref_scale = ch / REFERENCE_HEIGHT;
    match generator {
        Generator::Text { content, style } => {
            let text = style.case.apply(content);
            if text.trim().is_empty() {
                return None;
            }
            Some(SceneLayer {
                source: LayerSource::Text {
                    content: text,
                    style: map_text_style(style, cw, ch),
                },
                center,
                size: SizeSpec::BitmapScaled(scale),
                rotation,
                opacity,
                uv,
            })
        }
        Generator::SolidColor { rgba } => Some(SceneLayer {
            source: LayerSource::Solid(*rgba),
            center,
            size: SizeSpec::Fixed([cw * scale, ch * scale]),
            rotation,
            opacity,
            uv,
        }),
        Generator::Shape {
            shape: Shape::Rectangle,
            rgba,
            width,
            height,
        } => Some(SceneLayer {
            source: LayerSource::Solid(*rgba),
            center,
            size: SizeSpec::Fixed([width * ref_scale * scale, height * ref_scale * scale]),
            rotation,
            opacity,
            uv,
        }),
        // Ellipses need a vector rasterizer; stickers/effects/filters/adjustment
        // layers are not composited yet. Skip rather than draw something wrong.
        Generator::Shape { .. }
        | Generator::Sticker
        | Generator::Effect
        | Generator::Filter
        | Generator::Adjustment => None,
    }
}

/// Map a model [`ModelTextStyle`] onto a [`cutlass_text`] render style.
///
/// Lossy in v1: stroke, background, shadow, and bold/italic are not rendered
/// yet. Font size is scaled from reference (1080-tall) pixels to the canvas.
///
/// The rasterized bitmap is kept tight to the glyphs (no canvas-width wrap):
/// the layer's placement centers it on the canvas, so passing the canvas width
/// as a wrap constraint here would only double-center the run. Multi-line wrap
/// within a fixed width is a follow-up.
fn map_text_style(style: &ModelTextStyle, _cw: f32, ch: f32) -> TextStyle {
    let font_size = style.size * (ch / REFERENCE_HEIGHT);
    let family = if style.font.is_empty() {
        FontFamily::SansSerif
    } else {
        FontFamily::Named(style.font.clone())
    };
    let align = match style.align_h {
        TextAlignH::Left => TextAlign::Left,
        TextAlignH::Center => TextAlign::Center,
        TextAlignH::Right => TextAlign::Right,
    };
    TextStyle::new(font_size)
        .with_color(style.fill)
        .with_family(family)
        .with_align(align)
        .with_line_height(font_size * style.line_spacing)
}

/// UV rect from a clip's crop, with axes reversed for mirror flags.
fn crop_flip_uv(clip: &cutlass_models::Clip) -> [f32; 4] {
    let c = clip.crop;
    let (mut u0, mut u1) = (c.x, c.x + c.w);
    let (mut v0, mut v1) = (c.y, c.y + c.h);
    if clip.flip_h {
        core::mem::swap(&mut u0, &mut u1);
    }
    if clip.flip_v {
        core::mem::swap(&mut v0, &mut v1);
    }
    [u0, v0, u1, v1]
}

/// Uniform "contain" scale fitting `nw`×`nh` content inside a `cw`×`ch` canvas.
fn fit_scale(nw: f32, nh: f32, cw: f32, ch: f32) -> f32 {
    if nw <= 0.0 || nh <= 0.0 {
        return 1.0;
    }
    (cw / nw).min(ch / nh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{LayerSource, SizeSpec};
    use cutlass_models::{
        CanvasAspect, CanvasSettings, ClipTransform, CropRect, Generator, MediaSource, Project,
        Rational, RationalTime, Shape, TextStyle as ModelTextStyle, TimeRange, TrackKind,
    };

    const FPS_24: Rational = Rational::FPS_24;

    fn rt(value: i64) -> RationalTime {
        RationalTime::new(value, FPS_24)
    }

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, FPS_24)
    }

    fn video(width: u32, height: u32) -> MediaSource {
        MediaSource::new("/tmp/v.mp4", width, height, FPS_24, 600, false)
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "expected ~{b}, got {a}");
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
    }

    #[test]
    fn empty_project_uses_default_canvas_and_has_no_layers() {
        let project = Project::new("p", FPS_24);
        let scene = resolve(&project, rt(0)).unwrap();
        assert_eq!((scene.width, scene.height), (1920, 1080));
        assert!(scene.layers.is_empty());
    }

    #[test]
    fn single_video_clip_is_centered_and_aspect_fit() {
        let mut project = Project::new("p", FPS_24);
        let media = project.add_media(video(1920, 1080));
        let track = project.add_track(TrackKind::Video, "V1");
        project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        assert_eq!((scene.width, scene.height), (1920, 1080));
        assert_eq!(scene.layers.len(), 1);

        let layer = &scene.layers[0];
        approx2(layer.center, [960.0, 540.0]);
        match layer.size {
            SizeSpec::Fixed(size) => approx2(size, [1920.0, 1080.0]),
            other => panic!("expected fixed size, got {other:?}"),
        }
        approx(layer.opacity, 1.0);
        match &layer.source {
            LayerSource::Media { media: m, source_time } => {
                assert_eq!(*m, media);
                assert_eq!(source_time.value, 5);
            }
            other => panic!("expected media source, got {other:?}"),
        }
    }

    #[test]
    fn clip_inactive_at_time_produces_no_layer() {
        let mut project = Project::new("p", FPS_24);
        let media = project.add_media(video(1920, 1080));
        let track = project.add_track(TrackKind::Video, "V1");
        project.add_clip(track, media, tr(0, 10), rt(0)).unwrap();

        let scene = resolve(&project, rt(50)).unwrap();
        assert!(scene.layers.is_empty());
    }

    #[test]
    fn tracks_stack_bottom_to_top() {
        let mut project = Project::new("p", FPS_24);
        let bottom = project.add_track(TrackKind::Sticker, "S1");
        let top = project.add_track(TrackKind::Sticker, "S2");
        project
            .add_generated(bottom, Generator::SolidColor { rgba: [255, 0, 0, 255] }, tr(0, 100))
            .unwrap();
        project
            .add_generated(top, Generator::SolidColor { rgba: [0, 0, 255, 255] }, tr(0, 100))
            .unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        assert_eq!(scene.layers.len(), 2);
        assert_eq!(scene.layers[0].source, LayerSource::Solid([255, 0, 0, 255]));
        assert_eq!(scene.layers[1].source, LayerSource::Solid([0, 0, 255, 255]));
    }

    #[test]
    fn text_generator_maps_style_and_defers_size() {
        let mut project = Project::new("p", FPS_24);
        let track = project.add_track(TrackKind::Text, "T1");
        let style = ModelTextStyle {
            size: 90.0,
            fill: [255, 0, 0, 255],
            ..ModelTextStyle::default()
        };
        project
            .add_generated(
                track,
                Generator::Text { content: "Hi".into(), style },
                tr(0, 100),
            )
            .unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        assert_eq!((scene.width, scene.height), (1920, 1080));
        assert_eq!(scene.layers.len(), 1);
        let layer = &scene.layers[0];
        approx2(layer.center, [960.0, 540.0]);
        assert_eq!(layer.size, SizeSpec::BitmapScaled(1.0));
        match &layer.source {
            LayerSource::Text { content, style } => {
                assert_eq!(content, "Hi");
                approx(style.font_size, 90.0);
                assert_eq!(style.color, [255, 0, 0, 255]);
            }
            other => panic!("expected text source, got {other:?}"),
        }
    }

    #[test]
    fn solid_generator_fills_canvas() {
        let mut project = Project::new("p", FPS_24);
        let track = project.add_track(TrackKind::Sticker, "S1");
        project
            .add_generated(track, Generator::SolidColor { rgba: [10, 20, 30, 255] }, tr(0, 100))
            .unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        let layer = &scene.layers[0];
        match layer.size {
            SizeSpec::Fixed(size) => approx2(size, [1920.0, 1080.0]),
            other => panic!("expected fixed size, got {other:?}"),
        }
    }

    #[test]
    fn crop_and_horizontal_flip_set_uv() {
        let mut project = Project::new("p", FPS_24);
        let media = project.add_media(video(1920, 1080));
        let track = project.add_track(TrackKind::Video, "V1");
        let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
        project
            .set_clip_crop(
                clip,
                CropRect {
                    x: 0.25,
                    y: 0.1,
                    w: 0.5,
                    h: 0.8,
                },
                true,
                false,
            )
            .unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        let uv = scene.layers[0].uv;
        // base [0.25, 0.1, 0.75, 0.9]; flip_h swaps the u axis.
        approx(uv[0], 0.75);
        approx(uv[1], 0.1);
        approx(uv[2], 0.25);
        approx(uv[3], 0.9);
    }

    #[test]
    fn fixed_aspect_presets_resolve_to_1080_baseline() {
        let mut project = Project::new("p", FPS_24);
        project.timeline_mut().set_canvas(CanvasSettings {
            aspect: CanvasAspect::Tall9x16,
            background: [0, 0, 0],
        });
        assert_eq!(canvas_size(&project), (1080, 1920));

        project.timeline_mut().set_canvas(CanvasSettings {
            aspect: CanvasAspect::Wide16x9,
            background: [0, 0, 0],
        });
        assert_eq!(canvas_size(&project), (1920, 1080));

        project.timeline_mut().set_canvas(CanvasSettings {
            aspect: CanvasAspect::Square1x1,
            background: [0, 0, 0],
        });
        assert_eq!(canvas_size(&project), (1080, 1080));
    }

    #[test]
    fn transform_offsets_center_and_scales_size() {
        let mut project = Project::new("p", FPS_24);
        let media = project.add_media(video(1920, 1080));
        let track = project.add_track(TrackKind::Video, "V1");
        let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
        project
            .set_transform(
                clip,
                ClipTransform {
                    position: [0.25, -0.1],
                    scale: 2.0,
                    opacity: 0.5,
                    ..ClipTransform::IDENTITY
                },
                None,
            )
            .unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        let layer = &scene.layers[0];
        approx2(layer.center, [1920.0 * 0.75, 1080.0 * 0.4]);
        match layer.size {
            SizeSpec::Fixed(size) => approx2(size, [3840.0, 2160.0]),
            other => panic!("expected fixed size, got {other:?}"),
        }
        approx(layer.opacity, 0.5);
    }

    #[test]
    fn audio_track_is_skipped() {
        let mut project = Project::new("p", FPS_24);
        let video_track = project.add_track(TrackKind::Video, "V1");
        let media = project.add_media(video(1920, 1080));
        project.add_clip(video_track, media, tr(0, 100), rt(0)).unwrap();

        let audio_track = project.add_track(TrackKind::Audio, "A1");
        let song = project.add_media(MediaSource::new("/tmp/a.mp3", 0, 0, FPS_24, 600, true));
        project.add_clip(audio_track, song, tr(0, 100), rt(0)).unwrap();

        let scene = resolve(&project, rt(5)).unwrap();
        assert_eq!(scene.layers.len(), 1);
        assert!(matches!(scene.layers[0].source, LayerSource::Media { .. }));
    }

    #[test]
    fn rectangle_renders_but_ellipse_is_skipped() {
        let mut project = Project::new("p", FPS_24);
        let track = project.add_track(TrackKind::Sticker, "S1");
        project
            .add_generated(track, Generator::shape(Shape::Rectangle, [255, 255, 255, 255]), tr(0, 100))
            .unwrap();
        let scene = resolve(&project, rt(5)).unwrap();
        assert_eq!(scene.layers.len(), 1);
        assert!(matches!(scene.layers[0].source, LayerSource::Solid(_)));

        let mut project = Project::new("p", FPS_24);
        let track = project.add_track(TrackKind::Sticker, "S1");
        project
            .add_generated(track, Generator::shape(Shape::Ellipse, [255, 255, 255, 255]), tr(0, 100))
            .unwrap();
        let scene = resolve(&project, rt(5)).unwrap();
        assert!(scene.layers.is_empty());
    }
}
