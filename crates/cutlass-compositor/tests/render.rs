//! End-to-end compositor checks: build a scene, render it on a real (headless)
//! GPU, and assert the read-back pixels.
//!
//! Color tests compare against a CPU reimplementation of the shader math within
//! a small tolerance (f32 + 8-bit unorm rounding differs slightly across
//! drivers, including software adapters), rather than pinning exact goldens.
//! If no GPU adapter is available the test skips rather than fails.

use cutlass_compositor::{
    ColorGrade, CompositeLayer, Compositor, CompositorConfig, CompositorLayer, GpuContext,
    LayerChromaKey, LayerEffects, LayerMask, LayerPlacement, PassInstance, RgbaImage, mask_kind,
};
use cutlass_core::{
    ColorRange, ColorSpace, CpuImage, FrameData, MatrixCoefficients, PixelFormat, Plane, Rational,
    RationalTime, Rect, Rotation, VideoFrame,
};

/// Try to bring up a headless GPU; `None` (skip) if the host has no adapter.
fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping compositor test: no GPU adapter ({e})");
            None
        }
    }
}

macro_rules! gpu_or_skip {
    () => {
        match try_gpu() {
            Some(g) => g,
            None => return,
        }
    };
}

fn frame(format: PixelFormat, color: ColorSpace, w: u32, h: u32, planes: Vec<Plane>) -> VideoFrame {
    VideoFrame::new(
        RationalTime::new(0, Rational::FPS_30),
        format,
        color,
        (w, h),
        Rect::from_size(w, h),
        Rotation::None,
        FrameData::Cpu(CpuImage::new(planes)),
    )
}

/// A `w×h` NV12 frame with every pixel set to the same (Y, U, V).
fn nv12_uniform(w: u32, h: u32, y: u8, u: u8, v: u8, color: ColorSpace) -> VideoFrame {
    let yp = Plane::new(vec![y; (w * h) as usize], w as usize, h as usize);
    let (cw, ch) = (w / 2, h / 2);
    let mut uv = Vec::with_capacity((cw * ch * 2) as usize);
    for _ in 0..(cw * ch) {
        uv.push(u);
        uv.push(v);
    }
    let uvp = Plane::new(uv, (cw * 2) as usize, ch as usize);
    frame(PixelFormat::Nv12, color, w, h, vec![yp, uvp])
}

/// A `w×h` I420 frame with every pixel set to the same (Y, U, V).
fn i420_uniform(w: u32, h: u32, y: u8, u: u8, v: u8, color: ColorSpace) -> VideoFrame {
    let yp = Plane::new(vec![y; (w * h) as usize], w as usize, h as usize);
    let (cw, ch) = (w / 2, h / 2);
    let up = Plane::new(vec![u; (cw * ch) as usize], cw as usize, ch as usize);
    let vp = Plane::new(vec![v; (cw * ch) as usize], cw as usize, ch as usize);
    frame(PixelFormat::Yuv420p, color, w, h, vec![yp, up, vp])
}

/// CPU mirror of `yuv.wgsl`'s `yuv_to_rgb`, for tolerance comparisons.
fn yuv_ref(y: u8, cb: u8, cr: u8, color: ColorSpace) -> [u8; 3] {
    let (kr, kb): (f32, f32) = match color.resolved().matrix {
        MatrixCoefficients::Bt601 => (0.299, 0.114),
        MatrixCoefficients::Bt2020Ncl => (0.2627, 0.0593),
        _ => (0.2126, 0.0722),
    };
    let (ys, cbs, crs) = (
        f32::from(y) / 255.0,
        f32::from(cb) / 255.0,
        f32::from(cr) / 255.0,
    );
    let (yv, cbv, crv) = if color.range.is_full() {
        (ys, cbs - 128.0 / 255.0, crs - 128.0 / 255.0)
    } else {
        (
            (ys - 16.0 / 255.0) * (255.0 / 219.0),
            (cbs - 128.0 / 255.0) * (255.0 / 224.0),
            (crs - 128.0 / 255.0) * (255.0 / 224.0),
        )
    };
    let kg = 1.0 - kr - kb;
    let r = yv + 2.0 * (1.0 - kr) * crv;
    let b = yv + 2.0 * (1.0 - kb) * cbv;
    let g = yv - (2.0 * (1.0 - kr) * kr / kg) * crv - (2.0 * (1.0 - kb) * kb / kg) * cbv;
    [quant(r), quant(g), quant(b)]
}

fn quant(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// CPU mirror of `grade.wgsl`'s `apply_grade` — the canonical version lives
/// on [`ColorGrade::apply`] so the LUT baker shares it.
fn grade_ref(rgb: [f32; 3], grade: ColorGrade) -> [f32; 3] {
    grade.apply(rgb)
}

fn grade_ref_u8(rgba: [u8; 4], grade: ColorGrade) -> [u8; 4] {
    let rgb = grade_ref(
        [
            f32::from(rgba[0]) / 255.0,
            f32::from(rgba[1]) / 255.0,
            f32::from(rgba[2]) / 255.0,
        ],
        grade,
    );
    [quant(rgb[0]), quant(rgb[1]), quant(rgb[2]), rgba[3]]
}

#[track_caller]
fn assert_px(img: &RgbaImage, x: u32, y: u32, expect: [u8; 4], tol: i32) {
    let got = img.pixel(x, y);
    for ch in 0..4 {
        let d = i32::from(got[ch]) - i32::from(expect[ch]);
        assert!(
            d.abs() <= tol,
            "pixel({x},{y}) = {got:?}, expected ~{expect:?} (channel {ch} off by {d}, tol {tol})"
        );
    }
}

/// Render a single full-canvas frame layer and read it back.
fn render_frame(
    gpu: &GpuContext,
    comp: &mut Compositor,
    w: u32,
    h: u32,
    f: &VideoFrame,
) -> RgbaImage {
    let config = CompositorConfig::new(w, h);
    let layer = CompositeLayer::frame(f, LayerPlacement::full_canvas(&config));
    comp.render(gpu, &config, &[layer]).expect("render")
}

#[test]
fn solid_fill_covers_canvas() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(32, 32); // default black background
    let red = CompositeLayer::solid([255, 0, 0, 255], LayerPlacement::full_canvas(&config));
    let img = comp.render(&gpu, &config, &[red]).expect("render");

    assert_px(&img, 0, 0, [255, 0, 0, 255], 1);
    assert_px(&img, 16, 16, [255, 0, 0, 255], 1);
    assert_px(&img, 31, 31, [255, 0, 0, 255], 1);
}

#[test]
fn empty_scene_is_background() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(8, 8).with_background([10, 20, 30, 255]);
    let img = comp.render(&gpu, &config, &[]).expect("render");
    assert_px(&img, 4, 4, [10, 20, 30, 255], 1);
}

#[test]
fn repeated_renders_reuse_the_cached_target() {
    // `render` caches the canvas texture + readback buffer across calls (the
    // preview renders one size every frame). Same-size renders must stay
    // independent — the pass clears, so no state leaks between frames — and a
    // size change must rebuild the target at the new dimensions.
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);

    let config = CompositorConfig::new(32, 32).with_background([10, 20, 30, 255]);
    let first = comp.render(&gpu, &config, &[]).expect("render 1");
    assert_px(&first, 16, 16, [10, 20, 30, 255], 1);

    // Second render at the same size (cache hit) with different content.
    let red = CompositeLayer::solid([255, 0, 0, 255], LayerPlacement::full_canvas(&config));
    let second = comp.render(&gpu, &config, &[red]).expect("render 2");
    assert_px(&second, 16, 16, [255, 0, 0, 255], 1);

    // Third render back to an empty scene: nothing of the red frame survives.
    let third = comp.render(&gpu, &config, &[]).expect("render 3");
    assert_px(&third, 16, 16, [10, 20, 30, 255], 1);

    // Size change (also exercises a non-256-multiple row pitch): the target
    // is rebuilt and the readback matches the new dimensions.
    let smaller = CompositorConfig::new(17, 9).with_background([1, 2, 3, 255]);
    let resized = comp.render(&gpu, &smaller, &[]).expect("render 4");
    assert_eq!((resized.width, resized.height), (17, 9));
    assert_px(&resized, 16, 8, [1, 2, 3, 255], 1);

    // And back to the original size.
    let again = comp.render(&gpu, &config, &[]).expect("render 5");
    assert_eq!((again.width, again.height), (32, 32));
    assert_px(&again, 31, 31, [10, 20, 30, 255], 1);
}

#[test]
fn solid_placement_lands_in_the_right_quadrant() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(100, 100); // black background

    // A 20×20 red square centered in the top-left quadrant at (25, 25).
    let placement = LayerPlacement {
        center: [25.0, 25.0],
        size: [20.0, 20.0],
        rotation: 0.0,
        opacity: 1.0,
    };
    let layer = CompositeLayer::solid([255, 0, 0, 255], placement);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    assert_px(&img, 25, 25, [255, 0, 0, 255], 1); // inside, top-left
    assert_px(&img, 75, 25, [0, 0, 0, 255], 1); // top-right: background
    assert_px(&img, 25, 75, [0, 0, 0, 255], 1); // bottom-left: background
    assert_px(&img, 75, 75, [0, 0, 0, 255], 1); // bottom-right: background
}

#[test]
fn opacity_blends_over_background() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    // Opaque blue canvas, half-opacity red on top → ~purple (gamma-space blend).
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 255, 255]);
    let mut placement = LayerPlacement::full_canvas(&config);
    placement.opacity = 0.5;
    let layer = CompositeLayer::solid([255, 0, 0, 255], placement);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    assert_px(&img, 8, 8, [128, 0, 128, 255], 2);
}

#[test]
fn nv12_neutral_is_gray() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    // Full-range Y=128, neutral chroma → mid gray, alpha opaque.
    let full_gray = ColorSpace {
        range: ColorRange::Full,
        ..ColorSpace::BT709
    };
    let f = nv12_uniform(16, 16, 128, 128, 128, full_gray);
    let img = render_frame(&gpu, &mut comp, 16, 16, &f);
    assert_px(&img, 8, 8, [128, 128, 128, 255], 2);
}

#[test]
fn nv12_limited_luma_endpoints() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);

    // Limited range: Y=16 is black, Y=235 is white (neutral chroma).
    let black = nv12_uniform(16, 16, 16, 128, 128, ColorSpace::BT709);
    let white = nv12_uniform(16, 16, 235, 128, 128, ColorSpace::BT709);
    assert_px(
        &render_frame(&gpu, &mut comp, 16, 16, &black),
        8,
        8,
        [0, 0, 0, 255],
        2,
    );
    assert_px(
        &render_frame(&gpu, &mut comp, 16, 16, &white),
        8,
        8,
        [255, 255, 255, 255],
        2,
    );
}

#[test]
fn nv12_color_matches_cpu_reference() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);

    // A saturated, non-neutral sample under several color spaces.
    let (y, u, v) = (150u8, 60u8, 200u8);
    let spaces = [
        ColorSpace::BT709,
        ColorSpace::BT601,
        ColorSpace {
            range: ColorRange::Full,
            ..ColorSpace::BT709
        },
        ColorSpace {
            range: ColorRange::Full,
            ..ColorSpace::BT601
        },
    ];
    for color in spaces {
        let f = nv12_uniform(16, 16, y, u, v, color);
        let img = render_frame(&gpu, &mut comp, 16, 16, &f);
        let [r, g, b] = yuv_ref(y, u, v, color);
        assert_px(&img, 8, 8, [r, g, b, 255], 2);
    }
}

#[test]
fn nv12_and_i420_agree() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let (y, u, v) = (90u8, 200u8, 70u8);
    let color = ColorSpace::BT709;

    let from_nv12 = render_frame(
        &gpu,
        &mut comp,
        16,
        16,
        &nv12_uniform(16, 16, y, u, v, color),
    );
    let from_i420 = render_frame(
        &gpu,
        &mut comp,
        16,
        16,
        &i420_uniform(16, 16, y, u, v, color),
    );

    let a = from_nv12.pixel(8, 8);
    assert_px(&from_i420, 8, 8, a, 1);
}

#[test]
fn matrix_choice_actually_changes_output() {
    // Guards against the YUV matrix being hardcoded: BT.601 and BT.709 must
    // produce visibly different RGB for chroma-heavy input.
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let (y, u, v) = (128u8, 128u8, 210u8); // strong Cr, neutral Cb

    let bt709 = render_frame(
        &gpu,
        &mut comp,
        16,
        16,
        &nv12_uniform(16, 16, y, u, v, ColorSpace::BT709),
    );
    let bt601 = render_frame(
        &gpu,
        &mut comp,
        16,
        16,
        &nv12_uniform(16, 16, y, u, v, ColorSpace::BT601),
    );

    let p709 = bt709.pixel(8, 8);
    let p601 = bt601.pixel(8, 8);
    let max_diff = (0..3)
        .map(|c| (i32::from(p709[c]) - i32::from(p601[c])).abs())
        .max()
        .unwrap();
    assert!(
        max_diff > 2,
        "BT.601 and BT.709 should differ: 709={p709:?} 601={p601:?}"
    );
}

/// A `w×h` RGBA bitmap with every pixel set to `rgba` (straight alpha).
fn rgba_uniform(w: u32, h: u32, rgba: [u8; 4]) -> RgbaImage {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        pixels.extend_from_slice(&rgba);
    }
    RgbaImage::new(w, h, pixels)
}

#[test]
fn rgba_opaque_layer_draws_over_background() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 0, 255]);
    let bmp = rgba_uniform(16, 16, [10, 200, 60, 255]);
    let layer = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config));
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    assert_px(&img, 8, 8, [10, 200, 60, 255], 1);
}

#[test]
fn rgba_fully_transparent_shows_background() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16).with_background([7, 8, 9, 255]);
    // Opaque-colored RGB but zero alpha: must not tint the background.
    let bmp = rgba_uniform(16, 16, [255, 0, 0, 0]);
    let layer = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config));
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    assert_px(&img, 8, 8, [7, 8, 9, 255], 1);
}

#[test]
fn rgba_semi_alpha_blends_premultiplied() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    // 50%-alpha red over opaque blue → ~half-and-half (premultiplied src-over).
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 255, 255]);
    let bmp = rgba_uniform(16, 16, [255, 0, 0, 128]);
    let layer = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config));
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    assert_px(&img, 8, 8, [128, 0, 127, 255], 3);
}

#[test]
fn rgba_layer_opacity_scales_coverage() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    // Opaque red bitmap, but the *layer* opacity is 0.5 → same as 50% alpha.
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 255, 255]);
    let bmp = rgba_uniform(16, 16, [255, 0, 0, 255]);
    let mut placement = LayerPlacement::full_canvas(&config);
    placement.opacity = 0.5;
    let layer = CompositeLayer::rgba(&bmp, placement);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    assert_px(&img, 8, 8, [128, 0, 127, 255], 3);
}

#[test]
fn rgba_placement_lands_in_the_right_quadrant() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(100, 100); // black background

    let bmp = rgba_uniform(8, 8, [255, 255, 255, 255]);
    let placement = LayerPlacement {
        center: [25.0, 25.0],
        size: [20.0, 20.0],
        rotation: 0.0,
        opacity: 1.0,
    };
    let layer = CompositeLayer::rgba(&bmp, placement);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    assert_px(&img, 25, 25, [255, 255, 255, 255], 1); // inside, top-left
    assert_px(&img, 75, 25, [0, 0, 0, 255], 1);
    assert_px(&img, 25, 75, [0, 0, 0, 255], 1);
}

#[test]
fn chroma_key_keys_green_nv12_to_transparent() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64).with_background([0, 0, 255, 255]);
    // Full-range BT.709 green (#00FF00).
    let full = ColorSpace {
        range: ColorRange::Full,
        ..ColorSpace::BT709
    };
    let f = nv12_uniform(64, 64, 182, 30, 12, full);
    let placement = LayerPlacement::full_canvas(&config);
    let effects = LayerEffects {
        mask: None,
        chroma_key: Some(LayerChromaKey {
            rgb: [0.0, 1.0, 0.0],
            strength: 0.5,
            shadow: 0.0,
        }),
    };
    let layer = CompositeLayer::frame(&f, placement).with_fx(effects);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    // Keyed-out green reveals the blue background.
    assert_px(&img, 32, 32, [0, 0, 255, 255], 8);
}

#[test]
fn combined_mask_and_chroma_on_rgba_layer() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64).with_background([0, 0, 255, 255]);
    let bmp = rgba_uniform(64, 64, [0, 255, 0, 255]);
    let placement = LayerPlacement::full_canvas(&config);
    let effects = LayerEffects {
        mask: Some(LayerMask::new(mask_kind::CIRCLE)),
        chroma_key: Some(LayerChromaKey {
            rgb: [0.0, 1.0, 0.0],
            strength: 0.5,
            shadow: 0.0,
        }),
    };
    let layer = CompositeLayer::rgba(&bmp, placement).with_fx(effects);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    // Center: inside circle but green keyed out → background.
    assert_px(&img, 32, 32, [0, 0, 255, 255], 8);
    // Corner: outside circle → background regardless.
    assert_px(&img, 2, 2, [0, 0, 255, 255], 2);
}

#[test]
fn circle_mask_cuts_rgba_corners() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64).with_background([0, 0, 255, 255]);
    let bmp = rgba_uniform(64, 64, [255, 0, 0, 255]);
    let placement = LayerPlacement::full_canvas(&config);
    let effects = LayerEffects {
        mask: Some(LayerMask::new(mask_kind::CIRCLE)),
        chroma_key: None,
    };
    let layer = CompositeLayer::rgba(&bmp, placement).with_fx(effects);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    // Center is inside the circle mask.
    assert_px(&img, 32, 32, [255, 0, 0, 255], 2);
    // Corners are outside the circle — background shows through.
    assert_px(&img, 2, 2, [0, 0, 255, 255], 2);
    assert_px(&img, 61, 61, [0, 0, 255, 255], 2);
}

#[test]
fn rgba_undersized_bitmap_is_rejected() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(8, 8);
    // Claims 4×4 (64 bytes) but only carries 16.
    let bad = RgbaImage::new(4, 4, vec![0u8; 16]);
    let layer = CompositeLayer::rgba(&bad, LayerPlacement::full_canvas(&config));
    assert!(comp.render(&gpu, &config, &[layer]).is_err());
}

// --- SDF shapes ---------------------------------------------------------
//
// shape.wgsl and cutlass-shapes/src/sdf.rs implement the same math by
// contract. These goldens render each shape through both — the GPU SDF
// pipeline, and the CPU reference raster composited as an RGBA layer — and
// assert the canvases agree pixel-for-pixel within rounding tolerance.

use cutlass_compositor::{SdfLayer, SdfParams, Stroke};
use cutlass_shapes::ShapeStyle;

/// Composite one layer on a small canvas sized to the CPU raster.
fn render_shape_pair(
    gpu: &GpuContext,
    comp: &mut Compositor,
    params: SdfParams,
    half: [f32; 2],
    style: ShapeStyle,
    opacity: f32,
) -> (RgbaImage, RgbaImage) {
    let shape = params.with_half(half);
    let cpu_raster = cutlass_shapes::raster(&shape, &style);
    let (w, h) = (cpu_raster.width, cpu_raster.height);
    let config = CompositorConfig::new(w, h).with_background([40, 60, 80, 255]);
    let placement = LayerPlacement {
        center: [w as f32 / 2.0, h as f32 / 2.0],
        size: [w as f32, h as f32],
        rotation: 0.0,
        opacity,
    };

    let sdf_layer = CompositeLayer::sdf(
        SdfLayer {
            shape,
            fill: style.fill.unwrap_or([0, 0, 0, 0]),
            stroke: style.stroke,
        },
        placement,
    );
    let gpu_img = comp.render(gpu, &config, &[sdf_layer]).expect("gpu sdf");

    let cpu_layer = CompositeLayer::rgba(&cpu_raster, placement);
    let cpu_img = comp.render(gpu, &config, &[cpu_layer]).expect("cpu ref");
    (gpu_img, cpu_img)
}

#[track_caller]
fn assert_images_agree(gpu_img: &RgbaImage, cpu_img: &RgbaImage, what: &str, tol: i32) {
    assert_eq!(
        (gpu_img.width, gpu_img.height),
        (cpu_img.width, cpu_img.height)
    );
    let mut worst = 0i32;
    let mut worst_at = (0u32, 0u32);
    for y in 0..gpu_img.height {
        for x in 0..gpu_img.width {
            let a = gpu_img.pixel(x, y);
            let b = cpu_img.pixel(x, y);
            for c in 0..4 {
                let d = (i32::from(a[c]) - i32::from(b[c])).abs();
                if d > worst {
                    worst = d;
                    worst_at = (x, y);
                }
            }
        }
    }
    assert!(
        worst <= tol,
        "{what}: GPU and CPU disagree by {worst} at {worst_at:?} \
         (gpu {:?} vs cpu {:?}, tol {tol})",
        gpu_img.pixel(worst_at.0, worst_at.1),
        cpu_img.pixel(worst_at.0, worst_at.1),
    );
}

fn fill_only(rgba: [u8; 4]) -> ShapeStyle {
    ShapeStyle {
        fill: Some(rgba),
        stroke: None,
    }
}

#[test]
fn sdf_shapes_match_cpu_reference() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);

    let cases: Vec<(&str, SdfParams, [f32; 2])> = vec![
        ("rect", SdfParams::RoundedRect { radius: 0.0 }, [40.0, 25.0]),
        (
            "rounded_rect",
            SdfParams::RoundedRect { radius: 12.0 },
            [40.0, 25.0],
        ),
        ("ellipse", SdfParams::Ellipse, [35.0, 22.0]),
        ("triangle", SdfParams::polygon(3, 0.0), [40.0, 40.0]),
        ("hexagon", SdfParams::polygon(6, 4.0), [40.0, 40.0]),
        (
            "star5",
            SdfParams::Star {
                points: 5,
                inner: 0.5,
                round: 0.0,
            },
            [40.0, 40.0],
        ),
        ("line", SdfParams::Line, [45.0, 5.0]),
        ("arrow", SdfParams::Arrow, [45.0, 20.0]),
        ("heart", SdfParams::Heart, [38.0, 36.0]),
    ];
    for (name, params, half) in cases {
        let (gpu_img, cpu_img) = render_shape_pair(
            &gpu,
            &mut comp,
            params,
            half,
            fill_only([230, 80, 40, 255]),
            1.0,
        );
        assert_images_agree(&gpu_img, &cpu_img, name, 4);
    }
}

#[test]
fn sdf_stroke_and_translucency_match_cpu_reference() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);

    // Fill + stroke.
    let stroked = ShapeStyle {
        fill: Some([20, 120, 240, 255]),
        stroke: Some(Stroke {
            rgba: [255, 230, 0, 255],
            width: 8.0,
        }),
    };
    let (g, c) = render_shape_pair(
        &gpu,
        &mut comp,
        SdfParams::Ellipse,
        [30.0, 30.0],
        stroked,
        1.0,
    );
    assert_images_agree(&g, &c, "stroked ellipse", 4);

    // Stroke-only (fill alpha 0) with a translucent stroke.
    let outline = ShapeStyle {
        fill: None,
        stroke: Some(Stroke {
            rgba: [255, 255, 255, 128],
            width: 5.0,
        }),
    };
    let (g, c) = render_shape_pair(
        &gpu,
        &mut comp,
        SdfParams::RoundedRect { radius: 10.0 },
        [30.0, 20.0],
        outline,
        1.0,
    );
    assert_images_agree(&g, &c, "outline rect", 4);

    // Translucent fill under layer opacity.
    let (g, c) = render_shape_pair(
        &gpu,
        &mut comp,
        SdfParams::polygon(5, 0.0),
        [30.0, 30.0],
        fill_only([200, 40, 200, 180]),
        0.6,
    );
    assert_images_agree(&g, &c, "translucent pentagon at 0.6 opacity", 4);
}

#[test]
fn sdf_quad_padding_keeps_stroke_unclipped() {
    // The stroke overhangs the shape edge by width/2; the CPU raster (and the
    // resolver) pad the quad accordingly. Verify ink reaches beyond the shape
    // box but stays inside the padded quad.
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let style = ShapeStyle {
        fill: Some([255, 0, 0, 255]),
        stroke: Some(Stroke {
            rgba: [0, 255, 0, 255],
            width: 10.0,
        }),
    };
    let (g, _) = render_shape_pair(
        &gpu,
        &mut comp,
        SdfParams::RoundedRect { radius: 0.0 },
        [20.0, 15.0],
        style,
        1.0,
    );
    let (cx, cy) = (g.width / 2, g.height / 2);
    // 23px right of center: inside the stroke ring (edge at 20 + 5 overhang).
    assert_px(&g, cx + 23, cy, [0, 255, 0, 255], 3);
    // At the canvas corner: untouched background.
    assert_px(&g, 0, 0, [40, 60, 80, 255], 1);
}

// --- effects (M4) -------------------------------------------------------

#[test]
fn vignette_darkens_corners_vs_center() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64).with_background([0, 0, 0, 255]);
    let white = CompositeLayer::solid([255, 255, 255, 255], LayerPlacement::full_canvas(&config));
    let baseline = comp.render(&gpu, &config, &[white]).expect("baseline");

    let vignette = PassInstance {
        id: "vignette",
        params: &[0.8],
    };
    let effects = [vignette];
    let effected =
        CompositeLayer::solid([255, 255, 255, 255], LayerPlacement::full_canvas(&config))
            .with_effects(&effects);
    let dark = comp.render(&gpu, &config, &[effected]).expect("vignette");

    let center = baseline.pixel(32, 32);
    let corner = dark.pixel(2, 2);
    assert!(
        corner[0] < center[0] - 20,
        "corner should darken: {corner:?} vs {center:?}"
    );
}

#[test]
fn gaussian_blur_radius_softens_a_hard_edge() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64).with_background([0, 0, 0, 255]);
    // Half white / half black vertical split — blur should mix across the seam.
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for _y in 0..64 {
        for x in 0..64 {
            let v = if x < 32 { 255 } else { 0 };
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let bmp = RgbaImage::new(64, 64, pixels);
    let sharp = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config));
    let sharp_img = comp.render(&gpu, &config, &[sharp]).expect("sharp");

    let blur = PassInstance {
        id: "gaussian_blur",
        params: &[8.0],
    };
    let effects = [blur];
    let soft =
        CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config)).with_effects(&effects);
    let soft_img = comp.render(&gpu, &config, &[soft]).expect("blur");

    let sharp_seam = sharp_img.pixel(31, 32)[0] as i16;
    let soft_seam = soft_img.pixel(31, 32)[0] as i16;
    assert!(
        (soft_seam - sharp_seam).abs() > 10,
        "blur should pull dark ink across the seam: sharp={sharp_seam} soft={soft_seam}"
    );
}

#[test]
fn pixelate_produces_uniform_blocks() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64);
    // Horizontal gradient bitmap.
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for _y in 0..64 {
        for x in 0..64 {
            let v = (x * 4) as u8;
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let bmp = RgbaImage::new(64, 64, pixels);
    let smooth = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config));
    let smooth_img = comp.render(&gpu, &config, &[smooth]).expect("smooth");

    let pixelate = PassInstance {
        id: "pixelate",
        params: &[8.0],
    };
    let effects = [pixelate];
    let blocky_layer =
        CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config)).with_effects(&effects);
    let blocky = comp
        .render(&gpu, &config, &[blocky_layer])
        .expect("pixelate");

    // Within an 8px block, colors should match after pixelate.
    let a = blocky.pixel(10, 10);
    let b = blocky.pixel(11, 10);
    assert_eq!(a, b, "pixelate should flatten local variation");
    // And differ from the smooth original at the same coords.
    assert_ne!(smooth_img.pixel(10, 10), a);
}

#[test]
fn canvas_pass_pixelates_the_composited_stack() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 64);
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for _y in 0..64 {
        for x in 0..64 {
            let v = (x * 4) as u8;
            pixels.extend_from_slice(&[v, 255u8.saturating_sub(v), 32, 255]);
        }
    }
    let bmp = RgbaImage::new(64, 64, pixels);
    let gradient = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config));
    let mut overlay_place = LayerPlacement::full_canvas(&config);
    overlay_place.opacity = 0.5;
    let overlay = CompositeLayer::solid([0, 0, 255, 255], overlay_place);

    let baseline = comp
        .render_compositor_layers(
            &gpu,
            &config,
            &[
                CompositorLayer::layer(&gradient),
                CompositorLayer::layer(&overlay),
            ],
        )
        .expect("baseline");

    let pixelate = PassInstance {
        id: "pixelate",
        params: &[8.0],
    };
    let effects = [pixelate];
    let blocky = comp
        .render_compositor_layers(
            &gpu,
            &config,
            &[
                CompositorLayer::layer(&gradient),
                CompositorLayer::layer(&overlay),
                CompositorLayer::CanvasPass {
                    effects: &effects,
                    grade: None,
                    lut: None,
                },
            ],
        )
        .expect("canvas pass pixelate");

    assert_eq!(
        blocky.pixel(10, 10),
        blocky.pixel(11, 10),
        "canvas pass should flatten local variation"
    );
    assert_ne!(baseline.pixel(10, 10), blocky.pixel(10, 10));
}

// --- Color grade ---------------------------------------------------------

#[test]
fn identity_grade_matches_ungraded_solid() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16);
    let placement = LayerPlacement::full_canvas(&config);
    let rgba = [200, 80, 40, 255];

    let baseline = comp
        .render(&gpu, &config, &[CompositeLayer::solid(rgba, placement)])
        .expect("baseline");

    let explicit = comp
        .render(
            &gpu,
            &config,
            &[CompositeLayer::solid(rgba, placement).with_grade(ColorGrade::IDENTITY)],
        )
        .expect("identity");

    for y in 0..baseline.height {
        for x in 0..baseline.width {
            assert_eq!(baseline.pixel(x, y), explicit.pixel(x, y));
        }
    }
}

#[test]
fn solid_saturation_minus_one_desaturates_red() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16);
    let grade = ColorGrade {
        saturation: -1.0,
        ..ColorGrade::IDENTITY
    };
    let rgba = [255, 0, 0, 255];
    let layer = CompositeLayer::solid(rgba, LayerPlacement::full_canvas(&config)).with_grade(grade);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    let expect = grade_ref_u8(rgba, grade);
    assert_px(&img, 8, 8, expect, 3);
}

#[test]
fn solid_exposure_plus_one_brightens_mid_gray() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16);
    let grade = ColorGrade {
        exposure: 1.0,
        ..ColorGrade::IDENTITY
    };
    let rgba = [128, 128, 128, 255];
    let layer = CompositeLayer::solid(rgba, LayerPlacement::full_canvas(&config)).with_grade(grade);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    let expect = grade_ref_u8(rgba, grade);
    assert_px(&img, 8, 8, expect, 2);
}

#[test]
fn canvas_pass_grade_replaces_translucent_canvas() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 0, 0]);
    let translucent = CompositeLayer::solid([255, 0, 0, 128], LayerPlacement::full_canvas(&config));
    let grade = ColorGrade {
        saturation: -1.0,
        ..ColorGrade::IDENTITY
    };

    let img = comp
        .render_compositor_layers(
            &gpu,
            &config,
            &[
                CompositorLayer::layer(&translucent),
                CompositorLayer::CanvasPass {
                    effects: &[],
                    grade: Some(grade),
                    lut: None,
                },
            ],
        )
        .expect("canvas pass grade");

    let expect = grade_ref_u8([128, 0, 0, 128], grade);
    assert_px(&img, 8, 8, expect, 3);
}

/// Left half opaque red, right half fully transparent tinted red.
fn rgba_opaque_and_transparent(w: u32, h: u32) -> RgbaImage {
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if x < w / 2 {
                pixels[i..i + 4].copy_from_slice(&[255, 0, 0, 255]);
            } else {
                pixels[i..i + 4].copy_from_slice(&[255, 0, 0, 0]);
            }
        }
    }
    RgbaImage::new(w, h, pixels)
}

#[test]
fn rgba_grade_skips_transparent_and_grades_opaque() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16).with_background([7, 8, 9, 255]);
    let grade = ColorGrade {
        saturation: -1.0,
        ..ColorGrade::IDENTITY
    };
    let bmp = rgba_opaque_and_transparent(16, 16);
    let layer = CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config)).with_grade(grade);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    let opaque_expect = grade_ref_u8([255, 0, 0, 255], grade);
    assert_px(&img, 4, 8, opaque_expect, 3);
    assert_px(&img, 12, 8, [7, 8, 9, 255], 1);
}

#[test]
fn yuv_grade_saturation_minus_one_matches_cpu_reference() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let (y, u, v) = (150u8, 60u8, 200u8);
    let color = ColorSpace::BT709;
    let grade = ColorGrade {
        saturation: -1.0,
        ..ColorGrade::IDENTITY
    };

    let config = CompositorConfig::new(16, 16);
    let f = nv12_uniform(16, 16, y, u, v, color);
    let layer = CompositeLayer::frame(&f, LayerPlacement::full_canvas(&config)).with_grade(grade);
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    let [r, g, b] = yuv_ref(y, u, v, color);
    let graded = grade_ref(
        [
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
        ],
        grade,
    );
    assert_px(
        &img,
        8,
        8,
        [quant(graded[0]), quant(graded[1]), quant(graded[2]), 255],
        3,
    );
}
