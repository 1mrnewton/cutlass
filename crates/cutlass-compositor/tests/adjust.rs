//! GPU readback tests for the six new color-adjust sliders.

use cutlass_compositor::{
    ColorGrade, CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};

fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping compositor adjust test: no GPU adapter ({e})");
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

fn quant(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn grade_ref_u8(rgba: [u8; 4], grade: ColorGrade) -> [u8; 4] {
    let rgb = grade.apply([
        f32::from(rgba[0]) / 255.0,
        f32::from(rgba[1]) / 255.0,
        f32::from(rgba[2]) / 255.0,
    ]);
    [quant(rgb[0]), quant(rgb[1]), quant(rgb[2]), rgba[3]]
}

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

fn solid_graded(gpu: &GpuContext, rgba: [u8; 4], grade: ColorGrade) -> RgbaImage {
    let mut comp = Compositor::new(gpu);
    let config = CompositorConfig::new(16, 16);
    let layer = CompositeLayer::solid(rgba, LayerPlacement::full_canvas(&config)).with_grade(grade);
    comp.render(gpu, &config, &[layer]).expect("render")
}

fn rgba_graded(gpu: &GpuContext, bmp: &RgbaImage, grade: ColorGrade) -> RgbaImage {
    let mut comp = Compositor::new(gpu);
    let config = CompositorConfig::new(bmp.width, bmp.height).with_background([0, 0, 0, 255]);
    let layer = CompositeLayer::rgba(bmp, LayerPlacement::full_canvas(&config)).with_grade(grade);
    comp.render(gpu, &config, &[layer]).expect("render")
}

#[test]
fn tint_plus_one_shifts_green_channel() {
    let gpu = gpu_or_skip!();
    let grade = ColorGrade {
        tint: 1.0,
        ..ColorGrade::IDENTITY
    };
    let rgba = [128, 128, 128, 255];
    let img = solid_graded(&gpu, rgba, grade);
    let expect = grade_ref_u8(rgba, grade);
    assert_px(&img, 8, 8, expect, 3);
    assert!(img.pixel(8, 8)[1] > rgba[1], "positive tint lifts green");
}

#[test]
fn hue_plus_one_rotates_red_toward_yellow() {
    let gpu = gpu_or_skip!();
    let grade = ColorGrade {
        hue: 1.0,
        ..ColorGrade::IDENTITY
    };
    let rgba = [255, 0, 0, 255];
    let img = solid_graded(&gpu, rgba, grade);
    let expect = grade_ref_u8(rgba, grade);
    assert_px(&img, 8, 8, expect, 4);
    let px = img.pixel(8, 8);
    assert!(
        px[1] > 20,
        "hue +1 should introduce green into red (got {px:?})"
    );
    assert!(px[0] > px[2], "red should still dominate blue after +30°");
}

#[test]
fn highlights_brighten_bright_gray_more_than_dark() {
    let gpu = gpu_or_skip!();
    let grade = ColorGrade {
        highlights: 1.0,
        ..ColorGrade::IDENTITY
    };
    let bright = solid_graded(&gpu, [200, 200, 200, 255], grade);
    let dark = solid_graded(&gpu, [40, 40, 40, 255], grade);
    let b = bright.pixel(8, 8)[0] as i32 - 200;
    let d = dark.pixel(8, 8)[0] as i32 - 40;
    assert!(
        b > d,
        "highlights should lift bright gray more (Δbright={b}, Δdark={d})"
    );
    assert_px(&bright, 8, 8, grade_ref_u8([200, 200, 200, 255], grade), 3);
}

#[test]
fn shadows_minus_one_darkens_dark_gray_more() {
    let gpu = gpu_or_skip!();
    let grade = ColorGrade {
        shadows: -1.0,
        ..ColorGrade::IDENTITY
    };
    let bright = solid_graded(&gpu, [200, 200, 200, 255], grade);
    let dark = solid_graded(&gpu, [40, 40, 40, 255], grade);
    let b = 200 - bright.pixel(8, 8)[0] as i32;
    let d = 40 - dark.pixel(8, 8)[0] as i32;
    assert!(
        d > b,
        "shadows −1 should darken dark gray more (Δdark={d}, Δbright={b})"
    );
    assert_px(&dark, 8, 8, grade_ref_u8([40, 40, 40, 255], grade), 3);
}

/// Soft step edge (not 0/255) so unsharp overshoot is visible before clamp.
fn half_tone_step(w: u32, h: u32) -> RgbaImage {
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let v = if x < w / 2 { 80 } else { 180 };
            pixels[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    RgbaImage::new(w, h, pixels)
}

#[test]
fn vignette_darkens_corner_more_than_center() {
    let gpu = gpu_or_skip!();
    let grade = ColorGrade {
        vignette: 1.0,
        ..ColorGrade::IDENTITY
    };
    let mut pixels = vec![0u8; 32 * 32 * 4];
    for px in pixels.chunks_exact_mut(4) {
        px.copy_from_slice(&[200, 200, 200, 255]);
    }
    let bmp = RgbaImage::new(32, 32, pixels.clone());
    let img = rgba_graded(&gpu, &bmp, grade);
    let center = img.pixel(16, 16)[0];
    let corner = img.pixel(1, 1)[0];
    assert!(
        corner < center,
        "vignette should darken corner ({corner}) more than center ({center})"
    );

    // CPU apply_image parity at center / corner.
    let mut cpu = pixels;
    grade.apply_image(32, 32, &mut cpu);
    let cpu_center = cpu[(16 * 32 + 16) * 4];
    let cpu_corner = cpu[(32 + 1) * 4];
    assert!((i32::from(center) - i32::from(cpu_center)).abs() <= 4);
    assert!((i32::from(corner) - i32::from(cpu_corner)).abs() <= 4);
}

#[test]
fn sharpness_increases_edge_contrast() {
    let gpu = gpu_or_skip!();
    let bmp = half_tone_step(32, 32);
    let baseline = rgba_graded(&gpu, &bmp, ColorGrade::IDENTITY);
    let grade = ColorGrade {
        sharpness: 1.0,
        ..ColorGrade::IDENTITY
    };
    let sharp = rgba_graded(&gpu, &bmp, grade);

    // Probe just left and right of the vertical edge at x=16.
    let base_l = baseline.pixel(15, 16)[0] as i32;
    let base_r = baseline.pixel(16, 16)[0] as i32;
    let sharp_l = sharp.pixel(15, 16)[0] as i32;
    let sharp_r = sharp.pixel(16, 16)[0] as i32;
    let base_contrast = (base_r - base_l).abs();
    let sharp_contrast = (sharp_r - sharp_l).abs();
    assert!(
        sharp_contrast > base_contrast,
        "sharpness should increase edge contrast ({sharp_contrast} vs {base_contrast}; \
         L/R baseline {base_l}/{base_r}, sharp {sharp_l}/{sharp_r})"
    );
}
