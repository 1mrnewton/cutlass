//! Cutlass desktop editor (Slice 1).
//!
//! A native Rust/Slint frontend that links [`cutlass_engine`] **directly** —
//! the same headless engine the iOS/Android apps reach through the
//! `cutlass-mobile` C-ABI/JNI bridge. Proving the engine drives a third,
//! unrelated frontend unchanged is the point: the editor core in `crates/`
//! carries no UI or platform assumptions, and `apps/` just adds consumers.
//!
//! This slice mirrors the mobile harness's preview: build a synthetic demo
//! project, then scrub it — every slider tick calls [`Engine::get_frame`] and
//! shows the composited RGBA frame. Opening real media and the timeline editing
//! UI come in later slices.

use std::cell::RefCell;
use std::rc::Rc;

use cutlass_engine::{Engine, EngineConfig};
use cutlass_models::{Generator, Project, Rational, RationalTime, TimeRange, TrackKind};
use cutlass_render::RgbaImage;

slint::include_modules!();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fps = Rational::FPS_30;
    let total_frames = 180i64; // ~6s @ 30fps

    let engine = Engine::with_project(EngineConfig::default(), demo_project(fps, total_frames))?;
    let engine = Rc::new(RefCell::new(engine));

    let window = AppWindow::new()?;
    window.set_duration_seconds(total_frames as f32 / fps.as_f64() as f32);
    render_into(&window, &mut engine.borrow_mut(), fps, total_frames, 0.0);

    {
        let engine = Rc::clone(&engine);
        let weak = window.as_weak();
        window.on_scrubbed(move |seconds| {
            let Some(window) = weak.upgrade() else {
                return;
            };
            render_into(&window, &mut engine.borrow_mut(), fps, total_frames, seconds);
        });
    }

    window.run()?;
    Ok(())
}

/// Render the frame at `seconds` through the engine and push it to the preview.
fn render_into(
    window: &AppWindow,
    engine: &mut Engine,
    fps: Rational,
    total_frames: i64,
    seconds: f32,
) {
    let frame = (f64::from(seconds.max(0.0)) * fps.as_f64()).round() as i64;
    let frame = frame.clamp(0, (total_frames - 1).max(0));
    match engine.get_frame(RationalTime::new(frame, fps)) {
        Ok(image) => window.set_preview(to_slint_image(&image)),
        Err(err) => eprintln!("preview render failed at frame {frame}: {err}"),
    }
}

/// Copy an engine [`RgbaImage`] (RGBA8) into a Slint image. Preview frames are
/// opaque, so straight vs. premultiplied alpha is a no-op here.
fn to_slint_image(frame: &RgbaImage) -> slint::Image {
    let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(frame.width, frame.height);
    buffer.make_mut_bytes().copy_from_slice(&frame.pixels);
    slint::Image::from_rgba8(buffer)
}

/// A synthetic, file-free project: a full-canvas color sweeping through the hue
/// wheel, so the preview + scrub path works with no assets (mirrors the mobile
/// `CutlassPreview::demo`). Temporary demo content for slice 1.
fn demo_project(fps: Rational, total_frames: i64) -> Project {
    let step = 6i64; // a new color every 0.2s @ 30fps
    let mut project = Project::new("cutlass-desktop-demo", fps);
    let track = project.add_track(TrackKind::Sticker, "BG");
    let mut start = 0i64;
    while start < total_frames {
        let len = step.min(total_frames - start);
        let rgba = hue_sweep(start as f32 / total_frames as f32);
        let span = TimeRange::at_rate(start, len, fps);
        let _ = project.add_generated(track, Generator::SolidColor { rgba }, span);
        start += step;
    }
    project
}

/// An opaque RGBA sweep through the hue wheel as `t` runs 0->1.
fn hue_sweep(t: f32) -> [u8; 4] {
    let h = (t.rem_euclid(1.0) * 6.0).rem_euclid(6.0);
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
        255,
    ]
}
