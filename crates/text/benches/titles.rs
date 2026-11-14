//! How long a title takes to open and to draw.
//!
//! `cargo bench -p cutlass-text --bench titles`
//!
//! A title that moves, and every caption, is opened and drawn for each
//! frame, so these two figures are what a frame of text costs. "open" is
//! from the font's bytes to glyphs laid out with their images; "draw" is
//! the picture from an open title. Each figure is the median of 31 runs
//! after one warmup. The fixture face is always measured; the system
//! faces are measured where this machine has them.

#![allow(clippy::print_stdout)]

use std::time::{Duration, Instant};

use cutlass_model::{BloomParams, LetterPose, MaterialText, Micros, Segment, TimeRange};
use cutlass_picture::PictureSource;
use cutlass_text::{NoFaces, TextCache, TextSource};

const RUNS: usize = 31;
const FIXTURE: &[u8] = include_bytes!("../../font/fixtures/shaping.ttf");

fn median(mut run: impl FnMut()) -> Duration {
    run();
    let mut times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[RUNS / 2]
}

fn title(content: &str) -> MaterialText {
    let mut text = MaterialText::new("title");
    text.content = content.to_owned();
    text.font_size = 72.0;
    text.line_max_width = 900.0;
    text.text_color = "#FFFFFF".to_owned();
    text
}

fn measure(name: &str, bytes: &[u8], content: &str) {
    let text = title(content);
    let segment = Segment::new("segment", "title", TimeRange::new(0, 1_000_000));
    // Each open with nothing kept from the last, as before the cache.
    let cold = median(|| {
        let cache = TextCache::new();
        let opened = TextSource::open_cached(bytes, None, &text, None, &segment, &NoFaces, &cache);
        std::hint::black_box(opened.is_ok());
    });
    // Each open after the first, with the glyphs and the layout kept.
    let cache = TextCache::new();
    let warm = median(|| {
        let opened = TextSource::open_cached(bytes, None, &text, None, &segment, &NoFaces, &cache);
        std::hint::black_box(opened.is_ok());
    });
    let Ok(source) = TextSource::open_cached(bytes, None, &text, None, &segment, &NoFaces, &cache)
    else {
        println!("{name}: does not open");
        return;
    };
    let draw = median(|| {
        std::hint::black_box(source.sample(Micros(1_000)));
    });
    let (width, height) = source.size();
    println!(
        "{name}: {} glyphs, {width}x{height}; open {:.0} us cold, {:.0} us warm; draw {:.0} us",
        source.records().len(),
        cold.as_secs_f64() * 1e6,
        warm.as_secs_f64() * 1e6,
        draw.as_secs_f64() * 1e6,
    );
}

/// A frame of the heaviest title: large letters with an outline, a
/// shadow and a glow, each letter in a pose of its own, as a title is
/// while its letters move. This is what such a title costs at each frame.
fn measure_moving(name: &str, bytes: &[u8], content: &str) {
    let mut text = title(content);
    text.font_size = 144.0;
    text.line_max_width = 1_800.0;
    text.border_color = "#000000".to_owned();
    text.border_alpha = 1.0;
    text.border_width = 0.08;
    text.has_shadow = true;
    text.shadow_color = "#000000".to_owned();
    text.shadow_alpha = 0.8;
    text.shadow_angle = -45.0;
    text.shadow_distance = 8.0;
    text.shadow_smoothing = 0.5;
    let glow = BloomParams {
        color: "#FFCC00".to_owned(),
        strength: 0.8,
        range: 0.5,
        ..BloomParams::default()
    };
    let segment = Segment::new("segment", "title", TimeRange::new(0, 1_000_000));
    let cache = TextCache::new();
    let open =
        || TextSource::open_cached(bytes, None, &text, Some(&glow), &segment, &NoFaces, &cache);
    if open().is_err() {
        println!("{name}: does not open");
        return;
    }
    let mut size = (0, 0);
    let mut glyphs = 0;
    let mut step = 0_u32;
    // Opened from what is kept, posed and drawn, as each frame is.
    let frame = median(|| {
        let Ok(mut source) = open() else {
            return;
        };
        step += 1;
        let poses = (0..source.letters().len())
            .map(|letter| {
                let phase = f64::from(step) * 0.3 + f64::from(u32::try_from(letter).unwrap_or(0));
                LetterPose {
                    alpha: 0.5 + 0.5 * phase.sin().abs(),
                    dx: 0.0,
                    dy: 12.0 * phase.sin(),
                    scale: 1.0 + 0.1 * phase.cos(),
                }
            })
            .collect();
        source.set_poses(poses);
        size = source.size();
        glyphs = source.records().len();
        std::hint::black_box(source.sample(Micros(1_000)));
    });
    println!(
        "{name}: {glyphs} glyphs, {}x{}; a frame with every letter moving {:.0} us",
        size.0,
        size.1,
        frame.as_secs_f64() * 1e6,
    );
}

fn main() {
    measure(
        "fixture, 48 letters",
        FIXTURE,
        "fie fie AVA eve fife five if a vie fief eve AVA fie",
    );
    let system: [(&str, &str, &str); 6] = [
        (
            "Latin, Arial",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "The quick brown fox jumps over the lazy dog, twice over.",
        ),
        (
            "Arabic, Arial",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "\u{645}\u{631}\u{62D}\u{628}\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645} \u{627}\u{644}\u{633}\u{644}\u{627}\u{645} \u{639}\u{644}\u{64A}\u{643}\u{645}",
        ),
        (
            "Devanagari, Arial Unicode",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947} \u{926}\u{941}\u{928}\u{93F}\u{92F}\u{93E} \u{939}\u{93F}\u{928}\u{94D}\u{926}\u{940} \u{92D}\u{93E}\u{937}\u{93E}",
        ),
        (
            "Khmer, Khmer Sangam MN",
            "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf",
            "\u{1797}\u{17B6}\u{179F}\u{17B6}\u{1781}\u{17D2}\u{1798}\u{17C2}\u{179A} \u{179F}\u{17D2}\u{179A}\u{17A1}\u{17B6}\u{1789}\u{17CB} \u{1780}\u{1798}\u{17D2}\u{1796}\u{17BB}\u{1787}\u{17B6} \u{1780}\u{17D2}\u{179A}\u{17BB}\u{1784} \u{1796}\u{17D2}\u{179A}\u{17C7}\u{179A}\u{17B6}\u{1787}\u{17B6}\u{178E}\u{17B6}\u{1785}\u{1780}\u{17D2}\u{179A}",
        ),
        (
            "Javanese, Noto Sans Javanese",
            "/System/Library/Fonts/Supplemental/NotoSansJavanese-Regular.otf",
            "\u{A9A7}\u{A9B1}\u{A997}\u{A9AE} \u{A9B2}\u{A98F}\u{A9C0}\u{A9B1}\u{A9AB} \u{A98F}\u{A9BF}\u{A9A0}\u{A9BA}\u{A9B4}\u{A9A4}\u{A9C0} \u{A994}\u{A9AA}\u{A9BA}\u{A9B4}\u{A992}\u{A9BE}\u{A98F}\u{A982}\u{A9A0}",
        ),
        (
            "mixed, Arial",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "Hello \u{645}\u{631}\u{62D}\u{628}\u{627} 123 (\u{5E9}\u{5DC}\u{5D5}\u{5DD}) office",
        ),
    ];
    for (name, path, content) in system {
        match std::fs::read(path) {
            Ok(bytes) => measure(name, &bytes, content),
            Err(_) => println!("{name}: {path} is not on this machine"),
        }
    }
    measure_moving(
        "fixture, moving, outline, shadow and glow",
        FIXTURE,
        "fie fie AVA eve fife five if a vie",
    );
    let arial = "/System/Library/Fonts/Supplemental/Arial.ttf";
    match std::fs::read(arial) {
        Ok(bytes) => measure_moving(
            "Latin, Arial, moving, outline, shadow and glow",
            &bytes,
            "The quick brown fox jumps over",
        ),
        Err(_) => println!("moving: {arial} is not on this machine"),
    }
}
