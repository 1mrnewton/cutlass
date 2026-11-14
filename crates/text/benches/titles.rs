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

use cutlass_model::{MaterialText, Micros, Segment, TimeRange};
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

fn main() {
    measure(
        "fixture, 48 letters",
        FIXTURE,
        "fie fie AVA eve fife five if a vie fief eve AVA fie",
    );
    let system: [(&str, &str, &str); 4] = [
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
}
