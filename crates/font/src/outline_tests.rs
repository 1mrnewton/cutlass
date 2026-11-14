//! A check of outlines against HarfBuzz over faces named by the caller.
//!
//! `CUTLASS_OUTLINE_FACES` holds paths with `:` between them, each with
//! `#` and a face index after it for a collection. Every glyph of each
//! face is outlined, and in a face of CFF outlines the boxes of the glyphs
//! a spread of characters map to are compared with what
//! `hb-shape --show-extents` prints. `hb-shape` does not take a path that
//! is not ASCII: name such a file by a link to it.

use crate::{Font, Segment, read, read_collection};

/// The box of a glyph's outline as HarfBuzz reports one: the left edge
/// rounded down, the top up, and the width and the height to the far
/// edges rounded away. Control points count as part of the outline for a
/// cubic curve, as HarfBuzz counts them, and a quadratic curve is measured
/// where it goes.
fn extents(font: &Font<'_>, glyph: u16) -> Option<[i32; 4]> {
    let outline = font.outline(glyph).ok()?;
    let (mut left, mut bottom, mut right, mut top) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let mut seen = false;
    let mut add = |x: f32, y: f32| {
        left = left.min(x);
        right = right.max(x);
        bottom = bottom.min(y);
        top = top.max(y);
        seen = true;
    };
    for contour in &outline.contours {
        add(contour.x, contour.y);
        let (mut at_x, mut at_y) = (contour.x, contour.y);
        for segment in &contour.segments {
            match *segment {
                Segment::Line { x, y } => {
                    add(x, y);
                    (at_x, at_y) = (x, y);
                }
                Segment::Quad { cx, cy, x, y } => {
                    add(x, y);
                    // Where the curve turns, on each axis.
                    for (from, control, to, across) in [(at_x, cx, x, true), (at_y, cy, y, false)] {
                        let spread = from - 2.0 * control + to;
                        if spread != 0.0 {
                            let time = (from - control) / spread;
                            if time > 0.0 && time < 1.0 {
                                let back = 1.0 - time;
                                let point = |from: f32, control: f32, to: f32| {
                                    back * back * from
                                        + 2.0 * back * time * control
                                        + time * time * to
                                };
                                let (px, py) = (point(at_x, cx, x), point(at_y, cy, y));
                                if across {
                                    add(px, at_y.min(y).max(py.min(at_y.max(y))));
                                } else {
                                    add(at_x.min(x).max(px.min(at_x.max(x))), py);
                                }
                            }
                        }
                    }
                    (at_x, at_y) = (x, y);
                }
                Segment::Cubic {
                    c1x,
                    c1y,
                    c2x,
                    c2y,
                    x,
                    y,
                } => {
                    add(c1x, c1y);
                    add(c2x, c2y);
                    add(x, y);
                    (at_x, at_y) = (x, y);
                }
            }
        }
    }
    if !seen {
        return Some([0, 0, 0, 0]);
    }
    #[allow(clippy::cast_possible_truncation)]
    let (left, bottom, right, top) = (
        left.floor() as i32,
        bottom.floor() as i32,
        right.ceil() as i32,
        top.ceil() as i32,
    );
    Some([left, top, right - left, bottom - top])
}

#[test]
#[ignore = "reads the faces named in CUTLASS_OUTLINE_FACES and runs hb-shape"]
#[allow(clippy::print_stdout)]
fn outlines_agree_with_harfbuzz() {
    let Ok(faces) = std::env::var("CUTLASS_OUTLINE_FACES") else {
        return;
    };
    let mut wrong = 0;
    for named in faces.split(':') {
        let (path, index) = named.split_once('#').unwrap_or((named, "0"));
        let index: u32 = index.parse().expect("index");
        let bytes = std::fs::read(path).expect("face file");
        let font = if bytes.starts_with(b"ttcf") {
            read_collection(&bytes, index)
        } else {
            read(&bytes)
        }
        .unwrap_or_else(|error| panic!("{named}: {error:?}"));
        let started = std::time::Instant::now();
        let mut failed = 0;
        for glyph in 0..font.glyph_count() {
            if let Err(error) = font.outline(glyph) {
                if failed < 5 {
                    println!("  {named} glyph {glyph}: {error:?}");
                }
                failed += 1;
            }
        }
        let outlined = started.elapsed();
        // A spread of the characters the face maps.
        let characters: Vec<char> = (0x20_u32..0x3_0000)
            .filter_map(char::from_u32)
            .filter(|character| matches!(font.glyph_id(*character), Ok(Some(_))))
            .collect();
        let step = characters.len().div_ceil(1_500).max(1);
        let picked: Vec<String> = characters
            .iter()
            .step_by(step)
            .map(|character| format!("U+{:04X}", u32::from(*character)))
            .collect();
        let output = std::process::Command::new("hb-shape")
            .arg(format!("--font-file={path}"))
            .arg(format!("--face-index={index}"))
            .args([
                "--no-glyph-names",
                "--no-clusters",
                "--no-positions",
                "--show-extents",
            ])
            .arg(format!("--unicodes={}", picked.join(",")))
            .output()
            .expect("hb-shape");
        let printed = String::from_utf8_lossy(&output.stdout);
        let mut compared = 0;
        let mut differ = 0;
        for item in printed.trim().trim_matches(['[', ']']).split('|') {
            let Some((glyph, rest)) = item.split_once('<') else {
                continue;
            };
            let glyph: u16 = glyph.parse().expect("glyph");
            let theirs: Vec<i32> = rest
                .trim_end_matches('>')
                .split(',')
                .map(|value| value.parse().expect("extent"))
                .collect();
            // A TrueType glyph's box is what its header says, which need
            // not be the box of its points: only CFF boxes are compared.
            if font.kind() != crate::FontKind::OpenType {
                continue;
            }
            compared += 1;
            let ours = extents(&font, glyph);
            if ours.map(|ours| ours.to_vec()) != Some(theirs.clone()) {
                if differ < 5 {
                    println!("  {named} glyph {glyph}: ours {ours:?} theirs {theirs:?}");
                }
                differ += 1;
            }
        }
        println!(
            "{named}: {} glyphs outlined in {outlined:?}, {failed} failed; {differ} of {compared} boxes differ",
            font.glyph_count()
        );
        wrong += failed + differ;
    }
    assert_eq!(wrong, 0);
}

/// Open every face of every font file under the folders named in
/// `CUTLASS_FACE_FOLDERS` (with `:` between them) and outline every glyph,
/// and say which faces do not open or have glyphs that do not outline.
/// Nothing is asserted: a system has faces in forms this reader does not
/// take, and the list is what to look at.
#[test]
#[ignore = "reads the font folders named in CUTLASS_FACE_FOLDERS"]
#[allow(clippy::print_stdout)]
fn the_faces_of_the_named_folders_open() {
    fn walk(folder: &std::path::Path, depth: u8, files: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(folder) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if depth < 4 {
                    walk(&path, depth + 1, files);
                }
            } else if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    ["ttf", "otf", "ttc", "otc"].contains(&extension.to_lowercase().as_str())
                })
            {
                files.push(path);
            }
        }
    }
    let Ok(folders) = std::env::var("CUTLASS_FACE_FOLDERS") else {
        return;
    };
    let mut files = Vec::new();
    for folder in folders.split(':') {
        walk(std::path::Path::new(folder), 0, &mut files);
    }
    files.sort();
    let (mut faces, mut opened, mut glyphs, mut failed) = (0, 0, 0_u64, 0_u64);
    let started = std::time::Instant::now();
    for path in &files {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let count = if bytes.starts_with(b"ttcf") && bytes.len() >= 12 {
            u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]).min(64)
        } else {
            1
        };
        for index in 0..count {
            faces += 1;
            let font = if bytes.starts_with(b"ttcf") {
                read_collection(&bytes, index)
            } else {
                read(&bytes)
            };
            let font = match font {
                Ok(font) => font,
                Err(error) => {
                    println!("{}#{index}: {error:?}", path.display());
                    continue;
                }
            };
            opened += 1;
            if !font.has_outlines() {
                continue;
            }
            let mut bad = 0;
            let mut first = None;
            for glyph in 0..font.glyph_count() {
                glyphs += 1;
                if let Err(error) = font.outline(glyph) {
                    bad += 1;
                    first.get_or_insert((glyph, error));
                }
            }
            if let Some((glyph, error)) = first {
                println!(
                    "{}#{index}: {bad} glyphs do not outline, the first {glyph}: {error:?}",
                    path.display()
                );
                failed += bad;
            }
        }
    }
    println!(
        "{} files, {opened} of {faces} faces open, {failed} of {glyphs} glyphs do not outline, {:?}",
        files.len(),
        started.elapsed()
    );
}
