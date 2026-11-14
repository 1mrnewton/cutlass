//! Checks of variable faces against HarfBuzz, over faces named by the
//! caller.
//!
//! `CUTLASS_VARIABLE_FACES` holds paths with `:` between them, each with
//! `#` and a face index after it for a collection. `hb-shape` and
//! `hb-subset` do not take a path that is not ASCII: name such a file by a
//! link to it.

use crate::vary::nearest;
use crate::{Font, Segment, Tag, read, read_collection};

/// The corners of the box of every point of a glyph's outline, control
/// points too; `None` for a glyph that draws nothing.
fn corners(font: &Font<'_>, glyph: u16) -> Option<[f32; 4]> {
    let outline = font.outline(glyph).ok()?;
    let mut found: Option<[f32; 4]> = None;
    let mut add = |x: f32, y: f32| {
        found = Some(match found {
            None => [x, y, x, y],
            Some([left, bottom, right, top]) => {
                [left.min(x), bottom.min(y), right.max(x), top.max(y)]
            }
        });
    };
    for contour in &outline.contours {
        add(contour.x, contour.y);
        for segment in &contour.segments {
            match *segment {
                Segment::Line { x, y } => add(x, y),
                Segment::Quad { cx, cy, x, y } => {
                    add(cx, cy);
                    add(x, y);
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
                }
            }
        }
    }
    found
}

/// The box as HarfBuzz reports one for a glyph of a face that is set
/// somewhere: its left edge and top rounded, and its width and height
/// rounded from there.
fn extents(font: &Font<'_>, glyph: u16) -> [i32; 4] {
    let Some([left, bottom, right, top]) = corners(font, glyph) else {
        return [0, 0, 0, 0];
    };
    #[allow(clippy::cast_possible_truncation)]
    let whole = |value: f32| nearest(value) as i32;
    let (x, y) = (nearest(left), nearest(top));
    [whole(x), whole(y), whole(right - x), whole(bottom - y)]
}

fn open<'a>(bytes: &'a [u8], index: u32, named: &str) -> Font<'a> {
    if bytes.starts_with(b"ttcf") {
        read_collection(bytes, index)
    } else {
        read(bytes)
    }
    .unwrap_or_else(|error| panic!("{named}: {error:?}"))
}

fn tag_name(tag: Tag) -> String {
    String::from_utf8_lossy(&tag).into_owned()
}

/// Settings to try a face at: each axis at its ends and partway to each
/// with the others at their defaults, every axis off its default at once,
/// and up to eight of its named instances.
fn settings(font: &Font<'_>) -> Vec<Vec<(Tag, f32)>> {
    // `CUTLASS_VARIABLE_SETTINGS` names the settings to try instead, as
    // `wght=700,wdth=80;wght=300`.
    if let Ok(named) = std::env::var("CUTLASS_VARIABLE_SETTINGS") {
        return named
            .split(';')
            .map(|setting| {
                setting
                    .split(',')
                    .filter_map(|one| {
                        let (tag, value) = one.split_once('=')?;
                        let tag: Tag = tag.as_bytes().try_into().ok()?;
                        Some((tag, value.parse().ok()?))
                    })
                    .collect()
            })
            .collect();
    }
    let axes = font.axes();
    let mut sets = Vec::new();
    for axis in &axes {
        for value in [
            axis.min,
            axis.max,
            axis.default + (axis.max - axis.default) * 0.37,
            axis.default - (axis.default - axis.min) * 0.61,
        ] {
            if (value - axis.default).abs() > f32::EPSILON {
                sets.push(vec![(axis.tag, value)]);
            }
        }
    }
    sets.push(
        axes.iter()
            .map(|axis| (axis.tag, axis.default + (axis.max - axis.default) * 0.8))
            .collect(),
    );
    sets.push(
        axes.iter()
            .map(|axis| (axis.tag, axis.default - (axis.default - axis.min) * 0.45))
            .collect(),
    );
    let instances = font.instances();
    let step = instances.len().div_ceil(8).max(1);
    for instance in instances.iter().step_by(step) {
        sets.push(
            axes.iter()
                .zip(&instance.coordinates)
                .map(|(axis, value)| (axis.tag, *value))
                .collect(),
        );
    }
    sets
}

fn printed(setting: &[(Tag, f32)]) -> String {
    setting
        .iter()
        .map(|(tag, value)| format!("{}={value}", tag_name(*tag)))
        .collect::<Vec<_>>()
        .join(",")
}

/// For each named face and each of [`settings`], compare the advance and
/// the box of the glyphs a spread of characters map to with what
/// `hb-shape --variations` prints for them, shaped without the face's
/// layout so each character is its own glyph.
#[test]
#[ignore = "reads the faces named in CUTLASS_VARIABLE_FACES and runs hb-shape"]
#[allow(clippy::print_stdout)]
fn varied_glyphs_agree_with_harfbuzz() {
    let Ok(faces) = std::env::var("CUTLASS_VARIABLE_FACES") else {
        return;
    };
    let show: usize = std::env::var("CUTLASS_VARIABLE_SHOW")
        .ok()
        .and_then(|show| show.parse().ok())
        .unwrap_or(3);
    let mut wrong = 0;
    for named in faces.split(':') {
        let (path, index) = named.split_once('#').unwrap_or((named, "0"));
        let index: u32 = index.parse().expect("index");
        let bytes = std::fs::read(path).expect("face file");
        let plain = open(&bytes, index, named);
        assert!(plain.is_variable(), "{named} does not vary");
        let characters: Vec<char> = (0x20_u32..0x3_0000)
            .filter_map(char::from_u32)
            .filter(|character| matches!(plain.glyph_id(*character), Ok(Some(_))))
            .collect();
        let step = characters.len().div_ceil(1_200).max(1);
        let picked: Vec<String> = characters
            .iter()
            .step_by(step)
            .map(|character| format!("U+{:04X}", u32::from(*character)))
            .collect();
        for setting in settings(&plain) {
            let font = plain.varied(&setting).expect("varied");
            let output = std::process::Command::new("hb-shape")
                .arg(format!("--font-file={path}"))
                .arg(format!("--face-index={index}"))
                .arg(format!("--variations={}", printed(&setting)))
                .args([
                    "--shapers=fallback",
                    "--no-glyph-names",
                    "--no-clusters",
                    "--show-extents",
                ])
                .arg(format!("--unicodes={}", picked.join(",")))
                .output()
                .expect("hb-shape");
            let text = String::from_utf8_lossy(&output.stdout);
            let (mut compared, mut advances, mut boxes) = (0, 0, 0);
            for item in text.trim().trim_matches(['[', ']']).split('|') {
                let Some((head, rest)) = item.split_once('<') else {
                    continue;
                };
                // A glyph, sometimes where it is moved to, and its advance.
                let (glyph, advance) = head.split_once('+').expect("advance");
                let glyph = glyph.split('@').next().expect("glyph");
                let glyph: u16 = glyph.parse().expect("glyph");
                let advance: i32 = advance
                    .split(',')
                    .next()
                    .expect("advance")
                    .parse()
                    .expect("advance");
                let theirs: Vec<i32> = rest
                    .trim_end_matches('>')
                    .split(',')
                    .map(|value| value.parse().expect("extent"))
                    .collect();
                compared += 1;
                let ours = i32::from(font.advance(glyph).expect("advance"));
                // A mark's advance is set to nothing by that shaper.
                if ours != advance && advance != 0 {
                    if advances < show {
                        println!("  glyph {glyph}: advance {ours}, theirs {advance}");
                    }
                    advances += 1;
                }
                // At its default a face's glyphs have the boxes their
                // records store, which need not be the boxes of their
                // points.
                let ours = extents(&font, glyph);
                if ours.to_vec() != theirs && font.variations().iter().any(|place| *place != 0) {
                    if boxes < show {
                        println!(
                            "  glyph {glyph}: box {ours:?}, theirs {theirs:?}, of corners {:?}",
                            corners(&font, glyph)
                        );
                    }
                    boxes += 1;
                }
            }
            println!(
                "{named} at {}: {advances} advances and {boxes} boxes of {compared} differ",
                printed(&setting)
            );
            wrong += advances + boxes;
        }
    }
    assert_eq!(wrong, 0);
}

/// The points of each contour of a glyph's outline, in order.
fn contours(font: &Font<'_>, glyph: u16) -> Option<Vec<Vec<(f32, f32)>>> {
    let outline = font.outline(glyph).ok()?;
    let mut contours = Vec::new();
    for contour in &outline.contours {
        let mut points = vec![(contour.x, contour.y)];
        for segment in &contour.segments {
            match *segment {
                Segment::Line { x, y } => points.push((x, y)),
                Segment::Quad { cx, cy, x, y } => points.extend([(cx, cy), (x, y)]),
                Segment::Cubic {
                    c1x,
                    c1y,
                    c2x,
                    c2y,
                    x,
                    y,
                } => points.extend([(c1x, c1y), (c2x, c2y), (x, y)]),
            }
        }
        contours.push(points);
    }
    Some(contours)
}

/// How far apart the farthest two matching points of two outlines are,
/// along either axis; `None` when the outlines are not of the same
/// points. A contour is closed by a line back to its start only when its
/// last point is not its first, which rounding its points can change:
/// where one contour has one point more than the other and that point is
/// back at its start, it is that line, and is left out.
fn apart(one: &[Vec<(f32, f32)>], other: &[Vec<(f32, f32)>]) -> Option<f32> {
    if one.len() != other.len() {
        return None;
    }
    let mut furthest = 0.0_f32;
    for (one, other) in one.iter().zip(other) {
        let closes = |points: &[(f32, f32)]| {
            points
                .first()
                .zip(points.last())
                .is_some_and(|(first, last)| (first.0 - last.0, first.1 - last.1) == (0.0, 0.0))
        };
        let (one, other) = if one.len() == other.len() + 1 && closes(one) {
            (&one[..other.len()], other.as_slice())
        } else if other.len() == one.len() + 1 && closes(other) {
            (one.as_slice(), &other[..one.len()])
        } else if one.len() == other.len() {
            (one.as_slice(), other.as_slice())
        } else {
            return None;
        };
        for (one, other) in one.iter().zip(other) {
            furthest = furthest
                .max((one.0 - other.0).abs())
                .max((one.1 - other.1).abs());
        }
    }
    Some(furthest)
}

/// The advance of `glyph` as the moves of its edges give it, which is
/// where `hb-subset` takes the advances it writes from, and not as the
/// face's table of advances does.
fn edge_advance(font: &Font<'_>, glyph: u16) -> Option<u16> {
    let crate::Outlines::TrueType(source) = &font.outlines else {
        return None;
    };
    let advance = crate::ttf::advance(source, glyph, &font.reading()).ok()?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let whole = advance.clamp(0.0, 65_535.0) as u16;
    Some(whole)
}

/// For each named face and each of [`settings`], have `hb-subset` write
/// the face at that setting as a face that does not vary, and compare
/// every glyph of it, point for point and advance for advance, with the
/// same glyph of the variable face set there. `hb-subset` writes whole
/// numbers: a point may differ by the half unit of that rounding, and
/// one of a glyph made of parts by half a unit more for each depth of
/// parts, as each part's place is rounded as well as its points. Two
/// units, which is parts three deep, is the most allowed. An advance is
/// compared as the face's table of advances gives it and, where that
/// differs, as the moves of the glyph's edges give it: `hb-subset` writes
/// the second, and a face whose two tables disagree is counted apart.
/// An advance that is neither is counted as differing; where that has
/// been seen, in glyphs made of parts in one face, `hb-shape` gave the
/// advance this reader gives and `hb-subset` had left the glyph's
/// advance as it is at the default.
#[test]
#[ignore = "reads the faces named in CUTLASS_VARIABLE_FACES and runs hb-subset"]
#[allow(clippy::print_stdout)]
fn varied_outlines_agree_with_an_instance_harfbuzz_writes() {
    let Ok(faces) = std::env::var("CUTLASS_VARIABLE_FACES") else {
        return;
    };
    let show: usize = std::env::var("CUTLASS_VARIABLE_SHOW")
        .ok()
        .and_then(|show| show.parse().ok())
        .unwrap_or(3);
    let folder = std::env::temp_dir().join(format!("cutlass-instances-{}", std::process::id()));
    std::fs::create_dir_all(&folder).expect("folder");
    let mut wrong = 0;
    for named in faces.split(':') {
        let (path, index) = named.split_once('#').unwrap_or((named, "0"));
        let index: u32 = index.parse().expect("index");
        let bytes = std::fs::read(path).expect("face file");
        let plain = open(&bytes, index, named);
        for setting in settings(&plain) {
            // Every axis is pinned, so what is written does not vary.
            let pinned: Vec<(Tag, f32)> = plain
                .axes()
                .iter()
                .map(|axis| {
                    let value = setting
                        .iter()
                        .find(|(tag, _)| *tag == axis.tag)
                        .map_or(axis.default, |(_, value)| *value);
                    (axis.tag, value)
                })
                .collect();
            let written = folder.join("instance.ttf");
            let status = std::process::Command::new("hb-subset")
                .arg(format!("--font-file={path}"))
                .arg(format!("--face-index={index}"))
                .arg(format!("--variations={}", printed(&pinned)))
                .args(["--glyphs=*", "--retain-gids", "--notdef-outline"])
                .arg(format!("--output-file={}", written.display()))
                .status()
                .expect("hb-subset");
            if !status.success() {
                println!("{named} at {}: hb-subset failed", printed(&setting));
                wrong += 1;
                continue;
            }
            let instance = std::fs::read(&written).expect("instance");
            let theirs = read(&instance).expect("instance face");
            let ours = plain.varied(&setting).expect("varied");
            let (mut glyphs, mut advances, mut near, mut far) = (0, 0, 0, 0);
            let mut disagree = 0;
            let mut furthest = 0.0_f32;
            for glyph in 0..ours.glyph_count().min(theirs.glyph_count()) {
                glyphs += 1;
                if ours.advance(glyph) != theirs.advance(glyph)
                    && edge_advance(&ours, glyph).map(Ok) == Some(theirs.advance(glyph))
                {
                    disagree += 1;
                } else if ours.advance(glyph) != theirs.advance(glyph) {
                    if advances < show {
                        println!(
                            "  glyph {glyph}: advance {:?}, theirs {:?}; its edges give {:?}",
                            ours.advance(glyph),
                            theirs.advance(glyph),
                            edge_advance(&ours, glyph)
                        );
                    }
                    advances += 1;
                }
                let (Some(one), Some(other)) = (contours(&ours, glyph), contours(&theirs, glyph))
                else {
                    far += 1;
                    continue;
                };
                let Some(apart) = apart(&one, &other) else {
                    if far < show {
                        println!("  glyph {glyph}: not the same points as theirs");
                    }
                    far += 1;
                    continue;
                };
                furthest = furthest.max(apart);
                if apart > 2.001 {
                    if far < show {
                        println!("  glyph {glyph}: a point {apart} apart");
                    }
                    far += 1;
                } else if apart > 0.501 {
                    near += 1;
                }
            }
            println!(
                "{named} at {}: of {glyphs} glyphs {advances} advances differ and {disagree} more are the face's own two answers, {near} have a point over half a unit apart and {far} over two; the furthest {furthest}",
                printed(&setting)
            );
            wrong += advances + far;
        }
    }
    let _removed = std::fs::remove_dir_all(&folder);
    assert_eq!(wrong, 0);
}

/// For each named face, how long it takes to set it, and to outline and
/// to measure every glyph at its default and set to the first of
/// [`settings`]: the least of five runs of each.
#[test]
#[ignore = "reads the faces named in CUTLASS_VARIABLE_FACES"]
#[allow(clippy::print_stdout)]
fn varied_glyphs_are_timed() {
    let Ok(faces) = std::env::var("CUTLASS_VARIABLE_FACES") else {
        return;
    };
    let least = |run: &mut dyn FnMut()| {
        (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                run();
                started.elapsed()
            })
            .min()
            .unwrap_or_default()
    };
    for named in faces.split(':') {
        let (path, index) = named.split_once('#').unwrap_or((named, "0"));
        let index: u32 = index.parse().expect("index");
        let bytes = std::fs::read(path).expect("face file");
        let plain = open(&bytes, index, named);
        let Some(setting) = settings(&plain).into_iter().next() else {
            continue;
        };
        let set = plain.varied(&setting).expect("varied");
        let setting_takes = least(&mut || {
            for _ in 0..100 {
                std::hint::black_box(plain.varied(&setting).expect("varied"));
            }
        }) / 100;
        let outlines = |font: &Font<'_>| {
            for glyph in 0..font.glyph_count() {
                std::hint::black_box(font.outline(glyph).ok());
            }
        };
        let advances = |font: &Font<'_>| {
            for glyph in 0..font.glyph_count() {
                std::hint::black_box(font.advance(glyph).ok());
            }
        };
        println!(
            "{named} at {}: {} glyphs; setting it takes {setting_takes:?}; outlines {:?} at the default and {:?} set; advances {:?} and {:?}",
            printed(&setting),
            plain.glyph_count(),
            least(&mut || outlines(&plain)),
            least(&mut || outlines(&set)),
            least(&mut || advances(&plain)),
            least(&mut || advances(&set)),
        );
    }
}
