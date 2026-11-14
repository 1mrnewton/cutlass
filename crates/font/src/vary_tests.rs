//! Tests of a variable face built in code, [`crate::fixture_var`].
//!
//! Each number an assertion expects is what HarfBuzz 14.2 gives for the
//! fixture file: `hb-shape --variations=... --shapers=fallback
//! --show-extents` for advances and boxes, `hb-shape --variations=...`
//! for kerning and marks, and `hb-vector --variations=...` for outlines.

use crate::fixture_var::{
    A, A_HEAVY, B, C, D, MARK, NARROW, Place, SPACE, bends_of, composite, definitions_of,
    moving_anchor, outlines_of, positions_of, store, table_of, variable, variable_edges,
    variable_with, variable_with_tables,
};
use crate::{Adjust, Anchor, Axis, Font, FontError, Instance, Positioning, Segment, Tag, read};

const VARIABLE: &[u8] = include_bytes!("../fixtures/variable.ttf");
const EDGES: &[u8] = include_bytes!("../fixtures/variable-edges.ttf");
const STATIC: &[u8] = include_bytes!("../fixtures/sample.ttf");

const WGHT: Tag = *b"wght";
const WDTH: Tag = *b"wdth";

fn set<'a>(bytes: &'a [u8], settings: &[(Tag, f32)]) -> Font<'a> {
    read(bytes).expect("face").varied(settings).expect("set")
}

/// A glyph's outline as `hb-vector` writes one.
fn drawn(font: &Font<'_>, glyph: u16) -> String {
    use std::fmt::Write;
    // It writes each number to two decimal places.
    let n = |value: f32| (value * 100.0).round() / 100.0;
    let mut path = String::new();
    for contour in &font.outline(glyph).expect("outline").contours {
        write!(path, "M{},{}", n(contour.x), n(contour.y)).expect("write");
        for segment in &contour.segments {
            match *segment {
                Segment::Line { x, y } => write!(path, "L{},{}", n(x), n(y)),
                Segment::Quad { cx, cy, x, y } => {
                    write!(path, "Q{},{} {},{}", n(cx), n(cy), n(x), n(y))
                }
                Segment::Cubic { .. } => write!(path, "C"),
            }
            .expect("write");
        }
        path.push('Z');
    }
    path
}

fn advances(font: &Font<'_>) -> Vec<u16> {
    [A, B, C, SPACE, MARK, D]
        .iter()
        .map(|glyph| font.advance(*glyph).expect("advance"))
        .collect()
}

#[test]
fn a_face_tells_its_axes_and_named_instances() {
    let face = read(VARIABLE).expect("face");
    assert!(face.is_variable());
    assert_eq!(
        face.axes(),
        [
            Axis {
                tag: WGHT,
                min: 100.0,
                default: 400.0,
                max: 900.0,
                hidden: false,
                name: Some("Weight".to_owned()),
            },
            Axis {
                tag: WDTH,
                min: 50.0,
                default: 100.0,
                max: 150.0,
                hidden: false,
                name: Some("Width".to_owned()),
            },
        ]
    );
    assert_eq!(
        face.instances(),
        [
            Instance {
                name: Some("Regular".to_owned()),
                postscript_name: None,
                coordinates: vec![400.0, 100.0],
            },
            Instance {
                name: Some("Bold".to_owned()),
                postscript_name: None,
                coordinates: vec![700.0, 100.0],
            },
            Instance {
                name: Some("Condensed Bold".to_owned()),
                postscript_name: Some("Fixture-CondensedBold".to_owned()),
                coordinates: vec![700.0, 75.0],
            },
        ]
    );
    assert_eq!(face.variations(), [0, 0]);
    // A face that does not vary has none of this, and setting it is
    // nothing.
    let mut plain = read(STATIC).expect("face");
    assert!(!plain.is_variable());
    assert!(plain.axes().is_empty());
    assert!(plain.instances().is_empty());
    assert!(plain.variations().is_empty());
    let before = plain.fingerprint();
    assert_eq!(plain.set_variations(&[(WGHT, 700.0)]), Ok(()));
    assert_eq!(plain.fingerprint(), before);
}

#[test]
fn a_setting_becomes_a_place_along_each_axis() {
    let places = |settings: &[(Tag, f32)]| set(VARIABLE, settings).variations().to_vec();
    assert_eq!(places(&[]), [0, 0]);
    assert_eq!(places(&[(WGHT, 400.0), (WDTH, 100.0)]), [0, 0]);
    // The ends of each axis, and past them.
    assert_eq!(places(&[(WGHT, 900.0)]), [16_384, 0]);
    assert_eq!(places(&[(WGHT, 100.0)]), [-16_384, 0]);
    assert_eq!(places(&[(WGHT, 2_000.0), (WDTH, 10.0)]), [16_384, -16_384]);
    assert_eq!(places(&[(WDTH, 75.0)]), [0, -8_192]);
    assert_eq!(places(&[(WDTH, 125.0)]), [0, 8_192]);
    // Weight is bent: halfway along its heavy side is a quarter of the
    // way, and four fifths of the way is seven tenths.
    assert_eq!(places(&[(WGHT, 650.0)]), [4_096, 0]);
    assert_eq!(places(&[(WGHT, 800.0)]), [11_469, 0]);
    assert_eq!(places(&[(WGHT, 250.0)]), [-8_192, 0]);
    // An axis the face lacks is passed over, the later of two settings
    // counts, and a value that is no number is the default.
    assert_eq!(places(&[(*b"slnt", -10.0), (WGHT, 900.0)]), [16_384, 0]);
    assert_eq!(places(&[(WGHT, 900.0), (WGHT, 650.0)]), [4_096, 0]);
    assert_eq!(places(&[(WGHT, f32::NAN), (WDTH, f32::INFINITY)]), [0, 0]);
}

#[test]
fn advances_change_by_the_table_of_advances() {
    let at = |settings: &[(Tag, f32)]| advances(&set(VARIABLE, settings));
    assert_eq!(at(&[]), [500, 600, 500, 250, 0, 700]);
    assert_eq!(at(&[(WGHT, 900.0)]), [600, 680, 600, 300, 0, 810]);
    assert_eq!(at(&[(WGHT, 650.0)]), [525, 620, 525, 263, 0, 728]);
    assert_eq!(at(&[(WGHT, 800.0)]), [570, 656, 570, 285, 0, 777]);
    assert_eq!(at(&[(WGHT, 100.0)]), [500, 600, 500, 250, 0, 700]);
    assert_eq!(at(&[(WDTH, 50.0)]), [400, 540, 400, 225, 0, 600]);
    // A change of twelve and a half is rounded up, to twelve less.
    assert_eq!(at(&[(WDTH, 75.0)]), [450, 570, 450, 238, 0, 650]);
    assert_eq!(at(&[(WDTH, 150.0)]), [500, 600, 500, 250, 0, 700]);
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)]),
        [500, 620, 500, 275, 0, 710]
    );
    assert_eq!(
        at(&[(WGHT, 700.0), (WDTH, 75.0)]),
        [490, 602, 490, 258, 0, 694]
    );
    // The replaced `A`, and the glyph past the end of the map, which
    // has the last row.
    let heavy = set(VARIABLE, &[(WGHT, 900.0)]);
    assert_eq!(heavy.advance(A_HEAVY), Ok(630));
    assert_eq!(heavy.advance(D), Ok(810));
    assert_eq!(heavy.advance(8), Err(FontError::UnknownGlyph));
}

#[test]
fn advances_follow_a_glyphs_edges_without_that_table() {
    let at = |settings: &[(Tag, f32)]| advances(&set(EDGES, settings));
    assert_eq!(at(&[]), [500, 600, 500, 250, 0, 700]);
    // Only the space's edge moves with weight; `A`'s moves with width,
    // and `C`, made of `A`, has `A`'s.
    assert_eq!(at(&[(WGHT, 900.0)]), [500, 600, 500, 300, 0, 700]);
    assert_eq!(at(&[(WGHT, 650.0)]), [500, 600, 500, 263, 0, 700]);
    assert_eq!(at(&[(WDTH, 50.0)]), [400, 600, 400, 225, 0, 700]);
    assert_eq!(at(&[(WDTH, 75.0)]), [450, 600, 450, 238, 0, 700]);
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)]),
        [400, 600, 400, 275, 0, 700]
    );
}

#[test]
fn points_move_and_the_points_between_follow() {
    let at = |settings: &[(Tag, f32)], glyph: u16| drawn(&set(VARIABLE, settings), glyph);
    assert_eq!(at(&[], A), "M50,0L450,0L450,700L250,700L50,700L50,0Z");
    // The bottom right corner moves 100 and the bottom left stays: the
    // top right goes with the one, the top left with the other, and the
    // top middle halfway.
    assert_eq!(
        at(&[(WGHT, 900.0)], A),
        "M50,0L550,0L550,700L300,700L50,700L50,0Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0)], A),
        "M50,0L475,0L475,700L262.5,700L50,700L50,0Z"
    );
    // Between a quarter of the way and the end, the top middle rises.
    assert_eq!(
        at(&[(WGHT, 800.0)], A),
        "M50,0L520,0L520,700L285,724L50,700L50,0Z"
    );
    // A tuple that moves every point.
    assert_eq!(
        at(&[(WDTH, 50.0)], A),
        "M40,0L360,0L360,700L200,700L40,700L40,0Z"
    );
    assert_eq!(
        at(&[(WDTH, 75.0)], A),
        "M45,0L405,0L405,700L225,700L45,700L45,0Z"
    );
    // In the heavy narrow corner a tuple for the corner adds its own.
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)], A),
        "M60,0L480,0L480,700L270,700L60,700L60,0Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0), (WDTH, 75.0)], A),
        "M47.5,0L432.5,0L432.5,700L240,700L47.5,700L47.5,0Z"
    );
    assert_eq!(
        at(&[(WGHT, 700.0), (WDTH, 75.0)], A),
        "M49,0L449,0L449,700L249,724L49,700L49,0Z"
    );
    // Tuples that share one list of points.
    assert_eq!(at(&[(WGHT, 900.0)], B), "M50,0L630,0L630,700L50,700L50,0Z");
    assert_eq!(at(&[(WGHT, 800.0)], B), "M50,0L606,0L606,700L50,700L50,0Z");
    assert_eq!(at(&[(WDTH, 50.0)], B), "M50,0L490,0L490,700L50,700L50,0Z");
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)], B),
        "M50,0L570,0L570,700L50,700L50,0Z"
    );
    // The mark rises; the glyphs with no moves stay, and the space
    // draws nothing.
    assert_eq!(
        at(&[(WGHT, 900.0)], MARK),
        "M100,770L300,770L300,920L100,920L100,770Z"
    );
    assert_eq!(
        at(&[(WGHT, 900.0)], A_HEAVY),
        "M50,0L470,0L470,700L50,700L50,0Z"
    );
    assert_eq!(at(&[(WGHT, 900.0)], D), "M50,0L650,0L650,700L50,700L50,0Z");
    assert_eq!(at(&[(WGHT, 900.0)], SPACE), "");
}

#[test]
fn a_glyph_of_parts_moves_each_part_and_its_place() {
    let at = |settings: &[(Tag, f32)]| drawn(&set(VARIABLE, settings), C);
    assert_eq!(
        at(&[]),
        "M50,0L450,0L450,700L250,700L50,700L50,0ZM200,950L400,950L400,1100L200,1100L200,950Z"
    );
    // `A` as it is at the setting, and the mark both risen and moved.
    assert_eq!(
        at(&[(WGHT, 900.0)]),
        "M50,0L550,0L550,700L300,700L50,700L50,0ZM230,1020L430,1020L430,1170L230,1170L230,1020Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0)]),
        "M50,0L475,0L475,700L262.5,700L50,700L50,0ZM207.5,967.5L407.5,967.5L407.5,1117.5L207.5,1117.5L207.5,967.5Z"
    );
    assert_eq!(
        at(&[(WGHT, 800.0)]),
        "M50,0L520,0L520,700L285,724L50,700L50,0ZM221,999L421,999L421,1149L221,1149L221,999Z"
    );
    assert_eq!(
        at(&[(WDTH, 50.0)]),
        "M40,0L360,0L360,700L200,700L40,700L40,0ZM200,950L400,950L400,1100L200,1100L200,950Z"
    );
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)]),
        "M60,0L480,0L480,700L270,700L60,700L60,0ZM230,1020L430,1020L430,1170L230,1170L230,1020Z"
    );
}

/// The kerning of `A` before `B`, and the anchors that set the mark on
/// `A`: the mark's own and `A`'s.
fn placed(font: &Font<'_>, varied: bool) -> (Adjust, (Anchor, Anchor)) {
    let table = font.gpos().expect("positioning");
    let subtable = |lookup: u16| {
        let lookup = table.lookup(lookup).expect("lookup");
        if varied {
            table.positioning(lookup.kind, lookup.subtables[0])
        } else {
            Positioning::read(lookup.kind, lookup.subtables[0])
        }
        .expect("subtable")
    };
    let Positioning::Pair(pairs) = subtable(0) else {
        panic!("not pairs");
    };
    let Positioning::MarkToBase(marks) = subtable(1) else {
        panic!("not marks");
    };
    (
        pairs.adjust(A, B).expect("pair").expect("kerned").0,
        marks.anchors(MARK, A).expect("anchors").expect("set"),
    )
}

#[test]
fn kerning_and_anchors_move_with_the_setting() {
    let at = |settings: &[(Tag, f32)]| {
        let (kern, (mark, base)) = placed(&set(VARIABLE, settings), true);
        assert_eq!((kern.x, kern.y, kern.y_advance), (0, 0, 0));
        assert_eq!(mark, Anchor { x: 200, y: 750 });
        (kern.x_advance, base.x, base.y)
    };
    assert_eq!(at(&[]), (-50, 250, 700));
    // HarfBuzz shapes `A B` to an advance of 510 for `A`, which is 600
    // less 90, and sets the mark 100 right of and 20 below its own
    // anchor: an anchor at 300, 730.
    assert_eq!(at(&[(WGHT, 900.0)]), (-90, 300, 730));
    // 525 less 60 is its 465; the anchor's twelve and a half and seven
    // and a half are each rounded up.
    assert_eq!(at(&[(WGHT, 650.0)]), (-60, 263, 708));
    assert_eq!(at(&[(WGHT, 800.0)]), (-78, 285, 721));
    assert_eq!(at(&[(WGHT, 700.0), (WDTH, 75.0)]), (-66, 270, 712));
    // Width alone moves none of them, nor does the light side of weight.
    assert_eq!(at(&[(WDTH, 50.0)]), (-50, 250, 700));
    assert_eq!(at(&[(WGHT, 100.0)]), (-50, 250, 700));
    // A subtable read without the face's setting has the default values.
    let (kern, (_, base)) = placed(&set(VARIABLE, &[(WGHT, 900.0)]), false);
    assert_eq!((kern.x_advance, base), (-50, Anchor { x: 250, y: 700 }));
}

#[test]
fn a_feature_runs_other_lookups_where_the_face_is_set() {
    let lookups = |settings: &[(Tag, f32)]| {
        let font = set(VARIABLE, settings);
        let table = font.gsub().expect("substitutions");
        assert_eq!(table.feature_tag(0), Ok(*b"rvrn"));
        table
            .feature_lookups(0)
            .expect("feature")
            .iter()
            .collect::<Vec<u16>>()
    };
    assert_eq!(lookups(&[]), [0_u16; 0]);
    // From halfway along the heavy side, after the bending: 650 is a
    // quarter of the way, 700 two fifths, 800 seven tenths.
    assert_eq!(lookups(&[(WGHT, 650.0)]), [0_u16; 0]);
    assert_eq!(lookups(&[(WGHT, 700.0)]), [0_u16; 0]);
    assert_eq!(lookups(&[(WGHT, 800.0)]), [0]);
    assert_eq!(lookups(&[(WGHT, 900.0), (WDTH, 50.0)]), [0]);
    assert_eq!(lookups(&[(WGHT, 100.0)]), [0_u16; 0]);
    // Set back, the feature runs nothing again.
    let mut font = set(VARIABLE, &[(WGHT, 900.0)]);
    font.set_variations(&[]).expect("set");
    let table = font.gsub().expect("substitutions");
    assert_eq!(table.feature_lookups(0).expect("feature").len(), 0);
}

#[test]
fn a_face_set_somewhere_is_another_face_to_what_keeps_glyphs() {
    let plain = read(VARIABLE).expect("face");
    let bold = plain.varied(&[(WGHT, 700.0)]).expect("set");
    let other = plain.varied(&[(WGHT, 700.0), (WDTH, 75.0)]).expect("set");
    assert_ne!(plain.fingerprint(), bold.fingerprint());
    assert_ne!(bold.fingerprint(), other.fingerprint());
    // The same setting is the same number, however it is come by.
    let again = read(VARIABLE)
        .expect("face")
        .varied(&[(WDTH, 100.0), (WGHT, 700.0)])
        .expect("set");
    assert_eq!(bold.fingerprint(), again.fingerprint());
    // Setting a copy leaves the face it was copied from alone.
    assert_eq!(plain.variations(), [0, 0]);
    assert_eq!(plain.advance(A), Ok(500));
    // Set back to the default it is the face it was.
    let mut back = bold.clone();
    back.set_variations(&[]).expect("set");
    assert_eq!(back.fingerprint(), plain.fingerprint());
    assert_eq!(drawn(&back, A), drawn(&plain, A));
    assert_eq!(back.advance(A), Ok(500));
    // And a setting that is the default is the default.
    let same = plain.varied(&[(WGHT, 400.0), (WDTH, 100.0)]).expect("set");
    assert_eq!(same.fingerprint(), plain.fingerprint());
}

#[test]
fn the_files_are_the_faces_the_builder_makes() {
    assert_eq!(VARIABLE, variable());
    assert_eq!(EDGES, variable_edges());
}

/// Ask a face everything a setting changes, at a few settings. Nothing
/// is expected but an answer.
fn exercise(bytes: &[u8]) {
    let Ok(plain) = read(bytes) else {
        return;
    };
    let _axes = plain.axes();
    let _instances = plain.instances();
    for settings in [
        &[(WGHT, 900.0), (WDTH, 50.0)][..],
        &[(WGHT, 650.0)],
        &[(WDTH, 125.0)],
    ] {
        let Ok(font) = plain.varied(settings) else {
            continue;
        };
        for glyph in 0..font.glyph_count() {
            let _advance = font.advance(glyph);
            let _outline = font.outline(glyph);
        }
        if let Some(table) = font.gpos() {
            for index in 0..table.lookup_count() {
                let Ok(lookup) = table.lookup(index) else {
                    continue;
                };
                for data in &lookup.subtables {
                    match table.positioning(lookup.kind, data) {
                        Ok(Positioning::Pair(pairs)) => {
                            let _adjust = pairs.adjust(A, B);
                        }
                        Ok(Positioning::MarkToBase(marks)) => {
                            let _anchors = marks.anchors(MARK, A);
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(table) = font.gsub() {
            let _lookups = table.feature_lookups(0);
        }
    }
}

#[test]
fn torn_and_scrambled_tables_are_answered_without_a_panic() {
    let started = std::time::Instant::now();
    for name in [
        b"fvar", b"avar", b"gvar", b"HVAR", b"GDEF", b"GPOS", b"GSUB",
    ] {
        let whole = table_of(name);
        // Cut short at every length.
        for length in 0..whole.len() {
            exercise(&variable_with(name, Some(whole[..length].to_vec())));
        }
        // Every byte in turn set to each of a few values.
        for at in 0..whole.len() {
            for value in [0x00, 0x01, 0x7F, 0x80, 0xFF] {
                let mut table = whole.clone();
                table[at] = value;
                exercise(&variable_with(name, Some(table)));
            }
        }
        // And without the table at all.
        exercise(&variable_with(name, None));
    }
    assert!(started.elapsed() < std::time::Duration::from_secs(60));
}

#[test]
fn a_table_that_does_not_read_leaves_the_rest_working() {
    // Axes of a version this reader does not know: the face does not
    // vary, and still opens.
    let mut axes = table_of(b"fvar");
    axes[1] = 2;
    let bytes = variable_with(b"fvar", Some(axes));
    let font = read(&bytes).expect("face");
    assert!(!font.is_variable());
    let font = font.varied(&[(WGHT, 900.0)]).expect("set");
    assert_eq!(font.advance(A), Ok(500));
    // No axes at all is the same.
    let mut axes = table_of(b"fvar");
    axes[8..10].copy_from_slice(&[0, 0]);
    assert!(
        !read(&variable_with(b"fvar", Some(axes)))
            .expect("face")
            .is_variable()
    );
    // Moves written for another count of axes move nothing; advances
    // still change by their own table.
    let mut moves = table_of(b"gvar");
    moves[5] = 3;
    let bytes = variable_with(b"gvar", Some(moves));
    let font = set(&bytes, &[(WGHT, 900.0)]);
    assert_eq!(drawn(&font, A), "M50,0L450,0L450,700L250,700L50,700L50,0Z");
    assert_eq!(font.advance(A), Ok(600));
    // A table of advances that does not read leaves the glyphs' edges
    // to say.
    let mut advances = table_of(b"HVAR");
    advances[1] = 9;
    let bytes = variable_with(b"HVAR", Some(advances));
    let font = set(&bytes, &[(WGHT, 900.0), (WDTH, 50.0)]);
    assert_eq!(font.advance(A), Ok(400));
    assert_eq!(font.advance(SPACE), Ok(275));
    // Bending of a version this reader does not know is no bending.
    let mut bends = table_of(b"avar");
    bends[1] = 7;
    let bytes = variable_with(b"avar", Some(bends));
    assert_eq!(set(&bytes, &[(WGHT, 650.0)]).variations(), [8_192, 0]);
    // A glyph whose moves are torn is an error for that glyph alone.
    let mut moves = table_of(b"gvar");
    let length = moves.len();
    moves.truncate(length - 4);
    let bytes = variable_with(b"gvar", Some(moves));
    let font = set(&bytes, &[(WGHT, 900.0)]);
    assert_eq!(font.outline(MARK), Err(FontError::Truncated));
    assert!(font.outline(A).is_ok());
}

/// A face of one axis whose glyph 1 has `points` points and `tuples`
/// tuples that each move every one of them by nothing.
fn heavy_moves(points: u16, tuples: u16) -> Vec<u8> {
    use crate::fixture::{be, boxed, font};
    let mut glyph = vec![0, 1, 0, 0, 0, 0];
    glyph.extend(points.to_be_bytes());
    glyph.extend(points.to_be_bytes());
    glyph.extend((points - 1).to_be_bytes());
    glyph.extend([0, 0]);
    let count = usize::from(points);
    glyph.extend(std::iter::repeat_n(0x37, count));
    glyph.extend(std::iter::repeat_n(1, count * 2));
    let mut loca = vec![0; 8];
    loca.extend(u32::try_from(glyph.len()).expect("length").to_be_bytes());
    // One axis, from 0 through 0 to 1.
    let mut axes = be(&[1, 0, 16, 2, 1, 20, 0, 8]);
    axes.extend(b"wght");
    axes.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);
    axes.extend(be(&[0, 256]));
    // Each tuple: every point, and runs of 64 zeros for x and for y.
    let moved = count + 4;
    let mut zeros = vec![0xBF_u8; moved / 64];
    if moved % 64 != 0 {
        zeros.push(0x80 | u8::try_from(moved % 64 - 1).expect("run"));
    }
    let mut one = vec![0_u8];
    one.extend(&zeros);
    one.extend(&zeros);
    let mut data = be(&[tuples, 4 + tuples * 6]);
    for _ in 0..tuples {
        data.extend(be(&[
            u16::try_from(one.len()).expect("size"),
            0xA000,
            0x4000,
        ]));
    }
    for _ in 0..tuples {
        data.extend(&one);
    }
    let mut moves = be(&[1, 0, 1, 0]);
    moves.extend(32_u32.to_be_bytes());
    moves.extend(be(&[2, 1]));
    moves.extend(32_u32.to_be_bytes());
    moves.extend(0_u32.to_be_bytes());
    moves.extend(0_u32.to_be_bytes());
    moves.extend(u32::try_from(data.len()).expect("length").to_be_bytes());
    moves.extend(data);
    font(
        &[boxed(500), boxed(500)],
        &[('A', 1)],
        &[
            (b"glyf", glyph),
            (b"loca", loca),
            (b"fvar", axes),
            (b"gvar", moves),
        ],
    )
}

#[test]
fn a_glyph_of_many_points_and_many_tuples_is_moved_in_time() {
    let bytes = heavy_moves(5_000, 1_000);
    let font = set(&bytes, &[(WGHT, 1.0)]);
    assert_eq!(font.variations(), [16_384]);
    let started = std::time::Instant::now();
    let outline = font.outline(1).expect("outline");
    assert!(outline.contours[0].segments.len() >= 4_999);
    assert_eq!(font.advance(1), Ok(500));
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
    // Setting a face is no work for each glyph: many settings are quick.
    let face = read(VARIABLE).expect("face");
    let started = std::time::Instant::now();
    for step in 0..2_000_u16 {
        let font = face
            .varied(&[(WGHT, 100.0 + f32::from(step) * 0.4)])
            .expect("set");
        assert_eq!(font.variations().len(), 2);
    }
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[test]
fn counts_past_a_table_are_refused_and_not_walked() {
    let started = std::time::Instant::now();
    // More instances than the table holds: none are listed.
    let mut axes = table_of(b"fvar");
    axes[12..14].copy_from_slice(&[0xFF, 0xFF]);
    let bytes = variable_with(b"fvar", Some(axes));
    assert!(!read(&bytes).expect("face").is_variable());
    // More axes than a face may have.
    let mut axes = table_of(b"fvar");
    axes[8..10].copy_from_slice(&[0xFF, 0xFF]);
    assert!(
        !read(&variable_with(b"fvar", Some(axes)))
            .expect("face")
            .is_variable()
    );
    // A store that says it has more regions and groups than it has.
    let mut advances = table_of(b"HVAR");
    advances[26..28].copy_from_slice(&[0xFF, 0xFF]);
    let bytes = variable_with(b"HVAR", Some(advances));
    let font = set(&bytes, &[(WGHT, 900.0)]);
    assert_eq!(font.advance(SPACE), Ok(300));
    // A glyph with more tuples than it has bytes for.
    let mut moves = table_of(b"gvar");
    let array = usize::try_from(u32::from_be_bytes([
        moves[16], moves[17], moves[18], moves[19],
    ]))
    .expect("offset");
    moves[array..array + 2].copy_from_slice(&[0x0F, 0xFF]);
    let bytes = variable_with(b"gvar", Some(moves));
    let font = set(&bytes, &[(WGHT, 900.0)]);
    assert!(font.outline(A).is_err());
    // A feature table with more records of conditions than it has.
    let mut substitutions = table_of(b"GSUB");
    let variations = usize::try_from(u32::from_be_bytes([
        substitutions[10],
        substitutions[11],
        substitutions[12],
        substitutions[13],
    ]))
    .expect("offset");
    substitutions[variations + 4..variations + 8].copy_from_slice(&[0xFF; 4]);
    let bytes = variable_with(b"GSUB", Some(substitutions));
    let font = set(&bytes, &[(WGHT, 100.0)]);
    let table = font.gsub().expect("substitutions");
    assert_eq!(table.feature_lookups(0).expect("feature").len(), 0);
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

/// The fixture with its mark's own anchor moving too: a quarter of a
/// unit right and three quarters up for each quarter of the way to the
/// heavy end, so that the two anchors lie between whole units.
fn moving_anchors() -> Vec<u8> {
    variable_with_tables(vec![
        (b"GDEF", definitions_of(&[&[-40], &[50], &[30], &[1], &[3]])),
        (b"GPOS", positions_of(moving_anchor((200, 750), 3, 4))),
    ])
}

/// The fixture with the mark of `C` put where its first point lands on
/// the top middle point of `A`, and scaled by `scale` when that is given.
fn matched(scale: Option<i16>) -> Vec<u8> {
    let made = composite(
        [50, 0, 450, 850],
        &[
            (A, Place::At(0, 0), true, None),
            (MARK, Place::Matched(3, 0), false, scale),
        ],
    );
    variable_with_tables(outlines_of(made).into_iter().collect())
}

/// The fixture with bending of the second version: after each axis's
/// own, weight is moved half the way to its heavy end at the narrow end
/// of width.
fn bent_by_width() -> Vec<u8> {
    variable_with(b"avar", Some(bent_by_width_table()))
}

/// The `avar` table of that face.
fn bent_by_width_table() -> Vec<u8> {
    let mut table = bends_of(2);
    // No map of axes to rows: each axis has the row of its number.
    table.extend(0_u32.to_be_bytes());
    table.extend(
        u32::try_from(table.len() + 4)
            .expect("offset")
            .to_be_bytes(),
    );
    table.extend(store(&[NARROW], 1, &[&[8_192], &[0]]));
    table
}

/// The faces above, by a file name for each.
fn variants() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("anchors.ttf", moving_anchors()),
        ("matched.ttf", matched(None)),
        ("matched-scaled.ttf", matched(Some(4_096))),
        ("bent-by-width.ttf", bent_by_width()),
    ]
}

/// Write the faces of [`variants`] into the folder `CUTLASS_VARIANTS`
/// names, to ask HarfBuzz what it makes of them: the tests that use them
/// expect what it says.
#[test]
#[ignore = "writes the variant faces into the folder named in CUTLASS_VARIANTS"]
fn write_the_variant_faces() {
    let Ok(folder) = std::env::var("CUTLASS_VARIANTS") else {
        return;
    };
    for (name, bytes) in variants() {
        std::fs::write(std::path::Path::new(&folder).join(name), bytes).expect("write");
    }
}

#[test]
fn each_of_a_marks_two_anchors_is_rounded_by_itself() {
    // Both anchors lie between whole units part of the way to the heavy
    // end. HarfBuzz shapes `A` and the mark to a mark moved, after `A`'s
    // advance is taken off, by these distances between the two anchors:
    // each anchor's own change rounded, and not the distance.
    let bytes = moving_anchors();
    let apart = |settings: &[(Tag, f32)]| {
        let (_, (mark, base)) = placed(&set(&bytes, settings), true);
        (base.x - mark.x, base.y - mark.y)
    };
    assert_eq!(apart(&[]), (50, -50));
    // A quarter of the way: 262.5 and 200.25 are 62.25 apart, and each
    // rounded they are 63.
    assert_eq!(apart(&[(WGHT, 650.0)]), (63, -43));
    assert_eq!(apart(&[(WGHT, 700.0)]), (70, -39));
    assert_eq!(apart(&[(WGHT, 800.0)]), (84, -31));
    assert_eq!(apart(&[(WGHT, 900.0)]), (99, -23));
    assert_eq!(apart(&[(WGHT, 700.0), (WDTH, 75.0)]), (70, -39));
    let (_, (mark, _)) = placed(&set(&bytes, &[(WGHT, 650.0)]), true);
    assert_eq!(mark, Anchor { x: 200, y: 751 });
}

#[test]
fn a_part_placed_by_matching_points_follows_the_points() {
    // The mark's first point on the top middle point of `A`, wherever
    // the face's setting puts the two; the move the glyph's own tuple
    // has for the part counts for nothing.
    let bytes = matched(None);
    let at = |settings: &[(Tag, f32)]| drawn(&set(&bytes, settings), C);
    assert_eq!(
        at(&[]),
        "M50,0L450,0L450,700L250,700L50,700L50,0ZM250,700L450,700L450,850L250,850L250,700Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0)]),
        "M50,0L475,0L475,700L262.5,700L50,700L50,0ZM262.5,700L462.5,700L462.5,850L262.5,850L262.5,700Z"
    );
    assert_eq!(
        at(&[(WGHT, 800.0)]),
        "M50,0L520,0L520,700L285,724L50,700L50,0ZM285,724L485,724L485,874L285,874L285,724Z"
    );
    assert_eq!(
        at(&[(WGHT, 900.0)]),
        "M50,0L550,0L550,700L300,700L50,700L50,0ZM300,700L500,700L500,850L300,850L300,700Z"
    );
    assert_eq!(
        at(&[(WDTH, 50.0)]),
        "M40,0L360,0L360,700L200,700L40,700L40,0ZM200,700L400,700L400,850L200,850L200,700Z"
    );
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)]),
        "M60,0L480,0L480,700L270,700L60,700L60,0ZM270,700L470,700L470,850L270,850L270,700Z"
    );
    assert_eq!(set(&bytes, &[(WGHT, 900.0)]).advance(C), Ok(600));
    // A part that is scaled is matched by where its matrix puts its
    // point, and lands there exactly, between whole units or not.
    let bytes = matched(Some(4_096));
    let at = |settings: &[(Tag, f32)]| drawn(&set(&bytes, settings), C);
    assert_eq!(
        at(&[]),
        "M50,0L450,0L450,700L250,700L50,700L50,0ZM250,700L300,700L300,737.5L250,737.5L250,700Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0)]),
        "M50,0L475,0L475,700L262.5,700L50,700L50,0ZM262.5,700L312.5,700L312.5,737.5L262.5,737.5L262.5,700Z"
    );
    assert_eq!(
        at(&[(WGHT, 800.0)]),
        "M50,0L520,0L520,700L285,724L50,700L50,0ZM285,724L335,724L335,761.5L285,761.5L285,724Z"
    );
    assert_eq!(
        at(&[(WGHT, 900.0)]),
        "M50,0L550,0L550,700L300,700L50,700L50,0ZM300,700L350,700L350,737.5L300,737.5L300,700Z"
    );
    // A point the glyph does not have is an error, not a guess.
    let made = composite(
        [50, 0, 450, 850],
        &[
            (A, Place::At(0, 0), true, None),
            (MARK, Place::Matched(9, 0), false, None),
        ],
    );
    let bytes = variable_with_tables(outlines_of(made).into_iter().collect());
    assert_eq!(
        set(&bytes, &[(WGHT, 900.0)]).outline(C),
        Err(FontError::BadTable)
    );
}

#[test]
fn bending_of_the_second_version_moves_an_axis_by_the_others() {
    let bytes = bent_by_width();
    let places = |settings: &[(Tag, f32)]| set(&bytes, settings).variations().to_vec();
    assert_eq!(places(&[]), [0, 0]);
    // At the narrow end weight is half the way to its heavy end, and
    // halfway there a quarter.
    assert_eq!(places(&[(WDTH, 50.0)]), [8_192, -16_384]);
    assert_eq!(places(&[(WDTH, 75.0)]), [4_096, -8_192]);
    assert_eq!(places(&[(WDTH, 150.0)]), [0, 16_384]);
    // After weight's own bending, and no further than its end.
    assert_eq!(places(&[(WGHT, 650.0), (WDTH, 75.0)]), [8_192, -8_192]);
    assert_eq!(places(&[(WGHT, 900.0), (WDTH, 50.0)]), [16_384, -16_384]);
    assert_eq!(places(&[(WGHT, 100.0), (WDTH, 50.0)]), [-8_192, -16_384]);
    assert_eq!(places(&[(WGHT, 250.0), (WDTH, 75.0)]), [-4_096, -8_192]);
    // The advances HarfBuzz gives the face there.
    let at = |settings: &[(Tag, f32)]| {
        let font = set(&bytes, settings);
        [A, B, SPACE, D].map(|glyph| font.advance(glyph).expect("advance"))
    };
    assert_eq!(at(&[]), [500, 600, 250, 700]);
    assert_eq!(at(&[(WDTH, 50.0)]), [450, 580, 250, 655]);
    assert_eq!(at(&[(WDTH, 75.0)]), [475, 590, 250, 678]);
    assert_eq!(at(&[(WDTH, 150.0)]), [500, 600, 250, 700]);
    assert_eq!(at(&[(WGHT, 650.0), (WDTH, 75.0)]), [500, 610, 263, 705]);
    assert_eq!(at(&[(WGHT, 900.0), (WDTH, 50.0)]), [500, 620, 275, 710]);
    assert_eq!(at(&[(WGHT, 100.0), (WDTH, 50.0)]), [400, 540, 225, 600]);
    assert_eq!(at(&[(WGHT, 250.0), (WDTH, 75.0)]), [450, 570, 238, 650]);
    // And its outlines, and the feature that draws another glyph for
    // `A` from halfway along the heavy side.
    let font = set(&bytes, &[(WDTH, 50.0)]);
    assert_eq!(drawn(&font, A), "M50,0L420,0L420,700L235,740L50,700L50,0Z");
    let table = font.gsub().expect("substitutions");
    assert_eq!(table.feature_lookups(0).expect("feature").len(), 1);
    assert_eq!(
        drawn(&set(&bytes, &[(WGHT, 650.0), (WDTH, 75.0)]), A),
        "M50,0L460,0L460,700L255,740L50,700L50,0Z"
    );
    assert_eq!(
        drawn(&set(&bytes, &[(WGHT, 100.0), (WDTH, 50.0)]), A),
        "M40,0L360,0L360,700L200,700L40,700L40,0Z"
    );
    // A table of that version cut short, or with a store that is not
    // one, cannot set the face, and one with no store bends as the
    // first version does.
    let whole = bent_by_width_table();
    for length in 30..whole.len() {
        let bytes = variable_with(b"avar", Some(whole[..length].to_vec()));
        let face = read(&bytes).expect("face");
        let _set = face.varied(&[(WDTH, 50.0)]);
    }
    let mut none = bends_of(2);
    none.extend([0; 8]);
    let bytes = variable_with(b"avar", Some(none));
    assert_eq!(
        set(&bytes, &[(WGHT, 650.0), (WDTH, 50.0)]).variations(),
        [4_096, -16_384]
    );
}
