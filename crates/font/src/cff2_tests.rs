//! Tests of a variable face of `CFF2` outlines built in code,
//! [`crate::fixture_cff2`].
//!
//! Each outline and advance an assertion expects is what HarfBuzz 14.2
//! gives for the fixture file: `hb-vector --variations=...` for outlines,
//! and `hb-shape --variations=... --shapers=fallback --show-extents` for
//! advances and for the boxes of the two glyphs no character maps to.

use crate::fixture_cff2::{Op, Parts, Selector, face, program, table, variable_cff2};
use crate::fixture_var::{A, A_HEAVY, B, C, D, MARK, SPACE};
use crate::{Font, FontError, Segment, Tag, pin_instance, read};

const CFF2: &[u8] = include_bytes!("../fixtures/cff2.otf");

const WGHT: Tag = *b"wght";
const WDTH: Tag = *b"wdth";

const CALLSUBR: u8 = 10;
const HLINETO: u8 = 6;
const VLINETO: u8 = 7;
const RMOVETO: u8 = 21;
const VSINDEX: u8 = 15;
const BLEND: u8 = 16;

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
                Segment::Cubic {
                    c1x,
                    c1y,
                    c2x,
                    c2y,
                    x,
                    y,
                } => write!(
                    path,
                    "C{},{} {},{} {},{}",
                    n(c1x),
                    n(c1y),
                    n(c2x),
                    n(c2y),
                    n(x),
                    n(y)
                ),
            }
            .expect("write");
        }
        path.push('Z');
    }
    path
}

/// A box from 50 to `right` across and 0 to 700 up.
fn boxed(right: f32) -> String {
    format!("M50,0L{right},0L{right},700L50,700L50,0Z")
}

#[test]
fn the_file_is_the_face_the_builder_makes() {
    assert_eq!(CFF2, variable_cff2());
}

#[test]
fn a_face_of_these_outlines_is_drawn_at_its_default() {
    let font = read(CFF2).expect("face");
    assert!(font.is_variable());
    assert!(font.has_outlines());
    assert_eq!(font.kind(), crate::FontKind::OpenType);
    assert_eq!(drawn(&font, 0), boxed(450.0));
    assert_eq!(drawn(&font, A), boxed(450.0));
    // A subroutine of the glyph's font dictionary.
    assert_eq!(drawn(&font, B), boxed(550.0));
    // Another dictionary's, and one every glyph shares.
    assert_eq!(
        drawn(&font, C),
        "M50,0L450,0L450,700L50,700L50,0ZM200,950L400,950L400,1100L200,1100L200,950Z"
    );
    // A program of no bytes draws nothing.
    assert_eq!(drawn(&font, SPACE), "");
    assert_eq!(
        drawn(&font, MARK),
        "M100,750L300,750L300,900L100,900L100,750Z"
    );
    // Hints and a hint mask are stepped over.
    assert_eq!(drawn(&font, A_HEAVY), boxed(470.0));
    assert_eq!(
        drawn(&font, D),
        "M50,0C350,0 500,100 650,350L650,700L50,700L50,0Z"
    );
    assert_eq!(font.outline(8), Err(FontError::UnknownGlyph));
}

#[test]
fn blends_follow_the_setting() {
    let at = |settings: &[(Tag, f32)], glyph: u16| drawn(&set(CFF2, settings), glyph);
    // Blended over both regions: the heavy end and the narrow one.
    assert_eq!(at(&[(WGHT, 900.0)], A), boxed(550.0));
    assert_eq!(at(&[(WGHT, 650.0)], A), boxed(475.0));
    assert_eq!(at(&[(WGHT, 800.0)], A), boxed(520.0));
    assert_eq!(at(&[(WGHT, 100.0)], A), boxed(450.0));
    assert_eq!(at(&[(WDTH, 50.0)], A), boxed(370.0));
    assert_eq!(at(&[(WDTH, 75.0)], A), boxed(410.0));
    assert_eq!(at(&[(WDTH, 150.0)], A), boxed(450.0));
    assert_eq!(at(&[(WGHT, 900.0), (WDTH, 50.0)], A), boxed(470.0));
    assert_eq!(at(&[(WGHT, 650.0), (WDTH, 75.0)], A), boxed(435.0));
    // A glyph that says it blends over the heavy end alone: width does
    // not move it.
    assert_eq!(at(&[(WGHT, 900.0)], B), boxed(630.0));
    assert_eq!(at(&[(WGHT, 800.0)], B), boxed(606.0));
    assert_eq!(at(&[(WDTH, 50.0)], B), boxed(550.0));
    assert_eq!(at(&[(WGHT, 900.0), (WDTH, 50.0)], B), boxed(630.0));
    // A font dictionary that says so for its glyphs.
    assert_eq!(
        at(&[(WGHT, 900.0)], C),
        "M50,0L550,0L550,700L50,700L50,0ZM230,1000L430,1000L430,1150L230,1150L230,1000Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0)], C),
        "M50,0L475,0L475,700L50,700L50,0ZM207.5,962.5L407.5,962.5L407.5,1112.5L207.5,1112.5L207.5,962.5Z"
    );
    assert_eq!(
        at(&[(WGHT, 800.0)], C),
        "M50,0L520,0L520,700L50,700L50,0ZM221,985L421,985L421,1135L221,1135L221,985Z"
    );
    assert_eq!(
        at(&[(WDTH, 50.0)], C),
        "M50,0L450,0L450,700L50,700L50,0ZM200,950L400,950L400,1100L200,1100L200,950Z"
    );
    // Two numbers blended at once.
    assert_eq!(
        at(&[(WGHT, 900.0)], MARK),
        "M100,770L300,770L300,920L100,920L100,770Z"
    );
    assert_eq!(
        at(&[(WGHT, 650.0)], MARK),
        "M100,755L300,755L300,905L100,905L100,755Z"
    );
    // After hints.
    assert_eq!(at(&[(WGHT, 900.0)], A_HEAVY), boxed(580.0));
    assert_eq!(at(&[(WDTH, 50.0)], A_HEAVY), boxed(380.0));
    // And glyphs with no blends stay.
    assert_eq!(
        at(&[(WGHT, 900.0), (WDTH, 50.0)], D),
        "M50,0C350,0 500,100 650,350L650,700L50,700L50,0Z"
    );
    assert_eq!(at(&[(WGHT, 900.0)], 0), boxed(450.0));
}

#[test]
fn advances_and_the_fingerprint_follow_the_setting_too() {
    let plain = read(CFF2).expect("face");
    let advances = |font: &Font<'_>| -> Vec<u16> {
        [A, B, C, SPACE, MARK, D]
            .iter()
            .map(|glyph| font.advance(*glyph).expect("advance"))
            .collect()
    };
    assert_eq!(advances(&plain), [500, 600, 500, 250, 0, 700]);
    let heavy = set(CFF2, &[(WGHT, 900.0)]);
    assert_eq!(advances(&heavy), [600, 680, 600, 300, 0, 810]);
    assert_eq!(
        advances(&set(CFF2, &[(WGHT, 650.0), (WDTH, 75.0)])),
        [475, 590, 475, 250, 0, 678]
    );
    assert_ne!(plain.fingerprint(), heavy.fingerprint());
    // Written as one instance, the face opens set to it.
    let pinned = pin_instance(CFF2, &[(WGHT, 900.0)]).expect("pinned");
    let pinned = read(&pinned).expect("face");
    assert_eq!(pinned.variations(), [16_384, 0]);
    for glyph in 0..8 {
        assert_eq!(drawn(&pinned, glyph), drawn(&heavy, glyph));
        assert_eq!(pinned.advance(glyph), heavy.advance(glyph));
    }
    // Set back, it is the face it was.
    let mut back = heavy.clone();
    back.set_variations(&[]).expect("set");
    assert_eq!(back.fingerprint(), plain.fingerprint());
    assert_eq!(drawn(&back, A), boxed(450.0));
}

#[test]
fn each_way_of_naming_a_glyphs_dictionary_is_read() {
    let ranges = read(CFF2).expect("face");
    for selector in [Selector::Each, Selector::Wide] {
        let mut parts = Parts::new();
        parts.selector = selector;
        let bytes = face(table(&parts));
        for settings in [&[][..], &[(WGHT, 900.0), (WDTH, 50.0)]] {
            let font = set(&bytes, settings);
            let same = ranges.varied(settings).expect("set");
            for glyph in 0..8 {
                assert_eq!(drawn(&font, glyph), drawn(&same, glyph));
            }
        }
    }
}

/// The face with glyph 1 the program `ops`, at the heavy end.
fn with_glyph(ops: &[Op]) -> Vec<u8> {
    let mut parts = Parts::new();
    parts.glyphs[1] = program(ops);
    face(table(&parts))
}

fn heavy_outline(bytes: &[u8]) -> Result<crate::Outline, FontError> {
    set(bytes, &[(WGHT, 900.0)]).outline(A)
}

#[test]
fn a_glyph_that_breaks_the_rules_of_blending_is_refused() {
    use Op::{N, O};
    let start = [N(50), N(0), O(RMOVETO)];
    let with = |rest: &[Op]| {
        let mut ops = start.to_vec();
        ops.extend_from_slice(rest);
        heavy_outline(&with_glyph(&ops))
    };
    // A blend of more values than the glyph holds numbers for.
    assert_eq!(
        with(&[N(400), N(100), N(2), O(BLEND), O(HLINETO)]),
        Err(FontError::BadTable)
    );
    assert_eq!(with(&[N(513), O(BLEND)]), Err(FontError::BadTable));
    assert_eq!(with(&[N(32_767), O(BLEND)]), Err(FontError::BadTable));
    assert_eq!(with(&[N(-1), O(BLEND)]), Err(FontError::BadTable));
    assert_eq!(with(&[O(BLEND)]), Err(FontError::BadTable));
    // Saying which group to blend over after a blend, or with no number.
    assert_eq!(
        with(&[
            N(400),
            N(100),
            N(-80),
            N(1),
            O(BLEND),
            N(1),
            O(VSINDEX),
            O(HLINETO)
        ]),
        Err(FontError::BadTable)
    );
    assert_eq!(with(&[O(VSINDEX)]), Err(FontError::BadTable));
    assert_eq!(with(&[N(-1), O(VSINDEX)]), Err(FontError::BadTable));
    // A group the store lacks has no regions: a blend over it is its
    // defaults.
    let outline = with(&[
        N(7),
        O(VSINDEX),
        N(400),
        N(1),
        O(BLEND),
        O(HLINETO),
        N(700),
        O(VLINETO),
    ])
    .expect("outline");
    assert_eq!(outline.contours.len(), 1);
    // More numbers than a glyph may hold at once.
    let mut many = start.to_vec();
    many.extend(std::iter::repeat_n(N(1), 514));
    assert_eq!(heavy_outline(&with_glyph(&many)), Err(FontError::BadTable));
    // A line before any move.
    assert_eq!(
        heavy_outline(&with_glyph(&[N(400), O(HLINETO)])),
        Err(FontError::BadTable)
    );
    // The other glyphs of such a face are still drawn.
    let bytes = with_glyph(&[O(BLEND)]);
    assert_eq!(drawn(&set(&bytes, &[(WGHT, 900.0)]), B), boxed(630.0));
}

#[test]
fn subroutines_that_call_without_end_are_cut_short() {
    use Op::{N, O};
    let started = std::time::Instant::now();
    // One that calls itself.
    let mut parts = Parts::new();
    parts.first = vec![program(&[N(-107), O(CALLSUBR)])];
    parts.glyphs[1] = program(&[N(-107), O(CALLSUBR)]);
    let bytes = face(table(&parts));
    assert_eq!(heavy_outline(&bytes), Err(FontError::BadTable));
    // Ten, each calling the next a hundred times, the last doing
    // nothing: a hundred to the ninth calls, no deeper than is allowed.
    let mut parts = Parts::new();
    parts.first = (0..10_i16)
        .map(|level| {
            if level == 9 {
                Vec::new()
            } else {
                let call = [N(level + 1 - 107), O(CALLSUBR)];
                program(&call.repeat(100))
            }
        })
        .collect();
    parts.glyphs[1] = program(&[N(-107), O(CALLSUBR)]);
    let bytes = face(table(&parts));
    assert_eq!(heavy_outline(&bytes), Err(FontError::BadTable));
    // A long program that does end is drawn.
    let mut long = vec![N(50), N(0), O(RMOVETO)];
    for _ in 0..7_000 {
        long.extend([N(1), O(HLINETO), N(1), O(VLINETO)]);
    }
    let outline = heavy_outline(&with_glyph(&long)).expect("outline");
    assert!(outline.contours[0].segments.len() >= 14_000);
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
}

/// Ask a face for every glyph at a few settings. Nothing is expected but
/// an answer.
fn exercise(bytes: &[u8]) {
    let Ok(plain) = read(bytes) else {
        return;
    };
    for settings in [&[][..], &[(WGHT, 900.0), (WDTH, 50.0)], &[(WGHT, 650.0)]] {
        let Ok(font) = plain.varied(settings) else {
            continue;
        };
        for glyph in 0..font.glyph_count() {
            let _outline = font.outline(glyph);
            let _advance = font.advance(glyph);
        }
    }
}

#[test]
fn a_torn_or_scrambled_table_is_answered_without_a_panic() {
    let started = std::time::Instant::now();
    let whole = table(&Parts::new());
    for length in 0..whole.len() {
        exercise(&face(whole[..length].to_vec()));
    }
    for at in 0..whole.len() {
        for value in [0x00, 0x01, 0x10, 0x7F, 0x80, 0xFF] {
            let mut table = whole.clone();
            table[at] = value;
            exercise(&face(table));
        }
    }
    assert!(started.elapsed() < std::time::Duration::from_secs(120));
}

#[test]
fn a_table_that_does_not_read_leaves_the_face_open_with_nothing_to_draw() {
    // Another version, no font dictionaries, and a store cut short: the
    // face opens, as it did before these outlines were read.
    let whole = table(&Parts::new());
    let mut other = whole.clone();
    other[0] = 3;
    let mut bare = whole.clone();
    // The third entry of the top dictionary, where the font dictionaries
    // are, becomes a number with nothing it is for.
    bare[5 + 12 + 5] = 139;
    bare[5 + 12 + 6] = 139;
    for table in [other, bare, whole[..40].to_vec()] {
        let bytes = face(table);
        let font = read(&bytes).expect("face");
        assert!(!font.has_outlines());
        assert_eq!(font.outline(A), Err(FontError::Unsupported));
        // Its advances are still read, and still vary.
        assert_eq!(font.advance(A), Ok(500));
        let heavy = font.varied(&[(WGHT, 900.0)]).expect("set");
        assert_eq!(heavy.advance(A), Ok(600));
    }
    // With no axes the same outlines are drawn at their defaults.
    let count = usize::from(u16::from_be_bytes([CFF2[4], CFF2[5]]));
    let mut tables = Vec::new();
    for slot in 0..count {
        let record = &CFF2[12 + slot * 16..28 + slot * 16];
        let number = |at: usize| {
            u32::from_be_bytes([record[at], record[at + 1], record[at + 2], record[at + 3]])
                as usize
        };
        let tag: [u8; 4] = record[..4].try_into().expect("tag");
        if &tag != b"fvar" {
            tables.push((tag, 0, CFF2[number(8)..number(8) + number(12)].to_vec()));
        }
    }
    let fixed = crate::faces::assemble(0x4F54_544F, &tables).expect("font");
    let font = read(&fixed).expect("face");
    assert!(!font.is_variable());
    assert_eq!(drawn(&font, A), boxed(450.0));
    assert_eq!(
        drawn(&font, C),
        "M50,0L450,0L450,700L50,700L50,0ZM200,950L400,950L400,1100L200,1100L200,950Z"
    );
}
