//! Outlines of a CFF face keyed by character ids, as the faces of Han
//! characters are stored: glyphs in groups, each group with subroutines
//! of its own. The face is the OpenType sample with a `CFF ` table built
//! here in place of its own.

use crate::{FontError, Segment, read};

const SAMPLE: &[u8] = include_bytes!("../fixtures/sample.otf");

/// An `INDEX` with two-byte offsets.
fn index(items: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = u16::try_from(items.len())
        .expect("count")
        .to_be_bytes()
        .to_vec();
    if items.is_empty() {
        return bytes;
    }
    bytes.push(2);
    let mut at = 1_u16;
    bytes.extend(at.to_be_bytes());
    for item in items {
        at += u16::try_from(item.len()).expect("item");
        bytes.extend(at.to_be_bytes());
    }
    for item in items {
        bytes.extend(item);
    }
    bytes
}

/// A number in a dictionary, always five bytes so offsets can be known
/// before the tables are laid out.
fn number(value: usize) -> Vec<u8> {
    let mut bytes = vec![29];
    bytes.extend(i32::try_from(value).expect("number").to_be_bytes());
    bytes
}

/// A subroutine that moves to (`at`, `at`) and draws a square of `side`.
/// Each number is one byte, or two from 108.
fn square(at: i16, side: i16) -> Vec<u8> {
    let operand = |value: i16| -> Vec<u8> {
        match value {
            -107..=107 => vec![u8::try_from(value + 139).expect("small")],
            108..=1_131 => {
                let rest = value - 108;
                vec![
                    u8::try_from(247 + rest / 256).expect("high"),
                    u8::try_from(rest % 256).expect("low"),
                ]
            }
            _ => {
                let rest = -value - 108;
                vec![
                    u8::try_from(251 + rest / 256).expect("high"),
                    u8::try_from(rest % 256).expect("low"),
                ]
            }
        }
    };
    let mut program = Vec::new();
    program.extend(operand(at));
    program.extend(operand(at));
    program.push(21);
    program.extend(operand(side));
    program.push(6);
    program.extend(operand(side));
    program.push(7);
    program.extend(operand(-side));
    program.extend([6, 11]);
    program
}

/// How the glyphs are assigned to the two font dictionaries.
enum Chosen {
    /// Format 0, a byte for each of the four glyphs.
    Each([u8; 4]),
    /// Format 3: ranges of a first glyph and a dictionary, and the glyph
    /// after the last.
    Ranges(Vec<(u16, u8)>, u16),
    /// Bytes as they are.
    Raw(Vec<u8>),
}

/// A `CFF ` table of four glyphs: 0 is empty, 1 and 2 call the first
/// local subroutine, and 3 the first global one. The first dictionary's
/// subroutine draws a square of 100 at (100, 100), the second's one of
/// 300 at (300, 300), and the global one of 50 at (50, 50).
fn table(chosen: &Chosen, dictionaries: usize) -> Vec<u8> {
    let call_local = vec![32, 10, 14];
    let charstrings = index(&[vec![14], call_local.clone(), call_local, vec![32, 29, 14]]);
    let select = match chosen {
        Chosen::Each(groups) => {
            let mut bytes = vec![0];
            bytes.extend(groups);
            bytes
        }
        Chosen::Ranges(ranges, end) => {
            let mut bytes = vec![3];
            bytes.extend(u16::try_from(ranges.len()).expect("ranges").to_be_bytes());
            for (first, group) in ranges {
                bytes.extend(first.to_be_bytes());
                bytes.push(*group);
            }
            bytes.extend(end.to_be_bytes());
            bytes
        }
        Chosen::Raw(bytes) => bytes.clone(),
    };
    let subrs = [index(&[square(100, 100)]), index(&[square(300, 300)])];
    // A private dictionary is the offset of its subroutines from its own
    // start and the operator for that: six bytes.
    const PRIVATE: usize = 6;
    // A font dictionary is a private dictionary's size and place: eleven.
    const FONT: usize = 11;
    let mut start = vec![1, 0, 4, 2];
    start.extend(index(&[b"Keyed".to_vec()]));
    // The top dictionary: registry, ordering and supplement, then where
    // the charstrings, the dictionaries and the selector are.
    let top_length = 5 + 6 + 7 + 7;
    let globals = index(&[square(50, 50)]);
    let charstrings_at = start.len() + (2 + 1 + 4 + top_length) + 2 + globals.len();
    let select_at = charstrings_at + charstrings.len();
    let array_at = select_at + select.len();
    let array_length = if dictionaries == 0 {
        2
    } else {
        2 + 1 + (dictionaries + 1) * 2 + dictionaries * FONT
    };
    let privates_at = array_at + array_length;
    let subrs_at = privates_at + 2 * PRIVATE;
    let mut top = vec![139, 139, 139, 12, 30];
    top.extend(number(charstrings_at));
    top.push(17);
    top.extend(number(array_at));
    top.extend([12, 36]);
    top.extend(number(select_at));
    top.extend([12, 37]);
    assert_eq!(top.len(), top_length);
    let mut fonts = Vec::new();
    let mut privates = Vec::new();
    let mut after = subrs_at;
    for (which, subr) in subrs.iter().enumerate() {
        let private_at = privates_at + which * PRIVATE;
        let mut font = number(PRIVATE);
        font.extend(number(private_at));
        font.push(18);
        fonts.push(font);
        privates.extend(number(after - private_at));
        privates.push(19);
        after += subr.len();
    }
    fonts.truncate(dictionaries);
    let mut bytes = start;
    bytes.extend(index(&[top]));
    bytes.extend(index(&[]));
    bytes.extend(globals);
    assert_eq!(bytes.len(), charstrings_at);
    bytes.extend(charstrings);
    bytes.extend(select);
    assert_eq!(bytes.len(), array_at);
    bytes.extend(index(&fonts));
    assert_eq!(bytes.len(), privates_at);
    bytes.extend(privates);
    for subr in subrs {
        bytes.extend(subr);
    }
    bytes
}

/// The sample face with `cff` as its `CFF ` table: the table goes at the
/// end of the file and the directory names it there.
fn face(cff: &[u8]) -> Vec<u8> {
    let mut bytes = SAMPLE.to_vec();
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let count = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
    let record = (0..count)
        .map(|index| 12 + index * 16)
        .find(|at| &bytes[*at..*at + 4] == b"CFF ")
        .expect("CFF");
    let at = u32::try_from(bytes.len()).expect("offset");
    let length = u32::try_from(cff.len()).expect("length");
    bytes[record + 8..record + 12].copy_from_slice(&at.to_be_bytes());
    bytes[record + 12..record + 16].copy_from_slice(&length.to_be_bytes());
    bytes.extend(cff);
    bytes
}

/// The corners of a glyph's one contour.
fn corners(bytes: &[u8], glyph: u16) -> Vec<(f32, f32)> {
    let font = read(bytes).expect("face");
    let outline = font.outline(glyph).expect("outline");
    assert_eq!(outline.contours.len(), 1, "glyph {glyph}");
    let contour = &outline.contours[0];
    let mut points = vec![(contour.x, contour.y)];
    for segment in &contour.segments {
        let Segment::Line { x, y } = *segment else {
            panic!("a curve in glyph {glyph}");
        };
        points.push((x, y));
    }
    points.truncate(4);
    points
}

fn square_at(at: f32, side: f32) -> Vec<(f32, f32)> {
    vec![
        (at, at),
        (at + side, at),
        (at + side, at + side),
        (at, at + side),
    ]
}

#[test]
fn each_glyph_calls_the_subroutines_of_its_own_group() {
    for chosen in [
        Chosen::Each([0, 0, 1, 1]),
        Chosen::Ranges(vec![(0, 0), (2, 1)], 4),
    ] {
        let bytes = face(&table(&chosen, 2));
        let font = read(&bytes).expect("face");
        assert!(font.has_outlines());
        assert!(font.outline(0).expect("empty").contours.is_empty());
        // The same program draws what its group's subroutine draws.
        assert_eq!(corners(&bytes, 1), square_at(100.0, 100.0));
        assert_eq!(corners(&bytes, 2), square_at(300.0, 300.0));
        // The global subroutines are shared by every group.
        assert_eq!(corners(&bytes, 3), square_at(50.0, 50.0));
    }
    // The groups the other way round, and a range for every glyph.
    let bytes = face(&table(&Chosen::Each([1, 1, 0, 0]), 2));
    assert_eq!(corners(&bytes, 1), square_at(300.0, 300.0));
    assert_eq!(corners(&bytes, 2), square_at(100.0, 100.0));
    let each = Chosen::Ranges(vec![(0, 1), (1, 0), (2, 1), (3, 0)], 4);
    let bytes = face(&table(&each, 2));
    assert_eq!(corners(&bytes, 1), square_at(100.0, 100.0));
    assert_eq!(corners(&bytes, 2), square_at(300.0, 300.0));
}

#[test]
fn a_hostile_selector_or_dictionary_list_is_an_error() {
    let outline = |chosen: &Chosen, dictionaries: usize, glyph: u16| {
        let bytes = face(&table(chosen, dictionaries));
        read(&bytes).and_then(|font| font.outline(glyph).map(|_| ()))
    };
    // A group the face has no dictionary for.
    assert_eq!(
        outline(&Chosen::Each([0, 0, 9, 0]), 2, 2),
        Err(FontError::BadTable)
    );
    assert_eq!(outline(&Chosen::Each([0, 0, 9, 0]), 2, 1), Ok(()));
    assert_eq!(
        outline(&Chosen::Each([0, 0, 1, 1]), 1, 2),
        Err(FontError::BadTable)
    );
    // Ranges that start after the glyph, and that end before it.
    let late = Chosen::Ranges(vec![(2, 0), (3, 1)], 4);
    assert_eq!(outline(&late, 2, 1), Err(FontError::BadTable));
    assert_eq!(outline(&late, 2, 2), Ok(()));
    let early = Chosen::Ranges(vec![(0, 0), (1, 1)], 2);
    assert_eq!(outline(&early, 2, 2), Err(FontError::BadTable));
    assert_eq!(outline(&early, 2, 1), Ok(()));
    // No ranges, a format this reader does not know, and a selector
    // shorter than it says: the face does not open.
    let opened = |chosen: &Chosen, dictionaries: usize| {
        read(&face(&table(chosen, dictionaries))).map(|_| ())
    };
    assert_eq!(
        opened(&Chosen::Ranges(Vec::new(), 4), 2),
        Err(FontError::BadTable)
    );
    assert_eq!(
        opened(&Chosen::Raw(vec![9, 0, 0, 0, 0]), 2),
        Err(FontError::Unsupported)
    );
    assert_eq!(
        opened(&Chosen::Each([0, 0, 1, 1]), 0),
        Err(FontError::BadTable)
    );
    // Every cut of the table opens or is an error, and so is every glyph.
    let whole = table(&Chosen::Ranges(vec![(0, 0), (2, 1)], 4), 2);
    for length in 0..whole.len() {
        let bytes = face(&whole[..length]);
        if let Ok(font) = read(&bytes) {
            for glyph in 0..4 {
                let _ = font.outline(glyph);
            }
        }
    }
}

#[test]
fn a_glyph_of_a_keyed_face_is_outlined_quickly() {
    let bytes = face(&table(&Chosen::Ranges(vec![(0, 0), (2, 1)], 4), 2));
    let font = read(&bytes).expect("face");
    let started = std::time::Instant::now();
    for round in 0..100_000_u32 {
        let glyph = u16::try_from(round % 4).expect("glyph");
        assert!(font.outline(glyph).is_ok());
    }
    // A search of the ranges and a short program each time.
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}
