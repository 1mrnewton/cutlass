//! A variable font of `CFF2` outlines built in code, for tests.
//!
//! `cff2.otf` has the axes, the bending of weight, the names and the
//! advances of [`crate::fixture_var`]'s face, and the same eight glyphs
//! drawn another way: as programs whose numbers are blends. The table's
//! store has two groups of regions: the heavy end of weight with the
//! narrow end of width, and the heavy end alone. Its glyphs:
//!
//! - 0, the missing glyph: a box that does not vary;
//! - 1, `A`: a box that widens by 100 at the heavy end and narrows by 80
//!   at the narrow one, by blends over the first group;
//! - 2, `B`: says it blends over the second group and calls a subroutine
//!   of its font dictionary, a box that widens by 80 at the heavy end;
//! - 3, `C`: of the second font dictionary, whose blends are over the
//!   second group and which has a subroutine of its own for a box, with
//!   a mark over it from a subroutine every glyph shares; the mark's
//!   place moves;
//! - 4, the space: a program of no bytes;
//! - 5, the mark: its start blended two numbers at once, the box from
//!   the shared subroutine;
//! - 6, the other `A`: with hints and a hint mask before its path;
//! - 7, `D`: a curve.

use crate::fixture::be;
use crate::fixture_var::variable;

/// 1 in the 2.14 fixed-point form places along an axis are stored in.
const ONE: i16 = 16_384;

const RMOVETO: u8 = 21;
const HLINETO: u8 = 6;
const VLINETO: u8 = 7;
const RRCURVETO: u8 = 8;
const CALLSUBR: u8 = 10;
const CALLGSUBR: u8 = 29;
const VSINDEX: u8 = 15;
const BLEND: u8 = 16;
const HSTEM: u8 = 1;
const VSTEM: u8 = 3;
const HINTMASK: u8 = 19;

/// One piece of a glyph's program, or of a dictionary.
#[derive(Clone, Copy)]
pub(crate) enum Op {
    /// A number.
    N(i16),
    /// An operator of one byte.
    O(u8),
    /// A byte as it is: the bits of a hint mask.
    Raw(u8),
}

use Op::{N, O, Raw};

/// The bytes of a program.
pub(crate) fn program(ops: &[Op]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for op in ops {
        match *op {
            N(value @ -107..=107) => bytes.push(u8::try_from(value + 139).expect("small")),
            N(value) => {
                bytes.push(28);
                bytes.extend(value.to_be_bytes());
            }
            O(operator) | Raw(operator) => bytes.push(operator),
        }
    }
    bytes
}

/// An `INDEX` as a `CFF2` table has one: a count of four bytes, and
/// offsets of two.
pub(crate) fn index(items: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = u32::try_from(items.len())
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
fn offset(value: usize) -> Vec<u8> {
    let mut bytes = vec![29];
    bytes.extend(i32::try_from(value).expect("offset").to_be_bytes());
    bytes
}

/// A box from where the pen is: right `width`, up `height`, and back.
fn boxed(width: i16, height: i16) -> [Op; 6] {
    [
        N(width),
        O(HLINETO),
        N(height),
        O(VLINETO),
        N(-width),
        O(HLINETO),
    ]
}

/// The programs of the eight glyphs.
fn glyphs() -> Vec<Vec<u8>> {
    let mut notdef = vec![N(50), N(0), O(RMOVETO)];
    notdef.extend(boxed(400, 700));
    let a = [
        N(50),
        N(0),
        O(RMOVETO),
        // 400, and 100 more at the heavy end, 80 less at the narrow.
        N(400),
        N(100),
        N(-80),
        N(1),
        O(BLEND),
        O(HLINETO),
        N(700),
        O(VLINETO),
        N(-400),
        N(-100),
        N(80),
        N(1),
        O(BLEND),
        O(HLINETO),
    ];
    let b = [N(1), O(VSINDEX), N(-107), O(CALLSUBR)];
    let c = [
        N(-107),
        O(CALLSUBR),
        // Up to the mark: two blends of one number each.
        N(150),
        N(30),
        N(1),
        O(BLEND),
        N(250),
        N(50),
        N(1),
        O(BLEND),
        O(RMOVETO),
        N(-107),
        O(CALLGSUBR),
    ];
    let mark = [
        // Two numbers blended at once: each default, then each one's
        // change at the heavy end and at the narrow.
        N(100),
        N(750),
        N(0),
        N(0),
        N(20),
        N(0),
        N(2),
        O(BLEND),
        O(RMOVETO),
        N(-107),
        O(CALLGSUBR),
    ];
    let other = [
        N(0),
        N(700),
        O(HSTEM),
        N(50),
        N(420),
        O(VSTEM),
        O(HINTMASK),
        Raw(0xC0),
        N(50),
        N(0),
        O(RMOVETO),
        N(420),
        N(110),
        N(-90),
        N(1),
        O(BLEND),
        O(HLINETO),
        N(700),
        O(VLINETO),
        N(-420),
        N(-110),
        N(90),
        N(1),
        O(BLEND),
        O(HLINETO),
    ];
    let d = [
        N(50),
        N(0),
        O(RMOVETO),
        N(300),
        N(0),
        N(150),
        N(100),
        N(150),
        N(250),
        O(RRCURVETO),
        N(350),
        O(VLINETO),
        N(-600),
        O(HLINETO),
    ];
    vec![
        program(&notdef),
        program(&a),
        program(&b),
        program(&c),
        Vec::new(),
        program(&mark),
        program(&other),
        program(&d),
    ]
}

/// The subroutine every glyph shares: a box 200 by 150 from the pen.
fn shared() -> Vec<u8> {
    program(&boxed(200, 150))
}

/// The first font dictionary's subroutine: a box that widens by 80 at
/// the heavy end, blended over the second group.
fn first_subroutine() -> Vec<u8> {
    program(&[
        N(50),
        N(0),
        O(RMOVETO),
        N(500),
        N(80),
        N(1),
        O(BLEND),
        O(HLINETO),
        N(700),
        O(VLINETO),
        N(-500),
        N(-80),
        N(1),
        O(BLEND),
        O(HLINETO),
    ])
}

/// The second font dictionary's: a box that widens by 100.
fn second_subroutine() -> Vec<u8> {
    program(&[
        N(50),
        N(0),
        O(RMOVETO),
        N(400),
        N(100),
        N(1),
        O(BLEND),
        O(HLINETO),
        N(700),
        O(VLINETO),
        N(-400),
        N(-100),
        N(1),
        O(BLEND),
        O(HLINETO),
    ])
}

/// The table's store: the regions of the heavy end of weight and of the
/// narrow end of width, and two groups with no rows, the first over both
/// regions and the second over the heavy end alone. Its length in two
/// bytes goes before it.
fn store() -> Vec<u8> {
    let regions: [[(i16, i16, i16); 2]; 2] =
        [[(0, ONE, ONE), (0, 0, 0)], [(0, 0, 0), (-ONE, -ONE, 0)]];
    let mut list = be(&[2, 2]);
    for region in regions {
        for (start, peak, end) in region {
            for value in [start, peak, end] {
                list.extend(value.to_be_bytes());
            }
        }
    }
    let groups = [be(&[0, 0, 2, 0, 1]), be(&[0, 0, 1, 0])];
    // The header and the two groups' offsets, the regions, the groups.
    let mut table = be(&[1]);
    table.extend(16_u32.to_be_bytes());
    table.extend(be(&[2]));
    let first = 16 + list.len();
    for at in [first, first + groups[0].len()] {
        table.extend(u32::try_from(at).expect("offset").to_be_bytes());
    }
    table.extend(list);
    for group in groups {
        table.extend(group);
    }
    let mut sized = be(&[u16::try_from(table.len()).expect("length")]);
    sized.extend(table);
    sized
}

/// How the table says which font dictionary each glyph has.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selector {
    /// A byte for each glyph.
    Each,
    /// Ranges of glyphs, in two bytes.
    Ranges,
    /// Ranges of glyphs, in four bytes.
    Wide,
}

/// The table that gives glyph 3 the second font dictionary and every
/// other glyph the first.
fn selector(kind: Selector) -> Vec<u8> {
    match kind {
        Selector::Each => vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
        Selector::Ranges => {
            let mut table = vec![3, 0, 3];
            for (first, dictionary) in [(0_u16, 0_u8), (3, 1), (4, 0)] {
                table.extend(first.to_be_bytes());
                table.push(dictionary);
            }
            table.extend(8_u16.to_be_bytes());
            table
        }
        Selector::Wide => {
            let mut table = vec![4];
            table.extend(3_u32.to_be_bytes());
            for (first, dictionary) in [(0_u32, 0_u16), (3, 1), (4, 0)] {
                table.extend(first.to_be_bytes());
                table.extend(dictionary.to_be_bytes());
            }
            table.extend(8_u32.to_be_bytes());
            table
        }
    }
}

/// What a `CFF2` table is built of, for a test to change one part.
pub(crate) struct Parts {
    pub(crate) glyphs: Vec<Vec<u8>>,
    pub(crate) shared: Vec<Vec<u8>>,
    pub(crate) first: Vec<Vec<u8>>,
    pub(crate) second: Vec<Vec<u8>>,
    pub(crate) selector: Selector,
}

impl Parts {
    pub(crate) fn new() -> Self {
        Self {
            glyphs: glyphs(),
            shared: vec![shared()],
            first: vec![first_subroutine()],
            second: vec![second_subroutine()],
            selector: Selector::Ranges,
        }
    }
}

/// The `CFF2` table of `parts`.
pub(crate) fn table(parts: &Parts) -> Vec<u8> {
    let shared = index(&parts.shared);
    let store = store();
    let glyphs = index(&parts.glyphs);
    let select = selector(parts.selector);
    let subroutines = [index(&parts.first), index(&parts.second)];
    // The first private dictionary has hinting numbers of which one is a
    // blend, and says where its subroutines are, from its own start;
    // the second says too that its glyphs blend over the second group.
    let mut first = program(&[N(0), N(10), N(1), N(1), N(1), O(23), O(6)]);
    first.extend(offset(first.len() + 6));
    first.push(19);
    let mut second = program(&[N(1), O(22)]);
    second.extend(offset(second.len() + 6));
    second.push(19);
    let privates = [first, second];
    // The top dictionary is four offsets of five bytes and what each is:
    // the glyphs, the store, the font dictionaries, the selector.
    let top_length = 4 * 5 + 2 + 2 * 2;
    let store_at = 5 + top_length + shared.len();
    let glyphs_at = store_at + store.len();
    // A font dictionary is the size and the place of its private one.
    let dictionary_length = 5 + 5 + 1;
    let dictionaries_at = glyphs_at + glyphs.len();
    let select_at = dictionaries_at + 4 + 1 + 3 * 2 + 2 * dictionary_length;
    let mut private_at = select_at + select.len();
    let mut dictionaries = Vec::new();
    let mut tail: Vec<u8> = Vec::new();
    for (private, subroutines) in privates.iter().zip(&subroutines) {
        let mut dictionary = offset(private.len());
        dictionary.extend(offset(private_at));
        dictionary.push(18);
        dictionaries.push(dictionary);
        tail.extend(private);
        tail.extend(subroutines);
        private_at += private.len() + subroutines.len();
    }
    let mut table = vec![2, 0, 5];
    table.extend(u16::try_from(top_length).expect("length").to_be_bytes());
    table.extend(offset(glyphs_at));
    table.push(17);
    table.extend(offset(store_at));
    table.push(24);
    table.extend(offset(dictionaries_at));
    table.extend([12, 36]);
    table.extend(offset(select_at));
    table.extend([12, 37]);
    table.extend(shared);
    table.extend(store);
    table.extend(glyphs);
    table.extend(index(&dictionaries));
    table.extend(select);
    table.extend(tail);
    table
}

/// The face of [`crate::fixture_var`] with `cff2` for its outlines: its
/// character map, metrics, names, axes and advances, and none of its
/// TrueType outlines or their moves.
pub(crate) fn face(cff2: Vec<u8>) -> Vec<u8> {
    let truetype = variable();
    let count = usize::from(u16::from_be_bytes([truetype[4], truetype[5]]));
    let mut tables: Vec<([u8; 4], u32, Vec<u8>)> = Vec::new();
    for slot in 0..count {
        let record = &truetype[12 + slot * 16..28 + slot * 16];
        let number = |at: usize| {
            u32::from_be_bytes([record[at], record[at + 1], record[at + 2], record[at + 3]])
                as usize
        };
        let tag: [u8; 4] = record[..4].try_into().expect("tag");
        if [
            b"cmap", b"head", b"hhea", b"hmtx", b"name", b"fvar", b"avar", b"HVAR",
        ]
        .contains(&&tag)
        {
            tables.push((tag, 0, truetype[number(8)..number(8) + number(12)].to_vec()));
        }
    }
    // The short form of the table of counts, for a face with no
    // TrueType outlines.
    tables.push((*b"maxp", 0, vec![0, 0, 0x50, 0, 0, 8]));
    tables.push((*b"CFF2", 0, cff2));
    tables.sort_by_key(|table| table.0);
    crate::faces::assemble(0x4F54_544F, &tables).expect("font")
}

/// The face.
pub(crate) fn variable_cff2() -> Vec<u8> {
    face(table(&Parts::new()))
}

fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

#[test]
fn the_cff2_fixture_file_is_what_the_builder_writes() {
    let stored = std::fs::read(path("cff2.otf")).expect("fixture");
    assert!(
        stored == variable_cff2(),
        "cff2.otf is stale: write the fixtures again"
    );
    let face = crate::read(&stored).expect("font");
    assert!(face.is_variable());
    assert!(face.has_outlines());
}

#[test]
#[ignore = "writes the CFF2 fixture font"]
fn write_the_cff2_fixture_font() {
    std::fs::write(path("cff2.otf"), variable_cff2()).expect("write");
}
