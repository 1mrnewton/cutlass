//! Variable fonts built in code, for tests.
//!
//! `variable.ttf` varies along two axes: weight from 100 through 400 to
//! 900, and width from 50 through 100 to 150. Weight is bent so that
//! halfway to its heavy end counts as a quarter of the way. Its glyphs:
//!
//! - 0, the missing glyph, a box that does not vary;
//! - 1, `A`: five points, of which the heavy tuple moves two and leaves
//!   three to follow them; a tuple for the narrow end moves every point;
//!   one for the middle of the heavy side lifts the top middle point;
//!   and one for the heavy narrow corner moves two points more;
//! - 2, `B`: a box whose tuples share one list of point numbers;
//! - 3, `C`: `A` with the mark over it, as parts; the mark's place moves,
//!   and the glyph has the metrics of its `A`;
//! - 4, the space: no points, an advance that varies;
//! - 5, the mark: a small box that rises with weight;
//! - 6, the `A` the face draws in its place from the middle of the heavy
//!   side on;
//! - 7, `D`: a box past the end of the map of advances.
//!
//! Advances vary by `HVAR`; kerning of `A` before `B` and the anchor of
//! the mark on `A` vary by the store in `GDEF`; and the feature `rvrn`
//! draws glyph 6 for `A` where weight is past the middle of its heavy
//! side. `variable-edges.ttf` is the same face without `HVAR`, so its
//! advances are what its glyphs' edges give.

use crate::fixture::{
    Glyph, LookupSpec, Off, V, be, boxed, classes, coverage, font, layout_table, signed, single,
    table,
};

pub(crate) const A: u16 = 1;
pub(crate) const B: u16 = 2;
pub(crate) const C: u16 = 3;
pub(crate) const SPACE: u16 = 4;
pub(crate) const MARK: u16 = 5;
pub(crate) const A_HEAVY: u16 = 6;
pub(crate) const D: u16 = 7;

/// 1 in the 2.14 fixed-point form places along an axis are stored in.
const ONE: i16 = 16_384;

fn count(length: usize) -> u16 {
    u16::try_from(length).expect("count")
}

fn long(value: usize) -> [u8; 4] {
    u32::try_from(value).expect("offset").to_be_bytes()
}

fn signed_bytes(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

/// A glyph of one contour through `points`, each on the curve.
fn simple(points: &[(i16, i16)]) -> Vec<u8> {
    let xs = points.iter().map(|point| point.0);
    let ys = points.iter().map(|point| point.1);
    let mut record = signed_bytes(&[
        1,
        xs.clone().min().expect("points"),
        ys.clone().min().expect("points"),
        xs.clone().max().expect("points"),
        ys.clone().max().expect("points"),
    ]);
    record.extend(be(&[count(points.len()) - 1, 0]));
    record.extend(std::iter::repeat_n(1, points.len()));
    // Each coordinate is stored as its distance from the one before.
    for coordinates in [xs.collect::<Vec<_>>(), ys.collect::<Vec<_>>()] {
        let mut last = 0;
        for coordinate in coordinates {
            record.extend((coordinate - last).to_be_bytes());
            last = coordinate;
        }
    }
    record
}

/// Where a part of a glyph goes.
#[derive(Clone, Copy)]
pub(crate) enum Place {
    /// So far right and up.
    At(i16, i16),
    /// Where a point of the part, the second number, lands on a point of
    /// the glyph so far, the first.
    Matched(u16, u16),
}

/// One part of a glyph of parts: a glyph, where it goes, whether the
/// whole takes its metrics, and how much it is scaled by, as a 2.14
/// number.
pub(crate) type Part = (u16, Place, bool, Option<i16>);

/// A glyph of parts.
pub(crate) fn composite(corners: [i16; 4], parts: &[Part]) -> Vec<u8> {
    let mut record = signed_bytes(&[-1, corners[0], corners[1], corners[2], corners[3]]);
    for (index, (glyph, place, metrics, scale)) in parts.iter().enumerate() {
        // The place is two words: a place, or two points to match.
        let mut flags = match place {
            Place::At(..) => 0x0003,
            Place::Matched(..) => 0x0001,
        };
        if index + 1 < parts.len() {
            flags |= 0x0020;
        }
        if *metrics {
            flags |= 0x0200;
        }
        if scale.is_some() {
            flags |= 0x0008;
        }
        let (first, second) = match *place {
            Place::At(x, y) => (signed(x), signed(y)),
            Place::Matched(onto, point) => (onto, point),
        };
        record.extend(be(&[flags, *glyph, first, second]));
        if let Some(scale) = scale {
            record.extend(scale.to_be_bytes());
        }
    }
    record
}

/// The points of `A`.
pub(crate) const A_POINTS: [(i16, i16); 5] = [(50, 0), (450, 0), (450, 700), (250, 700), (50, 700)];

/// The `glyf`, `loca` and `hmtx` tables.
fn outlines() -> [(&'static [u8; 4], Vec<u8>); 3] {
    outlines_of(composite(
        [50, 0, 450, 1100],
        &[
            (A, Place::At(0, 0), true, None),
            (MARK, Place::At(100, 200), false, None),
        ],
    ))
}

/// The same tables with `made` for `C`, the glyph of parts.
pub(crate) fn outlines_of(made: Vec<u8>) -> [(&'static [u8; 4], Vec<u8>); 3] {
    let square = |left: i16, bottom: i16, right: i16, top: i16| {
        simple(&[(left, bottom), (right, bottom), (right, top), (left, top)])
    };
    // Each glyph's record, advance and left side bearing.
    let glyphs: [(Vec<u8>, u16, i16); 8] = [
        (square(50, 0, 450, 700), 500, 50),
        (simple(&A_POINTS), 500, 50),
        (square(50, 0, 550, 700), 600, 50),
        (made, 500, 50),
        (Vec::new(), 250, 0),
        (square(100, 750, 300, 900), 0, 100),
        (square(50, 0, 470, 700), 520, 50),
        (square(50, 0, 650, 700), 700, 50),
    ];
    let (mut glyf, mut loca, mut hmtx) = (Vec::new(), Vec::new(), Vec::new());
    for (record, advance, bearing) in glyphs {
        loca.extend(long(glyf.len()));
        glyf.extend(record);
        hmtx.extend(advance.to_be_bytes());
        hmtx.extend(bearing.to_be_bytes());
    }
    loca.extend(long(glyf.len()));
    [(b"glyf", glyf), (b"loca", loca), (b"hmtx", hmtx)]
}

/// A 16.16 fixed-point number.
fn fixed(value: f32) -> [u8; 4] {
    #[allow(clippy::cast_possible_truncation)]
    let raw = (value * 65_536.0).round() as i32;
    raw.to_be_bytes()
}

/// The names the face's axes and instances have, by name id.
const NAMES: [(u16, &str); 6] = [
    (256, "Weight"),
    (257, "Width"),
    (258, "Regular"),
    (259, "Bold"),
    (260, "Condensed Bold"),
    (261, "Fixture-CondensedBold"),
];

/// A `name` table of Windows English names, by name id.
pub(crate) fn names_table(names: &[(u16, &str)]) -> Vec<u8> {
    let entries = count(names.len());
    let mut table = be(&[0, entries, 6 + entries * 12]);
    let mut strings = Vec::new();
    for (id, text) in names {
        let encoded: Vec<u8> = text.encode_utf16().flat_map(u16::to_be_bytes).collect();
        table.extend(be(&[
            3,
            1,
            0x0409,
            *id,
            count(encoded.len()),
            count(strings.len()),
        ]));
        strings.extend(encoded);
    }
    table.extend(strings);
    table
}

fn names() -> Vec<u8> {
    names_table(&NAMES)
}

/// One axis of an `fvar` table: its tag, its least, default and greatest
/// values, and the id of its name.
pub(crate) type AxisSpec<'a> = (&'a [u8; 4], f32, f32, f32, u16);
/// One instance: the id of its name, a value for each axis, and the id
/// of its name as a font of its own, or 0xFFFF.
pub(crate) type InstanceSpec<'a> = (u16, &'a [f32], u16);

/// An `fvar` table of `axes` and `instances`.
pub(crate) fn fvar_table(axes: &[AxisSpec<'_>], instances: &[InstanceSpec<'_>]) -> Vec<u8> {
    let mut table = be(&[
        1,
        0,
        16,
        2,
        count(axes.len()),
        20,
        count(instances.len()),
        count(4 + axes.len() * 4 + 2),
    ]);
    for (tag, min, default, max, name) in axes {
        table.extend(*tag);
        table.extend(fixed(*min));
        table.extend(fixed(*default));
        table.extend(fixed(*max));
        table.extend(be(&[0, *name]));
    }
    for (name, values, postscript) in instances {
        table.extend(be(&[*name, 0]));
        for value in *values {
            table.extend(fixed(*value));
        }
        table.extend(postscript.to_be_bytes());
    }
    table
}

/// `fvar`: the two axes, and three named instances, each with a name of
/// its own as a font; only the last has one.
fn axes() -> Vec<u8> {
    fvar_table(
        &[
            (b"wght", 100.0, 400.0, 900.0, 256),
            (b"wdth", 50.0, 100.0, 150.0, 257),
        ],
        &[
            (258, &[400.0, 100.0], 0xFFFF),
            (259, &[700.0, 100.0], 0xFFFF),
            (260, &[700.0, 75.0], 261),
        ],
    )
}

/// `avar`: halfway along the heavy side of weight is a quarter of the
/// way; width is left straight.
fn bends() -> Vec<u8> {
    bends_of(1)
}

/// `avar` of version `version`, without what a later version has after
/// each axis's own bending.
pub(crate) fn bends_of(version: u16) -> Vec<u8> {
    let mut table = be(&[version, 0, 0, 2]);
    table.extend(be(&[4]));
    table.extend(signed_bytes(&[
        -ONE,
        -ONE,
        0,
        0,
        ONE / 2,
        ONE / 4,
        ONE,
        ONE,
    ]));
    table.extend(be(&[3]));
    table.extend(signed_bytes(&[-ONE, -ONE, 0, 0, ONE, ONE]));
    table
}

/// Point numbers, packed: a count, then runs of the distance from each
/// number to the one before.
fn packed_points(numbers: &[u16]) -> Vec<u8> {
    let mut packed = vec![u8::try_from(numbers.len()).expect("few points")];
    if numbers.is_empty() {
        return packed;
    }
    packed.push(u8::try_from(numbers.len() - 1).expect("one run"));
    let mut last = 0;
    for number in numbers {
        packed.push(u8::try_from(number - last).expect("near"));
        last = *number;
    }
    packed
}

/// Deltas, packed: runs of zeros, of bytes and of words.
fn packed_deltas(deltas: &[i16]) -> Vec<u8> {
    let kind = |delta: i16| match delta {
        0 => 0x80_u8,
        -128..=127 => 0x00,
        _ => 0x40,
    };
    let mut packed = Vec::new();
    let mut at = 0;
    while at < deltas.len() {
        let control = kind(deltas[at]);
        let mut end = at + 1;
        while end < deltas.len() && end - at < 64 && kind(deltas[end]) == control {
            end += 1;
        }
        packed.push(control | u8::try_from(end - at - 1).expect("run"));
        for delta in &deltas[at..end] {
            match control {
                0x80 => {}
                0x00 => packed.extend(i8::try_from(*delta).expect("byte").to_be_bytes()),
                _ => packed.extend(delta.to_be_bytes()),
            }
        }
        at = end;
    }
    packed
}

/// Where a tuple's region peaks.
enum Peak {
    /// At one of the peaks the table's glyphs share.
    Shared(u16),
    /// At a place of its own along each axis.
    Own([i16; 2]),
}

/// One tuple of a glyph's variation data.
struct Tuple {
    peak: Peak,
    /// A start and an end along each axis, for a region that does not
    /// start at the default.
    between: Option<([i16; 2], [i16; 2])>,
    /// The points the tuple moves: `None` for the list the glyph's
    /// tuples share, and an empty list for every point.
    points: Option<Vec<u16>>,
    dx: Vec<i16>,
    dy: Vec<i16>,
}

/// One glyph's variation data: the point numbers its tuples share, when
/// they share any, and its tuples.
fn glyph_variations(shared: Option<&[u16]>, tuples: &[Tuple]) -> Vec<u8> {
    let mut headers = Vec::new();
    let mut data = shared.map_or_else(Vec::new, packed_points);
    for tuple in tuples {
        let mut own = tuple.points.as_deref().map_or_else(Vec::new, packed_points);
        own.extend(packed_deltas(&tuple.dx));
        own.extend(packed_deltas(&tuple.dy));
        let mut index = match &tuple.peak {
            Peak::Shared(index) => *index,
            Peak::Own(_) => 0x8000,
        };
        if tuple.between.is_some() {
            index |= 0x4000;
        }
        if tuple.points.is_some() {
            index |= 0x2000;
        }
        headers.extend(be(&[count(own.len()), index]));
        if let Peak::Own(peak) = &tuple.peak {
            headers.extend(signed_bytes(peak));
        }
        if let Some((start, end)) = &tuple.between {
            headers.extend(signed_bytes(start));
            headers.extend(signed_bytes(end));
        }
        data.extend(own);
    }
    let mut flags = count(tuples.len());
    if shared.is_some() {
        flags |= 0x8000;
    }
    let mut record = be(&[flags, count(4 + headers.len())]);
    record.extend(headers);
    record.extend(data);
    record
}

/// `gvar`: one shared peak, the heavy end of weight, and each glyph's
/// tuples.
fn moves() -> Vec<u8> {
    let heavy = Peak::Shared(0);
    let narrow = || Peak::Own([0, -ONE]);
    let glyphs: [Vec<u8>; 8] = [
        Vec::new(),
        glyph_variations(
            None,
            &[
                // The bottom left stays and the bottom right goes right:
                // the points above follow by where they are between.
                Tuple {
                    peak: Peak::Shared(0),
                    between: None,
                    points: Some(vec![0, 1]),
                    dx: vec![0, 100],
                    dy: vec![0, 0],
                },
                // Every point, and the right edge after them.
                Tuple {
                    peak: narrow(),
                    between: None,
                    points: Some(Vec::new()),
                    dx: vec![-10, -90, -90, -50, -10, 0, -100, 0, 0],
                    dy: vec![0; 9],
                },
                // From a quarter of the way to the heavy end, most at
                // the middle: the top middle point rises, between two
                // that stay.
                Tuple {
                    peak: Peak::Own([ONE / 2, 0]),
                    between: Some(([ONE / 4, 0], [ONE, 0])),
                    points: Some(vec![2, 3, 4]),
                    dx: vec![0, 0, 0],
                    dy: vec![0, 40, 0],
                },
                // The heavy narrow corner.
                Tuple {
                    peak: Peak::Own([ONE, -ONE]),
                    between: None,
                    points: Some(vec![1, 2]),
                    dx: vec![20, 20],
                    dy: vec![0, 0],
                },
            ],
        ),
        glyph_variations(
            Some(&[0, 1, 2]),
            &[
                Tuple {
                    peak: heavy,
                    between: None,
                    points: None,
                    dx: vec![0, 80, 80],
                    dy: vec![0, 0, 0],
                },
                Tuple {
                    peak: narrow(),
                    between: None,
                    points: None,
                    dx: vec![0, -60, -60],
                    dy: vec![0, 0, 0],
                },
            ],
        ),
        // The second part, the mark, moves.
        glyph_variations(
            None,
            &[Tuple {
                peak: Peak::Shared(0),
                between: None,
                points: Some(vec![1]),
                dx: vec![30],
                dy: vec![50],
            }],
        ),
        // No points: the right edge alone.
        glyph_variations(
            None,
            &[
                Tuple {
                    peak: Peak::Shared(0),
                    between: None,
                    points: Some(Vec::new()),
                    dx: vec![0, 50, 0, 0],
                    dy: vec![0; 4],
                },
                Tuple {
                    peak: narrow(),
                    between: None,
                    points: Some(Vec::new()),
                    dx: vec![0, -25, 0, 0],
                    dy: vec![0; 4],
                },
            ],
        ),
        glyph_variations(
            None,
            &[Tuple {
                peak: Peak::Shared(0),
                between: None,
                points: Some(Vec::new()),
                dx: vec![0; 8],
                dy: vec![20, 20, 20, 20, 0, 0, 0, 0],
            }],
        ),
        Vec::new(),
        Vec::new(),
    ];
    let shared = signed_bytes(&[ONE, 0]);
    let offsets_at = 20;
    let shared_at = offsets_at + (glyphs.len() + 1) * 4;
    let array_at = shared_at + shared.len();
    let mut table = be(&[1, 0, 2, 1]);
    table.extend(long(shared_at));
    table.extend(be(&[count(glyphs.len()), 1]));
    table.extend(long(array_at));
    let mut at = 0;
    for glyph in &glyphs {
        table.extend(long(at));
        at += glyph.len();
    }
    table.extend(long(at));
    table.extend(shared);
    for glyph in glyphs {
        table.extend(glyph);
    }
    table
}

/// An item variation store over the two axes: `regions`, each a start,
/// a peak and an end along each axis, and one group of rows with a
/// number for each region, the first `words` of them words and the rest
/// bytes.
pub(crate) fn store(regions: &[[(i16, i16, i16); 2]], words: u16, rows: &[&[i16]]) -> Vec<u8> {
    let mut list = be(&[2, count(regions.len())]);
    for region in regions {
        for (start, peak, end) in region {
            list.extend(signed_bytes(&[*start, *peak, *end]));
        }
    }
    let mut group = be(&[count(rows.len()), words, count(regions.len())]);
    group.extend(be(&(0..count(regions.len())).collect::<Vec<_>>()));
    for row in rows {
        for (column, delta) in row.iter().enumerate() {
            if column < usize::from(words) {
                group.extend(delta.to_be_bytes());
            } else {
                group.extend(i8::try_from(*delta).expect("byte").to_be_bytes());
            }
        }
    }
    // The header, the one group's offset, the regions, the group.
    let mut table = be(&[1]);
    table.extend(long(12));
    table.extend(be(&[1]));
    table.extend(long(12 + list.len()));
    table.extend(list);
    table.extend(group);
    table
}

/// The region of the heavy side of weight, and of the narrow side of
/// width.
pub(crate) const HEAVY: [(i16, i16, i16); 2] = [(0, ONE, ONE), (0, 0, 0)];
pub(crate) const NARROW: [(i16, i16, i16); 2] = [(0, 0, 0), (-ONE, -ONE, 0)];

/// `HVAR`: how much wider each glyph is at the heavy end and at the
/// narrow one. The map of glyphs to rows stops at glyph 6, so glyph 7
/// has glyph 6's row.
fn advances() -> Vec<u8> {
    let store = store(
        &[HEAVY, NARROW],
        1,
        &[
            &[0, 0],
            &[100, -100],
            &[80, -60],
            &[100, -100],
            &[50, -25],
            &[0, 0],
            &[110, -100],
        ],
    );
    // Entries of one byte, all of it the row.
    let mut map = vec![0, 0x07];
    map.extend(be(&[7]));
    map.extend([0, 1, 2, 3, 4, 5, 6]);
    let mut table = be(&[1, 0]);
    table.extend(long(20));
    table.extend(long(20 + store.len()));
    table.extend(long(0));
    table.extend(long(0));
    table.extend(store);
    table.extend(map);
    table
}

/// `GDEF`, version 1.3: the glyphs' classes, and the store the
/// positioning table's values name rows of: the kerning, and the two
/// numbers of the anchor on `A`, at the heavy end.
fn definitions() -> Vec<u8> {
    definitions_of(&[&[-40], &[50], &[30]])
}

/// `GDEF` with a store of `rows`, each a number for the heavy end.
pub(crate) fn definitions_of(rows: &[&[i16]]) -> Vec<u8> {
    let classes = classes(&[(A, C, 1), (MARK, MARK, 3), (A_HEAVY, D, 1)]);
    let store = store(&[HEAVY], 0, rows);
    let mut table = be(&[1, 3, 18, 0, 0, 0, 0]);
    table.extend(long(18 + classes.len()));
    table.extend(classes);
    table.extend(store);
    table
}

/// A table that names row `row` of the store's one group.
fn row(row: u16) -> Vec<u8> {
    be(&[0, row, 0x8000])
}

/// `GPOS`: `A` before `B` kerned by -50, less 40 more at the heavy end;
/// and the mark set on `A` at an anchor that moves.
fn positions() -> Vec<u8> {
    positions_of(be(&[1, 200, 750]))
}

/// An anchor that rows `x` and `y` of the store move.
pub(crate) fn moving_anchor(at: (i16, i16), x: u16, y: u16) -> Vec<u8> {
    table(vec![
        V(3),
        V(signed(at.0)),
        V(signed(at.1)),
        Off(row(x)),
        Off(row(y)),
    ])
}

/// `GPOS` with `mark_anchor` for the mark's own anchor.
pub(crate) fn positions_of(mark_anchor: Vec<u8>) -> Vec<u8> {
    // The first glyph's advance, and what changes it.
    let pair = table(vec![
        V(1),
        Off(coverage(&[A])),
        V(0x0044),
        V(0),
        V(1),
        Off(table(vec![V(1), V(B), V(signed(-50)), Off(row(0))])),
    ]);
    let mark = table(vec![
        V(1),
        Off(coverage(&[MARK])),
        Off(coverage(&[A])),
        V(1),
        Off(table(vec![V(1), V(0), Off(mark_anchor)])),
        Off(table(vec![V(1), Off(moving_anchor((250, 700), 1, 2))])),
    ]);
    layout_table(
        &[(b"DFLT", &[0, 1])],
        &[(b"kern", &[0]), (b"mark", &[1])],
        vec![
            LookupSpec::new(2, vec![pair]),
            LookupSpec::new(4, vec![mark]),
        ],
    )
}

/// `GSUB`, version 1.1: the feature `rvrn` runs nothing at the default,
/// and from halfway along the heavy side of weight runs the lookup that
/// draws glyph 6 for `A`.
fn substitutions() -> Vec<u8> {
    let plain = layout_table(
        &[(b"DFLT", &[0])],
        &[(b"rvrn", &[])],
        vec![LookupSpec::new(1, vec![single(&[(A, A_HEAVY)])])],
    );
    // The header grows by the offset of the table of variations, which
    // goes after everything else.
    let lists: Vec<u16> = plain[4..10]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_be_bytes(*pair) + 4)
        .collect();
    let mut table = be(&[1, 1, lists[0], lists[1], lists[2]]);
    table.extend(long(plain.len() + 4));
    table.extend(&plain[10..]);
    // One record: a set of one condition, and the features it swaps.
    let mut variations = be(&[1, 0]);
    variations.extend(long(1));
    variations.extend(long(16));
    variations.extend(long(16 + 6 + 8));
    variations.extend(be(&[1]));
    variations.extend(long(6));
    variations.extend(be(&[1, 0]));
    variations.extend(signed_bytes(&[ONE / 2, ONE]));
    // Feature 0 becomes a feature that runs lookup 0.
    variations.extend(be(&[1, 0, 1, 0]));
    variations.extend(long(12));
    variations.extend(be(&[0, 1, 0]));
    table.extend(variations);
    table
}

/// The tables of the face that are not in every font, by name.
fn tables() -> Vec<(&'static [u8; 4], Vec<u8>)> {
    let mut tables: Vec<(&[u8; 4], Vec<u8>)> = outlines().into_iter().collect();
    tables.extend([
        (b"fvar", axes()),
        (b"avar", bends()),
        (b"gvar", moves()),
        (b"name", names()),
        (b"GDEF", definitions()),
        (b"GPOS", positions()),
        (b"GSUB", substitutions()),
        (b"HVAR", advances()),
    ]);
    tables
}

fn built(tables: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let glyphs: Vec<Glyph> = vec![boxed(500); 8];
    let map = [
        (' ', SPACE),
        ('A', A),
        ('B', B),
        ('C', C),
        ('D', D),
        ('\u{301}', MARK),
    ];
    font(&glyphs, &map, tables)
}

/// The face, with its table of advances.
pub(crate) fn variable() -> Vec<u8> {
    built(&tables())
}

/// The face without its table of advances.
pub(crate) fn variable_edges() -> Vec<u8> {
    variable_with(b"HVAR", None)
}

/// The table `name` of the face.
pub(crate) fn table_of(name: &[u8; 4]) -> Vec<u8> {
    tables()
        .into_iter()
        .find(|(tag, _)| *tag == name)
        .map(|(_, bytes)| bytes)
        .expect("table")
}

/// The face with `table` in place of its table `name`, or without that
/// table.
pub(crate) fn variable_with_tables(replaced: Vec<(&'static [u8; 4], Vec<u8>)>) -> Vec<u8> {
    let mut tables = tables();
    tables.retain(|(tag, _)| !replaced.iter().any(|(name, _)| name == tag));
    tables.extend(replaced);
    built(&tables)
}

/// The face with `table` in place of its table `name`, or without that
/// table.
pub(crate) fn variable_with(name: &'static [u8; 4], table: Option<Vec<u8>>) -> Vec<u8> {
    let mut tables = tables();
    tables.retain(|(tag, _)| *tag != name);
    if let Some(table) = table {
        tables.push((name, table));
    }
    built(&tables)
}

/// A fixture's file name and what builds it.
type Built = (&'static str, fn() -> Vec<u8>);

const FILES: [Built; 2] = [
    ("variable.ttf", variable),
    ("variable-edges.ttf", variable_edges),
];

fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

#[test]
fn the_variable_fixture_files_are_what_the_builder_writes() {
    for (name, build) in FILES {
        let stored = std::fs::read(path(name)).expect("fixture");
        assert!(
            stored == build(),
            "{name} is stale: write the fixtures again"
        );
        let face = crate::read(&stored).expect("font");
        assert!(face.is_variable());
    }
}

#[test]
#[ignore = "writes the variable fixture fonts"]
fn write_the_var_fixture_fonts() {
    for (name, build) in FILES {
        std::fs::write(path(name), build()).expect("write");
    }
}
