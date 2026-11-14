//! A face of layered color glyphs, built in code: its `COLR` table of
//! both versions, its palette, and paints that are wrong on purpose.

use crate::fixture::{Glyph, boxed, font};

/// One field of a paint table.
pub(crate) enum Field {
    U8(u8),
    U16(u16),
    I16(i16),
    U32(u32),
    /// A value with 14 fraction bits.
    F14(f32),
    /// A value with 16 fraction bits.
    Fixed(f32),
    /// A table this one points to with a 24-bit offset from its own start.
    Child(Vec<u8>),
}

use Field::{Child, F14, Fixed, I16, U8, U16, U32};

fn f14(value: f32) -> i16 {
    (value * 16_384.0).round() as i16
}

fn fixed(value: f32) -> i32 {
    (value * 65_536.0).round() as i32
}

/// The table `fields` describe: the fields in order, then each child.
pub(crate) fn paint_table(fields: Vec<Field>) -> Vec<u8> {
    let head: usize = fields
        .iter()
        .map(|field| match field {
            U8(_) => 1,
            U16(_) | I16(_) | F14(_) => 2,
            Child(_) => 3,
            U32(_) | Fixed(_) => 4,
        })
        .sum();
    let mut out = Vec::new();
    let mut tail = Vec::new();
    for field in fields {
        match field {
            U8(value) => out.push(value),
            U16(value) => out.extend(value.to_be_bytes()),
            I16(value) => out.extend(value.to_be_bytes()),
            U32(value) => out.extend(value.to_be_bytes()),
            F14(value) => out.extend(f14(value).to_be_bytes()),
            Fixed(value) => out.extend(fixed(value).to_be_bytes()),
            Child(bytes) => {
                let at = u32::try_from(head + tail.len()).expect("offset");
                out.extend(&at.to_be_bytes()[1..]);
                tail.extend(bytes);
            }
        }
    }
    out.extend(tail);
    out
}

/// A paint that fills with palette entry `index`.
pub(crate) fn solid(index: u16, alpha: f32) -> Vec<u8> {
    paint_table(vec![U8(2), U16(index), F14(alpha)])
}

/// A color line: how it extends, and each stop's offset, palette entry, and
/// alpha. A variable line's stops carry four more bytes.
pub(crate) fn color_line(extend: u8, stops: &[(f32, u16, f32)], variable: bool) -> Vec<u8> {
    let mut fields = vec![U8(extend), U16(u16::try_from(stops.len()).expect("stops"))];
    for (offset, index, alpha) in stops {
        fields.extend([F14(*offset), U16(*index), F14(*alpha)]);
        if variable {
            fields.push(U32(u32::MAX));
        }
    }
    paint_table(fields)
}

/// A paint of `format` that holds one other paint, and then `rest`.
pub(crate) fn around(format: u8, paint: Vec<u8>, rest: Vec<Field>) -> Vec<u8> {
    let mut fields = vec![U8(format), Child(paint)];
    fields.extend(rest);
    paint_table(fields)
}

/// `paint` inside the outline of `glyph`.
pub(crate) fn fill(glyph: u16, paint: Vec<u8>) -> Vec<u8> {
    around(10, paint, vec![U16(glyph)])
}

/// A gradient of `format` along `line`, and then `rest`.
fn gradient(format: u8, line: Vec<u8>, rest: Vec<Field>) -> Vec<u8> {
    around(format, line, rest)
}

fn points(values: &[i16]) -> Vec<Field> {
    values.iter().map(|value| I16(*value)).collect()
}

/// `count` entries of the layer list from `first`.
fn layers(count: u8, first: u32) -> Vec<u8> {
    paint_table(vec![U8(1), U8(count), U32(first)])
}

/// The color glyph of `glyph`, used as a paint.
fn reuse(glyph: u16) -> Vec<u8> {
    paint_table(vec![U8(11), U16(glyph)])
}

/// What goes into a `COLR` table.
#[derive(Default)]
struct Spec {
    /// Version 0: a glyph and its layers, each a glyph and a palette entry.
    flat: Vec<(u16, Vec<(u16, u16)>)>,
    /// Version 1: a glyph and its paint, in glyph order.
    roots: Vec<(u16, Vec<u8>)>,
    /// The layer list.
    layers: Vec<Vec<u8>>,
    /// A first glyph, a last, and their clip box.
    clips: Vec<(u16, u16, [i16; 4])>,
}

fn be32(value: usize) -> [u8; 4] {
    u32::try_from(value).expect("32 bits").to_be_bytes()
}

fn colr_table(spec: &Spec) -> Vec<u8> {
    let mut bases = Vec::new();
    let mut flat_layers = Vec::new();
    for (glyph, parts) in &spec.flat {
        bases.extend(glyph.to_be_bytes());
        bases.extend(
            u16::try_from(flat_layers.len() / 4)
                .expect("first")
                .to_be_bytes(),
        );
        bases.extend(u16::try_from(parts.len()).expect("count").to_be_bytes());
        for (layer, index) in parts {
            flat_layers.extend(layer.to_be_bytes());
            flat_layers.extend(index.to_be_bytes());
        }
    }
    // A list of 32-bit offsets from its own start, then what they point to.
    let listed = |entries: &[&[u8]], record: &dyn Fn(usize, usize) -> Vec<u8>, each: usize| {
        let mut out = be32(entries.len()).to_vec();
        let mut at = 4 + entries.len() * each;
        let mut tail: Vec<u8> = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            out.extend(record(index, at));
            at += entry.len();
            tail.extend(*entry);
        }
        out.extend(tail);
        out
    };
    let root_paints: Vec<&[u8]> = spec.roots.iter().map(|root| root.1.as_slice()).collect();
    let base_list = listed(
        &root_paints,
        &|index, at| {
            let mut record = spec.roots[index].0.to_be_bytes().to_vec();
            record.extend(be32(at));
            record
        },
        6,
    );
    let layer_paints: Vec<&[u8]> = spec.layers.iter().map(Vec::as_slice).collect();
    let layer_list = listed(&layer_paints, &|_, at| be32(at).to_vec(), 4);
    let mut clip_list = vec![1];
    clip_list.extend(be32(spec.clips.len()));
    let mut boxes = Vec::new();
    for (first, last, corners) in &spec.clips {
        clip_list.extend(first.to_be_bytes());
        clip_list.extend(last.to_be_bytes());
        let at = 5 + spec.clips.len() * 7 + boxes.len();
        clip_list.extend(&be32(at)[1..]);
        boxes.push(1);
        for corner in corners {
            boxes.extend(corner.to_be_bytes());
        }
    }
    clip_list.extend(boxes);

    let bases_at = 34;
    let flat_at = bases_at + bases.len();
    let base_list_at = flat_at + flat_layers.len();
    let layer_list_at = base_list_at + base_list.len();
    let clip_list_at = layer_list_at + layer_list.len();
    let mut out = vec![0, 1];
    out.extend(u16::try_from(spec.flat.len()).expect("bases").to_be_bytes());
    out.extend(be32(bases_at));
    out.extend(be32(flat_at));
    out.extend(
        u16::try_from(flat_layers.len() / 4)
            .expect("layers")
            .to_be_bytes(),
    );
    out.extend(be32(base_list_at));
    out.extend(be32(layer_list_at));
    out.extend(be32(clip_list_at));
    out.extend([0; 8]);
    out.extend(bases);
    out.extend(flat_layers);
    out.extend(base_list);
    out.extend(layer_list);
    out.extend(clip_list);
    out
}

/// One palette of colors, each red, green, blue, alpha.
fn cpal_table(colors: &[[u8; 4]]) -> Vec<u8> {
    let count = u16::try_from(colors.len()).expect("colors");
    let mut out = Vec::new();
    for value in [0, count, 1, count] {
        out.extend(value.to_be_bytes());
    }
    out.extend(14_u32.to_be_bytes());
    out.extend(0_u16.to_be_bytes());
    for [red, green, blue, alpha] in colors {
        out.extend([*blue, *green, *red, *alpha]);
    }
    out
}

/// A glyph that is one box.
fn block(corners: [i16; 4]) -> Glyph {
    Glyph {
        advance: 1_000,
        outline: Some(corners),
    }
}

/// The palette entries of the face.
pub(crate) const RED: u16 = 0;
pub(crate) const BLUE: u16 = 1;
pub(crate) const HALF_GREEN: u16 = 2;
/// The palette index that means the color of the text.
pub(crate) const TEXT: u16 = 0xFFFF;

/// The whole em square, as a clip box and as a gradient's frame.
const EM: [i16; 4] = [0, 0, 1_000, 1_000];

fn two_stops(extend: u8, variable: bool) -> Vec<u8> {
    color_line(extend, &[(0.0, RED, 1.0), (1.0, BLUE, 1.0)], variable)
}

/// A face whose glyphs are drawn in layered color.
///
/// Its outlines are boxes in an em of 1000 units: 2 the left half, 3 the
/// right half, 4 the whole em, 5 the lower left quarter, 6 the middle
/// half. Glyph 1 is `A`, with no color. The palette is red, blue, green
/// at half alpha, and white.
///
/// Version 0 glyphs: 10 is red on the left and blue on the right; 11 is
/// the whole em in the color of the text with the half green middle over
/// it.
///
/// Version 1 glyphs, each filling the whole em unless said: 12 a linear
/// gradient from red at the left to blue at the right; 13 a radial one
/// from red at the center to blue 500 units out; 14 a sweep from red to
/// blue once around the center; 15 the quarter box moved to the upper
/// right, red; 16 the red left half kept only where the blue middle is;
/// 17 glyph 12 used again; 18 two layers, red left and blue right; 19 a
/// gradient 250 units long that repeats; 20 the same, reflected; 21 to 29
/// the quarter box in red under each kind of transform; 30 and 31 the
/// variable forms of a translate, a solid, and a linear gradient.
///
/// Glyphs that are wrong on purpose: 40 names itself; 41 is a layer that
/// lists itself; 42 nests 40 paints; 43 holds 70 layers of 70 layers; 44
/// fills a glyph the face does not have; 45 is a paint format that does
/// not exist; 46 a palette entry that does not; 47 a blend mode that does
/// not; 48 and 49 name each other; 50 names a glyph with no paint; 51
/// points at itself.
pub(crate) fn layered() -> Vec<u8> {
    let mut glyphs = vec![
        Glyph {
            advance: 1_000,
            outline: None,
        };
        52
    ];
    glyphs[0] = boxed(500);
    glyphs[1] = boxed(600);
    glyphs[2] = block([0, 0, 500, 1_000]);
    glyphs[3] = block([500, 0, 1_000, 1_000]);
    glyphs[4] = block(EM);
    glyphs[5] = block([0, 0, 500, 500]);
    glyphs[6] = block([250, 250, 750, 750]);
    let mut map = vec![('A', 1)];
    map.extend((10..52).map(|glyph| {
        let character = char::from_u32(0xE000 + u32::from(glyph)).expect("private use");
        (character, glyph)
    }));

    let red = || solid(RED, 1.0);
    let quarter = || fill(5, red());
    let frame = || points(&[0, 0, 1_000, 0, 0, 1_000]);
    let short = || points(&[0, 0, 250, 0, 0, 250]);
    let mut deep = fill(4, red());
    for _ in 0..40 {
        deep = around(14, deep, points(&[1, 1]));
    }
    let mut spec = Spec {
        flat: vec![
            (10, vec![(2, RED), (3, BLUE)]),
            (11, vec![(4, TEXT), (6, HALF_GREEN)]),
        ],
        roots: vec![
            (12, fill(4, gradient(4, two_stops(0, false), frame()))),
            (
                13,
                fill(
                    4,
                    gradient(
                        6,
                        two_stops(0, false),
                        vec![I16(500), I16(500), U16(0), I16(500), I16(500), U16(500)],
                    ),
                ),
            ),
            (
                14,
                fill(
                    4,
                    gradient(
                        8,
                        two_stops(0, false),
                        vec![I16(500), I16(500), F14(-1.0), F14(1.0)],
                    ),
                ),
            ),
            (15, around(14, quarter(), points(&[500, 500]))),
            (
                16,
                paint_table(vec![
                    U8(32),
                    Child(fill(2, red())),
                    U8(5),
                    Child(fill(6, solid(BLUE, 1.0))),
                ]),
            ),
            (17, reuse(12)),
            (18, layers(2, 0)),
            (19, fill(4, gradient(4, two_stops(1, false), short()))),
            (20, fill(4, gradient(4, two_stops(2, false), short()))),
            (
                21,
                paint_table(vec![
                    U8(12),
                    Child(quarter()),
                    Child(paint_table(vec![
                        Fixed(0.5),
                        Fixed(0.0),
                        Fixed(0.0),
                        Fixed(0.5),
                        Fixed(250.0),
                        Fixed(250.0),
                    ])),
                ]),
            ),
            (22, around(16, quarter(), vec![F14(1.5), F14(0.5)])),
            (
                23,
                around(18, quarter(), vec![F14(1.5), F14(0.5), I16(500), I16(500)]),
            ),
            (24, around(20, quarter(), vec![F14(0.5)])),
            (
                25,
                around(22, quarter(), vec![F14(0.5), I16(500), I16(500)]),
            ),
            (26, around(24, quarter(), vec![F14(0.5)])),
            (
                27,
                around(26, quarter(), vec![F14(0.5), I16(500), I16(500)]),
            ),
            (28, around(28, quarter(), vec![F14(0.25), F14(0.0)])),
            (
                29,
                around(30, quarter(), vec![F14(0.0), F14(0.25), I16(500), I16(500)]),
            ),
            (
                30,
                around(
                    15,
                    fill(
                        5,
                        paint_table(vec![U8(3), U16(RED), F14(0.5), U32(u32::MAX)]),
                    ),
                    vec![I16(500), I16(500), U32(u32::MAX)],
                ),
            ),
            (
                31,
                fill(4, {
                    let mut rest = frame();
                    rest.push(U32(u32::MAX));
                    gradient(5, two_stops(0, true), rest)
                }),
            ),
            (40, reuse(40)),
            (41, layers(1, 2)),
            (42, deep),
            (43, layers(70, 73)),
            (44, fill(999, red())),
            (45, vec![99, 0, 0, 0]),
            (46, fill(4, solid(9, 1.0))),
            (
                47,
                paint_table(vec![
                    U8(32),
                    Child(fill(2, red())),
                    U8(99),
                    Child(fill(6, red())),
                ]),
            ),
            (48, reuse(49)),
            (49, reuse(48)),
            (50, reuse(1)),
            (51, vec![10, 0, 0, 0, 0, 4]),
        ],
        layers: vec![fill(2, red()), fill(3, solid(BLUE, 1.0)), layers(1, 2)],
        clips: vec![(15, 16, EM), (21, 30, [-1_000, -1_000, 1_500, 1_500])],
    };
    // Layers 3 to 72 are solids, and 73 to 142 each list all of those.
    spec.layers.extend((0..70).map(|_| red()));
    spec.layers.extend((0..70).map(|_| layers(70, 3)));
    let palette = cpal_table(&[
        [255, 0, 0, 255],
        [0, 0, 255, 255],
        [0, 255, 0, 128],
        [255, 255, 255, 255],
    ]);
    font(
        &glyphs,
        &map,
        &[(b"COLR", colr_table(&spec)), (b"CPAL", palette)],
    )
}
