//! The color fonts under `fixtures/` that are built in code: what each
//! holds, and a check that the files are what this builder writes.
//!
//! To write them again after a change here:
//! `cargo test -p cutlass-font -- --ignored write_the_color_fixture_fonts`.

use crate::fixture::{Glyph, boxed, font};

/// A glyph that draws nothing of its own: an empty outline, one em wide.
fn blank() -> Glyph {
    Glyph {
        advance: 1_000,
        outline: None,
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut low, mut high) = (1_u32, 0_u32);
    for byte in bytes {
        low = (low + u32::from(*byte)) % 65_521;
        high = (high + low) % 65_521;
    }
    (high << 16) | low
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(data.len())
        .expect("chunk")
        .to_be_bytes()
        .to_vec();
    let mut body = kind.to_vec();
    body.extend(data);
    out.extend(&body);
    out.extend(crc32(&body).to_be_bytes());
    out
}

fn header(width: u32, height: u32, depth: u8, color_type: u8) -> Vec<u8> {
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = width.to_be_bytes().to_vec();
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([depth, color_type, 0, 0, 0]);
    out.extend(chunk(b"IHDR", &ihdr));
    out
}

/// A PNG picture. `rows` are the packed rows, top first, with no filter
/// bytes; they are stored without compression. `extra` chunks come before
/// the pixels.
pub(crate) fn png(
    size: (u32, u32),
    depth: u8,
    color_type: u8,
    rows: &[&[u8]],
    extra: &[(&[u8; 4], &[u8])],
) -> Vec<u8> {
    let mut out = header(size.0, size.1, depth, color_type);
    for (kind, data) in extra {
        out.extend(chunk(kind, data));
    }
    let mut raw = Vec::new();
    for row in rows {
        raw.push(0);
        raw.extend(*row);
    }
    let len = u16::try_from(raw.len()).expect("one stored block");
    let mut stream = vec![0x78, 0x01, 0x01];
    stream.extend(len.to_le_bytes());
    stream.extend((!len).to_le_bytes());
    stream.extend(&raw);
    stream.extend(adler32(&raw).to_be_bytes());
    out.extend(chunk(b"IDAT", &stream));
    out.extend(chunk(b"IEND", &[]));
    out
}

/// Two by two, 8-bit color with alpha: red, half green, blue, clear.
pub(crate) fn rgba_picture() -> Vec<u8> {
    png(
        (2, 2),
        8,
        6,
        &[
            &[255, 0, 0, 255, 0, 255, 0, 128],
            &[0, 0, 255, 255, 0, 0, 0, 0],
        ],
        &[],
    )
}

/// Two-bit indices into red, green, blue, yellow, whose alphas are full,
/// half, none, and, past the table, full.
pub(crate) fn palette_picture() -> Vec<u8> {
    png(
        (2, 2),
        2,
        3,
        &[&[0b0001_0000], &[0b1011_0000]],
        &[
            (b"PLTE", &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]),
            (b"tRNS", &[255, 128, 0]),
        ],
    )
}

/// 8-bit gray: black, mid, white, and the value 7, which is the clear one.
fn gray_picture() -> Vec<u8> {
    png((2, 2), 8, 0, &[&[0, 128], &[255, 7]], &[(b"tRNS", &[0, 7])])
}

/// 16-bit color with alpha: red, half green at half alpha, blue, clear.
fn deep_picture() -> Vec<u8> {
    png(
        (2, 2),
        16,
        6,
        &[
            &[
                0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF, 0, 0, 0x80, 0, 0, 0, 0x80, 0,
            ],
            &[0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        &[],
    )
}

/// 8-bit gray with alpha: white, 100 at half alpha, black, clear.
pub(crate) fn gray_alpha_picture() -> Vec<u8> {
    png(
        (2, 2),
        8,
        4,
        &[&[255, 255, 100, 128], &[0, 255, 50, 0]],
        &[],
    )
}

/// 8-bit color with no alpha.
fn opaque_picture() -> Vec<u8> {
    png(
        (2, 2),
        8,
        2,
        &[&[10, 20, 30, 40, 50, 60], &[70, 80, 90, 100, 110, 120]],
        &[],
    )
}

/// One-bit gray: white, black, black, white.
fn mono_picture() -> Vec<u8> {
    png((2, 2), 1, 0, &[&[0b1000_0000], &[0b0100_0000]], &[])
}

/// A header that declares a hundred thousand pixels each way.
fn bomb_picture() -> Vec<u8> {
    let mut out = header(100_000, 100_000, 8, 6);
    out.extend(chunk(b"IDAT", &[0x78, 0x01, 0x03, 0, 0, 0, 0, 1]));
    out.extend(chunk(b"IEND", &[]));
    out
}

/// A header alone, for a picture `edge` pixels each way.
fn header_only(edge: u32) -> Vec<u8> {
    let mut out = header(edge, edge, 8, 6);
    out.extend(chunk(b"IEND", &[]));
    out
}

fn truncated_picture() -> Vec<u8> {
    let mut out = rgba_picture();
    out.truncate(out.len() - 20);
    out
}

fn bad_crc_picture() -> Vec<u8> {
    let mut out = rgba_picture();
    let at = out.len() - 20;
    out[at] ^= 0x40;
    out
}

/// One `sbix` glyph record: where its left and bottom edges sit, what it
/// is, and its bytes.
fn record(origin: (i16, i16), kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = origin.0.to_be_bytes().to_vec();
    out.extend(origin.1.to_be_bytes());
    out.extend(kind);
    out.extend(data);
    out
}

/// The glyphs of one `sbix` strike that have a record, in glyph order.
type Records = Vec<(u16, Vec<u8>)>;

fn sbix_table(glyph_count: u16, strikes: &[(u16, Records)]) -> Vec<u8> {
    let mut out = vec![0, 1, 0, 1];
    out.extend(u32::try_from(strikes.len()).expect("strikes").to_be_bytes());
    let mut at = 8 + strikes.len() * 4;
    let mut bodies = Vec::new();
    for (ppem, records) in strikes {
        out.extend(u32::try_from(at).expect("strike").to_be_bytes());
        let mut body = ppem.to_be_bytes().to_vec();
        body.extend(72_u16.to_be_bytes());
        let mut data_at = 4 + (usize::from(glyph_count) + 1) * 4;
        let mut data: Vec<u8> = Vec::new();
        for glyph in 0..glyph_count {
            body.extend(u32::try_from(data_at).expect("record").to_be_bytes());
            if let Some((_, bytes)) = records.iter().find(|(id, _)| *id == glyph) {
                data_at += bytes.len();
                data.extend(bytes);
            }
        }
        body.extend(u32::try_from(data_at).expect("record").to_be_bytes());
        body.extend(data);
        at += body.len();
        bodies.extend(body);
    }
    out.extend(bodies);
    out
}

/// The character a test font maps to `glyph`.
fn private(glyph: u16) -> char {
    char::from_u32(0xE000 + u32::from(glyph)).expect("private use")
}

/// A face with `glyf` and `sbix`, as a color emoji face has: every picture
/// glyph has an empty outline.
///
/// Glyph 1 is `A`, a box with no picture. Glyphs 2 to 15 have a two by two
/// picture in the strike of 2 pixels to the em: 2 color with alpha (it is
/// also U+1F600), 3 a palette with alpha, 4 gray with a clear value, 5
/// 16-bit color with alpha, 6 gray with alpha, 7 glyph 2 flipped, 8 glyph
/// 2 again, 9 a header that declares a huge picture, 10 a picture cut
/// short, 11 a picture with a bad check value, 12 color with no alpha, 13 a
/// JPEG record, 14 glyph 2's picture set 3 right and 1 down, 15 one-bit
/// gray. Glyph 2 also has a one-pixel picture in a strike of 1 pixel to
/// the em, and a header for a 600-pixel picture in a strike of 600.
pub(crate) fn sbix() -> Vec<u8> {
    let mut glyphs = vec![blank(); 16];
    glyphs[0] = boxed(500);
    glyphs[1] = boxed(600);
    let mut map = vec![('A', 1)];
    map.extend((3..16).map(|glyph| (private(glyph), glyph)));
    map.push(('\u{1F600}', 2));
    let main: Records = vec![
        (2, record((0, 0), b"png ", &rgba_picture())),
        (3, record((0, 0), b"png ", &palette_picture())),
        (4, record((0, 0), b"png ", &gray_picture())),
        (5, record((0, 0), b"png ", &deep_picture())),
        (6, record((0, 0), b"png ", &gray_alpha_picture())),
        (7, record((0, 0), b"flip", &[0, 2])),
        (8, record((0, 0), b"dupe", &[0, 2])),
        (9, record((0, 0), b"png ", &bomb_picture())),
        (10, record((0, 0), b"png ", &truncated_picture())),
        (11, record((0, 0), b"png ", &bad_crc_picture())),
        (12, record((0, 0), b"png ", &opaque_picture())),
        (13, record((0, 0), b"jpg ", &[0xFF, 0xD8, 0xFF, 0xD9])),
        (14, record((3, -1), b"png ", &rgba_picture())),
        (15, record((0, 0), b"png ", &mono_picture())),
    ];
    let small: Records = vec![(
        2,
        record(
            (0, 0),
            b"png ",
            &png((1, 1), 8, 6, &[&[255, 255, 255, 255]], &[]),
        ),
    )];
    let large: Records = vec![(2, record((0, 0), b"png ", &header_only(600)))];
    let table = sbix_table(16, &[(1, small), (2, main), (600, large)]);
    font(&glyphs, &map, &[(b"sbix", table)])
}

/// A face with `CBLC` and `CBDT` at 2 pixels to the em, each glyph a two by
/// two PNG picture: 1 color with alpha after small metrics, 2 a palette
/// after big metrics, set 1 right with its top 3 up, 3 gray with alpha
/// whose metrics its index shares, set 1 left with its top 2 up.
pub(crate) fn cbdt() -> Vec<u8> {
    let glyphs = vec![blank(); 4];
    let map: Vec<(char, u16)> = (1..4).map(|glyph| (private(glyph), glyph)).collect();
    let sized = |picture: &[u8]| {
        let mut out = u32::try_from(picture.len())
            .expect("picture")
            .to_be_bytes()
            .to_vec();
        out.extend(picture);
        out
    };
    let mut first = vec![2, 2, 0, 2, 2];
    first.extend(sized(&rgba_picture()));
    let mut second = vec![2, 2, 1, 3, 2, 0, 0, 0];
    second.extend(sized(&palette_picture()));
    let third = sized(&gray_alpha_picture());
    let lengths =
        [first.len(), second.len(), third.len()].map(|len| u32::try_from(len).expect("len"));
    let mut data = vec![0, 3, 0, 0];
    data.extend(&first);
    data.extend(&second);
    data.extend(&third);

    // Three index subtables: 32-bit offsets, 16-bit offsets, and one size
    // with shared metrics.
    let mut one = vec![0, 1, 0, 17];
    one.extend(4_u32.to_be_bytes());
    one.extend(0_u32.to_be_bytes());
    one.extend(lengths[0].to_be_bytes());
    let mut two = vec![0, 3, 0, 18];
    two.extend((4 + lengths[0]).to_be_bytes());
    two.extend(0_u16.to_be_bytes());
    two.extend(u16::try_from(lengths[1]).expect("len").to_be_bytes());
    let mut three = vec![0, 2, 0, 19];
    three.extend((4 + lengths[0] + lengths[1]).to_be_bytes());
    three.extend(lengths[2].to_be_bytes());
    three.extend([2, 2, 0xFF, 2, 2, 0, 0, 0]);
    let subtables = [one, two, three];
    let mut list = Vec::new();
    let mut at = subtables.len() * 8;
    for (glyph, subtable) in (1_u16..).zip(&subtables) {
        list.extend(glyph.to_be_bytes());
        list.extend(glyph.to_be_bytes());
        list.extend(u32::try_from(at).expect("subtable").to_be_bytes());
        at += subtable.len();
    }
    for subtable in &subtables {
        list.extend(subtable);
    }
    let mut location = vec![0, 3, 0, 0];
    location.extend(1_u32.to_be_bytes());
    location.extend(56_u32.to_be_bytes());
    location.extend(u32::try_from(list.len()).expect("list").to_be_bytes());
    location.extend(3_u32.to_be_bytes());
    location.extend([0; 28]);
    location.extend(1_u16.to_be_bytes());
    location.extend(3_u16.to_be_bytes());
    location.extend([2, 2, 32, 1]);
    location.extend(list);
    font(&glyphs, &map, &[(b"CBDT", data), (b"CBLC", location)])
}

/// A fixture's file name and what builds it.
type Built = (&'static str, fn() -> Vec<u8>);

const FILES: [Built; 2] = [("sbix-png.ttf", sbix), ("cbdt-png.ttf", cbdt)];

fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

#[test]
fn the_color_fixture_files_are_what_the_builder_writes() {
    for (name, build) in FILES {
        let stored = std::fs::read(path(name)).expect("fixture");
        assert!(
            stored == build(),
            "{name} is stale: write the fixtures again"
        );
        crate::read(&stored).expect("font");
    }
}

#[test]
#[ignore = "writes the color fixture fonts"]
fn write_the_color_fixture_fonts() {
    for (name, build) in FILES {
        std::fs::write(path(name), build()).expect("write");
    }
}
