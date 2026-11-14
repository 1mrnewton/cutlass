//! Checks of the color glyph reader against the face `fixture_colr` builds.

use crate::{
    Affine, ClipBox, Color, ColorGlyph, ColorLine, ColorStop, CompositeMode, Extend, FontError,
    Paint, read,
};

const COLR: &[u8] = include_bytes!("../fixtures/colr.ttf");
const PLAIN: &[u8] = include_bytes!("../fixtures/sample.ttf");

const RED: Color = Color {
    rgb: Some([255, 0, 0]),
    alpha: 1.0,
};
const BLUE: Color = Color {
    rgb: Some([0, 0, 255]),
    alpha: 1.0,
};

fn glyph(glyph_id: u16) -> ColorGlyph {
    let font = read(COLR).expect("font");
    font.color_glyph(glyph_id).expect("glyph").expect("color")
}

fn refusal(glyph_id: u16) -> FontError {
    let font = read(COLR).expect("font");
    font.color_glyph(glyph_id).expect_err("refused")
}

fn filled(glyph_id: u16, paint: Paint) -> Paint {
    Paint::Glyph {
        glyph_id,
        paint: Box::new(paint),
    }
}

fn red_to_blue(extend: Extend) -> ColorLine {
    ColorLine {
        extend,
        stops: vec![
            ColorStop {
                offset: 0.0,
                color: RED,
            },
            ColorStop {
                offset: 1.0,
                color: BLUE,
            },
        ],
    }
}

fn across(extend: Extend, length: f32) -> Paint {
    filled(
        4,
        Paint::LinearGradient {
            line: red_to_blue(extend),
            p0: [0.0, 0.0],
            p1: [length, 0.0],
            p2: [0.0, length],
        },
    )
}

#[test]
fn a_version_0_glyph_is_its_layers_in_palette_colors() {
    let font = read(COLR).expect("font");
    assert!(font.has_color_layers());
    assert_eq!(
        glyph(10),
        ColorGlyph {
            paint: Paint::Layers(vec![
                filled(2, Paint::Solid(RED)),
                filled(3, Paint::Solid(BLUE)),
            ]),
            clip: None,
        }
    );
}

#[test]
fn the_color_of_the_text_is_named_and_a_palette_alpha_is_kept() {
    let Paint::Layers(layers) = glyph(11).paint else {
        panic!("layers");
    };
    assert_eq!(
        layers[0],
        filled(
            4,
            Paint::Solid(Color {
                rgb: None,
                alpha: 1.0
            })
        )
    );
    assert_eq!(
        layers[1],
        filled(
            6,
            Paint::Solid(Color {
                rgb: Some([0, 255, 0]),
                alpha: 128.0 / 255.0
            })
        )
    );
}

#[test]
fn gradients_read_their_points_stops_and_extend() {
    assert_eq!(glyph(12).paint, across(Extend::Pad, 1_000.0));
    assert_eq!(glyph(19).paint, across(Extend::Repeat, 250.0));
    assert_eq!(glyph(20).paint, across(Extend::Reflect, 250.0));
    assert_eq!(
        glyph(13).paint,
        filled(
            4,
            Paint::RadialGradient {
                line: red_to_blue(Extend::Pad),
                c0: [500.0, 500.0],
                r0: 0.0,
                c1: [500.0, 500.0],
                r1: 500.0,
            }
        )
    );
    // Stored as -1 and 1 half turns less one: once around from the x axis.
    assert_eq!(
        glyph(14).paint,
        filled(
            4,
            Paint::SweepGradient {
                line: red_to_blue(Extend::Pad),
                center: [500.0, 500.0],
                start: 0.0,
                end: 360.0,
            }
        )
    );
}

#[test]
fn layers_a_clip_box_and_a_glyph_used_again_are_read() {
    assert_eq!(
        glyph(18),
        ColorGlyph {
            paint: Paint::Layers(vec![
                filled(2, Paint::Solid(RED)),
                filled(3, Paint::Solid(BLUE)),
            ]),
            clip: None,
        }
    );
    let moved = glyph(15);
    assert_eq!(
        moved.clip,
        Some(ClipBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1_000.0,
            y_max: 1_000.0,
        })
    );
    assert_eq!(
        moved.paint,
        Paint::Transform {
            transform: Affine {
                dx: 500.0,
                dy: 500.0,
                ..Affine::IDENTITY
            },
            paint: Box::new(filled(5, Paint::Solid(RED))),
        }
    );
    assert_eq!(glyph(17).paint, Paint::ColorGlyph(Box::new(glyph(12))));
    assert_eq!(glyph(17).clip, None);
}

#[test]
fn a_composite_reads_its_source_mode_and_backdrop() {
    assert_eq!(
        glyph(16).paint,
        Paint::Composite {
            source: Box::new(filled(2, Paint::Solid(RED))),
            mode: CompositeMode::SrcIn,
            backdrop: Box::new(filled(6, Paint::Solid(BLUE))),
        }
    );
}

#[test]
fn each_kind_of_transform_is_one_matrix() {
    let expected = [
        // A matrix as stored.
        (21, [0.5, 0.0, 0.0, 0.5, 250.0, 250.0]),
        // A scale, and the same about the middle of the em.
        (22, [1.5, 0.0, 0.0, 0.5, 0.0, 0.0]),
        (23, [1.5, 0.0, 0.0, 0.5, -250.0, 250.0]),
        (24, [0.5, 0.0, 0.0, 0.5, 0.0, 0.0]),
        (25, [0.5, 0.0, 0.0, 0.5, 250.0, 250.0]),
        // A quarter turn counter-clockwise.
        (26, [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
        (27, [0.0, 1.0, -1.0, 0.0, 1_000.0, 0.0]),
        // A lean of 45 degrees along x, and one along y.
        (28, [1.0, 0.0, -1.0, 1.0, 0.0, 0.0]),
        (29, [1.0, 1.0, 0.0, 1.0, 0.0, -500.0]),
    ];
    for (glyph_id, matrix) in expected {
        let Paint::Transform { transform, paint } = glyph(glyph_id).paint else {
            panic!("glyph {glyph_id} is not a transform");
        };
        let read = [
            transform.xx,
            transform.yx,
            transform.xy,
            transform.yy,
            transform.dx,
            transform.dy,
        ];
        for (got, want) in read.iter().zip(matrix) {
            assert!((got - want).abs() < 1.0e-3, "glyph {glyph_id}: {read:?}");
        }
        assert_eq!(*paint, filled(5, Paint::Solid(RED)));
    }
    // The quarter turn sends the box's far corner up and to the left.
    let Paint::Transform { transform, .. } = glyph(26).paint else {
        panic!("transform");
    };
    let (x, y) = transform.apply(500.0, 500.0);
    assert!((x + 500.0).abs() < 1.0e-2 && (y - 500.0).abs() < 1.0e-2);
    let twice = transform.then_after(&transform);
    let (x, y) = twice.apply(500.0, 0.0);
    assert!((x + 500.0).abs() < 1.0e-2 && y.abs() < 1.0e-2);
}

#[test]
fn variable_paints_read_as_their_default_values() {
    assert_eq!(
        glyph(30).paint,
        Paint::Transform {
            transform: Affine {
                dx: 500.0,
                dy: 500.0,
                ..Affine::IDENTITY
            },
            paint: Box::new(filled(
                5,
                Paint::Solid(Color {
                    rgb: Some([255, 0, 0]),
                    alpha: 0.5
                })
            )),
        }
    );
    assert_eq!(glyph(31).paint, glyph(12).paint);
}

#[test]
fn a_paint_graph_that_loops_or_grows_without_end_is_a_bad_table() {
    // A glyph that names itself, a layer that lists itself, two glyphs
    // that name each other, and a paint whose child is itself.
    for glyph_id in [40, 41, 48, 49, 51] {
        assert_eq!(refusal(glyph_id), FontError::BadTable, "glyph {glyph_id}");
    }
    // Forty paints deep, and seventy layers of seventy layers.
    assert_eq!(refusal(42), FontError::BadTable);
    assert_eq!(refusal(43), FontError::BadTable);
}

#[test]
fn a_paint_that_names_what_the_face_lacks_is_an_error() {
    // An outline, a palette entry, and a color glyph that are not there.
    assert_eq!(refusal(44), FontError::BadTable);
    assert_eq!(refusal(46), FontError::BadTable);
    assert_eq!(refusal(50), FontError::BadTable);
    // A paint format and a blend mode that do not exist.
    assert_eq!(refusal(45), FontError::Unsupported);
    assert_eq!(refusal(47), FontError::Unsupported);
    // The rest of the face still reads.
    assert!(read(COLR).expect("font").color_glyph(10).is_ok());
}

#[test]
fn a_face_or_a_glyph_without_color_layers_has_none() {
    let plain = read(PLAIN).expect("font");
    assert!(!plain.has_color_layers());
    assert_eq!(plain.color_glyph(2), Ok(None));
    let font = read(COLR).expect("font");
    assert_eq!(font.color_glyph(1), Ok(None));
    assert_eq!(font.color_glyph(52), Err(FontError::UnknownGlyph));
    // A table of a version this reader does not know is passed over.
    let mut later = COLR.to_vec();
    let at = table_at(COLR, b"COLR").0;
    later[at + 1] = 2;
    let later = read(&later).expect("font");
    assert!(!later.has_color_layers());
    assert_eq!(later.color_glyph(10), Ok(None));
}

#[test]
fn a_torn_color_table_is_an_error_and_never_a_panic() {
    let (_, length) = table_at(COLR, b"COLR");
    // An empty table is no table, and a header cut short does not read.
    let mut none = COLR.to_vec();
    set_table_length(&mut none, b"COLR", 0);
    assert!(!read(&none).expect("font").has_color_layers());
    for cut in 1..34 {
        let mut torn = COLR.to_vec();
        set_table_length(&mut torn, b"COLR", cut);
        assert_eq!(read(&torn).err(), Some(FontError::Truncated), "cut {cut}");
    }
    // Cut anywhere after the header, every glyph answers or is refused.
    let mut refused = 0;
    for cut in (34..length).step_by(3) {
        let mut torn = COLR.to_vec();
        set_table_length(&mut torn, b"COLR", cut);
        let Ok(font) = read(&torn) else {
            refused += 1;
            continue;
        };
        for glyph_id in 0..font.glyph_count() {
            if font.color_glyph(glyph_id).is_err() {
                refused += 1;
            }
        }
    }
    assert!(refused > 0);
    // The last glyph's paint is at the end of what the lists point to.
    let mut torn = COLR.to_vec();
    set_table_length(&mut torn, b"COLR", 40);
    assert!(matches!(
        read(&torn).and_then(|font| font.color_glyph(12)),
        Err(FontError::Truncated)
    ));
    // A palette cut short does not read either.
    let mut torn = COLR.to_vec();
    set_table_length(&mut torn, b"CPAL", 20);
    assert_eq!(read(&torn).err(), Some(FontError::Truncated));
}

#[test]
fn hostile_bytes_in_the_color_table_never_panic() {
    let (at, length) = table_at(COLR, b"COLR");
    let mut answered = 0_u32;
    for index in 0..length {
        for value in [0x00, 0x7F, 0xFF] {
            let mut bytes = COLR.to_vec();
            if bytes[at + index] == value {
                continue;
            }
            bytes[at + index] = value;
            let Ok(font) = read(&bytes) else {
                continue;
            };
            for glyph_id in [10, 12, 13, 14, 16, 17, 18, 21, 31, 41, 42, 43] {
                if font.color_glyph(glyph_id).is_ok() {
                    answered += 1;
                }
            }
        }
    }
    assert!(answered > 0);
}

/// Where a table starts, and its length.
fn table_at(bytes: &[u8], tag: &[u8; 4]) -> (usize, usize) {
    let at = record_at(bytes, tag);
    let field = |at: usize| {
        usize::try_from(u32::from_be_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]))
        .expect("field")
    };
    (field(at + 8), field(at + 12))
}

fn record_at(bytes: &[u8], tag: &[u8; 4]) -> usize {
    let count = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
    (0..count)
        .map(|index| 12 + index * 16)
        .find(|at| &bytes[*at..*at + 4] == tag)
        .expect("table")
}

fn set_table_length(bytes: &mut [u8], tag: &[u8; 4], length: usize) {
    let at = record_at(bytes, tag);
    bytes[at + 12..at + 16].copy_from_slice(&u32::try_from(length).expect("length").to_be_bytes());
}
