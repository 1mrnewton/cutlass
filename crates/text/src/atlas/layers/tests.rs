//! Layered color glyphs of the built face, painted and read back by pixel.

use cutlass_font::{Affine, Color, ColorGlyph, CompositeMode, Font, FontError, Paint, read};

use super::{COLOR_EM, blend, coverage, paint};
use crate::TextError;
use crate::atlas::{Atlas, AtlasRoute, Edge, GlyphImage, GlyphKind, Point, route};

const COLR: &[u8] = include_bytes!("../../../../font/fixtures/colr.ttf");
const BOTH: &[u8] = include_bytes!("../../../../font/fixtures/colr-sbix.ttf");

const RED: [u8; 4] = [255, 0, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const CLEAR: [u8; 4] = [0, 0, 0, 0];

fn image(glyph_id: u16) -> GlyphImage {
    let font = read(COLR).expect("font");
    let mut atlas = Atlas::new();
    let image = atlas.glyph(&font, glyph_id).expect("glyph").clone();
    assert_eq!(image.kind, GlyphKind::Color, "glyph {glyph_id}");
    assert!(image.sdf.is_empty());
    image
}

/// The premultiplied color of the pixel that holds a point in font units.
fn at(image: &GlyphImage, x: f32, y: f32) -> [u8; 4] {
    let column = ((x - image.left) / image.unit).floor();
    let row = ((image.top - y) / image.unit).floor();
    assert!(column >= 0.0 && row >= 0.0, "point outside the image");
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (column, row) = (column as u32, row as u32);
    assert!(column < image.width && row < image.height);
    let offset = usize::try_from((row * image.width + column) * 4).expect("offset");
    [
        image.color[offset],
        image.color[offset + 1],
        image.color[offset + 2],
        image.color[offset + 3],
    ]
}

/// Whether every channel of `got` is within `slack` of `want`.
fn near(got: [u8; 4], want: [u8; 4], slack: u8) -> bool {
    got.iter()
        .zip(want)
        .all(|(got, want)| got.abs_diff(want) <= slack)
}

#[track_caller]
fn expect_near(image: &GlyphImage, x: f32, y: f32, want: [u8; 4], slack: u8) {
    let got = at(image, x, y);
    assert!(
        near(got, want, slack),
        "at ({x}, {y}): {got:?}, not {want:?}"
    );
}

#[test]
fn a_two_layer_glyph_has_each_color_in_its_place() {
    let font = read(COLR).expect("font");
    assert_eq!(route(&font), AtlasRoute::SignedDistance);
    let mut atlas = Atlas::new();
    // The same two layers as a version 0 glyph and as a version 1 glyph.
    for glyph_id in [10, 18] {
        let image = atlas.glyph(&font, glyph_id).expect("glyph").clone();
        assert_eq!(image.kind, GlyphKind::Color);
        // An em of 128 pixels and one clear pixel all round.
        assert_eq!((image.width, image.height), (130, 130));
        assert_eq!(image.pixels_per_em, COLOR_EM);
        assert_eq!(image.unit, 1_000.0 / 128.0);
        assert_eq!((image.left, image.top), (-image.unit, 1_000.0 + image.unit));
        assert_eq!(at(&image, 250.0, 500.0), RED);
        assert_eq!(at(&image, 750.0, 500.0), BLUE);
        assert_eq!(at(&image, 4.0, 996.0), RED);
        assert_eq!(at(&image, 996.0, 4.0), BLUE);
        // The edge of the image is clear, and the rest is covered.
        assert_eq!(at(&image, -4.0, 500.0), CLEAR);
        assert_eq!(at(&image, 500.0, 1_004.0), CLEAR);
        let covered = image
            .color
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|pixel| pixel[3] == 255)
            .count();
        assert_eq!(covered, 128 * 128);
    }
    // Each glyph is painted once.
    assert_eq!(atlas.paints(), 2);
    let _cached = atlas.glyph(&font, 10).expect("cached");
    assert_eq!(atlas.paints(), 2);
    // A letter of the same face is still a distance field.
    let letter = atlas.glyph(&font, 1).expect("letter");
    assert_eq!(letter.kind, GlyphKind::Distance);
}

#[test]
fn the_color_of_the_text_is_black_and_a_palette_alpha_blends_over_it() {
    let image = image(11);
    assert_eq!(at(&image, 100.0, 100.0), [0, 0, 0, 255]);
    assert_eq!(at(&image, 500.0, 500.0), [0, 128, 0, 255]);
    // The palette's alpha and the paint's multiply: half green over nothing.
    assert_eq!(at(&self::image(38), 500.0, 500.0), [0, 128, 0, 128]);
    // A paint's own alpha: red at half.
    assert_eq!(at(&self::image(30), 750.0, 750.0), [128, 0, 0, 128]);
}

#[test]
fn a_linear_gradient_runs_along_its_line_and_extends_as_it_says() {
    let plain = image(12);
    expect_near(&plain, 250.0, 500.0, [191, 0, 64, 255], 3);
    expect_near(&plain, 750.0, 100.0, [64, 0, 191, 255], 3);
    expect_near(&plain, 4.0, 900.0, RED, 3);
    // The variable form and a glyph used again draw the same picture.
    assert_eq!(image(31).color, plain.color);
    assert_eq!(image(17).color, plain.color);
    // 250 units long, repeated: the middle of the first and second run.
    let repeated = image(19);
    expect_near(&repeated, 125.0, 500.0, [128, 0, 128, 255], 5);
    expect_near(&repeated, 375.0, 500.0, [128, 0, 128, 255], 5);
    expect_near(&repeated, 262.0, 500.0, [243, 0, 12, 255], 8);
    // Reflected: the second run goes back from blue to red.
    let reflected = image(20);
    expect_near(&reflected, 262.0, 500.0, [12, 0, 243, 255], 8);
    expect_near(&reflected, 512.0, 500.0, [243, 0, 12, 255], 8);
    // Stripes that lean: the same color along the line from p0 to p2.
    let leaning = image(36);
    expect_near(&leaning, 500.0, 4.0, [128, 0, 128, 255], 4);
    expect_near(&leaning, 750.0, 500.0, [128, 0, 128, 255], 4);
    expect_near(&leaning, 500.0, 996.0, RED, 4);
    // Three stops from a quarter of the way: padded before and after.
    let three = image(37);
    expect_near(&three, 100.0, 500.0, RED, 0);
    expect_near(&three, 375.0, 500.0, [255, 128, 128, 255], 5);
    expect_near(&three, 500.0, 500.0, [255, 255, 255, 255], 5);
    expect_near(&three, 900.0, 500.0, BLUE, 0);
    // Red to clear blue: channels and alpha change evenly, then multiply.
    expect_near(&image(32), 500.0, 500.0, [64, 0, 64, 128], 3);
}

#[test]
fn radial_and_sweep_gradients_follow_their_circles_and_angles() {
    let radial = image(13);
    expect_near(&radial, 503.0, 503.0, RED, 5);
    expect_near(&radial, 500.0, 900.0, [51, 0, 204, 255], 4);
    expect_near(&radial, 950.0, 950.0, BLUE, 0);
    // Two circles with different centers: halfway is the circle of radius
    // 175 about (400, 500).
    let cone = image(35);
    expect_near(&cone, 400.0, 675.0, [128, 0, 128, 255], 6);
    expect_near(&cone, 575.0, 500.0, [128, 0, 128, 255], 6);
    // Once round from the positive x axis, counter-clockwise with y up.
    let sweep = image(14);
    expect_near(&sweep, 900.0, 520.0, RED, 8);
    expect_near(&sweep, 900.0, 480.0, BLUE, 8);
    expect_near(&sweep, 500.0, 900.0, [191, 0, 64, 255], 4);
    expect_near(&sweep, 100.0, 500.0, [128, 0, 128, 255], 4);
    expect_near(&sweep, 500.0, 100.0, [64, 0, 191, 255], 4);
    // A quarter turn, padded: the last color the rest of the way round.
    let sector = image(33);
    expect_near(&sector, 900.0, 900.0, [128, 0, 128, 255], 4);
    expect_near(&sector, 100.0, 500.0, BLUE, 0);
    expect_near(&sector, 500.0, 100.0, BLUE, 0);
    // Repeated: the next quarter starts again from red.
    let turns = image(34);
    expect_near(&turns, 100.0, 900.0, [128, 0, 128, 255], 4);
    expect_near(&turns, 480.0, 900.0, RED, 12);
}

#[test]
fn transforms_move_the_outline_and_the_clip_box_sets_the_frame() {
    // The quarter box moved up and right, in a clip box of the em.
    let moved = image(15);
    assert_eq!((moved.width, moved.height), (130, 130));
    assert_eq!(at(&moved, 750.0, 750.0), RED);
    assert_eq!(at(&moved, 504.0, 504.0), RED);
    assert_eq!(at(&moved, 250.0, 250.0), CLEAR);
    assert_eq!(at(&moved, 750.0, 250.0), CLEAR);
    assert_eq!(at(&moved, 496.0, 750.0), CLEAR);
    // A clip box 2500 units across does not fit at 128 pixels to the em,
    // so the glyph is painted smaller, to fill the cell.
    let wide = image(21);
    assert_eq!((wide.width, wide.height), (256, 256));
    assert!((wide.pixels_per_em - 101.6).abs() < 0.01);
    assert!((wide.unit * wide.pixels_per_em - 1_000.0).abs() < 0.01);
    // Each transform puts the red box where its matrix says.
    let inside_and_outside = [
        (21, (375.0, 375.0), (125.0, 125.0)),
        (22, (700.0, 100.0), (700.0, 400.0)),
        (23, (-200.0, 375.0), (600.0, 375.0)),
        (24, (125.0, 125.0), (375.0, 375.0)),
        (25, (375.0, 375.0), (125.0, 125.0)),
        (26, (-250.0, 250.0), (250.0, 250.0)),
        (27, (750.0, 250.0), (250.0, 250.0)),
        (28, (-200.0, 400.0), (300.0, 400.0)),
        (29, (250.0, 0.0), (250.0, 600.0)),
    ];
    for (glyph_id, inside, outside) in inside_and_outside {
        let image = image(glyph_id);
        assert_eq!(at(&image, inside.0, inside.1), RED, "glyph {glyph_id}");
        assert_eq!(at(&image, outside.0, outside.1), CLEAR, "glyph {glyph_id}");
    }
}

#[test]
fn a_composite_keeps_what_its_mode_says() {
    // The red left half, kept only where the blue middle is.
    let kept = image(16);
    assert_eq!(at(&kept, 375.0, 500.0), RED);
    assert_eq!(at(&kept, 125.0, 500.0), CLEAR);
    assert_eq!(at(&kept, 625.0, 500.0), CLEAR);
    assert_eq!(at(&kept, 375.0, 900.0), CLEAR);
    // Orange at three quarters over gray, one glyph for each mode: where
    // only the gray is, where both are, and where only the orange is.
    let gray = [128, 128, 128, 255];
    let orange = [191, 96, 48, 191];
    let expected = [
        (60, CLEAR, CLEAR, CLEAR),
        (61, CLEAR, orange, orange),
        (62, gray, gray, CLEAR),
        (63, gray, [223, 128, 80, 255], orange),
        (64, gray, gray, orange),
        (65, CLEAR, orange, CLEAR),
        (67, CLEAR, CLEAR, orange),
        (83, gray, [128, 80, 56, 255], orange),
    ];
    for (glyph_id, under, both, over) in expected {
        let image = image(glyph_id);
        expect_near(&image, 125.0, 500.0, under, 1);
        expect_near(&image, 375.0, 500.0, both, 1);
        expect_near(&image, 625.0, 500.0, over, 1);
    }
}

/// `over` onto `under` as straight colors with alphas, rounded to bytes.
fn blended(mode: CompositeMode, over: [f32; 4], under: [f32; 4]) -> [u8; 4] {
    let premultiplied =
        |[red, green, blue, alpha]: [f32; 4]| [red * alpha, green * alpha, blue * alpha, alpha];
    blend(mode, premultiplied(over), premultiplied(under)).map(super::byte)
}

#[test]
fn each_blend_mode_follows_its_formula() {
    let orange = [1.0, 0.5, 0.25, 1.0];
    let gray = [0.5, 0.5, 0.5, 1.0];
    // Modes that mix two opaque colors, channel by channel.
    let mixes = [
        (CompositeMode::Multiply, [128, 64, 32, 255]),
        (CompositeMode::Screen, [255, 191, 159, 255]),
        (CompositeMode::Overlay, [255, 128, 64, 255]),
        (CompositeMode::Darken, [128, 128, 64, 255]),
        (CompositeMode::Lighten, [255, 128, 128, 255]),
        (CompositeMode::ColorDodge, [255, 255, 170, 255]),
        (CompositeMode::ColorBurn, [128, 0, 0, 255]),
        (CompositeMode::HardLight, [255, 128, 64, 255]),
        (CompositeMode::SoftLight, [180, 128, 96, 255]),
        (CompositeMode::Difference, [128, 0, 64, 255]),
        (CompositeMode::Exclusion, [128, 128, 128, 255]),
        // The backdrop is gray, so it has no hue and no saturation to give.
        (CompositeMode::Hue, [128, 128, 128, 255]),
        (CompositeMode::Saturation, [128, 128, 128, 255]),
        (CompositeMode::Color, [224, 96, 33, 255]),
        (CompositeMode::Luminosity, [159, 159, 159, 255]),
    ];
    for (mode, want) in mixes {
        let got = blended(mode, orange, gray);
        assert!(near(got, want, 1), "{mode:?}: {got:?}, not {want:?}");
    }
    // Modes that keep or drop: half red onto opaque blue.
    let red = [1.0, 0.0, 0.0, 0.5];
    let blue = [0.0, 0.0, 1.0, 1.0];
    let keeps = [
        (CompositeMode::Clear, [0, 0, 0, 0]),
        (CompositeMode::Src, [128, 0, 0, 128]),
        (CompositeMode::Dest, [0, 0, 255, 255]),
        (CompositeMode::SrcOver, [128, 0, 128, 255]),
        (CompositeMode::DestOver, [0, 0, 255, 255]),
        (CompositeMode::SrcIn, [128, 0, 0, 128]),
        (CompositeMode::DestIn, [0, 0, 128, 128]),
        (CompositeMode::SrcOut, [0, 0, 0, 0]),
        (CompositeMode::DestOut, [0, 0, 128, 128]),
        (CompositeMode::SrcAtop, [128, 0, 128, 255]),
        (CompositeMode::DestAtop, [0, 0, 128, 128]),
        (CompositeMode::Xor, [0, 0, 128, 128]),
        (CompositeMode::Plus, [128, 0, 255, 255]),
    ];
    for (mode, want) in keeps {
        let got = blended(mode, red, blue);
        assert!(near(got, want, 1), "{mode:?}: {got:?}, not {want:?}");
    }
    // Nothing onto nothing is nothing, in every mode, and never not a number.
    for (mode, _) in mixes.iter().chain(&keeps) {
        assert_eq!(blended(*mode, [0.0; 4], [0.0; 4]), [0, 0, 0, 0], "{mode:?}");
        let alone = blend(*mode, [0.5, 0.25, 0.0, 0.5], [0.0; 4]);
        assert!(alone.iter().all(|channel| channel.is_finite()), "{mode:?}");
    }
}

fn polygon(points: &[(f32, f32)]) -> Vec<Edge> {
    (0..points.len())
        .map(|index| {
            let (start, end) = (points[index], points[(index + 1) % points.len()]);
            Edge {
                start: Point {
                    x: start.0,
                    y: start.1,
                },
                end: Point { x: end.0, y: end.1 },
            }
        })
        .collect()
}

#[test]
fn coverage_is_anti_aliased_and_follows_the_nonzero_rule() {
    // A box from 1.5 to 3.25 across and 1 to 2.5 down, in a 5 by 4 image.
    let mask = coverage(
        &polygon(&[(1.5, 1.0), (3.25, 1.0), (3.25, 2.5), (1.5, 2.5)]),
        5,
        4,
    );
    let cover = |column: usize, row: usize| mask[row * 5 + column];
    assert_eq!(cover(2, 1), 1.0);
    assert_eq!(cover(1, 1), 0.5);
    assert_eq!(cover(3, 1), 0.25);
    assert_eq!(cover(2, 2), 0.5);
    assert_eq!(cover(1, 2), 0.25);
    assert_eq!(cover(0, 1), 0.0);
    assert_eq!(cover(2, 0), 0.0);
    assert_eq!(cover(2, 3), 0.0);
    // Two boxes wound the same way overlap: covered once, not twice.
    let mut twice = polygon(&[(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]);
    twice.extend(polygon(&[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)]));
    let mask = coverage(&twice, 4, 4);
    assert!(mask.iter().all(|cover| *cover <= 1.0));
    assert_eq!(mask[5], 1.0);
    assert_eq!(mask[3], 0.0);
    // A box wound the other way inside the first cuts a hole.
    let mut ring = polygon(&[(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]);
    ring.extend(polygon(&[(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)]));
    let mask = coverage(&ring, 3, 3);
    assert_eq!(mask[4], 0.0);
    assert_eq!(mask[0], 1.0);
    // A slope covers its pixels in part, and nothing is past one.
    let slope = coverage(&polygon(&[(0.0, 0.0), (4.0, 4.0), (0.0, 4.0)]), 4, 4);
    assert!((slope[0] - 0.5).abs() < 0.07);
    assert_eq!(slope[1], 0.0);
    assert_eq!(slope[4], 1.0);
    // Edges far outside the image, or none at all, are not a panic.
    let far = coverage(
        &polygon(&[(-1.0e9, -1.0e9), (1.0e9, -1.0e9), (0.0, 1.0e9)]),
        3,
        3,
    );
    assert!(far.iter().all(|cover| (0.0..=1.0).contains(cover)));
    assert!(coverage(&[], 3, 3).iter().all(|cover| *cover == 0.0));
    assert!(coverage(&twice, 0, 0).is_empty());
}

#[test]
fn a_leaning_edge_is_soft_and_every_pixel_is_premultiplied() {
    let leaning = image(28);
    let pixels = leaning.color.as_chunks::<4>().0;
    assert!(pixels.iter().any(|pixel| pixel[3] > 0 && pixel[3] < 255));
    for glyph_id in [14, 28, 32, 35, 63, 77] {
        let image = image(glyph_id);
        for [red, green, blue, alpha] in image.color.as_chunks::<4>().0 {
            assert!(red <= alpha && green <= alpha && blue <= alpha);
        }
    }
}

#[test]
fn a_hostile_paint_graph_is_an_error_and_paints_nothing() {
    let font = read(COLR).expect("font");
    let mut atlas = Atlas::new();
    // Loops, a graph too deep, and one too wide.
    for glyph_id in [40, 41, 42, 43, 48, 49, 51] {
        assert!(
            matches!(
                atlas.glyph(&font, glyph_id),
                Err(TextError::Font(FontError::BadTable))
            ),
            "glyph {glyph_id}"
        );
    }
    // A glyph, a palette entry, and a color glyph the face does not have.
    for glyph_id in [44, 46, 50] {
        assert!(matches!(
            atlas.glyph(&font, glyph_id),
            Err(TextError::Font(FontError::BadTable))
        ));
    }
    // A paint format and a blend mode that do not exist.
    for glyph_id in [45, 47] {
        assert!(matches!(
            atlas.glyph(&font, glyph_id),
            Err(TextError::Font(FontError::Unsupported))
        ));
    }
    assert_eq!(atlas.paints(), 0);
    assert!(atlas.image(40).is_none());
}

/// The whole em in red, under `transform`.
fn red_box(font: &Font<'_>, transform: Affine) -> Result<GlyphImage, TextError> {
    let red = Paint::Solid(Color {
        rgb: Some([255, 0, 0]),
        alpha: 1.0,
    });
    paint(
        font,
        &ColorGlyph {
            paint: Paint::Transform {
                transform,
                paint: Box::new(Paint::Glyph {
                    glyph_id: 4,
                    paint: Box::new(red),
                }),
            },
            clip: None,
        },
    )
}

#[test]
fn a_transform_that_blows_up_is_a_limit_and_a_huge_glyph_still_fits() {
    let font = read(COLR).expect("font");
    let scale = |by: f32| Affine {
        xx: by,
        yy: by,
        ..Affine::IDENTITY
    };
    // Past what a number holds, and not a number at all.
    assert!(matches!(
        red_box(&font, scale(f32::MAX)),
        Err(TextError::Limit)
    ));
    assert!(matches!(
        red_box(&font, scale(f32::NAN)),
        Err(TextError::Limit)
    ));
    assert!(matches!(
        red_box(&font, scale(f32::INFINITY)),
        Err(TextError::Limit)
    ));
    // A glyph a million ems across is painted small enough for the cell.
    let huge = red_box(&font, scale(1.0e6)).expect("huge");
    assert_eq!((huge.width, huge.height), (256, 256));
    assert_eq!(at(&huge, 5.0e8, 5.0e8), RED);
    // Flattened to nothing, the glyph is blank.
    let flat = red_box(&font, scale(0.0)).expect("flat");
    assert_eq!((flat.width, flat.height), (0, 0));
    assert!(flat.color.is_empty());
}

#[test]
fn layers_come_before_a_strike_and_a_strike_before_an_outline() {
    let font = read(BOTH).expect("font");
    let mut atlas = Atlas::new();
    // Glyph 10 has layers and a picture: the layers are painted.
    let layered = atlas.glyph(&font, 10).expect("layers").clone();
    assert_eq!((layered.width, layered.height), (130, 130));
    assert_eq!(layered.color, image(10).color);
    // The letter has a picture and an outline: the picture is taken.
    let letter = atlas.glyph(&font, 1).expect("picture");
    assert_eq!(letter.kind, GlyphKind::Color);
    assert_eq!((letter.width, letter.height), (2, 2));
    assert_eq!(&letter.color[..4], &RED);
    // A box with neither is its outline.
    let outline = atlas.glyph(&font, 4).expect("outline");
    assert_eq!(outline.kind, GlyphKind::Distance);
    assert!(outline.color.is_empty());
}
