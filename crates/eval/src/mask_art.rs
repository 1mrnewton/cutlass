//! Drawn masks' pictures: a pen outline filled, brush strokes painted and
//! rubbed out in order, or a text's letters filled, each drawn into the
//! mask's box with room round it for the feather, then blurred by the
//! feather.
//!
//! A picture is about as many pixels across as the box covers on the
//! canvas, in steps so a keyed size does not draw it again at every
//! frame, and at most [`MAX_ART_EDGE`] each way, so eight of them stacked
//! fit in one GPU texture. Pictures are kept by what they were drawn
//! from, so scrubbing, or the CPU and GPU passes of one frame, draw a
//! mask once.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use cutlass_effects::{ArtPicture, Mask, MaskArt};
use cutlass_lottie::{Edge, LineCap, LineJoin, Outline, Pen, coverage, fill_edges, stroke_edges};

/// The widest or tallest a drawn mask's picture is.
pub(crate) const MAX_ART_EDGE: u32 = 1_024;
/// The smallest.
const MIN_ART_EDGE: u32 = 16;
/// Picture sizes go up in steps of this many pixels.
const ART_STEP: u32 = 32;
/// The most pictures kept; past it they all go.
const MAX_KEPT: usize = 64;
/// The farthest past its box a feather reaches, as a part of the box.
const MAX_SOFT: f32 = 1.0;

fn kept() -> &'static Mutex<HashMap<u64, ArtPicture>> {
    static KEPT: OnceLock<Mutex<HashMap<u64, ArtPicture>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `art`'s picture for `mask` on a `canvas`-sized frame, the clip taken
/// to fill the canvas. `None` when it draws nothing.
pub(crate) fn art_picture(art: &MaskArt, mask: &Mask, canvas: (u32, u32)) -> Option<ArtPicture> {
    // The feather in box parts, as a shape's feather is measured.
    let soft = (mask.feather * 0.5 / mask.width.min(mask.height).max(1e-3)).clamp(0.0, MAX_SOFT);
    let reach = 1.0 + soft;
    #[allow(clippy::cast_precision_loss)]
    let side = |box_part: f32, canvas: u32| {
        let pixels = (box_part * canvas as f32 * reach).clamp(1.0, f32::from(u16::MAX));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pixels = pixels.ceil() as u32;
        pixels
            .div_ceil(ART_STEP)
            .saturating_mul(ART_STEP)
            .clamp(MIN_ART_EDGE, MAX_ART_EDGE)
    };
    let (width, height) = (side(mask.width, canvas.0), side(mask.height, canvas.1));
    let key = {
        let mut hasher = DefaultHasher::new();
        hash_art(art, &mut hasher);
        (width, height, soft.to_bits()).hash(&mut hasher);
        hasher.finish()
    };
    if let Ok(kept) = kept().lock()
        && let Some(picture) = kept.get(&key)
    {
        return Some(picture.clone());
    }
    let picture = draw(art, width, height, reach, soft)?;
    if let Ok(mut kept) = kept().lock() {
        if kept.len() >= MAX_KEPT {
            kept.clear();
        }
        kept.insert(key, picture.clone());
    }
    Some(picture)
}

/// Feed everything `art` is drawn from to `hasher`.
fn hash_art(art: &MaskArt, hasher: &mut DefaultHasher) {
    let pair = |hasher: &mut DefaultHasher, [x, y]: [f32; 2]| {
        x.to_bits().hash(hasher);
        y.to_bits().hash(hasher);
    };
    match art {
        MaskArt::Pen { points } => {
            0_u8.hash(hasher);
            for point in points {
                for value in point
                    .at
                    .iter()
                    .chain(&point.handle_in)
                    .chain(&point.handle_out)
                {
                    value.to_bits().hash(hasher);
                }
            }
        }
        MaskArt::Brush { strokes } => {
            1_u8.hash(hasher);
            for stroke in strokes {
                (stroke.size.to_bits(), stroke.erase, stroke.points.len()).hash(hasher);
                for point in &stroke.points {
                    pair(hasher, *point);
                }
            }
        }
        MaskArt::Text { outlines, .. } => {
            2_u8.hash(hasher);
            for outline in outlines {
                outline.len().hash(hasher);
                for point in outline {
                    pair(hasher, *point);
                }
            }
        }
    }
}

fn draw(art: &MaskArt, width: u32, height: u32, reach: f32, soft: f32) -> Option<ArtPicture> {
    #[allow(clippy::cast_precision_loss)]
    let (w, h) = (width as f32, height as f32);
    // A box point, -1 to 1 with y up, as a picture pixel.
    let to_pixel = |[x, y]: [f32; 2]| {
        [
            f64::from((x / reach + 1.0) * 0.5 * w),
            f64::from((1.0 - y / reach) * 0.5 * h),
        ]
    };
    let mut cover: Vec<f32> = vec![0.0; width as usize * height as usize];
    let mut add = |edges: &[Edge], erase: bool| {
        let fill = coverage(edges, width, height);
        for (kept, new) in cover.iter_mut().zip(fill) {
            let new = f32::from(new) / 255.0;
            *kept = if erase {
                *kept * (1.0 - new)
            } else {
                *kept + new - *kept * new
            };
        }
    };
    match art {
        MaskArt::Pen { points } => {
            #[allow(clippy::cast_possible_truncation)]
            let point = |at: [f64; 2]| to_pixel([at[0] as f32, at[1] as f32]);
            let cubics = (0..points.len())
                .map(|index| {
                    let (a, b) = (&points[index], &points[(index + 1) % points.len()]);
                    let offset = |at: [f64; 2], by: [f64; 2]| [at[0] + by[0], at[1] + by[1]];
                    [
                        point(a.at),
                        point(offset(a.at, a.handle_out)),
                        point(offset(b.at, b.handle_in)),
                        point(b.at),
                    ]
                })
                .collect();
            add(
                &fill_edges(&[Outline {
                    cubics,
                    closed: true,
                }]),
                false,
            );
        }
        MaskArt::Brush { strokes } => {
            for stroke in strokes {
                let mut line: Vec<[f64; 2]> =
                    stroke.points.iter().map(|point| to_pixel(*point)).collect();
                // A dab is a stroke of no length, drawn by its round ends.
                if line.len() == 1 {
                    line.push([line[0][0] + 0.01, line[0][1]]);
                }
                let cubics = line
                    .windows(2)
                    .map(|pair| [pair[0], pair[0], pair[1], pair[1]])
                    .collect();
                let pen = Pen {
                    width: f64::from(stroke.size * w * 0.5 / reach),
                    cap: LineCap::Round,
                    join: LineJoin::Round,
                    miter_limit: 4.0,
                };
                let outline = Outline {
                    cubics,
                    closed: false,
                };
                add(&stroke_edges(&[outline], &pen, None), stroke.erase);
            }
        }
        MaskArt::Text { outlines, .. } => {
            let shapes: Vec<Outline> = outlines
                .iter()
                .filter(|outline| outline.len() >= 3)
                .map(|outline| {
                    let corners: Vec<[f64; 2]> =
                        outline.iter().map(|point| to_pixel(*point)).collect();
                    Outline {
                        cubics: (0..corners.len())
                            .map(|index| {
                                let (a, b) = (corners[index], corners[(index + 1) % corners.len()]);
                                [a, a, b, b]
                            })
                            .collect(),
                        closed: true,
                    }
                })
                .collect();
            add(&fill_edges(&shapes), false);
        }
    }
    // The feather's blur: about the width a shape's feather spreads.
    let radius = soft * w * 0.5 / reach * 0.6;
    let cover = if radius >= 0.5 {
        cutlass_picture::blur_mask(&cover, width as usize, height as usize, radius)
    } else {
        cover
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let alpha: Arc<[u8]> = cover
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let picture = ArtPicture {
        width,
        height,
        alpha,
        reach,
    };
    picture.is_valid().then_some(picture)
}

/// A text mask's drawing: `text` set in the face in `bytes`, named `font`,
/// its letters stretched to fill the mask's box. Its `aspect` says how
/// wide the words are for their height, to size the box by. `None` when
/// the face does not read or the text is too long.
#[must_use]
pub fn text_mask_art(bytes: &[u8], font: &str, text: &str) -> Option<MaskArt> {
    if text.chars().count() > cutlass_effects::MAX_MASK_TEXT {
        return None;
    }
    let (polygons, [x0, y0, x1, y1]) = cutlass_text::text_outlines(bytes, text).ok()?;
    let (across, up) = ((x1 - x0).max(1e-3), (y1 - y0).max(1e-3));
    let (mid_x, mid_y) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    // Four places are finer than a pixel on any screen, and keep the
    // record small.
    let round = |value: f32| (value * 10_000.0).round() / 10_000.0;
    let outlines = polygons
        .into_iter()
        .map(|polygon| {
            polygon
                .into_iter()
                .map(|[x, y]| {
                    [
                        round((x - mid_x) / across * 2.0),
                        round((y - mid_y) / up * 2.0),
                    ]
                })
                .collect()
        })
        .collect();
    let art = MaskArt::Text {
        text: text.to_owned(),
        font: font.to_owned(),
        aspect: if x1 > x0 { across / up } else { 1.0 },
        outlines,
    };
    art.is_valid().then_some(art)
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{BrushStroke, MaskShape};
    use cutlass_model::ShapePoint;

    use super::*;

    fn diamond() -> MaskArt {
        let at = |x: f64, y: f64| ShapePoint {
            at: [x, y],
            ..ShapePoint::default()
        };
        MaskArt::Pen {
            points: vec![at(0.0, 0.8), at(0.8, 0.0), at(0.0, -0.8), at(-0.8, 0.0)],
        }
    }

    #[test]
    fn a_pen_outline_keeps_inside_and_not_outside() {
        let mask = Mask::new(MaskShape::Pen);
        let picture = art_picture(&diamond(), &mask, (320, 320)).unwrap();
        assert!(picture.coverage(0.0, 0.0) > 0.99);
        assert!(picture.coverage(0.35, 0.35) > 0.9);
        assert!(picture.coverage(0.6, 0.6) < 0.01);
        assert!(picture.coverage(0.95, 0.0) < 0.01);
    }

    #[test]
    fn a_brush_paints_its_strokes_and_rubs_out_with_an_eraser() {
        let line = |y: f32, size: f32, erase: bool| BrushStroke {
            points: vec![[-0.6, y], [0.6, y]],
            size,
            erase,
        };
        let art = MaskArt::Brush {
            strokes: vec![
                line(0.0, 0.4, false),
                line(0.0, 0.1, true),
                line(0.5, 0.1, false),
            ],
        };
        let picture = art_picture(&art, &Mask::new(MaskShape::Brush), (400, 400)).unwrap();
        // Painted either side of the rubbed-out middle.
        assert!(picture.coverage(0.0, 0.15) > 0.9);
        assert!(picture.coverage(0.0, -0.15) > 0.9);
        assert!(picture.coverage(0.0, 0.0) < 0.05);
        assert!(picture.coverage(0.3, 0.5) > 0.9);
        assert!(picture.coverage(0.0, 0.8) < 0.01);
        // A single dab is a round spot.
        let dab = MaskArt::Brush {
            strokes: vec![BrushStroke {
                points: vec![[0.0, 0.0]],
                size: 0.5,
                erase: false,
            }],
        };
        let spot = art_picture(&dab, &Mask::new(MaskShape::Brush), (400, 400)).unwrap();
        assert!(spot.coverage(0.0, 0.0) > 0.99);
        assert!(spot.coverage(0.2, 0.0) > 0.9);
        assert!(spot.coverage(0.2, 0.2) < 0.05);
    }

    #[test]
    fn a_feather_softens_the_edge_and_room_is_left_for_it() {
        let sharp = Mask::new(MaskShape::Pen);
        let soft = Mask {
            feather: 0.4,
            ..sharp
        };
        let hard = art_picture(&diamond(), &sharp, (320, 320)).unwrap();
        let blurred = art_picture(&diamond(), &soft, (320, 320)).unwrap();
        assert!(blurred.reach > hard.reach);
        // Just inside the edge is softer, just outside it is no longer bare.
        assert!(blurred.coverage(0.35, 0.35) < hard.coverage(0.35, 0.35));
        assert!(blurred.coverage(0.45, 0.45) > 0.02);
        assert!(blurred.coverage(0.0, 0.0) > 0.95);
    }

    #[test]
    fn a_picture_is_drawn_once_for_a_mask_that_does_not_change() {
        let mask = Mask {
            x: 0.123,
            ..Mask::new(MaskShape::Pen)
        };
        let art = MaskArt::Pen {
            points: vec![
                ShapePoint {
                    at: [-0.4, -0.3],
                    ..ShapePoint::default()
                },
                ShapePoint {
                    at: [0.41, -0.3],
                    ..ShapePoint::default()
                },
                ShapePoint {
                    at: [0.0, 0.52],
                    ..ShapePoint::default()
                },
            ],
        };
        let first = art_picture(&art, &mask, (640, 360)).unwrap();
        // Moved and turned, the same picture serves; only its size or its
        // drawing draws it again.
        let moved = Mask {
            x: -0.4,
            rotation: 30.0,
            ..mask
        };
        for _ in 0..10 {
            let again = art_picture(&art, &moved, (640, 360)).unwrap();
            assert!(Arc::ptr_eq(&again.alpha, &first.alpha));
        }
        let grown = Mask { width: 2.0, ..mask };
        let larger = art_picture(&art, &grown, (640, 360)).unwrap();
        assert!(larger.width > first.width);
        assert!(!Arc::ptr_eq(&larger.alpha, &first.alpha));
    }

    #[test]
    fn a_text_mask_holds_its_letters_in_its_box() {
        const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
        let art = text_mask_art(TTF, "sample.ttf", "AB").unwrap();
        let MaskArt::Text {
            aspect, outlines, ..
        } = &art
        else {
            panic!("not text: {art:?}");
        };
        assert!(*aspect > 0.5, "{aspect}");
        let points = outlines.iter().flatten();
        assert!(
            points
                .clone()
                .all(|[x, y]| x.abs() <= 1.0 && y.abs() <= 1.0)
        );
        assert!(points.clone().any(|[x, _]| *x < -0.99));
        assert!(points.clone().any(|[_, y]| *y > 0.99));
        let picture = art_picture(&art, &Mask::new(MaskShape::Text), (640, 360)).unwrap();
        let covered = picture.alpha.iter().filter(|value| **value > 128).count();
        assert!(covered > picture.alpha.len() / 10, "{covered}");
        assert!(covered < picture.alpha.len() * 9 / 10, "{covered}");
        assert_eq!(text_mask_art(b"not a font", "x", "AB"), None);
    }
}
