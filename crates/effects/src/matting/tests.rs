use cutlass_model::{Draft, RawMaterial, Segment, TimeRange};
use serde_json::json;

use super::*;

fn custom() -> Matting {
    Matting {
        mode: MattingMode::Custom,
        picks: vec![Pick {
            frame: Micros(40_000),
            clicks: vec![PickClick {
                x: 0.5,
                y: 0.25,
                keep: true,
            }],
            brushes: vec![Brush {
                points: vec![[0.1, 0.1], [0.2, 0.2]],
                radius: 0.02,
                keep: false,
                smart: true,
            }],
        }],
        ..Matting::auto()
    }
}

#[test]
fn settings_round_trip_through_the_record() {
    for matting in [
        Matting::auto(),
        custom(),
        Matting {
            feather: 0.4,
            clean_up: 0.2,
            stroke: Some(Stroke::new(StrokeStyle::Dotted)),
            ..Matting::auto()
        },
    ] {
        let fields = matting.to_fields();
        assert_eq!(Matting::from_fields(&fields), Some(matting));
    }
}

#[test]
fn bad_settings_are_refused() {
    let bad = [
        Matting {
            feather: 1.5,
            ..Matting::auto()
        },
        Matting {
            clean_up: -0.1,
            ..Matting::auto()
        },
        Matting {
            stroke: Some(Stroke {
                color: "red".into(),
                ..Stroke::new(StrokeStyle::Solid)
            }),
            ..Matting::auto()
        },
        // Picks belong to Custom.
        Matting {
            mode: MattingMode::Auto,
            ..custom()
        },
        Matting {
            picks: vec![Pick {
                frame: Micros(0),
                clicks: vec![PickClick {
                    x: 1.5,
                    y: 0.0,
                    keep: true,
                }],
                brushes: Vec::new(),
            }],
            ..custom()
        },
        // An empty pick, and picks out of time order.
        Matting {
            picks: vec![Pick {
                frame: Micros(0),
                clicks: Vec::new(),
                brushes: Vec::new(),
            }],
            ..custom()
        },
        Matting {
            picks: vec![custom().picks[0].clone(), custom().picks[0].clone()],
            ..custom()
        },
    ];
    for matting in bad {
        assert!(!matting.is_valid(), "{matting:?}");
        assert_eq!(Matting::from_fields(&matting.to_fields()), None);
    }
}

#[test]
fn hostile_fields_read_as_none() {
    for fields in [
        json!({ "mode": "sideways" }),
        json!({ "mode": "custom", "picks": "many" }),
        json!({ "mode": "auto", "feather": "soft" }),
        json!({ "mode": "custom", "picks": [{ "frame": -5, "clicks": [{ "x": 0.5, "y": 0.5, "keep": true }] }] }),
    ] {
        let Value::Object(fields) = fields else {
            unreachable!()
        };
        assert_eq!(Matting::from_fields(&fields), None, "{fields:?}");
    }
    let too_many = vec![custom().picks[0].clone(); MAX_PICKS + 1];
    let matting = Matting {
        picks: too_many,
        ..custom()
    };
    assert!(!matting.is_valid());
}

#[test]
fn a_custom_matte_is_named_by_its_picks() {
    assert_eq!(Matting::auto().picks_key(), "");
    let one = custom();
    let mut moved = custom();
    moved.picks[0].clicks[0].x = 0.6;
    assert_ne!(one.picks_key(), moved.picks_key());
    assert_eq!(one.picks_key(), custom().picks_key());
    assert!(
        !Matting {
            picks: Vec::new(),
            ..custom()
        }
        .ready()
    );
    assert!(one.ready() && Matting::auto().ready());
}

#[test]
fn a_clip_finds_its_matting_beside_other_masks() {
    let mut draft = Draft::empty("D");
    let mut segment = Segment::new("a", "m", TimeRange::new(0, 1_000_000));
    segment.extra_material_refs = vec!["sel".into(), "mat".into()];
    draft.materials.common_mask.push(RawMaterial {
        id: "sel".into(),
        kind: crate::SELECTION.to_owned(),
        ..RawMaterial::default()
    });
    draft.materials.common_mask.push(RawMaterial {
        id: "mat".into(),
        kind: MATTING.to_owned(),
        fields: custom().to_fields(),
    });
    assert_eq!(segment_matting(&draft, &segment), Some(custom()));
    segment.extra_material_refs.pop();
    assert_eq!(segment_matting(&draft, &segment), None);
}

#[test]
fn cut_scales_every_channel_by_the_matte() {
    let mut rgba = vec![200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255];
    cut(&mut rgba, &[255, 128, 0]);
    assert_eq!(&rgba[0..4], &[200, 100, 50, 255]);
    assert_eq!(&rgba[4..8], &[100, 50, 25, 128]);
    assert_eq!(&rgba[8..12], &[0, 0, 0, 0]);
}

/// A 40 by 40 matte with a kept 20 by 20 square in the middle.
fn square() -> Vec<u8> {
    (0..40 * 40)
        .map(|i| {
            let (x, y) = (i % 40, i / 40);
            if (10..30).contains(&x) && (10..30).contains(&y) {
                255
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn refine_with_nothing_set_leaves_the_matte() {
    let mut alpha = square();
    refine(&mut alpha, 40, 40, 0.0, 0.0);
    assert_eq!(alpha, square());
}

#[test]
fn clean_up_pulls_the_edge_in() {
    // At a 1000 pixel short side, full clean up is 15 pixels.
    let (w, h) = (1000_u32, 1000_u32);
    let mut alpha: Vec<u8> = (0..w * h)
        .map(|i| {
            if (300..700).contains(&(i % w)) {
                255
            } else {
                0
            }
        })
        .collect();
    refine(&mut alpha, w, h, 1.0, 0.0);
    let row = &alpha[500 * w as usize..501 * w as usize];
    assert_eq!(row[314], 0);
    assert_eq!(row[315], 255);
    assert_eq!(row[684], 255);
    assert_eq!(row[685], 0);
}

#[test]
fn feather_softens_the_edge_and_keeps_the_middle() {
    let (w, h) = (400_u32, 400_u32);
    let mut alpha: Vec<u8> = (0..w * h)
        .map(|i| if i % w >= 200 { 255 } else { 0 })
        .collect();
    refine(&mut alpha, w, h, 0.0, 1.0);
    let row = &alpha[200 * w as usize..201 * w as usize];
    assert_eq!(row[0], 0);
    assert_eq!(row[399], 255);
    assert!(row[200] > 100 && row[200] < 200, "{}", row[200]);
    assert!(row[195] > 0 && row[205] < 255);
    // Monotone across the edge.
    assert!(row.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn distances_are_euclidean_to_the_nearest_kept_pixel() {
    let (w, h) = (9, 9);
    let mut alpha = vec![0_u8; w * h];
    alpha[4 * w + 4] = 255;
    let distance = outside_distance(&alpha, w, h);
    assert_eq!(distance[4 * w + 4], 0.0);
    assert!((distance[4 * w + 7] - 3.0).abs() < 1e-9);
    assert!((distance[0] - (32.0_f64).sqrt()).abs() < 1e-9);
    assert!((distance[w + 8] - 5.0).abs() < 1e-9);
}

#[test]
fn distances_with_nothing_kept_stay_far() {
    let distance = outside_distance(&[0; 16], 4, 4);
    assert!(distance.iter().all(|d| *d > 1e9));
}

#[test]
fn every_stroke_draws_outside_the_subject_only() {
    let alpha = square();
    for style in StrokeStyle::ALL {
        let stroke = Stroke {
            size: 1.0,
            ..Stroke::new(style)
        };
        let layer = stroke_layer(&alpha, 40, 40, &stroke);
        assert_eq!(layer.len(), 40 * 40 * 4);
        let drawn = layer.chunks(4).filter(|px| px[3] > 0).count();
        assert!(drawn > 0, "{style:?} drew nothing");
        if style != StrokeStyle::Offset {
            // Nothing under the kept square.
            for y in 10..30 {
                for x in 10..30 {
                    assert_eq!(layer[(y * 40 + x) * 4 + 3], 0, "{style:?} at {x},{y}");
                }
            }
        }
        // Premultiplied: no channel above its alpha.
        assert!(
            layer
                .chunks(4)
                .all(|px| px[0..3].iter().all(|c| *c <= px[3]))
        );
    }
}

#[test]
fn a_solid_stroke_is_as_wide_as_its_size() {
    // Full size is 4% of the short side: 4 pixels of 100.
    let mut alpha = vec![0_u8; 100 * 100];
    for y in 40..60 {
        for x in 40..60 {
            alpha[y * 100 + x] = 255;
        }
    }
    let stroke = Stroke {
        size: 1.0,
        ..Stroke::new(StrokeStyle::Solid)
    };
    let layer = stroke_layer(&alpha, 100, 100, &stroke);
    let at = |x: usize| layer[(50 * 100 + x) * 4 + 3];
    assert_eq!(at(60), 255);
    assert_eq!(at(63), 255);
    assert_eq!(at(65), 0);
    let spaced = Stroke {
        spacing: 1.0,
        ..stroke
    };
    let layer = stroke_layer(&alpha, 100, 100, &spaced);
    // A 3 pixel gap first.
    assert_eq!(layer[(50 * 100 + 61) * 4 + 3], 0);
    assert_eq!(layer[(50 * 100 + 64) * 4 + 3], 255);
}

#[test]
fn a_bad_size_draws_nothing_and_does_not_panic() {
    let stroke = Stroke::new(StrokeStyle::Glow);
    assert!(
        stroke_layer(&[255; 3], 2, 2, &stroke)
            .iter()
            .all(|b| *b == 0)
    );
    let mut alpha = vec![255; 3];
    refine(&mut alpha, 2, 2, 1.0, 1.0);
    assert_eq!(alpha, vec![255; 3]);
}

#[test]
fn a_1080p_refine_and_stroke_stay_linear() {
    // A coarse bound: both passes are linear in the pixels, so a frame
    // takes well under a second even in a debug build.
    let (w, h) = (1920_u32, 1080_u32);
    let mut alpha: Vec<u8> = (0..w * h)
        .map(|i| if (i % w).abs_diff(960) < 300 { 255 } else { 0 })
        .collect();
    let start = std::time::Instant::now();
    refine(&mut alpha, w, h, 1.0, 1.0);
    let _ = stroke_layer(&alpha, w, h, &Stroke::new(StrokeStyle::Paper));
    assert!(start.elapsed().as_secs_f64() < 8.0, "{:?}", start.elapsed());
}
