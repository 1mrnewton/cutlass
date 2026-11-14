use std::path::{Path, PathBuf};

use safetensors::SafeTensors;

use super::*;

const REQUIRE: &str = "CUTLASS_REQUIRE_DETECT_MODELS";

fn local_models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/detect")
}

fn found(x0: f32, y0: f32, x1: f32, y1: f32, score: f32) -> Found {
    Found {
        x0,
        y0,
        x1,
        y1,
        score,
    }
}

#[test]
fn the_anchors_are_mediapipes() {
    let body = ssd_anchors(BODY_SIDE, &[(8, 2), (16, 2), (32, 6)]);
    assert_eq!(body.len(), 2254);
    assert_eq!(body[0], (0.5 / 28.0, 0.5 / 28.0));
    assert_eq!(body[2], (1.5 / 28.0, 0.5 / 28.0));
    assert_eq!(body[2253], (6.5 / 7.0, 6.5 / 7.0));
    let hand = ssd_anchors(HAND_SIDE, &[(8, 2), (16, 6)]);
    assert_eq!(hand.len(), 2016);
    assert_eq!(hand[2015], (11.5 / 12.0, 11.5 / 12.0));
}

#[test]
fn an_overlapping_weaker_box_goes_and_a_separate_one_stays() {
    let kept = suppress(vec![
        found(0.0, 0.0, 10.0, 10.0, 0.7),
        found(1.0, 1.0, 11.0, 11.0, 0.9),
        found(50.0, 50.0, 60.0, 60.0, 0.6),
    ]);
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].score, 0.9);
    assert_eq!(kept[1].x0, 50.0);
}

#[test]
fn a_body_reaches_from_past_the_head_to_as_far_below_the_hips() {
    let person = Person {
        found: found(0.0, 0.0, 1.0, 1.0, 0.8),
        points: [
            (100.0, 300.0),
            (100.0, 100.0),
            (100.0, 200.0),
            (100.0, 50.0),
        ],
    };
    let b = body(&person);
    assert_eq!((b.y0, b.y1), (100.0, 500.0));
    assert_eq!((b.x0, b.x1), (25.0, 175.0));
    assert_eq!(b.score, 0.8);
}

#[test]
fn a_frame_of_the_wrong_size_is_refused() {
    let rgb = vec![0; 10];
    assert!(
        check(&Image {
            width: 2,
            height: 2,
            rgb: &rgb
        })
        .is_err()
    );
}

/// The frames and what onnxruntime and OpenCV made of them, from
/// `tools/vision-models/scripts/fetch_detect.sh`.
fn reference() -> Option<Vec<u8>> {
    let dir = local_models();
    let file = dir.join("reference_detect.safetensors");
    let here = [FACE_WEIGHTS, BODY_WEIGHTS, HAND_WEIGHTS]
        .iter()
        .all(|w| dir.join(w).is_file());
    if !here || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the detector weights or {} are missing",
            file.display()
        );
        return None;
    }
    Some(std::fs::read(&file).unwrap())
}

fn floats(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    tensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("{name}"))
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn relative(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f32::max)
}

fn frames(tensors: &SafeTensors<'_>) -> Vec<(u32, u32, Vec<u8>)> {
    (0..)
        .map_while(|k| tensors.tensor(&format!("{k}.rgb")).ok())
        .map(|rgb| {
            let shape = rgb.shape();
            (shape[1] as u32, shape[0] as u32, rgb.data().to_vec())
        })
        .collect()
}

#[test]
fn the_networks_give_what_onnxruntime_gives() {
    let Some(bytes) = reference() else { return };
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let count = frames(&tensors).len();
    assert!(count >= 4);
    let dir = local_models();
    let face = Detector::load(&dir, Part::Face, Compute::Cpu).unwrap();
    let body = Detector::load(&dir, Part::Body, Compute::Cpu).unwrap();
    let hand = Detector::load(&dir, Part::Hand, Compute::Cpu).unwrap();
    let tensor = |values: Vec<f32>, shape: [usize; 4], device: &Device| {
        Tensor::<4>::from_data(TensorData::new(values, shape), device)
    };
    for k in 0..count {
        let Net::Face(model) = &face.net else {
            panic!()
        };
        let input = floats(&tensors, &format!("{k}.face.input"));
        let out = model.forward(tensor(input, [1, 3, 640, 640], &face.device));
        let got = [
            out.0, out.1, out.2, out.3, out.4, out.5, out.6, out.7, out.8, out.9, out.10, out.11,
        ];
        let names = ["cls", "obj", "bbox", "kps"]
            .iter()
            .flat_map(|head| [8, 16, 32].map(|s| format!("{head}_{s}")));
        for (name, out) in names.zip(got) {
            let expected = floats(&tensors, &format!("{k}.face.{name}"));
            let got = values(out.into_data()).unwrap();
            let error = worst(&got, &expected);
            assert!(error < 2e-3, "frame {k} face {name}: {error}");
        }
        let Net::Body(model) = &body.net else {
            panic!()
        };
        let input = floats(&tensors, &format!("{k}.person.input"));
        let (scores, boxes) = model.forward(tensor(input, [1, 3, 224, 224], &body.device));
        // Scores far below zero run to the hundred thousands; compare them
        // as a share of their size.
        let e = relative(
            &values(scores.into_data()).unwrap(),
            &floats(&tensors, &format!("{k}.person.scores_out")),
        );
        let b = worst(
            &values(boxes.into_data()).unwrap(),
            &floats(&tensors, &format!("{k}.person.boxes_out")),
        );
        assert!(
            e < 2e-2 && b < 2e-2,
            "frame {k} person: scores {e}, boxes {b}"
        );
        let Net::Hand(model) = &hand.net else {
            panic!()
        };
        let input = floats(&tensors, &format!("{k}.palm.input"));
        let (boxes, scores) = model.forward(tensor(input, [1, 192, 192, 3], &hand.device));
        let e = worst(
            &values(scores.into_data()).unwrap(),
            &floats(&tensors, &format!("{k}.palm.scores_out")),
        );
        let b = worst(
            &values(boxes.into_data()).unwrap(),
            &floats(&tensors, &format!("{k}.palm.boxes_out")),
        );
        assert!(
            e < 2e-2 && b < 2e-2,
            "frame {k} palm: scores {e}, boxes {b}"
        );
    }
}

#[test]
fn the_squares_are_made_as_opencv_makes_them() {
    let Some(bytes) = reference() else { return };
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    for (k, (width, height, rgb)) in frames(&tensors).into_iter().enumerate() {
        let image = Image {
            width,
            height,
            rgb: &rgb,
        };
        let fit = Letterbox::new(&image, BODY_SIDE);
        let hwc = fit.pixels(&image, |v| (v / 255.0 - 0.5) * 2.0);
        let expected = floats(&tensors, &format!("{k}.person.input"));
        let plane = BODY_SIDE * BODY_SIDE;
        let mut errors: Vec<f32> = hwc
            .iter()
            .enumerate()
            .map(|(i, v)| (v - expected[(i % 3) * plane + i / 3]).abs())
            .collect();
        errors.sort_by(f32::total_cmp);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        let p99 = errors[errors.len() * 99 / 100];
        eprintln!(
            "frame {k} body square: mean {mean}, 99th {p99}, worst {}",
            errors[errors.len() - 1]
        );
        // Two steps of 255 on the -1 to 1 scale.
        assert!(
            mean < 2e-3 && p99 < 4.0 / 255.0 * 2.0,
            "frame {k}: {mean} {p99}"
        );
        let (input, _) = face_input(&image);
        let expected = floats(&tensors, &format!("{k}.face.input"));
        let mean = input
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / input.len() as f32;
        assert!(mean < 0.5, "frame {k} face square: {mean}");
    }
}

/// Every expected box has a found one over it.
fn covers(found: &[Found], expected: &[Found], least: f32, what: &str) {
    assert_eq!(
        found.len(),
        expected.len(),
        "{what}: {found:?} for {expected:?}"
    );
    for e in expected {
        let best = found.iter().map(|f| f.overlap(e)).fold(0.0, f32::max);
        assert!(
            best > least,
            "{what}: {e:?} best overlap {best} in {found:?}"
        );
    }
}

#[test]
fn faces_and_people_are_found_where_opencv_finds_them() {
    let Some(bytes) = reference() else { return };
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let dir = local_models();
    let face = Detector::load(&dir, Part::Face, Compute::Cpu).unwrap();
    let body = Detector::load(&dir, Part::Body, Compute::Cpu).unwrap();
    let Net::Body(people) = &body.net else {
        panic!()
    };
    for (k, (width, height, rgb)) in frames(&tensors).into_iter().enumerate() {
        let image = Image {
            width,
            height,
            rgb: &rgb,
        };
        let rows = |name: &str, n: usize| -> Vec<Vec<f32>> {
            floats(&tensors, &format!("{k}.{name}"))
                .chunks(n)
                .map(<[f32]>::to_vec)
                .collect()
        };
        let expected: Vec<Found> = rows("face.boxes", 5)
            .iter()
            .map(|r| found(r[0], r[1], r[2], r[3], r[4]))
            .collect();
        covers(
            &face.find(&image).unwrap(),
            &expected,
            0.85,
            &format!("frame {k} faces"),
        );
        let expected = rows("person.found", 13);
        let got = body.people(people, &image).unwrap();
        covers(
            &got.iter().map(|p| p.found).collect::<Vec<_>>(),
            &expected
                .iter()
                .map(|r| found(r[0], r[1], r[2], r[3], r[12]))
                .collect::<Vec<_>>(),
            0.8,
            &format!("frame {k} people"),
        );
        #[allow(clippy::cast_precision_loss)]
        let near = width.max(height) as f32 * 0.03;
        for e in &expected {
            let person = got
                .iter()
                .max_by(|a, b| {
                    a.found
                        .overlap(&found(e[0], e[1], e[2], e[3], 0.0))
                        .total_cmp(&b.found.overlap(&found(e[0], e[1], e[2], e[3], 0.0)))
                })
                .unwrap();
            for (i, (x, y)) in person.points.iter().enumerate() {
                let (ex, ey) = (e[4 + 2 * i], e[5 + 2 * i]);
                assert!(
                    (x - ex).abs() < near && (y - ey).abs() < near,
                    "frame {k} point {i}: ({x}, {y}) for ({ex}, {ey})"
                );
            }
        }
    }
}

#[test]
fn the_walking_man_is_boxed_head_to_foot_and_the_dancers_hands_are_found() {
    let Some(bytes) = reference() else { return };
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let frames = frames(&tensors);
    let dir = local_models();
    let body = Detector::load(&dir, Part::Body, Compute::Cpu).unwrap();
    // The fourth frame: the man walking away down the pier, 1920 by 1080,
    // his shirt about (997, 517).
    let (width, height, rgb) = &frames[3];
    let image = Image {
        width: *width,
        height: *height,
        rgb,
    };
    let bodies = body.find(&image).unwrap();
    let man = bodies.first().expect("the man");
    assert!(
        man.x0 < 997.0 && 997.0 < man.x1 && man.y0 < 450.0 && man.y1 > 900.0,
        "{man:?}"
    );
    // The second frame: the dancer holding her headphones, 1280 by 720.
    // Squares of the frame find her palms near (632, 277) and (496, 229).
    let hand = Detector::load(&dir, Part::Hand, Compute::Cpu).unwrap();
    let (width, height, rgb) = &frames[1];
    let hands = hand
        .find(&Image {
            width: *width,
            height: *height,
            rgb,
        })
        .unwrap();
    for (x, y) in [(632.0, 277.0), (496.0, 229.0)] {
        assert!(
            hands.iter().any(|h| {
                ((h.x0 + h.x1) / 2.0 - x).abs() < 15.0 && ((h.y0 + h.y1) / 2.0 - y).abs() < 15.0
            }),
            "a hand near ({x}, {y}) in {hands:?}"
        );
    }
    // No palm on the face of the woman in the first frame.
    let (width, height, rgb) = &frames[0];
    let none = hand
        .find(&Image {
            width: *width,
            height: *height,
            rgb,
        })
        .unwrap();
    assert!(none.is_empty(), "{none:?}");
}
