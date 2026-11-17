use std::path::PathBuf;
use std::time::Instant;

use safetensors::SafeTensors;

use super::*;
use crate::matte_frame::real;
use crate::{Click, MATTE_MAX_SIDE};

const REQUIRE: &str = "CUTLASS_REQUIRE_FOLLOW_MODEL";

fn local_models() -> PathBuf {
    real::models("follow")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-follow-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn image(width: u32, height: u32, rgb: &[u8]) -> Image<'_> {
    Image { width, height, rgb }
}

/// A frame with something in it: a slope of light and a square.
fn frame(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let inside = x > width / 3 && x < width * 2 / 3 && y > height / 3;
            let shade = (x * 255 / width.max(1)) as u8;
            rgb.extend_from_slice(&if inside {
                [200, 60, 40]
            } else {
                [shade, 120, 255 - shade]
            });
        }
    }
    rgb
}

fn click(x: f32, y: f32, keep: bool) -> Click {
    Click { x, y, keep }
}

fn clicked(clicks: Vec<Click>) -> Prompt {
    Prompt { clicks, mask: None }
}

/// The frames, prompts and answers of `tools/vision-models/scripts/fetch_follow.sh`.
struct Reference {
    bytes: Vec<u8>,
}

impl Reference {
    fn load() -> Option<Self> {
        let dir = local_models();
        let file = dir.join("reference_frames.safetensors");
        if !dir.join(FOLLOW_WEIGHTS).is_file() || !file.is_file() {
            assert!(
                std::env::var_os(REQUIRE).is_none(),
                "{REQUIRE} is set and the follow weights or {} are missing",
                file.display()
            );
            return None;
        }
        Some(Self {
            bytes: std::fs::read(&file).unwrap(),
        })
    }

    fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.bytes).unwrap()
    }

    fn bytes(&self, name: &str) -> Vec<u8> {
        self.tensors().tensor(name).unwrap().data().to_vec()
    }

    fn floats(&self, name: &str) -> Vec<f32> {
        self.bytes(name)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn clicks(&self, run: &str) -> Prompt {
        let clicks = self
            .floats(&format!("clicks_{run}"))
            .as_chunks::<3>()
            .0
            .iter()
            .map(|[x, y, keep]| click(*x, *y, *keep > 0.5))
            .collect();
        clicked(clicks)
    }
}

/// How far a run's logits are from transformers', frame by frame: the
/// worst and the mean difference, the share of pixels on the other side of
/// 0, and the difference in the object score.
fn against_transformers(
    compute: Compute,
    reference: &Reference,
    run: &str,
    prompt: &Prompt,
    frames: usize,
) -> Vec<(f32, f32, f32, f32)> {
    let mut model = FollowModel::load(&local_models(), compute).unwrap();
    let side = FOLLOW_SIDE;
    (0..frames)
        .map(|i| {
            let rgb = reference.bytes(&format!("rgb_{i}"));
            let step = model
                .advance(&image(side, side, &rgb), (i == 0).then_some(prompt))
                .unwrap();
            let got = matte_frame::values(step.logits).unwrap();
            let want = reference.floats(&format!("{run}_logits_{i}"));
            assert_eq!(got.len(), want.len());
            let kept = want.iter().filter(|v| **v > 0.0).count();
            assert!(kept > want.len() / 100, "transformers kept {kept}");
            let errors: Vec<f32> = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).collect();
            let worst = errors.iter().copied().fold(0.0, f32::max);
            let mean = errors.iter().sum::<f32>() / errors.len() as f32;
            let flipped = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| (**a > 0.0) != (**b > 0.0))
                .count() as f32
                / got.len() as f32;
            let score = reference.floats(&format!("{run}_score_{i}"));
            let object = (step.object - score.first().copied().unwrap()).abs();
            real::say(&format!(
                "sam2 {run} frame {i} on {compute:?}: worst {worst}, mean {mean}, \
                 flipped {flipped}, object score off by {object}"
            ));
            (worst, mean, flipped, object)
        })
        .collect()
}

/// Checks the runs named in `runs` against transformers.
fn check_runs(compute: Compute, runs: &[&str]) {
    let Some(reference) = Reference::load() else {
        return;
    };
    let painted = Prompt {
        clicks: Vec::new(),
        mask: Matte::new(FOLLOW_SIDE, FOLLOW_SIDE, reference.bytes("mask")),
    };
    let wanted = runs;
    let runs = [
        ("a", reference.clicks("a"), 3),
        ("b", painted, 2),
        ("c", reference.clicks("c"), 1),
    ];
    for (run, prompt, frames) in runs.into_iter().filter(|(run, ..)| wanted.contains(run)) {
        for (i, (worst, mean, flipped, object)) in
            against_transformers(compute, &reference, run, &prompt, frames)
                .into_iter()
                .enumerate()
        {
            assert!(
                worst < 0.02 && mean < 1e-3 && flipped < 1e-4 && object < 1e-3,
                "{run} frame {i}: worst {worst}, mean {mean}, flipped {flipped}, object {object}"
            );
        }
    }
}

/// The CPU checks the clicked run alone, a prompted frame and two
/// followed, which reads every part of the model: each frame takes a while
/// on the CPU unoptimised.
#[test]
fn the_cpu_follows_as_transformers_does() {
    check_runs(Compute::Cpu, &["a"]);
}

#[test]
#[ignore = "needs a GPU"]
fn the_gpu_follows_as_transformers_does() {
    // Trusted as the editor trusts a GPU that passed the check.
    crate::trust_gpu(crate::check_gpu().ok());
    check_runs(Compute::Gpu, &["a", "b", "c"]);
}

#[test]
fn the_image_features_match_transformers() {
    let Some(reference) = Reference::load() else {
        return;
    };
    let model = FollowModel::load(&local_models(), Compute::Cpu).unwrap();
    let rgb = reference.bytes("rgb_0");
    let side = FOLLOW_SIDE;
    let frame = matte_frame::input(
        &image(side, side, &rgb),
        (side, side),
        MEAN,
        SPREAD,
        &model.device,
    )
    .unwrap();
    let features = model.model.features(frame, &model.fixed).unwrap();
    for (name, level) in [
        ("s0", features.s0),
        ("s1", features.s1),
        ("top", features.top),
    ] {
        let [_, c, h, w] = level.dims();
        let got = matte_frame::values(level.reshape([c, h * w]).swap_dims(0, 1)).unwrap();
        let want = reference.floats(&format!("{name}_0"));
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        real::say(&format!("sam2 features {name}: worst {worst}"));
        assert!(worst < 1e-2, "{name}: worst {worst}");
    }
}

#[test]
fn bfloat16_rounds_to_the_nearest_ties_to_even() {
    assert_eq!(bfloat16(1.0), 1.0);
    // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties go to the even 1.
    assert_eq!(bfloat16(1.0 + 1.0 / 256.0), 1.0);
    assert_eq!(bfloat16(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
    assert_eq!(bfloat16(-2.5), -2.5);
    assert!(bfloat16(f32::NAN).is_nan());
}

#[test]
fn resampling_shrinks_by_averaging_and_keeps_a_flat_plane() {
    let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let small = resample(&values, (4, 4), (1, 1));
    assert!((small[0] - 7.5).abs() < 1e-5);
    let flat = resample(&[3.0; 6 * 4], (6, 4), (13, 9));
    assert!(flat.iter().all(|v| (v - 3.0).abs() < 1e-5));
    let same = resample(&values, (4, 4), (4, 4));
    assert!(same.iter().zip(&values).all(|(a, b)| (a - b).abs() < 1e-5));
}

#[test]
fn bicubic_keeps_a_flat_plane_and_its_samples() {
    let flat = bicubic(&[2.0; 2 * 3 * 3], 2, 3, 8);
    assert_eq!(flat.len(), 8 * 8 * 2);
    assert!(flat.iter().all(|v| (v - 2.0).abs() < 1e-5));
    let values: Vec<f32> = (0..9).map(|v| v as f32).collect();
    let same = bicubic(&values, 1, 3, 3);
    assert!(same.iter().zip(&values).all(|(a, b)| (a - b).abs() < 1e-5));
}

#[test]
fn the_blocks_fall_into_their_stages() {
    assert_eq!(stage_ends(), [0, 2, 9, 11]);
    assert_eq!(stage_of(0), (0, true));
    assert_eq!(stage_of(1), (1, true));
    assert_eq!(stage_of(2), (1, false));
    assert_eq!(stage_of(3), (2, true));
    assert_eq!(stage_of(10), (3, true));
    assert_eq!(stage_of(11), (3, false));
}

#[test]
fn windows_come_apart_and_back_together() {
    let device = Device::flex();
    let values: Vec<f32> = (0..2 * 5 * 3).map(|v| v as f32).collect();
    let x = Tensor::<4>::from_data(TensorData::new(values.clone(), [1, 5, 3, 2]), &device);
    let (windows, padded) = partition(x, 2);
    assert_eq!(padded, (6, 4));
    assert_eq!(windows.dims(), [6, 2, 2, 2]);
    let back = unpartition(windows, 2, padded, (5, 3));
    assert_eq!(matte_frame::values(back).unwrap(), values);
}

#[test]
fn the_position_tables_have_their_shapes_and_first_values() {
    let sines = sine_grid(4, 2);
    assert_eq!(sines.len(), 2 * 2 * 8);
    // The first token: y and x both one step of two, scaled to 2 pi; each
    // frequency serves a sine and a cosine.
    let at = 1.0 / 2.0 * 2.0 * std::f32::consts::PI;
    assert!((sines[0] - at.sin()).abs() < 1e-5);
    assert!((sines[1] - at.cos()).abs() < 1e-5);
    assert!((sines[2] - (at / 100.0).sin()).abs() < 1e-5);
    assert!((sines[4] - at.sin()).abs() < 1e-5);
    let pointers = pointer_sines(&[0, 15]);
    assert_eq!(pointers.len(), 2 * HIDDEN);
    assert_eq!(pointers[0], 0.0);
    assert_eq!(pointers[HIDDEN / 2], 1.0);
    assert!((pointers[HIDDEN] - 1.0_f32.sin()).abs() < 1e-6);
    let (cos, sin) = rotary(64);
    assert_eq!((cos.len(), sin.len()), (64 * 64 * HIDDEN, 64 * 64 * HIDDEN));
    // The second token is one column along: its first channels turn by 1.
    assert!((cos[HIDDEN] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((sin[HIDDEN + 1] - 1.0_f32.sin()).abs() < 1e-6);
    assert_eq!(sin[HIDDEN + HIDDEN / 2], 0.0);
}

#[test]
fn a_run_needs_a_prompt_first_and_again_after_a_reset() {
    let mut model = FollowModel::untrained(Compute::Cpu, 256).unwrap();
    assert_eq!(model.id(), FOLLOW_MODEL);
    let rgb = frame(48, 32);
    let frame = image(48, 32, &rgb);
    assert!(matches!(
        model.step(&frame, None),
        Err(VisionError::Clicks { .. })
    ));
    let prompt = clicked(vec![click(24.0, 20.0, true)]);
    let matte = model.step(&frame, Some(&prompt)).unwrap();
    assert_eq!((matte.width, matte.height), (48, 32));
    assert_eq!(matte.alpha.len(), 48 * 32);
    for _ in 0..2 {
        let matte = model.step(&frame, None).unwrap();
        assert_eq!((matte.width, matte.height), (48, 32));
    }
    model.reset();
    assert!(matches!(
        model.step(&frame, None),
        Err(VisionError::Clicks { .. })
    ));
}

#[test]
fn a_painted_mask_alone_or_under_clicks_starts_a_run() {
    let mut model = FollowModel::untrained(Compute::Cpu, 256).unwrap();
    let rgb = frame(40, 30);
    let frame = image(40, 30, &rgb);
    let alpha: Vec<u8> = (0..40 * 30)
        .map(|i| if i % 40 > 20 { 255 } else { 0 })
        .collect();
    let painted = Prompt {
        clicks: Vec::new(),
        mask: Matte::new(40, 30, alpha.clone()),
    };
    // A painted mask is taken as the frame's answer.
    let matte = model.step(&frame, Some(&painted)).unwrap();
    assert!(
        matte.alpha[0] < 5 && matte.alpha[39] > 250,
        "{:?}",
        &matte.alpha[..40]
    );
    let both = Prompt {
        clicks: vec![click(30.0, 15.0, true), click(5.0, 5.0, false)],
        mask: Matte::new(40, 30, alpha),
    };
    assert!(model.step(&frame, Some(&both)).is_ok());
    assert!(model.step(&frame, None).is_ok());
}

#[test]
fn a_bad_frame_or_prompt_is_refused() {
    let mut model = FollowModel::untrained(Compute::Cpu, 256).unwrap();
    let prompt = clicked(vec![click(1.0, 1.0, true)]);
    for (w, h, bytes) in [
        (0, 4, 0),
        (4, 4, 47),
        (MATTE_MAX_SIDE + 1, 1, 3 * (MATTE_MAX_SIDE as usize + 1)),
    ] {
        let rgb = vec![0_u8; bytes];
        assert!(
            matches!(
                model.step(&image(w, h, &rgb), Some(&prompt)),
                Err(VisionError::Image { .. })
            ),
            "{w}x{h} with {bytes}"
        );
    }
    let rgb = frame(16, 16);
    let frame = image(16, 16, &rgb);
    for bad in [
        Prompt::default(),
        clicked(vec![click(17.0, 1.0, true)]),
        Prompt {
            clicks: Vec::new(),
            mask: Matte::new(8, 8, vec![255; 64]),
        },
    ] {
        assert!(
            matches!(
                model.step(&frame, Some(&bad)),
                Err(VisionError::Clicks { .. })
            ),
            "{bad:?}"
        );
    }
    // Nothing refused was remembered.
    assert!(model.step(&frame, None).is_err());
}

#[test]
fn saved_weights_load_back_and_damaged_or_missing_ones_are_refused() {
    let dir = scratch("round-trip");
    let mut saved = FollowModel::untrained(Compute::Cpu, 256).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(FOLLOW_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let device = Device::flex();
    let mut model = Sam2::new(&device);
    load_weights(&mut model, &dir.join(FOLLOW_WEIGHTS)).unwrap();
    let mut loaded = FollowModel::on(device, model, 256).unwrap();
    let rgb = frame(32, 24);
    let prompt = clicked(vec![click(16.0, 16.0, true)]);
    assert_eq!(
        saved.step(&image(32, 24, &rgb), Some(&prompt)).unwrap(),
        loaded.step(&image(32, 24, &rgb), Some(&prompt)).unwrap()
    );
    std::fs::write(dir.join(FOLLOW_WEIGHTS), b"not weights").unwrap();
    assert!(matches!(
        FollowModel::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    std::fs::remove_file(dir.join(FOLLOW_WEIGHTS)).unwrap();
    assert!(matches!(
        FollowModel::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    assert!(convert_sam2(&dir.join("none.safetensors"), &dir.join("out.bpk")).is_err());
    std::fs::write(dir.join("bad.safetensors"), b"nope").unwrap();
    assert!(convert_sam2(&dir.join("bad.safetensors"), &dir.join("out.bpk")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Where to click on each real frame: a share of its width and height.
fn real_click(name: &str) -> (f32, f32) {
    match name {
        "dock" => (0.515, 0.55),
        "party" => (0.3, 0.5),
        _ => (0.47, 0.7),
    }
}

/// Times a prompted frame, then frames followed until the memory is full,
/// each real frame standing in for a still clip.
fn time_real_frames(compute: Compute) {
    let Ok(mut model) = FollowModel::load(&local_models(), compute) else {
        return;
    };
    for frame in real::frames() {
        let image = image(frame.width, frame.height, &frame.rgb);
        let (fx, fy) = real_click(&frame.name);
        let prompt = clicked(vec![click(
            fx * frame.width as f32,
            fy * frame.height as f32,
            true,
        )]);
        // One run to warm up.
        model.reset();
        model.step(&image, Some(&prompt)).unwrap();
        model.step(&image, None).unwrap();
        model.reset();
        let started = Instant::now();
        let first = model.step(&image, Some(&prompt)).unwrap();
        let prompted = started.elapsed();
        let mut followed = Vec::new();
        let mut last = first.clone();
        for _ in 0..POINTERS {
            let started = Instant::now();
            last = model.step(&image, None).unwrap();
            followed.push(started.elapsed());
        }
        let full = followed[MEMORIES..].iter().sum::<std::time::Duration>()
            / (followed.len() - MEMORIES) as u32;
        real::say(&format!(
            "sam2 {} {}x{} on {compute:?}: prompted {prompted:?}, first followed {:?}, \
             followed with a full memory {full:?}; kept {:.3} then {:.3}",
            frame.name,
            frame.width,
            frame.height,
            followed[0],
            first.coverage(),
            last.coverage()
        ));
        real::save(&frame.name, &format!("{FOLLOW_MODEL}-prompted"), &first);
        real::save(&frame.name, &format!("{FOLLOW_MODEL}-followed"), &last);
    }
}

#[test]
#[ignore = "timing on real frames; run with --release --ignored --nocapture"]
fn follow_real_frames_on_the_gpu() {
    time_real_frames(Compute::Gpu);
}

#[test]
#[ignore = "timing on real frames; run with --release --ignored --nocapture"]
fn follow_real_frames_on_the_cpu() {
    time_real_frames(Compute::Cpu);
}

/// Follows a clip: the frames in `CUTLASS_MATTE_FRAMES` in name order, the
/// first clicked where `CUTLASS_FOLLOW_CLICK` says (`x,y` shares of the
/// frame, several split by `;`, the middle if unset), each matte saved
/// beside its frame.
#[test]
#[ignore = "a real clip; run with --release --ignored --nocapture"]
fn follow_a_real_clip_on_the_gpu() {
    let Ok(mut model) = FollowModel::load(&local_models(), Compute::Gpu) else {
        return;
    };
    let spots: Vec<(f32, f32)> = std::env::var("CUTLASS_FOLLOW_CLICK")
        .unwrap_or_else(|_| "0.5,0.5".into())
        .split(';')
        .filter_map(|spot| {
            let (x, y) = spot.split_once(',')?;
            Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
        })
        .collect();
    for (i, frame) in real::frames().iter().enumerate() {
        let image = image(frame.width, frame.height, &frame.rgb);
        let (w, h) = (frame.width as f32, frame.height as f32);
        let prompt = clicked(
            spots
                .iter()
                .map(|(fx, fy)| click(fx * w, fy * h, true))
                .collect(),
        );
        let started = Instant::now();
        let matte = model.step(&image, (i == 0).then_some(&prompt)).unwrap();
        real::say(&format!(
            "sam2 clip {} kept {:.3} in {:?}",
            frame.name,
            matte.coverage(),
            started.elapsed()
        ));
        real::save(&frame.name, FOLLOW_MODEL, &matte);
    }
}
