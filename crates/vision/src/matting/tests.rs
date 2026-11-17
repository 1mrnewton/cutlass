use std::path::PathBuf;
use std::time::Instant;

use super::*;
use crate::MATTE_MAX_SIDE;
use crate::matte_frame::real;

const REQUIRE: &str = "CUTLASS_REQUIRE_MATTING_MODEL";

fn local_models() -> PathBuf {
    real::models("matting")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-matting-{name}-{}", std::process::id()));
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

/// The frame drawn at `side` and what PyTorch made of it, from
/// `tools/vision-models/scripts/fetch_matting.sh`.
fn reference(side: u32) -> Option<(Vec<u8>, Vec<f32>)> {
    let dir = local_models();
    let file = dir.join("reference_frame.safetensors");
    if !dir.join(MATTING_WEIGHTS).is_file() || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the matting weights or {} are missing",
            file.display()
        );
        return None;
    }
    let bytes = std::fs::read(&file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let side = side as usize;
    let rgb = tensors.tensor(&format!("rgb_{side}")).unwrap();
    assert_eq!(rgb.shape(), [side, side, 3]);
    let matte = tensors.tensor(&format!("matte_{side}")).unwrap();
    assert_eq!(matte.shape(), [side, side]);
    let matte = matte
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Some((rgb.data().to_vec(), matte))
}

/// The worst and the mean difference from PyTorch's share kept, and the
/// share of pixels on the other side of one half.
fn against_pytorch(compute: Compute, side: u32) -> Option<(f32, f32, f32)> {
    let (rgb, expected) = reference(side)?;
    let model = Matting::load_at(&local_models(), compute, side).unwrap();
    let got = matte_frame::values(model.kept(&image(side, side, &rgb)).unwrap()).unwrap();
    assert_eq!(got.len(), expected.len());
    let kept = expected.iter().filter(|v| **v > 0.5).count();
    assert!(
        kept > expected.len() / 10,
        "PyTorch kept too little: {kept}"
    );
    let errors: Vec<f32> = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let worst = errors.iter().copied().fold(0.0, f32::max);
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    let flipped = got
        .iter()
        .zip(&expected)
        .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
        .count() as f32
        / got.len() as f32;
    real::say(&format!(
        "birefnet {side} on {compute:?}: worst {worst}, mean {mean}, flipped {flipped}"
    ));
    Some((worst, mean, flipped))
}

#[test]
fn the_cpu_matches_pytorch() {
    if let Some((worst, mean, _)) = against_pytorch(Compute::Cpu, 512) {
        assert!(worst < 1e-3 && mean < 1e-5, "worst {worst}, mean {mean}");
    }
}

#[test]
#[ignore = "needs a Metal GPU"]
fn the_gpu_matches_pytorch() {
    for side in [512, 1024] {
        if let Some((worst, mean, _)) = against_pytorch(Compute::Gpu, side) {
            assert!(
                worst < 1e-3 && mean < 1e-5,
                "{side}: worst {worst}, mean {mean}"
            );
        }
    }
}

#[test]
fn the_window_mask_keeps_shifted_neighbours_apart() {
    // A 14 by 14 grid is four windows; only the last row and column of
    // windows hold tokens that wrapped round in the shift.
    let mask = shift_mask(14, 14);
    let area = WINDOW * WINDOW;
    assert_eq!(mask.len(), 4 * area * area);
    assert!(mask[..area * area].iter().all(|v| *v == 0.0));
    let last = &mask[3 * area * area..];
    // Token 0 (top left) and token 48 (bottom right) of the last window
    // come from different corners of the frame.
    assert_eq!(last[48], MASKED);
    assert_eq!(last[0], 0.0);
    let index = relative_index();
    assert_eq!(index.len(), area * area);
    assert_eq!(index[0], 84);
    assert_eq!(index[48], 0);
    assert_eq!(index[48 * area], 168);
}

#[test]
fn the_frame_is_cut_into_tiles_channel_then_row_then_column() {
    let device = Device::flex();
    let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Tensor::<4>::from_data(TensorData::new(values, [1, 1, 4, 4]), &device);
    let cut = tiles(&x, 2);
    assert_eq!(cut.dims(), [1, 4, 2, 2]);
    let got = matte_frame::values(cut).unwrap();
    assert_eq!(
        got,
        [
            0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0, 15.0
        ]
    );
}

#[test]
fn a_side_off_the_steps_is_refused() {
    for side in [0, 255, 300, 4096] {
        assert!(Matting::untrained(Compute::Cpu, side).is_err(), "{side}");
    }
}

#[test]
fn a_matte_comes_back_at_the_frame_size() {
    let mut model = Matting::untrained(Compute::Cpu, 256).unwrap();
    assert_eq!(model.id(), "birefnet-lite-256");
    assert_eq!(model.side(), 256);
    let rgb = frame(90, 50);
    let matte = model.matte(&image(90, 50, &rgb)).unwrap();
    assert_eq!((matte.width, matte.height), (90, 50));
    assert_eq!(matte.alpha.len(), 90 * 50);
}

#[test]
fn an_empty_wrong_sized_or_huge_frame_is_refused() {
    let mut model = Matting::untrained(Compute::Cpu, 256).unwrap();
    for (w, h, bytes) in [
        (0, 4, 0),
        (4, 4, 47),
        (MATTE_MAX_SIDE + 1, 1, 3 * (MATTE_MAX_SIDE as usize + 1)),
    ] {
        let rgb = vec![0_u8; bytes];
        assert!(
            matches!(
                model.matte(&image(w, h, &rgb)),
                Err(VisionError::Image { .. })
            ),
            "{w}x{h} with {bytes}"
        );
    }
}

#[test]
fn saved_weights_load_back_and_damaged_or_missing_ones_are_refused() {
    let dir = scratch("round-trip");
    let mut saved = Matting::untrained(Compute::Cpu, 256).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(MATTING_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let mut loaded = Matting::load_at(&dir, Compute::Cpu, 256).unwrap();
    let rgb = frame(64, 48);
    assert_eq!(
        saved.matte(&image(64, 48, &rgb)).unwrap(),
        loaded.matte(&image(64, 48, &rgb)).unwrap()
    );
    std::fs::write(dir.join(MATTING_WEIGHTS), b"not weights").unwrap();
    assert!(matches!(
        Matting::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    std::fs::remove_file(dir.join(MATTING_WEIGHTS)).unwrap();
    assert!(matches!(
        Matting::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    assert!(convert_birefnet(&dir.join("none.safetensors"), &dir.join("out.bpk")).is_err());
    std::fs::write(dir.join("bad.safetensors"), b"nope").unwrap();
    assert!(convert_birefnet(&dir.join("bad.safetensors"), &dir.join("out.bpk")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

fn time_real_frames(compute: Compute, sides: &[u32]) {
    for side in sides {
        let Ok(mut model) = Matting::load_at(&local_models(), compute, *side) else {
            return;
        };
        for frame in real::frames() {
            let image = image(frame.width, frame.height, &frame.rgb);
            let matte = model.matte(&image).unwrap();
            let runs = 3;
            let started = Instant::now();
            for _ in 0..runs {
                model.matte(&image).unwrap();
            }
            let each = started.elapsed() / runs;
            real::say(&format!(
                "birefnet {} {}x{} read at {side} on {compute:?}: {each:?} a frame",
                frame.name, frame.width, frame.height
            ));
            real::save(&frame.name, model.id(), &matte);
        }
    }
}

#[test]
#[ignore = "timing on real frames; run with --release --ignored --nocapture"]
fn matting_real_frames_on_the_gpu() {
    time_real_frames(Compute::Gpu, &[1024, 768]);
}

#[test]
#[ignore = "timing on real frames; run with --release --ignored --nocapture"]
fn matting_real_frames_on_the_cpu() {
    time_real_frames(Compute::Cpu, &[1024, 768]);
}
