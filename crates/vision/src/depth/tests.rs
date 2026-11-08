use std::path::{Path, PathBuf};
use std::time::Instant;

use super::*;

const REQUIRE: &str = "CUTLASS_REQUIRE_DEPTH_MODEL";

fn local_models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/relight")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-depth-{name}-{}", std::process::id()));
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

/// The frame and what transformers made of it, from
/// `tools/vision-models/scripts/fetch_depth.sh`.
fn reference() -> Option<(Vec<u8>, Vec<f32>)> {
    let dir = local_models();
    let file = dir.join("reference_frame.safetensors");
    if !dir.join(DEPTH_WEIGHTS).is_file() || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the depth weights or {} are missing",
            file.display()
        );
        return None;
    }
    let bytes = std::fs::read(&file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let rgb = tensors.tensor("rgb").unwrap();
    assert_eq!(rgb.shape(), [518, 686, 3]);
    let depth = tensors.tensor("depth").unwrap();
    assert_eq!(depth.shape(), [518, 686]);
    let depth = depth
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Some((rgb.data().to_vec(), depth))
}

/// The worst difference from transformers, as a share of its depth's
/// spread, and the mean.
fn against_transformers(compute: Compute) -> Option<(f32, f32)> {
    let (rgb, expected) = reference()?;
    let model = DepthModel::load(&local_models(), compute).unwrap();
    let got = model.estimate(&image(686, 518, &rgb)).unwrap();
    assert_eq!((got.width, got.height), (686, 518));
    let spread = expected.iter().copied().fold(f32::MIN, f32::max)
        - expected.iter().copied().fold(f32::MAX, f32::min);
    assert!(spread > 1.0, "transformers saw no depth: {spread}");
    let errors: Vec<f32> = got
        .values
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs() / spread)
        .collect();
    let worst = errors.iter().copied().fold(0.0, f32::max);
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    Some((worst, mean))
}

#[test]
fn the_cpu_matches_transformers() {
    if let Some((worst, mean)) = against_transformers(Compute::Cpu) {
        assert!(worst < 2e-3 && mean < 2e-4, "worst {worst}, mean {mean}");
    }
}

#[test]
#[ignore = "needs a Metal GPU"]
fn the_gpu_matches_transformers() {
    if let Some((worst, mean)) = against_transformers(Compute::Gpu) {
        assert!(worst < 5e-3 && mean < 5e-4, "worst {worst}, mean {mean}");
    }
}

#[test]
fn a_frame_is_read_at_whole_patches_with_its_short_side_at_518() {
    assert_eq!(DepthModel::grid(518, 518), (37, 37));
    assert_eq!(DepthModel::grid(1920, 1080), (66, 37));
    assert_eq!(DepthModel::grid(1080, 1920), (37, 66));
    assert_eq!(DepthModel::grid(100, 50), (74, 37));
    assert_eq!(DepthModel::grid(1, 1), (37, 37));
    assert_eq!(DepthModel::grid_at(1920, 1080, 364), (46, 26));
    assert_eq!(DepthModel::grid_at(1920, 1080, 0), (2, 1));
    let model = DepthModel::untrained(Compute::Cpu).unwrap();
    let rgb = frame(64, 36);
    let map = model.estimate(&image(64, 36, &rgb)).unwrap();
    let (pw, ph) = DepthModel::grid(64, 36);
    assert_eq!((map.width, map.height), (pw * 14, ph * 14));
    assert_eq!(map.values.len(), (map.width * map.height) as usize);
    assert!(map.values.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn resampling_the_learned_grid_to_its_own_size_keeps_it() {
    let side = 3;
    let from: Vec<f32> = (0..side * side * WIDTH).map(|i| (i % 97) as f32).collect();
    let same = bicubic(&from, side, side, side);
    assert!(from.iter().zip(&same).all(|(a, b)| (a - b).abs() < 1e-4));
    // A flat grid stays flat at any size, its weights summing to one.
    let flat = vec![2.5_f32; side * side * WIDTH];
    assert!(
        bicubic(&flat, side, 7, 5)
            .iter()
            .all(|v| (v - 2.5).abs() < 1e-5)
    );
}

#[test]
fn an_empty_wrong_sized_or_huge_frame_is_refused() {
    let model = DepthModel::untrained(Compute::Cpu).unwrap();
    for (w, h, bytes) in [
        (0, 4, 0),
        (4, 4, 47),
        (DEPTH_MAX_SIDE + 1, 1, 3 * (DEPTH_MAX_SIDE as usize + 1)),
    ] {
        let rgb = vec![0_u8; bytes];
        assert!(
            matches!(
                model.estimate(&image(w, h, &rgb)),
                Err(VisionError::Image { .. })
            ),
            "{w}x{h} with {bytes}"
        );
    }
}

#[test]
fn saved_weights_load_back_and_damaged_or_missing_ones_are_refused() {
    let dir = scratch("round-trip");
    let saved = DepthModel::untrained(Compute::Cpu).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(DEPTH_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let loaded = DepthModel::load(&dir, Compute::Cpu).unwrap();
    let rgb = frame(28, 28);
    assert_eq!(
        saved.estimate(&image(28, 28, &rgb)).unwrap(),
        loaded.estimate(&image(28, 28, &rgb)).unwrap()
    );
    std::fs::write(dir.join(DEPTH_WEIGHTS), b"not weights").unwrap();
    assert!(matches!(
        DepthModel::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    std::fs::remove_file(dir.join(DEPTH_WEIGHTS)).unwrap();
    assert!(matches!(
        DepthModel::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    assert!(convert_depth_anything(&dir.join("none.safetensors"), &dir.join("out.bpk")).is_err());
    std::fs::write(dir.join("bad.safetensors"), b"nope").unwrap();
    assert!(convert_depth_anything(&dir.join("bad.safetensors"), &dir.join("out.bpk")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

fn time_1080p(compute: Compute, side: u32) {
    let model = DepthModel::untrained(compute).unwrap();
    let rgb = frame(1920, 1080);
    let started = Instant::now();
    model.estimate_at(&image(1920, 1080, &rgb), side).unwrap();
    let first = started.elapsed();
    let started = Instant::now();
    let runs = 5;
    for _ in 0..runs {
        model.estimate_at(&image(1920, 1080, &rgb), side).unwrap();
    }
    let each = started.elapsed() / runs;
    let _ = std::io::Write::write_all(
        &mut std::io::stderr(),
        format!(
            "depth 1080p read at {side} on {compute:?}: first {first:?}, then {each:?} a frame\n"
        )
        .as_bytes(),
    );
}

#[test]
#[ignore = "timing; run with --release --ignored --nocapture"]
fn depth_1080p_on_the_gpu() {
    for side in [DEPTH_SIDE, 364, 280] {
        time_1080p(Compute::Gpu, side);
    }
}

#[test]
#[ignore = "timing; run with --release --ignored --nocapture"]
fn depth_1080p_on_the_cpu() {
    time_1080p(Compute::Cpu, 364);
}
