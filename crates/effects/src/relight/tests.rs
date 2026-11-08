use std::collections::HashSet;

use super::*;

fn relight(preset: &str, strength: f64) -> Relight {
    Relight {
        preset: preset.into(),
        strength,
        color: String::new(),
        angle: Relight::DEFAULT_ANGLE,
    }
}

/// A grey ball filling a `n` by `n` picture, and its depth.
fn ball(n: usize) -> (Vec<u8>, Vec<u8>) {
    let mut rgba = Vec::new();
    let mut depth = Vec::new();
    let r = n as f32 * 0.45;
    for y in 0..n {
        for x in 0..n {
            let dx = (x as f32 + 0.5 - n as f32 / 2.0) / r;
            let dy = (y as f32 + 0.5 - n as f32 / 2.0) / r;
            let d2 = dx * dx + dy * dy;
            rgba.extend_from_slice(&[128, 128, 128, 200]);
            depth.push(if d2 < 1.0 {
                (100.0 + 155.0 * (1.0 - d2).sqrt()) as u8
            } else {
                100
            });
        }
    }
    (rgba, depth)
}

fn mean(rgba: &[u8], n: usize, columns: std::ops::Range<usize>) -> f32 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for y in n / 4..3 * n / 4 {
        for x in columns.clone() {
            sum += f32::from(rgba[(y * n + x) * 4]);
            count += 1.0;
        }
    }
    sum / count
}

fn one(toward: [f32; 3], reach: f32) -> Lighting {
    Lighting {
        ambient: [0.5; 3],
        lights: vec![light([1.0, 1.0, 1.0], toward, 1.0, reach)],
        strength: 1.0,
    }
}

#[test]
fn every_preset_has_its_own_id_and_custom_and_unknown_ones_read_as_they_should() {
    let ids: HashSet<_> = RELIGHT_PRESETS.iter().map(|preset| preset.id).collect();
    assert_eq!(ids.len(), RELIGHT_PRESETS.len());
    assert!(!ids.contains(Relight::CUSTOM));
    for group in RelightGroup::ALL {
        assert!(RELIGHT_PRESETS.iter().any(|preset| preset.group == group));
    }
    for preset in &RELIGHT_PRESETS {
        assert!(
            preset
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b == b'_')
        );
        let lit = lighting(&relight(preset.id, 0.5), 1.0).unwrap();
        assert!(
            !lit.lights.is_empty() && lit.lights.len() <= MAX_LIGHTS,
            "{}",
            preset.id
        );
        assert_eq!(lit.strength, 0.5);
    }
    assert!(lighting(&relight("not_a_light", 1.0), 0.0).is_none());
    let custom = Relight {
        color: "#ff0000".into(),
        angle: 0.0,
        ..relight(Relight::CUSTOM, 1.0)
    };
    let lit = lighting(&custom, 0.0).unwrap();
    assert_eq!(lit.lights[0].color, [1.0, 0.0, 0.0]);
    assert!(lit.lights[0].toward[0] > 0.7, "0 degrees is from the right");
    let default = lighting(&relight(Relight::CUSTOM, 1.0), 0.0).unwrap();
    assert_eq!(default.lights[0].color, CUSTOM_COLOR);
}

#[test]
fn only_creative_presets_move_over_time() {
    for preset in &RELIGHT_PRESETS {
        let at = |t| lighting(&relight(preset.id, 1.0), t).unwrap();
        let changed = (1..20).any(|k| at(0.0) != at(f64::from(k) * 0.37));
        assert_eq!(changed, preset.moves(), "{}", preset.id);
    }
    // A time before the clip or not a number is its start.
    let at = |t| lighting(&relight("rainbow_spots", 1.0), t).unwrap();
    assert_eq!(at(-3.0), at(0.0));
    assert_eq!(at(f64::NAN), at(0.0));
}

#[test]
fn a_light_from_the_left_brightens_the_side_that_faces_it() {
    let n = 96;
    let (mut rgba, depth) = ball(n);
    relight_rgba(
        &mut rgba,
        n as u32,
        n as u32,
        &depth,
        &one([-1.0, 0.0, 0.3], 0.0),
    );
    let (left, right) = (mean(&rgba, n, 10..30), mean(&rgba, n, 66..86));
    assert!(left > right + 20.0, "left {left}, right {right}");
    // Alpha stays.
    assert!(rgba.chunks(4).all(|p| p[3] == 200));
}

#[test]
fn a_light_from_behind_outlines_the_edges_not_the_middle() {
    let n = 96;
    let (mut rgba, depth) = ball(n);
    relight_rgba(
        &mut rgba,
        n as u32,
        n as u32,
        &depth,
        &one([0.8, 0.0, -0.6], 0.0),
    );
    let edge = mean(&rgba, n, 84..88);
    let middle = mean(&rgba, n, 46..50);
    assert!(edge > middle + 20.0, "edge {edge}, middle {middle}");
}

#[test]
fn a_near_reaching_light_brightens_what_is_near_more_than_what_is_far() {
    let n = 64;
    let rgba: Vec<u8> = [120, 120, 120, 255].repeat(n * n);
    // Near on the left half, far on the right, each flat.
    let depth: Vec<u8> = (0..n * n)
        .map(|i| if i % n < n / 2 { 250 } else { 5 })
        .collect();
    let mut lit = rgba.clone();
    relight_rgba(
        &mut lit,
        n as u32,
        n as u32,
        &depth,
        &one([0.0, 0.0, 1.0], 1.0),
    );
    let (near, far) = (mean(&lit, n, 4..24), mean(&lit, n, 40..60));
    assert!(near > far + 30.0, "near {near}, far {far}");
    let mut even = rgba;
    relight_rgba(
        &mut even,
        n as u32,
        n as u32,
        &depth,
        &one([0.0, 0.0, 1.0], 0.0),
    );
    assert!((mean(&even, n, 4..24) - mean(&even, n, 40..60)).abs() < 2.0);
}

#[test]
fn black_stays_black_and_no_strength_or_a_wrong_size_changes_nothing() {
    let n = 32;
    let (_, depth) = ball(n);
    let mut black: Vec<u8> = [0, 0, 0, 255].repeat(n * n);
    let bright = Lighting {
        ambient: [3.0; 3],
        ..one([0.0, 0.0, 1.0], 0.0)
    };
    relight_rgba(&mut black, n as u32, n as u32, &depth, &bright);
    assert!(black.chunks(4).all(|p| p == [0, 0, 0, 255]));
    let (rgba, _) = ball(n);
    let mut off = rgba.clone();
    let none = Lighting {
        strength: 0.0,
        ..bright.clone()
    };
    relight_rgba(&mut off, n as u32, n as u32, &depth, &none);
    assert_eq!(off, rgba);
    let mut wrong = rgba.clone();
    relight_rgba(&mut wrong, n as u32, n as u32, &depth[1..], &bright);
    relight_rgba(&mut wrong, 0, 0, &[], &bright);
    assert_eq!(wrong, rgba);
    // White light at an ambient of one with nothing facing it keeps a
    // picture as it was, within rounding.
    let mut kept = rgba.clone();
    let neutral = Lighting {
        ambient: [1.0; 3],
        lights: Vec::new(),
        strength: 1.0,
    };
    relight_rgba(&mut kept, n as u32, n as u32, &depth, &neutral);
    assert!(kept.iter().zip(&rgba).all(|(a, b)| a.abs_diff(*b) <= 1));
}

#[test]
fn half_strength_lies_between_the_clip_and_the_full_light() {
    let n = 48;
    let (rgba, depth) = ball(n);
    let at = |strength| {
        let mut out = rgba.clone();
        let lit = Lighting {
            strength,
            ..one([-1.0, 0.2, 0.4], 0.0)
        };
        relight_rgba(&mut out, n as u32, n as u32, &depth, &lit);
        mean(&out, n, 8..20)
    };
    let (none, half, full) = (at(0.0), at(0.5), at(1.0));
    assert!(none < half && half < full, "{none} {half} {full}");
}

#[test]
fn each_preset_draws_its_own_tile() {
    let tiles: Vec<Vec<u8>> = RELIGHT_PRESETS
        .iter()
        .map(|preset| relight_tile(&lighting(&relight(preset.id, 1.0), 0.0).unwrap(), 40))
        .collect();
    assert!(tiles.iter().all(|tile| tile.len() == 40 * 40 * 4));
    let distinct: HashSet<_> = tiles.iter().collect();
    assert_eq!(distinct.len(), tiles.len());
}

#[test]
#[ignore = "timing; run with --release --ignored --nocapture"]
fn relight_1080p_timing() {
    let (w, h) = (1920_usize, 1080_usize);
    let mut rgba: Vec<u8> = (0..w * h * 4).map(|i| (i % 251) as u8).collect();
    let depth: Vec<u8> = (0..w * h).map(|i| ((i % w) * 255 / w) as u8).collect();
    let lit = lighting(&relight("rainbow_spots", 1.0), 2.0).unwrap();
    let started = std::time::Instant::now();
    relight_rgba(&mut rgba, w as u32, h as u32, &depth, &lit);
    let _ = std::io::Write::write_all(
        &mut std::io::stderr(),
        format!("relight 1080p: {:?}\n", started.elapsed()).as_bytes(),
    );
}
