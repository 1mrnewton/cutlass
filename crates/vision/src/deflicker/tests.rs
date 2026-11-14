use super::{DEFLICKER_REACH, Deflicker, mean_brightness};

/// A grey frame of 64 pixels at `value`.
fn grey(value: u8) -> Vec<u8> {
    vec![value; 64 * 3]
}

/// Each frame of a clip of grey frames at `values`, evened out, and how
/// many reads it took.
fn run(values: &[u8]) -> (Vec<Vec<u8>>, usize) {
    let mut deflicker = Deflicker::new(values.len());
    let mut reads = 0;
    let mut read = |k: usize| -> Result<Vec<u8>, String> {
        reads += 1;
        values
            .get(k)
            .map(|value| grey(*value))
            .ok_or_else(|| format!("no frame {k}"))
    };
    let frames = (0..values.len())
        .map(|k| deflicker.frame(k, &mut read).unwrap())
        .collect();
    (frames, reads)
}

fn brightness(frames: &[Vec<u8>]) -> Vec<f32> {
    frames.iter().map(|frame| mean_brightness(frame)).collect()
}

/// How far the brightness moves from one frame to the next, at most.
fn largest_step(values: &[f32]) -> f32 {
    values
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0, f32::max)
}

#[test]
fn a_steady_clip_is_left_as_it_is_and_each_frame_is_read_once() {
    let values = [120; 20];
    let (frames, reads) = run(&values);
    assert_eq!(reads, values.len());
    for frame in frames {
        assert_eq!(frame, grey(120));
    }
}

#[test]
fn brightness_that_alternates_is_evened_out() {
    // A swing of 20% each way, frame by frame.
    let values: Vec<u8> = (0..30_u8)
        .map(|k| if k.is_multiple_of(2) { 120 } else { 80 })
        .collect();
    let before = largest_step(&values.iter().map(|v| f32::from(*v)).collect::<Vec<_>>());
    let (frames, _) = run(&values);
    let after = brightness(&frames);
    // Away from the ends, where the window is whole, little is left.
    let middle = &after[DEFLICKER_REACH..after.len() - DEFLICKER_REACH];
    assert!(
        largest_step(middle) < before / 6.0,
        "{before} to {middle:?}"
    );
}

#[test]
fn brightness_that_jumps_at_random_is_evened_out() {
    // A timelapse whose exposure wanders by up to 15%.
    let jitter = [
        0, 12, -9, 15, -14, 3, 8, -11, 14, -6, 0, 10, -15, 5, -3, 13, -12, 7, -8, 11,
    ];
    let values: Vec<u8> = jitter.iter().map(|j| (100 + j) as u8).collect();
    let (frames, _) = run(&values);
    let after = brightness(&frames);
    let spread = |values: &[f32]| {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
    };
    let before: Vec<f32> = values.iter().map(|v| f32::from(*v)).collect();
    let middle = |values: &[f32]| values[DEFLICKER_REACH..values.len() - DEFLICKER_REACH].to_vec();
    assert!(
        spread(&middle(&after)) < spread(&middle(&before)) / 2.0,
        "{before:?} to {after:?}"
    );
}

#[test]
fn a_fade_keeps_its_brightness() {
    let values: Vec<u8> = (0..25).map(|k| (20 + k * 8) as u8).collect();
    let (frames, _) = run(&values);
    for (frame, value) in frames.iter().zip(&values) {
        let shown = f32::from(frame[0]);
        assert!(
            (shown - f32::from(*value)).abs() <= 1.0,
            "{value} became {shown}"
        );
    }
}

#[test]
fn the_frames_beside_a_cut_are_not_pulled_toward_the_other_shot() {
    let mut values = vec![60; 12];
    values.extend([150; 12]);
    let (frames, _) = run(&values);
    for (frame, value) in frames.iter().zip(&values) {
        assert_eq!(frame, &grey(*value));
    }
}

#[test]
fn black_is_left_black_and_the_first_and_last_frames_are_kept() {
    let (frames, _) = run(&[0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(frames.iter().all(|frame| frame == &grey(0)));
    let (frames, _) = run(&[100, 60, 100, 60, 100, 60, 100]);
    assert_eq!(frames[0], grey(100));
    assert_eq!(frames[6], grey(100));
}

#[test]
fn a_frame_asked_for_again_is_read_again_and_evened_the_same() {
    let values = [100_u8, 80, 100, 80, 100, 80, 100, 80, 100, 80, 100, 80];
    let mut deflicker = Deflicker::new(values.len());
    let mut reads = Vec::new();
    let mut read = |k: usize| -> Result<Vec<u8>, String> {
        reads.push(k);
        Ok(grey(values[k]))
    };
    let first: Vec<_> = (0..6)
        .map(|k| deflicker.frame(k, &mut read).unwrap())
        .collect();
    let again = deflicker.frame(4, &mut read).unwrap();
    assert_eq!(again, first[4]);
    assert_eq!(reads.iter().filter(|k| **k == 4).count(), 2);
}

#[test]
fn a_read_that_fails_is_handed_back() {
    let mut deflicker = Deflicker::new(10);
    let mut read = |k: usize| -> Result<Vec<u8>, String> {
        if k == 3 {
            Err("frame 3 is broken".into())
        } else {
            Ok(grey(100))
        }
    };
    assert_eq!(
        deflicker.frame(0, &mut read),
        Err("frame 3 is broken".to_owned())
    );
}

#[test]
fn brightness_weighs_green_most() {
    let red = [255, 0, 0].repeat(4);
    let green = [0, 255, 0].repeat(4);
    let blue = [0, 0, 255].repeat(4);
    assert!(mean_brightness(&green) > mean_brightness(&red));
    assert!(mean_brightness(&red) > mean_brightness(&blue));
    assert!((mean_brightness(&grey(200)) - 200.0).abs() < 0.01);
    assert_eq!(mean_brightness(&[]), 0.0);
}
