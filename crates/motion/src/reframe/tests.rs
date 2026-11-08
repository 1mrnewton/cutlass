use super::*;

fn at(x: f64) -> Subject {
    Subject {
        x,
        y: 0.5,
        weight: 1.0,
    }
}

const TALL: (f64, f64) = (0.316, 1.0);

#[test]
fn a_tall_window_in_a_wide_frame_covers_its_height_and_a_wide_one_its_width() {
    let (w, h) = window_size(16.0 / 9.0, 9.0 / 16.0);
    assert!((w - 0.31640625).abs() < 1e-9 && h == 1.0);
    let (w, h) = window_size(9.0 / 16.0, 16.0 / 9.0);
    assert!(w == 1.0 && (h - 0.31640625).abs() < 1e-9);
    assert_eq!(window_size(1.0, 1.0), (1.0, 1.0));
    assert_eq!(window_size(0.0, 1.0), (1.0, 1.0));
    assert_eq!(window_size(1.0, f64::NAN), (1.0, 1.0));
}

#[test]
fn the_window_follows_a_subject_to_the_side_and_stays_inside_the_frame() {
    let subjects = [vec![at(0.5); 10], vec![at(0.95); 60]].concat();
    let planned = plan_reframe(&subjects, 5.0, TALL, Follow::from_settings(2, 2));
    assert_eq!(planned.len(), subjects.len());
    let (x0, _) = planned[0];
    let (last, y) = planned[planned.len() - 1];
    assert!((x0 - 0.5).abs() < 0.1, "{x0}");
    assert!((last - (1.0 - TALL.0 / 2.0)).abs() < 1e-6, "{last}");
    assert_eq!(y, 0.5, "a full-height window does not move up or down");
    assert!(
        planned
            .iter()
            .all(|(x, _)| *x >= TALL.0 / 2.0 - 1e-9 && *x <= 1.0 - TALL.0 / 2.0 + 1e-9)
    );
}

#[test]
fn a_slow_camera_moves_no_faster_than_its_speed() {
    let subjects = [vec![at(0.2); 5], vec![at(0.8); 60]].concat();
    for (speed, most) in [(3, 0.12), (1, 0.6)] {
        let planned = plan_reframe(&subjects, 5.0, TALL, Follow::from_settings(1, speed));
        let fastest = planned
            .windows(2)
            .map(|pair| (pair[1].0 - pair[0].0).abs() * 5.0)
            .fold(0.0, f64::max);
        assert!(fastest <= most * 1.05, "speed {speed}: {fastest}");
    }
}

#[test]
fn a_stable_window_ignores_a_subject_shaking_about_its_middle() {
    let shaky: Vec<Subject> = (0..80)
        .map(|k| at(0.5 + if k % 2 == 0 { 0.03 } else { -0.03 }))
        .collect();
    let planned = plan_reframe(&shaky, 5.0, TALL, Follow::from_settings(3, 2));
    let xs = planned.iter().map(|(x, _)| *x);
    let spread = xs.clone().fold(f64::MIN, f64::max) - xs.fold(f64::MAX, f64::min);
    assert!(spread < 0.005, "{spread}");
    assert!((planned[40].0 - 0.5).abs() < 0.05);
}

#[test]
fn frames_with_nothing_found_or_broken_values_hold_the_last_place() {
    let mut subjects = vec![at(0.3); 10];
    subjects.extend(vec![Subject::NONE; 20]);
    subjects.push(Subject {
        x: f64::NAN,
        y: 0.1,
        weight: 1.0,
    });
    let planned = plan_reframe(&subjects, 5.0, TALL, Follow::from_settings(1, 1));
    assert!(planned.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
    assert!(
        planned.iter().all(|(x, _)| (x - 0.3).abs() < 0.05),
        "{planned:?}"
    );
    assert!(plan_reframe(&[], 5.0, TALL, Follow::from_settings(2, 2)).is_empty());
    // A window as big as the frame sits in its middle.
    let whole = plan_reframe(&subjects, 0.0, (1.0, 1.0), Follow::from_settings(2, 2));
    assert!(whole.iter().all(|p| *p == (0.5, 0.5)));
}

#[test]
fn an_hour_of_samples_plans_quickly() {
    let subjects: Vec<Subject> = (0..18_000)
        .map(|k| at((k as f64 * 0.01).sin() * 0.4 + 0.5))
        .collect();
    let started = std::time::Instant::now();
    let planned = plan_reframe(&subjects, 5.0, TALL, Follow::from_settings(2, 2));
    assert_eq!(planned.len(), subjects.len());
    assert!(started.elapsed().as_millis() < 500);
}
