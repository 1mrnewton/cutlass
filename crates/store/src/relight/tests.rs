use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::{EnhanceKey, enhanced_file};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-relight-store-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn project() -> (Temp, RelightKey) {
    let temp = Temp::new();
    let source = temp.0.join("walk.mp4");
    fs::write(&source, b"a movie").unwrap();
    let key = RelightKey {
        source,
        model: "depth-anything-v2-small-364".into(),
        rate: 30.0,
    };
    (temp, key)
}

#[test]
fn a_depth_video_serves_any_clip_inside_its_stretch_in_its_own_folder() {
    let (temp, key) = project();
    let held = TimeRange::new(1_000, 4_000);
    let file = relight_file(&temp.0, &key, &held).unwrap();
    fs::write(&file, b"depth").unwrap();
    assert!(
        file.starts_with(
            temp.0
                .canonicalize()
                .unwrap()
                .join("Resources/video_relight")
        )
    );
    assert_eq!(
        find_relight(&temp.0, &key, &TimeRange::new(2_000, 1_000)),
        Some((file, held.clone()))
    );
    assert_eq!(
        find_relight(&temp.0, &key, &TimeRange::new(500, 1_000)),
        None
    );
    for other in [
        RelightKey {
            model: "other".into(),
            ..key.clone()
        },
        RelightKey {
            rate: 25.0,
            ..key.clone()
        },
    ] {
        assert_eq!(find_relight(&temp.0, &other, &held), None);
    }
}

#[test]
fn a_changed_or_missing_source_or_a_bad_stretch_finds_and_makes_nothing() {
    let (temp, key) = project();
    let held = TimeRange::new(0, 4_000);
    fs::write(relight_file(&temp.0, &key, &held).unwrap(), b"depth").unwrap();
    fs::write(&key.source, b"another movie").unwrap();
    assert_eq!(find_relight(&temp.0, &key, &held), None);
    fs::remove_file(&key.source).unwrap();
    assert!(relight_file(&temp.0, &key, &held).is_err());
    assert_eq!(find_relight(&temp.0, &key, &held), None);
    fs::write(&key.source, b"a movie").unwrap();
    assert!(relight_file(&temp.0, &key, &TimeRange::new(-1, 5)).is_err());
    assert!(relight_file(&temp.0, &key, &TimeRange::new(0, 0)).is_err());
}

#[test]
fn a_depth_video_and_an_enhanced_copy_never_share_a_name() {
    let (temp, key) = project();
    let enhance = EnhanceKey {
        source: key.source.clone(),
        level: 0,
        denoise: 0,
        interpolate: 0,
        motion_blur: String::new(),
        model: key.model.clone(),
        rate: key.rate,
    };
    let held = TimeRange::new(0, 4_000);
    let a = relight_file(&temp.0, &key, &held).unwrap();
    let b = enhanced_file(&temp.0, &enhance, &held).unwrap();
    assert_ne!(a.file_name(), b.file_name());
}
