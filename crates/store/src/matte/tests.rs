use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::{RelightKey, relight_file};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-matte-store-{}-{}",
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

fn project() -> (Temp, MatteKey) {
    let temp = Temp::new();
    let source = temp.0.join("walk.mp4");
    fs::write(&source, b"a movie").unwrap();
    let key = MatteKey {
        source,
        model: "birefnet-lite-1024".into(),
        rate: 30.0,
        picks: String::new(),
    };
    (temp, key)
}

#[test]
fn a_matte_video_serves_any_clip_inside_its_stretch_in_its_own_folder() {
    let (temp, key) = project();
    let held = TimeRange::new(1_000, 4_000);
    let file = matte_file(&temp.0, &key, &held).unwrap();
    fs::write(&file, b"matte").unwrap();
    assert!(file.starts_with(temp.0.canonicalize().unwrap().join("Resources/video_matte")));
    assert_eq!(
        find_matte(&temp.0, &key, &TimeRange::new(2_000, 1_000)),
        Some((file, held.clone()))
    );
    assert_eq!(find_matte(&temp.0, &key, &TimeRange::new(500, 1_000)), None);
    for other in [
        MatteKey {
            model: "other".into(),
            ..key.clone()
        },
        MatteKey {
            rate: 25.0,
            ..key.clone()
        },
        MatteKey {
            picks: "0123456789abcdef".into(),
            ..key.clone()
        },
    ] {
        assert_eq!(find_matte(&temp.0, &other, &held), None);
    }
}

#[test]
fn a_changed_or_missing_source_or_a_bad_stretch_finds_and_makes_nothing() {
    let (temp, key) = project();
    let held = TimeRange::new(0, 4_000);
    fs::write(matte_file(&temp.0, &key, &held).unwrap(), b"matte").unwrap();
    fs::write(&key.source, b"another movie").unwrap();
    assert_eq!(find_matte(&temp.0, &key, &held), None);
    fs::remove_file(&key.source).unwrap();
    assert!(matte_file(&temp.0, &key, &held).is_err());
    assert_eq!(find_matte(&temp.0, &key, &held), None);
    fs::write(&key.source, b"a movie").unwrap();
    assert!(matte_file(&temp.0, &key, &TimeRange::new(-1, 5)).is_err());
    assert!(matte_file(&temp.0, &key, &TimeRange::new(0, 0)).is_err());
}

#[test]
fn a_matte_video_and_a_depth_video_never_share_a_name() {
    let (temp, key) = project();
    let relight = RelightKey {
        source: key.source.clone(),
        model: key.model.clone(),
        rate: key.rate,
    };
    let held = TimeRange::new(0, 4_000);
    let a = matte_file(&temp.0, &key, &held).unwrap();
    let b = relight_file(&temp.0, &relight, &held).unwrap();
    assert_ne!(a.file_name(), b.file_name());
}

#[test]
fn only_the_mattes_kept_and_other_files_stay() {
    let (temp, key) = project();
    let used = matte_file(&temp.0, &key, &TimeRange::new(0, 1_000)).unwrap();
    let old = matte_file(&temp.0, &key, &TimeRange::new(0, 5_000)).unwrap();
    let half = old.with_extension("part");
    let other = used.with_file_name("notes.txt");
    for file in [&used, &old, &half, &other] {
        fs::write(file, b"x").unwrap();
    }
    assert_eq!(remove_mattes_but(&temp.0, std::slice::from_ref(&used)).unwrap(), 2);
    assert!(used.is_file() && other.is_file());
    assert!(!old.exists() && !half.exists());
    assert_eq!(remove_mattes_but(&temp.0, std::slice::from_ref(&used)).unwrap(), 0);
    let empty = Temp::new();
    assert_eq!(
        remove_mattes_but(&empty.0, &[]).unwrap(),
        0,
        "no folder yet"
    );
}
