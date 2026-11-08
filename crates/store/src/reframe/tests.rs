use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-reframe-store-{}-{}",
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

fn project() -> (Temp, ReframeKey) {
    let temp = Temp::new();
    let source = temp.0.join("walk.mp4");
    fs::write(&source, b"a movie").unwrap();
    let key = ReframeKey {
        source,
        range: TimeRange::new(0, 4_000_000),
        model: "depth-anything-v2-small-252".into(),
        rate: 5.0,
    };
    (temp, key)
}

fn path() -> SubjectPath {
    SubjectPath {
        width: 640,
        height: 360,
        subjects: vec![
            Subject {
                x: 0.25,
                y: 0.5,
                weight: 0.75,
            },
            Subject::NONE,
        ],
    }
}

#[test]
fn a_kept_path_reads_back_and_another_key_or_a_changed_source_finds_none() {
    let (temp, key) = project();
    assert_eq!(load_subject_path(&temp.0, &key), None);
    save_subject_path(&temp.0, &key, &path()).unwrap();
    assert_eq!(load_subject_path(&temp.0, &key), Some(path()));
    let folder = temp.0.join("Resources/video_reframe");
    assert_eq!(fs::read_dir(&folder).unwrap().count(), 1);
    for other in [
        ReframeKey {
            rate: 10.0,
            ..key.clone()
        },
        ReframeKey {
            model: "other".into(),
            ..key.clone()
        },
        ReframeKey {
            range: TimeRange::new(0, 1),
            ..key.clone()
        },
    ] {
        assert_eq!(load_subject_path(&temp.0, &other), None);
    }
    fs::write(&key.source, b"another movie").unwrap();
    assert_eq!(load_subject_path(&temp.0, &key), None);
}

#[test]
fn a_broken_sample_is_kept_as_nothing_found_and_a_damaged_file_reads_as_none() {
    let (temp, key) = project();
    let mut broken = path();
    broken.subjects[0].x = f64::NAN;
    save_subject_path(&temp.0, &key, &broken).unwrap();
    assert_eq!(
        load_subject_path(&temp.0, &key).unwrap().subjects[0],
        Subject::NONE
    );
    let file = fs::read_dir(temp.0.join("Resources/video_reframe"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    for damaged in [
        &b"not json"[..],
        br#"{"version":9,"width":1,"height":1,"subjects":[[0,0,0]]}"#,
        br#"{"version":1,"width":0,"height":1,"subjects":[[0,0,0]]}"#,
    ] {
        fs::write(&file, damaged).unwrap();
        assert_eq!(load_subject_path(&temp.0, &key), None);
    }
    fs::remove_file(&key.source).unwrap();
    assert!(save_subject_path(&temp.0, &key, &path()).is_err());
}
