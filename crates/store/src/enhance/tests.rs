use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-enhance-store-{}-{}",
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

fn project() -> (Temp, EnhanceKey) {
    let temp = Temp::new();
    let source = temp.0.join("walk.mp4");
    fs::write(&source, b"a movie").unwrap();
    let key = EnhanceKey {
        source,
        level: 1,
        denoise: 0,
        interpolate: 0,
        motion_blur: String::new(),
        model: "realesr-general-x4v3".into(),
        rate: 30.0,
    };
    (temp, key)
}

/// Makes the copy of `range` as a job would, once it finished.
fn keep(temp: &Temp, key: &EnhanceKey, range: &TimeRange) -> PathBuf {
    let file = enhanced_file(&temp.0, key, range).unwrap();
    fs::write(&file, b"enhanced").unwrap();
    file
}

#[test]
fn a_copy_serves_any_clip_inside_its_stretch() {
    let (temp, key) = project();
    let held = TimeRange::new(1_000_000, 4_000_000);
    let file = keep(&temp, &key, &held);
    assert!(
        file.starts_with(
            temp.0
                .canonicalize()
                .unwrap()
                .join("Resources/video_enhance")
        )
    );
    for inside in [
        held.clone(),
        TimeRange::new(1_000_000, 1),
        TimeRange::new(2_000_000, 3_000_000),
    ] {
        assert_eq!(
            find_enhanced(&temp.0, &key, &inside),
            Some((file.clone(), held.clone()))
        );
    }
    for outside in [
        TimeRange::new(999_999, 10),
        TimeRange::new(4_000_000, 1_000_001),
    ] {
        assert_eq!(find_enhanced(&temp.0, &key, &outside), None, "{outside:?}");
    }
}

#[test]
fn the_shortest_covering_copy_wins() {
    let (temp, key) = project();
    keep(&temp, &key, &TimeRange::new(0, 9_000_000));
    let short = keep(&temp, &key, &TimeRange::new(2_000_000, 2_000_000));
    assert_eq!(
        find_enhanced(&temp.0, &key, &TimeRange::new(2_500_000, 1_000_000)).map(|found| found.0),
        Some(short)
    );
}

#[test]
fn another_level_model_rate_or_a_changed_source_finds_nothing() {
    let (temp, key) = project();
    let range = TimeRange::new(0, 1_000_000);
    keep(&temp, &key, &range);
    assert!(find_enhanced(&temp.0, &key, &range).is_some());
    for other in [
        EnhanceKey {
            level: 2,
            ..key.clone()
        },
        EnhanceKey {
            model: "better".into(),
            ..key.clone()
        },
        EnhanceKey {
            rate: 60.0,
            ..key.clone()
        },
        EnhanceKey {
            denoise: 1,
            ..key.clone()
        },
    ] {
        assert_eq!(find_enhanced(&temp.0, &other, &range), None, "{other:?}");
    }
    fs::write(&key.source, b"a longer movie than before").unwrap();
    assert_eq!(find_enhanced(&temp.0, &key, &range), None);
    fs::remove_file(&key.source).unwrap();
    assert_eq!(find_enhanced(&temp.0, &key, &range), None);
    assert!(enhanced_file(&temp.0, &key, &range).is_err());
}

#[test]
fn no_folder_strange_names_and_bad_stretches_find_nothing() {
    let (temp, key) = project();
    let range = TimeRange::new(0, 1_000_000);
    assert_eq!(find_enhanced(&temp.0, &key, &range), None);
    let folder = keep(&temp, &key, &TimeRange::new(5_000_000, 1))
        .parent()
        .unwrap()
        .to_path_buf();
    let stem = key.stem().unwrap();
    for name in [
        format!("{stem}-0-1000000.mov"),
        format!("{stem}-x-1000000.mp4"),
        format!("{stem}--5-2000000.mp4"),
        format!("{stem}-0-0.mp4"),
        format!("{stem}0-2000000.mp4"),
        "notes.txt".to_owned(),
    ] {
        fs::write(folder.join(name), b"x").unwrap();
    }
    fs::create_dir(folder.join(format!("{stem}-0-3000000.mp4"))).unwrap();
    assert_eq!(find_enhanced(&temp.0, &key, &range), None);
    for bad in [TimeRange::new(-1, 10), TimeRange::new(0, 0)] {
        assert!(enhanced_file(&temp.0, &key, &bad).is_err());
    }
}

#[cfg(unix)]
#[test]
fn a_folder_that_leads_out_of_the_project_is_refused() {
    let (temp, key) = project();
    let outside = Temp::new();
    fs::create_dir_all(temp.0.join("Resources")).unwrap();
    std::os::unix::fs::symlink(&outside.0, temp.0.join("Resources/video_enhance")).unwrap();
    let range = TimeRange::new(0, 1_000_000);
    assert!(enhanced_file(&temp.0, &key, &range).is_err());
    let stem = key.stem().unwrap();
    fs::write(outside.0.join(format!("{stem}-0-1000000.mp4")), b"x").unwrap();
    assert_eq!(find_enhanced(&temp.0, &key, &range), None);
}

#[test]
fn a_copy_that_is_only_denoised_keeps_its_own_folder() {
    let (temp, key) = project();
    let denoised = EnhanceKey {
        level: 0,
        denoise: 2,
        ..key.clone()
    };
    let range = TimeRange::new(0, 1_000_000);
    let file = keep(&temp, &denoised, &range);
    assert!(file.parent().unwrap().ends_with("Resources/video_denoise"));
    assert_eq!(
        find_enhanced(&temp.0, &denoised, &range).map(|f| f.0),
        Some(file)
    );
    assert_eq!(find_enhanced(&temp.0, &key, &range), None);
    let both = EnhanceKey { denoise: 2, ..key };
    assert!(
        keep(&temp, &both, &range)
            .parent()
            .unwrap()
            .ends_with("Resources/video_enhance")
    );
    assert_ne!(
        find_enhanced(&temp.0, &both, &range),
        find_enhanced(&temp.0, &denoised, &range)
    );
}

#[test]
fn a_copy_that_is_only_enhanced_keeps_the_name_it_had_before_denoising() {
    // The hash as it was before keys had a noise level, which copies made
    // then are named by.
    let (_temp, key) = project();
    let meta = fs::metadata(&key.source).unwrap();
    let modified = meta
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(key.source.to_string_lossy().as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&meta.len().to_le_bytes());
    bytes.extend_from_slice(&modified.to_le_bytes());
    bytes.extend_from_slice(&key.level.to_le_bytes());
    bytes.extend_from_slice(key.model.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&key.rate.to_bits().to_le_bytes());
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    assert_eq!(key.stem().unwrap(), format!("{hash:016x}"));
}

#[test]
fn a_copy_that_is_only_interpolated_has_its_own_folder_and_name() {
    let (temp, key) = project();
    let range = TimeRange::new(0, 1_000_000);
    let flowed = EnhanceKey {
        level: 0,
        interpolate: 60,
        motion_blur: String::new(),
        model: "rife-v4.26".into(),
        rate: 120.0,
        ..key.clone()
    };
    let file = keep(&temp, &flowed, &range);
    assert!(
        file.parent()
            .unwrap()
            .ends_with("Resources/video_interpolate")
    );
    assert_eq!(
        find_enhanced(&temp.0, &flowed, &range).map(|f| f.0),
        Some(file)
    );
    let at_30 = EnhanceKey {
        interpolate: 30,
        motion_blur: String::new(),
        ..flowed.clone()
    };
    assert_eq!(find_enhanced(&temp.0, &at_30, &range), None);
    // Interpolated and enhanced goes with the enhanced copies, named apart
    // from a copy that is only enhanced.
    let both = EnhanceKey {
        interpolate: 60,
        motion_blur: String::new(),
        ..key.clone()
    };
    assert!(
        keep(&temp, &both, &range)
            .parent()
            .unwrap()
            .ends_with("Resources/video_enhance")
    );
    assert_ne!(both.stem().unwrap(), key.stem().unwrap());
    let denoised = EnhanceKey {
        level: 0,
        denoise: 1,
        interpolate: 60,
        motion_blur: String::new(),
        ..key
    };
    assert!(
        keep(&temp, &denoised, &range)
            .parent()
            .unwrap()
            .ends_with("Resources/video_denoise")
    );
}

#[test]
fn a_blurred_copy_has_its_own_name_and_one_only_blurred_its_own_folder() {
    let (temp, key) = project();
    let range = TimeRange::new(0, 1_000_000);
    let blurred = EnhanceKey {
        motion_blur: "b1.000-m0.000-d0-x1".into(),
        ..key.clone()
    };
    assert_ne!(blurred.stem().unwrap(), key.stem().unwrap());
    let other = EnhanceKey {
        motion_blur: "b0.500-m0.000-d0-x1".into(),
        ..key.clone()
    };
    assert_ne!(blurred.stem().unwrap(), other.stem().unwrap());
    let only = EnhanceKey {
        level: 0,
        ..blurred
    };
    let file = enhanced_file(&temp.0, &only, &range).unwrap();
    assert!(
        file.parent()
            .unwrap()
            .ends_with("Resources/video_motion_blur")
    );
}
