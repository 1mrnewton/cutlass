//! Measured camera paths for steadying clips, kept in the project's
//! `Resources/video_stable` folder.
//!
//! Measuring a clip's shake is slow, so it is done once and kept with the
//! project: export and other machines read the same file, and no cache
//! eviction takes it away. A path is named by what was measured: the
//! source file as it was on disk (its size and when it last changed), the
//! source range, and the frames per second. A file that no longer matches
//! (the source changed, or two keys share a name) reads as missing, so the
//! clip is measured again. So does a file that is not one this build wrote.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use cutlass_model::Micros;
use cutlass_motion::{Move, Path as CameraPath};
use serde::{Deserialize, Serialize};

use crate::StoreError;
use crate::folder::{RESOURCES_DIR, io_error, stay_inside, write_bytes};

const STABLE_DIR: &str = "video_stable";

/// The version of the file this build writes and reads.
const VERSION: u32 = 1;

/// The largest file read back: a million frames and room to spare.
const MAX_BYTES: u64 = 96 * 1024 * 1024;

/// The most frames a kept path may hold.
const MAX_MOVES: usize = 1_000_000;

/// What was measured: a stretch of one source file at a frame rate.
#[derive(Clone, Debug, PartialEq)]
pub struct StableKey {
    /// The source file, as the shell opened it.
    pub source: PathBuf,
    /// The source range measured, in the file's own time.
    pub start: Micros,
    pub duration: Micros,
    /// Frames measured per second.
    pub rate: f64,
}

/// The source's identity on disk, and what was measured of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Identity {
    source: String,
    size: u64,
    /// Milliseconds since 1970 when the file last changed, where the disk
    /// keeps that.
    modified: Option<i64>,
    start: i64,
    duration: i64,
    rate: f64,
}

#[derive(Serialize, Deserialize)]
struct Stored {
    version: u32,
    identity: Identity,
    width: u32,
    height: u32,
    /// Each move as `[dx, dy, turn, scale, confidence]`.
    moves: Vec<[f64; 5]>,
}

impl StableKey {
    fn identity(&self) -> Result<Identity, StoreError> {
        let meta = fs::metadata(&self.source).map_err(|error| io_error(&self.source, error))?;
        let modified = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .and_then(|since| i64::try_from(since.as_millis()).ok());
        Ok(Identity {
            source: self.source.to_string_lossy().into_owned(),
            size: meta.len(),
            modified,
            start: self.start.0,
            duration: self.duration.0,
            rate: self.rate,
        })
    }
}

/// The file `identity` is kept in, under `project`.
fn file_for(project: &Path, identity: &Identity) -> PathBuf {
    let name = format!("{:016x}.json", fnv(&identity_bytes(identity)));
    project.join(RESOURCES_DIR).join(STABLE_DIR).join(name)
}

fn identity_bytes(identity: &Identity) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(identity.source.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&identity.size.to_le_bytes());
    bytes.extend_from_slice(&identity.modified.unwrap_or(i64::MIN).to_le_bytes());
    bytes.extend_from_slice(&identity.start.to_le_bytes());
    bytes.extend_from_slice(&identity.duration.to_le_bytes());
    bytes.extend_from_slice(&identity.rate.to_bits().to_le_bytes());
    bytes
}

/// FNV-1a, 64 bits: a file name, not a lock.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Keep `path`, measured as `key` says, in the project at `project`. A
/// move that is not a number is kept as still.
pub fn save_camera_path(
    project: &Path,
    key: &StableKey,
    path: &CameraPath,
) -> Result<(), StoreError> {
    let identity = key.identity()?;
    let project = project
        .canonicalize()
        .map_err(|error| io_error(project, error))?;
    let file = file_for(&project, &identity);
    let moves = path
        .moves
        .iter()
        .map(|m| {
            let row = [m.dx, m.dy, m.turn, m.scale, f64::from(m.confidence)];
            if row.iter().all(|v| v.is_finite()) {
                row
            } else {
                [0.0, 0.0, 0.0, 1.0, 0.0]
            }
        })
        .collect();
    let stored = Stored {
        version: VERSION,
        identity,
        width: path.width,
        height: path.height,
        moves,
    };
    let bytes = serde_json::to_vec(&stored).map_err(|error| StoreError::Json {
        path: file.clone(),
        message: error.to_string(),
    })?;
    write_bytes(&project, &file, &bytes)
}

/// The path kept for `key` in the project at `project`, or `None` when
/// there is none, the source changed since, or the file is not one this
/// build reads.
#[must_use]
pub fn load_camera_path(project: &Path, key: &StableKey) -> Option<CameraPath> {
    let identity = key.identity().ok()?;
    let project = project.canonicalize().ok()?;
    let file = file_for(&project, &identity);
    stay_inside(&project, &file).ok()?;
    let mut bytes = Vec::new();
    File::open(&file)
        .ok()?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_BYTES {
        return None;
    }
    let stored: Stored = serde_json::from_slice(&bytes).ok()?;
    if stored.version != VERSION
        || stored.identity != identity
        || stored.width == 0
        || stored.height == 0
        || stored.moves.len() > MAX_MOVES
    {
        return None;
    }
    let moves = stored
        .moves
        .iter()
        .map(|&[dx, dy, turn, scale, confidence]| Move {
            dx,
            dy,
            turn,
            scale,
            confidence: confidence as f32,
        })
        .collect();
    Some(CameraPath {
        width: stored.width,
        height: stored.height,
        rate: key.rate,
        moves,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cutlass-stable-{}-{}",
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

    /// A project folder and a source file beside it.
    fn setup() -> (Temp, PathBuf, StableKey) {
        let temp = Temp::new();
        let project = temp.0.join("project");
        fs::create_dir_all(&project).unwrap();
        let source = temp.0.join("walk.mp4");
        fs::write(&source, b"not really a movie").unwrap();
        let key = StableKey {
            source,
            start: Micros(1_000_000),
            duration: Micros(4_000_000),
            rate: 30.0,
        };
        (temp, project, key)
    }

    fn camera_path() -> CameraPath {
        CameraPath {
            width: 640,
            height: 360,
            rate: 30.0,
            moves: (0..120)
                .map(|i| Move {
                    dx: f64::from(i).sin() * 3.0,
                    dy: -0.25,
                    turn: 0.001 * f64::from(i % 7),
                    scale: 1.0 + 0.0001 * f64::from(i % 3),
                    confidence: 0.9,
                })
                .collect(),
        }
    }

    #[test]
    fn a_kept_path_reads_back_the_same() {
        let (_temp, project, key) = setup();
        let path = camera_path();
        save_camera_path(&project, &key, &path).unwrap();
        assert_eq!(load_camera_path(&project, &key), Some(path));
        let folder = project.join(RESOURCES_DIR).join(STABLE_DIR);
        assert_eq!(fs::read_dir(folder).unwrap().count(), 1);
    }

    #[test]
    fn nothing_kept_reads_as_missing() {
        let (_temp, project, key) = setup();
        assert_eq!(load_camera_path(&project, &key), None);
    }

    #[test]
    fn another_range_rate_or_a_changed_source_is_measured_again() {
        let (_temp, project, key) = setup();
        save_camera_path(&project, &key, &camera_path()).unwrap();
        let later = StableKey {
            start: Micros(2_000_000),
            ..key.clone()
        };
        assert_eq!(load_camera_path(&project, &later), None);
        let faster = StableKey {
            rate: 60.0,
            ..key.clone()
        };
        assert_eq!(load_camera_path(&project, &faster), None);
        fs::write(&key.source, b"a longer, different movie").unwrap();
        assert_eq!(load_camera_path(&project, &key), None);
    }

    #[test]
    fn a_missing_source_is_refused_on_save_and_missing_on_load() {
        let (_temp, project, mut key) = setup();
        key.source = key.source.with_file_name("gone.mp4");
        assert!(matches!(
            save_camera_path(&project, &key, &camera_path()),
            Err(StoreError::Io { .. })
        ));
        assert_eq!(load_camera_path(&project, &key), None);
    }

    #[test]
    fn a_damaged_or_foreign_file_reads_as_missing() {
        let (_temp, project, key) = setup();
        save_camera_path(&project, &key, &camera_path()).unwrap();
        let file = file_for(&project.canonicalize().unwrap(), &key.identity().unwrap());
        let good = fs::read(&file).unwrap();
        for bad in [
            b"{".to_vec(),
            good[..good.len() / 2].to_vec(),
            String::from_utf8(good.clone())
                .unwrap()
                .replace("\"version\":1", "\"version\":2")
                .into_bytes(),
            String::from_utf8(good.clone())
                .unwrap()
                .replace("\"width\":640", "\"width\":0")
                .into_bytes(),
            String::from_utf8(good.clone())
                .unwrap()
                .replace("\"size\":", "\"size\":1")
                .into_bytes(),
        ] {
            fs::write(&file, bad).unwrap();
            assert_eq!(load_camera_path(&project, &key), None);
        }
    }

    #[test]
    fn a_move_that_is_not_a_number_is_kept_as_still() {
        let (_temp, project, key) = setup();
        let mut path = camera_path();
        path.moves[3].dx = f64::NAN;
        path.moves[4].scale = f64::INFINITY;
        save_camera_path(&project, &key, &path).unwrap();
        let back = load_camera_path(&project, &key).unwrap();
        assert_eq!(back.moves[3], Move::STILL);
        assert_eq!(back.moves[4], Move::STILL);
        assert_eq!(back.moves[5], path.moves[5]);
    }

    #[cfg(unix)]
    #[test]
    fn a_resources_folder_that_leads_outside_is_refused() {
        let (temp, project, key) = setup();
        let outside = temp.0.join("outside");
        fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, project.join(RESOURCES_DIR)).unwrap();
        assert!(matches!(
            save_camera_path(&project, &key, &camera_path()),
            Err(StoreError::PathEscapes { .. })
        ));
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
    }
}
