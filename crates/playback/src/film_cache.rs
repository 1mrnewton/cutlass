//! Filmstrip thumbnails kept on disk, so a project opened again shows its
//! filmstrips without decoding them.
//!
//! Each source file has a folder named by the file as it is on disk, its
//! path and time of change, then its size and the thumbnail height, so a
//! changed file starts a new folder and is never shown through an old
//! picture. Each thumbnail is one file named by its source time in
//! microseconds: a JPEG, or a PNG when the picture is not opaque, as a
//! JPEG has no alpha.
//!
//! The thumbnail worker owns the cache and is its only caller. A file that
//! does not read back as a thumbnail is a miss, and the picture is decoded
//! again. The cache is held to [`FILM_CACHE_BYTES`]: the first time it is
//! used in a run, the folders written longest ago go until it fits.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use cutlass_engine::{THUMB_HEIGHT, Thumb, ThumbCache, ThumbKey};
use image::{ExtendedColorType, ImageEncoder, ImageFormat};

/// The most the cache keeps on disk, in bytes.
pub const FILM_CACHE_BYTES: u64 = 512 << 20;

/// JPEG quality out of 100.
const JPEG_QUALITY: u8 = 90;

/// The largest file read back as a thumbnail. One is a few tens of
/// kilobytes; a larger file is not one this cache wrote.
const MAX_FILE: u64 = 4 << 20;

/// The widest thumbnail read back: a picture sixteen times as wide as tall.
const MAX_WIDTH: u32 = THUMB_HEIGHT * 16;

/// Filmstrip thumbnails under one folder.
pub struct FilmCache {
    root: PathBuf,
    budget: u64,
    /// Each source's folder, worked out once. `None` for a source that
    /// cannot be read.
    folders: HashMap<PathBuf, Option<PathBuf>>,
    pruned: bool,
}

impl FilmCache {
    /// A cache under `root`, which is made when the first thumbnail is
    /// kept.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self::with_budget(root, FILM_CACHE_BYTES)
    }

    /// [`Self::new`], held to `budget` bytes.
    #[must_use]
    pub fn with_budget(root: PathBuf, budget: u64) -> Self {
        Self {
            root,
            budget,
            folders: HashMap::new(),
            pruned: false,
        }
    }

    /// The folder that holds `source`'s thumbnails.
    fn folder(&mut self, source: &Path) -> Option<PathBuf> {
        if let Some(known) = self.folders.get(source) {
            return known.clone();
        }
        let folder = folder_name(source).map(|name| self.root.join(name));
        self.folders.insert(source.to_path_buf(), folder.clone());
        folder
    }

    /// Hold the cache to its budget, once a run.
    fn prune_once(&mut self) {
        if !std::mem::replace(&mut self.pruned, true) {
            prune(&self.root, self.budget);
        }
    }
}

/// The folder name of `source` as it is on disk, or `None` when the file
/// cannot be read.
fn folder_name(source: &Path) -> Option<String> {
    let canonical = source.canonicalize().ok()?;
    let meta = fs::metadata(&canonical).ok()?;
    let changed = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    let seed = format!("{}\n{changed}", canonical.display());
    // FNV-1a, the same on every run.
    let key = seed.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    Some(format!("{key:016x}_{}_{THUMB_HEIGHT}", meta.len()))
}

/// Delete the folders of `root` written longest ago until what is left
/// takes no more than `budget` bytes.
fn prune(root: &Path, budget: u64) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut folders: Vec<(SystemTime, u64, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| {
            let path = entry.path();
            let written = entry
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let bytes: u64 = fs::read_dir(&path).map_or(0, |files| {
                files
                    .filter_map(Result::ok)
                    .filter_map(|file| file.metadata().ok())
                    .map(|meta| meta.len())
                    .sum()
            });
            (written, bytes, path)
        })
        .collect();
    let mut total: u64 = folders.iter().map(|(_, bytes, _)| *bytes).sum();
    folders.sort_by_key(|(written, _, _)| *written);
    for (_, bytes, path) in folders {
        if total <= budget {
            break;
        }
        if fs::remove_dir_all(&path).is_ok() {
            total = total.saturating_sub(bytes);
        }
    }
}

/// `thumb` as the bytes of a file, and that file's extension: a JPEG when
/// every pixel is opaque, else a PNG.
fn encoded(thumb: &Thumb) -> Option<(Vec<u8>, &'static str)> {
    let mut bytes = Vec::new();
    let (pixels, _) = thumb.rgba.as_chunks::<4>();
    let opaque = pixels.iter().all(|pixel| pixel[3] == u8::MAX);
    if opaque {
        let rgb: Vec<u8> = pixels
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
            .write_image(&rgb, thumb.width, thumb.height, ExtendedColorType::Rgb8)
            .ok()?;
        Some((bytes, "jpg"))
    } else {
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(
                &thumb.rgba,
                thumb.width,
                thumb.height,
                ExtendedColorType::Rgba8,
            )
            .ok()?;
        Some((bytes, "png"))
    }
}

/// The thumbnail in the file at `path`, when it is one this cache could
/// have written.
fn read(path: &Path, format: ImageFormat) -> Option<Thumb> {
    if fs::metadata(path).ok()?.len() > MAX_FILE {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    let picture = image::load_from_memory_with_format(&bytes, format).ok()?;
    let (width, height) = (picture.width(), picture.height());
    if width == 0 || height == 0 || height > THUMB_HEIGHT || width > MAX_WIDTH {
        return None;
    }
    Some(Thumb {
        width,
        height,
        rgba: picture.into_rgba8().into_raw(),
    })
}

impl ThumbCache for FilmCache {
    fn load(&mut self, key: &ThumbKey) -> Option<Thumb> {
        let folder = self.folder(&key.path)?;
        let name = key.time.0;
        read(&folder.join(format!("{name}.jpg")), ImageFormat::Jpeg)
            .or_else(|| read(&folder.join(format!("{name}.png")), ImageFormat::Png))
    }

    fn keep(&mut self, key: &ThumbKey, thumb: &Thumb) {
        let Some(folder) = self.folder(&key.path) else {
            return;
        };
        let Some((bytes, extension)) = encoded(thumb) else {
            return;
        };
        if fs::create_dir_all(&folder).is_err() {
            return;
        }
        self.prune_once();
        // Written aside and moved into place, so a half-written file is
        // never read back.
        let name = key.time.0;
        let part = folder.join(format!(".{name}.part"));
        let written = fs::write(&part, bytes)
            .and_then(|()| fs::rename(&part, folder.join(format!("{name}.{extension}"))));
        if let Err(error) = written {
            let _ = fs::remove_file(&part);
            cutlass_logger::warn!("thumbnail not kept in {}: {error}", folder.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_engine::{Thumb, ThumbCache, ThumbKey};
    use cutlass_model::Micros;

    use super::{FilmCache, folder_name, prune};

    /// A folder of this test's own, with a source file in it.
    fn scratch(name: &str) -> (PathBuf, PathBuf) {
        let folder =
            std::env::temp_dir().join(format!("cutlass-film-cache-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        let source = folder.join("movie.mp4");
        std::fs::write(&source, b"a movie").unwrap();
        (folder, source)
    }

    fn key(source: &Path, time: i64) -> ThumbKey {
        ThumbKey {
            path: source.to_path_buf(),
            time: Micros(time),
        }
    }

    /// A picture that runs from dark to light, with `alpha` everywhere.
    fn ramp(width: u32, height: u32, alpha: u8) -> Thumb {
        let mut rgba = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let level = u8::try_from((x + y) * 255 / (width + height)).unwrap();
                rgba.extend([level, level / 2, 255 - level, alpha]);
            }
        }
        Thumb {
            width,
            height,
            rgba,
        }
    }

    #[test]
    fn a_thumbnail_comes_back_from_disk_in_a_later_run() {
        let (folder, source) = scratch("round");
        let root = folder.join("cache");
        let opaque = ramp(284, 160, 255);
        let clear = ramp(90, 160, 128);
        let mut cache = FilmCache::new(root.clone());
        assert!(
            cache.load(&key(&source, 0)).is_none(),
            "nothing is kept yet"
        );
        assert!(!root.exists(), "a look makes no folder");
        cache.keep(&key(&source, 0), &opaque);
        cache.keep(&key(&source, 2_000_000), &clear);

        // Another run: a new cache over the same folder.
        let mut later = FilmCache::new(root.clone());
        let back = later.load(&key(&source, 0)).expect("the opaque one");
        assert_eq!((back.width, back.height), (284, 160));
        // A JPEG is close, not exact, and stays opaque.
        let apart: u64 = back
            .rgba
            .iter()
            .zip(&opaque.rgba)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        assert!(apart / (opaque.rgba.len() as u64) < 4, "{apart}");
        assert!(
            back.rgba
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| pixel[3] == 255)
        );
        // One that is not opaque is kept whole, alpha and all.
        assert_eq!(later.load(&key(&source, 2_000_000)), Some(clear));
        assert!(later.load(&key(&source, 1_000_000)).is_none());

        // One folder for the file, named by it, with a file a thumbnail.
        let folders: Vec<_> = std::fs::read_dir(&root).unwrap().collect();
        assert_eq!(folders.len(), 1);
        let mut names: Vec<String> = std::fs::read_dir(folders[0].as_ref().unwrap().path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["0.jpg", "2000000.png"]);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_changed_file_is_not_shown_through_its_old_thumbnails() {
        let (folder, source) = scratch("changed");
        let mut cache = FilmCache::new(folder.join("cache"));
        cache.keep(&key(&source, 0), &ramp(64, 32, 255));
        let before = folder_name(&source).unwrap();
        assert_eq!(folder_name(&source), Some(before.clone()));
        assert!(before.ends_with("_7_160"), "{before}");

        // The file grows: a cache opened after that has nothing for it.
        std::fs::write(&source, b"a longer movie").unwrap();
        assert_ne!(folder_name(&source), Some(before));
        let mut later = FilmCache::new(folder.join("cache"));
        assert!(later.load(&key(&source, 0)).is_none());
        // A file that is not there has no folder, and keeping does nothing.
        let gone = folder.join("gone.mp4");
        assert_eq!(folder_name(&gone), None);
        later.keep(&key(&gone, 0), &ramp(64, 32, 255));
        assert!(later.load(&key(&gone, 0)).is_none());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_file_that_is_not_a_thumbnail_is_a_miss() {
        let (folder, source) = scratch("hostile");
        let root = folder.join("cache");
        let mut cache = FilmCache::new(root.clone());
        cache.keep(&key(&source, 0), &ramp(64, 32, 255));
        let held = root.join(folder_name(&source).unwrap());

        // Cut short, not a picture at all, and a picture far too tall.
        let whole = std::fs::read(held.join("0.jpg")).unwrap();
        std::fs::write(held.join("1.jpg"), &whole[..whole.len() / 2]).unwrap();
        std::fs::write(held.join("2.jpg"), b"not a picture").unwrap();
        let mut tall = FilmCache::new(folder.join("other"));
        let big = Thumb {
            width: 8,
            height: 2000,
            rgba: vec![255; 8 * 2000 * 4],
        };
        tall.keep(&key(&source, 3), &big);
        let made = folder
            .join("other")
            .join(folder_name(&source).unwrap())
            .join("3.jpg");
        std::fs::copy(made, held.join("3.jpg")).unwrap();
        for time in [1, 2, 3] {
            assert!(cache.load(&key(&source, time)).is_none(), "{time}");
        }
        assert!(
            cache.load(&key(&source, 0)).is_some(),
            "the good one still reads"
        );
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn the_folders_written_longest_ago_go_first_when_the_cache_is_full() {
        let (folder, _) = scratch("prune");
        let root = folder.join("cache");
        for (name, age) in [("old", 300), ("middle", 200), ("new", 100)] {
            let held = root.join(name);
            std::fs::create_dir_all(&held).unwrap();
            std::fs::write(held.join("0.jpg"), vec![0_u8; 1000]).unwrap();
            let when = std::time::SystemTime::now() - std::time::Duration::from_secs(age);
            std::fs::File::open(&held)
                .unwrap()
                .set_modified(when)
                .unwrap();
        }
        let left = |root: &Path| {
            let mut names: Vec<String> = std::fs::read_dir(root)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        };
        // Room for all three: nothing goes.
        prune(&root, 3000);
        assert_eq!(left(&root), ["middle", "new", "old"]);
        // Room for two, then for one: the oldest go.
        prune(&root, 2500);
        assert_eq!(left(&root), ["middle", "new"]);
        prune(&root, 1000);
        assert_eq!(left(&root), ["new"]);
        // A cache folder that is not there is left alone.
        prune(&folder.join("nowhere"), 0);
        std::fs::remove_dir_all(&folder).unwrap();
    }
}
