//! Proxies: a smaller copy of a source movie that the preview plays in its
//! place, so 4K and long clips scrub and play smoothly.
//!
//! A proxy is named by the source file as it is on disk, its path, size and
//! time of change, so a changed file is not shown through a stale proxy. It
//! has pictures only, at the source's own aspect and rate; the sound still
//! comes from the source. Export never reads a proxy.

use std::path::Path;

use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange, Track};

use crate::ExportRequest;
use crate::prerender::render_request;

/// The short side a proxy is drawn at: the smallest export step.
pub const PROXY_SIDE: u32 = crate::VIDEO_SHORT_SIDES[0];

/// A source whose short side is above this wants a proxy.
const LARGE_SIDE: u32 = 1080;

/// A source at least this long wants a proxy, whatever its size.
const LONG: Micros = Micros(20 * 60 * 1_000_000);

/// Whether a movie of `width` by `height` lasting `duration` is heavy
/// enough that the preview should play a proxy of it.
#[must_use]
pub fn wants_proxy(width: u32, height: u32, duration: Micros) -> bool {
    width.min(height) > LARGE_SIDE || duration.0 >= LONG.0
}

/// The file name the proxy of `source` takes, or `None` when the file
/// cannot be read.
#[must_use]
pub fn proxy_name(source: &Path) -> Option<String> {
    let canonical = source.canonicalize().ok()?;
    let meta = std::fs::metadata(&canonical).ok()?;
    let changed = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    let seed = format!("{}\n{}\n{changed}", canonical.display(), meta.len());
    // FNV-1a, the same on every run.
    let key = seed.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    Some(format!("{key:016x}-{PROXY_SIDE}.mp4"))
}

/// A draft that plays all of `source`, `width` by `height` and `duration`
/// long at `fps`, on a canvas its own size: what a proxy is drawn from.
#[must_use]
pub fn proxy_draft(source: &Path, width: u32, height: u32, duration: Micros, fps: f64) -> Draft {
    let mut draft = Draft::empty("proxy");
    let width = i32::try_from(width).unwrap_or(i32::MAX);
    let height = i32::try_from(height).unwrap_or(i32::MAX);
    draft.canvas_config.width = width;
    draft.canvas_config.height = height;
    draft.fps = fps;
    let mut movie = MaterialVideo::new("source");
    movie.kind = "video".to_owned();
    movie.path = source.to_string_lossy().into_owned();
    movie.width = width;
    movie.height = height;
    movie.duration = duration;
    draft.materials.videos.push(movie);
    let mut clip = Segment::new("source", "source", TimeRange::new(0, duration.0));
    clip.source_range = Some(TimeRange::new(0, duration.0));
    let mut track = Track::main("source");
    track.segments.push(clip);
    draft.tracks = vec![track];
    draft.duration = duration;
    draft
}

/// What a proxy of `draft`, made by [`proxy_draft`], writes.
#[must_use]
pub fn proxy_request(draft: &Draft) -> ExportRequest {
    render_request(draft, TimeRange::new(0, draft.duration.0), PROXY_SIDE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_and_long_movies_want_a_proxy() {
        assert!(wants_proxy(3840, 2160, Micros(1_000_000)));
        assert!(wants_proxy(2160, 3840, Micros(1_000_000)));
        assert!(!wants_proxy(1920, 1080, Micros(60_000_000)));
        assert!(wants_proxy(1280, 720, LONG));
    }

    #[test]
    fn the_name_follows_the_file_as_it_is() {
        let folder = std::env::temp_dir().join(format!("cutlass-proxy-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let file = folder.join("clip.mp4");
        std::fs::write(&file, b"one").unwrap();
        let first = proxy_name(&file).unwrap();
        assert_eq!(proxy_name(&file), Some(first.clone()));
        assert!(first.ends_with("-480.mp4"));
        std::fs::write(&file, b"longer").unwrap();
        assert_ne!(proxy_name(&file), Some(first));
        assert_eq!(proxy_name(&folder.join("gone.mp4")), None);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_proxy_draft_plays_the_whole_source_on_its_own_canvas() {
        let draft = proxy_draft(
            Path::new("/m/clip.mp4"),
            3840,
            2160,
            Micros(5_000_000),
            25.0,
        );
        assert_eq!(
            (draft.canvas_config.width, draft.canvas_config.height),
            (3840, 2160)
        );
        assert_eq!(draft.materials.videos[0].path, "/m/clip.mp4");
        assert_eq!(draft.tracks[0].segments[0].target.duration.0, 5_000_000);
        let request = proxy_request(&draft);
        assert_eq!(request.short_side, PROXY_SIDE);
        assert!(request.audio.is_none());
        assert_eq!(request.range, TimeRange::new(0, 5_000_000));
        assert!(crate::validate(&request).is_ok());
    }
}
