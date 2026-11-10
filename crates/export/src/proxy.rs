//! Proxies: a smaller copy of a source movie that the preview plays in its
//! place, so large clips scrub and play smoothly.
//!
//! A proxy is named by the source file as it is on disk, its path, size and
//! time of change, so a changed file is not shown through a stale proxy. It
//! has pictures only, at the source's own aspect and rate; the sound still
//! comes from the source. Export never reads a proxy.

use std::path::Path;

use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange, Track};

use crate::ExportRequest;
use crate::prerender::render_request;

/// The short sides a proxy can be drawn at, largest first.
pub const PROXY_SIDES: [u32; 2] = crate::config::PROXY_SHORT_SIDES;

/// The short side a proxy is drawn at when nothing else is chosen.
pub const DEFAULT_PROXY_SIDE: u32 = PROXY_SIDES[0];

/// `side` when it is one of [`PROXY_SIDES`], and the default otherwise, so
/// a stored size this build does not draw is not taken.
#[must_use]
pub fn proxy_side(side: u32) -> u32 {
    if PROXY_SIDES.contains(&side) {
        side
    } else {
        DEFAULT_PROXY_SIDE
    }
}

/// Whether a movie of `width` by `height` is larger than a proxy with the
/// short side `side`, so its proxy is smaller than it.
#[must_use]
pub fn wants_proxy(width: u32, height: u32, side: u32) -> bool {
    width.min(height) > side
}

/// The bits a second a proxy with the short side `side` is coded at: small
/// enough to read fast, and enough for a viewer.
#[must_use]
pub fn proxy_bitrate(side: u32) -> u32 {
    if side > 540 { 4_000_000 } else { 2_500_000 }
}

/// The file name the proxy of `source` with the short side `side` takes,
/// or `None` when the file cannot be read.
#[must_use]
pub fn proxy_name(source: &Path, side: u32) -> Option<String> {
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
    Some(format!("{key:016x}-{side}.mp4"))
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

/// What a proxy of `draft`, made by [`proxy_draft`], writes with the short
/// side `side`.
#[must_use]
pub fn proxy_request(draft: &Draft, side: u32) -> ExportRequest {
    ExportRequest {
        bitrate: proxy_bitrate(side),
        ..render_request(draft, TimeRange::new(0, draft.duration.0), side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_movie_larger_than_the_proxy_wants_one() {
        assert!(wants_proxy(3840, 2160, 720));
        assert!(wants_proxy(2160, 3840, 720));
        assert!(wants_proxy(1920, 1080, 720));
        assert!(!wants_proxy(1280, 720, 720));
        assert!(wants_proxy(1280, 720, 540));
        assert!(!wants_proxy(960, 540, 540));
        assert!(!wants_proxy(0, 0, 540));
    }

    #[test]
    fn a_size_this_build_does_not_draw_becomes_the_default() {
        assert_eq!(PROXY_SIDES, [720, 540]);
        for side in PROXY_SIDES {
            assert_eq!(proxy_side(side), side);
        }
        for side in [0, 480, 1080, u32::MAX] {
            assert_eq!(proxy_side(side), DEFAULT_PROXY_SIDE);
        }
    }

    #[test]
    fn the_name_follows_the_file_as_it_is() {
        let folder = std::env::temp_dir().join(format!("cutlass-proxy-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let file = folder.join("clip.mp4");
        std::fs::write(&file, b"one").unwrap();
        let first = proxy_name(&file, 720).unwrap();
        assert_eq!(proxy_name(&file, 720), Some(first.clone()));
        assert!(first.ends_with("-720.mp4"));
        let smaller = proxy_name(&file, 540).unwrap();
        assert!(smaller.ends_with("-540.mp4"));
        assert_eq!(smaller[..16], first[..16], "the same source");
        std::fs::write(&file, b"longer").unwrap();
        assert_ne!(proxy_name(&file, 720), Some(first));
        assert_eq!(proxy_name(&folder.join("gone.mp4"), 720), None);
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
        for side in PROXY_SIDES {
            let request = proxy_request(&draft, side);
            assert_eq!(request.short_side, side);
            assert!(request.audio.is_none());
            assert_eq!(request.range, TimeRange::new(0, 5_000_000));
            assert!(crate::validate(&request).is_ok(), "{side}");
            // Well under an export of that size, so the file reads fast.
            assert!(request.bitrate < crate::video_bitrate(side));
        }
        assert!(proxy_bitrate(540) < proxy_bitrate(720));
    }
}
