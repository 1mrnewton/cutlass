//! A clip's filmstrip: which picture each tile shows.
//!
//! The editor places the tiles. This maps each tile's timeline time through
//! the clip's speed to a source frame, and keys it by the clip's file. A
//! still shows its one picture in every tile. The pictures come from a
//! [`Thumbnails`] worker; a tile whose picture is not stored yet borrows
//! the stored one of its file nearest in time, and the caller asks for its
//! key.
//!
//! A tile that covers an eighth of a second or more of its file shows a
//! frame from a grid of times instead of the one at its exact left edge:
//! the largest step of an eighth of a second doubled that fits in what the
//! tile covers. Each step holds every time of the steps above it, so
//! zooming, trimming and a tile a little wider or narrower ask for frames
//! that are already made, here or in the cache on disk.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_editor::{FilmTile, film_tiles};
use cutlass_engine::{Thumb, ThumbKey, Thumbnails};
use cutlass_model::{Draft, FrameRate, MaterialRef, MetaType, Micros, SegmentId};

/// One filmstrip tile of a clip, and its picture once stored.
#[derive(Clone, Debug)]
pub struct FilmCell {
    pub tile: FilmTile,
    pub key: ThumbKey,
    /// The tile's own picture, or while that is on its way the stored one
    /// of its file nearest in time.
    pub thumb: Option<Arc<Thumb>>,
    /// `thumb` is not this tile's own, so its key is still to be asked for.
    pub waiting: bool,
}

/// The finest step of the grid of times a tile's frame is taken from.
const FILM_STEP: i64 = 125_000;

/// The source time a tile shows for the frame at `source`, when the tile
/// covers `span` microseconds of its file: `source` moved back onto the
/// grid whose step is the largest doubling of [`FILM_STEP`] no longer than
/// `span`, or on to the grid time just ahead when it is all but there. A
/// tile that covers less than the finest step shows `source`.
fn film_time(source: Micros, span: i64) -> Micros {
    if span < FILM_STEP {
        return source;
    }
    let doublings = (span / FILM_STEP).ilog2();
    let step = FILM_STEP.checked_shl(doublings).unwrap_or(i64::MAX);
    // A tile's edge falls a hair short of a grid time as often as on it,
    // pixels being fractions; within a hundredth of a step it is on it.
    let near = source.0.saturating_add(step / 128);
    Micros(near - near.rem_euclid(step))
}

/// The cells of clip `id`, `clip_width` pixels wide and `height` tall, that
/// meet `visible`, pixels from the clip's left edge. `pixel_size` is pixels
/// per microsecond. A tile is as wide as the file's upright picture at that
/// height. `project` resolves a file stored relative to the project. A clip
/// with no picture file, or an unknown id, has no filmstrip.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn film_cells(
    draft: &Draft,
    project: Option<&Path>,
    thumbnails: &Thumbnails,
    id: &SegmentId,
    clip_width: f64,
    height: f64,
    visible: Range<f64>,
    pixel_size: f64,
) -> Vec<FilmCell> {
    let Some(segment) = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == id)
    else {
        return Vec::new();
    };
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return Vec::new();
    };
    let Some(path) = media_file(project, cutlass_engine::media_path(draft, segment)) else {
        return Vec::new();
    };
    let Ok(map) = cutlass_engine::placement_of(draft, segment) else {
        return Vec::new();
    };
    let still = video.kind == MetaType::Photo.as_str();
    // The file's upright shape once the worker has opened it; the stored
    // material size until then.
    let (shape_w, shape_h) = thumbnails.shape(&path).unwrap_or((
        u32::try_from(video.width).unwrap_or(0),
        u32::try_from(video.height).unwrap_or(0),
    ));
    let aspect = if shape_w > 0 && shape_h > 0 {
        f64::from(shape_w) / f64::from(shape_h)
    } else {
        16.0 / 9.0
    };
    let rate = draft.frame_rate().unwrap_or(FrameRate::FPS_30);
    // What a whole tile covers of the file, at the clip's speed.
    let tile_width = height * aspect;
    let speed = if segment.speed.is_finite() && segment.speed > 0.0 {
        segment.speed
    } else {
        1.0
    };
    // A point more, so a tile sized to a step exactly is not counted short.
    let span = (tile_width / pixel_size * speed).round() + 1.0;
    let span = if span.is_finite() && span >= 0.0 && span < i64::MAX as f64 {
        span as i64
    } else {
        0
    };
    let mut cells = Vec::new();
    for tile in film_tiles(clip_width, tile_width, visible, pixel_size) {
        let time = Micros(segment.target.start.0.saturating_add(tile.offset.0));
        let Ok(source) = cutlass_engine::map_timeline(&map, time) else {
            continue;
        };
        let source = if still {
            Micros(0)
        } else {
            let source = film_time(source, span);
            rate.frame_index(source)
                .and_then(|frame| rate.frame_start(frame))
                .unwrap_or(source)
        };
        let key = ThumbKey {
            path: path.clone(),
            time: source,
        };
        let own = thumbnails.get(&key);
        let waiting = own.is_none();
        let thumb = own.or_else(|| thumbnails.nearest(&key));
        cells.push(FilmCell {
            tile,
            key,
            thumb,
            waiting,
        });
    }
    cells
}

/// The file a stored media path names: itself when absolute, else inside
/// `project`.
#[must_use]
pub fn media_file(project: Option<&Path>, stored: Option<&str>) -> Option<PathBuf> {
    let stored = stored?;
    if Path::new(stored).is_absolute() {
        return Some(PathBuf::from(stored));
    }
    cutlass_engine::media_in_project(project?, stored)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_engine::{ReaderFactory, Thumbnails};
    use cutlass_model::{MaterialVideo, MetaType, Micros, Segment, SegmentId, TimeRange};

    use super::{FILM_STEP, film_cells, film_time, media_file};

    fn pattern() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/fixtures/pattern.png")
    }

    #[test]
    fn a_still_shows_its_one_picture_in_every_tile() {
        let mut draft = crate::color_session(4_000_000, 4_000_000).draft().clone();
        draft.materials.videos.push(MaterialVideo {
            id: "still".into(),
            kind: MetaType::Photo.as_str().to_owned(),
            path: pattern().display().to_string(),
            width: 64,
            height: 32,
            ..MaterialVideo::default()
        });
        let mut segment = Segment::new("photo", "still", TimeRange::new(0, 4_000_000));
        segment.source_range = Some(TimeRange::new(0, 4_000_000));
        draft.tracks[0].segments = vec![segment];
        let thumbnails = Thumbnails::start(ReaderFactory::default(), || {});
        // 1 pixel per 10 ms: the clip is 400 wide, a tile 2:1 at 40 tall.
        let cells = film_cells(
            &draft,
            None,
            &thumbnails,
            &SegmentId::new("photo"),
            400.0,
            40.0,
            0.0..400.0,
            0.0001,
        );
        assert_eq!(cells.len(), 5);
        assert!(cells.iter().all(|cell| cell.key.time == Micros(0)));
        assert!(cells.iter().all(|cell| cell.key.path == pattern()));
        assert_eq!(cells[1].tile.left, 80.0);
    }

    #[test]
    fn a_tile_shows_a_frame_from_a_grid_that_zooming_keeps() {
        // A tile that covers less than the finest step shows its own frame.
        assert_eq!(
            film_time(Micros(1_234_567), FILM_STEP - 1),
            Micros(1_234_567)
        );
        // From there the step is the finest doubled as often as fits.
        assert_eq!(film_time(Micros(1_234_567), 125_000), Micros(1_125_000));
        assert_eq!(film_time(Micros(1_234_567), 249_999), Micros(1_125_000));
        // A hair short of a grid time is that time, not a step before it.
        assert_eq!(film_time(Micros(3_999_999), 4_000_000), Micros(4_000_000));
        assert_eq!(film_time(Micros(3_900_000), 4_000_000), Micros(0));
        assert_eq!(film_time(Micros(1_234_567), 250_000), Micros(1_000_000));
        assert_eq!(film_time(Micros(1_234_567), 1_900_000), Micros(1_000_000));
        assert_eq!(film_time(Micros(1_234_567), 2_000_000), Micros(0));
        assert_eq!(film_time(Micros(0), 2_000_000), Micros(0));
        // A time before the file's start goes back, not toward zero.
        assert_eq!(film_time(Micros(-100_000), 250_000), Micros(-250_000));
        // A span no clip has does not overflow.
        assert_eq!(film_time(Micros(5), i64::MAX), Micros(0));

        // Every time of a coarse grid is a time of each finer one, so the
        // frames made at one zoom are the frames wanted at the next.
        for doubling in 0..12 {
            let coarse = FILM_STEP << (doubling + 1);
            let fine = FILM_STEP << doubling;
            for tile in 0..40 {
                let at = film_time(Micros(tile * coarse + coarse / 3), coarse);
                assert_eq!(film_time(at, fine), at, "{coarse} over {fine}");
            }
        }
    }

    #[test]
    fn a_movies_tiles_keep_their_frames_through_a_zoom_and_a_trim() {
        let mut draft = crate::color_session(60_000_000, 60_000_000).draft().clone();
        draft.materials.videos.push(MaterialVideo {
            id: "movie".into(),
            kind: MetaType::Video.as_str().to_owned(),
            path: "/movie.mp4".to_owned(),
            width: 1920,
            height: 1080,
            duration: Micros(60_000_000),
            ..MaterialVideo::default()
        });
        let mut segment = Segment::new("clip", "movie", TimeRange::new(0, 60_000_000));
        segment.source_range = Some(TimeRange::new(0, 60_000_000));
        draft.tracks[0].segments = vec![segment];
        let thumbnails = Thumbnails::start(ReaderFactory::default(), || {});
        let times = |draft: &cutlass_model::Draft, pixel_size: f64| -> Vec<i64> {
            let clip_width = 60_000_000.0 * pixel_size;
            film_cells(
                draft,
                None,
                &thumbnails,
                &SegmentId::new("clip"),
                clip_width,
                45.0,
                0.0..clip_width,
                pixel_size,
            )
            .into_iter()
            .map(|cell| cell.key.time.0)
            .collect()
        };
        // An 80-point tile covers 4 s here, so frames come every 4 s.
        let wide = times(&draft, 0.000_02);
        assert_eq!(&wide[..4], [0, 4_000_000, 8_000_000, 12_000_000]);
        // Zoomed in a little, a tile covers 3.2 s: the step is 2 s, and
        // every frame wanted is on that grid.
        let closer = times(&draft, 0.000_025);
        assert!(
            closer.iter().all(|time| time % 2_000_000 == 0),
            "{closer:?}"
        );
        // Zoomed to twice the size, half the frames were made already.
        let twice = times(&draft, 0.000_04);
        let made = twice.iter().filter(|time| wide.contains(time)).count();
        assert!(made * 2 >= twice.len() - 1, "{made} of {}", twice.len());

        // The clip's start trimmed by a third of a second: the tiles move,
        // and still show the frames they had.
        let mut trimmed = draft.clone();
        trimmed.tracks[0].segments[0].target = TimeRange::new(0, 59_650_000);
        trimmed.tracks[0].segments[0].source_range = Some(TimeRange::new(350_000, 59_650_000));
        let after = times(&trimmed, 0.000_02);
        let kept = after.iter().filter(|time| wide.contains(time)).count();
        assert!(
            kept + 1 >= after.len(),
            "{kept} of {}: {after:?}",
            after.len()
        );
    }

    #[test]
    fn a_colour_clip_or_an_unknown_id_has_no_filmstrip() {
        let draft = crate::color_session(1_000_000, 1_000_000).draft().clone();
        let thumbnails = Thumbnails::start(ReaderFactory::default(), || {});
        for id in ["segment", "missing"] {
            let cells = film_cells(
                &draft,
                None,
                &thumbnails,
                &SegmentId::new(id),
                400.0,
                40.0,
                0.0..400.0,
                0.0001,
            );
            assert!(cells.is_empty(), "{id}");
        }
    }

    #[test]
    fn a_stored_path_stays_inside_the_project() {
        let project = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(
            media_file(None, Some("/absolute/clip.mov")),
            Some(PathBuf::from("/absolute/clip.mov"))
        );
        assert_eq!(media_file(None, Some("relative.mov")), None);
        assert_eq!(media_file(Some(project), None), None);
        assert_eq!(media_file(Some(project), Some("../escape.mov")), None);
        assert_eq!(
            media_file(Some(project), Some("Cargo.toml")),
            Some(project.canonicalize().unwrap().join("Cargo.toml"))
        );
    }
}
