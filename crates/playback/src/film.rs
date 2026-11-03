//! A clip's filmstrip: which picture each tile shows.
//!
//! The editor places the tiles. This maps each tile's timeline time through
//! the clip's speed to a source frame, and keys it by the clip's file. A
//! still shows its one picture in every tile. The pictures come from a
//! [`Thumbnails`] worker; a tile whose picture is not stored yet has none,
//! and the caller asks for its key.

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
    pub thumb: Option<Arc<Thumb>>,
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
    let mut cells = Vec::new();
    for tile in film_tiles(clip_width, height * aspect, visible, pixel_size) {
        let time = Micros(segment.target.start.0.saturating_add(tile.offset.0));
        let Ok(source) = cutlass_engine::map_timeline(&map, time) else {
            continue;
        };
        let source = if still {
            Micros(0)
        } else {
            rate.frame_index(source)
                .and_then(|frame| rate.frame_start(frame))
                .unwrap_or(source)
        };
        let key = ThumbKey {
            path: path.clone(),
            time: source,
        };
        let thumb = thumbnails.get(&key);
        cells.push(FilmCell { tile, key, thumb });
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

    use super::{film_cells, media_file};

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
