//! Import a file the phone picked, and add a library item to the end of its lane.
//!
//! The file is probed by `cutlass-probe`, the same way the desktop probes
//! before import. It is copied into the project's `Resources` folder and the
//! draft stores that relative path.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cutlass_editor::{AppendLane, append_placement, material_span};
use cutlass_model::{
    MaterialAudio, MaterialId, MaterialVideo, MetaType, Segment, TimeRange, TrackAttribute,
    TrackType,
};
use cutlass_probe::Probed;
use cutlass_session::{Draft, ImportedMedium, Micros, Request, SegmentId, TrackId};

use crate::error::FfiError;

fn duration(probed: &Probed) -> i64 {
    match probed {
        Probed::Video { duration, .. } | Probed::Audio { duration } => *duration,
    }
}

/// The import request for `probed`, stored at `path` inside the project.
pub(crate) fn imported(id: &str, path: String, probed: Probed) -> ImportedMedium {
    match probed {
        Probed::Video {
            meta,
            width,
            height,
            duration,
        } => {
            let mut material = MaterialVideo::new(id);
            material.kind = meta.as_str().to_owned();
            material.path = path;
            material.duration = Micros(duration);
            material.width = width;
            material.height = height;
            ImportedMedium::Video(Box::new(material))
        }
        Probed::Audio { duration } => ImportedMedium::Audio(Box::new(MaterialAudio {
            id: MaterialId::new(id),
            kind: MetaType::Music.as_str().to_owned(),
            path,
            duration: Micros(duration),
            ..MaterialAudio::default()
        })),
    }
}

/// Refuse a probe the draft cannot place.
pub(crate) fn check(probed: &Probed) -> Result<(), FfiError> {
    if duration(probed) <= 0 {
        return Err(FfiError::Refused {
            reason: "the media has no duration".to_owned(),
        });
    }
    if let Probed::Video { width, height, .. } = probed
        && (*width <= 0 || *height <= 0)
    {
        return Err(FfiError::Refused {
            reason: "the picture has no size".to_owned(),
        });
    }
    Ok(())
}

/// Copy `source` into `root/Resources` under a name not already taken.
/// Returns the stored relative path and the copy.
pub(crate) fn copy_in(root: &Path, source: &Path) -> Result<(String, PathBuf), FfiError> {
    let io = |error: std::io::Error| FfiError::Refused {
        reason: error.to_string(),
    };
    if !source.is_file() {
        return Err(FfiError::Refused {
            reason: format!("{} is not a file", source.display()),
        });
    }
    let name = source
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| FfiError::Refused {
            reason: "the file has no name".to_owned(),
        })?;
    let resources = root.join("Resources");
    std::fs::create_dir_all(&resources).map_err(io)?;
    let (stem, extension) = match name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() => (stem, Some(extension)),
        _ => (name, None),
    };
    let mut candidate = name.to_owned();
    let mut n = 1_u32;
    while resources.join(&candidate).exists() {
        n = n.saturating_add(1);
        candidate = match extension {
            Some(extension) => format!("{stem} {n}.{extension}"),
            None => format!("{stem} {n}"),
        };
    }
    let destination = resources.join(&candidate);
    std::fs::copy(source, &destination).map_err(io)?;
    Ok((format!("Resources/{candidate}"), destination))
}

/// The requests that add the whole of `material_id` at the end of its lane.
pub(crate) fn append_requests(
    draft: &Draft,
    material_id: &str,
) -> Result<(Vec<Request>, SegmentId), FfiError> {
    let (kind, duration) = material_span(draft, material_id).ok_or_else(|| FfiError::Refused {
        reason: "the media is not in the library".to_owned(),
    })?;
    if duration.0 <= 0 {
        return Err(FfiError::Refused {
            reason: "the media has no duration".to_owned(),
        });
    }
    let taken = ids(draft);
    let append = append_placement(draft, kind);
    let mut requests = Vec::new();
    let track_id = match append.lane {
        AppendLane::Track(track_id) => track_id,
        AppendLane::New(kind) => {
            let track_id = TrackId::new(fresh(&taken, "track"));
            requests.push(Request::InsertTrack {
                track_id: track_id.clone(),
                kind,
            });
            track_id
        }
    };
    let segment_id = SegmentId::new(fresh(&taken, "seg"));
    let mut segment = Segment::new(
        segment_id.clone(),
        material_id,
        TimeRange::new(append.start.0, duration.0),
    );
    segment.source_range = Some(TimeRange::new(0, duration.0));
    requests.push(Request::Place {
        track_id,
        segment: Box::new(segment),
    });
    requests.extend(cutlass_session::follow_first_picture(draft, material_id));
    Ok((requests, segment_id))
}

/// Every track, segment, and material id in the draft.
pub(crate) fn ids(draft: &Draft) -> HashSet<String> {
    let mut taken = HashSet::new();
    for track in &draft.tracks {
        taken.insert(track.id.as_str().to_owned());
        for segment in &track.segments {
            taken.insert(segment.id.as_str().to_owned());
        }
    }
    for video in &draft.materials.videos {
        taken.insert(video.id.as_str().to_owned());
    }
    for audio in &draft.materials.audios {
        taken.insert(audio.id.as_str().to_owned());
    }
    for text in &draft.materials.texts {
        taken.insert(text.id.as_str().to_owned());
    }
    for effect in &draft.materials.video_effects {
        taken.insert(effect.id.as_str().to_owned());
    }
    taken
}

/// `prefix-N` for the first N not in `taken`.
/// The first unlocked `kind` row with nothing between `at` and `at +
/// length`, else a new row id, then a new clip id, and every id now taken.
pub(crate) fn room(
    draft: &Draft,
    kind: TrackType,
    at: Micros,
    length: Micros,
) -> (TrackId, SegmentId, HashSet<String>) {
    let at = Micros(at.0.max(0));
    let end = at.0.saturating_add(length.0);
    let mut taken = ids(draft);
    let free = draft.tracks.iter().find(|track| {
        track.kind == kind
            && !track.attribute.contains(TrackAttribute::LOCK)
            && track.segments.iter().all(|segment| {
                let start = segment.target.start.0;
                let stop = start.saturating_add(segment.target.duration.0);
                stop <= at.0 || start >= end
            })
    });
    let track_id = match free {
        Some(track) => track.id.clone(),
        None => TrackId::new(fresh(&taken, "track")),
    };
    taken.insert(track_id.as_str().to_owned());
    let segment_id = SegmentId::new(fresh(&taken, "seg"));
    taken.insert(segment_id.as_str().to_owned());
    (track_id, segment_id, taken)
}

pub(crate) fn fresh(taken: &HashSet<String>, prefix: &str) -> String {
    (1_u64..)
        .map(|n| format!("{prefix}-{n}"))
        .find(|id| !taken.contains(id))
        .unwrap_or_else(|| format!("{prefix}-0"))
}
