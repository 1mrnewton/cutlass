//! Import a file the phone picked, and add a library item to the end of its lane.
//!
//! The phone shell probes the file with its own media framework, the way
//! the desktop probes before import. The file is copied into the project's
//! `Resources` folder and the draft stores that relative path.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cutlass_editor::{AppendLane, append_placement, material_span};
use cutlass_model::{MaterialAudio, MaterialId, MaterialVideo, MetaType, Segment, TimeRange};
use cutlass_session::{Draft, ImportedMedium, Micros, Request, SegmentId, TrackId};

use crate::error::FfiError;

/// What the phone read from a file before import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probed {
    /// A movie, or a still when `still` is true. Size in pixels, upright.
    Picture {
        still: bool,
        width: i32,
        height: i32,
        duration: Micros,
    },
    /// A sound file.
    Sound { duration: Micros },
}

impl Probed {
    fn duration(self) -> Micros {
        match self {
            Self::Picture { duration, .. } | Self::Sound { duration } => duration,
        }
    }
}

/// The import request for `probed`, stored at `path` inside the project.
pub(crate) fn imported(id: &str, path: String, probed: Probed) -> ImportedMedium {
    match probed {
        Probed::Picture {
            still,
            width,
            height,
            duration,
        } => {
            let mut material = MaterialVideo::new(id);
            let meta = if still {
                MetaType::Photo
            } else {
                MetaType::Video
            };
            material.kind = meta.as_str().to_owned();
            material.path = path;
            material.duration = duration;
            material.width = width;
            material.height = height;
            ImportedMedium::Video(Box::new(material))
        }
        Probed::Sound { duration } => ImportedMedium::Audio(Box::new(MaterialAudio {
            id: MaterialId::new(id),
            kind: MetaType::Music.as_str().to_owned(),
            path,
            duration,
            ..MaterialAudio::default()
        })),
    }
}

/// Refuse a probe the draft cannot place.
pub(crate) fn check(probed: Probed) -> Result<(), FfiError> {
    if probed.duration().0 <= 0 {
        return Err(FfiError::Refused {
            reason: "the media has no duration".to_owned(),
        });
    }
    if let Probed::Picture { width, height, .. } = probed
        && (width <= 0 || height <= 0)
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
    taken
}

/// `prefix-N` for the first N not in `taken`.
pub(crate) fn fresh(taken: &HashSet<String>, prefix: &str) -> String {
    (1_u64..)
        .map(|n| format!("{prefix}-{n}"))
        .find(|id| !taken.contains(id))
        .unwrap_or_else(|| format!("{prefix}-0"))
}
