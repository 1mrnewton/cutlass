//! What the phone timeline paints besides the clips: the ruler, the zoom
//! limits, each clip's filmstrip, and its waveform.
//!
//! The phone works in points and microseconds. A pixel size is points per
//! microsecond. Filmstrip pictures come from a worker that starts with the
//! first ask; a tile whose picture is not stored yet says so, and the phone
//! asks again a moment later. Waveforms come from a second worker the same
//! way, through the code the desktop draws its waves with.

use std::collections::HashMap;
use std::path::Path;

use cutlass_editor::{Zoom, ZoomBounds, ruler_ticks};
use cutlass_eval::ReaderFactory;
use cutlass_model::{Draft, FrameRate, TrackType};
use cutlass_playback::{ThumbKey, Thumbnails, Waveforms, clip_wave, film_cells};
use cutlass_session::{Micros, SegmentId};

use crate::preview::Picture;

/// One ruler tick. A major tick carries its label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneTick {
    pub time: Micros,
    pub major: bool,
    pub label: Option<String>,
}

/// One filmstrip tile, in points from the clip's left edge. `key` names the
/// picture for [`Strip::thumb`].
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneTile {
    pub left: f64,
    pub width: f64,
    pub key: String,
    pub ready: bool,
}

/// The tiles of one clip.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneFilm {
    pub id: SegmentId,
    pub tiles: Vec<PhoneTile>,
}

/// One clip's sound over the columns asked for: a level, 0 to 1, per
/// point from `first`, points from the clip's left edge, at the clip's
/// volume. `ready` is false while the file is still being read, so the
/// phone asks again.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneWave {
    pub id: SegmentId,
    pub first: u32,
    pub levels: Vec<f32>,
    /// The sound of a video clip, drawn in a band over its picture's foot.
    pub under_picture: bool,
    pub muted: bool,
    pub ready: bool,
}

/// How much closer a phone's labels sit than the desktop's. The editor
/// picks its step for a wide window; a phone strip is about a quarter as wide.
pub const LABEL_DENSITY: f64 = 4.0;

/// The ticks from `from` through `to` at `pixel_size`, at the draft's rate,
/// with labels [`LABEL_DENSITY`] times closer than the desktop's.
#[must_use]
pub fn ruler(draft: &Draft, pixel_size: f64, from: Micros, to: Micros) -> Vec<PhoneTick> {
    let rate = draft.frame_rate().unwrap_or(FrameRate::FPS_30);
    ruler_ticks(rate, false, pixel_size * LABEL_DENSITY, from, to)
        .into_iter()
        .map(|tick| PhoneTick {
            time: tick.time,
            major: tick.major,
            label: tick.label,
        })
        .collect()
}

/// `pixel_size` times `factor`, held between the finest step the draft's
/// rate draws and a quarter of `lane_width` for the whole cut. A factor or
/// size that is not finite and positive keeps `pixel_size`.
#[must_use]
pub fn zoom(draft: &Draft, pixel_size: f64, factor: f64, lane_width: f64) -> f64 {
    let bounds = ZoomBounds {
        lane_width,
        duration: draft.duration,
        rate: draft.frame_rate().unwrap_or(FrameRate::FPS_30),
    };
    Zoom::new(pixel_size)
        .and_then(|zoom| zoom.scale(factor, bounds))
        .map_or(pixel_size, Zoom::pixel_size)
}

/// The filmstrip worker and the picture names handed out.
#[derive(Default)]
pub struct Strip {
    thumbnails: Option<Thumbnails>,
    keys: HashMap<String, ThumbKey>,
    waveforms: Option<Waveforms>,
}

impl Strip {
    /// The tiles of every picture clip that meets `from` to `to`, each tile
    /// `height` points tall. Pictures not stored yet are asked for, the
    /// nearest first.
    #[allow(clippy::too_many_arguments)]
    pub fn film(
        &mut self,
        draft: &Draft,
        project: &Path,
        readers: &ReaderFactory,
        from: Micros,
        to: Micros,
        pixel_size: f64,
        height: f64,
    ) -> Vec<PhoneFilm> {
        if !pixel_size.is_finite() || pixel_size <= 0.0 || to.0 < from.0 {
            return Vec::new();
        }
        let thumbnails = self
            .thumbnails
            .get_or_insert_with(|| Thumbnails::start(readers.clone(), || {}));
        self.keys.clear();
        let mut asks = Vec::new();
        let mut films = Vec::new();
        for track in draft
            .tracks
            .iter()
            .filter(|track| track.kind == TrackType::Video)
        {
            for segment in &track.segments {
                let start = segment.target.start.0;
                let end = start.saturating_add(segment.target.duration.0);
                if end < from.0 || start > to.0 {
                    continue;
                }
                let points = |time: i64| (time - start) as f64 * pixel_size;
                let cells = film_cells(
                    draft,
                    Some(project),
                    thumbnails,
                    &segment.id,
                    points(end),
                    height,
                    points(from.0)..points(to.0),
                    pixel_size,
                );
                if cells.is_empty() {
                    continue;
                }
                let tiles = cells
                    .into_iter()
                    .map(|cell| {
                        let name = format!("{}#{}", cell.key.path.display(), cell.key.time.0);
                        // A nearby picture stands in on the desktop, but the
                        // phone keeps each picture by name, so a tile is
                        // ready only once its own is stored.
                        if cell.waiting {
                            asks.push(cell.key.clone());
                        }
                        self.keys.insert(name.clone(), cell.key);
                        PhoneTile {
                            left: cell.tile.left,
                            width: cell.tile.width,
                            key: name,
                            ready: !cell.waiting,
                        }
                    })
                    .collect();
                films.push(PhoneFilm {
                    id: segment.id.clone(),
                    tiles,
                });
            }
        }
        thumbnails.want(asks);
        films
    }

    /// The stored picture a tile named, or `None` before it is stored or for
    /// a name the last [`Self::film`] did not hand out.
    #[must_use]
    pub fn thumb(&self, key: &str) -> Option<Picture> {
        let thumb = self.thumbnails.as_ref()?.get(self.keys.get(key)?)?;
        Some(Picture {
            width: thumb.width,
            height: thumb.height,
            rgba: thumb.rgba.clone(),
        })
    }

    /// The waveform of every clip with sound that meets `from` to `to`, one
    /// level per point at `pixel_size`, over the part inside that window.
    /// Files not read yet are asked for, and their clips say so.
    pub fn waves(
        &mut self,
        draft: &Draft,
        project: &Path,
        readers: &ReaderFactory,
        from: Micros,
        to: Micros,
        pixel_size: f64,
    ) -> Vec<PhoneWave> {
        if !pixel_size.is_finite() || pixel_size <= 0.0 || to.0 < from.0 {
            return Vec::new();
        }
        let waveforms = self
            .waveforms
            .get_or_insert_with(|| Waveforms::start(readers.clone(), || {}));
        let mut asks = Vec::new();
        let mut waves = Vec::new();
        for segment in draft.tracks.iter().flat_map(|track| &track.segments) {
            let start = segment.target.start.0;
            let end = start.saturating_add(segment.target.duration.0);
            if end < from.0 || start > to.0 {
                continue;
            }
            let points = |time: i64| (time - start) as f64 * pixel_size;
            let asked = asks.len();
            let Some(wave) = clip_wave(
                draft,
                Some(project),
                waveforms,
                &segment.id,
                pixel_size,
                points(end),
                points(from.0)..points(to.0),
                &mut asks,
            ) else {
                continue;
            };
            waves.push(PhoneWave {
                id: segment.id.clone(),
                first: wave.first,
                levels: wave.levels,
                under_picture: wave.under_picture,
                muted: wave.muted,
                ready: asks.len() == asked,
            });
        }
        waveforms.want(asks);
        waves
    }

    /// Stop decoding tiles and sound while playing, or carry on.
    pub fn set_paused(&self, paused: bool) {
        if let Some(thumbnails) = self.thumbnails.as_ref() {
            thumbnails.set_paused(paused);
        }
        if let Some(waveforms) = self.waveforms.as_ref() {
            waveforms.set_paused(paused);
        }
    }
}
