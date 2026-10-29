//! The preview surface.
//!
//! The shell presents the device. Callers that need pixels use the CPU
//! readback from [`cutlass_eval::frame`]. A cached frame is keyed by the
//! draft revision, the timeline time, and the canvas size. A new revision
//! drops the cache.

use std::collections::VecDeque;

use cutlass_eval::Snapshot;
use cutlass_gpu::Device;
use cutlass_model::{DraftId, Micros};
use cutlass_session::Session;

use crate::playhead::{PlayheadError, Readback};

/// Why a seek or a present did not return a picture.
#[derive(Debug, PartialEq, Eq)]
pub enum PresentError<E> {
    /// Eval refused the frame.
    Picture(cutlass_eval::EvalError<E>),
    /// The playhead time was refused.
    Playhead(PlayheadError),
}

impl<E: std::fmt::Display> std::fmt::Display for PresentError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Picture(error) => write!(formatter, "{error}"),
            Self::Playhead(error) => write!(formatter, "{error}"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for PresentError<E> {}

impl<E> From<PlayheadError> for PresentError<E> {
    fn from(error: PlayheadError) -> Self {
        Self::Playhead(error)
    }
}

/// How many composited readbacks the preview keeps.
///
/// A scrub looks at a handful of times. Four frames is enough to reuse the
/// last ones, and the search stays a short scan.
const CACHE_CAP: usize = 4;

/// The playhead frame was shown, or it arrived after the audio clock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Presentation {
    /// CPU readback for this time.
    Shown(Readback),
    /// The picture missed its time. The audio clock was not moved.
    Dropped { audio_time: Micros },
}

pub(crate) struct Cache {
    draft: Option<DraftId>,
    revision: Option<u64>,
    frames: VecDeque<Entry>,
}

struct Entry {
    time: i64,
    width: i32,
    height: i32,
    readback: Readback,
}

impl Cache {
    pub(crate) fn new() -> Self {
        Self {
            draft: None,
            revision: None,
            frames: VecDeque::new(),
        }
    }

    fn reset(&mut self, draft: DraftId, revision: u64) {
        self.draft = Some(draft);
        self.revision = Some(revision);
        self.frames.clear();
    }

    pub(crate) fn lookup(
        &mut self,
        draft: &DraftId,
        revision: u64,
        time: Micros,
        width: i32,
        height: i32,
    ) -> Option<Readback> {
        if self.draft.as_ref() != Some(draft) || self.revision != Some(revision) {
            self.reset(draft.clone(), revision);
            return None;
        }
        self.frames
            .iter()
            .find(|entry| entry.time == time.0 && entry.width == width && entry.height == height)
            .map(|entry| entry.readback.clone())
    }

    pub(crate) fn store(
        &mut self,
        draft: DraftId,
        revision: u64,
        time: Micros,
        width: i32,
        height: i32,
        readback: Readback,
    ) {
        if self.draft.as_ref() != Some(&draft) || self.revision != Some(revision) {
            self.reset(draft, revision);
        }
        if let Some(existing) = self
            .frames
            .iter_mut()
            .find(|entry| entry.time == time.0 && entry.width == width && entry.height == height)
        {
            existing.readback = readback;
            return;
        }
        if self.frames.len() == CACHE_CAP {
            self.frames.pop_front();
        }
        self.frames.push_back(Entry {
            time: time.0,
            width,
            height,
            readback,
        });
    }
}

pub(crate) fn evaluate<D: Device>(
    cache: &mut Cache,
    session: &Session,
    device: &mut D,
    time: Micros,
) -> Result<Readback, PresentError<D::Error>> {
    let draft = session.draft();
    let revision = session.revision();
    let width = draft.canvas_config.width;
    let height = draft.canvas_config.height;
    if let Some(hit) = cache.lookup(&draft.id, revision, time, width, height) {
        return Ok(hit);
    }
    let frame =
        cutlass_eval::frame(&Snapshot::new(draft), time, device).map_err(PresentError::Picture)?;
    let readback = Readback {
        width: frame.extent.width,
        height: frame.extent.height,
        rgba: frame.rgba,
    };
    cache.store(
        draft.id.clone(),
        revision,
        time,
        width,
        height,
        readback.clone(),
    );
    Ok(readback)
}

pub(crate) fn accept(audio_time: Micros, requested: Micros, readback: Readback) -> Presentation {
    if audio_time.0 > requested.0 {
        return Presentation::Dropped { audio_time };
    }
    Presentation::Shown(readback)
}

/// The stored duration is usable, and `time` sits on `[0, duration]`.
pub(crate) fn check_time(duration: Micros, time: Micros) -> Result<(), PlayheadError> {
    if duration.0 < 0 {
        return Err(PlayheadError::Duration);
    }
    if time.0 < 0 || time.0 > duration.0 {
        return Err(PlayheadError::Time);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, Pass, TexelSource};
    use cutlass_model::{DraftId, Micros, TrackAttribute};
    use cutlass_renderer_software::{SoftwareDevice, SoftwareError};
    use cutlass_session::Request;

    use super::*;
    use crate::color_session;
    use crate::playhead::{Player, Readback};

    #[test]
    fn the_cache_key_is_revision_time_and_canvas() {
        let mut cache = Cache::new();
        let id = DraftId::new("draft");
        let frame = Readback {
            width: 2,
            height: 1,
            rgba: vec![1, 2, 3, 4],
        };
        cache.store(id.clone(), 0, Micros(10), 2, 1, frame.clone());
        assert_eq!(
            cache.lookup(&id, 0, Micros(10), 2, 1).as_ref(),
            Some(&frame)
        );
        assert!(cache.lookup(&id, 0, Micros(11), 2, 1).is_none());
        assert!(cache.lookup(&id, 0, Micros(10), 4, 1).is_none());
        assert!(cache.lookup(&id, 1, Micros(10), 2, 1).is_none());
        let other = DraftId::new("other");
        assert!(cache.lookup(&other, 1, Micros(10), 2, 1).is_none());
    }

    struct Counting {
        inner: SoftwareDevice,
        created: u32,
    }

    impl Device for Counting {
        type Error = SoftwareError;

        fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error> {
            self.created += 1;
            self.inner.create_framebuffer(extent)
        }

        fn destroy_framebuffer(&mut self, framebuffer: Framebuffer) -> Result<(), Self::Error> {
            self.inner.destroy_framebuffer(framebuffer)
        }

        fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, Self::Error> {
            self.inner.begin(framebuffer)
        }

        fn blend(&mut self, pass: Pass, blend: Blend) -> Result<(), Self::Error> {
            self.inner.blend(pass, blend)
        }

        fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error> {
            self.inner.clear(pass, color)
        }

        fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error> {
            self.inner.draw(pass, source)
        }

        fn end(&mut self, pass: Pass) -> Result<(), Self::Error> {
            self.inner.end(pass)
        }

        fn submit(&mut self, pass: Pass) -> Result<(), Self::Error> {
            self.inner.submit(pass)
        }

        fn read_rgba(&self, framebuffer: Framebuffer) -> Result<&[u8], Self::Error> {
            self.inner.read_rgba(framebuffer)
        }
    }

    #[test]
    fn a_repeated_seek_reuses_the_cached_frame_until_the_revision_changes() {
        let mut session = color_session(1_000_000, 1_000_000);
        let mut player = Player::new();
        let mut device = Counting {
            inner: SoftwareDevice::new(),
            created: 0,
        };
        let first = player
            .seek(&session, &mut device, Micros(10))
            .expect("first");
        assert_eq!(device.created, 1);
        let second = player
            .seek(&session, &mut device, Micros(10))
            .expect("cached");
        assert_eq!(device.created, 1);
        assert_eq!(first.rgba, second.rgba);
        session
            .apply(
                0,
                &[Request::SetTrackAttribute {
                    track_id: session.draft().tracks[0].id.clone(),
                    bit: TrackAttribute::HIDE,
                    on: true,
                }],
            )
            .expect("hide");
        let hidden = player
            .seek(&session, &mut device, Micros(10))
            .expect("revision");
        assert_eq!(device.created, 2);
        assert_eq!(hidden.rgba, vec![0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
