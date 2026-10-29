//! Playback and export do not share a clock.
//!
//! A cache miss does not move the audio clock backward and does not stop it.
//! Playback drops a picture that is ready only after the audio clock has
//! passed the request. Export waits: a late picture is still the frame.

use cutlass_model::Micros;
use cutlass_picture::{CpuBytes, Picture, Sample};

use crate::error::EngineError;

/// Which clock is asking for the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockKind {
    /// Drop a picture that arrives after the audio clock passed the request.
    Playback,
    /// Wait for the frame. A late arrival is still delivered.
    Export,
}

/// Audio time, advanced in sample frames, independent of picture arrival.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaClock {
    kind: ClockKind,
    audio: Micros,
}

impl MediaClock {
    #[must_use]
    pub fn new(kind: ClockKind) -> Self {
        Self {
            kind,
            audio: Micros(0),
        }
    }

    #[must_use]
    pub fn kind(&self) -> ClockKind {
        self.kind
    }

    #[must_use]
    pub fn audio_time(&self) -> Micros {
        self.audio
    }

    /// Moves the audio clock forward. A missing picture does not stop this.
    pub fn advance_audio(&mut self, frames: u32, rate: u32) -> Result<Micros, EngineError> {
        if rate == 0 {
            return Err(EngineError::NoClock);
        }
        let step = i128::from(frames) * 1_000_000 / i128::from(rate);
        let step = i64::try_from(step).unwrap_or(i64::MAX);
        self.audio.0 = self.audio.0.saturating_add(step);
        Ok(self.audio)
    }

    /// Playback returns [`Sample::Miss`] when `ready_at` is after `requested`.
    /// Export returns the picture either way.
    #[must_use]
    pub fn accept_picture(&self, requested: Micros, ready_at: Micros, picture: Picture) -> Sample {
        if self.kind == ClockKind::Playback && ready_at.0 > requested.0 {
            return Sample::Miss;
        }
        Sample::Ready(picture)
    }

    /// Playback does not call `decode` on a miss. Export does, and uses the picture.
    pub fn take<F>(&self, cached: Option<Picture>, decode: F) -> Result<Sample, EngineError>
    where
        F: FnOnce() -> Result<Picture, EngineError>,
    {
        if let Some(picture) = cached {
            return Ok(Sample::Ready(picture));
        }
        match self.kind {
            ClockKind::Playback => Ok(Sample::Miss),
            ClockKind::Export => decode().map(Sample::Ready),
        }
    }

    /// Keeps a decoded RGBA picture for export, including one that arrived late.
    ///
    /// A playback clock does not use this path. A cache miss still runs the
    /// decode that produced `rgba`. The bytes are returned only when the
    /// export clock accepts the picture.
    pub fn export_rgba(
        &self,
        requested: Micros,
        ready_at: Micros,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> Result<Vec<u8>, EngineError> {
        if self.kind != ClockKind::Export {
            return Err(EngineError::NotReady);
        }
        let picture = Picture::Cpu(CpuBytes::new(width, height, rgba).map_err(|_| {
            EngineError::ColorMiss {
                layout: "rgba".to_owned(),
            }
        })?);
        let produced = self.take(None, || Ok(picture))?;
        let Sample::Ready(picture) = produced else {
            return Err(EngineError::NotReady);
        };
        match self.accept_picture(requested, ready_at, picture) {
            Sample::Ready(Picture::Cpu(cpu)) => Ok(cpu.into_rgba()),
            Sample::Ready(Picture::Platform(_)) => Err(EngineError::ColorMiss {
                layout: "platform".to_owned(),
            }),
            Sample::Miss => Err(EngineError::NotReady),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_gpu::Extent;
    use cutlass_picture::{CpuBytes, Picture};

    fn pixel() -> Picture {
        Picture::Cpu(CpuBytes::new(1, 1, vec![1, 2, 3, 4]).expect("pixel"))
    }

    #[test]
    fn a_miss_does_not_stall_the_audio_clock() {
        let mut clock = MediaClock::new(ClockKind::Playback);
        let mut decodes = 0_u32;
        let sample = clock
            .take(None, || {
                decodes += 1;
                Ok(pixel())
            })
            .expect("miss");
        assert_eq!(sample, Sample::Miss);
        assert_eq!(decodes, 0);
        let time = clock.advance_audio(48, 48_000).expect("audio");
        assert_eq!(time, Micros(1_000));
        assert_eq!(clock.audio_time(), Micros(1_000));
    }

    #[test]
    fn playback_drops_a_late_picture_and_export_waits() {
        let playback = MediaClock::new(ClockKind::Playback);
        let late = playback.accept_picture(Micros(0), Micros(1), pixel());
        assert_eq!(late, Sample::Miss);
        let on_time = playback.accept_picture(Micros(10), Micros(10), pixel());
        assert!(matches!(on_time, Sample::Ready(_)));

        let export = MediaClock::new(ClockKind::Export);
        let waited = export.accept_picture(Micros(0), Micros(50), pixel());
        assert!(matches!(waited, Sample::Ready(_)));
        let mut decodes = 0_u32;
        let sample = export
            .take(None, || {
                decodes += 1;
                Ok(pixel())
            })
            .expect("wait");
        assert_eq!(decodes, 1);
        assert!(matches!(sample, Sample::Ready(_)));
        let _ = Extent {
            width: 1,
            height: 1,
        };
    }

    #[test]
    fn export_keeps_a_late_rgba_picture_and_playback_does_not_take_it() {
        let export = MediaClock::new(ClockKind::Export);
        let kept = export
            .export_rgba(Micros(0), Micros(50), 1, 1, vec![9, 8, 7, 6])
            .expect("late");
        assert_eq!(kept, vec![9, 8, 7, 6]);

        let playback = MediaClock::new(ClockKind::Playback);
        assert_eq!(
            playback.export_rgba(Micros(0), Micros(0), 1, 1, vec![9, 8, 7, 6]),
            Err(EngineError::NotReady)
        );
        assert_eq!(
            export.export_rgba(Micros(0), Micros(0), 1, 1, vec![1, 2, 3]),
            Err(EngineError::ColorMiss {
                layout: "rgba".to_owned(),
            })
        );
    }
}
