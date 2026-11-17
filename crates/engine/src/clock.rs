//! Playback and export do not share a clock.
//!
//! Playback drops a picture that is ready only after the time it was asked
//! for. Export waits: a late picture is still the frame.

use cutlass_model::Micros;
use cutlass_picture::{CpuBytes, Picture, Sample};

use crate::error::EngineError;

/// Which clock is asking for the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockKind {
    /// Drop a picture that arrives after the time it was asked for.
    Playback,
    /// Wait for the frame. A late arrival is still delivered.
    Export,
}

/// The rule a clock applies to a picture that arrives late.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaClock {
    kind: ClockKind,
}

impl MediaClock {
    #[must_use]
    pub fn new(kind: ClockKind) -> Self {
        Self { kind }
    }

    #[must_use]
    pub fn kind(&self) -> ClockKind {
        self.kind
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

    /// Keeps a decoded RGBA picture for export, including one that arrived late.
    ///
    /// A playback clock does not use this path. The bytes are returned only
    /// when the export clock accepts the picture.
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
    use cutlass_picture::{CpuBytes, Picture};

    fn pixel() -> Picture {
        Picture::Cpu(CpuBytes::new(1, 1, vec![1, 2, 3, 4]).expect("pixel"))
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
