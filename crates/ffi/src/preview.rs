//! The picture at a timeline time, drawn on the software device.
//!
//! Media paths in the draft are read from the project folder. A phone
//! registers the Apple decoders first, then the Rust codecs. Text segments
//! are not drawn yet: the phone has no font table.

use std::path::Path;

use cutlass_eval::{NoFonts, ReaderFactory};
use cutlass_playback::FrameCache;
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{Micros, Session};

use crate::error::FfiError;

/// A composited picture, eight-bit RGBA, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The readers and the picture cache behind [`crate::PhoneHost::frame`].
pub struct Preview {
    readers: ReaderFactory,
    cache: FrameCache,
}

impl Preview {
    /// The phone's codecs and an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::with_readers(crate::codecs::readers())
    }

    #[must_use]
    pub fn with_readers(readers: ReaderFactory) -> Self {
        Self {
            readers,
            cache: FrameCache::new(),
        }
    }

    /// The readers playback mixes sound with.
    #[must_use]
    pub fn readers(&self) -> &ReaderFactory {
        &self.readers
    }

    /// The picture at `time`, waiting for every clip's picture. A time
    /// outside the cut draws the canvas alone.
    pub fn frame(
        &mut self,
        session: &Session,
        root: &Path,
        time: Micros,
    ) -> Result<Picture, FfiError> {
        if time.0 < 0 {
            return Err(FfiError::Frame {
                reason: "the time is before the start".to_owned(),
            });
        }
        let mut device = SoftwareDevice::new();
        self.cache.set_patient(true);
        let readback = self
            .cache
            .frame(
                session,
                &mut device,
                time,
                &self.readers,
                Some(root),
                &NoFonts,
            )
            .map_err(|error| FfiError::Frame {
                reason: error.to_string(),
            })?;
        Ok(Picture {
            width: readback.width,
            height: readback.height,
            rgba: readback.rgba,
        })
    }
}

impl Default for Preview {
    fn default() -> Self {
        Self::new()
    }
}
