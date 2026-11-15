//! The picture at a timeline time, drawn on the GPU.
//!
//! Media paths in the draft are read from the project folder. A phone
//! registers the Apple decoders first, then the Rust codecs. Eval hands the
//! frame over as layers; [`cutlass_render`] draws them on the phone's GPU
//! (Metal on Apple) and the canvas is read back for the shell to show.
//! There is no CPU fallback: a phone without a GPU device gets an error.

use std::path::Path;

use cutlass_eval::{Fonts, ReaderFactory};
use cutlass_playback::FrameCache;
use cutlass_render::{Canvas, RenderError, Renderer};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{Micros, Session};

use crate::error::FfiError;

/// A composited picture, eight-bit premultiplied RGBA, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// What the last picture showed, so the same ask is not drawn twice.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Drawn {
    draft: String,
    revision: u64,
    time: Micros,
}

/// The readers, the open clips and the GPU behind [`crate::PhoneHost::frame`].
pub struct Preview {
    readers: ReaderFactory,
    cache: FrameCache,
    /// Opened on the first frame.
    renderer: Option<Renderer>,
    canvas: Option<Canvas>,
    last: Option<(Drawn, Picture)>,
    /// The last picture showed a clip early. See [`Preview::pending`].
    pending: bool,
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
            renderer: None,
            canvas: None,
            last: None,
            pending: false,
        }
    }

    /// The readers playback mixes sound with.
    #[must_use]
    pub fn readers(&self) -> &ReaderFactory {
        &self.readers
    }

    /// The GPU the preview draws on, opened now if no frame has been drawn.
    pub(crate) fn renderer(&mut self) -> Result<&Renderer, FfiError> {
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::headless().map_err(drawing)?);
        }
        self.renderer
            .as_ref()
            .ok_or_else(|| drawing(RenderError::NoDevice("not opened".to_owned())))
    }

    /// The picture at `time`, waiting for every clip's picture. A time
    /// outside the cut draws the canvas alone.
    pub fn frame(
        &mut self,
        session: &Session,
        root: &Path,
        fonts: &dyn Fonts,
        time: Micros,
    ) -> Result<Picture, FfiError> {
        self.draw(session, root, fonts, time, true)
    }

    /// The picture at `time` during play, without waiting for a clip whose
    /// picture is still decoding: that clip shows the picture before it,
    /// and its thread moves on to the newest time asked. A late picture is
    /// never shown after its time has passed; the play goes on at the
    /// sound's pace. See [`Self::pending`].
    pub fn playing_frame(
        &mut self,
        session: &Session,
        root: &Path,
        fonts: &dyn Fonts,
        time: Micros,
    ) -> Result<Picture, FfiError> {
        self.draw(session, root, fonts, time, false)
    }

    /// Whether the last picture showed a clip at an earlier time than asked,
    /// because [`Self::playing_frame`] did not wait. It is not kept as the
    /// picture for that time, so asking again draws it again.
    #[must_use]
    pub fn pending(&self) -> bool {
        self.pending
    }

    fn draw(
        &mut self,
        session: &Session,
        root: &Path,
        fonts: &dyn Fonts,
        time: Micros,
        patient: bool,
    ) -> Result<Picture, FfiError> {
        if time.0 < 0 {
            return Err(FfiError::Frame {
                reason: "the time is before the start".to_owned(),
            });
        }
        let key = Drawn {
            draft: session.draft().id.to_string(),
            revision: session.revision(),
            time,
        };
        if let Some((drawn, picture)) = &self.last
            && *drawn == key
        {
            self.pending = false;
            return Ok(picture.clone());
        }
        // Eval's CPU device draws the layers the GPU does not, such as text.
        let mut device = SoftwareDevice::new();
        self.cache.set_patient(patient);
        let layers = self
            .cache
            .layers(session, &mut device, time, &self.readers, Some(root), fonts)
            .map_err(|error| FfiError::Frame {
                reason: error.to_string(),
            })?;
        let stale = self.cache.take_stale();
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::headless().map_err(drawing)?);
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return Err(drawing(RenderError::NoDevice("not opened".to_owned())));
        };
        let size = (layers.width, layers.height);
        if self
            .canvas
            .as_ref()
            .is_none_or(|canvas| canvas.size() != size)
        {
            self.canvas = Some(renderer.canvas(size.0, size.1).map_err(drawing)?);
        }
        let Some(canvas) = self.canvas.as_ref() else {
            return Err(drawing(RenderError::Size {
                width: size.0,
                height: size.1,
            }));
        };
        renderer.draw_frame(canvas, &layers).map_err(drawing)?;
        let picture = Picture {
            width: size.0,
            height: size.1,
            rgba: renderer.read(canvas).map_err(drawing)?,
        };
        self.pending = stale;
        self.last = (!stale).then(|| (key, picture.clone()));
        Ok(picture)
    }
}

fn drawing(error: RenderError) -> FfiError {
    FfiError::Frame {
        reason: error.to_string(),
    }
}

impl Default for Preview {
    fn default() -> Self {
        Self::new()
    }
}
