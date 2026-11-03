//! Each clip's sound, read from its file, for [`crate::export`].

use std::path::Path;

use cutlass_encode::{ChannelLayout, Pcm};
use cutlass_eval::{ClipAsk, ClipPcm, Heard, ReaderFactory};
use cutlass_model::{Draft, Micros, SegmentId};

use crate::ExportError;

/// The audio callback [`crate::export`] takes, reading each clip's file
/// through `readers`. A relative media path resolves against `root` and must
/// stay inside it.
///
/// A clip with no sound, such as a picture, or a file that cannot be opened,
/// is silence, as in playback. The file is read on the spot, so there is
/// nothing to wait for.
pub fn file_audio<'a>(
    readers: &'a ReaderFactory,
    draft: &'a Draft,
    root: Option<&'a Path>,
) -> impl FnMut(&SegmentId, Micros, u32, u32) -> Result<ClipPcm, ExportError> + 'a {
    let mut heard = Heard::new();
    move |segment, time, frames, rate| {
        let pcm = heard
            .clip(ClipAsk {
                readers,
                draft,
                revision: 0,
                segment,
                time,
                frames,
                rate,
                root,
            })
            .map_err(ExportError::Audio)?;
        match pcm {
            ClipPcm::Missing => silence(frames, rate),
            ready => Ok(ready),
        }
    }
}

fn silence(frames: u32, rate: u32) -> Result<ClipPcm, ExportError> {
    let count = usize::try_from(frames).map_err(|_| ExportError::Limit)?;
    Pcm::from_f32(rate, ChannelLayout::mono(), vec![0.0; count])
        .map(ClipPcm::Ready)
        .map_err(|_| ExportError::Limit)
}
