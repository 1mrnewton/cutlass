//! The containers and codecs a phone opens.
//!
//! The Apple decoders go first on an Apple phone. A codec they decline falls
//! through to the Rust decoder of the same id.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// ISOBMFF, WAV, AIFF, and stills, decoded with [`decoders`].
#[must_use]
pub fn readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_still::probe, open_still);
    *readers.decoders_mut() = decoders();
    readers
}

/// The platform decoders for this OS, then PNG, JPEG, AAC, H.264, and PCM.
#[must_use]
pub fn decoders() -> DecodeFactory {
    let mut decoders = DecodeFactory::new();
    #[cfg(target_vendor = "apple")]
    cutlass_platform_apple::register_decoders(&mut decoders);
    cutlass_png::register_decoders(&mut decoders);
    cutlass_jpeg::register_decoders(&mut decoders);
    cutlass_aac::register_decoders(&mut decoders);
    cutlass_h264::register_decoders(&mut decoders);
    cutlass_pcm::register_decoders(&mut decoders);
    decoders
}

fn probe_isobmff(header: &[u8]) -> ProbeResult {
    if header.len() < 8 {
        return ProbeResult::NeedBytes {
            have: header.len(),
            need: 8,
        };
    }
    if header.get(4..8) == Some(b"ftyp") {
        ProbeResult::Accept {
            family: ContainerId::Isobmff,
            examined: 8,
        }
    } else {
        ProbeResult::Refuse { examined: 8 }
    }
}

fn open_file(path: &Path) -> Result<(File, u64), OpenError> {
    let source = |error: std::io::Error| OpenError::Source {
        message: error.to_string(),
    };
    let file = File::open(path).map_err(source)?;
    let length = file.metadata().map_err(source)?.len();
    Ok((file, length))
}

fn open_isobmff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, _) = open_file(path)?;
    Ok(Box::new(IsoBmffFile::open(file)?))
}

fn open_wav(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = open_file(path)?;
    Ok(Box::new(cutlass_riff::WavDemux::open(file, length)?))
}

fn open_aiff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = open_file(path)?;
    Ok(Box::new(cutlass_aiff::AiffDemux::open(file, length)?))
}

fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = open_file(path)?;
    Ok(Box::new(StillDemux::open(Input::random(
        file,
        length,
        still_limits(),
    ))?))
}
