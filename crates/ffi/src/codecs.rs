//! The containers and codecs a phone opens.
//!
//! The Apple decoders go first on an Apple phone, except WebP: its Rust
//! decoder goes ahead of them so a WebP decodes the same everywhere and an
//! animated one plays. A codec the Apple decoders decline falls through to
//! the Rust decoder of the same id.

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

/// WebP, the platform decoders for this OS, then PNG, GIF, JPEG, AAC,
/// H.264, HEVC, and PCM.
#[must_use]
pub fn decoders() -> DecodeFactory {
    let mut decoders = DecodeFactory::new();
    // Ahead of the platform: ImageIO gives one picture per packet and does
    // not keep an animation's canvas.
    cutlass_webp::register_decoders(&mut decoders);
    #[cfg(target_vendor = "apple")]
    cutlass_platform_apple::register_decoders(&mut decoders);
    cutlass_png::register_decoders(&mut decoders);
    cutlass_gif::register_decoders(&mut decoders);
    cutlass_jpeg::register_decoders(&mut decoders);
    cutlass_aac::register_decoders(&mut decoders);
    cutlass_h264::register_decoders(&mut decoders);
    // HEVC in Rust, after the platform decoders.
    cutlass_hevc::register_decoders(&mut decoders);
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

#[cfg(test)]
mod tests {
    use cutlass_decode::{
        CodecId, CodecParams, Decoder, Output, Packet, PictureSamples, PixelFormat,
    };

    /// A phone decodes WebP with the Rust decoder even where ImageIO is
    /// registered: the canvas an animation's first packet starts is kept
    /// for the frames in the next packet.
    #[test]
    fn a_phone_decodes_webp_in_rust_ahead_of_the_platform() {
        let file = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../codecs/webp/tests/fixtures/anim-mux.webp"),
        )
        .expect("fixture");
        let second = file
            .windows(4)
            .enumerate()
            .filter(|(_, word)| *word == b"ANMF")
            .nth(1)
            .map(|(at, _)| at)
            .expect("two frames");
        let params = CodecParams::with_format(CodecId::Webp, 40, 30, PixelFormat::Rgba8);
        let mut decoder = super::decoders().open(&params).expect("webp");
        assert_eq!(Decoder::codec(decoder.as_ref()), CodecId::Webp);
        let mut frames = 0;
        for data in [&file[..second], &file[second..]] {
            for output in decoder
                .push(&Packet::audio(data.to_vec(), 0, 0, 1000))
                .expect("packet")
            {
                let Output::Picture(picture) = output else {
                    panic!("picture");
                };
                assert!(matches!(picture.samples(), PictureSamples::Cpu(_)));
                assert_eq!((picture.info().width, picture.info().height), (40, 30));
                frames += 1;
            }
        }
        assert_eq!(frames, 6);
    }
}
