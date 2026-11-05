//! The containers and codecs a phone opens.
//!
//! Rust audio decoders go first, so a file decodes the same on every
//! platform. The Apple decoders come next on an Apple phone; a codec they
//! decline falls through to the Rust decoder of the same id.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// ISOBMFF, WAV, AIFF, Ogg Opus, and stills, decoded with [`decoders`].
#[must_use]
pub fn readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_ogg::probe, open_ogg);
    readers.register_container(cutlass_still::probe, open_still);
    *readers.decoders_mut() = decoders();
    readers
}

/// Opus, the platform decoders for this OS, then PNG, JPEG, AAC, H.264,
/// and PCM.
#[must_use]
pub fn decoders() -> DecodeFactory {
    let mut decoders = DecodeFactory::new();
    // Rust audio codecs go ahead of the platform ones, so Opus decodes the
    // same on the phone as on the desktop.
    cutlass_opus::register_decoders(&mut decoders);
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

fn open_ogg(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = open_file(path)?;
    Ok(Box::new(cutlass_ogg::OggDemux::open(
        std::io::BufReader::new(file),
        length,
    )?))
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
    use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Output};
    use cutlass_encode::{AudioInput, Encoder, Pcm};

    /// The phone encodes and decodes Opus with the Rust codec: the same
    /// packets and samples as `OpusEncoder` and `OpusDecoder` themselves.
    /// On an Apple phone the platform codecs, put first, give other bytes
    /// and samples, so this fails if the order is reversed.
    #[test]
    fn the_rust_opus_codec_goes_ahead_of_the_platform_one() {
        let mut spec = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
        spec.bitrate = 96_000;
        let samples: Vec<f32> = (0..2 * 9600)
            .map(|i| ((i as f32) * 0.017).sin() * 0.4)
            .collect();
        let input = AudioInput {
            pcm: Pcm::from_f32(48_000, ChannelLayout::stereo(), samples).expect("pcm"),
        };
        let encode = |encoder: &mut dyn Encoder| {
            let mut packets = encoder.push(&input).expect("push");
            packets.extend(encoder.drain().expect("drain"));
            packets
        };
        let packets = encode(&mut cutlass_opus::OpusEncoder::open(&spec).expect("encoder"));
        let mut phone = crate::export::encoders()
            .open(&spec)
            .expect("phone encoder");
        assert_eq!(encode(phone.as_mut()), packets);
        let decode = |decoder: &mut dyn Decoder| {
            let mut out: Vec<f32> = Vec::new();
            for packet in &packets {
                for output in decoder.push(packet).expect("push") {
                    if let Output::Audio(audio) = output {
                        out.extend(audio.pcm.as_f32().expect("f32"));
                    }
                }
            }
            out
        };
        let mut rust = cutlass_opus::OpusDecoder::open(&spec).expect("rust");
        let expected = decode(&mut rust);
        let mut phone = super::decoders().open(&spec).expect("phone");
        assert_eq!(decode(phone.as_mut()), expected);
        #[cfg(target_vendor = "apple")]
        {
            let mut reversed = cutlass_decode::DecodeFactory::new();
            cutlass_platform_apple::register_decoders(&mut reversed);
            cutlass_opus::register_decoders(&mut reversed);
            let mut platform = reversed.open(&spec).expect("platform");
            assert_ne!(decode(platform.as_mut()), expected);
            let mut reversed = cutlass_encode::EncodeFactory::new();
            cutlass_platform_apple::register_encoders(&mut reversed);
            cutlass_opus::register_encoders(&mut reversed);
            let mut platform = reversed.open(&spec).expect("platform encoder");
            assert_ne!(encode(platform.as_mut()), packets);
        }
    }
}
