//! The containers and codecs a phone opens.
//!
//! The Apple decoders go first on an Apple phone. A codec they decline falls
//! through to the Rust decoder of the same id. FLAC is the exception: the
//! Rust decoder goes first everywhere, so a FLAC file decodes bit-exactly
//! and the same on every platform.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// ISOBMFF, WAV, AIFF, FLAC, and stills, decoded with [`decoders`].
#[must_use]
pub fn readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_flac_container::probe, open_flac);
    readers.register_container(cutlass_still::probe, open_still);
    *readers.decoders_mut() = decoders();
    readers
}

/// FLAC, the platform decoders for this OS, then PNG, JPEG, AAC, H.264,
/// and PCM.
#[must_use]
pub fn decoders() -> DecodeFactory {
    let mut decoders = DecodeFactory::new();
    cutlass_flac::register_decoders(&mut decoders);
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

fn open_flac(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = open_file(path)?;
    Ok(Box::new(cutlass_flac_container::FlacDemux::open(
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
    use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
    use cutlass_container::{Mux, NewStream, TrackDuration};

    #[test]
    fn a_flac_file_opens_and_decodes_exactly() {
        let path =
            std::env::temp_dir().join(format!("cutlass-ffi-flac-{}.flac", std::process::id()));
        let rate = 48_000u32;
        let source: Vec<i32> = (0..rate as usize)
            .flat_map(|i| {
                let left = (i as i32 % 400 - 200) * 60;
                [left, -left / 3]
            })
            .collect();
        let settings = cutlass_flac::Settings::level(5).expect("level");
        let mut encoder = cutlass_flac::FlacEncoder::new(rate, 2, 16, settings).expect("encoder");
        let mut frames = encoder.encode(&source).expect("encode");
        frames.extend(encoder.finish().expect("finish"));
        let mut mux =
            cutlass_flac_container::FlacMux::new(std::fs::File::create(&path).expect("create"));
        let params = StreamParams::try_audio(
            StreamCodec::Known(CodecId::Flac),
            rate,
            2,
            encoder.stream_info().to_bytes().to_vec(),
        )
        .expect("params");
        let track = mux
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(rate),
                language: None,
            })
            .expect("stream");
        for frame in frames {
            let pts = i64::try_from(frame.first_sample).expect("pts");
            let timestamp =
                Timestamp::try_new(Some(pts), Some(pts), Some(i64::from(frame.samples)))
                    .expect("timestamp");
            let packet = Packet::try_new(0, frame.data, timestamp, Timescale::new(rate), true)
                .expect("packet");
            mux.write_packet(&packet).expect("write");
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(rate),
            ticks: i64::from(rate),
        }])
        .expect("finish");
        drop(mux);
        let readers = super::readers();
        let info = readers.probe(&path).expect("probe");
        assert_eq!(info.duration.map(|d| d.0), Some(1_000_000));
        let mut media = readers.open(&path).expect("open");
        let pcm = media
            .audio_from(cutlass_model::Micros(0), 4096)
            .expect("audio");
        std::fs::remove_file(&path).expect("remove");
        let got = pcm.as_f32().expect("f32");
        assert!(!got.is_empty());
        for (index, value) in got.iter().enumerate() {
            let expected = source[index] as f32 / 32_767.0;
            assert!((value - expected).abs() < 1e-6, "sample {index}");
        }
    }
}
