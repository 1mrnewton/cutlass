//! Containers and decoders the preview opens.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// The codecs [`crate::register_codecs`] installs, plus ISOBMFF, WAV, AIFF,
/// Ogg Opus, and stills.
#[cfg(test)]
pub(crate) fn installed_readers() -> ReaderFactory {
    readers_with(crate::register_codecs().0)
}

/// ISOBMFF, WAV, AIFF, Ogg Opus, and stills, decoded with `decoders`.
pub(crate) fn readers_with(decoders: DecodeFactory) -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_ogg::probe, open_ogg);
    readers.register_container(cutlass_still::probe, open_still);
    *readers.decoders_mut() = decoders;
    readers
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

fn open_isobmff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    Ok(Box::new(IsoBmffFile::open(file)?))
}

fn open_wav(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(cutlass_riff::WavDemux::open(file, length)?))
}

fn open_aiff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(cutlass_aiff::AiffDemux::open(file, length)?))
}

fn open_ogg(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(cutlass_ogg::OggDemux::open(
        std::io::BufReader::new(file),
        length,
    )?))
}

fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let len = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(StillDemux::open(Input::random(
        file,
        len,
        still_limits(),
    ))?))
}

#[cfg(test)]
mod tests {
    use super::installed_readers;
    use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
    use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};

    #[test]
    fn an_aiff_frame_decodes_as_big_endian_s16() {
        let path =
            std::env::temp_dir().join(format!("cutlass-readers-aiff-{}", std::process::id()));
        std::fs::write(&path, aiff_s16(&[0x40, 0x00])).expect("write");
        let mut media = installed_readers().open(&path).expect("open");
        let ready = media.pull_audio().expect("pull").expect("pcm");
        std::fs::remove_file(&path).expect("remove");
        let samples = ready.pcm.as_i16().expect("i16");
        assert_eq!(samples, &[0x4000]);
    }

    /// An `.opus` file opens, plays from the start exactly what the Rust
    /// decoder gives after the pre-skip, and converges on it after a seek.
    #[test]
    fn an_opus_file_opens_seeks_and_decodes_after_the_pre_skip() {
        use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};

        let path =
            std::env::temp_dir().join(format!("cutlass-readers-opus-{}.opus", std::process::id()));
        let samples: Vec<f32> = (0..2 * 72_000)
            .map(|i| ((i as f32) * 0.011).sin() * 0.4)
            .collect();
        let mut spec = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
        spec.bitrate = 128_000;
        let mut encoder = cutlass_opus::OpusEncoder::open(&spec).expect("encoder");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        let mut mux = cutlass_ogg::OggMux::new(std::fs::File::create(&path).expect("create"));
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Opus), 48_000, 2, Vec::new())
                .expect("params");
        let track = mux
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(48_000),
                language: None,
            })
            .expect("stream");
        let mut ticks = 0;
        for packet in &packets {
            for side in &packet.side {
                if let SideData::Config(head) = side {
                    mux.set_config(track, head).expect("head");
                }
            }
            let timestamp =
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .expect("timestamp");
            let out = Packet::try_new(
                0,
                packet.data.clone(),
                timestamp,
                Timescale::new(48_000),
                true,
            )
            .expect("packet");
            mux.write_packet(&out).expect("write");
            ticks += packet.duration;
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks,
        }])
        .expect("finish");
        drop(mux);

        let mut decoder = cutlass_opus::OpusDecoder::open(&spec).expect("decoder");
        let mut expected = Vec::new();
        for packet in &packets {
            expected.extend(decoder.decode(&packet.data).expect("decode"));
        }
        let readers = installed_readers();
        let info = readers.probe(&path).expect("probe");
        // The probe takes the last packet's end, which counts the 2.5 ms
        // pre-skip, as it counts AAC priming in an MP4.
        assert_eq!(info.duration.map(|d| d.0), Some(1_502_500));
        let mut media = readers.open(&path).expect("open");
        // From the start, the reader plays exactly the samples after the
        // pre-skip.
        let first = media
            .audio_from(cutlass_model::Micros(0), 4800)
            .expect("audio");
        let first = first.as_f32().expect("f32");
        assert_eq!(first.len(), 2 * 4800);
        for (index, value) in first.iter().enumerate() {
            assert!(
                (value - expected[2 * 120 + index]).abs() < 1e-6,
                "sample {index}"
            );
        }
        // A seek decodes from 80 ms early with a fresh decoder, so the
        // samples converge on the continuous decode rather than equal it.
        let later = media
            .audio_from(cutlass_model::Micros(1_000_000), 4800)
            .expect("audio");
        std::fs::remove_file(&path).expect("remove");
        let later = later.as_f32().expect("f32");
        assert_eq!(later.len(), 2 * 4800);
        let start = 2 * (120 + 48_000);
        let (mut signal, mut noise) = (0.0f64, 0.0f64);
        for (index, value) in later.iter().enumerate() {
            let want = f64::from(expected[start + index]);
            signal += want * want;
            noise += (f64::from(*value) - want).powi(2);
        }
        let snr = 10.0 * (signal / noise.max(1e-30)).log10();
        assert!(snr > 30.0, "SNR {snr:.1} dB after a seek");
    }

    #[test]
    fn an_iphone_movie_with_spatial_audio_plays_its_picture_and_aac_sound() {
        // Recent iPhones store an `apac` spatial-audio track beside the AAC.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/IMG_9197.mov");
        if !path.is_file() {
            eprintln!("skipped: {} is missing", path.display());
            return;
        }
        let mut media = installed_readers().open(&path).expect("open");
        let held = media
            .decode_at(
                cutlass_model::Micros(1_000_000),
                &cutlass_model::TimeRange::new(0, 10_000_000),
            )
            .expect("picture");
        assert_eq!((held.frame.width(), held.frame.height()), (3840, 2160));
        let sound = media.pull_audio().expect("pull").expect("pcm");
        assert_eq!(sound.pcm.rate, 48_000);
    }

    #[test]
    fn a_fragmented_movie_without_sidx_or_mfra_decodes_the_picture_at_two_seconds() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/16929689_1920_1080_30fps.mp4");
        if !path.is_file() {
            eprintln!("skipped: {} is missing", path.display());
            return;
        }
        let mut media = installed_readers().open(&path).expect("open");
        let at = cutlass_model::Micros(2_000_000);
        let held = media
            .decode_at(at, &cutlass_model::TimeRange::new(0, 10_000_000))
            .expect("picture at 2 s");
        assert!(held.start.0 <= at.0 && at.0 < held.end.0);
        assert_eq!(held.frame.width(), 1920);
        assert_eq!(held.frame.height(), 1080);
    }

    fn aiff_s16(frame: &[u8]) -> Vec<u8> {
        let mut mux = cutlass_aiff::AiffMux::new(std::io::Cursor::new(Vec::new()));
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 8_000, 1, vec![1, 1, 0])
                .expect("params");
        mux.add_stream(NewStream {
            params,
            timescale: Timescale::new(8_000),
            language: None,
        })
        .expect("stream");
        let packet = Packet::try_new(
            0,
            frame.to_vec(),
            Timestamp::try_new(Some(0), Some(0), Some(1)).expect("time"),
            Timescale::new(8_000),
            true,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
        mux.finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(8_000),
            ticks: 1,
        }])
        .expect("finish");
        mux.into_inner().into_inner()
    }
}
