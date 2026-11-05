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
/// FLAC, and stills.
#[cfg(test)]
pub(crate) fn installed_readers() -> ReaderFactory {
    readers_with(crate::register_codecs().0)
}

/// ISOBMFF, WAV, AIFF, FLAC, and stills, decoded with `decoders`.
pub(crate) fn readers_with(decoders: DecodeFactory) -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_flac_container::probe, open_flac);
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

fn open_flac(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(cutlass_flac_container::FlacDemux::open(
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

    #[test]
    fn a_flac_file_opens_seeks_and_decodes_exactly() {
        let path =
            std::env::temp_dir().join(format!("cutlass-readers-flac-{}.flac", std::process::id()));
        let rate = 44_100u32;
        let source: Vec<i32> = (0..rate as usize * 3 / 2)
            .flat_map(|i| {
                let t = i as f64 / f64::from(rate);
                let left = (8000.0 * (t * 440.0 * std::f64::consts::TAU).sin()) as i32;
                [left, -left / 2]
            })
            .collect();
        let settings = cutlass_flac::Settings::level(5).expect("level");
        let mut encoder = cutlass_flac::FlacEncoder::new(rate, 2, 16, settings).expect("encoder");
        let mut frames = encoder.encode(&source).expect("encode");
        frames.extend(encoder.finish().expect("finish"));
        let file = std::fs::File::create(&path).expect("create");
        let mut mux = cutlass_flac_container::FlacMux::new(file);
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
        let mut ticks = 0;
        for frame in frames {
            let pts = i64::try_from(frame.first_sample).expect("pts");
            let duration = i64::from(frame.samples);
            let timestamp =
                Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("timestamp");
            let packet = Packet::try_new(0, frame.data, timestamp, Timescale::new(rate), true)
                .expect("packet");
            mux.write_packet(&packet).expect("write");
            ticks += duration;
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(rate),
            ticks,
        }])
        .expect("finish");
        drop(mux);
        let readers = installed_readers();
        let info = readers.probe(&path).expect("probe");
        assert_eq!(info.duration.map(|d| d.0), Some(1_500_000));
        let mut media = readers.open(&path).expect("open");
        let pcm = media
            .audio_from(cutlass_model::Micros(1_000_000), 1000)
            .expect("audio");
        std::fs::remove_file(&path).expect("remove");
        let got = pcm.as_f32().expect("f32");
        let start = rate as usize * 2;
        for (index, value) in got.iter().enumerate() {
            let expected = source[start + index] as f32 / 32_767.0;
            assert!((value - expected).abs() < 1e-6, "sample {index}");
        }
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
