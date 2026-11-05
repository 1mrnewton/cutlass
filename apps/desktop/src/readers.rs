//! Containers and decoders the preview opens.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// The codecs [`crate::register_codecs`] installs, plus ISOBMFF, WAV, AIFF, MP3, and stills.
#[cfg(test)]
pub(crate) fn installed_readers() -> ReaderFactory {
    readers_with(crate::register_codecs().0)
}

/// ISOBMFF, WAV, AIFF, MP3, and stills, decoded with `decoders`.
pub(crate) fn readers_with(decoders: DecodeFactory) -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_still::probe, open_still);
    readers.register_container(probe_mp3, open_mp3);
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

/// An ID3v2 tag longer than the header window, which cover art often makes,
/// is taken as MP3: the demuxer skips the tag and refuses the file when no
/// Layer III frame follows it. MP3 is registered last, so a family that
/// names the header itself still wins.
fn probe_mp3(header: &[u8]) -> ProbeResult {
    let accept = ProbeResult::Accept {
        family: ContainerId::Mp3,
        examined: header.len(),
    };
    match cutlass_mp3_container::probe(header) {
        cutlass_mp3_container::Probe::Matched => accept,
        cutlass_mp3_container::Probe::NeedMore { .. } if header.starts_with(b"ID3") => accept,
        cutlass_mp3_container::Probe::NeedMore { minimum } => ProbeResult::NeedBytes {
            have: header.len(),
            need: usize::try_from(minimum).unwrap_or(usize::MAX),
        },
        cutlass_mp3_container::Probe::Reject => ProbeResult::Refuse {
            examined: header.len(),
        },
    }
}

fn open_mp3(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let demux =
        cutlass_mp3_container::Mp3Demux::open(std::io::BufReader::new(file)).map_err(|error| {
            OpenError::Source {
                message: error.to_string(),
            }
        })?;
    Ok(Box::new(demux))
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

    fn mp3_fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/codecs/mp3/tests/fixtures/sine_44100_j128_crc.mp3")
    }

    /// Amplitude and phase lag, in samples, of a 1 kHz sine fitted to the
    /// left channel of interleaved stereo that should start at `t0` seconds.
    fn sine_fit(samples: &[f32], rate: f64, t0: f64) -> (f64, f64) {
        let w = 2.0 * std::f64::consts::PI * 1000.0 / rate;
        let (mut xs, mut xc, mut ss, mut cc) = (0.0, 0.0, 0.0, 0.0);
        for (i, sample) in samples.iter().step_by(2).enumerate() {
            let t = (i as f64 + t0 * rate) * w;
            xs += f64::from(*sample) * t.sin();
            xc += f64::from(*sample) * t.cos();
            ss += t.sin() * t.sin();
            cc += t.cos() * t.cos();
        }
        let (a, b) = (xs / ss, xc / cc);
        (a.hypot(b), -b.atan2(a) / w)
    }

    #[test]
    fn an_mp3_plays_through_the_rust_decoder_from_its_first_valid_sample() {
        let mut decoders = cutlass_decode::DecodeFactory::new();
        cutlass_mp3::register_decoders(&mut decoders);
        let mut media = super::readers_with(decoders)
            .open(&mp3_fixture())
            .expect("open");
        // The source is a 1 kHz half-scale sine that starts at phase zero.
        // LAME's priming and the decoder delay are trimmed, so it starts
        // at zero here too, and again after a seek.
        for (at, t0) in [(0, 0.0), (250_000, 0.25), (100_000, 0.1)] {
            let pcm = media
                .audio_from(cutlass_model::Micros(at), 2_205)
                .expect("audio");
            assert_eq!(pcm.rate, 44_100);
            let samples = pcm.as_f32().expect("f32");
            assert_eq!(samples.len(), 2 * 2_205);
            let (amplitude, lag) = sine_fit(samples, 44_100.0, t0);
            assert!(amplitude > 0.4, "at {at}: amplitude {amplitude}");
            assert!(lag.abs() < 0.05, "at {at}: lag {lag}");
        }
    }

    #[test]
    fn the_installed_readers_open_an_mp3() {
        let mut media = installed_readers().open(&mp3_fixture()).expect("open");
        let ready = media.pull_audio().expect("pull").expect("pcm");
        assert_eq!(ready.pcm.rate, 44_100);
        assert_eq!(ready.pcm.layout.len(), 2);
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
