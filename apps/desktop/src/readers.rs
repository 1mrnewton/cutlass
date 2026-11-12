//! Containers and decoders the preview opens.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// The codecs [`crate::register_codecs`] installs, plus ISOBMFF, AVI, WAV,
/// AIFF, FLAC, Ogg (Opus and Vorbis), MP3, and stills.
#[cfg(test)]
pub(crate) fn installed_readers() -> ReaderFactory {
    readers_with(crate::register_codecs().0)
}

/// ISOBMFF, AVI, WAV, AIFF, FLAC, Ogg (Opus and Vorbis), MP3, and stills,
/// decoded with `decoders`.
pub(crate) fn readers_with(decoders: DecodeFactory) -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_avi::probe, open_avi);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_flac_container::probe, open_flac);
    readers.register_container(cutlass_ogg::probe, open_ogg);
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

fn open_avi(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(cutlass_avi::AviDemux::open(file, length)?))
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

    /// An `.ogg` Vorbis file opens, probes, and plays the decoder's own
    /// samples from the start and after a seek. There is no encoder delay.
    #[test]
    fn an_ogg_vorbis_file_opens_seeks_and_decodes() {
        use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};

        let path =
            std::env::temp_dir().join(format!("cutlass-readers-vorbis-{}.ogg", std::process::id()));
        let rate = 44_100_u32;
        let frames = rate as usize * 2;
        let samples: Vec<f32> = (0..frames * 2)
            .map(|i| ((i as f32) * 0.02).sin() * 0.25)
            .collect();
        let spec = CodecParams::audio(CodecId::Vorbis, rate, ChannelLayout::stereo());
        let mut encoder = cutlass_vorbis::VorbisEncoder::open(&spec).expect("encoder");
        let pcm = Pcm::from_f32(rate, ChannelLayout::stereo(), samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        let config = packets
            .iter()
            .find_map(|packet| {
                packet.side.iter().find_map(|side| match side {
                    SideData::Config(config) => Some(config.clone()),
                    _ => None,
                })
            })
            .expect("headers");
        let mut mux = cutlass_ogg::OggMux::new(std::fs::File::create(&path).expect("create"));
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Vorbis), rate, 2, config.clone())
                .expect("params");
        let track = mux
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(rate),
                language: None,
            })
            .expect("stream");
        let mut ticks = 0;
        for packet in &packets {
            let timestamp =
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .expect("timestamp");
            let out = Packet::try_new(
                track.get(),
                packet.data.clone(),
                timestamp,
                Timescale::new(rate),
                true,
            )
            .expect("packet");
            mux.write_packet(&out).expect("write");
            ticks += packet.duration;
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(rate),
            ticks,
        }])
        .expect("finish");
        drop(mux);

        let mut decoder = cutlass_vorbis::VorbisDecoder::from_config(&config).expect("decoder");
        let mut expected = Vec::new();
        for packet in &packets {
            expected.extend(decoder.decode(&packet.data).expect("decode"));
        }
        expected.truncate(frames * 2);

        let readers = installed_readers();
        let info = readers.probe(&path).expect("probe");
        assert_eq!(info.duration.map(|d| d.0), Some(2_000_000));
        let mut media = readers.open(&path).expect("open");
        let window = 4_096_usize;
        let first = media
            .audio_from(cutlass_model::Micros(0), window as u32)
            .expect("audio");
        let first = first.as_f32().expect("f32");
        assert_eq!(first, &expected[..window * 2]);
        let later = media
            .audio_from(cutlass_model::Micros(1_000_000), window as u32)
            .expect("seek");
        std::fs::remove_file(&path).expect("remove");
        let later = later.as_f32().expect("f32");
        let at = rate as usize * 2;
        assert_eq!(later, &expected[at..at + window * 2]);
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

    /// An AC-3 or E-AC-3 movie opens, plays the tone, seeks, and opens again.
    #[test]
    fn an_ac3_movie_opens_seeks_and_reopens() {
        use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};

        for codec in [CodecId::Ac3, CodecId::Eac3] {
            let path = std::env::temp_dir().join(format!(
                "cutlass-readers-{codec:?}-{}.mp4",
                std::process::id()
            ));
            let rate = 48_000u32;
            let layout = ChannelLayout::stereo();
            let samples: Vec<f32> = (0..rate as usize)
                .flat_map(|i| {
                    let t = i as f32 / rate as f32;
                    let left = (t * 1_000.0 * std::f32::consts::TAU).sin() * 0.4;
                    [left, left * 0.5]
                })
                .collect();
            let mut spec = CodecParams::audio(codec, rate, layout.clone());
            spec.bitrate = 192_000;
            let mut encoder = cutlass_ac3::Ac3Encoder::open(&spec).expect("encoder");
            let pcm = Pcm::from_f32(rate, layout, samples).expect("pcm");
            let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
            packets.extend(encoder.drain().expect("drain"));
            let mut mux =
                cutlass_isobmff::Mp4Mux::new(std::fs::File::create(&path).expect("create"));
            let params = StreamParams::try_audio(StreamCodec::Known(codec), rate, 2, Vec::new())
                .expect("params");
            let track = mux
                .add_stream(NewStream {
                    params,
                    timescale: Timescale::new(rate),
                    language: None,
                })
                .expect("stream");
            let mut ticks = 0i64;
            for packet in &packets {
                for side in &packet.side {
                    if let SideData::Config(config) = side {
                        mux.set_config(track, config).expect("config");
                    }
                }
                let timestamp =
                    Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                        .expect("time");
                let out = Packet::try_new(
                    track.get(),
                    packet.data.clone(),
                    timestamp,
                    Timescale::new(rate),
                    true,
                )
                .expect("packet");
                mux.write_packet(&out).expect("write");
                ticks += packet.duration;
            }
            mux.finish(&[TrackDuration {
                track,
                timescale: Timescale::new(rate),
                ticks,
            }])
            .expect("finish");
            drop(mux);

            let window = |at| {
                let mut media = installed_readers().open(&path).expect("open");
                media
                    .audio_from(cutlass_model::Micros(at), 2_048)
                    .expect("audio")
            };
            let start = window(100_000);
            let later = window(500_000);
            let info = installed_readers().probe(&path).expect("probe");
            assert!(info.duration.is_some(), "{codec:?}");
            installed_readers().open(&path).expect("reopen");
            std::fs::remove_file(&path).expect("remove");
            for (name, pcm) in [("start", &start), ("seek", &later)] {
                let samples = pcm.as_f32().expect("f32");
                assert_eq!(samples.len(), 2 * 2_048, "{codec:?} {name}");
                assert!(
                    tone_margin(samples, rate, 1_000.0) > 20.0,
                    "{codec:?} {name}"
                );
            }
        }
    }

    /// A CRC-failed syncframe in the middle of an AC-3 file is passed over.
    /// Its time is silence, and the sweep after it plays at its own time,
    /// sample for sample as the intact file plays it.
    #[test]
    fn a_bad_ac3_frame_is_skipped_and_the_rest_plays() {
        use cutlass_decode::Decoder;
        use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};

        let rate = 48_000u32;
        let layout = ChannelLayout::stereo();
        // A sweep from 200 Hz to 4 kHz: one frame early is a different pitch.
        let samples: Vec<f32> = (0..rate as usize)
            .flat_map(|i| {
                let t = f64::from(i as u32) / f64::from(rate);
                let phase = std::f64::consts::TAU * (200.0 * t + 1_900.0 * t * t);
                let left = (phase.sin() * 0.4) as f32;
                [left, left * 0.5]
            })
            .collect();
        let mut spec = CodecParams::audio(CodecId::Ac3, rate, layout.clone());
        spec.bitrate = 192_000;
        let mut encoder = cutlass_ac3::Ac3Encoder::open(&spec).expect("encoder");
        let pcm = Pcm::from_f32(rate, layout, samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        assert!(packets.len() > 8, "need a frame in the middle");
        let mut damaged = packets.clone();
        damaged[3].data[20] ^= 0xff;
        let mut check = cutlass_ac3::Ac3Decoder::open(&spec).expect("decoder");
        for (index, packet) in damaged.iter().enumerate().take(5) {
            let pushed = check.push(packet);
            if index == 3 {
                assert!(pushed.is_err(), "the damaged frame was accepted");
            } else {
                pushed.expect("a good frame");
            }
        }
        let write = |name: &str, packets: &[cutlass_encode::Packet]| {
            let path = std::env::temp_dir().join(format!(
                "cutlass-readers-{name}-ac3-{}.mp4",
                std::process::id()
            ));
            let mut mux =
                cutlass_isobmff::Mp4Mux::new(std::fs::File::create(&path).expect("create"));
            let params =
                StreamParams::try_audio(StreamCodec::Known(CodecId::Ac3), rate, 2, Vec::new())
                    .expect("params");
            let track = mux
                .add_stream(NewStream {
                    params,
                    timescale: Timescale::new(rate),
                    language: None,
                })
                .expect("stream");
            let mut ticks = 0i64;
            for packet in packets {
                for side in &packet.side {
                    if let SideData::Config(config) = side {
                        mux.set_config(track, config).expect("config");
                    }
                }
                let timestamp =
                    Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                        .expect("time");
                let out = Packet::try_new(
                    track.get(),
                    packet.data.clone(),
                    timestamp,
                    Timescale::new(rate),
                    true,
                )
                .expect("packet");
                mux.write_packet(&out).expect("write");
                ticks += packet.duration;
            }
            mux.finish(&[TrackDuration {
                track,
                timescale: Timescale::new(rate),
                ticks,
            }])
            .expect("finish");
            path
        };
        let clean_path = write("clean", &packets);
        let bad_path = write("bad", &damaged);

        // One window, then the same span in 1024-frame windows that run on
        // without a seek.
        let frames = 8 * 1_536;
        let read = |path: &std::path::Path, step: u32| {
            let mut media = installed_readers().open(path).expect("open");
            let mut out = Vec::new();
            let mut at = 0u32;
            while at < frames {
                let micros = i64::from(at) * 1_000_000 / i64::from(rate);
                let pcm = media
                    .audio_from(cutlass_model::Micros(micros), step)
                    .expect("audio");
                out.extend_from_slice(pcm.as_f32().expect("f32"));
                at += step;
            }
            out
        };
        let clean = read(&clean_path, frames);
        let whole = read(&bad_path, frames);
        let stepped = read(&bad_path, 1_024);
        let seek = |path: &std::path::Path| {
            installed_readers()
                .open(path)
                .expect("open")
                .audio_from(cutlass_model::Micros(250_000), 2_048)
                .expect("audio")
                .as_f32()
                .expect("f32")
                .to_vec()
        };
        let clean_later = seek(&clean_path);
        let bad_later = seek(&bad_path);
        std::fs::remove_file(&clean_path).expect("remove");
        std::fs::remove_file(&bad_path).expect("remove");

        // The decoder trims the 256-sample delay from the head of the
        // stream, so syncframe `k`'s own sound plays from `1536 k`.
        let frame = |k: usize| 1_536 * k * 2;
        let peak = |samples: &[f32]| samples.iter().fold(0.0f32, |top, x| top.max(x.abs()));
        for (name, bad) in [("one window", &whole), ("stepped", &stepped)] {
            assert_eq!(bad.len(), clean.len(), "{name}");
            assert_eq!(bad[..frame(3)], clean[..frame(3)], "{name} before");
            assert_eq!(peak(&bad[frame(3)..frame(4)]), 0.0, "{name} hole");
            // Syncframe 4 overlaps onto syncframe 2, not 3. From syncframe 5
            // on the decoder is back on the intact file's samples.
            let after = &bad[frame(5)..];
            let reference = &clean[frame(5)..];
            let worst = after
                .iter()
                .zip(reference)
                .fold(0.0f32, |top, (a, b)| top.max((a - b).abs()));
            assert!(worst < 1e-4, "{name} after the hole is off by {worst}");
            assert!(peak(after) > 0.1, "{name} after the hole is quiet");
        }
        assert_eq!(bad_later, clean_later, "seek past the hole");
        assert_eq!(bad_later.len(), 2 * 2_048);
    }

    /// A real recording is imported, exported as AC-3 and E-AC-3, then
    /// reopened and seeked. The file's edit skips the encoder delay, so
    /// playback lines up with the source.
    #[test]
    fn a_real_recording_exports_as_ac3_and_eac3_and_plays_back() {
        use cutlass_encode::EncodeFactory;
        use cutlass_eval::NoFonts;
        use cutlass_export::{AudioExport, ExportRequest, MovieSources, write_sound};
        use cutlass_model::{
            Draft, FrameRate, MaterialAudio, MaterialId, Micros, Segment, TimeRange, Track,
            TrackType,
        };

        let Some(root) = local_media() else {
            return;
        };
        let name = "06 First Person Shooter (feat. J. Cole).m4a";
        let source_path = root.join(name);
        if !source_path.is_file() {
            if std::env::var_os("AC3_REQUIRE_ORACLES").is_some_and(|value| value == "1") {
                panic!("missing {}", source_path.display());
            }
            return;
        }
        let source_at = |at| {
            installed_readers()
                .open(&source_path)
                .expect("import")
                .audio_from(Micros(at), 16_384)
                .expect("source")
        };
        let start = source_at(40_000_000);
        let later = source_at(41_000_000);
        assert_eq!(start.rate, 48_000);
        assert!(
            rms(start.as_f32().expect("f32")) > 0.01,
            "the imported window is silent"
        );

        let mut draft = Draft::empty("song");
        draft.duration = Micros(2_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            path: name.to_owned(),
            duration: Micros(42_000_000),
            ..MaterialAudio::default()
        });
        let mut track = Track::main("song");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("song", "song", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(40_000_000, 2_000_000));
        track.segments.push(segment);
        draft.tracks.push(track);

        for codec in [CodecId::Ac3, CodecId::Eac3] {
            let bitrate = if codec == CodecId::Ac3 {
                192_000
            } else {
                128_000
            };
            let dir =
                std::env::temp_dir().join(format!("cutlass-real-{codec:?}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("dir");
            let path = dir.join("cut.m4a");
            let mut encoders = EncodeFactory::new();
            cutlass_ac3::register_encoders(&mut encoders);
            let readers = installed_readers();
            let sources = MovieSources {
                draft: &draft,
                root: Some(&root),
                readers: &readers,
                fonts: &NoFonts,
                renderer: None,
                encoders: &encoders,
            };
            let request = ExportRequest {
                range: TimeRange::new(0, 2_000_000),
                short_side: 0,
                codec,
                bitrate: 0,
                frame_rate: FrameRate::FPS_30,
                container: cutlass_codec::ContainerId::Isobmff,
                audio: Some(AudioExport {
                    codec,
                    bitrate,
                    sample_rate: 48_000,
                    channels: 2,
                    limiter: 0,
                }),
                sidecar: None,
                late_by: 0,
                constant_rate: false,
                prores: cutlass_export::ProresProfile::Standard,
            };
            write_sound(&sources, &request, &path, &mut |_, _| true).expect("export");

            let file = cutlass_isobmff::IsoBmffFile::open(
                std::fs::File::open(&path).expect("open export"),
            )
            .expect("movie");
            let edit = cutlass_container::Demux::streams(&file)[0].edits()[0];
            assert_eq!(
                edit.media_time(),
                Some(i64::from(cutlass_ac3::ENCODER_DELAY)),
                "{codec:?}"
            );

            let played = installed_readers()
                .open(&path)
                .expect("playback")
                .audio_from(Micros(0), 16_384)
                .expect("play");
            let seeked = installed_readers()
                .open(&path)
                .expect("reopen")
                .audio_from(Micros(1_000_000), 16_384)
                .expect("seek");
            installed_readers().open(&path).expect("reopen");
            std::fs::remove_dir_all(&dir).expect("remove");
            assert_aligned(&start, &played, codec);
            assert_aligned(&later, &seeked, codec);
        }
    }

    fn local_media() -> Option<std::path::PathBuf> {
        let path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        if path.is_dir() {
            return Some(path);
        }
        if std::env::var_os("AC3_REQUIRE_ORACLES").is_some_and(|value| value == "1") {
            panic!("missing {}", path.display());
        }
        None
    }

    fn rms(samples: &[f32]) -> f64 {
        let energy: f64 = samples
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum();
        (energy / f64::from(u32::try_from(samples.len()).unwrap_or(1).max(1))).sqrt()
    }

    fn left(pcm: &cutlass_codec::access::Pcm) -> Vec<f32> {
        let channels = pcm.layout.len().max(1);
        pcm.as_f32()
            .expect("f32")
            .iter()
            .step_by(channels)
            .copied()
            .collect()
    }

    fn assert_aligned(
        source: &cutlass_codec::access::Pcm,
        played: &cutlass_codec::access::Pcm,
        codec: CodecId,
    ) {
        assert_eq!(
            (played.rate, played.layout.len()),
            (source.rate, 2),
            "{codec:?}"
        );
        let reference = left(source);
        let got = left(played);
        let mut best_lag = 0;
        let mut best = f64::MIN;
        for lag in -8..=8 {
            let score = lagged_snr(&reference, &got, lag);
            if score > best {
                best = score;
                best_lag = lag;
            }
        }
        let at_zero = lagged_snr(&reference, &got, 0);
        assert_eq!(
            best_lag, 0,
            "{codec:?} lines up at lag {best_lag} ({best:.1} dB), not at 0 ({at_zero:.1} dB)"
        );
        assert!(at_zero > 15.0, "{codec:?} {at_zero:.1} dB");
    }

    /// A positive `lag` drops the start of `got`: the playback is that many
    /// samples late.
    fn lagged_snr(reference: &[f32], got: &[f32], lag: i32) -> f64 {
        let (reference, got) = aligned(reference, got, lag);
        let n = reference.len().min(got.len());
        if n == 0 {
            return f64::MIN;
        }
        let (mut signal, mut noise) = (0.0, 0.0);
        for index in 0..n {
            let sample = f64::from(reference[index]);
            let error = sample - f64::from(got[index]);
            signal += sample * sample;
            noise += error * error;
        }
        if noise == 0.0 {
            return 200.0;
        }
        10.0 * (signal / noise).log10()
    }

    fn aligned<'a>(reference: &'a [f32], got: &'a [f32], lag: i32) -> (&'a [f32], &'a [f32]) {
        let shift = usize::try_from(lag.unsigned_abs()).unwrap_or(0);
        if lag > 0 {
            let got = got.get(shift..).unwrap_or(&[]);
            let reference = reference
                .get(..reference.len().saturating_sub(shift))
                .unwrap_or(&[]);
            (reference, got)
        } else {
            let reference = reference.get(shift..).unwrap_or(&[]);
            let got = got.get(..got.len().saturating_sub(shift)).unwrap_or(&[]);
            (reference, got)
        }
    }

    /// A 5.1 AC-3 or E-AC-3 movie keeps each speaker, including LFE, at
    /// the start and after a seek. Surrounds stay side channels.
    #[test]
    fn a_surround_movie_keeps_each_speaker_through_seek() {
        use cutlass_decode::Channel::{
            self, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft, SideRight,
        };
        use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};

        let speakers = [
            (FrontLeft, 300.0),
            (FrontRight, 600.0),
            (FrontCenter, 900.0),
            (Lfe, 50.0),
            (SideLeft, 1_200.0),
            (SideRight, 1_500.0),
        ];
        let order: Vec<Channel> = speakers.iter().map(|(channel, _)| *channel).collect();
        for codec in [CodecId::Ac3, CodecId::Eac3] {
            let path = std::env::temp_dir().join(format!(
                "cutlass-readers-51-{codec:?}-{}.mp4",
                std::process::id()
            ));
            let rate = 48_000u32;
            let layout = ChannelLayout::new(order.clone()).expect("layout");
            let frames = rate as usize;
            let mut samples = Vec::with_capacity(frames * speakers.len());
            for index in 0..frames {
                let t = index as f32 / rate as f32;
                for (_, hz) in &speakers {
                    samples.push((t * *hz as f32 * std::f32::consts::TAU).sin() * 0.4);
                }
            }
            let bitrate = if codec == CodecId::Ac3 {
                448_000
            } else {
                384_000
            };
            let mut spec = CodecParams::audio(codec, rate, layout.clone());
            spec.bitrate = bitrate;
            let mut encoder = cutlass_ac3::Ac3Encoder::open(&spec).expect("encoder");
            let pcm = Pcm::from_f32(rate, layout, samples).expect("pcm");
            let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
            packets.extend(encoder.drain().expect("drain"));
            let mut mux =
                cutlass_isobmff::Mp4Mux::new(std::fs::File::create(&path).expect("create"));
            let params = StreamParams::try_audio(
                StreamCodec::Known(codec),
                rate,
                u16::try_from(speakers.len()).expect("channels"),
                Vec::new(),
            )
            .expect("params");
            let track = mux
                .add_stream(NewStream {
                    params,
                    timescale: Timescale::new(rate),
                    language: None,
                })
                .expect("stream");
            let mut ticks = 0i64;
            for packet in &packets {
                for side in &packet.side {
                    match side {
                        SideData::Config(config) => mux.set_config(track, config).expect("config"),
                        SideData::Gapless {
                            encoder_delay,
                            trailing_padding,
                            ..
                        } => mux
                            .set_trim(
                                track,
                                cutlass_container::Trim::new(
                                    Some(u64::from(*encoder_delay)),
                                    Some(u64::from(*trailing_padding)),
                                ),
                            )
                            .expect("trim"),
                        _ => {}
                    }
                }
                let timestamp =
                    Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                        .expect("time");
                let out = Packet::try_new(
                    track.get(),
                    packet.data.clone(),
                    timestamp,
                    Timescale::new(rate),
                    true,
                )
                .expect("packet");
                mux.write_packet(&out).expect("write");
                ticks += packet.duration;
            }
            mux.finish(&[TrackDuration {
                track,
                timescale: Timescale::new(rate),
                ticks,
            }])
            .expect("finish");
            drop(mux);

            let info = installed_readers().probe(&path).expect("probe");
            assert_eq!(info.streams[0].channels, Some(6), "{codec:?}");
            for at in [100_000, 500_000] {
                let played = installed_readers()
                    .open(&path)
                    .expect("open")
                    .audio_from(cutlass_model::Micros(at), 4_096)
                    .expect("audio");
                assert_eq!(
                    played.layout.channels(),
                    order.as_slice(),
                    "{codec:?} at {at}"
                );
                let floats = played.as_f32().expect("f32");
                let n = speakers.len();
                assert_eq!(floats.len(), n * 4_096, "{codec:?}");
                for (index, (channel, hz)) in speakers.iter().enumerate() {
                    let mine: Vec<f32> = floats.iter().skip(index).step_by(n).copied().collect();
                    let own = goertzel(&mine, rate, *hz);
                    for (_, other) in speakers.iter().filter(|(_, tone)| *tone != *hz) {
                        let leak = goertzel(&mine, rate, *other);
                        let margin = if *channel == Lfe { 10.0 } else { 50.0 };
                        assert!(
                            own > leak * margin,
                            "{codec:?} {channel:?} at {at} carries {other} Hz"
                        );
                    }
                }
            }
            installed_readers().open(&path).expect("reopen");
            std::fs::remove_file(&path).expect("remove");
        }
    }

    /// A 7.1 E-AC-3 movie keeps the side pair and the back pair after a seek.
    #[test]
    fn a_seven_point_one_movie_keeps_each_speaker_through_seek() {
        use cutlass_decode::Channel::{
            self, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft, SideRight,
        };
        use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};

        let speakers = [
            (FrontLeft, 300.0),
            (FrontRight, 600.0),
            (FrontCenter, 900.0),
            (Lfe, 50.0),
            (BackLeft, 1_200.0),
            (BackRight, 1_500.0),
            (SideLeft, 1_800.0),
            (SideRight, 2_100.0),
        ];
        let order: Vec<Channel> = speakers.iter().map(|(channel, _)| *channel).collect();
        let path =
            std::env::temp_dir().join(format!("cutlass-readers-71-{}.mp4", std::process::id()));
        let rate = 48_000u32;
        let layout = ChannelLayout::new(order.clone()).expect("layout");
        let frames = rate as usize;
        let mut samples = Vec::with_capacity(frames * speakers.len());
        for index in 0..frames {
            let t = index as f32 / rate as f32;
            for (_, hz) in &speakers {
                samples.push((t * *hz as f32 * std::f32::consts::TAU).sin() * 0.25);
            }
        }
        let mut spec = CodecParams::audio(CodecId::Eac3, rate, layout.clone());
        spec.bitrate = 1_024_000;
        let mut encoder = cutlass_ac3::Ac3Encoder::open(&spec).expect("encoder");
        let pcm = Pcm::from_f32(rate, layout, samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        let mut mux = cutlass_isobmff::Mp4Mux::new(std::fs::File::create(&path).expect("create"));
        let params = StreamParams::try_audio(
            StreamCodec::Known(CodecId::Eac3),
            rate,
            u16::try_from(speakers.len()).expect("channels"),
            Vec::new(),
        )
        .expect("params");
        let track = mux
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(rate),
                language: None,
            })
            .expect("stream");
        let mut ticks = 0i64;
        for packet in &packets {
            for side in &packet.side {
                match side {
                    SideData::Config(config) => mux.set_config(track, config).expect("config"),
                    SideData::Gapless {
                        encoder_delay,
                        trailing_padding,
                        ..
                    } => mux
                        .set_trim(
                            track,
                            cutlass_container::Trim::new(
                                Some(u64::from(*encoder_delay)),
                                Some(u64::from(*trailing_padding)),
                            ),
                        )
                        .expect("trim"),
                    _ => {}
                }
            }
            let timestamp =
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .expect("time");
            let out = Packet::try_new(
                track.get(),
                packet.data.clone(),
                timestamp,
                Timescale::new(rate),
                true,
            )
            .expect("packet");
            mux.write_packet(&out).expect("write");
            ticks += packet.duration;
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(rate),
            ticks,
        }])
        .expect("finish");
        drop(mux);

        let info = installed_readers().probe(&path).expect("probe");
        assert_eq!(info.streams[0].channels, Some(8));
        for at in [100_000, 500_000] {
            let played = installed_readers()
                .open(&path)
                .expect("open")
                .audio_from(cutlass_model::Micros(at), 4_096)
                .expect("audio");
            assert_eq!(played.layout.channels(), order.as_slice());
            let floats = played.as_f32().expect("f32");
            let n = speakers.len();
            assert_eq!(floats.len(), n * 4_096);
            for (index, (channel, hz)) in speakers.iter().enumerate() {
                let mine: Vec<f32> = floats.iter().skip(index).step_by(n).copied().collect();
                let own = goertzel(&mine, rate, *hz);
                for (_, other) in speakers.iter().filter(|(_, tone)| *tone != *hz) {
                    let leak = goertzel(&mine, rate, *other);
                    let margin = if *channel == Lfe { 10.0 } else { 50.0 };
                    assert!(
                        own > leak * margin,
                        "{channel:?} at {at} carries {other} Hz"
                    );
                }
            }
        }
        installed_readers().open(&path).expect("reopen");
        std::fs::remove_file(&path).expect("remove");
    }

    fn goertzel(samples: &[f32], rate: u32, hz: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * hz / f64::from(rate);
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for sample in samples {
            let s = f64::from(*sample) + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        s2 * s2 + s1 * s1 - coeff * s1 * s2
    }

    /// How far, in dB, the left channel's energy at `freq` sits above the
    /// energy an octave up.
    fn tone_margin(interleaved: &[f32], rate: u32, freq: f64) -> f64 {
        let left: Vec<f32> = interleaved.iter().step_by(2).copied().collect();
        let at = |hz: f64| {
            let w = 2.0 * std::f64::consts::PI * hz / f64::from(rate);
            let coeff = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for sample in &left {
                let s = f64::from(*sample) + coeff * s1 - s2;
                s2 = s1;
                s1 = s;
            }
            s2 * s2 + s1 * s1 - coeff * s1 * s2
        };
        10.0 * (at(freq) / at(freq * 2.0).max(1e-12)).log10()
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
