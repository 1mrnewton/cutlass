//! The containers and codecs a phone opens.
//!
//! Rust audio decoders (FLAC, ALAC, MP3, AC-3, E-AC-3, Opus and Vorbis) go
//! first, so a file decodes the same on every platform, and a lossless file
//! bit-exactly.
//! WebP goes first too: ImageIO gives one picture per packet and does not
//! keep an animation's canvas. The Apple decoders come next on an Apple
//! phone; a codec they decline falls through to the Rust decoder of the
//! same id.

use std::fs::File;
use std::path::Path;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::DecodeFactory;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_still::{StillDemux, still_limits};

/// ISOBMFF, AVI, WAV, AIFF, FLAC, Ogg (Opus and Vorbis), MP3, and stills,
/// decoded with [`decoders`].
#[must_use]
pub fn readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_avi::probe, open_avi);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_flac_container::probe, open_flac);
    readers.register_container(cutlass_ogg::probe, open_ogg);
    readers.register_container(cutlass_still::probe, open_still);
    readers.register_container(probe_mp3, open_mp3);
    *readers.decoders_mut() = decoders();
    readers
}

/// FLAC, ALAC, MP3, AC-3, E-AC-3, Opus, Vorbis, WebP and Motion JPEG, the
/// platform decoders for this OS, then PNG, GIF, JPEG, AAC, H.264, HEVC,
/// MPEG-2, MPEG-4 Visual, PCM and ProRes.
#[must_use]
pub fn decoders() -> DecodeFactory {
    let mut decoders = DecodeFactory::new();
    // Rust audio codecs go ahead of the platform ones, so FLAC, ALAC, MP3,
    // AC-3, E-AC-3, Opus and Vorbis decode the same on the phone as on the
    // desktop. WebP goes ahead too: ImageIO gives one picture per packet and
    // does not keep an animation's canvas. Motion JPEG goes ahead because
    // VideoToolbox refuses 4:2:2 frames after it opens and weaves no fields.
    cutlass_flac::register_decoders(&mut decoders);
    cutlass_alac::register_decoders(&mut decoders);
    cutlass_mp3::register_decoders(&mut decoders);
    cutlass_ac3::register_decoders(&mut decoders);
    cutlass_opus::register_decoders(&mut decoders);
    cutlass_vorbis::register_decoders(&mut decoders);
    cutlass_webp::register_decoders(&mut decoders);
    cutlass_mjpeg::register_decoders(&mut decoders);
    #[cfg(target_vendor = "apple")]
    cutlass_platform_apple::register_decoders(&mut decoders);
    cutlass_png::register_decoders(&mut decoders);
    cutlass_gif::register_decoders(&mut decoders);
    cutlass_jpeg::register_decoders(&mut decoders);
    cutlass_aac::register_decoders(&mut decoders);
    cutlass_h264::register_decoders(&mut decoders);
    // HEVC, MPEG-2 and MPEG-4 Visual in Rust, after the platform decoders.
    // On Apple the platform claims MPEG-4 Visual. Its opener here decodes
    // that codec when the platform opener is absent.
    cutlass_hevc::register_decoders(&mut decoders);
    cutlass_mpeg2::register_decoders(&mut decoders);
    cutlass_mpeg4::register_engine_decoders(&mut decoders);
    cutlass_pcm::register_decoders(&mut decoders);
    cutlass_prores::register_decoders(&mut decoders);
    decoders
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

fn open_avi(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = open_file(path)?;
    Ok(Box::new(cutlass_avi::AviDemux::open(file, length)?))
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
    use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
    use cutlass_container::{Mux, NewStream, TrackDuration};
    use cutlass_decode::{ChannelLayout, CodecParams, Decoder, Output};
    use cutlass_encode::{AudioInput, Encoder, Pcm};

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

    #[test]
    fn a_phone_opens_and_hears_an_mp3() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../codecs/mp3/tests/fixtures/sweep_22050_mono64.mp3");
        let mut media = super::readers().open(&path).expect("open");
        let ready = media.pull_audio().expect("pull").expect("pcm");
        assert_eq!(ready.pcm.rate, 22_050);
        assert_eq!(ready.pcm.layout.len(), 1);
    }

    /// Every sample `decoders` makes of the fixture's frames.
    fn decode_mp3(decoders: &cutlass_decode::DecodeFactory) -> Vec<f32> {
        use cutlass_mp3_container::{Demux, Mp3Demux};

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../codecs/mp3/tests/fixtures/sine_44100_j128_crc.mp3");
        let bytes = std::fs::read(path).expect("fixture");
        let mut demux = Mp3Demux::open(std::io::Cursor::new(bytes)).expect("demux");
        let stream = &demux.streams()[0].params;
        let rate = stream.sample_rate;
        let layout = if stream.channels == 1 {
            ChannelLayout::mono()
        } else {
            ChannelLayout::stereo()
        };
        let params = CodecParams::audio(CodecId::Mp3, rate, layout);
        let mut decoder = decoders.open(&params).expect("decoder");
        let mut pcm = Vec::new();
        let mut keep = |outputs: Vec<Output>| {
            for output in outputs {
                if let Output::Audio(audio) = output {
                    pcm.extend_from_slice(audio.pcm.as_f32().expect("f32"));
                }
            }
        };
        while let Some(packet) = demux.next_packet().expect("packet") {
            let packet =
                cutlass_decode::Packet::audio(packet.data, packet.pts, packet.duration, rate);
            keep(decoder.push(&packet).expect("decode"));
        }
        keep(decoder.drain().expect("drain"));
        pcm
    }

    #[test]
    fn mp3_is_decoded_by_the_rust_decoder_ahead_of_the_platform() {
        let mut rust = cutlass_decode::DecodeFactory::new();
        cutlass_mp3::register_decoders(&mut rust);
        let expected = decode_mp3(&rust);
        assert!(!expected.is_empty());
        assert!(
            decode_mp3(&super::decoders()) == expected,
            "the Rust decoder decodes MP3"
        );

        // With the platform first, an Apple phone decodes MP3 itself, and
        // differently: the check above would fail if the order were reversed.
        #[cfg(target_vendor = "apple")]
        {
            let mut reversed = cutlass_decode::DecodeFactory::new();
            cutlass_platform_apple::register_decoders(&mut reversed);
            cutlass_mp3::register_decoders(&mut reversed);
            assert!(
                decode_mp3(&reversed) != expected,
                "the platform decoder was not picked"
            );
        }
    }

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

    /// The phone encodes and decodes Vorbis with the Rust codec: the same
    /// packets and samples as `VorbisEncoder` and `VorbisDecoder` opened
    /// directly, so a platform codec registered ahead of it fails here.
    #[test]
    fn the_rust_vorbis_codec_goes_ahead_of_the_platform_one() {
        let mut spec = CodecParams::audio(CodecId::Vorbis, 48_000, ChannelLayout::stereo());
        spec.bitrate = 128_000;
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
        let packets = encode(&mut cutlass_vorbis::VorbisEncoder::open(&spec).expect("encoder"));
        let mut phone = crate::export::encoders()
            .open(&spec)
            .expect("phone encoder");
        assert_eq!(phone.codec(), CodecId::Vorbis);
        assert_eq!(encode(phone.as_mut()), packets);
        let mut listen = CodecParams::audio(CodecId::Vorbis, 48_000, ChannelLayout::stereo());
        listen.extra = packets
            .iter()
            .flat_map(|packet| &packet.side)
            .find_map(|side| match side {
                cutlass_encode::SideData::Config(config) => Some(config.clone()),
                _ => None,
            })
            .expect("vorbis headers");
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
        let mut rust = cutlass_vorbis::VorbisDecoder::open(&listen).expect("rust");
        let expected = decode(&mut rust);
        assert!(expected.iter().any(|sample| sample.abs() > 0.1));
        let mut phone = super::decoders().open(&listen).expect("phone");
        assert_eq!(phone.codec(), CodecId::Vorbis);
        assert_eq!(decode(phone.as_mut()), expected);
    }

    /// The phone decodes ALAC with the Rust decoder: the same buffers as
    /// `AlacDecoder`, a 5.1 stream in ALAC's own channel order, and the same
    /// refusal of a damaged frame. On an Apple phone the platform decoder,
    /// put first, decodes the same samples but hands back sound for the
    /// damaged frame, so this fails if the order is reversed.
    #[test]
    fn the_rust_alac_decoder_goes_ahead_of_the_platform_one() {
        let layout = ChannelLayout::five_one();
        let mut spec = CodecParams::audio(CodecId::Alac, 48_000, layout.clone());
        spec.sample_format = Some(cutlass_decode::SampleFormat::I16);
        let samples: Vec<i16> = (0..6 * 9000)
            .map(|i| ((((i % 6) + 1) as f32 * (i / 6) as f32 * 0.011).sin() * 9000.0) as i16)
            .collect();
        let input = AudioInput {
            pcm: Pcm::from_i16(48_000, layout, samples).expect("pcm"),
        };
        let mut encoder = cutlass_alac::AlacEncoder::open(&spec).expect("encoder");
        let mut packets = encoder.push(&input).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        spec.extra = encoder.cookie().to_bytes();
        let decode = |decoder: &mut dyn Decoder| {
            let mut out = Vec::new();
            for packet in &packets {
                for output in decoder.push(packet).expect("push") {
                    if let Output::Audio(audio) = output {
                        out.push(audio.pcm);
                    }
                }
            }
            out
        };
        let damaged = cutlass_decode::Packet::audio(vec![0xff; 64], 0, 4096, 48_000);

        let mut rust = cutlass_alac::AlacDecoder::open(&spec).expect("rust");
        let expected = decode(&mut rust);
        assert!(!expected.is_empty());
        assert!(rust.push(&damaged).is_err(), "the Rust decoder refuses it");
        let mut phone = super::decoders().open(&spec).expect("phone");
        assert!(
            decode(phone.as_mut()) == expected,
            "the Rust decoder decodes ALAC"
        );
        assert!(
            phone.push(&damaged).is_err(),
            "the damaged frame is refused"
        );
        #[cfg(target_vendor = "apple")]
        {
            let mut reversed = cutlass_decode::DecodeFactory::new();
            cutlass_platform_apple::register_decoders(&mut reversed);
            cutlass_alac::register_decoders(&mut reversed);
            let mut platform = reversed.open(&spec).expect("platform");
            assert!(
                platform.push(&damaged).is_ok(),
                "the platform decoder was not picked"
            );
        }
    }

    fn ac3_round_trip(
        encoder: &mut dyn Encoder,
        decoder: &mut dyn Decoder,
    ) -> (Vec<Vec<u8>>, Vec<f32>) {
        let rate = 48_000u32;
        let frames = 1536 * 2;
        let samples: Vec<f32> = (0..frames)
            .flat_map(|i| {
                let t = i as f32 / rate as f32;
                [
                    0.3 * (t * 440.0 * std::f32::consts::TAU).sin(),
                    0.2 * (t * 660.0 * std::f32::consts::TAU).sin(),
                ]
            })
            .collect();
        let pcm = Pcm::from_f32(rate, ChannelLayout::stereo(), samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("encode");
        packets.extend(encoder.drain().expect("drain"));
        let mut decoded = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).expect("decode") {
                if let Output::Audio(audio) = output {
                    decoded.extend_from_slice(audio.pcm.as_f32().expect("f32"));
                }
            }
        }
        for output in decoder.drain().expect("drain") {
            if let Output::Audio(audio) = output {
                decoded.extend_from_slice(audio.pcm.as_f32().expect("f32"));
            }
        }
        let bytes = packets.into_iter().map(|packet| packet.data).collect();
        (bytes, decoded)
    }

    #[test]
    fn the_phone_codes_ac3_and_eac3_with_the_rust_codec() {
        for codec in [CodecId::Ac3, CodecId::Eac3] {
            let mut spec = CodecParams::audio(codec, 48_000, ChannelLayout::stereo());
            spec.bitrate = 192_000;
            let mut rust_encoder = cutlass_ac3::Ac3Encoder::open(&spec).expect("rust encoder");
            let mut rust_decoder = cutlass_ac3::Ac3Decoder::open(&spec).expect("rust decoder");
            let rust = ac3_round_trip(&mut rust_encoder, &mut rust_decoder);
            assert!(!rust.0.is_empty() && !rust.1.is_empty(), "{codec:?}");
            let mut phone_encoder = crate::export::encoders()
                .open(&spec)
                .expect("phone encoder");
            let mut phone_decoder = super::decoders().open(&spec).expect("phone decoder");
            let phone = ac3_round_trip(phone_encoder.as_mut(), phone_decoder.as_mut());
            assert!(phone.0 == rust.0, "{codec:?} encodes with the Rust encoder");
            assert!(phone.1 == rust.1, "{codec:?} decodes with the Rust decoder");

            #[cfg(target_vendor = "apple")]
            {
                // AudioToolbox registers both ids. When it accepts the
                // parameters its bytes and samples differ, so a reversed
                // registration fails here. A refusal falls through to Rust.
                let mut only = cutlass_encode::EncodeFactory::new();
                cutlass_platform_apple::register_encoders(&mut only);
                if let Ok(mut platform) = only.open(&spec) {
                    let mut again = cutlass_ac3::Ac3Decoder::open(&spec).expect("decoder");
                    let encoded = ac3_round_trip(platform.as_mut(), &mut again);
                    assert!(
                        encoded.0 != rust.0,
                        "{codec:?} platform encoder matched the Rust encoder"
                    );
                }
                let mut only = cutlass_decode::DecodeFactory::new();
                cutlass_platform_apple::register_decoders(&mut only);
                let mut platform = only.open(&spec).expect("platform decoder");
                let mut rust_encoder = cutlass_ac3::Ac3Encoder::open(&spec).expect("encoder");
                let decoded = ac3_round_trip(&mut rust_encoder, platform.as_mut());
                assert!(
                    decoded.1 != rust.1,
                    "{codec:?} platform decoder matched the Rust decoder"
                );
            }
        }
    }

    /// The phone plays MPEG-2: the platform decoder where it takes the
    /// stream, else the Rust one, gives every picture of a stream with B
    /// pictures once, in display order.
    #[test]
    fn the_phone_decodes_mpeg2() {
        use cutlass_encode::{VideoEncoder, VideoInput};
        let (width, height) = (64, 48);
        let mut encoder =
            cutlass_mpeg2::Mpeg2Encoder::open(&CodecParams::video(CodecId::Mpeg2, width, height))
                .expect("encoder");
        let mut packets = Vec::new();
        for t in 0..8u8 {
            let luma: Vec<u8> = (0..width * height)
                .map(|i| (i % width) as u8 * 3 + t * 4)
                .collect();
            let chroma = vec![128; (width * height / 4) as usize];
            let picture =
                cutlass_decode::Yuv420p::try_new(width, height, luma, chroma.clone(), chroma)
                    .expect("picture");
            let input = VideoInput {
                picture,
                pts: i64::from(t),
                duration: 1,
                timescale: 25,
                force_keyframe: false,
            };
            packets.extend(encoder.push(&input).expect("push"));
        }
        packets.extend(VideoEncoder::drain(&mut encoder).expect("drain"));
        let mut params = CodecParams::video(CodecId::Mpeg2, width, height);
        params.extra = encoder.extradata().to_vec();
        let mut phone = super::decoders().open(&params).expect("phone");
        let mut times = Vec::new();
        for packet in &packets {
            for output in phone.push(packet).expect("push") {
                if let Output::Picture(picture) = output {
                    times.push(picture.pts());
                }
            }
        }
        for output in phone.drain().expect("drain") {
            if let Output::Picture(picture) = output {
                times.push(picture.pts());
            }
        }
        assert_eq!(times, (0..8).collect::<Vec<i64>>());
    }

    /// A lossless 1x1 WebP of one opaque red pixel, as `cwebp -lossless` 1.6.0
    /// wrote it.
    const RED: [u8; 36] = [
        0x52, 0x49, 0x46, 0x46, 0x1c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4c, 0x0f, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x00, 0x07, 0x10, 0xfd, 0x8f, 0xfe,
        0x07, 0x22, 0xa2, 0xff, 0x01, 0x00,
    ];

    #[test]
    fn the_phone_decodes_webp_in_rust() {
        let mut decoder = super::decoders()
            .open(&CodecParams::video(CodecId::Webp, 1, 1))
            .expect("webp decoder");
        let packet = cutlass_decode::Packet::video(RED.to_vec(), 0, 0, 0, true);
        let pictures = decoder.decode_picture(&packet).expect("decode");
        assert_eq!(pictures[0].image.samples[..3], [255, 0, 0]);
    }

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
        let params =
            CodecParams::with_format(CodecId::Webp, 40, 30, cutlass_decode::PixelFormat::Rgba8);
        let mut decoder = super::decoders().open(&params).expect("webp");
        assert_eq!(Decoder::codec(decoder.as_ref()), CodecId::Webp);
        let mut frames = 0;
        for data in [&file[..second], &file[second..]] {
            for output in decoder
                .push(&cutlass_decode::Packet::audio(data.to_vec(), 0, 0, 1000))
                .expect("packet")
            {
                let Output::Picture(picture) = output else {
                    panic!("picture");
                };
                assert!(matches!(
                    picture.samples(),
                    cutlass_decode::PictureSamples::Cpu(_)
                ));
                assert_eq!((picture.info().width, picture.info().height), (40, 30));
                frames += 1;
            }
        }
        assert_eq!(frames, 6);
    }

    /// A phone decodes and encodes Motion JPEG with the Rust codec: the
    /// same RGBA and the same packet bytes. The frame is 4:2:2, which the
    /// Apple platform decoder, put first, opens for and then refuses.
    #[test]
    fn a_phone_codes_motion_jpeg_in_rust_ahead_of_the_platform() {
        use cutlass_decode::PixelFormat;
        use cutlass_encode::{CpuFrame, FrameInput, FrameTime};

        let rgba: Vec<u8> = (0..64u32 * 48)
            .flat_map(|i| [(i % 64 * 4) as u8, (i / 64 * 5) as u8, 90, 255])
            .collect();
        let params = CodecParams::with_format(CodecId::Mjpeg, 64, 48, PixelFormat::Rgba8);
        let frame = cutlass_mjpeg::MjpegEncoder::open(&params)
            .expect("encoder")
            .with_sampling(cutlass_mjpeg::Sampling::Yuv422)
            .encode_rgba(64, 48, &rgba, 1, 30)
            .expect("frame");
        let packet = cutlass_decode::Packet::video(frame, 0, 1, 30, true);
        let decode = |decoders: &cutlass_decode::DecodeFactory| {
            let mut decoder = decoders.open(&params)?;
            let mut pictures = Vec::new();
            for output in decoder.push(&packet)? {
                if let Output::Picture(picture) = output {
                    pictures.push(picture.into_cpu().ok_or(
                        cutlass_decode::CodecError::InvalidApi {
                            what: "not a cpu picture",
                        },
                    )?);
                }
            }
            Ok::<_, cutlass_decode::CodecError>(pictures)
        };
        let mut rust = cutlass_decode::DecodeFactory::new();
        cutlass_mjpeg::register_decoders(&mut rust);
        let expected = decode(&rust).expect("rust");
        assert_eq!(expected.len(), 1);
        assert_eq!(decode(&super::decoders()).expect("phone"), expected);

        let time = FrameTime::new(0, 1, 30).expect("time");
        let input = FrameInput::Cpu(
            CpuFrame::new(64, 48, PixelFormat::Rgba8, rgba.clone(), time).expect("frame"),
        );
        let mut phone = crate::export::encoders()
            .open(&params)
            .expect("phone encoder");
        let packets = phone.push_frame(&input).expect("encode");
        let ours = cutlass_mjpeg::MjpegEncoder::open(&params)
            .expect("encoder")
            .encode_rgba(64, 48, &rgba, 1, 30)
            .expect("frame");
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data, ours);

        #[cfg(target_vendor = "apple")]
        {
            let mut reversed = cutlass_decode::DecodeFactory::new();
            cutlass_platform_apple::register_decoders(&mut reversed);
            cutlass_mjpeg::register_decoders(&mut reversed);
            assert!(
                decode(&reversed).ok() != Some(expected),
                "the platform decoder was not picked"
            );
        }
    }
}
