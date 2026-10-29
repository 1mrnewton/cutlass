//! Behavior locked for the PCM codec. Expected samples are written here,
//! not taken from the encoder under test.

use std::sync::atomic::{AtomicU64, Ordering};

static ORACLE_DIR: AtomicU64 = AtomicU64::new(0);

use cutlass_decode::{
    Channel, ChannelLayout, CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output,
    Packet, Pcm, SampleFormat, SideData,
};
use cutlass_encode::{AudioInput, EncodeFactory, Encoder};

use crate::limit::{self, MAX_RATE};
use crate::{
    Arrangement, Coding, Endian, Layout, PcmDecoder, PcmEncoder, register_decoders,
    register_encoders,
};

fn layout(coding: Coding, endian: Endian, arrangement: Arrangement) -> Layout {
    Layout::new(coding, endian, arrangement)
}

fn params(spec: Layout, rate: u32, channels: ChannelLayout) -> CodecParams {
    let mut params = CodecParams::audio(CodecId::Pcm, rate, channels);
    params.extra = spec.to_extra().to_vec();
    params.sample_format = Some(spec.coding().output_format());
    params
}

fn open_decoder(spec: Layout, channels: ChannelLayout) -> PcmDecoder {
    PcmDecoder::open(&params(spec, 48_000, channels)).expect("decoder")
}

fn open_encoder(spec: Layout, channels: ChannelLayout) -> PcmEncoder {
    PcmEncoder::open(&params(spec, 48_000, channels)).expect("encoder")
}

fn audio(decoder: &mut PcmDecoder, bytes: Vec<u8>) -> Result<Pcm, CodecError> {
    let packet = Packet::audio(bytes, 5, 1, 48_000);
    let mut outputs = decoder.push(&packet)?;
    let Some(Output::Audio(audio)) = outputs.pop() else {
        return Err(CodecError::InvalidApi { what: "audio" });
    };
    Ok(audio.pcm)
}

fn le_i16(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn unit(value: i32, full: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        (f64::from(value) / full) as f32
    }
}

#[test]
fn s16le_interleaved_is_the_mix_sample() {
    let bytes = le_i16(&[0, 32_767, -32_768, -1]);
    let mut decoder = open_decoder(
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        ChannelLayout::stereo(),
    );
    let pcm = audio(&mut decoder, bytes).expect("decode");
    assert_eq!(pcm.format, SampleFormat::I16);
    assert_eq!(pcm.samples, 2);
    assert_eq!(pcm.as_i16(), Some(&[0, 32_767, -32_768, -1][..]));
    assert!(pcm.as_f32().is_none());
}

#[test]
fn an_independent_s16_stream_is_not_produced_by_this_encoder() {
    let bytes = vec![0x02, 0x01, 0x00, 0x80];
    let mut decoder = open_decoder(
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let pcm = audio(&mut decoder, bytes).expect("decode");
    assert_eq!(pcm.format, SampleFormat::I16);
    assert_eq!(pcm.samples, 2);
    assert_eq!(pcm.as_i16(), Some(&[0x0102, i16::MIN][..]));
}

#[test]
fn endian_is_a_parameter_for_integer_pcm() {
    let mut be = open_decoder(
        layout(Coding::S16, Endian::Big, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let pcm = audio(&mut be, vec![0x01, 0x02]).expect("be");
    assert_eq!(pcm.as_i16(), Some(&[0x0102][..]));

    let mut s24 = open_decoder(
        layout(Coding::S24, Endian::Little, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let pcm = audio(&mut s24, vec![0xFF, 0xFF, 0x7F]).expect("s24 max");
    let samples = pcm.as_f32().expect("f32");
    assert_eq!(pcm.samples, 1);
    assert_eq!(samples, &[1.0]);
    let pcm = audio(&mut s24, vec![0x00, 0x00, 0x80]).expect("s24 min");
    let min = pcm.as_f32().expect("f32")[0];
    assert!(min < -1.0, "s24 minimum was clipped to {min}");
    assert_eq!(min, unit(-8_388_608, limit::S24_FULL));

    let mut s32 = open_decoder(
        layout(Coding::S32, Endian::Big, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let pcm = audio(&mut s32, i32::MAX.to_be_bytes().to_vec()).expect("s32");
    assert_eq!(pcm.format, SampleFormat::F32);
    assert_eq!(pcm.samples, 1);
    assert_eq!(pcm.as_f32(), Some(&[1.0][..]));
    let pcm = audio(&mut s32, 1_000_000i32.to_be_bytes().to_vec()).expect("mid");
    assert_eq!(pcm.as_f32(), Some(&[unit(1_000_000, limit::S32_FULL)][..]));
    let pcm = audio(&mut s32, i32::MIN.to_be_bytes().to_vec()).expect("s32 min");
    assert_eq!(pcm.as_f32(), Some(&[-1.0][..]));
}

#[test]
fn a_non_finite_float_is_an_error_and_a_loud_finite_sample_is_kept() {
    let spec = layout(Coding::F32, Endian::Little, Arrangement::Interleaved);
    let mut decoder = open_decoder(spec, ChannelLayout::mono());
    let err = audio(&mut decoder, f32::NAN.to_le_bytes().to_vec()).expect_err("nan");
    assert_eq!(err, CodecError::NonFinite);
    let err = audio(&mut decoder, f32::INFINITY.to_le_bytes().to_vec()).expect_err("inf");
    assert_eq!(err, CodecError::NonFinite);
    let pcm = audio(&mut decoder, 2.0f32.to_le_bytes().to_vec()).expect("loud");
    assert_eq!(pcm.as_f32().expect("f32")[0].to_bits(), 2.0f32.to_bits());

    let mut wide = open_decoder(
        layout(Coding::F64, Endian::Big, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let err = audio(&mut wide, f64::NAN.to_be_bytes().to_vec()).expect_err("f64 nan");
    assert_eq!(err, CodecError::NonFinite);
    let err = audio(&mut wide, f64::MAX.to_be_bytes().to_vec()).expect_err("overflow");
    assert_eq!(err, CodecError::NonFinite);
    let pcm = audio(&mut wide, f64::from(-0.0f32).to_be_bytes().to_vec()).expect("neg zero");
    assert_eq!(pcm.samples, 1);
    assert_eq!(pcm.as_f32().expect("f32")[0].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn planar_pcm_decodes_to_interleaved_frames() {
    let mut decoder = open_decoder(
        layout(Coding::S16, Endian::Little, Arrangement::Planar),
        ChannelLayout::stereo(),
    );
    let mut bytes = le_i16(&[1, 2]);
    bytes.extend(le_i16(&[3, 4]));
    let pcm = audio(&mut decoder, bytes).expect("planar");
    assert_eq!(pcm.format, SampleFormat::I16);
    assert_eq!(pcm.samples, 2);
    assert_eq!(pcm.as_i16(), Some(&[1, 3, 2, 4][..]));

    let mut floats = open_decoder(
        layout(Coding::F32, Endian::Little, Arrangement::Planar),
        ChannelLayout::stereo(),
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0.25f32.to_le_bytes());
    bytes.extend_from_slice(&(-0.5f32).to_le_bytes());
    let pcm = audio(&mut floats, bytes).expect("planar f32");
    assert_eq!(pcm.samples, 1);
    assert_eq!(pcm.as_f32(), Some(&[0.25, -0.5][..]));
}

#[test]
fn a_partial_frame_is_an_error_and_not_silence() {
    let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
    let mut decoder = open_decoder(spec, ChannelLayout::stereo());
    let err = audio(&mut decoder, vec![0x01, 0x00, 0x02]).expect_err("partial");
    assert!(matches!(
        err,
        CodecError::Truncated {
            what: "pcm frame",
            ..
        }
    ));
    let err = audio(&mut decoder, Vec::new()).expect_err("empty");
    assert_eq!(
        err,
        CodecError::InvalidApi {
            what: "empty packet"
        }
    );
    let pcm = audio(&mut decoder, le_i16(&[7, 8])).expect("next");
    assert_eq!(pcm.as_i16(), Some(&[7, 8][..]));
    assert_ne!(pcm.as_i16(), Some(&[0, 0][..]));
}

#[test]
fn a_short_buffer_and_an_overflowing_length_are_rejected() {
    let mut decoder = open_decoder(
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let err = audio(&mut decoder, vec![0x7F]).expect_err("short");
    assert!(matches!(err, CodecError::Truncated { .. }));
    let err = limit::reserve::<u16>(usize::MAX).expect_err("overflow");
    assert_eq!(err, CodecError::Limit { what: "allocation" });
    let err = limit::packet_len(limit::MAX_PACKET + 1).expect_err("cap");
    assert_eq!(
        err,
        CodecError::Limit {
            what: "packet bytes"
        }
    );
    let pcm = audio(&mut decoder, le_i16(&[9])).expect("recovered");
    assert_eq!(pcm.as_i16(), Some(&[9][..]));
}

#[test]
fn a_failed_frame_does_not_poison_the_next_frame() {
    let spec = layout(Coding::F32, Endian::Little, Arrangement::Interleaved);
    let mut decoder = open_decoder(spec, ChannelLayout::mono());
    let mut packet = Packet::audio(f32::NAN.to_le_bytes().to_vec(), 0, 1, 48_000);
    packet.side.push(SideData::Config(vec![1, 0, 0]));
    assert_eq!(
        decoder.push(&packet).expect_err("config"),
        CodecError::Unsupported {
            what: "configuration change"
        }
    );
    assert_eq!(
        decoder
            .push(&Packet::audio(
                f32::NAN.to_le_bytes().to_vec(),
                0,
                1,
                48_000
            ))
            .expect_err("nan"),
        CodecError::NonFinite
    );
    let outputs = decoder
        .push(&Packet::audio(0.5f32.to_le_bytes().to_vec(), 3, 1, 48_000))
        .expect("next");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio");
    };
    assert_eq!(audio.pts, 3);
    assert_eq!(audio.pcm.as_f32(), Some(&[0.5][..]));
    assert!(decoder.drain().expect("drain").is_empty());
    assert!(decoder.drain().expect("second").is_empty());
    decoder.reset();
    let outputs = decoder
        .push(&Packet::audio(0.25f32.to_le_bytes().to_vec(), 0, 1, 48_000))
        .expect("after reset");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio");
    };
    assert_eq!(audio.pcm.samples, 1);
}

#[test]
fn the_decoder_is_registered_and_refuses_a_bad_open() {
    let mut factory = DecodeFactory::new();
    register_decoders(&mut factory);
    register_decoders(&mut factory);
    let opened = factory
        .open(&params(
            layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
            48_000,
            ChannelLayout::mono(),
        ))
        .expect("open");
    assert_eq!(Decoder::codec(opened.as_ref()), CodecId::Pcm);

    let mut wrong = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
    wrong.codec = CodecId::Pcm;
    wrong.sample_rate = 0;
    assert!(matches!(
        PcmDecoder::open(&wrong),
        Err(CodecError::InvalidApi {
            what: "sample rate"
        })
    ));
    wrong.sample_rate = MAX_RATE + 1;
    assert!(matches!(
        PcmDecoder::open(&wrong),
        Err(CodecError::Limit {
            what: "sample rate"
        })
    ));
    let mut missing = params(
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        48_000,
        ChannelLayout::mono(),
    );
    missing.layout = None;
    assert!(matches!(
        PcmDecoder::open(&missing),
        Err(CodecError::InvalidApi {
            what: "channel layout"
        })
    ));
    let err = PcmDecoder::open(&CodecParams::audio(
        CodecId::Aac,
        48_000,
        ChannelLayout::mono(),
    ))
    .expect_err("codec");
    assert_eq!(err, CodecError::Unsupported { what: "pcm" });
    let eight = ChannelLayout::new(vec![Channel::FrontLeft; 8]).expect("layout");
    let mut wide = open_decoder(
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        eight,
    );
    let pcm = audio(&mut wide, le_i16(&[1, 2, 3, 4, 5, 6, 7, 8])).expect("eight");
    assert_eq!(pcm.samples, 1);
    assert_eq!(pcm.as_i16().expect("i16").len(), 8);
}

#[test]
fn encode_round_trips_the_original_samples() {
    let spec = layout(Coding::S16, Endian::Big, Arrangement::Planar);
    let channels = ChannelLayout::stereo();
    let source =
        Pcm::from_i16(48_000, channels.clone(), vec![0, -32_768, 32_767, -1]).expect("pcm");
    let mut encoder = open_encoder(spec, channels.clone());
    let packets = encoder
        .push(&AudioInput {
            pcm: source.clone(),
        })
        .expect("encode");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].pts, 0);
    assert_eq!(packets[0].duration, 2);
    assert_eq!(packets[0].timescale, 48_000);
    assert!(packets[0].keyframe);
    let mut decoder = open_decoder(spec, channels);
    let outputs = decoder.push(&packets[0]).expect("decode");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio");
    };
    assert_eq!(audio.pcm, source);

    let spec = layout(Coding::F32, Endian::Little, Arrangement::Interleaved);
    let source = Pcm::from_f32(
        48_000,
        ChannelLayout::mono(),
        vec![-0.0, 0.5, 1.5, f32::from_bits(1)],
    )
    .expect("pcm");
    let mut encoder = open_encoder(spec, ChannelLayout::mono());
    let packets = encoder
        .push(&AudioInput {
            pcm: source.clone(),
        })
        .expect("encode");
    let mut decoder = open_decoder(spec, ChannelLayout::mono());
    let outputs = decoder.push(&packets[0]).expect("decode");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio");
    };
    let got = audio.pcm.as_f32().expect("f32");
    let want = source.as_f32().expect("f32");
    assert_eq!(got.len(), want.len());
    for (got, want) in got.iter().zip(want) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    let spec = layout(Coding::F64, Endian::Little, Arrangement::Interleaved);
    let mut encoder = open_encoder(spec, ChannelLayout::mono());
    let packets = encoder
        .push(&AudioInput {
            pcm: source.clone(),
        })
        .expect("f64");
    assert_eq!(packets[0].data.len(), 4 * 8);
    let mut decoder = open_decoder(spec, ChannelLayout::mono());
    let outputs = decoder.push(&packets[0]).expect("decode f64");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio");
    };
    let got = audio.pcm.as_f32().expect("f32");
    for (got, want) in got.iter().zip(want) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
}

#[test]
fn export_selects_the_encoder_by_codec_id() {
    let mut factory = EncodeFactory::new();
    register_encoders(&mut factory);
    let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
    let opened = factory
        .open(&params(spec, 48_000, ChannelLayout::mono()))
        .expect("pcm encoder");
    assert_eq!(Encoder::codec(opened.as_ref()), CodecId::Pcm);
    let err = PcmEncoder::open(&params(
        layout(Coding::S24, Endian::Little, Arrangement::Interleaved),
        48_000,
        ChannelLayout::mono(),
    ))
    .expect_err("s24");
    assert_eq!(err, CodecError::Unsupported { what: "s24 encode" });
    let err = PcmEncoder::open(&params(
        layout(Coding::S32, Endian::Little, Arrangement::Interleaved),
        48_000,
        ChannelLayout::mono(),
    ))
    .expect_err("s32");
    assert_eq!(err, CodecError::Unsupported { what: "s32 encode" });
}

#[test]
fn a_bad_encode_adds_nothing_and_the_next_push_still_works() {
    let spec = layout(Coding::F32, Endian::Little, Arrangement::Interleaved);
    let mut encoder = open_encoder(spec, ChannelLayout::mono());
    let nan = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![f32::NAN]).expect("nan");
    assert_eq!(
        encoder.push(&AudioInput { pcm: nan }),
        Err(CodecError::NonFinite)
    );
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    let sample = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.25]).expect("sample");
    assert_eq!(
        encoder.push(&AudioInput { pcm: sample }),
        Err(CodecError::Cancelled)
    );
    encoder.cancel_flag().store(false, Ordering::Relaxed);
    let wrong = Pcm::from_f32(44_100, ChannelLayout::mono(), vec![0.25]).expect("rate");
    assert_eq!(
        encoder.push(&AudioInput { pcm: wrong }),
        Err(CodecError::InvalidApi {
            what: "audio format"
        })
    );
    let sample = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.25, -0.5]).expect("ok");
    let packets = encoder.push(&AudioInput { pcm: sample }).expect("push");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].pts, 0);
    assert_eq!(packets[0].duration, 2);
    let more = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![1.0]).expect("more");
    let packets = encoder.push(&AudioInput { pcm: more }).expect("second");
    assert_eq!(packets[0].pts, 2);
    encoder.reset();
    let again = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.5]).expect("again");
    let packets = encoder.push(&AudioInput { pcm: again }).expect("reset");
    assert_eq!(packets[0].pts, 0);
    assert!(encoder.drain().expect("drain").is_empty());
    let empty = Pcm::from_f32(48_000, ChannelLayout::mono(), Vec::new()).expect("empty");
    assert!(
        encoder
            .push(&AudioInput { pcm: empty })
            .expect("empty")
            .is_empty()
    );
    assert_eq!(
        encoder.push_frame(&cutlass_encode::FrameInput::Platform(
            cutlass_encode::PlatformFrame::new(
                1,
                1,
                1,
                cutlass_encode::FrameTime::new(0, 1, 1).expect("time")
            )
            .expect("frame")
        )),
        Err(CodecError::Unsupported { what: "picture" })
    );
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 7;
        state ^= state >> 9;
        state ^= state << 8;
        self.0 = state;
        state
    }

    fn index(&mut self, len: usize) -> usize {
        let width = u64::try_from(len).unwrap_or(1).max(1);
        usize::try_from(self.next() % width).unwrap_or(0)
    }

    fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[0]
    }

    fn sample(&mut self) -> i16 {
        let bytes = self.next().to_le_bytes();
        i16::from_le_bytes([bytes[0], bytes[1]])
    }
}

#[test]
fn random_packets_fail_closed_and_round_trips_keep_every_sample() {
    let mut rng = Rng(0x5043_4D00);
    let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
    for _ in 0..32 {
        let frames = rng.index(64) + 1;
        let samples: Vec<i16> = (0..frames).map(|_| rng.sample()).collect();
        let source = Pcm::from_i16(48_000, ChannelLayout::mono(), samples).expect("pcm");
        let mut encoder = open_encoder(spec, ChannelLayout::mono());
        let packets = encoder
            .push(&AudioInput {
                pcm: source.clone(),
            })
            .expect("encode");
        let mut decoder = open_decoder(spec, ChannelLayout::mono());
        let outputs = decoder.push(&packets[0]).expect("decode");
        let Output::Audio(audio) = &outputs[0] else {
            panic!("audio");
        };
        assert_eq!(audio.pcm, source);
    }
    let mut decoder = open_decoder(spec, ChannelLayout::mono());
    for _ in 0..64 {
        let len = rng.index(24);
        let bytes: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        let result = decoder.push(&Packet::audio(bytes, 0, 1, 48_000));
        if len == 0 || !len.is_multiple_of(2) {
            assert!(result.is_err());
        }
    }
    let pcm = audio(&mut decoder, le_i16(&[11])).expect("alive");
    assert_eq!(pcm.as_i16(), Some(&[11][..]));
}

fn ffmpeg_bin() -> Option<std::path::PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let path = std::path::PathBuf::from(candidate);
        let ok = std::process::Command::new(&path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success());
        if ok {
            return Some(path);
        }
    }
    None
}

fn require_or_skip(tool: &str) -> bool {
    if std::env::var_os("PCM_REQUIRE_ORACLES").is_some() {
        panic!("{tool} is required");
    }
    eprintln!("skipped {tool} oracle: {tool} was not found");
    false
}

fn ffmpeg_convert(
    ffmpeg: &std::path::Path,
    input_format: &str,
    output_format: &str,
    channels: u8,
    bytes: &[u8],
) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-pcm-oracle-{}-{}",
        std::process::id(),
        ORACLE_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("temp");
    let input = dir.join("in.pcm");
    let output = dir.join("out.pcm");
    std::fs::write(&input, bytes).expect("write");
    let status = std::process::Command::new(ffmpeg)
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            input_format,
            "-ar",
            "48000",
            "-ac",
        ])
        .arg(channels.to_string())
        .arg("-i")
        .arg(&input)
        .args(["-f", output_format])
        .arg(&output)
        .output()
        .expect("ffmpeg");
    assert!(
        status.status.success(),
        "{input_format} -> {output_format} {status:?}"
    );
    let decoded = std::fs::read(&output).expect("read");
    let _ = std::fs::remove_dir_all(&dir);
    decoded
}

#[test]
fn ffmpeg_reads_an_independent_bitstream() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        if !require_or_skip("ffmpeg") {
            return;
        }
        return;
    };
    let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
    let source = Pcm::from_i16(
        48_000,
        ChannelLayout::mono(),
        vec![0, 1000, -32_768, 32_767],
    )
    .expect("pcm");
    let mut encoder = open_encoder(spec, ChannelLayout::mono());
    let packets = encoder
        .push(&AudioInput {
            pcm: source.clone(),
        })
        .expect("encode");
    let foreign = ffmpeg_convert(&ffmpeg, "s16le", "s16le", 1, &packets[0].data);
    assert_eq!(foreign, packets[0].data);
    assert_eq!(foreign.len() / 2, source.as_i16().expect("i16").len());

    let spec = layout(Coding::F32, Endian::Little, Arrangement::Interleaved);
    let source = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.0, -0.25, 2.0]).expect("pcm");
    let mut encoder = open_encoder(spec, ChannelLayout::mono());
    let packets = encoder.push(&AudioInput { pcm: source }).expect("encode");
    let foreign = ffmpeg_convert(&ffmpeg, "f32le", "f32le", 1, &packets[0].data);
    assert_eq!(foreign, packets[0].data);

    let packed = vec![0x00, 0x00, 0x00, 0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80];
    let foreign = ffmpeg_convert(&ffmpeg, "s24le", "s32le", 1, &packed);
    assert_eq!(foreign.len(), 12);
    let mut ints = Vec::new();
    for chunk in foreign.as_chunks::<4>().0 {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        ints.push(i32::from_le_bytes(bytes) >> 8);
    }
    let mut decoder = open_decoder(
        layout(Coding::S24, Endian::Little, Arrangement::Interleaved),
        ChannelLayout::mono(),
    );
    let pcm = audio(&mut decoder, packed).expect("s24");
    let got = pcm.as_f32().expect("f32");
    assert_eq!(pcm.samples, 3);
    assert_eq!(got.len(), ints.len());
    for (got, integer) in got.iter().zip(ints) {
        assert_eq!(*got, unit(integer, limit::S24_FULL));
    }
}

fn local_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files")
}

fn ffprobe_audio(ffprobe: &std::path::Path, media: &std::path::Path) -> Option<(u32, usize)> {
    let output = std::process::Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0:s=x",
        ])
        .arg(media)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let mut parts = text.trim().split('x');
    let rate: u32 = parts.next()?.parse().ok()?;
    let channels: usize = parts.next()?.parse().ok()?;
    if rate == 0 || channels == 0 || channels > 8 {
        return None;
    }
    Some((rate, channels))
}

fn channel_layout(channels: usize) -> ChannelLayout {
    let labels = [
        Channel::FrontLeft,
        Channel::FrontRight,
        Channel::FrontCenter,
        Channel::Lfe,
        Channel::BackLeft,
        Channel::BackRight,
        Channel::SideLeft,
        Channel::SideRight,
    ];
    ChannelLayout::new(labels[..channels].to_vec()).expect("channels")
}

#[test]
fn local_files_match_an_independent_pcm_dump() {
    let root = local_root();
    if !root.is_dir() {
        eprintln!("skipped local files: {} is missing", root.display());
        return;
    }
    let Some(ffmpeg) = ffmpeg_bin() else {
        if !require_or_skip("ffmpeg") {
            return;
        }
        return;
    };
    let ffprobe = ffmpeg.with_file_name("ffprobe");
    let names = [
        "GOOGLE IS BACK! (Gemini 3.8 Flash).wav",
        "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a",
    ];
    let mut exercised = 0;
    for name in names {
        let media = root.join(name);
        if !media.is_file() {
            eprintln!("skipped local file: {name} is missing");
            continue;
        }
        let Some((rate, channels)) = ffprobe_audio(&ffprobe, &media) else {
            eprintln!("skipped local file: {name} has no audio stream");
            continue;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-pcm-local-{}", exercised));
        std::fs::create_dir_all(&dir).expect("temp");
        let raw = dir.join("clip.s16le");
        let status = std::process::Command::new(&ffmpeg)
            .args(["-v", "error", "-y", "-t", "0.20", "-i"])
            .arg(&media)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&raw)
            .output()
            .expect("ffmpeg");
        assert!(status.status.success(), "{name} {status:?}");
        let bytes = std::fs::read(&raw).expect("read");
        let _ = std::fs::remove_dir_all(&dir);
        let width = channels * 2;
        assert!(
            !bytes.is_empty() && bytes.len().is_multiple_of(width),
            "{name}"
        );
        let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
        let mut opened =
            PcmDecoder::open(&params(spec, rate, channel_layout(channels))).expect("open");
        let pcm = audio(&mut opened, bytes.clone()).expect("decode");
        assert_eq!(pcm.format, SampleFormat::I16);
        assert_eq!(
            usize::try_from(pcm.samples).expect("samples"),
            bytes.len() / width
        );
        assert_eq!(pcm.as_i16().expect("i16").len(), bytes.len() / 2);
        let foreign: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| i16::from_le_bytes(*chunk))
            .collect();
        assert_eq!(pcm.as_i16(), Some(foreign.as_slice()));
        exercised += 1;
    }
    assert!(exercised >= 1, "no local audio file was decoded");
}

#[test]
#[ignore = "pcm decoder fuzz; runs at least 30 minutes"]
fn thirty_minute_decoder_campaign() {
    let seconds: u64 = std::env::var("PCM_FUZZ_SECS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(30 * 60);
    let seed = 0x5043_4D5F_4655_5A5Au64;
    let mut rng = Rng(seed);
    let specs = [
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        layout(Coding::S16, Endian::Big, Arrangement::Planar),
        layout(Coding::S24, Endian::Little, Arrangement::Interleaved),
        layout(Coding::S32, Endian::Big, Arrangement::Interleaved),
        layout(Coding::F32, Endian::Little, Arrangement::Planar),
        layout(Coding::F64, Endian::Little, Arrangement::Interleaved),
    ];
    let mut decoders: Vec<PcmDecoder> = specs
        .into_iter()
        .map(|spec| open_decoder(spec, ChannelLayout::stereo()))
        .collect();
    let started = std::time::Instant::now();
    let mut inputs = 0u64;
    let mut errors = 0u64;
    while started.elapsed().as_secs() < seconds {
        let index = rng.index(decoders.len());
        let bytes = if inputs.is_multiple_of(5) {
            let mut packet = le_i16(&[1, -1, 2, -2]);
            let at = rng.index(packet.len());
            packet[at] ^= rng.byte() | 1;
            packet
        } else {
            let len = rng.index(96);
            (0..len).map(|_| rng.byte()).collect()
        };
        match decoders[index].push(&Packet::audio(bytes, 0, 1, 48_000)) {
            Ok(outputs) => {
                for output in outputs {
                    let Output::Audio(audio) = output else {
                        panic!("finding: non-audio output");
                    };
                    let channels = audio.pcm.layout.len();
                    let stored = audio
                        .pcm
                        .as_i16()
                        .map(|s| s.len())
                        .or_else(|| audio.pcm.as_f32().map(|s| s.len()));
                    let Some(stored) = stored else {
                        panic!("finding: empty storage");
                    };
                    assert_eq!(
                        stored,
                        usize::try_from(audio.pcm.samples).expect("samples") * channels
                    );
                }
            }
            Err(_) => errors += 1,
        }
        if inputs.is_multiple_of(10_000) {
            let pcm = audio(&mut decoders[0], le_i16(&[4, 5])).expect("known frame");
            assert_eq!(pcm.as_i16(), Some(&[4, 5][..]));
        }
        inputs += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "pcm fuzz seed {seed:#x} seconds {elapsed:.2} inputs {inputs} errors {errors} findings 0"
    );
}

#[test]
#[ignore = "pcm encoder fuzz; bounded"]
fn bounded_encoder_campaign() {
    let seconds: u64 = std::env::var("PCM_ENCODER_FUZZ_SECS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(120);
    let seed = 0x5043_4D5F_454E_4300u64;
    let mut rng = Rng(seed);
    let specs = [
        layout(Coding::S16, Endian::Little, Arrangement::Interleaved),
        layout(Coding::S16, Endian::Big, Arrangement::Planar),
        layout(Coding::F32, Endian::Little, Arrangement::Interleaved),
        layout(Coding::F64, Endian::Big, Arrangement::Planar),
    ];
    let started = std::time::Instant::now();
    let mut inputs = 0u64;
    let mut errors = 0u64;
    let mut round_trips = 0u64;
    while started.elapsed().as_secs() < seconds {
        let spec = specs[rng.index(specs.len())];
        let mut opened = open_encoder(spec, ChannelLayout::stereo());
        let frames = rng.index(32) + 1;
        let result = if spec.coding() == Coding::S16 {
            let samples: Vec<i16> = (0..frames * 2).map(|_| rng.sample()).collect();
            let pcm = Pcm::from_i16(48_000, ChannelLayout::stereo(), samples).expect("pcm");
            opened.push(&AudioInput { pcm })
        } else if inputs.is_multiple_of(7) {
            let samples = vec![0.25f32, -0.5, f32::NAN, 1.0];
            let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), samples).expect("pcm");
            opened.push(&AudioInput { pcm })
        } else {
            let samples: Vec<f32> = (0..frames * 2)
                .map(|_| f32::from(rng.sample()) / 32_768.0_f32)
                .collect();
            let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), samples).expect("pcm");
            opened.push(&AudioInput { pcm })
        };
        match result {
            Ok(packets) => {
                let mut decoder = open_decoder(spec, ChannelLayout::stereo());
                let outputs = decoder.push(&packets[0]).expect("round trip");
                let Output::Audio(audio) = &outputs[0] else {
                    panic!("finding: non-audio output");
                };
                assert_eq!(usize::try_from(audio.pcm.samples).expect("samples"), frames);
                round_trips += 1;
            }
            Err(_) => errors += 1,
        }
        if inputs.is_multiple_of(1_000) {
            let known = Pcm::from_i16(48_000, ChannelLayout::stereo(), vec![1, 2]).expect("known");
            let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
            let mut encoder = open_encoder(spec, ChannelLayout::stereo());
            let packets = encoder.push(&AudioInput { pcm: known }).expect("known");
            assert_eq!(packets[0].data, le_i16(&[1, 2]));
        }
        let mut hostile = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        hostile.extra = vec![rng.byte(), rng.byte(), rng.byte()];
        let _ = PcmEncoder::open(&hostile);
        inputs += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "pcm encoder fuzz seed {seed:#x} seconds {elapsed:.2} inputs {inputs} errors {errors} round_trips {round_trips} findings 0"
    );
}

#[test]
#[ignore = "release pcm throughput; not a debug measurement"]
fn release_decode_and_encode_throughput() {
    if cfg!(debug_assertions) {
        panic!("release build required");
    }
    let frames = 48_000u32 * 10;
    let frame_count = usize::try_from(frames).expect("frames");
    let mut samples = Vec::with_capacity(frame_count * 2);
    let mut state = 0x1234_5678u32;
    for _ in 0..frame_count * 2 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let bytes = state.to_le_bytes();
        samples.push(i16::from_le_bytes([bytes[0], bytes[1]]));
    }
    let spec = layout(Coding::S16, Endian::Little, Arrangement::Interleaved);
    let channels = ChannelLayout::stereo();
    let source = Pcm::from_i16(48_000, channels.clone(), samples.clone()).expect("pcm");
    let bytes = le_i16(&samples);
    let mut decoder = open_decoder(spec, channels.clone());
    let pcm = audio(&mut decoder, bytes.clone()).expect("check");
    assert_eq!(pcm.as_i16(), Some(samples.as_slice()));
    let mut encoder = open_encoder(spec, channels);
    let packets = encoder
        .push(&AudioInput {
            pcm: source.clone(),
        })
        .expect("check encode");
    assert_eq!(packets[0].data, bytes);

    let warmup = 2;
    let runs = 8;
    for _ in 0..warmup {
        let _ = audio(&mut decoder, bytes.clone()).expect("warmup");
        let _ = encoder
            .push(&AudioInput {
                pcm: source.clone(),
            })
            .expect("warmup");
    }
    let rss_before = resident_kib();
    let started = std::time::Instant::now();
    for _ in 0..runs {
        let pcm = audio(&mut decoder, bytes.clone()).expect("decode");
        assert_eq!(pcm.samples, frames);
    }
    let decode_elapsed = started.elapsed();
    let started = std::time::Instant::now();
    for _ in 0..runs {
        let packets = encoder
            .push(&AudioInput {
                pcm: source.clone(),
            })
            .expect("encode");
        assert_eq!(packets[0].data.len(), bytes.len());
    }
    let encode_elapsed = started.elapsed();
    let rss_after = resident_kib();
    let audio_seconds = 10.0 * f64::from(runs);
    let decode_rt = audio_seconds / decode_elapsed.as_secs_f64();
    let encode_rt = audio_seconds / encode_elapsed.as_secs_f64();
    eprintln!(
        "pcm release frames {frames} channels 2 runs {runs} decode_s {:.6} decode_rt {decode_rt:.1} encode_s {:.6} encode_rt {encode_rt:.1} rss_before {rss_before:?} rss_after {rss_after:?}",
        decode_elapsed.as_secs_f64(),
        encode_elapsed.as_secs_f64()
    );
    assert!(decode_rt >= 10.0, "decode realtime {decode_rt}");
    assert!(encode_rt >= 10.0, "encode realtime {encode_rt}");
    if let (Some(before), Some(after)) = (rss_before, rss_after) {
        let growth = after.saturating_sub(before);
        assert!(growth < 64 * 1024, "rss grew {growth} KiB");
    }
}

fn resident_kib() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}
