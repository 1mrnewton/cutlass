use cutlass_decode::{
    ChannelLayout, CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet, Pcm,
    SideData,
};
use cutlass_encode::{AudioInput, EncodeFactory, Encoder};

use super::audio::{open_decoder, open_encoder};

const RATE: u32 = 48_000;

fn params(rate: u32, layout: ChannelLayout) -> CodecParams {
    let mut params = CodecParams::audio(CodecId::Aac, rate, layout);
    params.bitrate = 128_000;
    params
}

/// A rising tone with a little noise, so a lag search has one best answer.
fn signal(frames: usize) -> Vec<f32> {
    let mut state = 0x2468_ace1_u32;
    (0..frames)
        .map(|index| {
            let time = index as f32 / RATE as f32;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 9) as f32 / (1 << 23) as f32 - 0.5;
            0.4 * (2.0 * std::f32::consts::PI * (300.0 + 400.0 * time) * time).sin() + 0.05 * noise
        })
        .collect()
}

fn stereo(mono: &[f32]) -> Vec<f32> {
    mono.iter().flat_map(|sample| [*sample, *sample]).collect()
}

fn encode(encoder: &mut dyn Encoder, mono: &[f32]) -> Vec<Packet> {
    let mut packets = Vec::new();
    for chunk in mono.chunks(4_000) {
        let pcm = Pcm::from_f32(RATE, ChannelLayout::stereo(), stereo(chunk)).expect("pcm");
        packets.extend(encoder.push(&AudioInput { pcm }).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    packets
}

/// The left channel of every decoded buffer, in order.
fn decode(decoder: &mut dyn Decoder, packets: &[Packet]) -> Vec<f32> {
    let mut outputs = Vec::new();
    for packet in packets {
        outputs.extend(decoder.push(packet).expect("decode"));
    }
    outputs.extend(decoder.drain().expect("drain"));
    let mut left = Vec::new();
    for output in outputs {
        let Output::Audio(audio) = output else {
            panic!("audio");
        };
        let samples = audio.pcm.as_f32().expect("f32");
        let channels = audio.pcm.layout.len();
        left.extend(samples.iter().step_by(channels).copied());
    }
    left
}

/// The lag of `heard` behind `sent`, and the signal-to-noise ratio there.
fn best_lag(sent: &[f32], heard: &[f32]) -> (usize, f64) {
    let span = 24_000;
    let mut best = (0, f64::MIN);
    for lag in 0..4_096 {
        if heard.len() < lag + span || sent.len() < 4_096 + span {
            break;
        }
        let (mut signal, mut error) = (0.0_f64, 0.0_f64);
        for index in 4_096..4_096 + span {
            let want = f64::from(sent[index]);
            let got = f64::from(heard[index + lag]);
            signal += want * want;
            error += (want - got) * (want - got);
        }
        let snr = 10.0 * (signal / error.max(1e-12)).log10();
        if snr > best.1 {
            best = (lag, snr);
        }
    }
    best
}

fn config(packets: &[Packet]) -> Vec<u8> {
    packets[0]
        .side
        .iter()
        .find_map(|side| match side {
            SideData::Config(bytes) => Some(bytes.clone()),
            _ => None,
        })
        .expect("config on the first packet")
}

#[test]
fn aac_round_trips_a_tone_through_windows() {
    let sent = signal(RATE as usize);
    let mut encoder = open_encoder(&params(RATE, ChannelLayout::stereo())).expect("encoder");
    let packets = encode(encoder.as_mut(), &sent);
    assert!(packets.len() >= 47, "{} packets", packets.len());
    for (index, packet) in packets.iter().enumerate() {
        assert_eq!(
            (packet.pts, packet.duration, packet.timescale),
            (index as i64 * 1024, 1024, RATE)
        );
    }
    let asc = config(&packets);
    assert_eq!(asc[0] >> 3, 2, "AAC-LC object type");
    let mut decode_params = params(RATE, ChannelLayout::stereo());
    decode_params.extra = asc;
    let mut decoder = open_decoder(&decode_params).expect("decoder");
    let heard = decode(decoder.as_mut(), &packets);
    let (lag, snr) = best_lag(&sent, &heard);
    eprintln!("windows to windows: lag {lag}, snr {snr:.1} dB");
    assert!(snr > 20.0, "snr {snr:.1} at lag {lag}");
}

#[test]
fn the_encoder_adds_no_priming() {
    let sent = signal(RATE as usize);
    let mut encoder = open_encoder(&params(RATE, ChannelLayout::stereo())).expect("encoder");
    let packets = encode(encoder.as_mut(), &sent);
    assert!(
        packets.iter().all(|packet| !packet
            .side
            .iter()
            .any(|side| matches!(side, SideData::Gapless { .. }))),
        "no gapless delay"
    );
    // The Rust decoder keeps every sample, so the lag it hears is the
    // encoder's whole priming.
    let mut rust = DecodeFactory::new();
    cutlass_aac::register_decoders(&mut rust);
    let mut decode_params = params(RATE, ChannelLayout::stereo());
    decode_params.extra = config(&packets);
    let mut decoder = rust.open(&decode_params).expect("rust decoder");
    let heard = decode(decoder.as_mut(), &packets);
    let (lag, snr) = best_lag(&sent, &heard);
    eprintln!("windows to rust: lag {lag}, snr {snr:.1} dB");
    assert!(snr > 20.0, "snr {snr:.1} at lag {lag}");
    assert_eq!(lag, 0, "priming");
}

#[test]
fn rust_aac_decodes_on_windows() {
    let sent = signal(RATE as usize);
    let mut rust = EncodeFactory::new();
    cutlass_aac::register_encoders(&mut rust);
    let mut encoder = rust
        .open(&params(RATE, ChannelLayout::stereo()))
        .expect("rust encoder");
    let packets = encode(encoder.as_mut(), &sent);
    let mut decode_params = params(RATE, ChannelLayout::stereo());
    decode_params.extra = config(&packets);
    let mut decoder = open_decoder(&decode_params).expect("decoder");
    let heard = decode(decoder.as_mut(), &packets);
    let (lag, snr) = best_lag(&sent, &heard);
    eprintln!("rust to windows: lag {lag}, snr {snr:.1} dB");
    assert!(snr > 20.0, "snr {snr:.1} at lag {lag}");
}

#[test]
fn a_rate_or_layout_the_encoder_lacks_is_declined() {
    assert!(matches!(
        open_encoder(&params(32_000, ChannelLayout::stereo())),
        Err(CodecError::Unsupported { .. })
    ));
    let mut opus = params(RATE, ChannelLayout::stereo());
    opus.codec = CodecId::Opus;
    assert!(matches!(
        open_encoder(&opus),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        open_decoder(&opus),
        Err(CodecError::Unsupported { .. })
    ));
}

#[test]
fn a_bad_sample_adds_nothing() {
    let mut encoder = open_encoder(&params(RATE, ChannelLayout::stereo())).expect("encoder");
    let nan = Pcm::from_f32(RATE, ChannelLayout::stereo(), vec![0.0, f32::NAN]).expect("pcm");
    assert_eq!(
        encoder.push(&AudioInput { pcm: nan }),
        Err(CodecError::NonFinite)
    );
    let loud = Pcm::from_f32(RATE, ChannelLayout::stereo(), vec![0.0, 1.5]).expect("pcm");
    assert_eq!(
        encoder.push(&AudioInput { pcm: loud }),
        Err(CodecError::OutOfRange)
    );
    let mono = Pcm::from_f32(RATE, ChannelLayout::mono(), vec![0.0]).expect("pcm");
    assert!(matches!(
        encoder.push(&AudioInput { pcm: mono }),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(
        encoder.drain().expect("drain").is_empty(),
        "nothing was accepted"
    );
}

#[test]
fn the_factory_uses_windows_aac_first() {
    let mut encoders = EncodeFactory::new();
    super::register_encoders_with(&mut encoders, false);
    let encoder = encoders.open(&params(RATE, ChannelLayout::stereo()));
    assert!(
        encoder.is_ok(),
        "audio stays registered with the picture codecs off"
    );
}
