//! The decoder on checked-in fixtures (see `fixtures/README.md`) through
//! the [`Decoder`] contract: output sizes and times, end trim, refusal,
//! failure, reset, drain, configuration changes, chained streams, and
//! truncated packets.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use common::{OggPacket, decode_file, ogg_packets, split_stream};
use cutlass_decode::{
    Channel, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, SideData,
};
use cutlass_vorbis::VorbisDecoder;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// Every fixture with one link, its rate, channels, and end granule.
const FIXTURES: [(&str, u32, usize, usize); 5] = [
    ("stereo_44100_q4.ogg", 44_100, 2, 22_050),
    ("clicks_48000_q9.ogg", 48_000, 2, 48_000),
    ("sweep_8000_mono_q0.ogg", 8_000, 1, 4_000),
    ("surround51_48000_q4.ogg", 48_000, 6, 12_000),
    ("ffmpeg_stereo_44100.ogg", 44_100, 2, 22_080),
];

fn open(config: &[u8]) -> VorbisDecoder {
    let mut params = CodecParams::audio(CodecId::Vorbis, 44_100, ChannelLayout::stereo());
    params.extra = config.to_vec();
    VorbisDecoder::open(&params).unwrap()
}

fn packet(data: &[u8], pts: i64, rate: u32) -> Packet {
    Packet::audio(data.to_vec(), pts, 0, rate)
}

/// The one audio output of a push, or `None` for no output.
fn audio(outputs: Vec<Output>) -> Option<cutlass_decode::AudioOutput> {
    assert!(outputs.len() <= 1, "one push gives at most one output");
    outputs.into_iter().next().map(|output| match output {
        Output::Audio(audio) => audio,
        other => panic!("not audio: {other:?}"),
    })
}

fn samples(outputs: Vec<Output>) -> Vec<f32> {
    audio(outputs).map_or_else(Vec::new, |audio| audio.pcm.as_f32().unwrap().to_vec())
}

/// Every packet's output through `push`, in order.
fn push_all(decoder: &mut VorbisDecoder, packets: &[OggPacket], rate: u32) -> Vec<Vec<f32>> {
    packets
        .iter()
        .map(|p| samples(decoder.push(&packet(&p.data, 0, rate)).unwrap()))
        .collect()
}

#[test]
fn fixtures_decode_to_their_granule_length() {
    for (name, rate, channels, length) in FIXTURES {
        let (got_rate, got_channels, pcm) = decode_file(&fixture(name));
        assert_eq!((got_rate, got_channels), (rate, channels), "{name}");
        assert_eq!(pcm.len(), length * channels, "{name}");
        assert!(
            pcm.iter().all(|s| s.is_finite() && s.abs() < 2.0),
            "{name}: range"
        );
        for channel in 0..channels {
            let energy: f64 = pcm
                .iter()
                .skip(channel)
                .step_by(channels)
                .map(|&s| f64::from(s) * f64::from(s))
                .sum();
            assert!(energy > 1.0, "{name}: channel {channel} is silent");
        }
    }
}

#[test]
fn each_packet_outputs_the_overlap_of_its_block_and_the_last() {
    for (name, rate, channels, _) in FIXTURES {
        let (config, packets) = split_stream(&fixture(name));
        let mut decoder = open(&config);
        let [short, long] = decoder.identification().blocksizes;
        let sizes = [short / 2, (short + long) / 4, long / 2];
        let mut short_long = 0;
        for (index, p) in packets.iter().enumerate() {
            let pts = 1_000 + index as i64;
            let out = audio(decoder.push(&packet(&p.data, pts, rate)).unwrap());
            if index == 0 {
                assert!(out.is_none(), "{name}: the first packet outputs nothing");
                continue;
            }
            let out = out.unwrap();
            assert_eq!(
                (out.pts, out.timescale),
                (pts, rate),
                "{name}: time passes through"
            );
            assert_eq!((out.skip_leading, out.skip_trailing), (0, 0));
            assert!(!out.concealed);
            assert_eq!(out.pcm.rate, rate);
            assert_eq!(out.pcm.layout.len(), channels);
            let frames = out.pcm.samples as usize;
            assert!(
                sizes.contains(&frames),
                "{name}: packet {index} gave {frames}"
            );
            short_long += usize::from(frames == sizes[1]);
        }
        if name.starts_with("clicks") {
            assert!(short_long > 4, "{name}: block switching is exercised");
        }
    }
}

#[test]
fn a_short_stamped_last_packet_trims_its_end() {
    let (config, packets) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let mut decoder = open(&config);
    let (last, rest) = packets.split_last().unwrap();
    push_all(&mut decoder, rest, 44_100);
    let mut stamped = packet(&last.data, 0, 44_100);
    stamped.duration = 100;
    let out = audio(decoder.push(&stamped).unwrap()).unwrap();
    assert_eq!(out.skip_trailing, out.pcm.samples - 100);
    // A duration in another timescale converts at the sample rate.
    let mut decoder = open(&config);
    push_all(&mut decoder, rest, 44_100);
    let mut stamped = Packet::audio(last.data.clone(), 0, 1_000, 1_000_000);
    stamped.duration = 1_000;
    let out = audio(decoder.push(&stamped).unwrap()).unwrap();
    assert_eq!(out.skip_trailing, out.pcm.samples - 44);
}

#[test]
fn open_refuses_another_codec_and_a_bad_configuration() {
    let (config, _) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
    params.extra = config.clone();
    assert!(matches!(
        VorbisDecoder::open(&params),
        Err(CodecError::Unsupported { .. })
    ));
    params.codec = CodecId::Vorbis;
    params.extra = config[..config.len() / 2].to_vec();
    assert!(VorbisDecoder::open(&params).is_err());
    params.extra = Vec::new();
    assert!(VorbisDecoder::open(&params).is_err());
}

#[test]
fn an_empty_packet_is_refused() {
    let (config, _) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let mut decoder = open(&config);
    assert!(matches!(
        decoder.push(&packet(&[], 0, 44_100)),
        Err(CodecError::InvalidApi { .. })
    ));
}

#[test]
fn a_failed_packet_leaves_the_decoder_as_it_was() {
    let (config, packets) = split_stream(&fixture("clicks_48000_q9.ogg"));
    let mut clean = open(&config);
    let want = push_all(&mut clean, &packets, 48_000);
    let mut decoder = open(&config);
    let bad_setup = {
        let mut setup = split_stream(&fixture("clicks_48000_q9.ogg")).1[0]
            .data
            .clone();
        setup[0] = 5;
        setup
    };
    let mut bad_config = config.clone();
    bad_config.truncate(config.len() - 40);
    // A configuration that opens, on a packet that then fails.
    let (other_config, _) = split_stream(&fixture("sweep_8000_mono_q0.ogg"));
    let failures: [Box<dyn Fn() -> Packet>; 5] = [
        Box::new(|| packet(&[1, b'x', b'x'], 0, 48_000)),
        Box::new(|| packet(&[7, 0, 0, 0], 0, 48_000)),
        Box::new(move || packet(&bad_setup, 0, 48_000)),
        Box::new(move || {
            let mut p = packet(&[0, 0], 0, 48_000);
            p.side.push(SideData::Config(bad_config.clone()));
            p
        }),
        Box::new(move || {
            let mut p = packet(&[1, b'x', b'x'], 0, 48_000);
            p.side.push(SideData::Config(other_config.clone()));
            p
        }),
    ];
    for (index, p) in packets.iter().enumerate() {
        if index % 5 == 3 {
            let failure = &failures[index / 5 % failures.len()];
            assert!(decoder.push(&failure()).is_err(), "packet {index}: refused");
        }
        let got = samples(decoder.push(&packet(&p.data, 0, 48_000)).unwrap());
        assert_eq!(got, want[index], "packet {index}");
    }
}

#[test]
fn reset_drops_the_overlap_and_keeps_the_configuration() {
    let (config, packets) = split_stream(&fixture("clicks_48000_q9.ogg"));
    let mut decoder = open(&config);
    push_all(&mut decoder, &packets[..20], 48_000);
    decoder.reset();
    let after = push_all(&mut decoder, &packets[20..], 48_000);
    let mut fresh = open(&config);
    let want = push_all(&mut fresh, &packets[20..], 48_000);
    assert!(
        after[0].is_empty(),
        "the first packet after a reset outputs nothing"
    );
    assert_eq!(after, want);
}

#[test]
fn drain_outputs_nothing_and_decoding_continues() {
    let (config, packets) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let mut clean = open(&config);
    let want = push_all(&mut clean, &packets, 44_100);
    let mut decoder = open(&config);
    let mut got = push_all(&mut decoder, &packets[..10], 44_100);
    assert!(decoder.drain().unwrap().is_empty());
    assert!(decoder.drain().unwrap().is_empty());
    got.extend(push_all(&mut decoder, &packets[10..], 44_100));
    assert_eq!(got, want);
}

#[test]
fn the_same_configuration_again_changes_nothing() {
    let (config, packets) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let mut clean = open(&config);
    let want = push_all(&mut clean, &packets, 44_100);
    let mut decoder = open(&config);
    for (index, p) in packets.iter().enumerate() {
        let mut stamped = packet(&p.data, 0, 44_100);
        stamped.side.push(SideData::Config(config.clone()));
        assert_eq!(samples(decoder.push(&stamped).unwrap()), want[index]);
        // In-band header packets of the same stream change nothing either.
        if index == 5 {
            let headers = ogg_packets(&fixture("stereo_44100_q4.ogg"));
            for header in &headers[..3] {
                assert!(
                    decoder
                        .push(&packet(&header.data, 0, 44_100))
                        .unwrap()
                        .is_empty()
                );
            }
        }
    }
}

#[test]
fn a_new_configuration_switches_streams() {
    let (stereo, stereo_packets) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let (mono, mono_packets) = split_stream(&fixture("sweep_8000_mono_q0.ogg"));
    let mut fresh = open(&mono);
    let want = push_all(&mut fresh, &mono_packets, 8_000);
    let mut decoder = open(&stereo);
    push_all(&mut decoder, &stereo_packets[..6], 44_100);
    let mut got = Vec::new();
    for (index, p) in mono_packets.iter().enumerate() {
        let mut stamped = packet(&p.data, 0, 8_000);
        if index == 0 {
            stamped.side.push(SideData::Config(mono.clone()));
        }
        let out = audio(decoder.push(&stamped).unwrap());
        if let Some(out) = &out {
            assert_eq!(out.pcm.layout.channels(), [Channel::FrontCenter]);
            assert_eq!(out.pcm.rate, 8_000);
        }
        got.push(out.map_or_else(Vec::new, |o| o.pcm.as_f32().unwrap().to_vec()));
    }
    assert_eq!(got, want);
    assert_eq!(decoder.identification().channels, 1);
}

#[test]
fn a_chained_stream_takes_its_new_headers_in_band() {
    let chained = ogg_packets(&fixture("chained_mono_stereo.ogg"));
    let mono = ogg_packets(&fixture("sweep_8000_mono_q0.ogg"));
    let (stereo_config, stereo) = split_stream(&fixture("stereo_44100_q4.ogg"));
    assert_eq!(chained.len(), mono.len() + stereo.len() + 3);
    let (first_config, _) = split_stream(&fixture("sweep_8000_mono_q0.ogg"));
    let mut decoder = open(&first_config);
    let first = push_all(&mut decoder, &chained[3..mono.len()], 8_000);
    assert!(first.iter().skip(1).all(|s| !s.is_empty()));
    let headers = push_all(&mut decoder, &chained[mono.len()..mono.len() + 3], 44_100);
    assert!(headers.iter().all(Vec::is_empty));
    assert_eq!(decoder.layout(), &ChannelLayout::stereo());
    let second = push_all(&mut decoder, &chained[mono.len() + 3..], 44_100);
    let mut fresh = open(&stereo_config);
    assert_eq!(second, push_all(&mut fresh, &stereo, 44_100));
}

#[test]
fn truncated_and_damaged_packets_fail_or_decode_without_panic() {
    for name in [
        "clicks_48000_q9.ogg",
        "surround51_48000_q4.ogg",
        "ffmpeg_stereo_44100.ogg",
    ] {
        let (config, packets) = split_stream(&fixture(name));
        let mut decoder = open(&config);
        let [short, long] = decoder.identification().blocksizes;
        let sizes = [0, short / 2, (short + long) / 4, long / 2];
        let channels = usize::from(decoder.identification().channels);
        for p in packets.iter().take(12) {
            for len in 1..p.data.len() {
                let mut damaged = p.data[..len].to_vec();
                let flip = len * 7 % damaged.len();
                if len % 2 == 0 {
                    damaged[flip] ^= 0x5a;
                }
                if let Ok(out) = decoder.decode(&damaged) {
                    assert!(sizes.contains(&(out.len() / channels)), "{name}");
                    assert!(out.iter().all(|s| s.is_finite()), "{name}");
                }
            }
            decoder.decode(&p.data).unwrap();
        }
    }
}

#[test]
fn decoding_is_deterministic() {
    let bytes = fixture("surround51_48000_q4.ogg");
    assert_eq!(decode_file(&bytes), decode_file(&bytes));
}

/// `bytes` with its last page's granule set to the one before it, so every
/// packet ending on that page lies wholly past the end, and its checksum
/// written again.
fn end_at_the_page_before(bytes: &[u8]) -> (Vec<u8>, i64) {
    let mut pages = Vec::new();
    let mut at = 0;
    while at + 27 <= bytes.len() {
        let header = cutlass_ogg::page::parse_header(&bytes[at..]).unwrap();
        let table = 27 + usize::from(header.segments);
        let body: usize = bytes[at + 27..at + table]
            .iter()
            .map(|&l| usize::from(l))
            .sum();
        pages.push((at, table, body, header.granule));
        at += table + body;
    }
    let (last, table, body, _) = *pages.last().unwrap();
    let end = pages[..pages.len() - 1]
        .iter()
        .rev()
        .find_map(|&(_, _, _, granule)| (granule != -1).then_some(granule))
        .unwrap();
    let mut out = bytes.to_vec();
    out[last + 6..last + 14].copy_from_slice(&end.to_le_bytes());
    let crc = cutlass_ogg::crc::page_crc(
        &out[last..last + table],
        &out[last + table..last + table + body],
    );
    out[last + 22..last + 26].copy_from_slice(&crc.to_le_bytes());
    (out, end)
}

/// Packets the Ogg demuxer stamps with a duration of 0, because the
/// end-of-stream granule falls before them, are trimmed whole: what plays
/// is the edit's length, not the decoded one.
#[test]
fn a_packet_wholly_past_the_end_granule_is_trimmed_whole() {
    use cutlass_container::Demux;

    let (bytes, end) = end_at_the_page_before(&fixture("stereo_44100_q4.ogg"));
    let length = bytes.len() as u64;
    let mut demux = cutlass_ogg::OggDemux::open(std::io::Cursor::new(bytes), length).unwrap();
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.edits()[0].duration(), end);
    let mut decoder = VorbisDecoder::from_config(stream.params().extradata()).unwrap();
    let (mut kept, mut whole) = (0_i64, 0);
    while let Some(packet) = demux.read_packet().unwrap() {
        let duration = packet.timestamp().duration().unwrap();
        let access = Packet::audio(
            packet.bytes().to_vec(),
            packet.timestamp().presentation_or_decode().unwrap(),
            duration,
            packet.timescale().ticks_per_second(),
        );
        let Some(out) = audio(decoder.push(&access).unwrap()) else {
            continue;
        };
        if duration == 0 {
            assert_eq!(out.skip_trailing, out.pcm.samples, "trimmed whole");
            whole += 1;
        }
        kept += i64::from(out.pcm.samples - out.skip_trailing);
    }
    assert!(
        whole > 0,
        "the last page holds a packet wholly past the end"
    );
    assert_eq!(kept, end);

    // A source that stamps no lengths gets every sample of a packet.
    let (config, packets) = split_stream(&fixture("stereo_44100_q4.ogg"));
    let mut decoder = open(&config);
    push_all(&mut decoder, &packets[..2], 44_100);
    let out = audio(decoder.push(&packet(&packets[2].data, 0, 44_100)).unwrap()).unwrap();
    assert_eq!(out.skip_trailing, 0);
}
