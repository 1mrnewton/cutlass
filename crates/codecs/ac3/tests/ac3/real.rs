//! Program material from `../local-files`, plus silence.
//!
//! The folder is not in CI. A missing folder skips the real-file test,
//! and `AC3_REQUIRE_ORACLES=1` turns that skip into a failure. Files stay
//! where they are: ffmpeg writes a few seconds of PCM into a temp dir.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_ac3::{Ac3Decoder, Ac3Encoder, ENCODER_DELAY, EncoderConfig, Format};
use cutlass_codec::{
    CodecId, Packet as FilePacket, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{Demux, Landing, Mux, NewStream, SeekTo, TrackDuration};
use cutlass_decode::{ChannelLayout, Decoder, Output, Packet, Pcm};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_isobmff::{IsoBmffFile, Mp4Mux};

use crate::common::{apple_decode, ffmpeg_decode, oracle, snr, temp_dir};

struct Clip {
    kind: &'static str,
    file: &'static str,
    start: f64,
    seconds: f64,
    channels: u16,
    rate: u32,
}

const CLIPS: [Clip; 4] = [
    Clip {
        kind: "speech",
        file: "GOOGLE IS BACK! (Gemini 3.8 Flash).wav",
        start: 60.0,
        seconds: 4.0,
        channels: 1,
        rate: 48_000,
    },
    Clip {
        kind: "music",
        file: "06 First Person Shooter (feat. J. Cole).m4a",
        start: 40.0,
        seconds: 4.0,
        channels: 2,
        rate: 48_000,
    },
    Clip {
        kind: "talk",
        file: "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a",
        start: 15.0,
        seconds: 4.0,
        channels: 2,
        rate: 44_100,
    },
    Clip {
        kind: "thunder",
        file: "thunder_distant_10s.mp3",
        start: 1.0,
        seconds: 4.0,
        channels: 2,
        rate: 48_000,
    },
];

fn local_files() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if path.is_dir() {
        return Some(path);
    }
    let required = std::env::var_os("AC3_REQUIRE_ORACLES").is_some_and(|value| value == "1");
    assert!(!required, "local-files is required and missing");
    eprintln!("skipped: local-files is not next to this checkout");
    None
}

fn layout(channels: u16) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("pcm");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn load_pcm(ffmpeg: &str, clip: &Clip, root: &Path) -> Vec<f32> {
    let dir = temp_dir("real");
    let output = dir.join(format!("{}.f32", clip.kind));
    let status = Command::new(ffmpeg)
        .args([
            "-v",
            "error",
            "-y",
            "-ss",
            &clip.start.to_string(),
            "-t",
            &clip.seconds.to_string(),
            "-i",
        ])
        .arg(root.join(clip.file))
        .args([
            "-ac",
            &clip.channels.to_string(),
            "-ar",
            &clip.rate.to_string(),
            "-f",
            "f32le",
            "-c:a",
            "pcm_f32le",
        ])
        .arg(&output)
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "ffmpeg could not read {}", clip.file);
    read_f32(&output)
}

fn encode(format: Format, rate: u32, channels: u16, pcm: &[f32]) -> Vec<Packet> {
    let config = EncoderConfig::new(format, rate, layout(channels));
    let mut encoder = Ac3Encoder::new(config).expect("encoder");
    let pcm = Pcm::from_f32(rate, layout(channels), pcm.to_vec()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    packets
}

fn codec_of(format: Format) -> CodecId {
    match format {
        Format::Ac3 => CodecId::Ac3,
        Format::Eac3 => CodecId::Eac3,
    }
}

fn audible(codec: CodecId, packets: &[Packet]) -> Vec<f32> {
    let params = cutlass_decode::CodecParams::audio(codec, 0, ChannelLayout::stereo());
    let mut decoder = Ac3Decoder::open(&params).expect("decoder");
    let mut pcm = Vec::new();
    let mut take = |outputs: Vec<Output>| {
        for output in outputs {
            let Output::Audio(audio) = output else {
                continue;
            };
            let samples = audio.pcm.as_f32().expect("f32");
            let channels = audio.pcm.layout.len();
            let start = audio.skip_leading as usize * channels;
            let end = samples.len() - audio.skip_trailing as usize * channels;
            pcm.extend_from_slice(&samples[start..end]);
        }
    };
    for packet in packets {
        take(decoder.push(packet).expect("decode"));
    }
    take(decoder.drain().expect("drain"));
    pcm
}

fn bitrate(packets: &[Packet], rate: u32) -> f64 {
    let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
    let samples: i64 = packets.iter().map(|packet| packet.duration).sum();
    bytes as f64 * 8.0 * f64::from(rate) / samples as f64
}

fn middle(len: usize) -> std::ops::Range<usize> {
    let margin = 1536.min(len / 4);
    margin..len - margin
}

fn stream(packets: &[Packet]) -> Vec<u8> {
    packets
        .iter()
        .flat_map(|packet| packet.data.clone())
        .collect()
}

fn check(ffmpeg: &str, kind: &str, format: Format, rate: u32, channels: u16, source: &[f32]) {
    let packets = encode(format, rate, channels, source);
    let codec = codec_of(format);
    let ours = audible(codec, &packets);
    assert_eq!(ours.len(), source.len(), "{kind} {format:?} length");
    let target = packets[0].data.len() as f64 * 8.0 * f64::from(rate) / packets[0].duration as f64;
    let actual = bitrate(&packets, rate);
    assert!(
        (actual - target).abs() / target < 0.01,
        "{kind} {format:?}: {actual:.0} against a {target:.0} frame"
    );
    let foreign = ffmpeg_decode(ffmpeg, &stream(&packets), &[]);
    let skip = ENCODER_DELAY as usize * usize::from(channels);
    let foreign = &foreign[skip.min(foreign.len())..];
    let n = ours.len().min(foreign.len());
    let span = middle(n);
    // A one-sample slip would put the best lag off zero. Dither can hold
    // the zero-lag figure under the sine-wave gate without a slip.
    let mut best_lag = 0i32;
    let mut best = f64::NEG_INFINITY;
    for lag in -8..=8 {
        let (a, b) = if lag >= 0 {
            (
                &ours[span.start + lag as usize..span.end],
                &foreign[span.clone()],
            )
        } else {
            (
                &ours[span.clone()],
                &foreign[span.start + (-lag) as usize..span.end],
            )
        };
        let quality = snr(a, b);
        if quality > best {
            best = quality;
            best_lag = lag;
        }
    }
    let interop = snr(&ours[span.clone()], &foreign[span.clone()]);
    assert_eq!(
        best_lag, 0,
        "{kind} {format:?} best lag {best_lag} at {best:.1} dB"
    );
    let quality = snr(&ours[span.clone()], &source[span.clone()]);
    // Measured on the four local clips: the low end is music E-AC-3 at
    // 32 dB against FFmpeg. A broken coupling frame landed at 22 dB.
    assert!(interop > 30.0, "{kind} {format:?} ffmpeg {interop:.1} dB");
    assert!(quality > 15.0, "{kind} {format:?} source {quality:.1} dB");
    if let Some(afconvert) = oracle("afconvert", "-h") {
        let (_count, apple) = apple_decode(&afconvert, &stream(&packets));
        let n = ours.len().min(apple.len());
        let apple_span = middle(n);
        let apple_snr = snr(&ours[apple_span.clone()], &apple[apple_span]);
        assert!(
            apple_snr > 30.0,
            "{kind} {format:?} apple {apple_snr:.1} dB"
        );
    }
    if kind == "music" {
        reopen(codec, rate, channels, &packets);
    }
}

fn reopen(codec: CodecId, rate: u32, channels: u16, packets: &[Packet]) {
    let mut mux = Mp4Mux::new(std::io::Cursor::new(Vec::new()));
    let params = StreamParams::try_audio(StreamCodec::Known(codec), rate, channels, Vec::new())
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
            if let cutlass_decode::SideData::Config(config) = side {
                mux.set_config(track, config).expect("config");
            }
        }
        let timestamp =
            Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                .expect("time");
        let stored = FilePacket::try_new(
            track.get(),
            packet.data.clone(),
            timestamp,
            Timescale::new(rate),
            true,
        )
        .expect("packet");
        mux.write_packet(&stored).expect("write");
        ticks += packet.duration;
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(rate),
        ticks,
    }])
    .expect("finish");
    let bytes = mux.into_writer().expect("writer").into_inner();
    let mut file = IsoBmffFile::open(std::io::Cursor::new(bytes.clone())).expect("open");
    let mut read = Vec::new();
    while let Some(stored) = Demux::read_packet(&mut file).expect("read") {
        read.push(stored.bytes().to_vec());
    }
    assert_eq!(read.len(), packets.len());
    for (stored, packet) in read.iter().zip(packets) {
        assert_eq!(stored, &packet.data);
    }
    let mut file = IsoBmffFile::open(std::io::Cursor::new(bytes)).expect("reopen");
    let landing = Demux::seek(
        &mut file,
        SeekTo {
            track: track.get(),
            time: i64::from(rate),
            end: None,
        },
    )
    .expect("seek");
    let Landing::Packet { time, .. } = landing.landing else {
        panic!("seek past the end");
    };
    assert!(
        time <= i64::from(rate) && i64::from(rate) - time < 1536 * 2,
        "landed at {time}"
    );
    let stored = Demux::read_packet(&mut file)
        .expect("read")
        .expect("packet");
    assert_eq!(&stored.bytes()[..2], &[0x0B, 0x77]);
    let packet = Packet::audio(
        stored.bytes().to_vec(),
        stored.timestamp().presentation().unwrap_or(0),
        stored.timestamp().duration().unwrap_or(1536),
        rate,
    );
    let outputs = Ac3Decoder::open(&cutlass_decode::CodecParams::audio(
        codec,
        rate,
        layout(channels),
    ))
    .expect("decoder")
    .push(&packet)
    .expect("decode");
    assert!(
        outputs
            .iter()
            .any(|output| matches!(output, Output::Audio(_)))
    );
}

#[test]
fn silence_stays_quiet_for_both_formats() {
    let rate = 48_000u32;
    let frames = rate as usize;
    let source = vec![0.0f32; frames * 2];
    for format in [Format::Ac3, Format::Eac3] {
        let pcm = audible(codec_of(format), &encode(format, rate, 2, &source));
        assert_eq!(pcm.len(), source.len());
        let energy: f64 = pcm.iter().map(|sample| f64::from(*sample).powi(2)).sum();
        let rms = (energy / pcm.len() as f64).sqrt();
        assert!(rms < 1.0e-3, "{format:?} silence rms {rms}");
    }
}

#[test]
fn real_speech_music_and_thunder_encode_and_reopen() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    let Some(root) = local_files() else {
        return;
    };
    for clip in &CLIPS {
        let path = root.join(clip.file);
        if !path.is_file() {
            let required = std::env::var_os("AC3_REQUIRE_ORACLES").is_some_and(|v| v == "1");
            assert!(!required, "missing {}", clip.file);
            eprintln!("skipped: {} is not in local-files", clip.file);
            continue;
        }
        let pcm = load_pcm(&ffmpeg, clip, &root);
        assert!(pcm.len() > clip.channels as usize * 8_000, "{}", clip.kind);
        for format in [Format::Ac3, Format::Eac3] {
            check(&ffmpeg, clip.kind, format, clip.rate, clip.channels, &pcm);
        }
    }
}
