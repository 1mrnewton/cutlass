//! Opus files other tools write, read by the Cutlass demuxers and decoded
//! here; and Opus files written here, read by ffmpeg.
//!
//! 1. ffmpeg with libopus (CELT, SILK, hybrid, 2.5 to 60 ms frames, 5.1)
//!    and with its native encoder writes `.opus` and `.mp4`. `OggDemux` or
//!    `IsoBmffFile` reads each, this crate decodes every packet, and the
//!    first edit cuts the pre-skip and the end trim. The result has the
//!    sample count libopus (through ffmpeg) gives for the same file, and
//!    matches its samples to at least `MIN_SNR_DB`.
//!    The same encode in MP4 gives exactly the Ogg file's samples.
//! 2. This crate's encoder, muxed by `OggMux` and `Mp4Mux`, reads back in
//!    ffprobe with the pre-skip, and both ffmpeg decoders give back
//!    exactly the input's sample count (for MP4, ffprobe's duration: its
//!    MP4 reader does not apply the end trim) and libopus agrees with this
//!    crate's decode of the same file.
//!
//! Without ffmpeg built with libopus the tests skip. The source is ten
//! seconds of speech from `local-files` when it is there (named by
//! `CUTLASS_LOCAL_FILES`), else a synthetic chord with noise. Files go to a
//! temporary folder and are removed.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm, SideData};
use cutlass_isobmff::{IsoBmffFile, Mp4Mux};
use cutlass_ogg::{OggDemux, OggMux};
use cutlass_opus::{Mode, OpusDecoder, OpusEncoder, OpusHead, Toc};

/// libopus's float decoder and this one differ by rounding only; the
/// vectors' worst stereo case is far above this.
const MIN_SNR_DB: f64 = 60.0;
const SECONDS: u32 = 10;

fn has_libopus() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("libopus"))
}

fn scratch(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("cutlass-opus-interop-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn speech() -> Option<PathBuf> {
    let root = std::env::var_os("CUTLASS_LOCAL_FILES")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/Users/mrnewton/dev/cutlass/local-files")))?;
    let file = root.join("GOOGLE IS BACK! (Gemini 3.8 Flash).wav");
    file.is_file().then_some(file)
}

/// Writes the ten-second stereo 48 kHz source as a WAV in `dir`.
fn source(dir: &Path) -> PathBuf {
    let out = dir.join("source.wav");
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-y"]);
    if let Some(file) = speech() {
        command
            .args(["-ss", "30", "-t", &SECONDS.to_string(), "-i"])
            .arg(file);
    } else {
        command.args([
            "-f",
            "lavfi",
            "-i",
            &format!(
                "aevalsrc=0.3*sin(2*PI*220*t)+0.2*sin(2*PI*1330*t)+0.05*(random(0)-0.5)|\
                 0.3*sin(2*PI*330*t)+0.1*sin(2*PI*5000*t):s=48000:d={SECONDS}"
            ),
        ]);
    }
    command
        .args(["-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le"])
        .arg(&out);
    assert!(command.status().expect("ffmpeg").success(), "source");
    out
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// ffmpeg's decode with `decoder` (`opus` or `libopus`), in the file's own
/// channel count, as floats.
fn ffmpeg_decode(file: &Path, decoder: &str) -> Vec<f32> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-c:a", decoder, "-i"])
        .arg(file)
        .args(["-f", "f32le", "-"])
        .output()
        .expect("ffmpeg");
    assert!(
        out.status.success(),
        "{decoder} decode of {}: {}",
        file.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    floats(&out.stdout)
}

struct Ours {
    channels: usize,
    pcm: Vec<f32>,
    pre_skip: u16,
    modes: [usize; 3],
    frame_sizes: Vec<u32>,
    /// The track's ticks per second.
    scale: i64,
}

/// Reads `file` with the Cutlass demuxer for its extension, decodes every
/// packet, and keeps what the first edit plays.
fn ours(file: &Path) -> Ours {
    let reader = File::open(file).expect("open");
    let length = reader.metadata().expect("metadata").len();
    let mut demux: Box<dyn Demux> = if file.extension().is_some_and(|e| e == "mp4") {
        Box::new(IsoBmffFile::open(reader).expect("mp4"))
    } else {
        Box::new(OggDemux::open(reader, length).expect("ogg"))
    };
    let stream = demux.streams()[0].clone();
    // Ogg counts at 48 kHz; an MP4 may count at the input's rate.
    let scale = i64::from(stream.timescale().ticks_per_second());
    let head = OpusHead::parse(stream.params().extradata()).expect("head");
    let channels = usize::from(head.channels);
    assert_eq!(
        stream.trim().encoder_delay(),
        Some(u64::from(head.pre_skip))
    );
    let mut decoder = OpusDecoder::with_head(head.clone(), 48_000).expect("decoder");
    let mut decoded = Vec::new();
    let mut first = None;
    let mut modes = [0; 3];
    let mut frame_sizes = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        if packet.stream() != stream.id().get() {
            continue;
        }
        first.get_or_insert(packet.timestamp().presentation_or_decode().expect("pts"));
        let toc = Toc::parse(packet.bytes()[0]);
        modes[match toc.mode {
            Mode::SilkOnly => 0,
            Mode::Hybrid => 1,
            Mode::CeltOnly => 2,
        }] += 1;
        if !frame_sizes.contains(&toc.frame_samples) {
            frame_sizes.push(toc.frame_samples);
        }
        decoded.extend(decoder.decode(packet.bytes()).expect("decode"));
    }
    let edit = stream.edits()[0];
    let start = (edit.media_time().expect("media time") - first.expect("packets")) * 48_000 / scale;
    let start = usize::try_from(start).expect("start");
    let length = usize::try_from(edit.duration() * 48_000 / scale).expect("duration");
    let pcm = decoded[start * channels..(start + length) * channels].to_vec();
    Ours {
        channels,
        pcm,
        pre_skip: head.pre_skip,
        modes,
        frame_sizes,
        scale,
    }
}

fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (&r, &t) in reference.iter().zip(test) {
        signal += f64::from(r) * f64::from(r);
        noise += f64::from(r - t) * f64::from(r - t);
    }
    if noise == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal / noise).log10()
}

/// Encodes `source` with ffmpeg into `out`, with `args` after the input.
fn ffmpeg_encode(source: &Path, args: &[&str], out: &Path) {
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(source)
        .args(args)
        .arg(out)
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "encode {}", out.display());
}

#[test]
fn files_ffmpeg_writes_decode_like_libopus() {
    if !has_libopus() {
        eprintln!("skipped: ffmpeg with libopus is missing");
        return;
    }
    let dir = scratch("read");
    let source = source(&dir);
    let cases: [(&str, &[&str]); 9] = [
        ("celt-96k", &["-c:a", "libopus", "-b:a", "96k"]),
        (
            "silk-nb-mono",
            &[
                "-ac",
                "1",
                "-ar",
                "8000",
                "-c:a",
                "libopus",
                "-b:a",
                "8k",
                "-application",
                "voip",
            ],
        ),
        (
            "silk-wb-mono",
            &[
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libopus",
                "-b:a",
                "16k",
                "-application",
                "voip",
            ],
        ),
        (
            "hybrid-32k",
            &["-c:a", "libopus", "-b:a", "32k", "-application", "voip"],
        ),
        (
            "lowdelay-2.5ms",
            &[
                "-c:a",
                "libopus",
                "-b:a",
                "128k",
                "-frame_duration",
                "2.5",
                "-application",
                "lowdelay",
            ],
        ),
        (
            "silk-60ms",
            &[
                "-ac",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                "12k",
                "-frame_duration",
                "60",
                "-application",
                "voip",
            ],
        ),
        (
            "lossy-fec",
            &[
                "-c:a",
                "libopus",
                "-b:a",
                "24k",
                "-fec",
                "1",
                "-packet_loss",
                "20",
                "-application",
                "voip",
            ],
        ),
        ("native", &["-c:a", "opus", "-strict", "-2", "-b:a", "96k"]),
        (
            "surround",
            &[
                "-af",
                "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.2*c0|BL=0.7*c0|BR=0.7*c1",
                "-c:a",
                "libopus",
                "-b:a",
                "256k",
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (name, args) in cases {
        let ogg = dir.join(format!("{name}.opus"));
        ffmpeg_encode(&source, args, &ogg);
        let ours_ogg = ours(&ogg);
        let mut reference = ffmpeg_decode(&ogg, "libopus");
        if ours_ogg.channels == 6 {
            // ffmpeg's 5.1 order (FL FR FC LFE BL BR) from Vorbis order
            // (FL FC FR BL BR LFE), which this decoder keeps.
            reference = reference
                .as_chunks::<6>()
                .0
                .iter()
                .flat_map(|f| [f[0], f[2], f[1], f[4], f[5], f[3]])
                .collect();
        }
        let snr = snr_db(&reference, &ours_ogg.pcm);
        eprintln!(
            "{name}.opus: {} ch, pre-skip {}, SILK/hybrid/CELT packets {:?}, frame sizes {:?}, \
             {} samples (libopus {}), SNR {snr:.1} dB",
            ours_ogg.channels,
            ours_ogg.pre_skip,
            ours_ogg.modes,
            ours_ogg.frame_sizes,
            ours_ogg.pcm.len() / ours_ogg.channels,
            reference.len() / ours_ogg.channels,
        );
        if reference.len() != ours_ogg.pcm.len() || snr < MIN_SNR_DB {
            failures.push(format!("{name}.opus"));
        }
        if ours_ogg.channels > 2 {
            // ffmpeg writes more than two Opus channels in MP4 only
            // behind -strict -2.
            continue;
        }
        // The same encode in MP4 must give exactly the Ogg file's samples.
        // ffmpeg's MP4 reader is no reference for the count: it plays the
        // last packet whole, past the end trim; and when ffmpeg wrote the
        // track at the input's rate (8 or 16 kHz here) it skips the edit's
        // media time as if counted at 48 kHz (52 samples, not 312). Its
        // samples are compared only for 48 kHz tracks.
        let mp4 = dir.join(format!("{name}.mp4"));
        ffmpeg_encode(&source, args, &mp4);
        let ours_mp4 = ours(&mp4);
        let reference = ffmpeg_decode(&mp4, "libopus");
        let snr = snr_db(
            &reference[..ours_mp4.pcm.len().min(reference.len())],
            &ours_mp4.pcm,
        );
        eprintln!(
            "{name}.mp4: {} Hz track, {} samples, the Ogg file's samples {}, SNR {snr:.1} dB",
            ours_mp4.scale,
            ours_mp4.pcm.len() / ours_mp4.channels,
            if ours_mp4.pcm == ours_ogg.pcm {
                "exactly"
            } else {
                "NOT"
            },
        );
        if ours_mp4.pcm != ours_ogg.pcm || (ours_mp4.scale == 48_000 && snr < MIN_SNR_DB) {
            failures.push(format!("{name}.mp4"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{failures:?}");
}

fn read_source(file: &Path) -> Vec<f32> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(file)
        .args(["-f", "f32le", "-"])
        .output()
        .expect("ffmpeg");
    floats(&out.stdout)
}

fn encode(pcm: &[f32]) -> Vec<cutlass_encode::Packet> {
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
    params.bitrate = 96_000;
    let mut encoder = OpusEncoder::open(&params).expect("encoder");
    let mut packets = Vec::new();
    // Uneven pushes, as a timeline export gives them.
    for chunk in pcm.chunks(2 * 1777) {
        let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), chunk.to_vec()).expect("pcm");
        packets.extend(encoder.push(&AudioInput { pcm }).expect("push"));
    }
    packets.extend(encoder.drain().expect("drain"));
    packets
}

fn mux(packets: &[cutlass_encode::Packet], mux: &mut dyn Mux) {
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Opus),
                48_000,
                2,
                Vec::new(),
            )
            .expect("params"),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect("stream");
    let mut ticks = 0;
    for packet in packets {
        for side in &packet.side {
            if let SideData::Config(config) = side {
                mux.set_config(track, config).expect("config");
            }
        }
        let time = Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
            .expect("time");
        let out = Packet::try_new(
            track.get(),
            packet.data.clone(),
            time,
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
}

fn ffprobe(file: &Path, entries: &str) -> String {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            entries,
            "-of",
            "default=nw=1",
        ])
        .arg(file)
        .output()
        .expect("ffprobe");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn files_written_here_open_in_ffmpeg_with_every_sample() {
    if !has_libopus() {
        eprintln!("skipped: ffmpeg with libopus is missing");
        return;
    }
    let dir = scratch("write");
    let input = read_source(&source(&dir));
    // An odd length, so the last packet carries an end trim.
    let input = &input[..input.len() - 2 * 333];
    let samples = input.len() / 2;
    let packets = encode(input);

    let ogg = dir.join("ours.opus");
    let mut writer = OggMux::new(File::create(&ogg).expect("create"));
    mux(&packets, &mut writer);
    drop(writer);
    let mp4 = dir.join("ours.mp4");
    let mut writer = Mp4Mux::new(File::create(&mp4).expect("create"));
    mux(&packets, &mut writer);
    writer.into_writer().expect("mp4");

    for file in [&ogg, &mp4] {
        let probe = ffprobe(
            file,
            "stream=codec_name,sample_rate,channels,initial_padding",
        );
        eprintln!("{}: {}", file.display(), probe.replace('\n', " "));
        assert!(probe.contains("codec_name=opus"), "{probe}");
        assert!(probe.contains("sample_rate=48000"), "{probe}");
        assert!(probe.contains("channels=2"), "{probe}");
        // The pre-skip, from the OpusHead or the dOps box.
        assert!(probe.contains("initial_padding=120"), "{probe}");
        let mut native = ffmpeg_decode(file, "opus");
        let mut libopus = ffmpeg_decode(file, "libopus");
        let mine = ours(file);
        assert_eq!(mine.pre_skip, 120);
        if file == &mp4 {
            // ffmpeg's MP4 reader plays the last packet whole; the edit
            // ffprobe reads ends at the input's last sample.
            let played = packets.len() * 960 - 120;
            assert_eq!(native.len() / 2, played);
            assert_eq!(libopus.len() / 2, played);
            let probe = ffprobe(file, "stream=duration_ts");
            assert!(probe.contains(&format!("duration_ts={samples}")), "{probe}");
            native.truncate(2 * samples);
            libopus.truncate(2 * samples);
        }
        eprintln!(
            "  native {} samples, libopus {}, ours {}, input {samples}; libopus vs ours SNR {:.1} dB",
            native.len() / 2,
            libopus.len() / 2,
            mine.pcm.len() / 2,
            snr_db(&libopus, &mine.pcm)
        );
        assert_eq!(native.len() / 2, samples, "{}", file.display());
        assert_eq!(libopus.len() / 2, samples, "{}", file.display());
        assert_eq!(mine.pcm.len() / 2, samples, "{}", file.display());
        assert!(snr_db(&libopus, &mine.pcm) >= MIN_SNR_DB);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
