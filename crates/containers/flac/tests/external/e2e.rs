//! A real recording from `../local-files` through every FLAC stage, the way
//! the engine and export run them: demux, decode to the mix format, encode,
//! mux. `flac -t` and ffmpeg then read the result.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use cutlass_codec::{CodecId, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_decode::{ChannelLayout, CodecParams, Decoder, Output, SampleFormat, SideData};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_flac::{FlacDecoder, FlacEncoder};
use cutlass_flac_container::{FlacDemux, FlacMux};

use crate::support::{ffmpeg_samples, have, path_str, run, scratch, succeeds};

fn local_files() -> Option<std::path::PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if root.is_dir() {
        Some(root)
    } else {
        eprintln!("skipped: ../local-files is missing");
        None
    }
}

/// Demux, decode, encode, and mux `source` into `out`, as `format`.
fn transcode(source: &Path, out: &Path, format: SampleFormat) -> (u64, u32) {
    let file = File::open(source).expect("open");
    let length = file.metadata().expect("metadata").len();
    let mut demux = FlacDemux::open(file, length).expect("demux");
    let stream = demux.streams()[0].clone();
    let rate = stream.params().sample_rate().expect("rate");
    let channels = stream.params().channels().expect("channels");
    let layout = match channels {
        1 => ChannelLayout::mono(),
        _ => ChannelLayout::stereo(),
    };
    let mut decode_params = CodecParams::audio(CodecId::Flac, rate, layout.clone());
    decode_params.extra = stream.params().extradata().to_vec();
    decode_params.sample_format = Some(format);
    let mut decoder = FlacDecoder::open(&decode_params).expect("decoder");

    let mut encode_params = CodecParams::audio(CodecId::Flac, rate, layout);
    encode_params.sample_format = Some(format);
    if format == SampleFormat::F32 {
        // Keep the source depth: a 16-bit source stays 16-bit.
        encode_params.extra = stream.params().extradata().to_vec();
    }
    let mut encoder = FlacEncoder::open(&encode_params).expect("encoder");

    let mut mux = FlacMux::new(File::create(out).expect("create"));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Flac),
                rate,
                channels,
                Vec::new(),
            )
            .expect("params"),
            timescale: Timescale::new(rate),
            language: None,
        })
        .expect("stream");
    let mut ticks = 0i64;
    let mut write = |mux: &mut FlacMux<File>, packets: Vec<cutlass_decode::Packet>| {
        for packet in packets {
            for side in &packet.side {
                if let SideData::Config(config) = side {
                    mux.set_config(track, config).expect("config");
                }
            }
            let timestamp =
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .expect("timestamp");
            let file_packet = cutlass_codec::Packet::try_new(
                0,
                packet.data,
                timestamp,
                Timescale::new(packet.timescale),
                true,
            )
            .expect("packet");
            mux.write_packet(&file_packet).expect("write");
            ticks += packet.duration;
        }
    };
    let mut frames = 0u32;
    while let Some(packet) = demux.read_packet().expect("read") {
        let access = cutlass_decode::Packet::audio(
            packet.bytes().to_vec(),
            packet.timestamp().presentation().expect("pts"),
            packet.timestamp().duration().expect("duration"),
            rate,
        );
        for output in decoder.push(&access).expect("decode") {
            let Output::Audio(audio) = output else {
                panic!("audio");
            };
            let packets = encoder
                .push(&AudioInput { pcm: audio.pcm })
                .expect("encode");
            write(&mut mux, packets);
        }
        frames += 1;
    }
    let packets = encoder.drain().expect("drain");
    write(&mut mux, packets);
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(rate),
        ticks,
    }])
    .expect("finish");
    (u64::try_from(ticks).expect("ticks"), frames)
}

#[test]
fn a_real_recording_survives_demux_decode_encode_and_mux() {
    let Some(root) = local_files() else {
        return;
    };
    if !have("flac") || !have("ffmpeg") || !have("metaflac") {
        return;
    }
    let dir = scratch("e2e");
    let picks = [
        ("m4a-stereo", "videoplayback.m4a"),
        ("speech-mono", "GOOGLE IS BACK! (Gemini 3.8 Flash).wav"),
    ];
    for (name, file) in picks {
        let input = root.join(file);
        if !input.is_file() {
            eprintln!("skipped {name}: {file} is missing");
            continue;
        }
        // The source FLAC, written by libFLAC from the first 90 seconds.
        let wav = dir.join(format!("{name}.wav"));
        run(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-t",
                "90",
                "-i",
                path_str(&input),
                "-c:a",
                "pcm_s16le",
                path_str(&wav),
            ],
        );
        let source = dir.join(format!("{name}.flac"));
        run(
            "flac",
            &["-s", "-f", "-5", path_str(&wav), "-o", path_str(&source)],
        );
        let expected = ffmpeg_samples(&source, 16);
        for format in [SampleFormat::I16, SampleFormat::F32] {
            let out = dir.join(format!("{name}-{format:?}.flac"));
            let started = Instant::now();
            let (samples, frames) = transcode(&source, &out, format);
            let elapsed = started.elapsed();
            let (ok, stderr) = succeeds("flac", &["-s", "-t", path_str(&out)]);
            assert!(ok, "{name} {format:?}: flac -t refused: {stderr}");
            assert!(
                ffmpeg_samples(&out, 16) == expected,
                "{name} {format:?}: samples differ"
            );
            let md5 = |path: &Path| run("metaflac", &["--show-md5sum", path_str(path)]);
            assert_eq!(md5(&out), md5(&source), "{name} {format:?}: md5");
            let sizes = (
                std::fs::metadata(&source).expect("size").len(),
                std::fs::metadata(&out).expect("size").len(),
            );
            eprintln!(
                "{name} {format:?}: {frames} frames, {samples} samples per channel in {elapsed:?}; \
                 libflac -5 {} bytes, ours {} bytes; flac -t, ffmpeg, md5 agree",
                sizes.0, sizes.1
            );
        }
    }
}
