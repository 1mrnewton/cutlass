//! WAV files written from the PCM decoder's three layout bytes, as an MP4,
//! CAF or Matroska demuxer hands them over, read back by this crate,
//! ffprobe, ffmpeg and `afinfo`.

use std::io::Cursor;

use crate::support::{ffprobe_stream, output, tool_on_path};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_riff::{WavDemux, WavMux};

/// Layout bytes, bytes per sample, channels, rate, ffprobe codec name,
/// `afinfo` format text.
type Case = ([u8; 3], usize, u16, u32, &'static str, &'static str);

const CASES: &[Case] = &[
    (
        [1, 0, 0],
        2,
        2,
        44_100,
        "pcm_s16le",
        "2 ch,  44100 Hz, Int16",
    ),
    (
        [2, 0, 0],
        3,
        2,
        48_000,
        "pcm_s24le",
        "2 ch,  48000 Hz, lpcm (0x0000000C) 24-bit little-endian signed integer",
    ),
    (
        [3, 0, 0],
        4,
        1,
        48_000,
        "pcm_s32le",
        "1 ch,  48000 Hz, Int32",
    ),
    (
        [9, 0, 0],
        8,
        1,
        48_000,
        "pcm_s64le",
        "1 ch,  48000 Hz, lpcm (0x0000000C) 64-bit little-endian signed integer",
    ),
    (
        [11, 0, 0],
        1,
        2,
        22_050,
        "pcm_u8",
        "2 ch,  22050 Hz, lpcm (0x00000008) 8-bit unsigned integer",
    ),
    (
        [11, 1, 0],
        1,
        1,
        8_000,
        "pcm_u8",
        "1 ch,   8000 Hz, lpcm (0x00000008) 8-bit unsigned integer",
    ),
    (
        [4, 0, 0],
        4,
        2,
        48_000,
        "pcm_f32le",
        "2 ch,  48000 Hz, Float32",
    ),
    (
        [5, 0, 0],
        8,
        1,
        44_100,
        "pcm_f64le",
        "1 ch,  44100 Hz, Float64",
    ),
    ([7, 0, 0], 1, 1, 8_000, "pcm_alaw", "1 ch,   8000 Hz, alaw"),
    ([8, 1, 0], 1, 1, 8_000, "pcm_mulaw", "1 ch,   8000 Hz, ulaw"),
    (
        [2, 0, 0],
        3,
        6,
        96_000,
        "pcm_s24le",
        "6 ch,  96000 Hz, lpcm (0x0000000C) 24-bit little-endian signed integer",
    ),
];

const FRAMES: usize = 480;

fn payload(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| u8::try_from((index * 37 + 11) % 251).expect("byte"))
        .collect()
}

fn wav(layout: [u8; 3], width: usize, channels: u16, rate: u32) -> Result<Vec<u8>, String> {
    let params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Pcm),
        rate,
        channels,
        layout.to_vec(),
    )
    .map_err(|error| error.to_string())?;
    let frames = i64::try_from(FRAMES).expect("frames");
    let timestamp = Timestamp::try_new(Some(0), Some(0), Some(frames)).expect("time");
    let bytes = payload(FRAMES * width * usize::from(channels));
    let packet = Packet::try_new(0, bytes, timestamp, Timescale::new(rate), true).expect("packet");
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(rate),
        language: None,
    })
    .map_err(|error| error.to_string())?;
    mux.write_packet(&packet)
        .map_err(|error| error.to_string())?;
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks: frames,
    }])
    .map_err(|error| format!("{error:?}"))?;
    Ok(mux.into_inner().into_inner())
}

#[test]
fn every_wav_layout_reads_back_as_written() {
    for &(layout, width, channels, rate, _, _) in CASES {
        let bytes = wav(layout, width, channels, rate).expect("mux");
        let len = u64::try_from(bytes.len()).expect("len");
        let mut demux = WavDemux::open(Cursor::new(bytes), len).expect("demux");
        let format = demux.format().clone();
        assert_eq!(format.channels(), channels, "{layout:?}");
        assert_eq!(format.sample_rate(), rate, "{layout:?}");
        assert_eq!(usize::from(format.bits()), width * 8, "{layout:?} bits");
        assert_eq!(format.is_float(), matches!(layout[0], 4 | 5), "{layout:?}");
        let mut gathered = Vec::new();
        while let Some(packet) = demux.read_packet().expect("packet") {
            gathered.extend_from_slice(packet.bytes());
        }
        assert_eq!(
            gathered,
            payload(FRAMES * width * usize::from(channels)),
            "{layout:?}"
        );
    }
}

#[test]
fn wav_refuses_layouts_it_cannot_state() {
    for layout in [
        [6, 0, 0],
        [1, 1, 0],
        [4, 1, 0],
        [1, 0, 1],
        [10, 0, 0],
        [1, 2, 0],
    ] {
        let refused = wav(layout, 2, 2, 48_000);
        assert!(refused.is_err(), "{layout:?} was written");
    }
}

#[test]
fn ffmpeg_and_afinfo_read_each_wav_layout_as_written() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg and ffprobe are not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutlass-riff-layouts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp");
    for (index, &(layout, width, channels, rate, codec, afinfo)) in CASES.iter().enumerate() {
        let path = dir.join(format!("layout-{index}.wav"));
        std::fs::write(&path, wav(layout, width, channels, rate).expect("mux")).expect("write");
        let shown = path.display().to_string();
        let probed = ffprobe_stream(&path);
        assert_eq!(probed.codec_name, codec, "{layout:?}");
        assert_eq!(probed.sample_rate, rate, "{layout:?}");
        assert_eq!(probed.channels, channels, "{layout:?}");
        assert_eq!(
            probed.duration_ts,
            Some(u64::try_from(FRAMES).expect("frames")),
            "{layout:?}"
        );
        let copied = output(
            "ffmpeg",
            &[
                "-v", "error", "-i", &shown, "-map", "0:a", "-c", "copy", "-f", "data", "-",
            ],
        );
        assert!(
            copied.status.success(),
            "{layout:?}: {}",
            String::from_utf8_lossy(&copied.stderr)
        );
        assert_eq!(
            copied.stdout,
            payload(FRAMES * width * usize::from(channels)),
            "{layout:?} bytes"
        );
        if tool_on_path("afinfo") {
            let info = output("afinfo", &[&shown]);
            let text = String::from_utf8_lossy(&info.stdout);
            assert!(text.contains(afinfo), "{layout:?}: {text}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
