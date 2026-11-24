//! Files in `tests/fixtures` that are unfinished, cut, or use a corner of
//! the format: each opens and plays what it holds, as libavformat plays
//! it. Every fixture was written from the same 100 samples, [`sequence`],
//! so the 16-bit sound each must give is known here without a decoder,
//! and is compared with `ffmpeg -f s16le` when `ffmpeg` is on the `PATH`.

use std::io::Cursor;
use std::process::Command;

use cutlass_aiff::AiffDemux;
use cutlass_container::{Demux, Landing, SeekTo, TrackId};

fn path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(path(name)).unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// The samples the fixtures were written from: a linear congruential
/// sequence.
fn sequence() -> Vec<i16> {
    let mut state = 12_345_u64;
    (0..100)
        .map(|_| {
            state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF;
            let sample = i32::try_from((state >> 8) & 0xFFFF).expect("sample") - 32_768;
            i16::try_from(sample).expect("i16")
        })
        .collect()
}

/// The top byte of each sample, as an 8-bit file keeps it.
fn top_byte(samples: &[i16]) -> Vec<i16> {
    samples.iter().map(|sample| sample & !0xFF).collect()
}

struct Expect {
    name: &'static str,
    /// The PCM layout bytes the stream must carry.
    extra: [u8; 3],
    rate: u32,
    cut: bool,
    /// The sound as 16-bit samples.
    sound: Vec<i16>,
}

fn expectations() -> Vec<Expect> {
    let whole = sequence();
    let row = |name, extra, sound: Vec<i16>| Expect {
        name,
        extra,
        rate: 8_000,
        cut: false,
        sound,
    };
    vec![
        // Offset binary, 128 for silence.
        row("raw-unsigned.aifc", [11, 1, 0], top_byte(&whole)),
        // 99 one-byte frames; the pad byte after the odd `SSND` is absent.
        row("no-final-pad.aiff", [6, 1, 0], top_byte(&whole[..99])),
        // The file stops inside frame 70 of 100.
        Expect {
            cut: true,
            ..row("ssnd-past-end.aiff", [1, 1, 0], whole[..69].to_vec())
        },
        // The form states 1,000 bytes more than the file holds.
        row("form-past-end.aiff", [1, 1, 0], whole.clone()),
        // `COMM` counts no frames; `SSND` holds 100.
        row("zero-frames.aiff", [1, 1, 0], whole.clone()),
        // 244800 / 11 Hz.
        Expect {
            rate: 22_255,
            ..row("rate-22254.aiff", [1, 1, 0], whole.clone())
        },
        // 12 bits in two bytes: the low four bits are zero.
        row(
            "depth-12.aiff",
            [1, 1, 0],
            whole.iter().map(|sample| sample & !0xF).collect(),
        ),
        // 20 bits in three bytes.
        row("depth-20.aiff", [2, 1, 0], whole.clone()),
        row("fl32-upper.aifc", [4, 1, 0], whole.clone()),
        row("fl64-upper.aifc", [5, 1, 0], whole),
    ]
}

/// Packet bytes as 16-bit samples, the way `ffmpeg -f s16le` narrows
/// them: the top two bytes of a wider integer, a float times 2^15.
fn to_s16(extra: [u8; 3], bytes: &[u8]) -> Vec<i16> {
    assert_eq!(extra[1], 1, "big-endian");
    assert_eq!(extra[2], 0, "interleaved");
    let float = |value: f64| -> i16 {
        let scaled = (value * 32_768.0).round().clamp(-32_768.0, 32_767.0);
        #[allow(clippy::cast_possible_truncation)]
        {
            scaled as i16
        }
    };
    match extra[0] {
        1 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_be_bytes(*pair))
            .collect(),
        2 => bytes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|three| i16::from_be_bytes([three[0], three[1]]))
            .collect(),
        4 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|four| float(f64::from(f32::from_be_bytes(*four))))
            .collect(),
        5 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|eight| float(f64::from_be_bytes(*eight)))
            .collect(),
        6 => bytes
            .iter()
            .map(|byte| i16::from(i8::from_ne_bytes([*byte])) << 8)
            .collect(),
        11 => bytes
            .iter()
            .map(|byte| (i16::from(*byte) - 128) << 8)
            .collect(),
        other => panic!("coding {other}"),
    }
}

fn open(bytes: &[u8]) -> AiffDemux<Cursor<Vec<u8>>> {
    let len = u64::try_from(bytes.len()).expect("len");
    AiffDemux::open(Cursor::new(bytes.to_vec()), len).expect("open")
}

fn sound<R: std::io::Read + std::io::Seek + Send>(demux: &mut AiffDemux<R>) -> (Vec<u8>, i64) {
    let mut bytes = Vec::new();
    let mut frames = 0;
    while let Some(packet) = demux.read_packet().expect("packet") {
        assert_eq!(packet.timestamp().presentation(), Some(frames));
        frames += packet.timestamp().duration().expect("duration");
        bytes.extend_from_slice(packet.bytes());
    }
    (bytes, frames)
}

#[test]
fn every_fixture_opens_and_plays_the_frames_it_holds() {
    for expect in expectations() {
        let name = expect.name;
        let frames = i64::try_from(expect.sound.len()).expect("frames");
        let mut demux = open(&fixture(name));
        let extra: [u8; 3] = demux.streams()[0]
            .params()
            .extradata()
            .try_into()
            .expect("layout");
        assert_eq!(extra, expect.extra, "{name}");
        assert_eq!(demux.sample_rate(), expect.rate, "{name}");
        assert_eq!(demux.is_cut(), expect.cut, "{name}");
        assert_eq!(
            demux.duration_samples(),
            u64::try_from(frames).expect("frames"),
            "{name}"
        );
        assert_eq!(demux.indexed_end(TrackId::new(0)), Some(frames), "{name}");
        let (bytes, counted) = sound(&mut demux);
        assert_eq!(counted, frames, "{name}");
        assert_eq!(to_s16(extra, &bytes), expect.sound, "{name}");

        // The last whole frame is a landing; one past it is the end.
        let last = demux
            .seek(SeekTo {
                track: 0,
                time: frames - 1,
                end: None,
            })
            .expect("seek");
        assert!(
            matches!(last.landing, Landing::Packet { time, .. } if time == frames - 1),
            "{name}"
        );
        let packet = demux.read_packet().expect("read").expect("last frame");
        assert_eq!(
            to_s16(extra, packet.bytes()),
            &expect.sound[expect.sound.len() - 1..],
            "{name}"
        );
        assert!(demux.read_packet().expect("end").is_none(), "{name}");

        // A forward reader plays the same sound, and has learnt of a cut
        // by the time the input ends.
        let mut forward = AiffDemux::open_forward(Cursor::new(fixture(name))).expect("forward");
        let (bytes, counted) = sound(&mut forward);
        assert_eq!(counted, frames, "{name}");
        assert_eq!(to_s16(extra, &bytes), expect.sound, "{name}");
        assert_eq!(forward.is_cut(), expect.cut, "{name}");
    }
}

#[test]
fn ffmpeg_plays_the_same_samples_from_every_fixture() {
    let found = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !found {
        eprintln!("skipped: ffmpeg is not on PATH");
        return;
    }
    for expect in expectations() {
        let name = expect.name;
        let file = path(name);
        let out = Command::new("ffmpeg")
            .args(["-v", "quiet", "-i"])
            .arg(&file)
            .args(["-f", "s16le", "-"])
            .output()
            .expect("ffmpeg");
        let theirs: Vec<i16> = out
            .stdout
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        assert!(!theirs.is_empty(), "{name}");
        assert_eq!(theirs, expect.sound, "{name}");
        let probed = Command::new("ffprobe")
            .args(["-v", "error", "-show_entries", "stream=sample_rate"])
            .args(["-of", "csv=p=0"])
            .arg(&file)
            .output();
        if let Ok(probed) = probed
            && probed.status.success()
        {
            let rate = String::from_utf8_lossy(&probed.stdout);
            assert_eq!(rate.trim(), expect.rate.to_string(), "{name}");
        }
    }
}
