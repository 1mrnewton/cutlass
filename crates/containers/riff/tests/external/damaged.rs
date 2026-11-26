//! Files a recorder, a pipe or a download left unfinished, in
//! `tests/fixtures`. Each opens and plays what it holds, as libavformat
//! plays it. The sound of every fixture is 16-bit little-endian, so the
//! joined packets are what `ffmpeg -f s16le` writes; that is compared
//! when `ffmpeg` is on the `PATH`.
//!
//! `mono` below is the 200 frames every mono fixture was written from and
//! `stereo` the 100 frames of the stereo one.

use std::io::Cursor;

use crate::support::{output, tool_on_path};
use cutlass_container::{Demux, OpenError, ProbeResult, ReadError, SeekError, SeekTo, TrackId};
use cutlass_riff::{Brand, WavDemux, probe};

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The generator the fixtures were written from: 16-bit samples of a
/// linear congruential sequence.
fn sequence(samples: usize) -> Vec<u8> {
    let mut state = 12_345_u64;
    let mut bytes = Vec::with_capacity(samples * 2);
    for _ in 0..samples {
        state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF;
        let sample = i32::try_from((state >> 8) & 0xFFFF).expect("sample") - 32_768;
        bytes.extend_from_slice(&i16::try_from(sample).expect("i16").to_le_bytes());
    }
    bytes
}

fn open(bytes: &[u8]) -> WavDemux<Cursor<Vec<u8>>> {
    let len = u64::try_from(bytes.len()).expect("len");
    WavDemux::open(Cursor::new(bytes.to_vec()), len).expect("open")
}

fn sound<R: std::io::Read + std::io::Seek + Send>(demux: &mut WavDemux<R>) -> (Vec<u8>, i64) {
    let mut bytes = Vec::new();
    let mut frames = 0;
    while let Some(packet) = demux.read_packet().expect("packet") {
        assert_eq!(packet.timestamp().presentation(), Some(frames));
        frames += packet.timestamp().duration().expect("duration");
        bytes.extend_from_slice(packet.bytes());
    }
    (bytes, frames)
}

/// What one fixture must play, on a seekable reader and on a forward one.
struct Expect {
    name: &'static str,
    sound: Vec<u8>,
    frames: i64,
    cut: bool,
    open_ended: bool,
    /// A forward reader learns of a cut only when the input ends.
    forward_cut: bool,
}

fn expectations() -> Vec<Expect> {
    let mono = sequence(200);
    let stereo = sequence(200);
    vec![
        Expect {
            name: "streamed-ffmpeg-pipe.wav",
            sound: mono.clone(),
            frames: 200,
            cut: false,
            open_ended: true,
            forward_cut: false,
        },
        Expect {
            name: "data-size-zero.wav",
            sound: mono.clone(),
            frames: 200,
            cut: false,
            open_ended: true,
            forward_cut: false,
        },
        Expect {
            name: "data-past-end.wav",
            sound: mono[..300].to_vec(),
            frames: 150,
            cut: true,
            open_ended: false,
            forward_cut: true,
        },
        Expect {
            name: "data-partial-frame.wav",
            sound: stereo,
            frames: 100,
            cut: true,
            open_ended: false,
            forward_cut: true,
        },
        Expect {
            name: "rf64-sample-count-zero.wav",
            sound: mono.clone(),
            frames: 200,
            cut: false,
            open_ended: false,
            forward_cut: false,
        },
        Expect {
            name: "bw64.wav",
            sound: mono.clone(),
            frames: 200,
            cut: false,
            open_ended: false,
            forward_cut: false,
        },
        Expect {
            name: "trailing-chunk-cut.wav",
            sound: mono,
            frames: 200,
            cut: false,
            open_ended: false,
            forward_cut: false,
        },
    ]
}

#[test]
fn an_unfinished_file_plays_the_whole_frames_it_holds() {
    for expect in expectations() {
        let name = expect.name;
        let bytes = fixture(name);
        let mut demux = open(&bytes);
        assert_eq!(demux.is_cut(), expect.cut, "{name}");
        assert_eq!(demux.is_open_ended(), expect.open_ended, "{name}");
        let frames = u64::try_from(expect.frames).expect("frames");
        assert_eq!(demux.duration_samples(), Some(frames), "{name}");
        assert_eq!(
            demux.indexed_end(TrackId::new(0)),
            Some(expect.frames),
            "{name}"
        );
        let (played, counted) = sound(&mut demux);
        assert_eq!(counted, expect.frames, "{name}");
        assert!(played == expect.sound, "{name}");

        // The last whole frame is a landing, and one past it is the end.
        let last = demux
            .seek(SeekTo {
                track: 0,
                time: expect.frames - 1,
                end: None,
            })
            .expect("seek");
        assert!(
            matches!(last.landing, cutlass_container::Landing::Packet { time, .. } if time == expect.frames - 1),
            "{name}"
        );
        let packet = demux.read_packet().expect("read").expect("last frame");
        let frame = expect.sound.len() / usize::try_from(expect.frames).expect("frames");
        assert_eq!(
            packet.bytes(),
            &expect.sound[expect.sound.len() - frame..],
            "{name}"
        );
        assert!(demux.read_packet().expect("end").is_none(), "{name}");
    }
}

#[test]
fn a_forward_reader_plays_an_unfinished_file_to_the_end_of_the_input() {
    for expect in expectations() {
        let name = expect.name;
        let mut demux = WavDemux::open_forward(Cursor::new(fixture(name))).expect("forward");
        assert_eq!(demux.is_open_ended(), expect.open_ended, "{name}");
        if expect.open_ended {
            // Nothing states the length before the input ends.
            assert_eq!(demux.duration_samples(), None, "{name}");
            assert_eq!(demux.indexed_end(TrackId::new(0)), None, "{name}");
        }
        let (played, counted) = sound(&mut demux);
        assert_eq!(counted, expect.frames, "{name}");
        assert!(played == expect.sound, "{name}");
        assert_eq!(demux.is_cut(), expect.forward_cut, "{name}");
        assert!(
            demux.read_packet().expect("still ended").is_none(),
            "{name}"
        );
        assert!(matches!(
            demux.seek(SeekTo {
                track: 0,
                time: 0,
                end: None
            }),
            Err(ReadError::Seek(SeekError::ForwardOnly))
        ));
    }
}

#[test]
fn bw64_is_its_own_brand_and_the_trailing_chunk_is_named() {
    let bytes = fixture("bw64.wav");
    assert!(matches!(probe(&bytes[..12]), ProbeResult::Accept { .. }));
    assert_eq!(open(&bytes).brand(), Brand::Bw64);
    assert_eq!(
        open(&fixture("rf64-sample-count-zero.wav")).brand(),
        Brand::Rf64
    );

    let cut = open(&fixture("trailing-chunk-cut.wav"));
    let dropped: Vec<[u8; 4]> = cut.dropped().iter().map(|chunk| chunk.id()).collect();
    assert_eq!(dropped, [*b"LIST"]);
}

/// The damage that is still refused: a header that is not all there, and
/// a size that is zero in a file that is otherwise finished.
#[test]
fn a_cut_header_is_refused_and_an_empty_finished_file_is_empty() {
    let whole = fixture("data-past-end.wav");
    for keep in [0, 11, 12, 20, 35, 43] {
        let len = u64::try_from(keep).expect("len");
        let opened = WavDemux::open(Cursor::new(whole[..keep].to_vec()), len);
        let error = match opened {
            Ok(_) => panic!("{keep} bytes opened"),
            Err(error) => error,
        };
        assert!(!error.streams_published(), "{keep}");
        assert!(
            matches!(
                error,
                OpenError::Truncated { .. }
                    | OpenError::Source { .. }
                    | OpenError::HostileSize { .. }
            ),
            "{keep}: {error}"
        );
    }
    // The header alone: the sound chunk is there and holds nothing.
    let mut empty = open(&whole[..44]);
    assert!(empty.is_cut());
    assert_eq!(empty.duration_samples(), Some(0));
    assert!(empty.read_packet().expect("read").is_none());

    // A finished file whose `data` is empty and is followed by a chunk:
    // the size is zero because there is no sound, and the chunk after it
    // is not played.
    let mut finished = fixture("data-size-zero.wav")[..44].to_vec();
    finished.extend_from_slice(b"LIST\x04\x00\x00\x00INFO");
    let form = u32::try_from(finished.len() - 8).expect("form");
    finished[4..8].copy_from_slice(&form.to_le_bytes());
    let mut demux = open(&finished);
    assert!(!demux.is_open_ended());
    assert_eq!(demux.duration_samples(), Some(0));
    assert!(demux.read_packet().expect("read").is_none());
}

#[test]
fn ffmpeg_plays_the_same_samples_from_every_unfinished_file() {
    if !tool_on_path("ffmpeg") {
        eprintln!("skipped: ffmpeg is not on PATH");
        return;
    }
    for expect in expectations() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(expect.name);
        let out = output(
            "ffmpeg",
            &[
                "-v",
                "quiet",
                "-i",
                &path.display().to_string(),
                "-f",
                "s16le",
                "-",
            ],
        );
        let (ours, _) = sound(&mut open(&fixture(expect.name)));
        assert!(!out.stdout.is_empty(), "{}", expect.name);
        assert!(out.stdout == ours, "{}", expect.name);
    }
}

/// A reader that is a header and then silence, for a file too long to hold.
struct Sparse {
    header: Vec<u8>,
    len: u64,
    at: u64,
}

impl std::io::Read for Sparse {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.len.saturating_sub(self.at);
        let take = usize::try_from(left).unwrap_or(usize::MAX).min(buf.len());
        for (index, slot) in buf[..take].iter_mut().enumerate() {
            let at = usize::try_from(self.at)
                .unwrap_or(usize::MAX)
                .saturating_add(index);
            *slot = self.header.get(at).copied().unwrap_or(0);
        }
        self.at += u64::try_from(take).expect("take");
        Ok(take)
    }
}

impl std::io::Seek for Sparse {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        if let std::io::SeekFrom::Start(at) = to {
            self.at = at;
        }
        Ok(self.at)
    }
}

/// Mono 8-bit sound of 2^32 + 10 frames. A seek past frame 2^32 lands on
/// the frame; the ordinal, which is 32 bits, stops at its largest value.
#[test]
fn a_seek_past_two_to_the_32_frames_lands_on_the_frame() {
    let frames = (1_u64 << 32) + 10;
    let mut header = b"RF64\xff\xff\xff\xffWAVEds64".to_vec();
    header.extend_from_slice(&28_u32.to_le_bytes());
    header.extend_from_slice(&(frames + 72).to_le_bytes());
    header.extend_from_slice(&frames.to_le_bytes());
    header.extend_from_slice(&frames.to_le_bytes());
    header.extend_from_slice(&0_u32.to_le_bytes());
    header.extend_from_slice(b"fmt \x10\x00\x00\x00");
    header.extend_from_slice(&1_u16.to_le_bytes());
    header.extend_from_slice(&1_u16.to_le_bytes());
    header.extend_from_slice(&8_000_u32.to_le_bytes());
    header.extend_from_slice(&8_000_u32.to_le_bytes());
    header.extend_from_slice(&1_u16.to_le_bytes());
    header.extend_from_slice(&8_u16.to_le_bytes());
    header.extend_from_slice(b"data\xff\xff\xff\xff");
    let start = u64::try_from(header.len()).expect("header");
    let len = start + frames;
    let reader = Sparse { header, len, at: 0 };
    let mut demux = WavDemux::open(reader, len).expect("open");
    assert_eq!(demux.duration_samples(), Some(frames));
    let target = i64::try_from(frames - 5).expect("target");
    let landed = demux
        .seek(SeekTo {
            track: 0,
            time: target,
            end: None,
        })
        .expect("seek");
    match landed.landing {
        cutlass_container::Landing::Packet {
            ordinal,
            time,
            preroll,
            byte_position,
            ..
        } => {
            assert_eq!(ordinal, u32::MAX);
            assert_eq!(time, target);
            assert_eq!(preroll, 0);
            assert_eq!(byte_position, start + frames - 5);
        }
        cutlass_container::Landing::End => panic!("end"),
    }
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.timestamp().presentation(), Some(target));
    assert_eq!(packet.timestamp().duration(), Some(5));
    assert_eq!(packet.bytes().len(), 5);
    assert!(demux.read_packet().expect("end").is_none());
}
