//! In-process random-mutation campaigns over the demuxer and the muxer.
//!
//! cargo-fuzz is not used: it is not installed here, and libfuzzer-sys
//! builds C++ through `cc`, which this workspace's rules do not allow. The
//! campaigns are deterministic (a fixed xorshift generator per seed), so a
//! failure names the seed and mutant that reproduce it.
//!
//! Demux: valid files (written by [`MatroskaMux`], and hand-built ones with
//! lacing, BlockGroups, unknown sizes, header stripping, Cues before the
//! Clusters, and Tracks found through the SeekHead) are mutated: bit flips,
//! interesting bytes (VINT markers, all-ones sizes), inserts, deletes, cuts,
//! duplicated ranges, and eight-octet sizes written over element headers.
//! Each mutant is probed, opened random-access and forward, read to the
//! end, and seeked. Nothing may panic, no read may return more than the
//! packet limit, and every call returns.
//!
//! Mux: random track setups and packet sequences, including invalid ones.
//! A refused call must leave the mux failed; a mux that finishes must give
//! a file this crate reads back to the same packets per track.
//!
//! `fuzz_seeds_*` run on every `cargo test`. The long campaigns are
//! `#[ignore]`; `MKV_FUZZ_SECS` sets their length (default 60 s) and
//! `MKV_FUZZ_ORIGIN` their first seed (default 1,000 for the demuxer and
//! 100,000 for the muxer), so a new campaign covers new mutants. The demux
//! campaign names its slowest input, and `MKV_FUZZ_REPLAY=seed:mutant`
//! with the ignored `fuzz_replay` times that one input again.

use std::io::Cursor;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Limits, Mux, MuxStatus, NewStream, SeekTo, TrackDuration, TrackId};
use cutlass_matroska::{MatroskaDemux, MatroskaMux, ids, probe};

use crate::build::{
    Track, bytes, cluster, cues, ebml_laced, element, file, fixed, group, info, live_file, simple,
    tracks, uint, unknown_cluster, with_seek_head, xiph,
};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            usize::try_from(self.next() % bound as u64).unwrap_or(0)
        }
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// Seed files for the demux campaign.
fn seeds() -> &'static Vec<Vec<u8>> {
    static SEEDS: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    SEEDS.get_or_init(|| {
        let mut out = vec![muxed(false), muxed(true)];
        let audio = Track::audio(2, "A_MPEG/L3", 48_000.0, 2).with_default_duration(24_000_000);
        let video = Track::video(1, "V_MPEG4/ISO/AVC").with_private(&[1, 0x64, 0, 0x1F, 0xFF]);
        let blocks = vec![
            simple(1, 0, true, &[1; 40]),
            xiph(2, 0, &[b"aaa", b"bbbbbbbb", b"c"]),
            ebml_laced(2, 72, &[b"dd", b"eeeee", b"f"]),
            fixed(2, 144, &[b"gggg", b"hhhh"]),
            group(1, 40, &[2; 30], Some(40), Some(-40)),
            group(2, 200, b"last", Some(5), None),
        ];
        out.push(file(
            "matroska",
            &[
                info(1_000_000, Some(500.0)),
                tracks(&[video.clone(), audio.clone()]),
                cues(&[(0, 1, 0)]),
                cluster(0, &blocks),
                cluster(250, &blocks),
            ],
        ));
        out.push(live_file(
            "webm",
            &[
                info(1_000_000, None),
                tracks(&[Track::video(1, "V_VP8"), audio.clone()]),
                unknown_cluster(0, &blocks),
                unknown_cluster(300, &blocks),
            ],
        ));
        let mut compression = uint(ids::CONTENT_COMP_ALGO, 3);
        compression.extend(bytes(ids::CONTENT_COMP_SETTINGS, &[0xFF, 0xFB]));
        let stripped = element(
            ids::CONTENT_ENCODINGS,
            &element(
                ids::CONTENT_ENCODING,
                &element(ids::CONTENT_COMPRESSION, &compression),
            ),
        );
        out.push(with_seek_head(
            &[ids::INFO, ids::TRACKS],
            &[
                cluster(7, &blocks),
                info(3, None),
                tracks(&[audio.with_extra(&stripped)]),
            ],
        ));
        out
    })
}

/// A two-track file from the muxer, with Cues, BlockGroups and padding.
fn muxed(webm: bool) -> Vec<u8> {
    let writer = Cursor::new(Vec::new());
    let mut mux = if webm {
        MatroskaMux::webm(writer)
    } else {
        MatroskaMux::new(writer)
    };
    let video_codec = if webm { CodecId::Vp9 } else { CodecId::H264 };
    let audio_codec = if webm { CodecId::Opus } else { CodecId::Aac };
    let size = FrameSize::try_new(64, 48).expect("size");
    let picture = mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Known(video_codec), size, vec![1, 2, 3]),
            timescale: Timescale::new(1000),
            language: None,
        })
        .expect("video");
    let config = if webm {
        let mut head = b"OpusHead".to_vec();
        head.extend_from_slice(&[1, 2, 0x38, 1, 0x80, 0xBB, 0, 0, 0, 0, 0]);
        head
    } else {
        vec![0x11, 0x90]
    };
    let sound = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(StreamCodec::Known(audio_codec), 48_000, 2, config)
                .expect("audio"),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect("audio");
    let mut video_ticks = 0;
    let mut audio_ticks = 0;
    for index in 0..60_i64 {
        let time = Timestamp::try_new(None, Some(index * 40), Some(40)).expect("time");
        let packet = Packet::try_new(
            picture.get(),
            vec![7; 30],
            time,
            Timescale::new(1000),
            index % 12 == 0,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("video packet");
        video_ticks += 40;
        for part in 0..2 {
            let pts = (index * 2 + part) * 960;
            let duration = if index == 59 && part == 1 { 500 } else { 960 };
            let time = Timestamp::try_new(None, Some(pts), Some(duration)).expect("time");
            let packet =
                Packet::try_new(sound.get(), vec![5; 12], time, Timescale::new(48_000), true)
                    .expect("packet");
            mux.write_packet(&packet).expect("audio packet");
            audio_ticks += duration;
        }
    }
    mux.finish(&[
        TrackDuration {
            track: picture,
            timescale: Timescale::new(1000),
            ticks: video_ticks,
        },
        TrackDuration {
            track: sound,
            timescale: Timescale::new(48_000),
            ticks: audio_ticks,
        },
    ])
    .expect("finish");
    mux.into_writer().expect("writer").into_inner()
}

const INTERESTING: [u8; 9] = [0x00, 0x01, 0x10, 0x40, 0x7F, 0x80, 0x81, 0xFE, 0xFF];

fn mutate(rng: &mut Rng, input: &[u8]) -> Vec<u8> {
    let mut data = input.to_vec();
    let rounds = 1 + rng.below(4);
    for _ in 0..rounds {
        if data.is_empty() {
            data.push(0x1A);
        }
        let at = rng.below(data.len());
        match rng.below(8) {
            0 => data[at] ^= 1 << rng.below(8),
            1 => data[at] = INTERESTING[rng.below(INTERESTING.len())],
            2 => {
                let count = 1 + rng.below(16);
                for _ in 0..count {
                    let byte = u8::try_from(rng.next() & 0xFF).unwrap_or(0);
                    data.insert(at.min(data.len()), byte);
                }
            }
            3 => {
                let end = (at + 1 + rng.below(64)).min(data.len());
                data.drain(at..end);
            }
            4 => data.truncate(at),
            5 => {
                let end = (at + 1 + rng.below(256)).min(data.len());
                let copy = data[at..end].to_vec();
                let to = rng.below(data.len());
                data.splice(to..to, copy);
            }
            // An eight-octet size, often huge or unknown.
            6 => {
                let mut size = [0x01, 0, 0, 0, 0, 0, 0, 0];
                for byte in size.iter_mut().skip(1) {
                    *byte = if rng.chance(50) {
                        0xFF
                    } else {
                        u8::try_from(rng.next() & 0xFF).unwrap_or(0)
                    };
                }
                let end = (at + 8).min(data.len());
                data.splice(at..end, size);
            }
            // A one-octet size set to the largest or to the unknown value.
            _ => data[at] = if rng.chance(50) { 0xFE } else { 0xFF },
        }
    }
    data
}

/// What the mutants reached.
#[derive(Clone, Debug, Default)]
struct Reach {
    mutants: u64,
    opened: u64,
    packets: u64,
    seeks: u64,
    slowest: Duration,
    /// The slowest inputs, slowest first: time, seed and mutant.
    slowest_at: Vec<(Duration, u64, usize)>,
}

/// Probes, opens both ways, reads and seeks one input.
fn exercise(data: &[u8], reach: &mut Reach) {
    let started = Instant::now();
    let _ = probe(&data[..data.len().min(64)]);
    let limit = Limits::standard().max_packet_bytes;
    let length = data.len() as u64;
    reach.mutants += 1;
    if let Ok(mut demux) = MatroskaDemux::open(Cursor::new(data.to_vec()), length) {
        reach.opened += 1;
        let tracks: Vec<u32> = demux.streams().iter().map(|s| s.id().get()).collect();
        for _ in 0..5_000 {
            match demux.read_packet() {
                Ok(Some(packet)) => {
                    assert!(packet.bytes().len() as u64 <= limit);
                    reach.packets += 1;
                }
                Ok(None) | Err(_) => break,
            }
        }
        for track in tracks.iter().take(4) {
            for time in [-5, 0, 100, 250, 1_000, 9_999_999] {
                if demux
                    .seek(SeekTo {
                        track: *track,
                        time,
                        end: None,
                    })
                    .is_ok()
                {
                    reach.seeks += 1;
                    for _ in 0..3 {
                        if !matches!(demux.read_packet(), Ok(Some(_))) {
                            break;
                        }
                    }
                }
            }
        }
        let _ = demux.seek(SeekTo {
            track: 99,
            time: 0,
            end: None,
        });
    }
    if let Ok(mut demux) = MatroskaDemux::open_forward(Cursor::new(data.to_vec())) {
        for _ in 0..5_000 {
            if !matches!(demux.read_packet(), Ok(Some(_))) {
                break;
            }
        }
    }
    let took = started.elapsed();
    reach.slowest = reach.slowest.max(took);
}

/// Slowest inputs a campaign names, to replay.
const SLOWEST_KEPT: usize = 8;

/// The `index`th mutant of campaign `seed`.
fn mutant(seed: u64, index: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let seeds = seeds();
    let mut data = Vec::new();
    for _ in 0..=index {
        let base = &seeds[rng.below(seeds.len())];
        data = mutate(&mut rng, base);
    }
    data
}

fn demux_campaign(seed: u64, mutants: usize, reach: &mut Reach) {
    let mut rng = Rng::new(seed);
    let seeds = seeds();
    for index in 0..mutants {
        let base = &seeds[rng.below(seeds.len())];
        let data = mutate(&mut rng, base);
        let before = Instant::now();
        exercise(&data, reach);
        let took = before.elapsed();
        if reach.slowest_at.len() < SLOWEST_KEPT
            || reach.slowest_at.last().is_some_and(|last| took > last.0)
        {
            reach.slowest_at.push((took, seed, index));
            reach.slowest_at.sort_by_key(|a| std::cmp::Reverse(a.0));
            reach.slowest_at.truncate(SLOWEST_KEPT);
        }
        assert!(
            took < Duration::from_secs(5),
            "seed {seed} mutant {index} took {took:?}"
        );
    }
}

#[test]
fn fuzz_seeds_read_back_unmutated() {
    for seed in seeds() {
        let length = seed.len() as u64;
        let mut demux = MatroskaDemux::open(Cursor::new(seed.clone()), length).expect("seed opens");
        let mut count = 0;
        while demux.read_packet().expect("seed reads").is_some() {
            count += 1;
        }
        assert!(count > 0);
    }
}

#[test]
fn fuzz_seeds_demux() {
    let mut reach = Reach::default();
    for seed in 0..12 {
        demux_campaign(seed, 250, &mut reach);
    }
    // The mutants reach the readers, not only the header checks.
    assert!(reach.opened * 4 > reach.mutants, "{reach:?}");
    assert!(reach.packets > reach.mutants, "{reach:?}");
}

/// A random stream setup and packet sequence. Some of it is invalid on
/// purpose: unknown tracks, wrong timescales, negative or huge times.
fn mux_campaign(seed: u64) -> bool {
    let mut rng = Rng::new(seed);
    let webm = rng.chance(30);
    // One sequence in ten breaks the rules on purpose.
    let hostile = rng.chance(10);
    let writer = Cursor::new(Vec::new());
    let mut mux = if webm {
        MatroskaMux::webm(writer)
    } else {
        MatroskaMux::new(writer)
    };
    let codecs = [
        CodecId::Mp3,
        CodecId::Vp9,
        CodecId::Pcm,
        CodecId::Opus,
        CodecId::H264,
        CodecId::Vp8,
        CodecId::Vp9,
        CodecId::Av1,
        CodecId::Prores,
        CodecId::Mjpeg,
        CodecId::Aac,
        CodecId::Opus,
        CodecId::Vorbis,
        CodecId::Flac,
        CodecId::Mp3,
        CodecId::Pcm,
    ];
    let mut added: Vec<(TrackId, Timescale)> = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let codec = if hostile && rng.chance(10) {
            CodecId::Png
        } else {
            codecs[rng.below(codecs.len())]
        };
        let scale = [1000_u32, 48_000, 90_000, 25, 1][rng.below(5)];
        let extra = match codec {
            _ if hostile && rng.chance(30) => Vec::new(),
            CodecId::Opus => {
                let mut head = b"OpusHead".to_vec();
                head.extend_from_slice(&[1, 2, 0x38, 1, 0x80, 0xBB, 0, 0, 0, 0, 0]);
                head
            }
            CodecId::Pcm => vec![1, 0, 0],
            _ => vec![1, 2, 3, 4],
        };
        let params = if matches!(
            codec,
            CodecId::Aac
                | CodecId::Opus
                | CodecId::Vorbis
                | CodecId::Flac
                | CodecId::Mp3
                | CodecId::Pcm
        ) {
            StreamParams::try_audio(StreamCodec::Known(codec), 48_000, 2, extra).expect("audio")
        } else {
            StreamParams::video(
                StreamCodec::Known(codec),
                FrameSize::try_new(64, 48).expect("size"),
                extra,
            )
        };
        match mux.add_stream(NewStream {
            params,
            timescale: Timescale::new(scale),
            language: None,
        }) {
            Ok(id) => added.push((id, Timescale::new(scale))),
            Err(_) => {
                assert_eq!(mux.status(), MuxStatus::Failed);
                return false;
            }
        }
    }
    let mut sent: Vec<Packet> = Vec::new();
    let mut ticks = vec![0_i64; added.len()];
    let mut clock = vec![0_i64; added.len()];
    for _ in 0..rng.below(200) {
        let pick = if hostile && rng.chance(2) {
            added.len()
        } else {
            rng.below(added.len())
        };
        let (track, scale) = added
            .get(pick)
            .copied()
            .unwrap_or((TrackId::new(77), Timescale::new(1000)));
        let scale = if hostile && rng.chance(2) {
            Timescale::new(7)
        } else {
            scale
        };
        let duration = i64::try_from(rng.below(5_000)).unwrap_or(0);
        let index = pick.min(added.len() - 1);
        let pts = if hostile && rng.chance(2) {
            i64::try_from(rng.next() >> 20).unwrap_or(0) - (1 << 40)
        } else {
            clock[index] + i64::try_from(rng.below(10)).unwrap_or(0) - 3
        };
        clock[index] = pts.max(clock[index]) + duration;
        let time = Timestamp::try_new(None, Some(pts), Some(duration)).expect("time");
        let size = rng.below(300);
        let packet = Packet::try_new(track.get(), vec![0xAB; size], time, scale, rng.chance(30))
            .expect("packet");
        if hostile
            && rng.chance(5)
            && let Some((config_track, _)) = added.first()
        {
            let _ = mux.set_config(*config_track, &[1, 2, 3, 4]);
        }
        match mux.write_packet(&packet) {
            Ok(()) => {
                ticks[pick] += duration;
                sent.push(packet);
            }
            Err(_) => {
                assert_eq!(mux.status(), MuxStatus::Failed, "seed {seed}");
                return false;
            }
        }
    }
    let durations: Vec<TrackDuration> = added
        .iter()
        .zip(&ticks)
        .map(|((track, timescale), ticks)| TrackDuration {
            track: *track,
            timescale: *timescale,
            ticks: *ticks,
        })
        .collect();
    if mux.finish(&durations).is_err() {
        assert_eq!(mux.status(), MuxStatus::Failed, "seed {seed}");
        return false;
    }
    let bytes = mux.into_writer().expect("writer").into_inner();
    let length = bytes.len() as u64;
    let mut demux = MatroskaDemux::open(Cursor::new(bytes), length).expect("written file opens");
    let mut read = Vec::new();
    while let Some(packet) = demux.read_packet().expect("written file reads") {
        read.push(packet);
    }
    for (track, _) in &added {
        let theirs: Vec<&[u8]> = sent
            .iter()
            .filter(|p| p.stream() == track.get())
            .map(Packet::bytes)
            .collect();
        let ours: Vec<Vec<u8>> = read
            .iter()
            .filter(|p| p.stream() == track.get())
            .map(|p| p.bytes().to_vec())
            .collect();
        assert_eq!(
            theirs.len(),
            ours.len(),
            "seed {seed} track {}",
            track.get()
        );
        for (a, b) in theirs.iter().zip(&ours) {
            // A ProRes frame comes back with its first atom (size and
            // `icpf`) whether or not it was sent with one.
            let b = if b.get(4..8) == Some(b"icpf") && a.get(4..8) != Some(b"icpf") {
                &b[8..]
            } else {
                b.as_slice()
            };
            assert_eq!(*a, b, "seed {seed}");
        }
    }
    true
}

#[test]
fn fuzz_seeds_mux() {
    let finished = (0..400).filter(|seed| mux_campaign(*seed)).count();
    assert!(finished > 200, "{finished} of 400 finished");
}

fn campaign_origin(default: u64) -> u64 {
    std::env::var("MKV_FUZZ_ORIGIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn campaign_length() -> Duration {
    let secs = std::env::var("MKV_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60);
    Duration::from_secs(secs)
}

#[test]
#[ignore = "long campaign; run with MKV_FUZZ_SECS and --ignored"]
fn fuzz_campaign_demux() {
    let length = campaign_length();
    let started = Instant::now();
    let origin = campaign_origin(1_000);
    let mut seed = origin;
    let mut reach = Reach::default();
    while started.elapsed() < length {
        demux_campaign(seed, 100, &mut reach);
        seed += 1;
    }
    eprintln!(
        "demux campaign: {} mutants over seeds {origin}..{seed} in {:.1} s; {} opened, {} packets read, {} seeks landed; slowest input {:.1} ms; slowest (ms seed:mutant) {}",
        reach.mutants,
        started.elapsed().as_secs_f64(),
        reach.opened,
        reach.packets,
        reach.seeks,
        reach.slowest.as_secs_f64() * 1000.0,
        reach
            .slowest_at
            .iter()
            .map(|(took, seed, index)| format!("{:.1} {seed}:{index}", took.as_secs_f64() * 1000.0))
            .collect::<Vec<_>>()
            .join(", "),
    );
}

/// Times demux mutants ten times each: `MKV_FUZZ_REPLAY=seed:mutant`, or
/// several separated by commas.
#[test]
#[ignore = "replays one input named by MKV_FUZZ_REPLAY"]
fn fuzz_replay() {
    let Ok(list) = std::env::var("MKV_FUZZ_REPLAY") else {
        return;
    };
    for one in list.split(',') {
        let Some((seed, index)) = one
            .trim()
            .split_once(':')
            .and_then(|(seed, index)| Some((seed.parse().ok()?, index.parse().ok()?)))
        else {
            continue;
        };
        replay(seed, index);
    }
}

fn replay(seed: u64, index: usize) {
    let data = mutant(seed, index);
    let mut times = Vec::new();
    for _ in 0..10 {
        let mut reach = Reach::default();
        let before = Instant::now();
        exercise(&data, &mut reach);
        times.push(before.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "seed {seed} mutant {index}: {} bytes, {:.2} ms median, {:.2} ms fastest",
        data.len(),
        times[5],
        times[0]
    );
    if let Ok(path) = std::env::var("MKV_FUZZ_SAVE") {
        std::fs::write(path, &data).expect("save");
    }
}

#[test]
#[ignore = "long campaign; run with MKV_FUZZ_SECS and --ignored"]
fn fuzz_campaign_mux() {
    let length = campaign_length();
    let started = Instant::now();
    let origin = campaign_origin(100_000);
    let mut seed = origin;
    let mut finished = 0;
    while started.elapsed() < length {
        finished += u64::from(mux_campaign(seed));
        seed += 1;
    }
    eprintln!(
        "mux campaign: {} sequences (seeds {origin}..{seed}) in {:.1} s, {finished} finished and read back",
        seed - origin,
        started.elapsed().as_secs_f64()
    );
}
