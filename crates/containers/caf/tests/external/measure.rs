//! Seek cost on long packet tables, and the memory a long table takes.
//!
//! The memory half is a run, not an assertion: under `/usr/bin/time -l`,
//! `CAF_MEASURE_PACKETS=16000000 cargo test --release -p cutlass-caf --test external long_table_open -- --ignored --nocapture`
//! writes a file whose table has that many size and frame pairs, streams
//! it to disk, then opens it and reports the open time.

use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::time::Instant;

use cutlass_caf::CafDemux;
use cutlass_container::{Demux, Landing, SeekTo};

use crate::fuzz::Rng;
use crate::support::{TempDir, path};

fn varint(value: u32, out: &mut Vec<u8>) {
    let mut groups = vec![value & 0x7F];
    let mut rest = value >> 7;
    while rest != 0 {
        groups.push(rest & 0x7F);
        rest >>= 7;
    }
    for (index, group) in groups.iter().enumerate().rev() {
        let more = if index == 0 { 0 } else { 0x80 };
        out.push(u8::try_from(*group).expect("group") | more);
    }
}

fn header(rate: f64, format: &[u8; 4], frames_per_packet: u32) -> Vec<u8> {
    let mut out = b"caff\x00\x01\x00\x00desc".to_vec();
    out.extend(32_i64.to_be_bytes());
    out.extend(rate.to_bits().to_be_bytes());
    out.extend(format);
    for field in [0_u32, 0, frames_per_packet, 2, 0] {
        out.extend(field.to_be_bytes());
    }
    out
}

fn size_of(index: u64) -> u32 {
    1 + u32::try_from(index % 3).expect("small")
}

/// Ten hours of 48 kHz AAC: 1,687,500 packets of one to three bytes, the
/// table before the data.
fn ten_hours() -> (Vec<u8>, u64) {
    let packets = 1_687_500_u64;
    let mut table = Vec::new();
    let mut payload_len = 0_u64;
    for index in 0..packets {
        varint(size_of(index), &mut table);
        payload_len += u64::from(size_of(index));
    }
    let mut out = header(48_000.0, b"aac ", 1024);
    // An ES_Descriptor with one-byte lengths around AudioSpecificConfig
    // 0x1190 (AAC-LC, 48 kHz, stereo).
    let cookie: Vec<u8> = (0..27)
        .map(|at| {
            u8::from_str_radix(
                &"031900000004114015000000000000000000000005021190060102"[at * 2..at * 2 + 2],
                16,
            )
            .expect("hex")
        })
        .collect();
    out.extend(b"kuki");
    out.extend(i64::try_from(cookie.len()).expect("len").to_be_bytes());
    out.extend(&cookie);
    out.extend(b"pakt");
    out.extend(i64::try_from(24 + table.len()).expect("len").to_be_bytes());
    out.extend(i64::try_from(packets).expect("packets").to_be_bytes());
    out.extend(i64::try_from(packets * 1024).expect("valid").to_be_bytes());
    out.extend([0; 8]);
    out.extend(table);
    out.extend(b"data");
    out.extend(i64::try_from(4 + payload_len).expect("len").to_be_bytes());
    out.extend(1_u32.to_be_bytes());
    for index in 0..packets {
        let size = usize::try_from(size_of(index)).expect("size");
        out.extend(std::iter::repeat_n(
            u8::try_from(index % 251).expect("byte"),
            size,
        ));
    }
    (out, packets)
}

#[test]
fn a_seek_in_ten_hours_of_aac_reads_no_payload_and_visits_few_entries() {
    let (bytes, packets) = ten_hours();
    let len = u64::try_from(bytes.len()).expect("len");
    let started = Instant::now();
    let mut demux = CafDemux::open(Cursor::new(bytes), len).expect("open");
    let opened = started.elapsed();
    assert_eq!(demux.packet_count(), Some(packets));
    let mut rng = Rng(0x5EE4);
    let mut widest = 0;
    let started = Instant::now();
    for _ in 0..1_000 {
        let target = i64::try_from(rng.below(usize::try_from(packets * 1024).expect("frames")))
            .expect("time");
        let before = demux.consumed();
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        assert_eq!(demux.consumed(), before, "seek read payload");
        widest = widest.max(landing.entries_visited);
        let Landing::Packet { ordinal, time, .. } = landing.landing else {
            panic!("end");
        };
        assert_eq!(i64::from(ordinal) * 1024, time);
        assert!(time <= target && target < time + 1024);
        let packet = demux.read_packet().expect("read").expect("packet");
        let index = u64::from(ordinal);
        assert_eq!(
            packet.bytes().len(),
            usize::try_from(size_of(index)).expect("size")
        );
        assert_eq!(packet.bytes()[0], u8::try_from(index % 251).expect("byte"));
    }
    let seeks = started.elapsed();
    assert!(widest < 64, "a seek visited {widest} entries");
    eprintln!(
        "ten hours of AAC ({packets} packets): open {opened:?}, 1000 seeks with a read {seeks:?}, \
         at most {widest} table entries visited a seek"
    );
}

#[test]
#[ignore = "writes a large file; set CAF_MEASURE_PACKETS, default 16000000"]
fn long_table_open() {
    let packets: u64 = std::env::var("CAF_MEASURE_PACKETS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16_000_000);
    let dir = TempDir::new("measure");
    let file = dir.join("long.caf");
    {
        let mut out = BufWriter::new(File::create(&file).expect("create"));
        let mut table_len = 0_u64;
        let mut scratch = Vec::new();
        for index in 0..packets {
            scratch.clear();
            varint(size_of(index), &mut scratch);
            varint(
                1000 + u32::try_from(index % 200).expect("frames"),
                &mut scratch,
            );
            table_len += u64::try_from(scratch.len()).expect("len");
        }
        out.write_all(&header(48_000.0, b"zzzz", 0)).expect("write");
        out.write_all(b"pakt").expect("write");
        out.write_all(&i64::try_from(24 + table_len).expect("len").to_be_bytes())
            .expect("write");
        out.write_all(&i64::try_from(packets).expect("packets").to_be_bytes())
            .expect("write");
        out.write_all(&[0; 16]).expect("write");
        let mut payload = 0_u64;
        for index in 0..packets {
            scratch.clear();
            varint(size_of(index), &mut scratch);
            varint(
                1000 + u32::try_from(index % 200).expect("frames"),
                &mut scratch,
            );
            out.write_all(&scratch).expect("write");
            payload += u64::from(size_of(index));
        }
        out.write_all(b"data").expect("write");
        out.write_all(&i64::try_from(4 + payload).expect("len").to_be_bytes())
            .expect("write");
        out.write_all(&1_u32.to_be_bytes()).expect("write");
        let block = vec![7_u8; 1 << 20];
        let mut left = payload;
        while left > 0 {
            let take = left.min(1 << 20);
            out.write_all(&block[..usize::try_from(take).expect("take")])
                .expect("write");
            left -= take;
        }
    }
    let size = std::fs::metadata(&file).expect("meta").len();
    let handle = File::open(&file).expect("open");
    let started = Instant::now();
    let mut demux =
        CafDemux::open(handle, size).unwrap_or_else(|error| panic!("{}: {error}", path(&file)));
    let opened = started.elapsed();
    assert_eq!(demux.packet_count(), Some(packets));
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: i64::try_from(demux.total_frames().expect("frames") / 2).expect("time"),
            end: None,
        })
        .expect("seek");
    eprintln!(
        "{packets} size and frame pairs, file {size} bytes: open {opened:?}, a middle seek visited {} entries",
        landing.entries_visited
    );
}
