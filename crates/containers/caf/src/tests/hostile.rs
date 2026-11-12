//! Hostile and broken input: every case returns an error or a bounded
//! result, never a panic and never an allocation the file asked for.

use std::io::{Cursor, Read, Seek, SeekFrom};

use cutlass_container::{Demux, Limits, OpenError, ReadError, SeekTo};

use super::build::{
    aac_cookie, chunk, data_body, desc_body, file, open, open_forward, pakt_body, payload_of,
    pcm_file, sized_chunk, varint,
};
use crate::CafDemux;

fn aac_desc() -> Vec<u8> {
    chunk(b"desc", &desc_body(48_000.0, b"aac ", 0, 0, 1024, 2, 0))
}

fn refused(bytes: &[u8]) -> OpenError {
    open(bytes).err().expect("refused")
}

/// Reads every packet and seeks around; errors are fine, panics are not.
fn drain(bytes: &[u8]) {
    for forward in [false, true] {
        let opened = if forward {
            open_forward(bytes)
        } else {
            open(bytes)
        };
        let Ok(mut demux) = opened else { continue };
        let mut count = 0;
        while let Ok(Some(_)) = demux.read_packet() {
            count += 1;
            assert!(count < 1_000_000, "runaway reader");
        }
        for time in [-1, 0, 1, 1023, 1024, 4_000_000, i64::MAX, i64::MIN] {
            let _ = demux.seek(SeekTo {
                track: 0,
                time,
                end: None,
            });
            let _ = demux.read_packet();
        }
    }
}

#[test]
fn files_that_are_not_caf_version_one_are_refused() {
    assert!(matches!(refused(b""), OpenError::Truncated { .. }));
    assert!(matches!(refused(b"caf"), OpenError::Truncated { .. }));
    assert!(matches!(
        refused(b"RIFF\0\0\0\0WAVE"),
        OpenError::Unrecognized { .. }
    ));
    let mut two = file(&[aac_desc()]);
    two[5] = 2;
    assert!(matches!(refused(&two), OpenError::UnsupportedBrand { .. }));
    let first = file(&[chunk(b"free", &[0; 4]), aac_desc()]);
    assert!(refused(&first).to_string().contains("desc"));
    let short = file(&[chunk(b"desc", &[0; 31])]);
    assert!(open(&short).is_err());
}

#[test]
fn a_broken_description_is_refused() {
    for (rate, bpp, fpp, channels, bits) in [
        (f64::NAN, 4, 1, 2, 16),
        (f64::INFINITY, 4, 1, 2, 16),
        (0.0, 4, 1, 2, 16),
        (-48_000.0, 4, 1, 2, 16),
        (44_100.5, 4, 1, 2, 16),
        (1e12, 4, 1, 2, 16),
        (48_000.0, 4, 1, 0, 16),
        (48_000.0, 4, 1, 65, 16),
        (48_000.0, 4, 1, u32::MAX, 16),
        (48_000.0, 4, u32::MAX, 2, 16),
        (48_000.0, u32::MAX, 1, 2, 16),
        (48_000.0, 4, 1, 2, 65),
    ] {
        let bytes = file(&[
            chunk(
                b"desc",
                &desc_body(rate, b"lpcm", 2, bpp, fpp, channels, bits),
            ),
            chunk(b"data", &data_body(&[0; 64])),
        ]);
        assert!(
            open(&bytes).is_err(),
            "{rate} {bpp} {fpp} {channels} {bits}"
        );
    }
}

#[test]
fn pcm_that_does_not_add_up_is_unknown_not_misread() {
    // 2 channels of 16 bits in 5 bytes, and 3 frames per packet: framed by
    // the description, but not handed to a PCM decoder.
    for (bpp, fpp) in [(5, 1), (12, 3)] {
        let bytes = file(&[
            chunk(b"desc", &desc_body(48_000.0, b"lpcm", 2, bpp, fpp, 2, 16)),
            chunk(b"data", &data_body(&[0; 60])),
        ]);
        let demux = open(&bytes).expect("open");
        assert!(
            demux.streams()[0].params().codec().is_unknown(),
            "{bpp} {fpp}"
        );
        drain(&bytes);
    }
}

#[test]
fn variable_packets_without_a_table_are_refused() {
    let bytes = file(&[
        aac_desc(),
        chunk(b"kuki", &aac_cookie()),
        chunk(b"data", &data_body(&[0; 100])),
    ]);
    assert!(refused(&bytes).to_string().contains("pakt"));
}

#[test]
fn chunk_sizes_that_overrun_or_go_negative_are_refused_before_data() {
    for size in [-2, -1, i64::MAX, 1 << 40] {
        let bytes = file(&[aac_desc(), sized_chunk(b"kuki", size, &aac_cookie())]);
        assert!(open(&bytes).is_err(), "kuki {size}");
        let bytes = file(&[aac_desc(), sized_chunk(b"free", size, &[0; 8])]);
        assert!(open(&bytes).is_err(), "free {size}");
        drain(&bytes);
    }
    let huge = file(&[
        chunk(b"desc", &desc_body(48_000.0, b"lpcm", 2, 4, 1, 2, 16)),
        sized_chunk(b"data", 1 << 50, &data_body(&[0; 16])),
    ]);
    assert!(matches!(refused(&huge), OpenError::HostileSize { .. }));
}

#[test]
fn a_cookie_larger_than_the_limit_is_not_read() {
    let mut limits = Limits::standard();
    limits.max_extradata_bytes = 16;
    let bytes = file(&[
        aac_desc(),
        chunk(b"kuki", &[0; 17]),
        chunk(b"data", &data_body(&[])),
    ]);
    let length = u64::try_from(bytes.len()).expect("len");
    let error = CafDemux::open_limited(Cursor::new(bytes), length, limits)
        .err()
        .expect("refused");
    assert!(matches!(error, OpenError::HostileSize { .. }));
}

#[test]
fn a_cookie_that_is_not_an_es_descriptor_is_refused() {
    for cookie in [
        vec![],
        vec![0x03],
        vec![0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F],
        vec![0x04, 0x05, 0, 0, 0, 0, 0],
        vec![0x03, 0x05, 0, 0, 0, 0x04, 0x7F],
    ] {
        let bytes = file(&[
            aac_desc(),
            chunk(b"kuki", &cookie),
            chunk(b"pakt", &pakt_body(0, 0, 0, 0, &[])),
            chunk(b"data", &data_body(&[])),
        ]);
        assert!(open(&bytes).is_err(), "{cookie:02x?}");
    }
}

fn aac_with_pakt(pakt: Vec<u8>, payload: &[u8]) -> Vec<u8> {
    file(&[
        aac_desc(),
        chunk(b"kuki", &aac_cookie()),
        chunk(b"pakt", &pakt),
        chunk(b"data", &data_body(payload)),
    ])
}

#[test]
fn a_packet_count_that_lies_is_refused_without_allocating_it() {
    // 2^40 packets in a table of three bytes.
    let bytes = aac_with_pakt(pakt_body(1 << 40, 0, 0, 0, &[1, 1, 1]), &[0; 3]);
    assert!(matches!(refused(&bytes), OpenError::HostileSize { .. }));
    // More packets than entries.
    let bytes = aac_with_pakt(pakt_body(4, 4096, 0, 0, &[1, 1, 1]), &[0; 3]);
    assert!(open(&bytes).is_err());
    // Negative counts and frame totals.
    for (packets, valid, priming, remainder) in [
        (-1, 0, 0, 0),
        (1, -1, 0, 0),
        (1, 1024, -1, 0),
        (1, 1024, 0, -1),
    ] {
        let bytes = aac_with_pakt(pakt_body(packets, valid, priming, remainder, &[1]), &[0]);
        assert!(
            open(&bytes).is_err(),
            "{packets} {valid} {priming} {remainder}"
        );
    }
    // A table over the entry limit.
    let mut limits = Limits::standard();
    limits.max_index_entries = 10;
    let sizes = [1_u32; 11];
    let bytes = aac_with_pakt(pakt_body(11, 0, 0, 0, &sizes), &payload_of(&sizes));
    let length = u64::try_from(bytes.len()).expect("len");
    assert!(CafDemux::open_limited(Cursor::new(bytes), length, limits).is_err());
}

#[test]
fn varints_that_are_too_long_or_too_large_are_refused() {
    let mut table = pakt_body(1, 1024, 0, 0, &[]);
    table.extend([0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    assert!(open(&aac_with_pakt(table, &[0; 4])).is_err());
    let mut table = pakt_body(1, 1024, 0, 0, &[]);
    table.extend([0x9F, 0xFF, 0xFF, 0xFF, 0x7F]);
    assert!(open(&aac_with_pakt(table, &[0; 4])).is_err());
    // A varint that ends with the chunk.
    let mut table = pakt_body(1, 1024, 0, 0, &[]);
    table.extend([0x81]);
    assert!(open(&aac_with_pakt(table, &[0; 4])).is_err());
}

#[test]
fn packet_sizes_beyond_the_data_or_the_packet_limit_are_refused() {
    let bytes = aac_with_pakt(pakt_body(2, 2048, 0, 0, &[10, 10]), &[0; 15]);
    assert!(matches!(refused(&bytes), OpenError::HostileSize { .. }));
    let mut big = Vec::new();
    varint(20 * 1024 * 1024, &mut big);
    let mut table = pakt_body(1, 1024, 0, 0, &[]);
    table.extend(big);
    assert!(matches!(
        refused(&aac_with_pakt(table, &[0; 4])),
        OpenError::HostileSize { .. }
    ));
}

#[test]
fn entries_of_zero_frames_are_read_and_empty_ones_skipped() {
    // ffmpeg ends a FLAC table with an entry of no bytes and no frames.
    let bytes = file(&[
        chunk(b"desc", &desc_body(44_100.0, b"flac", 0, 0, 0, 2, 0)),
        chunk(
            b"pakt",
            &pakt_body(5, 20, 0, 0, &[1, 0, 2, 10, 0, 0, 3, 10, 0, 0]),
        ),
        chunk(b"data", &data_body(&payload_of(&[1, 2, 0, 3]))),
    ]);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.packet_count(), Some(5));
    let mut seen = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        seen.push((
            packet.bytes().len(),
            packet.timestamp().presentation().expect("pts"),
            packet.timestamp().duration().expect("duration"),
        ));
    }
    assert_eq!(seen, [(1, 0, 0), (2, 0, 10), (3, 10, 10)]);
    for (time, ordinal) in [(0, 1), (5, 1), (10, 3), (19, 3)] {
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time,
                end: None,
            })
            .expect("seek");
        assert!(
            matches!(landing.landing, cutlass_container::Landing::Packet { ordinal: got, .. } if got == ordinal),
            "{time}: {landing:?}"
        );
    }
    drain(&bytes);
}

#[test]
fn priming_and_remainder_beyond_the_audio_do_not_break_the_edit() {
    let sizes = [5_u32, 5];
    let bytes = aac_with_pakt(
        pakt_body(2, 1 << 40, i32::MAX, i32::MAX, &sizes),
        &payload_of(&sizes),
    );
    if let Ok(demux) = open(&bytes) {
        for edit in demux.streams()[0].edits() {
            assert!(edit.duration() <= 2048);
        }
    }
    drain(&bytes);
}

#[test]
fn duplicate_chunks_are_refused() {
    let (pcm, _) = pcm_file(4);
    let data_at = pcm
        .windows(4)
        .position(|window| window == b"data")
        .expect("data");
    for twin in [
        chunk(b"desc", &desc_body(48_000.0, b"lpcm", 2, 4, 1, 2, 16)),
        chunk(b"chan", &[0, 0x65, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        let mut bytes = pcm[..data_at].to_vec();
        bytes.extend(&twin);
        bytes.extend(&twin);
        bytes.extend(&pcm[data_at..]);
        assert!(open(&bytes).is_err());
    }
    let mut bytes = pcm.clone();
    bytes.extend(chunk(b"data", &data_body(&[0; 4])));
    assert!(open(&bytes).is_err());
}

#[test]
fn a_channel_layout_that_lies_is_refused() {
    let mut chan = vec![0, 0, 0, 0, 0, 0, 0, 0];
    chan.extend(1000_u32.to_be_bytes());
    let (pcm, _) = pcm_file(4);
    let data_at = pcm
        .windows(4)
        .position(|window| window == b"data")
        .expect("data");
    let mut bytes = pcm[..data_at].to_vec();
    bytes.extend(chunk(b"chan", &chan));
    bytes.extend(&pcm[data_at..]);
    assert!(open(&bytes).is_err());
}

#[test]
fn too_many_chunks_are_refused() {
    let mut chunks = vec![chunk(
        b"desc",
        &desc_body(48_000.0, b"lpcm", 2, 4, 1, 2, 16),
    )];
    for _ in 0..crate::MAX_CHUNKS {
        chunks.push(chunk(b"free", &[]));
    }
    chunks.push(chunk(b"data", &data_body(&[0; 4])));
    assert!(open(&file(&chunks)).is_err());
}

#[test]
fn a_missing_data_chunk_is_refused() {
    let bytes = file(&[chunk(
        b"desc",
        &desc_body(48_000.0, b"lpcm", 2, 4, 1, 2, 16),
    )]);
    assert!(open(&bytes).is_err());
    assert!(open_forward(&bytes).is_err());
}

fn sample_files() -> Vec<Vec<u8>> {
    let sizes: Vec<u32> = (0..70).map(|i| 3 + i % 9).collect();
    let mut flac_table = Vec::new();
    for size in &sizes {
        flac_table.push(*size);
        flac_table.push(1152 + size);
    }
    let mut info = 1_u32.to_be_bytes().to_vec();
    info.extend(b"k\0v\0");
    vec![
        pcm_file(40).0,
        aac_with_pakt(
            pakt_body(70, 70 * 1024 - 100, 50, 50, &sizes),
            &payload_of(&sizes),
        ),
        file(&[
            chunk(b"desc", &desc_body(44_100.0, b"flac", 0, 0, 0, 2, 0)),
            chunk(b"info", &info),
            chunk(b"data", &data_body(&payload_of(&sizes))),
            chunk(b"pakt", &pakt_body(70, 0, 0, 0, &flac_table)),
        ]),
        file(&[
            chunk(b"desc", &desc_body(8_000.0, b"ulaw", 0, 1, 1, 1, 8)),
            sized_chunk(b"data", -1, &data_body(&[0x7F; 50])),
        ]),
    ]
}

#[test]
fn every_prefix_of_a_good_file_is_safe() {
    for whole in sample_files() {
        for end in 0..whole.len() {
            drain(&whole[..end]);
        }
    }
}

#[test]
fn single_byte_changes_are_safe() {
    // Deterministic: every byte of the headers takes a handful of values.
    for whole in sample_files() {
        let header = whole.len().min(400);
        for at in 0..header {
            for value in [0x00, 0x01, 0x7F, 0x80, 0xFF, whole[at] ^ 0x10] {
                let mut bytes = whole.clone();
                bytes[at] = value;
                drain(&bytes);
            }
        }
    }
}

/// A reader that fails after a number of bytes.
struct Failing {
    inner: Cursor<Vec<u8>>,
    left: usize,
}

impl Read for Failing {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.left == 0 {
            return Err(std::io::Error::other("unplugged"));
        }
        let take = out.len().min(self.left);
        let read = self.inner.read(&mut out[..take])?;
        self.left -= read;
        Ok(read)
    }
}

impl Seek for Failing {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(to)
    }
}

#[test]
fn a_failing_reader_gives_an_error() {
    let whole = sample_files().swap_remove(1);
    let length = u64::try_from(whole.len()).expect("len");
    for left in 0..whole.len() {
        let reader = Failing {
            inner: Cursor::new(whole.clone()),
            left,
        };
        match CafDemux::open(reader, length) {
            Err(_) => {}
            Ok(mut demux) => loop {
                match demux.read_packet() {
                    Ok(Some(_)) => {}
                    Ok(None) => panic!("{left}: ended cleanly with the source failing"),
                    Err(error) => {
                        assert!(!matches!(error, ReadError::Cancelled));
                        break;
                    }
                }
            },
        }
    }
}
