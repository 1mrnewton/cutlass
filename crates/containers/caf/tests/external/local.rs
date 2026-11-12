//! Real CAF files: the system sounds, ringtones, speech and Apple Loops a
//! Mac ships with, and any `.caf` under `../local-files`. Each one opens,
//! reads to the end with times that add up, and seeks without reading the
//! sound; every 25th is also checked against afinfo and ffprobe when they
//! are on PATH. `CUTLASS_CAF_ROOTS` (colon separated) replaces the folders.
//! The test skips, and says so, when no folder has a CAF file.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use cutlass_caf::CafDemux;
use cutlass_container::{Demux, Landing, OpenError, SeekTo};

use crate::support::{afinfo, ffprobe, tool};

const DEFAULT_ROOTS: &[&str] = &[
    "/System/Library/Components/CoreAudio.component",
    "/System/Library/PrivateFrameworks/ToneLibrary.framework",
    "/System/Library/PrivateFrameworks/TelephonyUtilities.framework",
    "/System/Library/PrivateFrameworks/SiriTTSService.framework",
    "/Library/Audio/Apple Loops",
];

fn roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = match std::env::var("CUTLASS_CAF_ROOTS") {
        Ok(list) => list.split(':').map(PathBuf::from).collect(),
        Err(_) => DEFAULT_ROOTS.iter().map(PathBuf::from).collect(),
    };
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files"));
    roots.retain(|root| root.is_dir());
    roots
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect(&path, out);
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("caf"))
        {
            out.push(path);
        }
    }
}

#[derive(Default)]
struct Tally {
    files: u32,
    packets: u64,
    bytes: u64,
    oracle: u32,
    stopped: u32,
    open_ended: u32,
    widest_open: u64,
    misnamed: u32,
    forward: u32,
}

#[test]
fn real_caf_files_read_and_seek_without_reading_the_sound() {
    let mut files = Vec::new();
    for root in roots() {
        collect(&root, &mut files);
    }
    files.sort();
    if files.is_empty() {
        eprintln!("skipped: no .caf files under the system folders or ../local-files");
        return;
    }
    let oracles = tool("afinfo") && tool("ffprobe") && tool("ffmpeg");
    let mut tally = Tally::default();
    for (index, file) in files.iter().enumerate() {
        let mut head = [0_u8; 4];
        let read = File::open(file).and_then(|mut handle| handle.read_exact(&mut head));
        if read.is_err() || &head != b"caff" {
            // Some system files named .caf are MPEG-4 audio: they must be
            // refused as not CAF, not misread.
            if read.is_ok() {
                let handle = File::open(file).expect("open");
                let len = handle.metadata().expect("meta").len();
                let refused = CafDemux::open(handle, len).err();
                assert!(
                    matches!(refused, Some(OpenError::Unrecognized { .. })),
                    "{}",
                    file.display()
                );
            }
            tally.misnamed += 1;
            continue;
        }
        check(file, &mut tally);
        if oracles && index % 25 == 0 {
            tally.oracle += u32::from(compare(file));
        }
    }
    eprintln!(
        "{} real CAF files: {} packets, {} payload bytes, {} with a broken chunk after data, \
         {} with data of unknown size, {} checked against afinfo and ffprobe, open read at most {} bytes; \
         {} files named .caf are not CAF; {} also read forward",
        tally.files,
        tally.packets,
        tally.bytes,
        tally.stopped,
        tally.open_ended,
        tally.oracle,
        tally.widest_open,
        tally.misnamed,
        tally.forward
    );
}

fn open(file: &Path) -> CafDemux<File> {
    let handle = File::open(file).expect("open");
    let len = handle.metadata().expect("meta").len();
    CafDemux::open(handle, len).unwrap_or_else(|error| panic!("{}: {error}", file.display()))
}

fn check(file: &Path, tally: &mut Tally) {
    let name = file.display();
    let mut demux = open(file);
    tally.files += 1;
    tally.stopped += u32::from(demux.stopped_at().is_some());
    tally.open_ended += u32::from(demux.is_open_ended());
    let header = demux.consumed();
    tally.widest_open = tally.widest_open.max(header);
    let mut times = Vec::new();
    let mut expect = 0;
    while let Some(packet) = demux
        .read_packet()
        .unwrap_or_else(|error| panic!("{name}: {error}"))
    {
        let time = packet.timestamp();
        assert_eq!(time.presentation(), Some(expect), "{name}");
        times.push((expect, packet.bytes().to_vec()));
        expect += time.duration().expect("duration");
        tally.packets += 1;
        tally.bytes += u64::try_from(packet.bytes().len()).expect("len");
    }
    assert_eq!(
        u64::try_from(expect).expect("frames"),
        demux.total_frames().expect("frames"),
        "{name}"
    );
    // A forward reader, as for a file still being copied in, gives the same
    // packets wherever the table sits before the data.
    let handle = File::open(file).expect("open");
    match CafDemux::open_forward(handle) {
        Ok(mut forward) => {
            tally.forward += 1;
            for (at, (_, bytes)) in times.iter().enumerate() {
                let packet = forward
                    .read_packet()
                    .unwrap_or_else(|error| panic!("{name}: forward {error}"))
                    .unwrap_or_else(|| panic!("{name}: forward read ended at {at}"));
                assert_eq!(packet.bytes(), &bytes[..], "{name}: forward packet {at}");
            }
            assert!(forward.read_packet().expect("end").is_none(), "{name}");
        }
        Err(error) => {
            let table_after = demux.packet_table().is_some() && !demux.description().is_constant();
            assert!(table_after, "{name}: forward open failed: {error}");
        }
    }
    if times.is_empty() {
        return;
    }
    for pick in [0, times.len() / 2, times.len() - 1] {
        let (start, bytes) = &times[pick];
        let before = demux.consumed();
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: *start,
                end: None,
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(demux.consumed(), before, "{name}: seek read payload");
        assert_eq!(landing.payload_scans, 0);
        let Landing::Packet { time, preroll, .. } = landing.landing else {
            panic!("{name}: seek to {start} landed at the end");
        };
        assert_eq!((time, preroll), (*start, 0), "{name}");
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(
            packet.bytes(),
            &bytes[..],
            "{name}: packet {pick} after seek"
        );
    }
}

/// afinfo's counts and libavformat's packets, as in the oracle tests.
fn compare(file: &Path) -> bool {
    let name = file.display();
    let demux = open(file);
    let (priming, remainder) = demux.packet_table().map_or((0, 0), |header| {
        (
            i64::from(header.priming_frames),
            i64::from(header.remainder_frames),
        )
    });
    let Some(info) = afinfo(file) else {
        return false;
    };
    assert_eq!(Some(info.packets), demux.packet_count(), "{name}");
    if let Some(header) = demux.packet_table() {
        assert_eq!(info.valid, header.valid_frames, "{name}");
        assert_eq!(info.priming, u64::from(header.priming_frames), "{name}");
    }
    let Some(probed) = ffprobe(file) else {
        eprintln!("ffprobe cannot read {name}");
        return false;
    };
    let mut demux = open(file);
    let mut index = 0;
    let mut payload: Vec<u8> = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        let (pts, _, size) = probed.packets[index];
        assert_eq!(
            pts,
            packet.timestamp().presentation().expect("pts") - priming,
            "{name}"
        );
        assert_eq!(size, packet.bytes().len(), "{name}");
        payload.extend(packet.bytes());
        index += 1;
    }
    assert_eq!(index, probed.packets.len(), "{name}");
    assert!(payload == probed.payload, "{name}: payload");
    let _ = remainder;
    true
}
