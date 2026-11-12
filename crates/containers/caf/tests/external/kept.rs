//! Kept chunks in real files: what this crate keeps is what AudioToolbox
//! finds (`afinfo -c <type>` prints the first chunk of a type), and a remux
//! through this crate carries each one so that afinfo finds the same bytes
//! and lists the same markers. Skips, and says so, without afinfo or files.

use std::path::{Path, PathBuf};

use std::fs::File;

use cutlass_caf::{CafMux, KeptChunk};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};

use crate::local::{collect, roots};
use crate::support::{TempDir, open, path, remux, run, tool};

/// The hex afinfo prints for the first chunk of `kind`, if it finds one.
fn afinfo_chunk(file: &Path, kind: [u8; 4]) -> Option<String> {
    let kind = String::from_utf8(kind.to_vec()).expect("ascii");
    let out = run("afinfo", &["-c", &kind, path(file)])?;
    let text = String::from_utf8_lossy(&out).into_owned();
    let tag = format!("chunk '{kind}': ");
    text.lines()
        .find_map(|line| line.strip_prefix(&tag).map(str::to_owned))
}

/// afinfo's marker and region lines, which it reads from `mark`, `regn`
/// and the names in `strg`.
fn afinfo_markers(file: &Path) -> Vec<String> {
    let out = run("afinfo", &[path(file)]).expect("afinfo");
    String::from_utf8_lossy(&out)
        .lines()
        .filter(|line| !line.starts_with("File:"))
        .filter(|line| line.contains("marker") || line.contains("region"))
        .map(str::to_owned)
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn kept_chunks_match_audiotoolbox_and_survive_a_remux() {
    if !tool("afinfo") {
        eprintln!("skipped: afinfo is not on PATH (it comes with macOS)");
        return;
    }
    let mut files = Vec::new();
    for root in roots() {
        collect(&root, &mut files);
    }
    files.push(PathBuf::from(
        "/System/Library/PrivateFrameworks/ScreenReader.framework/Versions/A/Resources/Sounds/Indentation.caf",
    ));
    files.sort();
    files.dedup();
    let dir = TempDir::new("kept");
    let (mut checked, mut chunks, mut markers) = (0_u32, 0_u32, 0_u32);
    let mut common = 0_u32;
    for file in &files {
        let Ok(head) = std::fs::read(file).map(|bytes| bytes.get(..4).map(<[u8]>::to_vec)) else {
            continue;
        };
        if head.as_deref() != Some(b"caff") {
            continue;
        }
        let kept = open(file).kept_chunks().to_vec();
        if kept.is_empty() {
            continue;
        }
        // Every file with a marker, region or string chunk; one in ten of
        // those with only uuid and midi chunks (Apple Loops), to keep the
        // run short.
        let rare = kept
            .iter()
            .any(|chunk| matches!(&chunk.kind(), b"mark" | b"regn" | b"strg"));
        if !rare {
            common += 1;
            if common % 10 != 1 {
                continue;
            }
        }
        let name = file.display();
        let out = dir.join(&format!("kept-{checked}.caf"));
        remux(file, &out).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(open(&out).kept_chunks(), kept.as_slice(), "{name}");
        let mut kinds: Vec<[u8; 4]> = kept.iter().map(|chunk| chunk.kind()).collect();
        kinds.dedup();
        for kind in kinds {
            let first = kept
                .iter()
                .find(|chunk| chunk.kind() == kind)
                .expect("kind");
            let theirs = afinfo_chunk(file, kind);
            assert_eq!(
                theirs.as_deref(),
                Some(hex(first.body()).as_str()),
                "{name}"
            );
            assert_eq!(afinfo_chunk(&out, kind), theirs, "{name}: remuxed");
            chunks += 1;
        }
        let listed = afinfo_markers(file);
        assert_eq!(afinfo_markers(&out), listed, "{name}: markers");
        markers += u32::try_from(listed.len()).expect("count");
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipped: no CAF file with kept chunks was found");
        return;
    }
    eprintln!(
        "{checked} files with kept chunks remuxed ({common} have only uuid or midi ones, one in ten taken): \
         {chunks} chunk types match afinfo before and after, {markers} marker lines unchanged"
    );
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
        .collect()
}

/// `mark`, `strg` and `regn` as AudioToolbox wrote them (through
/// `AudioFileSetProperty` with a marker list and a region list), written
/// again by this crate's muxer: afinfo lists the same region and markers.
/// No system file has a `regn` chunk, hence this one.
#[test]
fn markers_and_regions_this_crate_writes_are_listed_by_afinfo() {
    if !tool("afinfo") {
        eprintln!("skipped: afinfo is not on PATH (it comes with macOS)");
        return;
    }
    let mark = "000000000000000200000000408f400000000000000000010000000000000000000000000000000040dd4c000000000000000002000000000000000000000000";
    let strg = "0000000500000001000000000000000000000002000000000000000600000005000000000000000d00000003000000000000001200000004000000000000001856657273650043686f727573004c6f6f7000737461727400656e6400";
    let regn = "000000000000000100000005000000010000000200000000409f400000000000000000030000000000000000000000000000000040d388000000000000000004000000000000000000000000";
    let dir = TempDir::new("kept-regions");
    let file = dir.join("regions.caf");
    let rate = Timescale::new(44_100);
    let mut mux = CafMux::new(File::create(&file).expect("create"));
    for (kind, body) in [(*b"mark", mark), (*b"strg", strg), (*b"regn", regn)] {
        mux.add_chunk(KeptChunk::new(kind, unhex(body)).expect("kept"))
            .expect("add chunk");
    }
    mux.add_stream(NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 2, vec![1, 1, 0])
            .expect("params"),
        timescale: rate,
        language: None,
    })
    .expect("add");
    let frames = 30_001;
    let time = Timestamp::try_new(Some(0), Some(0), Some(frames)).expect("time");
    let bytes = vec![0; usize::try_from(frames).expect("frames") * 4];
    mux.write_packet(&Packet::try_new(0, bytes, time, rate, true).expect("packet"))
        .expect("write");
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: rate,
        ticks: frames,
    }])
    .expect("finish");
    let listed: Vec<String> = afinfo_markers(&file)
        .iter()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    assert_eq!(
        listed,
        [
            "1 regions; SMPTE time type 0 (none)",
            "region 5, \"Loop\", flags 00000001",
            "marker 3, \"start\": frame 2000 SMPTE time 0:00:00:00/0 type 0 (....) channel 0",
            "marker 4, \"end\": frame 20000 SMPTE time 0:00:00:00/0 type 0 (....) channel 0",
            "2 markers; SMPTE time type 0 (none)",
            "marker 1, \"Verse\": frame 1000 SMPTE time 0:00:00:00/0 type 0 (....) channel 0",
            "marker 2, \"Chorus\": frame 30000 SMPTE time 0:00:00:00/0 type 0 (....) channel 0",
        ]
    );
}
