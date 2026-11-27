//! The stateless front end against the software decoder, with an
//! accelerator that records what it is handed: on the fixture stream and
//! on every x265 configuration the oracle and malformed tests encode, the
//! same pictures come out in the same order with the same timing, and each
//! reference picture list names the pictures reconstruction reads. The
//! x265 streams are skipped with a message when ffmpeg or its libx265
//! encoder is missing.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder};

use super::{Accelerator, Picture, RpsSet, StatelessDecoder};
use crate::HevcDecoder;
use crate::nal;
use crate::ps::{ScalingList, Sps};
use crate::slice::{self, SliceHeader};

/// A decoded picture: its order count and a number no other surface has.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Surface {
    poc: i32,
    id: u64,
}

/// A picture as decoded: its order count and, for each slice segment, the
/// order counts of its two reference picture lists.
type Trace = (i32, Vec<[Vec<i32>; 2]>);

#[derive(Default)]
struct Recorder {
    next_id: u64,
    sequences: usize,
    /// Surfaces made for missing references.
    missing: Vec<Surface>,
    trace: Vec<Trace>,
    /// Refuse every sequence.
    refuse: bool,
}

impl Recorder {
    fn surface(&mut self, poc: i32) -> Surface {
        self.next_id += 1;
        Surface {
            poc,
            id: self.next_id,
        }
    }
}

impl Accelerator for Recorder {
    type Surface = Surface;

    fn sequence(&mut self, _sps: &Sps) -> Result<(), CodecError> {
        if self.refuse {
            return Err(CodecError::Unsupported { what: "test" });
        }
        self.sequences += 1;
        Ok(())
    }

    fn missing(&mut self, _sps: &Sps, poc: i32) -> Result<Surface, CodecError> {
        let surface = self.surface(poc);
        self.missing.push(surface.clone());
        Ok(surface)
    }

    fn decode(&mut self, picture: &Picture<'_, Surface>) -> Result<Surface, CodecError> {
        check_picture(picture);
        let lists = picture
            .slices
            .iter()
            .map(|s| {
                s.ref_lists
                    .each_ref()
                    .map(|list| list.iter().map(|&i| picture.refs[i].poc).collect())
            })
            .collect();
        self.trace.push((picture.poc, lists));
        Ok(self.surface(picture.poc))
    }
}

/// Emulation prevention bytes in `coded` (clause 7.4.2): a 3 after two
/// zeros, the zeros counted again from the byte after it.
fn emulation_bytes(coded: &[u8]) -> usize {
    let (mut count, mut zeros) = (0, 0);
    for &b in coded {
        if zeros >= 2 && b == 3 {
            count += 1;
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    count
}

/// What a picture handed to the accelerator must hold.
fn check_picture(picture: &Picture<'_, Surface>) {
    let (sps, pps) = (picture.sps, picture.pps);
    for r in picture.refs {
        assert_eq!(r.surface.poc, r.poc, "a reference's surface");
        match r.set {
            RpsSet::StCurrBefore => assert!(r.poc < picture.poc && !r.long_term),
            RpsSet::StCurrAfter => assert!(r.poc > picture.poc && !r.long_term),
            RpsSet::LtCurr => assert!(r.long_term),
            RpsSet::Foll => {}
        }
    }
    let mut ids: Vec<u64> = picture.refs.iter().map(|r| r.surface.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), picture.refs.len(), "a picture referenced twice");
    assert_eq!(picture.column_widths.iter().sum::<u32>(), sps.ctb_width());
    assert_eq!(picture.row_heights.iter().sum::<u32>(), sps.ctb_height());
    assert!(!picture.slices.is_empty());
    let sets = |id: u8| {
        (id == pps.pps_pic_parameter_set_id).then(|| (Arc::new(sps.clone()), Arc::new(pps.clone())))
    };
    let mut independent: Option<SliceHeader> = None;
    for slice in picture.slices {
        let header = slice.header;
        assert_eq!(
            (slice.nal[0] >> 1) & 0x3f,
            picture.nal_unit_type,
            "the NAL unit type"
        );
        // The header parses again from the coded NAL unit to the same
        // values and the same length.
        let unit = nal::parse(slice.nal).unwrap();
        let parsed = slice::parse_header(&unit, &sets, independent.as_ref()).unwrap();
        assert_eq!(&parsed.header, header);
        assert_eq!(slice.header_bits % 8, 0);
        assert_eq!(slice.header_bits as usize, (parsed.data_offset + 2) * 8);
        if header.dependent_slice_segment_flag {
            assert!(independent.is_some());
        } else {
            assert_eq!(slice.st_rps_bits, parsed.st_rps_bits);
            if header.short_term_ref_pic_set_sps_flag {
                assert_eq!(slice.st_rps_bits, 0);
            }
            independent = Some(header.clone());
        }
        // The coded offset is the RBSP one plus the emulation prevention
        // bytes before it, and the slice data starts there: the first nine
        // bits of CABAC data are an `ivlOffset` below 510 (clause 9.3.2.5).
        let rbsp_bytes = slice.header_bits as usize / 8;
        assert!(slice.data_offset < slice.nal.len());
        assert_eq!(
            slice.data_offset - rbsp_bytes,
            emulation_bytes(&slice.nal[2..slice.data_offset])
        );
        let data = &unit.rbsp[rbsp_bytes - 2..];
        if let [a, b, ..] = data {
            let offset = (u32::from(*a) << 1) | u32::from(*b >> 7);
            assert!(offset < 510, "ivlOffset {offset}");
        }
        // Lists of the sizes the header gives, of references the current
        // picture uses.
        for (list, size) in slice.ref_lists.iter().zip(header.num_ref_idx_active()) {
            assert_eq!(list.len(), size);
            for &i in list {
                assert_ne!(picture.refs[i].set, RpsSet::Foll);
            }
        }
    }
}

/// The NAL units of an Annex B stream, without start codes.
fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    nal::annex_b_ranges(data)
        .into_iter()
        .map(|(start, end)| &data[start..end])
        .collect()
}

/// An Annex B stream split into access units (clause 7.4.2.4.4).
fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut units: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut has_slice = false;
    for nal in nal_units(data) {
        if nal.len() < 3 {
            continue;
        }
        let kind = (nal[0] >> 1) & 0x3f;
        let slice = kind <= 21;
        let starts_unit = if slice {
            nal[2] & 0x80 != 0
        } else {
            matches!(kind, 32..=35 | 39 | 41..=44 | 48..=55)
        };
        if starts_unit && has_slice {
            units.push(std::mem::take(&mut current));
            has_slice = false;
        }
        current.extend_from_slice(&[0, 0, 0, 1]);
        current.extend_from_slice(nal);
        has_slice |= slice;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}

/// Annex B to four-byte length prefixes, without the parameter sets when
/// `keep_params` is false.
fn length_prefixed(unit: &[u8], keep_params: bool) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nal_units(unit) {
        if !keep_params && matches!((nal[0] >> 1) & 0x3f, 32..=34) {
            continue;
        }
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

fn packet(unit: &[u8], n: usize) -> Packet {
    Packet::video(unit.to_vec(), 1000 + n as i64, 1, 30, n == 0)
}

fn params(prefix: u8, extra: Vec<u8>) -> CodecParams {
    let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
    params.nal_length_size = prefix;
    params.extra = extra;
    params
}

/// What a decoder made of a run of pushes: the timestamps of the pictures
/// out, in order, and each push's error.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    pts: Vec<i64>,
    errors: Vec<Option<CodecError>>,
}

fn software(units: &[Vec<u8>], params: &CodecParams) -> (Run, Vec<Trace>) {
    let mut dec = HevcDecoder::open_with_threads(params, 1).unwrap();
    let mut run = Run {
        pts: Vec::new(),
        errors: Vec::new(),
    };
    for (n, unit) in units.iter().enumerate() {
        match dec.push(&packet(unit, n)) {
            Ok(out) => {
                run.pts.extend(out.iter().map(|o| o.pts));
                run.errors.push(None);
            }
            Err(e) => run.errors.push(Some(e)),
        }
    }
    run.pts.extend(dec.drain().unwrap().iter().map(|o| o.pts));
    assert!(dec.drain().unwrap().is_empty());
    (run, std::mem::take(&mut dec.ref_trace))
}

fn stateless(units: &[Vec<u8>], params: &CodecParams) -> (Run, Recorder) {
    let mut dec = StatelessDecoder::open(params, Recorder::default()).unwrap();
    let mut run = Run {
        pts: Vec::new(),
        errors: Vec::new(),
    };
    let mut seen = Vec::new();
    for (n, unit) in units.iter().enumerate() {
        match dec.push(&packet(unit, n)) {
            Ok(out) => {
                run.pts.extend(out.iter().map(|o| o.pts));
                seen.extend(out.into_iter().map(|o| o.surface));
                run.errors.push(None);
            }
            Err(e) => run.errors.push(Some(e)),
        }
    }
    let out = dec.drain().unwrap();
    run.pts.extend(out.iter().map(|o| o.pts));
    seen.extend(out.into_iter().map(|o| o.surface));
    assert!(dec.drain().unwrap().is_empty(), "a second drain");
    let StatelessDecoder { accelerator, .. } = dec;
    // Every picture out is one the accelerator decoded, once; none is a
    // stand-in for a missing reference.
    for s in &seen {
        assert!(!accelerator.missing.contains(s), "a missing picture out");
    }
    let mut ids: Vec<u64> = seen.iter().map(|s| s.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), seen.len(), "a picture out twice");
    (run, accelerator)
}

/// Both decoders on `units`: the same pictures out in the same order and
/// the same reference picture lists. Returns what the stateless decoder
/// made of the pushes and the recorder.
fn compare(name: &str, units: &[Vec<u8>], params: &CodecParams) -> (Run, Recorder) {
    let (want, want_trace) = software(units, params);
    let (got, recorder) = stateless(units, params);
    assert_eq!(got, want, "{name}: pictures out");
    assert_eq!(recorder.trace.len(), want_trace.len(), "{name}: pictures");
    for (index, (a, b)) in recorder.trace.iter().zip(&want_trace).enumerate() {
        assert_eq!(a, b, "{name}: picture {index} in decoding order");
    }
    (got, recorder)
}

fn fixture() -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/testsrc-64x48.hevc");
    std::fs::read(path).unwrap()
}

#[test]
fn the_fixture_matches_the_software_decoder() {
    let units = access_units(&fixture());
    assert!(!units.is_empty());
    let (_, recorder) = compare("fixture", &units, &params(0, Vec::new()));
    assert_eq!(recorder.sequences, 1);
    assert_eq!(recorder.trace.len(), units.len());
    let prefixed: Vec<Vec<u8>> = units.iter().map(|u| length_prefixed(u, true)).collect();
    compare("fixture, prefixed", &prefixed, &params(4, Vec::new()));
}

/// An `hvcC` record holding the parameter sets of `unit`, with four-byte
/// lengths.
fn hvcc(unit: &[u8]) -> Vec<u8> {
    let mut record = vec![1u8; 21];
    record.push(0xfc | 3);
    let params: Vec<&[u8]> = nal_units(unit)
        .into_iter()
        .filter(|n| matches!((n[0] >> 1) & 0x3f, 32..=34))
        .collect();
    record.push(params.len() as u8);
    for nal in params {
        record.push((nal[0] >> 1) & 0x3f);
        record.extend_from_slice(&1u16.to_be_bytes());
        record.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        record.extend_from_slice(nal);
    }
    record
}

#[test]
fn hvcc_extradata_carries_the_parameter_sets() {
    let units = access_units(&fixture());
    let samples: Vec<Vec<u8>> = units.iter().map(|u| length_prefixed(u, false)).collect();
    // The record's length size replaces the parameters' 0.
    let params = params(0, hvcc(&units[0]));
    let (_, recorder) = compare("hvcC", &samples, &params);
    assert_eq!(recorder.trace.len(), units.len());
}

#[test]
fn drain_and_reset_keep_their_contracts() {
    let units = access_units(&fixture());
    let params = params(0, Vec::new());
    let mut dec = StatelessDecoder::open(&params, Recorder::default()).unwrap();
    assert!(dec.drain().unwrap().is_empty());
    assert!(matches!(
        dec.push(&Packet::video(Vec::new(), 0, 1, 30, true)),
        Err(CodecError::InvalidApi { .. })
    ));
    let mut first = Vec::new();
    for (n, unit) in units.iter().enumerate() {
        first.extend(dec.push(&packet(unit, n)).unwrap());
    }
    first.extend(dec.drain().unwrap());
    assert!(!first.is_empty());
    assert!(dec.drain().unwrap().is_empty(), "a second drain");
    assert_eq!(dec.accelerator().sequences, 1);

    // After a reset, a keyframe decodes again and the accelerator hears
    // of its sequence again.
    dec.reset();
    let mut again = Vec::new();
    for (n, unit) in units.iter().enumerate() {
        again.extend(dec.push(&packet(unit, n)).unwrap());
    }
    again.extend(dec.drain().unwrap());
    assert_eq!(
        again.iter().map(|o| o.pts).collect::<Vec<_>>(),
        first.iter().map(|o| o.pts).collect::<Vec<_>>()
    );
    assert_eq!(dec.accelerator().sequences, 2);
    let sps = &again[0].sps;
    assert_eq!(sps.pic_width_in_luma_samples, 64);
    assert_eq!(sps.pic_height_in_luma_samples, 48);
    assert!(again[0].keyframe);
    dec.accelerator_mut().trace.clear();
    assert!(dec.accelerator().trace.is_empty());
}

#[test]
fn a_refused_sequence_fails_the_push_and_changes_nothing() {
    let units = access_units(&fixture());
    let params = params(0, Vec::new());
    let mut dec = StatelessDecoder::open(&params, Recorder::default()).unwrap();
    dec.accelerator_mut().refuse = true;
    assert!(matches!(
        dec.push(&packet(&units[0], 0)),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(dec.accelerator().trace.is_empty());
    // The failed push stored nothing: once accepted, the sequence starts
    // with the same keyframe.
    dec.accelerator_mut().refuse = false;
    let mut out = dec.push(&packet(&units[0], 0)).unwrap();
    out.extend(dec.drain().unwrap());
    assert_eq!(out.len(), 1);
    assert_eq!(dec.accelerator().sequences, 1);
}

fn ffmpeg() -> Option<PathBuf> {
    let found = Command::new("ffmpeg").arg("-version").output().ok()?;
    if !found.status.success() {
        return None;
    }
    let encoders = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&encoders.stdout)
        .contains("libx265")
        .then(|| PathBuf::from("ffmpeg"))
}

/// One x265 stream: size, frames, pixel format, parameters, filter.
struct Case {
    name: &'static str,
    size: (usize, usize),
    frames: usize,
    pix: &'static str,
    params: String,
    filter: &'static str,
}

const BASE: &str = "log-level=error:keyint=30:min-keyint=30:scenecut=0:wpp=0";

fn case(name: &'static str, extra: &str) -> Case {
    Case {
        name,
        size: (208, 120),
        frames: 10,
        pix: "yuv420p",
        params: format!("{BASE}:{extra}"),
        filter: "",
    }
}

/// A stream the malformed tests mangle, without the oracle's base
/// parameters.
fn malformed(name: &'static str, pix: &'static str, params: &str) -> Case {
    Case {
        name,
        size: (128, 96),
        frames: 10,
        pix,
        params: format!("log-level=error:{params}"),
        filter: "fade=in:0:6",
    }
}

/// The streams the oracle tests decode, and those the malformed tests
/// mangle.
fn cases() -> Vec<Case> {
    let big = (640, 352);
    vec![
        case("intra", "keyint=1:sao=0:deblock=0"),
        case("intra filters", "keyint=1"),
        case("p refs", "bframes=0:ref=3"),
        case("b pyramid", "bframes=4:b-pyramid=1:ref=4"),
        case("amp", "rect=1:amp=1:bframes=2"),
        case("no tmvp", "temporal-mvp=0:bframes=3"),
        Case {
            filter: "fade=in:0:8",
            ..case("weighted", "weightp=1:weightb=1:bframes=3")
        },
        Case {
            pix: "yuv420p10le",
            filter: "fade=in:0:8",
            ..case("weighted main10", "weightp=1:weightb=1:bframes=3")
        },
        Case {
            frames: 26,
            ..case(
                "open gop",
                "keyint=8:min-keyint=8:open-gop=1:bframes=3:ref=3",
            )
        },
        Case {
            frames: 26,
            ..case(
                "closed gop",
                "keyint=8:min-keyint=8:open-gop=0:bframes=3:ref=3",
            )
        },
        case(
            "many refs",
            "ref=6:bframes=5:b-adapt=2:rect=1:max-merge=5:subme=7:me=star",
        ),
        case("sao off", "sao=0:bframes=2"),
        case("deblock offsets", "deblock=-3,4:bframes=2"),
        case("deblock off", "deblock=0:bframes=2"),
        case("wpp", "wpp=1:bframes=2"),
        case("slices", "wpp=1:slices=3:bframes=2"),
        case("transform skip", "tskip=1:rdoq-level=2:bframes=2"),
        case("scaling lists", "scaling-list=default:bframes=2"),
        case("lossless", "lossless=1:bframes=2"),
        case(
            "constrained intra",
            "constrained-intra=1:signhide=1:bframes=2",
        ),
        case(
            "small ctu",
            "ctu=16:min-cu-size=8:tu-intra-depth=3:tu-inter-depth=3:bframes=2:sao=0",
        ),
        case(
            "small ctu sao",
            "ctu=16:min-cu-size=8:tu-intra-depth=3:tu-inter-depth=3:bframes=2",
        ),
        Case {
            pix: "yuv420p10le",
            ..case("main10", "bframes=3:ref=3")
        },
        Case {
            size: big,
            frames: 12,
            ..case("threads wpp", "wpp=1:ctu=32:bframes=3:ref=3")
        },
        Case {
            size: big,
            frames: 12,
            pix: "yuv420p10le",
            ..case("threads wpp slices", "wpp=1:ctu=32:slices=4:bframes=2")
        },
        Case {
            size: big,
            frames: 8,
            ..case("threads filters", "ctu=32:bframes=2")
        },
        Case {
            pix: "yuv420p12le",
            ..case("main12", "bframes=3:ref=3")
        },
        Case {
            pix: "yuv420p10le",
            ..case("main10 intra", "keyint=1")
        },
        malformed(
            "malformed 0",
            "yuv420p",
            "bframes=2:ref=2:keyint=8:wpp=1:amp=1:weightp=1",
        ),
        malformed(
            "malformed 1",
            "yuv420p10le",
            "bframes=3:ref=3:keyint=4:open-gop=1:slices=2:wpp=1:tskip=1",
        ),
        malformed(
            "malformed 2",
            "yuv420p",
            "lossless=1:ctu=16:scaling-list=default:keyint=8:wpp=0:temporal-mvp=0",
        ),
    ]
}

/// Encode a case with libx265; `None` when ffmpeg fails.
fn encode(ff: &Path, case: &Case) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("cutlass-hevc-stateless-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let tag: String = case
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let path = dir.join(format!("{tag}.hevc"));
    let (w, h) = case.size;
    let source = if case.filter.is_empty() {
        format!("testsrc2=size={w}x{h}:rate=30")
    } else {
        format!("testsrc2=size={w}x{h}:rate=30,{}", case.filter)
    };
    let encoded = Command::new(ff)
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", &source])
        .args(["-frames:v", &case.frames.to_string(), "-pix_fmt", case.pix])
        .args([
            "-c:v",
            "libx265",
            "-x265-params",
            &case.params,
            "-f",
            "hevc",
        ])
        .arg(&path)
        .output()
        .ok()?;
    assert!(
        encoded.status.success(),
        "{}: x265 refused the parameters: {}",
        case.name,
        String::from_utf8_lossy(&encoded.stderr)
    );
    let bytes = std::fs::read(&path).ok();
    let _ = std::fs::remove_file(&path);
    bytes
}

#[test]
fn x265_streams_match_the_software_decoder() {
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: no ffmpeg with libx265");
        return;
    };
    for case in cases() {
        let Some(bytes) = encode(&ff, &case) else {
            eprintln!("skipped: could not run ffmpeg");
            return;
        };
        let units = access_units(&bytes);
        let (run, recorder) = compare(case.name, &units, &params(0, Vec::new()));
        // x265 writes two slices at 128×96 with slice data the software
        // decoder finds short (ffmpeg overreads their headers too), so for
        // that stream only the agreement above counts.
        if case.name != "malformed 1" {
            assert_eq!(run.errors, vec![None; units.len()], "{}", case.name);
            assert_eq!(recorder.trace.len(), case.frames, "{}", case.name);
            assert!(recorder.missing.is_empty(), "{}", case.name);
        }
        if case.name == "b pyramid" {
            let prefixed: Vec<Vec<u8>> = units.iter().map(|u| length_prefixed(u, true)).collect();
            compare("b pyramid, prefixed", &prefixed, &params(4, Vec::new()));
        }
    }
}

/// A reference dropped from the stream is generated for the pictures that
/// name it, the same by both decoders; a push that fails changes nothing.
#[test]
fn missing_references_and_failed_pushes_match_the_software_decoder() {
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: no ffmpeg with libx265");
        return;
    };
    let Some(bytes) = encode(&ff, &case("p refs", "bframes=0:ref=3")) else {
        eprintln!("skipped: could not run ffmpeg");
        return;
    };
    let mut units = access_units(&bytes);
    units.remove(3);
    let (_, recorder) = compare("dropped reference", &units, &params(0, Vec::new()));
    assert!(recorder.missing.iter().any(|s| s.poc == 3));
    assert!(
        recorder
            .trace
            .iter()
            .any(|(_, slices)| slices.iter().any(|lists| lists[0].contains(&3)))
    );

    // Cut the slice of one access unit inside its header: that push fails
    // in both decoders and the next ones go on as if it never came.
    let mut units = access_units(&bytes);
    let unit = &mut units[5];
    let ranges = nal::annex_b_ranges(unit);
    let (start, _) = ranges[ranges.len() - 1];
    unit.truncate(start + 4);
    let (run, recorder) = compare("failed push", &units, &params(0, Vec::new()));
    assert!(run.errors[5].is_some());
    assert_eq!(run.errors.iter().filter(|e| e.is_some()).count(), 1);
    assert_eq!(recorder.trace.len(), units.len() - 1);
}

#[test]
fn both_decoders_refuse_four_two_two() {
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: no ffmpeg with libx265");
        return;
    };
    let Some(bytes) = encode(
        &ff,
        &Case {
            pix: "yuv422p",
            frames: 2,
            ..case("422", "bframes=0")
        },
    ) else {
        eprintln!("skipped: could not run ffmpeg");
        return;
    };
    let units = access_units(&bytes);
    let params = params(0, Vec::new());
    let mut soft = HevcDecoder::open_with_threads(&params, 1).unwrap();
    let mut hard = StatelessDecoder::open(&params, Recorder::default()).unwrap();
    assert!(matches!(
        soft.push(&packet(&units[0], 0)),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        hard.push(&packet(&units[0], 0)),
        Err(CodecError::Unsupported { .. })
    ));
    assert_eq!(hard.accelerator().sequences, 0);
}

/// The scaling lists a picture uses are handed over in coded order: the
/// default lists (x265's `scaling-list=default`) start 16, 16, 16 and end
/// with the largest factor.
#[test]
fn scaling_lists_reach_the_accelerator_in_coded_order() {
    struct Lists(Vec<([[u8; 64]; 6], [u8; 6])>);
    impl Accelerator for Lists {
        type Surface = ();
        fn sequence(&mut self, _sps: &Sps) -> Result<(), CodecError> {
            Ok(())
        }
        fn missing(&mut self, _sps: &Sps, _poc: i32) -> Result<(), CodecError> {
            Ok(())
        }
        fn decode(&mut self, picture: &Picture<'_, ()>) -> Result<(), CodecError> {
            if let Some(lists) = ScalingList::active(picture.sps, picture.pps) {
                self.0
                    .push((lists.scaling_list_16x16(), lists.scaling_list_dc_16x16()));
            }
            Ok(())
        }
    }
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: no ffmpeg with libx265");
        return;
    };
    let Some(bytes) = encode(
        &ff,
        &case("scaling lists", "scaling-list=default:bframes=0"),
    ) else {
        eprintln!("skipped: could not run ffmpeg");
        return;
    };
    let mut dec = StatelessDecoder::open(&params(0, Vec::new()), Lists(Vec::new())).unwrap();
    for (n, unit) in access_units(&bytes).iter().enumerate() {
        dec.push(&packet(unit, n)).unwrap();
    }
    let seen = &dec.accelerator().0;
    assert!(!seen.is_empty());
    for (lists, dc) in seen {
        assert_eq!(dc, &[16; 6]);
        assert_eq!(&lists[0][..3], &[16, 16, 16]);
        assert_eq!(lists[0][63], 115);
        assert_eq!(lists[3][63], 91);
    }
}
