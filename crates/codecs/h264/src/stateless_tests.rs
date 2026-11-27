//! Tests of [`super::StatelessDecoder`]. A recording [`Accelerator`] stands
//! in for the hardware: its surface is the picture's order count and a
//! number of its own, and it checks each slice it is handed. Every stream
//! is also decoded by [`H264Decoder`], whose output order, times and
//! reference lists the front end has to match.

use std::path::PathBuf;
use std::process::Command;

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};

use super::{Accelerator, Output, Picture, Slice, Sps, StatelessDecoder};
use crate::bits::Bits;
use crate::dec::list_pocs;
use crate::header::{self, Context};
use crate::nal;
use crate::{H264Decoder, H264Encoder};

/// What the recorder hands back for a picture.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Surface {
    /// The order count of the picture, or `i32::MIN` for a missing frame.
    poc: i32,
    /// Unique to each surface made.
    id: u64,
}

#[derive(Debug, Default)]
struct Recorder {
    made: u64,
    /// The coded size of each sequence begun.
    sequences: Vec<(u32, u32)>,
    /// The `frame_num` of each missing frame asked for.
    missing: Vec<u32>,
    /// The order counts of each slice's reference lists.
    lists: Vec<[Vec<i32>; 2]>,
    pictures: usize,
    /// Fail the picture after this many.
    fail_after: Option<usize>,
}

impl Accelerator for Recorder {
    type Surface = Surface;

    fn sequence(&mut self, sps: &Sps) -> Result<(), CodecError> {
        self.sequences.push(sps.coded_size());
        Ok(())
    }

    fn missing(&mut self, _sps: &Sps, frame_num: u32) -> Result<Surface, CodecError> {
        self.made += 1;
        self.missing.push(frame_num);
        Ok(Surface {
            poc: i32::MIN,
            id: self.made,
        })
    }

    fn decode(&mut self, picture: &Picture<'_, Surface>) -> Result<Surface, CodecError> {
        if self.fail_after == Some(self.pictures) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "recorder",
            });
        }
        assert!(!picture.slices.is_empty());
        let poc = picture
            .top_field_order_cnt
            .min(picture.bottom_field_order_cnt);
        for reference in picture.refs {
            assert_eq!(reference.non_existing, reference.surface.poc == i32::MIN);
            if !reference.non_existing {
                let counted = reference
                    .top_field_order_cnt
                    .min(reference.bottom_field_order_cnt);
                assert_eq!(reference.surface.poc, counted);
            }
        }
        for slice in picture.slices {
            check_slice(picture, slice);
            self.lists.push(slice.ref_lists.clone().map(|list| {
                list.iter()
                    .map(|&at| picture.refs[at].surface.poc)
                    .collect()
            }));
        }
        self.made += 1;
        self.pictures += 1;
        Ok(Surface { poc, id: self.made })
    }
}

/// The slice's NAL unit says what its header says, the header read again
/// from the NAL unit ends at `header_bits`, and a CABAC slice's data starts
/// at the next byte after alignment bits that are all 1.
fn check_slice(picture: &Picture<'_, Surface>, slice: &Slice<'_>) {
    let kind = slice.nal[0] & 0x1f;
    assert!(kind == 1 || kind == 5);
    assert_eq!(kind == 5, slice.header.idr_pic_flag);
    assert_eq!(kind == 5, picture.idr);
    assert_eq!((slice.nal[0] >> 5) & 3, slice.header.nal_ref_idc);
    assert_eq!(slice.header.nal_ref_idc, picture.nal_ref_idc);
    assert_eq!(slice.header.frame_num, picture.frame_num);
    let rbsp = nal::rbsp(&slice.nal[1..]).expect("rbsp");
    let mut bits = Bits::new(&rbsp);
    let start = header::read_start(&mut bits).expect("slice start");
    let again = header::read_rest(
        &mut bits,
        start,
        picture.sps,
        picture.pps,
        Context::annex_a(kind == 5, slice.header.nal_ref_idc),
    )
    .expect("slice header");
    assert_eq!(&again, slice.header);
    assert_eq!(8 + bits.pos(), slice.header_bits as usize);
    if picture.pps.entropy_coding_mode_flag {
        let mut align = Bits::at(&rbsp, bits.pos()).expect("slice data");
        while !align.pos().is_multiple_of(8) {
            assert!(align.bit().expect("alignment bit"));
        }
    }
    let active = [
        usize::from(slice.header.num_ref_idx_l0_active_minus1) + 1,
        usize::from(slice.header.num_ref_idx_l1_active_minus1) + 1,
    ];
    let kind = slice.header.slice_kind();
    let used = [!matches!(kind, 2 | 4), kind == 1];
    for list in 0..2 {
        if used[list] {
            assert!(slice.ref_lists[list].len() <= active[list]);
        } else {
            assert!(slice.ref_lists[list].is_empty());
        }
        for &at in &slice.ref_lists[list] {
            assert!(!picture.refs[at].non_existing);
        }
    }
}

/// What the software decoder shows of a stream: each picture's time and
/// key flag, and the order counts of every slice's reference lists.
type Shown = Vec<(i64, i64, bool)>;

fn software(params: &CodecParams, packets: &[Packet]) -> (Shown, Vec<[Vec<i32>; 2]>) {
    list_pocs::start();
    let mut decoder = H264Decoder::open(params).expect("software open");
    let mut shown = Vec::new();
    for packet in packets {
        let pictures = decoder.push(packet).expect("software push");
        shown.extend(pictures.iter().map(|o| (o.pts, o.duration, o.keyframe)));
    }
    let rest = decoder.drain().expect("software drain");
    shown.extend(rest.iter().map(|o| (o.pts, o.duration, o.keyframe)));
    (shown, list_pocs::take())
}

/// Decodes `packets` with both front ends and checks that they agree.
fn same_as_software(name: &str, params: &CodecParams, packets: &[Packet]) -> Recorder {
    let (want, want_lists) = software(params, packets);
    let mut decoder = StatelessDecoder::open(params, Recorder::default()).expect("open");
    let mut got: Vec<Output<Surface>> = Vec::new();
    for (n, packet) in packets.iter().enumerate() {
        let out = decoder
            .push(packet)
            .unwrap_or_else(|error| panic!("{name}: packet {n}: {error}"));
        got.extend(out);
    }
    got.extend(decoder.drain().expect("drain"));
    assert!(
        decoder.drain().expect("second drain").is_empty(),
        "{name}: a second drain returned pictures"
    );
    let shown: Shown = got
        .iter()
        .map(|o| (o.pts, o.duration, o.keyframe))
        .collect();
    assert_eq!(shown, want, "{name}: output order and times");
    let mut ids: Vec<u64> = got.iter().map(|o| o.surface.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), got.len(), "{name}: a surface was output twice");
    let recorder = std::mem::take(decoder.accelerator_mut());
    assert_eq!(recorder.pictures, got.len(), "{name}: pictures decoded");
    assert_eq!(recorder.lists.len(), want_lists.len(), "{name}: slices");
    for (at, (got, want)) in recorder.lists.iter().zip(&want_lists).enumerate() {
        assert_eq!(got, want, "{name}: reference lists of slice {at}");
    }
    assert!(!recorder.sequences.is_empty());
    recorder
}

/// Pictures with motion in every direction, from the crate's encoder:
/// I, P and B pictures whose 4-bit order counts and frame numbers wrap.
fn encoded(
    width: u32,
    height: u32,
    count: usize,
    keyframes: &[usize],
    length_size: u8,
) -> (CodecParams, Vec<Packet>) {
    let mut params = CodecParams::video(CodecId::H264, width, height);
    params.nal_length_size = length_size;
    params.bitrate = 400_000;
    let mut encoder = H264Encoder::open(&params).expect("encoder");
    let (w, h) = (width as usize, height as usize);
    let mut packets = Vec::new();
    for index in 0..count {
        let shift = index * 3;
        let luma = |x: usize, y: usize| {
            let (x, y) = (x + shift, y + shift / 2);
            (((x / 8) ^ (y / 8)) * 37 + x * 2 + y) as u8
        };
        let y: Vec<u8> = (0..w * h).map(|i| luma(i % w, i / w)).collect();
        let u: Vec<u8> = (0..w * h / 4)
            .map(|i| luma(i % (w / 2), i / (w / 2)) / 2 + 60)
            .collect();
        let v: Vec<u8> = (0..w * h / 4)
            .map(|i| 200 - luma(i / (w / 2), i % (w / 2)) / 2)
            .collect();
        let input = VideoInput {
            picture: Yuv420p::try_new(width, height, y, u, v).expect("picture"),
            pts: index as i64,
            duration: 1,
            timescale: 30,
            force_keyframe: index == 0 || keyframes.contains(&index),
        };
        packets.extend(encoder.push(&input).expect("encode"));
    }
    packets.extend(encoder.drain().expect("encoder drain"));
    if length_size != 0 {
        params.extra = encoder.extradata().to_vec();
    }
    (params, packets)
}

#[test]
fn the_crate_encoders_pictures_come_out_as_the_software_decoder_shows_them() {
    let (params, packets) = encoded(64, 48, 40, &[17], 0);
    let recorder = same_as_software("annex b", &params, &packets);
    assert!(recorder.missing.is_empty());
    // The SPS repeats in every packet; a sequence begins once.
    assert_eq!(recorder.sequences, vec![(64, 48)]);
}

#[test]
fn an_avcc_record_and_length_prefixes_decode_the_same() {
    let (params, packets) = encoded(48, 32, 12, &[], 4);
    assert_eq!(params.extra.first(), Some(&1));
    let recorder = same_as_software("avcC", &params, &packets);
    let (_, annex) = encoded(48, 32, 12, &[], 0);
    let mut annex_params = params.clone();
    annex_params.extra.clear();
    annex_params.nal_length_size = 0;
    let from_annex = same_as_software("annex b", &annex_params, &annex);
    assert_eq!(recorder.lists, from_annex.lists);
}

/// Sets bit `at` of the SPS in every packet, and returns the packets
/// written as Annex B.
fn with_sps_bit(packets: &[Packet], at: usize) -> Vec<Packet> {
    packets
        .iter()
        .map(|packet| {
            let units =
                nal::access_unit_with(&packet.data, 0, |raw| Ok(raw.to_vec())).expect("nal units");
            let mut data = Vec::new();
            for unit in units {
                if unit[0] & 0x1f == 7 {
                    let mut rbsp = nal::rbsp(&unit[1..]).expect("rbsp");
                    rbsp[at / 8] |= 0x80 >> (at % 8);
                    data.extend(nal::annex_b_nal(unit[0] >> 5, 7, &rbsp));
                } else {
                    data.extend([0, 0, 0, 1]);
                    data.extend(unit);
                }
            }
            Packet {
                data,
                ..packet.clone()
            }
        })
        .collect()
}

#[test]
fn a_frame_num_gap_asks_for_missing_frames_and_keeps_them_out_of_the_lists() {
    let (params, packets) = encoded(64, 48, 16, &[], 0);
    // The encoder's SPS has `gaps_in_frame_num_value_allowed_flag` at bit 38.
    let allowed = with_sps_bit(&packets, 38);
    let units = nal::access_unit(&allowed[0].data, 0).expect("units");
    let sps = crate::set::parse_sps(&units[0].rbsp).expect("sps");
    assert!(sps.gaps_in_frame_num_value_allowed_flag);
    // Packets run I0, P2, B1, P4, B3, P6, B5 ... Without P4 and B3 the
    // next reference skips a frame number.
    let mut gapped = allowed;
    gapped.drain(3..5);
    let recorder = same_as_software("gap", &params, &gapped);
    assert_eq!(recorder.missing, vec![2]);
}

#[test]
fn drain_reset_and_a_failed_push_keep_their_contracts() {
    let (params, packets) = encoded(48, 32, 20, &[9], 0);
    let mut fresh = StatelessDecoder::open(&params, Recorder::default()).expect("open");
    let mut want = Vec::new();
    for packet in &packets {
        want.extend(fresh.push(packet).expect("push"));
    }
    want.extend(fresh.drain().expect("drain"));
    let want: Vec<(i64, i32)> = want.iter().map(|o| (o.pts, o.surface.poc)).collect();

    // Reset part of the way through, then decode from the first keyframe.
    let mut decoder = StatelessDecoder::open(&params, Recorder::default()).expect("open");
    for packet in &packets[..7] {
        decoder.push(packet).expect("push");
    }
    decoder.reset();
    assert!(decoder.drain().expect("drain after reset").is_empty());
    let mut got = Vec::new();
    for (n, packet) in packets.iter().enumerate() {
        // A packet whose slice header is cut short fails and changes
        // nothing, though its parameter sets were read.
        if n == 5 {
            let mut broken = packet.clone();
            let slice = broken
                .data
                .windows(4)
                .rposition(|bytes| bytes == [0, 0, 0, 1])
                .expect("slice start code");
            broken.data.truncate(slice + 6);
            assert!(decoder.push(&broken).is_err());
        }
        // So does a picture the accelerator fails.
        if n == 11 {
            let pictures = decoder.accelerator().pictures;
            decoder.accelerator_mut().fail_after = Some(pictures);
            assert!(decoder.push(packet).is_err());
            decoder.accelerator_mut().fail_after = None;
        }
        got.extend(decoder.push(packet).expect("push"));
    }
    got.extend(decoder.drain().expect("drain"));
    let got: Vec<(i64, i32)> = got.iter().map(|o| (o.pts, o.surface.poc)).collect();
    assert_eq!(got, want);
    assert!(decoder.drain().expect("second drain").is_empty());
    // After the reset the first picture begins a sequence again.
    assert_eq!(decoder.accelerator().sequences.len(), 2);

    // Drained, decoding goes on with the next packet.
    let mut decoder = StatelessDecoder::open(&params, Recorder::default()).expect("open");
    let mut count = 0;
    for packet in &packets[..9] {
        count += decoder.push(packet).expect("push").len();
    }
    count += decoder.drain().expect("drain").len();
    for packet in &packets[9..] {
        count += decoder.push(packet).expect("push").len();
    }
    count += decoder.drain().expect("drain").len();
    assert_eq!(count, want.len());

    let empty = Packet::video(Vec::new(), 0, 1, 30, true);
    assert!(matches!(
        decoder.push(&empty),
        Err(CodecError::InvalidApi { .. })
    ));
    let other = CodecParams::video(CodecId::Hevc, 16, 16);
    assert!(matches!(
        StatelessDecoder::open(&other, Recorder::default()),
        Err(CodecError::Unsupported { .. })
    ));
}

/// Exp-Golomb and fixed-length bits, ended by `rbsp_trailing_bits()`.
#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
    acc: u8,
    n: u32,
}

impl Writer {
    fn bit(&mut self, one: bool) {
        self.acc = (self.acc << 1) | u8::from(one);
        self.n += 1;
        if self.n == 8 {
            self.bytes.push(self.acc);
            self.acc = 0;
            self.n = 0;
        }
    }

    fn u(&mut self, n: u32, value: u32) {
        for shift in (0..n).rev() {
            self.bit((value >> shift) & 1 == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let width = 31 - code.leading_zeros();
        self.u(width, 0);
        self.u(width + 1, code);
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            2 * value.unsigned_abs() - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(code);
    }

    fn finish(mut self) -> Vec<u8> {
        self.bit(true);
        while self.n != 0 {
            self.bit(false);
        }
        self.bytes
    }
}

/// How a hand-written SPS differs from a 16×16 progressive 8-bit 4:2:0 one.
struct Shape {
    profile: u32,
    chroma: u32,
    depth: u32,
    frame_mbs_only: bool,
    /// An SPS 4×4 Intra Y list of 20s.
    scaling: bool,
}

const PLAIN: Shape = Shape {
    profile: 100,
    chroma: 1,
    depth: 0,
    frame_mbs_only: true,
    scaling: false,
};

fn sps_rbsp(shape: &Shape) -> Vec<u8> {
    let mut bits = Writer::default();
    bits.u(8, shape.profile);
    bits.u(8, 0);
    bits.u(8, 30);
    bits.ue(0);
    if matches!(shape.profile, 100 | 110 | 122 | 244) {
        bits.ue(shape.chroma);
        if shape.chroma == 3 {
            bits.bit(false);
        }
        bits.ue(shape.depth);
        bits.ue(shape.depth);
        bits.bit(false);
        bits.bit(shape.scaling);
        if shape.scaling {
            // List 0 is 20 throughout: the first delta goes from 8 to 20.
            bits.bit(true);
            bits.se(12);
            for _ in 1..16 {
                bits.se(0);
            }
            for _ in 1..8 {
                bits.bit(false);
            }
        }
    }
    bits.ue(0);
    bits.ue(0);
    bits.ue(0);
    bits.ue(1);
    bits.bit(false);
    bits.ue(0);
    bits.ue(0);
    bits.bit(shape.frame_mbs_only);
    if !shape.frame_mbs_only {
        bits.bit(false);
    }
    bits.bit(true);
    bits.bit(false);
    bits.bit(false);
    bits.finish()
}

/// A PPS for SPS 0 with `groups` slice groups and, when `scaling`, a
/// scaling matrix that codes no list.
fn pps_rbsp(groups: u32, scaling: bool) -> Vec<u8> {
    let mut bits = Writer::default();
    bits.ue(0);
    bits.ue(0);
    bits.bit(false);
    bits.bit(false);
    bits.ue(groups - 1);
    if groups > 1 {
        bits.ue(0);
        for _ in 0..groups {
            bits.ue(0);
        }
    }
    bits.ue(0);
    bits.ue(0);
    bits.bit(false);
    bits.u(2, 0);
    bits.se(0);
    bits.se(0);
    bits.se(0);
    bits.bit(true);
    bits.bit(false);
    bits.bit(false);
    if scaling {
        bits.bit(true);
        bits.bit(true);
        for _ in 0..8 {
            bits.bit(false);
        }
        bits.se(0);
    }
    bits.finish()
}

fn unsupported(result: Result<impl Sized, CodecError>) -> bool {
    matches!(result, Err(CodecError::Unsupported { .. }))
}

/// An `avcC` record holding one SPS and one PPS.
fn avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut record = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
    record.extend((sps.len() as u16).to_be_bytes());
    record.extend(sps);
    record.push(1);
    record.extend((pps.len() as u16).to_be_bytes());
    record.extend(pps);
    record
}

#[test]
fn parameter_sets_out_of_scope_are_refused() {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let shapes = [
        ("in scope", PLAIN, true),
        (
            "baseline",
            Shape {
                profile: 66,
                ..PLAIN
            },
            true,
        ),
        (
            "extended",
            Shape {
                profile: 88,
                ..PLAIN
            },
            false,
        ),
        (
            "fields",
            Shape {
                frame_mbs_only: false,
                ..PLAIN
            },
            false,
        ),
        (
            "4:2:2",
            Shape {
                profile: 122,
                chroma: 2,
                ..PLAIN
            },
            false,
        ),
        (
            "10-bit",
            Shape {
                profile: 110,
                depth: 2,
                ..PLAIN
            },
            false,
        ),
    ];
    for (name, shape, accepted) in shapes {
        let sps = sps_rbsp(&shape);
        assert!(crate::set::parse_sps(&sps).is_ok(), "{name}: not an SPS");
        let mut decoder = StatelessDecoder::open(&params, Recorder::default()).expect("open");
        let packet = Packet::video(nal::annex_b_nal(3, 7, &sps), 0, 1, 30, true);
        let pushed = decoder.push(&packet);
        if accepted {
            assert!(pushed.is_ok_and(|out| out.is_empty()), "{name}");
        } else {
            assert!(unsupported(pushed), "{name}: pushed");
        }
        let mut configured = params.clone();
        let ebsp = nal::annex_b_nal(3, 7, &sps);
        let pps = nal::annex_b_nal(3, 8, &pps_rbsp(1, false));
        configured.extra = avcc(&ebsp[4..], &pps[4..]);
        let opened = StatelessDecoder::open(&configured, Recorder::default());
        if accepted {
            assert!(opened.is_ok(), "{name}: open");
        } else {
            assert!(unsupported(opened), "{name}: open");
        }
    }
    // Slice groups are refused when their PPS arrives.
    let mut decoder = StatelessDecoder::open(&params, Recorder::default()).expect("open");
    let mut data = nal::annex_b_nal(
        3,
        7,
        &sps_rbsp(&Shape {
            profile: 66,
            ..PLAIN
        }),
    );
    data.extend(nal::annex_b_nal(3, 8, &pps_rbsp(2, false)));
    assert!(unsupported(
        decoder.push(&Packet::video(data, 0, 1, 30, true))
    ));
}

/// Clause 7.4.2.2, Table 7-2: a PPS list that is not coded falls back to
/// the SPS's (rule B) when the SPS has lists, and to the default (rule A)
/// when it has none.
#[test]
fn a_pps_without_lists_falls_back_to_the_sequence_lists() {
    const DEFAULT_INTRA: [u8; 16] = [
        6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
    ];
    const DEFAULT_INTER: [u8; 16] = [
        10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
    ];
    for (sequence, intra) in [(true, [20u8; 16]), (false, DEFAULT_INTRA)] {
        let sps = crate::set::parse_sps(&sps_rbsp(&Shape {
            scaling: sequence,
            ..PLAIN
        }))
        .expect("sps");
        let pps = crate::set::parse_pps_with(&pps_rbsp(1, true), |id| (id == 0).then_some(&sps))
            .expect("pps");
        let lists = pps.scaling_list_4x4();
        assert_eq!(lists[0], intra, "sequence lists {sequence}");
        assert_eq!(lists[1], intra);
        assert_eq!(lists[2], intra);
        assert_eq!(lists[3], DEFAULT_INTER);
        if sequence {
            assert_eq!(sps.scaling_list_4x4()[0], [20; 16]);
            assert_eq!(sps.scaling_list_4x4()[3], DEFAULT_INTER);
        } else {
            assert_eq!(sps.scaling_list_4x4(), [[16; 16]; 6]);
        }
        assert_eq!(pps.scaling_list_8x8()[0][..3], [6, 10, 10]);
        assert_eq!(pps.scaling_list_8x8()[1][..3], [9, 13, 13]);
    }
}

fn ffmpeg() -> Option<PathBuf> {
    ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
}

/// An Annex B stream from ffmpeg's libx264, or `None` when there is none.
fn x264(dir: &std::path::Path, name: &str, args: &[&str]) -> Option<Vec<u8>> {
    let ff = ffmpeg()?;
    let path = dir.join(format!("{name}.h264"));
    let status = Command::new(ff)
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("testsrc2=size=96x64:rate=30")
        .args(["-frames:v", "30", "-c:v", "libx264"])
        .args(args)
        .args(["-f", "h264"])
        .arg(&path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    std::fs::read(&path).ok()
}

/// Splits an Annex B stream before each access unit delimiter and before
/// each slice whose `first_mb_in_slice` is 0 that follows a slice.
fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let header = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            starts.push((header, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut saw_vcl = false;
    for (n, &(header, body)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map_or(data.len(), |(next, _)| *next);
        let kind = data[body] & 31;
        let vcl = kind == 1 || kind == 5;
        let first_slice = data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (vcl && saw_vcl && first_slice)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_vcl = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_vcl |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-h264-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

#[test]
fn x264_streams_come_out_as_the_software_decoder_shows_them() {
    let dir = scratch("stateless");
    let cases = [
        (
            "baseline",
            "baseline",
            "cabac=0:bframes=0:ref=4:keyint=30:scenecut=0",
        ),
        (
            "main weighted pyramid",
            "main",
            "cabac=1:8x8dct=0:ref=4:weightp=2:bframes=3:b-pyramid=normal:keyint=30:scenecut=0",
        ),
        (
            "high sixteen references",
            "high",
            "cabac=1:8x8dct=1:ref=16:weightp=2:bframes=3:b-pyramid=normal:keyint=30:scenecut=0",
        ),
        (
            "high cavlc",
            "high",
            "cabac=0:8x8dct=1:bframes=2:ref=3:keyint=30:scenecut=0",
        ),
        (
            "main slices",
            "main",
            "cabac=1:slices=4:bframes=2:ref=2:keyint=30:scenecut=0",
        ),
        (
            "high short gops",
            "high",
            "cabac=1:bframes=2:b-pyramid=strict:keyint=8:min-keyint=8:scenecut=0",
        ),
        (
            "high scaling matrix",
            "high",
            "cabac=1:bframes=2:ref=3:cqm=jvt:keyint=30:scenecut=0",
        ),
    ];
    for (index, (name, profile, options)) in cases.into_iter().enumerate() {
        let args = [
            "-profile:v",
            profile,
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            options,
        ];
        let Some(bytes) = x264(&dir, &index.to_string(), &args) else {
            eprintln!("skipped: no ffmpeg with libx264");
            return;
        };
        let packets: Vec<Packet> = access_units(&bytes)
            .into_iter()
            .enumerate()
            .map(|(n, unit)| Packet::video(unit, n as i64, 1, 30, n == 0))
            .collect();
        let mut params = CodecParams::video(CodecId::H264, 96, 64);
        params.nal_length_size = 0;
        let recorder = same_as_software(name, &params, &packets);
        assert_eq!(recorder.pictures, 30, "{name}");
        let longest = |list: usize| recorder.lists.iter().map(|lists| lists[list].len()).max();
        assert!(
            longest(0) > Some(1),
            "{name}: list 0 never held two pictures"
        );
        if options.contains("bframes=0") {
            assert_eq!(longest(1), Some(0), "{name}");
        } else {
            assert!(longest(1) > Some(0), "{name}: no B slice");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn x264_streams_out_of_scope_are_refused() {
    let dir = scratch("stateless-refused");
    let cases: [(&str, &[&str]); 3] = [
        (
            "interlaced",
            &["-pix_fmt", "yuv420p", "-x264-params", "interlaced=1"],
        ),
        ("4:2:2", &["-pix_fmt", "yuv422p", "-profile:v", "high422"]),
        (
            "10-bit",
            &["-pix_fmt", "yuv420p10le", "-profile:v", "high10"],
        ),
    ];
    for (index, (name, args)) in cases.into_iter().enumerate() {
        let Some(bytes) = x264(&dir, &index.to_string(), args) else {
            eprintln!("skipped {name}: ffmpeg could not make it");
            continue;
        };
        let units = access_units(&bytes);
        let mut params = CodecParams::video(CodecId::H264, 96, 64);
        params.nal_length_size = 0;
        let mut decoder = StatelessDecoder::open(&params, Recorder::default()).expect("open");
        let pushed = decoder.push(&Packet::video(units[0].clone(), 0, 1, 30, true));
        assert!(unsupported(pushed), "{name}");
        assert_eq!(decoder.accelerator().pictures, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
