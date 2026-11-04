//! Data partitions and SP/SI slices.
//!
//! FFmpeg 8.1.1 does not decode data partitions, so those pictures follow the
//! recommendation: I_PCM bytes, an I_16x16 DC picture, a P skip, and one coded
//! residual. I macroblocks inside SP and SI match FFmpeg. SP P macroblocks and
//! the SI macroblock type follow clause 8.6, which FFmpeg does not run.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

const LUMA: usize = 16 * 16;
const CHROMA: usize = 8 * 8;

#[test]
fn partition_i_pcm_and_i16_follow_the_recommendation() {
    let pcm = access(&[nal(2, &partition_i(25, true)), nal(3, &partition_pcm(40))]);
    let flat = access(&[nal(2, &partition_i(3, false)), nal(3, &partition_dc())]);
    let (y, u, v) = decode_one(&pcm);
    assert!(y.iter().all(|sample| *sample == 40));
    assert!(u.iter().all(|sample| *sample == 128));
    assert!(v.iter().all(|sample| *sample == 128));
    let (y, u, v) = decode_one(&flat);
    assert!(
        y.iter()
            .chain(u.iter())
            .chain(v.iter())
            .all(|sample| *sample == 128)
    );
}

#[test]
fn partition_p_skip_copies_and_partition_c_carries_the_residual() {
    let idr = idr_flat();
    let skipped = access(&[nal(2, &partition_p_skip())]);
    let (y, u, v) = decode_two(&idr, &skipped);
    assert!(
        y.iter()
            .chain(u.iter())
            .chain(v.iter())
            .all(|sample| *sample == 128)
    );
    let coded = access(&[nal(2, &partition_p_dc()), nal(4, &partition_dc_coeff())]);
    let (y, u, v) = decode_two(&idr, &coded);
    for row in 0..16 {
        for col in 0..16 {
            let sample = y[row * 16 + col];
            let want = if row < 4 && col < 4 { 131 } else { 128 };
            assert_eq!(sample, want, "luma {row},{col}");
        }
    }
    assert!(u.iter().chain(v.iter()).all(|sample| *sample == 128));
}

#[test]
fn sp_residual_follows_the_switch_flag() {
    let idr = idr_flat();
    for (switch, coded) in [(false, 131u8), (true, 128)] {
        let annex = access(&[nal(1, &sp_dc(switch))]);
        let (y, u, v) = decode_two(&idr, &annex);
        for row in 0..16 {
            for col in 0..16 {
                let want = if row < 4 && col < 4 { coded } else { 128 };
                assert_eq!(y[row * 16 + col], want, "switch {switch} {row},{col}");
            }
        }
        assert!(u.iter().chain(v.iter()).all(|sample| *sample == 128));
    }
}

#[test]
fn sp_skip_quantises_the_prediction_and_si_does_the_same() {
    let idr = idr_flat();
    for switch in [false, true] {
        let kept = access(&[nal(1, &sp_skip(switch, -26))]);
        let (y, u, v) = decode_two(&idr, &kept);
        assert!(
            y.iter()
                .chain(u.iter())
                .chain(v.iter())
                .all(|sample| *sample == 128),
            "switch {switch}"
        );
        let dropped = access(&[nal(1, &sp_skip(switch, 0))]);
        let (y, u, v) = decode_two(&idr, &dropped);
        assert!(
            y.iter().all(|sample| *sample == 127),
            "switch {switch} luma"
        );
        assert!(u.iter().chain(v.iter()).all(|sample| *sample == 128));
    }
    let si = access(&[nal(1, &si_macroblock())]);
    let (y, u, v) = decode_one(&si);
    assert!(y.iter().all(|sample| *sample == 127));
    assert!(u.iter().chain(v.iter()).all(|sample| *sample == 128));
}

#[test]
fn sp_and_si_intra_pcm_match_ffmpeg() {
    let idr = idr_flat();
    let sp = access(&[nal(1, &sp_pcm(false))]);
    let si = access(&[nal(1, &si_pcm())]);
    let frame = LUMA + CHROMA * 2;
    let want = [vec![40u8; LUMA], vec![128; CHROMA], vec![128; CHROMA]].concat();
    for annex in [&sp, &si] {
        let (y, u, v) = decode_two(&idr, annex);
        assert!(y.iter().all(|sample| *sample == 40));
        assert!(u.iter().chain(v.iter()).all(|sample| *sample == 128));
        if let Some(ffmpeg) = ffmpeg_bin() {
            let raw = ffmpeg_raw(&ffmpeg, &concat(&idr, annex));
            assert_eq!(raw.len(), frame * 2);
            assert_eq!(&raw[frame..], want.as_slice());
        }
    }
}

#[test]
fn a_bad_partition_or_qs_rejects_and_a_later_idr_decodes() {
    let cases = [
        (access(&[nal(3, &trailing_only())]), "partition"),
        (
            access(&[
                nal(2, &i_header()),
                nal(4, &trailing_id()),
                nal(3, &trailing_id()),
            ]),
            "partition",
        ),
        (cabac_partition(), "data partition"),
        (access(&[nal(2, &partition_i(25, true))]), "partition"),
        (
            access(&[
                nal(2, &partition_i(3, false)),
                nal(3, &partition_dc()),
                nal(4, &trailing_id()),
            ]),
            "partition",
        ),
        (access(&[nal(2, &i_header_id(1))]), "partition"),
        (access(&[nal(1, &sp_skip(false, 26))]), "slice qs"),
        (bad_init_qs(), "init qs"),
    ];
    for (annex, what) in cases {
        let mut decoder = open();
        let err = decoder
            .push(&Packet::video(annex, 0, 1, 30, false))
            .unwrap_err();
        assert!(
            matches!(err, CodecError::Malformed { what: got, .. } if got == what),
            "{err:?}"
        );
        let mut pictures = decoder
            .push(&Packet::video(idr_flat(), 1, 1, 30, true))
            .unwrap();
        pictures.extend(decoder.drain().unwrap());
        assert_eq!(pictures.len(), 1, "{what}");
        assert!(pictures[0].picture.y().iter().all(|sample| *sample == 128));
    }
}

fn open() -> H264Decoder {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    H264Decoder::open(&params).unwrap()
}

fn decode_one(annex: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut decoder = open();
    let mut pictures = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .unwrap();
    pictures.extend(decoder.drain().unwrap());
    assert_eq!(pictures.len(), 1);
    planes(&pictures[0])
}

fn decode_two(first: &[u8], second: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut decoder = open();
    let mut pictures = decoder
        .push(&Packet::video(first.to_vec(), 0, 1, 30, true))
        .unwrap();
    pictures.extend(
        decoder
            .push(&Packet::video(second.to_vec(), 1, 1, 30, false))
            .unwrap(),
    );
    pictures.extend(decoder.drain().unwrap());
    assert_eq!(pictures.len(), 2);
    planes(&pictures[1])
}

fn planes(picture: &cutlass_decode::VideoOutput) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        picture.picture.y().to_vec(),
        picture.picture.u().to_vec(),
        picture.picture.v().to_vec(),
    )
}

fn access(nals: &[Vec<u8>]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps());
    out.extend(annex_b_nal(3, 8, &pps(false)));
    for nal in nals {
        out.extend(nal);
    }
    out
}

fn nal(kind: u8, rbsp: &[u8]) -> Vec<u8> {
    annex_b_nal(0, kind, rbsp)
}

fn idr_flat() -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 2,
            idr: true,
            frame_num: 0,
            poc: 0,
            switch: false,
            qs: 0,
            referenced: true,
        },
    );
    w.ue(3);
    w.ue(0);
    w.se(0);
    w.bit(true);
    w.trailing();
    let mut out = annex_b_nal(3, 7, &sps());
    out.extend(annex_b_nal(3, 8, &pps(false)));
    out.extend(annex_b_nal(3, 5, &w.bytes));
    out
}

fn concat(first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut out = first.to_vec();
    out.extend_from_slice(second);
    out
}

fn sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(88, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn pps(cabac: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    w.bit(cabac);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn bad_init_qs() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(26);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    let mut out = annex_b_nal(3, 7, &sps());
    out.extend(annex_b_nal(3, 8, &w.bytes));
    out
}

fn cabac_partition() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps());
    out.extend(annex_b_nal(3, 8, &pps(true)));
    out.extend(nal(2, &i_header()));
    out
}

struct Head {
    kind: u32,
    idr: bool,
    frame_num: u32,
    poc: u32,
    switch: bool,
    qs: i32,
    referenced: bool,
}

fn header(w: &mut Writer, head: Head) {
    w.ue(0);
    w.ue(head.kind);
    w.ue(0);
    w.bits(head.frame_num, 4);
    if head.idr {
        w.ue(0);
    }
    w.bits(head.poc, 4);
    let intra = head.kind == 2 || head.kind == 4;
    if !intra {
        w.bit(false);
        w.bit(false);
    }
    if head.referenced {
        if head.idr {
            w.bit(false);
            w.bit(false);
        } else {
            w.bit(false);
        }
    }
    w.se(0);
    if head.kind == 3 || head.kind == 4 {
        if head.kind == 3 {
            w.bit(head.switch);
        }
        w.se(head.qs);
    }
    w.ue(1);
}

fn i_header() -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 2,
            idr: false,
            frame_num: 0,
            poc: 0,
            switch: false,
            qs: 0,
            referenced: false,
        },
    );
    w.trailing();
    w.bytes
}

fn i_header_id(slice_id: u32) -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 2,
            idr: false,
            frame_num: 0,
            poc: 0,
            switch: false,
            qs: 0,
            referenced: false,
        },
    );
    w.ue(slice_id);
    w.trailing();
    w.bytes
}

fn partition_i(mb_type: u32, pcm: bool) -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 2,
            idr: false,
            frame_num: 0,
            poc: 0,
            switch: false,
            qs: 0,
            referenced: false,
        },
    );
    w.ue(0);
    w.ue(mb_type);
    if !pcm {
        w.ue(0);
        w.se(0);
    }
    w.trailing();
    w.bytes
}

fn partition_pcm(luma: u8) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(luma, 256));
    w.bytes.extend(std::iter::repeat_n(128, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn partition_dc() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn partition_p_skip() -> Vec<u8> {
    let mut w = Writer::new();
    p_header(&mut w, 0);
    w.ue(0);
    w.ue(1);
    w.trailing();
    w.bytes
}

fn partition_p_dc() -> Vec<u8> {
    let mut w = Writer::new();
    p_header(&mut w, 0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.trailing();
    w.bytes
}

fn partition_dc_coeff() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn p_header(w: &mut Writer, kind: u32) {
    header(
        w,
        Head {
            kind,
            idr: false,
            frame_num: 1,
            poc: 1,
            switch: false,
            qs: 0,
            referenced: false,
        },
    );
}

fn sp_dc(switch: bool) -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 3,
            idr: false,
            frame_num: 1,
            poc: 1,
            switch,
            qs: -26,
            referenced: false,
        },
    );
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn sp_skip(switch: bool, qs: i32) -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 3,
            idr: false,
            frame_num: 1,
            poc: 1,
            switch,
            qs,
            referenced: false,
        },
    );
    w.ue(1);
    w.trailing();
    w.bytes
}

fn sp_pcm(switch: bool) -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 3,
            idr: false,
            frame_num: 1,
            poc: 1,
            switch,
            qs: 0,
            referenced: false,
        },
    );
    w.ue(0);
    w.ue(30);
    pcm_body(&mut w, 40);
    w.bytes
}

fn si_pcm() -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 4,
            idr: false,
            frame_num: 1,
            poc: 1,
            switch: false,
            qs: 0,
            referenced: false,
        },
    );
    w.ue(26);
    pcm_body(&mut w, 40);
    w.bytes
}

fn si_macroblock() -> Vec<u8> {
    let mut w = Writer::new();
    header(
        &mut w,
        Head {
            kind: 4,
            idr: false,
            frame_num: 0,
            poc: 0,
            switch: false,
            qs: 0,
            referenced: false,
        },
    );
    w.ue(0);
    for _ in 0..16 {
        w.bit(true);
    }
    w.ue(0);
    w.ue(3);
    w.trailing();
    w.bytes
}

fn pcm_body(w: &mut Writer, luma: u8) {
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(luma, 256));
    w.bytes.extend(std::iter::repeat_n(128, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
}

fn trailing_only() -> Vec<u8> {
    let mut w = Writer::new();
    w.trailing();
    w.bytes
}

fn trailing_id() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.trailing();
    w.bytes
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn ffmpeg_raw(ffmpeg: &std::path::Path, annex: &[u8]) -> Vec<u8> {
    let mut child = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), annex).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

struct Writer {
    bytes: Vec<u8>,
    bit: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    fn bit(&mut self, value: bool) {
        let index = self.bit / 8;
        if index == self.bytes.len() {
            self.bytes.push(0);
        }
        if value {
            self.bytes[index] |= 1 << (7 - (self.bit % 8));
        }
        self.bit += 1;
    }

    fn bits(&mut self, value: u32, n: u32) {
        for shift in (0..n).rev() {
            self.bit(((value >> shift) & 1) == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let zeros = 31 - code.leading_zeros();
        for _ in 0..zeros {
            self.bit(false);
        }
        self.bits(code, zeros + 1);
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            value.unsigned_abs() * 2 - 1
        } else {
            value.unsigned_abs() * 2
        };
        self.ue(code);
    }

    fn align_zeros(&mut self) {
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }

    fn trailing(&mut self) {
        self.bit(true);
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }
}
