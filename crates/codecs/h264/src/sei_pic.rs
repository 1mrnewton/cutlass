//! Picture timing and film grain.
//!
//! Picture timing changes the output time and leaves the samples alone, so
//! those samples match FFmpeg. Film grain follows H.274 equations 22–32.
//! FFmpeg's grain synthesis is a different approximation, and the
//! recommendation does not require the display method to match the model, so
//! grain pictures are not compared with FFmpeg. A P skip after a grained
//! picture stays flat: grain is applied at display, not written into the
//! reference.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

const LUMA: usize = 16 * 16;
const CHROMA: usize = 8 * 8;

#[test]
fn picture_timing_changes_the_output_time() {
    let plain = access(&[], head_idr(), 1, 30);
    let still = decode(&[&plain], 10, 2);
    assert_eq!(still[0].pts, 10);
    assert_eq!(still[0].duration, 2);
    assert!(flat(&still[0]));

    let doubled = access(&[sei(&[(1, &timing(0, 4, 7))])], head_idr(), 1, 30);
    let picture = decode(&[&doubled], 10, 2);
    assert_eq!(picture[0].pts, 14);
    assert_eq!(picture[0].duration, 4);
    assert!(flat(&picture[0]));

    let pulled = access(&[sei(&[(1, &timing(0, 0, 5))])], head_idr(), 1, 30);
    assert_eq!(decode(&[&pulled], 10, 2)[0].duration, 3);
    let tripled = access(&[sei(&[(1, &timing(0, 0, 8))])], head_idr(), 1, 30);
    assert_eq!(decode(&[&tripled], 10, 2)[0].duration, 6);
    let field = access(&[sei(&[(1, &timing(0, 0, 1))])], head_idr(), 1, 30);
    assert_eq!(decode(&[&field], 10, 2)[0].duration, 1);
    let frame = access(&[sei(&[(1, &timing(0, 0, 0))])], head_idr(), 1, 30);
    let kept = decode(&[&frame], 10, 2);
    assert_eq!(kept[0].pts, 10);
    assert_eq!(kept[0].duration, 2);

    let inexact = access(&[sei(&[(1, &timing(0, 2, 7))])], head_idr(), 1, 50);
    let held = decode(&[&inexact], 10, 2);
    assert_eq!(held[0].pts, 10, "a fractional clock tick is not rounded");
    assert_eq!(held[0].duration, 4);
    let exact = access(&[sei(&[(1, &timing(0, 5, 7))])], head_idr(), 1, 50);
    assert_eq!(decode(&[&exact], 10, 2)[0].pts, 13);

    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &doubled);
        assert_eq!(
            raw.len(),
            LUMA + CHROMA * 2,
            "ffmpeg returned {} bytes",
            raw.len()
        );
        assert_eq!(raw, vec![128u8; LUMA + CHROMA * 2]);
    }
}

#[test]
fn other_sei_leaves_the_picture_alone() {
    let filler = access(&[sei(&[(3, &[0xFF])])], head_idr(), 1, 30);
    let unknown = access(&[sei(&[(180, &[0x11, 0x22])])], head_idr(), 1, 30);
    let both = access(
        &[sei(&[(180, &[0x11]), (1, &timing(0, 0, 7))])],
        head_idr(),
        1,
        30,
    );
    for annex in [&filler, &unknown] {
        let picture = decode(&[annex], 10, 2);
        assert!(flat(&picture[0]));
        assert_eq!(picture[0].pts, 10);
        assert_eq!(picture[0].duration, 2);
    }
    let picture = decode(&[&both], 10, 2);
    assert!(flat(&picture[0]));
    assert_eq!(picture[0].duration, 4);
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &filler);
        assert_eq!(raw, vec![128u8; LUMA + CHROMA * 2]);
    }
}

#[test]
fn frequency_and_autoregressive_grain_change_the_display() {
    let frequency = access(&[sei(&[(19, &grain(0, 0, 0, 64))])], head_idr(), 1, 30);
    let again = access(&[sei(&[(19, &grain(0, 0, 0, 64))])], head_idr(), 1, 30);
    let first = decode(&[&frequency], 0, 1);
    let second = decode(&[&again], 0, 1);
    assert_eq!(first[0].picture.y(), second[0].picture.y());
    assert!(first[0].picture.y().iter().any(|sample| *sample != 128));
    assert!(first[0].picture.u().iter().all(|sample| *sample == 128));
    assert!(first[0].picture.v().iter().all(|sample| *sample == 128));

    let auto = access(&[sei(&[(19, &grain(1, 0, 0, 64))])], head_idr(), 1, 30);
    let auto = decode(&[&auto], 0, 1);
    assert!(auto[0].picture.y().iter().any(|sample| *sample != 128));
    assert_ne!(auto[0].picture.y(), first[0].picture.y());

    let scaled = access(&[sei(&[(19, &grain(0, 1, 0, 64))])], head_idr(), 1, 30);
    let scaled = decode(&[&scaled], 0, 1);
    assert!(scaled[0].picture.y().iter().any(|sample| *sample != 128));
    assert_ne!(scaled[0].picture.y(), first[0].picture.y());

    let cancel = access(&[sei(&[(19, &[0x80])])], head_idr(), 1, 30);
    assert!(flat(&decode(&[&cancel], 0, 1)[0]));
    let reserved = access(&[sei(&[(19, &grain(2, 0, 0, 64))])], head_idr(), 1, 30);
    assert!(flat(&decode(&[&reserved], 0, 1)[0]));
    let blend = access(&[sei(&[(19, &grain(0, 2, 0, 64))])], head_idr(), 1, 30);
    assert!(flat(&decode(&[&blend], 0, 1)[0]));
    let other_depth = access(&[sei(&[(19, &grain_other_depth())])], head_idr(), 1, 30);
    assert!(flat(&decode(&[&other_depth], 0, 1)[0]));
}

#[test]
fn grain_is_not_stored_in_the_reference() {
    let idr = access(&[sei(&[(19, &grain(0, 0, 0, 64))])], head_idr(), 1, 30);
    let skipped = access(&[], head_p(), 1, 30);
    let pictures = decode(&[&idr, &skipped], 0, 1);
    assert_eq!(pictures.len(), 2);
    assert!(pictures[0].picture.y().iter().any(|sample| *sample != 128));
    assert!(flat(&pictures[1]));
}

#[test]
fn grain_period_one_persists_until_an_idr() {
    let first = access(&[sei(&[(19, &grain(0, 0, 1, 64))])], head_idr(), 1, 30);
    let next = access(&[], head_i(1), 1, 30);
    let pictures = decode(&[&first, &next], 0, 1);
    assert_eq!(pictures[0].picture.y(), pictures[1].picture.y());
    assert!(pictures[0].picture.y().iter().any(|sample| *sample != 128));

    let once = access(&[sei(&[(19, &grain(0, 0, 0, 64))])], head_idr(), 1, 30);
    let clean = decode(&[&once, &next], 0, 1);
    assert!(clean[0].picture.y().iter().any(|sample| *sample != 128));
    assert!(flat(&clean[1]));

    let reset = access(&[], head_idr(), 1, 30);
    let ended = decode(&[&first, &reset], 0, 1);
    assert!(ended[0].picture.y().iter().any(|sample| *sample != 128));
    assert!(flat(&ended[1]));

    let stop = access(&[sei(&[(19, &[0x80])])], head_i(1), 1, 30);
    let cancelled = decode(&[&first, &stop, &next], 0, 1);
    assert!(flat(&cancelled[1]));
    assert!(flat(&cancelled[2]));
}

#[test]
fn a_bad_sei_rejects_and_a_later_idr_decodes() {
    let cases = [
        sei(&[(1, &[0x00])]),
        sei(&[(1, &timing(0, 0, 9))]),
        sei(&[(19, &grain(0, 0, 0, -1))]),
        sei(&[(19, &grain(0, 0, 16_385, 64))]),
        vec![0x01, 0x0A, 0x00, 0x80],
    ];
    let mut decoder = open();
    for payload in cases {
        let annex = access(&[payload], head_idr(), 1, 30);
        let err = decoder
            .push(&Packet::video(annex, 0, 1, 30, false))
            .unwrap_err();
        assert!(
            matches!(
                err,
                CodecError::Malformed { what, .. } if what == "sei" || what == "pic struct"
            ),
            "{err:?}"
        );
        let pictures = {
            let mut out = decoder
                .push(&Packet::video(
                    access(&[], head_idr(), 1, 30),
                    1,
                    1,
                    30,
                    true,
                ))
                .unwrap();
            out.extend(decoder.drain().unwrap());
            out
        };
        assert!(pictures.iter().any(flat));
    }
}

fn decode(annexes: &[&[u8]], pts: i64, duration: i64) -> Vec<VideoOutput> {
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in annexes {
        pictures.extend(
            decoder
                .push(&Packet::video(annex.to_vec(), pts, duration, 30, true))
                .unwrap(),
        );
    }
    pictures.extend(decoder.drain().unwrap());
    assert_eq!(pictures.len(), annexes.len());
    pictures
}

fn flat(picture: &VideoOutput) -> bool {
    picture
        .picture
        .y()
        .iter()
        .chain(picture.picture.u())
        .chain(picture.picture.v())
        .all(|sample| *sample == 128)
}

fn open() -> H264Decoder {
    H264Decoder::open(&CodecParams::video(CodecId::H264, 16, 16)).unwrap()
}

fn access(messages: &[Vec<u8>], head: Head, units: u32, scale: u32) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(units, scale));
    out.extend(annex_b_nal(3, 8, &pps()));
    for message in messages {
        out.extend(annex_b_nal(0, 6, message));
    }
    let kind = if head.idr { 5 } else { 1 };
    let ref_idc = if head.idr { 3 } else { 0 };
    out.extend(annex_b_nal(ref_idc, kind, &slice(head)));
    out
}

fn head_idr() -> Head {
    Head {
        kind: 2,
        idr: true,
        frame_num: 0,
        poc: 0,
        referenced: true,
    }
}

fn head_i(poc: u32) -> Head {
    Head {
        kind: 2,
        idr: false,
        frame_num: 1,
        poc,
        referenced: false,
    }
}

fn head_p() -> Head {
    Head {
        kind: 0,
        idr: false,
        frame_num: 1,
        poc: 1,
        referenced: false,
    }
}

fn sei(messages: &[(u32, &[u8])]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (kind, payload) in messages {
        push_code(&mut bytes, *kind);
        push_code(&mut bytes, u32::try_from(payload.len()).unwrap_or(0));
        bytes.extend_from_slice(payload);
    }
    bytes.push(0x80);
    bytes
}

fn push_code(out: &mut Vec<u8>, mut value: u32) {
    while value >= 255 {
        out.push(0xFF);
        value -= 255;
    }
    let Ok(byte) = u8::try_from(value) else {
        return;
    };
    out.push(byte);
}

fn timing(cpb: u32, dpb: u32, pic_struct: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(cpb, 5);
    w.bits(dpb, 5);
    w.bits(pic_struct, 4);
    let clocks = [1u32, 1, 1, 2, 2, 3, 3, 2, 3];
    if let Some(count) = clocks.get(pic_struct as usize) {
        for _ in 0..*count {
            w.bit(false);
        }
    }
    w.align_zeros();
    w.bytes
}

fn grain(model: u32, blend: u32, period: u32, sigma: i32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bit(false);
    w.bits(model, 2);
    if model > 1 {
        w.align_zeros();
        return w.bytes;
    }
    w.bit(false);
    w.bits(blend, 2);
    if blend > 1 {
        w.align_zeros();
        return w.bytes;
    }
    w.bits(0, 4);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bits(0, 8);
    w.bits(0, 3);
    w.bits(0, 8);
    w.bits(255, 8);
    w.se(sigma);
    w.ue(period);
    w.align_zeros();
    w.bytes
}

fn grain_other_depth() -> Vec<u8> {
    let mut w = Writer::new();
    w.bit(false);
    w.bits(0, 2);
    w.bit(true);
    w.bits(2, 3);
    w.bits(2, 3);
    w.bit(false);
    w.bits(1, 8);
    w.bits(1, 8);
    w.bits(1, 8);
    w.bits(0, 2);
    w.bits(0, 4);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bits(0, 8);
    w.bits(0, 3);
    w.bits(0, 8);
    w.bits(255, 8);
    w.se(80);
    w.ue(0);
    w.align_zeros();
    w.bytes
}

struct Head {
    kind: u32,
    idr: bool,
    frame_num: u32,
    poc: u32,
    referenced: bool,
}

fn slice(head: Head) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(head.kind);
    w.ue(0);
    w.bits(head.frame_num, 4);
    if head.idr {
        w.ue(0);
    }
    w.bits(head.poc, 4);
    let intra = head.kind == 2;
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
    w.ue(1);
    if intra {
        w.ue(3);
        w.ue(0);
        w.se(0);
        w.bit(true);
    } else {
        w.ue(1);
    }
    w.trailing();
    w.bytes
}

fn sps(units: u32, scale: u32) -> Vec<u8> {
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
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bits(units, 32);
    w.bits(scale, 32);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.bits(0, 4);
    w.bits(0, 4);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 5);
    w.bits(4, 5);
    w.bits(4, 5);
    w.bits(0, 5);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.trailing();
    w.bytes
}

fn pps() -> Vec<u8> {
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
    w.se(0);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
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
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vsync",
            "0",
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
