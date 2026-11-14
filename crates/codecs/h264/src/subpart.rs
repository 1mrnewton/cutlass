//! B macroblocks whose 8×8 blocks are split further.
//!
//! libx264 never splits an 8×8 block of a B macroblock, so no stream it
//! writes asks the question these do: whether the block above and to the
//! right of a sub-macroblock partition is there when it belongs to a later
//! 8×8 block of the same macroblock. Clause 6.4.11.7 says it is not, and
//! clause 8.4.1.3.2 then takes the block above and to the left instead.
//! Encoders in phones do write such macroblocks.
//!
//! The stream is written here: an intra picture of I_PCM macroblocks whose
//! luma is a ramp across the picture, a P picture that skips every
//! macroblock, and a B picture between them with no residual and no
//! deblocking. Every vector is a whole number of samples to the right, so a
//! sample of the B picture names the vector that predicted it.
//! FFmpeg's software decoder is the picture oracle when `ffmpeg` is
//! installed; the vectors the recommendation gives are checked either way.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

const WIDTH: usize = 48;
const HEIGHT: usize = 16;

/// The samples the B picture's luma moved to the right, by region. The two
/// regions marked are the partitions whose above-right neighbour is a later
/// 8×8 block; a decoder that reads that block moves them by one sample.
fn shift(x: usize, y: usize) -> usize {
    match (x / 8, y / 4) {
        // Macroblock 0, B_L1_16x16.
        (0 | 1, _) => 2,
        // Macroblock 1: two list 0 blocks, a list 1 8×4 pair, a list 1 block.
        (2, 0 | 1) => 0,
        (3, 0 | 1) => 2,
        (2, 2) => 1,
        // Above-right is the fourth block, list 1 and not yet decoded.
        (2, _) => 2,
        (3, _) => 4,
        // Macroblock 2: a list 0 8×4 pair, a direct block, two list 0 blocks.
        (4, 0) => 1,
        // Above-right is the second block, direct with a zero vector.
        (4, 1) => 2,
        _ => 0,
    }
}

fn ramp(x: usize) -> u8 {
    20 + 4 * x.min(WIDTH - 1) as u8
}

#[test]
fn a_later_block_of_the_macroblock_is_not_a_neighbour() {
    let units = units();
    let frames = decode(&units, 1);
    assert_eq!(frames.len(), 3, "picture count");
    for (index, name) in [(0, "intra"), (2, "skipped P")] {
        let luma = frames[index].picture.y();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                assert_eq!(luma[y * WIDTH + x], ramp(x), "{name} luma at ({x}, {y})");
            }
        }
    }
    let luma = frames[1].picture.y();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            assert_eq!(
                luma[y * WIDTH + x],
                ramp(x + shift(x, y)),
                "B luma at ({x}, {y}) moved by the wrong vector"
            );
        }
    }
    for frame in &frames {
        assert!(frame.picture.u().iter().all(|sample| *sample == 128), "cb");
        assert!(frame.picture.v().iter().all(|sample| *sample == 128), "cr");
    }
    let ours = pictures(&frames);
    assert_eq!(
        pictures(&decode(&units, 4)),
        ours,
        "threads changed the picture"
    );
    let Some(ffmpeg) = ffmpeg_bin() else {
        return;
    };
    let theirs = ffmpeg_raw(&ffmpeg, &units.concat());
    assert_eq!(theirs.len(), ours.len(), "ffmpeg picture count");
    assert!(theirs == ours, "pictures differ from ffmpeg");
}

fn pictures(frames: &[VideoOutput]) -> Vec<u8> {
    let mut out = Vec::new();
    for frame in frames {
        out.extend_from_slice(frame.picture.y());
        out.extend_from_slice(frame.picture.u());
        out.extend_from_slice(frame.picture.v());
    }
    out
}

fn decode(units: &[Vec<u8>], threads: usize) -> Vec<VideoOutput> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH as u32, HEIGHT as u32);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params)
        .expect("open")
        .with_threads(threads);
    let mut frames = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        frames.extend(
            decoder
                .push(&Packet::video(
                    unit.clone(),
                    index as i64,
                    1,
                    30,
                    index == 0,
                ))
                .unwrap_or_else(|err| panic!("unit {index}: {err}")),
        );
    }
    frames.extend(decoder.drain().expect("drain"));
    frames
}

/// The three access units in decoding order: intra, P, B.
fn units() -> Vec<Vec<u8>> {
    let mut first = annex_b_nal(3, 7, &sps());
    first.extend(annex_b_nal(3, 8, &pps()));
    first.extend(annex_b_nal(3, 5, &intra_slice()));
    vec![
        first,
        annex_b_nal(2, 1, &p_slice()),
        annex_b_nal(0, 1, &b_slice()),
    ]
}

fn sps() -> Vec<u8> {
    let mut w = Writer::new();
    // Main profile, level 3.
    w.bits(77, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    // Four bits of frame_num, picture order count type 0 with four bits.
    w.ue(0);
    w.ue(0);
    w.ue(0);
    // Two reference frames, no gaps.
    w.ue(2);
    w.bit(false);
    w.ue((WIDTH / 16 - 1) as u32);
    w.ue((HEIGHT / 16 - 1) as u32);
    // Frame macroblocks only, 8×8 direct inference, no cropping.
    w.bit(true);
    w.bit(true);
    w.bit(false);
    // Annex E, for the bitstream restriction alone: one picture is reordered.
    w.bit(true);
    for _ in 0..9 {
        w.bit(false);
    }
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(16);
    w.ue(16);
    w.ue(1);
    w.ue(2);
    w.trailing();
    w.bytes
}

fn pps() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    // CAVLC, one slice group, one reference in each list.
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    // No weights.
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    // The slice header carries the deblocking switch.
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn intra_slice() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(7);
    w.ue(0);
    // frame_num 0, idr_pic_id 0, order count 0.
    w.bits(0, 4);
    w.ue(0);
    w.bits(0, 4);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
    for mb in 0..WIDTH / 16 {
        // I_PCM.
        w.ue(25);
        w.align_zeros();
        for _ in 0..16 {
            for x in 0..16 {
                w.bits(u32::from(ramp(mb * 16 + x)), 8);
            }
        }
        for _ in 0..128 {
            w.bits(128, 8);
        }
    }
    w.trailing();
    w.bytes
}

fn p_slice() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(5);
    w.ue(0);
    // frame_num 1, order count 4.
    w.bits(1, 4);
    w.bits(4, 4);
    // Default reference count, list as built, sliding window.
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
    // Every macroblock is skipped.
    w.ue((WIDTH / 16) as u32);
    w.trailing();
    w.bytes
}

fn b_slice() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(6);
    w.ue(0);
    // frame_num 2, order count 2: between the two references.
    w.bits(2, 4);
    w.bits(2, 4);
    // Spatial direct, default reference counts, both lists as built.
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);

    // Macroblock 0: B_L1_16x16, two samples to the right. It gives the next
    // macroblock a list 1 vector on its left and above-left.
    w.ue(0);
    w.ue(2);
    mvd(&mut w, 8);
    w.ue(0);

    // Macroblock 1: B_8x8 of B_L0_8x8, B_L0_8x8, B_L1_8x4, B_L1_8x8.
    w.ue(0);
    w.ue(22);
    for kind in [1, 1, 6, 2] {
        w.ue(kind);
    }
    // List 0: the first block stays, the second moves two samples and
    // gives the next macroblock a list 0 vector on its left.
    mvd(&mut w, 0);
    mvd(&mut w, 8);
    // List 1. The upper 8×4 is predicted two samples from the left and
    // moves one. The lower 8×4 codes no difference: its left and above-left
    // say two and the half above says one, so the median is two, unless the
    // fourth block is read for a zero vector it does not have yet.
    mvd(&mut w, -4);
    mvd(&mut w, 0);
    // The fourth block is predicted one sample from its left and moves four.
    mvd(&mut w, 12);
    w.ue(0);

    // Macroblock 2: B_8x8 of B_L0_8x4, B_Direct_8x8, B_L0_8x8, B_L0_8x8.
    w.ue(0);
    w.ue(22);
    for kind in [4, 0, 1, 1] {
        w.ue(kind);
    }
    // List 0. The upper 8×4 is predicted two samples from the left and
    // moves one. The lower 8×4 codes no difference: the median is two
    // again, unless the direct block, whose vector the co-located skipped
    // macroblock makes zero, is read as the above-right neighbour.
    mvd(&mut w, -4);
    mvd(&mut w, 0);
    mvd(&mut w, 0);
    mvd(&mut w, 0);
    w.ue(0);

    w.trailing();
    w.bytes
}

/// A motion vector difference of `x` quarter samples across and none down.
fn mvd(w: &mut Writer, x: i32) {
    w.se(x);
    w.se(0);
}

fn ffmpeg_raw(ffmpeg: &PathBuf, annex: &[u8]) -> Vec<u8> {
    let path =
        std::env::temp_dir().join(format!("cutlass-h264-subpart-{}.h264", std::process::id()));
    std::fs::write(&path, annex).expect("annex");
    let raw = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-i",
        ])
        .arg(&path)
        .args(["-an", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stdout(Stdio::piped())
        .output()
        .expect("ffmpeg raw");
    let _ = std::fs::remove_file(&path);
    assert!(raw.status.success(), "ffmpeg refused the stream");
    raw.stdout
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return Some(PathBuf::from(candidate));
        }
    }
    eprintln!("skipped: ffmpeg is not installed");
    None
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
