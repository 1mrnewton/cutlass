//! Random valid Layer III streams for the parts of the syntax the encoder
//! never writes: intensity stereo (MPEG-1 and LSF positions, illegal
//! positions, per-window bounds), mixed blocks, `scfsi`, `preflag`, the
//! LSF partitions of the intensity right channel, subblock gain, every
//! Huffman table, and count1 table B.
//!
//! Each frame has `main_data_begin` 0 and a high bitrate, so it stands
//! alone. The streams are decoded by this crate and, when they are
//! installed, by ffmpeg and mpg123, and the outputs must agree within the
//! ISO/IEC 11172-4 bounds.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Output, Packet};

use crate::Mp3Decoder;
use crate::bits::BitWriter;
use crate::header::{FrameHeader, Mode, Version};
use crate::huffenc::{long_bounds, write_count1, write_pairs};
use crate::layer3::lsf_slen;
use crate::sideinfo::{GranuleInfo, SHORT, SideInfo};
use crate::tables::{LSF_PARTITIONS, SHORT_WIDTHS, SLEN, TABLE_SELECT, long_starts, short_starts};

const ISO_RMS: f64 = 8.809_142_651_444_272e-6;
const ISO_MAX: f64 = 6.103_515_625e-5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }

    fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }
}

/// The largest magnitude `table_select` can code, kept small for linbits
/// tables so the output stays near full scale.
fn table_max(select: u8) -> i32 {
    let (table, linbits) = TABLE_SELECT[usize::from(select)];
    match table {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        5 | 6 => 3,
        7..=9 => 5,
        10..=12 => 7,
        13 | 15 => 15,
        _ => 15 + ((1i32 << linbits) - 1).min(48),
    }
}

fn random_table(rng: &mut Rng) -> u8 {
    loop {
        let select = rng.below(32) as u8;
        if TABLE_SELECT[usize::from(select)].0 != 255 {
            return select;
        }
    }
}

/// One granule of one channel, ready to write.
struct Granule {
    info: GranuleInfo,
    /// Scalefactor fields as `(value, bits)`.
    part2: Vec<(u32, u32)>,
    ix: [i32; 576],
    bounds: [(usize, usize); 3],
    big_end: usize,
    count1_end: usize,
}

impl Granule {
    fn write(&self, writer: &mut BitWriter) {
        for &(value, bits) in &self.part2 {
            writer.put(value, bits);
        }
        for (region, (from, to)) in self.bounds.iter().copied().enumerate() {
            write_pairs(writer, &self.ix, from, to, self.info.table_select[region]);
        }
        write_count1(
            writer,
            &self.ix,
            self.big_end,
            self.count1_end,
            self.info.count1_table_b,
        );
    }

    fn bits(&self) -> usize {
        let mut writer = BitWriter::new();
        self.write(&mut writer);
        writer.bits()
    }
}

/// The block shape both channels of a joint-stereo granule share.
#[derive(Clone, Copy)]
struct Shape {
    window_switching: bool,
    block_type: u8,
    mixed: bool,
}

const NORMAL: Shape = Shape {
    window_switching: false,
    block_type: 0,
    mixed: false,
};

/// A block shape that may follow `previous`: a short block only after a
/// start or short block, a long one only after a normal or stop block, as
/// an encoder must switch, and no pure short block after a mixed one.
/// `allow_mixed` off leaves mixed blocks out.
/// ffmpeg relies on both (it drops the long overlap tail under a short
/// window), so a stream that breaks them is not one the references agree
/// on.
fn random_shape(rng: &mut Rng, previous: Shape, allow_mixed: bool) -> Shape {
    let (block_type, mixed) = if previous.block_type == 1 || previous.block_type == SHORT {
        match rng.below(3) {
            0 if !previous.mixed => (SHORT, false),
            0 | 1 => (SHORT, allow_mixed),
            _ => (3, false),
        }
    } else if rng.chance(2) {
        return NORMAL;
    } else {
        (1, false)
    };
    Shape {
        window_switching: true,
        block_type,
        mixed,
    }
}

/// Corners of the syntax where ffmpeg and mpg123 disagree with each other.
/// Off, a stream uses neither, and both references must match it.
#[derive(Clone, Copy, Default)]
struct Disputed {
    /// LSF intensity positions from 16 up to one below the largest value
    /// of the field. ISO/IEC 13818-3 makes only the largest value illegal;
    /// ffmpeg treats 16 and above as illegal, and mpg123 treats 7 as
    /// illegal (7 is never written).
    lsf_high_positions: bool,
    /// MPEG-1 intensity positions 8 to 15, illegal like 7 here and in
    /// ffmpeg; mpg123 takes their tangent, which divides by zero at 9.
    mpeg1_high_positions: bool,
    /// `scfsi` on the right channel of an intensity-stereo frame, where
    /// mpg123 differs from this crate and ffmpeg.
    intensity_scfsi: bool,
    /// Mixed blocks at the MPEG-2.5 rates, where mpg123 differs from this
    /// crate and ffmpeg.
    mpeg25_mixed: bool,
}

struct Spec<'a> {
    header: &'a FrameHeader,
    shape: Shape,
    /// The right channel of an intensity-stereo frame.
    intensity_right: bool,
    /// MPEG-1 granule 1: the `scfsi` groups reused from granule 0.
    scfsi: [bool; 4],
    max_pairs: u64,
    disputed: Disputed,
}

fn granule(rng: &mut Rng, spec: &Spec<'_>) -> Granule {
    let header = spec.header;
    let lsf = header.version.lsf();
    let table = header.rate_table();
    let shape = spec.shape;
    let short = shape.window_switching && shape.block_type == SHORT;
    let mut info = GranuleInfo {
        global_gain: 110 + rng.below(45) as u8,
        window_switching: shape.window_switching,
        block_type: shape.block_type,
        mixed: shape.mixed,
        scalefac_scale: rng.chance(2),
        count1_table_b: rng.chance(2),
        ..GranuleInfo::default()
    };
    if shape.window_switching {
        for gain in &mut info.subblock_gain {
            *gain = rng.below(8) as u8;
        }
    } else {
        info.region0_count = rng.below(16) as u8;
        info.region1_count = rng.below(8) as u8;
    }

    // Scalefactors.
    let mut part2 = Vec::new();
    let (long_end, short_start) = match (short, shape.mixed) {
        (true, true) => (if lsf { 6 } else { 8 }, 3),
        (true, false) => (0, 0),
        _ => (22, 13),
    };
    if lsf {
        info.scalefac_compress = rng.below(512) as u16;
        let (slen, case) = lsf_slen(&mut info, spec.intensity_right);
        let block = match (short, shape.mixed) {
            (true, true) => 2,
            (true, false) => 1,
            _ => 0,
        };
        for (part, &count) in LSF_PARTITIONS[case][block].iter().enumerate() {
            let bits = u32::from(slen[part]);
            for _ in 0..count {
                let value = if spec.intensity_right {
                    let largest = (1u64 << bits) - 1;
                    let limit = if spec.disputed.lsf_high_positions {
                        largest
                    } else {
                        largest.min(16)
                    };
                    let value = rng.below(limit);
                    if value == 7 { 6 } else { value }
                } else {
                    rng.below(1 << bits)
                };
                part2.push((value as u32, bits));
            }
        }
    } else {
        info.scalefac_compress = rng.below(16) as u16;
        let slen1 = u32::from(SLEN[0][usize::from(info.scalefac_compress)]);
        let slen2 = u32::from(SLEN[1][usize::from(info.scalefac_compress)]);
        info.preflag = !short && rng.chance(3);
        let narrow = spec.intensity_right && !spec.disputed.mpeg1_high_positions;
        let mut value = |bits: u32| {
            let limit = 1u64 << bits;
            let limit = if narrow { limit.min(8) } else { limit };
            (rng.below(limit) as u32, bits)
        };
        if short {
            for _ in 0..long_end {
                part2.push(value(slen1));
            }
            for band in short_start..12 {
                let bits = if band < 6 { slen1 } else { slen2 };
                for _ in 0..3 {
                    part2.push(value(bits));
                }
            }
        } else {
            const GROUPS: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
            for (group, (from, to)) in GROUPS.iter().copied().enumerate() {
                if spec.scfsi[group] {
                    continue;
                }
                let bits = if group < 2 { slen1 } else { slen2 };
                for _ in from..to {
                    part2.push(value(bits));
                }
            }
        }
    }

    // Spectrum.
    let mut ix = [0i32; 576];
    let big_values = rng.below(spec.max_pairs.min(288) + 1) as u16;
    info.big_values = big_values;
    let big_end = usize::from(big_values) * 2;
    let bounds = if shape.window_switching {
        let r0 = if short {
            if table == 8 { 72 } else { 36 }
        } else if !lsf {
            36
        } else if table == 8 {
            108
        } else {
            54
        };
        let r0 = r0.min(big_end);
        [(0, r0), (r0, big_end), (big_end, big_end)]
    } else {
        let (r0, r1) = long_bounds(table, info.region0_count, info.region1_count, big_end);
        [(0, r0), (r0, r1), (r1, big_end)]
    };
    for (region, (from, to)) in bounds.iter().copied().enumerate() {
        if region == 2 && shape.window_switching {
            break;
        }
        let select = if rng.chance(6) { 0 } else { random_table(rng) };
        info.table_select[region] = select;
        let max = table_max(select);
        for value in &mut ix[from..to] {
            if max > 0 && !rng.chance(3) {
                let magnitude = if rng.chance(4) {
                    rng.below(max as u64 + 1) as i32
                } else {
                    rng.below(max.min(3) as u64 + 1) as i32
                };
                *value = if rng.chance(2) { -magnitude } else { magnitude };
            }
        }
    }
    let quads = rng.below(((576 - big_end) / 4) as u64 + 1) as usize;
    let count1_end = big_end + 4 * quads;
    for value in &mut ix[big_end..count1_end] {
        *value = match rng.below(4) {
            0 => 1,
            1 => -1,
            _ => 0,
        };
    }

    // The intensity right channel: zeros above a bound, per window for
    // short blocks, so the decoder's bound search sees every case.
    if spec.intensity_right {
        let starts = short_starts(table);
        let widths = SHORT_WIDTHS[table];
        let short_line = if short {
            if shape.mixed {
                usize::from(long_starts(table)[long_end])
            } else {
                0
            }
        } else {
            576
        };
        let cut = rng.below(short_line as u64 + 1) as usize;
        ix[cut..short_line].fill(0);
        if short {
            for w in 0..3 {
                let from_band = short_start + rng.below((13 - short_start) as u64 + 1) as usize;
                for band in from_band..13 {
                    let width = usize::from(widths[band]);
                    let at = 3 * usize::from(starts[band]) + w * width;
                    ix[at..at + width].fill(0);
                }
            }
        }
    }

    Granule {
        info,
        part2,
        ix,
        bounds,
        big_end,
        count1_end,
    }
}

/// A random stream of `frames` standalone frames.
fn stream(
    seed: u64,
    version: Version,
    rate_index: u8,
    mode: Mode,
    frames: usize,
    disputed: Disputed,
) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    let allow_mixed = version != Version::Mpeg25 || disputed.mpeg25_mixed;
    // The block shape of each channel's last granule.
    let mut previous = [NORMAL; 2];
    for _ in 0..frames {
        let mode_extension = if mode == Mode::JointStereo {
            rng.below(4) as u8
        } else {
            0
        };
        let header = FrameHeader {
            version,
            layer: 3,
            protected: false,
            bitrate_index: 14,
            bitrate_kbps: 0,
            rate_index,
            sample_rate: 0,
            padding: false,
            private: false,
            mode,
            mode_extension,
            copyright: false,
            original: false,
            emphasis: 0,
        };
        let header = match FrameHeader::parse(header.to_bytes()) {
            Ok(parsed) => parsed,
            Err(error) => panic!("header: {error:?}"),
        };
        let Some(frame_bytes) = header.frame_bytes() else {
            panic!("free format");
        };
        let channels = header.channels();
        let granules = header.granules();
        let side_bytes = header.side_info_bytes();
        let budget = (frame_bytes - 4 - side_bytes) * 8 / (channels * granules);
        let intensity = mode == Mode::JointStereo && mode_extension & 1 != 0;
        let mut side = SideInfo::default();
        let mut coded: Vec<Vec<Granule>> = Vec::new();
        // MPEG-1 with long blocks in both granules, so `scfsi` can apply.
        let long_pair = !header.version.lsf()
            && previous[..channels]
                .iter()
                .all(|shape| shape.block_type == 0 || shape.block_type == 3)
            && rng.chance(2);
        for gr in 0..granules {
            let joint = random_shape(&mut rng, previous[0], allow_mixed);
            let mut row = Vec::new();
            for (ch, last) in previous.iter_mut().enumerate().take(channels) {
                let shape = if long_pair {
                    NORMAL
                } else if mode == Mode::JointStereo {
                    joint
                } else {
                    random_shape(&mut rng, *last, allow_mixed)
                };
                *last = shape;
                let scfsi = if gr == 1 && long_pair {
                    side.scfsi[ch]
                } else {
                    [false; 4]
                };
                let mut max_pairs = 288;
                let made = loop {
                    let spec = Spec {
                        header: &header,
                        shape,
                        intensity_right: intensity && ch == 1,
                        scfsi,
                        max_pairs,
                        disputed,
                    };
                    let made = granule(&mut rng, &spec);
                    if made.bits() <= budget.min(4095) {
                        break made;
                    }
                    max_pairs /= 2;
                };
                let reuse = !(intensity && ch == 1) || disputed.intensity_scfsi;
                if gr == 0 && long_pair && reuse {
                    side.scfsi[ch] = std::array::from_fn(|_| rng.chance(2));
                }
                row.push(made);
            }
            coded.push(row);
        }
        let mut main = BitWriter::new();
        for (gr, row) in coded.iter_mut().enumerate() {
            for (ch, made) in row.iter_mut().enumerate() {
                made.info.part2_3_length = made.bits() as u16;
                made.write(&mut main);
                side.granules[gr][ch] = made.info;
            }
        }
        let mut head = BitWriter::new();
        for byte in header.to_bytes() {
            head.put(u32::from(byte), 8);
        }
        side.write(&header, &mut head);
        let mut frame = head.finish();
        frame.extend(main.finish());
        assert!(
            frame.len() <= frame_bytes,
            "{} > {frame_bytes}",
            frame.len()
        );
        frame.resize(frame_bytes, 0);
        out.push(frame);
    }
    out.concat()
}

/// Split a stream of fixed-size frames and decode it with this crate.
fn decode(bytes: &[u8]) -> Vec<f32> {
    let mut params = CodecParams::audio(CodecId::Mp3, 0, ChannelLayout::stereo());
    params.layout = None;
    let mut decoder = match Mp3Decoder::open(&params) {
        Ok(decoder) => decoder,
        Err(error) => panic!("open: {error:?}"),
    };
    let mut pcm = Vec::new();
    let mut at = 0;
    let mut pts = 0;
    while at + 4 <= bytes.len() {
        let header =
            match FrameHeader::parse([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) {
                Ok(header) => header,
                Err(error) => panic!("header at {at}: {error:?}"),
            };
        let size = header.frame_bytes().unwrap_or(bytes.len() - at);
        let packet = Packet::audio(
            bytes[at..at + size].to_vec(),
            pts,
            header.samples() as i64,
            header.sample_rate,
        );
        pts += header.samples() as i64;
        at += size;
        let outputs = match decoder.push(&packet) {
            Ok(outputs) => outputs,
            Err(error) => panic!("decode: {error:?}"),
        };
        for output in outputs {
            if let Output::Audio(audio) = output {
                pcm.extend_from_slice(audio.pcm.as_f32().unwrap_or(&[]));
            }
        }
    }
    pcm
}

fn tool(name: &str) -> Option<PathBuf> {
    ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
        .iter()
        .map(|dir| Path::new(dir).join(name))
        .find(|path| path.exists())
}

/// Decode with ffmpeg or mpg123 to interleaved `f32`.
fn reference(name: &str, path: &Path, bytes: &[u8], label: &str) -> Vec<f32> {
    static RUN: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutlass-mp3-synthetic-{label}-{name}-{}-{}",
        std::process::id(),
        RUN.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::create_dir_all(&dir);
    let input = dir.join("in.mp3");
    if let Err(error) = std::fs::write(&input, bytes) {
        panic!("write: {error}");
    }
    let output = if name == "ffmpeg" {
        Command::new(path)
            .args(["-v", "error", "-i"])
            .arg(&input)
            .args(["-f", "f32le", "-"])
            .output()
    } else {
        Command::new(path)
            .args(["-q", "-e", "f32", "-s"])
            .arg(&input)
            .output()
    };
    let _ = std::fs::remove_dir_all(&dir);
    let output = match output {
        Ok(output) => output,
        Err(error) => panic!("{name}: {error}"),
    };
    assert!(output.status.success(), "{name} {label}: {output:?}");
    output
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn difference(ours: &[f32], reference: &[f32]) -> (f64, f64, usize) {
    let mut sum = 0f64;
    let mut max = 0f64;
    let mut worst = 0;
    for (i, (a, b)) in ours.iter().zip(reference).enumerate() {
        let d = (f64::from(*a) - f64::from(*b)).abs();
        sum += d * d;
        if d > max {
            max = d;
            worst = i;
        }
    }
    ((sum / ours.len().max(1) as f64).sqrt(), max, worst)
}

const RATES: [(Version, u8, &str); 9] = [
    (Version::Mpeg1, 0, "44100"),
    (Version::Mpeg1, 1, "48000"),
    (Version::Mpeg1, 2, "32000"),
    (Version::Mpeg2, 0, "22050"),
    (Version::Mpeg2, 1, "24000"),
    (Version::Mpeg2, 2, "16000"),
    (Version::Mpeg25, 0, "11025"),
    (Version::Mpeg25, 1, "12000"),
    (Version::Mpeg25, 2, "8000"),
];

const MODES: [(Mode, &str); 4] = [
    (Mode::JointStereo, "joint"),
    (Mode::Stereo, "stereo"),
    (Mode::Mono, "mono"),
    (Mode::DualChannel, "dual"),
];

fn seed(rate: usize, mode: usize) -> u64 {
    rate as u64 * 7 + mode as u64 + 1
}

#[test]
fn random_streams_decode_without_error() {
    let disputed = Disputed {
        lsf_high_positions: true,
        mpeg1_high_positions: true,
        intensity_scfsi: true,
        mpeg25_mixed: true,
    };
    for (r, (version, rate_index, rate)) in RATES.iter().copied().enumerate() {
        for (m, (mode, name)) in MODES.iter().copied().enumerate() {
            let bytes = stream(seed(r, m), version, rate_index, mode, 40, disputed);
            let pcm = decode(&bytes);
            assert!(!pcm.is_empty(), "{rate} {name}");
            assert!(pcm.iter().all(|s| s.is_finite()), "{rate} {name}");
        }
    }
}

/// Decode streams of every rate and `modes` with this crate and the named
/// references, and require the ISO/IEC 11172-4 bounds. A missing reference
/// is skipped with a note.
fn check(names: &[&'static str], versions: &[Version], modes: &[(Mode, &str)], disputed: Disputed) {
    let oracles: Vec<(&str, PathBuf)> = names
        .iter()
        .filter_map(|&name| match tool(name) {
            Some(path) => Some((name, path)),
            None => {
                eprintln!("skipped the {name} reference: {name} was not found");
                None
            }
        })
        .collect();
    let mut failures = Vec::new();
    for (r, (version, rate_index, rate)) in RATES.iter().copied().enumerate() {
        if !versions.contains(&version) {
            continue;
        }
        for (m, (mode, name)) in modes.iter().copied().enumerate() {
            let label = format!("{rate}-{name}");
            let bytes = stream(seed(r, m), version, rate_index, mode, 40, disputed);
            let ours = decode(&bytes);
            for (oracle, path) in &oracles {
                let theirs = reference(oracle, path, &bytes, &label);
                let (rms, max, worst) = difference(&ours, &theirs);
                if ours.len() != theirs.len() || rms >= ISO_RMS || max > ISO_MAX {
                    failures.push(format!(
                        "{label} vs {oracle}: {} vs {} samples, rms {rms:e}, max {max:e} at {worst}",
                        ours.len(),
                        theirs.len()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

const ALL: [Version; 3] = [Version::Mpeg1, Version::Mpeg2, Version::Mpeg25];

#[test]
fn random_streams_match_ffmpeg_and_mpg123() {
    check(&["ffmpeg", "mpg123"], &ALL, &MODES, Disputed::default());
}

#[test]
fn mpeg1_positions_above_7_intensity_scfsi_and_mpeg25_mixed_blocks_match_ffmpeg() {
    let disputed = Disputed {
        mpeg1_high_positions: true,
        intensity_scfsi: true,
        mpeg25_mixed: true,
        ..Disputed::default()
    };
    check(&["ffmpeg"], &ALL, &MODES, disputed);
}

#[test]
fn lsf_intensity_positions_from_16_match_mpg123() {
    let disputed = Disputed {
        lsf_high_positions: true,
        ..Disputed::default()
    };
    check(
        &["mpg123"],
        &[Version::Mpeg2, Version::Mpeg25],
        &MODES[..1],
        disputed,
    );
}
