//! Decode streams from three encoders and compare with ffmpeg's decoder.
//!
//! Gates: every Y, Cb and Cr sample within 1 of ffmpeg's and alpha
//! identical after ffmpeg's own scaling to the output depth. The first gate
//! also capped the share of samples off by 1 at 1%; ffmpeg's int16 IDCT
//! misses that against Apple's own decoder too (about 2.5% at 10 bits, 14%
//! with a few 2s at 12 bits), so the share gate is now relative: on a Mac,
//! against VideoToolbox's decode, this crate may be off in no more samples
//! than ffmpeg is, and by no more than 1 or ffmpeg's own largest miss. Against ffmpeg the share stays under 5%
//! at 10 bits and 25% at 12. The multi-thread decode must equal the
//! single-thread decode byte for byte.

use std::path::Path;

use cutlass_decode::{CodecId, CodecParams, PixelFormat};
use cutlass_prores::{AlphaBits, ProresDecoder};

use crate::support::{Diff, HD, Scratch, UHD, ffmpeg, frames, local, reference, run, samples16};

const ALPHA_RAMP: &str =
    "format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(X,W/2),255,255*Y/H)'";
const ALPHA_FINE: &str =
    "format=rgba64le,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='mod(X*977+Y*331,65536)'";

struct Case {
    name: &'static str,
    source: &'static str,
    frames: u32,
    filter: Option<&'static str>,
    args: &'static [&'static str],
    /// ffmpeg's native output for the stream, so no conversion runs.
    reference: &'static str,
}

const fn case(
    name: &'static str,
    filter: Option<&'static str>,
    args: &'static [&'static str],
    reference: &'static str,
) -> Case {
    Case {
        name,
        source: HD,
        frames: 2,
        filter,
        args,
        reference,
    }
}

const P422: &str = "yuv422p10le";
const P444: &str = "yuv444p12le";
const P444A: &str = "yuva444p12le";

const KOSTYA: &[Case] = &[
    case(
        "ks proxy",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "0",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks lt",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "1",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks 422",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "2",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks hq",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks 4444",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuv444p10le",
        ],
        P444,
    ),
    case(
        "ks xq",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "5",
            "-pix_fmt",
            "yuv444p10le",
        ],
        P444,
    ),
    case(
        "ks 4444 alpha 8",
        Some(ALPHA_RAMP),
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "8",
        ],
        P444A,
    ),
    case(
        "ks 4444 alpha 16",
        Some(ALPHA_FINE),
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "16",
        ],
        P444A,
    ),
    case(
        "ks xq alpha 16",
        Some(ALPHA_FINE),
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "5",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "16",
        ],
        P444A,
    ),
    case(
        "ks hq interlaced",
        None,
        &[
            "-flags",
            "+ildct",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks hq interlaced tff",
        Some("setfield=tff"),
        &[
            "-flags",
            "+ildct",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks 4444 interlaced alpha",
        Some(ALPHA_RAMP),
        &[
            "-flags",
            "+ildct",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "8",
        ],
        P444A,
    ),
    case(
        "ks hq 1917x1077",
        Some("scale=1917:1077"),
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks 4444 33x17 alpha",
        Some("scale=33:17,format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='X*7+Y*3'"),
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "8",
        ],
        P444A,
    ),
    case(
        "ks hq interlaced 721x487",
        Some("scale=721:487"),
        &[
            "-flags",
            "+ildct",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks hq 2 mb slices",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-mbs_per_slice",
            "2",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks hq 1 mb slices",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-mbs_per_slice",
            "1",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks proxy 4 mb slices",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "0",
            "-mbs_per_slice",
            "4",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks 422 default matrix",
        None,
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "2",
            "-quant_mat",
            "default",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "ks lt hdr tags",
        Some("setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc"),
        &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "1",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
];

const ANATOLIY: &[Case] = &[
    case(
        "aw proxy",
        None,
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "0",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "aw lt",
        None,
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "1",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "aw 422",
        None,
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "2",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "aw hq",
        None,
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "aw 4444",
        None,
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuv444p10le",
        ],
        P444,
    ),
    case(
        "aw xq",
        None,
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "5",
            "-pix_fmt",
            "yuv444p10le",
        ],
        P444,
    ),
    case(
        "aw 4444 alpha",
        Some(ALPHA_FINE),
        &[
            "-c:v",
            "prores_aw",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
        ],
        P444A,
    ),
    case(
        "aw hq interlaced",
        None,
        &[
            "-flags",
            "+ildct",
            "-c:v",
            "prores_aw",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
];

const TOOLBOX: &[Case] = &[
    case(
        "vt proxy",
        None,
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "proxy",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "vt lt",
        None,
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "lt",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "vt 422",
        None,
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "standard",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "vt hq",
        None,
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "hq",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
    case(
        "vt 4444",
        None,
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuv444p10le",
        ],
        P444,
    ),
    case(
        "vt xq",
        None,
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "xq",
            "-pix_fmt",
            "yuv444p10le",
        ],
        P444,
    ),
    case(
        "vt 4444 alpha",
        Some(ALPHA_RAMP),
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p16le",
        ],
        P444A,
    ),
    case(
        "vt 4444 alpha 1917x1077",
        Some("scale=1917:1077,format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='X*255/W'"),
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p16le",
        ],
        P444A,
    ),
    case(
        "vt hq 1917x1077",
        Some("scale=1917:1077"),
        &[
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "hq",
            "-pix_fmt",
            "yuv422p10le",
        ],
        P422,
    ),
];

fn make(ffmpeg: &Path, case: &Case, scratch: &Scratch) -> Option<std::path::PathBuf> {
    let source = local(case.source)?;
    let out = scratch.path("stream.mov");
    let frames = case.frames.to_string();
    let mut args = vec![
        "-v",
        "error",
        "-y",
        "-i",
        source.to_str()?,
        "-frames:v",
        &frames,
    ];
    if let Some(filter) = case.filter {
        args.extend(["-vf", filter]);
    }
    args.extend_from_slice(case.args);
    args.push(out.to_str()?);
    run(ffmpeg, &args).then_some(out)
}

/// ffmpeg's alpha scaling to `depth` bits.
fn scale_alpha(value: u16, bits: u8, depth: u8) -> u16 {
    match (bits, depth) {
        (16, 12) => value >> 4,
        (16, _) => value >> 6,
        (_, 12) => (value << 4) | (value >> 4),
        _ => (value << 2) | (value >> 6),
    }
}

/// Decodes every case, prints the comparison, and returns the failures.
fn compare(cases: &[Case], ffmpeg: &Path, required: bool) -> Vec<String> {
    let mut failures = Vec::new();
    for case in cases {
        let scratch = Scratch::new("oracle");
        let Some(movie) = make(ffmpeg, case, &scratch) else {
            if required {
                failures.push(format!("{}: ffmpeg could not make the stream", case.name));
            } else {
                eprintln!("{}: skipped, the encoder is not available", case.name);
            }
            continue;
        };
        let coded = frames(ffmpeg, &movie, &scratch);
        let raw = reference(ffmpeg, &movie, case.reference, &scratch, false);
        let apple = cfg!(target_vendor = "apple")
            .then(|| reference(ffmpeg, &movie, case.reference, &scratch, true));
        assert!(!coded.is_empty(), "{}: no frames", case.name);
        let frame_len = raw.len() / coded.len();
        let params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
        let mut parallel = ProresDecoder::open(&params).expect("open");
        let mut single = ProresDecoder::open(&params).expect("open").with_threads(1);
        let mut worst = [Diff::default(); 4];
        // Against VideoToolbox: this crate, then ffmpeg.
        let mut toolbox = [Diff::default(); 2];
        for (index, data) in coded.iter().enumerate() {
            let ours = parallel.decode(data).expect(case.name);
            let alone = single.decode(data).expect(case.name);
            assert_eq!(
                ours.frame, alone.frame,
                "{}: threads changed the picture",
                case.name
            );
            let frame = &ours.frame;
            let depth = if matches!(
                frame.format,
                PixelFormat::Yuv422p10 | PixelFormat::Yuv444p10
            ) {
                10
            } else {
                12
            };
            let peak = f64::from((1u32 << depth) - 1);
            let reference = samples16(&raw[index * frame_len..(index + 1) * frame_len]);
            let mut ours_color = Vec::new();
            let mut at = 0;
            for (slot, plane) in [&frame.y, &frame.u, &frame.v].into_iter().enumerate() {
                let n = (plane.width * plane.height) as usize;
                let test = samples16(&plane.samples);
                let diff = Diff::between(&test, &reference[at..at + n], peak);
                at += n;
                if diff.max >= worst[slot].max || diff.psnr < worst[slot].psnr {
                    worst[slot] = diff;
                }
                ours_color.extend(test);
            }
            if let Some(apple) = &apple {
                let vt = samples16(&apple[index * frame_len..(index + 1) * frame_len]);
                let color = ours_color.len();
                for (slot, test) in [&ours_color[..], &reference[..color]]
                    .into_iter()
                    .enumerate()
                {
                    let diff = Diff::between(test, &vt[..color], peak);
                    if diff.mismatched >= toolbox[slot].mismatched {
                        toolbox[slot] = diff;
                    }
                }
            }
            if let Some(alpha) = &frame.alpha {
                let bits = alpha.bit_depth;
                let test: Vec<u16> = if bits > 8 {
                    samples16(&alpha.plane.samples)
                } else {
                    alpha.plane.samples.iter().map(|&b| u16::from(b)).collect()
                };
                let test: Vec<u16> = test.iter().map(|&a| scale_alpha(a, bits, depth)).collect();
                let n = test.len();
                worst[3] = Diff::between(&test, &reference[at..at + n], peak);
                assert_ne!(ours.header.alpha, AlphaBits::None);
            }
        }
        let header = parallel.decode(&coded[0]).expect("header").header;
        eprintln!(
            "{:<28} {}x{} {:?} {:?} {}\n    Y {}\n    U {}\n    V {}{}{}",
            case.name,
            header.width,
            header.height,
            header.chroma,
            header.interlace,
            String::from_utf8_lossy(&header.encoder),
            worst[0],
            worst[1],
            worst[2],
            if header.alpha == AlphaBits::None {
                String::new()
            } else {
                format!("\n    A {}", worst[3])
            },
            if apple.is_some() {
                format!(
                    "\n    vs VideoToolbox: ours {}\n                     ffmpeg {}",
                    toolbox[0], toolbox[1]
                )
            } else {
                String::new()
            },
        );
        let depth12 =
            header.chroma == cutlass_prores::Chroma::Yuv444 || header.alpha != AlphaBits::None;
        let cap = if depth12 { 0.25 } else { 0.05 };
        for (slot, diff) in worst.iter().enumerate().take(3) {
            let share = diff.mismatched as f64 / diff.count.max(1) as f64;
            if diff.max > 1 || share > cap {
                failures.push(format!("{} plane {slot}: {diff}", case.name));
            }
        }
        // When ffmpeg equals VideoToolbox exactly, the hardware path fell back
        // to ffmpeg's own decoder (tiny frames do), so there is nothing to learn.
        let fell_back = toolbox[1].mismatched == 0;
        let ours = toolbox[0];
        let theirs = toolbox[1];
        if apple.is_some()
            && !fell_back
            && (ours.max > theirs.max.max(1) || ours.mismatched > theirs.mismatched)
        {
            failures.push(format!(
                "{} vs VideoToolbox: ours {} ffmpeg {}",
                case.name, toolbox[0], toolbox[1]
            ));
        }
        if header.alpha != AlphaBits::None && worst[3].max != 0 {
            failures.push(format!("{} alpha: {}", case.name, worst[3]));
        }
    }
    failures
}

#[test]
fn prores_ks_streams_match_ffmpeg() {
    let Some(ffmpeg) = ffmpeg() else { return };
    if local(HD).is_none() {
        return;
    }
    let failures = compare(KOSTYA, &ffmpeg, true);
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn prores_aw_streams_match_ffmpeg() {
    let Some(ffmpeg) = ffmpeg() else { return };
    if local(HD).is_none() {
        return;
    }
    let failures = compare(ANATOLIY, &ffmpeg, true);
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn videotoolbox_streams_match_ffmpeg() {
    let Some(ffmpeg) = ffmpeg() else { return };
    if local(HD).is_none() {
        return;
    }
    let failures = compare(TOOLBOX, &ffmpeg, cfg!(target_vendor = "apple"));
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_4k_stream_matches_ffmpeg() {
    let Some(ffmpeg) = ffmpeg() else { return };
    if local(UHD).is_none() {
        return;
    }
    let cases = [
        Case {
            source: UHD,
            frames: 1,
            ..case(
                "ks hq 4k",
                None,
                &[
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "3",
                    "-pix_fmt",
                    "yuv422p10le",
                ],
                P422,
            )
        },
        Case {
            source: UHD,
            frames: 1,
            ..case(
                "ks 4444 4k",
                None,
                &[
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "4",
                    "-pix_fmt",
                    "yuv444p10le",
                ],
                P444,
            )
        },
    ];
    let failures = compare(&cases, &ffmpeg, true);
    assert!(failures.is_empty(), "{failures:#?}");
}
