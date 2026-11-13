//! x265 streams decoded here and by ffmpeg: every picture must match
//! exactly, in output order. Each test skips with a message when ffmpeg or
//! its libx265 encoder is missing.

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod oracle {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};
    use cutlass_hevc::HevcDecoder;

    fn ffmpeg() -> Option<PathBuf> {
        let found = Command::new("ffmpeg").arg("-version").output().ok()?;
        found.status.success().then(|| PathBuf::from("ffmpeg"))
    }

    fn has_x265(ff: &Path) -> bool {
        Command::new(ff)
            .args(["-hide_banner", "-encoders"])
            .output()
            .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("libx265"))
    }

    /// Annex B NAL units as (type, bytes without the start code).
    fn nal_units(data: &[u8]) -> Vec<&[u8]> {
        let mut starts = Vec::new();
        let mut i = 0;
        while i + 3 <= data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                starts.push(i + 3);
                i += 3;
            } else {
                i += 1;
            }
        }
        let mut out = Vec::new();
        for (n, &start) in starts.iter().enumerate() {
            let mut end = starts.get(n + 1).map_or(data.len(), |&next| next - 3);
            while end > start && data[end - 1] == 0 {
                end -= 1;
            }
            out.push(&data[start..end]);
        }
        out
    }

    /// Split an Annex B stream into access units (clause 7.4.2.4.4).
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

    /// Annex B to four-byte length prefixes.
    fn length_prefixed(unit: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nal_units(unit) {
            out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            out.extend_from_slice(nal);
        }
        out
    }

    pub struct Case<'a> {
        pub name: &'a str,
        pub size: (usize, usize),
        pub frames: usize,
        /// 8, 10, or 12.
        pub bits: u8,
        pub params: &'a str,
        pub source: &'a str,
        /// Filters after the source, such as a fade for weighted prediction.
        pub filter: &'a str,
    }

    fn work_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cutlass-hevc-oracle-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// Encode with libx265 and decode with ffmpeg. `None` when the encoder is
    /// missing.
    fn encode(ff: &Path, case: &Case<'_>, tag: &str) -> Option<(Vec<u8>, Vec<u8>)> {
        let dir = work_dir();
        let hevc = dir.join(format!("{tag}.hevc"));
        let yuv = dir.join(format!("{tag}.yuv"));
        let pix = match case.bits {
            10 => "yuv420p10le",
            12 => "yuv420p12le",
            _ => "yuv420p",
        };
        let (w, h) = case.size;
        let encoded = Command::new(ff)
            .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
            .arg(if case.filter.is_empty() {
                format!("{}=size={w}x{h}:rate=30", case.source)
            } else {
                format!("{}=size={w}x{h}:rate=30,{}", case.source, case.filter)
            })
            .args([
                "-frames:v",
                &case.frames.to_string(),
                "-pix_fmt",
                pix,
                "-c:v",
                "libx265",
            ])
            .args(["-x265-params", case.params, "-f", "hevc"])
            .arg(&hevc)
            .output()
            .ok()?;
        if !encoded.status.success() {
            panic!(
                "{}: x265 refused the parameters: {}",
                case.name,
                String::from_utf8_lossy(&encoded.stderr)
            );
        }
        let decoded = Command::new(ff)
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&hevc)
            .args(["-f", "rawvideo", "-pix_fmt", pix])
            .arg(&yuv)
            .status()
            .ok()?;
        assert!(decoded.success(), "{}: ffmpeg did not decode", case.name);
        Some((std::fs::read(&hevc).ok()?, std::fs::read(&yuv).ok()?))
    }

    fn decode_all(bytes: &[u8], size: (usize, usize), prefixed: bool) -> Vec<VideoOutput> {
        let mut params = CodecParams::video(CodecId::Hevc, size.0 as u32, size.1 as u32);
        params.nal_length_size = if prefixed { 4 } else { 0 };
        let mut decoder = HevcDecoder::open(&params).expect("open");
        let mut got = Vec::new();
        for (n, unit) in access_units(bytes).iter().enumerate() {
            let data = if prefixed {
                length_prefixed(unit)
            } else {
                unit.clone()
            };
            let pictures = decoder
                .push(&Packet::video(data, n as i64, 1, 30, n == 0))
                .unwrap_or_else(|err| panic!("access unit {n}: {err:?}"));
            got.extend(pictures);
        }
        got.extend(decoder.drain().expect("drain"));
        got
    }

    /// Every picture of `bytes` decoded with `threads` workers: the 8-bit
    /// planes and, above 8 bits, the full samples.
    fn decode_with_threads(bytes: &[u8], threads: usize) -> Vec<Vec<Vec<u16>>> {
        decode_with_split(bytes, threads, None)
    }

    /// Decoded with `threads`, or with `split` pictures at once and threads
    /// for the rows of each.
    fn decode_with_split(
        bytes: &[u8],
        threads: usize,
        split: Option<(usize, usize)>,
    ) -> Vec<Vec<Vec<u16>>> {
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.nal_length_size = 0;
        let mut decoder = HevcDecoder::open_with_threads(&params, threads).expect("open");
        if let Some((frames, rows)) = split {
            decoder.set_threads(frames, rows);
        }
        let mut got = Vec::new();
        for (n, unit) in access_units(bytes).iter().enumerate() {
            let pictures = decoder
                .push(&Packet::video(unit.clone(), n as i64, 1, 30, n == 0))
                .unwrap_or_else(|err| panic!("access unit {n}: {err:?}"));
            got.extend(pictures);
        }
        got.extend(decoder.drain().expect("drain"));
        got.iter()
            .map(|frame| {
                let p = &frame.picture;
                let mut planes: Vec<Vec<u16>> = [p.y(), p.u(), p.v()]
                    .iter()
                    .map(|plane| plane.iter().map(|&b| u16::from(b)).collect())
                    .collect();
                if let Some(p) = &frame.yuv420p10 {
                    planes.extend([p.y().to_vec(), p.u().to_vec(), p.v().to_vec()]);
                }
                if let Some(p) = &frame.yuv420_high {
                    planes.extend([p.y().to_vec(), p.u().to_vec(), p.v().to_vec()]);
                }
                planes
            })
            .collect()
    }

    /// The case matches ffmpeg on the process pool, and two, five and
    /// sixteen workers return exactly what one returns.
    fn check_threads(case: &Case<'_>) {
        if !check(case) {
            return;
        }
        let Some(ff) = ffmpeg() else {
            return;
        };
        let tag: String = case
            .name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let Some((bytes, _)) = encode(&ff, case, &tag) else {
            return;
        };
        let one = decode_with_threads(&bytes, 1);
        assert!(!one.is_empty(), "{}: no pictures", case.name);
        // The default splits, then rows of a picture on several threads.
        let splits = [
            (2, None),
            (5, None),
            (16, None),
            (3, Some((1, 3))),
            (4, Some((2, 2))),
            (8, Some((2, 4))),
        ];
        for (threads, split) in splits {
            let many = decode_with_split(&bytes, threads, split);
            assert_eq!(many.len(), one.len(), "{}: picture count", case.name);
            for (index, (a, b)) in one.iter().zip(&many).enumerate() {
                assert!(
                    a == b,
                    "{}: picture {index} differs with {threads} workers ({split:?})",
                    case.name
                );
            }
        }
    }

    fn first_difference(
        name: &str,
        frame: usize,
        plane: &str,
        width: usize,
        got: &[u16],
        want: &[u16],
    ) {
        assert_eq!(
            got.len(),
            want.len(),
            "{name} frame {frame} {plane}: plane size"
        );
        if let Ok(dir) = std::env::var("HEVC_DUMP") {
            let bytes = |v: &[u16]| v.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>();
            let _ = std::fs::write(format!("{dir}/{frame}-{plane}-got.raw"), bytes(got));
            let _ = std::fs::write(format!("{dir}/{frame}-{plane}-want.raw"), bytes(want));
        }
        if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
            let wrong = got.iter().zip(want).filter(|(a, b)| a != b).count();
            panic!(
                "{name} frame {frame} {plane}: first difference at ({}, {}): {} vs ffmpeg {}, {wrong} samples differ",
                at % width,
                at / width,
                got[at],
                want[at]
            );
        }
    }

    /// Every picture equal to ffmpeg's, in output order.
    pub fn check(case: &Case<'_>) -> bool {
        let Some(ff) = ffmpeg() else {
            eprintln!("skipped: no ffmpeg");
            return false;
        };
        if !has_x265(&ff) {
            eprintln!("skipped: ffmpeg has no libx265");
            return false;
        }
        let tag: String = case
            .name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let Some((bytes, reference)) = encode(&ff, case, &tag) else {
            eprintln!("skipped: could not run ffmpeg");
            return false;
        };
        let (w, h) = case.size;
        let sample = if case.bits > 8 { 2 } else { 1 };
        let frame_len = w * h * 3 / 2 * sample;
        let want_frames = reference.len() / frame_len;
        for prefixed in [false, true] {
            let got = decode_all(&bytes, case.size, prefixed);
            assert_eq!(got.len(), want_frames, "{}: picture count", case.name);
            for (index, picture) in got.iter().enumerate() {
                let at = &reference[index * frame_len..(index + 1) * frame_len];
                let want: Vec<u16> = if case.bits > 8 {
                    at.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| u16::from_le_bytes(b))
                        .collect()
                } else {
                    at.iter().map(|&b| u16::from(b)).collect()
                };
                let (gy, gu, gv): (Vec<u16>, Vec<u16>, Vec<u16>) = if case.bits == 10 {
                    let p = picture.yuv420p10.as_ref().expect("10-bit output");
                    (p.y().to_vec(), p.u().to_vec(), p.v().to_vec())
                } else if case.bits > 8 {
                    let p = picture.yuv420_high.as_ref().expect("high-bit-depth output");
                    assert_eq!(p.bit_depth(), case.bits);
                    (p.y().to_vec(), p.u().to_vec(), p.v().to_vec())
                } else {
                    let p = &picture.picture;
                    (
                        p.y().iter().map(|&b| u16::from(b)).collect(),
                        p.u().iter().map(|&b| u16::from(b)).collect(),
                        p.v().iter().map(|&b| u16::from(b)).collect(),
                    )
                };
                let (ly, lc) = (w * h, (w / 2) * (h / 2));
                first_difference(case.name, index, "y", w, &gy, &want[..ly]);
                first_difference(case.name, index, "u", w / 2, &gu, &want[ly..ly + lc]);
                first_difference(case.name, index, "v", w / 2, &gv, &want[ly + lc..]);
                if case.bits > 8 {
                    let shift = case.bits - 8;
                    let eight: Vec<u8> = gy.iter().map(|&s| (s >> shift) as u8).collect();
                    assert_eq!(picture.picture.y(), &eight[..], "{}: 8-bit view", case.name);
                }
            }
        }
        true
    }

    const BASE: &str = "log-level=error:keyint=30:min-keyint=30:scenecut=0:wpp=0";

    fn case<'a>(name: &'a str, params: &'a str) -> Case<'a> {
        Case {
            name,
            size: (208, 120),
            frames: 10,
            bits: 8,
            params,
            source: "testsrc2",
            filter: "",
        }
    }

    fn run(name: &str, extra: &str) {
        let params = format!("{BASE}:{extra}");
        check(&case(name, &params));
    }

    #[test]
    fn intra_only() {
        run("intra", "keyint=1:sao=0:deblock=0");
    }

    #[test]
    fn intra_with_loop_filters() {
        run("intra filters", "keyint=1");
    }

    #[test]
    fn p_frames_with_several_references() {
        run("p refs", "bframes=0:ref=3");
    }

    #[test]
    fn b_frames_with_pyramid() {
        run("b pyramid", "bframes=4:b-pyramid=1:ref=4");
    }

    #[test]
    fn amp_and_rectangular_partitions() {
        run("amp", "rect=1:amp=1:bframes=2");
    }

    #[test]
    fn temporal_mvp_off() {
        run("no tmvp", "temporal-mvp=0:bframes=3");
    }

    #[test]
    fn weighted_prediction() {
        let params = format!("{BASE}:weightp=1:weightb=1:bframes=3");
        check(&Case {
            filter: "fade=in:0:8",
            ..case("weighted", &params)
        });
    }

    #[test]
    fn open_gop_with_leading_pictures() {
        let params = format!("{BASE}:keyint=8:min-keyint=8:open-gop=1:bframes=3:ref=3");
        check(&Case {
            frames: 26,
            ..case("open gop", &params)
        });
    }

    #[test]
    fn closed_gop_with_idr_refresh() {
        let params = format!("{BASE}:keyint=8:min-keyint=8:open-gop=0:bframes=3:ref=3");
        check(&Case {
            frames: 26,
            ..case("closed gop", &params)
        });
    }

    #[test]
    fn many_references_and_merge_candidates() {
        run(
            "many refs",
            "ref=6:bframes=5:b-adapt=2:rect=1:max-merge=5:subme=7:me=star",
        );
    }

    #[test]
    fn sao_off() {
        run("sao off", "sao=0:bframes=2");
    }

    #[test]
    fn deblocking_offsets() {
        run("deblock offsets", "deblock=-3,4:bframes=2");
    }

    #[test]
    fn deblocking_off() {
        run("deblock off", "deblock=0:bframes=2");
    }

    #[test]
    fn wavefronts() {
        run("wpp", "wpp=1:bframes=2");
    }

    #[test]
    fn several_slices() {
        run("slices", "wpp=1:slices=3:bframes=2");
    }

    #[test]
    fn transform_skip_and_rdoq() {
        run("transform skip", "tskip=1:rdoq-level=2:bframes=2");
    }

    #[test]
    fn scaling_lists() {
        run("scaling lists", "scaling-list=default:bframes=2");
    }

    #[test]
    fn lossless() {
        run("lossless", "lossless=1:bframes=2");
    }

    #[test]
    fn constrained_intra_and_sign_hiding() {
        run(
            "constrained intra",
            "constrained-intra=1:signhide=1:bframes=2",
        );
    }

    /// SAO is off: ffmpeg 8.1 runs chroma SAO of a 16×16 CTB before the
    /// horizontal chroma edges to its right are deblocked and so differs from
    /// the standard there. `small_ctbs_with_sao_match_x265` covers SAO.
    #[test]
    fn smallest_blocks_and_deep_transform_tree() {
        run(
            "small ctu",
            "ctu=16:min-cu-size=8:tu-intra-depth=3:tu-inter-depth=3:bframes=2:sao=0",
        );
    }

    /// 16×16 CTBs with deblocking and SAO, against the x265 command line's own
    /// reconstruction (`--recon`), which follows the standard where ffmpeg 8.1
    /// does not.
    #[test]
    fn small_ctbs_with_sao_match_x265() {
        let Some(ff) = ffmpeg() else {
            eprintln!("skipped: no ffmpeg");
            return;
        };
        let has_cli = Command::new("x265")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !has_cli {
            eprintln!("skipped: no x265 command");
            return;
        }
        let dir = work_dir();
        let (w, h, frames) = (208usize, 120usize, 10usize);
        let src = dir.join("ctu16-src.yuv");
        let recon = dir.join("ctu16-recon.yuv");
        let hevc = dir.join("ctu16.hevc");
        let made = Command::new(&ff)
            .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size={w}x{h}:rate=30"))
            .args(["-frames:v", &frames.to_string(), "-pix_fmt", "yuv420p"])
            .arg(&src)
            .status()
            .expect("ffmpeg");
        assert!(made.success());
        let encoded = Command::new("x265")
            .args(["--log-level", "error", "--input"])
            .arg(&src)
            .args([
                "--input-res",
                &format!("{w}x{h}"),
                "--fps",
                "30",
                "--frames",
                &frames.to_string(),
            ])
            .args([
                "--ctu",
                "16",
                "--min-cu-size",
                "8",
                "--bframes",
                "2",
                "--keyint",
                "30",
                "--no-wpp",
            ])
            .args(["--tu-intra-depth", "3", "--tu-inter-depth", "3", "--recon"])
            .arg(&recon)
            .arg("-o")
            .arg(&hevc)
            .output()
            .expect("x265");
        assert!(
            encoded.status.success(),
            "x265: {}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        let bytes = std::fs::read(&hevc).expect("stream");
        let reference = std::fs::read(&recon).expect("recon");
        let got = decode_all(&bytes, (w, h), false);
        let frame_len = w * h * 3 / 2;
        assert_eq!(got.len(), reference.len() / frame_len, "picture count");
        for (index, picture) in got.iter().enumerate() {
            let want: Vec<u16> = reference[index * frame_len..(index + 1) * frame_len]
                .iter()
                .map(|&b| u16::from(b))
                .collect();
            let p = &picture.picture;
            let to16 = |v: &[u8]| v.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
            let (ly, lc) = (w * h, (w / 2) * (h / 2));
            first_difference("ctu16 sao", index, "y", w, &to16(p.y()), &want[..ly]);
            first_difference(
                "ctu16 sao",
                index,
                "u",
                w / 2,
                &to16(p.u()),
                &want[ly..ly + lc],
            );
            first_difference(
                "ctu16 sao",
                index,
                "v",
                w / 2,
                &to16(p.v()),
                &want[ly + lc..],
            );
        }
    }

    #[test]
    fn main_ten() {
        let params = format!("{BASE}:bframes=3:ref=3");
        check(&Case {
            bits: 10,
            ..case("main10", &params)
        });
    }

    /// Wavefronts over many CTB rows, so rows decode on several workers at
    /// once: 32×32 CTBs in 640×352 make 11 rows of 20.
    #[test]
    fn wavefront_rows_on_several_workers() {
        let params = format!("{BASE}:wpp=1:ctu=32:bframes=3:ref=3");
        check_threads(&Case {
            size: (640, 352),
            frames: 12,
            ..case("threads wpp", &params)
        });
    }

    /// Wavefronts in four slices, each starting on a row, at 10 bits.
    #[test]
    fn wavefront_slices_on_several_workers() {
        let params = format!("{BASE}:wpp=1:ctu=32:slices=4:bframes=2");
        check_threads(&Case {
            size: (640, 352),
            frames: 12,
            bits: 10,
            ..case("threads wpp slices", &params)
        });
    }

    /// Without wavefronts only the loop filters spread over the workers.
    #[test]
    fn loop_filters_on_several_workers() {
        let params = format!("{BASE}:ctu=32:bframes=2");
        check_threads(&Case {
            size: (640, 352),
            frames: 8,
            ..case("threads filters", &params)
        });
    }

    /// Main 12, a format-range-extensions profile with none of its tools on.
    #[test]
    fn main_twelve() {
        let params = format!("{BASE}:bframes=3:ref=3");
        check(&Case {
            bits: 12,
            ..case("main12", &params)
        });
    }

    #[test]
    fn main_ten_intra() {
        let params = format!("{BASE}:keyint=1");
        check(&Case {
            bits: 10,
            ..case("main10 intra", &params)
        });
    }

    /// A case named by the environment, for chasing a difference:
    /// `HEVC_PARAMS`, `HEVC_SOURCE`, `HEVC_SIZE` (WxH), `HEVC_FRAMES`, `HEVC_BITS`, `HEVC_FILTER`.
    #[test]
    #[ignore = "run by hand"]
    fn from_environment() {
        let var =
            |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_owned());
        let params = format!("{BASE}:{}", var("HEVC_PARAMS", ""));
        let source = var("HEVC_SOURCE", "testsrc2");
        let size = var("HEVC_SIZE", "64x64");
        let (w, h) = size.split_once('x').expect("WxH");
        check(&Case {
            name: "environment",
            size: (w.parse().expect("width"), h.parse().expect("height")),
            frames: var("HEVC_FRAMES", "1").parse().expect("frames"),
            bits: var("HEVC_BITS", "8").parse().expect("bits"),
            params: &params,
            source: &source,
            filter: &var("HEVC_FILTER", ""),
        });
    }

    /// Decode speed and equality on a stream named by `HEVC_BENCH` (Annex B),
    /// against the raw ffmpeg decode in `HEVC_BENCH_YUV` when given.
    #[test]
    #[ignore = "run by hand"]
    fn bench_from_environment() {
        let Ok(path) = std::env::var("HEVC_BENCH") else {
            eprintln!("skipped: HEVC_BENCH is not set");
            return;
        };
        let bytes = std::fs::read(&path).expect("stream");
        let units = access_units(&bytes);
        let params = {
            let mut p = CodecParams::video(CodecId::Hevc, 0, 0);
            p.nal_length_size = 0;
            p
        };
        // HEVC_BENCH_THREADS picks the worker count; unset is the process pool.
        let mut decoder = match std::env::var("HEVC_BENCH_THREADS") {
            Ok(threads) => {
                HevcDecoder::open_with_threads(&params, threads.parse().expect("threads"))
            }
            Err(_) => HevcDecoder::open(&params),
        }
        .expect("open");
        let start = std::time::Instant::now();
        let mut got = Vec::new();
        for (n, unit) in units.iter().enumerate() {
            got.extend(
                decoder
                    .push(&Packet::video(unit.clone(), n as i64, 1, 30, n == 0))
                    .expect("decode"),
            );
        }
        got.extend(decoder.drain().expect("drain"));
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "{} pictures in {seconds:.2} s: {:.1} fps",
            got.len(),
            got.len() as f64 / seconds
        );
        if let Ok(yuv) = std::env::var("HEVC_BENCH_YUV") {
            let reference = std::fs::read(yuv).expect("yuv");
            let mut at = 0;
            for (index, picture) in got.iter().enumerate() {
                let p = &picture.picture;
                for (plane, data) in [("y", p.y()), ("u", p.u()), ("v", p.v())] {
                    let want = &reference[at..at + data.len()];
                    if let Some(i) = data.iter().zip(want).position(|(a, b)| a != b) {
                        panic!("picture {index} {plane}: first difference at sample {i}");
                    }
                    at += data.len();
                }
            }
            eprintln!("all {} pictures equal ffmpeg", got.len());
        }
    }
}
