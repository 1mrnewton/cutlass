//! Phone and camera HEVC from `../local-files`, demuxed by the ISO BMFF
//! reader exactly as the engine does (the `hvcC` payload as extradata,
//! length-prefixed samples), against ffmpeg's decode. Skips when the folder
//! or ffmpeg is missing.

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod local_media {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use cutlass_codec::StreamCodec;
    use cutlass_container::Demux;
    use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder};
    use cutlass_hevc::HevcDecoder;
    use cutlass_isobmff::IsoBmffFile;

    /// Access units pushed, and pictures compared from the start of output.
    const PUSHED: usize = 40;
    const COMPARED: usize = 30;

    fn local_files() -> Option<PathBuf> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
        dir.is_dir().then_some(dir)
    }

    fn has_ffmpeg() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// The first `COMPARED` pictures of ffmpeg's decode, samples as `u16`.
    fn ffmpeg_frames(path: &Path, ten_bit: bool) -> Vec<u16> {
        let out = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-ignore_editlist", "1", "-i"])
            .arg(path)
            .args([
                "-map",
                "0:v:0",
                "-frames:v",
                &COMPARED.to_string(),
                "-fps_mode",
                "passthrough",
            ])
            .args(["-f", "rawvideo", "-"])
            .output()
            .expect("ffmpeg");
        assert!(
            out.status.success(),
            "ffmpeg: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        if ten_bit {
            out.stdout
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| u16::from_le_bytes(b))
                .collect()
        } else {
            out.stdout.iter().map(|&b| u16::from(b)).collect()
        }
    }

    fn check(name: &str) {
        let Some(dir) = local_files() else {
            eprintln!("skipped: no local-files folder");
            return;
        };
        let path = dir.join(name);
        if !path.is_file() {
            eprintln!("skipped: {name} is not in local-files");
            return;
        }
        if !has_ffmpeg() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let file = std::fs::File::open(&path).expect("open");
        let mut demux = IsoBmffFile::open(std::io::BufReader::new(file)).expect("demux");
        let Some(stream) = demux
            .streams()
            .iter()
            .find(|s| matches!(s.params().codec(), StreamCodec::Known(CodecId::Hevc)))
            .cloned()
        else {
            panic!("{name}: no HEVC track");
        };
        let size = stream.params().size();
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.extra = stream.params().extradata().to_vec();
        let mut decoder = HevcDecoder::open(&params).expect("decoder");
        let mut got = Vec::new();
        let mut pushed = 0;
        let start = std::time::Instant::now();
        while pushed < PUSHED {
            let Some(packet) = demux.read_packet().expect("packet") else {
                break;
            };
            if packet.stream() != stream.id().get() {
                continue;
            }
            let unit = Packet::video(
                packet.bytes().to_vec(),
                pushed as i64,
                1,
                30,
                packet.keyframe(),
            );
            got.extend(
                decoder
                    .push(&unit)
                    .unwrap_or_else(|e| panic!("{name}: access unit {pushed}: {e:?}")),
            );
            pushed += 1;
        }
        got.extend(decoder.drain().expect("drain"));
        let seconds = start.elapsed().as_secs_f64();
        eprintln!("{name}: {pushed} access units in {seconds:.2} s ({size:?})");
        let ten_bit = got.first().is_some_and(|p| p.yuv420p10.is_some());
        let reference = ffmpeg_frames(&path, ten_bit);
        let mut at = 0;
        for (index, picture) in got.iter().take(COMPARED).enumerate() {
            let planes: [Vec<u16>; 3] = if let Some(p) = &picture.yuv420p10 {
                [p.y().to_vec(), p.u().to_vec(), p.v().to_vec()]
            } else {
                let p = &picture.picture;
                [p.y(), p.u(), p.v()].map(|v| v.iter().map(|&b| u16::from(b)).collect())
            };
            for (plane, data) in ["y", "u", "v"].iter().zip(&planes) {
                let want = &reference[at..at + data.len()];
                if let Some(i) = data.iter().zip(want).position(|(a, b)| a != b) {
                    let wrong = data.iter().zip(want).filter(|(a, b)| a != b).count();
                    panic!(
                        "{name} picture {index} {plane}: first difference at sample {i}, {wrong} differ"
                    );
                }
                at += data.len();
            }
        }
        assert_eq!(
            at,
            reference.len(),
            "{name}: ffmpeg returned more samples than compared"
        );
    }

    /// The first pictures of an iPhone 4K clip (wavefronts, 68 CTB rows)
    /// with one worker and with eight: every sample the same.
    #[test]
    fn iphone_threads_match_one_thread() {
        let Some(dir) = local_files() else {
            eprintln!("skipped: no local-files folder");
            return;
        };
        let path = dir.join("IMG_9197.mov");
        if !path.is_file() {
            eprintln!("skipped: IMG_9197.mov is not in local-files");
            return;
        }
        let decode = |threads: usize| -> Vec<Vec<u16>> {
            let file = std::fs::File::open(&path).expect("open");
            let mut demux = IsoBmffFile::open(std::io::BufReader::new(file)).expect("demux");
            let stream = demux
                .streams()
                .iter()
                .find(|s| matches!(s.params().codec(), StreamCodec::Known(CodecId::Hevc)))
                .cloned()
                .expect("HEVC track");
            let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
            params.extra = stream.params().extradata().to_vec();
            let mut decoder = HevcDecoder::open_with_threads(&params, threads).expect("decoder");
            let mut got = Vec::new();
            let mut pushed = 0;
            while pushed < 12 {
                let Some(packet) = demux.read_packet().expect("packet") else {
                    break;
                };
                if packet.stream() != stream.id().get() {
                    continue;
                }
                let unit = Packet::video(packet.bytes().to_vec(), pushed, 1, 30, packet.keyframe());
                got.extend(decoder.push(&unit).expect("push"));
                pushed += 1;
            }
            got.extend(decoder.drain().expect("drain"));
            got.iter()
                .map(|f| {
                    let p = f.yuv420p10.as_ref().expect("10-bit");
                    [p.y(), p.u(), p.v()].concat()
                })
                .collect()
        };
        let one = decode(1);
        let eight = decode(8);
        assert!(!one.is_empty());
        assert_eq!(one.len(), eight.len());
        for (index, (a, b)) in one.iter().zip(&eight).enumerate() {
            assert!(a == b, "picture {index} differs with eight workers");
        }
    }

    #[test]
    fn iphone_main10_uhd() {
        check("IMG_9197.mov");
    }

    #[test]
    fn iphone_main10_uhd_long() {
        check("IMG_9196.MOV");
    }

    #[test]
    fn screen_recording_full_range() {
        check("ScreenRecording_11-03-2026 09-19-08_1.MP4");
    }

    #[test]
    fn screen_recording_large() {
        check("ScreenRecording_06-18-2026 20-15-57_1.MP4");
    }

    #[test]
    fn main_720p() {
        check("7688387224072670472.mp4");
    }
}
