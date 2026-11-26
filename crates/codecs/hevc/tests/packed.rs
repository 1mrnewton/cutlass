//! The pictures the engine takes (`HevcDecoder::push_pictures`, and the
//! opener `register_decoders` gives a `DecodeFactory`): the samples at the
//! depth of the stream and what the stream says about them.
//!
//! The streams in `tests/fixtures` were written by libx265 through the
//! `ffmpeg` program. Beside each is an `.expect` file: for every picture,
//! in output order, a hash of the samples `ffmpeg -f rawvideo` wrote for
//! it and the lines `ffprobe -show_frames` printed for it. The tests
//! compare the decoder with those files, so they run with no FFmpeg; when
//! `ffmpeg` and `ffprobe` are installed they also decode the streams
//! again with them and compare every byte and every line.
//!
//! `HEVC_FIXTURES=write cargo test -p cutlass-hevc --test packed` writes
//! the streams and the `.expect` files again (it needs `ffmpeg` with
//! libx265, and `ffprobe`).

#[cfg(test)]
#[path = "../examples/common/mod.rs"]
mod common;

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod packed {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;

    use cutlass_decode::{
        ChromaSamplePosition, CodecId, CodecParams, ColorMatrix, ColorPrimaries, ColorRange,
        ContentLight, DecodeFactory, Output, Packet, PictureSamples, PixelFormat, Transfer,
        VideoDecoder,
    };
    use cutlass_hevc::{HevcDecoder, HevcPicture, PicTiming};

    use super::common;

    struct Fixture {
        name: &'static str,
        size: (usize, usize),
        /// The format `ffmpeg` encodes from and decodes to.
        pix: &'static str,
        params: &'static str,
        /// An HDR10+ message is put before the slices of every picture.
        hdr10plus: bool,
    }

    const COMMON: &str =
        "log-level=error:keyint=4:min-keyint=4:bframes=2:b-adapt=0:no-scenecut=1:info=0";

    const FIXTURES: &[Fixture] = &[
        Fixture {
            name: "main-limited",
            size: (64, 48),
            pix: "yuv420p",
            params: "range=limited:colorprim=bt709:transfer=bt709:colormatrix=bt709:chromaloc=0",
            hdr10plus: false,
        },
        Fixture {
            name: "main-full",
            size: (64, 48),
            pix: "yuvj420p",
            params: "range=full:colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m",
            hdr10plus: false,
        },
        // HDR10: PQ with a mastering display and light levels, and HDR10+.
        Fixture {
            name: "main10-pq",
            size: (64, 48),
            pix: "yuv420p10le",
            params: "range=limited:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:\
                     chromaloc=2:hdr10=1:max-cll=1000,400:\
                     master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,50)",
            hdr10plus: true,
        },
        // Broadcast HLG: the VUI says BT.2020 10-bit and the alternative
        // transfer characteristics message says HLG. 70×46 is cropped
        // from 72×48.
        Fixture {
            name: "main10-hlg",
            size: (70, 46),
            pix: "yuv420p10le",
            params: "range=limited:colorprim=bt2020:transfer=bt2020-10:colormatrix=bt2020nc:\
                     atc-sei=18",
            hdr10plus: false,
        },
        Fixture {
            name: "main12",
            size: (64, 48),
            pix: "yuv420p12le",
            params: "range=full:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
            hdr10plus: false,
        },
        // Fields, top first: every picture timing message has `pic_struct`.
        Fixture {
            name: "main-fields",
            size: (64, 48),
            pix: "yuv420p",
            params: "interlace=tff",
            hdr10plus: false,
        },
    ];

    const FRAMES: usize = 6;

    /// The lines of `ffprobe -show_frames` the decoder answers for.
    const KEYS: &[&str] = &[
        "width",
        "height",
        "pix_fmt",
        "color_range",
        "color_space",
        "color_primaries",
        "color_transfer",
        "chroma_location",
        "interlaced_frame",
        "top_field_first",
        "red_x",
        "red_y",
        "green_x",
        "green_y",
        "blue_x",
        "blue_y",
        "white_point_x",
        "white_point_y",
        "min_luminance",
        "max_luminance",
        "max_content",
        "max_average",
        "hdr10plus",
    ];

    type Lines = BTreeMap<String, String>;

    fn dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn fixture(name: &str) -> &'static Fixture {
        FIXTURES
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no fixture {name}"))
    }

    fn stream(f: &Fixture) -> PathBuf {
        dir().join(format!("{}.hevc", f.name))
    }

    fn runs(program: &str) -> bool {
        Command::new(program)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// The access units of a fixture.
    fn units(f: &Fixture) -> Vec<Vec<u8>> {
        let path = stream(f);
        let (_, units) = common::read(path.to_str().expect("path")).expect("fixture");
        assert_eq!(units.len(), FRAMES, "{}: access units", f.name);
        units
    }

    fn annex_b() -> CodecParams {
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.nal_length_size = 0;
        params
    }

    /// Every picture of `units` as the engine takes it. The access unit a
    /// picture came from is its `pts`.
    fn decode(params: &CodecParams, units: &[Vec<u8>], threads: usize) -> Vec<HevcPicture> {
        let mut decoder = HevcDecoder::open_with_threads(params, threads).expect("open");
        let mut got = Vec::new();
        for (n, unit) in units.iter().enumerate() {
            let packet = Packet::video(unit.clone(), n as i64, 1, 30, n == 0);
            got.extend(
                decoder
                    .push_pictures(&packet)
                    .unwrap_or_else(|e| panic!("access unit {n}: {e:?}")),
            );
        }
        got.extend(decoder.drain_pictures().expect("drain"));
        assert!(decoder.drain_pictures().expect("drain").is_empty());
        got
    }

    fn bytes(picture: &HevcPicture) -> &[u8] {
        match picture.picture.samples() {
            PictureSamples::Cpu(bytes) => bytes,
            other => panic!("not on the CPU: {other:?}"),
        }
    }

    // ---- What FFmpeg calls each value -------------------------------

    fn primaries_name(code: u8) -> &'static str {
        match code {
            1 => "bt709",
            5 => "bt470bg",
            6 => "smpte170m",
            9 => "bt2020",
            _ => "unknown",
        }
    }

    fn transfer_name(code: u8) -> &'static str {
        match code {
            1 => "bt709",
            6 => "smpte170m",
            14 => "bt2020-10",
            16 => "smpte2084",
            18 => "arib-std-b67",
            _ => "unknown",
        }
    }

    fn matrix_name(code: u8) -> &'static str {
        match code {
            1 => "bt709",
            5 => "bt470bg",
            6 => "smpte170m",
            9 => "bt2020nc",
            _ => "unknown",
        }
    }

    /// The HDR10+ message put into access unit `index`: SMPTE ST 2094-40
    /// application 4, version 1, one window, a target display of
    /// `1000 + index` candelas per square metre and no curves.
    fn hdr10plus(index: usize) -> Vec<u8> {
        let mut bits: Vec<bool> = Vec::new();
        let mut put = |value: u32, count: u32| {
            for bit in (0..count).rev() {
                bits.push(value >> bit & 1 == 1);
            }
        };
        put(0xb5, 8); // United States
        put(0x003c, 16); // SMPTE
        put(0x0001, 16); // ST 2094-40
        put(4, 8); // application_identifier
        put(1, 8); // application_version
        put(1, 2); // num_windows
        put(1000 + index as u32, 27); // targeted_system_display_maximum_luminance
        put(0, 1); // no actual peak luminance
        for maxscl in [900, 800, 700] {
            put(maxscl, 17);
        }
        put(300, 17); // average_maxrgb
        put(0, 4); // no percentiles
        put(5, 10); // fraction_bright_pixels
        put(0, 1); // no mastering display peak luminance
        put(0, 1); // no tone mapping
        put(0, 1); // no saturation mapping
        bits.chunks(8)
            .map(|byte| {
                byte.iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &bit)| acc | u8::from(bit) << (7 - i))
            })
            .collect()
    }

    /// What `ffprobe -show_frames` would print for a picture.
    fn describe(picture: &HevcPicture) -> Lines {
        let info = picture.picture.info();
        let mut lines = Lines::new();
        let mut put = |key: &str, value: String| {
            lines.insert(key.to_owned(), value);
        };
        put("width", info.width.to_string());
        put("height", info.height.to_string());
        let full = info.range == ColorRange::Full;
        let pix = match (info.pixel, full) {
            (PixelFormat::Yuv420p, false) => "yuv420p",
            (PixelFormat::Yuv420p, true) => "yuvj420p",
            (PixelFormat::Yuv420p10, _) => "yuv420p10le",
            (PixelFormat::Yuv420p12, _) => "yuv420p12le",
            other => panic!("pixel format {other:?}"),
        };
        put("pix_fmt", pix.to_owned());
        put("color_range", if full { "pc" } else { "tv" }.to_owned());
        put("color_space", matrix_name(info.cicp_matrix).to_owned());
        put(
            "color_primaries",
            primaries_name(info.cicp_primaries).to_owned(),
        );
        put(
            "color_transfer",
            transfer_name(info.cicp_transfer).to_owned(),
        );
        let chroma = match info.chroma_position {
            ChromaSamplePosition::Vertical => "left",
            ChromaSamplePosition::Colocated => "topleft",
            _ => "unspecified",
        };
        put("chroma_location", chroma.to_owned());
        // FFmpeg calls a picture interlaced when it is one field.
        let (field, top) = match picture.timing.map(|t| t.pic_struct) {
            Some(1 | 9 | 11) => (1, 1),
            Some(2 | 10 | 12) => (1, 0),
            _ => (0, 0),
        };
        put("interlaced_frame", field.to_string());
        put("top_field_first", top.to_string());
        if let Some(display) = info.mastering_display {
            // Back to the steps the message is coded in.
            let chroma = |v: u16| format!("{}/50000", (u64::from(v) * 50_000 + (1 << 15)) >> 16);
            let [red, green, blue] = display.primaries;
            for (name, point) in [
                ("red", red),
                ("green", green),
                ("blue", blue),
                ("white_point", display.white_point),
            ] {
                put(&format!("{name}_x"), chroma(point[0]));
                put(&format!("{name}_y"), chroma(point[1]));
            }
            let max = (u64::from(display.max_luminance) * 10_000 + (1 << 7)) >> 8;
            let min = (u64::from(display.min_luminance) * 10_000 + (1 << 13)) >> 14;
            put("max_luminance", format!("{max}/10000"));
            put("min_luminance", format!("{min}/10000"));
        }
        if let Some(light) = info.content_light {
            put("max_content", light.max_cll.to_string());
            put("max_average", light.max_fall.to_string());
        }
        let plus = picture.picture.t35().iter().any(|m| {
            m.country_code == 0xb5 && m.payload.starts_with(&[0x00, 0x3c, 0x00, 0x01, 0x04])
        });
        if plus {
            put("hdr10plus", "1".to_owned());
        }
        lines
    }

    /// One picture's lines of `ffprobe -show_frames -of default=nw=1`,
    /// cut down to [`KEYS`].
    fn probe_lines(text: &str) -> Vec<Lines> {
        let mut frames: Vec<Lines> = Vec::new();
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            if key == "media_type" {
                frames.push(Lines::new());
            }
            let Some(frame) = frames.last_mut() else {
                continue;
            };
            if key == "side_data_type" && value.contains("HDR10+") {
                frame.insert("hdr10plus".to_owned(), "1".to_owned());
            }
            if KEYS.contains(&key) {
                frame.insert(key.to_owned(), value.to_owned());
            }
        }
        frames
    }

    /// The pictures `ffmpeg` decodes from `path` and what `ffprobe` says
    /// of each.
    fn ffmpeg_decodes(f: &Fixture, path: &Path) -> (Vec<Vec<u8>>, Vec<Lines>) {
        let raw = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-"])
            .output()
            .expect("ffmpeg");
        assert!(raw.status.success(), "{}: ffmpeg", f.name);
        let (w, h) = f.size;
        let sample = if f.pix.ends_with("le") { 2 } else { 1 };
        let frames: Vec<Vec<u8>> = raw
            .stdout
            .chunks(w * h * 3 / 2 * sample)
            .map(<[u8]>::to_vec)
            .collect();
        let probe = Command::new("ffprobe")
            .args(["-v", "error", "-show_frames", "-of", "default=nw=1"])
            .arg(path)
            .output()
            .expect("ffprobe");
        assert!(probe.status.success(), "{}: ffprobe", f.name);
        (frames, probe_lines(&String::from_utf8_lossy(&probe.stdout)))
    }

    /// The stored expectation of a fixture: for each picture the hash of
    /// its samples and its lines.
    fn stored(f: &Fixture) -> Vec<(u64, Lines)> {
        let path = dir().join(format!("{}.expect", f.name));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let mut frames: Vec<(u64, Lines)> = Vec::new();
        for line in text.lines() {
            if line == "[frame]" {
                frames.push((0, Lines::new()));
                continue;
            }
            let (Some((key, value)), Some(frame)) = (line.split_once('='), frames.last_mut())
            else {
                continue;
            };
            if key == "samples_fnv1a64" {
                frame.0 = u64::from_str_radix(value, 16).expect("hash");
            } else {
                frame.1.insert(key.to_owned(), value.to_owned());
            }
        }
        frames
    }

    // ---- Writing the fixtures ----------------------------------------

    /// Annex B NAL units, without their start codes.
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
        starts
            .iter()
            .enumerate()
            .map(|(k, &s)| {
                let mut end = starts.get(k + 1).map_or(data.len(), |&n| n - 3);
                while end > s && data[end - 1] == 0 {
                    end -= 1;
                }
                &data[s..end]
            })
            .collect()
    }

    /// A prefix (39) or suffix (40) SEI NAL unit with one message, with
    /// its start code and emulation prevention.
    fn sei_nal(nal_type: u8, kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut rbsp = vec![kind, payload.len() as u8];
        rbsp.extend_from_slice(payload);
        rbsp.push(0x80);
        let mut out = vec![0, 0, 0, 1, nal_type << 1, 1];
        let mut zeros = 0;
        for byte in rbsp {
            if zeros >= 2 && byte <= 3 {
                out.push(3);
                zeros = 0;
            }
            zeros = if byte == 0 { zeros + 1 } else { 0 };
            out.push(byte);
        }
        out
    }

    /// `data` with `extra(n)` put before the first slice segment of
    /// picture `n` (in decoding order).
    fn before_each_picture(data: &[u8], mut extra: impl FnMut(usize) -> Vec<u8>) -> Vec<u8> {
        let mut out = Vec::new();
        let mut picture = 0;
        for nal in nal_units(data) {
            let slice = (nal[0] >> 1) & 0x3f <= 21;
            if slice && nal.get(2).is_some_and(|b| b & 0x80 != 0) {
                out.extend(extra(picture));
                picture += 1;
            }
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    fn write(f: &Fixture) {
        let (w, h) = f.size;
        let path = stream(f);
        let encoded = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size={w}x{h}:rate=30"))
            .args(["-frames:v", &FRAMES.to_string(), "-pix_fmt", f.pix])
            .args(["-c:v", "libx265", "-x265-params"])
            .arg(format!("{COMMON}:{}", f.params))
            .args(["-f", "hevc"])
            .arg(&path)
            .output()
            .expect("ffmpeg");
        assert!(
            encoded.status.success(),
            "{}: {}",
            f.name,
            String::from_utf8_lossy(&encoded.stderr)
        );
        if f.hdr10plus {
            let data = std::fs::read(&path).expect("read");
            let with = before_each_picture(&data, |n| sei_nal(39, 4, &hdr10plus(n)));
            std::fs::write(&path, with).expect("write");
        }
        let (frames, lines) = ffmpeg_decodes(f, &path);
        assert_eq!(frames.len(), FRAMES, "{}", f.name);
        assert_eq!(lines.len(), FRAMES, "{}", f.name);
        let version = Command::new("ffmpeg")
            .arg("-version")
            .output()
            .expect("ffmpeg");
        let version = String::from_utf8_lossy(&version.stdout);
        let mut text = format!(
            "# {}.hevc as FFmpeg decodes it: each picture's samples (FNV-1a, 64 bits, of\n\
             # `ffmpeg -f rawvideo`) and its lines of `ffprobe -show_frames`.\n\
             # Written by tests/packed.rs with {}.\n\
             # x265 parameters: {}\n",
            f.name,
            version.lines().next().unwrap_or_default(),
            f.params.replace(' ', ""),
        );
        for (frame, lines) in frames.iter().zip(&lines) {
            text.push_str("[frame]\n");
            text.push_str(&format!("samples_fnv1a64={:016x}\n", fnv(frame)));
            for key in KEYS {
                if let Some(value) = lines.get(*key) {
                    text.push_str(&format!("{key}={value}\n"));
                }
            }
        }
        std::fs::write(dir().join(format!("{}.expect", f.name)), text).expect("write");
    }

    /// With `HEVC_FIXTURES=write`, write every fixture again first.
    fn written() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if std::env::var("HEVC_FIXTURES").is_ok_and(|v| v == "write") {
                for f in FIXTURES {
                    write(f);
                }
            }
        });
    }

    // ---- The tests ---------------------------------------------------

    fn assert_lines(name: &str, index: usize, got: &Lines, want: &Lines, whose: &str) {
        for key in KEYS {
            assert_eq!(
                got.get(*key),
                want.get(*key),
                "{name} picture {index}: {key} against {whose}"
            );
        }
    }

    #[test]
    fn every_fixture_decodes_to_ffmpegs_samples_and_frame_data() {
        written();
        let live = runs("ffmpeg") && runs("ffprobe");
        if !live {
            eprintln!("no ffmpeg and ffprobe: compared with the stored expectations only");
        }
        for f in FIXTURES {
            let units = units(f);
            let want = stored(f);
            assert_eq!(want.len(), FRAMES, "{}: stored pictures", f.name);
            let theirs = live.then(|| ffmpeg_decodes(f, &stream(f)));
            for threads in [1, 2, 4] {
                let got = decode(&annex_b(), &units, threads);
                assert_eq!(got.len(), FRAMES, "{}: pictures", f.name);
                for (index, picture) in got.iter().enumerate() {
                    let lines = describe(picture);
                    assert_eq!(
                        fnv(bytes(picture)),
                        want[index].0,
                        "{} picture {index}: samples against the stored hash, {threads} threads",
                        f.name
                    );
                    assert_lines(f.name, index, &lines, &want[index].1, "the stored lines");
                    if let Some((frames, probed)) = &theirs {
                        assert!(
                            bytes(picture) == frames[index].as_slice(),
                            "{} picture {index}: samples differ from ffmpeg's",
                            f.name
                        );
                        assert_lines(f.name, index, &lines, &probed[index], "ffprobe");
                    }
                }
            }
        }
    }

    #[test]
    fn the_description_is_the_streams_own() {
        written();
        let first = |name: &str| decode(&annex_b(), &units(fixture(name)), 1).remove(0);

        let limited = first("main-limited").picture.info();
        assert_eq!(limited.pixel, PixelFormat::Yuv420p);
        assert_eq!(limited.range, ColorRange::Limited);
        assert_eq!(limited.primaries, ColorPrimaries::Bt709);
        assert_eq!(limited.transfer, Transfer::Bt709);
        assert_eq!(limited.matrix, ColorMatrix::Bt709);
        assert_eq!(
            (
                limited.cicp_primaries,
                limited.cicp_transfer,
                limited.cicp_matrix
            ),
            (1, 1, 1)
        );
        assert_eq!(limited.chroma_position, ChromaSamplePosition::Vertical);
        assert_eq!((limited.render_width, limited.render_height), (64, 48));
        // 30 pictures a second.
        assert_eq!(
            limited.time_scale,
            30 * limited.num_units_in_display_tick.max(1)
        );
        assert_eq!(limited.content_light, None);
        assert_eq!(limited.mastering_display, None);

        let full = first("main-full").picture.info();
        assert_eq!(full.range, ColorRange::Full);
        assert_eq!(full.matrix, ColorMatrix::Bt601);
        assert_eq!(
            (full.cicp_primaries, full.cicp_transfer, full.cicp_matrix),
            (6, 6, 6)
        );

        let pq = first("main10-pq").picture.info();
        assert_eq!(pq.pixel, PixelFormat::Yuv420p10);
        assert_eq!(pq.range, ColorRange::Limited);
        assert_eq!(pq.primaries, ColorPrimaries::Bt2020);
        assert_eq!(pq.transfer, Transfer::Pq);
        assert_eq!(pq.matrix, ColorMatrix::Bt2020);
        assert_eq!(
            (pq.cicp_primaries, pq.cicp_transfer, pq.cicp_matrix),
            (9, 16, 9)
        );
        assert_eq!(pq.chroma_position, ChromaSamplePosition::Colocated);
        assert_eq!(
            pq.content_light,
            Some(ContentLight {
                max_cll: 1000,
                max_fall: 400
            })
        );
        let display = pq.mastering_display.expect("mastering display");
        assert!((display.max_nits() - 1000.0).abs() < 1e-9);
        assert!((display.min_nits() - 0.005).abs() < 1e-4);
        // Red x 0.708, white x 0.3127.
        assert_eq!(display.primaries[0][0], 46399);
        assert_eq!(display.white_point[0], 20493);

        // The VUI says BT.2020 10-bit (14); the message prefers HLG (18).
        let hlg = first("main10-hlg").picture.info();
        assert_eq!(hlg.pixel, PixelFormat::Yuv420p10);
        assert_eq!((hlg.width, hlg.height), (70, 46));
        assert_eq!(hlg.transfer, Transfer::Hlg);
        assert_eq!(hlg.cicp_transfer, 18);
        assert_eq!(hlg.primaries, ColorPrimaries::Bt2020);

        let twelve = first("main12").picture.info();
        assert_eq!(twelve.pixel, PixelFormat::Yuv420p12);
        assert_eq!(twelve.range, ColorRange::Full);
    }

    #[test]
    fn the_description_lasts_past_the_picture_that_carries_it() {
        written();
        // x265 sends the SEI with each IDR picture, every fourth here: the
        // pictures between carry none and are described all the same.
        for name in ["main10-pq", "main10-hlg"] {
            let f = fixture(name);
            let units = units(f);
            let carries = |unit: &[u8], kind: u8| {
                nal_units(unit)
                    .iter()
                    .any(|nal| nal[0] >> 1 == 39 && nal[2] == kind)
            };
            let kind = if name == "main10-pq" { 137 } else { 147 };
            let with = units.iter().filter(|u| carries(u, kind)).count();
            assert!(with > 0 && with < FRAMES, "{name}: {with} units carry it");
            let got = decode(&annex_b(), &units, 2);
            let first = got[0].picture.info();
            for picture in &got {
                let info = picture.picture.info();
                assert_eq!(info.mastering_display, first.mastering_display);
                assert_eq!(info.content_light, first.content_light);
                assert_eq!(info.transfer, first.transfer);
            }
        }
    }

    #[test]
    fn hdr10plus_stays_with_its_picture_through_reordering() {
        written();
        let f = fixture("main10-pq");
        let got = decode(&annex_b(), &units(f), 4);
        let order: Vec<i64> = got.iter().map(|p| p.picture.pts()).collect();
        assert!(
            order.windows(2).any(|w| w[0] > w[1]),
            "output order is decoding order: {order:?}"
        );
        for picture in &got {
            let unit = picture.picture.pts() as usize;
            let want = hdr10plus(unit);
            let messages = picture.picture.t35();
            assert_eq!(messages.len(), 1, "access unit {unit}");
            assert_eq!(messages[0].country_code, 0xb5);
            assert_eq!(messages[0].country_code_extension, None);
            assert_eq!(messages[0].payload, &want[1..], "access unit {unit}");
        }
        // A stream without the message has none.
        let plain = decode(&annex_b(), &units(fixture("main10-hlg")), 1);
        assert!(plain.iter().all(|p| p.picture.t35().is_empty()));
    }

    #[test]
    fn t35_after_the_slices_belongs_to_the_picture_before_it() {
        written();
        let f = fixture("main-limited");
        let mut units = units(f);
        // A suffix SEI unit (type 40) at the end of each access unit.
        for (n, unit) in units.iter_mut().enumerate() {
            unit.extend(sei_nal(40, 4, &[0xb5, 0x00, 0x31, n as u8]));
        }
        let got = decode(&annex_b(), &units, 1);
        assert_eq!(got.len(), FRAMES);
        for picture in &got {
            let messages = picture.picture.t35();
            assert_eq!(messages.len(), 1);
            assert_eq!(
                messages[0].payload,
                [0x00, 0x31, picture.picture.pts() as u8]
            );
        }
    }

    #[test]
    fn fields_say_which_they_are() {
        written();
        let got = decode(&annex_b(), &units(fixture("main-fields")), 1);
        let structs: Vec<Option<PicTiming>> = got.iter().map(|p| p.timing).collect();
        for (index, timing) in structs.iter().enumerate() {
            let timing = timing.unwrap_or_else(|| panic!("picture {index}: no timing"));
            // Top fields and bottom fields in turn, top first.
            assert_eq!(timing.pic_struct, if index % 2 == 0 { 1 } else { 2 });
            assert!(!timing.duplicate);
        }
        // A progressive stream without the message says nothing.
        let frames = decode(&annex_b(), &units(fixture("main-limited")), 1);
        assert!(frames.iter().all(|p| p.timing.is_none()));
    }

    #[test]
    fn the_video_decoder_returns_the_same_samples() {
        written();
        for f in FIXTURES {
            let units = units(f);
            let packed = decode(&annex_b(), &units, 1);
            let mut decoder = HevcDecoder::open_with_threads(&annex_b(), 1).expect("open");
            let mut planes = Vec::new();
            for (n, unit) in units.iter().enumerate() {
                let packet = Packet::video(unit.clone(), n as i64, 1, 30, n == 0);
                planes.extend(decoder.push(&packet).expect("push"));
            }
            planes.extend(decoder.drain().expect("drain"));
            assert_eq!(planes.len(), packed.len(), "{}", f.name);
            for (index, (a, b)) in planes.iter().zip(&packed).enumerate() {
                let wide = |planes: [&[u16]; 3]| -> Vec<u8> {
                    planes
                        .iter()
                        .flat_map(|p| p.iter().flat_map(|s| s.to_le_bytes()))
                        .collect()
                };
                let want = if let Some(p) = &a.yuv420p10 {
                    wide([p.y(), p.u(), p.v()])
                } else if let Some(p) = &a.yuv420_high {
                    assert_eq!(p.bit_depth(), 12);
                    wide([p.y(), p.u(), p.v()])
                } else {
                    [a.picture.y(), a.picture.u(), a.picture.v()].concat()
                };
                assert!(bytes(b) == want.as_slice(), "{} picture {index}", f.name);
                let info = b.picture.info();
                assert_eq!(a.range, info.range, "{}", f.name);
                assert_eq!(a.primaries, info.primaries, "{}", f.name);
                assert_eq!(a.transfer, info.transfer, "{}", f.name);
                assert_eq!(a.matrix, info.matrix, "{}", f.name);
            }
        }
    }

    #[test]
    fn the_registered_opener_returns_deep_pictures() {
        written();
        let mut factory = DecodeFactory::new();
        cutlass_hevc::register_decoders(&mut factory);
        for (name, pixel) in [
            ("main-limited", PixelFormat::Yuv420p),
            ("main10-pq", PixelFormat::Yuv420p10),
            ("main12", PixelFormat::Yuv420p12),
        ] {
            let f = fixture(name);
            let want = stored(f);
            let mut decoder = factory.open(&annex_b()).expect("open");
            let mut got = Vec::new();
            for (n, unit) in units(f).iter().enumerate() {
                let packet = Packet::video(unit.clone(), n as i64, 1, 30, n == 0);
                got.extend(decoder.push(&packet).expect("push"));
            }
            got.extend(decoder.drain().expect("drain"));
            assert_eq!(got.len(), FRAMES, "{name}");
            for (index, output) in got.into_iter().enumerate() {
                let Output::Picture(picture) = output else {
                    panic!("{name}: not a picture");
                };
                assert_eq!(picture.info().pixel, pixel, "{name}");
                assert_eq!(picture.t35().is_empty(), !f.hdr10plus, "{name}");
                // Shared with the decoder, not copied out.
                let shared = picture.into_shared().expect("bytes");
                assert_eq!(fnv(&shared), want[index].0, "{name} picture {index}");
            }
        }
    }

    #[test]
    fn deep_pictures_let_go_are_filled_again() {
        written();
        for name in ["main-limited", "main10-pq", "main10-hlg", "main12"] {
            let f = fixture(name);
            let want = stored(f);
            let units = units(f);
            let mut decoder = HevcDecoder::open_with_threads(&annex_b(), 1).expect("open");
            let mut seen: Vec<*const Vec<u8>> = Vec::new();
            let mut again = 0;
            let mut count = 0;
            // The stream three times over: each starts with an IDR picture.
            for n in 0..3 * FRAMES {
                let unit = units[n % FRAMES].clone();
                let packet = Packet::video(unit, n as i64, 1, 30, n % FRAMES == 0);
                let mut pictures = decoder.push_pictures(&packet).expect("push");
                if n + 1 == 3 * FRAMES {
                    pictures.extend(decoder.drain_pictures().expect("drain"));
                }
                for picture in pictures {
                    let hash = want[count % FRAMES].0;
                    count += 1;
                    let shared: Arc<Vec<u8>> = picture.picture.into_shared().expect("bytes");
                    // A buffer filled again holds the new picture whole.
                    assert_eq!(fnv(&shared), hash, "{name} picture {count}");
                    let at = Arc::as_ptr(&shared);
                    if seen.contains(&at) {
                        again += 1;
                    } else {
                        seen.push(at);
                    }
                    // Let go here: the decoder may take it back.
                }
            }
            assert_eq!(count, 3 * FRAMES, "{name}");
            assert!(again > 0, "{name}: no buffer came back");
        }
    }

    #[test]
    fn broken_sei_does_not_stop_the_pictures() {
        written();
        let f = fixture("main10-pq");
        let want = stored(f);
        let units = units(f);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..40 {
            let noisy: Vec<Vec<u8>> = units
                .iter()
                .map(|unit| {
                    before_each_picture(unit, |_| {
                        // A message cut short, a size past the end, or noise.
                        let mut nal = match round % 4 {
                            0 => sei_nal(39, 137, &[0x12; 7]),
                            1 => vec![0, 0, 0, 1, 39 << 1, 1, 144, 250, 1, 2],
                            2 => vec![0, 0, 0, 1, 39 << 1, 1, 0xff, 0xff, 0xff],
                            _ => vec![0, 0, 0, 1, 39 << 1, 1],
                        };
                        for _ in 0..(next() % 24) {
                            nal.push((next() % 251) as u8 | 4);
                        }
                        nal
                    })
                })
                .collect();
            let got = decode(&annex_b(), &noisy, 1 + round % 3);
            assert_eq!(got.len(), FRAMES, "round {round}");
            for (index, picture) in got.iter().enumerate() {
                assert_eq!(fnv(bytes(picture)), want[index].0, "round {round}");
            }
        }
    }

    /// An `hvcC` record (ISO/IEC 14496-15 clause 8.3.3) holding `nals`,
    /// one array for each, with four-byte lengths.
    fn hvcc(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = vec![0u8; 23];
        out[0] = 1;
        out[21] = 0xfc | 3;
        out[22] = nals.len() as u8;
        for nal in nals {
            out.push(0x80 | ((nal[0] >> 1) & 0x3f));
            out.extend_from_slice(&1u16.to_be_bytes());
            out.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            out.extend_from_slice(nal);
        }
        out
    }

    fn length_prefixed(unit: &[u8], keep: impl Fn(u8) -> bool) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nal_units(unit) {
            if keep((nal[0] >> 1) & 0x3f) {
                out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                out.extend_from_slice(nal);
            }
        }
        out
    }

    #[test]
    fn an_hvcc_record_with_another_layers_sets_and_sei_opens() {
        written();
        let f = fixture("main-limited");
        let want = stored(f);
        let units = units(f);
        let first = nal_units(&units[0]);
        let set = |kind: u8| -> &[u8] {
            first
                .iter()
                .find(|nal| (nal[0] >> 1) & 0x3f == kind)
                .unwrap_or_else(|| panic!("no nal of type {kind}"))
        };
        // An SPS and a PPS of layer 1, as a multilayer file may list: their
        // syntax is another one, and read as layer 0 they are refused.
        let layer_one = |kind: u8| -> Vec<u8> {
            let mut nal = vec![kind << 1, 0x09];
            nal.extend_from_slice(&[0xff; 12]);
            nal
        };
        let (sps1, pps1) = (layer_one(33), layer_one(34));
        // Light levels the stream itself does not carry.
        let light = sei_nal(39, 144, &[0x02, 0x58, 0x00, 0x64]);
        let record = hvcc(&[set(32), set(33), &sps1, set(34), &pps1, &light[4..]]);
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.extra = record;
        // The samples hold slices only: the sets come from the record.
        let samples: Vec<Vec<u8>> = units
            .iter()
            .map(|unit| length_prefixed(unit, |kind| kind <= 21))
            .collect();
        for threads in [1, 3] {
            let got = decode(&params, &samples, threads);
            assert_eq!(got.len(), FRAMES);
            for (index, picture) in got.iter().enumerate() {
                assert_eq!(fnv(bytes(picture)), want[index].0, "picture {index}");
                assert_eq!(
                    picture.picture.info().content_light,
                    Some(ContentLight {
                        max_cll: 600,
                        max_fall: 100
                    })
                );
            }
        }
        // The same sets of layer 1 inside a sample are skipped too.
        let mut with = hvcc(&[set(32), set(33), set(34)]);
        std::mem::swap(&mut params.extra, &mut with);
        let mut sample = (sps1.len() as u32).to_be_bytes().to_vec();
        sample.extend_from_slice(&sps1);
        sample.extend_from_slice(&samples[0]);
        let mut all = samples.clone();
        all[0] = sample;
        assert_eq!(decode(&params, &all, 1).len(), FRAMES);
    }

    #[test]
    fn reset_keeps_what_the_extradata_says() {
        written();
        let f = fixture("main-limited");
        let units = units(f);
        let first = nal_units(&units[0]);
        let sets: Vec<&[u8]> = first
            .iter()
            .copied()
            .filter(|nal| (32..=34).contains(&((nal[0] >> 1) & 0x3f)))
            .collect();
        let light = sei_nal(39, 144, &[0x00, 0x64, 0x00, 0x32]);
        let mut nals = sets.clone();
        nals.push(&light[4..]);
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.extra = hvcc(&nals);
        let sample = length_prefixed(&units[0], |kind| kind <= 21);
        let mut decoder = HevcDecoder::open_with_threads(&params, 1).expect("open");
        for _ in 0..2 {
            let packet = Packet::video(sample.clone(), 0, 1, 30, true);
            let mut got = decoder.push_pictures(&packet).expect("push");
            got.extend(decoder.drain_pictures().expect("drain"));
            assert_eq!(got.len(), 1);
            assert_eq!(
                got[0].picture.info().content_light.map(|l| l.max_cll),
                Some(100)
            );
            decoder.reset();
        }
    }
}
