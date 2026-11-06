//! WebP files from libwebp's encoders, opened with the still container and
//! decoded here, checked pixel for pixel against independent decoders.
//!
//! Fixtures (`tests/fixtures`), all made with libwebp 1.6.0 tools from
//! small generated pictures (gradients, noise, shapes, alpha ramps,
//! palettes); FFmpeg 8.1.1 for the libvpx one:
//!
//! | File | Command | Exercises |
//! | --- | --- | --- |
//! | `lossy` | `cwebp -q 75` (61x45) | odd size, 4 segments, normal filter |
//! | `lossy-simple` | `cwebp -q 40 -nostrong -sharpness 5 -sns 100 -segments 2` | simple filter |
//! | `lossy-nofilter` | `cwebp -q 95 -f 0 -segments 1` (64x48) | no loop filter |
//! | `lossy-strong` | `cwebp -q 20 -f 100 -sharpness 2` | strong filter, sharpness |
//! | `libvpx` | `ffmpeg -c:v libvpx -slices 4`, frame wrapped in RIFF | libvpx, 4 token partitions |
//! | `lossy-alpha` | `cwebp -q 70` (53x39 RGBA) | `ALPH`, VP8L-compressed, no filter |
//! | `lossy-alpha-raw` | `cwebp -alpha_method 0 -alpha_filter none` | raw `ALPH` |
//! | `lossy-alpha-best` | `cwebp -alpha_filter best -alpha_q 50` | level-reduced alpha |
//! | `alpha-horizontal` | `cwebp -q 80 -alpha_filter fast` (33x27) | horizontal alpha filter |
//! | `alpha-vertical` | `cwebp -q 80 -alpha_filter fast` | vertical alpha filter |
//! | `alpha-gradient` | `cwebp -q 80 -alpha_filter best` | gradient alpha filter |
//! | `lossless` | `cwebp -lossless -z 6` | predictor, cross-color, cache |
//! | `lossless-fast` | `cwebp -lossless -q 0 -m 0` | subtract green |
//! | `lossless-alpha` | `cwebp -lossless -exact -z 9` | lossless alpha, meta codes |
//! | `pal2` to `pal200` | `cwebp -lossless -exact` (37x23) | color indexing, 8/4/2/1 per pixel |
//! | `anim` | `img2webp -lossy/-lossless` per frame | sub-rectangles, dispose, blend |
//! | `anim-mux` | `webpmux -frame ... +dispose+blend` | each flag, offsets, `ANIM` color |
//!
//! Each `.rgba` holds the expected straight RGBA. For still files it is
//! `dwebp -pam` with the PAM header removed; FFmpeg's own WebP decoder
//! wrote the same Y, U, and V planes as `dwebp -yuv` for every lossy file.
//! For the animations it is every frame from Pillow 11.3's `WebPImagePlugin`
//! (libwebp's animation decoder), one after the other.

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod oracle {
    use std::io::Cursor;

    use cutlass_container::{Demux, Input};
    use cutlass_decode::{
        CodecError, CodecId, CodecParams, Decoder, Output, Packet, PictureSamples, PixelFormat,
    };
    use cutlass_still::{StillDemux, still_limits};
    use cutlass_webp::WebpDecoder;

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    /// The packets the still container makes of a file, as the engine hands
    /// them to a decoder.
    fn packets(file: &[u8]) -> Vec<Packet> {
        let length = u64::try_from(file.len()).unwrap();
        let mut demux = StillDemux::open(Input::random(
            Cursor::new(file.to_vec()),
            length,
            still_limits(),
        ))
        .unwrap();
        let mut out = Vec::new();
        while let Some(packet) = demux.read_packet().unwrap() {
            out.push(Packet {
                stream: packet.stream(),
                data: packet.bytes().to_vec(),
                pts: packet.timestamp().presentation_or_decode().unwrap_or(0),
                duration: packet.timestamp().duration().unwrap_or(0),
                timescale: packet.timescale().ticks_per_second(),
                keyframe: packet.keyframe(),
                side: Vec::new(),
            });
        }
        out
    }

    fn open() -> WebpDecoder {
        WebpDecoder::open(&CodecParams::with_format(
            CodecId::Webp,
            1,
            1,
            PixelFormat::Rgba8,
        ))
        .unwrap()
    }

    fn rgba(output: &Output) -> (u32, u32, Vec<u8>) {
        let Output::Picture(picture) = output else {
            panic!("picture");
        };
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("cpu");
        };
        (picture.info().width, picture.info().height, bytes.to_vec())
    }

    /// Decode every packet of `name.webp` and return the pictures in order.
    fn decode(name: &str) -> Vec<(u32, u32, Vec<u8>)> {
        let mut decoder = open();
        let mut pictures = Vec::new();
        for packet in packets(&fixture(&format!("{name}.webp"))) {
            pictures.extend(decoder.push(&packet).unwrap().iter().map(rgba));
        }
        pictures
    }

    fn assert_same(name: &str, got: &[u8], expected: &[u8], width: u32) {
        assert_eq!(got.len(), expected.len(), "{name}: length");
        if let Some(at) = got.iter().zip(expected).position(|(a, b)| a != b) {
            let pixel = at / 4;
            let width = usize::try_from(width).unwrap();
            panic!(
                "{name}: first difference at x {} y {} channel {}: {} not {}",
                pixel % width,
                pixel / width,
                at % 4,
                got[at],
                expected[at]
            );
        }
    }

    fn still(name: &str, size: (u32, u32)) {
        let pictures = decode(name);
        assert_eq!(pictures.len(), 1, "{name}");
        let (width, height, bytes) = &pictures[0];
        assert_eq!((*width, *height), size, "{name}");
        assert_same(name, bytes, &fixture(&format!("{name}.rgba")), *width);
    }

    #[test]
    fn lossy_stills_match_dwebp_exactly() {
        still("lossy", (61, 45));
        still("lossy-simple", (61, 45));
        still("lossy-nofilter", (64, 48));
        still("lossy-strong", (64, 48));
        still("libvpx", (61, 45));
    }

    #[test]
    fn lossy_alpha_in_every_filter_and_compression_matches_dwebp() {
        still("lossy-alpha", (53, 39));
        still("lossy-alpha-raw", (53, 39));
        still("lossy-alpha-best", (53, 39));
        still("alpha-horizontal", (33, 27));
        still("alpha-vertical", (33, 27));
        still("alpha-gradient", (33, 27));
    }

    #[test]
    fn lossless_stills_match_dwebp_exactly() {
        still("lossless", (61, 45));
        still("lossless-fast", (61, 45));
        still("lossless-alpha", (53, 39));
        for palette in ["pal2", "pal4", "pal11", "pal200"] {
            still(palette, (37, 23));
        }
    }

    fn animation(name: &str, frames: usize, durations: &[i64]) {
        let file = fixture(&format!("{name}.webp"));
        let split = packets(&file);
        assert_eq!(split.len(), frames, "{name}: one packet per frame");
        let mut clock = 0;
        for (packet, duration) in split.iter().zip(durations) {
            assert_eq!(
                (packet.pts, packet.duration, packet.timescale),
                (clock, *duration, 1000)
            );
            clock += duration;
        }
        let pictures = decode(name);
        assert_eq!(pictures.len(), frames);
        let expected = fixture(&format!("{name}.rgba"));
        let frame_len = expected.len() / frames;
        for (n, (width, height, bytes)) in pictures.iter().enumerate() {
            assert_eq!((*width, *height), (40, 30));
            assert_same(
                &format!("{name} frame {n}"),
                bytes,
                &expected[n * frame_len..(n + 1) * frame_len],
                *width,
            );
        }
    }

    #[test]
    fn animations_composite_like_libwebp() {
        animation("anim", 4, &[100, 150, 80, 120]);
        animation("anim-mux", 6, &[100, 70, 90, 60, 50, 40]);
    }

    #[test]
    fn a_seek_starts_again_from_the_first_packet() {
        let split = packets(&fixture("anim-mux.webp"));
        let mut decoder = open();
        let first: Vec<_> = split
            .iter()
            .flat_map(|p| decoder.push(p).unwrap())
            .collect();
        // A later packet without the file header is refused after a reset,
        // and the same frames come back from the first packet again.
        decoder.reset();
        assert!(matches!(
            decoder.push(&split[3]),
            Err(CodecError::InvalidApi { .. })
        ));
        let again: Vec<_> = split
            .iter()
            .flat_map(|p| decoder.push(p).unwrap())
            .collect();
        assert_eq!(first, again);
        assert!(decoder.drain().unwrap().is_empty());
    }

    #[test]
    fn a_bad_packet_leaves_the_canvas_as_it_was() {
        let split = packets(&fixture("anim-mux.webp"));
        let mut decoder = open();
        decoder.push(&split[0]).unwrap();
        let mut broken = split[1].clone();
        let last = broken.data.len() - 1;
        broken.data.truncate(last - 10);
        assert!(decoder.push(&broken).is_err());
        let mut clean = open();
        clean.push(&split[0]).unwrap();
        assert_eq!(
            decoder.push(&split[1]).unwrap(),
            clean.push(&split[1]).unwrap()
        );
    }

    #[test]
    fn hostile_and_short_files_are_errors_not_panics() {
        let mut decoder = open();
        assert!(matches!(
            decoder.push(&Packet::audio(Vec::new(), 0, 0, 0)),
            Err(CodecError::InvalidApi { .. })
        ));
        for name in ["lossy", "lossless", "lossy-alpha", "pal4", "anim-mux"] {
            let file = fixture(&format!("{name}.webp"));
            // Every cut of the file fails cleanly (or, for a cut inside the
            // trailing padding, decodes); none panics.
            for cut in 0..file.len() {
                let _ = open().push(&Packet::audio(file[..cut].to_vec(), 0, 0, 0));
            }
            let mut flipped = file.clone();
            for at in 12..flipped.len() {
                flipped[at] ^= 0x5A;
                let _ = open().push(&Packet::audio(flipped.clone(), 0, 0, 0));
                flipped[at] ^= 0x5A;
            }
        }
        // A VP8X canvas of 16384 x 16384 is refused by the limit before any
        // pixel is allocated.
        let mut huge = b"RIFF\x16\0\0\0WEBPVP8X\x0a\0\0\0\x02\0\0\0".to_vec();
        huge.extend_from_slice(&[0xFF, 0x3F, 0, 0xFF, 0x3F, 0]);
        assert!(matches!(
            open().push(&Packet::audio(huge, 0, 0, 0)),
            Err(CodecError::Limit { .. })
        ));
        // So is a lossless picture that size.
        let mut lossless = b"RIFF\x12\0\0\0WEBPVP8L\x05\0\0\0".to_vec();
        lossless.extend_from_slice(&[0x2F, 0xFF, 0xFF, 0xFF, 0x0F, 0]);
        assert!(matches!(
            open().push(&Packet::audio(lossless, 0, 0, 0)),
            Err(CodecError::Limit { .. })
        ));
        // A frame placed past the canvas edge is malformed.
        let split = packets(&fixture("anim-mux.webp"));
        let mut decoder = open();
        decoder.push(&split[0]).unwrap();
        let mut outside = split[1].clone();
        // ANMF payload starts 8 bytes in; X offset / 2 is its first field.
        outside.data[8] = 30;
        assert!(matches!(
            decoder.push(&outside),
            Err(CodecError::Malformed { .. })
        ));
    }

    /// Every `name.webp` in `$CUTLASS_WEBP_ORACLE` against `name.rgba` beside
    /// it (all frames, one after another), with the decode time printed. For
    /// real pictures that do not belong in the repository:
    ///
    /// ```text
    /// cwebp -q 80 ../local-files/04-motion-graphics.png -o dir/motion.webp
    /// dwebp dir/motion.webp -pam -o dir/motion.pam   # header removed: motion.rgba
    /// CUTLASS_WEBP_ORACLE=dir cargo test --release -p cutlass-webp --test oracle -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs CUTLASS_WEBP_ORACLE, a folder of files and dwebp or Pillow output"]
    fn oracle_directory() {
        let Ok(dir) = std::env::var("CUTLASS_WEBP_ORACLE") else {
            println!("CUTLASS_WEBP_ORACLE is not set; nothing to compare");
            return;
        };
        let mut compared = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|ext| ext != "webp") {
                continue;
            }
            let expected = std::fs::read(path.with_extension("rgba")).unwrap();
            let file = std::fs::read(&path).unwrap();
            let split = packets(&file);
            let started = std::time::Instant::now();
            let mut decoder = open();
            let mut got = Vec::new();
            let mut size = (0, 0);
            for packet in &split {
                for output in decoder.push(packet).unwrap() {
                    let (width, height, bytes) = rgba(&output);
                    size = (width, height);
                    got.extend_from_slice(&bytes);
                }
            }
            let elapsed = started.elapsed();
            let name = path.display().to_string();
            assert_same(&name, &got, &expected, size.0);
            println!(
                "{name}: {}x{}, {} packets, {:.1} ms, identical",
                size.0,
                size.1,
                split.len(),
                elapsed.as_secs_f64() * 1000.0
            );
            compared += 1;
        }
        assert!(compared > 0, "no .webp files in {dir}");
    }
}
