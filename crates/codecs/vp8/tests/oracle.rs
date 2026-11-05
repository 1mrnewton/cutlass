//! Key frames from two independent encoders, checked plane for plane
//! against two independent decoders.
//!
//! Fixtures (`tests/fixtures`), made with libwebp 1.6.0 and FFmpeg 8.1.1:
//!
//! - `libvpx.vp8`: libvpx through `ffmpeg -c:v libvpx -slices 4 -crf 12
//!   -b:v 2M -frames:v 1`, 61x45, four token partitions, normal loop
//!   filter. The key frame was lifted out of the IVF file.
//! - `lossy-strong.vp8`: `cwebp -q 20 -f 100 -sharpness 2 -segments 4`,
//!   64x48, four segments, strong normal filter with sharpness.
//! - `lossy-simple.vp8`: `cwebp -q 40 -nostrong -sharpness 5 -sns 100
//!   -segments 2`, 61x45, simple loop filter.
//!
//! Each `.yuv` is `dwebp -yuv` of the WebP file that frame came from: the
//! Y plane, then U, then V. FFmpeg's own VP8 decoder (`-pix_fmt yuv420p`)
//! wrote the same bytes for all three, so the two reference decoders
//! agree and this crate must match them exactly.

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod oracle {
    use cutlass_vp8::{CodecError, decode_frame, frame_size};

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    fn planes(name: &str) -> (u32, u32, Vec<u8>) {
        let picture = decode_frame(&fixture(&format!("{name}.vp8")), 1 << 20).unwrap();
        let mut joined = picture.y.clone();
        joined.extend_from_slice(&picture.u);
        joined.extend_from_slice(&picture.v);
        assert_eq!(
            picture.u.len(),
            picture.chroma_width() * picture.chroma_height()
        );
        (picture.width, picture.height, joined)
    }

    #[test]
    fn libvpx_four_partition_key_frame_matches_the_reference_decoders() {
        let (width, height, joined) = planes("libvpx");
        assert_eq!((width, height), (61, 45));
        assert_eq!(joined, fixture("libvpx.yuv"));
    }

    #[test]
    fn segmented_strong_filter_key_frame_matches_the_reference_decoders() {
        let (width, height, joined) = planes("lossy-strong");
        assert_eq!((width, height), (64, 48));
        assert_eq!(joined, fixture("lossy-strong.yuv"));
    }

    #[test]
    fn simple_filter_key_frame_matches_the_reference_decoders() {
        let (width, height, joined) = planes("lossy-simple");
        assert_eq!((width, height), (61, 45));
        assert_eq!(joined, fixture("lossy-simple.yuv"));
    }

    #[test]
    fn bad_frames_are_errors_not_panics() {
        let frame = fixture("libvpx.vp8");
        assert_eq!(frame_size(&frame).unwrap(), (61, 45));
        // Every truncation fails cleanly, as it does in libwebp and FFmpeg.
        for cut in 0..frame.len() {
            let result = decode_frame(&frame[..cut], 1 << 20);
            assert!(result.is_err(), "cut at {cut} decoded");
        }
        // Flipped bytes past the header decode or fail, never panic.
        let mut flipped = frame.clone();
        for at in 10..flipped.len() {
            flipped[at] ^= 0xA5;
            let _ = decode_frame(&flipped, 1 << 20);
            flipped[at] ^= 0xA5;
        }
        // An inter frame, a hidden frame, a bad start code, a zero size.
        let mut inter = frame.clone();
        inter[0] |= 1;
        assert!(matches!(
            decode_frame(&inter, 1 << 20),
            Err(CodecError::Unsupported { .. })
        ));
        let mut hidden = frame.clone();
        hidden[0] &= !0x10;
        assert!(matches!(
            decode_frame(&hidden, 1 << 20),
            Err(CodecError::Unsupported { .. })
        ));
        let mut start = frame.clone();
        start[3] = 0;
        assert!(matches!(
            decode_frame(&start, 1 << 20),
            Err(CodecError::Malformed { .. })
        ));
        let mut empty = frame.clone();
        empty[6..8].fill(0);
        assert!(decode_frame(&empty, 1 << 20).is_err());
        // The largest coded size is refused by the pixel limit before any
        // plane is allocated.
        let mut huge = frame;
        huge[6..10].copy_from_slice(&[0xFF, 0x3F, 0xFF, 0x3F]);
        assert!(matches!(
            decode_frame(&huge, 1 << 24),
            Err(CodecError::Limit { .. })
        ));
    }
}
