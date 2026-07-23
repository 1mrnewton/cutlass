use super::*;
use crate::preview_select::viewport_mapping;

fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut px = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        px[i * 4..i * 4 + 4].copy_from_slice(&rgba);
    }
    px
}

fn checker_frame(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            px[i] = (x % 256) as u8;
            px[i + 1] = (y % 256) as u8;
            px[i + 2] = 0x40;
            px[i + 3] = 0xFF;
        }
    }
    px
}

#[test]
fn widget_to_frame_respects_letterbox() {
    // Canvas 1920×1080 in a wider 1000×540 viewport → horizontal bars.
    let (vw, vh) = (1000.0_f32, 540.0);
    let (cw, ch) = (1920.0_f32, 1080.0);
    let (fw, fh) = (192u32, 108);
    let (scale, ox, _oy) = viewport_mapping(cw, ch, vw, vh, 1.0, 0.0, 0.0);
    assert!((scale - 0.5).abs() < 1e-4);
    assert!((ox - 20.0).abs() < 1e-3);

    assert!(widget_to_frame_pixel(10.0, 270.0, vw, vh, 1.0, 0.0, 0.0, cw, ch, fw, fh).is_none());
    assert_eq!(
        widget_to_frame_pixel(500.0, 270.0, vw, vh, 1.0, 0.0, 0.0, cw, ch, fw, fh),
        Some((96, 54))
    );
    assert!(widget_to_frame_pixel(990.0, 270.0, vw, vh, 1.0, 0.0, 0.0, cw, ch, fw, fh).is_none());
}

#[test]
fn widget_to_frame_honors_zoom_and_pan() {
    // Square canvas in a matching viewport, zoomed 2×. With pan = +50 the
    // content's top-left lands at the viewport origin (see viewport_mapping).
    let (cw, ch) = (100.0_f32, 100.0);
    let (vw, vh) = (100.0_f32, 100.0);
    let (fw, fh) = (100u32, 100);
    let hit = widget_to_frame_pixel(0.0, 0.0, vw, vh, 2.0, 50.0, 50.0, cw, ch, fw, fh);
    assert_eq!(hit, Some((0, 0)));
    let mid = widget_to_frame_pixel(50.0, 50.0, vw, vh, 2.0, 50.0, 50.0, cw, ch, fw, fh);
    assert_eq!(mid, Some((25, 25)));
    // Pan the magnified content fully off to the right → miss.
    assert!(widget_to_frame_pixel(50.0, 50.0, vw, vh, 2.0, 200.0, 0.0, cw, ch, fw, fh).is_none());
}

#[test]
fn sample_color_returns_exact_rgba() {
    let mut px = solid_frame(4, 4, [0, 0, 0, 0xFF]);
    // Pixel (2, 1) → distinct color.
    let i = (6 * 4) as usize; // pixel (2, 1) in a 4-wide row
    px[i..i + 4].copy_from_slice(&[0x11, 0x22, 0x33, 0x80]);
    assert_eq!(
        sample_color(&px, 4, 4, 2, 1),
        Some([0x11, 0x22, 0x33, 0x80])
    );
    assert_eq!(sample_color(&px, 4, 4, 0, 0), Some([0, 0, 0, 0xFF]));
    assert!(sample_color(&px, 4, 4, 4, 0).is_none());
}

#[test]
fn sample_region_clamps_at_edges() {
    let px = checker_frame(8, 8);
    let region = sample_region(&px, 8, 8, 0, 0);
    assert_eq!(region.len(), (LOUPE_SIZE * LOUPE_SIZE * 4) as usize);
    let half = (LOUPE_SIZE / 2) as usize;
    // Center of loupe is the sampled pixel (0,0) → R=0, G=0.
    let c = (half * LOUPE_SIZE as usize + half) * 4;
    assert_eq!(&region[c..c + 4], &[0, 0, 0x40, 0xFF]);
    // Top-left loupe cell also clamps to (0,0).
    assert_eq!(&region[0..4], &[0, 0, 0x40, 0xFF]);
    // Right of center at loupe: frame (1,0).
    let right = (half * LOUPE_SIZE as usize + half + 1) * 4;
    assert_eq!(&region[right..right + 4], &[1, 0, 0x40, 0xFF]);
}

#[test]
fn session_arm_hover_commit_emits_one_edit() {
    let mut s = EyedropperSession::new();
    assert!(s.arm(7, true).is_none());
    assert!(s.is_armed());
    assert_eq!(
        s.hover(Some([10, 20, 30, 40])),
        Some(EyedropperAction::Preview {
            r: 10,
            g: 20,
            b: 30,
            a: 40
        })
    );
    assert_eq!(
        s.hover(Some([11, 22, 33, 44])),
        Some(EyedropperAction::Preview {
            r: 11,
            g: 22,
            b: 33,
            a: 44
        })
    );
    assert_eq!(
        s.commit(),
        Some(EyedropperAction::Commit {
            r: 11,
            g: 22,
            b: 33,
            a: 44
        })
    );
    assert!(!s.is_armed());
    // Second commit is a no-op.
    assert!(s.commit().is_none());
}

#[test]
fn session_cancel_reverts_after_preview() {
    let mut s = EyedropperSession::new();
    s.arm(1, false);
    assert_eq!(
        s.hover(Some([1, 2, 3, 4])),
        Some(EyedropperAction::Preview {
            r: 1,
            g: 2,
            b: 3,
            a: 0xFF // opaque forced
        })
    );
    assert_eq!(s.cancel(), Some(EyedropperAction::Cancel));
    assert!(!s.is_armed());
}

#[test]
fn session_cancel_without_hover_is_silent() {
    let mut s = EyedropperSession::new();
    s.arm(1, true);
    assert!(s.cancel().is_none());
    assert!(!s.is_armed());
}

#[test]
fn session_commit_requires_hit() {
    let mut s = EyedropperSession::new();
    s.arm(1, true);
    // Letterbox hover — no color.
    assert!(s.hover(None).is_none());
    assert!(s.commit().is_none());
    assert!(s.is_armed());
    s.cancel();
}

#[test]
fn rearm_cancels_prior_preview() {
    let mut s = EyedropperSession::new();
    s.arm(1, true);
    s.hover(Some([9, 9, 9, 9]));
    assert_eq!(s.arm(2, true), Some(EyedropperAction::Cancel));
    assert_eq!(s.consumer(), 2);
}

#[test]
fn sampling_held_frame_is_stable_while_live_mutates() {
    // Mirrors the Slint arm-time freeze: keep an Image handle, then replace
    // the "live" buffer with a different color and confirm the held copy
    // still samples the original pixel (SharedPixelBuffer retain).
    let red = solid_frame(4, 4, [0xFF, 0x00, 0x00, 0xFF]);
    let frozen = Image::from_rgba8(slint::SharedPixelBuffer::clone_from_slice(&red, 4, 4));
    let first = sample_preview(&frozen, 2.0, 2.0, 4.0, 4.0, 1.0, 0.0, 0.0, 4.0, 4.0, true);
    assert!(first.hit);
    assert_eq!(first.rgba, [0xFF, 0x00, 0x00, 0xFF]);

    let blue = solid_frame(4, 4, [0x00, 0x00, 0xFF, 0xFF]);
    let _live = Image::from_rgba8(slint::SharedPixelBuffer::clone_from_slice(&blue, 4, 4));
    let second = sample_preview(&frozen, 2.0, 2.0, 4.0, 4.0, 1.0, 0.0, 0.0, 4.0, 4.0, true);
    assert!(second.hit);
    assert_eq!(second.rgba, [0xFF, 0x00, 0x00, 0xFF]);
    assert_eq!(second.loupe_rgba[0..4], [0xFF, 0x00, 0x00, 0xFF]);
}
