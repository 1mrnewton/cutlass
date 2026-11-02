//! Lifecycle of portable Android framebuffers.

use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, MAX_EDGE, Pass, TexelSource};

use super::{AndroidDevice, AndroidError};

fn extent(width: u32, height: u32) -> Extent {
    Extent { width, height }
}

fn one_pixel() -> Extent {
    extent(1, 1)
}

#[test]
fn an_oversized_edge_does_not_consume_the_first_id() {
    let mut device = AndroidDevice::new();
    for size in [
        extent(MAX_EDGE + 1, 1),
        extent(1, MAX_EDGE + 1),
        extent(MAX_EDGE + 1, MAX_EDGE + 1),
    ] {
        assert_eq!(
            device.create_framebuffer(size).unwrap_err(),
            AndroidError::Extent
        );
    }
    let first = device.create_framebuffer(one_pixel()).unwrap();
    assert_eq!(first.id(), 1);
    let second = device.create_framebuffer(one_pixel()).unwrap();
    assert_eq!(second.id(), 2);
}

#[test]
fn a_max_edge_row_is_created() {
    let mut device = AndroidDevice::new();
    let size = extent(MAX_EDGE, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    assert_eq!(framebuffer.id(), 1);
    let edge = usize::try_from(MAX_EDGE).unwrap();
    let row = edge.checked_mul(4).unwrap();
    assert_eq!(size.rgba_len(), Some(row));
    let pass = device.begin(framebuffer).unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    let bytes = device.read_rgba(framebuffer).unwrap();
    assert_eq!(bytes.len(), row);
    assert!(bytes.iter().all(|byte| *byte == 0));
}

#[test]
fn unknown_framebuffers_are_missing() {
    let mut device = AndroidDevice::new();
    device.create_framebuffer(one_pixel()).unwrap();
    for id in [0_u64, 99] {
        let framebuffer = Framebuffer::new(id);
        assert_eq!(
            device.begin(framebuffer).unwrap_err(),
            AndroidError::MissingFramebuffer
        );
        assert_eq!(
            device.destroy_framebuffer(framebuffer).unwrap_err(),
            AndroidError::MissingFramebuffer
        );
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            AndroidError::MissingFramebuffer
        );
    }
}

#[test]
fn a_second_begin_is_refused_and_the_first_pass_submits() {
    let mut device = AndroidDevice::new();
    let first = device.create_framebuffer(one_pixel()).unwrap();
    let second = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(first).unwrap();
    assert_eq!(device.begin(first).unwrap_err(), AndroidError::PassOpen);
    assert_eq!(device.begin(second).unwrap_err(), AndroidError::PassOpen);
    device
        .clear(pass, ClearColor { rgba: [3, 1, 4, 1] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(first).unwrap(), &[3, 1, 4, 1]);
}

#[test]
fn another_pass_is_refused_while_one_is_drawing() {
    let mut device = AndroidDevice::new();
    let size = one_pixel();
    let kept = device.create_framebuffer(size).unwrap();
    let other = device.create_framebuffer(size).unwrap();
    let pass = device.begin(kept).unwrap();
    device
        .clear(pass, ClearColor { rgba: [2, 2, 2, 2] })
        .unwrap();
    let foreign = Pass::new(other);
    assert_eq!(device.end(foreign).unwrap_err(), AndroidError::NoPass);
    assert_eq!(
        device
            .clear(foreign, ClearColor { rgba: [1, 2, 3, 4] },)
            .unwrap_err(),
        AndroidError::NoPass
    );
    assert_eq!(
        device.blend(foreign, Blend::Over).unwrap_err(),
        AndroidError::NoPass
    );
    assert_eq!(
        device
            .draw(
                foreign,
                TexelSource::Rgba {
                    extent: size,
                    bytes: &[9, 9, 9, 9],
                },
            )
            .unwrap_err(),
        AndroidError::NoPass
    );
    assert_eq!(device.submit(foreign).unwrap_err(), AndroidError::NoPass);
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(kept).unwrap(), &[2, 2, 2, 2]);
}

#[test]
fn commands_after_end_are_ended() {
    let mut device = AndroidDevice::new();
    let size = one_pixel();
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.end(pass).unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: size,
                    bytes: &[1, 2, 3, 4],
                },
            )
            .unwrap_err(),
        AndroidError::Ended
    );
    assert_eq!(
        device
            .clear(pass, ClearColor { rgba: [1, 2, 3, 4] },)
            .unwrap_err(),
        AndroidError::Ended
    );
    assert_eq!(
        device.blend(pass, Blend::Over).unwrap_err(),
        AndroidError::Ended
    );
    assert_eq!(device.end(pass).unwrap_err(), AndroidError::Ended);
}

#[test]
fn submit_and_read_wait_for_their_turn() {
    let mut device = AndroidDevice::new();
    let untouched = device.create_framebuffer(one_pixel()).unwrap();
    let framebuffer = device.create_framebuffer(one_pixel()).unwrap();
    assert_eq!(
        device.read_rgba(untouched).unwrap_err(),
        AndroidError::NotSubmitted
    );
    let pass = device.begin(framebuffer).unwrap();
    assert_eq!(device.submit(pass).unwrap_err(), AndroidError::NotEnded);
    assert_eq!(
        device.read_rgba(framebuffer).unwrap_err(),
        AndroidError::NotSubmitted
    );
    device.end(pass).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap_err(),
        AndroidError::NotSubmitted
    );
}

#[test]
fn a_clear_reads_back_and_a_second_submit_is_refused() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.submit(pass).unwrap_err(), AndroidError::NoPass);
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 5, 6, 7]);
}

#[test]
fn destroying_the_open_framebuffer_drops_the_pass() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.destroy_framebuffer(framebuffer).unwrap();
    assert_eq!(device.submit(pass).unwrap_err(), AndroidError::NoPass);
    assert_eq!(
        device.read_rgba(framebuffer).unwrap_err(),
        AndroidError::MissingFramebuffer
    );
    let again = device.create_framebuffer(one_pixel()).unwrap();
    let _pass = device.begin(again).unwrap();
}

#[test]
fn one_framebuffer_does_not_publish_the_other() {
    let mut device = AndroidDevice::new();
    let first = device.create_framebuffer(one_pixel()).unwrap();
    let second = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(first).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 2, 3, 4] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(first).unwrap(), &[1, 2, 3, 4]);
    assert_eq!(
        device.read_rgba(second).unwrap_err(),
        AndroidError::NotSubmitted
    );
    device.destroy_framebuffer(first).unwrap();
    assert_eq!(
        device.read_rgba(first).unwrap_err(),
        AndroidError::MissingFramebuffer
    );
    let pass = device.begin(second).unwrap();
    device
        .clear(pass, ClearColor { rgba: [8, 7, 6, 5] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(second).unwrap(), &[8, 7, 6, 5]);
}

#[test]
fn begin_again_hides_the_bytes_until_submit() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);

    let pass = device.begin(framebuffer).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap_err(),
        AndroidError::NotSubmitted
    );
    device
        .clear(pass, ClearColor { rgba: [2, 3, 4, 5] })
        .unwrap();
    device.end(pass).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap_err(),
        AndroidError::NotSubmitted
    );
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[2, 3, 4, 5]);
}

#[test]
fn an_empty_extent_reads_back_no_bytes() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(extent(0, 0)).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [9, 9, 9, 9] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert!(device.read_rgba(framebuffer).unwrap().is_empty());
}

#[test]
fn commands_without_a_begin_are_refused() {
    let mut device = AndroidDevice::new();
    let size = one_pixel();
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = Pass::new(framebuffer);
    assert_eq!(device.end(pass).unwrap_err(), AndroidError::NoPass);
    assert_eq!(
        device.blend(pass, Blend::Over).unwrap_err(),
        AndroidError::NoPass
    );
    assert_eq!(
        device.clear(pass, ClearColor::TRANSPARENT).unwrap_err(),
        AndroidError::NoPass
    );
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: size,
                    bytes: &[0, 0, 0, 0],
                },
            )
            .unwrap_err(),
        AndroidError::NoPass
    );
    assert_eq!(device.submit(pass).unwrap_err(), AndroidError::NoPass);
}

#[test]
fn a_platform_source_follows_the_pass_then_is_refused() {
    let mut device = AndroidDevice::new();
    let size = one_pixel();
    let framebuffer = device.create_framebuffer(size).unwrap();
    assert_eq!(
        device
            .draw(
                Pass::new(framebuffer),
                TexelSource::Platform {
                    extent: size,
                    token: 0,
                },
            )
            .unwrap_err(),
        AndroidError::NoPass
    );
    let pass = device.begin(framebuffer).unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Platform {
                    extent: size,
                    token: 0,
                },
            )
            .unwrap_err(),
        AndroidError::PlatformBuffer
    );
    device.end(pass).unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Platform {
                    extent: size,
                    token: 0,
                },
            )
            .unwrap_err(),
        AndroidError::Ended
    );
}

#[test]
fn an_empty_draw_after_end_is_ended() {
    let mut device = AndroidDevice::new();
    let size = one_pixel();
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.end(pass).unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: size,
                    bytes: b"",
                },
            )
            .unwrap_err(),
        AndroidError::Ended
    );
}

#[test]
fn a_short_draw_leaves_the_clear_in_place() {
    let mut device = AndroidDevice::new();
    let size = one_pixel();
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [9, 8, 7, 6] })
        .unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: size,
                    bytes: &[1, 2, 3],
                },
            )
            .unwrap_err(),
        AndroidError::Bytes
    );
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[9, 8, 7, 6]);
}

#[test]
fn a_wrong_size_wins_over_a_short_slice() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [4, 3, 2, 1] })
        .unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: extent(2, 1),
                    bytes: &[1, 2, 3],
                },
            )
            .unwrap_err(),
        AndroidError::Size
    );
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 3, 2, 1]);
}

#[test]
fn submit_of_a_different_pass_after_end_is_not_open() {
    let mut device = AndroidDevice::new();
    let kept = device.create_framebuffer(one_pixel()).unwrap();
    let other = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(kept).unwrap();
    device.end(pass).unwrap();
    assert_eq!(
        device.submit(Pass::new(other)).unwrap_err(),
        AndroidError::NoPass
    );
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(kept).unwrap().len(), 4);
}

#[test]
fn every_error_displays_its_frozen_message() {
    let messages = [
        (AndroidError::Extent, "the frame size is past the limit"),
        (
            AndroidError::Size,
            "the picture size does not match the framebuffer",
        ),
        (
            AndroidError::Bytes,
            "the picture byte length does not match its size",
        ),
        (
            AndroidError::PlatformBuffer,
            "the portable device cannot sample a platform buffer",
        ),
        (
            AndroidError::MissingFramebuffer,
            "the framebuffer is not on this device",
        ),
        (AndroidError::PassOpen, "a pass is already open"),
        (AndroidError::NoPass, "this pass is not open"),
        (AndroidError::Ended, "this pass has ended"),
        (AndroidError::NotEnded, "this pass has not ended"),
        (
            AndroidError::NotSubmitted,
            "the frame has not been submitted",
        ),
        (AndroidError::Image, "the vulkan image token is not usable"),
        (AndroidError::Device, "the vulkan device is unavailable"),
        (
            AndroidError::GpuUnsupported,
            "vulkan is not compiled on this host",
        ),
    ];
    for (error, message) in messages {
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn a_clear_fills_a_two_by_two() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(extent(2, 2)).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 2, 3, 4] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap(),
        &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    );
}

#[test]
fn a_missing_destroy_keeps_the_open_pass() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(one_pixel()).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    for id in [0_u64, 99] {
        assert_eq!(
            device
                .destroy_framebuffer(Framebuffer::new(id))
                .unwrap_err(),
            AndroidError::MissingFramebuffer
        );
    }
    device
        .clear(pass, ClearColor { rgba: [3, 3, 3, 3] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[3, 3, 3, 3]);
}

#[test]
fn destroying_the_first_framebuffer_leaves_the_second() {
    let mut device = AndroidDevice::new();
    let first = device.create_framebuffer(one_pixel()).unwrap();
    let second = device.create_framebuffer(one_pixel()).unwrap();
    assert_eq!(first.id(), 1);
    assert_eq!(second.id(), 2);
    device.destroy_framebuffer(first).unwrap();
    assert_eq!(
        device.read_rgba(first).unwrap_err(),
        AndroidError::MissingFramebuffer
    );
    let pass = device.begin(second).unwrap();
    device
        .clear(pass, ClearColor { rgba: [5, 6, 7, 8] })
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(second).unwrap(), &[5, 6, 7, 8]);
}
