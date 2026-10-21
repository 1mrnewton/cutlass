//! The GPU picture matches the CPU conversion.

use cutlass_color::{Matrix, YuvPlanes, to_rgba};
use cutlass_compose::{Error, Gpu};
use cutlass_core::{ColorRange, PixelFormat};

#[test]
fn an_uploaded_nv12_frame_matches_the_cpu_conversion() {
    let Some(gpu) = open_gpu() else {
        return;
    };
    let y = [16u8, 235, 235, 16];
    let chroma = [90u8, 240];
    let planes = YuvPlanes {
        format: PixelFormat::Nv12,
        range: ColorRange::Limited,
        matrix: Matrix::Bt709,
        width: 2,
        height: 2,
        y: &y,
        y_stride: 2,
        chroma: &chroma,
        chroma_stride: 2,
    };
    let expected = to_rgba(&planes).expect("cpu conversion");
    let image = gpu.render(&planes).expect("upload");
    let actual = gpu.read_rgba(&image).expect("readback");
    assert_close(&actual, &expected);
}

#[test]
fn an_uploaded_p010_frame_matches_the_cpu_conversion() {
    let Some(gpu) = open_gpu() else {
        return;
    };
    let mut y = Vec::new();
    for code in [64u16, 940, 940, 64] {
        y.extend((code << 6).to_le_bytes());
    }
    let mut chroma = Vec::new();
    for code in [360u16, 960] {
        chroma.extend((code << 6).to_le_bytes());
    }
    let planes = YuvPlanes {
        format: PixelFormat::P010,
        range: ColorRange::Limited,
        matrix: Matrix::Bt709,
        width: 2,
        height: 2,
        y: &y,
        y_stride: 4,
        chroma: &chroma,
        chroma_stride: 4,
    };
    let expected = to_rgba(&planes).expect("cpu conversion");
    let image = gpu.render(&planes).expect("upload");
    let actual = gpu.read_rgba(&image).expect("readback");
    assert_close(&actual, &expected);
}

fn assert_close(actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len());
    for (got, want) in actual.chunks(4).zip(expected.chunks(4)) {
        for (channel, reference) in got.iter().zip(want) {
            let delta = i16::from(*channel) - i16::from(*reference);
            assert!(delta.abs() <= 1, "{got:?} vs {want:?}");
        }
    }
}

#[expect(
    clippy::print_stderr,
    reason = "a missing GPU is reported instead of a silent pass"
)]
fn report_missing_gpu() {
    eprintln!("skipping: no GPU adapter");
}

fn open_gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => Some(gpu),
        Err(Error::NoAdapter) => {
            report_missing_gpu();
            None
        }
        Err(error) => panic!("opening a GPU failed: {error}"),
    }
}
