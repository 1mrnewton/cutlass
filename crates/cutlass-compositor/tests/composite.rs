use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, CompositorError, GpuContext,
};

fn try_gpu() -> Option<GpuContext> {
    GpuContext::new_headless_blocking().ok()
}

fn solid_canvas(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    for chunk in bytes.chunks_exact_mut(4) {
        chunk.copy_from_slice(&rgba);
    }
    bytes
}

#[test]
fn solid_fills_canvas() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping solid_fills_canvas: no GPU adapter");
        return;
    };
    let mut compositor = Compositor::new(&gpu).expect("compositor");
    let config = CompositorConfig::new(64, 64);
    let image = compositor
        .composite(
            &gpu,
            &config,
            &[CompositeLayer::Solid {
                rgba: [200, 40, 10, 255],
            }],
        )
        .expect("composite");

    assert_eq!(image.width, 64);
    assert_eq!(image.height, 64);
    assert!(image.bytes.chunks_exact(4).all(|p| p == [200, 40, 10, 255]));
}

#[test]
fn rgba_over_solid_alpha_blends() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping rgba_over_solid_alpha_blends: no GPU adapter");
        return;
    };
    let mut compositor = Compositor::new(&gpu).expect("compositor");
    let config = CompositorConfig::new(4, 4);

    let top = solid_canvas(4, 4, [0, 255, 0, 128]);
    let image = compositor
        .composite(
            &gpu,
            &config,
            &[
                CompositeLayer::Solid {
                    rgba: [255, 0, 0, 255],
                },
                CompositeLayer::Rgba { bytes: top },
            ],
        )
        .expect("composite");

    let pixel = |x: u32, y: u32| {
        let i = ((y * 4 + x) * 4) as usize;
        [
            image.bytes[i],
            image.bytes[i + 1],
            image.bytes[i + 2],
            image.bytes[i + 3],
        ]
    };

    let p = pixel(1, 1);
    assert!(p[0] > 100 && p[0] < 200, "red channel blended: {p:?}");
    assert!(p[1] > 100, "green channel present: {p:?}");
    assert_eq!(p[3], 255, "opaque output alpha");
}

#[test]
fn empty_layers_yields_transparent_black() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping empty_layers_yields_transparent_black: no GPU adapter");
        return;
    };
    let mut compositor = Compositor::new(&gpu).expect("compositor");
    let config = CompositorConfig::new(8, 8);
    let image = compositor
        .composite(&gpu, &config, &[])
        .expect("composite");

    assert!(image.bytes.iter().all(|&b| b == 0));
}

#[test]
fn invalid_dimensions_error() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let mut compositor = Compositor::new(&gpu).expect("compositor");
    let err = compositor
        .composite(&gpu, &CompositorConfig::new(0, 64), &[])
        .unwrap_err();
    assert!(matches!(err, CompositorError::InvalidDimensions { .. }));
}
