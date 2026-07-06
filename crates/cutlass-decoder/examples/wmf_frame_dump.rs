//! Dump decoded frames as PPM for eyeballing decode correctness
//! (stride handling, aperture crop, color mapping).
//!
//! Usage: `cargo run --release -p cutlass-decoder --example wmf_frame_dump -- <file> [seconds...]`

#[cfg(target_os = "windows")]
fn main() {
    use std::path::Path;

    use cutlass_core::{FrameData, PixelFormat, RationalTime, VideoFrame};

    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: wmf_frame_dump <file> [seconds...]");
    let times: Vec<f64> = {
        let rest: Vec<f64> = args.filter_map(|s| s.parse().ok()).collect();
        if rest.is_empty() { vec![1.0] } else { rest }
    };
    let path = Path::new(&path);

    let mut decoder = cutlass_decoder::open_video_decoder(path, cutlass_decoder::OutputMode::Cpu)
        .expect("open decoder");
    let info = decoder.info().clone();
    println!(
        "coded {}x{}, display {}x{}, rotation {:?}, format {:?}",
        info.coded_size.0,
        info.coded_size.1,
        info.display_size.0,
        info.display_size.1,
        info.rotation,
        info.pixel_format,
    );

    for (index, seconds) in times.iter().enumerate() {
        let fps = info.frame_rate;
        let target = RationalTime::new((seconds * fps.as_f64()).round() as i64, fps);
        let frame = decoder
            .frame_at(target)
            .expect("frame_at")
            .expect("frame within stream");
        println!(
            "t={seconds}s -> pts {}/{} visible {:?}",
            frame.pts.value, frame.pts.rate.num, frame.visible
        );

        // NV12 -> RGB over the *visible* rect only.
        let VideoFrame { visible, data, .. } = &frame;
        let FrameData::Cpu(image) = data else {
            panic!("expected CPU frame")
        };
        assert_eq!(frame.format, PixelFormat::Nv12);
        let y_plane = &image.planes[0];
        let uv_plane = &image.planes[1];
        let (w, h) = (visible.width as usize, visible.height as usize);
        let (x0, y0) = (visible.x as usize, visible.y as usize);
        let mut rgb = vec![0u8; w * h * 3];
        for row in 0..h {
            let y_row = &y_plane.data[(y0 + row) * y_plane.stride..];
            let uv_row = &uv_plane.data[((y0 + row) / 2) * uv_plane.stride..];
            for col in 0..w {
                let y = f32::from(y_row[x0 + col]);
                let u = f32::from(uv_row[(x0 + col) / 2 * 2]) - 128.0;
                let v = f32::from(uv_row[(x0 + col) / 2 * 2 + 1]) - 128.0;
                // BT.709 limited-range approximation; good enough for eyeballs.
                let yl = (y - 16.0) * 1.164;
                let r = (yl + 1.793 * v).clamp(0.0, 255.0) as u8;
                let g = (yl - 0.213 * u - 0.533 * v).clamp(0.0, 255.0) as u8;
                let b = (yl + 2.112 * u).clamp(0.0, 255.0) as u8;
                let at = (row * w + col) * 3;
                rgb[at] = r;
                rgb[at + 1] = g;
                rgb[at + 2] = b;
            }
        }
        let out = std::env::temp_dir().join(format!("wmf_frame_{index}.png"));
        let file = std::fs::File::create(&out).expect("create png");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&rgb).expect("png pixels");
        writer.finish().expect("png finish");
        println!("wrote {}", out.display());
    }
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("wmf_frame_dump is Windows-only");
}
