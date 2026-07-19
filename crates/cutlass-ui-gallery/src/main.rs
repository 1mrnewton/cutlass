use std::borrow::Cow;
use std::rc::Rc;

use slint::BackendSelector;
use slint::DataTransfer;
use slint::Image;
use slint::Model;
use slint::ModelRc;
use slint::Rgba8Pixel;
use slint::SharedPixelBuffer;
use slint::SharedString;
use slint::VecModel;
use slint::wgpu_29::WGPUConfiguration;

slint::include_modules!();

fn contains_ci(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

/// Library chips encode `"label|duration-ms"` in the drag payload.
fn split_payload(payload: &str) -> (SharedString, i32) {
    match payload.split_once('|') {
        Some((label, duration)) => {
            let ms = duration.parse::<i32>().unwrap_or(2500).max(100);
            (SharedString::from(label), ms)
        }
        None => (SharedString::from(payload), 2500),
    }
}

fn wire_string_filter(app: &GalleryWindow) {
    let filter = app.global::<StringFilter>();

    filter.on_contains(|haystack, needle| contains_ci(haystack.as_str(), needle.as_str()));

    filter.on_index_of(|items, value| {
        let value = value.as_str();
        for i in 0..items.row_count() {
            if items.row_data(i).is_some_and(|row| row.as_str() == value) {
                return i as i32;
            }
        }
        -1
    });

    filter.on_next_match(|items, query, from, direction| {
        let len = items.row_count();
        if len == 0 || direction == 0 {
            return -1;
        }
        let query = query.as_str();
        let len_i = len as i32;
        let mut i = from + direction;
        for _ in 0..len {
            if i < 0 {
                i = len_i - 1;
            } else if i >= len_i {
                i = 0;
            }
            if let Some(row) = items.row_data(i as usize)
                && contains_ci(row.as_str(), query)
            {
                return i;
            }
            i += direction;
        }
        -1
    });
}

fn wire_data_transfer(app: &GalleryWindow) {
    let bridge = app.global::<DataTransferBridge>();

    bridge.on_plain_text(DataTransfer::from);

    bridge.on_has_plain_text(|data| data.has_plain_text());

    bridge.on_read_plain_text(|data| {
        data.plain_text()
            .unwrap_or_else(|_| SharedString::default())
    });
}

fn wire_timeline_demo(app: &GalleryWindow) {
    let demo = app.global::<TimelineDemo>();
    let clips = Rc::new(VecModel::<DemoClip>::default());
    demo.set_clips(ModelRc::from(clips.clone()));

    demo.on_payload_label(|payload| split_payload(payload.as_str()).0);
    demo.on_payload_duration_ms(|payload| split_payload(payload.as_str()).1);

    demo.on_push_clip({
        let clips = clips.clone();
        move |label, start_ms, duration_ms| {
            let hue = (clips.row_count() as i32) % 3;
            clips.push(DemoClip {
                label,
                start_ms: start_ms.max(0),
                duration_ms: duration_ms.max(100),
                hue,
            });
        }
    });

    demo.on_clear_clips({
        let clips = clips.clone();
        move || {
            clips.set_vec(Vec::new());
        }
    });
}

/// Procedural demo stills for the Image gallery section.
fn make_demo_image(width: u32, height: u32, kind: u8) -> Image {
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    let stride = width as usize;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let (r, g, b) = match kind {
                0 => {
                    // Aurora — cool vertical wash + soft band
                    let band = (1.0 - (fy - 0.35).abs() * 3.0).clamp(0.0, 1.0);
                    (
                        (20.0 + fx * 40.0 + band * 80.0) as u8,
                        (80.0 + fy * 100.0 + band * 120.0) as u8,
                        (140.0 + (1.0 - fx) * 90.0) as u8,
                    )
                }
                1 => {
                    // Dune — warm diagonal
                    let t = (fx * 0.65 + fy * 0.35).clamp(0.0, 1.0);
                    (
                        (180.0 + t * 60.0) as u8,
                        (110.0 + t * 40.0) as u8,
                        (70.0 + (1.0 - t) * 30.0) as u8,
                    )
                }
                _ => {
                    // Tide — horizontal ripples
                    let wave = ((fx * 12.0 + fy * 4.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
                    (
                        (30.0 + wave * 40.0) as u8,
                        (90.0 + fy * 80.0 + wave * 50.0) as u8,
                        (150.0 + (1.0 - fy) * 70.0) as u8,
                    )
                }
            };
            buffer.make_mut_slice()[y as usize * stride + x as usize] =
                Rgba8Pixel { r, g, b, a: 255 };
        }
    }
    Image::from_rgba8(buffer)
}

fn wire_demo_images(app: &GalleryWindow) {
    let images = app.global::<DemoImages>();
    images.set_aurora(make_demo_image(480, 320, 0));
    images.set_dune(make_demo_image(400, 400, 1));
    images.set_tide(make_demo_image(520, 280, 2));
}

fn wire_image_actions(app: &GalleryWindow) {
    let actions = app.global::<ImageActions>();

    actions.on_copy_image(|img| {
        let Some(buffer) = img.to_rgba8() else {
            return false;
        };
        let mut clipboard = match arboard::Clipboard::new() {
            Ok(c) => c,
            Err(_) => return false,
        };
        let data = arboard::ImageData {
            width: buffer.width() as usize,
            height: buffer.height() as usize,
            bytes: Cow::Borrowed(buffer.as_bytes()),
        };
        clipboard.set_image(data).is_ok()
    });

    actions.on_save_image(|img, suggested_name| {
        let Some(buffer) = img.to_rgba8() else {
            return false;
        };
        let name = if suggested_name.is_empty() {
            "image.png".to_string()
        } else if suggested_name.as_str().ends_with(".png") {
            suggested_name.to_string()
        } else {
            format!("{suggested_name}.png")
        };

        let Some(path) = rfd::FileDialog::new()
            .set_title("Save image")
            .set_file_name(&name)
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return false;
        };

        let rgba = match image::RgbaImage::from_raw(
            buffer.width(),
            buffer.height(),
            buffer.as_bytes().to_vec(),
        ) {
            Some(img) => img,
            None => return false,
        };
        rgba.save(path).is_ok()
    });
}

fn main() -> Result<(), slint::PlatformError> {
    BackendSelector::new()
        .require_wgpu_29(WGPUConfiguration::default())
        .select()?;

    let app = GalleryWindow::new()?;
    wire_string_filter(&app);
    wire_data_transfer(&app);
    wire_timeline_demo(&app);
    wire_demo_images(&app);
    wire_image_actions(&app);
    app.run()
}
