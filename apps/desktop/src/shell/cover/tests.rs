use std::time::{Duration, Instant};

use cutlass_model::{COVER_FILE, Cover, Micros};

use super::{cut_to, jpeg, shrink};
use crate::shell::tests::{Temp, land_job, movie_project, shell};
use cutlass_engine::Thumb;

/// The movie project's colour, 0x336699, give or take what JPEG and
/// averaging move it.
fn is_project_blue(pixel: &[u8]) -> bool {
    let near = |got: u8, want: u8| got.abs_diff(want) <= 6;
    near(pixel[0], 0x33) && near(pixel[1], 0x66) && near(pixel[2], 0x99)
}

fn jpeg_size(bytes: &[u8]) -> (u32, u32) {
    let info = cutlass_jpeg::read_jpeg_info(bytes).unwrap();
    (info.width, info.height)
}

#[test]
fn a_frame_cover_is_drawn_stored_and_followed_by_undo_and_redo() {
    let temp = Temp::new();
    movie_project(&temp.path);
    let mut desktop = shell();
    desktop.open_folder(&temp.path).unwrap();
    assert_eq!(desktop.cover(), None);
    assert_eq!(desktop.cover_file(), None);

    desktop.set_cover_frame(Micros(1_500_000)).unwrap();
    assert_eq!(desktop.job().map(|job| job.label), Some("Cover".to_owned()));
    // The cover is set only once its picture is written.
    assert_eq!(desktop.cover(), None);
    land_job(&mut desktop);
    assert_eq!(desktop.notice(), "Cover saved");
    let frame = Cover::Frame {
        time: Micros(1_500_000),
    };
    assert_eq!(desktop.cover(), Some(&frame));
    let file = temp.path.join(COVER_FILE);
    assert_eq!(desktop.cover_file(), Some(file.clone()));
    assert_eq!(jpeg_size(&std::fs::read(&file).unwrap()), (64, 32));
    let drawn = desktop.cover_version();

    // Undo takes the cover and its file away; redo draws it again.
    desktop.undo().unwrap();
    assert_eq!(desktop.cover(), None);
    assert!(!file.exists());
    assert_ne!(desktop.cover_version(), drawn);
    desktop.redo().unwrap();
    assert_eq!(desktop.cover(), Some(&frame));
    land_job(&mut desktop);
    assert_eq!(jpeg_size(&std::fs::read(&file).unwrap()), (64, 32));

    // A time past the end draws the last picture.
    desktop.set_cover_frame(Micros(60_000_000)).unwrap();
    land_job(&mut desktop);
    assert_eq!(
        desktop.cover(),
        Some(&Cover::Frame {
            time: Micros(1_999_999)
        })
    );
}

#[test]
fn a_picture_from_disk_is_cut_to_the_canvas_shape_from_its_middle() {
    let temp = Temp::new();
    movie_project(&temp.path);
    // 100 by 100: red rows above and below a blue band through the middle.
    let mut picture = image::RgbaImage::new(100, 100);
    for (_, y, pixel) in picture.enumerate_pixels_mut() {
        *pixel = if (25..75).contains(&y) {
            image::Rgba([0x33, 0x66, 0x99, 255])
        } else {
            image::Rgba([255, 0, 0, 255])
        };
    }
    let png = temp.path.join("poster.png");
    picture.save(&png).unwrap();
    let mut desktop = shell();
    desktop.open_folder(&temp.path).unwrap();

    // The canvas is 2:1, so the middle 100 by 50 is kept: all blue.
    let preview = desktop.cover_image_preview(&png, 1080).unwrap();
    assert_eq!((preview.width, preview.height), (100, 50));
    assert!(
        preview
            .rgba
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| is_project_blue(pixel))
    );

    desktop.set_cover_image(&png).unwrap();
    land_job(&mut desktop);
    assert_eq!(
        desktop.cover(),
        Some(&Cover::Image {
            path: png.display().to_string()
        })
    );
    let bytes = std::fs::read(temp.path.join(COVER_FILE)).unwrap();
    assert_eq!(jpeg_size(&bytes), (100, 50));

    let missing = desktop.set_cover_image(&temp.path.join("gone.png"));
    if missing.is_ok() {
        land_job(&mut desktop);
    }
    // A picture that does not open leaves the cover as it was.
    assert_eq!(
        desktop.cover(),
        Some(&Cover::Image {
            path: png.display().to_string()
        })
    );
}

#[test]
fn the_dialog_strip_draws_the_whole_timeline_on_its_own_thread() {
    let temp = Temp::new();
    movie_project(&temp.path);
    let mut desktop = shell();
    desktop.open_folder(&temp.path).unwrap();
    let until = Instant::now() + Duration::from_secs(60);
    loop {
        let (cells, _) = desktop.cover_strip(4, 8);
        assert_eq!(cells.len(), 4);
        if cells.iter().all(Option::is_some) {
            for cell in cells.iter().flatten() {
                assert_eq!((cell.width, cell.height), (16, 8));
                assert!(
                    cell.rgba
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|pixel| is_project_blue(pixel))
                );
            }
            break;
        }
        assert!(Instant::now() < until, "the strip was not drawn");
        std::thread::sleep(Duration::from_millis(10));
    }
    let (_, version) = desktop.cover_strip(4, 8);
    assert_eq!(version, 4);
    desktop.forget_cover_strip();
}

#[test]
fn a_cut_keeps_the_middle_in_the_canvas_shape() {
    // 4 by 2, columns 0 to 3 numbered in red, half transparent.
    let rgba: Vec<u8> = (0..2)
        .flat_map(|_| (0..4).flat_map(|x| [x * 10, 0, 0, 128]))
        .collect();
    let picture = Thumb {
        width: 4,
        height: 2,
        rgba,
    };
    // Square: the middle two columns, opaque, colour kept.
    let square = cut_to(&picture, (100, 100)).unwrap();
    assert_eq!((square.width, square.height), (2, 2));
    assert_eq!(&square.rgba[..8], &[10, 0, 0, 255, 20, 0, 0, 255]);
    // Taller than the picture's shape: the whole width, fewer rows.
    let tall = cut_to(&picture, (4, 1)).unwrap();
    assert_eq!((tall.width, tall.height), (4, 1));
    assert!(cut_to(&picture, (0, 1)).is_none());
}

#[test]
fn a_shrink_averages_and_keeps_the_shape() {
    // 4 by 4: the left half 0, the right half 200.
    let rgba: Vec<u8> = (0..4)
        .flat_map(|_| (0..4).flat_map(|x| if x < 2 { [0; 4] } else { [200; 4] }))
        .collect();
    let small = shrink(4, 4, &rgba, 2).unwrap();
    assert_eq!((small.width, small.height), (2, 2));
    assert_eq!(&small.rgba[..8], &[0, 0, 0, 0, 200, 200, 200, 200]);
    // No taller than asked: copied.
    assert_eq!(shrink(4, 4, &rgba, 8).unwrap().rgba, rgba);
    assert!(shrink(4, 4, &rgba[..8], 2).is_none());
}

#[test]
fn a_cover_jpeg_reads_back_at_its_size() {
    let picture = Thumb {
        width: 18,
        height: 10,
        rgba: [0x33, 0x66, 0x99, 255].repeat(180),
    };
    let bytes = jpeg(&picture).unwrap();
    assert_eq!(&bytes[..2], &[0xff, 0xd8]);
    assert_eq!(jpeg_size(&bytes), (18, 10));
}
