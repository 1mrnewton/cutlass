use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cutlass_model::{COVER_FILE, Draft, THUMBNAIL_FILE};
use cutlass_store::Folder;

use crate::shell::Shell;
use crate::shell::tests::{Temp, movie_project, shell};

/// Ask for `project`'s card picture until nothing is being drawn.
fn settled(desktop: &mut Shell, project: &Path) -> Option<PathBuf> {
    let until = Instant::now() + Duration::from_secs(60);
    loop {
        let (file, pending) = desktop.project_thumbnail(project);
        if !pending {
            return file;
        }
        assert!(Instant::now() < until, "the card picture was not drawn");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The picture's size and its middle pixel.
fn read(file: &Path) -> ((u32, u32), [u8; 3]) {
    let picture = image::open(file).unwrap().to_rgb8();
    let middle = picture
        .get_pixel(picture.width() / 2, picture.height() / 2)
        .0;
    (picture.dimensions(), middle)
}

fn near(got: [u8; 3], want: [u8; 3]) -> bool {
    got.iter()
        .zip(want)
        .all(|(got, want)| got.abs_diff(want) <= 8)
}

/// Save the project at `dir` again with its colour clip in `color`.
fn recolor(dir: &Path, color: &str) {
    let mut folder = Folder::read(dir).unwrap();
    folder.draft.materials.material_colors[0].solid_color = color.to_owned();
    // A save in the same instant as the last could keep its file time.
    std::thread::sleep(Duration::from_millis(20));
    folder.save(dir).unwrap();
}

#[test]
fn a_projects_card_shows_its_first_frame_drawn_again_after_each_save() {
    let temp = Temp::new();
    movie_project(&temp.path);
    // A picked cover of another colour is not what the card shows.
    std::fs::write(temp.path.join(COVER_FILE), b"not the card").unwrap();
    let mut desktop = shell();

    let file = settled(&mut desktop, &temp.path).expect("a card picture");
    assert_eq!(file, temp.path.join(THUMBNAIL_FILE));
    let (size, middle) = read(&file);
    assert_eq!(size, (64, 32));
    assert!(near(middle, [0x33, 0x66, 0x99]), "{middle:?}");
    let drawn = std::fs::metadata(&file).unwrap().modified().unwrap();

    // Not saved since: not drawn again.
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(settled(&mut desktop, &temp.path), Some(file.clone()));
    assert_eq!(std::fs::metadata(&file).unwrap().modified().unwrap(), drawn);

    // Saved with a red clip: the card follows.
    recolor(&temp.path, "0xFFCC2211");
    settled(&mut desktop, &temp.path);
    let (_, middle) = read(&file);
    assert!(near(middle, [0xCC, 0x22, 0x11]), "{middle:?}");
}

#[test]
fn a_project_with_an_empty_timeline_has_a_black_card_picture() {
    let temp = Temp::new();
    let mut draft = Draft::empty("draft");
    draft.canvas_config.width = 64;
    draft.canvas_config.height = 32;
    Folder::new(draft).save(&temp.path).unwrap();
    let mut desktop = shell();
    let file = settled(&mut desktop, &temp.path).expect("a card picture");
    let (size, middle) = read(&file);
    assert_eq!(size, (64, 32));
    assert!(near(middle, [0, 0, 0]), "{middle:?}");
}

#[test]
fn a_timeline_that_starts_with_a_gap_has_a_black_card_picture() {
    let temp = Temp::new();
    movie_project(&temp.path);
    // Move both clips a second later: nothing is at the start.
    let mut folder = Folder::read(&temp.path).unwrap();
    for segment in &mut folder.draft.tracks[0].segments {
        segment.target.start.0 += 1_000_000;
    }
    folder.draft.duration = cutlass_model::Micros(3_000_000);
    folder.save(&temp.path).unwrap();
    let mut desktop = shell();
    let file = settled(&mut desktop, &temp.path).expect("a card picture");
    let (size, middle) = read(&file);
    assert_eq!(size, (64, 32));
    assert!(near(middle, [0, 0, 0]), "{middle:?}");
}
