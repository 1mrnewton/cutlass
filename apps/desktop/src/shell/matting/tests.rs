use std::path::PathBuf;

use cutlass_effects::StrokeStyle;
use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange, Track};
use cutlass_settings::Settings;
use cutlass_store::matte_file;

use super::*;

/// A project with a movie clip `v` and a photo clip `p` on a 1920 by 1080
/// canvas, the movie's file holding a few bytes.
fn open(name: &str) -> (Shell, PathBuf) {
    let temp = std::env::temp_dir().join(format!("cutlass-matting-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let mut draft = Draft::empty("draft");
    draft.fps = 10.0;
    draft.duration = Micros(2_000_000);
    for (id, kind) in [("m", "video"), ("still", "photo")] {
        draft.materials.videos.push(MaterialVideo {
            id: id.into(),
            kind: kind.into(),
            width: 1920,
            height: 1080,
            duration: Micros(2_000_000),
            path: "walk.mp4".into(),
            ..MaterialVideo::default()
        });
    }
    let mut clip = Segment::new("v", "m", TimeRange::new(0, 2_000_000));
    clip.source_range = Some(TimeRange::new(0, 2_000_000));
    draft.tracks[0].segments.push(clip);
    let mut overlay = Track::main("overlay");
    overlay.kind = cutlass_model::TrackType::Video;
    overlay
        .segments
        .push(Segment::new("p", "still", TimeRange::new(0, 1_000_000)));
    draft.tracks.push(overlay);
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    std::fs::write(temp.join("walk.mp4"), b"a movie").unwrap();
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    (shell, temp)
}

/// Keep clip `v`'s matte as `matting` would have it made, as a finished
/// job would.
fn keep_matte(shell: &Shell, temp: &Path, matting: &Matting) {
    let ask = matte_ask_with(
        shell.draft().unwrap(),
        Some(temp),
        &"v".into(),
        matting.clone(),
    )
    .unwrap();
    let file = matte_file(temp, &ask.key(), &ask.range).unwrap();
    std::fs::write(file, b"a matte movie").unwrap();
}

#[test]
fn auto_removal_with_its_matte_takes_its_edge_and_outline_at_once_and_undo_returns() {
    let (mut shell, temp) = open("auto");
    keep_matte(&shell, &temp, &Matting::auto());
    shell.select(Some("v".into()));
    assert_eq!(shell.matting(), None);
    assert!(!shell.matte_made());
    shell.set_auto_removal(true).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    assert_eq!(shell.matting(), Some(Matting::auto()));
    assert!(shell.matte_made());
    shell.set_matting_edge(0.4, 0.2).unwrap();
    shell
        .set_matting_stroke(Some(Stroke::new(StrokeStyle::Glow)))
        .unwrap();
    let edged = shell.matting().unwrap();
    assert_eq!((edged.feather, edged.clean_up), (0.4, 0.2));
    assert_eq!(edged.stroke.unwrap().style, StrokeStyle::Glow);
    assert!(
        shell.job().is_none(),
        "the edge and outline need no new matte"
    );
    shell.set_auto_removal(false).unwrap();
    assert_eq!(shell.matting(), None);
    shell.undo().unwrap();
    assert_eq!(
        shell.matting().unwrap().stroke.unwrap().style,
        StrokeStyle::Glow
    );
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_nothing_selected_or_an_edge_with_no_removal_is_refused() {
    let (mut shell, temp) = open("refused");
    shell.select(Some("p".into()));
    assert!(shell.set_auto_removal(true).is_err());
    shell.select(None);
    assert!(shell.set_auto_removal(true).is_err());
    shell.select(Some("v".into()));
    assert!(shell.set_matting_edge(0.5, 0.0).is_err());
    assert!(shell.set_matting_stroke(None).is_err());
    assert_eq!(shell.matting(), None);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn without_the_model_a_matte_that_is_needed_stores_nothing() {
    let (mut shell, temp) = open("no-model");
    shell.select(Some("v".into()));
    if auto_model_here() {
        // This machine has the model: the job starts instead.
        let _ = std::fs::remove_dir_all(&temp);
        return;
    }
    let before = shell.draft().unwrap().clone();
    assert!(shell.set_auto_removal(true).is_err());
    assert_eq!(shell.draft().unwrap(), &before);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn painted_strokes_become_one_pick_on_the_painted_frame() {
    let (mut shell, temp) = open("paint");
    shell.select(Some("v".into()));
    shell.seek(Micros(500_000));
    shell.start_painting().unwrap();
    shell.set_brush_size(0.0);
    // Across the canvas's middle, on a clip that fills it.
    shell.paint_at(480.0, 540.0).unwrap();
    shell.paint_at(960.0, 540.0).unwrap();
    shell.paint_at(960.0, 540.0).unwrap();
    shell.end_stroke();
    shell.set_brush(true, false);
    shell.paint_at(100.0, 100.0).unwrap();
    shell.end_stroke();
    let painting = shell.painting().unwrap();
    assert_eq!(painting.brushes.len(), 2);
    assert_eq!(painting.brushes[0].points, vec![[0.25, 0.5], [0.5, 0.5]]);
    assert!(painting.brushes[0].keep && !painting.brushes[0].smart);
    assert!(!painting.brushes[1].keep && painting.brushes[1].smart);
    assert!((painting.brushes[0].radius - BRUSH_SMALLEST).abs() < 1e-12);
    // The preview's copy, in canvas pixels: the frame fills the canvas
    // pixel for pixel, so the brush is its share of 1080.
    assert_eq!(painting.trails.len(), 2);
    assert_eq!(
        painting.trails[0].points,
        vec![(480.0, 540.0), (960.0, 540.0)]
    );
    assert!((painting.trails[0].radius - BRUSH_SMALLEST * 1080.0).abs() < 1e-6);
    assert!(!painting.trails[1].keep);
    let reach = shell.brush_on_canvas(960.0, 540.0).unwrap();
    assert!((reach - BRUSH_SMALLEST * 1080.0).abs() < 1e-6, "{reach}");
    shell.undo_stroke();
    assert_eq!(shell.painting().unwrap().brushes.len(), 1);
    assert_eq!(shell.painting().unwrap().trails.len(), 1);

    let expected = Matting {
        mode: MattingMode::Custom,
        picks: vec![Pick {
            frame: shell.painting().unwrap().source,
            clicks: Vec::new(),
            brushes: shell.painting().unwrap().brushes.clone(),
        }],
        ..Matting::default()
    };
    keep_matte(&shell, &temp, &expected);
    shell.apply_painting().unwrap();
    assert!(shell.painting().is_none());
    assert_eq!(shell.matting(), Some(expected));
    assert!(shell.job().is_none(), "its matte was there");
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn painting_nothing_applies_nothing() {
    let (mut shell, temp) = open("paint-nothing");
    shell.select(Some("v".into()));
    shell.start_painting().unwrap();
    shell.apply_painting().unwrap();
    assert!(shell.painting().is_none());
    assert_eq!(shell.matting(), None);
    assert!(shell.paint_at(1.0, 1.0).is_err(), "not painting");
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn picks_merge_by_frame_in_time_order() {
    let at = |frame: i64, x: f64| Pick {
        frame: Micros(frame),
        clicks: vec![PickClick {
            x,
            y: 0.5,
            keep: true,
        }],
        brushes: Vec::new(),
    };
    let mut picks = Vec::new();
    merge_pick(&mut picks, at(300, 0.1));
    merge_pick(&mut picks, at(100, 0.2));
    merge_pick(&mut picks, at(300, 0.3));
    let frames: Vec<i64> = picks.iter().map(|pick| pick.frame.0).collect();
    assert_eq!(frames, vec![100, 300]);
    assert_eq!(picks[1].clicks.len(), 2);
}

#[test]
fn a_clip_point_is_a_unit_point_of_its_frame() {
    assert_eq!(unit_of_clip((-1.0, 1.0)), (0.0, 0.0));
    assert_eq!(unit_of_clip((1.0, -1.0)), (1.0, 1.0));
    assert_eq!(unit_of_clip((0.0, 0.0)), (0.5, 0.5));
    assert_eq!(unit_of_clip((3.0, -3.0)), (1.0, 1.0));
}

#[test]
fn auto_removal_names_its_mattes_by_the_model_that_makes_them() {
    // A matte is found by the model's name, so eval's name for Auto
    // removal's model has to be the one the loaded model gives, at the
    // side it is loaded at.
    assert_eq!(
        cutlass_eval::MATTE_AUTO_MODEL,
        format!("birefnet-lite-{AUTO_SIDE}")
    );
    assert_eq!(AUTO_WEIGHTS, cutlass_vision::MATTING_WEIGHTS);
    assert_eq!(
        cutlass_eval::MATTE_FOLLOW_MODEL,
        cutlass_vision::FOLLOW_MODEL
    );
    assert_eq!(FOLLOW_WEIGHTS, cutlass_vision::FOLLOW_WEIGHTS);
}

#[test]
fn a_missing_model_is_refused_with_a_reason() {
    let empty = std::env::temp_dir().join(format!("cutlass-no-matter-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&empty);
    let Err(reason) = load_matter(&empty, Compute::Cpu) else {
        panic!("an empty folder has no model");
    };
    assert!(!reason.is_empty());
    let Err(reason) = load_follower(&empty, Compute::Cpu) else {
        panic!("an empty folder has no follower");
    };
    assert!(!reason.is_empty());
    let _ = std::fs::remove_dir_all(&empty);
}

/// The whole of Auto removal on a real clip, as the window does it: the
/// switch starts the job with BiRefNet_lite, the job lands its matte, and
/// the preview shows the children with the room gone, then with an edge
/// and an outline. Slow, so run by hand with `--ignored`; set
/// `CUTLASS_MATTE_OUT` to a folder to keep the frames.
#[test]
#[ignore = "reads a real clip and runs the real model"]
fn auto_removal_cuts_a_real_clip_in_the_preview() {
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    let clip = local.join("7150172-hd_720_1280_50fps.mp4");
    if !clip.is_file() || !auto_model_here() {
        return;
    }
    let temp = std::env::temp_dir().join(format!("cutlass-matting-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let mut draft = Draft::empty("draft");
    draft.fps = 25.0;
    (draft.canvas_config.width, draft.canvas_config.height) = (720, 1280);
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        width: 720,
        height: 1280,
        duration: Micros(7_820_000),
        path: "party.mp4".into(),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 1_000_000));
    segment.source_range = Some(TimeRange::new(2_000_000, 1_000_000));
    draft.tracks[0].segments.push(segment);
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    std::fs::copy(&clip, temp.join("party.mp4")).unwrap();
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    shell.select(Some("v".into()));
    shell.seek(Micros(500_000));
    let grey = |picture: &crate::shell::Picture, x: u32, y: u32| {
        let at = ((y * picture.width + x) * 4) as usize;
        let px = &picture.rgba[at..at + 4];
        (u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2])) / 3
    };
    let keep = |picture: &crate::shell::Picture, name: &str| {
        if let Some(out) = std::env::var_os("CUTLASS_MATTE_OUT") {
            image::RgbaImage::from_raw(picture.width, picture.height, picture.rgba.clone())
                .unwrap()
                .save(PathBuf::from(out).join(name))
                .unwrap();
        }
    };
    // The wall above the children, and the front girl's face.
    let (wall, face) = ((360, 60), (315, 645));
    let before = shell.preview_bytes().unwrap();
    keep(&before, "preview-before.png");
    assert!(grey(&before, wall.0, wall.1) > 150, "the wall is light");

    shell.set_auto_removal(true).unwrap();
    assert!(
        shell.matte_progress().is_some(),
        "the switch starts the job"
    );
    let start = std::time::Instant::now();
    while shell.poll_job() && start.elapsed().as_secs() < 600 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let took = start.elapsed();
    assert!(shell.matte_made(), "{:?}", shell.notice());
    let cut = shell.preview_bytes().unwrap();
    keep(&cut, "preview-cut.png");
    assert!(grey(&cut, wall.0, wall.1) < 20, "the wall is gone");
    assert!(
        grey(&cut, face.0, face.1).abs_diff(grey(&before, face.0, face.1)) < 12,
        "the face stays"
    );

    shell.set_matting_edge(0.3, 0.2).unwrap();
    shell
        .set_matting_stroke(Some(Stroke::new(StrokeStyle::Paper)))
        .unwrap();
    assert!(
        shell.job().is_none(),
        "the edge and outline need no new matte"
    );
    let outlined = shell.preview_bytes().unwrap();
    keep(&outlined, "preview-outlined.png");
    assert_ne!(outlined.rgba, cut.rgba);
    eprintln!("the matte job took {took:.2?}");

    // A wider trim: the new stretch has no matte, so one is made by itself
    // and the frame past the old end is cut too.
    shell
        .apply(Request::Trim {
            segment_id: "v".into(),
            edge: cutlass_session::TrimEdge::End,
            time: Micros(1_500_000),
        })
        .unwrap();
    assert!(!shell.matte_made());
    let start = std::time::Instant::now();
    while shell.poll_job() && start.elapsed().as_secs() < 600 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(shell.matte_made(), "{:?}", shell.notice());
    shell.seek(Micros(1_300_000));
    let later = shell.preview_bytes().unwrap();
    keep(&later, "preview-after-trim.png");
    assert!(
        grey(&later, wall.0, wall.1) < 20,
        "the wall is gone past the old end"
    );
    let _ = std::fs::remove_dir_all(&temp);
}

/// The whole of Custom removal on a real clip: three clicks on a dancer
/// against a green wall on the clip's first frame are followed through a
/// second of it by SAM 2.1 tiny, and the preview half a second on shows
/// her with the wall gone. Slow, so run by hand with `--ignored`; set
/// `CUTLASS_MATTE_OUT` to a folder to keep the frames.
#[test]
#[ignore = "reads a real clip and runs the real model"]
fn custom_removal_follows_real_clicks_through_a_clip() {
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    let clip = local.join("6905095-hd_1920_1080_25fps.mp4");
    if !clip.is_file() || !follow_model_here() {
        return;
    }
    let temp = std::env::temp_dir().join(format!("cutlass-follow-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let mut draft = Draft::empty("draft");
    draft.fps = 25.0;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        width: 1920,
        height: 1080,
        duration: Micros(76_926_667),
        path: "dancer.mp4".into(),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 1_000_000));
    segment.source_range = Some(TimeRange::new(2_000_000, 1_000_000));
    draft.tracks[0].segments.push(segment);
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    std::fs::copy(&clip, temp.join("dancer.mp4")).unwrap();
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    shell.select(Some("v".into()));
    shell.seek(Micros(500_000));
    let keep = |picture: &crate::shell::Picture, name: &str| {
        if let Some(out) = std::env::var_os("CUTLASS_MATTE_OUT") {
            image::RgbaImage::from_raw(picture.width, picture.height, picture.rgba.clone())
                .unwrap()
                .save(PathBuf::from(out).join(name))
                .unwrap();
        }
    };
    // Face, torso and shorts on the frame at 2 s of the file.
    let click = |x: f64, y: f64| PickClick { x, y, keep: true };
    let pick = Pick {
        frame: Micros(2_000_000),
        clicks: vec![
            click(0.444, 0.333),
            click(0.417, 0.556),
            click(0.417, 0.796),
        ],
        brushes: Vec::new(),
    };
    shell.add_pick("v".into(), pick, Vec::new()).unwrap();
    assert!(shell.matte_progress().is_some(), "a pick starts the job");
    let start = std::time::Instant::now();
    while shell.poll_job() && start.elapsed().as_secs() < 900 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let took = start.elapsed();
    assert!(shell.matte_made(), "{:?}", shell.notice());
    let cut = shell.preview_bytes().unwrap();
    keep(&cut, "follow-cut.png");
    let (w, h) = (cut.width as usize, cut.height as usize);
    let opaque = |x: usize, y: usize| cut.rgba[(y * w + x) * 4 + 3] > 128;
    let kept = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| opaque(x, y))
        .count();
    let left = (0..h)
        .flat_map(|y| (0..w / 4).map(move |x| (x, y)))
        .filter(|&(x, y)| opaque(x, y))
        .count();
    let share = kept as f64 / (w * h) as f64;
    // She fills a slice of the frame's middle; the wall left of her is gone.
    assert!((0.03..0.3).contains(&share), "{share}");
    assert!(left < w * h / 400, "{left} kept left of her");
    eprintln!("the follow job took {took:.2?}, {share:.3} of the frame kept");
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_wider_trim_makes_the_matte_again_once() {
    let (mut shell, temp) = open("sweep");
    shell.select(Some("v".into()));
    let trim = |shell: &mut Shell, end: i64| {
        shell
            .apply(Request::Trim {
                segment_id: "v".into(),
                edge: cutlass_session::TrimEdge::End,
                time: Micros(end),
            })
            .unwrap();
    };
    trim(&mut shell, 1_000_000);
    keep_matte(&shell, &temp, &Matting::auto());
    if !auto_model_here() {
        let _ = std::fs::remove_dir_all(&temp);
        return;
    }
    shell.set_auto_removal(true).unwrap();
    assert!(shell.job().is_none(), "its matte was there");
    assert!(!shell.poll_job(), "nothing is missing");
    // Wider than the matte: the job starts on the next poll, by itself.
    trim(&mut shell, 2_000_000);
    assert!(!shell.matte_made());
    assert!(shell.poll_job(), "the missing matte is being made");
    assert!(shell.matte_progress().is_some());
    // The movie here is a few bytes, so the job fails, and is not tried
    // again on the next edit.
    let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while shell.poll_job() && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(shell.job().is_none());
    shell.set_matting_edge(0.5, 0.0).unwrap();
    assert!(!shell.poll_job(), "a failed matte waits for the user");
    shell.make_matte().unwrap();
    assert!(shell.matte_progress().is_some(), "Make matte tries again");
    while shell.poll_job() && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let _ = std::fs::remove_dir_all(&temp);
}
