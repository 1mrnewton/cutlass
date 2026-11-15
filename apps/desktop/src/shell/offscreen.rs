//! Exports with no window: a compositor on a device opened without one.
//!
//! The window draws its exports on the preview's device, so an effect clip
//! is in the file. `cutlass mcp` has no window and no preview, so it asks
//! for a device of its own, which the first export opens: a host that only
//! edits opens none. On a machine that gives it none the export is still
//! written, composed on the CPU, which draws no recipe effect; the shell
//! then names the effect clips the file is missing, so the host can say so.

use std::fmt;
use std::path::Path;

use cutlass_model::{Draft, SegmentId, TimeRange};
use cutlass_render::{RenderError, Renderer, wgpu};

use super::Shell;

/// Opens a compositor on a device with no window, or says why it cannot.
pub type OpenRenderer = fn() -> Result<Renderer, RenderError>;

/// Where exports draw when the preview has no device.
pub(crate) enum Offscreen {
    /// Not asked for: exports draw on the preview's device, when it has one.
    Off,
    /// Asked for, and opened by the first export.
    Unopened(OpenRenderer),
    Open(Box<Renderer>),
    /// This machine opened none, and why.
    Absent(String),
}

/// The effect clips an export left out of its picture, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeftOut {
    /// Back to front as the draft holds them, a compound's own in its place.
    pub clips: Vec<SegmentId>,
    pub reason: String,
}

/// How many clip ids [`LeftOut`] names before it counts the rest.
const NAMED: usize = 8;

impl fmt::Display for LeftOut {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.clips.len();
        let (clips, are) = if count == 1 {
            ("clip", "is")
        } else {
            ("clips", "are")
        };
        let mut names: Vec<String> = self
            .clips
            .iter()
            .take(NAMED)
            .map(ToString::to_string)
            .collect();
        if count > NAMED {
            names.push(format!("and {} more", count - NAMED));
        }
        write!(
            formatter,
            "{count} effect {clips} {are} not in the picture ({}): {}",
            names.join(", "),
            self.reason
        )
    }
}

/// What an export that wrote `path` answers the assistant: the file, and
/// the effect clips it was written without, if any.
pub(crate) fn wrote(path: &Path, left_out: Option<&LeftOut>) -> String {
    match left_out {
        None => format!("wrote {}", path.display()),
        Some(left_out) => format!(
            "wrote {}, but {left_out}. Effects are drawn on a GPU, and none could be opened \
             on this machine.",
            path.display()
        ),
    }
}

impl Shell {
    /// Draw exports on a device `open` opens with no window, from the
    /// first export on, as the host with no window does. A shell whose
    /// preview has a device keeps drawing exports on that one.
    pub fn draw_exports_offscreen(&mut self, open: OpenRenderer) {
        self.offscreen = Offscreen::Unopened(open);
    }

    /// Open the device asked for with [`Self::draw_exports_offscreen`],
    /// once, when the preview has none. A machine that gives none is asked
    /// only this once.
    pub(super) fn open_offscreen(&mut self) {
        let Offscreen::Unopened(open) = self.offscreen else {
            return;
        };
        if self.gpu.is_some() {
            return;
        }
        self.offscreen = match open() {
            Ok(renderer) => {
                cutlass_logger::info!("export: drawing on a GPU device opened with no window");
                Offscreen::Open(Box::new(renderer))
            }
            Err(error) => {
                cutlass_logger::warn!("export: effect clips will not be drawn: {error}");
                Offscreen::Absent(error.to_string())
            }
        };
    }

    /// The compositor exports draw through: the preview's, or with no
    /// window the one [`Self::open_offscreen`] opened. `None` composes on
    /// the CPU, which draws no recipe effect.
    pub(super) fn export_renderer(&self) -> Option<&Renderer> {
        match (&self.gpu, &self.offscreen) {
            (Some(preview), _) => Some(preview.renderer()),
            (None, Offscreen::Open(renderer)) => Some(renderer),
            (None, Offscreen::Off | Offscreen::Unopened(_) | Offscreen::Absent(_)) => None,
        }
    }

    /// [`Self::export_renderer`]'s device, for a job to draw on from its
    /// own thread.
    pub(super) fn export_gpu(&self) -> Option<(wgpu::Device, wgpu::Queue)> {
        self.export_renderer()
            .map(|renderer| (renderer.device().clone(), renderer.queue().clone()))
    }

    /// What an export of `ranges` of `draft` from this shell leaves out:
    /// the recipe effect clips there, when it has no compositor to draw
    /// them on. `None` when it has one, or nothing there needs one.
    pub(super) fn left_out(&self, draft: &Draft, ranges: &[&TimeRange]) -> Option<LeftOut> {
        if self.export_renderer().is_some() {
            return None;
        }
        let mut clips: Vec<SegmentId> = Vec::new();
        for range in ranges {
            for clip in cutlass_eval::recipe_effect_clips(draft, range) {
                if !clips.contains(&clip) {
                    clips.push(clip);
                }
            }
        }
        if clips.is_empty() {
            return None;
        }
        let reason = match &self.offscreen {
            Offscreen::Absent(reason) => reason.clone(),
            Offscreen::Off | Offscreen::Unopened(_) | Offscreen::Open(_) => {
                "no GPU device was opened to draw them on".to_owned()
            }
        };
        Some(LeftOut { clips, reason })
    }

    /// What the last finished export, which wrote `path`, answers the
    /// assistant: the file, and the effect clips it was written without.
    pub(crate) fn export_wrote(&self, path: &Path) -> String {
        wrote(path, self.export_left_out.as_ref())
    }

    /// The effect clips the last finished export left out of its picture,
    /// until the next export starts. `None` when it drew every clip.
    #[must_use]
    pub fn export_left_out(&self) -> Option<&LeftOut> {
        self.export_left_out.as_ref()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicU32, Ordering};

    use cutlass_decode::CodecId;
    use cutlass_model::{
        Draft, MaterialColorConfig, MaterialVideo, MaterialVideoEffect, Micros, Segment, SegmentId,
        TimeRange, Track, TrackType,
    };
    use cutlass_render::{RenderError, Renderer};
    use cutlass_store::Folder;

    use super::LeftOut;
    use crate::shell::tests::{Temp, land_job, shell};
    use crate::shell::{ExportRequest, Shell};

    /// A fifth of a second of grey on a 16 by 16 canvas, with the vignette
    /// clip `fx` over its first half.
    pub(crate) fn vignette_draft() -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 16;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "grey".into(),
            is_color_clip: true,
            solid_color: "0xFF808080".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("grey", "grey", TimeRange::new(0, 200_000)));
        draft.materials.video_effects.push(MaterialVideoEffect {
            id: "fx".into(),
            effect_id: "recipe/vignette".to_owned(),
            ..MaterialVideoEffect::default()
        });
        let mut row = Track::main("effects");
        row.kind = TrackType::Effect;
        let mut effect = Segment::new("fx", "fx", TimeRange::new(0, 100_000));
        effect.render_index = 11_000;
        row.segments.push(effect);
        draft.tracks.push(row);
        draft.duration = Micros(200_000);
        draft
    }

    /// The red of the middle and of a corner of `rgba`, `width` wide.
    fn middle_and_corner(width: u32, height: u32, rgba: &[u8]) -> (u32, u32) {
        let at = |x: u32, y: u32| u32::from(rgba[usize::try_from((y * width + x) * 4).unwrap()]);
        (at(width / 2, height / 2), at(0, 0))
    }

    /// [`middle_and_corner`] of the PNG at `path`.
    pub(crate) fn still_middle_and_corner(path: &Path) -> (u32, u32) {
        let picture = image::open(path).unwrap().to_rgba8();
        middle_and_corner(picture.width(), picture.height(), picture.as_raw())
    }

    /// [`middle_and_corner`] of the first picture of the movie at `path`,
    /// read as a clip of a project of its own.
    pub(crate) fn movie_middle_and_corner(path: &Path) -> (u32, u32) {
        let temp = Temp::new();
        let mut draft = Draft::empty("reads");
        draft.canvas_config.width = 64;
        draft.canvas_config.height = 64;
        let mut video = MaterialVideo::new("movie");
        video.path = path.display().to_string();
        video.duration = Micros(100_000);
        draft.materials.videos.push(video);
        let mut segment = Segment::new("movie", "movie", TimeRange::new(0, 100_000));
        segment.source_range = Some(TimeRange::new(0, 100_000));
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = desktop.preview_bytes().unwrap();
        assert!(
            picture.rgba.iter().any(|byte| *byte != 0),
            "the movie did not open"
        );
        middle_and_corner(picture.width, picture.height, &picture.rgba)
    }

    fn open(draft: Draft, temp: &Temp) -> Shell {
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    fn export(desktop: &mut Shell, path: &Path, codec: CodecId) {
        desktop
            .export(ExportRequest {
                path: path.to_path_buf(),
                short_side: 480,
                codec,
                ..ExportRequest::default()
            })
            .unwrap();
        land_job(desktop);
        assert!(path.is_file(), "{}", desktop.notice());
    }

    /// This machine has no GPU device.
    pub(crate) fn absent() -> Result<Renderer, RenderError> {
        Err(RenderError::NoDevice("none here".to_owned()))
    }

    fn fx_left_out(reason: &str) -> LeftOut {
        LeftOut {
            clips: vec![SegmentId::new("fx")],
            reason: reason.to_owned(),
        }
    }

    #[test]
    fn with_no_window_an_export_draws_an_effect_clip_on_a_device_of_its_own() {
        if Renderer::for_test().expect("a GPU is required").is_none() {
            eprintln!("skipped: no GPU");
            return;
        }
        let temp = Temp::new();
        let mut desktop = open(vignette_draft(), &temp);
        desktop.draw_exports_offscreen(Renderer::headless);
        let still = temp.path.join("out.png");
        export(&mut desktop, &still, CodecId::Png);
        let (middle, corner) = still_middle_and_corner(&still);
        assert!(corner + 40 < middle, "corner {corner}, middle {middle}");
        assert_eq!(desktop.export_left_out(), None);
        let movie = temp.path.join("out.mp4");
        export(&mut desktop, &movie, CodecId::H264);
        let (middle, corner) = movie_middle_and_corner(&movie);
        assert!(corner + 40 < middle, "corner {corner}, middle {middle}");
        assert_eq!(desktop.export_left_out(), None);
        // The picture saved at the playhead draws on that device too.
        let saved = temp.path.join("saved.png");
        desktop.export_still(&saved).unwrap();
        land_job(&mut desktop);
        let (middle, corner) = still_middle_and_corner(&saved);
        assert!(corner + 40 < middle, "corner {corner}, middle {middle}");
        assert_eq!(desktop.export_left_out(), None);
    }

    #[test]
    fn with_no_gpu_an_export_is_written_and_names_the_effect_clips_it_left_out() {
        static ASKED: AtomicU32 = AtomicU32::new(0);
        fn asked_and_absent() -> Result<Renderer, RenderError> {
            ASKED.fetch_add(1, Ordering::Relaxed);
            absent()
        }
        let temp = Temp::new();
        let mut desktop = open(vignette_draft(), &temp);
        desktop.draw_exports_offscreen(asked_and_absent);
        // Asking opens nothing: the first export does.
        assert_eq!(ASKED.load(Ordering::Relaxed), 0);
        let left_out = fx_left_out("no GPU device: none here");

        let still = temp.path.join("out.png");
        export(&mut desktop, &still, CodecId::Png);
        let (middle, corner) = still_middle_and_corner(&still);
        assert_eq!(corner, middle, "the vignette was drawn");
        assert_eq!(desktop.export_left_out(), Some(&left_out));

        let movie = temp.path.join("out.mp4");
        export(&mut desktop, &movie, CodecId::H264);
        assert_eq!(desktop.export_left_out(), Some(&left_out));

        let saved = temp.path.join("saved.png");
        desktop.export_still(&saved).unwrap();
        land_job(&mut desktop);
        assert!(saved.is_file());
        assert_eq!(desktop.export_left_out(), Some(&left_out));

        // Past the effect clip the frame is whole, and the last export's
        // clips are not carried over.
        desktop.seek(Micros(150_000));
        export(&mut desktop, &temp.path.join("late.png"), CodecId::Png);
        assert_eq!(desktop.export_left_out(), None);
        desktop
            .export_still(&temp.path.join("late-saved.png"))
            .unwrap();
        land_job(&mut desktop);
        assert_eq!(desktop.export_left_out(), None);

        // An export that fails names nothing, and leaves no file.
        desktop.seek(Micros(0));
        export(&mut desktop, &temp.path.join("again.png"), CodecId::Png);
        assert_eq!(desktop.export_left_out(), Some(&left_out));
        let refused = desktop.export(ExportRequest {
            path: still.clone(),
            short_side: 480,
            codec: CodecId::Png,
            ..ExportRequest::default()
        });
        assert!(refused.is_err(), "the file is there already");
        assert_eq!(desktop.export_left_out(), None);

        // The machine was asked for a device once, not once an export.
        assert_eq!(ASKED.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_draft_with_no_effect_clip_leaves_nothing_out_with_no_gpu() {
        let temp = Temp::new();
        let mut draft = vignette_draft();
        draft.tracks.truncate(1);
        let mut desktop = open(draft, &temp);
        desktop.draw_exports_offscreen(absent);
        export(&mut desktop, &temp.path.join("out.png"), CodecId::Png);
        assert_eq!(desktop.export_left_out(), None);
        export(&mut desktop, &temp.path.join("out.mp4"), CodecId::H264);
        assert_eq!(desktop.export_left_out(), None);
    }

    #[test]
    fn a_shell_not_asked_to_draw_offscreen_opens_no_device_and_says_so() {
        let temp = Temp::new();
        let mut desktop = open(vignette_draft(), &temp);
        let still = temp.path.join("out.png");
        export(&mut desktop, &still, CodecId::Png);
        let (middle, corner) = still_middle_and_corner(&still);
        assert_eq!(corner, middle);
        assert_eq!(
            desktop.export_left_out(),
            Some(&fx_left_out("no GPU device was opened to draw them on"))
        );
    }

    #[test]
    fn the_answer_names_the_clips_left_out_up_to_eight_then_counts_them() {
        let path = Path::new("/films/out.mp4");
        assert_eq!(super::wrote(path, None), "wrote /films/out.mp4");
        let one = fx_left_out("no GPU device: none here");
        assert_eq!(
            super::wrote(path, Some(&one)),
            "wrote /films/out.mp4, but 1 effect clip is not in the picture (fx): no GPU \
             device: none here. Effects are drawn on a GPU, and none could be opened on \
             this machine."
        );
        let many = LeftOut {
            clips: (0..11).map(|n| SegmentId::new(format!("fx{n}"))).collect(),
            reason: "why".to_owned(),
        };
        assert_eq!(
            many.to_string(),
            "11 effect clips are not in the picture (fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7, \
             and 3 more): why"
        );
    }
}
