//! What the shell keeps for the effect studio: the document, the studio
//! folder's reads, the pictures the preview runs over, and the preview
//! itself, drawn by the same GPU blocks as the timeline.
//!
//! The preview runs over a bundled scene, drawn larger than a card shows
//! it, or over the designer's own clip: a short run of its frames decoded
//! on another thread, then looped under the effect.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cutlass_effects::PreviewScene;
use cutlass_model::Micros;

use super::Shell;
use crate::studio::{Doc, Studio};

/// How often the studio folder is looked at for a change.
const FOLDER_EVERY: Duration = Duration::from_millis(750);
/// How much larger than a card's scene the studio draws one, each way.
const SCENE_SCALE: u32 = 3;
/// The rows a frame of the designer's own clip is shrunk to.
const CLIP_ROWS: u32 = 432;
/// The frames of the clip kept for a second of it.
pub const CLIP_RATE: u32 = 12;
/// The most seconds of the clip that are kept.
const CLIP_SECONDS: f32 = 4.0;

/// What the preview runs over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backdrop {
    /// One of the scenes the app ships, by name.
    Scene(String),
    /// The designer's own clip, once it is loaded.
    Clip,
}

/// The designer's own clip: a short run of its frames, all one size.
struct Clip {
    path: PathBuf,
    frames: Vec<PreviewScene>,
}

/// A clip on its way: the file, and where its frames arrive.
struct Loading {
    path: PathBuf,
    frames: mpsc::Receiver<Result<Vec<PreviewScene>, String>>,
}

/// The studio as the shell holds it.
pub(crate) struct StudioState {
    doc: Studio,
    checked: Option<Instant>,
    clip: Option<Clip>,
    loading: Option<Loading>,
    clip_error: Option<String>,
    /// The bundled scenes at the studio's size, made when first shown.
    scenes: HashMap<(String, u32, u32), PreviewScene>,
}

impl StudioState {
    pub(crate) fn new() -> Self {
        Self {
            doc: Studio::new(None),
            checked: None,
            clip: None,
            loading: None,
            clip_error: None,
            scenes: HashMap::new(),
        }
    }
}

/// `scene` as a picture `width` by `height`: scaled to cover it, the
/// overflow cropped evenly, each pixel a blend of the four nearest.
fn covered(scene: &PreviewScene, width: u32, height: u32) -> PreviewScene {
    let (source_w, source_h) = (scene.width.max(1) as f32, scene.height.max(1) as f32);
    let scale = (width as f32 / source_w).max(height as f32 / source_h);
    let (offset_x, offset_y) = (
        (source_w - width as f32 / scale) / 2.0,
        (source_h - height as f32 / scale) / 2.0,
    );
    let at = |x: usize, y: usize, channel: usize| {
        let index = (y * scene.width as usize + x) * 4 + channel;
        f32::from(scene.rgba.get(index).copied().unwrap_or(0))
    };
    let (last_x, last_y) = (
        scene.width.saturating_sub(1) as usize,
        scene.height.saturating_sub(1) as usize,
    );
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        let source_y = ((y as f32 + 0.5) / scale + offset_y - 0.5).clamp(0.0, last_y as f32);
        let (top, down) = (source_y.floor() as usize, source_y.fract());
        let bottom = (top + 1).min(last_y);
        for x in 0..width {
            let source_x = ((x as f32 + 0.5) / scale + offset_x - 0.5).clamp(0.0, last_x as f32);
            let (left, across) = (source_x.floor() as usize, source_x.fract());
            let right = (left + 1).min(last_x);
            for channel in 0..4 {
                let above =
                    at(left, top, channel) * (1.0 - across) + at(right, top, channel) * across;
                let below = at(left, bottom, channel) * (1.0 - across)
                    + at(right, bottom, channel) * across;
                rgba.push(
                    (above * (1.0 - down) + below * down)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                );
            }
        }
    }
    PreviewScene {
        width,
        height,
        rgba,
    }
}

/// The source times of the frames kept for a clip `length` long.
fn clip_times(length: Option<Micros>) -> Vec<Micros> {
    let seconds = length.map_or(CLIP_SECONDS, |length| {
        (length.0 as f32 / 1_000_000.0).clamp(0.0, CLIP_SECONDS)
    });
    let count = ((seconds * CLIP_RATE as f32).floor() as i64).max(1);
    (0..count)
        .map(|frame| Micros(frame * 1_000_000 / i64::from(CLIP_RATE)))
        .collect()
}

impl Shell {
    /// Read and save the studio's recipes under `folder` from now on, and
    /// read what it holds.
    pub fn keep_studio_folder(&mut self, folder: PathBuf) {
        self.studio.doc = Studio::new(Some(folder));
        self.studio.doc.reload();
        self.studio.checked = Some(Instant::now());
    }

    /// The studio's document and what is picked in it.
    #[must_use]
    pub fn studio(&self) -> &Studio {
        &self.studio.doc
    }

    pub fn studio_mut(&mut self) -> &mut Studio {
        &mut self.studio.doc
    }

    /// Whether anything is being made: a document is open, or the studio
    /// folder holds a file. A shell that draws only when asked keeps
    /// looking at the folder while this is so.
    #[must_use]
    pub fn studio_in_use(&self) -> bool {
        self.studio.doc.doc().is_some()
            || !self.studio.doc.refused().is_empty()
            || !cutlass_effects::studio_recipes().is_empty()
            || !cutlass_effects::studio_transitions().is_empty()
    }

    /// Look at the studio folder when it is time to, and take a clip's
    /// frames when they have arrived. True when either changed what there
    /// is to draw.
    pub fn poll_studio(&mut self) -> bool {
        let mut changed = false;
        let due = self
            .studio
            .checked
            .is_none_or(|checked| checked.elapsed() >= FOLDER_EVERY);
        if due {
            self.studio.checked = Some(Instant::now());
            changed |= self.studio.doc.reload();
        }
        if let Some(loading) = &self.studio.loading {
            match loading.frames.try_recv() {
                Ok(Ok(frames)) => {
                    self.studio.clip = Some(Clip {
                        path: loading.path.clone(),
                        frames,
                    });
                    self.studio.loading = None;
                    changed = true;
                }
                Ok(Err(error)) => {
                    self.studio.clip_error = Some(error);
                    self.studio.loading = None;
                    changed = true;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.studio.clip_error = Some("the clip could not be read".to_owned());
                    self.studio.loading = None;
                    changed = true;
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        changed
    }

    /// Decode the start of `path` on another thread as the designer's own
    /// clip. Its frames replace the clip in use once they have all arrived.
    pub fn load_studio_clip(&mut self, path: PathBuf) {
        let (sender, frames) = mpsc::channel();
        let readers = self.readers.clone();
        let file = path.clone();
        let spawned = std::thread::Builder::new()
            .name("cutlass-studio-clip".to_owned())
            .spawn(move || {
                let loaded = (|| {
                    let length = readers.open(&file).map_err(|error| error.to_string())?;
                    let times = clip_times(length.info().duration);
                    drop(length);
                    let frames = cutlass_engine::frames_of(&readers, &file, &times, CLIP_ROWS)
                        .map_err(|error| error.to_string())?;
                    Ok(frames
                        .into_iter()
                        .map(|frame| PreviewScene {
                            width: frame.width,
                            height: frame.height,
                            rgba: frame.rgba,
                        })
                        .collect())
                })();
                // The studio may have closed; then nobody is waiting.
                let _ = sender.send(loaded);
            });
        self.studio.clip_error = None;
        match spawned {
            Ok(_) => self.studio.loading = Some(Loading { path, frames }),
            Err(error) => self.studio.clip_error = Some(error.to_string()),
        }
    }

    /// The file of the designer's own clip, once its frames are in.
    #[must_use]
    pub fn studio_clip(&self) -> Option<&Path> {
        self.studio.clip.as_ref().map(|clip| clip.path.as_path())
    }

    /// The file whose frames are on their way.
    #[must_use]
    pub fn studio_clip_loading(&self) -> Option<&Path> {
        self.studio
            .loading
            .as_ref()
            .map(|loading| loading.path.as_path())
    }

    /// Why the last clip could not be used.
    #[must_use]
    pub fn studio_clip_error(&self) -> Option<&str> {
        self.studio.clip_error.as_deref()
    }

    /// The size the preview draws at over `backdrop`.
    fn studio_size(&self, backdrop: &Backdrop) -> Option<(u32, u32)> {
        match backdrop {
            Backdrop::Clip => {
                let frame = self.studio.clip.as_ref()?.frames.first()?;
                Some((frame.width, frame.height))
            }
            Backdrop::Scene(name) => {
                let scene = cutlass_effects::preview_scene(name)?;
                Some((scene.width * SCENE_SCALE, scene.height * SCENE_SCALE))
            }
        }
    }

    /// Make sure the scene `name` is held at `size`.
    fn hold_scene(&mut self, name: &str, size: (u32, u32)) -> Option<()> {
        let key = (name.to_owned(), size.0, size.1);
        if !self.studio.scenes.contains_key(&key) {
            let scene = cutlass_effects::preview_scene(name)?;
            // A handful of scenes at a size or two: the designer's clip
            // changing size is the only way to grow this.
            if self.studio.scenes.len() >= 48 {
                self.studio.scenes.clear();
            }
            self.studio
                .scenes
                .insert(key, covered(scene, size.0, size.1));
        }
        Some(())
    }

    /// The studio's document at `seconds` of its loop, drawn over
    /// `backdrop`, as a texture egui shows as is. With `plain` an effect is
    /// left off and the backdrop shows alone. A transition goes from
    /// `backdrop` to the scene `to` at `seconds` taken as its progress, 0
    /// to 1. None without a GPU, without a document, or while the clip is
    /// not loaded.
    pub fn studio_frame(
        &mut self,
        backdrop: &Backdrop,
        to: &str,
        seconds: f64,
        plain: bool,
    ) -> Option<(egui::TextureId, [u32; 2])> {
        let size = self.studio_size(backdrop)?;
        if let Backdrop::Scene(name) = backdrop {
            self.hold_scene(name, size)?;
        }
        let transition = matches!(self.studio.doc.doc()?, Doc::Transition(_));
        if transition {
            self.hold_scene(to, size)?;
        }
        let studio = &self.studio;
        let from = match backdrop {
            Backdrop::Clip => {
                let frames = &studio.clip.as_ref()?.frames;
                let frame = (seconds.max(0.0) * f64::from(CLIP_RATE)).floor() as usize;
                // A transition leaves the clip's first frame.
                let frame = if transition {
                    0
                } else {
                    frame % frames.len().max(1)
                };
                frames.get(frame)?
            }
            Backdrop::Scene(name) => studio.scenes.get(&(name.clone(), size.0, size.1))?,
        };
        let gpu = self.gpu.as_mut()?;
        if transition {
            let to = studio.scenes.get(&(to.to_owned(), size.0, size.1))?;
            let crossing = studio.doc.preview_crossing(seconds as f32)?;
            return gpu.studio_crossing(&crossing, from, to).ok();
        }
        let steps = if plain {
            Vec::new()
        } else {
            let time = Micros((seconds.max(0.0) * 1_000_000.0).round() as i64);
            studio
                .doc
                .preview_recipe()
                .map(|recipe| recipe.steps_at(time))
                .unwrap_or_default()
        };
        gpu.studio_effect(&steps, from).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{CLIP_RATE, clip_times, covered};
    use cutlass_effects::PreviewScene;
    use cutlass_model::Micros;

    #[test]
    fn a_scene_is_scaled_to_cover_the_size_asked() {
        // Left half red, right half blue.
        let mut rgba = Vec::new();
        for _ in 0..2 {
            rgba.extend_from_slice(&[255, 0, 0, 255, 255, 0, 0, 255]);
            rgba.extend_from_slice(&[0, 0, 255, 255, 0, 0, 255, 255]);
        }
        let scene = PreviewScene {
            width: 4,
            height: 2,
            rgba,
        };
        let larger = covered(&scene, 12, 6);
        assert_eq!((larger.width, larger.height), (12, 6));
        assert_eq!(larger.rgba.len(), 12 * 6 * 4);
        assert_eq!(&larger.rgba[..4], &[255, 0, 0, 255]);
        assert_eq!(&larger.rgba[11 * 4..12 * 4], &[0, 0, 255, 255]);
        // A square crop of the wide picture keeps its middle: half and half.
        let square = covered(&scene, 6, 6);
        assert_eq!(&square.rgba[..4], &[255, 0, 0, 255]);
        assert_eq!(&square.rgba[5 * 4..6 * 4], &[0, 0, 255, 255]);
        // An empty picture gives clear pixels, not a panic.
        let empty = PreviewScene {
            width: 0,
            height: 0,
            rgba: Vec::new(),
        };
        assert_eq!(covered(&empty, 2, 1).rgba, vec![0; 8]);
    }

    #[test]
    fn the_preview_draws_the_document_over_a_scene_and_over_the_designers_clip() {
        use super::Backdrop;
        use crate::studio::{CurveKind, Slot, Stack};

        let mut desktop = crate::shell::tests::shell();
        let dancer = Backdrop::Scene("dancer".to_owned());
        // Nothing open, and no GPU: nothing to draw, and nothing in use.
        assert!(desktop.studio_frame(&dancer, "beach", 0.0, false).is_none());
        assert!(!desktop.studio_in_use());
        assert!(!desktop.poll_studio(), "the studio folder is off in a test");
        desktop.studio_mut().new_effect();
        assert!(desktop.studio_in_use());
        assert!(desktop.studio_frame(&dancer, "beach", 0.0, false).is_none());

        // The designer's clip: a picture decodes to one frame on another
        // thread; a file that is not media is named as the reason.
        let picture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/effects/previews/beach.jpg");
        desktop.load_studio_clip(picture.clone());
        assert_eq!(desktop.studio_clip_loading(), Some(picture.as_path()));
        let wait = |desktop: &mut crate::shell::Shell| {
            let start = std::time::Instant::now();
            while desktop.studio_clip_loading().is_some() {
                assert!(start.elapsed().as_secs() < 30, "the clip never arrived");
                desktop.poll_studio();
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        };
        wait(&mut desktop);
        assert_eq!(desktop.studio_clip_error(), None);
        assert_eq!(desktop.studio_clip(), Some(picture.as_path()));
        desktop.load_studio_clip(std::path::PathBuf::from(file!()));
        wait(&mut desktop);
        assert!(desktop.studio_clip_error().is_some());
        assert_eq!(
            desktop.studio_clip(),
            Some(picture.as_path()),
            "a clip that fails leaves the one in use"
        );

        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(state) = pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) else {
            eprintln!("skipped: no GPU");
            return;
        };
        desktop.attach_gpu(&state);
        // A scene is drawn three times a card's size each way.
        let (_, size) = desktop
            .studio_frame(&dancer, "beach", 0.5, false)
            .expect("the effect over a scene");
        assert_eq!(size, [288 * 3, 192 * 3]);
        assert!(desktop.studio_frame(&dancer, "beach", 0.5, true).is_some());
        // A scene the app does not ship draws nothing.
        let nowhere = Backdrop::Scene("nowhere".to_owned());
        assert!(
            desktop
                .studio_frame(&nowhere, "beach", 0.5, false)
                .is_none()
        );
        // The clip is drawn at its own size, whatever time is asked.
        let (_, size) = desktop
            .studio_frame(&Backdrop::Clip, "beach", 123.0, false)
            .expect("the effect over the clip");
        assert_eq!(size, [288, 192]);
        // Every pass muted: the picture alone, not nothing.
        desktop.studio_mut().set_muted(Stack::Effect, 0, true);
        assert!(desktop.studio_frame(&dancer, "beach", 0.5, false).is_some());

        // A transition crosses from the backdrop to the scene named, and
        // from the clip to a scene made the clip's size.
        desktop.studio_mut().new_transition();
        desktop
            .studio_mut()
            .add_pass(Stack::Outgoing, cutlass_effects::Block::Blur);
        let blur = Slot::pass(Stack::Outgoing, 0, "radius");
        desktop.studio_mut().set_kind(&blur, CurveKind::Keys);
        desktop.studio_mut().move_key(&blur, 1, 1.0, 0.05);
        for progress in [0.0, 0.5, 1.0] {
            let (_, size) = desktop
                .studio_frame(&dancer, "beach", progress, false)
                .expect("the transition between scenes");
            assert_eq!(size, [288 * 3, 192 * 3]);
        }
        assert!(
            desktop
                .studio_frame(&Backdrop::Clip, "beach", 0.5, false)
                .is_some()
        );
        assert!(
            desktop
                .studio_frame(&dancer, "nowhere", 0.5, false)
                .is_none()
        );
    }

    #[test]
    fn a_clip_keeps_a_few_seconds_of_frames() {
        let times = clip_times(Some(Micros(1_000_000)));
        assert_eq!(times.len(), CLIP_RATE as usize);
        assert_eq!(times[0], Micros(0));
        assert!(times.windows(2).all(|pair| pair[0] < pair[1]));
        // A long clip is cut off, and one with no length or no frames
        // still gives a picture.
        assert_eq!(
            clip_times(Some(Micros(3_600_000_000))).len(),
            4 * CLIP_RATE as usize
        );
        assert_eq!(clip_times(None).len(), 4 * CLIP_RATE as usize);
        assert_eq!(clip_times(Some(Micros(0))), vec![Micros(0)]);
        assert!(clip_times(None).len() <= cutlass_engine::MAX_FRAMES);
    }
}
