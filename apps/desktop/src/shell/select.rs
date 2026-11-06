//! Click-to-select on the preview: the user clicks the object in the
//! selected clip, and the clip is cut down to it.
//!
//! Picking starts on the frame under the playhead. Each click adds a point
//! (a plain click keeps, a shift-click removes) and runs a job: the model
//! loads once and stays, the clip's own frame is decoded and read once per
//! frame, and the clicks so far become one `SetSelection`. A result for a
//! project that moved meanwhile is dropped.

use std::path::PathBuf;

use cutlass_engine::{DecodedFrame, DisplayQuarter, ReaderFactory};
use cutlass_model::{MaterialId, MaterialRef, Micros, SegmentId, TimeRange};
use cutlass_picture::Picture;
use cutlass_session::Request;
use cutlass_vision::{Click, Compute, Image, MobileSam, SelectJob, Selector};

use super::{Landing, Shell, ShellError};

/// The folder inside `Models` that holds the selection model.
pub(crate) const VISION_FOLDER: &str = "mobile-sam";

/// The selection model, and the frame it last read.
pub(crate) struct VisionModel {
    sam: MobileSam,
    read: Option<FrameKey>,
    /// The size of the frame it last read.
    size: (u32, u32),
}

/// One frame of one file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FrameKey {
    path: PathBuf,
    source: Micros,
}

/// The object being picked: the clip, the frame, and the clicks so far.
#[derive(Clone, Debug, PartialEq)]
pub struct Picking {
    pub segment_id: SegmentId,
    /// The timeline time picking started at.
    pub time: Micros,
    source: Micros,
    path: PathBuf,
    pub clicks: Vec<PickClick>,
}

/// One click: where it is on the canvas, where that is in the clip's own
/// picture (-1 to 1, y up), and whether it keeps or removes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickClick {
    pub canvas: (f64, f64),
    pub clip: (f64, f64),
    pub keep: bool,
}

/// The folder holding the selection model: the one in `Models`, or in a
/// debug build the checkout's `local-files` copy when `Models` has none.
pub(super) fn vision_model_dir() -> PathBuf {
    let installed = Shell::home().models().join(VISION_FOLDER);
    if has_weights(&installed) {
        return installed;
    }
    #[cfg(debug_assertions)]
    {
        let checkout =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/vision");
        if has_weights(&checkout) {
            return checkout;
        }
    }
    installed
}

fn has_weights(dir: &std::path::Path) -> bool {
    dir.join(cutlass_vision::ENCODER_WEIGHTS).is_file()
        && dir.join(cutlass_vision::DECODER_WEIGHTS).is_file()
}

/// Whether the selection model is on this machine.
pub(super) fn vision_model_here() -> bool {
    has_weights(&vision_model_dir())
}

impl Shell {
    /// The object being picked, for the preview to mark its clicks.
    pub fn picking(&self) -> Option<&Picking> {
        self.picking.as_ref()
    }

    /// Whether the selection model is on this machine.
    pub fn vision_model_ready(&self) -> bool {
        vision_model_here()
    }

    /// Whether the selected clip can be picked from: a video clip under the
    /// playhead, unlocked, with the selection model on this machine.
    pub fn pick_enabled(&self) -> bool {
        vision_model_here() && self.pick_target().is_ok()
    }

    /// Start picking an object in the selected clip, on the frame under the
    /// playhead.
    pub fn start_picking(&mut self) -> Result<(), ShellError> {
        if !vision_model_here() {
            return Err(ShellError::Unavailable(
                "the selection model is not on this machine".to_owned(),
            ));
        }
        self.picking = Some(self.pick_target()?);
        Ok(())
    }

    /// Stop picking. The selection made so far stays on the clip.
    pub fn stop_picking(&mut self) {
        self.picking = None;
    }

    /// Add a click at canvas pixel `x`, `y`, keeping the object under it or,
    /// with `keep` false, taking it out, and select again.
    pub fn pick_at(&mut self, x: f64, y: f64, keep: bool) -> Result<(), ShellError> {
        let Some(picking) = self.picking.clone() else {
            return Err(ShellError::Unavailable("not picking an object".to_owned()));
        };
        self.refuse_busy()?;
        let draft = self.session_draft()?;
        let Some(segment) = super::placed(draft, &picking.segment_id).map(|(segment, _)| segment)
        else {
            self.picking = None;
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        let canvas = (
            u32::try_from(draft.canvas_config.width.max(1)).unwrap_or(1),
            u32::try_from(draft.canvas_config.height.max(1)).unwrap_or(1),
        );
        let clip = cutlass_eval::clip_point(draft, segment, picking.time, canvas, x, y)
            .filter(|(u, v)| (-1.0..=1.0).contains(u) && (-1.0..=1.0).contains(v))
            .ok_or_else(|| ShellError::Unavailable("click on the clip".to_owned()))?;
        let mut next = picking;
        next.clicks.push(PickClick {
            canvas: (x, y),
            clip,
            keep,
        });
        self.run_pick(next)
    }

    /// Take the selection off the selected clip.
    pub fn clear_selection(&mut self) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a selection belongs on a video clip")?;
        self.refuse_locked(&segment_id)?;
        if let Some(picking) = self.picking.as_mut() {
            picking.clicks.clear();
        }
        let material_id = MaterialId::new(self.fresh("selection"));
        self.apply(Request::SetSelection {
            segment_id,
            material_id,
            selection: None,
        })
    }

    /// The selected clip as a picking target at the playhead.
    fn pick_target(&self) -> Result<Picking, ShellError> {
        let segment_id = self.picture_segment("a selection belongs on a video clip")?;
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        let Some(segment) = super::placed(draft, &segment_id).map(|(segment, _)| segment) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if !matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Video(_))
        ) {
            return Err(ShellError::Unavailable(
                "a selection belongs on a video clip".to_owned(),
            ));
        }
        let time = self.playhead();
        let source = cutlass_engine::placement_of(draft, segment)
            .ok()
            .and_then(|map| cutlass_engine::map_timeline(&map, time).ok())
            .ok_or_else(|| ShellError::Unavailable("move the playhead onto the clip".to_owned()))?;
        let path = self
            .media_file(cutlass_engine::media_path(draft, segment))
            .ok_or_else(|| ShellError::Unavailable("the clip has no media file".to_owned()))?;
        Ok(Picking {
            segment_id,
            time,
            source,
            path,
            clicks: Vec::new(),
        })
    }

    /// Select `picking`'s clicks on a job, and store the result.
    fn run_pick(&mut self, picking: Picking) -> Result<(), ShellError> {
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let job = SelectJob::start(session);
        let loaded = self.vision.take();
        let readers = self.readers.clone();
        let kernels = Shell::home().cache().join("kernels");
        let material_id = MaterialId::new(self.fresh("selection"));
        let asked = picking.clone();
        self.picking = Some(picking);
        let started = self.start_job("Select object", move |progress| {
            let model = match loaded {
                Some(model) => Ok(model),
                None => {
                    progress.stage("Loading the selection model");
                    load_vision(&kernels)
                }
            };
            let (model, selection) = match model {
                Ok(mut model) => {
                    progress.stage("Finding the object");
                    let selection = pick(&mut model, &readers, &job, &asked);
                    (Some(model), selection)
                }
                Err(error) => (None, Err(error)),
            };
            let landing: Landing = Box::new(move |shell: &mut Shell| {
                if model.is_some() {
                    shell.vision = model;
                }
                let selection = selection?;
                if shell.revision() != Some(job.revision()) {
                    return Err(ShellError::Unavailable(
                        "the project changed while the object was found".to_owned(),
                    ));
                }
                shell.apply(Request::SetSelection {
                    segment_id: asked.segment_id,
                    material_id,
                    selection: Some(selection),
                })
            });
            landing
        });
        if started.is_err() {
            // The job did not take the clicks; take this one back.
            if let Some(picking) = self.picking.as_mut() {
                picking.clicks.pop();
            }
        }
        started
    }
}

/// Load the selection model; slow, so a job does it.
fn load_vision(kernels: &std::path::Path) -> Result<VisionModel, ShellError> {
    let compute = Compute::best();
    if compute == Compute::Gpu {
        // A missing cache folder only costs the warm-up on every launch.
        if let Err(error) = cutlass_vision::keep_kernels_in(kernels) {
            cutlass_logger::warn!("kernel cache: {error}");
        }
    }
    let sam = MobileSam::load(&vision_model_dir(), compute).map_err(|error| {
        cutlass_logger::error!("selection model: {error}");
        ShellError::Unavailable("the selection model could not be loaded".to_owned())
    })?;
    Ok(VisionModel {
        sam,
        read: None,
        size: (0, 0),
    })
}

/// Read `picking`'s frame unless the model already has, then select its clicks.
fn pick(
    model: &mut VisionModel,
    readers: &ReaderFactory,
    job: &SelectJob,
    picking: &Picking,
) -> Result<cutlass_effects::Selection, ShellError> {
    let key = FrameKey {
        path: picking.path.clone(),
        source: picking.source,
    };
    let refused = |error: cutlass_vision::JobError| {
        cutlass_logger::error!("select object: {error:?}");
        ShellError::Unavailable("the object could not be found".to_owned())
    };
    if model.read.as_ref() == Some(&key) {
        // The model still holds this frame: only the clicks are new.
        let (width, height) = model.size;
        let clicks = source_clicks(picking, width, height);
        let mask = model
            .sam
            .select(&clicks)
            .map_err(|error| refused(cutlass_vision::JobError::Vision(error)))?;
        return cutlass_effects::Selection::from_alpha(
            mask.width,
            mask.height,
            &mask.alpha,
            key.source,
        )
        .ok_or_else(|| ShellError::Unavailable("the object could not be found".to_owned()));
    }
    let (width, height, rgb) = frame_rgb(readers, &key)?;
    let image = Image {
        width,
        height,
        rgb: &rgb,
    };
    model.read = None;
    let clicks = source_clicks(picking, width, height);
    let selection = job
        .select(&mut model.sam, &image, key.source, &clicks)
        .map_err(refused)?;
    model.read = Some(key);
    model.size = (width, height);
    Ok(selection)
}

/// `picking`'s clicks in the pixels of a `width` by `height` frame.
fn source_clicks(picking: &Picking, width: u32, height: u32) -> Vec<Click> {
    #[allow(clippy::cast_precision_loss)]
    let (w, h) = (width as f32, height as f32);
    picking
        .clicks
        .iter()
        .map(|click| {
            let (u, v) = click.clip;
            #[allow(clippy::cast_possible_truncation)]
            let (x, y) = (((u as f32) + 1.0) * 0.5 * w, (1.0 - (v as f32)) * 0.5 * h);
            Click {
                x: x.clamp(0.0, w),
                y: y.clamp(0.0, h),
                keep: click.keep,
            }
        })
        .collect()
}

/// The picture of `key` as RGB, upright as the file asks it to be shown.
fn frame_rgb(readers: &ReaderFactory, key: &FrameKey) -> Result<(u32, u32, Vec<u8>), ShellError> {
    let unreadable = || ShellError::Unavailable("the clip's frame could not be read".to_owned());
    let mut media = readers.open(&key.path).map_err(|_| unreadable())?;
    let display = media.info().display;
    let length = media
        .info()
        .duration
        .map_or(i64::MAX / 2, |end| end.0.max(1));
    let held = media
        .decode_at(key.source, &TimeRange::new(0, length))
        .map_err(|_| unreadable())?;
    let frame: &DecodedFrame = &held.frame;
    let Ok(Picture::Cpu(picture)) = cutlass_engine::decoded_picture(frame) else {
        return Err(unreadable());
    };
    let (width, height) = (picture.width(), picture.height());
    Ok(upright(width, height, picture.rgba(), display))
}

/// RGBA `width` by `height` as RGB, turned by `display`.
fn upright(width: u32, height: u32, rgba: &[u8], display: DisplayQuarter) -> (u32, u32, Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    let (out_w, out_h) = match display {
        DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (h, w),
        DisplayQuarter::Zero | DisplayQuarter::Half => (w, h),
    };
    let mut rgb = vec![0; out_w * out_h * 3];
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = match display {
                DisplayQuarter::Zero => (x, y),
                DisplayQuarter::Clockwise90 => (h - 1 - y, x),
                DisplayQuarter::Half => (w - 1 - x, h - 1 - y),
                DisplayQuarter::Clockwise270 => (y, w - 1 - x),
            };
            let from = (y * w + x) * 4;
            let to = (oy * out_w + ox) * 3;
            if let (Some(source), Some(target)) =
                (rgba.get(from..from + 3), rgb.get_mut(to..to + 3))
            {
                target.copy_from_slice(source);
            }
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    (out_w as u32, out_h as u32, rgb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_settings::Settings;

    #[test]
    fn only_a_video_clip_can_be_picked_and_clicks_need_picking() {
        let temp =
            std::env::temp_dir().join(format!("cutlass-select-object-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.duration = Micros(1_000_000);
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "solid".into(),
                is_color_clip: true,
                solid_color: "0xFFFF0000".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "solid",
            "solid",
            TimeRange::new(0, 1_000_000),
        ));
        cutlass_store::Folder::new(draft).save(&temp).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&temp).unwrap();
        assert!(shell.pick_at(1.0, 1.0, true).is_err());
        shell.select(Some("solid".into()));
        assert!(!shell.pick_enabled());
        assert!(shell.pick_target().is_err());
        assert!(shell.picking().is_none());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn a_quarter_turned_frame_is_read_upright() {
        // Two pixels side by side, red then green.
        let rgba = [255, 0, 0, 255, 0, 255, 0, 255];
        assert_eq!(
            upright(2, 1, &rgba, DisplayQuarter::Zero),
            (2, 1, vec![255, 0, 0, 0, 255, 0])
        );
        // Turned clockwise, red is on top.
        assert_eq!(
            upright(2, 1, &rgba, DisplayQuarter::Clockwise90),
            (1, 2, vec![255, 0, 0, 0, 255, 0])
        );
        assert_eq!(
            upright(2, 1, &rgba, DisplayQuarter::Half),
            (2, 1, vec![0, 255, 0, 255, 0, 0])
        );
        assert_eq!(
            upright(2, 1, &rgba, DisplayQuarter::Clockwise270),
            (1, 2, vec![0, 255, 0, 255, 0, 0])
        );
    }
}
