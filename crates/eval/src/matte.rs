//! Remove BG, as eval sees it: which matte video cuts a clip, and the cut.
//!
//! [`matte_ask`] names the matte video: the stretch of the clip's file its
//! timeline shows, at the project's frame rate, made by Auto removal's
//! model ([`MATTE_AUTO_MODEL`]) or by Custom removal's follower
//! ([`MATTE_FOLLOW_MODEL`]) from the clip's picks. The job reads that
//! stretch through [`crate::EnhanceFrames`] and writes a grey picture a
//! frame, white kept, where `cutlass_store::matte_file` says. Once it is
//! there, a clip with matting is cut by it in preview and export alike,
//! with its edge cleaned and feathered and its outline drawn; until then it
//! draws uncut.

use std::path::{Path, PathBuf};

use cutlass_effects::{Matting, MattingMode, segment_matting, stretch_axis};
use cutlass_model::{Draft, MaterialRef, Micros, SegmentId, TimeRange};
use cutlass_picture::CpuBytes;
use cutlass_store::{MatteKey, find_matte};

use crate::enhance::{EnhanceAsk, EnhanceError, clip_stretch};
use crate::gpu::{GpuLayer, GpuSource, SelectionCut};

/// The model Auto removal's mattes are made with, and the size it reads
/// at. A matte made by another is not used.
pub const MATTE_AUTO_MODEL: &str = "birefnet-lite-1024";

/// The model Custom removal's mattes are made with.
pub const MATTE_FOLLOW_MODEL: &str = "sam2.1-hiera-tiny-1024";

/// What a matte video is made from: a stretch of one file at a frame rate,
/// by one model, from Custom removal's picks.
#[derive(Clone, Debug, PartialEq)]
pub struct MatteAsk {
    /// The clip's file on disk.
    pub source: PathBuf,
    /// The stretch the clip shows, in the file's own time.
    pub range: TimeRange,
    /// Pictures per second the stretch is read at: a still's one picture
    /// lasts one. A movie's matte has a picture for each of the file's own
    /// frames instead, at its time, so it fits the clip's picture at any
    /// speed, backwards and between the project's frames.
    pub rate: f64,
    /// The clip's settings when the job was asked for.
    pub matting: Matting,
    /// The file is a still picture: its matte is one picture, made from
    /// [`Self::range`], its first frame, whatever stretch the clip shows.
    pub still: bool,
}

impl MatteAsk {
    /// The model that makes this matte.
    #[must_use]
    pub fn model(&self) -> &'static str {
        match self.matting.mode {
            MattingMode::Auto => MATTE_AUTO_MODEL,
            MattingMode::Custom => MATTE_FOLLOW_MODEL,
        }
    }

    /// The name its matte video is kept under.
    #[must_use]
    pub fn key(&self) -> MatteKey {
        MatteKey {
            source: self.source.clone(),
            model: self.model().into(),
            rate: if self.still { self.rate } else { 0.0 },
            picks: self.matting.picks_key(),
        }
    }

    /// The same stretch as [`crate::EnhanceFrames`] reads it: the clip's
    /// own frames, upright.
    #[must_use]
    pub fn frames(&self) -> EnhanceAsk {
        EnhanceAsk {
            source: self.source.clone(),
            range: self.range.clone(),
            rate: self.rate,
            level: 0,
            denoise: 0,
            deflicker: 0,
            interpolate: 0,
            motion_blur: None,
            keyed_blur: Vec::new(),
        }
    }
}

/// What cutting clip `segment_id` out reads. A relative file stays inside
/// `root`. Refused when the clip has no matting, or Custom removal has no
/// pick yet.
pub fn matte_ask(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Result<MatteAsk, EnhanceError> {
    let (_, _, _, segment) = clip_stretch(draft, root, segment_id, "cut out")?;
    let matting = segment_matting(draft, segment).ok_or_else(|| EnhanceError::Clip {
        reason: "the clip has no Remove BG".into(),
    })?;
    matte_ask_with(draft, root, segment_id, matting)
}

/// What cutting clip `segment_id` out with `matting` would read, before
/// the clip has it: so a shell can tell whether a new matte is needed.
pub fn matte_ask_with(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
    matting: Matting,
) -> Result<MatteAsk, EnhanceError> {
    let (source, range, rate, segment) = clip_stretch(draft, root, segment_id, "cut out")?;
    let still = matches!(
        draft.materials.get(segment.material_id.as_str()),
        Some(MaterialRef::Video(video)) if video.kind == "photo"
    );
    // A photo shows one picture however long the clip: one matte cuts it
    // wherever it is trimmed.
    #[allow(clippy::cast_possible_truncation)]
    let range = if still {
        TimeRange::new(0, (1e6 / rate).ceil().max(1.0) as i64)
    } else {
        range
    };
    if !matting.ready() {
        return Err(EnhanceError::Clip {
            reason: "Custom removal has nothing picked yet".into(),
        });
    }
    Ok(MatteAsk {
        source,
        range,
        rate,
        matting,
        still,
    })
}

/// The matte video that cuts `segment_id` and where in the file it starts,
/// when the clip has matting and its matte is in the project.
pub(crate) fn matte_video(
    draft: &Draft,
    root: Option<&Path>,
    segment_id: &SegmentId,
) -> Option<(PathBuf, Micros)> {
    let root = root?;
    let ask = matte_ask(draft, Some(root), segment_id).ok()?;
    let (path, held) = find_matte(root, &ask.key(), &ask.range)?;
    Some((path, held.start))
}

/// `picture`, upright premultiplied RGBA, cut down to `matte`, a grey
/// picture stretched to its size, with `matting`'s edge and outline. A
/// matte of no size leaves it uncut.
pub(crate) fn cut_out(picture: CpuBytes, matte: &CpuBytes, matting: &Matting) -> CpuBytes {
    let (w, h) = (picture.width(), picture.height());
    let Some(mut alpha) = stretched_grey(matte, w, h) else {
        return picture;
    };
    cutlass_effects::matting::refine(&mut alpha, w, h, matting.clean_up, matting.feather);
    let mut rgba = picture.rgba().to_vec();
    cutlass_effects::matting::cut(&mut rgba, &alpha);
    if let Some(stroke) = &matting.stroke {
        let under = cutlass_effects::matting::stroke_layer(&alpha, w, h, stroke);
        over(&mut rgba, &under);
    }
    CpuBytes::new(w, h, rgba).unwrap_or(picture)
}

/// The GPU passes that cut a clip's upright picture, `size`, to `matte`
/// as [`cut_out`] does on the CPU: the cut, over the picture, and the
/// outline to draw beneath the cut picture. A matte with no edge work is
/// stretched by the GPU; one that is cleaned, feathered or outlined is
/// stretched and worked here first, as the CPU does. `None` for a matte or
/// a picture of no size.
pub(crate) fn matte_passes(
    matte: &CpuBytes,
    matting: &Matting,
    size: (u32, u32),
) -> Option<(GpuLayer, Option<GpuLayer>)> {
    let (w, h) = size;
    let (mw, mh) = (matte.width(), matte.height());
    if w == 0 || h == 0 || mw == 0 || mh == 0 {
        return None;
    }
    let plain = matting.clean_up == 0.0 && matting.feather == 0.0 && matting.stroke.is_none();
    let (alpha, aw, ah) = if plain {
        let red: Vec<u8> = matte
            .rgba()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| px[0])
            .collect();
        (red.len() == mw as usize * mh as usize).then_some(())?;
        (red, mw, mh)
    } else {
        let mut alpha = stretched_grey(matte, w, h)?;
        cutlass_effects::matting::refine(&mut alpha, w, h, matting.clean_up, matting.feather);
        (alpha, w, h)
    };
    let under = match &matting.stroke {
        Some(stroke) => {
            let rgba = cutlass_effects::matting::stroke_layer(&alpha, w, h, stroke);
            Some(GpuLayer::canvas(CpuBytes::new(w, h, rgba).ok()?))
        }
        None => None,
    };
    // Each canvas pixel centre to the matte point it shows, -1 to 1 with y
    // up: the matte covers the canvas exactly.
    #[allow(clippy::cast_precision_loss)]
    let frame = [2.0 / w as f32, 0.0, -1.0, 0.0, -2.0 / h as f32, 1.0];
    let cut = GpuLayer::cut(
        GpuSource::Select(SelectionCut {
            width: aw,
            height: ah,
            alpha,
            invert: false,
            frame,
        }),
        w,
        h,
    );
    Some((cut, under))
}

/// `under` drawn beneath `top`, both premultiplied RGBA of one size.
fn over(top: &mut [u8], under: &[u8]) {
    for (pixel, below) in top
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(under.as_chunks::<4>().0)
    {
        if below[3] == 0 || pixel[3] == 255 {
            continue;
        }
        let rest = 255 - u16::from(pixel[3]);
        for channel in 0..4 {
            #[allow(clippy::cast_possible_truncation)]
            let add = ((u16::from(below[channel]) * rest + 127) / 255) as u8;
            pixel[channel] = pixel[channel].saturating_add(add);
        }
    }
}

/// The red of `grey` at `width` by `height`, bilinear, centres matched.
fn stretched_grey(grey: &CpuBytes, width: u32, height: u32) -> Option<Vec<u8>> {
    let (gw, gh) = (grey.width() as usize, grey.height() as usize);
    let (w, h) = (width as usize, height as usize);
    let src = grey.rgba();
    if gw == 0 || gh == 0 || w == 0 || h == 0 || src.len() < gw * gh * 4 {
        return None;
    }
    if (gw, gh) == (w, h) {
        return Some(src.as_chunks::<4>().0.iter().map(|px| px[0]).collect());
    }
    let (xs, ys) = (stretch_axis(w, gw), stretch_axis(h, gh));
    let red = |x: usize, y: usize| f32::from(src[(y * gw + x) * 4]);
    let mut out = Vec::with_capacity(w * h);
    for &(y0, y1, fy) in &ys {
        for &(x0, x1, fx) in &xs {
            let top = red(x0, y0) + (red(x1, y0) - red(x0, y0)) * fx;
            let bottom = red(x0, y1) + (red(x1, y1) - red(x0, y1)) * fx;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            out.push((top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests;
