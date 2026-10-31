//! Video pictures for one evaluated frame.
//!
//! A segment the factory cannot open is omitted. The other layers still draw.
//! Identity keeps the decoded size when it fits, centered on the canvas. A
//! larger picture is scaled down until it fits, then centered. The segment
//! clip then scales, rotates, flips, and moves that picture, and multiplies
//! its opacity. Rotation is degrees counterclockwise with positive Y up.
//! A translation of 1 is half the canvas. A file display quarter turn is
//! clockwise and is applied before that fit. Keyframes are not applied.
//!
//! [`OpenClips`] keeps a file open for one draft revision. A later time seeks
//! that file. A failed open is not kept, so the next time tries again.

use std::collections::HashMap;

use cutlass_engine::{ClockKind, DisplayQuarter, ReaderFactory, VideoSource, open_segment};
use cutlass_model::{
    Clip, Draft, DraftId, MaterialRef, Micros, SegmentId, TrackAttribute, UniformScale,
};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample, rgba_len};

/// Open video files for one draft revision.
pub struct OpenClips {
    draft: Option<DraftId>,
    revision: u64,
    bound: bool,
    sources: HashMap<SegmentId, VideoSource>,
}

impl OpenClips {
    #[must_use]
    pub fn new() -> Self {
        Self {
            draft: None,
            revision: 0,
            bound: false,
            sources: HashMap::new(),
        }
    }

    /// Drops every open file.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.draft = None;
        self.revision = 0;
        self.bound = false;
    }

    pub(crate) fn prepare(&mut self, draft: &Draft, revision: u64) {
        if self.bound && self.draft.as_ref() == Some(&draft.id) && self.revision == revision {
            return;
        }
        self.sources.clear();
        self.draft = Some(draft.id.clone());
        self.revision = revision;
        self.bound = true;
    }
}

impl Default for OpenClips {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct VideoLayer<'a> {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: Placed<'a>,
}

/// Scale, rotation, flip, translation, and opacity for one picture.
///
/// `scale_x` and `scale_y` are positive. A negative clip scale flips that
/// axis and uses the absolute value. `sin` and `cos` are the counterclockwise
/// rotation, with positive Y up. `dx` is pixels to the right and `dy` is
/// pixels up. One translation unit is half the canvas on that axis.
#[derive(Clone, Copy, Debug)]
struct PictureTransform {
    scale_x: f64,
    scale_y: f64,
    sin: f64,
    cos: f64,
    flip_h: bool,
    flip_v: bool,
    dx: f64,
    dy: f64,
    opacity: u8,
    geometry_identity: bool,
}

pub(crate) struct Placed<'a> {
    source: &'a VideoSource,
    width: u32,
    height: u32,
    transform: PictureTransform,
}

fn picture_transform(
    clip: &Clip,
    uniform: &UniformScale,
    canvas_w: u32,
    canvas_h: u32,
) -> Option<PictureTransform> {
    let (mut scale_x, mut scale_y) = if uniform.on {
        (uniform.value, uniform.value)
    } else {
        (clip.scale.x, clip.scale.y)
    };
    if !scale_x.is_finite()
        || !scale_y.is_finite()
        || !clip.rotation.is_finite()
        || !clip.transform.x.is_finite()
        || !clip.transform.y.is_finite()
        || !clip.alpha.is_finite()
    {
        return None;
    }
    let mut flip_h = clip.flip.horizontal;
    let mut flip_v = clip.flip.vertical;
    if scale_x < 0.0 {
        flip_h = !flip_h;
        scale_x = -scale_x;
    }
    if scale_y < 0.0 {
        flip_v = !flip_v;
        scale_y = -scale_y;
    }
    let degrees = clip.rotation.rem_euclid(360.0);
    let (sin, cos) = sin_cos(degrees);
    let dx = clip.transform.x * f64::from(canvas_w) / 2.0;
    let dy = clip.transform.y * f64::from(canvas_h) / 2.0;
    let opacity = opacity_byte(clip.alpha)?;
    let geometry_identity = scale_x == 1.0
        && scale_y == 1.0
        && degrees == 0.0
        && !flip_h
        && !flip_v
        && dx == 0.0
        && dy == 0.0;
    Some(PictureTransform {
        scale_x,
        scale_y,
        sin,
        cos,
        flip_h,
        flip_v,
        dx,
        dy,
        opacity,
        geometry_identity,
    })
}

fn sin_cos(degrees: f64) -> (f64, f64) {
    if degrees == 0.0 {
        (0.0, 1.0)
    } else if degrees == 90.0 {
        (1.0, 0.0)
    } else if degrees == 180.0 {
        (0.0, -1.0)
    } else if degrees == 270.0 {
        (-1.0, 0.0)
    } else {
        let radians = degrees.to_radians();
        (radians.sin(), radians.cos())
    }
}

fn opacity_byte(alpha: f64) -> Option<u8> {
    if !alpha.is_finite() {
        return None;
    }
    if alpha <= 0.0 {
        return Some(0);
    }
    if alpha >= 1.0 {
        return Some(255);
    }
    let scaled = (alpha * 255.0).round();
    let mut byte = 0_u8;
    let mut cursor = 0.0_f64;
    while cursor + 0.5 < scaled && byte < 255 {
        byte = byte.saturating_add(1);
        cursor += 1.0;
    }
    Some(byte)
}

/// Visible video segments whose target covers `time`.
///
/// A segment already in `clips` is sampled from that file. A failed open is
/// left out of `clips`.
pub(crate) fn video_layers<'a>(
    draft: &Draft,
    media: &ReaderFactory,
    width: u32,
    height: u32,
    time: Micros,
    clips: &'a mut OpenClips,
) -> Vec<VideoLayer<'a>> {
    let mut chosen = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible || segment.source_range.is_none() {
                continue;
            }
            if !covers(segment.target.start.0, segment.target.duration.0, time.0) {
                continue;
            }
            if !matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(_))
            ) {
                continue;
            }
            let Some(transform) =
                picture_transform(&segment.clip, &segment.uniform_scale, width, height)
            else {
                continue;
            };
            if transform.opacity == 0 || transform.scale_x == 0.0 || transform.scale_y == 0.0 {
                continue;
            }
            if !clips.sources.contains_key(&segment.id) {
                let Ok(source) = open_segment(media, draft, &segment.id, ClockKind::Export, 1)
                else {
                    continue;
                };
                clips.sources.insert(segment.id.clone(), source);
            }
            chosen.push((
                segment.render_index,
                track_index,
                segment_index,
                segment.id.clone(),
                transform,
            ));
        }
    }
    let mut layers = Vec::with_capacity(chosen.len());
    for (render_index, track_index, segment_index, id, transform) in chosen {
        let Some(source) = clips.sources.get(&id) else {
            continue;
        };
        layers.push(VideoLayer {
            render_index,
            track_index,
            segment_index,
            placed: Placed {
                source,
                width,
                height,
                transform,
            },
        });
    }
    layers
}

fn covers(start: i64, duration: i64, time: i64) -> bool {
    let Some(end) = start.checked_add(duration) else {
        return false;
    };
    time >= start && time < end
}

impl PictureSource for Placed<'_> {
    fn sample(&self, time: Micros) -> Sample {
        let Sample::Ready(picture) = self.source.sample(time) else {
            return Sample::Miss;
        };
        match picture {
            Picture::Cpu(bytes) => place_with(
                self.width,
                self.height,
                bytes,
                self.transform,
                self.source.display(),
            ),
            Picture::Platform(_) => Sample::Miss,
        }
    }
}

fn place_with(
    canvas_w: u32,
    canvas_h: u32,
    picture: CpuBytes,
    transform: PictureTransform,
    display: DisplayQuarter,
) -> Sample {
    if transform.opacity == 0 || transform.scale_x == 0.0 || transform.scale_y == 0.0 {
        return Sample::Miss;
    }
    let Some(picture) = orient(picture, display) else {
        return Sample::Miss;
    };
    if transform.geometry_identity {
        let sample = place(canvas_w, canvas_h, picture);
        return if transform.opacity == 255 {
            sample
        } else {
            fade(sample, transform.opacity)
        };
    }
    sample_transformed(canvas_w, canvas_h, &picture, transform)
}

fn orient(picture: CpuBytes, display: DisplayQuarter) -> Option<CpuBytes> {
    if display == DisplayQuarter::Zero {
        return Some(picture);
    }
    let src_w = picture.width();
    let src_h = picture.height();
    let (dst_w, dst_h) = match display {
        DisplayQuarter::Zero | DisplayQuarter::Half => (src_w, src_h),
        DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (src_h, src_w),
    };
    let len = rgba_len(dst_w, dst_h)?;
    let src = picture.rgba();
    let mut dst = vec![0_u8; len];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let (sx, sy) = turned_source(display, src_w, src_h, x, y)?;
            let from = pixel_at(src_w, sx, sy)?;
            let to = pixel_at(dst_w, x, y)?;
            dst.get_mut(to..to + 4)?
                .copy_from_slice(src.get(from..from + 4)?);
        }
    }
    CpuBytes::new(dst_w, dst_h, dst).ok()
}

fn turned_source(
    display: DisplayQuarter,
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
) -> Option<(u32, u32)> {
    match display {
        DisplayQuarter::Zero => Some((x, y)),
        DisplayQuarter::Clockwise90 => Some((y, src_h.checked_sub(x)?.checked_sub(1)?)),
        DisplayQuarter::Half => Some((
            src_w.checked_sub(x)?.checked_sub(1)?,
            src_h.checked_sub(y)?.checked_sub(1)?,
        )),
        DisplayQuarter::Clockwise270 => Some((src_w.checked_sub(y)?.checked_sub(1)?, x)),
    }
}

fn pixel_at(width: u32, x: u32, y: u32) -> Option<usize> {
    let stride = usize::try_from(width).ok()?.checked_mul(4)?;
    usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)
}

fn place(canvas_w: u32, canvas_h: u32, picture: CpuBytes) -> Sample {
    let src_w = picture.width();
    let src_h = picture.height();
    if src_w == canvas_w && src_h == canvas_h {
        return Sample::Ready(Picture::Cpu(picture));
    }
    let (draw_w, draw_h, pixels) = if src_w <= canvas_w && src_h <= canvas_h {
        (src_w, src_h, picture.rgba().to_vec())
    } else {
        let Some((draw_w, draw_h)) = contain_size(src_w, src_h, canvas_w, canvas_h) else {
            return Sample::Miss;
        };
        let Some(pixels) = nearest(src_w, src_h, picture.rgba(), draw_w, draw_h) else {
            return Sample::Miss;
        };
        (draw_w, draw_h, pixels)
    };
    let Some(canvas) = stamp(canvas_w, canvas_h, draw_w, draw_h, &pixels) else {
        return Sample::Miss;
    };
    match CpuBytes::new(canvas_w, canvas_h, canvas) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

fn fade(sample: Sample, opacity: u8) -> Sample {
    let Sample::Ready(Picture::Cpu(bytes)) = sample else {
        return sample;
    };
    if opacity == 255 {
        return Sample::Ready(Picture::Cpu(bytes));
    }
    let mut rgba = bytes.rgba().to_vec();
    for pixel in rgba.chunks_mut(4) {
        for channel in pixel {
            *channel = mix_channel(*channel, opacity);
        }
    }
    match CpuBytes::new(bytes.width(), bytes.height(), rgba) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

fn mix_channel(channel: u8, opacity: u8) -> u8 {
    let mixed = u16::from(channel) * u16::from(opacity) + 127;
    u8::try_from(mixed / 255).unwrap_or(0)
}

fn sample_transformed(
    canvas_w: u32,
    canvas_h: u32,
    picture: &CpuBytes,
    transform: PictureTransform,
) -> Sample {
    let src_w = picture.width();
    let src_h = picture.height();
    let Some((fitted_w, fitted_h)) = fitted_size(src_w, src_h, canvas_w, canvas_h) else {
        return Sample::Miss;
    };
    let draw = Draw {
        src_w,
        src_h,
        fitted_w: f64::from(fitted_w),
        fitted_h: f64::from(fitted_h),
        center_x: f64::from(canvas_w) / 2.0 + transform.dx,
        center_y: f64::from(canvas_h) / 2.0 - transform.dy,
        transform,
    };
    if draw.drawn_w() <= 0.0 || draw.drawn_h() <= 0.0 {
        return Sample::Miss;
    }
    let Some(canvas_len) = rgba_len(canvas_w, canvas_h) else {
        return Sample::Miss;
    };
    let mut canvas = vec![0_u8; canvas_len];
    let Some((x0, x1, y0, y1)) = cover_box(&draw, canvas_w, canvas_h) else {
        return Sample::Miss;
    };
    let src = picture.rgba();
    let mut wrote = false;
    for y in y0..y1 {
        for x in x0..x1 {
            let Some((sx, sy)) = source_pixel(x, y, &draw) else {
                continue;
            };
            let Some(pixel) = src_pixel(src, src_w, sx, sy) else {
                continue;
            };
            let Some(dest) = canvas_slot(&mut canvas, canvas_w, x, y) else {
                continue;
            };
            dest[0] = mix_channel(pixel[0], draw.transform.opacity);
            dest[1] = mix_channel(pixel[1], draw.transform.opacity);
            dest[2] = mix_channel(pixel[2], draw.transform.opacity);
            dest[3] = mix_channel(pixel[3], draw.transform.opacity);
            wrote = true;
        }
    }
    if !wrote {
        return Sample::Miss;
    }
    match CpuBytes::new(canvas_w, canvas_h, canvas) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

fn fitted_size(src_w: u32, src_h: u32, canvas_w: u32, canvas_h: u32) -> Option<(u32, u32)> {
    if src_w == 0 || src_h == 0 {
        return None;
    }
    if src_w <= canvas_w && src_h <= canvas_h {
        Some((src_w, src_h))
    } else {
        contain_size(src_w, src_h, canvas_w, canvas_h)
    }
}

struct Draw {
    src_w: u32,
    src_h: u32,
    fitted_w: f64,
    fitted_h: f64,
    center_x: f64,
    center_y: f64,
    transform: PictureTransform,
}

impl Draw {
    fn drawn_w(&self) -> f64 {
        self.fitted_w * self.transform.scale_x
    }

    fn drawn_h(&self) -> f64 {
        self.fitted_h * self.transform.scale_y
    }
}

fn cover_box(draw: &Draw, canvas_w: u32, canvas_h: u32) -> Option<(u32, u32, u32, u32)> {
    let half_w = draw.drawn_w() / 2.0;
    let half_h = draw.drawn_h() / 2.0;
    let corners = [
        (half_w, half_h),
        (half_w, -half_h),
        (-half_w, half_h),
        (-half_w, -half_h),
    ];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let rotated_x = x * draw.transform.cos - y * draw.transform.sin;
        let rotated_y = x * draw.transform.sin + y * draw.transform.cos;
        let canvas_x = draw.center_x + rotated_x;
        let canvas_y = draw.center_y - rotated_y;
        min_x = min_x.min(canvas_x);
        max_x = max_x.max(canvas_x);
        min_y = min_y.min(canvas_y);
        max_y = max_y.max(canvas_y);
    }
    let x0 = axis_bound(min_x.floor(), canvas_w)?;
    let y0 = axis_bound(min_y.floor(), canvas_h)?;
    let x1 = axis_bound(max_x.ceil(), canvas_w)?;
    let y1 = axis_bound(max_y.ceil(), canvas_h)?;
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, x1, y0, y1))
    }
}

fn axis_bound(value: f64, limit: u32) -> Option<u32> {
    if !value.is_finite() {
        return None;
    }
    if value <= 0.0 {
        return Some(0);
    }
    if value >= f64::from(limit) {
        return Some(limit);
    }
    u32::try_from(value as u64).ok()
}

fn source_pixel(x: u32, y: u32, draw: &Draw) -> Option<(u32, u32)> {
    let rel_x = f64::from(x) + 0.5 - draw.center_x;
    let rel_y = f64::from(y) + 0.5 - draw.center_y;
    let up_y = -rel_y;
    let local_x = (rel_x * draw.transform.cos + up_y * draw.transform.sin) / draw.transform.scale_x;
    let local_y =
        (-rel_x * draw.transform.sin + up_y * draw.transform.cos) / draw.transform.scale_y;
    let half_w = draw.fitted_w / 2.0;
    let half_h = draw.fitted_h / 2.0;
    // The covered span is half-open on the right and on the bottom, matching
    // the integer center used when the clip is identity.
    if local_x < -half_w || local_x >= half_w || local_y <= -half_h || local_y > half_h {
        return None;
    }
    let su = (local_x + half_w) / draw.fitted_w * f64::from(draw.src_w);
    let sv = (half_h - local_y) / draw.fitted_h * f64::from(draw.src_h);
    let mut sx = floor_index(su, draw.src_w)?;
    let mut sy = floor_index(sv, draw.src_h)?;
    if draw.transform.flip_h {
        sx = draw.src_w.saturating_sub(sx).saturating_sub(1);
    }
    if draw.transform.flip_v {
        sy = draw.src_h.saturating_sub(sy).saturating_sub(1);
    }
    Some((sx, sy))
}

fn floor_index(value: f64, limit: u32) -> Option<u32> {
    if !value.is_finite() || value < 0.0 || value >= f64::from(limit) {
        return None;
    }
    u32::try_from(value as u64).ok()
}

fn src_pixel(src: &[u8], src_w: u32, x: u32, y: u32) -> Option<[u8; 4]> {
    let stride = usize::try_from(src_w).ok()?.checked_mul(4)?;
    let index = usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)?;
    let pixel = src.get(index..index + 4)?;
    Some([pixel[0], pixel[1], pixel[2], pixel[3]])
}

fn canvas_slot(canvas: &mut [u8], canvas_w: u32, x: u32, y: u32) -> Option<&mut [u8]> {
    let stride = usize::try_from(canvas_w).ok()?.checked_mul(4)?;
    let index = usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)?;
    canvas.get_mut(index..index + 4)
}

fn contain_size(src_w: u32, src_h: u32, canvas_w: u32, canvas_h: u32) -> Option<(u32, u32)> {
    if src_w == 0 || src_h == 0 || canvas_w == 0 || canvas_h == 0 {
        return None;
    }
    let by_width = u64::from(canvas_w).checked_mul(u64::from(src_h))?;
    let by_height = u64::from(canvas_h).checked_mul(u64::from(src_w))?;
    if by_width <= by_height {
        let draw_h = u64::from(src_h)
            .checked_mul(u64::from(canvas_w))?
            .checked_div(u64::from(src_w))?;
        let draw_h = u32::try_from(draw_h).ok()?.clamp(1, canvas_h);
        Some((canvas_w, draw_h))
    } else {
        let draw_w = u64::from(src_w)
            .checked_mul(u64::from(canvas_h))?
            .checked_div(u64::from(src_h))?;
        let draw_w = u32::try_from(draw_w).ok()?.clamp(1, canvas_w);
        Some((draw_w, canvas_h))
    }
}

fn nearest(src_w: u32, src_h: u32, src: &[u8], dst_w: u32, dst_h: u32) -> Option<Vec<u8>> {
    if src.len() != rgba_len(src_w, src_h)? || dst_w == 0 || dst_h == 0 {
        return None;
    }
    let dst_len = rgba_len(dst_w, dst_h)?;
    let mut out = vec![0_u8; dst_len];
    let src_stride = usize::try_from(src_w).ok()?.checked_mul(4)?;
    let dst_stride = usize::try_from(dst_w).ok()?.checked_mul(4)?;
    let src_h_us = usize::try_from(src_h).ok()?;
    let src_w_us = usize::try_from(src_w).ok()?;
    for y in 0..dst_h {
        let sy = (u64::from(y) * u64::from(src_h) / u64::from(dst_h))
            .min(u64::from(src_h.saturating_sub(1)));
        let sy = usize::try_from(sy).ok()?;
        let src_row = sy.min(src_h_us.saturating_sub(1)).checked_mul(src_stride)?;
        let dst_row = usize::try_from(y).ok()?.checked_mul(dst_stride)?;
        for x in 0..dst_w {
            let sx = (u64::from(x) * u64::from(src_w) / u64::from(dst_w))
                .min(u64::from(src_w.saturating_sub(1)));
            let sx = usize::try_from(sx).ok()?.min(src_w_us.saturating_sub(1));
            let from = src_row.checked_add(sx.checked_mul(4)?)?;
            let to = dst_row.checked_add(usize::try_from(x).ok()?.checked_mul(4)?)?;
            let pixel = src.get(from..from + 4)?;
            out.get_mut(to..to + 4)?.copy_from_slice(pixel);
        }
    }
    Some(out)
}

fn stamp(canvas_w: u32, canvas_h: u32, draw_w: u32, draw_h: u32, pixels: &[u8]) -> Option<Vec<u8>> {
    if draw_w > canvas_w || draw_h > canvas_h || pixels.len() != rgba_len(draw_w, draw_h)? {
        return None;
    }
    let mut canvas = vec![0_u8; rgba_len(canvas_w, canvas_h)?];
    let x0 = canvas_w - draw_w;
    let y0 = canvas_h - draw_h;
    let x0 = x0 / 2;
    let y0 = y0 / 2;
    let canvas_stride = usize::try_from(canvas_w).ok()?.checked_mul(4)?;
    let draw_stride = usize::try_from(draw_w).ok()?.checked_mul(4)?;
    for row in 0..draw_h {
        let y = usize::try_from(y0.checked_add(row)?).ok()?;
        let dest = y
            .checked_mul(canvas_stride)?
            .checked_add(usize::try_from(x0).ok()?.checked_mul(4)?)?;
        let src_at = usize::try_from(row).ok()?.checked_mul(draw_stride)?;
        canvas
            .get_mut(dest..dest + draw_stride)?
            .copy_from_slice(pixels.get(src_at..src_at + draw_stride)?);
    }
    Some(canvas)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Flip, Point};

    use super::*;

    fn solid(width: u32, height: u32, pixel: [u8; 4]) -> CpuBytes {
        let mut rgba = Vec::new();
        for _ in 0..width * height {
            rgba.extend(pixel);
        }
        CpuBytes::new(width, height, rgba).unwrap()
    }

    #[test]
    fn a_matching_canvas_keeps_the_decoded_bytes() {
        let picture = solid(2, 1, [9, 8, 7, 6]);
        let Sample::Ready(Picture::Cpu(bytes)) = place(2, 1, picture) else {
            panic!("ready");
        };
        assert_eq!(bytes.rgba(), &[9, 8, 7, 6, 9, 8, 7, 6]);
    }

    #[test]
    fn a_smaller_picture_stays_native_and_centered() {
        let picture = solid(1, 1, [1, 2, 3, 4]);
        let Sample::Ready(Picture::Cpu(bytes)) = place(3, 3, picture) else {
            panic!("ready");
        };
        let rgba = bytes.rgba();
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
        assert_eq!(&rgba[16..20], &[1, 2, 3, 4]);
        assert_eq!(rgba.iter().filter(|byte| **byte != 0).count(), 4);
    }

    #[test]
    fn a_larger_picture_is_scaled_down_to_fit() {
        let mut rgba = Vec::new();
        for pixel in [1_u8, 2, 3, 4, 5, 6, 7, 8] {
            rgba.extend([pixel, 0, 0, 255]);
        }
        let picture = CpuBytes::new(4, 2, rgba).unwrap();
        let Sample::Ready(Picture::Cpu(bytes)) = place(2, 2, picture) else {
            panic!("ready");
        };
        let rgba = bytes.rgba();
        assert_eq!(&rgba[0..8], &[1, 0, 0, 255, 3, 0, 0, 255]);
        assert_eq!(&rgba[8..16], &[0, 0, 0, 0, 0, 0, 0, 0]);
    }

    fn transform_of(
        clip: Clip,
        uniform: UniformScale,
        width: u32,
        height: u32,
    ) -> PictureTransform {
        picture_transform(&clip, &uniform, width, height).expect("transform")
    }

    fn paint(
        width: u32,
        height: u32,
        picture: CpuBytes,
        clip: Clip,
        uniform: UniformScale,
    ) -> Vec<u8> {
        let transform = transform_of(clip, uniform, width, height);
        let Sample::Ready(Picture::Cpu(bytes)) =
            place_with(width, height, picture, transform, DisplayQuarter::Zero)
        else {
            panic!("ready");
        };
        bytes.into_rgba()
    }

    #[test]
    fn a_default_clip_matches_the_untransformed_placement() {
        let picture = solid(1, 1, [1, 2, 3, 4]);
        let clip = Clip::default();
        let uniform = UniformScale::default();
        let transform = transform_of(clip, uniform, 3, 3);
        assert!(transform.geometry_identity);
        let Sample::Ready(Picture::Cpu(bytes)) =
            place_with(3, 3, picture, transform, DisplayQuarter::Zero)
        else {
            panic!("ready");
        };
        assert_eq!(&bytes.rgba()[16..20], &[1, 2, 3, 4]);
    }

    #[test]
    fn opacity_scales_premultiplied_channels_and_zero_draws_nothing() {
        let clip = Clip {
            alpha: 0.5,
            ..Clip::default()
        };
        let rgba = paint(
            1,
            1,
            solid(1, 1, [200, 0, 0, 255]),
            clip,
            UniformScale::default(),
        );
        assert_eq!(rgba, vec![100, 0, 0, 128]);

        let clear = Clip {
            alpha: 0.0,
            ..Clip::default()
        };
        let transform = transform_of(clear, UniformScale::default(), 1, 1);
        assert!(matches!(
            place_with(
                1,
                1,
                solid(1, 1, [1, 2, 3, 4]),
                transform,
                DisplayQuarter::Zero,
            ),
            Sample::Miss
        ));
    }

    #[test]
    fn flip_scale_and_translation_move_the_source_pixels() {
        let flip = Clip {
            flip: Flip {
                horizontal: true,
                ..Flip::default()
            },
            ..Clip::default()
        };
        let row = CpuBytes::new(2, 1, vec![1, 0, 0, 255, 2, 0, 0, 255]).unwrap();
        assert_eq!(
            &paint(2, 1, row, flip, UniformScale::default())[..8],
            &[2, 0, 0, 255, 1, 0, 0, 255]
        );

        let wide = Clip {
            scale: Point::new(3.0, 1.0),
            ..Clip::default()
        };
        let separate = UniformScale {
            on: false,
            ..UniformScale::default()
        };
        let scaled = paint(3, 1, solid(1, 1, [4, 0, 0, 255]), wide, separate);
        assert_eq!(scaled, vec![4, 0, 0, 255, 4, 0, 0, 255, 4, 0, 0, 255]);

        let shift = Clip {
            transform: Point::new(1.0, 0.0),
            ..Clip::default()
        };
        let moved = paint(
            4,
            1,
            solid(1, 1, [9, 0, 0, 255]),
            shift,
            UniformScale::default(),
        );
        assert_eq!(&moved[0..12], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&moved[12..16], &[9, 0, 0, 255]);

        let up = Clip {
            transform: Point::new(0.0, 0.5),
            ..Clip::default()
        };
        let raised = paint(
            1,
            4,
            solid(1, 1, [8, 0, 0, 255]),
            up,
            UniformScale::default(),
        );
        assert_eq!(&raised[0..4], &[8, 0, 0, 255]);
        assert!(raised[4..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn a_quarter_turn_puts_the_right_pixel_on_top() {
        let clip = Clip {
            rotation: 90.0,
            ..Clip::default()
        };
        let picture = CpuBytes::new(3, 1, vec![1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255]).unwrap();
        let rgba = paint(3, 3, picture, clip, UniformScale::default());
        let column = [rgba[4], rgba[16], rgba[28]];
        assert_eq!(column, [3, 2, 1]);
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[8], 0);
    }

    #[test]
    fn uniform_scale_drives_both_axes_and_a_negative_scale_flips() {
        let clip = Clip {
            scale: Point::new(9.0, 1.0),
            ..Clip::default()
        };
        let uniform = UniformScale::default();
        let doubled = UniformScale {
            value: 3.0,
            ..UniformScale::default()
        };
        let wide = paint(3, 1, solid(1, 1, [4, 0, 0, 255]), clip.clone(), doubled);
        assert_eq!(wide, vec![4, 0, 0, 255, 4, 0, 0, 255, 4, 0, 0, 255]);
        let ignored = transform_of(clip, uniform, 3, 1);
        assert_eq!(ignored.scale_x, 1.0);

        let negative = Clip {
            scale: Point::new(-1.0, 1.0),
            ..Clip::default()
        };
        let separate = UniformScale {
            on: false,
            ..UniformScale::default()
        };
        let row = CpuBytes::new(2, 1, vec![1, 0, 0, 255, 2, 0, 0, 255]).unwrap();
        assert_eq!(
            &paint(2, 1, row, negative, separate)[..8],
            &[2, 0, 0, 255, 1, 0, 0, 255]
        );
    }

    #[test]
    fn an_angle_that_is_not_a_quarter_turn_keeps_the_center_pixel() {
        let clip = Clip {
            rotation: 45.0,
            ..Clip::default()
        };
        let rgba = paint(
            3,
            3,
            solid(1, 1, [6, 0, 0, 255]),
            clip,
            UniformScale::default(),
        );
        assert_eq!(&rgba[16..20], &[6, 0, 0, 255]);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn a_file_quarter_turn_precedes_the_clip() {
        let row = CpuBytes::new(2, 1, vec![1, 0, 0, 255, 2, 0, 0, 255]).unwrap();
        let upright = placed(
            1,
            2,
            row.clone(),
            Clip::default(),
            DisplayQuarter::Clockwise90,
        );
        assert_eq!(upright, vec![1, 0, 0, 255, 2, 0, 0, 255]);

        let half = placed(2, 1, row.clone(), Clip::default(), DisplayQuarter::Half);
        assert_eq!(&half[..8], &[2, 0, 0, 255, 1, 0, 0, 255]);

        let left = placed(
            1,
            2,
            row.clone(),
            Clip::default(),
            DisplayQuarter::Clockwise270,
        );
        assert_eq!(left, vec![2, 0, 0, 255, 1, 0, 0, 255]);

        let flip = Clip {
            flip: Flip {
                horizontal: true,
                ..Flip::default()
            },
            ..Clip::default()
        };
        let after = placed(1, 2, row, flip, DisplayQuarter::Clockwise90);
        assert_eq!(after, vec![1, 0, 0, 255, 2, 0, 0, 255]);
    }

    fn placed(
        width: u32,
        height: u32,
        picture: CpuBytes,
        clip: Clip,
        display: DisplayQuarter,
    ) -> Vec<u8> {
        let transform = transform_of(clip, UniformScale::default(), width, height);
        let Sample::Ready(Picture::Cpu(bytes)) =
            place_with(width, height, picture, transform, display)
        else {
            panic!("ready");
        };
        bytes.into_rgba()
    }

    #[test]
    fn a_non_finite_clip_is_not_a_transform() {
        let clip = Clip {
            rotation: f64::NAN,
            ..Clip::default()
        };
        assert!(picture_transform(&clip, &UniformScale::default(), 2, 2).is_none());
    }
}
