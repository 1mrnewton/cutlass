//! Video pictures for one evaluated frame.
//!
//! A segment the factory cannot open is omitted. The other layers still draw.
//! Identity keeps the decoded size when it fits, centered on the canvas. A
//! larger picture is scaled down until it fits, then centered. Clip scale,
//! rotation, flip, and position are not applied.
//!
//! [`OpenClips`] keeps a file open for one draft revision. A later time seeks
//! that file. A failed open is not kept, so the next time tries again.

use std::collections::HashMap;

use cutlass_engine::{ClockKind, ReaderFactory, VideoSource, open_segment};
use cutlass_model::{Draft, DraftId, MaterialRef, Micros, SegmentId, TrackAttribute};
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

pub(crate) struct Placed<'a> {
    source: &'a VideoSource,
    width: u32,
    height: u32,
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
            ));
        }
    }
    let mut layers = Vec::with_capacity(chosen.len());
    for (render_index, track_index, segment_index, id) in chosen {
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
            Picture::Cpu(bytes) => place(self.width, self.height, bytes),
            Picture::Platform(_) => Sample::Miss,
        }
    }
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
}
