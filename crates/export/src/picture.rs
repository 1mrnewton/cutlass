//! Which pictures belong on each frame, and how they sit on the canvas.
//!
//! The fit matches the preview: the picture keeps its shape inside the
//! canvas, then scale, position, and rotation move it. Opacity blends it
//! over what is already there. Crop, blend modes, and text are left out,
//! as they are on the preview.

use cutlass_core::Timestamp;
use cutlass_engine::frame_time;
use cutlass_project::{Channels, FrameRate, Project, Segment};

/// Sequence times of the frames whose start is still inside the cut.
pub(crate) fn frame_times(duration_us: i64, rate: FrameRate) -> Vec<i64> {
    let mut times = Vec::new();
    let mut index = 0i64;
    while let Some(time) = frame_time(index, rate) {
        if time >= duration_us || time < 0 {
            break;
        }
        times.push(time);
        if index == i64::MAX {
            break;
        }
        index += 1;
    }
    times
}

/// Even width and height. H.264 stores pairs of pixels. A side below two
/// pixels cannot be encoded.
pub(crate) fn encode_size(width: u32, height: u32) -> Option<(u32, u32)> {
    let width = width - width % 2;
    let height = height - height % 2;
    if width < 2 || height < 2 {
        None
    } else {
        Some((width, height))
    }
}

/// One video picture under the playhead, bottom track first.
pub(crate) struct Shot<'a> {
    pub source: &'a str,
    pub media: (u32, u32),
    pub time: Timestamp,
    pub position: (f64, f64),
    pub scale: (f64, f64),
    pub rotation: f64,
    pub opacity: f64,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
}

/// Visible video under `playhead_us`. Hidden lanes, stills, and sound are
/// left out.
pub(crate) fn shots(project: &Project, playhead_us: i64) -> Vec<Shot<'_>> {
    let mut layers = Vec::new();
    for track in project.timeline().tracks() {
        if track.hidden() {
            continue;
        }
        let Some(segment) = track
            .segments()
            .iter()
            .find(|segment| drawn(project, segment, playhead_us))
        else {
            continue;
        };
        let Some(shot) = shot(project, segment, playhead_us) else {
            continue;
        };
        layers.push(shot);
    }
    layers
}

fn drawn(project: &Project, segment: &Segment, playhead_us: i64) -> bool {
    let range = segment.timeline();
    let inside = range.start().value <= playhead_us && playhead_us < range.end().value;
    inside
        && segment.visible()
        && segment.channels() != Channels::Sound
        && segment
            .asset_id()
            .and_then(|id| project.asset(id))
            .is_some_and(|asset| asset.is_video())
}

fn shot<'a>(project: &'a Project, segment: &'a Segment, playhead_us: i64) -> Option<Shot<'a>> {
    let asset = project.asset(segment.asset_id()?)?;
    let transform = segment.transform();
    let (flip_horizontal, flip_vertical) = transform.flips();
    Some(Shot {
        source: asset.source(),
        media: asset.visual_size()?,
        time: media_time(segment, playhead_us)?,
        position: transform.position(),
        scale: transform.scale(),
        rotation: transform.rotation(),
        opacity: transform.opacity(),
        flip_horizontal,
        flip_vertical,
    })
}

/// Source time at `playhead_us`. The start of the clip is the source start.
fn media_time(segment: &Segment, playhead_us: i64) -> Option<Timestamp> {
    let source = segment.source()?;
    let timeline = segment.timeline();
    if playhead_us < timeline.start().value || playhead_us >= timeline.end().value {
        return None;
    }
    let offset = playhead_us.checked_sub(timeline.start().value)?;
    let ticks = segment.source_offset(offset).ok()?;
    let mut value = source.start().value.checked_add(ticks)?;
    let end = source.end().value;
    if value >= end {
        if end <= source.start().value {
            return None;
        }
        value = end - 1;
    }
    Some(Timestamp::new(value, source.start().timescale))
}

/// One RGBA picture, top row first, four bytes per pixel.
pub(crate) struct Plate {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub(crate) fn plate(width: u32, height: u32, color: (u8, u8, u8)) -> Plate {
    let count = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    let mut pixels = vec![0; count];
    let mut index = 0;
    while index + 3 < pixels.len() {
        pixels[index] = color.0;
        pixels[index + 1] = color.1;
        pixels[index + 2] = color.2;
        pixels[index + 3] = 255;
        index += 4;
    }
    Plate {
        width,
        height,
        pixels,
    }
}

/// A decoded picture to paint in the preview's box.
pub(crate) struct Sprite<'a> {
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub media: (u32, u32),
    pub position: (f64, f64),
    pub scale: (f64, f64),
    pub rotation: f64,
    pub opacity: f64,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
}

/// Paints `sprite` over `plate`. A pixel outside the picture is left alone.
pub(crate) fn paint(plate: &mut Plate, sprite: &Sprite<'_>) {
    if sprite.opacity <= 0.0 || sprite.width == 0 || sprite.height == 0 {
        return;
    }
    let needed = (sprite.width as usize)
        .saturating_mul(sprite.height as usize)
        .saturating_mul(4);
    if sprite.pixels.len() != needed {
        return;
    }
    let Some(place) = placement(
        (plate.width, plate.height),
        sprite.media,
        sprite.position,
        sprite.scale,
        sprite.rotation,
    ) else {
        return;
    };
    let (sin, cos) = (-sprite.rotation).to_radians().sin_cos();
    let width = plate.width;
    let height = plate.height;
    for row in 0..height {
        for col in 0..width {
            let world_x = f64::from(col) + 0.5 - place.center_x;
            let world_y = f64::from(row) + 0.5 - place.center_y;
            let local_x = world_x * cos - world_y * sin;
            let local_y = world_x * sin + world_y * cos;
            let mut u = local_x / place.size_x + 0.5;
            let mut v = local_y / place.size_y + 0.5;
            if sprite.flip_horizontal {
                u = 1.0 - u;
            }
            if sprite.flip_vertical {
                v = 1.0 - v;
            }
            let Some(sample) = sample(sprite, u, v) else {
                continue;
            };
            let index = ((row as usize) * (width as usize) + col as usize) * 4;
            blend(&mut plate.pixels[index..index + 4], sample, sprite.opacity);
        }
    }
}

struct Place {
    center_x: f64,
    center_y: f64,
    size_x: f64,
    size_y: f64,
}

fn placement(
    canvas: (u32, u32),
    media: (u32, u32),
    position: (f64, f64),
    scale: (f64, f64),
    rotation: f64,
) -> Option<Place> {
    let canvas_w = f64::from(canvas.0);
    let canvas_h = f64::from(canvas.1);
    let media_w = f64::from(media.0);
    let media_h = f64::from(media.1);
    if canvas_w <= 0.0
        || canvas_h <= 0.0
        || media_w <= 0.0
        || media_h <= 0.0
        || scale.0 <= 0.0
        || scale.1 <= 0.0
        || !scale.0.is_finite()
        || !scale.1.is_finite()
        || !position.0.is_finite()
        || !position.1.is_finite()
        || !rotation.is_finite()
    {
        return None;
    }
    let media_shape = media_w / media_h;
    let (fitted_w, fitted_h) = if canvas_w / canvas_h > media_shape {
        (canvas_h * media_shape, canvas_h)
    } else {
        (canvas_w, canvas_w / media_shape)
    };
    Some(Place {
        center_x: canvas_w / 2.0 + position.0 * canvas_w,
        center_y: canvas_h / 2.0 + position.1 * canvas_h,
        size_x: fitted_w * scale.0,
        size_y: fitted_h * scale.1,
    })
}

fn sample(sprite: &Sprite<'_>, u: f64, v: f64) -> Option<[u8; 4]> {
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None;
    }
    let x = (u * f64::from(sprite.width))
        .floor()
        .clamp(0.0, f64::from(sprite.width - 1));
    let y = (v * f64::from(sprite.height))
        .floor()
        .clamp(0.0, f64::from(sprite.height - 1));
    let x = x as usize;
    let y = y as usize;
    let index = (y * sprite.width as usize + x) * 4;
    let pixel = sprite.pixels.get(index..index + 4)?;
    Some([pixel[0], pixel[1], pixel[2], pixel[3]])
}

fn blend(dst: &mut [u8], src: [u8; 4], opacity: f64) {
    let opacity = opacity.clamp(0.0, 1.0);
    if opacity >= 1.0 {
        dst[..3].copy_from_slice(&src[..3]);
        return;
    }
    for channel in 0..3 {
        let mixed = f64::from(src[channel]) * opacity + f64::from(dst[channel]) * (1.0 - opacity);
        dst[channel] = mixed.round().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::{Sprite, paint, plate};

    #[test]
    fn a_full_picture_covers_the_canvas() {
        let mut canvas = plate(4, 4, (0, 0, 0));
        let red = [255, 0, 0, 255].repeat(4);
        paint(
            &mut canvas,
            &Sprite {
                pixels: &red,
                width: 2,
                height: 2,
                media: (2, 2),
                position: (0.0, 0.0),
                scale: (1.0, 1.0),
                rotation: 0.0,
                opacity: 1.0,
                flip_horizontal: false,
                flip_vertical: false,
            },
        );
        assert!(
            canvas
                .pixels
                .chunks(4)
                .all(|pixel| pixel == [255, 0, 0, 255])
        );
    }

    #[test]
    fn a_clear_picture_leaves_the_background() {
        let mut canvas = plate(2, 2, (0, 0, 8));
        let red = [255, 0, 0, 255].repeat(4);
        paint(
            &mut canvas,
            &Sprite {
                pixels: &red,
                width: 2,
                height: 2,
                media: (2, 2),
                position: (0.0, 0.0),
                scale: (1.0, 1.0),
                rotation: 0.0,
                opacity: 0.0,
                flip_horizontal: false,
                flip_vertical: false,
            },
        );
        assert!(canvas.pixels.chunks(4).all(|pixel| pixel == [0, 0, 8, 255]));
    }
}
