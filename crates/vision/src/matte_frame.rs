//! A frame going into a Remove BG model, and its matte coming back.
//!
//! Every matter reads the frame smaller than it is, at a size of its own,
//! and answers with how much of each pixel it keeps, 0 to 1, at that size.
//! Here the frame is checked, scaled with a triangle filter and normalized
//! as the model was trained, and the answer is scaled back to the frame's
//! size with half-pixel centres and kept as bytes.

use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};
use image::imageops::{self, FilterType};
use image::{ImageBuffer, Rgb};

use crate::{Image, Matte, VisionError};

/// The longest side a frame given to a matter may have. The matte comes
/// back at the frame's size, so this bounds what one call holds.
pub const MATTE_MAX_SIDE: u32 = 8192;

/// Refuses an empty frame, one larger than [`MATTE_MAX_SIDE`], or RGB that
/// is not three bytes a pixel.
pub(crate) fn check(image: &Image<'_>) -> Result<(u32, u32), VisionError> {
    let (width, height) = (image.width, image.height);
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3));
    if width == 0
        || height == 0
        || width > MATTE_MAX_SIDE
        || height > MATTE_MAX_SIDE
        || expected != Some(image.rgb.len())
    {
        return Err(VisionError::Image {
            width,
            height,
            bytes: image.rgb.len(),
        });
    }
    Ok((width, height))
}

/// The size a `width` by `height` frame is read at with its short side
/// `side`, each side rounded to a whole number of `multiple` pixels, and
/// the long side no more than `longest`, the short one shrinking with it.
pub(crate) fn read_size(
    width: u32,
    height: u32,
    side: u32,
    multiple: u32,
    longest: u32,
) -> (u32, u32) {
    let multiple = f64::from(multiple.max(1));
    let (short, long) = (width.min(height).max(1), width.max(height).max(1));
    let scale = (f64::from(side) / f64::from(short)).min(f64::from(longest) / f64::from(long));
    let fit = |s: u32| {
        let whole = (f64::from(s) * scale / multiple).round().max(1.0) * multiple;
        whole.min(f64::from(longest.max(multiple as u32))) as u32
    };
    (fit(width), fit(height))
}

/// `image` scaled to `size` and normalized channel by channel, as a
/// `[1, 3, height, width]` tensor on `device`.
pub(crate) fn input(
    image: &Image<'_>,
    size: (u32, u32),
    mean: [f32; 3],
    spread: [f32; 3],
    device: &Device,
) -> Result<Tensor<4>, VisionError> {
    let refused = || VisionError::Image {
        width: image.width,
        height: image.height,
        bytes: image.rgb.len(),
    };
    let (width, height) = size;
    let scaled;
    let rgb = if (width, height) == (image.width, image.height) {
        image.rgb
    } else {
        let source: ImageBuffer<Rgb<u8>, &[u8]> =
            ImageBuffer::from_raw(image.width, image.height, image.rgb).ok_or_else(refused)?;
        scaled = imageops::resize(&source, width, height, FilterType::Triangle).into_raw();
        &scaled
    };
    let plane = width as usize * height as usize;
    let mut chw = vec![0.0_f32; 3 * plane];
    for (index, pixel) in rgb.as_chunks::<3>().0.iter().enumerate() {
        for (c, byte) in pixel.iter().enumerate() {
            chw[c * plane + index] = (f32::from(*byte) / 255.0 - mean[c]) / spread[c];
        }
    }
    Ok(Tensor::<4>::from_data(
        TensorData::new(chw, [1, 3, height as usize, width as usize]),
        device,
    ))
}

/// `x` resized bilinearly to `size` with half-pixel centres, as PyTorch's
/// `interpolate(..., align_corners=False)`.
pub(crate) fn resize(x: Tensor<4>, size: [usize; 2]) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    if [h, w] == size {
        return x;
    }
    interpolate(
        x,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_align_corners(false)
            .with_output_size(size),
    )
}

/// Reads a tensor back as `f32`s.
pub(crate) fn values<const D: usize>(x: Tensor<D>) -> Result<Vec<f32>, VisionError> {
    x.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .map_err(|error| VisionError::Model {
            message: format!("{error:?}"),
        })
}

/// A `[1, 1, h, w]` share kept, 0 to 1, as a matte at `width` by `height`.
pub(crate) fn to_matte(kept: Tensor<4>, width: u32, height: u32) -> Result<Matte, VisionError> {
    let [batch, channels, _, _] = kept.dims();
    if [batch, channels] != [1, 1] {
        return Err(VisionError::Model {
            message: format!("a matte of shape {:?}", kept.dims()),
        });
    }
    let kept = resize(kept, [height as usize, width as usize])
        .clamp(0.0, 1.0)
        .mul_scalar(255.0)
        .round();
    let alpha = values(kept)?.into_iter().map(|v| v as u8).collect();
    Matte::new(width, height, alpha).ok_or_else(|| VisionError::Model {
        message: "a matte of another size than the frame".into(),
    })
}

/// Real frames for timing and for looking at a matter's mattes, which no
/// committed test can hold: `name_WIDTHxHEIGHT.rgb` files of raw RGB in the
/// folder `CUTLASS_MATTE_FRAMES` names (made with `ffmpeg -pix_fmt rgb24 -f
/// rawvideo` from a clip in `local-files`). Each matte is written beside
/// them as `name.MODEL.pgm`.
#[cfg(test)]
pub(crate) mod real {
    use std::path::PathBuf;

    use crate::Matte;

    pub(crate) struct Frame {
        pub name: String,
        pub width: u32,
        pub height: u32,
        pub rgb: Vec<u8>,
    }

    /// `local-files/models/<name>` beside the repository, found from any
    /// checkout, a worktree under `.claude/worktrees` too.
    pub(crate) fn models(name: &str) -> PathBuf {
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        here.ancestors()
            .map(|dir| dir.join("local-files"))
            .find(|dir| dir.is_dir())
            .unwrap_or_else(|| here.join("../../../local-files"))
            .join("models")
            .join(name)
    }

    pub(crate) fn folder() -> Option<PathBuf> {
        std::env::var_os("CUTLASS_MATTE_FRAMES").map(PathBuf::from)
    }

    /// Every frame in the folder, by name.
    pub(crate) fn frames() -> Vec<Frame> {
        let Some(dir) = folder() else {
            return Vec::new();
        };
        let mut frames: Vec<Frame> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let stem = path.file_name()?.to_str()?.strip_suffix(".rgb")?.to_owned();
                let (name, size) = stem.rsplit_once('_')?;
                let (w, h) = size.split_once('x')?;
                let (width, height) = (w.parse().ok()?, h.parse().ok()?);
                let rgb = std::fs::read(&path).ok()?;
                (rgb.len() == width as usize * height as usize * 3).then(|| Frame {
                    name: name.to_owned(),
                    width,
                    height,
                    rgb,
                })
            })
            .collect();
        frames.sort_by(|a, b| a.name.cmp(&b.name));
        frames
    }

    pub(crate) fn save(name: &str, model: &str, matte: &Matte) {
        let Some(dir) = folder() else {
            return;
        };
        let mut pgm = format!("P5\n{} {}\n255\n", matte.width, matte.height).into_bytes();
        pgm.extend_from_slice(&matte.alpha);
        std::fs::write(dir.join(format!("{name}.{model}.pgm")), pgm).unwrap();
    }

    pub(crate) fn say(line: &str) {
        let _ = std::io::Write::write_all(&mut std::io::stderr(), format!("{line}\n").as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_short_side_is_read_at_the_side_in_whole_multiples() {
        assert_eq!(read_size(1920, 1080, 512, 32, 2048), (896, 512));
        assert_eq!(read_size(1080, 1920, 512, 32, 2048), (512, 896));
        assert_eq!(read_size(512, 512, 512, 32, 2048), (512, 512));
        assert_eq!(read_size(100, 50, 512, 32, 2048), (1024, 512));
        assert_eq!(read_size(1, 1, 512, 32, 2048), (512, 512));
    }

    #[test]
    fn a_long_thin_frame_is_held_to_the_longest_side() {
        assert_eq!(read_size(8192, 1, 512, 32, 2048), (2048, 32));
        assert_eq!(read_size(4000, 1000, 512, 32, 1024), (1024, 256));
    }

    #[test]
    fn a_bad_frame_is_refused() {
        for (width, height, bytes) in [
            (0, 4, 0),
            (4, 4, 47),
            (MATTE_MAX_SIDE + 1, 1, 3 * (MATTE_MAX_SIDE as usize + 1)),
        ] {
            let rgb = vec![0_u8; bytes];
            assert!(matches!(
                check(&Image {
                    width,
                    height,
                    rgb: &rgb
                }),
                Err(VisionError::Image { .. })
            ));
        }
    }

    #[test]
    fn a_frame_is_normalized_and_a_share_comes_back_as_bytes() {
        let device = Device::flex();
        let rgb = [0, 255, 51, 255, 0, 51];
        let image = Image {
            width: 2,
            height: 1,
            rgb: &rgb,
        };
        let x = input(&image, (2, 1), [0.5; 3], [0.5; 3], &device).unwrap();
        assert_eq!(x.dims(), [1, 3, 1, 2]);
        let got = values(x).unwrap();
        let want = [-1.0, 1.0, 1.0, -1.0, -0.6, -0.6];
        assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6));
        let kept = Tensor::<4>::from_data(TensorData::new(vec![0.0, 1.0], [1, 1, 1, 2]), &device);
        let matte = to_matte(kept, 4, 1).unwrap();
        assert_eq!(matte.alpha, [0, 64, 191, 255]);
        let flat = Tensor::<4>::from_data(TensorData::new(vec![0.2_f32], [1, 1, 1, 1]), &device);
        assert_eq!(to_matte(flat, 3, 2).unwrap().alpha, [51; 6]);
    }

    #[test]
    fn a_matte_of_two_channels_is_refused() {
        let device = Device::flex();
        let two = Tensor::<4>::zeros([1, 2, 2, 2], &device);
        assert!(matches!(
            to_matte(two, 2, 2),
            Err(VisionError::Model { .. })
        ));
    }
}
