//! Composing animation frames onto the canvas (RFC 9649 section 2.7.1.1).
//!
//! The rules are libwebp's animation decoder's, so a file plays here the
//! way it plays in browsers built on libwebp:
//!
//! - The canvas starts transparent black. The `ANIM` background colour is
//!   a hint the format lets a decoder ignore, and libwebp ignores it.
//! - Disposing a frame to the background clears its rectangle to
//!   transparent black.
//! - A frame replaces its rectangle, and with blending on, each pixel that
//!   is not opaque is then blended over the canvas after the previous
//!   disposal, in straight alpha with libwebp's integer arithmetic.
//! - A frame that redraws the whole canvas, or that follows a full clear,
//!   starts from a transparent canvas and is not blended.

use cutlass_decode::CodecError;

use crate::limits::{canvas_ok, pixel_count};
use crate::picture::{self, Picture};
use crate::riff::Frame;

#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Rect {
    fn contains(&self, x: usize, y: usize) -> bool {
        (self.x..self.x + self.width).contains(&x) && (self.y..self.y + self.height).contains(&y)
    }
}

#[derive(Clone, Copy)]
struct Previous {
    rect: Rect,
    full: bool,
    dispose_background: bool,
    key: bool,
}

/// One frame decoded and checked against the canvas, ready to compose.
pub(crate) struct Decoded {
    rect: Rect,
    picture: Picture,
    blend: bool,
    has_alpha: bool,
    dispose_background: bool,
}

/// The canvas between frames.
pub(crate) struct Canvas {
    width: usize,
    height: usize,
    /// The canvas after the previous frame and its disposal, RGBA.
    disposed: Vec<u8>,
    previous: Option<Previous>,
}

impl Canvas {
    pub(crate) fn new(width: u32, height: u32) -> Result<Self, CodecError> {
        canvas_ok(width, height)?;
        let count = pixel_count(width, height)?;
        Ok(Self {
            width: width as usize,
            height: height as usize,
            disposed: vec![0; count * 4],
            previous: None,
        })
    }

    pub(crate) fn width(&self) -> u32 {
        self.width as u32
    }

    pub(crate) fn height(&self) -> u32 {
        self.height as u32
    }

    /// Decode one frame and check it fits. Leaves the canvas untouched.
    pub(crate) fn decode(&self, frame: &Frame<'_>) -> Result<Decoded, CodecError> {
        let rect = Rect {
            x: frame.x as usize,
            y: frame.y as usize,
            width: frame.width as usize,
            height: frame.height as usize,
        };
        if rect.x + rect.width > self.width || rect.y + rect.height > self.height {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "webp frame outside the canvas",
            });
        }
        let picture = picture::decode(frame.bitstream)?;
        if (picture.width, picture.height) != (frame.width, frame.height) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "webp frame size does not match its bitstream",
            });
        }
        Ok(Decoded {
            rect,
            picture,
            blend: frame.blend,
            has_alpha: frame.has_alpha,
            dispose_background: frame.dispose_background,
        })
    }

    /// Compose a decoded frame and return the canvas it shows, RGBA.
    pub(crate) fn compose(&mut self, frame: &Decoded) -> Vec<u8> {
        let rect = frame.rect;
        let full = rect.width == self.width && rect.height == self.height;
        let key = match self.previous {
            None => true,
            Some(previous) => {
                ((!frame.has_alpha || !frame.blend) && full)
                    || (previous.dispose_background && (previous.full || previous.key))
            }
        };
        let mut canvas = if key {
            vec![0; self.disposed.len()]
        } else {
            self.disposed.clone()
        };
        let row_bytes = rect.width * 4;
        for (row, line) in frame.picture.rgba.chunks_exact(row_bytes).enumerate() {
            let at = ((rect.y + row) * self.width + rect.x) * 4;
            canvas[at..at + row_bytes].copy_from_slice(line);
        }
        if let Some(previous) = self.previous
            && frame.blend
            && !key
        {
            let skip = previous.dispose_background.then_some(previous.rect);
            for y in rect.y..rect.y + rect.height {
                for x in rect.x..rect.x + rect.width {
                    if skip.is_some_and(|cleared| cleared.contains(x, y)) {
                        continue;
                    }
                    let at = (y * self.width + x) * 4;
                    if canvas[at + 3] != 255 {
                        let src = [canvas[at], canvas[at + 1], canvas[at + 2], canvas[at + 3]];
                        let dst = [
                            self.disposed[at],
                            self.disposed[at + 1],
                            self.disposed[at + 2],
                            self.disposed[at + 3],
                        ];
                        canvas[at..at + 4].copy_from_slice(&blend(src, dst));
                    }
                }
            }
        }
        self.disposed.clone_from(&canvas);
        if frame.dispose_background {
            for y in rect.y..rect.y + rect.height {
                let at = (y * self.width + rect.x) * 4;
                self.disposed[at..at + row_bytes].fill(0);
            }
        }
        self.previous = Some(Previous {
            rect,
            full,
            dispose_background: frame.dispose_background,
            key,
        });
        canvas
    }
}

/// `src` over `dst`, straight alpha, as libwebp's `BlendPixelNonPremult`.
fn blend(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let src_a = u32::from(src[3]);
    if src_a == 0 {
        return dst;
    }
    let dst_a = (u32::from(dst[3]) * (256 - src_a)) >> 8;
    let alpha = src_a + dst_a;
    let scale = (1u64 << 24) / u64::from(alpha);
    let channel = |c: usize| {
        let mixed = u64::from(src[c]) * u64::from(src_a) + u64::from(dst[c]) * u64::from(dst_a);
        ((mixed * scale) >> 24) as u8
    };
    [channel(0), channel(1), channel(2), alpha as u8]
}

#[cfg(test)]
mod tests {
    use super::blend;

    #[test]
    fn blending_follows_libwebp_rounding() {
        assert_eq!(blend([10, 20, 30, 0], [1, 2, 3, 4]), [1, 2, 3, 4]);
        // Over transparent black the colour loses a little to rounding.
        assert_eq!(blend([255, 255, 255, 3], [0, 0, 0, 0]), [254, 254, 254, 3]);
        assert_eq!(
            blend([200, 100, 0, 128], [0, 100, 200, 255]),
            [100, 99, 99, 255]
        );
    }
}
