//! Canvas, color, and the two edit switches.

use crate::Error;
use crate::rate::{FrameRate, SequenceRate};

/// Canvas shape. Original follows the first video's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio {
    Original,
    R16x9,
    R9x16,
    Square,
    R4x5,
    R4x3,
}

/// Color space for the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// The empty project's color space.
    Rec709,
    Rec2020Hlg,
    Rec2100Pq,
}

/// A canvas color. Channels are 0..=255.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl Color {
    pub const BLACK: Self = Self { r: 0, g: 0, b: 0 };

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn channels(self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }
}

/// Width, height, and a ratio. Original follows the first video's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Canvas {
    width: u32,
    height: u32,
    ratio: AspectRatio,
}

impl Canvas {
    pub fn new(width: u32, height: u32, ratio: AspectRatio) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::ZeroCanvas);
        }
        Ok(Self {
            width,
            height,
            ratio,
        })
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn height(self) -> u32 {
        self.height
    }

    pub const fn ratio(self) -> AspectRatio {
        self.ratio
    }
}

/// Project settings. Duration is not one of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    canvas: Canvas,
    background: Color,
    rate: SequenceRate,
    color_space: ColorSpace,
    magnet: bool,
    linkage: bool,
}

impl Settings {
    pub fn new(
        canvas: Canvas,
        background: Color,
        rate: SequenceRate,
        color_space: ColorSpace,
        magnet: bool,
        linkage: bool,
    ) -> Self {
        Self {
            canvas,
            background,
            rate,
            color_space,
            magnet,
            linkage,
        }
    }

    pub(crate) fn set_canvas(&mut self, canvas: Canvas) {
        self.canvas = canvas;
    }

    pub(crate) fn set_size(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.canvas = Canvas::new(width, height, self.canvas.ratio)?;
        Ok(())
    }

    pub(crate) fn set_background(&mut self, background: Color) {
        self.background = background;
    }

    pub(crate) fn set_rate(&mut self, rate: SequenceRate) {
        self.rate = rate;
    }

    pub(crate) fn set_color_space(&mut self, color_space: ColorSpace) {
        self.color_space = color_space;
    }

    pub(crate) fn set_magnet(&mut self, magnet: bool) {
        self.magnet = magnet;
    }

    pub(crate) fn set_linkage(&mut self, linkage: bool) {
        self.linkage = linkage;
    }

    pub const fn canvas(&self) -> Canvas {
        self.canvas
    }

    pub const fn width(&self) -> u32 {
        self.canvas.width
    }

    pub const fn height(&self) -> u32 {
        self.canvas.height
    }

    pub const fn ratio(&self) -> AspectRatio {
        self.canvas.ratio
    }

    pub const fn background(&self) -> Color {
        self.background
    }

    pub const fn rate(&self) -> SequenceRate {
        self.rate
    }

    pub const fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    pub const fn magnet(&self) -> bool {
        self.magnet
    }

    pub const fn linkage(&self) -> bool {
        self.linkage
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::new(
            Canvas::new(1920, 1080, AspectRatio::Original).expect("the empty canvas is non-zero"),
            Color::BLACK,
            FrameRate::Fps30.sequence(),
            ColorSpace::Rec709,
            true,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_project_uses_the_starting_settings() {
        let settings = Settings::default();
        assert_eq!(settings.width(), 1920);
        assert_eq!(settings.height(), 1080);
        assert_eq!(settings.ratio(), AspectRatio::Original);
        assert_eq!(settings.background(), Color::BLACK);
        assert_eq!(settings.rate().rate(), FrameRate::Fps30);
        assert!(!settings.rate().drop_frame());
        assert_eq!(settings.color_space(), ColorSpace::Rec709);
        assert!(settings.magnet());
        assert!(settings.linkage());
    }

    #[test]
    fn a_zero_canvas_is_rejected() {
        assert_eq!(
            Canvas::new(0, 1080, AspectRatio::R16x9).unwrap_err(),
            Error::ZeroCanvas
        );
        assert_eq!(
            Canvas::new(1920, 0, AspectRatio::R16x9).unwrap_err(),
            Error::ZeroCanvas
        );
    }
}
