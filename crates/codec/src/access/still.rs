//! Coded samples of one still, before color conversion.
//!
//! A decoder fills this from the file's declared color type and bit depth.
//! It does not scale sub-byte samples up to eight bits, expand a palette,
//! or apply gamma. An encoder writes the same domain back.

use std::fmt;

/// How the samples of one still are arranged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorType {
    Gray,
    Rgb,
    Indexed,
    GrayAlpha,
    Rgba,
}

impl ColorType {
    /// Samples stored for one pixel, in channel order.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray | Self::Indexed => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// Bits stored for each sample on the wire.
///
/// Indexed color uses this as the index width. Palette entries are eight bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitDepth {
    One,
    Two,
    Four,
    Eight,
    Sixteen,
}

impl BitDepth {
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }

    /// Largest sample this depth can store.
    #[must_use]
    pub const fn max_sample(self) -> u16 {
        match self {
            Self::Sixteen => u16::MAX,
            _ => (1u16 << self.bits()) - 1,
        }
    }
}

/// One 8-bit palette entry. The palette is eight bits even when indices are narrower.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rgb8 {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

/// Straight alpha that is not a channel in the sample array.
///
/// A color key matches whole samples. Palette alpha is one byte per leading
/// palette entry; entries past the table are opaque.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transparency {
    None,
    ColorKey(Vec<u16>),
    PaletteAlpha(Vec<u8>),
}

/// Pixels per unit, from a physical-resolution chunk.
///
/// `unit` is 0 when the unit is unknown and 1 when the unit is the meter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalPixel {
    pub x: u32,
    pub y: u32,
    pub unit: u8,
}

/// One text item. `text` is the uncompressed payload.
///
/// An empty language and translated keyword is a Latin-1 text chunk.
/// `compressed` asks the encoder to zlib-compress that payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextItem {
    pub keyword: Vec<u8>,
    pub language: Vec<u8>,
    pub translated: Vec<u8>,
    pub text: Vec<u8>,
    pub compressed: bool,
}

/// Why [`StillImage::checked`] refused an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageError {
    Empty,
    Dimension,
    Depth,
    Length,
    Range,
    Palette,
    Transparency,
    Metadata,
}

impl fmt::Display for ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "the still has an empty edge",
            Self::Dimension => "the still dimension is outside the png range",
            Self::Depth => "the bit depth is not legal for this color type",
            Self::Length => "the sample count does not match the still",
            Self::Range => "a sample is outside the bit depth",
            Self::Palette => "the palette does not match the still",
            Self::Transparency => "the transparency does not match the still",
            Self::Metadata => "the still metadata is not legal",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ImageError {}

/// One still in its coded sample domain.
///
/// `samples` is row-major, channel-interleaved, with the top row first.
/// Each value is in `0 ..= depth.max_sample()`. Build it with
/// [`Self::from_samples`] and [`Self::checked`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StillImage {
    pub width: u32,
    pub height: u32,
    pub color: ColorType,
    pub depth: BitDepth,
    pub samples: Vec<u16>,
    pub palette: Vec<Rgb8>,
    pub transparency: Transparency,
    /// Gamma times 100000. Absent when the file did not name one.
    pub gamma_times_100000: Option<u32>,
    /// White point, then red, green, and blue, each x then y, times 100000.
    pub chromaticity: Option<[u32; 8]>,
    /// Rendering intent 0 through 3.
    pub srgb_intent: Option<u8>,
    pub icc_name: Vec<u8>,
    pub icc_profile: Vec<u8>,
    /// Empty when the file did not name significant bits.
    pub significant_bits: Vec<u8>,
    /// Empty when the file did not name a background.
    pub background: Vec<u16>,
    pub physical: Option<PhysicalPixel>,
    pub texts: Vec<TextItem>,
}

impl StillImage {
    /// An image with no palette, transparency, or metadata.
    ///
    /// The result is not checked. Call [`Self::checked`] before encoding.
    #[must_use]
    pub fn from_samples(
        width: u32,
        height: u32,
        color: ColorType,
        depth: BitDepth,
        samples: Vec<u16>,
    ) -> Self {
        Self {
            width,
            height,
            color,
            depth,
            samples,
            palette: Vec::new(),
            transparency: Transparency::None,
            gamma_times_100000: None,
            chromaticity: None,
            srgb_intent: None,
            icc_name: Vec::new(),
            icc_profile: Vec::new(),
            significant_bits: Vec::new(),
            background: Vec::new(),
            physical: None,
            texts: Vec::new(),
        }
    }

    /// [`Self::validate`] as a result that returns the image.
    pub fn checked(self) -> Result<Self, ImageError> {
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ImageError> {
        if self.width == 0 || self.height == 0 {
            return Err(ImageError::Empty);
        }
        if self.width > 0x7fff_ffff || self.height > 0x7fff_ffff {
            return Err(ImageError::Dimension);
        }
        if !depth_allowed(self.color, self.depth) {
            return Err(ImageError::Depth);
        }
        let channels = self.color.channels();
        let count = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|pixels| pixels.checked_mul(channels as u64))
            .ok_or(ImageError::Length)?;
        if self.samples.len() as u64 != count {
            return Err(ImageError::Length);
        }
        let max = self.depth.max_sample();
        if self.samples.iter().any(|sample| *sample > max) {
            return Err(ImageError::Range);
        }
        self.validate_palette(max)?;
        self.validate_transparency(max)?;
        self.validate_metadata()?;
        Ok(())
    }

    fn validate_palette(&self, max: u16) -> Result<(), ImageError> {
        match self.color {
            ColorType::Gray | ColorType::GrayAlpha if !self.palette.is_empty() => {
                Err(ImageError::Palette)
            }
            ColorType::Indexed => {
                let limit = 1usize << self.depth.bits();
                if self.palette.is_empty() || self.palette.len() > 256 || self.palette.len() > limit
                {
                    return Err(ImageError::Palette);
                }
                let entries = self.palette.len() as u16;
                if self.samples.iter().any(|index| *index >= entries) {
                    return Err(ImageError::Palette);
                }
                Ok(())
            }
            ColorType::Rgb | ColorType::Rgba => {
                if self.palette.len() > 256 {
                    return Err(ImageError::Palette);
                }
                let _ = max;
                Ok(())
            }
            ColorType::Gray | ColorType::GrayAlpha => Ok(()),
        }
    }

    fn validate_transparency(&self, max: u16) -> Result<(), ImageError> {
        match &self.transparency {
            Transparency::None => Ok(()),
            Transparency::ColorKey(key) => match self.color {
                ColorType::Gray if key.len() == 1 && key[0] <= max => Ok(()),
                ColorType::Rgb if key.len() == 3 && key.iter().all(|sample| *sample <= max) => {
                    Ok(())
                }
                _ => Err(ImageError::Transparency),
            },
            Transparency::PaletteAlpha(alpha) => {
                if self.color != ColorType::Indexed
                    || alpha.is_empty()
                    || alpha.len() > self.palette.len()
                {
                    return Err(ImageError::Transparency);
                }
                Ok(())
            }
        }
    }

    fn validate_metadata(&self) -> Result<(), ImageError> {
        if self
            .gamma_times_100000
            .is_some_and(|gamma| gamma > 0x7fff_ffff)
        {
            return Err(ImageError::Metadata);
        }
        if self
            .chromaticity
            .is_some_and(|values| values.iter().any(|value| *value > 0x7fff_ffff))
        {
            return Err(ImageError::Metadata);
        }
        if self.srgb_intent.is_some_and(|intent| intent > 3) {
            return Err(ImageError::Metadata);
        }
        validate_icc(&self.icc_name, &self.icc_profile)?;
        validate_significant_bits(self.color, self.depth, &self.significant_bits)?;
        validate_background(self, &self.background)?;
        if let Some(physical) = self.physical
            && (physical.unit > 1 || physical.x > 0x7fff_ffff || physical.y > 0x7fff_ffff)
        {
            return Err(ImageError::Metadata);
        }
        for text in &self.texts {
            if !legal_keyword(&text.keyword)
                || text.language.contains(&0)
                || text.translated.contains(&0)
            {
                return Err(ImageError::Metadata);
            }
        }
        Ok(())
    }
}

fn depth_allowed(color: ColorType, depth: BitDepth) -> bool {
    match color {
        ColorType::Gray => true,
        ColorType::Indexed => !matches!(depth, BitDepth::Sixteen),
        ColorType::Rgb | ColorType::GrayAlpha | ColorType::Rgba => {
            matches!(depth, BitDepth::Eight | BitDepth::Sixteen)
        }
    }
}

fn validate_icc(name: &[u8], profile: &[u8]) -> Result<(), ImageError> {
    if name.is_empty() && profile.is_empty() {
        return Ok(());
    }
    if profile.is_empty() || !legal_icc_name(name) {
        return Err(ImageError::Metadata);
    }
    Ok(())
}

fn legal_icc_name(name: &[u8]) -> bool {
    if name.is_empty() || name.len() > 79 || name[0] == b' ' || name[name.len() - 1] == b' ' {
        return false;
    }
    let mut previous_space = false;
    for &byte in name {
        let printable = (0x20..=0x7e).contains(&byte) || (0xa1..=0xff).contains(&byte);
        if !printable {
            return false;
        }
        if byte == b' ' && previous_space {
            return false;
        }
        previous_space = byte == b' ';
    }
    true
}

fn legal_keyword(keyword: &[u8]) -> bool {
    if !(1..=79).contains(&keyword.len())
        || keyword[0] == b' '
        || keyword[keyword.len() - 1] == b' '
    {
        return false;
    }
    let mut previous_space = false;
    for &byte in keyword {
        let latin1 = (0x20..=0x7e).contains(&byte) || (0xa1..=0xff).contains(&byte);
        if !latin1 || (byte == b' ' && previous_space) {
            return false;
        }
        previous_space = byte == b' ';
    }
    true
}

fn validate_significant_bits(
    color: ColorType,
    depth: BitDepth,
    bits: &[u8],
) -> Result<(), ImageError> {
    if bits.is_empty() {
        return Ok(());
    }
    let expected = match color {
        ColorType::Gray => 1,
        ColorType::GrayAlpha => 2,
        ColorType::Rgb | ColorType::Indexed => 3,
        ColorType::Rgba => 4,
    };
    let max = if color == ColorType::Indexed {
        8
    } else {
        depth.bits()
    };
    if bits.len() != expected || bits.iter().any(|bit| *bit == 0 || *bit > max) {
        return Err(ImageError::Metadata);
    }
    Ok(())
}

fn validate_background(image: &StillImage, background: &[u16]) -> Result<(), ImageError> {
    if background.is_empty() {
        return Ok(());
    }
    let max = image.depth.max_sample();
    let ok = match image.color {
        ColorType::Gray | ColorType::GrayAlpha => background.len() == 1 && background[0] <= max,
        ColorType::Rgb | ColorType::Rgba => {
            background.len() == 3 && background.iter().all(|sample| *sample <= max)
        }
        ColorType::Indexed => {
            background.len() == 1 && (background[0] as usize) < image.palette.len()
        }
    };
    if ok {
        Ok(())
    } else {
        Err(ImageError::Metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitDepth, ColorType, ImageError, Rgb8, StillImage, Transparency};

    #[test]
    fn an_eight_bit_grey_pixel_checks() {
        let image = StillImage::from_samples(1, 1, ColorType::Gray, BitDepth::Eight, vec![12])
            .checked()
            .expect("grey");
        assert_eq!(image.samples, vec![12]);
        assert_eq!(image.color.channels(), 1);
    }

    #[test]
    fn a_one_bit_sample_stays_in_range() {
        let image = StillImage::from_samples(2, 1, ColorType::Gray, BitDepth::One, vec![1, 0])
            .checked()
            .expect("bit");
        assert_eq!(image.depth.max_sample(), 1);
        let mut wide = image.clone();
        wide.samples = vec![1, 2];
        assert_eq!(wide.validate(), Err(ImageError::Range));
    }

    #[test]
    fn truecolor_rejects_a_four_bit_depth_and_a_short_buffer() {
        assert_eq!(
            StillImage::from_samples(1, 1, ColorType::Rgb, BitDepth::Four, vec![1, 2, 3])
                .validate(),
            Err(ImageError::Depth)
        );
        assert_eq!(
            StillImage::from_samples(1, 1, ColorType::Rgb, BitDepth::Eight, vec![1, 2]).validate(),
            Err(ImageError::Length)
        );
        assert_eq!(
            StillImage::from_samples(0, 1, ColorType::Gray, BitDepth::Eight, Vec::new()).validate(),
            Err(ImageError::Empty)
        );
    }

    #[test]
    fn indexed_color_keeps_indices_and_refuses_a_missing_palette() {
        let mut image =
            StillImage::from_samples(1, 1, ColorType::Indexed, BitDepth::Eight, vec![1]);
        assert_eq!(image.validate(), Err(ImageError::Palette));
        image.palette = vec![
            Rgb8 {
                red: 0,
                green: 0,
                blue: 0,
            },
            Rgb8 {
                red: 9,
                green: 8,
                blue: 7,
            },
        ];
        image.transparency = Transparency::PaletteAlpha(vec![0, 255]);
        assert!(image.validate().is_ok());
    }

    #[test]
    fn alpha_color_rejects_a_transparency_key() {
        let mut image =
            StillImage::from_samples(1, 1, ColorType::Rgba, BitDepth::Eight, vec![1, 2, 3, 4]);
        image.transparency = Transparency::ColorKey(vec![1, 2, 3]);
        assert_eq!(image.validate(), Err(ImageError::Transparency));
    }

    #[test]
    fn a_text_keyword_is_latin1_without_a_broken_space() {
        let mut image = StillImage::from_samples(1, 1, ColorType::Gray, BitDepth::Eight, vec![1]);
        image.texts.push(super::TextItem {
            keyword: b"Comment".to_vec(),
            language: Vec::new(),
            translated: Vec::new(),
            text: b"ok".to_vec(),
            compressed: false,
        });
        assert!(image.validate().is_ok());
        image.texts[0].keyword = b" Title".to_vec();
        assert_eq!(image.validate(), Err(ImageError::Metadata));
        image.texts[0].keyword = b"A  B".to_vec();
        assert_eq!(image.validate(), Err(ImageError::Metadata));
        image.texts[0].keyword = vec![0x7f];
        assert_eq!(image.validate(), Err(ImageError::Metadata));
    }

    #[test]
    fn sixteen_bit_truecolor_accepts_the_full_range() {
        let image =
            StillImage::from_samples(1, 1, ColorType::Rgb, BitDepth::Sixteen, vec![0, 65535, 256])
                .checked()
                .expect("deep");
        assert_eq!(image.depth.bits(), 16);
    }
}
