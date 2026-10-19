//! Frame geometry: the visible region within a (possibly padded) decode buffer,
//! and the display rotation a container can attach.

/// An axis-aligned rectangle in pixels, used for the visible/crop region inside
/// a coded frame buffer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// A rectangle at the origin covering `width × height`.
    pub const fn from_size(width: u32, height: u32) -> Self {
        Self::new(0, 0, width, height)
    }

    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Display rotation applied to a frame before it's shown, in degrees clockwise.
///
/// Phones record sensor-native and stash the orientation in container metadata;
/// honoring it is the difference between upright and sideways playback, so it
/// rides with the frame rather than being baked into pixels at decode time.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Rotation {
    #[default]
    None,
    Cw90,
    Cw180,
    Cw270,
}

impl Rotation {
    /// True when the rotation swaps width and height (90° or 270°).
    pub fn swaps_axes(self) -> bool {
        matches!(self, Rotation::Cw90 | Rotation::Cw270)
    }

    /// Degrees clockwise (0/90/180/270).
    pub fn degrees(self) -> u32 {
        match self {
            Rotation::None => 0,
            Rotation::Cw90 => 90,
            Rotation::Cw180 => 180,
            Rotation::Cw270 => 270,
        }
    }

    /// Build from clockwise degrees, accepting any multiple of 90 (normalized
    /// into `0..360`). Returns `None` for non-multiples of 90.
    pub fn from_degrees(deg: i32) -> Option<Self> {
        match deg.rem_euclid(360) {
            0 => Some(Rotation::None),
            90 => Some(Rotation::Cw90),
            180 => Some(Rotation::Cw180),
            270 => Some(Rotation::Cw270),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_basics() {
        let r = Rect::from_size(1920, 1080);
        assert_eq!(r, Rect::new(0, 0, 1920, 1080));
        assert!(!r.is_empty());
        assert!(Rect::new(0, 0, 0, 100).is_empty());
    }

    #[test]
    fn rotation_axis_swap() {
        assert!(!Rotation::None.swaps_axes());
        assert!(Rotation::Cw90.swaps_axes());
        assert!(!Rotation::Cw180.swaps_axes());
        assert!(Rotation::Cw270.swaps_axes());
    }

    #[test]
    fn rotation_from_degrees_normalizes() {
        assert_eq!(Rotation::from_degrees(0), Some(Rotation::None));
        assert_eq!(Rotation::from_degrees(90), Some(Rotation::Cw90));
        assert_eq!(Rotation::from_degrees(-90), Some(Rotation::Cw270));
        assert_eq!(Rotation::from_degrees(450), Some(Rotation::Cw90));
        assert_eq!(Rotation::from_degrees(45), None);
    }
}
