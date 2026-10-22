//! Clockwise rotation of a video stream.
//!
//! The container's track matrix names one of four right angles. Any other
//! matrix stays [`Rotation::Other`].

/// Clockwise rotation stored in a track matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rotation {
    /// The identity matrix.
    #[default]
    Zero,
    /// 90 degrees clockwise.
    Clockwise90,
    /// 180 degrees clockwise.
    Clockwise180,
    /// 270 degrees clockwise.
    Clockwise270,
    /// A matrix that is not one of those four.
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_identity_matrix_is_the_default() {
        assert_eq!(Rotation::default(), Rotation::Zero);
    }
}
