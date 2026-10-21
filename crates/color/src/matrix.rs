//! H.273 matrix coefficients.

use crate::Error;

/// The luma and chroma weights for one video matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matrix {
    /// H.273 codes 5 and 6. Kr 0.299, Kb 0.114.
    Bt601,
    /// H.273 code 1. Kr 0.2126, Kb 0.0722.
    Bt709,
    /// H.273 code 9, non-constant luminance. Kr 0.2627, Kb 0.0593.
    Bt2020,
}

impl Matrix {
    /// `None` for code 0 (unspecified) and for codes this crate does not convert.
    pub const fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Bt709),
            5 | 6 => Some(Self::Bt601),
            9 => Some(Self::Bt2020),
            _ => None,
        }
    }

    /// The matrix used when the file leaves it unspecified.
    ///
    /// A frame is HD when its width is at least 1280 or its height is at least
    /// 720. HD uses BT.709. Anything smaller uses BT.601.
    pub const fn for_size(width: u32, height: u32) -> Self {
        if width >= 1280 || height >= 720 {
            Self::Bt709
        } else {
            Self::Bt601
        }
    }

    /// Resolves a container matrix code. Missing and 0 follow [`Self::for_size`].
    pub fn resolve(code: Option<u16>, width: u32, height: u32) -> Result<Self, Error> {
        match code {
            None | Some(0) => Ok(Self::for_size(width, height)),
            Some(code) => Self::from_code(code).ok_or(Error::UnsupportedMatrix(code)),
        }
    }

    pub const fn kr(self) -> f32 {
        match self {
            Self::Bt601 => 0.299,
            Self::Bt709 => 0.2126,
            Self::Bt2020 => 0.2627,
        }
    }

    pub const fn kb(self) -> f32 {
        match self {
            Self::Bt601 => 0.114,
            Self::Bt709 => 0.0722,
            Self::Bt2020 => 0.0593,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes_select_their_matrix() {
        assert_eq!(Matrix::from_code(1), Some(Matrix::Bt709));
        assert_eq!(Matrix::from_code(5), Some(Matrix::Bt601));
        assert_eq!(Matrix::from_code(6), Some(Matrix::Bt601));
        assert_eq!(Matrix::from_code(9), Some(Matrix::Bt2020));
        assert_eq!(Matrix::from_code(0), None);
        assert_eq!(Matrix::from_code(4), None);
    }

    #[test]
    fn an_unspecified_matrix_follows_the_frame_size() {
        assert_eq!(Matrix::resolve(None, 720, 480).expect("sd"), Matrix::Bt601);
        assert_eq!(
            Matrix::resolve(Some(0), 1280, 720).expect("hd"),
            Matrix::Bt709
        );
        assert_eq!(
            Matrix::resolve(Some(9), 320, 240).expect("tagged"),
            Matrix::Bt2020
        );
        assert_eq!(
            Matrix::resolve(Some(4), 320, 240),
            Err(Error::UnsupportedMatrix(4))
        );
    }
}
