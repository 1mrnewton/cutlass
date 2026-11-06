//! The frame header's H.273 color code points, both ways.

use cutlass_decode::{ColorPrimaries, MatrixCoefficients, Transfer};

pub(crate) fn primaries(code: u8) -> ColorPrimaries {
    match code {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Smpte428,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => ColorPrimaries::Unspecified,
    }
}

pub(crate) fn transfer(code: u8) -> Transfer {
    match code {
        1 | 6 => Transfer::Bt709,
        4 | 5 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        11 => Transfer::Iec61966_2_4,
        12 => Transfer::Bt1361,
        13 => Transfer::Srgb,
        14 | 15 => Transfer::Bt2020,
        16 => Transfer::Pq,
        17 => Transfer::Smpte428,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
    }
}

pub(crate) fn matrix(code: u8) -> MatrixCoefficients {
    match code {
        1 => MatrixCoefficients::Bt709,
        5 | 6 => MatrixCoefficients::Bt601,
        9 | 10 => MatrixCoefficients::Bt2020,
        _ => MatrixCoefficients::Unspecified,
    }
}

/// Code 2, "unspecified", for anything the frame does not name.
pub(crate) fn primaries_code(value: ColorPrimaries) -> u8 {
    match value {
        ColorPrimaries::Bt709 => 1,
        ColorPrimaries::Bt470m => 4,
        ColorPrimaries::Bt601 => 6,
        ColorPrimaries::Smpte240m => 7,
        ColorPrimaries::Film => 8,
        ColorPrimaries::Bt2020 => 9,
        ColorPrimaries::Smpte428 => 10,
        ColorPrimaries::Smpte431 => 11,
        ColorPrimaries::Smpte432 => 12,
        ColorPrimaries::Ebu3213 => 22,
        ColorPrimaries::Unspecified => 2,
    }
}

pub(crate) fn transfer_code(value: Transfer) -> u8 {
    match value {
        Transfer::Bt709 => 1,
        Transfer::Bt601 => 6,
        Transfer::Bt2020 => 14,
        Transfer::Smpte240m => 7,
        Transfer::Linear => 8,
        Transfer::Log100 => 9,
        Transfer::LogSqrt => 10,
        Transfer::Iec61966_2_4 => 11,
        Transfer::Bt1361 => 12,
        Transfer::Srgb => 13,
        Transfer::Pq => 16,
        Transfer::Smpte428 => 17,
        Transfer::Hlg => 18,
        Transfer::Unspecified => 2,
    }
}

pub(crate) fn matrix_code(value: MatrixCoefficients) -> u8 {
    match value {
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Bt601 => 6,
        MatrixCoefficients::Bt2020 => 9,
        MatrixCoefficients::Rgb | MatrixCoefficients::Unspecified => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::{matrix, matrix_code, primaries, primaries_code, transfer, transfer_code};

    #[test]
    fn named_codes_survive_a_round_trip() {
        for code in [1u8, 4, 6, 7, 8, 9, 10, 11, 12, 22] {
            assert_eq!(primaries_code(primaries(code)), code);
        }
        for code in [1u8, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18] {
            assert_eq!(transfer_code(transfer(code)), code);
        }
        for code in [1u8, 6, 9] {
            assert_eq!(matrix_code(matrix(code)), code);
        }
        assert_eq!(primaries_code(primaries(2)), 2);
    }
}
