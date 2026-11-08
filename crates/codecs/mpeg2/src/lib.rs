//! MPEG-2 video (ISO/IEC 13818-2, ITU-T H.262) and MPEG-1 video decoding,
//! and MPEG-2 Main profile encoding.

#![forbid(unsafe_code)]

mod bits;
mod dct;
mod headers;
mod tables;
