//! MPEG-4 Part 2 (MPEG-4 Visual, ISO/IEC 14496-2) decoder and encoder.

#![forbid(unsafe_code)]

mod bits;
mod block;
mod frame;
mod header;
mod idct;
mod mc;
mod tables;
mod vlc;
mod vop;
