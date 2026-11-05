//! Placeholder until the header writer lands.
#![allow(dead_code)]
use crate::dec::obu::{FrameHeader, HeaderContext, SequenceHeader};
#[derive(Default)]
pub struct FrameHeaderWriteOptions;
pub fn write_sequence_header(_s: &SequenceHeader) -> Vec<u8> {
    Vec::new()
}
pub fn write_frame_header(
    _h: &FrameHeader,
    _c: &HeaderContext<'_>,
    _o: &FrameHeaderWriteOptions,
) -> Vec<u8> {
    Vec::new()
}
