//! One block to one frame: the channel assignment, the subframes, and the
//! frame header and CRC-16 (RFC 9639 section 9.1).

use cutlass_decode::CodecError;

use super::lpc;
use super::settings::{Settings, Stereo};
use super::subframe::{self, Workspace};
use crate::bits::BitWriter;
use crate::crc::crc16;
use crate::header::{ChannelAssignment, FrameHeader};

/// Encodes blocks for one stream.
pub(crate) struct FrameEncoder {
    settings: Settings,
    rate: u32,
    channels: usize,
    bits: u8,
    window_len: usize,
    windows: Vec<Vec<f64>>,
    planes: Vec<Vec<i64>>,
    subframes: Vec<BitWriter>,
    work: Workspace,
}

impl FrameEncoder {
    pub(crate) fn new(settings: Settings, rate: u32, channels: u8, bits: u8) -> Self {
        let channels = usize::from(channels);
        let slots = channels.max(4);
        Self {
            settings,
            rate,
            channels,
            bits,
            window_len: 0,
            windows: Vec::new(),
            planes: vec![Vec::new(); slots],
            subframes: (0..slots).map(|_| BitWriter::with_capacity(0)).collect(),
            work: Workspace::default(),
        }
    }

    pub(crate) fn settings(&self) -> &Settings {
        &self.settings
    }

    fn prepare_windows(&mut self, len: usize) {
        if self.window_len == len {
            return;
        }
        self.windows.clear();
        if self.settings.max_lpc_order > 0 {
            for apodization in &self.settings.apodizations {
                lpc::windows(*apodization, len, &mut self.windows);
            }
        }
        self.window_len = len;
    }

    fn code(&mut self, plane: usize, bits: u32) {
        subframe::encode(
            &self.planes[plane],
            bits,
            &self.settings,
            &self.windows,
            &mut self.work,
            &mut self.subframes[plane],
        );
    }

    /// Frame `number` holding `interleaved`, a whole number of samples per
    /// channel, 1 to 65535 of them.
    pub(crate) fn encode(
        &mut self,
        interleaved: &[i32],
        number: u64,
    ) -> Result<Vec<u8>, CodecError> {
        let channels = self.channels;
        let block = interleaved.len() / channels;
        if block == 0 || block > 65_535 || !interleaved.len().is_multiple_of(channels) {
            return Err(CodecError::InvalidApi { what: "block size" });
        }
        self.prepare_windows(block);
        for (index, plane) in self.planes.iter_mut().enumerate().take(channels) {
            plane.clear();
            plane.extend(
                interleaved
                    .iter()
                    .skip(index)
                    .step_by(channels)
                    .map(|sample| i64::from(*sample)),
            );
        }
        let bits = u32::from(self.bits);
        let assignment = if channels == 2 && self.settings.stereo != Stereo::Independent {
            self.stereo(bits)
        } else {
            for plane in 0..channels {
                self.code(plane, bits);
            }
            ChannelAssignment::Independent(u8::try_from(channels).unwrap_or(1))
        };
        let header = FrameHeader {
            variable_block: false,
            block_size: u32::try_from(block).unwrap_or(0),
            sample_rate: Some(self.rate),
            assignment,
            bits: Some(self.bits),
            number,
            len: 0,
        };
        let total: usize = self.subframes.iter().map(|s| s.bytes().len()).sum();
        let mut out = BitWriter::with_capacity(total + 32);
        header.write(&mut out)?;
        // Plane order for each assignment: see `stereo`.
        let order: &[usize] = match assignment {
            ChannelAssignment::Independent(_) => &[0, 1, 2, 3, 4, 5, 6, 7],
            ChannelAssignment::LeftSide => &[0, 3],
            ChannelAssignment::SideRight => &[3, 1],
            ChannelAssignment::MidSide => &[2, 3],
        };
        for plane in order.iter().take(channels) {
            out.append(&self.subframes[*plane]);
        }
        out.align();
        let mut bytes = out.into_bytes();
        let crc = crc16(&bytes);
        bytes.extend_from_slice(&crc.to_be_bytes());
        Ok(bytes)
    }

    /// Codes planes 0 (left) and 1 (right), derives 2 (mid) and 3 (side),
    /// and picks the assignment.
    fn stereo(&mut self, bits: u32) -> ChannelAssignment {
        let (left, rest) = self.planes.split_at_mut(1);
        let (right, rest) = rest.split_at_mut(1);
        let (mid, side) = rest.split_at_mut(1);
        let (left, right, mid, side) = (&left[0], &right[0], &mut mid[0], &mut side[0]);
        mid.clear();
        side.clear();
        for (l, r) in left.iter().zip(right.iter()) {
            mid.push((l + r) >> 1);
            side.push(l - r);
        }
        let pairs = [
            (ChannelAssignment::Independent(2), 0, 1),
            (ChannelAssignment::LeftSide, 0, 3),
            (ChannelAssignment::SideRight, 3, 1),
            (ChannelAssignment::MidSide, 2, 3),
        ];
        let plane_bits = |plane: usize| if plane == 3 { bits + 1 } else { bits };
        if self.settings.stereo == Stereo::Exhaustive {
            for plane in 0..4 {
                self.code(plane, plane_bits(plane));
            }
            let cost = |plane: usize| self.subframes[plane].bit_len();
            let mut best = pairs[0];
            for pair in pairs {
                if cost(pair.1) + cost(pair.2) < cost(best.1) + cost(best.2) {
                    best = pair;
                }
            }
            return best.0;
        }
        // Estimate from the best fixed predictor's residual magnitude.
        let estimate: Vec<u64> = (0..4)
            .map(|plane| {
                let (_, sum) = lpc::best_fixed_order(&self.planes[plane]);
                sum
            })
            .collect();
        let mut best = pairs[0];
        for pair in pairs {
            if estimate[pair.1] + estimate[pair.2] < estimate[best.1] + estimate[best.2] {
                best = pair;
            }
        }
        self.code(best.1, plane_bits(best.1));
        self.code(best.2, plane_bits(best.2));
        best.0
    }
}
