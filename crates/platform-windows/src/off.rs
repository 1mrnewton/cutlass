//! WASAPI is not available on this host.

use std::mem::size_of_val;

use crate::format::float_stereo;
use crate::slice::{QueueError, Slice, layout};

/// A WASAPI output. This host cannot open one.
pub struct OutputQueue {
    slice: Slice,
}

impl OutputQueue {
    /// Open the default shared-mode output at `rate` hertz and `channels`.
    ///
    /// The preview format is 48 kHz stereo `f32`. This host has no WASAPI
    /// device, so a matching format is still [`QueueError::NoDevice`].
    pub fn open(rate: u32, channels: u16) -> Result<Self, QueueError> {
        let frame = layout(rate, channels)?;
        let format = float_stereo(rate);
        if size_of_val(&format) != 40 {
            return Err(QueueError::Format);
        }
        let slice = Slice::new(frame)?;
        if slice.stride() != frame {
            return Err(QueueError::Format);
        }
        Err(QueueError::NoDevice)
    }

    /// Replace the slice the callback walks.
    pub fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        self.slice.set_sample(sample)
    }

    /// Copy the next frames into `out`.
    pub fn fill(&self, out: &mut [u8]) -> usize {
        self.slice.pull(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_refuses_a_bad_format_and_has_no_device_here() {
        assert!(matches!(OutputQueue::open(0, 2), Err(QueueError::Format)));
        assert!(matches!(
            OutputQueue::open(48_000, 1),
            Err(QueueError::Format)
        ));
        assert!(matches!(
            OutputQueue::open(48_000, 2),
            Err(QueueError::NoDevice)
        ));
    }
}
