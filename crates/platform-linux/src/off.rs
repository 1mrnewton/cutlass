//! ALSA is not available on this host.

use crate::slice::{QueueError, Slice, layout};

/// An ALSA output. This host cannot open one.
pub struct OutputQueue {
    slice: Slice,
}

impl OutputQueue {
    /// Open the default ALSA output at `rate` hertz and `channels`.
    ///
    /// The preview format is 48 kHz stereo `f32`. This host has no ALSA, so
    /// a matching format is still [`QueueError::NoDevice`].
    pub fn open(rate: u32, channels: u16) -> Result<Self, QueueError> {
        let frame = layout(rate, channels)?;
        let _slice = Slice::new(frame)?;
        Err(QueueError::NoDevice)
    }

    /// Replace the slice the device walks.
    pub fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        self.slice.set_sample(sample)
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
