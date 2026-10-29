//! Preview audio.
//!
//! On Apple the callback copies one pipeline sample into a Core Audio queue
//! buffer. Linux has no output device, so playback stays silent until one
//! exists. That silence is not a missing build, and opening the preview is
//! not a failed open. WASAPI and AAudio are not in this pass.
//!
//! Eval's picture call does not hand back PCM. The slice queued with a frame
//! is empty, and the callback copies that as one silent sample. Playback does
//! not mix.

use std::fmt;

use crate::playhead::PlayheadError;

/// Preview sample rate, in hertz.
pub(crate) const RATE: u32 = 48_000;

/// Interleaved stereo.
pub(crate) const CHANNELS: u16 = 2;

/// One stereo `f32` frame.
pub(crate) const FRAME_BYTES: usize = 8;

/// Largest slice the callback will look at. A longer slice is refused
/// before any of it is copied.
pub const MAX_SLICE_BYTES: usize = 64 * 1024;

/// Why the preview is not writing samples to a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioGap {
    /// Linux has no output device. Playback stays silent until one exists.
    /// This names silence. It is not a missing build, and it is not a failed open.
    Linux,
    /// The host has no output device. WASAPI and AAudio are not in this pass,
    /// and a Core Audio queue that cannot be opened is the same result.
    NoDevice,
}

impl fmt::Display for AudioGap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Linux => "linux playback is silent until an output device exists",
            Self::NoDevice => "the audio output device is missing",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for AudioGap {}

/// Which output the caller asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioHost {
    /// A buffer in this process. No OS device is opened.
    Software,
    /// Core Audio queue.
    Apple,
    /// No output device yet.
    Linux,
    /// WASAPI and AAudio are not in this pass.
    Unnamed,
}

impl AudioHost {
    #[must_use]
    pub fn current() -> Self {
        #[cfg(target_vendor = "apple")]
        {
            Self::Apple
        }
        #[cfg(target_os = "linux")]
        {
            Self::Linux
        }
        #[cfg(all(not(target_vendor = "apple"), not(target_os = "linux")))]
        {
            Self::Unnamed
        }
    }
}

/// One sample landed in the buffer, or the output is silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyOutcome {
    /// The first pipeline sample was copied. A longer slice kept the rest.
    Copied(usize),
    /// No device received the sample.
    Silent(AudioGap),
}

/// The bytes the callback last copied, and the Apple queue when one is open.
struct SampleBuffer {
    copied: Vec<u8>,
    #[cfg(target_vendor = "apple")]
    live: Option<cutlass_platform_apple::OutputQueue>,
}

impl SampleBuffer {
    fn software() -> Self {
        Self {
            copied: Vec::new(),
            #[cfg(target_vendor = "apple")]
            live: None,
        }
    }

    #[cfg(target_vendor = "apple")]
    fn core_audio(queue: cutlass_platform_apple::OutputQueue) -> Self {
        Self {
            copied: Vec::new(),
            live: Some(queue),
        }
    }

    fn copy_one(&mut self, sample: &[u8], buffer: &mut [u8]) -> Result<usize, PlayheadError> {
        let mut one = [0_u8; FRAME_BYTES];
        let copied = copy_one_sample(sample, FRAME_BYTES, &mut one)?;
        if buffer.len() < copied {
            return Err(PlayheadError::Truncated);
        }
        self.store_live(&one[..copied])?;
        buffer[..copied].copy_from_slice(&one[..copied]);
        self.copied = one[..copied].to_vec();
        Ok(copied)
    }

    fn set_slice(&mut self, sample: &[u8]) {
        let _ = self.store_live(sample);
    }

    fn store_live(&self, sample: &[u8]) -> Result<(), PlayheadError> {
        #[cfg(target_vendor = "apple")]
        if let Some(queue) = &self.live {
            return queue.set_sample(sample).map_err(map_queue_error);
        }
        let _ = (self, sample);
        Ok(())
    }
}

enum OutputKind {
    /// Copies one pipeline sample into a buffer. Tests use this with no OS device.
    Buffer(SampleBuffer),
    /// Pictures still play. Samples are not written anywhere.
    Silent(AudioGap),
}

/// The preview's audio output.
pub struct AudioOutput {
    kind: OutputKind,
}

impl AudioOutput {
    #[must_use]
    pub(crate) fn software() -> Self {
        Self {
            kind: OutputKind::Buffer(SampleBuffer::software()),
        }
    }

    /// Open the output for `host`. Linux is silence, not an error.
    #[must_use]
    pub fn open(host: AudioHost) -> Self {
        match host {
            AudioHost::Software => Self::software(),
            AudioHost::Linux => Self {
                kind: OutputKind::Silent(AudioGap::Linux),
            },
            AudioHost::Apple => open_apple(),
            AudioHost::Unnamed => Self {
                kind: OutputKind::Silent(AudioGap::NoDevice),
            },
        }
    }

    pub(crate) fn copy_one(
        &mut self,
        sample: &[u8],
        buffer: &mut [u8],
    ) -> Result<CopyOutcome, PlayheadError> {
        match &mut self.kind {
            OutputKind::Silent(gap) => Ok(CopyOutcome::Silent(*gap)),
            OutputKind::Buffer(queue) => queue.copy_one(sample, buffer).map(CopyOutcome::Copied),
        }
    }

    pub(crate) fn set_slice(&mut self, sample: &[u8]) {
        if let OutputKind::Buffer(queue) = &mut self.kind {
            queue.set_slice(sample);
        }
    }

    #[must_use]
    pub fn gap(&self) -> Option<AudioGap> {
        match self.kind {
            OutputKind::Silent(gap) => Some(gap),
            OutputKind::Buffer(_) => None,
        }
    }
}

fn open_apple() -> AudioOutput {
    #[cfg(target_vendor = "apple")]
    {
        match cutlass_platform_apple::OutputQueue::open(RATE, CHANNELS) {
            Ok(queue) => AudioOutput {
                kind: OutputKind::Buffer(SampleBuffer::core_audio(queue)),
            },
            Err(_) => AudioOutput {
                kind: OutputKind::Silent(AudioGap::NoDevice),
            },
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        AudioOutput {
            kind: OutputKind::Silent(AudioGap::NoDevice),
        }
    }
}

#[cfg(target_vendor = "apple")]
fn map_queue_error(error: cutlass_platform_apple::QueueError) -> PlayheadError {
    match error {
        cutlass_platform_apple::QueueError::Truncated => PlayheadError::Truncated,
        cutlass_platform_apple::QueueError::Hostile => PlayheadError::Hostile,
        cutlass_platform_apple::QueueError::NoDevice
        | cutlass_platform_apple::QueueError::Format => PlayheadError::Hostile,
    }
}

/// Copy the first pipeline sample into `buffer`.
///
/// An empty slice is one silent sample, so a missing mix does not stall the
/// clock. A short non-empty slice, or a buffer shorter than one sample, is
/// truncated. A slice past [`MAX_SLICE_BYTES`] is hostile and is not copied.
pub fn copy_one_sample(
    sample: &[u8],
    frame_bytes: usize,
    buffer: &mut [u8],
) -> Result<usize, PlayheadError> {
    if frame_bytes == 0 || frame_bytes > FRAME_BYTES {
        return Err(PlayheadError::Hostile);
    }
    if sample.len() > MAX_SLICE_BYTES {
        return Err(PlayheadError::Hostile);
    }
    if buffer.len() < frame_bytes {
        return Err(PlayheadError::Truncated);
    }
    if !sample.is_empty()
        && (sample.len() < frame_bytes || !sample.len().is_multiple_of(frame_bytes))
    {
        return Err(PlayheadError::Truncated);
    }
    if sample.is_empty() {
        buffer[..frame_bytes].fill(0);
    } else {
        buffer[..frame_bytes].copy_from_slice(&sample[..frame_bytes]);
    }
    Ok(frame_bytes)
}

#[cfg(test)]
mod tests {
    use cutlass_model::Micros;
    use cutlass_renderer_software::SoftwareDevice;

    use super::*;
    use crate::color_session;
    use crate::playhead::Player;

    const COLOR: [u8; 4] = [0x33, 0x66, 0x99, 0xFF];

    #[test]
    fn the_callback_copies_one_pipeline_sample() {
        let mut player = Player::new();
        let mut buffer = [0_u8; 32];
        let sample = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let outcome = player
            .copy_pipeline_sample(&sample, &mut buffer)
            .expect("copy");
        assert_eq!(outcome, CopyOutcome::Copied(FRAME_BYTES));
        assert_eq!(&buffer[..FRAME_BYTES], &sample[..FRAME_BYTES]);
        assert_eq!(&buffer[FRAME_BYTES..], &[0_u8; 24]);
        assert_eq!(player.audio_time(), Micros(1_000_000 / i64::from(RATE)));
    }

    #[test]
    fn an_empty_slice_is_one_silent_sample() {
        let mut player = Player::new();
        let mut buffer = [9_u8; 16];
        let outcome = player
            .copy_pipeline_sample(&[], &mut buffer)
            .expect("silence");
        assert_eq!(outcome, CopyOutcome::Copied(FRAME_BYTES));
        assert_eq!(&buffer[..FRAME_BYTES], &[0_u8; FRAME_BYTES]);
        assert_eq!(&buffer[FRAME_BYTES..], &[9_u8; 8]);
        assert_eq!(player.audio_time(), Micros(20));
    }

    #[test]
    fn a_truncated_or_hostile_sample_is_an_error() {
        let mut player = Player::new();
        let mut buffer = [0_u8; 16];
        let error = player
            .copy_pipeline_sample(&[1, 2, 3], &mut buffer)
            .expect_err("short");
        assert_eq!(error, PlayheadError::Truncated);
        assert_eq!(buffer, [0_u8; 16]);
        assert_eq!(player.audio_time(), Micros(0));

        let mut tiny = [0_u8; 4];
        let error = player
            .copy_pipeline_sample(&[1, 2, 3, 4, 5, 6, 7, 8], &mut tiny)
            .expect_err("buffer");
        assert_eq!(error, PlayheadError::Truncated);
        assert_eq!(tiny, [0_u8; 4]);

        let hostile = vec![0_u8; MAX_SLICE_BYTES + 1];
        let error = player
            .copy_pipeline_sample(&hostile, &mut buffer)
            .expect_err("hostile");
        assert_eq!(error, PlayheadError::Hostile);
        assert_eq!(player.audio_time(), Micros(0));
        assert_eq!(buffer, [0_u8; 16]);
    }

    #[test]
    fn linux_stays_silent_and_pictures_still_play() {
        let session = color_session(1_000_000, 1_000_000);
        let mut player = Player::new();
        player.open_output(AudioHost::Linux);
        assert_eq!(player.output().gap(), Some(AudioGap::Linux));
        let text = AudioGap::Linux.to_string();
        assert!(text.contains("silent"));
        assert!(!text.contains("missing build"));
        let mut device = SoftwareDevice::new();
        let frame = player
            .seek(&session, &mut device, Micros(10))
            .expect("picture");
        assert_eq!(frame.rgba, COLOR.repeat(2));
        let mut buffer = [9_u8; 8];
        let outcome = player
            .copy_pipeline_sample(&[1, 2, 3, 4, 5, 6, 7, 8], &mut buffer)
            .expect("named");
        assert_eq!(outcome, CopyOutcome::Silent(AudioGap::Linux));
        assert_eq!(buffer, [9_u8; 8]);
        assert_eq!(player.audio_time(), Micros(0));
    }

    #[test]
    fn a_missing_device_is_a_named_result() {
        let output = AudioOutput::open(AudioHost::Unnamed);
        assert_eq!(output.gap(), Some(AudioGap::NoDevice));
        assert_eq!(
            AudioGap::NoDevice.to_string(),
            "the audio output device is missing"
        );
        #[cfg(not(target_vendor = "apple"))]
        {
            let apple = AudioOutput::open(AudioHost::Apple);
            assert_eq!(apple.gap(), Some(AudioGap::NoDevice));
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    #[ignore = "opens the default Core Audio output"]
    fn opening_core_audio_names_a_queue_or_a_missing_device() {
        let output = AudioOutput::open(AudioHost::Apple);
        assert!(
            output.gap().is_none() || output.gap() == Some(AudioGap::NoDevice),
            "apple open must name the queue or the missing device"
        );
    }

    #[test]
    fn the_current_host_matches_this_process() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(AudioHost::current(), AudioHost::Apple);
        #[cfg(target_os = "linux")]
        assert_eq!(AudioHost::current(), AudioHost::Linux);
        #[cfg(all(not(target_vendor = "apple"), not(target_os = "linux")))]
        assert_eq!(AudioHost::current(), AudioHost::Unnamed);
    }
}
