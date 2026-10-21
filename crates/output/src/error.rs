//! AudioQueue status codes mapped to Cutlass errors.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// This machine has no output device the queue can open.
    #[error("{call}: no audio output device (OSStatus {status})")]
    NoDevice { call: &'static str, status: i32 },
    #[error("{call} failed with OSStatus {status}")]
    Os { call: &'static str, status: i32 },
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("the audio output lock was poisoned")]
    Poisoned,
}

impl Error {
    /// The caller should skip playback. There is no device to open.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::NoDevice { .. })
    }
}

const NO_DEVICE: [i32; 3] = [
    objc2_audio_toolbox::kAudioQueueErr_InvalidDevice,
    objc2_audio_toolbox::kAudioQueueErr_CannotStart,
    objc2_audio_toolbox::kAudioQueueErr_Permissions,
];

pub(crate) fn check(call: &'static str, status: i32) -> Result<(), Error> {
    if status == 0 {
        Ok(())
    } else if NO_DEVICE.contains(&status) {
        Err(Error::NoDevice { call, status })
    } else {
        Err(Error::Os { call, status })
    }
}
