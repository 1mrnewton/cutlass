//! Cutlass timestamps to and from `CMTime`.

use std::num::NonZeroU32;

use cutlass_core::Timestamp;
use objc2_core_media::{CMTime, CMTimeFlags};

use crate::error::Error;

/// Equal to `kCMTimeInvalid`, which is all zeros.
pub(crate) const INVALID: CMTime = CMTime {
    value: 0,
    timescale: 0,
    flags: CMTimeFlags(0),
    epoch: 0,
};

pub(crate) fn to_cm_time(time: Timestamp) -> Result<CMTime, Error> {
    let timescale = i32::try_from(time.timescale.get())
        .map_err(|_| Error::InvalidInput("CMTime needs a timescale of at most i32::MAX"))?;
    Ok(CMTime {
        value: time.value,
        timescale,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    })
}

pub(crate) fn to_cm_time_or_invalid(time: Option<Timestamp>) -> Result<CMTime, Error> {
    time.map_or(Ok(INVALID), to_cm_time)
}

/// `None` for invalid, infinite, and indefinite times.
pub(crate) fn from_cm_time(time: CMTime) -> Option<Timestamp> {
    let flags = time.flags;
    if !flags.contains(CMTimeFlags::Valid) || flags.intersects(CMTimeFlags::ImpliedValueFlagsMask) {
        return None;
    }
    let timescale = NonZeroU32::new(u32::try_from(time.timescale).ok()?)?;
    Some(Timestamp::new(time.value, timescale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_timestamp() {
        let time = Timestamp::new(-1001, NonZeroU32::new(30000).unwrap());
        assert_eq!(from_cm_time(to_cm_time(time).unwrap()), Some(time));
    }

    #[test]
    fn rejects_timescale_above_i32_max() {
        let time = Timestamp::new(0, NonZeroU32::new(u32::MAX).unwrap());
        assert!(to_cm_time(time).is_err());
    }

    #[test]
    fn invalid_time_has_no_timestamp() {
        assert_eq!(from_cm_time(INVALID), None);
    }
}
