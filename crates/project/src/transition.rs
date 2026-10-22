//! A blend on a main-track cut.
//!
//! The two segments stay touching. Overlap does not move them. The project
//! does not open the plugin and does not store a shader.

use crate::Error;
use crate::attachment::Attachment;
use crate::id::SegmentId;

/// The blend stored on one cut. The left segment ends where the right starts.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    left: SegmentId,
    right: SegmentId,
    duration_us: i64,
    overlap: bool,
    attachment: Attachment,
}

impl Transition {
    pub(crate) fn new(
        left: SegmentId,
        right: SegmentId,
        duration_us: i64,
        overlap: bool,
        attachment: Attachment,
    ) -> Result<Self, Error> {
        if left == right {
            return Err(Error::Transition);
        }
        if duration_us < 0 {
            return Err(Error::NegativeTime);
        }
        if duration_us == 0 {
            return Err(Error::ZeroDuration);
        }
        if overlap && duration_us < 2 {
            return Err(Error::Transition);
        }
        Ok(Self {
            left,
            right,
            duration_us,
            overlap,
            attachment,
        })
    }

    pub const fn left(&self) -> SegmentId {
        self.left
    }

    pub const fn right(&self) -> SegmentId {
        self.right
    }

    pub const fn duration_us(&self) -> i64 {
        self.duration_us
    }

    pub const fn overlap(&self) -> bool {
        self.overlap
    }

    pub const fn attachment(&self) -> &Attachment {
        &self.attachment
    }

    /// Time taken from the left segment, then time taken from the right.
    ///
    /// Overlap off uses the whole duration on the left. Overlap on gives the
    /// left half, rounded down, and the remainder to the right.
    pub(crate) const fn sides(&self) -> (i64, i64) {
        if self.overlap {
            let before = self.duration_us / 2;
            (before, self.duration_us - before)
        } else {
            (self.duration_us, 0)
        }
    }

    pub(crate) const fn fits(&self, left_dur: i64, right_dur: i64) -> bool {
        let (before, after) = self.sides();
        left_dur >= before && right_dur >= after
    }

    /// `at` is strictly inside the blend around `cut`.
    pub(crate) const fn covers(&self, cut: i64, at: i64) -> bool {
        let (before, after) = self.sides();
        let Some(start) = cut.checked_sub(before) else {
            return false;
        };
        let Some(end) = cut.checked_add(after) else {
            return false;
        };
        at > start && at < end
    }

    pub(crate) const fn retarget_left(&mut self, left: SegmentId) {
        self.left = left;
    }
}
