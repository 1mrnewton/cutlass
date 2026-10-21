//! Decoded frames in presentation order, and the one the playhead should show.

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use cutlass_core::{Timestamp, VideoFrame};

/// Why [`Queue::push`] gave a frame back.
#[derive(Debug)]
pub enum PushError<I> {
    /// The queue is at its depth. The decoder should wait.
    Full(VideoFrame<I>),
    /// `pts` is not later than the last frame accepted since the last seek.
    OutOfOrder(VideoFrame<I>),
}

impl<I> std::fmt::Display for PushError<I> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Full(_) => "the playback queue is full",
            Self::OutOfOrder(_) => "a video frame arrived out of presentation order",
        })
    }
}

impl<I: std::fmt::Debug> std::error::Error for PushError<I> {}

/// Frames kept ahead of the playhead, including the one on screen.
pub struct Queue<I> {
    limit: usize,
    frames: VecDeque<VideoFrame<I>>,
    last: Option<Timestamp>,
}

impl<I> Queue<I> {
    /// `limit` is how many frames may sit here, including the one on screen.
    pub fn new(limit: NonZeroUsize) -> Self {
        Self {
            limit: limit.get(),
            frames: VecDeque::with_capacity(limit.get()),
            last: None,
        }
    }

    /// How many frames are queued.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// No frames are queued.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The queue is at the depth from [`Self::new`].
    pub fn is_full(&self) -> bool {
        self.frames.len() == self.limit
    }

    /// Appends `frame`. Presentation times must increase.
    ///
    /// On failure the frame is returned. A full queue means the decoder
    /// should wait, not that the frame is invalid.
    pub fn push(&mut self, frame: VideoFrame<I>) -> Result<(), PushError<I>> {
        if self.is_full() {
            return Err(PushError::Full(frame));
        }
        if self.last.is_some_and(|last| frame.pts <= last) {
            return Err(PushError::OutOfOrder(frame));
        }
        self.last = Some(frame.pts);
        self.frames.push_back(frame);
        Ok(())
    }

    /// The frame to draw at `time`, dropping any frame an earlier one has replaced.
    ///
    /// A frame that has not started stays queued. The frame on screen stays
    /// until a later frame is due.
    pub fn at(&mut self, time: Timestamp) -> Option<&VideoFrame<I>> {
        while self.frames.len() >= 2 && self.frames[1].pts <= time {
            self.frames.pop_front();
        }
        self.frames.front().filter(|frame| frame.pts <= time)
    }

    /// Drops every frame. The next [`Self::push`] may start at any time.
    pub fn seek(&mut self) {
        self.frames.clear();
        self.last = None;
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};

    use cutlass_core::{Timestamp, VideoFrame};

    use super::{PushError, Queue};

    fn frame(ms: i64) -> VideoFrame<i64> {
        VideoFrame {
            pts: Timestamp::new(ms, NonZeroU32::new(1_000).expect("timescale")),
            duration: None,
            image: ms,
        }
    }

    fn queue(limit: usize) -> Queue<i64> {
        Queue::new(NonZeroUsize::new(limit).expect("limit"))
    }

    #[test]
    fn a_future_frame_is_not_shown() {
        let mut frames = queue(4);
        frames.push(frame(100)).expect("push");
        assert!(frames.at(frame(0).pts).is_none());
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn the_frame_at_the_playhead_is_shown() {
        let mut frames = queue(4);
        frames.push(frame(0)).expect("push");
        frames.push(frame(40)).expect("push");
        let shown = frames.at(frame(40).pts).expect("frame");
        assert_eq!(shown.image, 40);
    }

    #[test]
    fn the_shown_frame_is_held_until_the_next_one() {
        let mut frames = queue(4);
        frames.push(frame(0)).expect("push");
        frames.push(frame(40)).expect("push");
        let shown = frames.at(frame(39).pts).expect("frame");
        assert_eq!(shown.image, 0);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn frames_behind_the_playhead_are_dropped() {
        let mut frames = queue(4);
        for ms in [0, 40, 80, 120] {
            frames.push(frame(ms)).expect("push");
        }
        let shown = frames.at(frame(80).pts).expect("frame");
        assert_eq!(shown.image, 80);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn a_full_queue_returns_the_frame() {
        let mut frames = queue(2);
        frames.push(frame(0)).expect("push");
        frames.push(frame(40)).expect("push");
        let error = frames.push(frame(80)).expect_err("full");
        assert!(matches!(error, PushError::Full(frame) if frame.image == 80));
        assert!(frames.is_full());
    }

    #[test]
    fn a_frame_out_of_order_is_returned() {
        let mut frames = queue(4);
        frames.push(frame(0)).expect("push");
        frames.push(frame(40)).expect("push");
        let _ = frames.at(frame(40).pts);
        let error = frames.push(frame(20)).expect_err("order");
        assert!(matches!(error, PushError::OutOfOrder(frame) if frame.image == 20));
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn seek_drops_queued_frames() {
        let mut frames = queue(4);
        frames.push(frame(0)).expect("push");
        frames.push(frame(40)).expect("push");
        frames.seek();
        assert!(frames.is_empty());
        frames.push(frame(0)).expect("push after seek");
        let shown = frames.at(frame(0).pts).expect("frame");
        assert_eq!(shown.image, 0);
    }
}
