//! Decode packets until the picture queue is full.

use std::num::NonZeroUsize;

use cutlass_core::{Packet, Timestamp, VideoFrame};

use crate::queue::{PushError, Queue};

/// A decoder `Ahead` can pull from. The caller owns it.
pub trait Decode {
    type Image;
    type Error;

    /// Sends one compressed packet, in decode order.
    fn send(&mut self, packet: &Packet) -> Result<(), Self::Error>;

    /// The next decoded frame, in presentation order, if one is ready.
    fn receive(&mut self) -> Result<Option<VideoFrame<Self::Image>>, Self::Error>;

    /// Tells the decoder no more packets are coming.
    fn flush(&mut self) -> Result<(), Self::Error>;
}

/// Why [`Ahead::fill`] stopped.
#[derive(Debug)]
pub enum FillError<I, E> {
    /// The decoder returned an error.
    Decode(E),
    /// A frame's presentation time did not follow the previous one.
    OutOfOrder(VideoFrame<I>),
}

impl<I, E: std::fmt::Display> std::fmt::Display for FillError<I, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(error) => write!(formatter, "{error}"),
            Self::OutOfOrder(_) => {
                formatter.write_str("a video frame arrived out of presentation order")
            }
        }
    }
}

impl<I: std::fmt::Debug, E: std::error::Error + 'static> std::error::Error for FillError<I, E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::OutOfOrder(_) => None,
        }
    }
}

/// Fills a [`Queue`] from a decoder, and no further than that queue allows.
pub struct Ahead<I> {
    queue: Queue<I>,
    held: Option<VideoFrame<I>>,
    flushed: bool,
}

impl<I> Ahead<I> {
    /// `limit` is how many frames may sit in the queue, including the one on screen.
    pub fn new(limit: NonZeroUsize) -> Self {
        Self {
            queue: Queue::new(limit),
            held: None,
            flushed: false,
        }
    }

    /// How many frames are queued. A frame held back from a full queue is not counted.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// No frames are queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// The picture queue is at its depth.
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }

    /// Sends from `pending` until the queue is full or the decoder is drained.
    ///
    /// `pending` keeps the packets the decoder has not taken. Call this again
    /// after [`Self::at`] drops a frame. [`Self::seek`] starts a new span of
    /// packets; the caller also resets the decoder.
    pub fn fill<D>(
        &mut self,
        decoder: &mut D,
        pending: &mut &[Packet],
    ) -> Result<(), FillError<I, D::Error>>
    where
        D: Decode<Image = I>,
    {
        if let Some(frame) = self.held.take() {
            self.accept(frame)?;
        }
        loop {
            if self.held.is_some() || self.queue.is_full() {
                return Ok(());
            }
            match decoder.receive().map_err(FillError::Decode)? {
                Some(frame) => self.accept(frame)?,
                None if pending.is_empty() && self.flushed => return Ok(()),
                None if pending.is_empty() => {
                    decoder.flush().map_err(FillError::Decode)?;
                    self.flushed = true;
                }
                None => {
                    let packet = &pending[0];
                    decoder.send(packet).map_err(FillError::Decode)?;
                    *pending = &pending[1..];
                }
            }
        }
    }

    /// The frame to draw at `time`. Frames the playhead has passed are dropped.
    pub fn at(&mut self, time: Timestamp) -> Option<&VideoFrame<I>> {
        self.queue.at(time)
    }

    /// Drops every frame. The caller resets the decoder and passes the packets
    /// that start at [`crate::keyframe_before`].
    pub fn seek(&mut self) {
        self.queue.seek();
        self.held = None;
        self.flushed = false;
    }

    fn accept<E>(&mut self, frame: VideoFrame<I>) -> Result<(), FillError<I, E>> {
        match self.queue.push(frame) {
            Ok(()) => Ok(()),
            Err(PushError::Full(frame)) => {
                self.held = Some(frame);
                Ok(())
            }
            Err(PushError::OutOfOrder(frame)) => Err(FillError::OutOfOrder(frame)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::num::{NonZeroU32, NonZeroUsize};

    use cutlass_core::{Packet, Timestamp, VideoFrame};

    use super::{Ahead, Decode, FillError};

    fn time(ms: i64) -> Timestamp {
        Timestamp::new(ms, NonZeroU32::new(1_000).expect("timescale"))
    }

    fn packet(ms: i64) -> Packet {
        Packet {
            data: Vec::new(),
            pts: time(ms),
            dts: None,
            duration: None,
            keyframe: ms == 0,
        }
    }

    fn frame(ms: i64) -> VideoFrame<i64> {
        VideoFrame {
            pts: time(ms),
            duration: None,
            image: ms,
        }
    }

    /// Emits a frame only once a later packet is in, then the rest on flush.
    struct Lag {
        waiting: VecDeque<VideoFrame<i64>>,
        flushing: bool,
    }

    impl Decode for Lag {
        type Image = i64;
        type Error = &'static str;

        fn send(&mut self, packet: &Packet) -> Result<(), Self::Error> {
            self.waiting.push_back(frame(packet.pts.value));
            Ok(())
        }

        fn receive(&mut self) -> Result<Option<VideoFrame<i64>>, Self::Error> {
            if self.flushing || self.waiting.len() > 1 {
                Ok(self.waiting.pop_front())
            } else {
                Ok(None)
            }
        }

        fn flush(&mut self) -> Result<(), Self::Error> {
            self.flushing = true;
            Ok(())
        }
    }

    struct Scripted {
        frames: VecDeque<VideoFrame<i64>>,
    }

    impl Decode for Scripted {
        type Image = i64;
        type Error = &'static str;

        fn send(&mut self, _packet: &Packet) -> Result<(), Self::Error> {
            Ok(())
        }

        fn receive(&mut self) -> Result<Option<VideoFrame<i64>>, Self::Error> {
            Ok(self.frames.pop_front())
        }

        fn flush(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn filling_stops_when_the_queue_is_full() {
        let packets = [packet(0), packet(10), packet(20), packet(30)];
        let mut pending = packets.as_slice();
        let mut ahead = Ahead::new(NonZeroUsize::new(2).expect("limit"));
        ahead
            .fill(
                &mut Lag {
                    waiting: VecDeque::new(),
                    flushing: false,
                },
                &mut pending,
            )
            .expect("fill");
        assert!(ahead.is_full());
        assert_eq!(pending.len(), 1);
        let shown = ahead.at(time(0)).expect("frame");
        assert_eq!(shown.image, 0);
    }

    #[test]
    fn the_last_delayed_frame_comes_out_on_flush() {
        let packets = [packet(0), packet(10), packet(20)];
        let mut pending = packets.as_slice();
        let mut ahead = Ahead::new(NonZeroUsize::new(4).expect("limit"));
        ahead
            .fill(
                &mut Lag {
                    waiting: VecDeque::new(),
                    flushing: false,
                },
                &mut pending,
            )
            .expect("fill");
        assert!(pending.is_empty());
        assert_eq!(ahead.len(), 3);
        assert_eq!(ahead.at(time(20)).expect("frame").image, 20);
    }

    #[test]
    fn a_frame_out_of_order_is_returned() {
        let packets = [packet(0)];
        let mut pending = packets.as_slice();
        let mut ahead = Ahead::new(NonZeroUsize::new(4).expect("limit"));
        let error = ahead
            .fill(
                &mut Scripted {
                    frames: VecDeque::from([frame(20), frame(10)]),
                },
                &mut pending,
            )
            .expect_err("order");
        assert!(matches!(error, FillError::OutOfOrder(frame) if frame.image == 10));
    }

    #[test]
    fn seek_lets_a_new_span_start_at_the_same_time() {
        let packets = [packet(0), packet(10)];
        let mut pending = packets.as_slice();
        let mut ahead = Ahead::new(NonZeroUsize::new(4).expect("limit"));
        let mut lag = Lag {
            waiting: VecDeque::new(),
            flushing: false,
        };
        ahead.fill(&mut lag, &mut pending).expect("fill");
        ahead.seek();
        let again = [packet(0)];
        let mut pending = again.as_slice();
        ahead
            .fill(
                &mut Lag {
                    waiting: VecDeque::new(),
                    flushing: false,
                },
                &mut pending,
            )
            .expect("fill after seek");
        assert_eq!(ahead.at(time(0)).expect("frame").image, 0);
    }
}
