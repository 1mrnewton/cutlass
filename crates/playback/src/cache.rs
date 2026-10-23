//! Decoded pictures kept by start time.
//!
//! A frame covers a half-open range. An overlapping insert clips the shared
//! boundary. Playback drops frames behind the playhead down to a floor. A seek
//! that needs room drops the end farther from the requested time. A seek that
//! misses leaves the stored frames where they are.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::ops::Bound;

use cutlass_core::{Timestamp, VideoFrame};

/// Frames one track has already decoded.
pub struct FrameCache<I> {
    frames: BTreeMap<Timestamp, Entry<I>>,
    maximum: usize,
    minimum: usize,
}

struct Entry<I> {
    end: Timestamp,
    /// `end` came from the frame's duration. A frame without one stretches to
    /// the next frame's start.
    bounded: bool,
    frame: VideoFrame<I>,
}

/// What [`FrameCache::insert`] did with the frames that did not fit.
pub struct Remember<I> {
    /// Frames the cache dropped to make room, earliest first.
    pub evicted: Vec<VideoFrame<I>>,
    /// `frame` was not stored. Playback only inserts when the count fits
    /// after the frames behind the playhead have been dropped.
    pub refused: Option<VideoFrame<I>>,
}

/// Decoded pictures from many sources, capped as one pool.
pub struct FramePool<I> {
    maximum: NonZeroUsize,
    clips: HashMap<String, FrameCache<I>>,
    stored: usize,
}

impl<I> FrameCache<I> {
    /// `maximum` is how many frames may be stored. `minimum` is the floor
    /// playback keeps, and it is raised to 1 and lowered to `maximum`.
    pub fn new(maximum: NonZeroUsize, minimum: usize) -> Self {
        let maximum = maximum.get();
        let minimum = minimum.clamp(1, maximum);
        Self {
            frames: BTreeMap::new(),
            maximum,
            minimum,
        }
    }

    /// How many frames are stored.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// No frames are stored.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The frame that covers `time`.
    ///
    /// `tolerance` extends the hit to a frame that starts after `time` by at
    /// most that much. A zero tolerance leaves the half-open range as the only
    /// hit. A miss does not drop stored frames.
    pub fn find(&self, time: Timestamp, tolerance: Timestamp) -> Option<&VideoFrame<I>> {
        if let Some((_, entry)) = self.frames.range(..=time).next_back() {
            if time >= entry.frame.pts && (time < entry.end || time == entry.frame.pts) {
                return Some(&entry.frame);
            }
        }
        if tolerance.value <= 0 {
            return None;
        }
        let limit = add_time(time, tolerance);
        let next = self
            .frames
            .range((Bound::Excluded(time), Bound::Unbounded))
            .next();
        if let Some((_, entry)) = next {
            if entry.frame.pts <= limit {
                return Some(&entry.frame);
            }
        }
        None
    }

    /// Drops frames behind `time` until [`Self::len`] is the floor.
    ///
    /// The frame that covers `time`, and every frame that starts at or after
    /// it, stay. The returned frames are the ones that were dropped.
    pub fn trim_play(&mut self, time: Timestamp) -> Vec<VideoFrame<I>> {
        let mut dropped = Vec::new();
        while self.frames.len() > self.minimum {
            let Some((start, entry)) = self.frames.iter().next() else {
                break;
            };
            if *start >= time || (entry.end > time && *start <= time) {
                break;
            }
            let start = *start;
            let Some((_, entry)) = self.frames.remove_entry(&start) else {
                break;
            };
            dropped.push(entry.frame);
        }
        dropped
    }

    /// Stores `frame`, clipping any range that overlaps it.
    ///
    /// During `play`, frames behind `around` are dropped to the floor first.
    /// The frame is stored only when the count is still under the maximum.
    /// During a seek, the end farther from `around` is dropped until the new
    /// frame fits.
    pub fn insert(&mut self, frame: VideoFrame<I>, around: Timestamp, play: bool) -> Remember<I> {
        let start = frame.pts;
        if self.frames.contains_key(&start) {
            self.place(frame);
            return Remember {
                evicted: Vec::new(),
                refused: None,
            };
        }
        let evicted = if play {
            self.trim_play(around)
        } else {
            self.evict_farthest(around)
        };
        if play && self.frames.len() >= self.maximum {
            return Remember {
                evicted,
                refused: Some(frame),
            };
        }
        self.place(frame);
        Remember {
            evicted,
            refused: None,
        }
    }

    /// Drops every frame.
    pub fn drain(&mut self) -> Vec<VideoFrame<I>> {
        std::mem::take(&mut self.frames)
            .into_values()
            .map(|entry| entry.frame)
            .collect()
    }

    /// The earliest and latest start, when the cache is not empty.
    pub fn ends(&self) -> (Option<Timestamp>, Option<Timestamp>) {
        (
            self.frames.keys().next().copied(),
            self.frames.keys().next_back().copied(),
        )
    }

    /// Removes the frame that starts at `start`.
    pub fn remove(&mut self, start: Timestamp) -> Option<VideoFrame<I>> {
        self.frames.remove(&start).map(|entry| entry.frame)
    }

    fn evict_farthest(&mut self, around: Timestamp) -> Vec<VideoFrame<I>> {
        let mut dropped = Vec::new();
        while self.frames.len() >= self.maximum {
            let (earliest, latest) = self.ends();
            let Some(earliest) = earliest else {
                break;
            };
            let latest = latest.unwrap_or(earliest);
            let start = farther(around, earliest, latest);
            let Some(entry) = self.frames.remove(&start) else {
                break;
            };
            dropped.push(entry.frame);
        }
        dropped
    }

    fn place(&mut self, frame: VideoFrame<I>) {
        let start = frame.pts;
        let (end, bounded) = range_end(&frame);
        self.frames.insert(
            start,
            Entry {
                end,
                bounded,
                frame,
            },
        );
        self.clip_previous(start);
        self.clip_next(start);
    }

    fn clip_previous(&mut self, start: Timestamp) {
        let Some(previous) = self.frames.range(..start).next_back().map(|(key, _)| *key) else {
            return;
        };
        let Some(entry) = self.frames.get_mut(&previous) else {
            return;
        };
        if !entry.bounded || entry.end > start {
            entry.end = start;
        }
    }

    fn clip_next(&mut self, start: Timestamp) {
        let next = self
            .frames
            .range((Bound::Excluded(start), Bound::Unbounded))
            .next()
            .map(|(key, _)| *key);
        let Some(entry) = self.frames.get_mut(&start) else {
            return;
        };
        if entry.bounded {
            if let Some(next) = next {
                if entry.end > next {
                    entry.end = next;
                }
            }
            return;
        }
        entry.end = next.unwrap_or_else(|| one_tick(start));
    }
}

impl<I> FramePool<I> {
    /// `maximum` is how many pictures the pool holds across every source.
    pub fn new(maximum: NonZeroUsize) -> Self {
        Self {
            maximum,
            clips: HashMap::new(),
            stored: 0,
        }
    }

    /// How many pictures are stored.
    pub fn len(&self) -> usize {
        self.stored
    }

    /// No pictures are stored.
    pub fn is_empty(&self) -> bool {
        self.stored == 0
    }

    /// The picture for `key` that covers `time`.
    pub fn find(&self, key: &str, time: Timestamp, tolerance: Timestamp) -> Option<&VideoFrame<I>> {
        self.clips.get(key)?.find(time, tolerance)
    }

    /// Stores `frame` under `key` when it is closer to `around` than the frame
    /// the pool would drop. A full pool leaves a farther frame unstored.
    pub fn put(&mut self, key: &str, frame: VideoFrame<I>, around: Timestamp) {
        if self.stored >= self.maximum.get() {
            if let Some((_, start)) = self.farthest(around) {
                if micros_between(around, frame.pts) >= micros_between(around, start) {
                    return;
                }
            }
        }
        let maximum = self.maximum;
        let cache = self
            .clips
            .entry(key.to_owned())
            .or_insert_with(|| FrameCache::new(maximum, 1));
        let _ = cache.insert(frame, around, false);
        self.recount();
        self.fit(around);
    }

    fn recount(&mut self) {
        self.stored = self.clips.values().map(FrameCache::len).sum();
    }

    fn fit(&mut self, around: Timestamp) {
        while self.stored > self.maximum.get() {
            let Some((key, start)) = self.farthest(around) else {
                break;
            };
            let removed = self
                .clips
                .get_mut(&key)
                .and_then(|cache| cache.remove(start));
            if removed.is_none() {
                break;
            }
            self.stored -= 1;
            if self.clips.get(&key).is_some_and(FrameCache::is_empty) {
                self.clips.remove(&key);
            }
        }
    }

    fn farthest(&self, around: Timestamp) -> Option<(String, Timestamp)> {
        let mut choice: Option<(String, Timestamp, u128)> = None;
        for (key, cache) in &self.clips {
            let (earliest, latest) = cache.ends();
            let Some(earliest) = earliest else {
                continue;
            };
            let latest = latest.unwrap_or(earliest);
            let start = farther(around, earliest, latest);
            let distance = micros_between(around, start);
            let replace = match &choice {
                None => true,
                Some((_, _, kept)) => distance > *kept,
            };
            if replace {
                choice = Some((key.clone(), start, distance));
            }
        }
        choice.map(|(key, start, _)| (key, start))
    }
}

fn range_end<I>(frame: &VideoFrame<I>) -> (Timestamp, bool) {
    match frame.duration {
        Some(duration) if duration.value > 0 => (add_time(frame.pts, duration), true),
        _ => (one_tick(frame.pts), false),
    }
}

fn one_tick(time: Timestamp) -> Timestamp {
    Timestamp::new(time.value.saturating_add(1), time.timescale)
}

fn add_time(start: Timestamp, span: Timestamp) -> Timestamp {
    let ticks = i128::from(span.value).saturating_mul(i128::from(start.timescale.get()))
        / i128::from(span.timescale.get());
    let sum = i128::from(start.value).saturating_add(ticks);
    let value = i64::try_from(sum).unwrap_or(if sum > 0 { i64::MAX } else { i64::MIN });
    Timestamp::new(value, start.timescale)
}

/// The end farther from `around`. An equal distance keeps the later end.
fn farther(around: Timestamp, earliest: Timestamp, latest: Timestamp) -> Timestamp {
    if micros_between(around, latest) > micros_between(around, earliest) {
        latest
    } else {
        earliest
    }
}

fn micros_between(left: Timestamp, right: Timestamp) -> u128 {
    let left = i128::from(left.value).saturating_mul(1_000_000) / i128::from(left.timescale.get());
    let right =
        i128::from(right.value).saturating_mul(1_000_000) / i128::from(right.timescale.get());
    left.abs_diff(right)
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};

    use cutlass_core::{Rotation, Timestamp, VideoFrame};

    use super::{FrameCache, FramePool};

    fn time(ms: i64) -> Timestamp {
        Timestamp::new(ms, NonZeroU32::new(1_000).expect("scale"))
    }

    fn frame(ms: i64, duration: i64) -> VideoFrame<i64> {
        VideoFrame {
            pts: time(ms),
            duration: Some(time(duration)),
            color: None,
            mastering: None,
            content_light: None,
            rotation: Rotation::Zero,
            image: ms,
        }
    }

    fn cache(maximum: usize, minimum: usize) -> FrameCache<i64> {
        FrameCache::new(NonZeroUsize::new(maximum).expect("maximum"), minimum)
    }

    #[test]
    fn a_time_inside_the_range_hits_and_a_seek_miss_keeps_the_frames() {
        let mut frames = cache(4, 2);
        let remembered = frames.insert(frame(0, 40), time(0), false);
        assert!(remembered.evicted.is_empty());
        assert!(frames.find(time(39), time(0)).is_some());
        assert!(frames.find(time(40), time(0)).is_none());
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn an_overlapping_frame_clips_the_earlier_end() {
        let mut frames = cache(4, 1);
        frames.insert(frame(0, 100), time(0), false);
        frames.insert(frame(40, 40), time(40), false);
        assert!(
            frames
                .find(time(39), time(0))
                .is_some_and(|found| found.image == 0)
        );
        assert!(
            frames
                .find(time(40), time(0))
                .is_some_and(|found| found.image == 40)
        );
    }

    #[test]
    fn a_frame_without_a_duration_stretches_to_the_next_one() {
        let mut frames = cache(4, 1);
        let mut open = frame(0, 0);
        open.duration = None;
        frames.insert(open, time(0), false);
        assert!(frames.find(time(10), time(0)).is_none());
        frames.insert(frame(40, 10), time(40), false);
        assert!(
            frames
                .find(time(39), time(0))
                .is_some_and(|found| found.image == 0)
        );
    }

    #[test]
    fn playback_drops_frames_behind_the_playhead_down_to_the_floor() {
        let mut frames = cache(8, 2);
        for ms in [0, 10, 20, 30] {
            frames.insert(frame(ms, 10), time(ms), false);
        }
        let dropped = frames.trim_play(time(30));
        assert_eq!(
            dropped.iter().map(|frame| frame.image).collect::<Vec<_>>(),
            vec![0, 10]
        );
        assert_eq!(frames.len(), 2);
        assert!(
            frames
                .find(time(30), time(0))
                .is_some_and(|found| found.image == 30)
        );
    }

    #[test]
    fn a_full_seek_drops_the_end_farther_from_the_request() {
        let mut frames = cache(3, 1);
        for ms in [0, 10, 20] {
            frames.insert(frame(ms, 10), time(0), false);
        }
        let remembered = frames.insert(frame(30, 10), time(30), false);
        assert_eq!(
            remembered
                .evicted
                .iter()
                .map(|frame| frame.image)
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert!(remembered.refused.is_none());
        assert!(frames.find(time(0), time(0)).is_none());
        assert!(frames.find(time(30), time(0)).is_some());
    }

    #[test]
    fn playback_refuses_a_frame_when_the_cache_stays_full() {
        let mut frames = cache(2, 2);
        frames.insert(frame(0, 10), time(0), true);
        frames.insert(frame(10, 10), time(0), true);
        let remembered = frames.insert(frame(20, 10), time(0), true);
        assert!(remembered.refused.is_some_and(|frame| frame.image == 20));
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn a_later_frame_within_tolerance_hits() {
        let frames_time = {
            let mut frames = cache(4, 1);
            frames.insert(frame(40, 10), time(40), false);
            frames
        };
        assert!(
            frames_time
                .find(time(30), time(10))
                .is_some_and(|found| found.image == 40)
        );
        assert!(frames_time.find(time(20), time(10)).is_none());
    }

    #[test]
    fn the_pool_drops_the_picture_farthest_from_the_new_one() {
        let mut pool = FramePool::new(NonZeroUsize::new(2).expect("maximum"));
        pool.put("a", frame(0, 10), time(0));
        pool.put("a", frame(100, 10), time(100));
        pool.put("a", frame(110, 10), time(110));
        assert_eq!(pool.len(), 2);
        assert!(pool.find("a", time(0), time(0)).is_none());
        assert!(pool.find("a", time(100), time(0)).is_some());
        assert!(pool.find("a", time(110), time(0)).is_some());
    }

    #[test]
    fn the_pool_keeps_a_near_picture_when_a_far_one_arrives() {
        let mut pool = FramePool::new(NonZeroUsize::new(2).expect("maximum"));
        pool.put("a", frame(100, 10), time(110));
        pool.put("a", frame(110, 10), time(110));
        pool.put("a", frame(0, 10), time(110));
        assert_eq!(pool.len(), 2);
        assert!(pool.find("a", time(0), time(0)).is_none());
        assert!(pool.find("a", time(100), time(0)).is_some());
        assert!(pool.find("a", time(110), time(0)).is_some());
    }
}
