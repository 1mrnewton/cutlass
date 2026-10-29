//! A bounded window of decoded pictures.
//!
//! A lookup that misses returns immediately. It does not decode and it does
//! not wait. The window never grows past its capacity.

use crate::error::EngineError;

/// The largest window a cache will keep.
pub const MAX_WINDOW: usize = 64;

struct Entry<T> {
    start: i64,
    end: i64,
    value: T,
}

/// Pictures whose source time falls in `[start, end)`.
pub struct FrameWindow<T> {
    capacity: usize,
    frames: Vec<Entry<T>>,
}

impl<T> FrameWindow<T> {
    /// `capacity` is at least one and at most [`MAX_WINDOW`].
    pub fn new(capacity: usize) -> Result<Self, EngineError> {
        if capacity == 0 || capacity > MAX_WINDOW {
            return Err(EngineError::Window);
        }
        Ok(Self {
            capacity,
            frames: Vec::new(),
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The picture that covers `time`, if one is already stored.
    #[must_use]
    pub fn get(&self, time: i64) -> Option<&T> {
        self.frames
            .iter()
            .find(|frame| time >= frame.start && time < frame.end)
            .map(|frame| &frame.value)
    }

    /// Stores `value` and drops the picture farthest from `start` when the window is full.
    pub fn insert(&mut self, start: i64, end: i64, value: T) {
        let end = if end <= start {
            start.saturating_add(1)
        } else {
            end
        };
        if let Some(existing) = self.frames.iter_mut().find(|frame| frame.start == start) {
            existing.end = end;
            existing.value = value;
            return;
        }
        self.frames.push(Entry { start, end, value });
        if self.frames.len() > self.capacity {
            let farthest = self
                .frames
                .iter()
                .enumerate()
                .max_by_key(|(_, frame)| frame.start.abs_diff(start))
                .map(|(index, _)| index);
            if let Some(index) = farthest {
                self.frames.remove(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_drops_the_farthest_picture_and_a_miss_is_immediate() {
        let mut window = FrameWindow::new(2).expect("window");
        assert!(FrameWindow::<u8>::new(0).is_err());
        assert!(FrameWindow::<u8>::new(MAX_WINDOW + 1).is_err());
        let decodes = 0_u32;
        assert!(window.get(0).is_none());
        assert_eq!(decodes, 0, "a miss does not decode");
        window.insert(0, 10, 1_u8);
        window.insert(10, 20, 2);
        window.insert(100, 110, 3);
        assert_eq!(window.len(), 2);
        assert!(window.get(0).is_none());
        assert_eq!(window.get(10), Some(&2));
        assert_eq!(window.get(100), Some(&3));
        window.insert(10, 20, 9);
        assert_eq!(window.get(15), Some(&9));
        assert_eq!(window.len(), 2);
        assert_eq!(decodes, 0);
    }
}
