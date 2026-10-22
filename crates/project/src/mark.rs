//! Sorted times for a ruler marker, a clip marker, or a beat.
//!
//! A repeated time is the same mark. The lists stay in order.

/// Inserts `time` if it is not already there. Returns whether the list grew.
pub(crate) fn insert(times: &mut Vec<i64>, time: i64) -> bool {
    let index = times.partition_point(|stored| *stored < time);
    if times.get(index) == Some(&time) {
        return false;
    }
    times.insert(index, time);
    true
}

/// Removes `time` when it is there. Returns whether the list shrank.
pub(crate) fn remove(times: &mut Vec<i64>, time: i64) -> bool {
    let index = times.partition_point(|stored| *stored < time);
    if times.get(index) == Some(&time) {
        times.remove(index);
        return true;
    }
    false
}

/// Offsets at or after `at` move to the returned list, measured from `at`.
pub(crate) fn split(times: &mut Vec<i64>, at: i64) -> Vec<i64> {
    let index = times.partition_point(|offset| *offset < at);
    let mut right = Vec::new();
    for offset in times.split_off(index) {
        if let Some(next) = offset.checked_sub(at) {
            right.push(next);
        }
    }
    right
}

/// Keeps offsets that sit strictly inside `span`.
pub(crate) fn retain_inside(times: &mut Vec<i64>, span: i64) {
    times.retain(|offset| *offset >= 0 && *offset < span);
}

/// Keeps times from zero through `end`. An empty timeline keeps none.
pub(crate) fn retain_through(times: &mut Vec<i64>, end: i64) {
    if end <= 0 {
        times.clear();
        return;
    }
    times.retain(|time| *time >= 0 && *time <= end);
}

/// Clip markers and beats. Absent when both lists are empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StampList {
    pub(crate) markers: Vec<i64>,
    pub(crate) beats: Vec<i64>,
}

impl StampList {
    pub(crate) fn is_empty(&self) -> bool {
        self.markers.is_empty() && self.beats.is_empty()
    }
}
