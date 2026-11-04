//! The J, K and L shuttle and the I and O marks, the same on every screen.

use cutlass_session::Micros;

/// The fastest shuttle rate, either way.
pub const SHUTTLE_TOP: i32 = 8;

/// A shuttle key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shuttle {
    /// J: play backward, faster each press.
    Back,
    /// K: stop.
    Stop,
    /// L: play forward, faster each press.
    Forward,
}

/// The rate after `key` at `rate`, a whole multiple of real time with
/// backward below zero. A press the other way from a moving shuttle, or
/// from a stop, starts at one; a press the same way doubles, up to
/// [`SHUTTLE_TOP`].
#[must_use]
pub fn shuttle_rate(rate: i32, key: Shuttle) -> i32 {
    let way = match key {
        Shuttle::Stop => return 0,
        Shuttle::Back => -1,
        Shuttle::Forward => 1,
    };
    if rate.signum() == way {
        (rate * 2).clamp(-SHUTTLE_TOP, SHUTTLE_TOP)
    } else {
        way
    }
}

/// The in and out marks on a timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Marks {
    pub mark_in: Option<Micros>,
    pub mark_out: Option<Micros>,
}

impl Marks {
    /// Mark in at `time`. An out at or before it goes, as it would end the
    /// span before it starts.
    pub fn set_in(&mut self, time: Micros) {
        self.mark_in = Some(time);
        if self.mark_out.is_some_and(|out| out <= time) {
            self.mark_out = None;
        }
    }

    /// Mark out at `time`. An in at or after it goes.
    pub fn set_out(&mut self, time: Micros) {
        self.mark_out = Some(time);
        if self.mark_in.is_some_and(|mark| mark >= time) {
            self.mark_in = None;
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The marked span on a timeline `length` long: from the in, or the
    /// start, to the out, or the end. `None` with no marks, or when the
    /// marks leave nothing.
    #[must_use]
    pub fn span(&self, length: Micros) -> Option<(Micros, Micros)> {
        if self.mark_in.is_none() && self.mark_out.is_none() {
            return None;
        }
        let start = self
            .mark_in
            .unwrap_or(Micros(0))
            .0
            .clamp(0, length.0.max(0));
        let end = self.mark_out.unwrap_or(length).0.clamp(0, length.0.max(0));
        (end > start).then_some((Micros(start), Micros(end)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_press_the_same_way_doubles_and_the_other_way_starts_over() {
        let mut rate = 0;
        let mut seen = Vec::new();
        for key in [Shuttle::Forward; 5] {
            rate = shuttle_rate(rate, key);
            seen.push(rate);
        }
        assert_eq!(seen, [1, 2, 4, 8, 8]);
        assert_eq!(shuttle_rate(rate, Shuttle::Back), -1);
        assert_eq!(shuttle_rate(-1, Shuttle::Back), -2);
        assert_eq!(shuttle_rate(-4, Shuttle::Stop), 0);
        assert_eq!(shuttle_rate(0, Shuttle::Back), -1);
    }

    #[test]
    fn marks_keep_the_in_before_the_out_and_fill_a_missing_side() {
        let length = Micros(10);
        let mut marks = Marks::default();
        assert_eq!(marks.span(length), None);
        marks.set_in(Micros(3));
        assert_eq!(marks.span(length), Some((Micros(3), Micros(10))));
        marks.set_out(Micros(7));
        assert_eq!(marks.span(length), Some((Micros(3), Micros(7))));
        // An in past the out drops the out.
        marks.set_in(Micros(8));
        assert_eq!(marks.mark_out, None);
        marks.set_out(Micros(2));
        assert_eq!(marks.mark_in, None);
        assert_eq!(marks.span(length), Some((Micros(0), Micros(2))));
        // Marks past a shortened timeline clamp to it.
        marks.set_in(Micros(1));
        assert_eq!(marks.span(Micros(1)), None);
        marks.clear();
        assert_eq!(marks, Marks::default());
    }
}
