//! The order clips are drawn in, bottom first.
//!
//! A clip's `render_index` comes first: a higher layer is drawn over a
//! lower one whatever rows they sit on. Clips on the same layer are drawn
//! by row, and the track list is the row order with index 0 the top row,
//! so the upper row is drawn last and shows on top, as the timeline shows
//! it. Clips on one row are drawn in their order on it.

use std::cmp::Reverse;

/// A clip's place in the drawing order. Sorting keys ascending gives the
/// order to draw in, bottom first.
pub type PaintOrder = (i32, Reverse<usize>, usize);

/// The drawing order of the clip at `segment_index` on the track at
/// `track_index` in the draft's track list, on layer `render_index`.
#[must_use]
pub fn paint_order(render_index: i32, track_index: usize, segment_index: usize) -> PaintOrder {
    (render_index, Reverse(track_index), segment_index)
}

#[cfg(test)]
mod tests {
    use super::paint_order;

    #[test]
    fn on_one_layer_the_upper_row_is_drawn_over_the_lower() {
        // Row 0 is the top of the timeline, so it comes later in the order.
        assert!(paint_order(0, 0, 0) > paint_order(0, 1, 0));
        assert!(paint_order(14_000, 2, 0) > paint_order(14_000, 7, 0));
    }

    #[test]
    fn a_higher_layer_is_drawn_over_any_row() {
        assert!(paint_order(1, 9, 0) > paint_order(0, 0, 0));
        assert!(paint_order(14_000, 5, 0) > paint_order(11_000, 0, 3));
    }

    #[test]
    fn clips_on_one_row_keep_their_order_on_it() {
        assert!(paint_order(0, 3, 1) > paint_order(0, 3, 0));
    }

    #[test]
    fn sorting_gives_the_bottom_first() {
        let mut keys = vec![
            ("main", paint_order(0, 2, 0)),
            ("title", paint_order(14_000, 0, 0)),
            ("overlay", paint_order(0, 1, 0)),
            ("effect", paint_order(11_000, 0, 0)),
        ];
        keys.sort_by_key(|(_, key)| *key);
        let order: Vec<&str> = keys.iter().map(|(name, _)| *name).collect();
        assert_eq!(order, ["main", "overlay", "effect", "title"]);
    }
}
