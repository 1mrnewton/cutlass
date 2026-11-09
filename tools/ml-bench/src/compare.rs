//! How close two runs are: the raw numbers, and the masks a user would see.

/// The largest and mean absolute difference between two runs of values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drift {
    pub max: f32,
    pub mean: f32,
}

/// `None` when the lengths differ or there is nothing to compare.
#[must_use]
pub fn drift(a: &[f32], b: &[f32]) -> Option<Drift> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut max = 0.0_f32;
    let mut sum = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = max.max(d);
        sum += f64::from(d);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let mean = (sum / a.len() as f64) as f32;
    Some(Drift { max, mean })
}

/// Intersection over union of the two masks' insides (logit above zero).
/// Two empty masks agree fully. `None` when the lengths differ.
#[must_use]
pub fn mask_iou(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let (mut both, mut either) = (0_u64, 0_u64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x > 0.0, *y > 0.0);
        both += u64::from(x && y);
        either += u64::from(x || y);
    }
    if either == 0 {
        return Some(1.0);
    }
    #[allow(clippy::cast_precision_loss)]
    Some(both as f32 / either as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_runs_have_no_drift() {
        let d = drift(&[1.0, -2.0], &[1.0, -2.0]);
        assert_eq!(
            d,
            Some(Drift {
                max: 0.0,
                mean: 0.0
            })
        );
    }

    #[test]
    fn drift_reports_the_largest_and_mean_gap() {
        let d = drift(&[0.0, 0.0], &[1.0, -3.0]);
        assert_eq!(
            d,
            Some(Drift {
                max: 3.0,
                mean: 2.0
            })
        );
    }

    #[test]
    fn runs_of_different_length_do_not_compare() {
        assert_eq!(drift(&[0.0], &[0.0, 1.0]), None);
        assert_eq!(drift(&[], &[]), None);
        assert_eq!(mask_iou(&[0.0], &[0.0, 1.0]), None);
    }

    #[test]
    fn mask_iou_counts_only_the_inside() {
        // Inside in a: 0, 1. Inside in b: 1, 2. Overlap 1 of 3.
        let iou = mask_iou(&[1.0, 1.0, -1.0, -1.0], &[-1.0, 1.0, 1.0, -1.0]);
        assert_eq!(iou, Some(1.0 / 3.0));
    }

    #[test]
    fn two_empty_masks_agree() {
        assert_eq!(mask_iou(&[-1.0, -2.0], &[-5.0, 0.0]), Some(1.0));
    }
}
