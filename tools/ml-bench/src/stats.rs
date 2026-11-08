//! Summaries of repeated timings.

use std::time::Duration;

/// Milliseconds over a run of timings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub runs: usize,
    pub min: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
}

/// `None` for no timings.
#[must_use]
pub fn summarize(timings: &[Duration]) -> Option<Summary> {
    let mut ms: Vec<f64> = timings.iter().map(|t| t.as_secs_f64() * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    let last = ms.len().checked_sub(1)?;
    let at = |part: f64| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let index = ((last as f64) * part).round() as usize;
        ms.get(index.min(last)).copied().unwrap_or_default()
    };
    Some(Summary {
        runs: ms.len(),
        min: at(0.0),
        median: at(0.5),
        p90: at(0.9),
        max: at(1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_timings_have_no_summary() {
        assert_eq!(summarize(&[]), None);
    }

    #[test]
    fn the_summary_ignores_order() {
        let timings: Vec<Duration> = [5, 1, 3, 2, 4]
            .iter()
            .map(|ms| Duration::from_millis(*ms))
            .collect();
        let s = summarize(&timings);
        assert_eq!(
            s,
            Some(Summary {
                runs: 5,
                min: 1.0,
                median: 3.0,
                p90: 5.0,
                max: 5.0
            })
        );
    }
}
