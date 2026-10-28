use crate::timestamp::{self, TimestampError};
use crate::words::RecognizedWord;

/// Greedy CTC. Index 0 is blank and is not a word.
pub fn ctc(
    rows: &[Vec<f32>],
    vocab: &[String],
    hop_ms: i64,
) -> Result<Vec<RecognizedWord>, SearchError> {
    if vocab.is_empty() {
        return Err(SearchError::Shape);
    }
    let mut words = Vec::new();
    let mut open: Option<Run> = None;
    for (frame, row) in rows.iter().enumerate() {
        if row.len() != vocab.len() {
            return Err(SearchError::Shape);
        }
        let token = argmax(row)?;
        match open {
            Some(run) if run.token == token => {
                open = Some(Run { end: frame, ..run });
            }
            Some(run) => {
                push_word(&mut words, vocab, run, hop_ms)?;
                open = (token != 0).then_some(Run {
                    token,
                    start: frame,
                    end: frame,
                });
            }
            None if token != 0 => {
                open = Some(Run {
                    token,
                    start: frame,
                    end: frame,
                });
            }
            None => {}
        }
    }
    if let Some(run) = open {
        push_word(&mut words, vocab, run, hop_ms)?;
    }
    Ok(words)
}

struct Run {
    token: usize,
    start: usize,
    end: usize,
}

fn push_word(
    words: &mut Vec<RecognizedWord>,
    vocab: &[String],
    run: Run,
    hop_ms: i64,
) -> Result<(), SearchError> {
    if run.token == 0 {
        return Ok(());
    }
    let Some(text) = vocab.get(run.token) else {
        return Err(SearchError::Shape);
    };
    if text.is_empty() {
        return Ok(());
    }
    let (start_ms, end_ms) = timestamp::span_ms(run.start, run.end, hop_ms)?;
    words.push(RecognizedWord {
        text: text.clone(),
        start_ms,
        end_ms,
    });
    Ok(())
}

fn argmax(row: &[f32]) -> Result<usize, SearchError> {
    let mut best_index = 0;
    let mut best = *row.first().ok_or(SearchError::Shape)?;
    if !best.is_finite() {
        return Err(SearchError::Logits);
    }
    for (index, value) in row.iter().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(SearchError::Logits);
        }
        if *value > best {
            best = *value;
            best_index = index;
        }
    }
    Ok(best_index)
}

/// Why token search did not produce words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchError {
    /// A row does not match the vocabulary.
    Shape,
    /// A logit was NaN or infinite.
    Logits,
    /// A frame span does not fit in milliseconds.
    Time,
}

impl From<TimestampError> for SearchError {
    fn from(_error: TimestampError) -> Self {
        Self::Time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vec<String> {
        vec![String::new(), "on".to_owned(), "off".to_owned()]
    }

    #[test]
    fn repeats_collapse_and_blank_splits_them() {
        let rows = vec![
            vec![0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ];
        let words = ctc(&rows, &vocab(), 10).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "on");
        assert_eq!((words[0].start_ms, words[0].end_ms), (0, 20));
        assert_eq!(words[1].text, "off");
        assert_eq!((words[1].start_ms, words[1].end_ms), (30, 40));
    }

    #[test]
    fn a_non_finite_logit_is_refused() {
        let rows = vec![vec![0.0, f32::NAN, 0.0]];
        assert_eq!(ctc(&rows, &vocab(), 10).unwrap_err(), SearchError::Logits);
    }
}
