use crate::timestamp::{self, TimestampError};
use crate::words::RecognizedWord;

/// One collapsed non-blank token. `end` is inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub token: usize,
    pub start: usize,
    pub end: usize,
}

/// Greedy CTC. Repeats collapse. `blank` is dropped, and it splits repeats.
pub fn collapse(rows: &[Vec<f32>], width: usize, blank: usize) -> Result<Vec<Span>, SearchError> {
    if width == 0 || blank >= width {
        return Err(SearchError::Shape);
    }
    let mut spans = Vec::new();
    let mut open: Option<Span> = None;
    for (frame, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(SearchError::Shape);
        }
        let token = argmax(row)?;
        match open {
            Some(span) if span.token == token => {
                open = Some(Span { end: frame, ..span });
            }
            Some(span) => {
                if span.token != blank {
                    spans.push(span);
                }
                open = (token != blank).then_some(Span {
                    token,
                    start: frame,
                    end: frame,
                });
            }
            None if token != blank => {
                open = Some(Span {
                    token,
                    start: frame,
                    end: frame,
                });
            }
            None => {}
        }
    }
    if let Some(span) = open.filter(|span| span.token != blank) {
        spans.push(span);
    }
    Ok(spans)
}

/// Greedy CTC. Index 0 is blank and is not a word.
pub fn ctc(
    rows: &[Vec<f32>],
    vocab: &[String],
    hop_ms: i64,
) -> Result<Vec<RecognizedWord>, SearchError> {
    let spans = collapse(rows, vocab.len(), 0)?;
    let mut words = Vec::new();
    for span in spans {
        push_word(&mut words, vocab, span, hop_ms)?;
    }
    Ok(words)
}

fn push_word(
    words: &mut Vec<RecognizedWord>,
    vocab: &[String],
    span: Span,
    hop_ms: i64,
) -> Result<(), SearchError> {
    let Some(text) = vocab.get(span.token) else {
        return Err(SearchError::Shape);
    };
    if text.is_empty() {
        return Ok(());
    }
    let (start_ms, end_ms) = timestamp::span_ms(span.start, span.end, hop_ms)?;
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

    #[test]
    fn a_blank_at_the_end_of_the_row_is_dropped() {
        let rows = vec![vec![0.0, 3.0, 1.0], vec![0.0, 0.0, 4.0]];
        let spans = collapse(&rows, 3, 2).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].token, 1);
        assert_eq!((spans[0].start, spans[0].end), (0, 0));
    }
}
