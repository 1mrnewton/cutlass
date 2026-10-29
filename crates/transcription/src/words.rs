use cutlass_model::WordColumns;

/// One recognized word. Times are milliseconds on the media range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecognizedWord {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Words from one stretch of audio, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Utterance {
    pub words: Vec<RecognizedWord>,
}

/// A recognizer returned a clock the draft cannot store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordError {
    /// A start is negative, or an end is before its start.
    Clock,
}

/// Flatten utterances into the parallel arrays stored on a text material.
pub fn columns(utterances: &[Utterance]) -> Result<WordColumns, WordError> {
    let mut start_time = Vec::new();
    let mut end_time = Vec::new();
    let mut text = Vec::new();
    for utterance in utterances {
        for word in &utterance.words {
            if word.start_ms < 0 || word.end_ms < word.start_ms {
                return Err(WordError::Clock);
            }
            start_time.push(word.start_ms);
            end_time.push(word.end_ms);
            text.push(word.text.clone());
        }
    }
    Ok(WordColumns {
        start_time,
        end_time,
        text,
        rest: Default::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utterances_stay_in_order() {
        let utterances = [
            Utterance {
                words: vec![RecognizedWord {
                    text: "hi".to_owned(),
                    start_ms: 0,
                    end_ms: 200,
                }],
            },
            Utterance {
                words: vec![RecognizedWord {
                    text: "there".to_owned(),
                    start_ms: 200,
                    end_ms: 500,
                }],
            },
        ];
        let words = columns(&utterances).unwrap();
        assert_eq!(words.text, ["hi", "there"]);
        assert_eq!(words.start_time, [0, 200]);
        assert_eq!(words.end_time, [200, 500]);
    }

    #[test]
    fn a_backwards_word_is_refused() {
        let utterances = [Utterance {
            words: vec![RecognizedWord {
                text: "no".to_owned(),
                start_ms: 40,
                end_ms: 10,
            }],
        }];
        assert_eq!(columns(&utterances).unwrap_err(), WordError::Clock);
    }
}
