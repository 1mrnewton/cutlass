use cutlass_model::{Draft, MaterialId, WordColumns};

use super::EditError;

/// Join word text with single spaces. Empty pieces are skipped.
fn caption_text(words: &[String]) -> String {
    let mut text = String::new();
    for word in words {
        if word.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(word);
    }
    text
}

pub fn set_words(
    draft: &mut Draft,
    material_id: &MaterialId,
    words: WordColumns,
) -> Result<(), EditError> {
    if words.start_time.len() != words.end_time.len() || words.start_time.len() != words.text.len()
    {
        return Err(EditError::WordShape);
    }
    for (start, end) in words.start_time.iter().zip(&words.end_time) {
        if *start < 0 || *end < *start {
            return Err(EditError::WordShape);
        }
    }
    let content = caption_text(&words.text);
    let text = draft
        .materials
        .texts
        .iter_mut()
        .find(|text| &text.id == material_id)
        .ok_or(EditError::MissingMaterial)?;
    text.words = words.clone();
    text.current_words = words;
    text.content.clone_from(&content);
    text.base_content = content;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::MaterialText;

    use super::*;

    fn draft_with_text() -> (Draft, MaterialId) {
        let mut draft = Draft::empty("D");
        let id = MaterialId::new("caption");
        draft.materials.texts.push(MaterialText::new(id.clone()));
        (draft, id)
    }

    fn columns(start: i64, end: i64, text: &str) -> WordColumns {
        WordColumns {
            start_time: vec![start],
            end_time: vec![end],
            text: vec![text.to_owned()],
            rest: Default::default(),
        }
    }

    #[test]
    fn recognized_words_fill_both_clocks_and_the_string() {
        let (mut draft, id) = draft_with_text();
        set_words(&mut draft, &id, columns(0, 400, "hello")).unwrap();
        let text = &draft.materials.texts[0];
        assert_eq!(text.words.text, ["hello"]);
        assert_eq!(text.current_words.start_time, [0]);
        assert_eq!(text.current_words.end_time, [400]);
        assert_eq!(text.content, "hello");
        assert_eq!(text.base_content, "hello");
    }

    #[test]
    fn a_missing_text_material_is_refused() {
        let mut draft = Draft::empty("D");
        let error =
            set_words(&mut draft, &MaterialId::new("gone"), columns(0, 1, "x")).unwrap_err();
        assert_eq!(error, EditError::MissingMaterial);
        assert!(draft.materials.texts.is_empty());
    }

    #[test]
    fn a_word_that_ends_before_it_starts_is_refused() {
        let (mut draft, id) = draft_with_text();
        let error = set_words(&mut draft, &id, columns(20, 10, "x")).unwrap_err();
        assert_eq!(error, EditError::WordShape);
        assert!(draft.materials.texts[0].content.is_empty());
    }

    #[test]
    fn columns_of_different_lengths_are_refused() {
        let (mut draft, id) = draft_with_text();
        let words = WordColumns {
            start_time: vec![0, 10],
            end_time: vec![10],
            text: vec!["a".to_owned()],
            rest: Default::default(),
        };
        let error = set_words(&mut draft, &id, words).unwrap_err();
        assert_eq!(error, EditError::WordShape);
        assert!(draft.materials.texts[0].words.text.is_empty());
    }
}
