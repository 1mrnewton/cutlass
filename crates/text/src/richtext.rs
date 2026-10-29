//! The characters layout shapes.
//!
//! Rich-text commands are a later pass. A material marked as rich text is
//! refused here instead of being drawn as raw command text.

use cutlass_model::MaterialText;

use crate::TextError;

pub(crate) fn plain(material: &MaterialText) -> Result<&str, TextError> {
    if material.is_rich_text {
        return Err(TextError::RichTextCommands);
    }
    Ok(material.content.as_str())
}
