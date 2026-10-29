//! Subtitle sidecars.
//!
//! SRT and plain text carry the words. Font, shadow, and motion stay out of
//! the file. ASS is refused.

use std::fs;
use std::path::{Path, PathBuf};

use cutlass_model::{Draft, MaterialRef, MaterialText, Segment, TimeRange};

use crate::config::{ExportError, SidecarKind};

struct Cue {
    start_ms: i64,
    end_ms: i64,
    text: String,
}

/// The sidecar body. ASS is an error. A truncated word list is an error.
pub fn sidecar_text(
    draft: &Draft,
    range: &TimeRange,
    kind: SidecarKind,
) -> Result<String, ExportError> {
    if kind == SidecarKind::Ass {
        return Err(ExportError::Ass);
    }
    if range.duration.0 <= 0 || range.start.0.checked_add(range.duration.0).is_none() {
        return Err(ExportError::Range);
    }
    let cues = collect_cues(draft, range)?;
    match kind {
        SidecarKind::Srt => render_srt(&cues),
        SidecarKind::Plain => Ok(render_plain(&cues)),
        SidecarKind::Ass => Err(ExportError::Ass),
    }
}

/// Writes `text` as `file_name` inside `dir`.
///
/// `file_name` is a single path component. A name that leaves `dir` is refused
/// before a file is created.
pub fn write_sidecar(dir: &Path, file_name: &str, text: &str) -> Result<PathBuf, ExportError> {
    if !single_component(file_name) || !dir.is_dir() {
        return Err(ExportError::Path);
    }
    let path = dir.join(file_name);
    let parent = path.parent().ok_or(ExportError::Path)?;
    let dir_canon = dir
        .canonicalize()
        .map_err(|error| ExportError::Io(error.to_string()))?;
    let parent_canon = parent
        .canonicalize()
        .map_err(|error| ExportError::Io(error.to_string()))?;
    if parent_canon != dir_canon {
        return Err(ExportError::Path);
    }
    fs::write(&path, text).map_err(|error| ExportError::Io(error.to_string()))?;
    Ok(path)
}

pub(crate) fn file_name(kind: SidecarKind, stem: &str) -> Result<String, ExportError> {
    if kind == SidecarKind::Ass {
        return Err(ExportError::Ass);
    }
    if !safe_stem(stem) {
        return Err(ExportError::Path);
    }
    let extension = match kind {
        SidecarKind::Srt => "srt",
        SidecarKind::Plain => "txt",
        SidecarKind::Ass => return Err(ExportError::Ass),
    };
    Ok(format!("{stem}.{extension}"))
}

fn collect_cues(draft: &Draft, range: &TimeRange) -> Result<Vec<Cue>, ExportError> {
    let mut cues = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            let Some(MaterialRef::Text(text)) = draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            cues.extend(cues_from_text(text, segment, range)?);
        }
    }
    cues.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then(left.end_ms.cmp(&right.end_ms))
    });
    Ok(cues)
}

fn cues_from_text(
    text: &MaterialText,
    segment: &Segment,
    range: &TimeRange,
) -> Result<Vec<Cue>, ExportError> {
    let words = &text.words;
    let empty = words.start_time.is_empty() && words.end_time.is_empty() && words.text.is_empty();
    if !empty {
        if words.start_time.len() != words.end_time.len()
            || words.start_time.len() != words.text.len()
        {
            return Err(ExportError::Truncated);
        }
        let mut cues = Vec::new();
        for index in 0..words.start_time.len() {
            let start_ms = words.start_time[index];
            let end_ms = words.end_time[index];
            if start_ms < 0 || end_ms <= start_ms {
                return Err(ExportError::Truncated);
            }
            if word_overlaps(start_ms, end_ms, range)? {
                cues.push(Cue {
                    start_ms,
                    end_ms,
                    text: words.text[index].clone(),
                });
            }
        }
        return Ok(cues);
    }
    content_cue(text, segment, range)
}

fn content_cue(
    text: &MaterialText,
    segment: &Segment,
    range: &TimeRange,
) -> Result<Vec<Cue>, ExportError> {
    if text.content.is_empty() {
        return Ok(Vec::new());
    }
    let start_us = segment.target.start.0;
    let end_us = segment.target.end().0;
    if start_us < 0 || end_us <= start_us {
        return Err(ExportError::Range);
    }
    let range_end = range
        .start
        .0
        .checked_add(range.duration.0)
        .ok_or(ExportError::Range)?;
    if start_us >= range_end || end_us <= range.start.0 {
        return Ok(Vec::new());
    }
    let start_ms = start_us / 1_000;
    let end_ms = end_us / 1_000;
    if end_ms <= start_ms {
        return Ok(Vec::new());
    }
    Ok(vec![Cue {
        start_ms,
        end_ms,
        text: text.content.clone(),
    }])
}

fn word_overlaps(start_ms: i64, end_ms: i64, range: &TimeRange) -> Result<bool, ExportError> {
    let start_us = start_ms.checked_mul(1_000).ok_or(ExportError::Range)?;
    let end_us = end_ms.checked_mul(1_000).ok_or(ExportError::Range)?;
    let range_end = range
        .start
        .0
        .checked_add(range.duration.0)
        .ok_or(ExportError::Range)?;
    Ok(start_us < range_end && end_us > range.start.0)
}

fn render_srt(cues: &[Cue]) -> Result<String, ExportError> {
    let mut out = String::new();
    for (index, cue) in cues.iter().enumerate() {
        let number = index + 1;
        out.push_str(&format!(
            "{number}\n{} --> {}\n{}\n\n",
            srt_stamp(cue.start_ms)?,
            srt_stamp(cue.end_ms)?,
            cue.text
        ));
    }
    Ok(out)
}

fn render_plain(cues: &[Cue]) -> String {
    let mut out = String::new();
    for (index, cue) in cues.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        out.push_str(&cue.text);
    }
    if !cues.is_empty() {
        out.push('\n');
    }
    out
}

fn srt_stamp(ms: i64) -> Result<String, ExportError> {
    if ms < 0 {
        return Err(ExportError::Range);
    }
    let hours = ms / 3_600_000;
    let minutes = (ms / 60_000) % 60;
    let seconds = (ms / 1_000) % 60;
    let millis = ms % 1_000;
    Ok(format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}"))
}

fn safe_stem(stem: &str) -> bool {
    !stem.is_empty()
        && stem.len() <= 128
        && stem.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '-' || character == '_'
        })
}

fn single_component(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
        && Path::new(name).components().count() == 1
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{file_name, sidecar_text, write_sidecar};
    use crate::config::{ExportError, SidecarKind};
    use cutlass_model::{Draft, MaterialText, Segment, TimeRange, WordColumns};

    fn draft_with_words() -> Draft {
        let mut draft = Draft::empty("draft");
        let mut text = MaterialText::new("words");
        text.font_name = "StyledFace".to_owned();
        text.shadow_color = "0xFF000000".to_owned();
        text.content = "styled motion".to_owned();
        text.words = WordColumns {
            start_time: vec![0, 1_000, 5_000],
            end_time: vec![500, 1_500, 6_000],
            text: vec!["Hello".to_owned(), "there".to_owned(), "later".to_owned()],
            ..WordColumns::default()
        };
        draft.materials.texts.push(text);
        draft.tracks[0].segments.push(Segment::new(
            "caption",
            "words",
            TimeRange::new(0, 2_000_000),
        ));
        draft
    }

    #[test]
    fn srt_and_plain_text_omit_style_and_cues_outside_the_range() {
        let draft = draft_with_words();
        let range = TimeRange::new(0, 2_000_000);
        let srt = sidecar_text(&draft, &range, SidecarKind::Srt).expect("srt");
        assert!(srt.contains("Hello"));
        assert!(srt.contains("there"));
        assert!(!srt.contains("later"));
        assert!(srt.contains("-->"));
        assert!(srt.contains("00:00:00,000 --> 00:00:00,500"));
        assert!(!srt.contains("StyledFace"));
        assert!(!srt.contains("styled motion"));
        assert!(!srt.contains("0xFF000000"));

        let plain = sidecar_text(&draft, &range, SidecarKind::Plain).expect("plain");
        assert_eq!(plain, "Hello\nthere\n");
        assert!(!plain.contains("-->"));
        assert!(!plain.contains("StyledFace"));
    }

    #[test]
    fn ass_and_truncated_words_are_refused() {
        let draft = draft_with_words();
        let range = TimeRange::new(0, 1_000);
        assert_eq!(
            sidecar_text(&draft, &range, SidecarKind::Ass),
            Err(ExportError::Ass)
        );
        assert_eq!(
            file_name(SidecarKind::Ass, "captions"),
            Err(ExportError::Ass)
        );

        let mut truncated = draft_with_words();
        truncated.materials.texts[0].words.end_time.pop();
        assert_eq!(
            sidecar_text(&truncated, &range, SidecarKind::Srt),
            Err(ExportError::Truncated)
        );
        assert_eq!(
            file_name(SidecarKind::Srt, "../secret"),
            Err(ExportError::Path)
        );
        assert_eq!(file_name(SidecarKind::Plain, ""), Err(ExportError::Path));
    }

    #[test]
    fn a_sidecar_is_written_inside_the_destination() {
        let dir =
            std::env::temp_dir().join(format!("cutlass-export-sidecar-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");
        let name = file_name(SidecarKind::Srt, "captions").expect("name");
        let path = write_sidecar(&dir, &name, "1\n").expect("write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "1\n");
        assert!(path.starts_with(&dir));
        assert_eq!(
            write_sidecar(&dir, "../secret.srt", "nope"),
            Err(ExportError::Path)
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
