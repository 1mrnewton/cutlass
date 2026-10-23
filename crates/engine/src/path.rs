//! A source string the shell can open.

/// `file://` drops that prefix. `content:` is not a file.
pub fn file_path(source: &str) -> Option<&str> {
    if let Some(rest) = source.strip_prefix("file://") {
        let rest = rest.strip_prefix("localhost").unwrap_or(rest);
        return (!rest.is_empty()).then_some(rest);
    }
    if source.starts_with("content:") || source.is_empty() {
        return None;
    }
    Some(source)
}

#[cfg(test)]
mod tests {
    use super::file_path;

    #[test]
    fn a_file_url_is_a_path_and_a_content_url_is_not() {
        assert_eq!(file_path("file:///Users/a.mp4"), Some("/Users/a.mp4"));
        assert_eq!(
            file_path("file://localhost/Users/a.mp4"),
            Some("/Users/a.mp4")
        );
        assert_eq!(file_path("/Users/a.mp4"), Some("/Users/a.mp4"));
        assert_eq!(file_path("content://media/1"), None);
        assert_eq!(file_path(""), None);
        assert_eq!(file_path("file://"), None);
    }
}
