use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Caption words as three parallel arrays.
///
/// `start_time` and `end_time` are milliseconds, not microseconds.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct WordColumns {
    #[serde(default)]
    pub start_time: Vec<i64>,
    #[serde(default)]
    pub end_time: Vec<i64>,
    #[serde(default)]
    pub text: Vec<String>,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_columns_keep_unnamed_keys() {
        let words: WordColumns = serde_json::from_str(
            r#"{"start_time":[0],"end_time":[100],"text":["hi"],"tag":["kw"]}"#,
        )
        .unwrap();
        assert_eq!(words.start_time, [0]);
        assert_eq!(serde_json::to_value(words).unwrap()["tag"][0], "kw");
    }
}
