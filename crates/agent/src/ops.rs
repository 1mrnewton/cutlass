//! Every session request the model may send in `draft.apply`, each with
//! what it does and one example that parses.
//!
//! `ops.txt` holds one entry per request kind: a line `name: what it does`,
//! then one line of JSON, then a blank line. The tests check that every
//! kind `Request` takes has exactly one entry and that each example parses
//! as that kind, so the list cannot fall behind the session.

/// One request kind: its `op` name, what it does, and an example.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    /// The `op` value, in snake case.
    pub name: &'static str,
    /// What the request does, in one line.
    pub about: &'static str,
    /// One request of this kind as JSON.
    pub example: &'static str,
}

const OPS: &str = include_str!("ops.txt");

/// Every request kind, in the order `ops.txt` lists them.
#[must_use]
pub fn ops() -> Vec<Op> {
    OPS.split("\n\n")
        .filter_map(|entry| {
            let mut lines = entry.trim().lines();
            let head = lines.next()?;
            let (name, about) = head.split_once(": ")?;
            let example = lines.next()?.trim();
            Some(Op {
                name: name.trim(),
                about: about.trim(),
                example,
            })
        })
        .collect()
}

/// `draft.query ops`: every kind with what it does, one to a line.
/// `draft.query ops split add_text`: those kinds with an example each.
pub fn describe(names: &[&str]) -> Result<String, String> {
    let all = ops();
    if names.is_empty() {
        let mut out = String::from(
            "Request kinds for draft.apply. Ask draft.query \"ops <name> ...\" for the JSON of any of them.\n",
        );
        for op in &all {
            out.push_str(op.name);
            out.push_str(": ");
            out.push_str(op.about);
            out.push('\n');
        }
        return Ok(out);
    }
    let mut out = String::new();
    for name in names {
        let wanted = name.trim().to_ascii_lowercase().replace('-', "_");
        let Some(op) = all.iter().find(|op| op.name == wanted) else {
            return Err(format!(
                "no request kind {name}; draft.query \"ops\" lists them all"
            ));
        };
        out.push_str(op.name);
        out.push_str(": ");
        out.push_str(op.about);
        out.push('\n');
        out.push_str(op.example);
        out.push_str("\n\n");
    }
    Ok(out.trim_end().to_owned())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use cutlass_session::Request;

    use super::{describe, ops};

    /// The kinds `Request` takes, read from serde's own refusal of an
    /// unknown one.
    fn request_kinds() -> BTreeSet<String> {
        let error = serde_json::from_str::<Request>(r#"{"op":"no_such_kind"}"#)
            .unwrap_err()
            .to_string();
        let list = error
            .split("expected one of ")
            .nth(1)
            .unwrap_or_else(|| panic!("serde named no kinds: {error}"));
        list.split(',')
            .map(|kind| {
                kind.trim()
                    .trim_matches(|c: char| c == '`' || c == ' ')
                    .split('`')
                    .next()
                    .unwrap_or_default()
                    .to_owned()
            })
            .filter(|kind| !kind.is_empty())
            .collect()
    }

    #[test]
    fn every_request_kind_has_one_entry() {
        let kinds = request_kinds();
        assert!(kinds.len() > 100, "{kinds:?}");
        let listed: Vec<&str> = ops().iter().map(|op| op.name).collect();
        let unique: BTreeSet<&str> = listed.iter().copied().collect();
        assert_eq!(unique.len(), listed.len(), "an entry is listed twice");
        let missing: Vec<&String> = kinds
            .iter()
            .filter(|kind| !unique.contains(kind.as_str()))
            .collect();
        assert!(missing.is_empty(), "ops.txt has no entry for {missing:?}");
        let extra: Vec<&&str> = unique
            .iter()
            .filter(|name| !kinds.contains(**name))
            .collect();
        assert!(extra.is_empty(), "ops.txt lists unknown kinds {extra:?}");
    }

    #[test]
    fn every_example_parses_as_its_kind() {
        for op in ops() {
            assert!(!op.about.is_empty(), "{} says nothing", op.name);
            let value: serde_json::Value = serde_json::from_str(op.example)
                .unwrap_or_else(|error| panic!("{}: not JSON: {error}", op.name));
            assert_eq!(value["op"], op.name, "{}", op.example);
            serde_json::from_value::<Request>(value)
                .unwrap_or_else(|error| panic!("{}: {error}\n{}", op.name, op.example));
        }
    }

    #[test]
    fn describe_lists_all_or_shows_examples() {
        let all = describe(&[]).unwrap();
        assert!(all.contains("split: "));
        assert!(!all.contains("{\"op\""));
        let two = describe(&["split", "set-opacity"]).unwrap();
        assert!(two.contains("{\"op\":\"split\""), "{two}");
        assert!(two.contains("{\"op\":\"set_opacity\""), "{two}");
        assert!(
            describe(&["frobnicate"])
                .unwrap_err()
                .contains("frobnicate")
        );
    }
}
