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

    use cutlass_session::{Draft, Request, Session};

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

    /// The draft every example is applied to, built by requests so it is a
    /// draft the session itself made. Its ids are the ones the examples
    /// use: picture clips `a` (0 to 5 s, material `m-a`, with sound) and
    /// `b` (5 to 8 s, `m-b`) touching on `main`; `m-c` in the library but
    /// not placed; the music clip `music-1` (`m-music`) on lane `music`;
    /// the title `title-1` on `text-1`; the caption `cap-1` of `a` on
    /// `captions`; the shape `shape-1`; the effect `fx-1`; the filter
    /// `filter-1`; and the adjustment layer `adj-1`.
    const FIXTURE: &str = r#"[
        {"op":"import","medium":{"video":{"id":"m-a","type":"video","path":"/media/a.mp4","width":1920,"height":1080,"duration":10000000,"has_audio":true}}},
        {"op":"import","medium":{"video":{"id":"m-b","type":"video","path":"/media/b.mp4","width":1920,"height":1080,"duration":10000000,"has_audio":true}}},
        {"op":"import","medium":{"video":{"id":"m-c","type":"video","path":"/media/c.mp4","width":1920,"height":1080,"duration":10000000,"has_audio":true}}},
        {"op":"import","medium":{"audio":{"id":"m-music","type":"music","path":"/media/song.m4a","duration":20000000}}},
        {"op":"place","track_id":"main","segment":{"id":"a","material_id":"m-a","target_timerange":{"start":0,"duration":5000000},"source_timerange":{"start":0,"duration":5000000}}},
        {"op":"place","track_id":"main","segment":{"id":"b","material_id":"m-b","target_timerange":{"start":5000000,"duration":3000000},"source_timerange":{"start":0,"duration":3000000}}},
        {"op":"insert_track","track_id":"music","kind":"audio"},
        {"op":"place","track_id":"music","segment":{"id":"music-1","material_id":"m-music","target_timerange":{"start":0,"duration":10000000},"source_timerange":{"start":0,"duration":10000000}}},
        {"op":"add_text","track_id":"text-1","segment_id":"title-1","material_id":"title-1-text","start":0,"duration":3000000,"content":"Summer trip","face":"Inter[opsz,wght].ttf"},
        {"op":"add_caption","source_id":"a","track_id":"captions","segment_id":"cap-1","material_id":"cap-1-text","words":{"start_time":[0,400],"end_time":[400,900],"text":["Hello","there"]},"span":{"start":0,"duration":900000},"face":"Inter[opsz,wght].ttf"},
        {"op":"add_shape","track_id":"shapes","segment_id":"shape-1","material_id":"shape-1-shape","start":0,"duration":3000000,"kind":"rectangle"},
        {"op":"add_effect","track_id":"fx","segment_id":"fx-1","material_id":"fx-1-effect","start":0,"duration":2000000,"effect_id":"recipe/shake"},
        {"op":"add_filter","track_id":"filters","segment_id":"filter-1","material_id":"filter-1-look","start":0,"duration":5000000,"filter_id":"look/warm"},
        {"op":"add_adjustment","track_id":"adjust-1","segment_id":"adj-1","material_id":"adj-1-adjust","start":0,"duration":5000000}
    ]"#;

    /// Examples that act on something another example makes: a removal
    /// needs the thing it removes. Each kind here runs the examples of
    /// the kinds listed after it on its copy of the fixture first, so
    /// the pairs in `ops.txt` are checked to fit together.
    const PRELUDES: &[(&str, &[&str])] = &[
        ("recover_audio", &["extract_audio"]),
        ("remove_transition", &["set_transition"]),
        ("untrack", &["track"]),
        ("unkey_values", &["key_values"]),
        ("remove_keyframe", &["add_keyframe"]),
        ("set_keyframe_graph", &["set_keyframes"]),
        ("ungroup", &["create_compound"]),
        ("set_inside", &["create_compound"]),
        ("remove_time_mark", &["add_time_mark"]),
        ("remove_beat", &["add_beat"]),
        ("arrange", &["set_free_layer"]),
    ];

    fn requests(json: &str) -> Vec<Request> {
        serde_json::from_str(json).unwrap_or_else(|error| panic!("{error}\n{json}"))
    }

    fn fixture() -> Session {
        let mut session = Session::open(Draft::empty("ops"));
        for (index, request) in requests(FIXTURE).iter().enumerate() {
            let revision = session.revision();
            session
                .apply(revision, std::slice::from_ref(request))
                .unwrap_or_else(|error| panic!("fixture request {index}: {error:?}"));
        }
        session
    }

    /// Every example is applied on its own copy of [`FIXTURE`], after its
    /// [`PRELUDES`], and must be taken and change the draft.
    #[test]
    fn every_example_applies_to_the_fixture() {
        let all = ops();
        let example = |name: &str| -> Request {
            let op = all
                .iter()
                .find(|op| op.name == name)
                .unwrap_or_else(|| panic!("no entry {name}"));
            serde_json::from_str(op.example).unwrap()
        };
        let base = fixture();
        let mut failed = Vec::new();
        for op in &all {
            let mut session = base.clone();
            let before: &[&str] = PRELUDES
                .iter()
                .find(|(name, _)| *name == op.name)
                .map_or(&[], |(_, before)| before);
            for name in before {
                let revision = session.revision();
                session
                    .apply(revision, &[example(name)])
                    .unwrap_or_else(|error| panic!("{} before {}: {error:?}", name, op.name));
            }
            let draft = session.draft().clone();
            let revision = session.revision();
            match session.apply(revision, &[example(op.name)]) {
                Err(error) => failed.push(format!("{}: {error:?}", op.name)),
                Ok(_) if session.draft() == &draft => {
                    failed.push(format!("{}: changed nothing", op.name));
                }
                Ok(_) => {}
            }
        }
        assert!(
            failed.is_empty(),
            "{} of {} examples did not apply:\n{}",
            failed.len(),
            all.len(),
            failed.join("\n")
        );
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
