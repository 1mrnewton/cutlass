//! Protocol upgrade.
//!
//! `new_version` is the protocol. The integer `version` is stored and is not
//! this number. Tasks run from the draft's protocol up to this build, in
//! version order. A version with no schema change has no task. A newer
//! protocol is refused. It is not rewritten into an older file.
//!
//! This build's protocol is 189.0.0. A later save at 192.0.0 is newer.
//! Nested drafts are upgraded here, not inside a task. No task uses the network.

use std::fmt;

use cutlass_model::Draft;

use crate::StoreError;

/// The protocol this build reads and writes.
pub const BUILD: Protocol = Protocol {
    major: 189,
    minor: 0,
    patch: 0,
};

/// One schema change. Versions with no change are absent from the list.
struct Task {
    version: Protocol,
    run: fn(&mut Draft),
}

/// No Cutlass schema change has been added. An older file opens as stored.
const TASKS: &[Task] = &[];

/// `major.minor.patch`, compared as those three numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Protocol {
    major: u32,
    minor: u32,
    patch: u32,
}

impl Protocol {
    fn parse(text: &str) -> Result<Self, ProtocolError> {
        if text.is_empty() {
            return Err(ProtocolError::Empty);
        }
        let parts: Vec<&str> = text.split('.').collect();
        let [major, minor, patch] = parts.as_slice() else {
            return Err(ProtocolError::Invalid);
        };
        Ok(Self {
            major: numeric(major)?,
            minor: numeric(minor)?,
            patch: numeric(patch)?,
        })
    }
}

impl fmt::Display for Protocol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug)]
enum ProtocolError {
    Empty,
    Invalid,
}

fn numeric(part: &str) -> Result<u32, ProtocolError> {
    if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ProtocolError::Invalid);
    }
    part.parse().map_err(|_| ProtocolError::Invalid)
}

/// Tasks whose version is above `current` and at most this build, oldest first.
fn pending(current: Protocol, tasks: &[Task]) -> Vec<&Task> {
    let mut chosen: Vec<&Task> = tasks
        .iter()
        .filter(|task| task.version > current && task.version <= BUILD)
        .collect();
    chosen.sort_by_key(|task| task.version);
    chosen
}

/// Return the draft this build can edit. The input draft is left as it was.
#[must_use = "a refused protocol is an error, not a draft"]
pub fn upgrade(draft: &Draft) -> Result<Draft, StoreError> {
    let mut next = draft.clone();
    upgrade_in_place(&mut next)?;
    Ok(next)
}

fn upgrade_in_place(draft: &mut Draft) -> Result<(), StoreError> {
    let protocol = Protocol::parse(&draft.new_version).map_err(|error| error.on(draft))?;
    if protocol > BUILD {
        return Err(StoreError::NewerProtocol {
            draft_id: draft.id.to_string(),
            found: draft.new_version.clone(),
        });
    }
    for task in pending(protocol, TASKS) {
        (task.run)(draft);
    }
    for material in &mut draft.materials.drafts {
        if let Some(child) = material.draft.as_mut() {
            upgrade_in_place(child)?;
        }
    }
    Ok(())
}

impl ProtocolError {
    fn on(self, draft: &Draft) -> StoreError {
        let draft_id = draft.id.to_string();
        match self {
            Self::Empty => StoreError::EmptyProtocol { draft_id },
            Self::Invalid => StoreError::InvalidProtocol {
                draft_id,
                found: draft.new_version.clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialDraft};

    use super::{BUILD, Protocol, TASKS, Task, pending};
    use crate::StoreError;
    use crate::upgrade::upgrade;

    #[test]
    fn this_build_speaks_189() {
        assert_eq!(BUILD.to_string(), "189.0.0");
        let draft = Draft::empty("D");
        assert_eq!(draft.new_version, "189.0.0");
        assert_eq!(draft.version, 360_000);
        let opened = upgrade(&draft).unwrap();
        assert_eq!(opened, draft);
    }

    #[test]
    fn a_newer_protocol_is_refused_and_the_draft_stays() {
        let mut draft = Draft::empty("D");
        draft.new_version = "192.0.0".to_owned();
        let before = draft.clone();
        let error = upgrade(&draft).unwrap_err();
        assert!(matches!(
            error,
            StoreError::NewerProtocol { ref found, ref draft_id }
                if found == "192.0.0" && draft_id == "D"
        ));
        assert_eq!(draft, before);
        draft.new_version = "189.0.1".to_owned();
        assert!(matches!(
            upgrade(&draft),
            Err(StoreError::NewerProtocol { .. })
        ));
    }

    #[test]
    fn an_older_protocol_is_not_rewritten() {
        let mut draft = Draft::empty("D");
        draft.new_version = "75.0.0".to_owned();
        draft.version = 360_000;
        let opened = upgrade(&draft).unwrap();
        assert_eq!(opened.new_version, "75.0.0");
        assert_eq!(opened.version, 360_000);
        assert_eq!(opened.tracks, draft.tracks);
        assert!(TASKS.is_empty());
    }

    #[test]
    fn an_empty_or_broken_protocol_is_refused() {
        let mut draft = Draft::empty("D");
        draft.new_version.clear();
        assert!(matches!(
            upgrade(&draft),
            Err(StoreError::EmptyProtocol { ref draft_id }) if draft_id == "D"
        ));
        draft.new_version = "189.0.0-beta".to_owned();
        assert!(matches!(
            upgrade(&draft),
            Err(StoreError::InvalidProtocol { ref found, .. }) if found == "189.0.0-beta"
        ));
        draft.new_version = "189.0".to_owned();
        assert!(matches!(
            upgrade(&draft),
            Err(StoreError::InvalidProtocol { .. })
        ));
    }

    #[test]
    fn a_newer_child_draft_refuses_the_parent() {
        let mut child = Draft::empty("child");
        child.new_version = "192.0.0".to_owned();
        let mut parent = Draft::empty("parent");
        parent.materials.drafts.push(MaterialDraft {
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let error = upgrade(&parent).unwrap_err();
        assert!(matches!(
            error,
            StoreError::NewerProtocol { ref draft_id, ref found }
                if draft_id == "child" && found == "192.0.0"
        ));
        assert_eq!(parent.new_version, "189.0.0");
    }

    #[test]
    fn an_older_child_draft_stays_at_its_protocol() {
        let mut child = Draft::empty("child");
        child.new_version = "116.0.1".to_owned();
        let mut parent = Draft::empty("parent");
        parent.materials.drafts.push(MaterialDraft {
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let opened = upgrade(&parent).unwrap();
        let nested = opened.materials.drafts[0].draft.as_ref().unwrap();
        assert_eq!(nested.new_version, "116.0.1");
        assert_eq!(opened.new_version, "189.0.0");
    }

    #[test]
    fn tasks_run_in_version_order_and_skip_the_current_and_the_future() {
        fn marker(_: &mut Draft) {}
        let tasks = [
            Task {
                version: Protocol::parse("189.0.0").unwrap(),
                run: marker,
            },
            Task {
                version: Protocol::parse("10.0.0").unwrap(),
                run: marker,
            },
            Task {
                version: Protocol::parse("200.0.0").unwrap(),
                run: marker,
            },
            Task {
                version: Protocol::parse("60.0.0").unwrap(),
                run: marker,
            },
        ];
        let chosen = pending(Protocol::parse("55.0.0").unwrap(), &tasks);
        assert_eq!(
            chosen.iter().map(|task| task.version).collect::<Vec<_>>(),
            vec![
                Protocol::parse("60.0.0").unwrap(),
                Protocol::parse("189.0.0").unwrap()
            ]
        );
    }
}
