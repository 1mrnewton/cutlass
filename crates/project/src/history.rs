//! Undo and redo for one editing session.
//!
//! A step stores the project on the other side of the edit. Undo and redo
//! swap that project back. They do not run the commands again. The id counter
//! stays at the higher of the two values.

use crate::Error;
use crate::command::{Command, Outcome};
use crate::project::Project;

/// The session's undo and redo stacks. They are not part of the saved project.
#[derive(Debug, Default)]
pub struct History {
    undo: Vec<Project>,
    redo: Vec<Project>,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies `commands` as one step.
    ///
    /// An empty slice changes nothing and is not recorded. If any command
    /// fails, the project stays as it was.
    pub fn apply(
        &mut self,
        project: &mut Project,
        commands: &[Command],
    ) -> Result<Vec<Outcome>, Error> {
        if commands.is_empty() {
            return Ok(Vec::new());
        }
        let mut next = project.clone();
        let mut outcomes = Vec::with_capacity(commands.len());
        for command in commands {
            outcomes.push(next.apply(command.clone())?);
        }
        let previous = std::mem::replace(project, next);
        self.undo.push(previous);
        self.redo.clear();
        Ok(outcomes)
    }

    pub fn undo(&mut self, project: &mut Project) -> Result<(), Error> {
        exchange(
            project,
            &mut self.undo,
            &mut self.redo,
            Error::NothingToUndo,
        )
    }

    pub fn redo(&mut self, project: &mut Project) -> Result<(), Error> {
        exchange(
            project,
            &mut self.redo,
            &mut self.undo,
            Error::NothingToRedo,
        )
    }
}

fn exchange(
    project: &mut Project,
    from: &mut Vec<Project>,
    to: &mut Vec<Project>,
    empty: Error,
) -> Result<(), Error> {
    let Some(mut stored) = from.pop() else {
        return Err(empty);
    };
    let high = project.next_id().max(stored.next_id());
    std::mem::swap(project, &mut stored);
    project.keep_at_least(high);
    stored.keep_at_least(high);
    to.push(stored);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::Asset;
    use crate::command::{Command, Outcome};
    use crate::id::ProjectId;
    use crate::segment::{Channels, Segment};
    use crate::settings::AspectRatio;
    use crate::time::{Speed, TimeRange};
    use cutlass_core::Timestamp;
    use std::num::NonZeroU32;

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "cut")
    }

    fn video(seconds: i64) -> Asset {
        Asset::video(
            "file:///v.mp4",
            Timestamp::new(seconds * 30_000, NonZeroU32::new(30_000).unwrap()),
            320,
            180,
            true,
        )
        .unwrap()
    }

    fn clip(project: &Project, asset: crate::AssetId, start: i64) -> Segment {
        let asset = project.asset(asset).unwrap();
        let media = asset.duration().unwrap();
        let source = TimeRange::new(
            Timestamp::new(0, media.timescale),
            Timestamp::new(30_000, media.timescale),
        )
        .unwrap();
        Segment::media(asset, source, start, Channels::Both, Speed::ONE).unwrap()
    }

    #[test]
    fn an_empty_step_is_not_recorded() {
        let mut history = History::new();
        let mut project = project();
        let before = project.next_id();
        assert!(history.apply(&mut project, &[]).unwrap().is_empty());
        assert_eq!(project.next_id(), before);
        assert_eq!(
            history.undo(&mut project).unwrap_err(),
            Error::NothingToUndo
        );
    }

    #[test]
    fn a_failed_step_changes_nothing() {
        let mut history = History::new();
        let mut project = project();
        let before = project.next_id();
        let error = history
            .apply(
                &mut project,
                &[
                    Command::AddAsset(video(10)),
                    Command::SetCanvas {
                        width: 0,
                        height: 10,
                        ratio: AspectRatio::Square,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, Error::ZeroCanvas);
        assert!(project.assets().is_empty());
        assert_eq!(project.next_id(), before);
        assert_eq!(
            history.undo(&mut project).unwrap_err(),
            Error::NothingToUndo
        );
    }

    #[test]
    fn one_step_undoes_every_command_in_it() {
        let mut history = History::new();
        let mut project = project();
        history
            .apply(
                &mut project,
                &[
                    Command::Rename("a".to_string()),
                    Command::Rename("b".to_string()),
                ],
            )
            .unwrap();
        assert_eq!(project.name(), "b");
        history.undo(&mut project).unwrap();
        assert_eq!(project.name(), "cut");
        history.redo(&mut project).unwrap();
        assert_eq!(project.name(), "b");
    }

    #[test]
    fn a_new_step_clears_redo() {
        let mut history = History::new();
        let mut project = project();
        history
            .apply(&mut project, &[Command::Rename("a".to_string())])
            .unwrap();
        history.undo(&mut project).unwrap();
        history
            .apply(&mut project, &[Command::Rename("b".to_string())])
            .unwrap();
        assert_eq!(
            history.redo(&mut project).unwrap_err(),
            Error::NothingToRedo
        );
        assert_eq!(project.name(), "b");
    }

    #[test]
    fn undo_does_not_reuse_an_id_and_redo_restores_the_same_one() {
        let mut history = History::new();
        let mut project = project();
        assert_eq!(project.next_id(), 3);
        let asset = match history
            .apply(&mut project, &[Command::AddAsset(video(10))])
            .unwrap()
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        assert_eq!(asset.get(), 3);
        assert_eq!(project.next_id(), 4);

        let segment = clip(&project, asset, 0);
        let main = project.timeline().main_track().id();
        let first = match history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .unwrap()
            .as_slice()
        {
            [Outcome::Segment(id)] => *id,
            other => panic!("expected a segment, got {other:?}"),
        };
        assert_eq!(first.get(), 4);
        assert_eq!(project.next_id(), 5);

        history.undo(&mut project).unwrap();
        assert!(project.timeline().segment(first).is_none());
        assert_eq!(project.next_id(), 5);
        history.redo(&mut project).unwrap();
        assert!(project.timeline().segment(first).is_some());
        assert_eq!(project.next_id(), 5);

        history.undo(&mut project).unwrap();
        let segment = clip(&project, asset, 0);
        let second = match history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .unwrap()
            .as_slice()
        {
            [Outcome::Segment(id)] => *id,
            other => panic!("expected a segment, got {other:?}"),
        };
        assert_eq!(second.get(), 5);
        assert_ne!(second, first);
        assert!(project.timeline().segment(first).is_none());
        assert_eq!(
            history.redo(&mut project).unwrap_err(),
            Error::NothingToRedo
        );
    }
}
