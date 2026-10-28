use cutlass_model::Draft;

use crate::edit::EditError;

#[derive(Clone, Debug)]
struct OpenHistory {
    draft: Draft,
    undo: Vec<Draft>,
    redo: Vec<Draft>,
}

#[derive(Clone, Debug, PartialEq)]
struct RecordedStep {
    before: Draft,
    after: Draft,
}

/// Undo and redo stacks, plus a transaction those stacks do not open.
///
/// `apply` still pushes the previous draft on success. `begin`, `record`,
/// `commit`, `amend`, and `rollback` are the other calls. `rollback` puts
/// the open draft, and those stacks, back where they were at `begin`.
#[derive(Clone, Debug, Default)]
pub struct History {
    undo: Vec<Draft>,
    redo: Vec<Draft>,
    open: Option<OpenHistory>,
    recording: bool,
    steps: Vec<RecordedStep>,
}

impl History {
    pub fn push_undo(&mut self, draft: Draft) {
        self.undo.push(draft);
    }

    pub fn pop_undo(&mut self) -> Option<Draft> {
        self.undo.pop()
    }

    pub fn push_redo(&mut self, draft: Draft) {
        self.redo.push(draft);
    }

    pub fn pop_redo(&mut self) -> Option<Draft> {
        self.redo.pop()
    }

    pub fn clear_redo(&mut self) {
        self.redo.clear();
    }

    pub fn begin(&mut self, draft: &Draft) -> Result<(), EditError> {
        if self.open.is_some() {
            return Err(EditError::HistoryOpen);
        }
        self.open = Some(OpenHistory {
            draft: draft.clone(),
            undo: self.undo.clone(),
            redo: self.redo.clone(),
        });
        Ok(())
    }

    /// Start a recording. The first call opens the transaction `commit` closes.
    pub fn record(&mut self, draft: &Draft) -> Result<(), EditError> {
        if self.open.is_none() {
            self.begin(draft)?;
        }
        self.recording = true;
        Ok(())
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Close the transaction. A draft that changed becomes one recorded step.
    pub fn commit(&mut self, draft: &Draft) -> Result<(), EditError> {
        let Some(open) = self.open.take() else {
            return Err(EditError::HistoryClosed);
        };
        self.recording = false;
        if open.draft != *draft {
            self.steps.push(RecordedStep {
                before: open.draft,
                after: draft.clone(),
            });
        }
        Ok(())
    }

    /// Fold the open draft into the last recorded step and close the transaction.
    pub fn amend(&mut self, draft: &Draft) -> Result<(), EditError> {
        {
            let Some(last) = self.steps.last_mut() else {
                return Err(EditError::NothingToAmend);
            };
            last.after = draft.clone();
        }
        self.open = None;
        self.recording = false;
        Ok(())
    }

    /// Restore the draft and the undo stacks captured by `begin` or `record`.
    pub fn rollback(&mut self) -> Result<Draft, EditError> {
        let Some(open) = self.open.take() else {
            return Err(EditError::HistoryClosed);
        };
        self.undo = open.undo;
        self.redo = open.redo;
        self.recording = false;
        Ok(open.draft)
    }

    #[must_use]
    pub fn recorded_steps(&self) -> usize {
        self.steps.len()
    }

    #[must_use]
    pub fn recorded_before(&self) -> Option<&Draft> {
        self.steps.last().map(|step| &step.before)
    }

    #[must_use]
    pub fn recorded_after(&self) -> Option<&Draft> {
        self.steps.last().map(|step| &step.after)
    }
}
