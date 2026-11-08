use cutlass_model::Draft;

use crate::edit::EditError;

#[derive(Clone, Debug)]
struct OpenHistory {
    /// The draft at `begin` or `record`.
    draft: Draft,
    /// The redo stack at that moment, which `rollback` gives back.
    redo: Vec<Draft>,
}

#[derive(Clone, Debug, PartialEq)]
struct RecordedStep {
    before: Draft,
    after: Draft,
}

/// Undo steps kept. Each step is a whole draft, so an uncapped stack grows
/// with project size times session length; the oldest step goes first.
pub const UNDO_LIMIT: usize = 100;

/// Undo and redo stacks, and one open transaction.
///
/// Outside a transaction each `apply` is one undo step, and the stack keeps
/// at most [`UNDO_LIMIT`] of them. `begin` or `record` opens a transaction:
/// the applies inside it push nothing, and `commit` turns the whole of it
/// into one undo step, or none when the draft ended where it began.
/// `amend` closes it into the undo step before it instead. `rollback` puts
/// the draft and the redo stack back where they were at `begin`.
#[derive(Clone, Debug, Default)]
pub struct History {
    undo: Vec<Draft>,
    redo: Vec<Draft>,
    open: Option<OpenHistory>,
    recording: bool,
    steps: Vec<RecordedStep>,
}

impl History {
    /// Push one step. Past [`UNDO_LIMIT`] the oldest step is dropped.
    /// Inside a transaction nothing is pushed: `commit` makes the step.
    pub fn push_undo(&mut self, draft: Draft) {
        if self.open.is_some() {
            return;
        }
        self.push_step(draft);
    }

    fn push_step(&mut self, draft: Draft) {
        if self.undo.len() >= UNDO_LIMIT {
            self.undo.remove(0);
        }
        self.undo.push(draft);
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
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

    /// Whether a transaction is open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    pub fn begin(&mut self, draft: &Draft) -> Result<(), EditError> {
        if self.open.is_some() {
            return Err(EditError::HistoryOpen);
        }
        self.open = Some(OpenHistory {
            draft: draft.clone(),
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

    /// Close the transaction. A draft that changed becomes one undo step,
    /// and one recorded step.
    pub fn commit(&mut self, draft: &Draft) -> Result<(), EditError> {
        let Some(open) = self.open.take() else {
            return Err(EditError::HistoryClosed);
        };
        self.recording = false;
        if open.draft != *draft {
            self.steps.push(RecordedStep {
                before: open.draft.clone(),
                after: draft.clone(),
            });
            self.push_step(open.draft);
        }
        Ok(())
    }

    /// Close the open transaction into the undo step before it, so one undo
    /// takes back both. With no step before it, this is `commit`.
    pub fn amend(&mut self, draft: &Draft) -> Result<(), EditError> {
        if self.open.is_none() {
            return Err(EditError::HistoryClosed);
        }
        if self.undo.is_empty() {
            return self.commit(draft);
        }
        self.open = None;
        self.recording = false;
        if let Some(last) = self.steps.last_mut() {
            last.after = draft.clone();
        }
        Ok(())
    }

    /// Close the transaction and hand back the draft from `begin` or
    /// `record`. The redo stack is as it was then; the undo stack was not
    /// touched inside the transaction.
    pub fn rollback(&mut self) -> Result<Draft, EditError> {
        let Some(open) = self.open.take() else {
            return Err(EditError::HistoryClosed);
        };
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
