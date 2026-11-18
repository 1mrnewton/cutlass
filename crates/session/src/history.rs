use std::sync::Arc;

use cutlass_model::Draft;

use crate::edit::EditError;

#[derive(Clone, Debug)]
struct OpenHistory {
    /// The draft at `begin` or `record`.
    draft: Draft,
    /// The redo stack at that moment, which `rollback` gives back.
    redo: Vec<Arc<Draft>>,
}

#[derive(Clone, Debug, PartialEq)]
struct RecordedStep {
    before: Arc<Draft>,
    after: Arc<Draft>,
}

/// Undo steps kept. Each step is a whole draft, so an uncapped stack grows
/// with project size times session length; the oldest step goes first.
pub const UNDO_LIMIT: usize = 100;

/// Bytes the undo stack may hold, measured as each step's draft written as
/// JSON. A big project reaches this before [`UNDO_LIMIT`]; the oldest steps
/// go first, and the newest step always stays.
pub const UNDO_BYTES: usize = 256 * 1024 * 1024;

/// One undo step and its size.
///
/// A step's draft is never changed once it is on the stack, so the copies
/// of a history share it: a host that edits a copy of the session and
/// hands it back, as the assistant's tools do on every call, copies the
/// open draft and not the hundred behind it.
#[derive(Clone, Debug)]
struct Step {
    draft: Arc<Draft>,
    bytes: usize,
}

/// The draft a step held, for the session to open: taken when this
/// history is the only one that holds it, copied when a copy of the
/// history holds it too.
fn opened(draft: Arc<Draft>) -> Draft {
    Arc::try_unwrap(draft).unwrap_or_else(|shared| (*shared).clone())
}

/// Undo and redo stacks, and one open transaction.
///
/// Outside a transaction each `apply` is one undo step, and the stack keeps
/// at most [`UNDO_LIMIT`] of them. `begin` or `record` opens a transaction:
/// the applies inside it push nothing, and `commit` turns the whole of it
/// into one undo step, or none when the draft ended where it began.
/// `amend` closes it into the undo step before it instead. `rollback` puts
/// the draft and the redo stack back where they were at `begin`.
#[derive(Clone, Debug)]
pub struct History {
    undo: Vec<Step>,
    /// The sum of the undo steps' sizes.
    undo_bytes: usize,
    /// The most the undo steps may add up to, [`UNDO_BYTES`] outside tests.
    byte_budget: usize,
    redo: Vec<Arc<Draft>>,
    open: Option<OpenHistory>,
    recording: bool,
    steps: Vec<RecordedStep>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            undo_bytes: 0,
            byte_budget: UNDO_BYTES,
            redo: Vec::new(),
            open: None,
            recording: false,
            steps: Vec::new(),
        }
    }
}

/// How many bytes `draft` takes written as JSON, counted without keeping
/// them. A draft that cannot be written counts as nothing.
pub(crate) fn json_bytes(draft: &Draft) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    match serde_json::to_writer(&mut count, draft) {
        Ok(()) => count.0,
        Err(_) => 0,
    }
}

impl History {
    /// Push one step. Past [`UNDO_LIMIT`] the oldest step is dropped.
    /// Inside a transaction nothing is pushed: `commit` makes the step.
    pub fn push_undo(&mut self, draft: Draft) {
        if self.open.is_some() {
            return;
        }
        self.push_step(Arc::new(draft));
    }

    fn push_step(&mut self, draft: Arc<Draft>) {
        let bytes = json_bytes(&draft);
        self.undo_bytes = self.undo_bytes.saturating_add(bytes);
        self.undo.push(Step { draft, bytes });
        while self.undo.len() > 1
            && (self.undo.len() > UNDO_LIMIT || self.undo_bytes > self.byte_budget)
        {
            let oldest = self.undo.remove(0);
            self.undo_bytes = self.undo_bytes.saturating_sub(oldest.bytes);
        }
    }

    /// Sets how many bytes the undo steps may add up to.
    #[cfg(test)]
    pub(crate) fn set_byte_budget(&mut self, bytes: usize) {
        self.byte_budget = bytes;
    }

    /// The undo steps' sizes added up.
    #[cfg(test)]
    pub(crate) fn undo_bytes(&self) -> usize {
        self.undo_bytes
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn pop_undo(&mut self) -> Option<Draft> {
        let step = self.undo.pop()?;
        self.undo_bytes = self.undo_bytes.saturating_sub(step.bytes);
        Some(opened(step.draft))
    }

    pub fn push_redo(&mut self, draft: Draft) {
        self.redo.push(Arc::new(draft));
    }

    pub fn pop_redo(&mut self) -> Option<Draft> {
        self.redo.pop().map(opened)
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
            // The step and the record of it hold the one draft.
            let before = Arc::new(open.draft);
            self.steps.push(RecordedStep {
                before: Arc::clone(&before),
                after: Arc::new(draft.clone()),
            });
            self.push_step(before);
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
            last.after = Arc::new(draft.clone());
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
        self.steps.last().map(|step| &*step.before)
    }

    #[must_use]
    pub fn recorded_after(&self) -> Option<&Draft> {
        self.steps.last().map(|step| &*step.after)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutlass_model::Draft;

    use super::History;

    fn named(name: &str) -> Draft {
        let mut draft = Draft::empty("draft");
        name.clone_into(&mut draft.name);
        draft
    }

    #[test]
    fn a_copy_of_a_history_shares_its_steps_and_neither_changes_the_others() {
        let mut history = History::default();
        for name in ["one", "two", "three"] {
            history.push_undo(named(name));
        }
        history.push_redo(named("ahead"));
        let mut copy = history.clone();
        // The copy holds the same drafts, not copies of them: copying a
        // session costs its open draft, however long its history.
        assert_eq!(copy.undo.len(), 3);
        for (ours, theirs) in history.undo.iter().zip(&copy.undo) {
            assert!(Arc::ptr_eq(&ours.draft, &theirs.draft));
        }
        assert!(Arc::ptr_eq(&history.redo[0], &copy.redo[0]));
        assert_eq!(copy.undo_bytes(), history.undo_bytes());
        // Each steps back and forward on its own, through the same drafts.
        assert_eq!(copy.pop_undo().unwrap().name, "three");
        assert_eq!(copy.pop_undo().unwrap().name, "two");
        assert_eq!(copy.pop_redo().unwrap().name, "ahead");
        assert_eq!(history.undo.len(), 3);
        assert_eq!(history.pop_undo().unwrap().name, "three");
        assert_eq!(history.pop_redo().unwrap().name, "ahead");
        assert_eq!(history.pop_undo().unwrap().name, "two");
        assert_eq!(history.pop_undo().unwrap().name, "one");
        assert!(history.pop_undo().is_none());
        assert_eq!(history.undo_bytes(), 0);
        // What the copy took stays as it was after the first is done.
        assert_eq!(copy.pop_undo().unwrap().name, "one");
        // A step made after the copy is the maker's alone.
        history.push_undo(named("later"));
        assert!(!copy.can_undo());
        // A transaction's step and the record of it are the one draft.
        let mut recorded = History::default();
        recorded.begin(&named("before")).unwrap();
        recorded.commit(&named("after")).unwrap();
        assert!(Arc::ptr_eq(
            &recorded.undo[0].draft,
            &recorded.steps[0].before
        ));
        assert_eq!(recorded.recorded_before().unwrap().name, "before");
        assert_eq!(recorded.recorded_after().unwrap().name, "after");
    }
}
