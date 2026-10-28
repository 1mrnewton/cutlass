use cutlass_model::Draft;

#[derive(Clone, Debug, Default)]
pub struct History {
    undo: Vec<Draft>,
    redo: Vec<Draft>,
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
}
