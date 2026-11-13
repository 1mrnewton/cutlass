//! The effect studio's document: one effect or one transition being made,
//! and every change the studio screen can make to it.
//!
//! The document is a recipe as the app ships them. Each change goes through
//! [`Studio`], which keeps the step to undo, so the screen only draws and
//! asks. Nothing here draws or touches the GPU.
//!
//! A saved document is a file in the studio folder, [`cutlass_effects`]
//! reads it there, and from then on the shelves, the inspector and the
//! timeline treat it as they treat a bundled recipe. A bundled recipe is
//! never changed: opening one opens a copy under a new id.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use cutlass_effects::{
    Block, CROSS, Control, ControlUnit, Crossing, Curve, Ease, Jitter, MAX_CONTROLS, MAX_KEYS,
    MAX_PASSES, Param, Pass, Recipe, Refused, Target, TransitionKind, TransitionRecipe, Wave,
    gap_ease,
};

/// How long an effect's preview plays before it starts again, in seconds,
/// until the designer picks another length. An effect dropped on the
/// timeline lasts this long too.
pub const DEFAULT_LOOP: f32 = 3.0;
/// The loop lengths the preview offers, in seconds.
pub const LOOPS: [f32; 5] = [1.0, 2.0, 3.0, 5.0, 10.0];
/// How many steps back the studio remembers.
const UNDO_LIMIT: usize = 200;

/// What is being made.
#[derive(Clone, Debug, PartialEq)]
pub enum Doc {
    Effect(Recipe),
    Transition(TransitionRecipe),
}

/// One of a document's lists of passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stack {
    /// An effect's passes.
    Effect,
    /// What happens to the picture a transition leaves.
    Outgoing,
    /// What happens to the picture a transition arrives at.
    Incoming,
}

/// One value of the document: a pass's parameter, or one of a transition's
/// `cross` parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Slot {
    Pass {
        stack: Stack,
        pass: usize,
        name: String,
    },
    Cross {
        name: String,
    },
}

impl Slot {
    #[must_use]
    pub fn pass(stack: Stack, pass: usize, name: &str) -> Self {
        Self::Pass {
            stack,
            pass,
            name: name.to_owned(),
        }
    }

    #[must_use]
    pub fn cross(name: &str) -> Self {
        Self::Cross {
            name: name.to_owned(),
        }
    }

    /// The parameter's name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Pass { name, .. } | Self::Cross { name } => name,
        }
    }
}

/// How a value moves over time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveKind {
    Fixed,
    Keys,
    Wave,
    Random,
}

impl CurveKind {
    pub const ALL: [Self; 4] = [Self::Fixed, Self::Keys, Self::Wave, Self::Random];

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "Fixed",
            Self::Keys => "Keys",
            Self::Wave => "Wave",
            Self::Random => "Random",
        }
    }

    #[must_use]
    pub fn of(curve: Option<&Curve>) -> Self {
        match curve {
            None | Some(Curve::Constant(_)) => Self::Fixed,
            Some(Curve::Keys { .. }) => Self::Keys,
            Some(Curve::Wave { .. }) => Self::Wave,
            Some(Curve::Jitter { .. }) => Self::Random,
        }
    }
}

/// How a control acts on a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    /// The control leaves the parameter alone.
    None,
    /// The control's value multiplies the parameter.
    Scale,
    /// The control's value is the parameter.
    Set,
}

/// One row of the studio's list of things to open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub category: String,
    /// Made in the studio folder, so it opens as itself. A bundled one
    /// opens as a copy.
    pub studio: bool,
}

impl Doc {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Effect(recipe) => &recipe.id,
            Self::Transition(recipe) => &recipe.id,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Effect(recipe) => &recipe.name,
            Self::Transition(recipe) => &recipe.name,
        }
    }

    #[must_use]
    pub fn category(&self) -> &str {
        match self {
            Self::Effect(recipe) => &recipe.category,
            Self::Transition(recipe) => &recipe.category,
        }
    }

    #[must_use]
    pub fn is_effect(&self) -> bool {
        matches!(self, Self::Effect(_))
    }

    /// The stacks this kind of document has, in the order the screen lists
    /// them.
    #[must_use]
    pub fn stacks(&self) -> &'static [Stack] {
        match self {
            Self::Effect(_) => &[Stack::Effect],
            Self::Transition(_) => &[Stack::Outgoing, Stack::Incoming],
        }
    }

    /// The passes of `stack`; none for a stack this document does not have.
    #[must_use]
    pub fn passes(&self, stack: Stack) -> &[Pass] {
        match (self, stack) {
            (Self::Effect(recipe), Stack::Effect) => &recipe.passes,
            (Self::Transition(recipe), Stack::Outgoing) => &recipe.outgoing,
            (Self::Transition(recipe), Stack::Incoming) => &recipe.incoming,
            _ => &[],
        }
    }

    fn passes_mut(&mut self, stack: Stack) -> Option<&mut Vec<Pass>> {
        match (self, stack) {
            (Self::Effect(recipe), Stack::Effect) => Some(&mut recipe.passes),
            (Self::Transition(recipe), Stack::Outgoing) => Some(&mut recipe.outgoing),
            (Self::Transition(recipe), Stack::Incoming) => Some(&mut recipe.incoming),
            _ => None,
        }
    }

    /// The parameter `slot` names: its default and its range.
    #[must_use]
    pub fn param(&self, slot: &Slot) -> Option<&'static Param> {
        match slot {
            Slot::Pass { stack, pass, name } => self
                .passes(*stack)
                .get(*pass)?
                .block
                .params()
                .iter()
                .find(|param| param.name == name),
            Slot::Cross { name } => match self {
                Self::Transition(_) => CROSS.iter().find(|param| param.name == name),
                Self::Effect(_) => None,
            },
        }
    }

    /// The curve written for `slot`; none when it is left at its default.
    #[must_use]
    pub fn curve(&self, slot: &Slot) -> Option<&Curve> {
        match (self, slot) {
            (_, Slot::Pass { stack, pass, name }) => {
                self.passes(*stack).get(*pass)?.params.get(name)
            }
            (Self::Transition(recipe), Slot::Cross { name }) => recipe.cross.get(name),
            (Self::Effect(_), Slot::Cross { .. }) => None,
        }
    }

    /// Write `curve` for `slot`, or with none leave it at its default.
    /// False when the document has no such parameter.
    fn set_curve(&mut self, slot: &Slot, curve: Option<Curve>) -> bool {
        if self.param(slot).is_none() {
            return false;
        }
        let params = match (&mut *self, slot) {
            (_, Slot::Pass { stack, pass, .. }) => self
                .passes_mut(*stack)
                .and_then(|passes| passes.get_mut(*pass))
                .map(|pass| &mut pass.params),
            (Self::Transition(recipe), Slot::Cross { .. }) => Some(&mut recipe.cross),
            (Self::Effect(_), Slot::Cross { .. }) => None,
        };
        let Some(params) = params else {
            return false;
        };
        match curve {
            Some(curve) => {
                params.insert(slot.name().to_owned(), curve);
            }
            None => {
                params.remove(slot.name());
            }
        }
        true
    }

    /// What `slot` is at `time`: seconds for an effect, progress from 0 to
    /// 1 for a transition. A transition's `progress` left out follows the
    /// transition itself.
    #[must_use]
    pub fn value_at(&self, slot: &Slot, time: f32) -> Option<f32> {
        let param = self.param(slot)?;
        Some(self.curve(slot).map_or_else(
            || {
                if matches!(slot, Slot::Cross { name } if name == "progress") {
                    time.clamp(param.min, param.max)
                } else {
                    param.default
                }
            },
            |curve| curve.at(time).clamp(param.min, param.max),
        ))
    }

    /// Why the document would not be saved, in the reader's own words.
    #[must_use]
    pub fn problem(&self) -> Option<String> {
        match self {
            Self::Effect(recipe) => recipe.to_json().err(),
            Self::Transition(recipe) => recipe.to_json().err(),
        }
        .map(|error| error.to_string())
    }
}

/// `text` as a recipe id: lowercase letters, digits and dashes.
#[must_use]
pub fn slug(text: &str) -> String {
    let mut id = String::new();
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            id.push(character.to_ascii_lowercase());
        } else if !id.is_empty() && !id.ends_with('-') {
            id.push('-');
        }
        if id.len() >= 48 {
            break;
        }
    }
    id.trim_end_matches('-').to_owned()
}

/// `base`, or `base-2`, `base-3` and on: the first that is not `taken`.
fn fresh(base: &str, taken: impl Fn(&str) -> bool) -> String {
    let base = if base.is_empty() { "untitled" } else { base };
    if !taken(base) {
        return base.to_owned();
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|id| !taken(id))
        .unwrap_or_else(|| base.to_owned())
}

fn effect_id_taken(id: &str) -> bool {
    cutlass_effects::recipe(&format!("{}{id}", cutlass_effects::RECIPE_PREFIX)).is_some()
}

fn transition_id_taken(id: &str) -> bool {
    TransitionKind::every()
        .into_iter()
        .any(|kind| kind.key() == id)
}

/// Keep `ease` one for each gap between `keys` keys, with the eases it
/// already means, or empty when every gap is a straight line.
fn tidy_ease(ease: &mut Vec<Ease>, keys: usize) {
    let gaps = keys.saturating_sub(1);
    let spelled: Vec<Ease> = (0..gaps).map(|gap| gap_ease(ease, gap)).collect();
    *ease = if spelled.iter().all(|ease| *ease == Ease::Linear) {
        Vec::new()
    } else {
        spelled
    };
}

/// Move a control's targets after pass `removed` of an effect has gone:
/// a target on it is dropped, and one on a later pass names it one earlier.
fn retarget_removed(controls: &mut [Control], removed: usize) {
    for control in controls {
        for targets in [&mut control.scale, &mut control.set] {
            targets.retain(|Target(pass, _)| *pass != removed);
            for Target(pass, _) in targets.iter_mut() {
                if *pass > removed {
                    *pass -= 1;
                }
            }
        }
    }
}

/// Move a control's targets after pass `from` of an effect moved to `to`.
fn retarget_moved(controls: &mut [Control], from: usize, to: usize) {
    let moved = |pass: usize| {
        if pass == from {
            to
        } else if from < to && pass > from && pass <= to {
            pass - 1
        } else if to < from && pass >= to && pass < from {
            pass + 1
        } else {
            pass
        }
    };
    for control in controls {
        for Target(pass, _) in control.scale.iter_mut().chain(&mut control.set) {
            *pass = moved(*pass);
        }
    }
}

/// The studio: its folder, the document open in it, and what is picked.
#[derive(Debug)]
pub struct Studio {
    /// Where recipes are saved; none when the studio folder is off, as in
    /// a test that keeps away from the user's own folder.
    folder: Option<PathBuf>,
    stamp: Option<u64>,
    refused: Vec<Refused>,
    doc: Option<Doc>,
    /// The document as its file holds it; none for one never saved.
    saved: Option<Doc>,
    /// Whether the id still follows the name: until the id is typed, or
    /// the document is saved.
    id_follows_name: bool,
    stack: Stack,
    pass: Option<usize>,
    slot: Option<Slot>,
    key: Option<usize>,
    /// Passes left out of the preview only.
    muted: BTreeSet<(Stack, usize)>,
    undo: Vec<Doc>,
    redo: Vec<Doc>,
    /// The stream of the edit in hand: more changes on the same stream are
    /// the same step to undo, until [`Studio::settle`].
    stream: Option<String>,
    loop_seconds: f32,
}

impl Studio {
    /// A studio that saves under `folder`. With none nothing is read or
    /// saved, and the document lives in memory only.
    #[must_use]
    pub fn new(folder: Option<PathBuf>) -> Self {
        Self {
            folder,
            stamp: None,
            refused: Vec::new(),
            doc: None,
            saved: None,
            id_follows_name: false,
            stack: Stack::Effect,
            pass: None,
            slot: None,
            key: None,
            muted: BTreeSet::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            stream: None,
            loop_seconds: DEFAULT_LOOP,
        }
    }

    #[must_use]
    pub fn folder(&self) -> Option<&Path> {
        self.folder.as_deref()
    }

    /// Read the studio folder again when a file in it was added, removed
    /// or rewritten since the last read. True when it was read. A document
    /// with no unsaved change follows its file; one with changes keeps them.
    pub fn reload(&mut self) -> bool {
        let Some(folder) = self.folder.clone() else {
            return false;
        };
        let stamp = cutlass_effects::studio_stamp(&folder);
        if self.stamp == Some(stamp) {
            return false;
        }
        self.stamp = Some(stamp);
        self.refused = cutlass_effects::load_studio(&folder).refused;
        if !self.is_dirty()
            && let Some(saved) = &self.saved
        {
            let on_disk = match saved {
                Doc::Effect(recipe) => cutlass_effects::studio_recipes()
                    .into_iter()
                    .find(|held| held.id == recipe.id)
                    .map(|held| Doc::Effect(held.clone())),
                Doc::Transition(recipe) => cutlass_effects::studio_transitions()
                    .into_iter()
                    .find(|held| held.id == recipe.id)
                    .map(|held| Doc::Transition(held.clone())),
            };
            if let Some(on_disk) = on_disk
                && Some(&on_disk) != self.saved.as_ref()
            {
                self.doc = Some(on_disk.clone());
                self.saved = Some(on_disk);
                self.forget_picks();
            }
        }
        true
    }

    /// The files the last read left out, and why.
    #[must_use]
    pub fn refused(&self) -> &[Refused] {
        &self.refused
    }

    /// Every effect that can be opened: the studio's first, then the
    /// bundled ones.
    #[must_use]
    pub fn effects() -> Vec<Entry> {
        let entry = |recipe: &Recipe, studio| Entry {
            id: recipe.id.clone(),
            name: recipe.name.clone(),
            category: recipe.category.clone(),
            studio,
        };
        cutlass_effects::studio_recipes()
            .into_iter()
            .map(|recipe| entry(recipe, true))
            .chain(
                cutlass_effects::bundled()
                    .iter()
                    .map(|recipe| entry(recipe, false)),
            )
            .collect()
    }

    /// Every transition recipe that can be opened: the studio's first,
    /// then the bundled ones. A built-in transition is not a recipe.
    #[must_use]
    pub fn transitions() -> Vec<Entry> {
        let entry = |recipe: &TransitionRecipe, studio| Entry {
            id: recipe.id.clone(),
            name: recipe.name.clone(),
            category: recipe.category.clone(),
            studio,
        };
        cutlass_effects::studio_transitions()
            .into_iter()
            .map(|recipe| entry(recipe, true))
            .chain(
                cutlass_effects::bundled_transitions()
                    .iter()
                    .map(|recipe| entry(recipe, false)),
            )
            .collect()
    }

    #[must_use]
    pub fn doc(&self) -> Option<&Doc> {
        self.doc.as_ref()
    }

    /// Whether the document differs from its file, or has none.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.doc.is_some() && self.doc != self.saved
    }

    /// Whether the document has a file in the studio folder.
    #[must_use]
    pub fn is_saved(&self) -> bool {
        self.saved.is_some()
    }

    /// Why the document cannot be saved now.
    #[must_use]
    pub fn problem(&self) -> Option<String> {
        let doc = self.doc.as_ref()?;
        if let Some(problem) = doc.problem() {
            return Some(problem);
        }
        // Its own file's id is its own; any other id in use is taken.
        let own = self
            .saved
            .as_ref()
            .is_some_and(|saved| saved.id() == doc.id());
        let taken = match doc {
            Doc::Effect(_) => effect_id_taken(doc.id()),
            Doc::Transition(_) => transition_id_taken(doc.id()),
        };
        (taken && !own).then(|| format!("the id {} is already used", doc.id()))
    }

    fn forget_picks(&mut self) {
        self.stack = match self.doc {
            Some(Doc::Transition(_)) => Stack::Outgoing,
            _ => Stack::Effect,
        };
        self.pass = self
            .doc
            .as_ref()
            .filter(|doc| !doc.passes(self.stack).is_empty())
            .map(|_| 0);
        self.slot = None;
        self.key = None;
        self.muted.clear();
        // The first value that moves, so the lane opens on a curve.
        let moving = self.doc.as_ref().and_then(|doc| {
            doc.stacks().iter().find_map(|stack| {
                doc.passes(*stack)
                    .iter()
                    .enumerate()
                    .find_map(|(index, pass)| {
                        let (name, _) = pass
                            .params
                            .iter()
                            .find(|(_, curve)| !matches!(curve, Curve::Constant(_)))?;
                        Some((*stack, index, name.clone()))
                    })
            })
        });
        if let Some((stack, pass, name)) = moving {
            self.stack = stack;
            self.pass = Some(pass);
            self.slot = Some(Slot::pass(stack, pass, &name));
        }
    }

    fn show(&mut self, doc: Doc, saved: Option<Doc>) {
        self.id_follows_name = saved.is_none();
        self.doc = Some(doc);
        self.saved = saved;
        self.undo.clear();
        self.redo.clear();
        self.stream = None;
        self.forget_picks();
    }

    /// Start a new effect: one blur pass, to have something to see.
    pub fn new_effect(&mut self) {
        let id = fresh("my-effect", effect_id_taken);
        self.show(
            Doc::Effect(Recipe {
                version: 1,
                id,
                name: "My effect".to_owned(),
                category: "Basic".to_owned(),
                preview: String::new(),
                poster: None,
                controls: Vec::new(),
                passes: vec![Pass {
                    block: Block::Blur,
                    params: [("radius".to_owned(), Curve::Constant(0.01))]
                        .into_iter()
                        .collect(),
                }],
            }),
            None,
        );
    }

    /// Start a new transition: a soft wipe.
    pub fn new_transition(&mut self) {
        let id = fresh("my-transition", transition_id_taken);
        self.show(
            Doc::Transition(TransitionRecipe {
                version: 1,
                id,
                name: "My transition".to_owned(),
                category: "Basic".to_owned(),
                preview: Vec::new(),
                duration: None,
                outgoing: Vec::new(),
                incoming: Vec::new(),
                cross: [("kind".to_owned(), Curve::Constant(0.0))]
                    .into_iter()
                    .collect(),
            }),
            None,
        );
    }

    /// Open the effect `id`: a studio one as itself, a bundled one as a
    /// copy under a new id. False when there is no such effect.
    pub fn open_effect(&mut self, id: &str) -> bool {
        let effect_id = format!("{}{id}", cutlass_effects::RECIPE_PREFIX);
        let Some(recipe) = cutlass_effects::recipe(&effect_id) else {
            return false;
        };
        if cutlass_effects::is_studio_recipe(&effect_id) {
            let doc = Doc::Effect(recipe.clone());
            self.show(doc.clone(), Some(doc));
        } else {
            let mut copy = recipe.clone();
            copy.id = fresh(&format!("{id}-copy"), effect_id_taken);
            copy.name = format!("{} copy", recipe.name);
            self.show(Doc::Effect(copy), None);
        }
        true
    }

    /// Open the transition recipe `id`, as [`Self::open_effect`] opens an
    /// effect. A built-in transition is not a recipe and does not open.
    pub fn open_transition(&mut self, id: &str) -> bool {
        let Some(recipe) = TransitionKind::every()
            .into_iter()
            .filter_map(TransitionKind::recipe)
            .find(|recipe| recipe.id == id)
        else {
            return false;
        };
        if cutlass_effects::is_studio_transition(&recipe.effect_id()) {
            let doc = Doc::Transition(recipe.clone());
            self.show(doc.clone(), Some(doc));
        } else {
            let mut copy = recipe.clone();
            copy.id = fresh(&format!("{id}-copy"), transition_id_taken);
            copy.name = format!("{} copy", recipe.name);
            self.show(Doc::Transition(copy), None);
        }
        true
    }

    /// Carry on with the open document as a copy under a new id, with no
    /// file of its own yet.
    pub fn duplicate(&mut self) {
        let Some(mut doc) = self.doc.clone() else {
            return;
        };
        match &mut doc {
            Doc::Effect(recipe) => {
                recipe.id = fresh(&format!("{}-copy", recipe.id), effect_id_taken);
                recipe.name = format!("{} copy", recipe.name);
            }
            Doc::Transition(recipe) => {
                recipe.id = fresh(&format!("{}-copy", recipe.id), transition_id_taken);
                recipe.name = format!("{} copy", recipe.name);
            }
        }
        let (stack, pass) = (self.stack, self.pass);
        self.show(doc, None);
        (self.stack, self.pass) = (stack, pass);
    }

    /// Put the document away. Unsaved changes go with it.
    pub fn close(&mut self) {
        self.doc = None;
        self.saved = None;
        self.undo.clear();
        self.redo.clear();
        self.stream = None;
        self.forget_picks();
    }

    /// Save the document to the studio folder and read the folder again,
    /// so the app runs what was saved. A document saved before under
    /// another id leaves that file behind it no longer.
    pub fn save(&mut self) -> Result<(), String> {
        let Some(doc) = self.doc.clone() else {
            return Err("nothing is open".to_owned());
        };
        if let Some(problem) = self.problem() {
            return Err(problem);
        }
        let Some(folder) = self.folder.clone() else {
            return Err("the studio folder is off".to_owned());
        };
        match &doc {
            Doc::Effect(recipe) => cutlass_effects::write_studio_recipe(&folder, recipe),
            Doc::Transition(recipe) => cutlass_effects::write_studio_transition(&folder, recipe),
        }
        .map_err(|error| error.to_string())?;
        if let Some(saved) = self.saved.as_ref().filter(|saved| saved.id() != doc.id()) {
            let removed = match saved {
                Doc::Effect(_) => cutlass_effects::remove_studio_recipe(&folder, saved.id()),
                Doc::Transition(_) => {
                    cutlass_effects::remove_studio_transition(&folder, saved.id())
                }
            };
            removed.map_err(|error| error.to_string())?;
        }
        // The file holds the lowest version that can hold the recipe.
        let stored = match doc {
            Doc::Effect(mut recipe) => {
                recipe.version = recipe.least_version();
                Doc::Effect(recipe)
            }
            Doc::Transition(mut recipe) => {
                recipe.version = recipe.least_version();
                Doc::Transition(recipe)
            }
        };
        self.doc = Some(stored.clone());
        self.saved = Some(stored);
        self.id_follows_name = false;
        self.stamp = None;
        self.reload();
        Ok(())
    }

    /// Delete the document's file and put the document away.
    pub fn delete(&mut self) -> Result<(), String> {
        if let (Some(saved), Some(folder)) = (self.saved.as_ref(), self.folder.clone()) {
            match saved {
                Doc::Effect(_) => cutlass_effects::remove_studio_recipe(&folder, saved.id()),
                Doc::Transition(_) => {
                    cutlass_effects::remove_studio_transition(&folder, saved.id())
                }
            }
            .map_err(|error| error.to_string())?;
        }
        self.close();
        self.stamp = None;
        self.reload();
        Ok(())
    }

    /// Change the document, as one step to undo. Changes on the same
    /// `stream` join that step until [`Self::settle`]: a slider dragged
    /// across its rail is one step, not one for each value it passed.
    fn edit(&mut self, stream: &str, change: impl FnOnce(&mut Doc)) {
        let Some(doc) = self.doc.as_mut() else {
            return;
        };
        let before = doc.clone();
        change(doc);
        if *doc == before {
            return;
        }
        if self.stream.as_deref() != Some(stream) {
            self.undo.push(before);
            if self.undo.len() > UNDO_LIMIT {
                self.undo.remove(0);
            }
            self.stream = Some(stream.to_owned());
        }
        self.redo.clear();
    }

    /// End the edit in hand: the next change is a new step to undo. The
    /// screen calls this once the pointer is up and no field has focus.
    pub fn settle(&mut self) {
        self.stream = None;
    }

    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self) {
        if let (Some(before), Some(doc)) = (self.undo.pop(), self.doc.as_mut()) {
            self.redo.push(std::mem::replace(doc, before));
            self.stream = None;
            self.keep_picks();
        }
    }

    pub fn redo(&mut self) {
        if let (Some(after), Some(doc)) = (self.redo.pop(), self.doc.as_mut()) {
            self.undo.push(std::mem::replace(doc, after));
            self.stream = None;
            self.keep_picks();
        }
    }

    /// Drop a pick that names something the document no longer has.
    fn keep_picks(&mut self) {
        let Some(doc) = self.doc.as_ref() else {
            return;
        };
        let passes = doc.passes(self.stack).len();
        if self.pass.is_some_and(|pass| pass >= passes) {
            self.pass = passes.checked_sub(1);
        }
        if self
            .slot
            .as_ref()
            .is_some_and(|slot| doc.param(slot).is_none())
        {
            self.slot = None;
        }
        let keys =
            self.slot
                .as_ref()
                .and_then(|slot| doc.curve(slot))
                .map_or(0, |curve| match curve {
                    Curve::Keys { keys, .. } => keys.len(),
                    _ => 0,
                });
        if self.key.is_some_and(|key| key >= keys) {
            self.key = None;
        }
        self.muted
            .retain(|(stack, pass)| *pass < doc.passes(*stack).len());
    }

    // What is picked.

    #[must_use]
    pub fn stack(&self) -> Stack {
        self.stack
    }

    /// The pass whose values the screen shows.
    #[must_use]
    pub fn picked_pass(&self) -> Option<(Stack, usize)> {
        self.pass.map(|pass| (self.stack, pass))
    }

    pub fn pick_pass(&mut self, stack: Stack, pass: usize) {
        let has = self
            .doc
            .as_ref()
            .is_some_and(|doc| pass < doc.passes(stack).len());
        if has {
            let moved = (self.stack, self.pass) != (stack, Some(pass));
            self.stack = stack;
            self.pass = Some(pass);
            if moved && matches!(self.slot, Some(Slot::Pass { .. })) {
                self.slot = None;
                self.key = None;
            }
        }
    }

    /// The value whose curve the lane shows.
    #[must_use]
    pub fn picked_slot(&self) -> Option<&Slot> {
        self.slot.as_ref()
    }

    pub fn pick_slot(&mut self, slot: Option<Slot>) {
        let known = slot.as_ref().is_none_or(|slot| {
            self.doc
                .as_ref()
                .is_some_and(|doc| doc.param(slot).is_some())
        });
        if known && self.slot != slot {
            self.slot = slot;
            self.key = None;
        }
    }

    #[must_use]
    pub fn picked_key(&self) -> Option<usize> {
        self.key
    }

    pub fn pick_key(&mut self, key: Option<usize>) {
        self.key = key;
        self.keep_picks();
    }

    #[must_use]
    pub fn is_muted(&self, stack: Stack, pass: usize) -> bool {
        self.muted.contains(&(stack, pass))
    }

    /// Leave a pass out of the preview, or put it back. The document is
    /// not changed: a saved recipe has every pass.
    pub fn set_muted(&mut self, stack: Stack, pass: usize, muted: bool) {
        if muted {
            self.muted.insert((stack, pass));
        } else {
            self.muted.remove(&(stack, pass));
        }
        self.keep_picks();
    }

    #[must_use]
    pub fn loop_seconds(&self) -> f32 {
        self.loop_seconds
    }

    pub fn set_loop_seconds(&mut self, seconds: f32) {
        if seconds.is_finite() {
            self.loop_seconds = seconds.clamp(0.25, 60.0);
        }
    }

    /// How long the lane's time axis is: an effect's loop in seconds, or a
    /// transition's whole progress, 1.
    #[must_use]
    pub fn span(&self) -> f32 {
        match self.doc {
            Some(Doc::Transition(_)) => 1.0,
            _ => self.loop_seconds,
        }
    }

    // The document's own fields.

    /// Name the document. Until its id is typed or it is saved, the id
    /// follows the name.
    pub fn set_name(&mut self, name: &str) {
        let follows = self.id_follows_name;
        let own = self.doc.as_ref().map(|doc| doc.id().to_owned());
        self.edit("name", |doc| {
            // Its own id is free for it to keep.
            let other = |id: &str| Some(id) != own.as_deref();
            match doc {
                Doc::Effect(recipe) => {
                    name.clone_into(&mut recipe.name);
                    if follows {
                        recipe.id = fresh(&slug(name), |id| other(id) && effect_id_taken(id));
                    }
                }
                Doc::Transition(recipe) => {
                    name.clone_into(&mut recipe.name);
                    if follows {
                        recipe.id = fresh(&slug(name), |id| other(id) && transition_id_taken(id));
                    }
                }
            }
        });
    }

    /// Type the document's id. It stops following the name.
    pub fn set_id(&mut self, id: &str) {
        self.id_follows_name = false;
        self.edit("id", |doc| match doc {
            Doc::Effect(recipe) => id.clone_into(&mut recipe.id),
            Doc::Transition(recipe) => id.clone_into(&mut recipe.id),
        });
    }

    pub fn set_category(&mut self, category: &str) {
        self.edit("category", |doc| match doc {
            Doc::Effect(recipe) => category.clone_into(&mut recipe.category),
            Doc::Transition(recipe) => category.clone_into(&mut recipe.category),
        });
    }

    /// The scene an effect's card shows it over.
    pub fn set_preview_scene(&mut self, scene: &str) {
        self.edit("preview", |doc| {
            if let Doc::Effect(recipe) = doc {
                recipe.preview = if scene == cutlass_effects::DEFAULT_PREVIEW_SCENE {
                    String::new()
                } else {
                    scene.to_owned()
                };
            }
        });
    }

    /// The two scenes a transition's card plays between.
    pub fn set_preview_pair(&mut self, from: &str, to: &str) {
        self.edit("preview", |doc| {
            if let Doc::Transition(recipe) = doc {
                recipe.preview = vec![from.to_owned(), to.to_owned()];
            }
        });
    }

    /// The second of an effect its card's still shows; none is the app's
    /// own choice.
    pub fn set_poster(&mut self, poster: Option<f32>) {
        self.edit("poster", |doc| {
            if let Doc::Effect(recipe) = doc {
                recipe.poster = poster.filter(|poster| poster.is_finite());
            }
        });
    }

    /// The seconds a transition is dropped with; none is the app's own.
    pub fn set_duration(&mut self, seconds: Option<f32>) {
        self.edit("duration", |doc| {
            if let Doc::Transition(recipe) = doc {
                recipe.duration = seconds
                    .filter(|seconds| seconds.is_finite())
                    .map(|seconds| seconds.clamp(0.1, 5.0));
            }
        });
    }

    // Passes.

    /// Add a pass of `block` at the end of `stack` and pick it. None when
    /// the stack is full.
    pub fn add_pass(&mut self, stack: Stack, block: Block) -> Option<usize> {
        let count = self.doc.as_ref()?.passes(stack).len();
        if !self.doc.as_ref()?.stacks().contains(&stack) || count >= MAX_PASSES {
            return None;
        }
        self.settle();
        self.edit("add pass", |doc| {
            if let Some(passes) = doc.passes_mut(stack) {
                passes.push(Pass {
                    block,
                    params: std::collections::BTreeMap::new(),
                });
            }
        });
        self.settle();
        self.pick_pass(stack, count);
        Some(count)
    }

    /// Take pass `pass` out of `stack`. A control that acted on it no
    /// longer does, and the others follow their passes.
    pub fn remove_pass(&mut self, stack: Stack, pass: usize) {
        self.settle();
        self.edit("remove pass", |doc| {
            let Some(passes) = doc.passes_mut(stack) else {
                return;
            };
            if pass >= passes.len() {
                return;
            }
            passes.remove(pass);
            if let Doc::Effect(recipe) = doc {
                retarget_removed(&mut recipe.controls, pass);
            }
        });
        self.settle();
        let muted: BTreeSet<(Stack, usize)> = self
            .muted
            .iter()
            .filter(|(at, index)| !(*at == stack && *index == pass))
            .map(|(at, index)| {
                if *at == stack && *index > pass {
                    (*at, index - 1)
                } else {
                    (*at, *index)
                }
            })
            .collect();
        self.muted = muted;
        self.slot = None;
        self.key = None;
        self.keep_picks();
    }

    /// Move pass `from` of `stack` to place `to`. Controls follow.
    pub fn move_pass(&mut self, stack: Stack, from: usize, to: usize) {
        let Some(count) = self.doc.as_ref().map(|doc| doc.passes(stack).len()) else {
            return;
        };
        if from >= count || to >= count || from == to {
            return;
        }
        self.settle();
        self.edit("move pass", |doc| {
            if let Some(passes) = doc.passes_mut(stack) {
                let pass = passes.remove(from);
                passes.insert(to, pass);
            }
            if let Doc::Effect(recipe) = doc {
                retarget_moved(&mut recipe.controls, from, to);
            }
        });
        self.settle();
        let was_muted = self.muted.remove(&(stack, from));
        let muted: BTreeSet<(Stack, usize)> = self
            .muted
            .iter()
            .map(|(at, index)| {
                let index = if *at != stack {
                    *index
                } else if from < to && *index > from && *index <= to {
                    index - 1
                } else if to < from && *index >= to && *index < from {
                    index + 1
                } else {
                    *index
                };
                (*at, index)
            })
            .collect();
        self.muted = muted;
        if was_muted {
            self.muted.insert((stack, to));
        }
        self.stack = stack;
        self.pass = Some(to);
        self.slot = None;
        self.key = None;
    }

    // Values.

    /// The stream a change to `slot` belongs to.
    fn slot_stream(slot: &Slot, what: &str) -> String {
        format!("{what} {slot:?}")
    }

    /// Set `slot` to one fixed value, kept to its range.
    pub fn set_fixed(&mut self, slot: &Slot, value: f32) {
        let Some(param) = self.doc.as_ref().and_then(|doc| doc.param(slot)) else {
            return;
        };
        if !value.is_finite() {
            return;
        }
        let value = value.clamp(param.min, param.max);
        self.edit(&Self::slot_stream(slot, "fixed"), |doc| {
            doc.set_curve(slot, Some(Curve::Constant(value)));
        });
    }

    /// Leave `slot` at its block's default: nothing is written for it.
    pub fn reset(&mut self, slot: &Slot) {
        self.settle();
        self.edit(&Self::slot_stream(slot, "reset"), |doc| {
            doc.set_curve(slot, None);
        });
        self.settle();
        self.keep_picks();
    }

    /// Change how `slot` moves, starting from what it is at the start so
    /// the picture does not jump.
    pub fn set_kind(&mut self, slot: &Slot, kind: CurveKind) {
        let Some(doc) = self.doc.as_ref() else {
            return;
        };
        if CurveKind::of(doc.curve(slot)) == kind {
            return;
        }
        let Some(start) = doc.value_at(slot, 0.0) else {
            return;
        };
        let end = self.span();
        let curve = match kind {
            CurveKind::Fixed => Curve::Constant(start),
            CurveKind::Keys => Curve::Keys {
                keys: vec![[0.0, start], [end, start]],
                repeat: None,
                ease: Vec::new(),
            },
            CurveKind::Wave => Curve::Wave {
                wave: Wave {
                    period: 1.0,
                    from: start,
                    to: start,
                },
            },
            CurveKind::Random => Curve::Jitter {
                jitter: Jitter {
                    rate: 8.0,
                    from: start,
                    to: start,
                    seed: 0,
                },
            },
        };
        self.settle();
        self.edit(&Self::slot_stream(slot, "kind"), |doc| {
            doc.set_curve(slot, Some(curve));
        });
        self.settle();
        self.key = None;
    }

    /// Change the keys of `slot`, when it has keys, on `stream`.
    fn edit_keys(
        &mut self,
        slot: &Slot,
        stream: &str,
        change: impl FnOnce(&mut Vec<[f32; 2]>, &mut Option<f32>, &mut Vec<Ease>),
    ) {
        self.edit(&Self::slot_stream(slot, stream), |doc| {
            let Some(Curve::Keys { keys, repeat, ease }) = doc.curve(slot) else {
                return;
            };
            let (mut keys, mut repeat, mut ease) = (keys.clone(), *repeat, ease.clone());
            // Each gap's ease spelled out, so a change to the keys can
            // keep each one with its gap.
            ease = (0..keys.len().saturating_sub(1))
                .map(|gap| gap_ease(&ease, gap))
                .collect();
            change(&mut keys, &mut repeat, &mut ease);
            tidy_ease(&mut ease, keys.len());
            doc.set_curve(slot, Some(Curve::Keys { keys, repeat, ease }));
        });
    }

    /// Add a key to `slot` at `time` with `value`, both kept in range, and
    /// pick it. The gap it splits keeps its ease on both sides. None when
    /// the slot has no keys or holds as many as a curve may.
    pub fn add_key(&mut self, slot: &Slot, time: f32, value: f32) -> Option<usize> {
        let doc = self.doc.as_ref()?;
        let param = doc.param(slot)?;
        let Some(Curve::Keys { keys, .. }) = doc.curve(slot) else {
            return None;
        };
        if keys.len() >= MAX_KEYS || !time.is_finite() || !value.is_finite() {
            return None;
        }
        let time = time.max(0.0);
        let value = value.clamp(param.min, param.max);
        let at = keys.iter().take_while(|key| key[0] <= time).count();
        self.settle();
        self.edit_keys(slot, "add key", |keys, _, ease| {
            keys.insert(at, [time, value]);
            // The new key splits gap `at - 1`; both halves move as it did.
            if at > 0 && at <= ease.len() {
                let split = ease[at - 1];
                ease.insert(at - 1, split);
            } else if at == 0 {
                ease.insert(0, Ease::Linear);
            } else {
                ease.push(ease.last().copied().unwrap_or_default());
            }
        });
        self.settle();
        self.slot = Some(slot.clone());
        self.key = Some(at);
        Some(at)
    }

    /// Move key `key` of `slot` to `time` and `value`. It stays between
    /// its neighbours, so keys never change order.
    pub fn move_key(&mut self, slot: &Slot, key: usize, time: f32, value: f32) {
        let Some(param) = self.doc.as_ref().and_then(|doc| doc.param(slot)) else {
            return;
        };
        if !time.is_finite() || !value.is_finite() {
            return;
        }
        self.edit_keys(slot, &format!("move key {key}"), |keys, _, _| {
            if key >= keys.len() {
                return;
            }
            let low = if key == 0 { 0.0 } else { keys[key - 1][0] };
            let high = keys.get(key + 1).map_or(f32::MAX, |next| next[0]);
            keys[key] = [time.clamp(low, high), value.clamp(param.min, param.max)];
        });
    }

    /// Take key `key` out of `slot`. The last key stays: keys with none
    /// are not a curve.
    pub fn remove_key(&mut self, slot: &Slot, key: usize) {
        self.settle();
        self.edit_keys(slot, "remove key", |keys, _, ease| {
            if keys.len() <= 1 || key >= keys.len() {
                return;
            }
            keys.remove(key);
            if !ease.is_empty() {
                ease.remove(key.min(ease.len() - 1));
            }
        });
        self.settle();
        self.key = None;
        self.keep_picks();
    }

    /// How the value moves across the gap after key `gap` of `slot`.
    pub fn set_ease(&mut self, slot: &Slot, gap: usize, to: Ease) {
        self.settle();
        self.edit_keys(slot, "ease", |_, _, ease| {
            if let Some(ease) = ease.get_mut(gap) {
                *ease = to;
            }
        });
        self.settle();
    }

    /// Wrap the keys of `slot` every `seconds`; none plays them once.
    pub fn set_repeat(&mut self, slot: &Slot, seconds: Option<f32>) {
        let seconds = seconds.filter(|seconds| seconds.is_finite() && *seconds > 0.0);
        self.edit_keys(slot, "repeat", |_, repeat, _| *repeat = seconds);
    }

    /// Set the wave of `slot`, when it is one: its ends kept to the range
    /// and its period above zero.
    pub fn set_wave(&mut self, slot: &Slot, wave: &Wave) {
        let Some(param) = self.doc.as_ref().and_then(|doc| doc.param(slot)) else {
            return;
        };
        if ![wave.period, wave.from, wave.to]
            .iter()
            .all(|value| value.is_finite())
        {
            return;
        }
        let wave = Wave {
            period: wave.period.max(0.01),
            from: wave.from.clamp(param.min, param.max),
            to: wave.to.clamp(param.min, param.max),
        };
        self.edit(&Self::slot_stream(slot, "wave"), |doc| {
            if matches!(doc.curve(slot), Some(Curve::Wave { .. })) {
                doc.set_curve(slot, Some(Curve::Wave { wave }));
            }
        });
    }

    /// Set the random steps of `slot`, when it is that: its ends kept to
    /// the range and its rate to what a recipe may ask.
    pub fn set_jitter(&mut self, slot: &Slot, jitter: &Jitter) {
        let Some(param) = self.doc.as_ref().and_then(|doc| doc.param(slot)) else {
            return;
        };
        if ![jitter.rate, jitter.from, jitter.to]
            .iter()
            .all(|value| value.is_finite())
        {
            return;
        }
        let jitter = Jitter {
            rate: jitter.rate.clamp(0.1, cutlass_effects::MAX_JITTER_RATE),
            from: jitter.from.clamp(param.min, param.max),
            to: jitter.to.clamp(param.min, param.max),
            seed: jitter.seed,
        };
        self.edit(&Self::slot_stream(slot, "random"), |doc| {
            if matches!(doc.curve(slot), Some(Curve::Jitter { .. })) {
                doc.set_curve(slot, Some(Curve::Jitter { jitter }));
            }
        });
    }

    // Controls: an effect's settings in the inspector.

    /// The effect's controls; none for a transition.
    #[must_use]
    pub fn controls(&self) -> &[Control] {
        match &self.doc {
            Some(Doc::Effect(recipe)) => &recipe.controls,
            _ => &[],
        }
    }

    /// Add a control that does nothing yet. None when the effect has as
    /// many as it may, or the document is a transition.
    pub fn add_control(&mut self) -> Option<usize> {
        let Some(Doc::Effect(recipe)) = self.doc.as_ref() else {
            return None;
        };
        let count = recipe.controls.len();
        if count >= MAX_CONTROLS {
            return None;
        }
        let used = |id: &str| recipe.controls.iter().any(|control| control.id == id);
        let id = std::iter::once("amount".to_owned())
            .chain((2..).map(|n| format!("amount_{n}")))
            .find(|id| !used(id))
            .unwrap_or_default();
        self.settle();
        self.edit("add control", |doc| {
            if let Doc::Effect(recipe) = doc {
                recipe.controls.push(Control {
                    id,
                    name: "Amount".to_owned(),
                    default: 1.0,
                    min: 0.0,
                    max: 2.0,
                    unit: ControlUnit::Percent,
                    speed: false,
                    scale: Vec::new(),
                    set: Vec::new(),
                    keyframe: String::new(),
                });
            }
        });
        self.settle();
        Some(count)
    }

    pub fn remove_control(&mut self, control: usize) {
        self.settle();
        self.edit("remove control", |doc| {
            if let Doc::Effect(recipe) = doc
                && control < recipe.controls.len()
            {
                recipe.controls.remove(control);
            }
        });
        self.settle();
    }

    /// Change control `control`'s own fields: its name, id, range, default,
    /// unit and whether it is a speed.
    pub fn set_control(&mut self, control: usize, change: impl FnOnce(&mut Control)) {
        self.edit(&format!("control {control}"), |doc| {
            if let Doc::Effect(recipe) = doc
                && let Some(control) = recipe.controls.get_mut(control)
            {
                change(control);
            }
        });
    }

    /// How control `control` acts on parameter `name` of the effect's pass
    /// `pass`.
    #[must_use]
    pub fn link(&self, control: usize, pass: usize, name: &str) -> Link {
        let Some(control) = self.controls().get(control) else {
            return Link::None;
        };
        let aims = |targets: &[Target]| {
            targets
                .iter()
                .any(|Target(at, param)| *at == pass && param == name)
        };
        if aims(&control.set) {
            Link::Set
        } else if aims(&control.scale) {
            Link::Scale
        } else {
            Link::None
        }
    }

    /// Make control `control` act on parameter `name` of pass `pass` as
    /// `link` says.
    pub fn set_link(&mut self, control: usize, pass: usize, name: &str, link: Link) {
        let known = matches!(&self.doc, Some(doc) if doc.param(&Slot::pass(Stack::Effect, pass, name)).is_some());
        if !known {
            return;
        }
        self.settle();
        self.edit("link", |doc| {
            let Doc::Effect(recipe) = doc else {
                return;
            };
            let Some(control) = recipe.controls.get_mut(control) else {
                return;
            };
            let same = |Target(at, param): &Target| *at == pass && param == name;
            control.scale.retain(|target| !same(target));
            control.set.retain(|target| !same(target));
            match link {
                Link::None => {}
                Link::Scale => control.scale.push(Target(pass, name.to_owned())),
                Link::Set => control.set.push(Target(pass, name.to_owned())),
            }
        });
        self.settle();
    }

    // What the preview draws.

    /// The effect as the preview runs it: without the muted passes, their
    /// controls following. None for a transition, or when every pass is
    /// muted.
    #[must_use]
    pub fn preview_recipe(&self) -> Option<Recipe> {
        let Some(Doc::Effect(recipe)) = &self.doc else {
            return None;
        };
        let mut shown = recipe.clone();
        for pass in (0..shown.passes.len()).rev() {
            if self.muted.contains(&(Stack::Effect, pass)) {
                shown.passes.remove(pass);
                retarget_removed(&mut shown.controls, pass);
            }
        }
        (!shown.passes.is_empty()).then_some(shown)
    }

    /// Moment `t`, 0 to 1, of the transition as the preview runs it:
    /// without the muted passes. None for an effect.
    #[must_use]
    pub fn preview_crossing(&self, t: f32) -> Option<Crossing> {
        let Some(Doc::Transition(recipe)) = &self.doc else {
            return None;
        };
        let mut shown = recipe.clone();
        for (stack, passes) in [
            (Stack::Outgoing, &mut shown.outgoing),
            (Stack::Incoming, &mut shown.incoming),
        ] {
            for pass in (0..passes.len()).rev() {
                if self.muted.contains(&(stack, pass)) {
                    passes.remove(pass);
                }
            }
        }
        Some(shown.at(t))
    }
}

#[cfg(test)]
mod tests {
    use super::{CurveKind, Doc, Link, Slot, Stack, Studio, slug};
    use cutlass_effects::{Block, Curve, Ease, Jitter, MAX_CONTROLS, MAX_PASSES, Target, Wave};

    fn effect(studio: &Studio) -> &cutlass_effects::Recipe {
        match studio.doc() {
            Some(Doc::Effect(recipe)) => recipe,
            other => panic!("not an effect: {other:?}"),
        }
    }

    fn radius() -> Slot {
        Slot::pass(Stack::Effect, 0, "radius")
    }

    fn keys(studio: &Studio, slot: &Slot) -> (Vec<[f32; 2]>, Option<f32>, Vec<Ease>) {
        match studio.doc().and_then(|doc| doc.curve(slot)) {
            Some(Curve::Keys { keys, repeat, ease }) => (keys.clone(), *repeat, ease.clone()),
            other => panic!("not keys: {other:?}"),
        }
    }

    #[test]
    fn a_new_effect_is_valid_and_unsaved() {
        let mut studio = Studio::new(None);
        assert!(studio.doc().is_none());
        assert!(!studio.is_dirty());
        studio.new_effect();
        let recipe = effect(&studio);
        assert_eq!(
            (recipe.id.as_str(), recipe.name.as_str()),
            ("my-effect", "My effect")
        );
        assert_eq!(studio.problem(), None);
        assert!(studio.is_dirty() && !studio.is_saved());
        assert_eq!(studio.picked_pass(), Some((Stack::Effect, 0)));
        // With the studio folder off there is nowhere to save.
        assert!(studio.save().is_err());
        studio.close();
        assert!(studio.doc().is_none());
    }

    #[test]
    fn a_bundled_effect_opens_as_a_copy_under_a_free_id() {
        let mut studio = Studio::new(None);
        assert!(studio.open_effect("shake"));
        let recipe = effect(&studio);
        assert_eq!(
            (recipe.id.as_str(), recipe.name.as_str()),
            ("shake-copy", "Shake copy")
        );
        assert_eq!(
            recipe.passes,
            cutlass_effects::bundled_recipe("recipe/shake")
                .unwrap()
                .passes
        );
        assert!(!studio.is_saved());
        assert_eq!(studio.problem(), None);
        // Shake moves, so the lane opens on the first value that does.
        let slot = studio.picked_slot().cloned().expect("a moving value");
        assert!(!matches!(
            studio.doc().unwrap().curve(&slot),
            None | Some(Curve::Constant(_))
        ));
        assert!(
            matches!(&slot, Slot::Pass { stack: Stack::Effect, pass, .. }
            if studio.picked_pass() == Some((Stack::Effect, *pass)))
        );
        assert!(!studio.open_effect("no-such-effect"));
        // The bundled id itself is taken.
        studio.set_id("shake");
        assert!(
            studio
                .problem()
                .is_some_and(|problem| problem.contains("shake"))
        );
        assert!(
            Studio::effects()
                .iter()
                .any(|entry| entry.id == "shake" && !entry.studio)
        );
    }

    #[test]
    fn the_id_follows_the_name_until_it_is_typed() {
        assert_eq!(slug("Violet  Haze! 2"), "violet-haze-2");
        assert_eq!(slug("  --  "), "");
        let mut studio = Studio::new(None);
        studio.new_effect();
        studio.set_name("Violet Haze");
        assert_eq!(effect(&studio).id, "violet-haze");
        // A name that slugs to a bundled id gets the next free one.
        studio.set_name("Shake");
        assert_eq!(effect(&studio).id, "shake-2");
        studio.set_id("mine");
        studio.set_name("Other");
        assert_eq!(effect(&studio).id, "mine");
        // A name with no letters is refused by the recipe's own check.
        studio.set_name(" ");
        assert!(studio.problem().is_some());
    }

    #[test]
    fn passes_are_added_moved_and_removed_and_controls_follow() {
        let mut studio = Studio::new(None);
        studio.new_effect();
        assert_eq!(studio.add_pass(Stack::Effect, Block::Grade), Some(1));
        assert_eq!(studio.add_pass(Stack::Effect, Block::Glow), Some(2));
        assert_eq!(studio.picked_pass(), Some((Stack::Effect, 2)));
        // A transition's stack is not an effect's.
        assert_eq!(studio.add_pass(Stack::Outgoing, Block::Blur), None);
        let control = studio.add_control().unwrap();
        studio.set_link(control, 0, "radius", Link::Scale);
        studio.set_link(control, 1, "contrast", Link::Set);
        assert_eq!(studio.link(control, 0, "radius"), Link::Scale);
        assert_eq!(studio.link(control, 1, "contrast"), Link::Set);
        // A parameter the pass does not have is not linked.
        studio.set_link(control, 0, "contrast", Link::Scale);
        assert_eq!(studio.link(control, 0, "contrast"), Link::None);

        // Blur, Grade, Glow -> Grade, Glow, Blur: the links follow.
        studio.set_muted(Stack::Effect, 0, true);
        studio.move_pass(Stack::Effect, 0, 2);
        let blocks: Vec<Block> = effect(&studio)
            .passes
            .iter()
            .map(|pass| pass.block)
            .collect();
        assert_eq!(blocks, [Block::Grade, Block::Glow, Block::Blur]);
        assert_eq!(studio.link(control, 2, "radius"), Link::Scale);
        assert_eq!(studio.link(control, 0, "contrast"), Link::Set);
        assert!(studio.is_muted(Stack::Effect, 2) && !studio.is_muted(Stack::Effect, 0));
        assert_eq!(studio.picked_pass(), Some((Stack::Effect, 2)));
        assert_eq!(studio.problem(), None);

        // Removing Grade drops its link and moves the other down.
        studio.remove_pass(Stack::Effect, 0);
        assert_eq!(effect(&studio).controls[0].set, Vec::<Target>::new());
        assert_eq!(studio.link(control, 1, "radius"), Link::Scale);
        assert!(studio.is_muted(Stack::Effect, 1));
        assert_eq!(studio.problem(), None);

        // The preview leaves the muted pass out; the document keeps it.
        let shown = studio.preview_recipe().unwrap();
        assert_eq!(shown.passes.len(), 1);
        assert_eq!(shown.passes[0].block, Block::Glow);
        assert!(shown.controls[0].scale.is_empty());
        assert_eq!(effect(&studio).passes.len(), 2);
        studio.set_muted(Stack::Effect, 0, true);
        assert!(studio.preview_recipe().is_none(), "every pass is muted");

        // The stack fills up and then refuses.
        while effect(&studio).passes.len() < MAX_PASSES {
            assert!(studio.add_pass(Stack::Effect, Block::Invert).is_some());
        }
        assert_eq!(studio.add_pass(Stack::Effect, Block::Invert), None);
        // An effect with no passes is not a recipe.
        for _ in 0..MAX_PASSES {
            studio.remove_pass(Stack::Effect, 0);
        }
        assert!(studio.problem().is_some());
    }

    #[test]
    fn a_value_is_fixed_keyed_waved_or_random_and_stays_in_range() {
        let mut studio = Studio::new(None);
        studio.new_effect();
        let slot = radius();
        // Blur's radius runs from 0 to 0.1.
        studio.set_fixed(&slot, 5.0);
        assert_eq!(studio.doc().unwrap().value_at(&slot, 0.0), Some(0.1));
        studio.set_fixed(&slot, f32::NAN);
        assert_eq!(studio.doc().unwrap().value_at(&slot, 0.0), Some(0.1));
        studio.set_fixed(&slot, 0.02);

        studio.set_kind(&slot, CurveKind::Keys);
        let (held, repeat, ease) = keys(&studio, &slot);
        assert_eq!(
            held,
            vec![[0.0, 0.02], [3.0, 0.02]],
            "the picture does not jump"
        );
        assert_eq!((repeat, ease), (None, Vec::new()));

        // A key in the middle, eased; keys stay in order and in range.
        assert_eq!(studio.add_key(&slot, 1.0, 9.0), Some(1));
        assert_eq!(studio.picked_key(), Some(1));
        studio.set_ease(&slot, 0, Ease::Out);
        studio.move_key(&slot, 1, 99.0, -4.0);
        let (held, _, ease) = keys(&studio, &slot);
        assert_eq!(held, vec![[0.0, 0.02], [3.0, 0.0], [3.0, 0.02]]);
        assert_eq!(ease, vec![Ease::Out, Ease::Linear]);
        studio.move_key(&slot, 1, 1.0, 0.1);
        // A key that splits an eased gap leaves both halves eased.
        assert_eq!(studio.add_key(&slot, 0.5, 0.05), Some(1));
        let (_, _, ease) = keys(&studio, &slot);
        assert_eq!(ease, vec![Ease::Out, Ease::Out, Ease::Linear]);
        studio.remove_key(&slot, 1);
        let (held, _, ease) = keys(&studio, &slot);
        assert_eq!(held.len(), 3);
        assert_eq!(ease, vec![Ease::Out, Ease::Linear]);
        studio.set_repeat(&slot, Some(2.0));
        studio.set_repeat(&slot, Some(-1.0));
        assert_eq!(keys(&studio, &slot).1, None);
        // Straight lines everywhere are written as no eases at all.
        studio.set_ease(&slot, 0, Ease::Linear);
        assert_eq!(keys(&studio, &slot).2, Vec::new());
        assert_eq!(studio.problem(), None);
        // The last key cannot be removed.
        studio.remove_key(&slot, 0);
        studio.remove_key(&slot, 0);
        studio.remove_key(&slot, 0);
        assert_eq!(keys(&studio, &slot).0.len(), 1);

        studio.set_kind(&slot, CurveKind::Wave);
        studio.set_wave(
            &slot,
            &Wave {
                period: -2.0,
                from: -1.0,
                to: 1.0,
            },
        );
        assert_eq!(
            studio.doc().unwrap().curve(&slot),
            Some(&Curve::Wave {
                wave: Wave {
                    period: 0.01,
                    from: 0.0,
                    to: 0.1
                }
            })
        );
        studio.set_kind(&slot, CurveKind::Random);
        studio.set_jitter(
            &slot,
            &Jitter {
                rate: 1e9,
                from: 0.0,
                to: 0.05,
                seed: 3,
            },
        );
        assert_eq!(
            studio.problem(),
            None,
            "the rate is kept to what a recipe may ask"
        );
        studio.reset(&slot);
        assert_eq!(studio.doc().unwrap().curve(&slot), None);
        assert_eq!(studio.doc().unwrap().value_at(&slot, 0.0), Some(0.0));
        // A slot the document does not have changes nothing.
        let before = studio.doc().cloned();
        studio.set_fixed(&Slot::pass(Stack::Effect, 7, "radius"), 0.01);
        studio.set_fixed(&Slot::cross("soft"), 0.5);
        assert_eq!(studio.doc().cloned(), before);
    }

    #[test]
    fn an_overshoot_past_the_range_is_named_as_the_problem() {
        let mut studio = Studio::new(None);
        studio.new_effect();
        let slot = radius();
        studio.set_kind(&slot, CurveKind::Keys);
        studio.move_key(&slot, 1, 1.0, 0.1);
        assert_eq!(studio.problem(), None);
        studio.set_ease(&slot, 0, Ease::Overshoot);
        let problem = studio.problem().expect("the overshoot leaves the range");
        assert!(problem.contains("radius"), "{problem}");
    }

    #[test]
    fn a_drag_is_one_step_to_undo_and_redo_brings_it_back() {
        let mut studio = Studio::new(None);
        studio.new_effect();
        let start = studio.doc().cloned();
        let slot = radius();
        for value in [0.02, 0.03, 0.04] {
            studio.set_fixed(&slot, value);
        }
        studio.settle();
        studio.set_fixed(&slot, 0.05);
        studio.settle();
        // A change that changes nothing is not a step.
        studio.set_fixed(&slot, 0.05);
        studio.undo();
        assert_eq!(studio.doc().unwrap().value_at(&slot, 0.0), Some(0.04));
        studio.undo();
        assert_eq!(studio.doc().cloned(), start);
        assert!(!studio.can_undo());
        studio.undo();
        studio.redo();
        studio.redo();
        assert_eq!(studio.doc().unwrap().value_at(&slot, 0.0), Some(0.05));
        assert!(!studio.can_redo());
        // A new change after an undo drops what could be redone.
        studio.undo();
        studio.set_fixed(&slot, 0.01);
        assert!(!studio.can_redo());
        // Undoing a pass that was added keeps the pick on a pass that is there.
        studio.add_pass(Stack::Effect, Block::Glow);
        studio.undo();
        assert_eq!(studio.picked_pass(), Some((Stack::Effect, 0)));
    }

    #[test]
    fn a_control_is_added_changed_and_removed_up_to_the_limit() {
        let mut studio = Studio::new(None);
        studio.new_effect();
        let first = studio.add_control().unwrap();
        studio.set_control(first, |control| {
            control.name = "Strength".to_owned();
            control.max = 3.0;
        });
        assert_eq!(studio.controls()[0].name, "Strength");
        let second = studio.add_control().unwrap();
        assert_ne!(studio.controls()[first].id, studio.controls()[second].id);
        assert_eq!(studio.problem(), None);
        while studio.controls().len() < MAX_CONTROLS {
            assert!(studio.add_control().is_some());
        }
        assert_eq!(studio.add_control(), None);
        assert_eq!(studio.problem(), None);
        studio.remove_control(0);
        assert_eq!(studio.controls().len(), MAX_CONTROLS - 1);
        // A range that does not hold its default is the recipe's problem.
        studio.set_control(0, |control| control.min = 5.0);
        assert!(studio.problem().is_some());
    }

    #[test]
    fn a_transition_has_two_stacks_and_cross_values_over_its_progress() {
        let mut studio = Studio::new(None);
        studio.new_transition();
        let doc = studio.doc().unwrap();
        assert!(!doc.is_effect());
        assert_eq!(doc.stacks(), [Stack::Outgoing, Stack::Incoming]);
        assert_eq!(studio.problem(), None);
        assert_eq!(studio.span(), 1.0);
        assert_eq!(studio.add_pass(Stack::Effect, Block::Blur), None);
        assert_eq!(studio.add_pass(Stack::Outgoing, Block::Blur), Some(0));
        assert_eq!(studio.add_pass(Stack::Incoming, Block::Transform), Some(0));
        assert_eq!(studio.add_control(), None, "a transition has no controls");

        // Progress left out follows the transition itself.
        let progress = Slot::cross("progress");
        assert_eq!(studio.doc().unwrap().value_at(&progress, 0.25), Some(0.25));
        let soft = Slot::cross("soft");
        studio.set_fixed(&soft, 0.4);
        assert_eq!(studio.doc().unwrap().value_at(&soft, 0.5), Some(0.4));
        // Keys on a transition run over its progress, 0 to 1.
        let blur = Slot::pass(Stack::Outgoing, 0, "radius");
        studio.set_kind(&blur, CurveKind::Keys);
        assert_eq!(keys(&studio, &blur).0, vec![[0.0, 0.0], [1.0, 0.0]]);
        studio.move_key(&blur, 1, 1.0, 0.05);
        studio.set_duration(Some(99.0));
        studio.set_preview_pair("city", "lake");
        assert_eq!(studio.problem(), None);

        let crossing = studio.preview_crossing(1.0).unwrap();
        assert_eq!(crossing.outgoing.len(), 1);
        assert!((crossing.outgoing[0].values[0] - 0.05).abs() < 1e-6);
        assert!(
            (crossing.cross[3] - 0.4).abs() < 1e-6,
            "soft is the fourth value"
        );
        studio.set_muted(Stack::Outgoing, 0, true);
        let crossing = studio.preview_crossing(1.0).unwrap();
        assert!(crossing.outgoing.is_empty());
        assert_eq!(crossing.incoming.len(), 1);
        assert!(studio.preview_recipe().is_none());

        assert!(studio.open_transition("photo-slide"));
        assert_eq!(studio.doc().unwrap().id(), "photo-slide-copy");
        assert!(
            !studio.open_transition("dissolve"),
            "a built-in is not a recipe"
        );
        assert!(
            Studio::transitions()
                .iter()
                .any(|entry| entry.id == "photo-slide")
        );
    }
}
