//! The studio folder: effects and transitions being made on this machine,
//! run beside the ones the app ships.
//!
//! The folder holds `effects/<id>.json` and `transitions/<id>.json`, each a
//! recipe as the app ships them. [`load_studio`] reads both and makes what
//! it read the studio's recipes; [`recipe`] and [`recipes`] then answer for
//! a bundled recipe and a studio one alike, so the shelf, the inspector and
//! a frame treat them the same. [`studio_stamp`] changes when a file is
//! added, removed or rewritten, so a shell reads the folder again only then.
//!
//! A file is untrusted: it is a regular file named after its id, no larger
//! than a recipe may be, and it passes every check a bundled recipe does.
//! One that fails is left out and named in [`Loaded::refused`]. An id the
//! app already ships is refused, so a studio file never changes what a
//! bundled effect draws.
//!
//! A recipe read from the folder is kept for as long as the app runs, so a
//! lookup hands out `&'static` as it does for a bundled one. Reading a file
//! that has not changed keeps the recipe already held; each changed version
//! is a few hundred bytes more.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{PoisonError, RwLock};

use crate::recipe::{MAX_RECIPE_BYTES, Recipe, RecipeError, bundled, bundled_recipe};
use crate::{
    BUILT_IN_PREFIX, RECIPE_PREFIX, TransitionKind, TransitionRecipe, bundled_transitions,
};

/// The folder of effect recipes inside the studio folder.
pub const STUDIO_EFFECTS: &str = "effects";
/// The folder of transition recipes inside the studio folder.
pub const STUDIO_TRANSITIONS: &str = "transitions";
/// The most files read from each of the two folders.
pub const MAX_STUDIO_FILES: usize = 256;

const EXTENSION: &str = "json";

/// Why a studio file was left out, or could not be written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudioError {
    /// The folder or the file could not be read or written.
    Io(String),
    /// Not a regular file: a folder, or a link to somewhere else.
    NotAFile,
    /// The file is not named `<id>.json` for the id inside it.
    FileName,
    /// The app ships an effect or a transition with this id.
    Taken(String),
    /// More than [`MAX_STUDIO_FILES`] files in the folder.
    TooMany,
    /// The id is not one a recipe may have.
    Id,
    /// The recipe itself is refused.
    Recipe(RecipeError),
}

impl fmt::Display for StudioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "{message}"),
            Self::NotAFile => write!(formatter, "not a regular file"),
            Self::FileName => write!(formatter, "the file is not named after its id"),
            Self::Taken(id) => write!(formatter, "the app already ships {id}"),
            Self::TooMany => write!(
                formatter,
                "the folder holds more than {MAX_STUDIO_FILES} recipes"
            ),
            Self::Id => write!(formatter, "not a recipe id"),
            Self::Recipe(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for StudioError {}

impl From<RecipeError> for StudioError {
    fn from(error: RecipeError) -> Self {
        Self::Recipe(error)
    }
}

/// A file [`load_studio`] left out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    /// The file, from the studio folder: `effects/shake.json`.
    pub file: String,
    pub error: StudioError,
}

/// What [`load_studio`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Loaded {
    /// How many effect recipes the studio now holds.
    pub effects: usize,
    /// How many transition recipes the studio now holds.
    pub transitions: usize,
    /// The files left out, effects first, in name order.
    pub refused: Vec<Refused>,
}

/// The studio's recipes while the app runs.
struct Registry {
    effects: Vec<&'static Recipe>,
    /// One place for each transition id seen so far, never moved or
    /// removed, so a [`TransitionKind::Recipe`] names the same transition
    /// for as long as the app runs. A transition whose file has gone holds
    /// none.
    transitions: Vec<(&'static str, Option<&'static TransitionRecipe>)>,
}

impl Registry {
    const fn new() -> Self {
        Self {
            effects: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Hold exactly `effects` and `transitions`. A recipe equal to one
    /// already held keeps that one. True when anything changed.
    fn install(&mut self, effects: Vec<Recipe>, transitions: Vec<TransitionRecipe>) -> bool {
        let before: Vec<*const Recipe> =
            self.effects.iter().map(|held| &raw const **held).collect();
        let effects: Vec<&'static Recipe> = effects
            .into_iter()
            .map(|recipe| {
                let held = self.effects.iter().find(|held| ***held == recipe);
                held.copied().unwrap_or_else(|| Box::leak(Box::new(recipe)))
            })
            .collect();
        let after: Vec<*const Recipe> = effects.iter().map(|held| &raw const **held).collect();
        let mut changed = before != after;
        self.effects = effects;

        let named: BTreeSet<&str> = transitions
            .iter()
            .map(|recipe| recipe.id.as_str())
            .collect();
        for (id, held) in &mut self.transitions {
            if held.is_some() && !named.contains(id) {
                *held = None;
                changed = true;
            }
        }
        for recipe in transitions {
            match self.transitions.iter_mut().find(|(id, _)| *id == recipe.id) {
                Some((_, held)) => {
                    if held.is_none_or(|held| *held != recipe) {
                        *held = Some(Box::leak(Box::new(recipe)));
                        changed = true;
                    }
                }
                None => {
                    let recipe: &'static TransitionRecipe = Box::leak(Box::new(recipe));
                    self.transitions.push((recipe.id.as_str(), Some(recipe)));
                    changed = true;
                }
            }
        }
        changed
    }
}

static REGISTRY: RwLock<Registry> = RwLock::new(Registry::new());
/// How many studio effects are held, so a frame of a draft with none
/// never takes the lock.
static EFFECTS: AtomicUsize = AtomicUsize::new(0);
static GENERATION: AtomicU64 = AtomicU64::new(0);

fn held<T>(read: impl FnOnce(&Registry) -> T) -> T {
    read(&REGISTRY.read().unwrap_or_else(PoisonError::into_inner))
}

/// A number that changes each time the studio's recipes do. A shell that
/// keeps cards or pictures of them draws those again when it moves.
#[must_use]
pub fn studio_generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// The recipe stored under `effect_id`, such as `recipe/shake`: one the app
/// ships, or one in the studio.
#[must_use]
pub fn recipe(effect_id: &str) -> Option<&'static Recipe> {
    if let Some(recipe) = bundled_recipe(effect_id) {
        return Some(recipe);
    }
    if EFFECTS.load(Ordering::Acquire) == 0 {
        return None;
    }
    let id = effect_id.strip_prefix(RECIPE_PREFIX)?;
    held(|registry| {
        registry
            .effects
            .iter()
            .copied()
            .find(|recipe| recipe.id == id)
    })
}

/// Every recipe the app can run, in picker order: the bundled ones, then
/// the studio's by id.
#[must_use]
pub fn recipes() -> Vec<&'static Recipe> {
    bundled().iter().chain(studio_recipes()).collect()
}

/// The studio's effect recipes, by id.
#[must_use]
pub fn studio_recipes() -> Vec<&'static Recipe> {
    if EFFECTS.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    held(|registry| registry.effects.clone())
}

/// Whether `effect_id` names a recipe from the studio folder.
#[must_use]
pub fn is_studio_recipe(effect_id: &str) -> bool {
    bundled_recipe(effect_id).is_none() && recipe(effect_id).is_some()
}

/// The studio transition at `place`, counted after the bundled ones.
pub(crate) fn studio_transition(place: usize) -> Option<&'static TransitionRecipe> {
    held(|registry| registry.transitions.get(place).and_then(|(_, held)| *held))
}

/// The places of the studio's transitions, counted after the bundled ones.
pub(crate) fn studio_transition_places() -> Vec<usize> {
    held(|registry| {
        registry
            .transitions
            .iter()
            .enumerate()
            .filter(|(_, (_, held))| held.is_some())
            .map(|(place, _)| place)
            .collect()
    })
}

/// The studio's transition recipes, by id.
#[must_use]
pub fn studio_transitions() -> Vec<&'static TransitionRecipe> {
    let mut held: Vec<&'static TransitionRecipe> = held(|registry| {
        registry
            .transitions
            .iter()
            .filter_map(|(_, held)| *held)
            .collect()
    });
    held.sort_by(|a, b| a.id.cmp(&b.id));
    held
}

/// Whether a transition material's `effect_id` names one from the studio.
#[must_use]
pub fn is_studio_transition(effect_id: &str) -> bool {
    effect_id.strip_prefix(BUILT_IN_PREFIX).is_some_and(|key| {
        held(|registry| {
            registry
                .transitions
                .iter()
                .any(|(id, held)| *id == key && held.is_some())
        })
    })
}

/// Whether the app ships an effect with this id.
fn effect_taken(id: &str) -> bool {
    bundled().iter().any(|recipe| recipe.id == id)
}

/// Whether the app ships a transition with this id, built in or bundled.
fn transition_taken(id: &str) -> bool {
    TransitionKind::ALL.iter().any(|kind| kind.key() == id)
        || bundled_transitions().iter().any(|recipe| recipe.id == id)
}

/// Whether `id` is one a recipe may have, and so a safe file name.
fn id_ok(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

/// One file's name and bytes, or why it was left out.
type Read1 = (String, Result<Vec<u8>, StudioError>);

/// The recipe files directly inside `folder`, in name order. A folder that
/// is not there holds none.
fn read_folder(folder: &Path) -> Vec<Read1> {
    let Ok(entries) = fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| {
            !name.starts_with('.')
                && Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension == EXTENSION)
        })
        .collect();
    names.sort();
    names
        .into_iter()
        .enumerate()
        .map(|(place, name)| {
            let bytes = if place < MAX_STUDIO_FILES {
                read_file(&folder.join(&name))
            } else {
                Err(StudioError::TooMany)
            };
            (name, bytes)
        })
        .collect()
}

/// The bytes of a regular file no larger than a recipe may be. A link is
/// not followed, and a larger file is not read.
fn read_file(path: &Path) -> Result<Vec<u8>, StudioError> {
    let io = |error: std::io::Error| StudioError::Io(error.to_string());
    let meta = fs::symlink_metadata(path).map_err(io)?;
    if !meta.file_type().is_file() {
        return Err(StudioError::NotAFile);
    }
    let limit = MAX_RECIPE_BYTES as u64;
    if meta.len() > limit {
        return Err(RecipeError::TooLarge.into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(io)?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() as u64 > limit {
        return Err(RecipeError::TooLarge.into());
    }
    Ok(bytes)
}

/// The id a file named `name` must hold.
fn named_id(name: &str) -> Option<&str> {
    name.strip_suffix(EXTENSION)?.strip_suffix('.')
}

/// Read every file of one of the two folders with `read`, keeping the
/// recipes whose id is the file's name and is not `taken`.
fn read_kind<T>(
    studio: &Path,
    kind: &str,
    read: impl Fn(&[u8]) -> Result<T, RecipeError>,
    id: impl Fn(&T) -> &str,
    taken: impl Fn(&str) -> bool,
    refused: &mut Vec<Refused>,
) -> Vec<T> {
    let mut kept = Vec::new();
    for (name, bytes) in read_folder(&studio.join(kind)) {
        let recipe = bytes.and_then(|bytes| {
            let recipe = read(&bytes)?;
            if named_id(&name) != Some(id(&recipe)) {
                return Err(StudioError::FileName);
            }
            if taken(id(&recipe)) {
                return Err(StudioError::Taken(id(&recipe).to_owned()));
            }
            Ok(recipe)
        });
        match recipe {
            Ok(recipe) => kept.push(recipe),
            Err(error) => refused.push(Refused {
                file: format!("{kind}/{name}"),
                error,
            }),
        }
    }
    kept
}

/// What the studio folder holds now, without changing what the app runs.
fn read_studio(studio: &Path) -> (Vec<Recipe>, Vec<TransitionRecipe>, Vec<Refused>) {
    let mut refused = Vec::new();
    let effects = read_kind(
        studio,
        STUDIO_EFFECTS,
        Recipe::read,
        |recipe| recipe.id.as_str(),
        effect_taken,
        &mut refused,
    );
    let transitions = read_kind(
        studio,
        STUDIO_TRANSITIONS,
        TransitionRecipe::read,
        |recipe| recipe.id.as_str(),
        transition_taken,
        &mut refused,
    );
    (effects, transitions, refused)
}

/// Read the studio folder and make what it holds the studio's recipes: a
/// file that has gone takes its recipe with it. A folder that is not there
/// holds none.
pub fn load_studio(studio: &Path) -> Loaded {
    let (effects, transitions, refused) = read_studio(studio);
    let loaded = Loaded {
        effects: effects.len(),
        transitions: transitions.len(),
        refused,
    };
    install(effects, transitions);
    loaded
}

/// Drop every studio recipe, as when the studio folder is turned off.
pub fn clear_studio() {
    install(Vec::new(), Vec::new());
}

fn install(effects: Vec<Recipe>, transitions: Vec<TransitionRecipe>) {
    let mut registry = REGISTRY.write().unwrap_or_else(PoisonError::into_inner);
    let changed = registry.install(effects, transitions);
    EFFECTS.store(registry.effects.len(), Ordering::Release);
    if changed {
        GENERATION.fetch_add(1, Ordering::AcqRel);
    }
}

/// A number that changes when a recipe file in the studio folder is added,
/// removed, renamed or rewritten. It is the same for a folder that is not
/// there as for an empty one.
#[must_use]
pub fn studio_stamp(studio: &Path) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    for kind in [STUDIO_EFFECTS, STUDIO_TRANSITIONS] {
        let Ok(entries) = fs::read_dir(studio.join(kind)) else {
            continue;
        };
        let mut files: Vec<(std::ffi::OsString, u64, Option<std::time::SystemTime>)> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let meta = fs::symlink_metadata(entry.path()).ok()?;
                Some((entry.file_name(), meta.len(), meta.modified().ok()))
            })
            .collect();
        files.sort();
        (kind, files).hash(&mut hasher);
    }
    hasher.finish()
}

/// Write `json` as `<folder>/<id>.json`, whole or not at all: to a
/// neighbour first, then renamed over the old file.
fn write_file(folder: &Path, id: &str, json: &str) -> Result<PathBuf, StudioError> {
    let io = |error: std::io::Error| StudioError::Io(error.to_string());
    fs::create_dir_all(folder).map_err(io)?;
    let path = folder.join(format!("{id}.{EXTENSION}"));
    let partial = folder.join(format!(".{id}.{EXTENSION}.partial"));
    fs::write(&partial, json).map_err(io)?;
    fs::rename(&partial, &path).map_err(|error| {
        // The half-written neighbour is not a recipe; leave nothing behind.
        let _ = fs::remove_file(&partial);
        io(error)
    })?;
    Ok(path)
}

/// Save `recipe` in the studio folder as `effects/<id>.json`, over the file
/// already there. A recipe that would not read back, or whose id the app
/// ships, is refused and nothing is written. [`load_studio`] makes it one
/// of the studio's recipes.
pub fn write_studio_recipe(studio: &Path, recipe: &Recipe) -> Result<PathBuf, StudioError> {
    let json = recipe.to_json()?;
    if effect_taken(&recipe.id) {
        return Err(StudioError::Taken(recipe.id.clone()));
    }
    write_file(&studio.join(STUDIO_EFFECTS), &recipe.id, &json)
}

/// Save `recipe` in the studio folder as `transitions/<id>.json`, as
/// [`write_studio_recipe`] saves an effect.
pub fn write_studio_transition(
    studio: &Path,
    recipe: &TransitionRecipe,
) -> Result<PathBuf, StudioError> {
    let json = recipe.to_json()?;
    if transition_taken(&recipe.id) {
        return Err(StudioError::Taken(recipe.id.clone()));
    }
    write_file(&studio.join(STUDIO_TRANSITIONS), &recipe.id, &json)
}

fn remove_file(folder: &Path, id: &str) -> Result<bool, StudioError> {
    if !id_ok(id) {
        return Err(StudioError::Id);
    }
    match fs::remove_file(folder.join(format!("{id}.{EXTENSION}"))) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(StudioError::Io(error.to_string())),
    }
}

/// Delete the studio's effect file for `id`. False when there was none.
pub fn remove_studio_recipe(studio: &Path, id: &str) -> Result<bool, StudioError> {
    remove_file(&studio.join(STUDIO_EFFECTS), id)
}

/// Delete the studio's transition file for `id`. False when there was none.
pub fn remove_studio_transition(studio: &Path, id: &str) -> Result<bool, StudioError> {
    remove_file(&studio.join(STUDIO_TRANSITIONS), id)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_STUDIO_FILES, Refused, Registry, STUDIO_EFFECTS, STUDIO_TRANSITIONS, StudioError,
        read_studio, remove_studio_recipe, remove_studio_transition, studio_stamp,
        write_studio_recipe, write_studio_transition,
    };
    use crate::TransitionRecipe;
    use crate::recipe::{MAX_RECIPE_BYTES, Recipe, RecipeError};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp(label: &str) -> PathBuf {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "cutlass-effects-studio-{label}-{}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn effect(id: &str, radius: f32) -> Recipe {
        Recipe::read(
            format!(
                r#"{{"version":1,"id":"{id}","name":"Mine","passes":[{{"block":"blur","params":{{"radius":{radius}}}}}]}}"#
            )
            .as_bytes(),
        )
        .unwrap()
    }

    fn transition(id: &str, kind: u32) -> TransitionRecipe {
        TransitionRecipe::read(
            format!(r#"{{"version":1,"id":"{id}","name":"Mine","cross":{{"kind":{kind}}}}}"#)
                .as_bytes(),
        )
        .unwrap()
    }

    fn refusal(refused: &[Refused], file: &str) -> StudioError {
        refused
            .iter()
            .find(|refused| refused.file == file)
            .unwrap_or_else(|| panic!("{file} was not refused: {refused:?}"))
            .error
            .clone()
    }

    #[test]
    fn a_saved_recipe_and_transition_read_back_from_the_folder() {
        let studio = temp("round");
        let mine = effect("studio-mine", 0.02);
        let cut = transition("studio-cut", 2);
        let path = write_studio_recipe(&studio, &mine).unwrap();
        assert_eq!(path, studio.join(STUDIO_EFFECTS).join("studio-mine.json"));
        write_studio_transition(&studio, &cut).unwrap();
        let (effects, transitions, refused) = read_studio(&studio);
        assert_eq!(effects, vec![mine.clone()]);
        assert_eq!(transitions, vec![cut]);
        assert_eq!(refused, Vec::new());
        // Saving again replaces the file and leaves no half-written one.
        write_studio_recipe(&studio, &effect("studio-mine", 0.05)).unwrap();
        let (effects, _, _) = read_studio(&studio);
        assert_eq!(effects, vec![effect("studio-mine", 0.05)]);
        let files: Vec<_> = fs::read_dir(studio.join(STUDIO_EFFECTS))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(files.len(), 1, "{files:?}");
        // Removing is by id; a second removal finds nothing.
        assert_eq!(remove_studio_recipe(&studio, "studio-mine"), Ok(true));
        assert_eq!(remove_studio_recipe(&studio, "studio-mine"), Ok(false));
        assert_eq!(remove_studio_transition(&studio, "studio-cut"), Ok(true));
        assert_eq!(read_studio(&studio), (Vec::new(), Vec::new(), Vec::new()));
        let _ = fs::remove_dir_all(studio);
    }

    #[test]
    fn a_folder_that_is_not_there_holds_nothing() {
        let studio = temp("absent").join("nowhere");
        assert_eq!(read_studio(&studio), (Vec::new(), Vec::new(), Vec::new()));
        assert_eq!(studio_stamp(&studio), studio_stamp(&temp("empty")));
    }

    #[test]
    fn a_hostile_file_is_left_out_and_named() {
        let studio = temp("hostile");
        let effects = studio.join(STUDIO_EFFECTS);
        fs::create_dir_all(&effects).unwrap();
        let good = effect("studio-good", 0.02).to_json().unwrap();
        fs::write(effects.join("studio-good.json"), &good).unwrap();
        fs::write(effects.join("broken.json"), "{ not json").unwrap();
        fs::write(effects.join("other-name.json"), &good).unwrap();
        fs::write(effects.join("huge.json"), vec![b' '; MAX_RECIPE_BYTES + 1]).unwrap();
        // The app ships `shake`; a studio file may not stand in for it.
        let shake = effect("shake", 0.02).to_json().unwrap();
        fs::write(effects.join("shake.json"), shake).unwrap();
        fs::write(
            effects.join("too-far.json"),
            r#"{"version":1,"id":"too-far","name":"A","passes":[{"block":"blur","params":{"radius":9}}]}"#,
        )
        .unwrap();
        fs::write(
            effects.join("newer.json"),
            r#"{"version":99,"id":"newer","name":"A","passes":[{"block":"blur"}]}"#,
        )
        .unwrap();
        // A folder, a hidden file and a file of another kind are not recipes.
        fs::create_dir_all(effects.join("folder.json")).unwrap();
        fs::write(effects.join(".hidden.json"), &good).unwrap();
        fs::write(effects.join("notes.txt"), "hello").unwrap();

        let (kept, _, refused) = read_studio(&studio);
        assert_eq!(kept, vec![effect("studio-good", 0.02)]);
        assert!(matches!(
            refusal(&refused, "effects/broken.json"),
            StudioError::Recipe(RecipeError::Shape(_))
        ));
        assert_eq!(
            refusal(&refused, "effects/other-name.json"),
            StudioError::FileName
        );
        assert_eq!(
            refusal(&refused, "effects/huge.json"),
            StudioError::Recipe(RecipeError::TooLarge)
        );
        assert_eq!(
            refusal(&refused, "effects/shake.json"),
            StudioError::Taken("shake".to_owned())
        );
        assert!(matches!(
            refusal(&refused, "effects/too-far.json"),
            StudioError::Recipe(RecipeError::Range { .. })
        ));
        assert_eq!(
            refusal(&refused, "effects/newer.json"),
            StudioError::Recipe(RecipeError::Newer(99))
        );
        assert_eq!(
            refusal(&refused, "effects/folder.json"),
            StudioError::NotAFile
        );
        assert_eq!(refused.len(), 7, "{refused:?}");
        let _ = fs::remove_dir_all(studio);
    }

    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_folder_is_not_followed() {
        let studio = temp("link");
        let effects = studio.join(STUDIO_EFFECTS);
        fs::create_dir_all(&effects).unwrap();
        let outside = temp("outside").join("studio-out.json");
        fs::write(&outside, effect("studio-out", 0.02).to_json().unwrap()).unwrap();
        std::os::unix::fs::symlink(&outside, effects.join("studio-out.json")).unwrap();
        let (kept, _, refused) = read_studio(&studio);
        assert_eq!(kept, Vec::new());
        assert_eq!(
            refusal(&refused, "effects/studio-out.json"),
            StudioError::NotAFile
        );
        let _ = fs::remove_dir_all(studio);
    }

    #[test]
    fn a_built_in_or_bundled_id_and_a_bad_recipe_are_not_written() {
        let studio = temp("refuse");
        assert_eq!(
            write_studio_recipe(&studio, &effect("shake", 0.02)),
            Err(StudioError::Taken("shake".to_owned()))
        );
        for taken in ["dissolve", crate::bundled_transitions()[0].id.as_str()] {
            assert_eq!(
                write_studio_transition(&studio, &transition(taken, 0)),
                Err(StudioError::Taken(taken.to_owned()))
            );
        }
        let mut broken = effect("studio-broken", 0.02);
        broken.passes.clear();
        assert_eq!(
            write_studio_recipe(&studio, &broken),
            Err(StudioError::Recipe(RecipeError::Passes))
        );
        // An id that is not one cannot name a file outside the folder.
        broken.id = "../escape".to_owned();
        assert_eq!(
            write_studio_recipe(&studio, &broken),
            Err(StudioError::Recipe(RecipeError::Name))
        );
        assert_eq!(
            remove_studio_recipe(&studio, "../escape"),
            Err(StudioError::Id)
        );
        assert_eq!(remove_studio_transition(&studio, ""), Err(StudioError::Id));
        assert!(!studio.join(STUDIO_EFFECTS).exists());
        assert!(!studio.join(STUDIO_TRANSITIONS).exists());
        let _ = fs::remove_dir_all(studio);
    }

    #[test]
    fn files_past_the_limit_are_refused() {
        let studio = temp("many");
        let effects = studio.join(STUDIO_EFFECTS);
        fs::create_dir_all(&effects).unwrap();
        for n in 0..MAX_STUDIO_FILES + 2 {
            let id = format!("studio-{n:04}");
            fs::write(
                effects.join(format!("{id}.json")),
                effect(&id, 0.02).to_json().unwrap(),
            )
            .unwrap();
        }
        let (kept, _, refused) = read_studio(&studio);
        assert_eq!(kept.len(), MAX_STUDIO_FILES);
        assert_eq!(refused.len(), 2);
        assert!(
            refused
                .iter()
                .all(|file| file.error == StudioError::TooMany)
        );
        let _ = fs::remove_dir_all(studio);
    }

    #[test]
    fn the_stamp_moves_when_a_file_is_added_rewritten_or_removed() {
        let studio = temp("stamp");
        let empty = studio_stamp(&studio);
        write_studio_recipe(&studio, &effect("studio-a", 0.02)).unwrap();
        let one = studio_stamp(&studio);
        assert_ne!(one, empty);
        assert_eq!(studio_stamp(&studio), one, "nothing changed");
        // A longer file: the size differs even inside one clock tick.
        let mut longer = effect("studio-a", 0.02);
        longer.name = "A much longer name".to_owned();
        write_studio_recipe(&studio, &longer).unwrap();
        let rewritten = studio_stamp(&studio);
        assert_ne!(rewritten, one);
        write_studio_transition(&studio, &transition("studio-t", 0)).unwrap();
        let two = studio_stamp(&studio);
        assert_ne!(two, rewritten);
        remove_studio_recipe(&studio, "studio-a").unwrap();
        assert_ne!(studio_stamp(&studio), two);
        let _ = fs::remove_dir_all(studio);
    }

    #[test]
    fn a_registry_keeps_an_unchanged_recipe_and_a_transitions_place() {
        let mut registry = Registry::new();
        assert!(registry.install(
            vec![effect("studio-a", 0.02)],
            vec![transition("studio-t", 0), transition("studio-u", 1)],
        ));
        let first = registry.effects[0];
        // The same files again: nothing changed, and the recipe held stays.
        assert!(!registry.install(
            vec![effect("studio-a", 0.02)],
            vec![transition("studio-t", 0), transition("studio-u", 1)],
        ));
        assert!(std::ptr::eq(registry.effects[0], first));
        // A rewritten effect is a new recipe; the first transition's file
        // has gone, and the second keeps its place.
        assert!(registry.install(
            vec![effect("studio-a", 0.05)],
            vec![transition("studio-u", 1)],
        ));
        assert!(!std::ptr::eq(registry.effects[0], first));
        assert_eq!(registry.transitions[0], ("studio-t", None));
        assert_eq!(registry.transitions[1].0, "studio-u");
        assert!(registry.transitions[1].1.is_some());
        // It comes back in the place it had.
        assert!(registry.install(Vec::new(), vec![transition("studio-t", 3)]));
        assert!(registry.effects.is_empty());
        assert_eq!(registry.transitions.len(), 2);
        assert!(registry.transitions[0].1.is_some());
        assert_eq!(registry.transitions[1], ("studio-u", None));
    }
}
