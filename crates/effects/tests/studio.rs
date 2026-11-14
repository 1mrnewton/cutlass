//! The studio folder as the app sees it: one test, in a process of its
//! own, because the studio's recipes are shared by everything in it.

// The helpers below are test code outside a test function.
#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;

use cutlass_effects::{
    Recipe, TransitionKind, TransitionRecipe, bundled, bundled_recipe, bundled_transitions,
    clear_studio, is_studio_recipe, is_studio_transition, load_studio, recipe, recipes,
    remove_studio_recipe, remove_studio_transition, studio_generation, studio_recipes,
    studio_transitions, write_studio_recipe, write_studio_transition,
};
use cutlass_model::MaterialTransition;

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

fn kind_of(effect_id: &str) -> Option<TransitionKind> {
    TransitionKind::of(&MaterialTransition {
        effect_id: effect_id.to_owned(),
        ..MaterialTransition::default()
    })
}

#[test]
fn the_studio_folder_runs_beside_what_the_app_ships() {
    let studio: PathBuf =
        std::env::temp_dir().join(format!("cutlass-effects-studio-{}", std::process::id()));
    let _ = fs::remove_dir_all(&studio);

    // Nothing in the folder: only what the app ships.
    let start = studio_generation();
    let loaded = load_studio(&studio);
    assert_eq!((loaded.effects, loaded.transitions), (0, 0));
    assert_eq!(studio_generation(), start, "nothing changed");
    assert_eq!(recipes().len(), bundled().len());
    assert_eq!(recipe("recipe/studio-mine"), None);
    let built_in = TransitionKind::every().len();
    assert_eq!(
        built_in,
        TransitionKind::ALL.len() + bundled_transitions().len()
    );

    // A saved effect and transition are found the way bundled ones are.
    write_studio_recipe(&studio, &effect("studio-mine", 0.02)).unwrap();
    write_studio_transition(&studio, &transition("studio-cut", 2)).unwrap();
    write_studio_transition(&studio, &transition("studio-fade", 0)).unwrap();
    let loaded = load_studio(&studio);
    assert_eq!((loaded.effects, loaded.transitions), (1, 2));
    assert_eq!(loaded.refused, Vec::new());
    let one = studio_generation();
    assert_ne!(one, start);
    let mine = recipe("recipe/studio-mine").expect("a studio recipe");
    assert_eq!(mine, &effect("studio-mine", 0.02));
    assert!(is_studio_recipe("recipe/studio-mine"));
    assert!(!is_studio_recipe("recipe/shake"));
    assert_eq!(recipe("studio-mine"), None, "an effect id has its prefix");
    assert_eq!(recipes().len(), bundled().len() + 1);
    assert_eq!(recipes().last().copied(), Some(mine));
    assert_eq!(studio_recipes(), vec![mine]);
    // A bundled recipe is still the bundled one.
    assert!(std::ptr::eq(
        recipe("recipe/shake").unwrap(),
        bundled_recipe("recipe/shake").unwrap()
    ));

    let cut = kind_of("cutlass-studio-cut").expect("a studio transition");
    assert_eq!(cut.key(), "studio-cut");
    assert_eq!(cut.label(), "Mine");
    assert_eq!(cut.effect_id(), "cutlass-studio-cut");
    assert_eq!(cut.recipe(), Some(&transition("studio-cut", 2)));
    assert!(is_studio_transition("cutlass-studio-cut"));
    assert!(!is_studio_transition("cutlass-dissolve"));
    assert_eq!(TransitionKind::every().len(), built_in + 2);
    assert_eq!(studio_transitions().len(), 2);

    // Reading the same folder again changes nothing.
    load_studio(&studio);
    assert_eq!(studio_generation(), one);
    assert!(std::ptr::eq(recipe("recipe/studio-mine").unwrap(), mine));

    // A rewritten file is the recipe from then on.
    write_studio_recipe(&studio, &effect("studio-mine", 0.05)).unwrap();
    load_studio(&studio);
    assert_ne!(studio_generation(), one);
    assert_eq!(
        recipe("recipe/studio-mine"),
        Some(&effect("studio-mine", 0.05))
    );

    // A transition whose file has gone is no longer drawn, and the other
    // one is still the kind it was.
    let fade = kind_of("cutlass-studio-fade").unwrap();
    remove_studio_transition(&studio, "studio-cut").unwrap();
    load_studio(&studio);
    assert_eq!(kind_of("cutlass-studio-cut"), None);
    assert_eq!(cut.recipe(), None);
    assert_eq!(cut.key(), "");
    assert_eq!(kind_of("cutlass-studio-fade"), Some(fade));
    assert_eq!(TransitionKind::every().len(), built_in + 1);

    // A removed effect goes too, and clearing drops the rest.
    remove_studio_recipe(&studio, "studio-mine").unwrap();
    load_studio(&studio);
    assert_eq!(recipe("recipe/studio-mine"), None);
    assert_eq!(recipes().len(), bundled().len());
    clear_studio();
    assert_eq!(TransitionKind::every().len(), built_in);
    assert_eq!(studio_transitions(), Vec::<&TransitionRecipe>::new());

    let _ = fs::remove_dir_all(studio);
}
