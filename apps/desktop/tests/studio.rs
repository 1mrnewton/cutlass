//! The studio saving to its folder: one test, in a process of its own,
//! because the studio's recipes are shared by everything in it.

use std::fs;

use cutlass::studio::{CurveKind, Doc, Slot, Stack, Studio};
use cutlass_effects::{Block, Ease, TransitionKind};

#[test]
fn a_document_is_saved_reopened_renamed_reloaded_and_deleted() {
    let folder =
        std::env::temp_dir().join(format!("cutlass-desktop-studio-{}", std::process::id()));
    let _ = fs::remove_dir_all(&folder);
    let effects = folder.join("effects");
    let mut studio = Studio::new(Some(folder.clone()));
    assert!(studio.reload(), "the first look reads the folder");
    assert!(!studio.reload(), "nothing changed since");

    // A new effect is saved under its id and the app can run it.
    studio.new_effect();
    studio.set_name("Soft Punch");
    let slot = Slot::pass(Stack::Effect, 0, "radius");
    studio.set_kind(&slot, CurveKind::Keys);
    studio.move_key(&slot, 1, 1.0, 0.04);
    studio.set_ease(&slot, 0, Ease::Out);
    assert!(studio.is_dirty());
    studio.save().expect("saved");
    assert!(!studio.is_dirty() && studio.is_saved());
    assert!(effects.join("soft-punch.json").is_file());
    let held = cutlass_effects::recipe("recipe/soft-punch").expect("the app runs it");
    assert_eq!(held.version, 2, "an eased key needs format 2");
    assert!(cutlass_effects::is_studio_recipe("recipe/soft-punch"));
    let listed = Studio::effects();
    assert_eq!(listed[0].id, "soft-punch");
    assert!(listed[0].studio, "the studio's come first");

    // Opened again it is itself, not a copy, and saving keeps the one file.
    studio.close();
    assert!(studio.open_effect("soft-punch"));
    assert_eq!(studio.doc().map(Doc::id), Some("soft-punch"));
    assert!(studio.is_saved() && !studio.is_dirty());
    studio.add_pass(Stack::Effect, Block::Glow);
    studio.save().expect("saved again");
    assert_eq!(
        cutlass_effects::recipe("recipe/soft-punch").map(|recipe| recipe.passes.len()),
        Some(2)
    );
    assert_eq!(fs::read_dir(&effects).unwrap().count(), 1);

    // A new id moves the file: the old id no longer names an effect.
    studio.set_id("hard-punch");
    studio.save().expect("renamed");
    assert!(!effects.join("soft-punch.json").exists());
    assert!(effects.join("hard-punch.json").is_file());
    assert_eq!(cutlass_effects::recipe("recipe/soft-punch"), None);
    assert!(cutlass_effects::recipe("recipe/hard-punch").is_some());

    // A second document may not take an id in use, bundled or the studio's.
    let mut other = Studio::new(Some(folder.clone()));
    other.new_effect();
    other.set_id("hard-punch");
    assert!(other.problem().is_some());
    assert!(other.save().is_err());
    other.set_id("shake");
    assert!(other.save().is_err());

    // A file rewritten by hand is the document, when nothing is unsaved.
    let path = effects.join("hard-punch.json");
    let by_hand = fs::read_to_string(&path)
        .unwrap()
        .replace("\"Soft Punch\"", "\"Renamed by hand\"");
    fs::write(&path, by_hand).unwrap();
    assert!(studio.reload());
    assert_eq!(studio.doc().map(Doc::name), Some("Renamed by hand"));
    assert!(!studio.is_dirty());
    // With an unsaved change the document keeps it.
    studio.set_name("Mine again");
    fs::write(
        &path,
        fs::read_to_string(&path)
            .unwrap()
            .replace("Renamed by hand", "Theirs"),
    )
    .unwrap();
    assert!(studio.reload());
    assert_eq!(studio.doc().map(Doc::name), Some("Mine again"));
    assert_eq!(
        cutlass_effects::recipe("recipe/hard-punch").map(|recipe| recipe.name.as_str()),
        Some("Theirs"),
        "the app runs the file"
    );

    // A broken file is left out and named.
    fs::write(effects.join("broken.json"), "{").unwrap();
    assert!(studio.reload());
    assert_eq!(studio.refused().len(), 1);
    assert_eq!(studio.refused()[0].file, "effects/broken.json");
    fs::remove_file(effects.join("broken.json")).unwrap();

    // A transition goes the same way, into its own folder.
    let mut cut = Studio::new(Some(folder.clone()));
    cut.new_transition();
    cut.set_name("Soft Cut");
    cut.set_fixed(&Slot::cross("soft"), 0.3);
    cut.save().expect("saved");
    assert!(folder.join("transitions").join("soft-cut.json").is_file());
    assert!(
        TransitionKind::every()
            .into_iter()
            .any(|kind| kind.key() == "soft-cut")
    );
    assert!(cut.open_transition("soft-cut"));
    assert!(cut.is_saved());
    assert_eq!(Studio::transitions()[0].id, "soft-cut");
    cut.delete().expect("deleted");
    assert!(cut.doc().is_none());
    assert!(
        !TransitionKind::every()
            .into_iter()
            .any(|kind| kind.key() == "soft-cut")
    );

    // Deleting removes the file and the effect.
    studio.delete().expect("deleted");
    assert!(!path.exists());
    assert_eq!(cutlass_effects::recipe("recipe/hard-punch"), None);
    assert!(studio.doc().is_none());

    cutlass_effects::clear_studio();
    let _ = fs::remove_dir_all(folder);
}
