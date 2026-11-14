//! The shelves with a studio folder loaded: one test, in a process of its
//! own, because the studio's recipes are shared by everything in it.

use cutlass_editor::{
    EFFECT_CARD_PREFIX, effect_cards, effect_of_card, transition_of_card, transition_offers,
};
use cutlass_effects::{
    Recipe, TransitionRecipe, bundled, clear_studio, load_studio, write_studio_recipe,
    write_studio_transition,
};

#[test]
fn the_shelves_list_the_studios_effects_and_transitions_after_the_bundled_ones() {
    let studio = std::env::temp_dir().join(format!("cutlass-editor-studio-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&studio);
    let shipped_effects = effect_cards().len();
    let shipped_transitions = transition_offers().len();
    assert_eq!(shipped_effects, bundled().len());
    assert_eq!(effect_of_card("effect:recipe/studio-soft"), None);
    assert_eq!(transition_of_card("transition:studio-cut"), None);

    let effect = Recipe::read(
        br#"{"version":1,"id":"studio-soft","name":"Soft","category":"Blur","passes":[{"block":"blur","params":{"radius":0.02}}]}"#,
    )
    .expect("an effect");
    let transition = TransitionRecipe::read(
        br#"{"version":1,"id":"studio-cut","name":"Cut","category":"Wipe","duration":0.8,"cross":{"kind":2}}"#,
    )
    .expect("a transition");
    write_studio_recipe(&studio, &effect).expect("saved");
    write_studio_transition(&studio, &transition).expect("saved");
    load_studio(&studio);

    let cards = effect_cards();
    assert_eq!(cards.len(), shipped_effects + 1);
    let card = cards.last().expect("the studio card");
    assert_eq!(card.id, format!("{EFFECT_CARD_PREFIX}recipe/studio-soft"));
    assert_eq!(
        (card.name.as_str(), card.category.as_str()),
        ("Soft", "Blur")
    );
    assert_eq!(
        effect_of_card(&card.id),
        Some("recipe/studio-soft".to_owned())
    );

    let offers = transition_offers();
    assert_eq!(offers.len(), shipped_transitions + 1);
    let offer = transition_of_card("transition:studio-cut").expect("the studio transition");
    assert_eq!(offers.last().copied(), Some(offer));
    assert_eq!((offer.name, offer.category), ("Cut", "Wipe"));
    assert_eq!(offer.duration, 800_000);
    assert_eq!(offer.kind.recipe(), Some(&transition));

    // With the folder turned off the shelves are what the app ships again.
    clear_studio();
    assert_eq!(effect_cards().len(), shipped_effects);
    assert_eq!(transition_offers().len(), shipped_transitions);
    assert_eq!(transition_of_card("transition:studio-cut"), None);
    let _ = std::fs::remove_dir_all(studio);
}
