//! An effect card's picture: the effect over its scene, a still on the
//! card and playing while the pointer rests on it. A transition card's
//! picture is the same, from one scene to another.

use cutlass_effects::{PreviewScene, preview_scene};
use cutlass_model::Micros;

use super::Shell;

/// The recipe and scene an effect card shows.
fn card_effect(card_id: &str) -> Option<(&'static cutlass_effects::Recipe, &'static PreviewScene)> {
    let effect_id = cutlass_editor::effect_of_card(card_id)?;
    let recipe = cutlass_effects::recipe(&effect_id)?;
    Some((recipe, preview_scene(recipe.preview_scene())?))
}

impl Shell {
    /// The still on an effect card, as straight RGBA, and its size. Without
    /// a GPU it is the scene alone.
    #[must_use]
    pub fn effect_still(&mut self, card_id: &str) -> Option<(Vec<u8>, [u32; 2])> {
        let (recipe, scene) = card_effect(card_id)?;
        let size = [scene.width, scene.height];
        let drawn = self
            .gpu
            .as_mut()
            .and_then(|gpu| gpu.effect_still(recipe, scene).ok());
        Some((drawn.unwrap_or_else(|| scene.rgba.clone()), size))
    }

    /// The effect card playing for `elapsed`, drawn on the GPU and shown in
    /// egui as is; none without a GPU.
    pub fn effect_frame(
        &mut self,
        card_id: &str,
        elapsed: Micros,
    ) -> Option<(egui::TextureId, [u32; 2])> {
        let (recipe, scene) = card_effect(card_id)?;
        let time = cutlass_effects::preview_time(elapsed);
        self.gpu.as_mut()?.effect_frame(recipe, scene, time).ok()
    }
}

/// The transition and the two scenes a transition card plays between.
fn card_transition(
    card_id: &str,
) -> Option<(
    cutlass_effects::TransitionKind,
    &'static PreviewScene,
    &'static PreviewScene,
)> {
    let kind = cutlass_editor::transition_of_card(card_id)?.kind;
    let (from, to) = cutlass_effects::transition_scenes(kind);
    Some((kind, preview_scene(from)?, preview_scene(to)?))
}

impl Shell {
    /// The still on a transition card, halfway between its scenes, as
    /// straight RGBA, and its size. Without a GPU it is the first scene.
    #[must_use]
    pub fn transition_still(&mut self, card_id: &str) -> Option<(Vec<u8>, [u32; 2])> {
        let (kind, from, to) = card_transition(card_id)?;
        let size = [from.width, from.height];
        let drawn = self
            .gpu
            .as_mut()
            .and_then(|gpu| gpu.transition_still(kind, from, to).ok());
        Some((drawn.unwrap_or_else(|| from.rgba.clone()), size))
    }

    /// The transition card playing for `elapsed`, a loop of the first scene
    /// held, the transition and the second held; none without a GPU.
    pub fn transition_frame(
        &mut self,
        card_id: &str,
        elapsed: Micros,
    ) -> Option<(egui::TextureId, [u32; 2])> {
        let (kind, from, to) = card_transition(card_id)?;
        let t = cutlass_effects::transition_preview_t(elapsed);
        self.gpu.as_mut()?.transition_frame(kind, from, to, t).ok()
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::Micros;

    use crate::shell::tests::shell;

    #[test]
    fn an_effect_card_shows_its_effect_over_its_scene() {
        let mut desktop = shell();
        let card = format!("{}recipe/black-white", cutlass_editor::EFFECT_CARD_PREFIX);
        // Without a GPU the card shows the scene alone, and nothing plays.
        let (plain, size) = desktop.effect_still(&card).unwrap();
        assert_eq!(size, [288, 192]);
        assert!(desktop.effect_frame(&card, Micros(0)).is_none());
        assert!(desktop.effect_still("effect:recipe/none").is_none());

        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(state) = pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) else {
            eprintln!("skipped: no GPU");
            return;
        };
        desktop.attach_gpu(&state);
        let (grey, _) = desktop.effect_still(&card).unwrap();
        assert_ne!(grey, plain);
        assert!(
            grey.chunks(4)
                .all(|pixel| pixel[0].abs_diff(pixel[1]) <= 2 && pixel[1].abs_diff(pixel[2]) <= 2),
            "black and white has no colour left"
        );
        let (_, frame) = desktop.effect_frame(&card, Micros(400_000)).unwrap();
        assert_eq!(frame, [288, 192]);
    }

    #[test]
    fn a_transition_card_shows_its_scenes_crossing() {
        let mut desktop = shell();
        let card = super::super::transition_offers()
            .iter()
            .find(|offer| offer.id == "circle-open")
            .unwrap()
            .card_id();
        let (plain, size) = desktop.transition_still(&card).unwrap();
        assert_eq!(size, [288, 192]);
        assert!(desktop.transition_frame(&card, Micros(0)).is_none());
        assert!(desktop.transition_still("transition:none").is_none());

        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(state) = pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) else {
            eprintln!("skipped: no GPU");
            return;
        };
        desktop.attach_gpu(&state);
        let (half, _) = desktop.transition_still(&card).unwrap();
        // Halfway, the middle shows the second scene and the corner the first.
        let at = |x: usize, y: usize| &half[(y * 288 + x) * 4..(y * 288 + x) * 4 + 3];
        let was = |x: usize, y: usize| &plain[(y * 288 + x) * 4..(y * 288 + x) * 4 + 3];
        assert_ne!(at(144, 96), was(144, 96));
        assert_eq!(at(1, 1), was(1, 1));
        let (_, frame) = desktop.transition_frame(&card, Micros(800_000)).unwrap();
        assert_eq!(frame, [288, 192]);
    }
}
