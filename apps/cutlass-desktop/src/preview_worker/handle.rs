use super::*;

mod cache_maintenance;
mod project;

impl WorkerHandle {
    pub fn request_frame(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::Frame(tick));
    }

    /// Report the preview panel's on-screen size (physical px). The worker
    /// fits every subsequent render inside it and repaints the current frame.
    pub fn set_viewport(&self, width: u32, height: u32) {
        let _ = self.tx.send(WorkerMsg::Viewport { width, height });
    }

    /// Synchronous round-trip: clone of the live project as of every edit
    /// sent before this call. `None` only if the worker thread is gone.
    pub fn snapshot_project(&self) -> Option<Project> {
        self.snapshot_project_with_revision()
            .map(|(project, _)| project)
    }

    /// Like [`Self::snapshot_project`], plus the live engine revision at the
    /// same instant (for parked-plan divergence checks).
    pub fn snapshot_project_with_revision(&self) -> Option<(Project, u64)> {
        let (reply, rx) = bounded(1);
        self.tx.send(WorkerMsg::SnapshotProject { reply }).ok()?;
        rx.recv().ok()
    }

    /// Synchronous round-trip: monotonic live-engine revision. Cheap — no
    /// project clone. `None` only if the worker thread is gone.
    pub fn project_revision(&self) -> Option<u64> {
        let (reply, rx) = bounded(1);
        self.tx.send(WorkerMsg::ProjectRevision { reply }).ok()?;
        rx.recv().ok()
    }

    /// Synchronous round-trip: replay a rehearsed agent plan, one undo
    /// entry per phase. `expected_seed_revision` is checked against the live
    /// engine revision inside the worker immediately before replay. `None`
    /// only if the worker thread is gone.
    pub fn agent_apply_plan(
        &self,
        phases: Vec<Vec<AgentPlanStep>>,
        expected_seed_revision: u64,
    ) -> Option<Result<(), String>> {
        let (reply, rx) = bounded(1);
        self.tx
            .send(WorkerMsg::AgentApplyPlan {
                phases,
                expected_seed_revision,
                reply,
            })
            .ok()?;
        rx.recv().ok()
    }

    pub fn export(&self, request: ExportRequest) {
        let _ = self.tx.send(WorkerMsg::Export(request));
    }

    pub fn cancel_export(&self) {
        let _ = self.tx.send(WorkerMsg::CancelExport);
    }

    pub fn add_clip(
        &self,
        media: String,
        track: String,
        start_tick: i64,
        drop_row: i64,
        insert: bool,
    ) {
        let _ = self.tx.send(WorkerMsg::AddClip {
            media,
            track,
            start_tick,
            drop_row,
            insert,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_generated(
        &self,
        generator: Generator,
        track: String,
        start_tick: i64,
        duration_ticks: i64,
        drop_row: i64,
        effect: Option<String>,
        animations: Vec<(String, String)>,
    ) {
        let _ = self.tx.send(WorkerMsg::AddGenerated {
            generator,
            track,
            start_tick,
            duration_ticks,
            drop_row,
            effect,
            animations,
        });
    }

    pub fn move_clip(
        &self,
        clip: String,
        track: String,
        insert_row: i64,
        start_tick: i64,
        insert: bool,
    ) {
        let _ = self.tx.send(WorkerMsg::MoveClip {
            clip,
            track,
            insert_row,
            start_tick,
            insert,
        });
    }

    pub fn move_group(&self, moves: Vec<GroupMove>) {
        let _ = self.tx.send(WorkerMsg::MoveGroup { moves });
    }

    pub fn trim_clip(&self, clip: String, start_tick: i64, duration_ticks: i64) {
        let _ = self.tx.send(WorkerMsg::TrimClip {
            clip,
            start_tick,
            duration_ticks,
        });
    }

    pub fn remove_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::RemoveClips { clips });
    }

    pub fn ripple_delete_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::RippleDeleteClips { clips });
    }

    pub fn reverse_clip(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ReverseClip { clip });
    }

    pub fn extract_audio(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ExtractAudio { clip });
    }

    pub fn split_clip(&self, clip: String, at_tick: i64) {
        let _ = self.tx.send(WorkerMsg::SplitClip { clip, at_tick });
    }

    pub fn add_marker(&self, at_tick: i64, name: String, color: String) {
        let _ = self.tx.send(WorkerMsg::AddMarker {
            at_tick,
            name,
            color,
        });
    }

    pub fn remove_marker(&self, marker: String) {
        let _ = self.tx.send(WorkerMsg::RemoveMarker { marker });
    }

    pub fn set_marker(&self, marker: String, at_tick: i64, name: String, color: String) {
        let _ = self.tx.send(WorkerMsg::SetMarker {
            marker,
            at_tick,
            name,
            color,
        });
    }

    pub fn remove_track_manual(&self, track: String) {
        let _ = self.tx.send(WorkerMsg::RemoveTrackManual { track });
    }

    pub fn move_track_manual(&self, track: String, index: usize) {
        let _ = self.tx.send(WorkerMsg::MoveTrackManual { track, index });
    }

    pub fn set_track_name(&self, track: String, name: String) {
        let _ = self.tx.send(WorkerMsg::SetTrackName { track, name });
    }

    pub fn set_generator(&self, clip: String, generator: Generator) {
        let _ = self.tx.send(WorkerMsg::SetGenerator { clip, generator });
    }

    pub fn set_shape_size(&self, clip: String, width: f32, height: f32) {
        let _ = self.tx.send(WorkerMsg::SetShapeSize {
            clip,
            width,
            height,
        });
    }

    pub fn set_clip_speed(&self, clip: String, num: i32, den: i32, reversed: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipSpeed {
            clip,
            num,
            den,
            reversed,
        });
    }

    pub fn set_clip_pitch(&self, clip: String, preserve: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipPitch { clip, preserve });
    }

    pub fn set_denoise(&self, clip: String, denoise: bool) {
        let _ = self.tx.send(WorkerMsg::SetDenoise { clip, denoise });
    }

    /// Resolve a speed-ramp preset name (CapCut speed curves, M2) and dispatch
    /// the edit. `""` / `"none"` / `"normal"` clears the ramp; an unknown name
    /// is dropped with a warning so a stray UI string can't apply garbage.
    pub fn set_speed_curve(&self, clip: String, preset: String) {
        let curve = match preset.trim() {
            "" | "none" | "normal" => None,
            name => match cutlass_models::speed_preset(name) {
                Some(curve) => Some(curve),
                None => {
                    warn!(preset = name, "set-speed-curve ignored: unknown preset");
                    return;
                }
            },
        };
        let _ = self.tx.send(WorkerMsg::SetSpeedCurve { clip, curve });
    }

    pub fn set_speed_curve_point(&self, clip: String, index: i32, value: f32) {
        let Ok(index) = usize::try_from(index) else {
            warn!(index, "set-speed-curve-point ignored: negative index");
            return;
        };
        let _ = self
            .tx
            .send(WorkerMsg::SetSpeedCurvePoint { clip, index, value });
    }

    /// Set the flat volume level + fades (CapCut's basic slider): `volume` is
    /// `Some`, flattening any envelope.
    pub fn set_clip_audio(&self, clip: String, volume: f32, fade_in_s: f32, fade_out_s: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipAudio {
            clip,
            volume: Some(volume),
            fade_in_s,
            fade_out_s,
        });
    }

    /// Duck `clip` (a music clip) under the voice-tagged lanes (M8 Phase 4).
    pub fn duck_under_voice(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DuckUnderVoice { clip });
    }

    /// Detect beat markers on `clip` (CapCut "Beat", M8 Phase 6).
    pub fn detect_beats(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DetectBeats { clip });
    }

    /// Clear `clip`'s detected beat markers (M8 Phase 6).
    pub fn clear_beats(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ClearBeats { clip });
    }

    /// Set only the fades, preserving the clip's gain (constant or a
    /// keyframed M8 envelope) — `volume` lowers to `None`.
    pub fn set_clip_fades(&self, clip: String, fade_in_s: f32, fade_out_s: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipAudio {
            clip,
            volume: None,
            fade_in_s,
            fade_out_s,
        });
    }

    pub fn set_clip_crop(
        &self,
        clip: String,
        crop: CropRect,
        flip_h: bool,
        flip_v: bool,
        tick: i64,
    ) {
        let _ = self.tx.send(WorkerMsg::SetClipCrop {
            clip,
            crop,
            flip_h,
            flip_v,
            tick,
        });
    }

    pub fn set_blend_mode(&self, clip: String, mode: String) {
        let _ = self.tx.send(WorkerMsg::SetBlendMode { clip, mode });
    }

    pub fn set_motion_blur(&self, clip: String, motion_blur: MotionBlur) {
        let _ = self.tx.send(WorkerMsg::SetMotionBlur { clip, motion_blur });
    }

    /// Full styles replace. Inspector toggles use [`Self::toggle_layer_style`].
    #[allow(dead_code)] // Retained for tests / direct full-replace commits.
    pub fn set_layer_styles(&self, clip: String, styles: LayerStyles) {
        let _ = self.tx.send(WorkerMsg::SetLayerStyles { clip, styles });
    }

    pub fn toggle_layer_style(&self, clip: String, block: String, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::ToggleLayerStyle {
            clip,
            block,
            enabled,
        });
    }

    /// Full mask replace. Inspector kind/invert use the merge helpers below.
    #[allow(dead_code)] // Retained for tests / direct full-replace commits.
    pub fn set_mask(&self, clip: String, mask: Option<Mask>) {
        let _ = self.tx.send(WorkerMsg::SetMask { clip, mask });
    }

    pub fn set_mask_kind(&self, clip: String, kind: String) {
        let _ = self.tx.send(WorkerMsg::SetMaskKind { clip, kind });
    }

    pub fn set_mask_invert(&self, clip: String, invert: bool) {
        let _ = self.tx.send(WorkerMsg::SetMaskInvert { clip, invert });
    }

    pub fn set_chroma(&self, clip: String, chroma: Option<ChromaKey>) {
        let _ = self.tx.send(WorkerMsg::SetChroma { clip, chroma });
    }

    pub fn set_chroma_color(&self, clip: String, rgb: [u8; 3]) {
        let _ = self.tx.send(WorkerMsg::SetChromaColor { clip, rgb });
    }

    pub fn set_clip_filter(&self, clip: String, filter_id: String, intensity: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipFilter {
            clip,
            filter_id,
            intensity,
        });
    }

    pub fn set_clip_lut(&self, clip: String, path: String, intensity: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipLut {
            clip,
            path,
            intensity,
        });
    }

    pub fn set_clip_adjust(&self, clip: String, adjust: ColorAdjustments) {
        let _ = self.tx.send(WorkerMsg::SetClipAdjust { clip, adjust });
    }

    pub fn set_agent_rules(&self, rules: String) {
        let _ = self.tx.send(WorkerMsg::SetAgentRules { rules });
    }

    pub fn set_clip_animation(
        &self,
        clip: String,
        slot: String,
        animation_id: String,
        speed: f32,
        intensity: f32,
        stagger: f32,
    ) {
        let _ = self.tx.send(WorkerMsg::SetClipAnimation {
            clip,
            slot,
            animation_id,
            speed,
            intensity,
            stagger,
        });
    }

    pub fn preview_clip_look(
        &self,
        clip: String,
        filter_id: String,
        intensity: f32,
        adjust: ColorAdjustments,
        tick: i64,
    ) {
        let _ = self.tx.send(WorkerMsg::PreviewClipLook {
            clip,
            filter_id,
            intensity,
            adjust,
            tick,
        });
    }

    /// Live style-param preview: send only the delta; the worker merges it
    /// against the clip's committed styles (no UI-thread project snapshot).
    pub fn preview_clip_style_delta(
        &self,
        clip: String,
        key: String,
        value_x: f32,
        value_y: f32,
        tick: i64,
    ) {
        let _ = self.tx.send(WorkerMsg::PreviewClipStyleDelta {
            clip,
            key,
            value_x,
            value_y,
            tick,
        });
    }

    pub fn clear_styles_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearStylesOverride { tick });
    }

    pub fn add_effect(&self, clip: String, effect_id: String) {
        let _ = self.tx.send(WorkerMsg::AddEffect { clip, effect_id });
    }

    pub fn remove_effect(&self, clip: String, index: u32) {
        let _ = self.tx.send(WorkerMsg::RemoveEffect { clip, index });
    }

    pub fn set_effect_param(&self, clip: String, index: u32, param: String, value: f32) {
        self.set_effect_param_value(clip, index, param, ParamValue::Scalar(value));
    }

    pub fn set_effect_param_value(
        &self,
        clip: String,
        index: u32,
        param: String,
        value: ParamValue,
    ) {
        let _ = self.tx.send(WorkerMsg::SetEffectParam {
            clip,
            index,
            param,
            value,
        });
    }

    pub fn add_transition(&self, clip: String, transition_id: String) {
        let _ = self.tx.send(WorkerMsg::AddTransition {
            clip,
            transition_id,
        });
    }

    pub fn remove_transition(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::RemoveTransition { clip });
    }

    pub fn set_transition(&self, clip: String, duration: i64) {
        let _ = self.tx.send(WorkerMsg::SetTransition { clip, duration });
    }

    pub fn set_canvas(&self, aspect_index: i32, background: [u8; 3]) {
        let _ = self.tx.send(WorkerMsg::SetCanvas {
            aspect_index,
            background,
        });
    }

    pub fn fit_clip(&self, clip: String, fill: bool, tick: i64) {
        let _ = self.tx.send(WorkerMsg::FitClip { clip, fill, tick });
    }

    pub fn set_param_keyframe(
        &self,
        clip: String,
        param: ClipParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
        tangents: Option<cutlass_models::SpatialTangents>,
    ) {
        let _ = self.tx.send(WorkerMsg::SetParamKeyframe {
            clip,
            param,
            tick,
            value,
            easing,
            tangents,
        });
    }

    pub fn set_param_keyframe_tangents(
        &self,
        clip: String,
        tick: i64,
        tangents: Option<cutlass_models::SpatialTangents>,
    ) {
        let _ = self.tx.send(WorkerMsg::SetParamKeyframeTangents {
            clip,
            tick,
            tangents,
        });
    }

    pub fn set_param_constant(&self, clip: String, param: ClipParam, value: ParamValue) {
        let _ = self
            .tx
            .send(WorkerMsg::SetParamConstant { clip, param, value });
    }

    pub fn remove_param_keyframe(&self, clip: String, param: ClipParam, tick: i64) {
        let _ = self
            .tx
            .send(WorkerMsg::RemoveParamKeyframe { clip, param, tick });
    }

    /// Graph-editor tick move: remove+set in one undoable history group.
    pub fn move_param_keyframe(&self, req: MoveParamKeyframeRequest) {
        let _ = self.tx.send(WorkerMsg::MoveParamKeyframe {
            clip: req.clip,
            param: req.param,
            from_tick: req.from_tick,
            to_tick: req.to_tick,
            value: req.value,
            easing: req.easing,
            tangents: req.tangents,
        });
    }

    /// Graph-editor piecewise easing preset (bounce / elastic / back).
    pub fn apply_easing_preset(
        &self,
        clip: String,
        param: ClipParam,
        tick: i64,
        preset: cutlass_models::PiecewiseEasingPreset,
    ) {
        let _ = self.tx.send(WorkerMsg::ApplyEasingPreset {
            clip,
            param,
            tick,
            preset,
        });
    }

    pub fn retime_keyframes(&self, clip: String, from_tick: i64, to_tick: i64) {
        let _ = self.tx.send(WorkerMsg::RetimeKeyframes {
            clip,
            from_tick,
            to_tick,
        });
    }

    pub fn remove_keyframes_at(&self, clip: String, tick: i64) {
        let _ = self.tx.send(WorkerMsg::RemoveKeyframesAt { clip, tick });
    }

    pub fn set_transform(&self, clip: String, transform: ClipTransform, tick: i64) {
        let _ = self.tx.send(WorkerMsg::SetTransform {
            clip,
            transform,
            tick,
        });
    }

    pub fn transform_override(&self, clip: String, transform: ClipTransform, tick: i64) {
        let _ = self.tx.send(WorkerMsg::TransformOverride {
            clip,
            transform,
            tick,
        });
    }

    pub fn begin_transform_gesture(&self, clip: String, tick: i64) {
        let _ = self
            .tx
            .send(WorkerMsg::BeginTransformGesture { clip, tick });
    }

    pub fn end_transform_gesture(&self) {
        let _ = self.tx.send(WorkerMsg::EndTransformGesture);
    }

    pub fn clear_transform_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearTransformOverride { tick });
    }

    pub fn generator_override(&self, clip: String, generator: Generator, tick: i64) {
        let _ = self.tx.send(WorkerMsg::GeneratorOverride {
            clip,
            generator,
            tick,
        });
    }

    pub fn clear_generator_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearGeneratorOverride { tick });
    }

    pub fn preview_shape_size(&self, clip: String, width: f32, height: f32, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PreviewShapeSize {
            clip,
            width,
            height,
            tick,
        });
    }

    pub fn undo(&self) {
        let _ = self.tx.send(WorkerMsg::Undo);
    }

    pub fn redo(&self) {
        let _ = self.tx.send(WorkerMsg::Redo);
    }

    pub fn copy_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::CopyClips { clips });
    }

    pub fn paste_at(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PasteAt { tick });
    }

    pub fn duplicate_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::DuplicateClips { clips });
    }

    pub fn unlink_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::UnlinkClips { clips });
    }

    pub fn set_main_magnet(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetMainMagnet(enabled));
    }

    pub fn set_linkage(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetLinkage(enabled));
    }

    pub fn set_track_flag(&self, track: String, flag: TrackFlag, value: bool) {
        let _ = self.tx.send(WorkerMsg::SetTrackFlag { track, flag, value });
    }
}
