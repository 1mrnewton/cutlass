//! Undoable timeline edits as inverse [`EditAction`](crate::action::EditAction) pairs.

pub mod add_clip;
pub mod add_generated;
pub mod add_track;
// `detect_beats` and `duck` need the decoder's audio reader/DSP — deferred on
// mobile-support (FFmpeg-free); their command arms return `Unsupported`.
pub mod insert_clip;
pub mod insert_media;
pub mod link_clips;
pub mod marker;
pub mod move_clip;
pub mod remove_clip;
pub mod remove_media;
pub mod restore_clip;
pub mod restore_transitions;
pub mod ripple_delete;
pub mod ripple_insert;
pub mod set_audio;
pub mod set_canvas;
pub mod set_clip_media;
pub mod set_crop;
pub mod set_denoise;
pub mod set_effect;
pub mod set_generator;
pub mod set_param;
pub mod set_pitch;
pub mod set_speed;
pub mod set_speed_curve;
pub mod set_track_flags;
pub mod set_transform;
pub mod set_transition;
pub mod shift_clips;
pub mod split_clip;
pub mod template_marks;
pub mod trim_clip;
