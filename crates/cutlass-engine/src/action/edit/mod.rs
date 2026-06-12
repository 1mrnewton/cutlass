//! Undoable timeline edits as inverse [`EditAction`](crate::action::EditAction) pairs.

pub mod add_clip;
pub mod add_generated;
pub mod add_track;
pub mod insert_clip;
pub mod insert_media;
pub mod link_clips;
pub mod move_clip;
pub mod remove_clip;
pub mod remove_media;
pub mod set_generator;
pub mod set_param;
pub mod set_track_flags;
pub mod set_transform;
pub mod restore_clip;
pub mod ripple_delete;
pub mod ripple_insert;
pub mod shift_clips;
pub mod split_clip;
pub mod trim_clip;
