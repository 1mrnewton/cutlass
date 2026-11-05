//! Timeline geometry. A gesture becomes one session request.
//!
//! A lane is the row drawn for a track. The draft has no lane record.
//! Zoom is a pixel size. Snap reports a time. This crate does not rewrite tracks.

#![forbid(unsafe_code)]

mod act;
mod animation;
mod append;
mod clip_rules;
mod clipboard;
mod controls;
mod effect_shelf;
mod filmstrip;
mod fling;
mod gesture;
mod group_move;
mod hit;
mod keys;
mod lane;
mod library;
mod media_place;
mod project_choices;
mod reorder;
mod ripple;
mod ruler;
mod snap;
mod stage;
mod title;
mod tools;
mod transition_shelf;
mod transport;
mod view_zoom;
mod zoom;

pub use act::{can_cut_at, clip_under, delete_side, frame_margin};
pub use animation::{ClipAnimation, ClipAnimations, clip_animations};
pub use append::{Append, AppendLane, append_placement, free_row, material_span};
pub use clip_rules::{can_turn, magnet_request};
pub use clipboard::{ClipboardMiss, CopiedClip, copy_clips, paste_requests};
pub use controls::{MAX_VOLUME, SPEED_MARKS, speed_at, speed_position, volume_at, volume_position};
pub use effect_shelf::{
    EFFECT_CARD_PREFIX, EFFECT_LENGTH, EffectCard, EffectPlace, effect_cards, effect_of_card,
    effect_requests, effect_row_for, effect_target,
};
pub use filmstrip::{FilmTile, MAX_TILES, film_tiles};
pub use fling::{
    FLING_DECAY, FLING_LIMIT, FLING_REST, FLING_START, FlingStep, fling_start, fling_step,
};
pub use gesture::{
    DropGesture, DropOutcome, GestureError, GestureOutcome, MoveGesture, Placement, TrimGesture,
    TrimPlan, drop_gesture, move_gesture, placement, trim_gesture,
};
pub use group_move::{ClipMove, MovePlan, move_requests, plan_move};
pub use hit::{BadScale, Hit, Pointer, TimelineScale, hit_test};
pub use keys::{ClipKeys, KeyGraph, clip_keys, key_graphs, move_key};
pub use lane::{Lane, lane_category, lanes, lanes_for};
pub use library::{
    LIBRARY_TOP, LibraryFile, LibraryShelf, LibrarySort, NEW_FOLDER_NAME, add_folder,
    delete_folder, folder_trail, hide_file, library_files, library_folders, next_folder_name,
    parent_of, place, rename_folder,
};
pub use media_place::{MediaPlace, extract_request, media_requests, sound_row};
pub use project_choices::{
    COLOR_SPACES, PROJECT_RATES, color_space_name, is_drop_rate, next_canvas, next_color_space,
    next_rate, rate_fps, rate_label,
};
pub use reorder::{place_of, reorder_start};
pub use ripple::{close_gap, gap_at, ripple_delete};
pub use ruler::{RulerStep, Tick, clock, ruler_step, ruler_ticks, zoom_ceiling};
pub use snap::{SnapKind, SnapTarget, snap, snap_guides, transition_middle};
pub use stage::{
    MAX_SCALE, MIN_SCALE, NoText, Pose, SCALE_REACH, SNAP_REACH, StageAlign, StageBox,
    StageGesture, StageGuides, StageStep, TURN_REACH, TextSizes, canvas_size, keyed_pose,
    stage_align, stage_box, stage_gesture, stage_pick,
};
pub use title::{
    TITLE_CARD_PREFIX, TITLE_LENGTH, TITLE_STYLES, TitlePlace, TitleStyle, title_requests,
    title_style,
};
pub use tools::{Bar, Tool, ToolId, clip_bar, tools};
pub use transition_shelf::{
    MainCut, TRANSITION_OFFERS, TransitionMiss, TransitionOffer, fit_transition, main_cut,
    transition_of_card, transition_request,
};
pub use transport::{Marks, SHUTTLE_TOP, Shuttle, shuttle_rate};
pub use view_zoom::{ACTION_SAFE, PREVIEW_ZOOMS, TITLE_SAFE, preview_zoom_step, safe_box};
pub use zoom::{
    Zoom, ZoomBounds, fit_pixel_size, last_accuracy_step, minimum_pixel_size, quarter_lane,
};
