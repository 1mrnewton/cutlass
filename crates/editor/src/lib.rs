//! Timeline geometry. A gesture becomes one session request.
//!
//! A lane is the row drawn for a track. The draft has no lane record.
//! Zoom is a pixel size. Snap reports a time. This crate does not rewrite tracks.

#![forbid(unsafe_code)]

mod act;
mod animation;
mod append;
mod caption_list;
mod caption_replace;
mod caption_templates;
mod caption_words;
mod catalog;
mod clip_rules;
mod clipboard;
mod controls;
mod effect_shelf;
mod filmstrip;
mod fling;
mod freeze;
mod gesture;
mod group_move;
mod hit;
mod inspector;
mod keys;
mod lane;
mod library;
mod media_place;
mod project_choices;
mod reorder;
mod ripple;
mod ruler;
mod saved_look;
mod shape_shelf;
mod snap;
mod stage;
mod subtitles;
mod template;
mod text_catalog;
mod text_effect;
mod title;
mod tools;
mod transition_shelf;
mod transport;
mod view_zoom;
mod zoom;

pub use act::{can_cut_at, clip_under, delete_side, frame_margin};
pub use animation::{ClipAnimation, ClipAnimations, clip_animations};
pub use append::{Append, AppendLane, append_placement, free_row, material_span};
pub use caption_list::{
    CaptionListMiss, CaptionRow, NEW_CAPTION_GAP, NEW_CAPTION_LENGTH, NEW_CAPTION_TEXT, Paragraph,
    add_after_requests, caption_paragraphs, caption_rows, caption_text_request, fill_caption_faces,
    for_every_caption, main_caption_lane, paragraph_request, paragraphs, rematched,
    remove_caption_request, row_at, rows_matching, words_over,
};
pub use caption_replace::{Replaced, replace_ignoring_case, replace_in_captions};
pub use caption_templates::{
    CAPTION_TEMPLATES, CaptionTemplate, TEMPLATE_CATEGORIES, caption_template, caption_template_of,
    caption_template_requests, plain_caption_requests,
};
pub use caption_words::{LINE_BREAK, breaks_before, join_words, retype_word, toggle_line_break};
pub use catalog::{catalog_entries, catalog_search, is_bundled};
pub use clip_rules::{can_turn, magnet_request};
pub use clipboard::{ClipboardMiss, CopiedClip, copy_clips, paste_requests};
pub use controls::{MAX_VOLUME, SPEED_MARKS, speed_at, speed_position, volume_at, volume_position};
pub use effect_shelf::{
    ADJUSTMENT_CARD, EFFECT_CARD_PREFIX, EFFECT_LENGTH, EffectCard, EffectPlace,
    FILTER_CARD_PREFIX, effect_cards, effect_of_card, effect_requests, effect_row_for,
    effect_target, filter_cards, filter_of_card,
};
pub use filmstrip::{FilmTile, MAX_TILES, film_tiles};
pub use fling::{
    FLING_DECAY, FLING_LIMIT, FLING_REST, FLING_START, FlingStep, fling_start, fling_step,
};
pub use freeze::{FreezeImage, freeze_requests};
pub use gesture::{
    DropGesture, DropOutcome, GestureError, GestureOutcome, MoveGesture, Placement, TrimGesture,
    TrimPlan, drop_gesture, move_gesture, placement, trim_gesture,
};
pub use group_move::{ClipMove, MovePlan, NewLane, move_requests, plan_move, plan_move_between};
pub use hit::{BadScale, Hit, Pointer, TimelineScale, hit_test};
pub use inspector::{
    BACKGROUND_ALPHA_START, BACKGROUND_ROUND_START, GLOW_RANGE_START, GLOW_STRENGTH_START,
    LetterCase, OUTLINE_START, Panel, SHADOW_ANGLE, SHADOW_BLUR_START, SHADOW_DISTANCE_START,
    SILENT_DB, TEXT_PRESETS, TextLook, TextPreset, canvas_pixels, gain_of_db, is_caption, panel_of,
    preset_requests, starting, stored_position, text_look, volume_db,
};
pub use keys::{ClipKeys, KeyGraph, clip_keys, key_graphs, move_key, value_keys};
pub use lane::{Lane, lane_category, lanes, lanes_for};
pub use library::{
    LIBRARY_TOP, LibraryFile, LibraryShelf, LibrarySort, NEW_FOLDER_NAME, add_folder,
    delete_folder, folder_trail, hide_file, library_files, library_folders, next_folder_name,
    parent_of, place, rename_folder,
};
pub use media_place::{MediaPlace, extract_request, media_requests, sound_row};
pub use project_choices::{
    COLOR_SPACES, PROJECT_RATES, PROJECT_RATIOS, PROJECT_RESOLUTIONS, ProjectSettings,
    color_space_name, is_drop_rate, next_canvas, next_color_space, next_rate, project_canvas,
    project_requests, project_settings, rate_fps, rate_label,
};
pub use reorder::{place_of, reorder_start};
pub use ripple::{close_gap, gap_at, ripple_delete};
pub use ruler::{RulerStep, Tick, clock, ruler_step, ruler_ticks, zoom_ceiling};
pub use saved_look::{SavedLook, look_requests, read_looks, saved_look, write_looks};
pub use shape_shelf::{SHAPE_CARD_PREFIX, shape_cards, shape_of_card};
pub use snap::{SnapKind, SnapTarget, snap, snap_span, transition_middle};
pub use stage::{
    MAX_SCALE, MIN_SCALE, NoText, Pose, SCALE_REACH, SNAP_REACH, StageAlign, StageBox,
    StageGesture, StageGuides, StageStep, TURN_REACH, TextSizes, canvas_size, keyed_pose,
    stage_align, stage_box, stage_gesture, stage_pick,
};
pub use subtitles::{SubtitleFormat, SubtitleMiss, subtitle_request};
pub use template::{
    TEMPLATE_CARD_PREFIX, TITLE_TEMPLATES, TemplatePlace, TitleTemplate, Widget, template_requests,
    title_template,
};
pub use text_catalog::{
    BUBBLE_CATEGORIES, BUBBLE_LOOKS, BubbleLook, CatalogTile, TILE_SIDE, bubble_look_of,
    bubble_look_requests, catalog_shows, tile_session,
};
pub use text_effect::{
    EFFECT_CATEGORIES, NO_TEXT_EFFECT, TEXT_EFFECTS, TextEffect, text_effect_of,
    text_effect_requests,
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
pub use view_zoom::{
    ACTION_SAFE, PREVIEW_ZOOM_RANGE, PREVIEW_ZOOMS, TITLE_SAFE, clamp_preview_pan, preview_zoom_at,
    preview_zoom_step, safe_box,
};
pub use zoom::{
    Zoom, ZoomBounds, fit_pixel_size, last_accuracy_step, minimum_pixel_size, quarter_lane,
};
