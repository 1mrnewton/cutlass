//! Frames of small hand-written animations, checked pixel by pixel.

use cutlass_lottie::{Animation, rasterize};

const SQUARE: &str = include_str!("fixtures/square.json");
const TRIM: &str = include_str!("fixtures/trim.json");
const MASKS: &str = include_str!("fixtures/masks.json");
const MATTE: &str = include_str!("fixtures/matte.json");
const PRECOMP: &str = include_str!("fixtures/precomp.json");
const STICKER: &str = include_str!("fixtures/sticker.json");

#[cfg(test)]
fn parse(text: &str) -> Animation {
    Animation::parse(text.as_bytes()).expect("the fixture parses")
}

/// The frame drawn at the animation's own size.
#[cfg(test)]
fn frame(text: &str, at: f64) -> (Vec<u8>, u32) {
    let animation = parse(text);
    let width = animation.width() as u32;
    let height = animation.height() as u32;
    let scene = animation.scene(at, width, height).expect("the frame draws");
    (rasterize(&scene), width)
}

fn pixel(frame: &(Vec<u8>, u32), x: u32, y: u32) -> [u8; 4] {
    let at = (y * frame.1 + x) as usize * 4;
    [
        frame.0[at],
        frame.0[at + 1],
        frame.0[at + 2],
        frame.0[at + 3],
    ]
}

/// `text` with `from` replaced once; the fixture must hold it.
fn with(text: &str, from: &str, to: &str) -> String {
    assert!(text.contains(from), "{from} is not in the fixture");
    text.replacen(from, to, 1)
}

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const CLEAR: [u8; 4] = [0, 0, 0, 0];

#[test]
fn the_file_reports_its_size_rate_and_length() {
    let animation = parse(SQUARE);
    assert_eq!((animation.width(), animation.height()), (100.0, 100.0));
    assert_eq!(animation.frame_rate(), 30.0);
    assert_eq!((animation.in_point(), animation.out_point()), (0.0, 30.0));
    assert!((animation.duration() - 1.0).abs() < 1e-12);
}

#[test]
fn the_square_moves_along_its_position_keys() {
    let start = frame(SQUARE, 0.0);
    assert_eq!(pixel(&start, 30, 50), RED);
    assert_eq!(pixel(&start, 20, 40), RED);
    assert_eq!(pixel(&start, 19, 50), CLEAR);
    assert_eq!(pixel(&start, 50, 50), CLEAR);
    let middle = frame(SQUARE, 10.0);
    assert_eq!(pixel(&middle, 50, 50), RED);
    assert_eq!(pixel(&middle, 30, 50), CLEAR);
    // Held at the last key after it.
    let late = frame(SQUARE, 25.0);
    assert_eq!(pixel(&late, 70, 50), RED);
    assert_eq!(pixel(&late, 50, 50), CLEAR);
}

#[test]
fn a_smaller_output_scales_the_whole_frame() {
    let animation = parse(SQUARE);
    let scene = animation.scene(0.0, 50, 50).unwrap();
    let small = (rasterize(&scene), 50);
    assert_eq!(pixel(&small, 15, 25), RED);
    assert_eq!(pixel(&small, 25, 25), CLEAR);
    assert!(animation.scene(0.0, 0, 50).is_err());
}

#[test]
fn a_trim_reveals_each_path_at_once_or_both_in_turn() {
    let none = frame(TRIM, 0.0);
    assert!(none.0.iter().all(|byte| *byte == 0));
    let half = frame(TRIM, 5.0);
    assert_eq!(pixel(&half, 30, 30), BLUE);
    assert_eq!(pixel(&half, 70, 30), CLEAR);
    assert_eq!(pixel(&half, 30, 70), BLUE);
    assert_eq!(pixel(&half, 70, 70), CLEAR);
    let full = frame(TRIM, 10.0);
    assert_eq!(pixel(&full, 85, 30), BLUE);
    assert_eq!(pixel(&full, 85, 70), BLUE);
    // In turn: half the run is the whole first line and none of the second.
    let in_turn = with(TRIM, r#""m":1"#, r#""m":2"#);
    let half = frame(&in_turn, 5.0);
    assert_eq!(pixel(&half, 70, 30), BLUE);
    assert_eq!(pixel(&half, 30, 70), CLEAR);
    let three_quarters = frame(&in_turn, 7.5);
    assert_eq!(pixel(&three_quarters, 30, 70), BLUE);
    assert_eq!(pixel(&three_quarters, 70, 70), CLEAR);
}

#[test]
fn a_trim_offset_turns_the_kept_part_round() {
    let offset = with(TRIM, r#""o":{"a":0,"k":0}"#, r#""o":{"a":0,"k":180}"#);
    let half = frame(&offset, 5.0);
    assert_eq!(pixel(&half, 30, 30), CLEAR);
    assert_eq!(pixel(&half, 70, 30), BLUE);
}

#[test]
fn masks_add_and_subtract() {
    let drawn = frame(MASKS, 0.0);
    assert_eq!(pixel(&drawn, 5, 50), RED);
    assert_eq!(pixel(&drawn, 45, 90), RED);
    assert_eq!(pixel(&drawn, 20, 20), CLEAR);
    assert_eq!(pixel(&drawn, 75, 50), CLEAR);
}

#[test]
fn an_inverted_mask_keeps_the_outside() {
    let inverted = with(MASKS, r#""inv":false"#, r#""inv":true"#);
    let drawn = frame(&inverted, 0.0);
    assert_eq!(pixel(&drawn, 5, 50), CLEAR);
    assert_eq!(pixel(&drawn, 75, 50), RED);
    assert_eq!(pixel(&drawn, 20, 20), CLEAR);
}

#[test]
fn an_intersecting_mask_keeps_only_the_overlap() {
    let intersect = with(MASKS, r#""mode":"s""#, r#""mode":"i""#);
    let drawn = frame(&intersect, 0.0);
    assert_eq!(pixel(&drawn, 20, 20), RED);
    assert_eq!(pixel(&drawn, 5, 50), CLEAR);
    assert_eq!(pixel(&drawn, 75, 50), CLEAR);
}

#[test]
fn mask_opacity_fades_and_expansion_grows_it() {
    let faded = with(MASKS, r#""o":{"a":0,"k":100}"#, r#""o":{"a":0,"k":50}"#);
    let drawn = frame(&faded, 0.0);
    assert_eq!(pixel(&drawn, 5, 50), [128, 0, 0, 128]);
    let grown = with(MASKS, r#""x":{"a":0,"k":0}"#, r#""x":{"a":0,"k":10}"#);
    let drawn = frame(&grown, 0.0);
    assert_eq!(pixel(&drawn, 55, 50), RED);
    assert_eq!(pixel(&drawn, 65, 50), CLEAR);
    let shrunk = with(MASKS, r#""x":{"a":0,"k":0}"#, r#""x":{"a":0,"k":-10}"#);
    let drawn = frame(&shrunk, 0.0);
    assert_eq!(pixel(&drawn, 45, 50), CLEAR);
    assert_eq!(pixel(&drawn, 35, 50), RED);
}

#[test]
fn an_alpha_matte_shows_the_layer_inside_the_matte_only() {
    let drawn = frame(MATTE, 0.0);
    assert_eq!(pixel(&drawn, 25, 50), BLUE, "the matte itself is not drawn");
    assert_eq!(pixel(&drawn, 75, 50), CLEAR);
    let inverted = frame(&with(MATTE, r#""tt":1"#, r#""tt":2"#), 0.0);
    assert_eq!(pixel(&inverted, 25, 50), CLEAR);
    assert_eq!(pixel(&inverted, 75, 50), BLUE);
    let luma = frame(&with(MATTE, r#""tt":1"#, r#""tt":3"#), 0.0);
    assert_eq!(pixel(&luma, 25, 50), BLUE);
    assert_eq!(pixel(&luma, 75, 50), CLEAR);
    let luma_inverted = frame(&with(MATTE, r#""tt":1"#, r#""tt":4"#), 0.0);
    assert_eq!(pixel(&luma_inverted, 25, 50), CLEAR);
    assert_eq!(pixel(&luma_inverted, 75, 50), BLUE);
    // A black matte has no luma: the layer does not show through it.
    let dark = with(
        MATTE,
        r#""c":{"a":0,"k":[1,1,1,1]}"#,
        r#""c":{"a":0,"k":[0,0,0,1]}"#,
    );
    let luma_dark = frame(&with(&dark, r#""tt":1"#, r#""tt":3"#), 0.0);
    assert_eq!(pixel(&luma_dark, 25, 50), CLEAR);
}

#[test]
fn a_matte_named_by_index_is_read_from_anywhere() {
    // The matte moves below the layer it limits and is named by `tp`.
    let animation = with(MATTE, r#""tt":1,"#, r#""tt":1,"tp":1,"#);
    let mut value: serde_json::Value = serde_json::from_str(&animation).unwrap();
    let layers = value["layers"].as_array_mut().unwrap();
    layers.swap(0, 1);
    let drawn = frame(&value.to_string(), 0.0);
    assert_eq!(pixel(&drawn, 25, 50), BLUE);
    assert_eq!(pixel(&drawn, 75, 50), CLEAR);
}

#[test]
fn a_precomposition_runs_at_its_start_and_stretch() {
    // The inner clock is (t - 10) / 2; the runner crosses 80 pixels in 40.
    let at_start = frame(PRECOMP, 10.0);
    assert_eq!(pixel(&at_start, 10, 50), GREEN);
    let middle = frame(PRECOMP, 50.0);
    assert_eq!(pixel(&middle, 50, 50), GREEN);
    assert_eq!(pixel(&middle, 10, 50), CLEAR);
    let late = frame(PRECOMP, 90.0);
    assert_eq!(pixel(&late, 90, 50), GREEN);
    // Before its in point the layer draws nothing.
    let early = frame(PRECOMP, 2.0);
    assert!(early.0.iter().all(|byte| *byte == 0));
    let after = frame(PRECOMP, 96.0);
    assert!(after.0.iter().all(|byte| *byte == 0));
}

#[test]
fn a_time_remap_picks_the_inner_frame() {
    // Four seconds of remap over 100 frames: at local frame 50, 2 s, which
    // is inner frame 20.
    let remapped = with(
        PRECOMP,
        r#""sr":2,"#,
        r#""sr":2,"tm":{"a":1,"k":[{"t":0,"s":[0]},{"t":100,"s":[4]}]},"#,
    );
    let drawn = frame(&remapped, 60.0);
    assert_eq!(pixel(&drawn, 50, 50), GREEN);
    assert_eq!(pixel(&drawn, 30, 50), CLEAR);
}

#[test]
fn a_precomposition_shows_only_inside_its_bounds() {
    let narrow = with(
        PRECOMP,
        r#""w":100,"h":100,"ip":5"#,
        r#""w":60,"h":100,"ip":5"#,
    );
    assert_eq!(pixel(&frame(&narrow, 50.0), 50, 50), GREEN);
    let late = frame(&narrow, 90.0);
    assert!(late.0.iter().all(|byte| *byte == 0));
}

fn layer_with(ks: &str, shapes: &str) -> String {
    format!(
        r#"{{"w":100,"h":100,"fr":10,"ip":0,"op":10,"layers":[
            {{"ty":4,"ind":2,"parent":1,"ip":0,"op":10,"ks":{ks},"shapes":{shapes}}},
            {{"ty":3,"ind":1,"ip":0,"op":10,"ks":{{"p":{{"k":[50,50]}},"r":{{"k":90}}}}}}
        ]}}"#
    )
}

const SMALL_RED_SQUARE: &str = r#"[
    {"ty":"rc","p":{"k":[0,0]},"s":{"k":[6,6]}},
    {"ty":"fl","c":{"k":[1,0,0,1]},"o":{"k":100}}
]"#;

#[test]
fn a_child_follows_its_parent_turn_and_place() {
    // The child sits 20 right of its parent; the parent turns a quarter
    // clockwise, so the child lands 20 below it.
    let drawn = frame(&layer_with(r#"{"p":{"k":[20,0]}}"#, SMALL_RED_SQUARE), 0.0);
    assert_eq!(pixel(&drawn, 50, 70), RED);
    assert_eq!(pixel(&drawn, 70, 50), CLEAR);
}

#[test]
fn layer_opacity_fades_the_layer_but_not_its_children() {
    let drawn = frame(
        &layer_with(r#"{"p":{"k":[20,0]},"o":{"k":50}}"#, SMALL_RED_SQUARE),
        0.0,
    );
    assert_eq!(pixel(&drawn, 50, 70), [128, 0, 0, 128]);
    let hidden = frame(&layer_with(r#"{"o":{"k":0}}"#, SMALL_RED_SQUARE), 0.0);
    assert!(hidden.0.iter().all(|byte| *byte == 0));
}

fn shapes_frame(shapes: &str) -> (Vec<u8>, u32) {
    let text = format!(
        r#"{{"w":100,"h":100,"fr":10,"ip":0,"op":10,"layers":[
            {{"ty":4,"ind":1,"ip":0,"op":10,"ks":{{}},"shapes":{shapes}}}]}}"#
    );
    frame(&text, 0.0)
}

#[test]
fn a_linear_gradient_ramps_across_the_rectangle() {
    let drawn = shapes_frame(
        r#"[{"ty":"rc","p":{"k":[50,50]},"s":{"k":[100,100]}},
            {"ty":"gf","t":1,"o":{"k":100},"s":{"k":[0,0]},"e":{"k":[100,0]},
             "g":{"p":2,"k":{"k":[0,0,0,0, 1,1,1,1]}}}]"#,
    );
    assert_eq!(pixel(&drawn, 0, 50), [1, 1, 1, 255]);
    let middle = pixel(&drawn, 50, 50);
    assert!((128..=130).contains(&middle[0]), "{middle:?}");
    assert_eq!(pixel(&drawn, 99, 10), [254, 254, 254, 255]);
}

#[test]
fn gradient_opacity_stops_fade_the_ramp() {
    let drawn = shapes_frame(
        r#"[{"ty":"rc","p":{"k":[50,50]},"s":{"k":[100,100]}},
            {"ty":"gf","t":1,"o":{"k":100},"s":{"k":[0,0]},"e":{"k":[100,0]},
             "g":{"p":2,"k":{"k":[0,1,0,0, 1,1,0,0, 0,1, 1,0]}}}]"#,
    );
    assert_eq!(pixel(&drawn, 0, 50)[3], 254);
    assert!(pixel(&drawn, 99, 50)[3] <= 2);
}

#[test]
fn a_radial_gradient_grows_from_its_center() {
    let drawn = shapes_frame(
        r#"[{"ty":"rc","p":{"k":[50,50]},"s":{"k":[100,100]}},
            {"ty":"gf","t":2,"o":{"k":100},"s":{"k":[50,50]},"e":{"k":[100,50]},
             "g":{"p":2,"k":{"k":[0,1,0,0, 1,0,0,1]}}}]"#,
    );
    let center = pixel(&drawn, 50, 50);
    assert!(center[0] > 250 && center[2] < 5, "{center:?}");
    let edge = pixel(&drawn, 99, 50);
    assert!(edge[2] > 245, "{edge:?}");
    let up = pixel(&drawn, 50, 25);
    assert!((125..=131).contains(&up[0]), "{up:?}");
}

#[test]
fn a_dashed_stroke_leaves_gaps() {
    let drawn = shapes_frame(
        r#"[{"ty":"sh","ks":{"k":{"c":false,"v":[[0,50],[100,50]],"i":[[0,0],[0,0]],"o":[[0,0],[0,0]]}}},
            {"ty":"st","c":{"k":[0,0,1,1]},"o":{"k":100},"w":{"k":4},"lc":1,"lj":1,
             "d":[{"n":"d","v":{"k":10}},{"n":"g","v":{"k":10}}]}]"#,
    );
    assert_eq!(pixel(&drawn, 5, 50), BLUE);
    assert_eq!(pixel(&drawn, 15, 50), CLEAR);
    assert_eq!(pixel(&drawn, 25, 50), BLUE);
    assert_eq!(pixel(&drawn, 5, 53), CLEAR);
}

#[test]
fn caps_reach_past_the_ends_and_joins_fill_the_corner() {
    let line = |cap: u8, join: u8| {
        shapes_frame(&format!(
            r#"[{{"ty":"sh","ks":{{"k":{{"c":false,"v":[[20,20],[80,20],[80,80]],"i":[[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0]]}}}}}},
                {{"ty":"st","c":{{"k":[0,0,1,1]}},"o":{{"k":100}},"w":{{"k":10}},"lc":{cap},"lj":{join}}}]"#
        ))
    };
    let butt = line(1, 3);
    assert_eq!(pixel(&butt, 17, 20), CLEAR);
    let square = line(3, 3);
    assert_eq!(pixel(&square, 17, 20), BLUE);
    assert_eq!(pixel(&square, 15, 16), BLUE);
    let round = line(2, 3);
    assert_eq!(pixel(&round, 17, 20), BLUE);
    assert_eq!(pixel(&round, 15, 16), CLEAR);
    // The outer corner: a miter reaches it, a bevel cuts it off.
    let miter = line(1, 1);
    assert_eq!(pixel(&miter, 84, 15), BLUE);
    let bevel = line(1, 3);
    assert_eq!(pixel(&bevel, 84, 15), CLEAR);
    let round_join = line(1, 2);
    assert_eq!(pixel(&round_join, 84, 15), CLEAR);
    assert_eq!(pixel(&round_join, 82, 17), BLUE);
}

#[test]
fn a_group_transform_and_opacity_apply_to_its_paints() {
    let drawn = shapes_frame(
        r#"[{"ty":"gr","it":[
              {"ty":"rc","p":{"k":[0,0]},"s":{"k":[10,10]}},
              {"ty":"fl","c":{"k":[1,0,0,1]},"o":{"k":100}},
              {"ty":"tr","p":{"k":[50,50]},"s":{"k":[200,100]},"o":{"k":50}}]}]"#,
    );
    assert_eq!(pixel(&drawn, 41, 50), [128, 0, 0, 128]);
    assert_eq!(pixel(&drawn, 50, 56), CLEAR);
}

#[test]
fn the_first_shape_item_is_drawn_on_top() {
    let drawn = shapes_frame(
        r#"[{"ty":"gr","it":[{"ty":"rc","p":{"k":[50,50]},"s":{"k":[20,20]}},
                             {"ty":"fl","c":{"k":[0,1,0,1]},"o":{"k":100}}]},
            {"ty":"gr","it":[{"ty":"rc","p":{"k":[50,50]},"s":{"k":[40,40]}},
                             {"ty":"fl","c":{"k":[1,0,0,1]},"o":{"k":100}}]}]"#,
    );
    assert_eq!(pixel(&drawn, 50, 50), GREEN);
    assert_eq!(pixel(&drawn, 35, 50), RED);
}

#[test]
fn a_paint_draws_the_paths_inside_groups_above_it() {
    let drawn = shapes_frame(
        r#"[{"ty":"gr","it":[{"ty":"el","p":{"k":[30,50]},"s":{"k":[20,20]}}]},
            {"ty":"rc","p":{"k":[70,50]},"s":{"k":[20,20]}},
            {"ty":"fl","c":{"k":[1,0,0,1]},"o":{"k":100}},
            {"ty":"el","p":{"k":[50,80]},"s":{"k":[10,10]}}]"#,
    );
    assert_eq!(pixel(&drawn, 30, 50), RED);
    assert_eq!(pixel(&drawn, 70, 50), RED);
    assert_eq!(
        pixel(&drawn, 50, 80),
        CLEAR,
        "a path below the paint is not drawn"
    );
}

#[test]
fn the_repeater_spaces_and_fades_its_copies() {
    let drawn = frame(STICKER, 0.0);
    let first = pixel(&drawn, 24, 112);
    let last = pixel(&drawn, 104, 112);
    assert!((200..=206).contains(&first[3]), "{first:?}");
    assert!((58..=64).contains(&last[3]), "{last:?}");
    assert_eq!(pixel(&drawn, 34, 112), CLEAR);
}

#[test]
fn a_star_and_a_polygon_draw_their_points() {
    let drawn = shapes_frame(
        r#"[{"ty":"sr","sy":1,"pt":{"k":5},"p":{"k":[50,50]},"r":{"k":0},"or":{"k":40},"ir":{"k":15}},
            {"ty":"fl","c":{"k":[1,0,0,1]},"o":{"k":100}}]"#,
    );
    assert_eq!(pixel(&drawn, 50, 13), RED, "the top point");
    assert_eq!(pixel(&drawn, 50, 50), RED);
    assert_eq!(pixel(&drawn, 30, 20), CLEAR, "between two points");
    let hexagon = shapes_frame(
        r#"[{"ty":"sr","sy":2,"pt":{"k":6},"p":{"k":[50,50]},"r":{"k":0},"or":{"k":40}},
            {"ty":"fl","c":{"k":[1,0,0,1]},"o":{"k":100}}]"#,
    );
    assert_eq!(pixel(&hexagon, 50, 13), RED);
    assert_eq!(pixel(&hexagon, 50, 88), RED);
    assert_eq!(pixel(&hexagon, 86, 50), CLEAR, "a flat side, not a point");
}

#[test]
fn a_hidden_layer_and_a_hidden_shape_draw_nothing() {
    let hidden = with(SQUARE, r#""ind":1,"#, r#""ind":1,"hd":true,"#);
    assert!(frame(&hidden, 0.0).0.iter().all(|byte| *byte == 0));
    let hidden_shape = with(SQUARE, r#"{"ty":"fl","#, r#"{"ty":"fl","hd":true,"#);
    assert!(frame(&hidden_shape, 0.0).0.iter().all(|byte| *byte == 0));
}

#[test]
fn the_sticker_frame_stays_within_its_edge_budget() {
    // A hot path bound: a 128 pixel sticker with a star, dashes, five dots
    // and a band is a few thousand edges, not tens of thousands.
    let animation = parse(STICKER);
    for at in [0.0, 15.0, 30.0, 59.0] {
        let scene = animation.scene(at, 128, 128).unwrap();
        let edges = scene.edge_count();
        assert!((200..6_000).contains(&edges), "frame {at}: {edges} edges");
    }
    // Twice the size cuts curves finer, but not four times as many edges.
    let small = animation.scene(0.0, 128, 128).unwrap().edge_count();
    let large = animation.scene(0.0, 256, 256).unwrap().edge_count();
    assert!(large < small * 2, "{small} then {large}");
}
