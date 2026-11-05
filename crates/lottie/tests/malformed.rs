//! Hostile and broken files fail closed: an error, never a panic, and no
//! frame past the limits.

use cutlass_lottie::limits::{MAX_BYTES, MAX_DEPTH, MAX_LAYERS, MAX_PATH_POINTS};
use cutlass_lottie::{Animation, LottieError, rasterize};

const STICKER: &str = include_str!("fixtures/sticker.json");

fn header(layers: &str) -> String {
    format!(r#"{{"w":64,"h":64,"fr":30,"ip":0,"op":30,"layers":{layers}}}"#)
}

#[test]
fn bytes_that_are_not_an_animation_are_refused() {
    for bad in [
        &b""[..],
        b"not json",
        b"[1, 2, 3]",
        b"{\"w\": 10}",
        b"{\"w\": 10, \"h\": 10, \"fr\": 30, \"ip\": 0}",
        b"{\"w\": 0, \"h\": 10, \"fr\": 30, \"ip\": 0, \"op\": 10}",
        b"{\"w\": 1e9, \"h\": 10, \"fr\": 30, \"ip\": 0, \"op\": 10}",
        b"{\"w\": 10, \"h\": 10, \"fr\": 0, \"ip\": 0, \"op\": 10}",
        b"{\"w\": 10, \"h\": 10, \"fr\": 30, \"ip\": 10, \"op\": 10}",
        b"{\"w\": \"10\", \"h\": 10, \"fr\": 30, \"ip\": 0, \"op\": 10}",
        b"\xff\xfe\x00",
    ] {
        assert!(
            Animation::parse(bad).is_err(),
            "{:?}",
            String::from_utf8_lossy(bad)
        );
    }
}

#[test]
fn every_cut_of_a_good_file_fails_or_draws_without_a_panic() {
    let bytes = STICKER.as_bytes();
    for end in 0..bytes.len() {
        if let Ok(animation) = Animation::parse(&bytes[..end]) {
            let _ = animation.scene(10.0, 32, 32).map(|scene| rasterize(&scene));
        }
    }
}

#[test]
fn wrong_types_everywhere_are_skipped_not_trusted() {
    let text = header(
        r##"[
        "a string",
        {"ty": 4, "ks": "nope", "shapes": {"not": "a list"}},
        {"ty": 4, "ip": 0, "op": 30, "ks": {"p": {"k": "x"}, "s": {"a": 1, "k": [{"t": "0"}]}},
         "shapes": [
            {"ty": "gr", "it": "nope"},
            {"ty": "sh", "ks": {"k": {"v": "nope"}}},
            {"ty": "sh", "ks": {"k": {"c": true, "v": [[1], [2, 3, 4], "x"], "i": [], "o": null}}},
            {"ty": "rc", "p": {"k": [1e308, -1e308]}, "s": {"k": [1e308, 1e308]}},
            {"ty": "el", "s": {"k": []}},
            {"ty": "sr", "pt": {"k": -5}, "or": {"k": 1e300}},
            {"ty": "fl", "c": {"k": "red"}},
            {"ty": "st", "w": {"k": -3}, "d": [{"n": "d", "v": {"k": 0}}, {"n": "x"}]},
            {"ty": "gf", "g": {"p": 1e9, "k": {"k": [0, 1]}}},
            {"ty": "tm", "s": {"k": "a"}, "e": {"k": 1e308}, "o": {"k": -1e308}},
            {"ty": "rp", "c": {"k": 1e12}, "tr": 7},
            {"ty": "fl", "c": {"k": [300, 2, 0.5]}, "o": {"k": 1e9}}
         ]},
        {"ty": 0, "refId": "missing"},
        {"ty": 1, "sc": "#zzzzzz", "sw": -10, "sh": 1e308, "ip": 0, "op": 30},
        {"ty": 2, "refId": "image"},
        {"ty": 5, "t": {}}
    ]"##,
    );
    let animation = Animation::parse(text.as_bytes()).expect("odd values are skipped");
    for at in [-1e308, -1.0, 0.0, 15.0, 1e308, f64::NAN] {
        match animation.scene(at, 64, 64) {
            Ok(scene) => {
                let _ = rasterize(&scene);
            }
            Err(error) => assert!(matches!(error, LottieError::Limit(_)), "{error}"),
        }
    }
}

#[test]
fn a_file_past_the_byte_limit_is_refused_unread() {
    let big = vec![b' '; MAX_BYTES + 1];
    assert_eq!(Animation::parse(&big), Err(LottieError::Limit("file size")));
}

#[test]
fn too_many_layers_are_refused() {
    let layer = r#"{"ty":3,"ip":0,"op":30,"ks":{}}"#;
    let layers = vec![layer; MAX_LAYERS + 1].join(",");
    let text = header(&format!("[{layers}]"));
    assert_eq!(
        Animation::parse(text.as_bytes()),
        Err(LottieError::Limit("layers"))
    );
}

#[test]
fn groups_nested_too_deep_are_refused() {
    let mut shapes = r#"{"ty":"rc","p":{"k":[0,0]},"s":{"k":[4,4]}}"#.to_owned();
    for _ in 0..=MAX_DEPTH + 1 {
        shapes = format!(r#"{{"ty":"gr","it":[{shapes}]}}"#);
    }
    let text = header(&format!(
        r#"[{{"ty":4,"ip":0,"op":30,"ks":{{}},"shapes":[{shapes}]}}]"#
    ));
    assert_eq!(
        Animation::parse(text.as_bytes()),
        Err(LottieError::Limit("nesting depth"))
    );
}

#[test]
fn json_nested_past_the_parser_limit_is_refused() {
    let deep = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
    let text = header(&deep);
    assert!(matches!(
        Animation::parse(text.as_bytes()),
        Err(LottieError::Json(_))
    ));
}

#[test]
fn a_path_with_too_many_points_is_refused() {
    let points = vec!["[0,0]"; MAX_PATH_POINTS + 1].join(",");
    let text = header(&format!(
        r#"[{{"ty":4,"ip":0,"op":30,"ks":{{}},"shapes":[{{"ty":"sh","ks":{{"k":{{"c":true,"v":[{points}]}}}}}}]}}]"#
    ));
    assert_eq!(
        Animation::parse(text.as_bytes()),
        Err(LottieError::Limit("points in a path"))
    );
}

#[test]
fn a_precomposition_that_holds_itself_stops_at_the_depth_limit() {
    let text = r#"{"w":64,"h":64,"fr":30,"ip":0,"op":30,
        "assets":[{"id":"loop","layers":[
            {"ty":0,"refId":"loop","ip":0,"op":30,"ks":{}},
            {"ty":4,"ip":0,"op":30,"ks":{},"shapes":[
                {"ty":"rc","p":{"k":[8,8]},"s":{"k":[4,4]}},{"ty":"fl","c":{"k":[1,0,0,1]}}]}]}],
        "layers":[{"ty":0,"refId":"loop","ip":0,"op":30,"ks":{}}]}"#;
    let animation = Animation::parse(text.as_bytes()).unwrap();
    assert_eq!(
        animation.scene(0.0, 64, 64),
        Err(LottieError::Limit("nesting depth"))
    );
}

#[test]
fn precompositions_that_fan_out_stop_at_the_frame_layer_limit() {
    // Ten layers of the next comp in each of eight comps: 10^8 instances.
    let mut assets = Vec::new();
    for level in 0..8 {
        let layer = if level == 7 {
            r#"{"ty":4,"ip":0,"op":30,"ks":{},"shapes":[{"ty":"rc","p":{"k":[8,8]},"s":{"k":[4,4]}},{"ty":"fl","c":{"k":[1,0,0,1]}}]}"#.to_owned()
        } else {
            format!(
                r#"{{"ty":0,"refId":"c{}","ip":0,"op":30,"ks":{{}}}}"#,
                level + 1
            )
        };
        let layers = vec![layer; 10].join(",");
        assets.push(format!(r#"{{"id":"c{level}","layers":[{layers}]}}"#));
    }
    let text = format!(
        r#"{{"w":64,"h":64,"fr":30,"ip":0,"op":30,"assets":[{}],
            "layers":[{{"ty":0,"refId":"c0","ip":0,"op":30,"ks":{{}}}}]}}"#,
        assets.join(",")
    );
    let animation = Animation::parse(text.as_bytes()).unwrap();
    assert_eq!(
        animation.scene(0.0, 64, 64),
        Err(LottieError::Limit("layers in a frame"))
    );
}

#[test]
fn repeaters_inside_repeaters_stop_at_the_draw_limit() {
    let mut shapes =
        r#"{"ty":"rc","p":{"k":[8,8]},"s":{"k":[4,4]}},{"ty":"fl","c":{"k":[1,0,0,1]}}"#.to_owned();
    for _ in 0..3 {
        shapes = format!(
            r#"{{"ty":"gr","it":[{shapes},{{"ty":"rp","c":{{"k":1000}},"tr":{{"p":{{"k":[1,0]}}}}}}]}}"#
        );
    }
    let text = header(&format!(
        r#"[{{"ty":4,"ip":0,"op":30,"ks":{{}},"shapes":[{shapes}]}}]"#
    ));
    let animation = Animation::parse(text.as_bytes()).unwrap();
    assert_eq!(
        animation.scene(0.0, 64, 64),
        Err(LottieError::Limit("draws in a frame"))
    );
}

#[test]
fn a_parent_loop_draws_without_hanging() {
    let text = header(
        r#"[{"ty":4,"ind":1,"parent":2,"ip":0,"op":30,"ks":{"p":{"k":[1,0]}},
             "shapes":[{"ty":"rc","p":{"k":[8,8]},"s":{"k":[4,4]}},{"ty":"fl","c":{"k":[1,0,0,1]}}]},
            {"ty":3,"ind":2,"parent":1,"ip":0,"op":30,"ks":{"p":{"k":[1,0]}}}]"#,
    );
    let animation = Animation::parse(text.as_bytes()).unwrap();
    let scene = animation.scene(0.0, 64, 64).unwrap();
    // Each layer's chain stops where it comes round: the square moves 2.
    let bytes = rasterize(&scene);
    let at = |x: usize, y: usize| bytes[(y * 64 + x) * 4 + 3];
    assert_eq!(at(9, 8), 255);
    assert_eq!(at(6, 8), 0);
}
