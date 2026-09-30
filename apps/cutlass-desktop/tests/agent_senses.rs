#[path = "../src/agent_senses.rs"]
mod agent_senses;
#[path = "../src/agent_vision.rs"]
mod agent_vision;
#[path = "../src/timeline_map.rs"]
mod timeline_map;

use std::collections::{BTreeSet, HashSet};

use agent_senses::{
    AgentSenses, MAX_STRIP_FRAMES, parse_pool_sheet_request, parse_preview_request,
    parse_strip_request, strip_sample_times,
};
use cutlass_ai::ToolTier;
use cutlass_models::{MediaSource, Project, Rational};
use serde_json::{Value, json};

const FPS: Rational = Rational::FPS_24;

fn dispatch_error(project: &Project, name: &str, arguments: Value) -> String {
    AgentSenses::new()
        .call(project, 3.25, name, &arguments)
        .expect_err("call should be rejected before rendering")
}

#[test]
fn registry_has_exact_unique_read_only_object_specs() {
    let specs = AgentSenses::specs();
    let names: Vec<_> = specs.iter().map(|spec| spec.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "media_timeline_map",
            "media_preview_frame",
            "media_pool_sheet",
            "media_asset_frame",
            "media_asset_strip",
        ]
    );
    assert_eq!(
        names.iter().copied().collect::<HashSet<_>>().len(),
        names.len()
    );

    for spec in &specs {
        assert_eq!(spec.tier, ToolTier::ReadOnly, "{}", spec.name);
        assert_eq!(spec.parameters["type"], "object", "{}", spec.name);
        assert_eq!(
            spec.parameters["additionalProperties"], false,
            "{}",
            spec.name
        );
        assert!(spec.parameters["properties"].is_object(), "{}", spec.name);
        assert!(!spec.description.trim().is_empty(), "{}", spec.name);
    }

    assert_eq!(
        property_names(&specs[0]),
        BTreeSet::from(["end_seconds", "playhead_seconds", "start_seconds", "width"])
    );
    assert_eq!(
        property_names(&specs[1]),
        BTreeSet::from(["max_height", "max_width", "seconds"])
    );
    assert_eq!(
        property_names(&specs[2]),
        BTreeSet::from(["max_width", "page"])
    );
    assert_eq!(
        property_names(&specs[3]),
        BTreeSet::from(["max_height", "max_width", "media_id", "seconds"])
    );
    assert_eq!(
        property_names(&specs[4]),
        BTreeSet::from([
            "count",
            "end_seconds",
            "max_height",
            "max_width",
            "media_id",
            "start_seconds",
        ])
    );
    assert_eq!(specs[3].parameters["required"], json!(["media_id"]));
    assert_eq!(
        specs[4].parameters["required"],
        json!(["media_id", "start_seconds", "end_seconds"])
    );
    assert!(
        specs
            .iter()
            .all(|spec| spec.parameters["properties"].get("path").is_none())
    );
}

#[test]
fn empty_project_timeline_map_dispatches_a_labeled_valid_png() {
    let project = Project::new("empty", FPS);
    let output = AgentSenses::default()
        .call(&project, 7.5, "media_timeline_map", &json!({}))
        .expect("CPU-only timeline map");

    assert_eq!(output.images.len(), 1);
    let image = &output.images[0];
    assert_eq!(image.media_type, "image/png");
    assert_eq!(&image.data[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(image.label, "timeline map 0.00s-1.00s");
    let decoded = cutlass_decoder::decode_image_bytes(image.data.as_slice()).expect("valid PNG");
    assert_eq!(decoded.width, 768);
    assert!(decoded.is_well_formed());
    assert!(output.text.contains(&image.label));
    assert!(output.text.contains("0.00s-1.00s"));
    assert!(output.text.contains("0 lanes"));
    assert!(output.text.contains("0 clips"));
    assert!(output.text.contains("no playhead requested"));
}

#[test]
fn empty_media_pool_sheet_is_cpu_only_labeled_and_valid() {
    let project = Project::new("empty", FPS);
    let output = AgentSenses::default()
        .call(&project, 0.0, "media_pool_sheet", &json!({}))
        .expect("empty contact sheet needs no renderer");

    assert_eq!(output.images.len(), 1);
    let image = &output.images[0];
    assert_eq!(image.media_type, "image/jpeg");
    assert_eq!(&image.data[..2], &[0xFF, 0xD8]);
    assert_eq!(image.label, "media pool sheet page 1 of 1");
    let decoded = cutlass_decoder::decode_image_bytes(image.data.as_slice()).expect("valid JPEG");
    assert_eq!(decoded.width, 1024);
    assert!(decoded.is_well_formed());
    assert!(output.text.contains("page 1 of 1"));
    assert!(output.text.contains("showing 0 of 0 visual item(s)"));
    assert!(output.text.contains("Pictured visual media: none"));
}

#[test]
fn dispatch_rejects_non_objects_malformed_fields_and_unknown_names() {
    let project = Project::new("empty", FPS);
    for arguments in [Value::Null, json!([]), json!("arguments"), json!(42)] {
        let error = dispatch_error(&project, "media_timeline_map", arguments);
        assert!(error.contains("JSON object"), "unexpected error: {error}");
    }

    let error = dispatch_error(&project, "media_preview_frame", json!({"seconds": null}));
    assert!(
        error.contains("invalid arguments"),
        "unexpected error: {error}"
    );

    let error = dispatch_error(&project, "media_preview_frame", json!({"seconds": "now"}));
    assert!(
        error.contains("invalid arguments"),
        "unexpected error: {error}"
    );

    let error = dispatch_error(&project, "media_not_registered", json!({}));
    assert!(
        error.contains("unknown media sense tool"),
        "unexpected error: {error}"
    );
}

#[test]
fn timeline_and_frame_requests_validate_times_and_dimensions() {
    let project = Project::new("empty", FPS);
    for arguments in [
        json!({"start_seconds": -0.01}),
        json!({"start_seconds": 2.0, "end_seconds": 2.0}),
        json!({"start_seconds": 2.0, "end_seconds": 1.0}),
        json!({"playhead_seconds": -0.01}),
        json!({"width": 319}),
        json!({"width": 1025}),
        json!({"width": 768.5}),
    ] {
        assert!(
            AgentSenses::new()
                .call(&project, 0.0, "media_timeline_map", &arguments)
                .is_err(),
            "accepted {arguments}"
        );
    }

    for arguments in [
        json!({"seconds": -0.01}),
        json!({"max_width": 0}),
        json!({"max_width": 63}),
        json!({"max_width": 769}),
        json!({"max_height": 0}),
        json!({"max_height": 63}),
        json!({"max_height": 769}),
    ] {
        assert!(
            parse_preview_request(&arguments, 0.0).is_err(),
            "accepted {arguments}"
        );
    }
    assert!(parse_preview_request(&json!({}), f64::NAN).is_err());
    assert!(parse_preview_request(&json!({}), f64::INFINITY).is_err());
    assert!(parse_preview_request(&json!({}), -0.1).is_err());

    assert_eq!(parse_pool_sheet_request(&json!({})).unwrap().page, 1);
    for arguments in [
        json!({"page": 0}),
        json!({"page": 1.5}),
        json!({"max_width": 319}),
        json!({"max_width": 1025}),
    ] {
        assert!(
            parse_pool_sheet_request(&arguments).is_err(),
            "accepted {arguments}"
        );
    }
}

#[test]
fn asset_requests_validate_ids_times_windows_dimensions_and_counts() {
    let project = Project::new("empty", FPS);
    for arguments in [
        json!({"media_id": 0}),
        json!({"media_id": 1.5}),
        json!({"media_id": 1, "seconds": -0.1}),
        json!({"media_id": 1, "max_width": 63}),
        json!({"media_id": 1, "max_height": 769}),
    ] {
        assert!(
            AgentSenses::new()
                .call(&project, 0.0, "media_asset_frame", &arguments)
                .is_err(),
            "accepted {arguments}"
        );
    }

    for arguments in [
        json!({"media_id": 1, "start_seconds": -0.1, "end_seconds": 2.0}),
        json!({"media_id": 1, "start_seconds": 2.0, "end_seconds": 2.0}),
        json!({"media_id": 1, "start_seconds": 3.0, "end_seconds": 2.0}),
        json!({"media_id": 1, "start_seconds": 0.0, "end_seconds": 2.0, "count": 0}),
        json!({"media_id": 1, "start_seconds": 0.0, "end_seconds": 2.0, "count": 9}),
        json!({"media_id": 1, "start_seconds": 0.0, "end_seconds": 2.0, "count": 2.5}),
        json!({"media_id": 1, "start_seconds": 0.0, "end_seconds": 2.0, "max_width": 63}),
        json!({"media_id": 1, "start_seconds": 0.0, "end_seconds": 2.0, "max_height": 769}),
        json!({"media_id": 1, "end_seconds": 2.0}),
        json!({"media_id": 1, "start_seconds": 0.0}),
    ] {
        assert!(
            parse_strip_request(&arguments).is_err(),
            "accepted {arguments}"
        );
    }
}

#[test]
fn unknown_media_and_arbitrary_paths_are_rejected_without_rendering() {
    let empty = Project::new("empty", FPS);
    let error = dispatch_error(&empty, "media_asset_frame", json!({"media_id": 999_999}));
    assert!(
        error.contains("unknown project media id"),
        "unexpected error: {error}"
    );

    let mut project = Project::new("pool", FPS);
    let media = project.add_media(MediaSource::new(
        "/project/pool/known.mov",
        1920,
        1080,
        FPS,
        240,
        false,
    ));
    for arguments in [
        json!({"path": "/outside/project/secret.mov"}),
        json!({
            "media_id": media.raw(),
            "path": "/outside/project/secret.mov"
        }),
    ] {
        let error = dispatch_error(&project, "media_asset_frame", arguments);
        assert!(error.contains("path"), "unexpected error: {error}");
        assert!(
            error.contains("invalid arguments"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn audio_only_media_is_rejected_before_decoder_or_renderer_startup() {
    let mut project = Project::new("audio", FPS);
    let media = project.add_media(MediaSource::new(
        "/file/does/not/need/to/exist.wav",
        0,
        0,
        Rational::new(1000, 1),
        30_000,
        true,
    ));

    let frame_error = dispatch_error(
        &project,
        "media_asset_frame",
        json!({"media_id": media.raw()}),
    );
    assert!(
        frame_error.contains("audio-only"),
        "unexpected error: {frame_error}"
    );

    let strip_error = dispatch_error(
        &project,
        "media_asset_strip",
        json!({
            "media_id": media.raw(),
            "start_seconds": 0.0,
            "end_seconds": 10.0
        }),
    );
    assert!(
        strip_error.contains("audio-only"),
        "unexpected error: {strip_error}"
    );

    let sheet = AgentSenses::new()
        .call(&project, 0.0, "media_pool_sheet", &json!({}))
        .expect("audio-only pools need no renderer");
    assert_eq!(sheet.images.len(), 1);
    assert!(sheet.text.contains("audio-only"));
    assert!(sheet.text.contains("exist.wav"));
    assert!(sheet.text.contains("not pictured"));
}

#[test]
fn preview_parser_routes_the_caller_playhead_and_honors_explicit_time() {
    let defaulted = parse_preview_request(&json!({}), 12.25).expect("default playhead");
    assert_eq!(defaulted.seconds, 12.25);
    assert_eq!(defaulted.max_width, 768);
    assert_eq!(defaulted.max_height, 768);

    let explicit = parse_preview_request(
        &json!({
            "seconds": 4.5,
            "max_width": 320,
            "max_height": 180
        }),
        f64::NAN,
    )
    .expect("explicit time does not consult the default");
    assert_eq!(explicit.seconds, 4.5);
    assert_eq!(explicit.max_width, 320);
    assert_eq!(explicit.max_height, 180);
}

#[test]
fn strip_sampling_is_ordered_deterministic_and_has_defined_endpoints() {
    let expected = vec![2.0, 4.0, 6.0, 8.0];
    let first = strip_sample_times(2.0, 8.0, 4, false).expect("samples");
    let second = strip_sample_times(2.0, 8.0, 4, false).expect("repeat");
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert_eq!(first.first(), Some(&2.0));
    assert_eq!(first.last(), Some(&8.0));
    assert!(first.windows(2).all(|pair| pair[0] < pair[1]));

    assert_eq!(
        strip_sample_times(2.0, 8.0, 1, false).expect("one sample"),
        vec![2.0]
    );
}

#[test]
fn strip_defaults_are_bounded_and_stills_are_deduplicated() {
    let request = parse_strip_request(&json!({
        "media_id": 7,
        "start_seconds": 1.0,
        "end_seconds": 4.0
    }))
    .expect("defaults");
    assert_eq!(request.count, 6);
    assert_eq!(request.max_width, 768);
    assert_eq!(request.max_height, 768);
    assert!(request.count <= MAX_STRIP_FRAMES);

    let samples = strip_sample_times(1.0, 4.0, MAX_STRIP_FRAMES, true).expect("deduplicated still");
    assert_eq!(samples, vec![1.0]);
    assert!(samples.len() < MAX_STRIP_FRAMES as usize);
}

fn property_names(spec: &cutlass_ai::HostToolSpec) -> BTreeSet<&str> {
    spec.parameters["properties"]
        .as_object()
        .expect("properties object")
        .keys()
        .map(String::as_str)
        .collect()
}
