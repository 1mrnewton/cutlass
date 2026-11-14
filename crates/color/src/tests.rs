use cutlass_codec::access::{
    ColorMatrix, ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer, Yuv420p,
};

use crate::{
    Cube, Curve, Depth, Error, Input, Layout, Lut, LutIssue, Matrix, Outcome, Picture,
    PlatformView, Request, SegmentHdr, StillMap, StillTags, ThreeDl, apply_lut, convert,
    curve_from_codec,
};

fn planes<'a>(
    layout: Layout,
    depth: Depth,
    width: u32,
    height: u32,
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
) -> Picture<'a> {
    Picture {
        layout,
        depth,
        width,
        height,
        y,
        u,
        v,
        y_stride: 0,
        chroma_stride: 0,
        range: ColorRange::Limited,
        matrix: Matrix::Bt709,
        primaries: ColorPrimaries::Unspecified,
        curve: Curve::Unspecified,
    }
}

fn at(picture: Picture<'_>, request: &Request, x: u32, y: u32) -> [u8; 4] {
    let Outcome::Working(working) = convert(Input::Cpu(picture), request).expect("convert") else {
        panic!("platform");
    };
    assert_eq!(working.draft, request.draft);
    assert_eq!(working.segment, request.segment);
    assert_eq!(working.format, PixelFormat::Rgba8);
    working.pixel(x, y).expect("pixel")
}

#[test]
fn limited_420_black_and_white_match_the_studio_swing() {
    let request = Request::draft(0);
    let black = planes(
        Layout::Yuv420,
        Depth::Eight,
        2,
        2,
        &[16, 16, 16, 16],
        &[128],
        &[128],
    );
    let white = planes(
        Layout::Yuv420,
        Depth::Eight,
        2,
        2,
        &[235, 235, 235, 235],
        &[128],
        &[128],
    );
    assert_eq!(at(black, &request, 1, 1), [0, 0, 0, 255]);
    assert_eq!(at(white, &request, 0, 0), [255, 255, 255, 255]);
}

#[test]
fn full_range_420_keeps_the_code_that_limited_range_expands() {
    let request = Request::draft(0);
    let mut full = planes(
        Layout::Yuv420,
        Depth::Eight,
        2,
        2,
        &[16, 16, 16, 16],
        &[128],
        &[128],
    );
    full.range = ColorRange::Full;
    assert_eq!(at(full, &request, 0, 0), [16, 16, 16, 255]);
    full.y = &[235, 235, 235, 235];
    assert_eq!(at(full, &request, 1, 1), [235, 235, 235, 255]);
}

#[test]
fn unspecified_range_uses_studio_swing() {
    let request = Request::draft(0);
    let mut picture = planes(
        Layout::Yuv420,
        Depth::Eight,
        2,
        2,
        &[16, 16, 16, 16],
        &[128],
        &[128],
    );
    picture.range = ColorRange::Unspecified;
    assert_eq!(at(picture, &request, 0, 0), [0, 0, 0, 255]);
    picture.range = ColorRange::Studio;
    assert_eq!(at(picture, &request, 0, 0), [0, 0, 0, 255]);
}

#[test]
fn yuv420_chroma_belongs_to_the_top_left_cell() {
    let request = Request::draft(0);
    let y = [235_u8; 8];
    let picture = planes(
        Layout::Yuv420,
        Depth::Eight,
        4,
        2,
        &y,
        &[128, 160],
        &[128, 128],
    );
    assert_eq!(at(picture, &request, 0, 0), [255, 255, 255, 255]);
    assert_eq!(at(picture, &request, 1, 1), [255, 255, 255, 255]);
    assert_eq!(at(picture, &request, 2, 0), [255, 248, 255, 255]);
    assert_eq!(at(picture, &request, 3, 1), [255, 248, 255, 255]);
}

#[test]
fn a_codec_420_picture_converts_with_the_same_matrix() {
    let frame = Yuv420p::solid(2, 2, 16, 128, 128).expect("frame");
    let picture = Picture::from_yuv420p(
        &frame,
        ColorRange::Limited,
        Matrix::from_coefficients(MatrixCoefficients::Bt709).expect("matrix"),
        ColorPrimaries::Bt709,
        curve_from_codec(Transfer::Bt709),
    );
    assert_eq!(at(picture, &Request::draft(0), 0, 0), [0, 0, 0, 255]);
    assert_eq!(
        Matrix::from_coefficients(MatrixCoefficients::Rgb),
        Err(Error::Matrix)
    );
    assert_eq!(curve_from_codec(Transfer::Bt601), Curve::Rec601);
    assert_eq!(curve_from_codec(Transfer::Bt2020), Curve::Rec709);
    assert_eq!(curve_from_codec(Transfer::Pq), Curve::Pq);
    assert_eq!(curve_from_codec(Transfer::Hlg), Curve::Hlg);
    assert_eq!(curve_from_codec(Transfer::Srgb), Curve::Unspecified);
    assert_eq!(curve_from_codec(Transfer::Linear), Curve::Unspecified);
}

#[test]
fn yuv422_rows_do_not_share_chroma() {
    let request = Request::draft(0);
    let y = [235_u8; 8];
    let picture = planes(
        Layout::Yuv422,
        Depth::Eight,
        4,
        2,
        &y,
        &[128, 128, 160, 160],
        &[128, 128, 128, 128],
    );
    assert_eq!(at(picture, &request, 0, 0), [255, 255, 255, 255]);
    assert_eq!(at(picture, &request, 3, 0), [255, 255, 255, 255]);
    assert_eq!(at(picture, &request, 0, 1), [255, 248, 255, 255]);
    assert_eq!(at(picture, &request, 2, 1), [255, 248, 255, 255]);
}

#[test]
fn yuv444_uses_one_chroma_sample_per_pixel() {
    let request = Request::draft(0);
    let picture = planes(
        Layout::Yuv444,
        Depth::Eight,
        2,
        1,
        &[235, 235],
        &[128, 160],
        &[128, 128],
    );
    assert_eq!(at(picture, &request, 0, 0), [255, 255, 255, 255]);
    assert_eq!(at(picture, &request, 1, 0), [255, 248, 255, 255]);
}

#[test]
fn bt601_and_bt2020_matrices_differ_from_bt709() {
    let request = Request::draft(0);
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[82], &[90], &[240]);
    picture.primaries = ColorPrimaries::Bt709;
    assert_eq!(at(picture, &request, 0, 0), [255, 25, 0, 255]);
    picture.matrix = Matrix::Bt601;
    assert_eq!(at(picture, &request, 0, 0), [255, 1, 0, 255]);
    picture.matrix = Matrix::Bt2020;
    assert_eq!(at(picture, &request, 0, 0), [255, 11, 0, 255]);
    picture.matrix = Matrix::Bt709;
    picture.primaries = ColorPrimaries::Bt601;
    let smpte170 = at(picture, &request, 0, 0);
    picture.primaries = ColorPrimaries::Smpte240m;
    assert_eq!(at(picture, &request, 0, 0), smpte170);
    picture.primaries = ColorPrimaries::Unspecified;
    picture.matrix = Matrix::Bt601;
    assert_eq!(at(picture, &request, 0, 0), [247, 21, 0, 255]);
    picture.primaries = ColorPrimaries::Film;
    assert_eq!(at(picture, &request, 0, 0), [247, 21, 0, 255]);
    picture.primaries = ColorPrimaries::Unspecified;
    picture.matrix = Matrix::Bt2020;
    assert_eq!(at(picture, &request, 0, 0), [255, 0, 0, 255]);
}

#[test]
fn nv12_is_named_and_refused() {
    let picture = planes(
        Layout::Nv12,
        Depth::Eight,
        2,
        2,
        &[16, 16, 16, 16],
        &[128],
        &[128],
    );
    let error = convert(Input::Cpu(picture), &Request::draft(0)).expect_err("nv12");
    assert_eq!(error, Error::Layout(Layout::Nv12));
}

#[test]
fn a_short_or_odd_plane_is_an_error() {
    let request = Request::draft(0);
    let short = planes(
        Layout::Yuv420,
        Depth::Eight,
        2,
        2,
        &[16, 16, 16],
        &[128],
        &[128],
    );
    assert_eq!(
        convert(Input::Cpu(short), &request).expect_err("short"),
        Error::Length
    );
    let odd = planes(
        Layout::Yuv420,
        Depth::Eight,
        3,
        2,
        &[0; 6],
        &[0, 0],
        &[0, 0],
    );
    assert_eq!(
        convert(Input::Cpu(odd), &request).expect_err("odd"),
        Error::Odd
    );
    let empty = planes(Layout::Yuv444, Depth::Eight, 0, 1, &[], &[], &[]);
    assert_eq!(
        convert(Input::Cpu(empty), &request).expect_err("empty"),
        Error::Empty
    );
    let wide = planes(
        Layout::Yuv444,
        Depth::Eight,
        crate::MAX_EDGE + 1,
        1,
        &[0],
        &[0],
        &[0],
    );
    assert_eq!(
        convert(Input::Cpu(wide), &request).expect_err("edge"),
        Error::Limit
    );
}

#[test]
fn stride_skips_padding_bytes() {
    let request = Request::draft(0);
    let mut picture = planes(
        Layout::Yuv420,
        Depth::Eight,
        2,
        2,
        &[16, 16, 9, 9, 16, 16, 9, 9],
        &[128],
        &[128],
    );
    picture.y_stride = 4;
    assert_eq!(at(picture, &request, 0, 0), [0, 0, 0, 255]);
    assert_eq!(at(picture, &request, 1, 1), [0, 0, 0, 255]);
}

#[test]
fn ten_bit_full_range_scales_instead_of_shifting() {
    // Code 3. Dropping the low bits (`3 >> 2`) would store 0.
    let request = Request::draft(0);
    let mut picture = planes(Layout::Yuv444, Depth::Ten, 1, 1, &[3, 0], &[0, 2], &[0, 2]);
    picture.range = ColorRange::Full;
    assert_eq!(at(picture, &request, 0, 0), [1, 1, 1, 255]);
}

#[test]
fn ten_bit_limited_neutral_is_not_the_shifted_code() {
    // Code 512. A right shift by 2 stores 128. The 10-bit studio level is 130.
    let request = Request::draft(0);
    let picture = planes(
        Layout::Yuv420,
        Depth::Ten,
        2,
        2,
        &[0, 2, 0, 2, 0, 2, 0, 2],
        &[0, 2],
        &[0, 2],
    );
    assert_eq!(at(picture, &request, 0, 1), [130, 130, 130, 255]);
}

#[test]
fn ten_bit_white_and_an_out_of_range_code() {
    let request = Request::draft(0);
    let mut white = planes(
        Layout::Yuv444,
        Depth::Ten,
        1,
        1,
        &[255, 3],
        &[0, 2],
        &[0, 2],
    );
    white.range = ColorRange::Full;
    assert_eq!(at(white, &request, 0, 0), [255, 255, 255, 255]);
    let overflow = planes(Layout::Yuv444, Depth::Ten, 1, 1, &[0, 4], &[0, 2], &[0, 2]);
    assert_eq!(
        convert(Input::Cpu(overflow), &request).expect_err("range"),
        Error::Range
    );
    let truncated = planes(Layout::Yuv444, Depth::Ten, 1, 1, &[3], &[0, 2], &[0, 2]);
    assert_eq!(
        convert(Input::Cpu(truncated), &request).expect_err("bytes"),
        Error::Length
    );
}

#[test]
fn rec709_lands_in_the_draft_working_space() {
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[16], &[128], &[144]);
    picture.range = ColorRange::Full;
    picture.primaries = ColorPrimaries::Bt709;
    picture.curve = Curve::Rec709;
    assert_eq!(at(picture, &Request::draft(0), 0, 0), [41, 9, 16, 255]);
    assert_eq!(at(picture, &Request::draft(-1), 0, 0), [41, 9, 16, 255]);
    assert_eq!(at(picture, &Request::draft(2), 0, 0), [31, 11, 16, 255]);
    let error = convert(Input::Cpu(picture), &Request::draft(1)).expect_err("draft");
    assert_eq!(error, Error::Draft(1));
}

#[test]
fn hlg_is_not_the_segment_flag_or_the_rec709_curve() {
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[128], &[128], &[128]);
    picture.range = ColorRange::Full;
    picture.primaries = ColorPrimaries::Bt2020;
    picture.curve = Curve::Hlg;
    assert_eq!(at(picture, &Request::draft(0), 0, 0), [67, 67, 67, 255]);
    assert_eq!(at(picture, &Request::draft(67), 0, 0), [128, 128, 128, 255]);
    picture.curve = Curve::Rec709;
    assert_eq!(at(picture, &Request::draft(0), 0, 0), [128, 128, 128, 255]);
    assert_eq!(at(picture, &Request::draft(67), 0, 0), [191, 191, 191, 255]);
}

#[test]
fn pq_uses_the_st2084_curve_and_the_segment_peak() {
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[128], &[128], &[128]);
    picture.range = ColorRange::Full;
    picture.primaries = ColorPrimaries::Bt2020;
    picture.curve = Curve::Pq;
    assert_eq!(at(picture, &Request::draft(0), 0, 0), [247, 247, 247, 255]);
    let mut peaked = Request::draft(0);
    peaked.segment.nits = 1000.0;
    assert_eq!(at(picture, &peaked, 0, 0), [71, 71, 71, 255]);
    picture.curve = Curve::Rec709;
    assert_eq!(at(picture, &Request::draft(0), 0, 0), [128, 128, 128, 255]);
}

#[test]
fn segment_hdr_changes_pixels_and_leaves_the_draft_code() {
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[128], &[128], &[128]);
    picture.range = ColorRange::Full;
    picture.primaries = ColorPrimaries::Bt709;
    picture.curve = Curve::Rec709;
    let mut display = Request::draft(67);
    display.segment.mode = 4;
    assert_eq!(at(picture, &display, 0, 0), [67, 67, 67, 255]);
    assert_eq!(at(picture, &Request::draft(67), 0, 0), [191, 191, 191, 255]);
    let mut to_pq = Request::draft(2084);
    to_pq.segment.mode = 6;
    picture.curve = Curve::Rec709;
    assert_eq!(at(picture, &to_pq, 0, 0), [247, 247, 247, 255]);
    let error = convert(
        Input::Cpu(picture),
        &Request {
            draft: 0,
            segment: SegmentHdr {
                mode: 12,
                intensity: 1.0,
                nits: 0.0,
            },
        },
    )
    .expect_err("mode");
    assert_eq!(error, Error::Segment(12));
    let mut weakened = Request::draft(0);
    weakened.segment.mode = 2;
    weakened.segment.intensity = 0.0;
    assert_eq!(at(picture, &weakened, 0, 0), [128, 128, 128, 255]);
    weakened.segment.intensity = 1.0;
    assert_eq!(at(picture, &weakened, 0, 0), [191, 191, 191, 255]);
    weakened.segment.mode = 1;
    assert_eq!(at(picture, &weakened, 0, 0), [128, 128, 128, 255]);
}

#[test]
fn a_bad_intensity_or_peak_is_refused() {
    let picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[128], &[128], &[128]);
    let mut request = Request::draft(0);
    request.segment.intensity = f64::NAN;
    assert_eq!(
        convert(Input::Cpu(picture), &request).expect_err("intensity"),
        Error::Intensity
    );
    request.segment.intensity = 1.0;
    request.segment.nits = -1.0;
    assert_eq!(
        convert(Input::Cpu(picture), &request).expect_err("nits"),
        Error::Nits
    );
}

#[test]
fn a_platform_buffer_is_not_pulled_onto_the_cpu() {
    let kept = convert(
        Input::Platform(PlatformView {
            token: 11,
            preview_accepts: true,
            encoder_accepts: true,
        }),
        &Request::draft(0),
    )
    .expect("skip");
    assert_eq!(kept, Outcome::Platform { token: 11 });
    let error = convert(
        Input::Platform(PlatformView {
            token: 11,
            preview_accepts: true,
            encoder_accepts: false,
        }),
        &Request::draft(0),
    )
    .expect_err("download");
    assert_eq!(error, Error::Platform);
}

#[test]
fn the_manifest_does_not_depend_on_gpu_session_or_a_container() {
    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("cutlass-codec"));
    assert!(!manifest.contains("cutlass-gpu"));
    assert!(!manifest.contains("cutlass-session"));
    assert!(!manifest.contains("cutlass-container"));
    assert!(!manifest.contains("cutlass-model"));
    assert!(!manifest.contains("cutlass-picture"));
}

#[test]
fn a_cube_remaps_rgb_and_a_truncated_cube_is_an_error() {
    let text = "\
TITLE \"swap\"
LUT_3D_SIZE 2
0 0 0
0 1 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
";
    let cube = Cube::parse(text).expect("cube");
    assert_eq!(cube.map([255, 0, 0]), [0, 255, 0]);
    assert!(ThreeDl::parse(text).is_err());
    let truncated = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n";
    assert_eq!(
        Cube::parse(truncated).expect_err("short"),
        Error::Lut(LutIssue::Truncated)
    );
    assert_eq!(
        Cube::parse("LUT_1D_SIZE 2\n0\n1\n").expect_err("1d"),
        Error::Lut(LutIssue::OneDimensional)
    );
}

#[test]
fn a_3dl_remaps_rgb_and_is_not_a_cube() {
    let text = "\
0 1023
0 0 0
0 0 1023
0 1023 0
0 1023 1023
1023 0 0
1023 0 1023
1023 1023 0
1023 1023 1023
";
    let table = ThreeDl::parse(text).expect("3dl");
    assert_eq!(table.map([255, 0, 0]), [255, 0, 0]);
    assert_eq!(table.map([0, 0, 255]), [0, 0, 255]);
    assert_eq!(table.map([0, 255, 0]), [0, 255, 0]);
    assert_eq!(
        Cube::parse(text).expect_err("not a cube"),
        Error::Lut(LutIssue::Header)
    );
    assert_eq!(
        ThreeDl::parse("0 1023\n0 0 0\n").expect_err("short"),
        Error::Lut(LutIssue::Truncated)
    );
}

#[test]
fn a_12_bit_3dl_sample_is_scaled_by_4095() {
    let text = "\
0 1023
0 0 0
0 0 0
0 0 0
0 0 0
1000 0 0
0 0 0
0 0 0
4095 0 0
";
    let table = ThreeDl::parse(text).expect("3dl");
    assert_eq!(table.map([255, 0, 0]), [62, 0, 0]);
}

#[test]
fn applying_a_lut_keeps_alpha_and_the_draft_code() {
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[16], &[128], &[128]);
    picture.range = ColorRange::Full;
    let Outcome::Working(mut working) =
        convert(Input::Cpu(picture), &Request::draft(-1)).expect("rgb")
    else {
        panic!("platform");
    };
    assert_eq!(working.pixel(0, 0), Some([16, 16, 16, 255]));
    let cube = Cube::parse(
        "\
LUT_3D_SIZE 2
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
",
    )
    .expect("identity");
    apply_lut(&mut working, &Lut::Cube(cube)).expect("lut");
    assert_eq!(working.pixel(0, 0), Some([16, 16, 16, 255]));
    assert_eq!(working.draft, -1);
    assert_eq!(working.rgba[3], 255);
}

fn tags<'a>(
    gamma: Option<u32>,
    chromaticity: Option<[u32; 8]>,
    srgb: Option<u8>,
    icc: &'a [u8],
) -> StillTags<'a> {
    StillTags {
        gamma_times_100000: gamma,
        chromaticity,
        srgb_intent: srgb,
        icc_profile: icc,
    }
}

fn mapped(tags: &StillTags<'_>, rgb: [u8; 3]) -> [u8; 3] {
    StillMap::prepare(tags).expect("map").map_rgb(rgb)
}

#[test]
fn reference_gamma_is_copied_and_linear_gamma_is_encoded() {
    let reference = tags(Some(45_455), None, None, &[]);
    let map = StillMap::prepare(&reference).expect("map");
    assert!(map.is_identity());
    assert_eq!(map.map_rgb([1, 2, 3]), [1, 2, 3]);

    let linear = tags(Some(100_000), None, None, &[]);
    assert_eq!(mapped(&linear, [128, 128, 128]), [186, 186, 186]);
    assert_eq!(mapped(&linear, [0, 255, 128]), [0, 255, 186]);
    assert_eq!(mapped(&linear, [1, 2, 3]), [21, 28, 34]);

    let unnamed = tags(None, None, None, &[]);
    assert!(StillMap::prepare(&unnamed).expect("map").is_identity());
    assert!(matches!(
        StillMap::prepare(&tags(Some(0), None, None, &[])),
        Err(Error::Gamma)
    ));
}

#[test]
fn an_srgb_intent_wins_over_gamma_and_a_broken_profile_does_not() {
    let intent = tags(Some(100_000), None, Some(0), &[1, 2, 3, 4]);
    let map = StillMap::prepare(&intent).expect("map");
    assert!(map.is_identity());
    assert_eq!(map.map_rgb([128, 10, 20]), [128, 10, 20]);
    assert!(
        StillMap::prepare(&tags(Some(0), None, Some(3), &[]))
            .expect("map")
            .is_identity()
    );

    let broken = tags(Some(100_000), None, None, &[1, 2, 3, 4]);
    assert_eq!(mapped(&broken, [128, 128, 128]), [186, 186, 186]);
}

#[test]
fn chromaticity_moves_a_colored_sample_and_bt709_does_not() {
    let bt709 = [
        31_270, 32_900, 64_000, 33_000, 30_000, 60_000, 15_000, 6_000,
    ];
    let tagged = tags(Some(45_455), Some(bt709), None, &[]);
    assert!(StillMap::prepare(&tagged).expect("map").is_identity());

    let mut shifted = bt709;
    shifted[2] = 50_000;
    shifted[3] = 30_000;
    let moved = tags(Some(45_455), Some(shifted), None, &[]);
    assert_eq!(mapped(&moved, [180, 80, 40]), [180, 93, 86]);

    let mut broken = bt709;
    broken[1] = 0;
    let fallback = tags(Some(100_000), Some(broken), None, &[]);
    assert_eq!(mapped(&fallback, [128, 0, 0]), [186, 0, 0]);
}

#[test]
fn a_linear_matrix_profile_overrides_reference_gamma() {
    let profile = linear_bt709_profile(&gamma_tag(256));
    let tagged = tags(Some(45_455), None, Some(1), &profile);
    assert_eq!(mapped(&tagged, [128, 128, 128]), [186, 186, 186]);
    assert_eq!(
        mapped(&tags(Some(0), None, None, &profile), [128, 128, 128]),
        [186, 186, 186]
    );

    let parametric = linear_bt709_profile(&para_gamma(1.0));
    assert_eq!(
        mapped(
            &tags(Some(45_455), None, None, &parametric),
            [128, 128, 128]
        ),
        [186, 186, 186]
    );

    // The quantized matrix leaves a tiny green residual. The working curve lifts it to 1.
    let identity = linear_bt709_profile(&curv_identity());
    assert_eq!(
        mapped(&tags(None, None, None, &identity), [128, 0, 255]),
        [186, 1, 255]
    );

    let ramp = linear_bt709_profile(&curv_ramp());
    assert_eq!(
        mapped(&tags(None, None, None, &ramp), [128, 0, 255]),
        [186, 1, 255]
    );
}

fn linear_bt709_profile(trc: &[u8]) -> Vec<u8> {
    let columns = [
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ];
    let illuminant = [0.95047, 1.0, 1.08883];
    let mut tags = Vec::new();
    for column in columns {
        tags.push(xyz_tag(column));
    }
    for _ in 0..3 {
        tags.push(trc.to_vec());
    }
    let signatures: [[u8; 4]; 6] = [*b"rXYZ", *b"gXYZ", *b"bXYZ", *b"rTRC", *b"gTRC", *b"bTRC"];
    let mut body = vec![0_u8; 128];
    body[16..20].copy_from_slice(b"RGB ");
    body[20..24].copy_from_slice(b"XYZ ");
    body[36..40].copy_from_slice(b"acsp");
    write_i32(&mut body, 68, fixed(illuminant[0]));
    write_i32(&mut body, 72, fixed(illuminant[1]));
    write_i32(&mut body, 76, fixed(illuminant[2]));
    let mut cursor = 132 + signatures.len() * 12;
    let mut entries = Vec::new();
    for (signature, tag) in signatures.iter().zip(tags) {
        entries.push((*signature, cursor, tag.len()));
        cursor = cursor.saturating_add(tag.len()).div_ceil(4) * 4;
    }
    body.extend(
        u32::try_from(signatures.len())
            .expect("count")
            .to_be_bytes(),
    );
    for (signature, offset, size) in &entries {
        body.extend(signature);
        body.extend(u32::try_from(*offset).expect("offset").to_be_bytes());
        body.extend(u32::try_from(*size).expect("size").to_be_bytes());
    }
    for (index, (signature, _, _)) in entries.iter().enumerate() {
        let _ = signature;
        let tag = match index {
            0..=2 => xyz_tag(columns[index]),
            _ => trc.to_vec(),
        };
        let pad = (4 - tag.len() % 4) % 4;
        body.extend(tag);
        body.extend(std::iter::repeat_n(0, pad));
    }
    let size = u32::try_from(body.len()).expect("size");
    body[..4].copy_from_slice(&size.to_be_bytes());
    body
}

fn xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend(b"XYZ ");
    tag.extend([0, 0, 0, 0]);
    for value in xyz {
        tag.extend(fixed(value).to_be_bytes());
    }
    tag
}

fn para_gamma(gamma: f64) -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend(b"para");
    tag.extend([0, 0, 0, 0]);
    tag.extend(0_u16.to_be_bytes());
    tag.extend(0_u16.to_be_bytes());
    tag.extend(fixed(gamma).to_be_bytes());
    tag
}

fn curv_ramp() -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend(b"curv");
    tag.extend([0, 0, 0, 0]);
    tag.extend(2_u32.to_be_bytes());
    tag.extend(0_u16.to_be_bytes());
    tag.extend(u16::MAX.to_be_bytes());
    tag
}

fn curv_identity() -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend(b"curv");
    tag.extend([0, 0, 0, 0]);
    tag.extend(0_u32.to_be_bytes());
    tag
}

fn gamma_tag(gamma_8_8: u16) -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend(b"curv");
    tag.extend([0, 0, 0, 0]);
    tag.extend(1_u32.to_be_bytes());
    tag.extend(gamma_8_8.to_be_bytes());
    tag
}

fn fixed(value: f64) -> i32 {
    value.mul_add(65_536.0, 0.0).round() as i32
}

fn write_i32(bytes: &mut [u8], at: usize, value: i32) {
    bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
}

#[test]
fn a_bt709_picture_keeps_its_signal() {
    let signal =
        crate::Signal::for_tags(ColorPrimaries::Bt709, Transfer::Bt709, Matrix::Bt709, 0).unwrap();
    assert!(signal.same_signal());
}

#[test]
fn a_bt2020_hlg_picture_is_decoded_adapted_and_encoded_as_rec709() {
    let signal =
        crate::Signal::for_tags(ColorPrimaries::Bt2020, Transfer::Hlg, Matrix::Bt709, 0).unwrap();
    assert!(!signal.same_signal());
    assert_eq!(signal.source, crate::SignalCurve::Hlg);
    assert_eq!(signal.dest, crate::SignalCurve::Rec709);
    let adapt = signal.adapt.expect("a gamut change");
    // Each row of a D65 to D65 adaptation sums to one: white stays white.
    for row in adapt {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1.0e-9);
    }
}

#[test]
fn an_unknown_draft_code_has_no_signal() {
    assert_eq!(
        crate::Signal::for_tags(ColorPrimaries::Bt709, Transfer::Bt709, Matrix::Bt709, 5,),
        Err(Error::Draft(5))
    );
}

#[test]
fn twelve_bit_levels_are_the_eight_bit_ones_times_sixteen() {
    let request = Request::draft(0);
    // Limited black 256, white 3760, and a neutral 2048 at the level a
    // 10-bit 512 has.
    let le = |code: u16| code.to_le_bytes();
    for (code, want) in [(256u16, 0u8), (3760, 255), (2048, 130)] {
        let y = le(code);
        let c = le(2048);
        let picture = planes(Layout::Yuv444, Depth::Twelve, 1, 1, &y, &c, &c);
        assert_eq!(at(picture, &request, 0, 0), [want, want, want, 255]);
    }
    // Full range ends at 4095, and a code past it is refused.
    let (y, c) = (le(4095), le(2048));
    let mut white = planes(Layout::Yuv444, Depth::Twelve, 1, 1, &y, &c, &c);
    white.range = ColorRange::Full;
    assert_eq!(at(white, &request, 0, 0), [255, 255, 255, 255]);
    let y = le(4096);
    let overflow = planes(Layout::Yuv444, Depth::Twelve, 1, 1, &y, &c, &c);
    assert_eq!(
        convert(Input::Cpu(overflow), &request).expect_err("range"),
        Error::Range
    );
    // A 10-bit picture still ends at 1023.
    let y = le(1024);
    let c = le(512);
    let ten = planes(Layout::Yuv444, Depth::Ten, 1, 1, &y, &c, &c);
    assert_eq!(
        convert(Input::Cpu(ten), &request).expect_err("range"),
        Error::Range
    );
}

#[test]
fn a_440_chroma_sample_covers_the_rows_above_and_below() {
    let request = Request::draft(0);
    // 2 wide, 4 tall: chroma is 2 by 2. The top pair of rows is neutral and
    // the bottom pair is red-difference heavy.
    let y = [126u8; 8];
    let u = [128, 128, 128, 128];
    let v = [128, 128, 240, 240];
    let picture = planes(Layout::Yuv440, Depth::Eight, 2, 4, &y, &u, &v);
    let grey = at(picture, &request, 1, 1);
    assert_eq!(grey[0], grey[1]);
    assert_eq!(grey[1], grey[2]);
    let red = at(picture, &request, 1, 2);
    assert_eq!(red, at(picture, &request, 0, 3));
    assert!(red[0] > grey[0] && red[1] < grey[1]);
    // An odd height keeps a chroma row for its last luma row.
    let y = [126u8; 6];
    let odd = planes(Layout::Yuv440, Depth::Eight, 2, 3, &y, &u, &v);
    assert_eq!(at(odd, &request, 0, 2), red);
    // One chroma row is too few for three luma rows.
    let short = planes(Layout::Yuv440, Depth::Eight, 2, 3, &y, &u[..2], &v[..2]);
    assert_eq!(
        convert(Input::Cpu(short), &request).expect_err("short"),
        Error::Length
    );
}

#[test]
fn an_odd_422_width_keeps_a_chroma_sample_for_its_last_column() {
    let request = Request::draft(0);
    let y = [126u8; 3];
    let u = [128, 128];
    let v = [128, 240];
    let picture = planes(Layout::Yuv422, Depth::Eight, 3, 1, &y, &u, &v);
    let grey = at(picture, &request, 1, 0);
    let red = at(picture, &request, 2, 0);
    assert_eq!(grey[0], grey[1]);
    assert!(red[0] > grey[0]);
    let short = planes(Layout::Yuv422, Depth::Eight, 3, 1, &y, &u[..1], &v[..1]);
    assert_eq!(
        convert(Input::Cpu(short), &request).expect_err("short"),
        Error::Length
    );
}

#[test]
fn gbr_planes_take_no_matrix() {
    let request = Request::draft(0);
    // Planes are G, B, R. Full range comes out as stored.
    let mut picture = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[10], &[20], &[30]);
    picture.matrix = Matrix::Identity;
    picture.range = ColorRange::Full;
    assert_eq!(at(picture, &request, 0, 0), [30, 10, 20, 255]);
    // Limited range expands each plane as luma is: 16 is 0 and 235 is 255.
    let mut limited = planes(Layout::Yuv444, Depth::Eight, 1, 1, &[16], &[235], &[126]);
    limited.matrix = Matrix::Identity;
    assert_eq!(at(limited, &request, 0, 0), [128, 0, 255, 255]);
    // Ten bits, full range: 1023 is 255.
    let (g, b, r) = (
        1023u16.to_le_bytes(),
        0u16.to_le_bytes(),
        512u16.to_le_bytes(),
    );
    let mut ten = planes(Layout::Yuv444, Depth::Ten, 1, 1, &g, &b, &r);
    ten.matrix = Matrix::Identity;
    ten.range = ColorRange::Full;
    assert_eq!(at(ten, &request, 0, 0), [128, 255, 0, 255]);
}

#[test]
fn a_decoded_matrix_maps_to_the_weights_this_crate_has() {
    assert_eq!(
        Matrix::from_matrix(ColorMatrix::Unspecified),
        Ok(Matrix::Bt709)
    );
    assert_eq!(Matrix::from_matrix(ColorMatrix::Bt709), Ok(Matrix::Bt709));
    assert_eq!(Matrix::from_matrix(ColorMatrix::Bt601), Ok(Matrix::Bt601));
    assert_eq!(Matrix::from_matrix(ColorMatrix::Bt2020), Ok(Matrix::Bt2020));
    assert_eq!(
        Matrix::from_matrix(ColorMatrix::Identity),
        Ok(Matrix::Identity)
    );
    for other in [
        ColorMatrix::Fcc,
        ColorMatrix::Smpte240m,
        ColorMatrix::Ycgco,
        ColorMatrix::Ictcp,
    ] {
        assert_eq!(Matrix::from_matrix(other), Err(Error::Matrix));
    }
}
