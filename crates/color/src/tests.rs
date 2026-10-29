use cutlass_codec::access::{
    ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer, Yuv420p,
};

use crate::{
    Cube, Curve, Depth, Error, Input, Layout, Lut, LutIssue, Matrix, Outcome, Picture,
    PlatformView, Request, SegmentHdr, ThreeDl, apply_lut, convert, curve_from_codec,
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
    picture.primaries = ColorPrimaries::Unspecified;
    picture.matrix = Matrix::Bt601;
    assert_eq!(at(picture, &request, 0, 0), [247, 21, 0, 255]);
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
