//! YUV 4:2:0 to RGB, the way `dwebp` and the browsers built on libwebp
//! show a lossy WebP: chroma upsampled with the "fancy" 9-3-3-1 filter and
//! BT.601 limited-range coefficients in 14-bit fixed point.

/// Decoded planes, cropped to the picture.
pub(crate) struct Planes {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub y_stride: usize,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub uv_stride: usize,
}

fn mult_hi(v: i32, coeff: i32) -> i32 {
    (v * coeff) >> 8
}

fn clip8(v: i32) -> u8 {
    if (0..256 << 6).contains(&v) {
        (v >> 6) as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

fn rgb(y: u8, u: i32, v: i32) -> [u8; 3] {
    let y = mult_hi(i32::from(y), 19_077);
    [
        clip8(y + mult_hi(v, 26_149) - 14_234),
        clip8(y - mult_hi(u, 6_419) - mult_hi(v, 13_320) + 8_708),
        clip8(y + mult_hi(u, 33_050) - 17_685),
    ]
}

/// The chroma rows nearest to one luma row: the first weighs 3, the other 1.
struct ChromaRows<'a> {
    near_u: &'a [u8],
    near_v: &'a [u8],
    far_u: &'a [u8],
    far_v: &'a [u8],
}

/// One output row. Each output pixel weighs the nearest chroma sample 9,
/// its two neighbours 3, and the diagonal 1, in libwebp's rounding order.
fn upsample_row(y_row: &[u8], rows: &ChromaRows<'_>, width: usize, out: &mut [u8]) {
    let sample = |x: usize| {
        (
            [i32::from(rows.near_u[x]), i32::from(rows.near_v[x])],
            [i32::from(rows.far_u[x]), i32::from(rows.far_v[x])],
        )
    };
    let edge = |near: [i32; 2], far: [i32; 2]| {
        [
            (3 * near[0] + far[0] + 2) >> 2,
            (3 * near[1] + far[1] + 2) >> 2,
        ]
    };
    let put = |out: &mut [u8], x: usize, uv: [i32; 2]| {
        let [r, g, b] = rgb(y_row[x], uv[0], uv[1]);
        out[x * 4..x * 4 + 4].copy_from_slice(&[r, g, b, 255]);
    };
    let (mut left_near, mut left_far) = sample(0);
    put(out, 0, edge(left_near, left_far));
    let pairs = (width - 1) >> 1;
    for x in 1..=pairs {
        let (near, far) = sample(x);
        let mut uv0 = [0; 2];
        let mut uv1 = [0; 2];
        for c in 0..2 {
            let avg = left_far[c] + far[c] + left_near[c] + near[c] + 8;
            // Each output pixel averages its nearest sample with the
            // diagonal pair that does not include it.
            let diag_left = (avg + 2 * (left_far[c] + near[c])) >> 3;
            let diag_right = (avg + 2 * (far[c] + left_near[c])) >> 3;
            uv0[c] = (diag_left + left_near[c]) >> 1;
            uv1[c] = (diag_right + near[c]) >> 1;
        }
        put(out, 2 * x - 1, uv0);
        put(out, 2 * x, uv1);
        left_near = near;
        left_far = far;
    }
    if width & 1 == 0 {
        put(out, width - 1, edge(left_near, left_far));
    }
}

/// Straight RGBA, opaque.
pub(crate) fn to_rgba(planes: &Planes) -> Vec<u8> {
    let (width, height) = (planes.width, planes.height);
    let mut rgba = vec![0u8; width * height * 4];
    let uv_row = |k: usize| {
        let start = k * planes.uv_stride;
        (
            &planes.u[start..start + planes.uv_stride],
            &planes.v[start..start + planes.uv_stride],
        )
    };
    for (row, out) in rgba.chunks_mut(width * 4).enumerate() {
        // Row 2k - 1 sits nearer chroma row k - 1, row 2k nearer row k.
        // The first row, and the last of an even height, pair a chroma row
        // with itself.
        let (near_index, far_index) = if row == 0 {
            (0, 0)
        } else if row % 2 == 1 {
            (row / 2, (row / 2 + 1).min((height - 1) / 2))
        } else {
            (row / 2, row / 2 - 1)
        };
        let (near_u, near_v) = uv_row(near_index);
        let (far_u, far_v) = uv_row(far_index);
        let y_row = &planes.y[row * planes.y_stride..row * planes.y_stride + width];
        upsample_row(
            y_row,
            &ChromaRows {
                near_u,
                near_v,
                far_u,
                far_v,
            },
            width,
            out,
        );
    }
    rgba
}
