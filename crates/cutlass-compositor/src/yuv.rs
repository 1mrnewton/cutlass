//! YUV420P plane buffers shared by GPU upload and readback.

/// Tight or strided YUV420P planes (decoder layout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420pLayer {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub y_stride: u32,
    pub u: Vec<u8>,
    pub u_stride: u32,
    pub v: Vec<u8>,
    pub v_stride: u32,
    /// Samples are full (JPEG, 0–255) range rather than limited (16–235); the
    /// GPU YUV→RGB shader picks its matrix from this. Defaults to `false`
    /// (limited), the standard for video; set true for `yuvj420p` sources.
    pub full_range: bool,
}

impl Yuv420pLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        y: Vec<u8>,
        y_stride: u32,
        u: Vec<u8>,
        u_stride: u32,
        v: Vec<u8>,
        v_stride: u32,
    ) -> Self {
        Self {
            width,
            height,
            y,
            y_stride,
            u,
            u_stride,
            v,
            v_stride,
            full_range: false,
        }
    }

    /// Mark these planes as full- or limited-range (builder style).
    pub fn with_full_range(mut self, full_range: bool) -> Self {
        self.full_range = full_range;
        self
    }

    /// Copy each row to a tight buffer suitable for `wgpu` texture upload.
    pub fn tight_y(&self) -> Vec<u8> {
        tight_plane(&self.y, self.width, self.height, self.y_stride)
    }

    pub fn tight_u(&self) -> Vec<u8> {
        let w = self.width / 2;
        let h = self.height / 2;
        tight_plane(&self.u, w, h, self.u_stride)
    }

    pub fn tight_v(&self) -> Vec<u8> {
        let w = self.width / 2;
        let h = self.height / 2;
        tight_plane(&self.v, w, h, self.v_stride)
    }
}

/// YUV420P read back from the GPU rgba→yuv pass (tight planes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420pImage {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

pub(crate) fn tight_plane(data: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let stride = stride as usize;
    let mut out = vec![0u8; w * h];
    for row in 0..h {
        let src = row * stride;
        let dst = row * w;
        out[dst..dst + w].copy_from_slice(&data[src..src + w]);
    }
    out
}

/// Legacy CPU RGBA8 → YUV420P (BT.601), for tests and fallback comparisons.
pub fn legacy_rgba_to_yuv420p(rgba: &[u8], width: u32, height: u32) -> Yuv420pImage {
    let w = width as usize;
    let h = height as usize;
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; (w / 2) * (h / 2)];
    let mut v_plane = vec![0u8; (w / 2) * (h / 2)];

    for row in 0..h {
        for col in 0..w {
            let i = (row * w + col) * 4;
            let r = i32::from(rgba[i]);
            let g = i32::from(rgba[i + 1]);
            let b = i32::from(rgba[i + 2]);
            y_plane[row * w + col] =
                (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(16, 235) as u8;
        }
    }

    for row in (0..h).step_by(2) {
        for col in (0..w).step_by(2) {
            let mut r_sum = 0i32;
            let mut g_sum = 0i32;
            let mut b_sum = 0i32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = ((row + dy) * w + (col + dx)) * 4;
                    r_sum += i32::from(rgba[i]);
                    g_sum += i32::from(rgba[i + 1]);
                    b_sum += i32::from(rgba[i + 2]);
                }
            }
            let r = r_sum / 4;
            let g = g_sum / 4;
            let b = b_sum / 4;
            let u = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(16, 240) as u8;
            let v = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(16, 240) as u8;
            let uv_row = row / 2;
            let uv_col = col / 2;
            u_plane[uv_row * (w / 2) + uv_col] = u;
            v_plane[uv_row * (w / 2) + uv_col] = v;
        }
    }

    Yuv420pImage {
        width,
        height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}
