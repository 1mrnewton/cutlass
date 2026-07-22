// yuv.wgsl — YUV 4:2:0 → RGB on a placed quad.
//
// Handles both decoder layouts behind one pipeline (selected by coeffs.w):
//   - planar I420: Y, U, V as three R8 textures.
//   - biplanar NV12: Y as R8, interleaved CbCr as one RG8 texture (bound to the
//     U slot; the V slot is bound to the same texture and left unsampled).
//
// The YUV→RGB matrix is *not* hardcoded: the luma coefficients (Kr, Kb) and the
// range flag come from the frame's own ColorSpace, so BT.601 / BT.709 / BT.2020
// and limited/full range all run through the same non-constant-luminance math.
// Output is the gamma-encoded R'G'B' written straight into an Rgba8Unorm target
// (SDR display-ready); transfer-accurate linearization + primaries adaptation
// for HDR/wide-gamut is a follow-up.

struct Placement {
    // Columns of the 2x2 unit-quad → clip-space linear part (m00, m10, m01, m11).
    linear: vec4<f32>,
    // Clip-space translation (x, y), layer opacity (z), pad (w).
    trans_opacity: vec4<f32>,
    // Sampled UV rect (u0, v0, u1, v1) across the quad.
    uv_rect: vec4<f32>,
    // Kr (x), Kb (y), full-range flag (z: 0=limited, 1=full), plane mode
    // (w: 0=planar I420, 1=biplanar NV12).
    coeffs: vec4<f32>,
    // Color grade: brightness, contrast, saturation, enabled (0 | 1).
    grade_adj0: vec4<f32>,
    // Color grade: exposure, temperature, tint, hue.
    grade_adj1: vec4<f32>,
    // Color grade: highlights, shadows, sharpness, vignette.
    grade_adj2: vec4<f32>,
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> p: Placement;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5),
        vec2(-0.5, 0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
    );
    return corners[vertex_index];
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let c = quad_corner(vertex_index);
    let m = p.linear;
    let t = p.trans_opacity;
    var out: VertexOutput;
    out.position = vec4(
        m.x * c.x + m.z * c.y + t.x,
        m.y * c.x + m.w * c.y + t.y,
        0.0,
        1.0,
    );
    out.uv = mix(p.uv_rect.xy, p.uv_rect.zw, c + vec2(0.5, 0.5));
    out.quad_uv = c + vec2(0.5, 0.5);
    return out;
}

fn yuv_to_rgb(ys: f32, cbs: f32, crs: f32, kr: f32, kb: f32, full: f32) -> vec3<f32> {
    var y: f32;
    var cb: f32;
    var cr: f32;
    if (full > 0.5) {
        // Full / JPEG range: luma spans 0..1, chroma neutral at 128/255.
        y = ys;
        cb = cbs - 128.0 / 255.0;
        cr = crs - 128.0 / 255.0;
    } else {
        // Limited / studio range: luma 16..235, chroma 16..240 (8-bit).
        y = (ys - 16.0 / 255.0) * (255.0 / 219.0);
        cb = (cbs - 128.0 / 255.0) * (255.0 / 224.0);
        cr = (crs - 128.0 / 255.0) * (255.0 / 224.0);
    }
    // Non-constant-luminance YUV→RGB from luma coefficients (Kg = 1 − Kr − Kb).
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = y - (2.0 * (1.0 - kr) * kr / kg) * cr - (2.0 * (1.0 - kb) * kb / kg) * cb;
    return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let ys = textureSample(y_tex, samp, in.uv).r;
    var cbs: f32;
    var crs: f32;
    if (p.coeffs.w > 0.5) {
        // NV12: both chroma samples interleaved in the U-slot texture (RG8).
        let cbcr = textureSample(u_tex, samp, in.uv).rg;
        cbs = cbcr.r;
        crs = cbcr.g;
    } else {
        // I420: separate U and V planes.
        cbs = textureSample(u_tex, samp, in.uv).r;
        crs = textureSample(v_tex, samp, in.uv).r;
    }
    let rgb = yuv_to_rgb(ys, cbs, crs, p.coeffs.x, p.coeffs.y, p.coeffs.z);
    var graded = apply_color_grade(rgb, p.grade_adj0, p.grade_adj1, p.grade_adj2);
    // YUV sharpness: 4-tap laplacian on converted neighbor RGB (same formula).
    let sharp = p.grade_adj2.z;
    if (sharp > 0.0) {
        let dims = vec2<f32>(textureDimensions(y_tex));
        let texel = vec2(1.0) / dims;
        let n0 = sample_yuv_rgb(in.uv + vec2(texel.x, 0.0));
        let n1 = sample_yuv_rgb(in.uv - vec2(texel.x, 0.0));
        let n2 = sample_yuv_rgb(in.uv + vec2(0.0, texel.y));
        let n3 = sample_yuv_rgb(in.uv - vec2(0.0, texel.y));
        let avg = (n0 + n1 + n2 + n3) * 0.25;
        graded = clamp(graded + sharp * 1.5 * (graded - avg), vec3(0.0), vec3(1.0));
    }
    graded = apply_grade_vignette(graded, in.quad_uv, p.grade_adj2.w);
    return vec4(graded, p.trans_opacity.z);
}

fn sample_yuv_rgb(uv: vec2<f32>) -> vec3<f32> {
    let ys = textureSample(y_tex, samp, uv).r;
    var cbs: f32;
    var crs: f32;
    if (p.coeffs.w > 0.5) {
        let cbcr = textureSample(u_tex, samp, uv).rg;
        cbs = cbcr.r;
        crs = cbcr.g;
    } else {
        cbs = textureSample(u_tex, samp, uv).r;
        crs = textureSample(v_tex, samp, uv).r;
    }
    return yuv_to_rgb(ys, cbs, crs, p.coeffs.x, p.coeffs.y, p.coeffs.z);
}
