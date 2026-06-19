// yuv_blit.wgsl — YUV420P → RGBA placed quad with scale/rotation
//
// Upload Y/U/V as R8Unorm textures (U/V at half resolution) and draw them as
// a positioned quad (LayerPlacement). The interpolated fragment UV is the
// correct normalized sample position for every plane: `textureSample` already
// handles texel-center alignment internally (sample point uv·size − 0.5 in
// texel space). At 1:1 full-canvas placement this lands exactly on texel
// centers — bit-exact reads, no filtering blur. (A previous version
// recomputed pixel coordinates with an extra half-texel offset, which
// bilinear-blurred every frame even at 1:1.)
//
// Geometry matches blit.wgsl: unit quad ([-0.5, 0.5]², +y down) mapped to
// clip space by the per-layer affine; corner (-0.5, -0.5) = content top-left
// = UV (0, 0). Layer opacity rides the output alpha (src-over blend).

struct Placement {
    // Columns of the 2x2 linear part mapping unit-quad corners to clip
    // space: (m00, m10, m01, m11).
    linear: vec4<f32>,
    // Clip-space translation (x, y), layer opacity, and a full-range flag
    // (w: 0 = limited/studio 16-235, 1 = full/JPEG 0-255).
    trans_opacity: vec4<f32>,
    // Content UV rect (u0, v0, u1, v1) interpolated across the quad:
    // sub-rects crop, reversed axes mirror (flip H/V).
    uv_rect: vec4<f32>,
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var yuv_sampler: sampler;
@group(0) @binding(4) var<uniform> placement: Placement;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
    let m = placement.linear;
    let t = placement.trans_opacity;
    var out: VertexOutput;
    out.position = vec4(
        m.x * c.x + m.z * c.y + t.x,
        m.y * c.x + m.w * c.y + t.y,
        0.0,
        1.0,
    );
    out.uv = mix(placement.uv_rect.xy, placement.uv_rect.zw, c + vec2(0.5, 0.5));
    return out;
}

fn yuv_to_rgb(yv: f32, uv: f32, vv: f32, full_range: f32) -> vec3<f32> {
    let u = uv - 128.0;
    let v = vv - 128.0;
    if (full_range > 0.5) {
        // Full-range (JPEG) BT.601: luma already spans 0..255, no 16/219
        // rescale. Same matrix basis as the limited path, just unscaled.
        let y = yv;
        let r = clamp(y + 1.402 * v, 0.0, 255.0);
        let g = clamp(y - 0.344136 * u - 0.714136 * v, 0.0, 255.0);
        let b = clamp(y + 1.772 * u, 0.0, 255.0);
        return vec3(r, g, b) / 255.0;
    }
    // Limited-range (studio) BT.601: y - 16, scaled by 298/256 ≈ 255/219.
    let y = yv - 16.0;
    let r = clamp((298.0 * y + 409.0 * v + 128.0) / 256.0, 0.0, 255.0);
    let g = clamp((298.0 * y - 100.0 * u - 208.0 * v + 128.0) / 256.0, 0.0, 255.0);
    let b = clamp((298.0 * y + 516.0 * u + 128.0) / 256.0, 0.0, 255.0);
    return vec3(r, g, b) / 255.0;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let yv = textureSample(y_tex, yuv_sampler, in.uv).r * 255.0;
    let uv = textureSample(u_tex, yuv_sampler, in.uv).r * 255.0;
    let vv = textureSample(v_tex, yuv_sampler, in.uv).r * 255.0;
    let rgb = yuv_to_rgb(yv, uv, vv, placement.trans_opacity.w);
    return vec4(rgb, placement.trans_opacity.z);
}
