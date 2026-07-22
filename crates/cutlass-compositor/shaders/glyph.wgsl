// glyph.wgsl — instanced glyph quads sampled from a shared atlas.
//
// Each instance carries its own on-canvas placement (center, size, rotation,
// opacity) and an atlas UV rect. The atlas is uploaded *premultiplied* (same
// contract as rgba.wgsl) so anti-aliased glyph edges blend cleanly. Per-layer
// color grade is applied in the fragment shader; layer opacity multiplies the
// instance opacity.

struct Globals {
    // Color grade: brightness, contrast, saturation, enabled (0 | 1).
    grade_adj0: vec4<f32>,
    // Color grade: exposure, temperature, tint, pad.
    grade_adj1: vec4<f32>,
    // Canvas width/height in pixels (x, y), layer opacity (z), pad (w).
    canvas: vec4<f32>,
}

@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> g: Globals;

struct InstanceIn {
    // center.xy (canvas px), size.xy (canvas px).
    @location(0) center_size: vec4<f32>,
    // rotation radians clockwise (x), opacity (y), pad, pad.
    @location(1) rot_opacity: vec4<f32>,
    // Atlas UV rect (u0, v0, u1, v1).
    @location(2) uv_rect: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opacity: f32,
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5),
        vec2(-0.5, 0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
    );
    return corners[vertex_index];
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32, inst: InstanceIn) -> VertexOutput {
    let c = quad_corner(vertex_index);
    let center = inst.center_size.xy;
    let size = inst.center_size.zw;
    let rotation = inst.rot_opacity.x;
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    // Canvas space (+y down): pos = center + R·(corner ⊙ size).
    let local = vec2(c.x * size.x, c.y * size.y);
    let rotated = vec2(
        cos_r * local.x - sin_r * local.y,
        sin_r * local.x + cos_r * local.y,
    );
    let canvas_pos = center + rotated;
    // Canvas px → clip space: x' = 2x/cw − 1, y' = 1 − 2y/ch.
    let cw = g.canvas.x;
    let ch = g.canvas.y;
    var out: VertexOutput;
    out.position = vec4(
        2.0 * canvas_pos.x / cw - 1.0,
        1.0 - 2.0 * canvas_pos.y / ch,
        0.0,
        1.0,
    );
    out.uv = mix(inst.uv_rect.xy, inst.uv_rect.zw, c + vec2(0.5, 0.5));
    out.opacity = inst.rot_opacity.y * g.canvas.z;
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let premul = textureSample(atlas, samp, in.uv);
    let layer_a = premul.a * in.opacity;
    if layer_a <= 0.0 {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    var rgb = premul.rgb;
    if premul.a > 0.0 {
        rgb = rgb / premul.a;
    }
    rgb = apply_color_grade(rgb, g.grade_adj0, g.grade_adj1);
    return vec4(rgb * layer_a, layer_a);
}
