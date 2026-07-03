// solid.wgsl — a solid RGBA fill on a placed quad (backgrounds, mattes, tests).

struct Solid {
    // Straight-alpha fill color (r, g, b, a) in 0..1.
    color: vec4<f32>,
    // Columns of the 2x2 unit-quad → clip-space linear part.
    linear: vec4<f32>,
    // Clip-space translation (x, y), layer opacity (z), pad (w).
    trans_opacity: vec4<f32>,
}

@group(0) @binding(0) var<uniform> s: Solid;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
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
    let m = s.linear;
    let t = s.trans_opacity;
    var out: VertexOutput;
    out.position = vec4(
        m.x * c.x + m.z * c.y + t.x,
        m.y * c.x + m.w * c.y + t.y,
        0.0,
        1.0,
    );
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let o = s.trans_opacity.z;
    return vec4(s.color.rgb, s.color.a * o);
}
