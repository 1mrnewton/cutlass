@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle + UVs that interpolate to (0,0)–(1,1) over the viewport.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );
    var out: VertexOutput;
    let p = positions[vid];
    out.clip_position = vec4<f32>(p.x, p.y, 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Flip Y so PNG row 0 (top) matches typical texture sampling with NDC y-up.
    let uv_tex = vec2<f32>(uv.x, 1.0 - uv.y);
    return textureSample(tex, samp, uv_tex);
}
