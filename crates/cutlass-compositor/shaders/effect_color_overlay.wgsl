// Color overlay tint with optional UV offset.
// Flatten (catalog order): color(4) @ params[0], offset.xy + amount @ params[1].xy/z.

struct EffectUniforms {
    texel_size: vec4<f32>,
    params: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: EffectUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let overlay = uniforms.params[0];
    let offset = uniforms.params[1].xy;
    let amount = clamp(uniforms.params[1].z, 0.0, 1.0);

    let src = textureSample(input_tex, input_sampler, in.uv - offset);
    let a = src.a;
    if (a <= 1e-5) {
        return vec4<f32>(0.0);
    }
    // Offscreen effect chain stores premultiplied RGBA.
    let straight = src.rgb / a;
    let t = clamp(amount * overlay.a, 0.0, 1.0);
    let mixed = mix(straight, overlay.rgb, t);
    return vec4<f32>(mixed * a, a);
}
