// Map luminance onto a shadow→highlight color ramp.
// Flatten (catalog order): shadow_color(4) @ params[0], highlight_color(4) @ params[1],
// intensity @ params[2].x.

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
    let shadow = uniforms.params[0];
    let highlight = uniforms.params[1];
    let intensity = clamp(uniforms.params[2].x, 0.0, 1.0);

    let src = textureSample(input_tex, input_sampler, in.uv);
    let a = src.a;
    if (a <= 1e-5) {
        return vec4<f32>(0.0);
    }
    let straight = src.rgb / a;
    let luma = dot(straight, vec3<f32>(0.299, 0.587, 0.114));
    let duo = mix(shadow.rgb, highlight.rgb, clamp(luma, 0.0, 1.0));
    let mixed = mix(straight, duo, intensity);
    return vec4<f32>(mixed * a, a);
}
