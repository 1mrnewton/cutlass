// Separable vertical box blur; radius in pixels (params[0].x).

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
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

fn sample_at(uv: vec2<f32>) -> vec4<f32> {
    return textureSample(input_tex, input_sampler, uv);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = max(uniforms.params[0].x, 0.0);
    if (r <= 0.0) {
        return sample_at(in.uv);
    }
    let step = uniforms.texel_size.xy;
    let taps = min(i32(r), 16);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -taps; i <= taps; i++) {
        let w = 1.0;
        let uv = in.uv + vec2<f32>(0.0, f32(i) * step.y);
        acc += sample_at(uv) * w;
        wsum += w;
    }
    return acc / wsum;
}
