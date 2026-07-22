// Weighted full-canvas blit for motion-blur accumulation.
// Samples are premultiplied; the pipeline blends One/One so
// `accum += sample * weight` with weight = 1/N.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Weight {
    // xyz unused; w = weight
    value: vec4<f32>,
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> weight: Weight;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
    var o: VsOut;
    o.pos = vec4(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * weight.value.x;
}
