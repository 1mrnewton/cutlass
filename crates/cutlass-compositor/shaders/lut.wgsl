// lut.wgsl — fullscreen 3D-LUT pass (.cube color lookup).
//
// The input is a premultiplied offscreen texture (a layer drawn offscreen or
// a canvas snapshot). Color is un-premultiplied, mapped through the LUT with
// trilinear sampling, blended back over the original by `intensity`, and
// re-premultiplied so the result composites like any other offscreen pass.
//
// `params0` = intensity, lut size N, pad, pad.
// `domain_lo` / `domain_scale` remap input color into LUT texture space:
// uvw = (c - lo) * scale, then shrunk to texel centers so the edge grid
// points land exactly on the first/last texels.

struct LutUniforms {
    params0: vec4<f32>,
    domain_lo: vec4<f32>,
    domain_scale: vec4<f32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var lut_tex: texture_3d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: LutUniforms;

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

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureSample(input_tex, input_sampler, in.uv);
    if (src.a <= 0.0) {
        return src;
    }
    let straight = clamp(src.rgb / src.a, vec3<f32>(0.0), vec3<f32>(1.0));

    let n = uniforms.params0.y;
    let unit = clamp(
        (straight - uniforms.domain_lo.rgb) * uniforms.domain_scale.rgb,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    // Map [0,1] onto texel centers: 0 -> 0.5/N, 1 -> (N-0.5)/N.
    let coord = (unit * (n - 1.0) + vec3<f32>(0.5)) / n;
    let graded = textureSampleLevel(lut_tex, lut_sampler, coord, 0.0).rgb;

    let mixed = mix(straight, graded, uniforms.params0.x);
    return vec4<f32>(mixed * src.a, src.a);
}
