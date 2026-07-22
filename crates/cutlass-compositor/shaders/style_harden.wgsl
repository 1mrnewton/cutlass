// Harden a blurred silhouette into a near-hard outside stroke (outline).
// Samples the blurred premultiplied silhouette and the original content;
// blur + threshold approximates morphological dilation (effective width is
// capped by the style blur iteration cap — see `run_style_blur`).

struct HardenUniforms {
    // Straight-alpha outline color, 0..1.
    rgba: vec4<f32>,
}

@group(0) @binding(0) var blurred_tex: texture_2d<f32>;
@group(0) @binding(1) var content_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: HardenUniforms;

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
    let blurred = textureSample(blurred_tex, samp, in.uv);
    // Coverage alpha of the blurred premultiplied silhouette.
    let halo_a = blurred.a;
    // Harden: dilated coverage ramps up quickly once any blurred alpha exists.
    let dilated = smoothstep(0.02, 0.35, halo_a);
    // Ring only: cut the part covered by the content itself.
    let content_a = textureSample(content_tex, samp, in.uv).a;
    let ring = dilated * (1.0 - content_a);
    let a = u.rgba.a * ring;
    return vec4<f32>(u.rgba.rgb * a, a);
}
