// blit.wgsl — full-canvas textured layer (RGBA upload)
//
// Used by CompositeLayer::Rgba: a CPU-decoded/resized RGBA8 canvas uploaded as a
// GPU texture each frame. Samples with linear filtering (mostly irrelevant at
// 1:1 canvas size; kept for future scaled layers).
//
// Pipeline: compositor.rs `blit_pipeline`
//   - Same render target and src-over blend as solid.wgsl
//   - Layer textures are Rgba8Unorm, COPY_DST + TEXTURE_BINDING
//
// UV mapping: clip-space Y is flipped so texture row 0 is the top of the image
// (matches engine RGBA row-major layout).

@group(0) @binding(0) var layer_tex: texture_2d<f32>;
@group(0) @binding(1) var layer_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4(x, y, 0.0, 1.0);
    // Map clip [-1,1] to UV [0,1]; flip Y for top-left image origin.
    out.uv = vec2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(layer_tex, layer_sampler, in.uv);
}
