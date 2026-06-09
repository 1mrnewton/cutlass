// solid.wgsl — full-canvas solid fill layer
//
// Used by CompositeLayer::Solid (e.g. Generator::SolidColor clips). Draws a
// single RGBA color across the entire render target using a 3-vertex fullscreen
// triangle (no vertex buffer).
//
// Pipeline: compositor.rs `solid_pipeline`
//   - Render target: Rgba8Unorm offscreen texture
//   - Load: Clear transparent on first layer, then Load for subsequent layers
//   - Blend (configured in Rust, not here): src-over
//       color:  SrcAlpha * src + (1 - SrcAlpha) * dst
//       alpha:  1 * src.a + (1 - SrcAlpha) * dst.a
//
// Input color is straight (non-premultiplied) RGBA in 0..1, uploaded from
// engine as u8 0–255 and normalized in compositor.rs.

struct Uniforms {
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

// Fullscreen triangle: vertex_index 0..2 cover clip space without a VBO.
// Positions: (-1,-1), (3,-1), (-1,3) — only the [-1,1] region is visible.
@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return uniforms.color;
}
