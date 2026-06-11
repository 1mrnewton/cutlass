// yuv_blit.wgsl — YUV420P → RGBA layer with optional upscale/downscale
//
// Upload Y/U/V as R8Unorm textures (U/V at half resolution). The layer is
// stretched over the whole canvas, so the interpolated fragment UV *is* the
// correct normalized sample position for every plane: `textureSample` already
// handles texel-center alignment internally (sample point uv·size − 0.5 in
// texel space). At 1:1 this lands exactly on texel centers — bit-exact reads,
// no filtering blur. (A previous version recomputed pixel coordinates with an
// extra half-texel offset, which bilinear-blurred every frame even at 1:1.)

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var yuv_sampler: sampler;

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
    out.uv = vec2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn yuv_to_rgb(yv: f32, uv: f32, vv: f32) -> vec3<f32> {
    let y = yv - 16.0;
    let u = uv - 128.0;
    let v = vv - 128.0;
    let r = clamp((298.0 * y + 409.0 * v + 128.0) / 256.0, 0.0, 255.0);
    let g = clamp((298.0 * y - 100.0 * u - 208.0 * v + 128.0) / 256.0, 0.0, 255.0);
    let b = clamp((298.0 * y + 516.0 * u + 128.0) / 256.0, 0.0, 255.0);
    return vec3(r, g, b) / 255.0;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let yv = textureSample(y_tex, yuv_sampler, in.uv).r * 255.0;
    let uv = textureSample(u_tex, yuv_sampler, in.uv).r * 255.0;
    let vv = textureSample(v_tex, yuv_sampler, in.uv).r * 255.0;
    let rgb = yuv_to_rgb(yv, uv, vv);
    return vec4(rgb, 1.0);
}
