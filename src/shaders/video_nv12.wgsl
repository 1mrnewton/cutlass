struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

// NV12 planes: full-res Y (R8), half-res interleaved UV (RG8).
@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var cbcr_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

/// When the framebuffer is `*_Srgb`, wgpu applies sRGB encode on write; our YUV→RGB is already
/// display/gamma-coded (708-ish), so we must output **linear** in that case to avoid double encoding.
/// Set `unpack_for_srgb_target` to `1` iff the swapchain attachment uses an sRGB format.
struct Nv12FragParams {
    unpack_for_srgb_target: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(3) var<uniform> nv12_params: Nv12FragParams;

fn srgb_to_linear(rgb: vec3<f32>) -> vec3<f32> {
    let low = rgb <= vec3<f32>(0.04045);
    let lo = rgb / 12.92;
    let hi = pow((rgb + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, low);
}

// BT.709 limited range — common for HD from VideoToolbox.
fn nv12_to_rgb709_limited(uv: vec2<f32>) -> vec3<f32> {
    let y_raw = textureSample(y_tex, samp, uv).r;
    let cbcr = textureSample(cbcr_tex, samp, uv).rg;

    let y = (y_raw - 16.0 / 255.0) * (255.0 / 219.0);
    let cb = (cbcr.r - 128.0 / 255.0) * (255.0 / 224.0);
    let cr = (cbcr.g - 128.0 / 255.0) * (255.0 / 224.0);

    let r = y + 1.5748 * cr;
    let g = y - 0.1873 * cb - 0.4681 * cr;
    let b = y + 1.8556 * cb;
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let rgb_disp = nv12_to_rgb709_limited(in.uv);
    let unpack = nv12_params.unpack_for_srgb_target != 0u;
    let out_rgb = select(rgb_disp, srgb_to_linear(rgb_disp), unpack);
    return vec4<f32>(out_rgb, 1.0);
}
