// yuv_fx.wgsl — YUV 4:2:0 → RGB with optional mask/chroma effects.

struct Placement {
    linear: vec4<f32>,
    trans_opacity: vec4<f32>,
    uv_rect: vec4<f32>,
    coeffs: vec4<f32>,
}

struct Effects {
    mask: vec4<f32>,
    chroma: vec4<f32>,
    chroma_params: vec4<f32>,
    half: vec4<f32>,
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> p: Placement;
@group(0) @binding(5) var<uniform> fx: Effects;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) local: vec2<f32>,
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5),
        vec2(-0.5, 0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
    );
    return corners[vertex_index];
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let c = quad_corner(vertex_index);
    let m = p.linear;
    let t = p.trans_opacity;
    var out: VertexOutput;
    out.position = vec4(
        m.x * c.x + m.z * c.y + t.x,
        m.y * c.x + m.w * c.y + t.y,
        0.0,
        1.0,
    );
    out.uv = mix(p.uv_rect.xy, p.uv_rect.zw, c + vec2(0.5, 0.5));
    out.local = c * 2.0 * fx.half.xy;
    return out;
}

fn yuv_to_rgb(ys: f32, cbs: f32, crs: f32, kr: f32, kb: f32, full: f32) -> vec3<f32> {
    var y: f32;
    var cb: f32;
    var cr: f32;
    if full > 0.5 {
        y = ys;
        cb = cbs - 128.0 / 255.0;
        cr = crs - 128.0 / 255.0;
    } else {
        y = (ys - 16.0 / 255.0) * (255.0 / 219.0);
        cb = (cbs - 128.0 / 255.0) * (255.0 / 224.0);
        cr = (crs - 128.0 / 255.0) * (255.0 / 224.0);
    }
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = y - (2.0 * (1.0 - kr) * kr / kg) * cr - (2.0 * (1.0 - kb) * kb / kg) * cb;
    return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let ys = textureSample(y_tex, samp, in.uv).r;
    var cbs: f32;
    var crs: f32;
    if p.coeffs.w > 0.5 {
        let cbcr = textureSample(u_tex, samp, in.uv).rg;
        cbs = cbcr.r;
        crs = cbcr.g;
    } else {
        cbs = textureSample(u_tex, samp, in.uv).r;
        crs = textureSample(v_tex, samp, in.uv).r;
    }
    let rgb = yuv_to_rgb(ys, cbs, crs, p.coeffs.x, p.coeffs.y, p.coeffs.z);
    var alpha = p.trans_opacity.z;

    if fx.chroma.w > 0.5 {
        alpha = alpha * chroma_alpha(rgb, fx.chroma.rgb, fx.chroma_params.x, fx.chroma_params.y);
    }

    if fx.mask.w > 0.5 {
        let malpha = mask_alpha(
            in.local,
            fx.half.xy,
            u32(fx.mask.x + 0.5),
            fx.mask.y,
            fx.mask.z,
        );
        alpha = alpha * malpha;
    }

    return vec4(rgb, alpha);
}
