// yuv_fx.wgsl — YUV 4:2:0 → RGB with optional mask/chroma effects.

struct Placement {
    linear: vec4<f32>,
    trans_opacity: vec4<f32>,
    uv_rect: vec4<f32>,
    coeffs: vec4<f32>,
    // Color grade: brightness, contrast, saturation, enabled (0 | 1).
    grade_adj0: vec4<f32>,
    // Color grade: exposure, temperature, tint, hue.
    grade_adj1: vec4<f32>,
    // Color grade: highlights, shadows, sharpness, vignette.
    grade_adj2: vec4<f32>,
}

struct Effects {
    mask: vec4<f32>,
    chroma: vec4<f32>,
    chroma_params: vec4<f32>,
    // half.xy, mask rotation_rad (z), mask roundness (w)
    half: vec4<f32>,
    // mask center xy, mask size xy
    mask_geo: vec4<f32>,
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
    @location(2) quad_uv: vec2<f32>,
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
    out.quad_uv = c + vec2(0.5, 0.5);
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
    var rgb = yuv_to_rgb(ys, cbs, crs, p.coeffs.x, p.coeffs.y, p.coeffs.z);
    var alpha = p.trans_opacity.z;

    // Chroma-key on the ungraded color (the key targets the source footage),
    // then grade the RGB, then mask/opacity shape the alpha.
    if fx.chroma.w > 0.5 {
        alpha = alpha * chroma_alpha(rgb, fx.chroma.rgb, fx.chroma_params.x, fx.chroma_params.y);
    }

    rgb = apply_color_grade(rgb, p.grade_adj0, p.grade_adj1, p.grade_adj2);
    let sharp = p.grade_adj2.z;
    if (sharp > 0.0) {
        let dims = vec2<f32>(textureDimensions(y_tex));
        let texel = vec2(1.0) / dims;
        let n0 = sample_yuv_rgb_fx(in.uv + vec2(texel.x, 0.0));
        let n1 = sample_yuv_rgb_fx(in.uv - vec2(texel.x, 0.0));
        let n2 = sample_yuv_rgb_fx(in.uv + vec2(0.0, texel.y));
        let n3 = sample_yuv_rgb_fx(in.uv - vec2(0.0, texel.y));
        let avg = (n0 + n1 + n2 + n3) * 0.25;
        rgb = clamp(rgb + sharp * 1.5 * (rgb - avg), vec3(0.0), vec3(1.0));
    }
    rgb = apply_grade_vignette(rgb, in.quad_uv, p.grade_adj2.w);

    if fx.mask.w > 0.5 {
        let malpha = mask_alpha(
            in.local,
            fx.half.xy,
            u32(fx.mask.x + 0.5),
            fx.mask.y,
            fx.mask.z,
            fx.mask_geo.xy,
            fx.mask_geo.zw,
            fx.half.z,
            fx.half.w,
        );
        alpha = alpha * malpha;
    }

    return vec4(rgb, alpha);
}

fn sample_yuv_rgb_fx(uv: vec2<f32>) -> vec3<f32> {
    let ys = textureSample(y_tex, samp, uv).r;
    var cbs: f32;
    var crs: f32;
    if p.coeffs.w > 0.5 {
        let cbcr = textureSample(u_tex, samp, uv).rg;
        cbs = cbcr.r;
        crs = cbcr.g;
    } else {
        cbs = textureSample(u_tex, samp, uv).r;
        crs = textureSample(v_tex, samp, uv).r;
    }
    return yuv_to_rgb(ys, cbs, crs, p.coeffs.x, p.coeffs.y, p.coeffs.z);
}
