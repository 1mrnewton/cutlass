// rgba_fx.wgsl — premultiplied RGBA bitmap with optional mask/chroma effects.

struct Placement {
    linear: vec4<f32>,
    trans_opacity: vec4<f32>,
    uv_rect: vec4<f32>,
    // Color grade: brightness, contrast, saturation, enabled (0 | 1).
    grade_adj0: vec4<f32>,
    // Color grade: exposure, temperature, tint, hue.
    grade_adj1: vec4<f32>,
    // Color grade: highlights, shadows, sharpness, vignette.
    grade_adj2: vec4<f32>,
}

struct Effects {
    // mask_kind, mask_feather, mask_invert (0/1), mask_enabled (0/1)
    mask: vec4<f32>,
    // chroma_rgb (normalized), chroma_enabled (0/1)
    chroma: vec4<f32>,
    // chroma_strength, chroma_shadow, pad, pad
    chroma_params: vec4<f32>,
    // quad half-extents px (x, y), mask rotation_rad (z), mask roundness (w)
    half: vec4<f32>,
    // mask center xy (fraction of layer size), mask size xy (fraction)
    mask_geo: vec4<f32>,
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> p: Placement;
@group(0) @binding(3) var<uniform> fx: Effects;

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

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    var premul = textureSample(tex, samp, in.uv);

    if fx.chroma.w > 0.5 {
        let a = premul.a;
        var straight = vec3(0.0);
        if a > 1e-4 {
            straight = premul.rgb / a;
        }
        let chroma_mul = chroma_alpha(straight, fx.chroma.rgb, fx.chroma_params.x, fx.chroma_params.y);
        premul = premul * chroma_mul;
    }

    // Grade the straight-alpha color after chroma keying (the key targets the
    // source footage), before mask/opacity shape the alpha.
    if premul.a > 0.0 {
        var graded = apply_color_grade(premul.rgb / premul.a, p.grade_adj0, p.grade_adj1, p.grade_adj2);
        graded = apply_grade_sharpness(graded, in.uv, tex, samp, p.grade_adj2.z);
        graded = apply_grade_vignette(graded, in.quad_uv, p.grade_adj2.w);
        premul = vec4(graded * premul.a, premul.a);
    }

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
        premul = premul * malpha;
    }

    return premul * p.trans_opacity.z;
}
