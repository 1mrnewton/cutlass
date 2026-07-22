// grade.wgsl — per-layer color grade in the fragment shader (no extra pass).
//
// Each pipeline appends `grade_adj0` / `grade_adj1` / `grade_adj2` to its uniform:
//   adj0 = brightness, contrast, saturation, enabled (0 | 1)
//   adj1 = exposure, temperature, tint, hue
//   adj2 = highlights, shadows, sharpness, vignette
//
// Params are signed strengths in roughly [-1, 1] (sharpness/vignette in [0, 1]);
// `enabled == 0` is the identity fast path (one branch, no math).
//
// Sharpness (4-tap laplacian) and vignette (radial darken) need UVs / texels and
// are applied by the textured callers via `apply_grade_sharpness` /
// `apply_grade_vignette` after this per-pixel grade.

fn apply_color_grade(rgb: vec3<f32>, adj0: vec4<f32>, adj1: vec4<f32>, adj2: vec4<f32>) -> vec3<f32> {
    if (adj0.w < 0.5) {
        return rgb;
    }
    var c = rgb;
    c *= exp2(2.0 * adj1.x);
    c.r += adj1.y * 0.25;
    c.b -= adj1.y * 0.25;
    c.g += adj1.z * 0.25;
    c += adj0.x * 0.25;
    c = (c - vec3(0.5)) * (1.0 + adj0.y) + vec3(0.5);
    let luma0 = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma0), c, 1.0 + adj0.z);

    // YIQ hue rotation: adj1.w (= hue) * 30° at ±1.
    let hue = adj1.w;
    if (abs(hue) > 1e-6) {
        let y = dot(c, vec3(0.299, 0.587, 0.114));
        let i = dot(c, vec3(0.596, -0.274, -0.322));
        let q = dot(c, vec3(0.211, -0.523, 0.312));
        // Negated so +hue rotates red toward yellow (standard editor feel).
        let angle = -hue * 0.5235987755982988; // −π/6 per unit
        let cos_a = cos(angle);
        let sin_a = sin(angle);
        let i2 = i * cos_a - q * sin_a;
        let q2 = i * sin_a + q * cos_a;
        c = vec3(
            y + 0.956 * i2 + 0.621 * q2,
            y - 0.272 * i2 - 0.647 * q2,
            y - 1.106 * i2 + 1.703 * q2,
        );
    }

    let luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    let hl = adj2.x;
    if (abs(hl) > 1e-6) {
        let w = smoothstep(0.5, 1.0, luma);
        let lift = select(c, vec3(1.0) - c, hl >= 0.0);
        c += vec3(hl * w * 0.35) * lift;
    }
    let sh = adj2.y;
    if (abs(sh) > 1e-6) {
        let w = 1.0 - smoothstep(0.0, 0.5, luma);
        let lift = select(c, vec3(1.0) - c, sh >= 0.0);
        c += vec3(sh * w * 0.35) * lift;
    }

    return clamp(c, vec3(0.0), vec3(1.0));
}

/// 4-tap unsharp mask on a straight-alpha RGBA texture. `amount` is adj2.z.
fn apply_grade_sharpness(
    rgb: vec3<f32>,
    uv: vec2<f32>,
    tex: texture_2d<f32>,
    samp: sampler,
    amount: f32,
) -> vec3<f32> {
    if (amount <= 0.0) {
        return rgb;
    }
    let dims = vec2<f32>(textureDimensions(tex));
    let texel = vec2(1.0) / dims;
    let s0 = textureSample(tex, samp, uv + vec2(texel.x, 0.0));
    let s1 = textureSample(tex, samp, uv - vec2(texel.x, 0.0));
    let s2 = textureSample(tex, samp, uv + vec2(0.0, texel.y));
    let s3 = textureSample(tex, samp, uv - vec2(0.0, texel.y));
    var n0 = s0.rgb;
    if (s0.a > 1e-4) { n0 = s0.rgb / s0.a; }
    var n1 = s1.rgb;
    if (s1.a > 1e-4) { n1 = s1.rgb / s1.a; }
    var n2 = s2.rgb;
    if (s2.a > 1e-4) { n2 = s2.rgb / s2.a; }
    var n3 = s3.rgb;
    if (s3.a > 1e-4) { n3 = s3.rgb / s3.a; }
    let avg = (n0 + n1 + n2 + n3) * 0.25;
    return clamp(rgb + amount * 1.5 * (rgb - avg), vec3(0.0), vec3(1.0));
}

/// Radial vignette from layer-UV center. `amount` is adj2.w.
/// `uv` is in 0..1 across the layer quad (or equivalent normalized coords).
fn apply_grade_vignette(rgb: vec3<f32>, uv: vec2<f32>, amount: f32) -> vec3<f32> {
    if (amount <= 0.0) {
        return rgb;
    }
    let dist_norm = length(uv - vec2(0.5)) / (0.5 * sqrt(2.0));
    return rgb * (1.0 - amount * smoothstep(0.4, 0.9, dist_norm));
}
