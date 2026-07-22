// Dst-sampling blend composite: combine a premultiplied source layer with a
// premultiplied canvas snapshot per W3C Compositing and Blending Level 1.
// Output is premultiplied; the pipeline disables fixed-function blending
// (Replace) because this pass computes the final composition itself.

struct BlendUniforms {
    // mode: 0=Normal … 12=Exclusion (see BlendMode::shader_id). xyzw pad to 16.
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var dst_tex: texture_2d<f32>;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: BlendUniforms;

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

fn unpremultiply(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return c.rgb / c.a;
}

fn multiply(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return cb * cs;
}

fn screen(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return cb + cs - cb * cs;
}

fn color_burn_ch(cb: f32, cs: f32) -> f32 {
    if (cb == 1.0) {
        return 1.0;
    }
    if (cs == 0.0) {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - cb) / cs);
}

fn color_dodge_ch(cb: f32, cs: f32) -> f32 {
    if (cb == 0.0) {
        return 0.0;
    }
    if (cs == 1.0) {
        return 1.0;
    }
    return min(1.0, cb / (1.0 - cs));
}

fn soft_light_d(cb: f32) -> f32 {
    if (cb <= 0.25) {
        return ((16.0 * cb - 12.0) * cb + 4.0) * cb;
    }
    return sqrt(cb);
}

fn soft_light_ch(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) {
        return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
    }
    return cb + (2.0 * cs - 1.0) * (soft_light_d(cb) - cb);
}

fn hard_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return select(
        screen(cb, 2.0 * cs - vec3<f32>(1.0)),
        multiply(cb, 2.0 * cs),
        cs <= vec3<f32>(0.5),
    );
}

fn blend_fn(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1u: { // Darken
            return min(cb, cs);
        }
        case 2u: { // Multiply
            return multiply(cb, cs);
        }
        case 3u: { // ColorBurn
            return vec3<f32>(
                color_burn_ch(cb.r, cs.r),
                color_burn_ch(cb.g, cs.g),
                color_burn_ch(cb.b, cs.b),
            );
        }
        case 4u: { // Lighten
            return max(cb, cs);
        }
        case 5u: { // Screen
            return screen(cb, cs);
        }
        case 6u: { // ColorDodge
            return vec3<f32>(
                color_dodge_ch(cb.r, cs.r),
                color_dodge_ch(cb.g, cs.g),
                color_dodge_ch(cb.b, cs.b),
            );
        }
        case 7u: { // Add (linear dodge)
            return min(vec3<f32>(1.0), cb + cs);
        }
        case 8u: { // Overlay = HardLight with args swapped
            return hard_light(cs, cb);
        }
        case 9u: { // SoftLight
            return vec3<f32>(
                soft_light_ch(cb.r, cs.r),
                soft_light_ch(cb.g, cs.g),
                soft_light_ch(cb.b, cs.b),
            );
        }
        case 10u: { // HardLight
            return hard_light(cb, cs);
        }
        case 11u: { // Difference
            return abs(cb - cs);
        }
        case 12u: { // Exclusion
            return cb + cs - 2.0 * cb * cs;
        }
        default: { // Normal / unknown: source color (no blend)
            return cs;
        }
    }
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let dst_pm = textureSample(dst_tex, samp, in.uv);
    let src_pm = textureSample(src_tex, samp, in.uv);
    let ab = dst_pm.a;
    let as_ = src_pm.a;
    let cb = unpremultiply(dst_pm);
    let cs = unpremultiply(src_pm);

    let b = blend_fn(u.mode, cb, cs);
    // W3C: Cs = (1 - αb) × Cs + αb × B(Cb, Cs); then simple alpha composite.
    let cs_mixed = (1.0 - ab) * cs + ab * b;
    let co = cs_mixed * as_ + cb * ab * (1.0 - as_);
    let ao = as_ + ab * (1.0 - as_);
    return vec4<f32>(co, ao);
}
