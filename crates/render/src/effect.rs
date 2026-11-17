//! Effect passes: one shader that runs any block of a recipe.
//!
//! An effect layer applies to everything drawn below it. Each step reads
//! the picture so far and writes a new one of the same size; two textures
//! take turns. Colours stay premultiplied between steps; a step that works
//! on colour divides by alpha first and multiplies again after. Sampling is
//! linear and mirrors at the edges, so a shaken or turned frame shows a
//! reflection at its border instead of a hole. A blur runs as two steps,
//! across and then down.
//!
//! The steps run on half floats, so nothing is rounded to bytes between
//! them, and each block works in the values its arithmetic is written for
//! ([`space`]). A blur averages linear light, so a bright edge stays
//! bright; a grade or a colour look was drawn by eye on Rec. 709 encoded
//! values, as the canvas holds them, and is handed those. The picture is
//! converted as a whole, once, wherever the chain moves from one to the
//! other: a run of colour blocks costs no more than before, and none of
//! them pays for the curve on every tap. A chain that starts on encoded
//! values reads the canvas picture itself, and the last draw writes the
//! canvas format itself, encoding light on the way, so a chain of colour
//! blocks draws no more passes than it has blocks. The curve is the Rec. 709 OETF and its inverse, the one the
//! layer shader converts transfers with.
//!
//! A step may keep its result for a later blend to read. Keeping costs no
//! copy: the picture is set aside as it is, in whichever values the chain
//! then holds, and the chain goes on in a new texture. The picture as it
//! entered the effect is always kept, as the canvas bytes it came in.

use crate::pool::{Bind, TextureKey};
use cutlass_effects::{Block, MAX_SAVED, MAX_VALUES, Step};

use crate::CANVAS_FORMAT;

/// The format the steps draw into: linear light needs more than eight
/// bits, and values past white survive between steps.
pub(crate) const LIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// What a block's arithmetic is written for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Space {
    /// Linear light: blocks that average or add light.
    Light,
    /// Rec. 709 encoded values, as the canvas holds them: blocks whose
    /// numbers were tuned by eye on an encoded picture.
    Display,
}

/// The values `block`'s arithmetic is written for. None for a block that
/// only moves pixels: it runs on whatever the chain holds, so it never
/// costs a conversion.
pub(crate) const fn space(block: Block) -> Option<Space> {
    match block {
        // Averages of light, and light laid over light.
        Block::Blur
        | Block::ZoomBlur
        | Block::MotionBlur
        | Block::Focus
        | Block::Glow
        | Block::Blend => Some(Space::Light),
        // Pixels moved, not mixed.
        Block::Transform
        | Block::RgbSplit
        | Block::Slices
        | Block::Mirror
        | Block::Pixelate
        | Block::Swirl
        | Block::Ripple
        | Block::Bulge
        | Block::Tile
        | Block::Grid
        | Block::Stretch
        | Block::Blocks
        | Block::Kaleido
        | Block::Fringe => None,
        // Colour tuned by eye, and light whose thresholds and strengths
        // were tuned on encoded values.
        Block::Grade
        | Block::Flash
        | Block::Vignette
        | Block::Noise
        | Block::Scanlines
        | Block::Invert
        | Block::LightLeak
        | Block::Hue
        | Block::Posterize
        | Block::Edges
        | Block::GradientMap
        | Block::Spotlight
        | Block::Reveal
        | Block::Dust
        | Block::Halftone
        | Block::Prints
        | Block::Rays
        | Block::Place => Some(Space::Display),
    }
}

/// The space each of `steps` runs in: its block's, or for a block that
/// only moves pixels, what the chain holds then. Steps before the first
/// block that cares hold what that block wants, or encoded values, as the
/// canvas does, when none cares.
fn spaces(steps: &[Step]) -> Vec<Space> {
    let mut holding = steps
        .iter()
        .find_map(|step| space(step.block))
        .unwrap_or(Space::Display);
    steps
        .iter()
        .map(|step| {
            if let Some(wanted) = space(step.block) {
                holding = wanted;
            }
            holding
        })
        .collect()
}

/// The passes between the canvas's bytes, encoded values and light: the
/// picture read texel for texel and converted as a whole.
const LIGHT: &str = r"
// Linear light from a Rec. 709 encoded value, and back. Past 1 the power
// segment carries on, so a highlight past white keeps its place.
fn to_light(encoded: vec3<f32>) -> vec3<f32> {
    let e = max(encoded, vec3<f32>(0.0));
    return select(pow((e + 0.099) / 1.099, vec3<f32>(1.0 / 0.45)), e / 4.5, e < vec3<f32>(0.081));
}

fn to_encoded(light: vec3<f32>) -> vec3<f32> {
    let l = max(light, vec3<f32>(0.0));
    return select(1.099 * pow(l, vec3<f32>(0.45)) - 0.099, 4.5 * l, l < vec3<f32>(0.018));
}

// A premultiplied colour from one side to the other, alpha untouched.
fn light_of(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(to_light(c.rgb / c.a) * c.a, c.a);
}

fn encoded_of(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(to_encoded(c.rgb / c.a) * c.a, c.a);
}

// What the canvas can hold: alpha from 0 to 1, no channel above it.
fn held(c: vec4<f32>) -> vec4<f32> {
    let a = clamp(c.a, 0.0, 1.0);
    return vec4<f32>(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(a)), a);
}
";

const CONVERT: &str = r"
@group(0) @binding(0) var picture: texture_2d<f32>;

struct Out {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    var out: Out;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn here(in: Out) -> vec4<f32> {
    return textureLoad(picture, vec2<i32>(in.position.xy), 0);
}

@fragment
fn fs_decode(in: Out) -> @location(0) vec4<f32> {
    return light_of(here(in));
}

@fragment
fn fs_encode(in: Out) -> @location(0) vec4<f32> {
    return encoded_of(here(in));
}

@fragment
fn fs_encode_out(in: Out) -> @location(0) vec4<f32> {
    return encoded_of(held(here(in)));
}

@fragment
fn fs_copy_out(in: Out) -> @location(0) vec4<f32> {
    return held(here(in));
}
";

const SHADER: &str = r"
struct Effect {
    // The kernel: a block's code, or one of a block's own draws past
    // them; then a small number it reads, such as a blur's direction (0
    // across, 1 down).
    code: vec2<u32>,
    // The size of the picture this draw writes, in pixels.
    size: vec2<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<uniform> fx: Effect;
// A second picture a draw may read: a working picture of its block's, or
// the picture again when the draw reads one only.
@group(0) @binding(3) var other: texture_2d<f32>;

struct Out {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    var out: Out;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// The picture so far at `uv`, already in the values the block works on.
fn fetch(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(source, linear_sampler, uv, 0.0);
}

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A number in [0, 1] from three whole numbers.
fn rand(a: u32, b: u32, seed: u32) -> f32 {
    return f32(pcg(a ^ pcg(b ^ pcg(seed)))) / 4294967295.0;
}

fn straight(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 0.0) {
        return vec3<f32>(0.0);
    }
    return c.rgb / c.a;
}

fn premultiplied(rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}

fn transform(uv: vec2<f32>) -> vec4<f32> {
    let scale = fx.a.x;
    let turn = radians(fx.a.w);
    let d = (uv - vec2<f32>(0.5) - fx.a.yz) * fx.size;
    let c = cos(turn);
    let s = sin(turn);
    let back = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y) / scale;
    return fetch(back / fx.size + vec2<f32>(0.5));
}

fn rgb_split(uv: vec2<f32>) -> vec4<f32> {
    let shift = fx.a.xy;
    let red = fetch(uv + shift);
    let middle = fetch(uv);
    let blue = fetch(uv - shift);
    return vec4<f32>(red.r, middle.g, blue.b, max(middle.a, max(red.a, blue.a)));
}

fn slices(uv: vec2<f32>) -> vec4<f32> {
    let count = max(floor(fx.a.x), 1.0);
    let band = u32(floor(uv.y * count));
    let seed = u32(fx.a.z);
    if (rand(band, 1u, seed) < 0.5) {
        return fetch(uv);
    }
    let shift = (rand(band, 2u, seed) * 2.0 - 1.0) * fx.a.y;
    return fetch(uv + vec2<f32>(shift, 0.0));
}

// A Gaussian along one direction (`fx.code.y` 0 across, 1 down) of
// `fx.a.x` texels of the picture read, which is the size of the one
// written. Two texels a fetch: a pair of weights is one bilinear sample
// placed between them by their ratio.
fn gauss(uv: vec2<f32>) -> vec4<f32> {
    let sigma = fx.a.x;
    if (sigma < 0.2) {
        return fetch(uv);
    }
    var direction = vec2<f32>(1.0 / fx.size.x, 0.0);
    if (fx.code.y == 1u) {
        direction = vec2<f32>(0.0, 1.0 / fx.size.y);
    }
    let reach = min(i32(ceil(sigma * 3.0)), 24);
    let spread = 2.0 * sigma * sigma;
    var sum = fetch(uv);
    var total = 1.0;
    for (var k = 1; k <= reach; k = k + 2) {
        let near = f32(k);
        let wa = exp(-near * near / spread);
        let wb = select(0.0, exp(-(near + 1.0) * (near + 1.0) / spread), k + 1 <= reach);
        let pair = wa + wb;
        let offset = direction * (near + wb / pair);
        sum = sum + (fetch(uv + offset) + fetch(uv - offset)) * pair;
        total = total + 2.0 * pair;
    }
    return sum / total;
}

// Half the size through a 1 3 3 1 filter each way: four bilinear samples
// three quarters of a texel either side of the texel's middle. Unlike the
// plain mean of the two by two under it, the filter keeps a thin line
// where it was, so a blur drawn small does not move it.
fn halve(uv: vec2<f32>) -> vec4<f32> {
    let d = 0.75 / vec2<f32>(textureDimensions(source));
    return (fetch(uv + vec2<f32>(-d.x, -d.y)) + fetch(uv + vec2<f32>(d.x, -d.y))
        + fetch(uv + vec2<f32>(-d.x, d.y)) + fetch(uv + vec2<f32>(d.x, d.y))) * 0.25;
}

// `picture`, smaller than the one drawn, at `uv` through a cubic B-spline:
// four bilinear samples, so no seam of its texels shows when it is
// brought up to size.
fn bspline(picture: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(picture));
    let p = uv * dims - vec2<f32>(0.5);
    let i = floor(p);
    let f = p - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (vec2<f32>(1.0) - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (vec2<f32>(4.0) - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (vec2<f32>(1.0) + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let low = (i - vec2<f32>(1.0) + w1 / g0 + vec2<f32>(0.5)) / dims;
    let high = (i + vec2<f32>(1.0) + w3 / g1 + vec2<f32>(0.5)) / dims;
    return (textureSampleLevel(picture, linear_sampler, vec2<f32>(low.x, low.y), 0.0) * g0.x
        + textureSampleLevel(picture, linear_sampler, vec2<f32>(high.x, low.y), 0.0) * g1.x) * g0.y
        + (textureSampleLevel(picture, linear_sampler, vec2<f32>(low.x, high.y), 0.0) * g0.x
        + textureSampleLevel(picture, linear_sampler, vec2<f32>(high.x, high.y), 0.0) * g1.x) * g1.y;
}

// The picture read, smaller than the one drawn, brought up to its size.
fn smooth_up(uv: vec2<f32>) -> vec4<f32> {
    return bspline(source, uv);
}

// How much of a colour in light spreads as glow: the part of its
// brightest channel above `threshold`, eased in over `knee` either side
// of it, the colour's hue kept.
fn bright_part(c: vec4<f32>, threshold: f32, knee: f32) -> vec4<f32> {
    let top = max(c.r, max(c.g, c.b));
    let soft = clamp(top - threshold + knee, 0.0, 2.0 * knee);
    let eased = soft * soft / (4.0 * knee + 0.00001);
    let over = max(eased, top - threshold);
    return vec4<f32>(c.rgb * (over / max(top, 0.00001)), 0.0);
}

// The glow's first picture: the picture's bright part, `fx.code.y` times
// smaller, each of four samples thresholded before they are averaged so a
// lone bright pixel does not flicker as it moves.
fn glow_bright(uv: vec2<f32>) -> vec4<f32> {
    let step = 0.25 * f32(fx.code.y) / vec2<f32>(textureDimensions(source));
    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let corner = vec2<f32>(f32(k & 1u), f32(k >> 1u)) * 2.0 - vec2<f32>(1.0);
        sum = sum + bright_part(fetch(uv + corner * step), fx.a.x, fx.a.y);
    }
    return sum * 0.25;
}

// One level of the glow on its way back up: the coarser levels, brought
// up and weighed by `fx.a.y`, and this level's own, weighed by `fx.a.x`.
fn glow_up(uv: vec2<f32>) -> vec4<f32> {
    return bspline(source, uv) * fx.a.y + textureSampleLevel(other, linear_sampler, uv, 0.0) * fx.a.x;
}

// The glow laid over the picture: its levels, brought up to size, scaled
// by `fx.a.x` and tinted by `fx.a.yzw`. Light is added; nothing is
// clipped until the picture leaves the chain.
fn glow_add(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let glow = bspline(other, uv).rgb * fx.a.x * fx.a.yzw;
    return vec4<f32>(c.rgb + glow * c.a, c.a);
}

// A kept picture, `other`, over the picture so far, both in light; code.y
// 1 says the kept picture holds encoded values. a.x is the mode: 0 adds,
// 1 screens, 2 multiplies, 3 keeps the lighter, 4 is the kept picture
// itself. a.y mixes from the picture so far to the mode's result.
fn blend(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    var o = textureSampleLevel(other, linear_sampler, uv, 0.0);
    if (fx.code.y == 1u) {
        o = light_of(o);
    }
    var r = o;
    switch (u32(round(fx.a.x))) {
        case 0u: { r = vec4<f32>(c.rgb + o.rgb, c.a); }
        case 1u: {
            // Screen, which past white is an add.
            let lit = clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
            r = vec4<f32>(c.rgb + o.rgb * (1.0 - lit), c.a);
        }
        case 2u: {
            // Where the kept picture is clear the picture so far stays.
            let top = o.rgb / max(o.a, 1e-6);
            r = vec4<f32>(c.rgb * mix(vec3<f32>(1.0), top, o.a), c.a);
        }
        case 3u: { r = vec4<f32>(max(c.rgb, o.rgb), c.a); }
        default: {}
    }
    return mix(c, r, fx.a.y);
}

fn grade(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    var rgb = straight(c) + vec3<f32>(fx.a.x);
    rgb = (rgb - vec3<f32>(0.5)) * fx.a.y + vec3<f32>(0.5);
    let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    rgb = mix(vec3<f32>(luma), rgb, fx.a.z);
    return premultiplied(rgb, c.a);
}

fn flash(uv: vec2<f32>) -> vec4<f32> {
    return mix(fetch(uv), vec4<f32>(fx.a.yzw, 1.0), fx.a.x);
}

fn vignette(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let d = length(uv - vec2<f32>(0.5)) * 1.4142;
    let dark = smoothstep(fx.a.y, fx.a.y + 0.5, d) * fx.a.x;
    return vec4<f32>(c.rgb * (1.0 - dark), c.a);
}

fn noise(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let pixel = vec2<u32>(uv * fx.size);
    let grain = (rand(pixel.x, pixel.y, u32(fx.a.y)) - 0.5) * fx.a.x;
    return vec4<f32>(clamp(c.rgb + vec3<f32>(grain * c.a), vec3<f32>(0.0), vec3<f32>(c.a)), c.a);
}

fn scanlines(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let line = u32(floor(uv.y * fx.a.y));
    if ((line & 1u) == 1u) {
        return vec4<f32>(c.rgb * (1.0 - fx.a.x), c.a);
    }
    return c;
}

fn mirror(uv: vec2<f32>) -> vec4<f32> {
    var p = uv;
    let side = u32(fx.a.x);
    if (side == 0u && p.x > 0.5) {
        p.x = 1.0 - p.x;
    } else if (side == 1u && p.x < 0.5) {
        p.x = 1.0 - p.x;
    } else if (side == 2u && p.y > 0.5) {
        p.y = 1.0 - p.y;
    } else if (side == 3u && p.y < 0.5) {
        p.y = 1.0 - p.y;
    }
    return fetch(p);
}

fn pixelate(uv: vec2<f32>) -> vec4<f32> {
    let cell = fx.a.x * min(fx.size.x, fx.size.y);
    if (cell < 1.0) {
        return fetch(uv);
    }
    let pixel = (floor(uv * fx.size / cell) + vec2<f32>(0.5)) * cell;
    return fetch(pixel / fx.size);
}

fn invert(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let rgb = straight(c);
    return premultiplied(mix(rgb, vec3<f32>(1.0) - rgb, fx.a.x), c.a);
}

// From the middle of the frame (or a point moved from it) to `uv`, in
// shorter sides; and back.
fn from_point(uv: vec2<f32>, point: vec2<f32>) -> vec2<f32> {
    return (uv - vec2<f32>(0.5) - point) * fx.size / min(fx.size.x, fx.size.y);
}

fn to_uv(p: vec2<f32>, point: vec2<f32>) -> vec2<f32> {
    return p * min(fx.size.x, fx.size.y) / fx.size + vec2<f32>(0.5) + point;
}

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn zoom_blur(uv: vec2<f32>) -> vec4<f32> {
    let centre = vec2<f32>(0.5) + fx.a.yz;
    let toward = (centre - uv) * fx.a.x;
    var sum = vec4<f32>(0.0);
    for (var k = 0; k < 32; k = k + 1) {
        sum = sum + fetch(uv + toward * (f32(k) / 31.0));
    }
    return sum / 32.0;
}

fn motion_blur(uv: vec2<f32>) -> vec4<f32> {
    let turn = radians(fx.a.y);
    let reach = fx.a.x * min(fx.size.x, fx.size.y);
    let step = vec2<f32>(cos(turn), -sin(turn)) * reach / fx.size / 16.0;
    var sum = vec4<f32>(0.0);
    for (var k = -16; k <= 16; k = k + 1) {
        sum = sum + fetch(uv + step * f32(k));
    }
    return sum / 33.0;
}

fn swirl(uv: vec2<f32>) -> vec4<f32> {
    let point = fx.a.zw;
    let p = from_point(uv, point);
    let r = length(p);
    let radius = max(fx.a.y, 0.0001);
    if (r >= radius) {
        return fetch(uv);
    }
    let falloff = 1.0 - r / radius;
    let turn = radians(fx.a.x) * falloff * falloff;
    let c = cos(turn);
    let s = sin(turn);
    return fetch(to_uv(vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y), point));
}

fn ripple(uv: vec2<f32>) -> vec4<f32> {
    let p = from_point(uv, vec2<f32>(0.0));
    let kind = u32(fx.a.w + 0.5);
    let wave = 6.2831853;
    var q = p;
    if (kind == 0u) {
        q.x = p.x + fx.a.x * sin(wave * (p.y / fx.a.y - fx.a.z));
    } else if (kind == 1u) {
        q.y = p.y + fx.a.x * sin(wave * (p.x / fx.a.y - fx.a.z));
    } else {
        let r = length(p);
        if (r > 0.0) {
            q = p + p / r * fx.a.x * sin(wave * (r / fx.a.y - fx.a.z));
        }
    }
    return fetch(to_uv(q, vec2<f32>(0.0)));
}

fn bulge(uv: vec2<f32>) -> vec4<f32> {
    let point = fx.a.zw;
    let p = from_point(uv, point);
    let n = length(p) / max(fx.a.y, 0.0001);
    if (n >= 1.0) {
        return fetch(uv);
    }
    let pull = 1.0 - fx.a.x * (1.0 - n * n);
    return fetch(to_uv(p * pull, point));
}

fn light_leak(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let p = from_point(uv, fx.a.yz);
    let d = length(p) / max(fx.a.w, 0.0001);
    let light = fx.b.xyz * fx.a.x * exp(-d * d * 2.0);
    // Screen: light only ever brightens.
    let rgb = vec3<f32>(1.0) - (vec3<f32>(1.0) - straight(c)) * (vec3<f32>(1.0) - light);
    return premultiplied(rgb, c.a);
}

fn hue(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let rgb = straight(c);
    // Turn about the grey axis, keeping brightness.
    let turn = radians(fx.a.x);
    let k = vec3<f32>(0.57735);
    let cs = cos(turn);
    let turned = rgb * cs + cross(k, rgb) * sin(turn) + k * dot(k, rgb) * (1.0 - cs);
    return premultiplied(turned, c.a);
}

fn tile(uv: vec2<f32>) -> vec4<f32> {
    let count = max(floor(fx.a.xy + vec2<f32>(0.5)), vec2<f32>(1.0));
    return fetch(fract(uv * count));
}

fn posterize(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let steps = max(floor(fx.a.x + 0.5), 2.0) - 1.0;
    return premultiplied(floor(straight(c) * steps + vec3<f32>(0.5)) / steps, c.a);
}

fn edges(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let px = 1.0 / fx.size;
    let tl = luma(fetch(uv + vec2<f32>(-px.x, -px.y)).rgb);
    let t = luma(fetch(uv + vec2<f32>(0.0, -px.y)).rgb);
    let tr = luma(fetch(uv + vec2<f32>(px.x, -px.y)).rgb);
    let l = luma(fetch(uv + vec2<f32>(-px.x, 0.0)).rgb);
    let r = luma(fetch(uv + vec2<f32>(px.x, 0.0)).rgb);
    let bl = luma(fetch(uv + vec2<f32>(-px.x, px.y)).rgb);
    let b = luma(fetch(uv + vec2<f32>(0.0, px.y)).rgb);
    let br = luma(fetch(uv + vec2<f32>(px.x, px.y)).rgb);
    let gx = tr + 2.0 * r + br - tl - 2.0 * l - bl;
    let gy = bl + 2.0 * b + br - tl - 2.0 * t - tr;
    let line = clamp(length(vec2<f32>(gx, gy)) * 2.0, 0.0, 1.0);
    let rgb = straight(c);
    let inked = rgb * (1.0 - line * fx.a.x);
    // Alone, each outline keeps the colour under it, lit up.
    let alone = min(rgb * 1.5 + vec3<f32>(0.2), vec3<f32>(1.0)) * line;
    return premultiplied(mix(inked, alone, fx.a.y), c.a);
}

fn gradient_map(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let rgb = straight(c);
    let level = clamp(luma(rgb), 0.0, 1.0);
    let palette = u32(fx.a.y + 0.5);
    var mapped = mix(vec3<f32>(fx.a.z, fx.a.w, fx.b.x), fx.b.yzw, level);
    if (palette == 1u) {
        var heat = array<vec3<f32>, 6>(
            vec3<f32>(0.0, 0.0, 0.0),
            vec3<f32>(0.1, 0.1, 0.8),
            vec3<f32>(0.8, 0.1, 0.6),
            vec3<f32>(1.0, 0.5, 0.0),
            vec3<f32>(1.0, 0.95, 0.2),
            vec3<f32>(1.0, 1.0, 1.0),
        );
        let at = level * 5.0;
        let index = min(u32(floor(at)), 4u);
        mapped = mix(heat[index], heat[index + 1u], at - f32(index));
    } else if (palette == 2u) {
        let hue = level * 0.85;
        mapped = clamp(
            abs(fract(vec3<f32>(hue) + vec3<f32>(1.0, 0.6666667, 0.3333333)) * 6.0 - vec3<f32>(3.0))
                - vec3<f32>(1.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
    }
    return premultiplied(mix(rgb, mapped, fx.a.x), c.a);
}

fn grid(uv: vec2<f32>) -> vec4<f32> {
    let count = max(floor(fx.a.xy + vec2<f32>(0.5)), vec2<f32>(1.0));
    let local = fract(uv * count);
    // Each cell shows the middle of the frame at the cell's own shape.
    let cell = fx.size / count;
    let fill = max(cell.x / fx.size.x, cell.y / fx.size.y);
    let region = cell / (fill * fx.size) / max(fx.a.z, 1.0);
    return fetch(vec2<f32>(0.5) + (local - vec2<f32>(0.5)) * region);
}

fn spotlight(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let d = length(from_point(uv, fx.a.yz));
    let lit = 1.0 - smoothstep(fx.a.w, fx.a.w + fx.b.x + 0.0001, d);
    return vec4<f32>(c.rgb * (1.0 - fx.a.x * (1.0 - lit)), c.a);
}

fn reveal(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let progress = fx.a.x;
    if (progress >= 1.0) {
        return c;
    }
    let kind = u32(fx.a.y + 0.5);
    let turn = radians(fx.a.z);
    let along = vec2<f32>(cos(turn), -sin(turn));
    // How far through the run this pixel is, 0 to 1.
    var run = 0.0;
    let p = from_point(uv, vec2<f32>(0.0));
    let half = fx.size / min(fx.size.x, fx.size.y) * 0.5;
    if (kind == 1u) {
        run = length(p) / length(half);
    } else if (kind == 3u) {
        let reach = (abs(along.x) + abs(along.y)) * 0.5;
        run = abs(dot(uv - vec2<f32>(0.5), along)) / reach;
    } else if (kind == 4u) {
        // Clockwise from straight up, then on from `angle`.
        let round = atan2(p.x, -p.y) / 6.2831853 - fx.a.z / 360.0;
        run = fract(round + 1.0);
    } else if (kind == 5u) {
        run = (abs(p.x) + abs(p.y)) / (half.x + half.y);
    } else if (kind == 6u) {
        let count = max(floor(fx.b.x + 0.5), 1.0);
        let at = uv * fx.size / (min(fx.size.x, fx.size.y) / count);
        let cell = floor(at);
        let odd = f32((u32(cell.x) + u32(cell.y)) & 1u);
        run = odd * 0.5 + fract(at.x) * 0.5;
    } else if (kind == 7u) {
        let grain = vec2<u32>(uv * fx.size / 3.0);
        run = rand(grain.x, grain.y, 7u);
    } else {
        let reach = (abs(along.x) + abs(along.y)) * 0.5;
        run = dot(uv - vec2<f32>(0.5), along) / reach * 0.5 + 0.5;
        if (kind == 2u) {
            run = fract(run * max(floor(fx.b.x + 0.5), 1.0));
        }
    }
    let soft = fx.a.w;
    let shown = 1.0 - smoothstep(progress * (1.0 + soft) - soft, progress * (1.0 + soft), run);
    return vec4<f32>(c.rgb * shown, c.a);
}

fn blocks(uv: vec2<f32>) -> vec4<f32> {
    let tall = max(fx.a.y * min(fx.size.x, fx.size.y), 2.0);
    let cell = floor(uv * fx.size / vec2<f32>(tall * 2.5, tall));
    let cx = u32(cell.x + 1000.0);
    let cy = u32(cell.y + 1000.0);
    let seed = u32(fx.a.w);
    if (rand(cx, cy, seed) >= fx.a.x) {
        return fetch(uv);
    }
    let push = vec2<f32>(rand(cx, cy, seed + 1u), rand(cx, cy, seed + 2u)) * 2.0 - vec2<f32>(1.0);
    let c = fetch(uv + push * fx.a.z * vec2<f32>(1.0, 0.3));
    if (rand(cx, cy, seed + 3u) < fx.b.x) {
        return vec4<f32>(c.b, c.r, c.g, c.a);
    }
    return c;
}

fn rays(uv: vec2<f32>) -> vec4<f32> {
    let base = fetch(uv);
    let toward = (vec2<f32>(0.5) + fx.a.yz - uv) * fx.a.w;
    var light = vec3<f32>(0.0);
    for (var k = 0; k < 40; k = k + 1) {
        let t = f32(k) / 39.0;
        let c = fetch(uv + toward * t);
        light = light + max(straight(c) - vec3<f32>(fx.b.x), vec3<f32>(0.0)) * c.a * (1.0 - t);
    }
    let rgb = base.rgb + light / 10.0 * fx.a.x;
    return vec4<f32>(min(rgb, vec3<f32>(base.a)), base.a);
}

fn focus(uv: vec2<f32>) -> vec4<f32> {
    let kind = u32(fx.a.y + 0.5);
    let p = from_point(uv, vec2<f32>(0.0));
    let sharp = fx.a.z * 0.5;
    var weight = 0.0;
    if (kind == 0u) {
        weight = smoothstep(sharp, sharp + 0.25, abs(uv.y - 0.5 - fx.a.w));
    } else {
        weight = smoothstep(sharp, sharp + 0.4, length(p));
    }
    var sum = vec4<f32>(0.0);
    if (kind == 2u) {
        // Taps along an arc around the middle.
        let spin = radians(fx.b.x) * weight * min(fx.a.x * 10.0, 1.0);
        for (var k = -16; k <= 16; k = k + 1) {
            let a = spin * f32(k) / 16.0;
            let q = vec2<f32>(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a));
            sum = sum + fetch(to_uv(q, vec2<f32>(0.0)));
        }
        return sum / 33.0;
    }
    let reach = fx.a.x * weight * min(fx.size.x, fx.size.y) / fx.size;
    for (var k = 0; k < 48; k = k + 1) {
        let r = sqrt((f32(k) + 0.5) / 48.0);
        let turn = f32(k) * 2.3999632;
        sum = sum + fetch(uv + vec2<f32>(cos(turn), sin(turn)) * r * reach);
    }
    return sum / 48.0;
}

fn place(uv: vec2<f32>) -> vec4<f32> {
    let short = min(fx.size.x, fx.size.y);
    // Behind the card: clear, or a soft, dim copy of the picture.
    var behind = vec4<f32>(0.0);
    if (fx.c.y > 0.0) {
        let reach = fx.c.y * short / fx.size;
        var sum = vec4<f32>(0.0);
        for (var k = 0; k < 48; k = k + 1) {
            let r = sqrt((f32(k) + 0.5) / 48.0);
            let turn = f32(k) * 2.3999632;
            sum = sum + fetch(uv + vec2<f32>(cos(turn), sin(turn)) * r * reach);
        }
        behind = sum / 48.0;
        behind = vec4<f32>(behind.rgb * (1.0 - fx.c.z), behind.a);
    }
    // Pixels from the placed middle, turned back and scaled back.
    let d = (uv - vec2<f32>(0.5) - fx.a.yz) * fx.size;
    let turn = radians(-fx.a.w);
    let r = vec2<f32>(d.x * cos(turn) - d.y * sin(turn), d.x * sin(turn) + d.y * cos(turn));
    let scale = fx.a.x * fx.b.xy;
    if (scale.x <= 0.0001 || scale.y <= 0.0001) {
        return behind;
    }
    let back = r / scale / fx.size + vec2<f32>(0.5);
    let edge = fx.b.w * short * fx.a.x;
    let foot = fx.c.x * short * fx.a.x;
    if (edge <= 0.0 && foot <= 0.0 && fx.c.w <= 0.0) {
        if (any(back < vec2<f32>(0.0)) || any(back > vec2<f32>(1.0))) {
            return behind;
        }
        let card = fetch(back) * fx.b.z;
        return card + behind * (1.0 - card.a);
    }
    // A print: the picture on white paper, the bottom edge `foot` deeper.
    let half = scale * fx.size * 0.5;
    let paper_middle = vec2<f32>(0.0, foot * 0.5);
    let paper_half = half + vec2<f32>(edge, edge + foot * 0.5);
    if (fx.c.w > 0.0) {
        // A soft shadow below and to the right, as light from above left
        // casts it.
        let soft = max(fx.c.w * short * fx.a.x, 1.0);
        let fall = vec2<f32>(0.3, 0.6) * soft;
        let distance = rounded_box(r - paper_middle - fall, paper_half, 0.0);
        let shade = (1.0 - smoothstep(-soft, soft, distance)) * 0.45 * fx.b.z;
        behind = vec4<f32>(behind.rgb * (1.0 - shade), shade + behind.a * (1.0 - shade));
    }
    let paper = 1.0 - smoothstep(-0.5, 0.5, rounded_box(r - paper_middle, paper_half, 0.0));
    let photo = 1.0 - smoothstep(-0.5, 0.5, rounded_box(r, half, 0.0));
    let picture = fetch(clamp(back, vec2<f32>(0.0), vec2<f32>(1.0)));
    let white = vec4<f32>(0.97, 0.97, 0.96, 1.0);
    let card = mix(white, picture, photo) * paper * fx.b.z;
    return card + behind * (1.0 - card.a);
}

fn stretch(uv: vec2<f32>) -> vec4<f32> {
    let scale = max(fx.a.xy, vec2<f32>(0.0001));
    let back = (uv - vec2<f32>(0.5)) / scale + vec2<f32>(0.5);
    let outside = any(back < vec2<f32>(0.0)) || any(back > vec2<f32>(1.0));
    if (outside && fx.a.z >= 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return fetch(back);
}

fn dust(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    var rgb = straight(c);
    let seed = u32(fx.a.z);
    let p = from_point(uv, vec2<f32>(0.0)) + vec2<f32>(4.0);
    let room = max(fx.a.w, 0.001);
    let cell = floor(p / room);
    let cx = u32(cell.x);
    let cy = u32(cell.y);
    if (rand(cx, cy, seed) < fx.a.x * 0.35) {
        let middle = (cell + vec2<f32>(rand(cx, cy, seed + 1u), rand(cx, cy, seed + 2u))) * room;
        let radius = room * (0.08 + 0.25 * rand(cx, cy, seed + 3u));
        let speck = 1.0 - smoothstep(radius * 0.6, radius, length(p - middle));
        let light = rand(cx, cy, seed + 4u) < 0.3;
        if (light) {
            rgb = mix(rgb, vec3<f32>(0.95), speck * 0.8);
        } else {
            rgb = mix(rgb, vec3<f32>(0.05), speck * 0.85);
        }
    }
    for (var k = 0u; k < 3u; k = k + 1u) {
        if (rand(k, 7u, seed) < fx.a.y) {
            let x = rand(k, 8u, seed) + 0.01 * sin(uv.y * 9.0 + f32(k));
            let width = 1.2 / fx.size.x;
            let line = 1.0 - smoothstep(0.0, width, abs(uv.x - x));
            rgb = mix(rgb, vec3<f32>(0.1), line * 0.6);
        }
    }
    return premultiplied(rgb, c.a);
}

fn kaleido(uv: vec2<f32>) -> vec4<f32> {
    let count = floor(fx.a.x + 0.5);
    if (count < 2.0) {
        return fetch(uv);
    }
    let point = fx.a.zw;
    let p = from_point(uv, point);
    let turn = radians(fx.a.y);
    let wedge = 6.2831853 / count;
    var angle = atan2(p.y, p.x) - turn;
    angle = angle - floor(angle / wedge) * wedge;
    if (angle > wedge * 0.5) {
        angle = wedge - angle;
    }
    angle = angle + turn;
    let r = length(p);
    return fetch(to_uv(vec2<f32>(cos(angle), sin(angle)) * r, point));
}

fn fringe(uv: vec2<f32>) -> vec4<f32> {
    let centre = vec2<f32>(0.5) + fx.a.yz;
    let out = centre + (uv - centre) * (1.0 - fx.a.x);
    let inside = centre + (uv - centre) * (1.0 + fx.a.x);
    let middle = fetch(uv);
    let red = fetch(out);
    let blue = fetch(inside);
    return vec4<f32>(red.r, middle.g, blue.b, max(middle.a, max(red.a, blue.a)));
}

fn halftone(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let short = min(fx.size.x, fx.size.y);
    let cell = fx.a.x * short;
    if (cell < 2.0) {
        return c;
    }
    // A grid turned 45 degrees, as printers turn theirs.
    let pixel = uv * fx.size;
    let k = 0.70710678;
    let turned = vec2<f32>(k * pixel.x - k * pixel.y, k * pixel.x + k * pixel.y);
    let middle_turned = (floor(turned / cell) + vec2<f32>(0.5)) * cell;
    let middle = vec2<f32>(k * middle_turned.x + k * middle_turned.y, -k * middle_turned.x + k * middle_turned.y);
    let sample = straight(fetch(middle / fx.size));
    let radius = sqrt(1.0 - clamp(luma(sample), 0.0, 1.0)) * cell * 0.7;
    // A dot under a pixel wide fades out rather than leaving half a pixel.
    let dot_cover = (1.0 - smoothstep(radius - 0.75, radius + 0.75, length(turned - middle_turned)))
        * min(radius / 0.75, 1.0);
    let ink = mix(vec3<f32>(0.05), sample, fx.a.z);
    let printed = mix(vec3<f32>(0.97), ink, dot_cover);
    return premultiplied(mix(straight(c), printed, fx.a.y), c.a);
}

// How far `p` is outside a rectangle `half` each way with corners rounded
// by `round`; negative inside.
fn rounded_box(p: vec2<f32>, half: vec2<f32>, round: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(round);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - round;
}

fn prints(uv: vec2<f32>) -> vec4<f32> {
    let count = min(u32(fx.a.x + 0.5), 4u);
    let input = fetch(uv);
    if (count == 0u) {
        return input;
    }
    let short = min(fx.size.x, fx.size.y);
    // Behind: a soft, pale copy of the picture.
    var back = input;
    if (fx.b.z > 0.0) {
        let reach = fx.b.z * short / fx.size;
        var sum = vec4<f32>(0.0);
        for (var k = 0; k < 48; k = k + 1) {
            let r = sqrt((f32(k) + 0.5) / 48.0);
            let turn = f32(k) * 2.3999632;
            sum = sum + fetch(uv + vec2<f32>(cos(turn), sin(turn)) * r * reach);
        }
        back = sum / 48.0;
    }
    var rgb = mix(straight(back), vec3<f32>(1.0), fx.b.w);
    let p = from_point(uv, vec2<f32>(0.0));
    let frame_half = fx.size / short * 0.5;
    let pixel = 1.0 / short;
    var places = array<vec2<f32>, 4>(
        vec2<f32>(0.02, -0.06),
        vec2<f32>(0.26, -0.1),
        vec2<f32>(-0.22, 0.1),
        vec2<f32>(0.1, 0.16),
    );
    var leans = array<f32, 4>(-1.0, 0.7, -0.45, 0.9);
    // Where each print flies in from, in shorter sides from its place,
    // and how far it is turned as it sets off.
    var starts = array<vec2<f32>, 4>(
        vec2<f32>(0.15, 1.3),
        vec2<f32>(1.5, 0.35),
        vec2<f32>(-1.5, -0.3),
        vec2<f32>(0.9, -1.3),
    );
    var spins = array<f32, 4>(-8.0, 28.0, -32.0, 20.0);
    let border = fx.b.y;
    // Back to front, each print over what is behind it.
    for (var i = 0u; i < count; i = i + 1u) {
        let flight = clamp(fx.c.x - 0.6 * f32(i), 0.0, 1.0);
        if (flight <= 0.0) {
            continue;
        }
        // Fast at first, slowing as it lands.
        let left = pow(1.0 - flight, 3.0);
        let half = frame_half * fx.a.y * (1.0 - 0.2 * f32(i)) * (1.0 + 0.15 * left);
        let turn = radians(fx.a.w + fx.b.x * leans[i] + spins[i] * left);
        let c = cos(turn);
        let s = sin(turn);
        let d = p - places[i] * fx.a.z - starts[i] * left;
        let local = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
        let edge = half + vec2<f32>(border);
        // A soft shadow below and to the right.
        let fall = d - vec2<f32>(0.01, 0.016);
        let shadow_local = vec2<f32>(c * fall.x + s * fall.y, -s * fall.x + c * fall.y);
        let shadow = 1.0 - smoothstep(-0.01, 0.05, rounded_box(shadow_local, edge, border * 0.4));
        rgb = rgb * (1.0 - shadow * 0.35);
        let card = 1.0 - smoothstep(-pixel, pixel, rounded_box(local, edge, border * 0.4));
        let photo = 1.0 - smoothstep(-pixel, pixel, rounded_box(local, half, border * 1.2));
        let picture = straight(fetch(local / half * 0.5 + vec2<f32>(0.5)));
        let face = mix(vec3<f32>(0.96, 0.96, 0.95), picture, photo);
        rgb = mix(rgb, face, card);
    }
    return premultiplied(rgb, max(input.a, back.a));
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return kernel(in.position.xy / fx.size);
}

// The chain's last draw, straight into the canvas format: from encoded
// values, or from light.
@fragment
fn fs_last_encoded(in: Out) -> @location(0) vec4<f32> {
    return held(kernel(in.position.xy / fx.size));
}

@fragment
fn fs_last_light(in: Out) -> @location(0) vec4<f32> {
    return encoded_of(held(kernel(in.position.xy / fx.size)));
}

// A step's last draw when the next step wants the other values.
@fragment
fn fs_into_encoded(in: Out) -> @location(0) vec4<f32> {
    return encoded_of(kernel(in.position.xy / fx.size));
}

@fragment
fn fs_into_light(in: Out) -> @location(0) vec4<f32> {
    return light_of(kernel(in.position.xy / fx.size));
}

fn kernel(uv: vec2<f32>) -> vec4<f32> {
    switch (fx.code.x) {
        case 0u: { return transform(uv); }
        case 1u: { return rgb_split(uv); }
        case 2u: { return slices(uv); }
        // 3, the blur, is drawn by its plan: GAUSS, HALVE and SMOOTH_UP.
        case 4u: { return grade(uv); }
        case 5u: { return flash(uv); }
        case 6u: { return vignette(uv); }
        case 7u: { return noise(uv); }
        case 8u: { return scanlines(uv); }
        case 9u: { return mirror(uv); }
        case 10u: { return pixelate(uv); }
        case 11u: { return invert(uv); }
        case 12u: { return zoom_blur(uv); }
        case 13u: { return motion_blur(uv); }
        case 14u: { return swirl(uv); }
        case 15u: { return ripple(uv); }
        case 16u: { return bulge(uv); }
        // 17, the glow, is drawn by its plan: GLOW_BRIGHT, HALVE, GLOW_UP
        // and GLOW_ADD.
        case 18u: { return light_leak(uv); }
        case 19u: { return hue(uv); }
        case 20u: { return tile(uv); }
        case 21u: { return posterize(uv); }
        case 22u: { return edges(uv); }
        case 23u: { return gradient_map(uv); }
        case 24u: { return grid(uv); }
        case 25u: { return spotlight(uv); }
        case 26u: { return reveal(uv); }
        case 27u: { return stretch(uv); }
        case 28u: { return dust(uv); }
        case 29u: { return kaleido(uv); }
        case 30u: { return fringe(uv); }
        case 31u: { return halftone(uv); }
        case 32u: { return prints(uv); }
        case 33u: { return blocks(uv); }
        case 34u: { return rays(uv); }
        case 35u: { return focus(uv); }
        case 36u: { return place(uv); }
        case 37u: { return blend(uv); }
        case 100u: { return gauss(uv); }
        case 101u: { return halve(uv); }
        case 102u: { return smooth_up(uv); }
        case 103u: { return glow_bright(uv); }
        case 104u: { return glow_up(uv); }
        case 105u: { return glow_add(uv); }
        default: { return fetch(uv); }
    }
}
";

/// Bytes in one draw's uniform block.
const PARAMS: u64 = 64;

/// A picture one draw of a block reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Picture {
    /// The chain's picture so far. A draw that writes it makes the next
    /// one, which the draws after it read.
    Current,
    /// One of the block's own working pictures, by its index in
    /// [`Plan::work`].
    Work(usize),
    /// A picture the chain kept: 0 the picture as it entered the effect,
    /// then those steps saved, as [`Step::with`] counts them.
    Kept(u8),
}

/// One draw of a block: a full-screen pass of one kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Draw {
    /// What the shader runs: a block's code, or one of the kernels past
    /// them.
    pub(crate) kernel: u32,
    /// A small number the kernel reads: a blur's direction, a level.
    pub(crate) arg: u32,
    /// The picture sampled as `source`.
    pub(crate) read: Picture,
    /// The picture sampled as `other`, when the draw reads two.
    pub(crate) other: Option<Picture>,
    /// The picture drawn.
    pub(crate) write: Picture,
    /// The kernel's twelve values.
    pub(crate) values: [f32; MAX_VALUES],
}

/// How a block is drawn: its working pictures and its draws in order.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct Plan {
    /// Each working picture, as how many times smaller than the picture
    /// it is across and down: a power of two, 1 the picture's own size.
    pub(crate) work: Vec<u32>,
    pub(crate) draws: Vec<Draw>,
}

/// The size of a picture `shrink` times smaller than `size`, rounded up,
/// never nothing.
pub(crate) fn shrunk_size(size: (u32, u32), shrink: u32) -> (u32, u32) {
    let shrink = shrink.max(1);
    (
        size.0.div_ceil(shrink).max(1),
        size.1.div_ceil(shrink).max(1),
    )
}

/// The kernels past the blocks' own codes.
const GAUSS: u32 = 100;
const HALVE: u32 = 101;
const SMOOTH_UP: u32 = 102;
const GLOW_BRIGHT: u32 = 103;
const GLOW_UP: u32 = 104;
const GLOW_ADD: u32 = 105;

/// The short side, in pixels, at which a glow's finest level is the
/// picture's own pixels. Its levels are set by this share of the short
/// side, not by pixels, so a glow is the same share of the picture at
/// every size: a larger picture starts on a smaller copy, and a smaller
/// one folds the levels finer than its pixels into its finest.
const GLOW_FINEST: f32 = 540.0;

/// The fewest texels across a glow level's short side.
const GLOW_SMALLEST: u32 = 2;

/// How strong a glow's added light is at `amount` 1: the bright part's own
/// light, spread, times this.
const GLOW_GAIN: f32 = 0.6;

/// The most times smaller than the picture a working picture is.
pub(crate) const MAX_SHRINK: u32 = 64;

/// The fewest texels of the working picture a blur's sigma spans: below
/// it the picture would be too coarse to hold the blur's shape.
const MIN_SIGMA_TEXELS: f32 = 2.0;

/// How `step` is drawn over a `size` picture.
pub(crate) fn plan(step: &Step, size: (u32, u32)) -> Plan {
    match step.block {
        Block::Blur => {
            let short = size.0.min(size.1) as f32;
            blur_plan(step.values[0] * short)
        }
        Block::Glow => glow_plan(&step.values, size),
        _ => Plan {
            work: Vec::new(),
            draws: vec![Draw {
                kernel: step.block.code(),
                arg: 0,
                read: Picture::Current,
                other: step.with.map(Picture::Kept),
                write: Picture::Current,
                values: step.values,
            }],
        },
    }
}

/// A draw of `kernel` with `values` set from its first few.
fn kernel(kernel: u32, arg: u32, read: Picture, write: Picture, first: &[f32]) -> Draw {
    let mut values = [0.0; MAX_VALUES];
    values[..first.len()].copy_from_slice(first);
    Draw {
        kernel,
        arg,
        read,
        other: None,
        write,
        values,
    }
}

/// Linear light of a Rec. 709 encoded level, as the shader's `to_light`.
fn to_light(encoded: f32) -> f32 {
    let e = encoded.max(0.0);
    if e < 0.081 {
        e / 4.5
    } else {
        ((e + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// A glow: the light of the picture above a threshold, spread by a stack
/// of ever smaller and softer copies of itself and added back. Each level
/// is half the last; their sum falls off fast near a highlight and lasts
/// far from it, as light scattered in a lens does. The levels reach as far
/// as `radius` of the short side, the last weighed by how far of the way
/// to the next it gets, so the reach grows smoothly with the knob.
///
/// The levels sit at fixed shares of the short side, the finest at a
/// [`GLOW_FINEST`]th of it, so the stack is the same at every size. Where
/// that is finer than a pixel the finest level stands in for the levels
/// it cannot hold, and carries their weight.
fn glow_plan(values: &[f32; MAX_VALUES], size: (u32, u32)) -> Plan {
    let [amount, radius, threshold, knee, red, green, blue, ..] = *values;
    let short = size.0.min(size.1);
    let reach = radius * short as f32;
    if amount <= 0.0 || reach < 0.5 {
        return Plan::default();
    }
    // Shrinks as powers of two: where the finest level would sit, where
    // the picture's finest level does, and where the top level sits.
    let nominal = (short as f32 / GLOW_FINEST).log2();
    let finest = nominal.round().max(0.0);
    let top = nominal + (radius * GLOW_FINEST / 2.0).max(1.0).log2();
    // The most halvings that keep a level GLOW_SMALLEST texels across.
    let room = (short / GLOW_SMALLEST).max(1).ilog2().saturating_sub(finest as u32);
    let halvings = ((top - finest).ceil().max(0.0) as u32).min(room);
    // The finest level carries the levels finer than it, the top one its
    // share of the way to the next.
    let finest_weight = (1.0 + finest - nominal).max(0.5);
    let top_weight = (top - finest - (halvings as f32 - 1.0)).clamp(0.0, 1.0);
    let total = if halvings == 0 {
        1.0
    } else {
        finest_weight + (halvings - 1) as f32 + top_weight
    };
    let finest = 1u32 << finest as u32;
    let level = |index: u32| finest << index;
    let threshold = to_light(threshold);
    let mut work = vec![level(0)];
    let mut draws = vec![kernel(
        GLOW_BRIGHT,
        finest,
        Picture::Current,
        Picture::Work(0),
        &[threshold, knee * threshold],
    )];
    for index in 1..=halvings {
        work.push(level(index));
        draws.push(kernel(
            HALVE,
            0,
            Picture::Work(index as usize - 1),
            Picture::Work(index as usize),
            &[],
        ));
    }
    // Back up: each level's own light and the coarser ones brought up.
    let mut coarser = Picture::Work(halvings as usize);
    for index in (0..halvings).rev() {
        work.push(level(index));
        let sum = Picture::Work(work.len() - 1);
        let own = if index == 0 { finest_weight } else { 1.0 };
        let brought = if index + 1 == halvings {
            top_weight
        } else {
            1.0
        };
        let mut up = kernel(GLOW_UP, 0, coarser, sum, &[own, brought]);
        up.other = Some(Picture::Work(index as usize));
        draws.push(up);
        coarser = sum;
    }
    let mut add = kernel(
        GLOW_ADD,
        0,
        Picture::Current,
        Picture::Current,
        &[amount * GLOW_GAIN / total, red, green, blue],
    );
    add.other = Some(coarser);
    draws.push(add);
    Plan { work, draws }
}

/// A Gaussian blur as wide as a box of `reach` pixels each side: the same
/// spread, `reach / sqrt(3)`, without the box's hard edge. A wide one is
/// drawn on a picture halved until the spread is a few of its texels,
/// then brought back up through a B-spline; the halving and the spline
/// blur too, and the Gaussian drawn is narrowed by what they add, so the
/// spread is the same at every size.
fn blur_plan(reach: f32) -> Plan {
    let sigma = reach / 3f32.sqrt();
    if sigma < 0.2 {
        return Plan::default();
    }
    let mut shrink = 1;
    while shrink < MAX_SHRINK && sigma / (shrink * 2) as f32 >= MIN_SIGMA_TEXELS {
        shrink *= 2;
    }
    if shrink == 1 {
        return Plan {
            work: Vec::new(),
            draws: vec![
                kernel(GAUSS, 0, Picture::Current, Picture::Current, &[sigma]),
                kernel(GAUSS, 1, Picture::Current, Picture::Current, &[sigma]),
            ],
        };
    }
    // In the picture's pixels, the 1 3 3 1 halvings add about shrink² / 4
    // to the variance, the B-spline shrink² / 3.
    let s = shrink as f32;
    let left = (sigma * sigma - s * s * (1.0 / 4.0 + 1.0 / 3.0)).max(0.0);
    let at_level = left.sqrt() / s;
    // Halvings into work 0, 1, ...; the last is at `shrink`, and one more
    // of that size for the Gaussian to go across into.
    let mut work = Vec::new();
    let mut draws = Vec::new();
    let mut level = 1;
    let mut read = Picture::Current;
    while level < shrink {
        level *= 2;
        work.push(level);
        let write = Picture::Work(work.len() - 1);
        draws.push(kernel(HALVE, 0, read, write, &[]));
        read = write;
    }
    work.push(shrink);
    let across = Picture::Work(work.len() - 1);
    draws.push(kernel(GAUSS, 0, read, across, &[at_level]));
    draws.push(kernel(GAUSS, 1, across, read, &[at_level]));
    draws.push(kernel(SMOOTH_UP, 0, read, Picture::Current, &[]));
    Plan { work, draws }
}

pub(crate) struct Effects {
    pipeline: wgpu::RenderPipeline,
    /// The chain's last draw, into the canvas format: from encoded values
    /// as they are, or from light.
    last_encoded: wgpu::RenderPipeline,
    last_light: wgpu::RenderPipeline,
    /// A step's last draw, converting for the step after it: into encoded
    /// values from light, and into light from encoded values.
    draw_into_encoded: wgpu::RenderPipeline,
    draw_into_light: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The whole picture between encoded values and light, into a half
    /// float picture: `into_light` from the canvas or from encoded values,
    /// `into_encoded` from light.
    into_light: wgpu::RenderPipeline,
    into_encoded: wgpu::RenderPipeline,
    /// The result back into the canvas format, from light or as it is.
    out_of_light: wgpu::RenderPipeline,
    out_of_encoded: wgpu::RenderPipeline,
    convert_layout: wgpu::BindGroupLayout,
    pool: std::sync::Arc<crate::pool::Pool>,
}

/// A full-screen pipeline over `layout` drawing `entry` of `module` into
/// `format`.
fn full_screen(
    device: &wgpu::Device,
    label: &'static str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry: &'static str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl Effects {
    pub(crate) fn new(device: &wgpu::Device, pool: std::sync::Arc<crate::pool::Pool>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render effect"),
            source: wgpu::ShaderSource::Wgsl(format!("{LIGHT}{SHADER}").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render effect"),
            entries: &[
                texture_entry(0, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS),
                    },
                    count: None,
                },
                texture_entry(3, true),
            ],
        });
        let pipeline = full_screen(
            device,
            "cutlass-render effect",
            &module,
            &layout,
            "fs",
            LIGHT_FORMAT,
        );
        let last_encoded = full_screen(
            device,
            "cutlass-render effect last",
            &module,
            &layout,
            "fs_last_encoded",
            CANVAS_FORMAT,
        );
        let last_light = full_screen(
            device,
            "cutlass-render effect last from light",
            &module,
            &layout,
            "fs_last_light",
            CANVAS_FORMAT,
        );
        let draw_into_encoded = full_screen(
            device,
            "cutlass-render effect into encoded",
            &module,
            &layout,
            "fs_into_encoded",
            LIGHT_FORMAT,
        );
        let draw_into_light = full_screen(
            device,
            "cutlass-render effect into light",
            &module,
            &layout,
            "fs_into_light",
            LIGHT_FORMAT,
        );
        let convert = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render effect light"),
            source: wgpu::ShaderSource::Wgsl(format!("{LIGHT}{CONVERT}").into()),
        });
        let convert_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render effect light"),
            entries: &[texture_entry(0, false)],
        });
        let converting = |label, entry, format| {
            full_screen(device, label, &convert, &convert_layout, entry, format)
        };
        let into_light = converting(
            "cutlass-render effect into light",
            "fs_decode",
            LIGHT_FORMAT,
        );
        let into_encoded = converting(
            "cutlass-render effect into encoded",
            "fs_encode",
            LIGHT_FORMAT,
        );
        let out_of_light = converting(
            "cutlass-render effect out of light",
            "fs_encode_out",
            CANVAS_FORMAT,
        );
        let out_of_encoded = converting(
            "cutlass-render effect copy out",
            "fs_copy_out",
            CANVAS_FORMAT,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass-render effect"),
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        Self {
            pool,
            pipeline,
            last_encoded,
            last_light,
            draw_into_encoded,
            draw_into_light,
            layout,
            sampler,
            into_light,
            into_encoded,
            out_of_light,
            out_of_encoded,
            convert_layout,
        }
    }

    /// Run `steps` over `input`, a `size` picture in the canvas format.
    /// Returns a new picture in the canvas format holding the result; with
    /// no steps, `input` itself.
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        size: (u32, u32),
        steps: &[Step],
    ) -> wgpu::TextureView {
        if steps.is_empty() {
            return input.clone();
        }
        let mut to = self.light(device, size);
        // The pictures kept for a blend, each with the values it holds;
        // the first is the input, as the canvas bytes it is.
        let mut kept: [Option<(wgpu::TextureView, Space)>; MAX_SAVED + 1] = Default::default();
        kept[0] = Some((input.clone(), Space::Display));
        // The canvas holds encoded values: a first step that wants them
        // reads the input itself, and one that wants light reads it
        // decoded. The input is never drawn over, so it is held.
        let spaces = spaces(steps);
        let mut current = spaces[0];
        let (mut from, mut held) = match current {
            Space::Display => (input.clone(), true),
            Space::Light => {
                let decoded = self.light(device, size);
                self.convert(device, encoder, &self.into_light, input, &decoded);
                (decoded, false)
            }
        };
        let plans: Vec<Plan> = steps.iter().map(|step| plan(step, size)).collect();
        // The last step's last draw writes the result into the canvas
        // format itself, unless it keeps a picture or draws nothing there.
        let folds = steps.last().is_some_and(|step| step.save.is_none())
            && plans
                .last()
                .and_then(|plan| plan.draws.last())
                .is_some_and(|draw| draw.write == Picture::Current);
        let output = crate::blur::scratch(&self.pool, device, size);
        for (index, (step, plan)) in steps.iter().zip(&plans).enumerate() {
            // Converted here only when the step before could not: it drew
            // nothing, or it kept its picture.
            let wanted = spaces[index];
            if wanted != current {
                let pipeline = match wanted {
                    Space::Light => &self.into_light,
                    Space::Display => &self.into_encoded,
                };
                self.convert(device, encoder, pipeline, &from, &to);
                self.turn(device, size, &mut from, &mut to, &mut held);
                current = wanted;
            }
            let work: Vec<(wgpu::TextureView, (u32, u32))> = plan
                .work
                .iter()
                .map(|shrink| {
                    let shrunk = shrunk_size(size, *shrink);
                    (self.light(device, shrunk), shrunk)
                })
                .collect();
            for (number, draw) in plan.draws.iter().enumerate() {
                let mut draw = *draw;
                let ends = number + 1 == plan.draws.len() && draw.write == Picture::Current;
                let last = folds && ends && index + 1 == steps.len();
                // The step's last draw hands the next step the values it
                // wants, unless the step keeps its picture as it is.
                let next = spaces
                    .get(index + 1)
                    .copied()
                    .filter(|next| ends && *next != current && step.save.is_none());
                let read = match draw.read {
                    Picture::Work(index) => &work[index].0,
                    _ => &from,
                };
                let other = match draw.other {
                    Some(Picture::Work(index)) => &work[index].0,
                    Some(Picture::Kept(slot)) => match kept.get(usize::from(slot)) {
                        Some(Some((view, space))) => {
                            // Tell the kernel when it reads encoded values.
                            draw.arg = u32::from(*space == Space::Display);
                            view
                        }
                        _ => &from,
                    },
                    Some(Picture::Current) => &from,
                    None => read,
                };
                let (target, target_size) = match draw.write {
                    Picture::Work(index) => (&work[index].0, work[index].1),
                    _ if last => (&output, size),
                    _ => (&to, size),
                };
                let pipeline = match (last, current, next) {
                    (true, Space::Display, _) => &self.last_encoded,
                    (true, Space::Light, _) => &self.last_light,
                    (false, _, Some(Space::Display)) => &self.draw_into_encoded,
                    (false, _, Some(Space::Light)) => &self.draw_into_light,
                    (false, _, None) => &self.pipeline,
                };
                self.pass(
                    device,
                    queue,
                    encoder,
                    pipeline,
                    read,
                    other,
                    target,
                    target_size,
                    &draw,
                );
                if last {
                    return output;
                }
                if draw.write == Picture::Current {
                    self.turn(device, size, &mut from, &mut to, &mut held);
                }
                if let Some(next) = next {
                    current = next;
                }
            }
            if let Some(slot) = step.save.and_then(|slot| kept.get_mut(usize::from(slot))) {
                // Set aside as it is; the chain goes on in a new texture.
                *slot = Some((from.clone(), current));
                held = true;
            }
        }
        let last = match current {
            Space::Light => &self.out_of_light,
            Space::Display => &self.out_of_encoded,
        };
        self.convert(device, encoder, last, &from, &output);
        output
    }

    /// Make what was just drawn into `to` the picture so far, and hand
    /// the old one back to be drawn into next, unless it is kept.
    fn turn(
        &self,
        device: &wgpu::Device,
        size: (u32, u32),
        from: &mut wgpu::TextureView,
        to: &mut wgpu::TextureView,
        held: &mut bool,
    ) {
        std::mem::swap(from, to);
        if *held {
            *to = self.light(device, size);
            *held = false;
        }
    }

    /// A `size` half-float picture the steps take turns drawing into.
    fn light(&self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
        self.pool
            .texture(
                device,
                "cutlass-render effect light",
                TextureKey {
                    size,
                    format: LIGHT_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                },
            )
            .1
    }

    /// Draw `input` through `pipeline` into `output`, texel for texel.
    fn convert(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) {
        let group = self.pool.group(
            device,
            "cutlass-render effect light",
            &self.convert_layout,
            &[(0, Bind::View(input))],
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render effect light"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// One draw: `read` and `other` sampled, `to`, a `size` picture,
    /// drawn.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        read: &wgpu::TextureView,
        other: &wgpu::TextureView,
        to: &wgpu::TextureView,
        size: (u32, u32),
        draw: &Draw,
    ) {
        let mut bytes = Vec::with_capacity(PARAMS as usize);
        bytes.extend_from_slice(&draw.kernel.to_le_bytes());
        bytes.extend_from_slice(&draw.arg.to_le_bytes());
        for value in [size.0 as f32, size.1 as f32] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in draw.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let buffer = self.pool.uniform(
            device,
            queue,
            "cutlass-render effect params",
            PARAMS,
            &bytes,
        );
        let group = self.pool.group(
            device,
            "cutlass-render effect",
            &self.layout,
            &[
                (0, Bind::View(read)),
                (1, Bind::Sampler(&self.sampler)),
                (2, Bind::Buffer(&buffer)),
                (3, Bind::View(other)),
            ],
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render effect"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: to,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{Block, MAX_VALUES, Step, bundled};
    use cutlass_gpu::Placement;
    use cutlass_model::Micros;

    use crate::{Layer, Renderer, Rgba};

    fn renderer() -> Option<Renderer> {
        let renderer = Renderer::for_test().expect("a GPU is required");
        if renderer.is_none() {
            eprintln!("skipped: no GPU");
        }
        renderer
    }

    fn step(block: Block, set: &[(usize, f32)]) -> Step {
        let mut values = [0.0; MAX_VALUES];
        for (slot, param) in values.iter_mut().zip(block.params()) {
            *slot = param.default;
        }
        for (index, value) in set {
            values[*index] = *value;
        }
        Step::new(block, values)
    }

    /// Red, green, blue, white, all opaque.
    const ROW: [u8; 16] = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];

    fn draw(renderer: &Renderer, bytes: &[u8], size: (u32, u32), steps: &[Step]) -> Vec<u8> {
        let canvas = renderer.canvas(size.0, size.1).expect("canvas");
        let picture = Rgba {
            width: size.0,
            height: size.1,
            bytes,
        };
        renderer
            .draw(
                &canvas,
                &[
                    Layer::Rgba(picture, Placement::canvas(size.0, size.1)),
                    Layer::Effect(steps, None),
                ],
            )
            .expect("draw");
        renderer.read(&canvas).expect("read")
    }

    fn close(got: &[u8], expected: &[u8]) {
        assert_eq!(got.len(), expected.len());
        for (index, (a, b)) in got.iter().zip(expected).enumerate() {
            assert!(a.abs_diff(*b) <= 2, "byte {index}: {got:?} vs {expected:?}");
        }
    }

    #[test]
    fn no_steps_and_resting_steps_leave_the_picture_alone() {
        let Some(renderer) = renderer() else { return };
        close(&draw(&renderer, &ROW, (4, 1), &[]), &ROW);
        let resting = [
            step(Block::Transform, &[]),
            step(Block::RgbSplit, &[]),
            step(Block::Blur, &[]),
            step(Block::Grade, &[]),
            step(Block::Flash, &[]),
            step(Block::Pixelate, &[]),
            step(Block::Invert, &[(0, 0.0)]),
            step(Block::ZoomBlur, &[]),
            step(Block::MotionBlur, &[]),
            step(Block::Swirl, &[]),
            step(Block::Ripple, &[]),
            step(Block::Bulge, &[]),
            step(Block::Glow, &[]),
            step(Block::LightLeak, &[]),
            step(Block::Hue, &[]),
            step(Block::Tile, &[]),
            step(Block::Posterize, &[]),
            step(Block::Edges, &[]),
            step(Block::GradientMap, &[]),
            step(Block::Grid, &[]),
            step(Block::Spotlight, &[]),
            step(Block::Reveal, &[]),
            step(Block::Stretch, &[]),
            step(Block::Dust, &[]),
            step(Block::Kaleido, &[]),
            step(Block::Fringe, &[]),
            step(Block::Halftone, &[]),
            step(Block::Prints, &[]),
            step(Block::Blocks, &[]),
            step(Block::Rays, &[]),
            step(Block::Focus, &[]),
            step(Block::Place, &[]),
        ];
        close(&draw(&renderer, &ROW, (4, 1), &resting), &ROW);
    }

    #[test]
    fn invert_grade_and_flash_change_colour_as_named() {
        let Some(renderer) = renderer() else { return };
        let inverted = draw(&renderer, &ROW, (4, 1), &[step(Block::Invert, &[])]);
        close(
            &inverted,
            &[
                0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 0, 0, 255,
            ],
        );
        let grey = draw(&renderer, &ROW, (4, 1), &[step(Block::Grade, &[(2, 0.0)])]);
        for pixel in grey.chunks(4) {
            assert!(pixel[0].abs_diff(pixel[1]) <= 1 && pixel[1].abs_diff(pixel[2]) <= 1);
        }
        let white = draw(&renderer, &ROW, (4, 1), &[step(Block::Flash, &[(0, 1.0)])]);
        close(&white, &[255; 16]);
    }

    /// Into light and back leaves every byte, at every alpha, where it
    /// was: the half-float picture between keeps all eight bits.
    #[test]
    fn every_byte_comes_back_from_light_as_it_went_in() {
        let Some(renderer) = renderer() else { return };
        let alphas = [255u8, 200, 128, 37, 1, 0];
        let (width, height) = (256u32, alphas.len() as u32);
        let mut scene = Vec::with_capacity((width * height * 4) as usize);
        for alpha in alphas {
            for value in 0..=255u8 {
                // Premultiplied: no channel above its alpha.
                let v = value.min(alpha);
                scene.extend_from_slice(&[v, v / 2, alpha - v / 3 * u8::from(alpha > 0), alpha]);
            }
        }
        // A step that moves nothing, in light; one at rest on encoded
        // values; and a chain that goes from one to the other and back.
        let (moving, grading) = (step(Block::Transform, &[]), step(Block::Grade, &[]));
        for chain in [
            vec![moving],
            vec![grading],
            vec![moving, grading, moving, grading],
        ] {
            let got = draw(&renderer, &scene, (width, height), &chain);
            for (index, (a, b)) in got.iter().zip(&scene).enumerate() {
                assert!(a.abs_diff(*b) <= 1, "{chain:?}, byte {index}: {a} vs {b}");
            }
        }
    }

    /// A blur averages light, not encoded values: black and white mixed
    /// half and half is half the light, which encodes far above half way.
    #[test]
    fn a_blur_averages_light_so_bright_detail_stays_bright() {
        let Some(renderer) = renderer() else { return };
        // Black and white columns, one pixel each.
        let (width, height) = (64u32, 4u32);
        let scene: Vec<u8> = (0..width * height)
            .flat_map(|index| {
                if index % 2 == 0 {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                }
            })
            .collect();
        let blur = step(Block::Blur, &[(0, 2.0)]);
        let got = draw(&renderer, &scene, (width, height), &[blur]);
        // About half of white's light, Rec. 709 encoded: 1.099 * 0.5^0.45
        // - 0.099 is 0.705, or 180 of 255; the 17 taps take 8 or 9 white
        // columns, 175 or 185. Averaging the bytes would give near 128.
        let middle = &got[((2 * width + 32) * 4) as usize..][..4];
        for channel in &middle[..3] {
            assert!((172..=188).contains(channel), "{middle:?}");
        }
        assert_eq!(middle[3], 255);
    }

    /// Every draw of every block reads only pictures already drawn, never
    /// the one it draws, and a block's working pictures are all drawn.
    fn check_plan(plan: &super::Plan, what: &str) {
        use super::Picture;
        let mut drawn = vec![false; plan.work.len()];
        for draw in &plan.draws {
            for read in std::iter::once(draw.read).chain(draw.other) {
                if let Picture::Work(index) = read {
                    assert!(
                        drawn[index],
                        "{what}: reads work {index} before it is drawn"
                    );
                }
                assert!(
                    read != draw.write || read == Picture::Current,
                    "{what}: draws the picture it reads"
                );
            }
            if let Picture::Work(index) = draw.write {
                drawn[index] = true;
            }
        }
        assert!(
            drawn.iter().all(|done| *done),
            "{what}: a working picture never drawn"
        );
        for shrink in &plan.work {
            assert!(
                shrink.is_power_of_two() && *shrink <= 4096,
                "{what}"
            );
        }
    }

    #[test]
    fn every_block_plan_reads_only_what_it_has_drawn() {
        use cutlass_effects::Block;
        for size in [(1, 1), (64, 4), (1920, 1080), (3840, 2160)] {
            for block in Block::ALL {
                for at in [0.0, 0.5, 1.0] {
                    let mut values = [0.0; MAX_VALUES];
                    for (slot, param) in values.iter_mut().zip(block.params()) {
                        *slot = param.min + (param.max - param.min) * at;
                    }
                    let plan = super::plan(&Step::new(block, values), size);
                    check_plan(&plan, &format!("{block:?} at {at} on {size:?}"));
                }
            }
        }
    }

    /// However wide, a blur is a handful of draws on a few pictures, none
    /// of whose kernels takes more taps as the blur grows.
    #[test]
    fn a_blur_plan_stays_small_however_wide() {
        // Up to the widest blur on an 8K picture: a tenth of its short
        // side, about 430 pixels.
        for reach in [0.0, 1.0, 5.0, 40.0, 200.0, 440.0] {
            let plan = super::blur_plan(reach);
            assert!(plan.draws.len() <= 9, "{reach}: {plan:?}");
            assert!(plan.work.len() <= 7, "{reach}: {plan:?}");
            for draw in &plan.draws {
                if draw.kernel == super::GAUSS {
                    // Sigma in texels of the picture it is drawn on.
                    assert!(draw.values[0] <= 8.0, "{reach}: {plan:?}");
                }
            }
        }
        assert!(super::blur_plan(0.0).draws.is_empty());
    }

    #[test]
    fn a_glow_plan_stays_small_on_every_picture() {
        let widest = Block::Glow.params()[1].max;
        for size in [(1, 1), (64, 64), (1920, 1080), (3840, 2160), (7680, 4320)] {
            for radius in [0.0, 0.01, 0.05, widest] {
                let mut values = [0.0; MAX_VALUES];
                values[..7].copy_from_slice(&[1.0, radius, 0.6, 0.5, 1.0, 1.0, 1.0]);
                let plan = super::glow_plan(&values, size);
                // Bright part, halvings, the way back up, the add.
                assert!(plan.draws.len() <= 14, "{size:?} {radius}: {plan:?}");
                assert!(plan.work.len() <= 13, "{size:?} {radius}: {plan:?}");
                // No level is smaller than two texels across.
                let short = size.0.min(size.1);
                assert!(
                    plan.work.iter().all(|shrink| short / shrink >= 2 || *shrink == 1),
                    "{size:?} {radius}: {plan:?}"
                );
            }
        }
        let mut off = [0.0; MAX_VALUES];
        off[1] = 0.05;
        assert!(super::glow_plan(&off, (1920, 1080)).draws.is_empty());
    }

    /// The light a strong glow adds to a white square a tenth of the
    /// picture wide, at points a fortieth and a twentieth of the picture
    /// beside it.
    fn glow_beside(renderer: &Renderer, side: u32) -> [f64; 2] {
        let square = side / 10;
        let start = side / 2 - square / 2;
        let inside = |v: u32| v >= start && v < start + square;
        let scene: Vec<u8> = (0..side * side)
            .flat_map(|index| {
                if inside(index % side) && inside(index / side) {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                }
            })
            .collect();
        let glow = step(Block::Glow, &[(0, 3.0), (1, 0.2)]);
        let got = draw(renderer, &scene, (side, side), &[glow]);
        let middle = side / 2;
        let edge = start + square;
        [side / 40, side / 20].map(|away| {
            let x = edge + away;
            light(got[((middle * side + x) * 4) as usize])
        })
    }

    #[test]
    fn a_glow_spreads_alike_at_every_size() {
        let Some(renderer) = renderer() else { return };
        // The small picture glows on its own pixels; the large one from a
        // halved copy, as a 4K frame does.
        let small = glow_beside(&renderer, 200);
        let large = glow_beside(&renderer, 1200);
        for (s, l) in small.iter().zip(&large) {
            assert!(*s > 0.01 && *l > 0.01, "{small:?} {large:?}");
            assert!((s / l - 1.0).abs() < 0.25, "{small:?} {large:?}");
        }
        // It falls off away from the light.
        assert!(small[1] < small[0] && large[1] < large[0]);
    }

    #[test]
    fn a_glow_grows_with_brightness_and_holds_back_below_its_threshold() {
        let Some(renderer) = renderer() else { return };
        let size = (32u32, 32u32);
        // The glow beside a patch in the middle, as linear light.
        let beside = |level: u8, threshold: f32| {
            let scene: Vec<u8> = (0..size.0 * size.1)
                .flat_map(|index| {
                    let (x, y) = (index % size.0, index / size.0);
                    let v = if (12..20).contains(&x) && (12..20).contains(&y) {
                        level
                    } else {
                        0
                    };
                    [v, v, v, 255]
                })
                .collect();
            let glow = step(Block::Glow, &[(0, 1.0), (1, 0.2), (2, threshold)]);
            let got = draw(&renderer, &scene, size, &[glow]);
            light(got[((16 * size.0 + 22) * 4) as usize])
        };
        let white = beside(255, 0.6);
        let light_grey = beside(220, 0.6);
        assert!(white > light_grey && light_grey > 0.0, "{white} {light_grey}");
        // Below the knee nothing glows; a higher threshold glows less.
        assert_eq!(beside(90, 0.6), 0.0);
        assert!(beside(255, 0.9) < white);
    }

    #[test]
    fn a_glow_takes_its_tint() {
        let Some(renderer) = renderer() else { return };
        let size = (16u32, 16u32);
        let spot: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                let (x, y) = (index % size.0, index / size.0);
                let v = if (6..10).contains(&x) && (6..10).contains(&y) {
                    255
                } else {
                    0
                };
                [v, v, v, 255]
            })
            .collect();
        // A red glow from white light.
        let red = step(
            Block::Glow,
            &[(0, 2.0), (1, 0.3), (4, 1.0), (5, 0.0), (6, 0.0)],
        );
        let got = draw(&renderer, &spot, size, &[red]);
        let beside = ((8 * size.0 + 12) * 4) as usize;
        assert!(got[beside] > 0, "{:?}", &got[beside..beside + 4]);
        assert_eq!(got[beside + 1], 0);
        assert_eq!(got[beside + 2], 0);
    }

    /// The blocks that average light work in light; the ones tuned by eye
    /// on encoded values are handed encoded values.
    #[test]
    fn blurs_work_in_light_and_colour_looks_on_encoded_values() {
        use super::{Space, space};
        for block in [
            Block::Blur,
            Block::ZoomBlur,
            Block::MotionBlur,
            Block::Focus,
            Block::Glow,
            Block::Blend,
        ] {
            assert_eq!(space(block), Some(Space::Light), "{block:?}");
        }
        for block in [
            Block::Grade,
            Block::Hue,
            Block::Posterize,
            Block::GradientMap,
            Block::Rays,
            Block::LightLeak,
        ] {
            assert_eq!(space(block), Some(Space::Display), "{block:?}");
        }
        for block in [Block::Transform, Block::Swirl, Block::Slices, Block::RgbSplit] {
            assert_eq!(space(block), None, "{block:?}");
        }
    }

    #[test]
    fn a_block_that_moves_pixels_runs_on_what_the_chain_holds() {
        use super::Space::{Display, Light};
        let chain = |blocks: &[Block]| {
            let steps: Vec<Step> = blocks.iter().map(|block| step(*block, &[])).collect();
            super::spaces(&steps)
        };
        // A VHS look: moved, then graded, all on encoded values.
        assert_eq!(
            chain(&[Block::Slices, Block::RgbSplit, Block::Grade, Block::Noise]),
            [Display; 4]
        );
        assert_eq!(chain(&[Block::Transform, Block::Blur]), [Light, Light]);
        assert_eq!(
            chain(&[Block::Blur, Block::Swirl, Block::Grade, Block::Mirror]),
            [Light, Light, Display, Display]
        );
        assert_eq!(chain(&[Block::Transform]), [Display]);
    }

    /// Linear light of a Rec. 709 encoded byte.
    fn light(byte: u8) -> f64 {
        let e = f64::from(byte) / 255.0;
        if e < 0.081 {
            e / 4.5
        } else {
            ((e + 0.099) / 1.099).powf(1.0 / 0.45)
        }
    }

    /// The spread of the light in `row` about `middle`, in pixels.
    fn spread(row: &[u8], middle: f64) -> f64 {
        let (mut total, mut moment) = (0.0, 0.0);
        for (x, byte) in row.iter().enumerate() {
            let weight = light(*byte);
            let d = x as f64 + 0.5 - middle;
            total += weight;
            moment += weight * d * d;
        }
        (moment / total).sqrt()
    }

    #[test]
    fn a_blur_spreads_a_line_evenly_with_no_gaps() {
        let Some(renderer) = renderer() else { return };
        // One white column in the middle of 128 black ones.
        let (width, height) = (128u32, 4u32);
        let scene: Vec<u8> = (0..width * height)
            .flat_map(|index| {
                if index % width == 64 {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                }
            })
            .collect();
        // A reach of 40 pixels: ten times the 4 pixel short side. Wide
        // enough to be drawn on a picture an eighth of the size.
        let blur = step(Block::Blur, &[(0, 10.0)]);
        let got = draw(&renderer, &scene, (width, height), &[blur]);
        let row: Vec<u8> = got[..(width * 4) as usize]
            .chunks(4)
            .map(|pixel| pixel[0])
            .collect();
        // Every column near the line gets some of it, the same either
        // side, less the further away, with no gap or step.
        for offset in 1..=38 {
            let (left, right) = (row[64 - offset], row[64 + offset]);
            assert!(
                left > 0 && right > 0,
                "a gap {offset} from the line: {row:?}"
            );
            assert!(left.abs_diff(right) <= 2, "lopsided at {offset}: {row:?}");
            assert!(
                right <= row[64 + offset - 1] + 1,
                "a step at {offset}: {row:?}"
            );
        }
        // Spread as a box of 40 each side would: 40 / sqrt(3).
        let got = spread(&row, 64.5);
        assert!((got - 23.1).abs() < 3.0, "spread {got}: {row:?}");
    }

    /// The same blur is the same share of the picture however large the
    /// picture is: drawn whole on a small one, on a smaller copy of a
    /// large one, the spread agrees.
    #[test]
    fn a_blur_spreads_alike_at_every_size() {
        let Some(renderer) = renderer() else { return };
        let mut spreads = Vec::new();
        for side in [64u32, 512] {
            // A white line a 64th of the picture wide, in the middle.
            let line = side / 64;
            let scene: Vec<u8> = (0..side * side)
                .flat_map(|index| {
                    let x = index % side;
                    if x >= side / 2 && x < side / 2 + line {
                        [255, 255, 255, 255]
                    } else {
                        [0, 0, 0, 255]
                    }
                })
                .collect();
            let blur = step(Block::Blur, &[(0, 0.08)]);
            let got = draw(&renderer, &scene, (side, side), &[blur]);
            let middle = ((side / 2) * side * 4) as usize;
            let row: Vec<u8> = got[middle..middle + (side * 4) as usize]
                .chunks(4)
                .map(|pixel| pixel[0])
                .collect();
            let centre = f64::from(side / 2) + f64::from(line) / 2.0;
            spreads.push(spread(&row, centre) / f64::from(side));
        }
        // 0.08 of the side each way as a box: 0.046 of the side.
        for got in &spreads {
            assert!((got - 0.0462).abs() < 0.004, "{spreads:?}");
        }
    }

    #[test]
    fn mirror_copies_the_left_half_onto_the_right() {
        let Some(renderer) = renderer() else { return };
        let mirrored = draw(&renderer, &ROW, (4, 1), &[step(Block::Mirror, &[])]);
        close(
            &mirrored,
            &[
                255, 0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 0, 0, 255,
            ],
        );
    }

    #[test]
    fn an_effect_changes_only_what_is_below_it() {
        let Some(renderer) = renderer() else { return };
        let canvas = renderer.canvas(4, 1).expect("canvas");
        let blue = [0, 0, 255, 255];
        let mut place = Placement::canvas(1, 1);
        place.to_picture = cutlass_gpu::Affine([1.0, 0.0, -3.0, 0.0, 1.0, 0.0]);
        let steps = [step(Block::Invert, &[])];
        renderer
            .draw(
                &canvas,
                &[
                    Layer::Rgba(
                        Rgba {
                            width: 4,
                            height: 1,
                            bytes: &ROW,
                        },
                        Placement::canvas(4, 1),
                    ),
                    Layer::Effect(&steps, None),
                    Layer::Rgba(
                        Rgba {
                            width: 1,
                            height: 1,
                            bytes: &blue,
                        },
                        place,
                    ),
                ],
            )
            .expect("draw");
        let got = renderer.read(&canvas).expect("read");
        close(
            &got[..12],
            &[0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255],
        );
        close(&got[12..], &blue);
    }

    #[test]
    fn colour_steps_keep_transparent_pixels_transparent() {
        let Some(renderer) = renderer() else { return };
        let clear = [0u8; 16];
        let steps = [
            step(Block::Grade, &[(0, 0.5), (1, 2.0)]),
            step(Block::Invert, &[]),
            step(Block::Noise, &[(0, 1.0), (1, 9.0)]),
            step(Block::Vignette, &[(0, 1.0)]),
        ];
        close(&draw(&renderer, &clear, (4, 1), &steps), &clear);
    }

    fn solid(size: (u32, u32), pixel: [u8; 4]) -> Vec<u8> {
        pixel.repeat((size.0 * size.1) as usize)
    }

    #[test]
    fn hue_posterize_and_tile_change_colour_and_layout_as_named() {
        let Some(renderer) = renderer() else { return };
        // A third of the way round the wheel, red becomes green.
        let red = [255, 0, 0, 255];
        close(
            &draw(&renderer, &red, (1, 1), &[step(Block::Hue, &[(0, 120.0)])]),
            &[0, 255, 0, 255],
        );
        let greys = [100, 100, 100, 255, 200, 200, 200, 255];
        close(
            &draw(
                &renderer,
                &greys,
                (2, 1),
                &[step(Block::Posterize, &[(0, 2.0)])],
            ),
            &[0, 0, 0, 255, 255, 255, 255, 255],
        );
        // Two copies across: red, green, blue, white, twice.
        let wide: Vec<u8> = ROW.chunks(4).flat_map(|pixel| pixel.repeat(2)).collect();
        let tiled = draw(&renderer, &wide, (8, 1), &[step(Block::Tile, &[(0, 2.0)])]);
        close(&tiled, &[ROW, ROW].concat());
    }

    #[test]
    fn moving_blocks_only_move_pixels_so_a_flat_colour_stays_flat() {
        let Some(renderer) = renderer() else { return };
        let size = (16, 12);
        let flat = solid(size, [40, 120, 200, 255]);
        for moved in [
            step(Block::ZoomBlur, &[(0, 0.4)]),
            step(Block::MotionBlur, &[(0, 0.2), (1, 30.0)]),
            step(Block::Swirl, &[(0, 360.0)]),
            step(Block::Ripple, &[(0, 0.05), (3, 2.0)]),
            step(Block::Bulge, &[(0, 0.8)]),
            step(Block::Bulge, &[(0, -0.8)]),
            step(Block::Grid, &[(0, 3.0), (1, 2.0), (2, 1.5)]),
            step(Block::Kaleido, &[(0, 6.0), (1, 30.0)]),
            step(Block::Fringe, &[(0, 0.1)]),
            step(Block::Stretch, &[(0, 1.5), (1, 0.7)]),
        ] {
            close(&draw(&renderer, &flat, size, &[moved]), &flat);
        }
    }

    #[test]
    fn a_motion_blur_smears_along_its_angle_only() {
        let Some(renderer) = renderer() else { return };
        let size = (16u32, 16u32);
        // One white column on black.
        let column: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                let v = if index % size.0 == 8 { 255 } else { 0 };
                [v, v, v, 255]
            })
            .collect();
        let across = draw(
            &renderer,
            &column,
            size,
            &[step(Block::MotionBlur, &[(0, 0.2)])],
        );
        assert!(across[4 * 6] > 0, "the light spreads sideways");
        let down = draw(
            &renderer,
            &column,
            size,
            &[step(Block::MotionBlur, &[(0, 0.2), (1, 90.0)])],
        );
        close(&down, &column);
    }

    #[test]
    fn glow_spreads_only_light_above_its_threshold() {
        let Some(renderer) = renderer() else { return };
        let size = (15u32, 15u32);
        // A white patch three pixels square in the middle of black.
        let spot: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                let (x, y) = (index % size.0, index / size.0);
                let v = if (6..9).contains(&x) && (6..9).contains(&y) {
                    255
                } else {
                    0
                };
                [v, v, v, 255]
            })
            .collect();
        let glowing = draw(
            &renderer,
            &spot,
            size,
            &[step(Block::Glow, &[(0, 3.0), (1, 0.3)])],
        );
        let beside = 4 * (7 * size.0 + 4) as usize;
        assert!(glowing[beside] > 0, "the light reaches past the patch");
        let dim = solid(size, [80, 80, 80, 255]);
        close(
            &draw(&renderer, &dim, size, &[step(Block::Glow, &[(0, 3.0)])]),
            &dim,
        );
    }

    /// A blend of mode `mode` and `amount` over the kept picture `with`.
    fn blend(mode: f32, amount: f32, with: u8) -> Step {
        let mut blend = step(Block::Blend, &[(0, mode), (1, amount)]);
        blend.with = Some(with);
        blend
    }

    /// `block` with `set`, keeping its result in `slot`.
    fn saving(block: Block, set: &[(usize, f32)], slot: u8) -> Step {
        let mut saving = step(block, set);
        saving.save = Some(slot);
        saving
    }

    /// Black, a mid grey and white, with a dim red, in one row.
    const LEVELS: [u8; 16] = [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 90, 20, 20, 255];

    #[test]
    fn a_blend_lays_the_original_back_by_its_mode_and_amount() {
        let Some(renderer) = renderer() else { return };
        let size = (4u32, 1u32);
        let invert = step(Block::Invert, &[(0, 1.0)]);
        let inverted = draw(&renderer, &LEVELS, size, &[invert]);
        // At amount 0 the picture so far; mixing all the way, the original.
        for mode in 0..=4 {
            let got = draw(&renderer, &LEVELS, size, &[invert, blend(mode as f32, 0.0, 0)]);
            close(&got, &inverted);
        }
        close(&draw(&renderer, &LEVELS, size, &[invert, blend(4.0, 1.0, 0)]), &LEVELS);
        // Each mode in light, against the inverted picture below.
        let by_mode = |mode: f32| draw(&renderer, &LEVELS, size, &[invert, blend(mode, 1.0, 0)]);
        let (added, screened, multiplied, lighter) =
            (by_mode(0.0), by_mode(1.0), by_mode(2.0), by_mode(3.0));
        for channel in (0..16).filter(|index| index % 4 != 3) {
            let (below, laid) = (light(inverted[channel]), light(LEVELS[channel]));
            let near = |got: u8, want: f64, what: &str| {
                let want = want.min(1.0);
                assert!(
                    (light(got) - want).abs() < 0.01 + want * 0.03,
                    "{what} channel {channel}: {} vs {want}",
                    light(got)
                );
            };
            near(added[channel], below + laid, "add");
            near(screened[channel], below + laid * (1.0 - below), "screen");
            near(multiplied[channel], below * laid, "multiply");
            near(lighter[channel], below.max(laid), "lighten");
        }
        // Alpha is the picture so far's.
        assert!(added.chunks(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn a_kept_picture_outlasts_the_steps_after_it_in_either_space() {
        let Some(renderer) = renderer() else { return };
        let size = (16u32, 8u32);
        let scene: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                let (x, y) = (index % size.0, index / size.0);
                [(x * 16) as u8, (y * 32) as u8, 200, 255]
            })
            .collect();
        let soft = [(0, 0.2)];
        // Kept in light, by a blur.
        let blurred = draw(&renderer, &scene, size, &[step(Block::Blur, &soft)]);
        let invert = step(Block::Invert, &[(0, 1.0)]);
        let swirl = step(Block::Swirl, &[(0, 2.0)]);
        let kept_light = [
            saving(Block::Blur, &soft, 1),
            invert,
            swirl,
            invert,
            swirl,
            blend(4.0, 1.0, 1),
        ];
        close(&draw(&renderer, &scene, size, &kept_light), &blurred);
        // Kept as encoded values, by an invert, and blended in light.
        let inverted = draw(&renderer, &scene, size, &[invert]);
        let kept_encoded = [
            saving(Block::Invert, &[(0, 1.0)], 2),
            step(Block::Blur, &soft),
            invert,
            blend(4.0, 1.0, 2),
        ];
        close(&draw(&renderer, &scene, size, &kept_encoded), &inverted);
        // Two kept at once, and the original besides.
        let both = [
            saving(Block::Blur, &soft, 1),
            saving(Block::Invert, &[(0, 1.0)], 2),
            swirl,
            blend(4.0, 1.0, 1),
            blend(4.0, 0.0, 2),
        ];
        close(&draw(&renderer, &scene, size, &both), &blurred);
        let back = [saving(Block::Invert, &[(0, 1.0)], 1), swirl, blend(4.0, 1.0, 0)];
        close(&draw(&renderer, &scene, size, &back), &scene);
    }

    #[test]
    fn a_chain_ends_alike_however_its_last_step_ends() {
        let Some(renderer) = renderer() else { return };
        let size = (4u32, 1u32);
        let invert = step(Block::Invert, &[(0, 1.0)]);
        let inverted = draw(&renderer, &LEVELS, size, &[invert]);
        // A last step that keeps its picture, or draws nothing, is still
        // the chain's result.
        let keeping = saving(Block::Invert, &[(0, 1.0)], 1);
        close(&draw(&renderer, &LEVELS, size, &[keeping]), &inverted);
        let unlit = step(Block::Glow, &[(0, 0.0)]);
        close(&draw(&renderer, &LEVELS, size, &[invert, unlit]), &inverted);
        // Ending in light or in encoded values, from either.
        let turn = step(Block::Transform, &[]);
        close(&draw(&renderer, &LEVELS, size, &[invert, turn]), &inverted);
        close(&draw(&renderer, &LEVELS, size, &[turn, invert]), &inverted);
        close(&draw(&renderer, &LEVELS, size, &[turn]), &LEVELS);
    }

    #[test]
    fn a_change_of_space_drawn_in_a_step_matches_one_drawn_alone() {
        let Some(renderer) = renderer() else { return };
        let size = (16u32, 8u32);
        let scene: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                let (x, y) = (index % size.0, index / size.0);
                [(x * 16) as u8, (y * 32) as u8, 200, 255]
            })
            .collect();
        let soft = [(0, 0.2)];
        let grade = [(1, 1.3), (2, 0.8)];
        // A step that keeps its picture leaves the change to a pass of
        // its own; one that does not makes it in its last draw.
        for (first, then) in [
            ((Block::Blur, &soft[..]), (Block::Grade, &grade[..])),
            ((Block::Grade, &grade[..]), (Block::Blur, &soft[..])),
        ] {
            let folded = [step(first.0, first.1), step(then.0, then.1)];
            let apart = [saving(first.0, first.1, 1), step(then.0, then.1)];
            close(
                &draw(&renderer, &scene, size, &folded),
                &draw(&renderer, &scene, size, &apart),
            );
        }
    }

    #[test]
    fn a_recipe_screens_a_soft_copy_over_the_sharp_picture() {
        let Some(renderer) = renderer() else { return };
        let recipe = cutlass_effects::Recipe::read(
            br#"{"version":3,"id":"a","name":"A","passes":[
                {"block":"blur","params":{"radius":0.1},"save":"soft"},
                {"block":"blend","params":{"mode":4}},
                {"block":"blend","params":{"mode":1,"amount":0.5},"with":"soft"}]}"#,
        )
        .expect("a version 3 recipe");
        let steps = recipe.steps_at(Micros(0));
        let size = (16u32, 16u32);
        let got = draw(&renderer, &SPOT16, size, &steps);
        // The sharp picture is back, with soft light around the spot.
        let middle = ((8 * size.0 + 8) * 4) as usize;
        assert!(got[middle] >= 250, "{}", got[middle]);
        let beside = ((8 * size.0 + 12) * 4) as usize;
        assert!(got[beside] > 0 && got[beside] < got[middle]);
        assert_eq!(got[3], 255);
    }

    /// A white spot four pixels square in the middle of 16 by 16 black.
    const SPOT16: [u8; 1024] = {
        let mut bytes = [0u8; 1024];
        let mut index = 0;
        while index < 256 {
            let (x, y) = (index % 16, index / 16);
            let lit = x >= 6 && x < 10 && y >= 6 && y < 10;
            let v = if lit { 255 } else { 0 };
            bytes[index * 4] = v;
            bytes[index * 4 + 1] = v;
            bytes[index * 4 + 2] = v;
            bytes[index * 4 + 3] = 255;
            index += 1;
        }
        bytes
    };

    #[test]
    fn a_light_leak_only_brightens_and_edges_alone_leave_flat_colour_black() {
        let Some(renderer) = renderer() else { return };
        let size = (8u32, 8u32);
        let scene: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| [(index * 4) as u8, 60, 200, 255])
            .collect();
        let lit = draw(
            &renderer,
            &scene,
            size,
            &[step(Block::LightLeak, &[(0, 1.0)])],
        );
        assert!(
            lit.iter()
                .zip(&scene)
                .all(|(after, before)| u16::from(*after) + 1 >= u16::from(*before))
        );
        assert!(lit.iter().zip(&scene).any(|(after, before)| after > before));
        let flat = solid(size, [40, 120, 200, 255]);
        close(
            &draw(&renderer, &flat, size, &[step(Block::Edges, &[(1, 1.0)])]),
            &solid(size, [0, 0, 0, 255]),
        );
    }

    #[test]
    fn a_gradient_map_colours_by_brightness() {
        let Some(renderer) = renderer() else { return };
        // Black to white is grey by brightness.
        let grey = draw(
            &renderer,
            &[255, 0, 0, 255],
            (1, 1),
            &[step(Block::GradientMap, &[(0, 1.0)])],
        );
        assert!(grey[0].abs_diff(grey[1]) <= 1 && grey[1].abs_diff(grey[2]) <= 1);
        // Heat runs from black to white.
        let ends = [0, 0, 0, 255, 255, 255, 255, 255];
        close(
            &draw(
                &renderer,
                &ends,
                (2, 1),
                &[step(Block::GradientMap, &[(0, 1.0), (1, 1.0)])],
            ),
            &ends,
        );
    }

    #[test]
    fn a_reveal_shows_only_what_it_has_reached_and_a_stretch_can_leave_black() {
        let Some(renderer) = renderer() else { return };
        let white = solid((8, 1), [255, 255, 255, 255]);
        let black = solid((8, 1), [0, 0, 0, 255]);
        close(
            &draw(
                &renderer,
                &white,
                (8, 1),
                &[step(Block::Reveal, &[(0, 0.0)])],
            ),
            &black,
        );
        let half = draw(
            &renderer,
            &white,
            (8, 1),
            &[step(Block::Reveal, &[(0, 0.5), (3, 0.0)])],
        );
        close(
            &half,
            &[solid((4, 1), [255; 4]), solid((4, 1), [0, 0, 0, 255])].concat(),
        );
        let narrow = draw(
            &renderer,
            &white,
            (8, 1),
            &[step(Block::Stretch, &[(0, 0.5), (2, 1.0)])],
        );
        close(
            &narrow,
            &[
                solid((2, 1), [0, 0, 0, 255]),
                solid((4, 1), [255; 4]),
                solid((2, 1), [0, 0, 0, 255]),
            ]
            .concat(),
        );
    }

    #[test]
    fn the_new_reveals_open_from_where_they_are_named() {
        let Some(renderer) = renderer() else { return };
        let size = (32u32, 32u32);
        let white = solid(size, [255, 255, 255, 255]);
        let lit = |picture: &[u8], x: u32, y: u32| picture[4 * (y * size.0 + x) as usize] > 128;
        let reveal = |kind: f32, progress: f32| {
            draw(
                &renderer,
                &white,
                size,
                &[step(
                    Block::Reveal,
                    &[(0, progress), (1, kind), (3, 0.0), (4, 4.0)],
                )],
            )
        };
        // Doors open from the middle line out.
        let doors = reveal(3.0, 0.4);
        assert!(lit(&doors, 16, 16) && !lit(&doors, 1, 16));
        // A clock hand sweeps from the top: a quarter shows the top right.
        let clock = reveal(4.0, 0.25);
        assert!(lit(&clock, 24, 8) && !lit(&clock, 8, 24) && !lit(&clock, 8, 8));
        // A diamond opens from the middle; its corners come last.
        let diamond = reveal(5.0, 0.5);
        assert!(lit(&diamond, 16, 16) && !lit(&diamond, 1, 1));
        // Half way, one colour of the checkerboard is in and the other not.
        let checker = reveal(6.0, 0.5);
        assert!(lit(&checker, 4, 4) && !lit(&checker, 12, 4));
        // Half way, a dissolve shows grains all over, about half of them.
        let grains = reveal(7.0, 0.5);
        let shown = (0..size.0 * size.1)
            .filter(|at| grains[4 * *at as usize] > 128)
            .count();
        assert!((300..=724).contains(&shown), "{shown}");
    }

    #[test]
    fn blocks_rays_and_focus_do_as_named() {
        let Some(renderer) = renderer() else { return };
        let size = (64u32, 32u32);
        // Red on the left, blue on the right.
        let halves: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|at| {
                if at % size.0 < size.0 / 2 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                }
            })
            .collect();
        // Every block out of place moves some colour across the line, and
        // swapped ones bring in green.
        let broken = draw(
            &renderer,
            &halves,
            size,
            &[step(
                Block::Blocks,
                &[(0, 1.0), (1, 0.1), (2, 0.3), (4, 1.0)],
            )],
        );
        assert_ne!(broken, halves);
        assert!(broken.chunks(4).any(|pixel| pixel[1] > 200));
        // Rays only add light, and only from what is bright enough.
        let dark = solid(size, [40, 40, 40, 255]);
        let lit = draw(&renderer, &dark, size, &[step(Block::Rays, &[(0, 3.0)])]);
        close(&lit, &dark);
        // A bright middle streaks out over the dark around it.
        let middle: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|at| {
                let (x, y) = (at % size.0, at / size.0);
                if (24..40).contains(&x) && (8..24).contains(&y) {
                    [255, 255, 255, 255]
                } else {
                    [40, 40, 40, 255]
                }
            })
            .collect();
        let shone = draw(
            &renderer,
            &middle,
            size,
            &[step(Block::Rays, &[(0, 2.0), (3, 0.8)])],
        );
        assert!(
            shone
                .iter()
                .zip(&middle)
                .all(|(after, before)| u16::from(*after) + 1 >= u16::from(*before))
        );
        let i = 4 * (16 * size.0 + 12) as usize;
        assert!(shone[i] > 56, "streaked: {}", shone[i]);
        // A tilt-shift keeps its band sharp and blurs the line elsewhere.
        let tilt = draw(
            &renderer,
            &halves,
            size,
            &[step(Block::Focus, &[(0, 0.1), (1, 0.0), (2, 0.2)])],
        );
        let at = |picture: &[u8], x: u32, y: u32| picture[4 * (y * size.0 + x) as usize];
        assert_eq!(at(&tilt, 30, 16), 255, "the band stays sharp");
        assert!(at(&tilt, 30, 1) < 250, "the top is blurred across the line");
        // A spin leaves the middle alone and smears the corners.
        let spun = draw(
            &renderer,
            &halves,
            size,
            &[step(
                Block::Focus,
                &[(0, 0.1), (1, 2.0), (2, 0.1), (4, 60.0)],
            )],
        );
        assert_ne!(spun, halves);
    }

    #[test]
    fn a_spotlight_halftone_and_dust_mark_the_picture_as_named() {
        let Some(renderer) = renderer() else { return };
        let size = (16u32, 16u32);
        let white = solid(size, [255, 255, 255, 255]);
        let lit = draw(
            &renderer,
            &white,
            size,
            &[step(Block::Spotlight, &[(0, 1.0), (3, 0.1), (4, 0.05)])],
        );
        let middle = 4 * (8 * size.0 + 8) as usize;
        assert!(lit[middle] > 240, "the middle stays lit");
        assert!(lit[0] < 15, "the corner goes dark");
        // White prints no dots; black prints mostly ink.
        let printed = draw(
            &renderer,
            &white,
            size,
            &[step(Block::Halftone, &[(0, 0.25)])],
        );
        assert!(printed.chunks(4).all(|pixel| pixel[0] > 230));
        let dark = solid(size, [0, 0, 0, 255]);
        let inked = draw(
            &renderer,
            &dark,
            size,
            &[step(Block::Halftone, &[(0, 0.25)])],
        );
        let mean = inked
            .chunks(4)
            .map(|pixel| u32::from(pixel[0]))
            .sum::<u32>()
            / 256;
        assert!(mean < 128, "{mean}");
        let grey = solid(size, [128, 128, 128, 255]);
        let dusty = draw(
            &renderer,
            &grey,
            size,
            &[step(Block::Dust, &[(0, 1.0), (1, 1.0)])],
        );
        assert_ne!(dusty, grey);
    }

    #[test]
    fn prints_show_the_whole_picture_in_a_white_edged_print_over_a_pale_copy() {
        let Some(renderer) = renderer() else { return };
        let size = (64u32, 48u32);
        // Left half red, right half blue.
        let scene: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                if index % size.0 < size.0 / 2 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                }
            })
            .collect();
        // One flat print in the middle, half the frame's size.
        let one = draw(
            &renderer,
            &scene,
            size,
            &[step(
                Block::Prints,
                &[
                    (0, 1.0),
                    (1, 0.5),
                    (2, 0.0),
                    (4, 0.0),
                    (5, 0.06),
                    (6, 0.0),
                    (7, 0.5),
                ],
            )],
        );
        let at = |x: u32, y: u32| {
            let i = 4 * (y * size.0 + x) as usize;
            [one[i], one[i + 1], one[i + 2]]
        };
        // The print holds the whole picture, shrunk: red then blue.
        let red = at(24, 24);
        let blue = at(40, 24);
        assert!(red[0] > 200 && red[2] < 60, "{red:?}");
        assert!(blue[2] > 200 && blue[0] < 60, "{blue:?}");
        // Its white edge, just outside the picture.
        let edge = at(32, 10);
        assert!(edge.iter().all(|c| *c > 220), "{edge:?}");
        // Behind, the copy paled halfway to white.
        let corner = at(1, 1);
        assert!(corner[0] > 200 && corner[1] > 100, "{corner:?}");
        // Before its turn, a print is not there: only the pale copy.
        let early = draw(
            &renderer,
            &scene,
            size,
            &[step(
                Block::Prints,
                &[
                    (0, 1.0),
                    (1, 0.5),
                    (2, 0.0),
                    (4, 0.0),
                    (5, 0.06),
                    (6, 0.0),
                    (7, 0.5),
                    (8, 0.0),
                ],
            )],
        );
        let i = 4 * (24 * size.0 + 24) as usize;
        assert!(
            early[i + 1] > 100,
            "pale, not the print's red: {:?}",
            &early[i..i + 3]
        );
    }

    #[test]
    fn place_can_lay_the_picture_on_a_white_print_with_a_shadow_over_a_dim_copy() {
        let Some(renderer) = renderer() else { return };
        let size = (64u32, 48u32);
        // Left half red, right half blue.
        let scene: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|index| {
                if index % size.0 < size.0 / 2 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                }
            })
            .collect();
        let pixel = |frame: &[u8], x: u32, y: u32| {
            let i = 4 * (y * size.0 + x) as usize;
            [frame[i], frame[i + 1], frame[i + 2], frame[i + 3]]
        };
        // Half size, an edge of 0.05 of 48 at half scale: 1.2 pixels; a
        // foot of 0.2: 4.8 more below.
        let print = |back: f32, shadow: f32| {
            draw(
                &renderer,
                &scene,
                size,
                &[step(
                    Block::Place,
                    &[
                        (0, 0.5),
                        (7, 0.05),
                        (8, 0.2),
                        (9, back),
                        (10, 0.5),
                        (11, shadow),
                    ],
                )],
            )
        };
        let plain = print(0.0, 0.0);
        // The picture, shrunk to the middle: red then blue.
        let red = pixel(&plain, 24, 24);
        let blue = pixel(&plain, 40, 24);
        assert!(red[0] > 200 && red[2] < 60, "{red:?}");
        assert!(blue[2] > 200 && blue[0] < 60, "{blue:?}");
        // White paper all round it, deeper below: the picture spans rows
        // 12 to 36, the paper on to row 41.
        for (x, y) in [(32, 11), (15, 24), (48, 24), (32, 38), (32, 40)] {
            let paper = pixel(&plain, x, y);
            assert!(paper.iter().all(|c| *c > 230), "({x}, {y}) {paper:?}");
        }
        // Past the paper, clear.
        assert_eq!(pixel(&plain, 32, 44), [0; 4]);
        assert_eq!(pixel(&plain, 2, 2), [0; 4]);
        // A shadow falls below and to the right of the print, not above.
        let shaded = print(0.0, 0.1);
        let below = pixel(&shaded, 50, 43);
        assert!(below[3] > 20 && below[0] < 5, "{below:?}");
        assert!(pixel(&shaded, 10, 3)[3] < below[3]);
        // `back` fills the rest with the picture, soft and dimmed by half.
        let backed = print(0.02, 0.0);
        let corner = pixel(&backed, 1, 24);
        assert_eq!(corner[3], 255, "{corner:?}");
        assert!(
            (100..160).contains(&corner[0]) && corner[2] < 30,
            "{corner:?}"
        );
        let far = pixel(&backed, 62, 24);
        assert!((100..160).contains(&far[2]) && far[0] < 30, "{far:?}");
        // The print over it is unchanged.
        assert_eq!(pixel(&backed, 24, 24), pixel(&plain, 24, 24));
    }

    #[test]
    fn every_bundled_recipe_runs_and_moves_the_picture() {
        let Some(renderer) = renderer() else { return };
        // Large enough that a gentle blur or a fine halftone still reaches
        // a whole pixel.
        let (width, height) = (160u32, 120u32);
        let scene: Vec<u8> = (0..width * height)
            .flat_map(|index| {
                let (x, y) = (index % width, index / width);
                [
                    (x * 255 / width) as u8,
                    (y * 255 / height) as u8,
                    ((x + y) * 255 / (width + height)) as u8,
                    255,
                ]
            })
            .collect();
        for recipe in bundled() {
            let mut changed = false;
            for tenth in 0..20 {
                let steps = recipe.steps_at(Micros(tenth * 100_000));
                let got = draw(&renderer, &scene, (width, height), &steps);
                assert_eq!(got.len(), scene.len(), "{}", recipe.id);
                changed |= got != scene;
            }
            assert!(changed, "{} never changed the picture", recipe.id);
        }
    }
}
