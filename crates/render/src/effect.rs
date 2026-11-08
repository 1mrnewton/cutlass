//! Effect passes: one shader that runs any block of a recipe.
//!
//! An effect layer applies to everything drawn below it. Each step reads
//! the picture so far and writes a new one of the same size; two textures
//! take turns. Colours stay premultiplied between steps; a step that works
//! on colour divides by alpha first and multiplies again after. Sampling is
//! linear and mirrors at the edges, so a shaken or turned frame shows a
//! reflection at its border instead of a hole. A blur runs as two steps,
//! across and then down.

use cutlass_effects::{Block, Step};

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
struct Effect {
    // The block, then for a blur 0 across and 1 down.
    code: vec2<u32>,
    size: vec2<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<uniform> fx: Effect;

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

fn blur(uv: vec2<f32>) -> vec4<f32> {
    let short = min(fx.size.x, fx.size.y);
    let reach = fx.a.x * short;
    if (reach < 0.5) {
        return fetch(uv);
    }
    var direction = vec2<f32>(1.0 / fx.size.x, 0.0);
    if (fx.code.y == 1u) {
        direction = vec2<f32>(0.0, 1.0 / fx.size.y);
    }
    // A tap a pixel or closer apart up to 32 pixels each side, so a wide
    // blur has no gaps; past that the 65 taps spread out.
    let half = min(i32(ceil(reach)), 32);
    let stride = direction * reach / f32(half);
    var sum = vec4<f32>(0.0);
    for (var k = -half; k <= half; k = k + 1) {
        sum = sum + fetch(uv + stride * f32(k));
    }
    return sum / f32(2 * half + 1);
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

fn glow(uv: vec2<f32>) -> vec4<f32> {
    let base = fetch(uv);
    let reach = fx.a.y * min(fx.size.x, fx.size.y) / fx.size;
    var light = vec3<f32>(0.0);
    // A disc of taps laid out on the golden angle, so none line up.
    for (var k = 0; k < 48; k = k + 1) {
        let r = sqrt((f32(k) + 0.5) / 48.0);
        let turn = f32(k) * 2.3999632;
        let c = fetch(uv + vec2<f32>(cos(turn), sin(turn)) * r * reach);
        light = light + max(straight(c) - vec3<f32>(fx.a.z), vec3<f32>(0.0)) * c.a;
    }
    let rgb = base.rgb + light / 48.0 * fx.a.x;
    return vec4<f32>(min(rgb, vec3<f32>(base.a)), base.a);
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
    if (kind == 1u) {
        let p = from_point(uv, vec2<f32>(0.0));
        let corner = length(fx.size / min(fx.size.x, fx.size.y)) * 0.5;
        run = length(p) / corner;
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
    let uv = in.position.xy / fx.size;
    switch (fx.code.x) {
        case 0u: { return transform(uv); }
        case 1u: { return rgb_split(uv); }
        case 2u: { return slices(uv); }
        case 3u: { return blur(uv); }
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
        case 17u: { return glow(uv); }
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
        default: { return fetch(uv); }
    }
}
";

/// Bytes in one step's uniform block.
const PARAMS: u64 = 64;

pub(crate) struct Effects {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl Effects {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render effect"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render effect"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render effect"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render effect"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: CANVAS_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass-render effect"),
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
        }
    }

    /// Run `steps` over `first`, using `second` in between. Returns the
    /// texture that holds the result.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run<'v>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        first: &'v wgpu::TextureView,
        second: &'v wgpu::TextureView,
        size: (u32, u32),
        steps: &[Step],
    ) -> &'v wgpu::TextureView {
        let (mut from, mut to) = (first, second);
        for step in steps {
            let directions: &[u32] = if step.block == Block::Blur {
                &[0, 1]
            } else {
                &[0]
            };
            for direction in directions {
                self.pass(device, queue, encoder, from, to, size, step, *direction);
                std::mem::swap(&mut from, &mut to);
            }
        }
        from
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        from: &wgpu::TextureView,
        to: &wgpu::TextureView,
        size: (u32, u32),
        step: &Step,
        direction: u32,
    ) {
        let mut bytes = Vec::with_capacity(PARAMS as usize);
        bytes.extend_from_slice(&step.block.code().to_le_bytes());
        bytes.extend_from_slice(&direction.to_le_bytes());
        for value in [size.0 as f32, size.1 as f32] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in step.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render effect params"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render effect"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(from),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
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
        pass.set_pipeline(&self.pipeline);
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
        Step { block, values }
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
        // A reach of 40 pixels: ten times the 4 pixel short side.
        let blur = step(Block::Blur, &[(0, 10.0)]);
        let got = draw(&renderer, &scene, (width, height), &[blur]);
        let row: Vec<u8> = got[..(width * 4) as usize]
            .chunks(4)
            .map(|pixel| pixel[0])
            .collect();
        // Every column the reach covers gets some of the line, the same
        // amount either side, and the rest stays black.
        for offset in 1..=38 {
            let (left, right) = (row[64 - offset], row[64 + offset]);
            assert!(
                left > 0 && right > 0,
                "a gap {offset} from the line: {row:?}"
            );
            assert!(left.abs_diff(right) <= 2, "lopsided at {offset}: {row:?}");
        }
        assert_eq!(row[64 - 44], 0, "{row:?}");
        assert_eq!(row[64 + 44], 0, "{row:?}");
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
