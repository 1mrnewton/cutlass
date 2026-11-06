// A Lottie frame on the GPU: the same samples, paint and masks as
// cutlass-lottie's CPU rasterizer. Everything is read from one RGBA32F
// texture packed by `pack`; each draw is one instance whose vertex
// attribute is its record's first texel.

@group(0) @binding(0) var data: texture_2d<f32>;
@group(1) @binding(0) var child: texture_2d<f32>;
@group(1) @binding(1) var matte: texture_2d<f32>;

const DATA_WIDTH: u32 = 2048u;

fn texel(index: u32) -> vec4<f32> {
    return textureLoad(data, vec2<i32>(i32(index % DATA_WIDTH), i32(index / DATA_WIDTH)), 0);
}

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) record: u32,
};

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @location(0) record: u32) -> Varyings {
    let size = texel(0u).xy;
    let bounds = texel(record);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let pixel = mix(bounds.xy, bounds.zw, corners[vertex % 6u]);
    var out: Varyings;
    out.position = vec4<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0, 0.0, 1.0);
    out.record = record;
    return out;
}

// The share of the pixel's sixteen samples inside `count` edges from `start`.
fn coverage(start: u32, count: u32, even_odd: bool, pixel: vec2<f32>) -> f32 {
    var winding: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        winding[i] = 0;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let e = texel(start + i);
        if (max(e.y, e.w) < pixel.y || min(e.y, e.w) >= pixel.y + 1.0) {
            continue;
        }
        let dir = select(-1, 1, e.w > e.y);
        for (var row = 0u; row < 4u; row = row + 1u) {
            let ys = pixel.y + (f32(row) + 0.5) * 0.25;
            if ((e.y <= ys && ys < e.w) || (e.w <= ys && ys < e.y)) {
                let x = e.x + (ys - e.y) * (e.z - e.x) / (e.w - e.y);
                for (var col = 0u; col < 4u; col = col + 1u) {
                    let xs = pixel.x + (f32(col) + 0.5) * 0.25;
                    if (x > xs) {
                        winding[row * 4u + col] = winding[row * 4u + col] + dir;
                    }
                }
            }
        }
    }
    var inside = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let w = winding[i];
        if ((even_odd && (w % 2) != 0) || (!even_odd && w != 0)) {
            inside = inside + 1u;
        }
    }
    return f32(inside) / 16.0;
}

fn stop_color(at: u32) -> vec4<f32> {
    let a = texel(at);
    let b = texel(at + 1u);
    return vec4<f32>(a.yzw, b.x);
}

fn ramp(start: u32, count: u32, t: f32) -> vec4<f32> {
    if (count == 0u) {
        return vec4<f32>(0.0);
    }
    if (t <= texel(start).x) {
        return stop_color(start);
    }
    for (var k = 0u; k + 1u < count; k = k + 1u) {
        let a = start + k * 2u;
        let b = a + 2u;
        let a_offset = texel(a).x;
        let b_offset = texel(b).x;
        if (t <= b_offset) {
            let span = b_offset - a_offset;
            var f = 1.0;
            if (span > 0.0) {
                f = (t - a_offset) / span;
            }
            let ca = stop_color(a);
            let cb = stop_color(b);
            return ca + (cb - ca) * f;
        }
    }
    return stop_color(start + (count - 1u) * 2u);
}

// The premultiplied paint of `record` at the output point `p`.
fn shade(record: u32, kind: u32, p: vec2<f32>) -> vec4<f32> {
    let extra = texel(record + 2u);
    var color = extra;
    if (kind != 0u) {
        let row0 = texel(record + 3u);
        let row1 = texel(record + 4u);
        let ends = texel(record + 5u);
        let focal = texel(record + 6u).xy;
        let gx = row0.x * p.x + row0.y * p.y + row0.z;
        let gy = row1.x * p.x + row1.y * p.y + row1.z;
        var t = 1.0;
        if (kind == 1u) {
            let dx = ends.z - ends.x;
            let dy = ends.w - ends.y;
            let length = dx * dx + dy * dy;
            if (length > 0.0) {
                t = ((gx - ends.x) * dx + (gy - ends.y) * dy) / length;
            }
        } else {
            let r = extra.w;
            let cx = ends.x - focal.x;
            let cy = ends.y - focal.y;
            let px = gx - focal.x;
            let py = gy - focal.y;
            let qa = cx * cx + cy * cy - r * r;
            let qb = px * cx + py * cy;
            let qc = px * px + py * py;
            if (r > 1e-6 && qa < -1e-6) {
                t = (qb - sqrt(max(qb * qb - qa * qc, 0.0))) / qa;
            }
        }
        color = ramp(u32(extra.x), u32(extra.y), clamp(t, 0.0, 1.0));
        color.a = color.a * extra.z;
    }
    return vec4<f32>(color.rgb * color.a, color.a);
}

@fragment
fn fs_fill(in: Varyings) -> @location(0) vec4<f32> {
    let pixel = floor(in.position.xy);
    let head = texel(in.record + 1u);
    let cover = coverage(u32(head.x), u32(head.y), head.z > 0.5, pixel);
    if (cover <= 0.0) {
        discard;
    }
    return shade(in.record, u32(head.w), pixel + vec2<f32>(0.5, 0.5)) * cover;
}

fn combine(mode: u32, mask: f32, cover: f32) -> f32 {
    switch mode {
        case 1u: { return mask * (1.0 - cover); }
        case 2u: { return mask * cover; }
        case 3u: { return max(mask, cover); }
        case 4u: { return min(mask, cover); }
        case 5u: { return abs(mask - cover); }
        default: { return mask + cover - mask * cover; }
    }
}

fn matte_factor(mode: u32, t: vec4<f32>) -> f32 {
    let luma = 0.2125 * t.r + 0.7154 * t.g + 0.0721 * t.b;
    switch mode {
        case 1u: { return t.a; }
        case 2u: { return 1.0 - t.a; }
        case 3u: { return luma; }
        default: { return 1.0 - luma; }
    }
}

@fragment
fn fs_group(in: Varyings) -> @location(0) vec4<f32> {
    let pixel = floor(in.position.xy);
    let head = texel(in.record + 1u);
    var factor = head.w;
    let count = u32(head.y);
    if (count > 0u) {
        var value = texel(in.record + 2u).x;
        for (var k = 0u; k < count; k = k + 1u) {
            let at = u32(head.x) + k * 2u;
            let a = texel(at);
            let b = texel(at + 1u);
            var cover = coverage(u32(a.x), u32(a.y), false, pixel);
            if (b.z > 0.5) {
                let extra = coverage(u32(b.y), u32(b.z), false, pixel);
                if (b.w > 0.5) {
                    cover = cover + extra - cover * extra;
                } else {
                    cover = cover * (1.0 - extra);
                }
            }
            var shown = cover;
            if (a.w > 0.5) {
                shown = 1.0 - cover;
            }
            value = combine(u32(a.z), value, shown * b.x);
        }
        factor = factor * value;
    }
    let mode = u32(head.z);
    if (mode > 0u) {
        factor = factor * matte_factor(mode, textureLoad(matte, vec2<i32>(pixel), 0));
    }
    return textureLoad(child, vec2<i32>(pixel), 0) * factor;
}
