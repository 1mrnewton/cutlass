// mask.wgsl — shared mask alpha helpers for rgba_fx / yuv_fx pipelines.
//
// Evaluated in quad-local pixels (origin at center, +y down). `half` is the
// layer's half-extents (`placement.size * 0.5`). Geometry (center / size /
// rotation / roundness) is applied in that same space before the SDF.

const BASE_AA: f32 = 1.0;
const PI: f32 = 3.14159265358979;

fn dot2(v: vec2<f32>) -> f32 {
    return dot(v, v);
}

fn coverage(d: f32, aa: f32) -> f32 {
    return clamp(0.5 - d / (2.0 * aa), 0.0, 1.0);
}

fn sd_round_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2(r, r);
    return length(max(q, vec2(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn sd_ellipse(p: vec2<f32>, r: vec2<f32>) -> f32 {
    let rr = max(r, vec2(1e-3, 1e-3));
    let k1 = length(p / rr);
    if k1 < 1e-6 {
        return -min(rr.x, rr.y);
    }
    let k2 = length(p / (rr * rr));
    return k1 * (k1 - 1.0) / k2;
}

fn star_vertex(k: u32, n: f32, inner: f32, hx: f32, hy: f32) -> vec2<f32> {
    let theta = -PI / 2.0 + (PI / n) * f32(k);
    var r = 1.0;
    if k % 2u == 1u {
        r = inner;
    }
    return vec2(cos(theta) * r * hx, sin(theta) * r * hy);
}

fn sd_star(p: vec2<f32>, half: vec2<f32>, points_f: f32, inner_f: f32) -> f32 {
    let n = clamp(points_f, 3.0, 20.0);
    let inner = clamp(inner_f, 0.05, 1.0);
    let hx = max(half.x, 0.5);
    let hy = max(half.y, 0.5);
    let count = 2u * u32(n);

    var d = dot2(p - star_vertex(0u, n, inner, hx, hy));
    var sgn = 1.0;
    var vj = star_vertex(count - 1u, n, inner, hx, hy);
    for (var i = 0u; i < count; i = i + 1u) {
        let vi = star_vertex(i, n, inner, hx, hy);
        let e = vj - vi;
        let w = p - vi;
        let t = clamp(dot(w, e) / max(dot2(e), 1e-12), 0.0, 1.0);
        let b = w - e * t;
        d = min(d, dot2(b));
        let c0 = p.y >= vi.y;
        let c1 = p.y < vj.y;
        let c2 = e.x * w.y > e.y * w.x;
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            sgn = -sgn;
        }
        vj = vi;
    }
    return sgn * sqrt(d);
}

fn sd_heart(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let lobe = 0.35355338;
    let unit_hw = 0.25 + lobe;
    let unit_h = 0.75 + lobe;
    let sx = half.x / unit_hw;
    let sy = half.y / (unit_h * 0.5);
    let hx = abs(p.x / sx);
    let hy = (half.y - p.y) / sy;

    var d_unit = 0.0;
    if hy + hx > 1.0 {
        d_unit = sqrt(dot2(vec2(hx - 0.25, hy - 0.75))) - lobe;
    } else {
        let a = dot2(vec2(hx, hy - 1.0));
        let m = max(0.5 * (hx + hy), 0.0);
        let b = dot2(vec2(hx - m, hy - m));
        var sgn = 1.0;
        if hx < hy {
            sgn = -1.0;
        }
        d_unit = sqrt(min(a, b)) * sgn;
    }
    return d_unit * min(sx, sy);
}

// Mask kind ids (must match `mask_kind` in layer.rs).
const MASK_LINEAR: u32 = 0u;
const MASK_MIRROR: u32 = 1u;
const MASK_CIRCLE: u32 = 2u;
const MASK_RECTANGLE: u32 = 3u;
const MASK_HEART: u32 = 4u;
const MASK_STAR: u32 = 5u;

// Map layer-local px → mask space. Identity geometry (center 0, size 1, rot 0)
// is an exact no-op: subtract 0, rotate by 0, divide by 1.
fn mask_space(
    local: vec2<f32>,
    half: vec2<f32>,
    center: vec2<f32>,
    size: vec2<f32>,
    rotation: f32,
) -> vec2<f32> {
    var p = local - center * (half * 2.0);
    let c = cos(-rotation);
    let s = sin(-rotation);
    p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
    return p / size;
}

fn mask_alpha(
    local: vec2<f32>,
    half: vec2<f32>,
    kind: u32,
    feather: f32,
    invert: f32,
    center: vec2<f32>,
    size: vec2<f32>,
    rotation: f32,
    roundness: f32,
) -> f32 {
    let p = mask_space(local, half, center, size, rotation);
    let aa = BASE_AA + feather * min(half.x, half.y);
    var alpha = 1.0;

    switch kind {
        case MASK_LINEAR: {
            // Soft ramp across the vertical center line (x = 0): right half visible.
            alpha = coverage(-p.x, aa);
        }
        case MASK_MIRROR: {
            // Keep the left half visible (x <= 0).
            alpha = coverage(p.x, aa);
        }
        case MASK_CIRCLE: {
            alpha = coverage(sd_ellipse(p, half), aa);
        }
        case MASK_RECTANGLE: {
            let r = roundness * 0.5 * min(half.x, half.y);
            alpha = coverage(sd_round_box(p, half, r), aa);
        }
        case MASK_HEART: {
            alpha = coverage(sd_heart(p, half), aa);
        }
        default: {
            // Star (5 points, inner 0.5).
            alpha = coverage(sd_star(p, half, 5.0, 0.5), aa);
        }
    }

    if invert > 0.5 {
        alpha = 1.0 - alpha;
    }
    return alpha;
}

// Returns an alpha multiplier: 0 keyed out, 1 kept. Chroma runs before mask.
fn chroma_alpha(rgb: vec3<f32>, key: vec3<f32>, strength: f32, shadow: f32) -> f32 {
    let dist = length(rgb - key);
    let tol = max(strength * 0.35, 1e-4);
    let edge = 0.08;
    var keyed = 1.0 - smoothstep(tol, tol + edge, dist);
    let luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    keyed = keyed * (1.0 - shadow * (1.0 - luma));
    return 1.0 - keyed;
}
