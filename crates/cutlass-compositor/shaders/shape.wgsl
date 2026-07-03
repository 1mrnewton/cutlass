// shape.wgsl — parametric vector shapes evaluated as signed-distance fields
// on a placed quad (rects, ellipses, polygons/stars, lines, arrows, hearts).
//
// CONTRACT: this file implements exactly the math in
// `crates/cutlass-shapes/src/sdf.rs` — same shape formulas, same fixed 1px
// linear anti-alias ramp, same stroke-over-fill shading. A golden test in
// this crate renders both and asserts per-pixel agreement; change the two
// files only in lockstep.
//
// Coordinates are quad-local canvas pixels (origin at the quad center, +y
// down). The quad composites 1:1 with canvas pixels (scale is folded into
// the pixel extents by the resolver; rotation is rigid), so a fixed 1px ramp
// anti-aliases correctly without derivative tricks.

const AA: f32 = 1.0;
const PI: f32 = 3.14159265358979;

struct Sdf {
    // Straight-alpha fill color; a == 0 means no fill (stroke-only).
    fill: vec4<f32>,
    // Straight-alpha stroke color.
    stroke_color: vec4<f32>,
    // Columns of the 2x2 unit-quad → clip-space linear part.
    linear: vec4<f32>,
    // Clip-space translation (x, y), layer opacity (z), stroke width px (w).
    trans_opacity: vec4<f32>,
    // Shape half-extents px (x, y), shape kind (z), corner radius px (w).
    // Kinds: 0 rounded rect, 1 ellipse, 2 star/polygon, 3 line, 4 arrow,
    // 5 heart.
    geo: vec4<f32>,
    // Star spike count (x), inner-radius fraction (y); quad half-extents px
    // (z, w) — the vertex shader's local-coordinate scale.
    star: vec4<f32>,
}

@group(0) @binding(0) var<uniform> s: Sdf;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Quad-local canvas pixels, origin at the quad center, +y down.
    @location(0) local: vec2<f32>,
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
    let m = s.linear;
    let t = s.trans_opacity;
    var out: VertexOutput;
    out.position = vec4(
        m.x * c.x + m.z * c.y + t.x,
        m.y * c.x + m.w * c.y + t.y,
        0.0,
        1.0,
    );
    out.local = c * 2.0 * s.star.zw;
    return out;
}

fn dot2(v: vec2<f32>) -> f32 {
    return dot(v, v);
}

// Fill coverage at signed distance d: a linear ramp across ±AA (see
// `coverage` in sdf.rs).
fn coverage(d: f32) -> f32 {
    return clamp(0.5 - d / (2.0 * AA), 0.0, 1.0);
}

// --- shape formulas (each has a Rust twin in sdf.rs) -------------------------

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

fn sd_capsule(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let r = min(half.y, half.x);
    let straight = max(half.x - r, 0.0);
    let qx = max(abs(p.x) - straight, 0.0);
    return sqrt(qx * qx + p.y * p.y) - r;
}

// Star vertex k of 2n, spike up: outer vertices on the (aspect-scaled) unit
// circle, odd (inner) vertices at `inner` of it.
fn star_vertex(k: u32, n: f32, inner: f32, hx: f32, hy: f32) -> vec2<f32> {
    let theta = -PI / 2.0 + (PI / n) * f32(k);
    var r = 1.0;
    if k % 2u == 1u {
        r = inner;
    }
    return vec2(cos(theta) * r * hx, sin(theta) * r * hy);
}

// Exact polygon SDF over the star's vertex loop (parity sign), minus the
// corner rounding. Polygons are stars whose inner vertices sit on the edge
// midpoints (see `SdfParams::polygon`).
fn sd_star(p: vec2<f32>, half: vec2<f32>, points_f: f32, inner_f: f32, round: f32) -> f32 {
    let n = clamp(points_f, 3.0, 20.0);
    let inner = clamp(inner_f, 0.05, 1.0);
    let hx = max(half.x - round, 0.5);
    let hy = max(half.y - round, 0.5);
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
    return sgn * sqrt(d) - round;
}

// Right-pointing arrow as an explicit 7-gon (same proportions as
// `arrow_vertices` in sdf.rs).
fn sd_arrow(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let hx = half.x;
    let hy = half.y;
    let head = min(hy, hx);
    let shaft = 0.4 * hy;
    var v = array<vec2<f32>, 7>(
        vec2(hx, 0.0),
        vec2(hx - head, -hy),
        vec2(hx - head, -shaft),
        vec2(-hx, -shaft),
        vec2(-hx, shaft),
        vec2(hx - head, shaft),
        vec2(hx - head, hy),
    );
    var d = dot2(p - v[0]);
    var sgn = 1.0;
    var j = 6u;
    for (var i = 0u; i < 7u; i = i + 1u) {
        let e = v[j] - v[i];
        let w = p - v[i];
        let t = clamp(dot(w, e) / max(dot2(e), 1e-12), 0.0, 1.0);
        let b = w - e * t;
        d = min(d, dot2(b));
        let c0 = p.y >= v[i].y;
        let c1 = p.y < v[j].y;
        let c2 = e.x * w.y > e.y * w.x;
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            sgn = -sgn;
        }
        j = i;
    }
    return sgn * sqrt(d);
}

// Heart (unit heart mapped to the box, upright); see `sd_heart` in sdf.rs.
fn sd_heart(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let lobe = 0.35355338; // sqrt(2) / 4
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

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let half = s.geo.xy;
    let kind = u32(s.geo.z + 0.5);
    let radius = clamp(s.geo.w, 0.0, min(half.x, half.y));

    var d = 0.0;
    switch kind {
        case 0u: {
            d = sd_round_box(in.local, half, radius);
        }
        case 1u: {
            d = sd_ellipse(in.local, half);
        }
        case 2u: {
            d = sd_star(in.local, half, s.star.x, s.star.y, radius);
        }
        case 3u: {
            d = sd_capsule(in.local, half);
        }
        case 4u: {
            d = sd_arrow(in.local, half);
        }
        default: {
            d = sd_heart(in.local, half);
        }
    }

    // Stroke ring over fill, straight alpha — mirrors `shade` in sdf.rs.
    let sw = s.trans_opacity.w;
    let fill_cov = coverage(d) * s.fill.a;
    var stroke_cov = 0.0;
    if sw > 0.0 {
        stroke_cov = coverage(abs(d) - sw * 0.5) * s.stroke_color.a;
    }
    let a = stroke_cov + fill_cov * (1.0 - stroke_cov);
    if a <= 0.0 {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    let rgb = (s.stroke_color.rgb * stroke_cov + s.fill.rgb * fill_cov * (1.0 - stroke_cov)) / a;
    return vec4(rgb, a * s.trans_opacity.z);
}
