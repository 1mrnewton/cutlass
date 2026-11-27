//! Blocks that draw: light, weather, film dirt and paper made from
//! arithmetic on a pixel's place, a time and a seed, with no picture
//! file behind them.
//!
//! Each reads the picture so far and lays what it draws over it: light
//! is screened on, so black changes nothing; dirt and paper are
//! multiplied on, so white changes nothing; grain is overlaid, so mid
//! grey changes nothing. Places and sizes are in shorter sides of the
//! frame, so a block looks the same on a wide frame, a tall one and a
//! square one, and the same at any size in pixels.
//!
//! Nothing here keeps anything between frames. A speck's place comes from
//! the time and the seed, so any frame can be drawn alone and the same
//! draft and time always give the same picture.

/// Random numbers and cloud patterns, shared by the effect shader and the
/// cross shader. Floats in, floats out: the same point always gives the
/// same number.
pub(crate) const NOISE: &str = r"
const TAU: f32 = 6.2831853;
const PI: f32 = 3.1415927;

// A random number 0..1 for a point.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// A random number 0..1 for a point in space, or a point and a time.
fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

// Two random numbers 0..1 for a point.
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Smooth random hills 0..1, one hill a unit.
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// The same hills in three dimensions: with time as the third, they boil.
fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = hash13(i);
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z,
    );
}

// Clouds 0..1: hills on hills, each layer half the size and strength.
fn fbm(at: vec2<f32>, layers: i32) -> f32 {
    var p = at;
    var sum = 0.0;
    var amp = 0.5;
    var total = 0.0;
    for (var i = 0; i < layers; i = i + 1) {
        sum = sum + amp * vnoise(p);
        total = total + amp;
        p = p * 2.03 + vec2<f32>(17.1, 9.7);
        amp = amp * 0.5;
    }
    return sum / total;
}

// Clouds that change with time, the third number.
fn fbm3(at: vec3<f32>, layers: i32) -> f32 {
    var p = at;
    var sum = 0.0;
    var amp = 0.5;
    var total = 0.0;
    for (var i = 0; i < layers; i = i + 1) {
        sum = sum + amp * vnoise3(p);
        total = total + amp;
        p = p * 2.03 + vec3<f32>(17.1, 9.7, 3.3);
        amp = amp * 0.5;
    }
    return sum / total;
}

// A point turned by an angle.
fn turn(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}
";

/// Mattes that grow with a `progress` of 0 to 1: 0 where nothing has been
/// reached yet, 1 where it has. A transition swaps its two pictures by
/// one, and the reveal block shows its picture through one. They need
/// [`NOISE`] and nothing else, so each shader passes in the frame's shape.
pub(crate) const MATTES: &str = r"
// Ink: a blot lands with a splash of spikes and flung drops, then creeps
// out with a wandering edge until it covers the frame.
//
// q: the pixel from the blot's middle, in shorter sides. far: how far the
// frame's farthest corner is from there. one: a pixel, in shorter sides.
fn ink_matte(q: vec2<f32>, progress: f32, seed: f32, rough: f32, far: f32, one: f32) -> f32 {
    if (progress <= 0.0) {
        return 0.0;
    }
    let since = progress * 3.0;
    let sd = seed * 3.1;
    let splash = 0.2;
    let burst = 1.0 - exp(-since * 11.0);
    let creep = pow(clamp(progress, 0.0, 1.0), 1.7);
    let radius = splash * burst + max(far * 1.5 - splash, 0.0) * creep;
    // The edge wanders: the whole shape is pushed about by slow clouds.
    var w = q + (vec2<f32>(
        fbm3(vec3<f32>(q * 2.1 + sd, since * 0.12), 4),
        fbm3(vec3<f32>(q * 2.1 + sd + 40.0, since * 0.12), 4),
    ) - 0.5) * 0.42 * rough * (0.35 + radius);
    w = w + (vec2<f32>(fbm(q * 9.0 + sd, 3), fbm(q * 9.0 + sd + 80.0, 3)) - 0.5) * 0.05 * rough;
    let d = length(w);
    let ray = normalize(q + vec2<f32>(0.00001));
    // Spikes: thin tongues, long and short, where the splash threw them.
    let tongue = pow(vnoise(ray * 8.0 + sd), 5.0) * 1.1 + pow(vnoise(ray * 21.0 + sd + 9.0), 7.0) * 0.9;
    let lobes = (vnoise(ray * 2.5 + sd + 30.0) - 0.5) * 0.35 * rough;
    let edge_r = radius * (1.0 + lobes) + splash * burst * tongue;
    let soft = max(0.0025, one);
    var ink = smoothstep(-soft, soft, edge_r - d);
    // Droplets: small drops flung past the edge, a touch longer along
    // the line of flight.
    for (var j = 0; j < 26; j = j + 1) {
        let fj = f32(j);
        let h = hash22(vec2<f32>(fj * 3.7 + 1.0, sd + 5.0));
        let h2 = hash22(vec2<f32>(fj * 9.1 + 4.0, sd + 11.0));
        if (h2.y > 0.8) {
            continue;
        }
        let a = h.x * TAU;
        let fly = vec2<f32>(cos(a), sin(a));
        let out = splash * (1.2 + 2.3 * h.y * h.y);
        let fat = splash * (0.02 + 0.1 * h2.x * h2.x) * smoothstep(0.02 + 0.1 * h.y, 0.1 + 0.16 * h.y, since);
        let o = q - fly * out;
        let e = vec2<f32>(dot(o, fly) / 1.5, dot(o, vec2<f32>(-fly.y, fly.x)));
        let wob = (vnoise(o * 90.0 + fj) - 0.5) * fat * 0.5;
        ink = max(ink, smoothstep(-soft, soft, fat - length(e) + wob));
    }
    // Whatever the edge left dry fills in at the very end.
    return max(ink, smoothstep(0.9, 1.0, progress));
}

// A brush: strokes laid one under the other, each the opposite way to
// the last, streaky at the leading end where the brush runs dry and
// solid behind it.
//
// p: the pixel from the frame's middle, in shorter sides. shape: the
// frame in shorter sides. angle: the strokes' tilt in radians. dry: how
// streaky they are.
fn brush_matte(p0: vec2<f32>, shape: vec2<f32>, progress: f32, seed: f32, strokes: f32, angle: f32, dry: f32) -> f32 {
    let p = turn(p0, -angle);
    let count = i32(clamp(strokes, 1.0, 8.0));
    let n = f32(count);
    // More than the frame's height, so the tilt leaves no corner.
    let cover = 1.25;
    let inset = -0.25;
    // The share of the whole run one stroke takes.
    let span = min(1.0, 1.45 / n);
    var m = 0.0;
    for (var i = 0; i < count; i = i + 1) {
        let fi = f32(i);
        let r = hash22(vec2<f32>(fi * 7.3 + 2.0, seed * 4.1 + 1.0));
        var begin = 0.0;
        if (n > 1.0) {
            begin = fi * (1.0 - span) / (n - 1.0);
        }
        let local = clamp((progress - begin) / span, 0.0, 1.0);
        // A stroke starts fast and slows as the brush lifts.
        let go = 1.0 - pow(1.0 - local, 2.2);
        var way = 1.0;
        if (i % 2 == 1) {
            way = -1.0;
        }
        // 0 at the side the stroke starts from, 1 at the far side.
        let u = p.x * way / shape.x + 0.5;
        let yc = (-cover * 0.5 + (fi + 0.5) * cover / n + (r.x - 0.5) * 0.03
            + 0.018 * sin(u * 3.0 + r.y * TAU)) * shape.y;
        let reach = cover / n * 0.5 * 1.5 * shape.y;
        // The stroke swells and thins along its length.
        let fat = 0.85 + 0.3 * vnoise(vec2<f32>(u * 3.0 + fi * 10.0, seed));
        let v = (p.y - yc) / (reach * fat);
        // Bristles: streaks down the stroke, in clumps and single hairs.
        let across = (p.y - yc) * 1080.0;
        let clumps = fbm(vec2<f32>(across * 0.085 + fi * 31.0 + seed * 3.0, u * 1.6), 3);
        let hairs = vnoise(vec2<f32>(across * 0.42 + fi * 17.0, u * 5.0 + seed));
        let bb = mix(clumps, hairs, 0.4);
        // The long edges: ragged, following the bristles.
        let wave = (fbm(vec2<f32>(u * 5.0 + fi * 13.0, seed + sign(v) * 9.0), 3) - 0.5) * 0.3;
        let edge = 1.0 - smoothstep(0.7, 1.0, abs(v) + wave + (bb - 0.5) * 0.75);
        // The brush head: some bristles lead, some lag.
        let head = mix(inset - 0.3, 1.0 - inset + 0.3, go);
        let behind = head - u - (bb - 0.5) * 0.24;
        var wet = smoothstep(0.0, 0.012, behind) * smoothstep(0.0, 0.012, u - inset - (hairs - 0.5) * 0.06);
        wet = wet * (1.0 - smoothstep(0.0, 0.012, u - (1.0 - inset) - (clumps - 0.5) * 0.08));
        // Dry brush: gaps between bristles just behind the head, out at
        // the edges, and a little all along where the paint is thin.
        let thin = vnoise(vec2<f32>(u * 2.2 + fi * 4.0, across * 0.01 + seed));
        let thirst = dry * (1.0 - smoothstep(0.0, 0.32, behind)) + dry * 0.75 * smoothstep(0.45, 1.0, abs(v))
            + dry * 0.55 * smoothstep(0.45, 0.8, thin);
        let paint = smoothstep(thirst * 0.85 - 0.08, thirst * 0.85 + 0.04, bb + 0.12);
        m = max(m, edge * wet * paint);
    }
    // The last streaks fill in at the very end.
    return max(m, smoothstep(0.86, 1.0, progress));
}

// A paper tear travelling across the frame along `dir`. Returns how much
// of the top sheet is gone, and how much of the paper's white core shows
// along the tear.
//
// p: the pixel from the frame's middle, in shorter sides. shape: the
// frame in shorter sides. one: a pixel, in shorter sides.
fn tear_matte(p: vec2<f32>, shape: vec2<f32>, progress: f32, seed: f32, rough: f32, dir: vec2<f32>, one: f32) -> vec2<f32> {
    let across = dot(p, dir);
    let along = dot(p, vec2<f32>(-dir.y, dir.x));
    let half_way = 0.5 * (abs(dir.x) * shape.x + abs(dir.y) * shape.y) + 0.2;
    // It starts gently, is still moving fast when it leaves, and is all
    // the way across a little before the end.
    let k = pow(clamp(progress / 0.9, 0.0, 1.0), 1.6);
    let pos = mix(-half_way, half_way, k);
    // The tear line: big wanders, small jags and a fine fray.
    let n = (fbm(vec2<f32>(along * 2.2, seed * 3.7), 4) - 0.5) * 0.26 * rough
        + (fbm(vec2<f32>(along * 14.0, seed * 9.1), 3) - 0.5) * 0.035 * rough
        + (fbm(vec2<f32>(along * 70.0, seed), 2) - 0.5) * 0.007 * rough;
    // Below 0 the top sheet is gone here.
    let d = across - pos - n;
    let gone = 1.0 - smoothstep(-one, one, d);
    // The core is wider in some places than others, its far edge ragged
    // too, and it is not flat white: paper has fibres.
    let wide = 16.0 / 1080.0 * (0.35 + 1.3 * fbm(vec2<f32>(along * 9.0, seed * 5.0 + 3.0), 3))
        + (fbm(vec2<f32>(along * 90.0, seed * 2.0 + 7.0), 2) - 0.5) * 0.006;
    var rim = smoothstep(-one, one, d) * (1.0 - smoothstep(wide - one, wide + one, d));
    rim = rim * (0.82 + 0.18 * fbm(p * 260.0 + seed, 2));
    return vec2<f32>(gone, rim);
}
";

/// The kernels. They read `fx`, `fetch`, `straight` and `premultiplied`
/// from the effect shader, and [`NOISE`].
pub(crate) const DRAWN: &str = r"
// The frame's sides in shorter sides: 1 for the shorter, more for the
// longer.
fn frame_shape() -> vec2<f32> {
    return fx.size / min(fx.size.x, fx.size.y);
}

// `uv` from the middle of the frame, in shorter sides.
fn from_middle(uv: vec2<f32>) -> vec2<f32> {
    return (uv - vec2<f32>(0.5)) * frame_shape();
}

// How far the frame's edge is from the middle, going along `dir`.
fn edge_reach(dir: vec2<f32>) -> f32 {
    let shape = frame_shape();
    return min(0.5 * shape.x / max(abs(dir.x), 0.001), 0.5 * shape.y / max(abs(dir.y), 0.001));
}

// The film frame a time falls in: marks that are new on every frame
// change this often.
fn film_frame(time: f32) -> f32 {
    return floor(time * 24.0 + 0.5);
}

// A pinch of noise, so soft dark light does not turn into steps.
fn dither(uv: vec2<f32>, time: f32) -> f32 {
    return (hash13(vec3<f32>(uv * fx.size, film_frame(time))) - 0.5) / 200.0;
}

// Light laid over the picture: it only ever brightens.
fn screened(c: vec4<f32>, light: vec3<f32>) -> vec4<f32> {
    let lit = clamp(light, vec3<f32>(0.0), vec3<f32>(1.0));
    return premultiplied(vec3<f32>(1.0) - (vec3<f32>(1.0) - straight(c)) * (vec3<f32>(1.0) - lit), c.a);
}

// Shade laid over the picture: it only ever darkens.
fn multiplied(c: vec4<f32>, shade: vec3<f32>) -> vec4<f32> {
    return premultiplied(straight(c) * clamp(shade, vec3<f32>(0.0), vec3<f32>(1.0)), c.a);
}

// Grey laid over the picture: lighter than mid grey lifts it, darker
// pulls it down, and pure black and white are left alone.
fn overlaid(c: vec4<f32>, grey: vec3<f32>) -> vec4<f32> {
    let base = straight(c);
    let low = 2.0 * base * grey;
    let high = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - base) * (vec3<f32>(1.0) - grey);
    return premultiplied(select(high, low, base < vec3<f32>(0.5)), c.a);
}

// The way light comes in from: `side` in turns (0 right, 0.25 bottom,
// 0.5 left, 0.75 top), or picked by the seed when it is under 0.
fn side_way(side: f32, seed: f32, salt: f32) -> vec2<f32> {
    var part = side;
    if (side < 0.0) {
        part = hash12(vec2<f32>(seed * 3.7, salt));
    }
    return vec2<f32>(cos(part * TAU), sin(part * TAU));
}

// amount, time, seed, side | second, red, green, blue | red2, green2, blue2
fn leak(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let amount = fx.a.x;
    if (amount <= 0.0) {
        return c;
    }
    let time = fx.a.y;
    let seed = fx.a.z;
    let tint = fx.b.yzw;
    let second_tint = fx.c.xyz;
    let p = from_middle(uv);
    let dir = side_way(fx.a.w, seed, 3.1);
    let along = vec2<f32>(-dir.y, dir.x);
    let reach = edge_reach(dir);
    var warm = 0.0;
    var pink = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let fi = f32(i);
        let r = hash22(vec2<f32>(fi * 13.7 + 1.0, seed * 5.3 + 1.0));
        let r2 = hash22(vec2<f32>(fi * 3.1 + 40.0, seed * 9.1 + 2.0));
        let tt = time * (0.22 + 0.3 * r2.x) + r.x * 10.0;
        // Each light sits near the edge, slides along it and leans in
        // and out of the frame.
        let lean = 0.95 + 0.5 * (vnoise(vec2<f32>(tt, fi * 7.0 + seed)) - 0.5);
        let slide = (r.y - 0.5) * 1.3 + (vnoise(vec2<f32>(tt * 0.8 + 50.0, fi + seed)) - 0.5) * 0.9;
        let q = p - (dir * reach * lean + along * slide);
        // Long along the edge, shorter into the picture.
        let da = dot(q, along) / (0.45 + 0.5 * r2.y);
        let dd = dot(q, dir) / (0.26 + 0.34 * r.x);
        let blob = exp(-(da * da + dd * dd));
        let breath = 0.5 + 0.5 * vnoise(vec2<f32>(tt * 1.9 + 9.0, fi * 3.0));
        if (i < 3) {
            warm = warm + blob * breath;
        } else {
            pink = pink + blob * breath;
        }
    }
    // Soft streaks along the edge, slowly changing.
    let cloud = fbm3(vec3<f32>(dot(p, along) * 0.9, dot(p, dir) * 2.4, time * 0.25) + vec3<f32>(seed), 3);
    let k = amount * (0.4 + 1.2 * cloud);
    // Faint light is the tint; stronger light lets the other colours
    // through too, to yellow and then white, as over-exposed film does.
    let a = vec3<f32>(1.0) - exp(-warm * k * 2.4 * tint);
    let b = vec3<f32>(1.0) - exp(-pink * k * 1.7 * fx.b.x * second_tint);
    let light = vec3<f32>(1.0) - (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b);
    return screened(c, light + vec3<f32>(dither(uv, time)) * min(amount, 1.0));
}

// amount, progress, time, seed | side, rough, flicker, red | green, blue
fn burn(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let amount = fx.a.x;
    if (amount <= 0.0) {
        return c;
    }
    let progress = fx.a.y;
    let time = fx.a.z;
    let seed = fx.a.w;
    let rough = fx.b.y;
    let flicker = fx.b.z;
    let tint = vec3<f32>(fx.b.w, fx.c.x, fx.c.y);
    let frame = film_frame(time);
    // The film slips in the gate a little on every frame.
    let jump = (hash22(vec2<f32>(frame * 1.7, seed * 3.3)) - 0.5) * 0.05 * flicker;
    let p = from_middle(uv) + jump;
    let dir = side_way(fx.b.x, seed, 8.1);
    // +1 at the edge the light comes in by, -1 at the far edge.
    var s = dot(p, dir) / edge_reach(dir);
    s = s + (fbm3(vec3<f32>(p * 1.5 + seed * 3.0, time * 0.9), 4) - 0.5) * 1.7 * rough;
    s = s + (fbm3(vec3<f32>(p * 5.5 + seed, time * 1.6), 3) - 0.5) * 0.4 * rough;
    // The front of the light crosses in the first half, its back in the
    // second: in the middle the whole frame is lit.
    let front = mix(2.1, -2.1, smoothstep(0.0, 0.5, progress));
    let back = mix(2.1, -2.1, smoothstep(0.5, 1.0, progress));
    let heat = min(s - front, back - s);
    var e = pow(smoothstep(-1.15, 0.8, heat), 3.4) * 40.0;
    // Hot spots that boil.
    e = e * (0.55 + 0.95 * fbm3(vec3<f32>(p * 2.6 - seed, time * 2.2), 3));
    e = e * amount * (1.0 + (hash12(vec2<f32>(frame, seed * 5.1)) - 0.5) * 2.0 * flicker);
    let light = vec3<f32>(1.0) - exp(-e * tint);
    // Nothing is drawn before the light arrives or after it has left.
    let lit = step(0.0001, progress) * step(progress, 0.9999);
    return screened(c, (light + vec3<f32>(dither(uv, time))) * lit);
}

fn rainbow(h: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(TAU * (vec3<f32>(h) + vec3<f32>(0.0, 0.33, 0.67)));
}

// One depth of out-of-focus discs: a grid of cells, a disc in some.
fn bokeh_discs(p: vec2<f32>, cells: f32, sd: f32) -> vec3<f32> {
    let time = fx.a.y;
    let fill = fx.a.w;
    let size = fx.b.x;
    let tint = fx.b.zwx;
    let g = p * cells;
    let id = floor(g);
    var sum = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cell = id + vec2<f32>(f32(dx), f32(dy));
            let h = hash22(cell + sd);
            let h2 = hash22(cell * 1.9 + sd + 7.0);
            if (h2.x > 0.5 * fill) {
                continue;
            }
            let wander = vec2<f32>(
                sin(time * (0.2 + 0.3 * h.x) + h.y * TAU),
                cos(time * (0.17 + 0.25 * h.y) + h.x * TAU),
            );
            let at = cell + 0.5 + (h - 0.5) * 0.6 + 0.14 * wander;
            let r = (0.16 + 0.3 * h2.y) * size;
            let d = length(g - at);
            let disc = 1.0 - smoothstep(r - 0.035, r, d);
            // A real out-of-focus disc is a little brighter at its rim.
            let rim = 0.55 + 0.45 * smoothstep(r * 0.55, r, d);
            let breathe = mix(0.35, 1.0, 0.5 + 0.5 * sin(time * (0.5 + h.x) + h2.y * TAU));
            let colour = mix(vec3<f32>(tint.x, tint.y, fx.c.x), rainbow(h.y), fx.b.y);
            sum = sum + colour * disc * rim * breathe;
        }
    }
    return sum;
}

// amount, time, seed, fill | size, colours, red, green | blue
fn bokeh(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let seed = fx.a.z;
    let p = uv * frame_shape();
    let light = bokeh_discs(p, 2.6, seed * 3.7) * 0.5
        + bokeh_discs(p + 5.0, 4.5, seed * 9.1 + 20.0) * 0.7
        + bokeh_discs(p + 9.0, 8.0, seed * 5.3 + 60.0) * 0.6;
    return screened(c, light * fx.a.x);
}

// amount, progress, time, seed | from_x, from_y, to_x, to_y |
// streak, rays, ghosts, red | green, blue
fn flare(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let amount = fx.a.x;
    if (amount <= 0.0) {
        return c;
    }
    let progress = fx.a.y;
    let time = fx.a.z;
    let seed = fx.a.w;
    let streak = fx.c.x;
    let rays = fx.c.y;
    let ghosts = fx.c.z;
    let tint = vec3<f32>(fx.c.w, fx.d.x, fx.d.y);
    let shape = frame_shape();
    let k = progress * progress * (3.0 - 2.0 * progress);
    let light = mix(fx.b.xy, fx.b.zw, k) * shape;
    let mid = 0.5 * shape;
    let p = uv * shape;
    let d = p - light;
    let r = length(d);
    // The light itself and its glow.
    var sum = tint * (0.0035 / (r * r + 0.0035) * 0.9 + exp(-r * 7.0) * 0.45);
    // The streak of a wide-screen lens: thin, long, a little blue.
    let blue = vec3<f32>(0.55, 0.7, 1.0);
    sum = sum + blue * exp(-abs(d.y) * 110.0) * exp(-abs(d.x) * 1.5) * 0.8 * streak;
    sum = sum + blue * exp(-abs(d.y) * 25.0) * exp(-abs(d.x) * 4.0) * 0.25 * streak;
    // Rays: uneven spokes that turn slowly.
    let a = atan2(d.y, d.x) + time * 0.05;
    let spoke = pow(vnoise(vec2<f32>(a * 7.0, seed * 3.0)), 3.0)
        + 0.5 * pow(vnoise(vec2<f32>(a * 19.0, seed * 5.0 + 3.0)), 4.0);
    sum = sum + tint * spoke * exp(-r * 4.5) * 0.7 * rays;
    // A faint rainbow ring round the light.
    sum = sum + rainbow(r * 6.0 + 0.1) * (1.0 - smoothstep(0.0, 0.025, abs(r - 0.36))) * 0.07 * ghosts;
    // Ghosts: discs and rings strung along the line through the middle
    // of the frame, where a real lens puts them.
    let axis = mid - light;
    for (var i = 0; i < 8; i = i + 1) {
        let fi = f32(i);
        let h = hash22(vec2<f32>(fi * 3.1 + 1.0, seed * 7.7));
        let along = 0.3 + fi * 0.3 + (h.x - 0.5) * 0.16;
        let rad = 0.012 + 0.07 * h.y * h.y + 0.012 * fi * step(5.0, fi);
        let gd = length(p - (light + axis * along));
        let disc = 1.0 - smoothstep(rad * 0.9, rad, gd);
        let rim = 0.35 + 0.65 * smoothstep(rad * 0.45, rad, gd);
        sum = sum + mix(tint, rainbow(h.x + fi * 0.21), 0.7) * disc * rim * (0.05 + 0.1 * h.y) * ghosts;
    }
    let flick = 0.93 + 0.07 * vnoise(vec2<f32>(time * 9.0, seed));
    return screened(c, sum * amount * flick);
}

// Specks without a simulation. The frame is cut into a grid of cells,
// a cell may hold one speck, and the speck's place, size and flicker
// come from the cell's own random numbers. The caller slides the whole
// grid to make them travel, so any frame can be drawn alone.
//
// p: the pixel, in shorter sides. cells: cells a shorter side. sd: seed.
// now: seconds. density: the share of cells holding a speck. size: its
// radius as a share of a cell. soft: 0 a hard dot, 1 all blur. stretch:
// squeezes it. wander: how far it strays. twinkle: how fast it flickers.
fn specks_layer(
    p: vec2<f32>,
    cells: f32,
    sd: f32,
    now: f32,
    density: f32,
    size: f32,
    soft: f32,
    stretch: vec2<f32>,
    wander: f32,
    twinkle: f32,
) -> f32 {
    let g = p * cells;
    let id = floor(g);
    // One pixel, in cells. A speck smaller than a pixel is drawn a pixel
    // wide and that much fainter, so a small frame shows the same light
    // as a large one instead of losing the speck between pixels.
    let pixel = cells / min(fx.size.x, fx.size.y);
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cell = id + vec2<f32>(f32(dx), f32(dy));
            if (hash12(cell * 1.7 + sd * 3.1) > density) {
                continue;
            }
            let h = hash22(cell + sd * 17.3);
            let h2 = hash22(cell * 2.3 + sd * 5.9 + 11.0);
            let stray = vec2<f32>(
                sin(now * (0.5 + h2.x) + h2.y * TAU),
                cos(now * (0.4 + h2.y) + h2.x * TAU),
            ) * 0.22 * wander;
            let at = cell + 0.5 + (h - 0.5) * 0.5 + stray;
            let true_r = size * (0.45 + 0.55 * h2.x);
            let r = max(true_r, pixel * 0.8);
            let fainter = (true_r * true_r) / (r * r);
            let d = length((g - at) * stretch);
            var dip = 0.0;
            if (twinkle > 0.0) {
                dip = 0.7;
            }
            let flick = 1.0 - dip * (0.5 + 0.5 * sin(now * twinkle * (0.5 + h.x) + h.y * TAU));
            let inner = min(r * (1.0 - soft), r - pixel * 0.5);
            sum = sum + (1.0 - smoothstep(inner, r, d)) * fainter * flick * (0.4 + 0.6 * h2.y);
        }
    }
    return sum;
}

// amount, time, seed, count | drift_x, drift_y, beam, beam_angle |
// beam_width, red, green, blue
fn motes(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let count = fx.a.w;
    let drift = fx.b.xy;
    let beam = fx.b.z;
    let p = from_middle(uv);
    let round = vec2<f32>(1.0);
    // Far to near: small sharp slow motes, then bigger, softer, faster.
    var v = specks_layer(p + drift * t * 0.5, 34.0, seed, t, 0.30 * count, 0.075, 0.6, round, 1.0, 1.3) * 1.5;
    v = v + specks_layer(p + drift * t * 0.9, 21.0, seed + 7.0, t, 0.28 * count, 0.08, 0.7, round, 1.0, 1.1) * 1.8;
    v = v + specks_layer(p + drift * t * 1.6, 11.0, seed + 13.0, t, 0.22 * count, 0.085, 0.85, round, 1.0, 0.8) * 1.4;
    v = v + specks_layer(p + drift * t * 2.8, 4.5, seed + 23.0, t, 0.2 * count, 0.14, 1.0, round, 1.0, 0.5) * 0.4;
    var lit = 1.0;
    if (beam > 0.0) {
        // Only the dust inside a slanting shaft of light shows.
        let q = turn(p, fx.b.w);
        let width = fx.c.x;
        let shaft = (1.0 - smoothstep(width * 0.4, width, abs(q.x + 0.1)))
            * (0.75 + 0.25 * fbm(vec2<f32>(q.x * 6.0, t * 0.1), 2));
        lit = mix(1.0, shaft, beam);
        v = v + shaft * beam * 0.08;
    }
    return screened(c, fx.c.yzw * v * lit * fx.a.x);
}

// amount, time, seed, count | rise, sway, height, red | green, blue
fn embers(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let count = fx.a.w;
    let rise = fx.b.x;
    let tint = vec3<f32>(fx.b.w, fx.c.x, fx.c.y);
    let p = from_middle(uv);
    // Hot air: the whole field is bent by slow rising clouds.
    let push = (vec2<f32>(
        fbm3(vec3<f32>(p * 1.6 + seed, t * 0.3), 3),
        fbm3(vec3<f32>(p * 1.6 + seed + 30.0, t * 0.3), 3),
    ) - 0.5) * 0.16 * fx.b.y;
    let q = p + push;
    var e = specks_layer(q + vec2<f32>(0.0, rise * 0.6) * t, 30.0, seed, t, 0.22 * count, 0.06, 0.6, vec2<f32>(1.0, 0.6), 1.2, 5.0) * 0.8;
    e = e + specks_layer(q + vec2<f32>(0.02, rise) * t, 17.0, seed + 5.0, t, 0.2 * count, 0.07, 0.7, vec2<f32>(1.0, 0.55), 1.2, 4.0);
    e = e + specks_layer(q + vec2<f32>(-0.02, rise * 1.7) * t, 8.0, seed + 11.0, t, 0.16 * count, 0.08, 0.9, vec2<f32>(1.0, 0.6), 1.0, 3.0) * 0.8;
    // Thicker low down, gone by the top.
    let height = fx.b.z;
    e = e * smoothstep(0.5 - height, 0.5 - height + 0.9, uv.y) * 1.3;
    // Faint ones are deep red, bright ones go orange, then yellow-white.
    return screened(c, vec3<f32>(1.0) - exp(-e * fx.a.x * 3.2 * tint));
}

// amount, time, seed, count | fall, wind, red, green | blue
fn snow(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let count = fx.a.w;
    let way = vec2<f32>(fx.b.y, fx.b.x) * t;
    let p = from_middle(uv);
    let round = vec2<f32>(1.0);
    var v = specks_layer(p - way * 0.45, 40.0, seed, t, 0.5 * count, 0.09, 0.6, round, 1.0, 0.0) * 0.9;
    v = v + specks_layer(p - way * 0.75, 24.0, seed + 3.0, t, 0.45 * count, 0.1, 0.65, round, 1.2, 0.0) * 1.1;
    v = v + specks_layer(p - way * 1.3, 13.0, seed + 9.0, t, 0.4 * count, 0.11, 0.8, round, 1.4, 0.0) * 1.1;
    v = v + specks_layer(p - way * 2.4, 5.0, seed + 17.0, t, 0.3 * count, 0.14, 1.0, round, 1.4, 0.0) * 0.5;
    return screened(c, vec3<f32>(fx.b.z, fx.b.w, fx.c.x) * v * fx.a.x);
}

// One depth of rain: thin columns, drops repeating down each.
fn rain_streaks(p: vec2<f32>, columns: f32, len: f32, sd: f32, speed: f32, density: f32) -> f32 {
    let t = fx.a.y;
    let q = vec2<f32>(p.x + p.y * fx.b.y, p.y);
    let col = floor(q.x * columns);
    let h = hash12(vec2<f32>(col, sd * 7.7));
    if (h > density) {
        return 0.0;
    }
    let h2 = hash12(vec2<f32>(col * 1.9 + 4.0, sd * 3.3));
    // Across the column: a thin line at a random place in it.
    // A line thinner than a pixel is drawn a pixel wide and fainter.
    let x = fract(q.x * columns) - (0.2 + 0.6 * h2);
    let wide = max(0.07, columns / min(fx.size.x, fx.size.y) * 0.8);
    let line = (1.0 - smoothstep(0.0, wide, abs(x))) * 0.07 / wide;
    // Down the column: each drop is bright at its head and fades up
    // its tail.
    let gap = len * (2.5 + 4.0 * h);
    let y = fract((q.y - t * speed * (0.8 + 0.4 * h2)) / gap + h * 13.0) * gap;
    let drop = smoothstep(0.0, len, y) * (1.0 - smoothstep(len, len * 1.08, y));
    return line * drop * (0.5 + 0.5 * h2);
}

// amount, time, seed, count | fall, slant, red, green | blue
fn rain(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let seed = fx.a.z;
    let count = fx.a.w;
    let fall = fx.b.x;
    let p = from_middle(uv);
    var v = rain_streaks(p, 150.0, 0.05, seed, fall * 0.6, 0.55 * count) * 0.35;
    v = v + rain_streaks(p, 90.0, 0.09, seed + 5.0, fall, 0.5 * count) * 0.45;
    v = v + rain_streaks(p, 45.0, 0.17, seed + 11.0, fall * 1.6, 0.4 * count) * 0.3;
    return screened(c, vec3<f32>(fx.b.z, fx.b.w, fx.c.x) * v * fx.a.x);
}

// Where a spark thrown at `v0` is after `age` seconds: air slows it,
// gravity pulls it.
fn spark_flight(v0: vec2<f32>, age: f32, drag: f32, gravity: f32) -> vec2<f32> {
    let k = (1.0 - exp(-drag * age)) / drag;
    return v0 * k + vec2<f32>(0.0, 0.9 * gravity) * age * age * 0.5;
}

// amount, time, seed, x | y, count, speed, gravity | life, aim, fan, every
fn sparks(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let speed = fx.b.z;
    let gravity = fx.b.w;
    let life = fx.c.x;
    let every = fx.c.w;
    let p = (uv - vec2<f32>(fx.a.w, fx.b.x)) * frame_shape();
    var burst = 0.0;
    var since = t;
    if (every > 0.0) {
        burst = floor(t / every);
        since = t - burst * every;
    }
    var light = vec3<f32>(0.0);
    let n = i32(clamp(fx.b.y, 1.0, 120.0));
    for (var i = 0; i < n; i = i + 1) {
        let fi = f32(i);
        let h = hash22(vec2<f32>(fi * 3.1 + 1.0, seed * 7.3 + burst * 13.0));
        let h2 = hash22(vec2<f32>(fi * 5.7 + 9.0, seed * 2.9 + burst * 5.0));
        let lives = life * (0.35 + 0.65 * h2.x);
        let age = since - h2.y * 0.08;
        if (age < 0.0 || age > lives) {
            continue;
        }
        let a = (fx.c.y + (h.x - 0.5) * fx.c.z) * TAU;
        let v0 = vec2<f32>(cos(a), sin(a)) * speed * (0.35 + 1.1 * h.y * h.y);
        let drag = 0.9 + 1.2 * h2.x;
        let head = spark_flight(v0, age, drag, gravity);
        let tail = spark_flight(v0, max(age - 0.06, 0.0), drag, gravity);
        // How far this pixel is from the streak between tail and head.
        let seg = head - tail;
        let along = clamp(dot(p - tail, seg) / max(dot(seg, seg), 0.000001), 0.0, 1.0);
        let d = length(p - tail - seg * along);
        let left = 1.0 - age / lives;
        let glow = exp(-d * d / (0.0000035 + 0.000004 * h2.y)) * (0.35 + 0.65 * along);
        // Hot at first, then cooling from white through yellow to red.
        let heat = mix(vec3<f32>(1.0, 0.25, 0.04), vec3<f32>(1.0, 0.85, 0.6), smoothstep(0.15, 0.7, left));
        light = light + heat * glow * smoothstep(0.0, 0.25, left) * 1.6;
    }
    // The flash at the point of the burst.
    let flash = exp(-since * 14.0) * exp(-dot(p, p) / 0.0015);
    light = light + vec3<f32>(1.0, 0.8, 0.5) * flash;
    return screened(c, light * fx.a.x);
}

// One bank of fog: clouds bent by slower clouds.
fn fog_bank(p: vec2<f32>, z: f32, sd: f32) -> f32 {
    let w = vec2<f32>(
        fbm3(vec3<f32>(p * 0.8 + sd, z), 3),
        fbm3(vec3<f32>(p * 0.8 + sd + 31.0, z), 3),
    ) - 0.5;
    return fbm3(vec3<f32>(p + w * 1.1, z * 1.6 + sd), 5);
}

// amount, time, seed, fill | scale, wind_x, wind_y, ground | red, green, blue
fn fog(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let wind = fx.b.yz;
    let p = uv * frame_shape() * 1.6 / max(fx.b.x, 0.01);
    let z = t * 0.06;
    // A far bank and a near one: the near one is bigger and faster.
    let back = fog_bank(p * 1.5 - wind * t * 0.025, z, seed * 7.1);
    let front = fog_bank(p * 0.8 - wind * t * 0.06 + 9.0, z * 1.3, seed * 3.3 + 50.0);
    let edge = 0.62 - 0.3 * fx.a.w;
    let v = smoothstep(edge, edge + 0.35, back) * 0.6 + smoothstep(edge, edge + 0.4, front) * 0.7;
    let lie = mix(1.0, smoothstep(0.25, 1.0, uv.y + (front - 0.5) * 0.3), fx.b.w);
    return screened(c, fx.c.xyz * clamp(v * lie * fx.a.x * 0.55, 0.0, 1.0));
}

// amount, time, seed, x | y, width, spread, height | rise, curl, red, green |
// blue
fn smoke(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let width = fx.b.y;
    let curl = fx.c.y;
    let p = (uv - vec2<f32>(fx.a.w, fx.b.x)) * frame_shape();
    // How far above the source this pixel is.
    let h = -p.y;
    let sd = fx.a.z * 11.3;
    let up = t * 0.35 * fx.c.x;
    // The whole plume sways, more the higher it gets.
    let sway = (fbm3(vec3<f32>(h * 1.4 - up * 0.7, sd, t * 0.08), 3) - 0.5) * 0.9 * curl * h;
    var q = vec2<f32>(p.x - sway, h);
    // Curls: the smoke is pushed about by clouds that rise with it.
    let push = vec2<f32>(
        fbm3(vec3<f32>(q * 2.6 + vec2<f32>(0.0, -up * 2.0) + sd, t * 0.15), 4),
        fbm3(vec3<f32>(q * 2.6 + vec2<f32>(40.0, -up * 2.0) + sd, t * 0.15), 4),
    ) - 0.5;
    q = q + push * 0.3 * curl * (0.3 + h);
    let wide = width + fx.b.z * max(h, 0.0);
    let across = q.x / wide;
    let body = exp(-across * across * 2.2);
    // Wisps: stretched upwards, streaming up.
    let n = fbm3(vec3<f32>(q.x * 4.5 / (0.6 + h), q.y * 2.2 - up * 2.4, t * 0.1 + sd), 5);
    let wisps = smoothstep(0.3, 0.72, n);
    let height = fx.b.w;
    let fade = smoothstep(0.0, 0.06, h) * (1.0 - smoothstep(height * 0.35, height, h));
    // Thinner as it spreads.
    let thin = width / wide * 2.2 + 0.25;
    let tint = vec3<f32>(fx.c.z, fx.c.w, fx.d.x);
    return screened(c, tint * clamp(body * wisps * fade * min(thin, 1.0) * fx.a.x * 1.28, 0.0, 1.0));
}

// Where a pixel is on the film, in the pixels of a frame 1080 on its
// shorter side: the measure film dirt and tape lines are sized in.
fn film_at(uv: vec2<f32>) -> vec2<f32> {
    return uv * frame_shape() * 1080.0;
}

// One layer of grain specks, about -1 to 1, for one colour layer on one
// frame.
fn grain_specks(at: vec2<f32>, cell: f32, frame: f32, channel: f32, seed: f32) -> f32 {
    let slide = hash22(vec2<f32>(frame * 1.37 + channel * 91.0, seed * 7.13 + cell)) * 4096.0;
    // A frame too small to show a speck draws specks a pixel wide and
    // that much weaker, as many small specks average out in one pixel.
    let shown = max(cell, 1080.0 / min(fx.size.x, fx.size.y) * 0.8);
    let q = at / shown + slide;
    // Two lots of hills, the second turned, so no grid shows.
    let n = vnoise(q) + vnoise(turn(q, 0.9) * 1.31 + 57.0) - 1.0;
    return n * 2.2 * cell / shown;
}

fn grain_layer(at: vec2<f32>, frame: f32, channel: f32) -> f32 {
    let size = fx.a.w;
    let clump = fx.b.x;
    let fine = grain_specks(at, size, frame, channel, fx.a.z);
    let big = grain_specks(at, size * 2.7, frame, channel + 3.0, fx.a.z);
    return (fine + big * clump) / (1.0 + clump * 0.6);
}

// amount, time, seed, size | clump, colour
fn film_grain(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let frame = film_frame(fx.a.y);
    let at = film_at(uv);
    var g = vec3<f32>(grain_layer(at, frame, 0.0));
    if (fx.b.y > 0.0) {
        let own = vec3<f32>(grain_layer(at, frame, 1.0), grain_layer(at, frame, 2.0), grain_layer(at, frame, 3.0));
        g = mix(g, own, fx.b.y);
    }
    return overlaid(c, clamp(vec3<f32>(0.5) + g * fx.a.x, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// Dust in one grid: `cell` is the cell size and `big` the largest
// radius, both in film pixels. `rag` is how ragged the bits are and
// `rare` how few of them are big.
fn film_dust(at: vec2<f32>, cell: f32, big: f32, density: f32, sd: f32, clumps: f32, size: f32) -> f32 {
    let g = at / cell;
    let id = floor(g);
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cl = id + vec2<f32>(f32(dx), f32(dy));
            // Patches: some parts of the frame are dirtier than others.
            let dirtier = mix(1.0, smoothstep(0.35, 0.75, fbm(cl * cell * 0.0035 + sd * 0.37, 3)) * 2.2, clumps);
            if (hash12(cl * 1.7 + sd * 3.1) > min(density * dirtier, 0.4)) {
                continue;
            }
            let h = hash22(cl + sd * 17.3);
            let h2 = hash22(cl * 2.3 + sd * 5.9 + 11.0);
            // Anywhere in its cell, so no rows or columns show.
            var q = (g - cl - h) * cell;
            // Most bits are tiny, a few are big.
            let r = mix(0.55, big, pow(h2.x, 5.0)) * size;
            // Ragged: the edge is pushed in and out, and the bit is
            // squashed a random way.
            q = turn(q, h2.y * TAU) * vec2<f32>(1.0, 0.55 + 0.9 * h.x);
            let rag = (vnoise(q * 0.9 / size + cl * 7.0) - 0.5) * r * 1.1;
            let d = length(q) + rag;
            sum = sum + (1.0 - smoothstep(r - 0.7, r + 0.5, d)) * (0.45 + 0.55 * h2.y);
        }
    }
    return sum;
}

// A hair or a nick: a thin line `len` long through `mid`, lying at
// `angle`, bent by `curl`.
fn film_fibre(at: vec2<f32>, mid: vec2<f32>, angle: f32, len: f32, thick: f32, curl: f32, sd: f32) -> f32 {
    let q = turn(at - mid, -angle);
    let along = q.x / (len * 0.5);
    if (abs(along) > 1.0) {
        return 0.0;
    }
    // The bend: a bow plus a wriggle, both scaled by the hair's length.
    let bend = curl * len * (0.22 * (along * along - 0.4) * sin(sd * 9.0)
        + 0.09 * sin(along * (2.5 + 4.0 * fract(sd * 3.7)) + sd * 20.0));
    let d = abs(q.y - bend);
    // Thinner towards the ends.
    let w = thick * (1.0 - 0.6 * along * along);
    return 1.0 - smoothstep(w - 0.5, w + 0.5, d);
}

// amount, time, seed, count | hold, specks, hairs, scratches | lines,
// clumps, size
fn film_damage(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let count = fx.a.w;
    let clumps = fx.c.y;
    let size = fx.c.z;
    let frame = film_frame(t);
    let beat = floor(frame / max(fx.b.x, 1.0));
    let at = film_at(uv);
    let full = frame_shape() * 1080.0;
    let sd = seed * 13.7 + beat * 7.31;
    var v = 0.0;
    // Specks: a fine layer of pin-points, laid twice with the second
    // turned so its grid cannot be seen, and a coarse layer of lint.
    if (fx.b.y > 0.0) {
        let dust = count * fx.b.y;
        v = v + film_dust(at, 31.0 * size, 2.2, 0.09 * dust, sd, clumps, size) * 0.9;
        v = v + film_dust(turn(at, 0.63) + 300.0, 23.0 * size, 2.0, 0.07 * dust, sd + 80.0, clumps, size) * 0.9;
        v = v + film_dust(turn(at, -0.31), 95.0 * size, 7.0, 0.17 * dust, sd + 40.0, clumps, size);
    }
    // Hairs and nicks: a handful a frame, each somewhere new.
    let hairs = i32(clamp(7.0 * count * fx.b.z, 0.0, 24.0));
    for (var i = 0; i < hairs; i = i + 1) {
        let fi = f32(i);
        let h = hash22(vec2<f32>(fi * 3.3 + 1.0, sd));
        let h2 = hash22(vec2<f32>(fi * 7.1 + 5.0, sd + 3.0));
        let len = mix(25.0, 150.0, h2.x * h2.x) * size;
        v = v + film_fibre(at, h * full, h2.y * TAU, len, 0.8 * size, 1.0 + 1.5 * h.x, sd + fi) * (0.4 + 0.5 * h.y);
    }
    let nicks = i32(clamp(6.0 * count * fx.b.w, 0.0, 24.0));
    for (var i = 0; i < nicks; i = i + 1) {
        let fi = f32(i);
        let h = hash22(vec2<f32>(fi * 4.7 + 9.0, sd + 21.0));
        let h2 = hash22(vec2<f32>(fi * 2.9 + 2.0, sd + 27.0));
        let len = mix(12.0, 70.0, h2.x * h2.x) * size;
        v = v + film_fibre(at, h * full, h2.y * TAU, len, (0.9 + 1.3 * h.x * h.x) * size, 0.0, sd + fi) * (0.6 + 0.4 * h.y);
    }
    // Lines: scratches down the reel. Each keeps its place for a second
    // or two, shivers from frame to frame, breaks up along its length
    // and comes and goes.
    if (fx.c.x > 0.0) {
        for (var i = 0; i < 4; i = i + 1) {
            let fi = f32(i);
            let lasts = 1.0 + 1.5 * hash12(vec2<f32>(fi, seed * 3.0));
            let turn_no = floor(t / lasts + fi * 0.37);
            let h = hash22(vec2<f32>(fi * 5.9 + turn_no * 1.3, seed * 2.3 + turn_no));
            if (h.y > 0.45 * fx.c.x * count) {
                continue;
            }
            let x = h.x * full.x + (hash12(vec2<f32>(frame, fi + seed)) - 0.5) * 2.0
                + (vnoise(vec2<f32>(at.y * 0.004, fi + turn_no)) - 0.5) * 6.0;
            let broken = smoothstep(0.3, 0.6, vnoise(vec2<f32>(at.y * 0.012 + frame * 3.0, fi * 9.0 + seed)));
            let flick = 0.35 + 0.65 * hash12(vec2<f32>(frame * 1.7, fi * 3.0 + seed));
            v = v + (1.0 - smoothstep(0.3 * size, 1.1 * size, abs(at.x - x))) * broken * flick * 0.45;
        }
    }
    return screened(c, vec3<f32>(v * fx.a.x));
}

fn grime_dust(at: vec2<f32>, cell: f32, big: f32, density: f32, sd: f32) -> f32 {
    let g = at / cell;
    let id = floor(g);
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cl = id + vec2<f32>(f32(dx), f32(dy));
            if (hash12(cl * 1.7 + sd * 3.1) > density) {
                continue;
            }
            let h = hash22(cl + sd * 17.3);
            let h2 = hash22(cl * 2.3 + sd * 5.9 + 11.0);
            var q = (g - cl - h) * cell;
            let r = mix(0.7, big, pow(h2.x, 4.0));
            q = turn(q, h2.y * TAU) * vec2<f32>(1.0, 0.6 + 0.8 * h.x);
            let d = length(q) + (vnoise(q * 0.8 + cl * 7.0) - 0.5) * r;
            sum = sum + (1.0 - smoothstep(r - 0.8, r + 0.6, d)) * (0.5 + 0.5 * h2.y);
        }
    }
    return sum;
}

// amount, time, seed, count | hold, stains, edges, specks | blotches,
// red, green, blue
fn grime(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let seed = fx.a.z;
    let count = fx.a.w;
    let stains = fx.b.y;
    let frame = film_frame(fx.a.y);
    let beat = floor(frame / max(fx.b.x, 1.0));
    let at = film_at(uv);
    let full = frame_shape() * 1080.0;
    let sd = seed * 13.7 + beat * 7.31;
    var dark = 0.0;
    // Stains: the inside a little darker, the rim clearly darker, like
    // a dried water mark. These stay put.
    let fine = fbm(at * 0.009 + seed * 2.1, 4);
    let s = fbm(at * 0.0022 + seed * 5.3, 5) + (fine - 0.5) * 0.22;
    dark = dark + smoothstep(0.58, 0.63, s) * 0.14 * stains * (0.5 + fine);
    dark = dark + (1.0 - smoothstep(0.0, 0.018, abs(s - 0.6))) * 0.2 * stains * (0.4 + fine);
    // Edges: the frame's border is dirtier, unevenly.
    let inside = min(min(at.x, full.x - at.x), min(at.y, full.y - at.y));
    let reach = 40.0 + 190.0 * fbm(at * 0.004 + seed * 8.0, 3);
    dark = dark + (1.0 - smoothstep(0.0, reach, inside)) * 0.6 * fx.b.z * fbm(at * 0.02 + seed * 9.0, 3);
    // Specks: dark dust and lint, new every few frames.
    dark = dark + grime_dust(at, 37.0, 2.4, 0.06 * count * fx.b.w, sd) * 0.8;
    dark = dark + grime_dust(turn(at, 0.5) + 200.0, 110.0, 8.0, 0.14 * count * fx.b.w, sd + 40.0) * 0.9;
    // Blotches: one frame in a few has a big ragged splotch.
    for (var i = 0; i < 2; i = i + 1) {
        let fi = f32(i);
        let h = hash22(vec2<f32>(fi * 3.3 + 1.0, sd));
        let h2 = hash22(vec2<f32>(fi * 7.1 + 5.0, sd + 3.0));
        if (h2.x > 0.22 * fx.c.x * count) {
            continue;
        }
        let q = at - h * full;
        let r = 18.0 + 60.0 * h2.y;
        let d = length(q * vec2<f32>(1.0, 0.6 + 0.8 * h.x)) + (fbm(q * 0.03 + fi * 9.0 + sd, 3) - 0.5) * r * 1.6;
        dark = dark + (1.0 - smoothstep(r * 0.7, r, d)) * 0.75;
    }
    return multiplied(c, mix(vec3<f32>(1.0), fx.c.yzw, clamp(dark * fx.a.x * 0.7, 0.0, 1.0)));
}

// amount, time, seed, hold | band, pulse
fn flicker(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let beat = floor(film_frame(t) / max(fx.a.w, 1.0));
    // Most frames are nearly clean, a few dip hard.
    let quick = pow(hash12(vec2<f32>(beat * 1.37, seed * 9.1)), 2.5);
    let slow = 0.5 + 0.5 * sin(t * 2.1 + seed) * sin(t * 0.77 + seed * 2.0);
    var dip = mix(quick, slow, fx.b.y * 0.5);
    // A soft band of shade rolling up the picture.
    let roll = 0.5 + 0.5 * sin((uv.y * 1.3 + t * 0.9 + seed) * TAU);
    dip = dip + fx.b.x * 0.5 * roll * (0.4 + 0.6 * hash12(vec2<f32>(beat, seed + 4.0)));
    return multiplied(c, vec3<f32>(1.0 - clamp(dip * fx.a.x, 0.0, 1.0)));
}

// amount, time, seed, wobble | tracking, switching, scan, snow
fn tape(uv: vec2<f32>) -> vec4<f32> {
    let amount = fx.a.x;
    if (amount <= 0.0) {
        return fetch(uv);
    }
    let t = fx.a.y;
    let seed = fx.a.z;
    let scan = fx.b.z;
    let snow = fx.b.w;
    let frame = film_frame(t);
    let at = film_at(uv);
    // Tape lines are two film pixels tall.
    let row = floor(at.y / 2.0);
    // The tracking band rolls up the frame about once every nine
    // seconds; the bottom strip tears where the heads switch.
    let band_y = fract(0.85 - t * 0.11 + seed * 0.37);
    let off = (uv.y - band_y) / 0.03;
    let band = exp(-off * off) * fx.b.x;
    let strip = step(0.977, uv.y) * fx.b.y;
    // How far this line is shoved sideways, in film pixels: lines
    // jitter, the picture bends, the band and the strip tear.
    var shove = fx.a.w * ((hash12(vec2<f32>(row, frame + seed)) - 0.5) * 1.6
        + sin(uv.y * 9.0 + t * 2.3) * 2.0 * vnoise(vec2<f32>(t * 0.7, seed)));
    shove = shove + band * (hash12(vec2<f32>(row * 1.3, frame * 1.7 + seed)) - 0.25) * 46.0;
    shove = shove + strip * (22.0 + 34.0 * hash12(vec2<f32>(row, frame * 2.1 + seed)));
    let c = fetch(uv + vec2<f32>(shove * amount / (1080.0 * frame_shape().x), 0.0));
    // Scan lines and a rolling shade. A frame too small to hold a line
    // every four pixels gets the shade without the lines.
    let fits = clamp(min(fx.size.x, fx.size.y) / 540.0 - 0.5, 0.0, 1.0);
    let line = mix(0.5, 0.5 + 0.5 * cos(at.y * PI * 0.5), fits);
    var shade = 1.0 - scan * 0.2 * line;
    shade = shade * (1.0 - 0.05 * scan * (0.5 + 0.5 * sin(uv.y * 5.0 - t * 1.7)));
    shade = shade * (1.0 - 0.25 * band);
    // Static: dropouts (a white dash with a tail), and snow in the
    // band and the strip.
    var v = 0.0;
    let h = hash22(vec2<f32>(floor(at.x / 90.0), row) + vec2<f32>(frame * 3.1, seed));
    if (h.x > 1.0 - 0.0016 * snow) {
        v = v + (1.0 - fract(at.x / 90.0)) * (0.5 + 0.5 * h.y);
    }
    let grit = hash12(vec2<f32>(floor(at.x / 2.0), row) + frame * 7.7);
    v = v + band * step(0.55, grit) * 0.7 * snow;
    v = v + strip * step(0.45, grit) * 0.55 * snow;
    let shaded = multiplied(c, vec3<f32>(mix(1.0, shade, min(amount, 1.0))));
    return screened(shaded, vec3<f32>(v * min(amount, 1.0)));
}

// amount, progress, time, seed | push, split, torn, hold | blocks
fn band_tear(uv: vec2<f32>) -> vec4<f32> {
    let amount = fx.a.x;
    if (amount <= 0.0) {
        return fetch(uv);
    }
    let seed = fx.a.w;
    let beat = floor(film_frame(fx.a.z) / max(fx.b.w, 1.0));
    // The burst: nothing at the ends, strongest in the middle, never
    // smooth.
    var swell = sin(PI * clamp(fx.a.y, 0.0, 1.0)) * amount;
    swell = swell * (0.55 + 0.45 * hash12(vec2<f32>(beat, seed * 3.0)));
    // Bands of uneven height: most stay still, a few tear.
    let rows = mix(6.0, 26.0, hash12(vec2<f32>(beat * 1.3, seed)));
    let band = floor(uv.y * rows + hash12(vec2<f32>(beat, seed + 5.0)) * 3.0);
    var shove = 0.0;
    if (hash12(vec2<f32>(band * 3.7, beat * 1.9 + seed)) < fx.b.z) {
        shove = (hash12(vec2<f32>(band, beat + seed * 7.0)) - 0.5) * 2.0;
    }
    // Small blocks that jump, on top of the bands.
    let cell = floor(uv * vec2<f32>(24.0, 14.0));
    if (hash12(cell + beat * 5.3 + seed) < 0.06 * fx.c.x) {
        shove = shove + (hash12(cell * 1.3 + beat) - 0.5) * 1.2;
    }
    // Film pixels to a share of the frame's width.
    let wide = 1080.0 * frame_shape().x;
    let base = shove * fx.b.x * swell / wide;
    let apart = fx.b.y * swell * (0.6 + 0.8 * hash12(vec2<f32>(beat, seed + 9.0))) / wide;
    let red = fetch(uv + vec2<f32>(base + apart, 0.0));
    let middle = fetch(uv + vec2<f32>(base, 0.0));
    let blue = fetch(uv + vec2<f32>(base - apart, 0.0));
    return vec4<f32>(red.r, middle.g, blue.b, max(middle.a, max(red.a, blue.a)));
}

// A sheet of paper worked out from where a pixel is: tooth, fibres,
// specks, soft clouds and creases. `at` is in film pixels. Returns a
// brightness around 1.
fn paper_sheet(at: vec2<f32>, sd: f32, tooth: f32, fibres: f32, mottle: f32, specks: f32, creases: f32) -> f32 {
    var v = 1.0;
    // Tooth: the fine roughness of the surface.
    v = v + (fbm(at * 0.8 + sd * 11.0, 3) - 0.5) * 0.09 * tooth;
    v = v + (hash12(at + sd * 3.0) - 0.5) * 0.03 * tooth;
    // Fibres: long thin hairs lying every which way, some lighter than
    // the sheet, some darker.
    for (var k = 0; k < 7; k = k + 1) {
        let fk = f32(k);
        let a = hash12(vec2<f32>(fk * 5.1, sd * 2.3)) * PI;
        var r = turn(at + vec2<f32>(fk * 311.0, sd * 97.0), a);
        // A slow bend, so the hairs curl instead of lying like scratches.
        r.y = r.y + (vnoise(r * 0.006 + fk * 3.0) - 0.5) * 70.0;
        let n = vnoise(vec2<f32>(r.x * 0.011, r.y * 0.4));
        var hair = 1.0 - smoothstep(0.0, 0.035, abs(n - 0.5));
        // Only here and there, not everywhere.
        hair = hair * smoothstep(0.5, 0.68, vnoise(r * 0.018 + fk * 9.0));
        var tone = -0.035;
        if (k % 2 == 0) {
            tone = 0.03;
        }
        v = v + hair * tone * fibres;
    }
    // Soft clouds: kept low, since big blotches show in flat skies.
    v = v + (fbm(at * 0.004 + sd * 5.0, 4) - 0.5) * 0.3 * mottle;
    // Specks: a few small dark bits pressed into the sheet.
    let cell = floor(at / 16.0);
    if (hash12(cell + sd * 13.0) < 0.05 * specks) {
        let spot = (0.15 + 0.7 * hash22(cell + sd * 29.0)) * 16.0;
        let rr = 0.5 + 1.7 * hash12(cell * 1.7 + 3.0);
        let d = length(at - cell * 16.0 - spot);
        v = v - (1.0 - smoothstep(rr - 0.6, rr + 0.6, d)) * (0.1 + 0.25 * hash12(cell + 71.0));
    }
    // Creases: straight fold lines, each with light on one side and
    // shade on the other.
    if (creases > 0.0) {
        for (var k = 0; k < 4; k = k + 1) {
            let fk = f32(k);
            let h = hash22(vec2<f32>(fk * 8.3 + 2.0, sd * 6.1 + 4.0));
            // A sheet is folded mostly square to its edges.
            var a = (h.x - 0.5) * 0.16;
            var far = 0.0;
            if (k % 2 == 1) {
                a = a + PI * 0.5;
                far = 500.0;
            }
            let nrm = vec2<f32>(-sin(a), cos(a));
            var d = abs(dot(at, nrm)) - mix(180.0, 900.0, h.y) - far;
            d = d + (vnoise(at * 0.01 + fk * 5.0) - 0.5) * 3.0;
            // The fold is a thin dark line; one side of it catches the
            // light, the other falls away.
            v = v - exp(-d * d / 2.5) * 0.06 * creases;
            v = v + sign(d) * exp(-abs(d) / 70.0) * 0.022 * creases;
        }
    }
    return v;
}

// amount, seed, scale, tooth | fibres, mottle, specks, creases | red,
// green, blue
fn paper(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let at = film_at(uv) / max(fx.a.z, 0.01);
    let v = paper_sheet(at, fx.a.y, fx.a.w, fx.b.x, fx.b.y, fx.b.z, fx.b.w);
    let sheet = clamp(fx.c.xyz * v, vec3<f32>(0.0), vec3<f32>(1.0));
    return multiplied(c, mix(vec3<f32>(1.0), sheet, fx.a.x));
}

// The distance to the nearest of a scatter of points: a field of cones
// with sharp valleys between them, which is what a crease is.
fn crumple_cones(p: vec2<f32>, sd: f32) -> f32 {
    let id = floor(p);
    var best = 9.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cl = id + vec2<f32>(f32(dx), f32(dy));
            best = min(best, length(p - cl - hash22(cl + sd)));
        }
    }
    return best;
}

// The height of crumpled paper at a point.
fn crumple_surface(p: vec2<f32>) -> f32 {
    let sd = fx.a.y * 7.3;
    // The cones are pushed about, so the creases are not quite straight.
    let w = p + (vec2<f32>(fbm(p * 0.7 + sd, 3), fbm(p * 0.7 + sd + 30.0, 3)) - 0.5) * 0.5;
    return crumple_cones(w * 2.2, sd)
        + crumple_cones(turn(w, 0.7) * 5.1 + 9.0, sd + 5.0) * 0.38
        + crumple_cones(turn(w, -0.4) * 11.0 + 3.0, sd + 9.0) * 0.12;
}

// amount, seed, facets, base | light
fn crumple(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    if (fx.a.x <= 0.0) {
        return c;
    }
    let p = uv * frame_shape() * fx.a.z * 1.6;
    let e = 0.003;
    let h = crumple_surface(p);
    let slope = vec2<f32>(crumple_surface(p + vec2<f32>(e, 0.0)) - h, crumple_surface(p + vec2<f32>(0.0, e)) - h) / e;
    let lit = dot(slope, vec2<f32>(cos(fx.b.x), sin(fx.b.x)));
    // Soft, so a steep facet does not go black or white.
    let shade = lit / (1.0 + abs(lit) * 0.6) * 0.085 * fx.a.x;
    return multiplied(c, vec3<f32>(mix(1.0, fx.a.w, min(fx.a.x, 1.0)) + shade));
}
";
