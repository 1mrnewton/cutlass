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
";
