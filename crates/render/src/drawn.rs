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
";
