//! Soft clipping of float output before integer conversion
//! (`opus_pcm_soft_clip`).

/// Bends samples beyond full scale back under it with a smooth
/// `x + a * x^2` curve between the zero crossings around each peak, so a
/// float decode can be converted to integers without hard clipping.
/// `x` is interleaved with `mem.len()` channels; `mem` carries each
/// channel's curve into the next call and starts at zero.
pub(crate) fn soft_clip(x: &mut [f32], mem: &mut [f32]) {
    let channels = mem.len();
    if channels == 0 || x.len() < channels {
        return;
    }
    let n = x.len() / channels;
    for v in x.iter_mut() {
        *v = if v.is_nan() { 0.0 } else { v.clamp(-2.0, 2.0) };
    }
    for (c, slot) in mem.iter_mut().enumerate() {
        let at = |i: usize| i * channels + c;
        let mut a = *slot;
        // Continue the previous frame's curve up to the first sign change.
        for i in 0..n {
            let v = x[at(i)];
            if v * a >= 0.0 {
                break;
            }
            x[at(i)] = v + a * v * v;
        }
        let mut curr = 0usize;
        let x0 = x[at(0)];
        loop {
            let Some(i) = (curr..n).find(|&i| x[at(i)].abs() > 1.0) else {
                a = 0.0;
                break;
            };
            let xi = x[at(i)];
            let mut peak_pos = i;
            let mut start = i;
            let mut end = i;
            let mut maxval = xi.abs();
            while start > 0 && xi * x[at(start - 1)] >= 0.0 {
                start -= 1;
            }
            while end < n && xi * x[at(end)] >= 0.0 {
                if x[at(end)].abs() > maxval {
                    maxval = x[at(end)].abs();
                    peak_pos = end;
                }
                end += 1;
            }
            let special = start == 0 && xi * x[at(0)] >= 0.0;
            // maxval + a * maxval^2 = 1.
            a = (maxval - 1.0) / (maxval * maxval);
            a += a * 2.4e-7;
            if xi > 0.0 {
                a = -a;
            }
            for k in start..end {
                let v = x[at(k)];
                x[at(k)] = v + a * v * v;
            }
            if special && peak_pos >= 2 {
                // A ramp from the first sample to the peak avoids a step at
                // the start of the frame.
                let mut offset = x0 - x[at(0)];
                let delta = offset / peak_pos as f32;
                for k in curr..peak_pos {
                    offset -= delta;
                    x[at(k)] = (x[at(k)] + offset).clamp(-1.0, 1.0);
                }
            }
            curr = end;
            if curr == n {
                break;
            }
        }
        *slot = a;
    }
}

/// Converts float samples to 16-bit the way `opus_demo` does after soft
/// clipping (`FLOAT2INT16`: scale, clamp, round half to even).
#[cfg(test)]
pub(crate) fn to_i16(x: f32) -> i16 {
    let v = (x * 32768.0).clamp(-32768.0, 32767.0);
    v.round_ties_even() as i16
}

#[cfg(test)]
mod tests {
    use super::soft_clip;

    #[test]
    fn the_curve_keeps_everything_within_full_scale() {
        let mut mem = [0f32; 2];
        for frame in 0..4 {
            let mut x: Vec<f32> = (0..960)
                .map(|i| 1.8 * ((i + frame * 480) as f32 * 0.01).sin())
                .collect();
            soft_clip(&mut x, &mut mem);
            assert!(x.iter().all(|v| v.abs() <= 1.0 + 1e-6));
        }
        let mut quiet = vec![0.5f32; 100];
        let mut mem = [0f32];
        soft_clip(&mut quiet, &mut mem);
        assert!(quiet.iter().all(|&v| v == 0.5));
        let mut odd = vec![f32::NAN, f32::INFINITY, -3.0];
        soft_clip(&mut odd, &mut mem);
        assert!(odd.iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6));
    }
}
