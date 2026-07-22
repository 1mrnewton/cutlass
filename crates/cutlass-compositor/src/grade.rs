//! Resolved per-layer color grade parameters consumed by the GPU shaders.
//!
//! The render bridge maps persisted [`cutlass_models::ColorAdjustments`] and
//! filter presets into this compact form. Per-clip effects, masks, and canvas
//! passes use adjacent compositor plumbing, but share this grade representation.
//!
//! # Spatial controls
//!
//! `sharpness` and `vignette` need neighboring texels / layer UVs. They are
//! packed into the grade uniform block and applied in the textured layer
//! shaders (`rgba` / `rgba_fx` / `yuv` / `yuv_fx`) and the canvas grade pass —
//! not in solid/shape/glyph paths (no useful neighbors). The CPU mirror
//! [`ColorGrade::apply`] covers the per-pixel ops; [`ColorGrade::apply_image`]
//! adds the same 4-tap sharpness and radial vignette for full-frame parity.

/// Manual color grade + filter-preset strengths, ready for the WGSL uniform block.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorGrade {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub exposure: f32,
    pub temperature: f32,
    pub tint: f32,
    /// Hue rotation; ±1 → ±30° via YIQ.
    pub hue: f32,
    /// Lift/compress tones above mid-luma.
    pub highlights: f32,
    /// Lift/compress tones below mid-luma.
    pub shadows: f32,
    /// Unsharp-mask strength (`0`…`1`); applied where texture neighbors exist.
    pub sharpness: f32,
    /// Radial darkening from layer center (`0`…`1`).
    pub vignette: f32,
}

impl ColorGrade {
    /// Identity grade: all controls are neutral and the shader output is unchanged.
    pub const IDENTITY: Self = Self {
        brightness: 0.0,
        contrast: 0.0,
        saturation: 0.0,
        exposure: 0.0,
        temperature: 0.0,
        tint: 0.0,
        hue: 0.0,
        highlights: 0.0,
        shadows: 0.0,
        sharpness: 0.0,
        vignette: 0.0,
    };

    /// True when every slider sits at neutral, so grade work can be skipped.
    pub const fn is_identity(&self) -> bool {
        self.brightness == 0.0
            && self.contrast == 0.0
            && self.saturation == 0.0
            && self.exposure == 0.0
            && self.temperature == 0.0
            && self.tint == 0.0
            && self.hue == 0.0
            && self.highlights == 0.0
            && self.shadows == 0.0
            && self.sharpness == 0.0
            && self.vignette == 0.0
    }

    /// CPU mirror of `grade.wgsl`'s `apply_color_grade` (per-pixel ops only —
    /// sharpness / vignette need [`Self::apply_image`]).
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut c = rgb;
        let exposure = 2f32.powf(2.0 * self.exposure);
        for ch in &mut c {
            *ch *= exposure;
        }
        c[0] += 0.25 * self.temperature;
        c[2] -= 0.25 * self.temperature;
        c[1] += 0.25 * self.tint;
        for ch in &mut c {
            *ch += 0.25 * self.brightness;
            *ch = (*ch - 0.5) * (1.0 + self.contrast) + 0.5;
        }
        let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let sat = 1.0 + self.saturation;
        c = c.map(|ch| luma + (ch - luma) * sat);

        // YIQ hue rotation: ±1 → ±30°.
        if self.hue != 0.0 {
            let y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
            let i = 0.596 * c[0] - 0.274 * c[1] - 0.322 * c[2];
            let q = 0.211 * c[0] - 0.523 * c[1] + 0.312 * c[2];
            // Negated so +hue rotates red toward yellow (matches the shader).
            let angle = -self.hue * std::f32::consts::PI / 6.0;
            let (sin_a, cos_a) = angle.sin_cos();
            let i2 = i * cos_a - q * sin_a;
            let q2 = i * sin_a + q * cos_a;
            c = [
                y + 0.956 * i2 + 0.621 * q2,
                y - 0.272 * i2 - 0.647 * q2,
                y - 1.106 * i2 + 1.703 * q2,
            ];
        }

        let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        if self.highlights != 0.0 {
            let w = smoothstep(0.5, 1.0, luma);
            let hl = self.highlights;
            for ch in &mut c {
                let lift = if hl >= 0.0 { 1.0 - *ch } else { *ch };
                *ch += hl * w * 0.35 * lift;
            }
        }
        if self.shadows != 0.0 {
            let w = 1.0 - smoothstep(0.0, 0.5, luma);
            let sh = self.shadows;
            for ch in &mut c {
                let lift = if sh >= 0.0 { 1.0 - *ch } else { *ch };
                *ch += sh * w * 0.35 * lift;
            }
        }

        c.map(|ch| ch.clamp(0.0, 1.0))
    }

    /// Full-image CPU mirror of the textured grade path: per-pixel [`Self::apply`],
    /// then 4-tap unsharp mask and radial vignette (same formulas as the shaders).
    pub fn apply_image(&self, width: u32, height: u32, rgba: &mut [u8]) {
        assert_eq!(rgba.len(), (width * height * 4) as usize);
        if self.is_identity() {
            return;
        }

        // Grade every pixel first (sharpness neighbors must see graded colors).
        let mut graded = vec![[0f32; 3]; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 4) as usize;
                let rgb = [
                    f32::from(rgba[i]) / 255.0,
                    f32::from(rgba[i + 1]) / 255.0,
                    f32::from(rgba[i + 2]) / 255.0,
                ];
                graded[(y * width + x) as usize] = self.apply(rgb);
            }
        }

        let sharp = self.sharpness.max(0.0);
        let vig = self.vignette.max(0.0);
        let half_diag = 0.5 * (2f32).sqrt();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let mut c = graded[idx];

                if sharp > 0.0 && width > 1 && height > 1 {
                    let sample = |sx: i32, sy: i32| {
                        let sx = sx.clamp(0, width as i32 - 1) as u32;
                        let sy = sy.clamp(0, height as i32 - 1) as u32;
                        graded[(sy * width + sx) as usize]
                    };
                    let xi = x as i32;
                    let yi = y as i32;
                    let avg = [
                        (sample(xi + 1, yi)[0]
                            + sample(xi - 1, yi)[0]
                            + sample(xi, yi + 1)[0]
                            + sample(xi, yi - 1)[0])
                            * 0.25,
                        (sample(xi + 1, yi)[1]
                            + sample(xi - 1, yi)[1]
                            + sample(xi, yi + 1)[1]
                            + sample(xi, yi - 1)[1])
                            * 0.25,
                        (sample(xi + 1, yi)[2]
                            + sample(xi - 1, yi)[2]
                            + sample(xi, yi + 1)[2]
                            + sample(xi, yi - 1)[2])
                            * 0.25,
                    ];
                    c = [
                        c[0] + sharp * 1.5 * (c[0] - avg[0]),
                        c[1] + sharp * 1.5 * (c[1] - avg[1]),
                        c[2] + sharp * 1.5 * (c[2] - avg[2]),
                    ];
                }

                if vig > 0.0 {
                    let uv = [
                        (x as f32 + 0.5) / width as f32,
                        (y as f32 + 0.5) / height as f32,
                    ];
                    let dist_norm = ((uv[0] - 0.5).hypot(uv[1] - 0.5)) / half_diag;
                    let mul = 1.0 - vig * smoothstep(0.4, 0.9, dist_norm);
                    c = [c[0] * mul, c[1] * mul, c[2] * mul];
                }

                let i = idx * 4;
                rgba[i] = quant(c[0]);
                rgba[i + 1] = quant(c[1]);
                rgba[i + 2] = quant(c[2]);
            }
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn quant(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}
