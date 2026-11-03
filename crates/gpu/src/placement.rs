//! Where a source lands on the canvas, as maps a GPU or CPU pass can apply.
//!
//! A canvas pixel center goes through three maps. Each result is floored to
//! a whole pixel and moved to that pixel's center before the next: canvas to
//! the placed picture, picture to the oriented source (a crop), and oriented
//! source to the stored picture (a quarter turn). A point outside the
//! picture or the oriented source is transparent.

/// `(x, y)` to `(a*x + b*y + c, d*x + e*y + f)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f64; 6]);

impl Affine {
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    #[must_use]
    pub fn apply(self, x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d, e, f] = self.0;
        (a * x + b * y + c, d * x + e * y + f)
    }
}

/// Where one source lands on the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// Canvas pixel center to placed-picture pixels.
    pub to_picture: Affine,
    pub picture: (u32, u32),
    /// Placed-picture pixel center to oriented-source pixels.
    pub to_oriented: Affine,
    pub oriented: (u32, u32),
    /// Oriented-source pixel center to texture pixels.
    pub to_texture: Affine,
    /// Multiplies the premultiplied color, 0 to 1.
    pub opacity: f64,
}

impl Placement {
    /// A source the size of the canvas, pixel for pixel.
    #[must_use]
    pub const fn canvas(width: u32, height: u32) -> Self {
        Self {
            to_picture: Affine::IDENTITY,
            picture: (width, height),
            to_oriented: Affine::IDENTITY,
            oriented: (width, height),
            to_texture: Affine::IDENTITY,
            opacity: 1.0,
        }
    }
}
