//! One evaluated frame: flat geometry in output pixels, in draw order.
//!
//! Every path is already transformed and cut into line edges. A fill
//! covers what its edges enclose under its rule; a stroke arrives as the
//! fill of its outline. A group draws its items on a clear canvas of its
//! own, then lands on what is below it times its opacity, its masks and its
//! matte. Colors are straight, not premultiplied, from 0 to 1.

/// One line edge, from `(x0, y0)` to `(x1, y1)`, in output pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientKind {
    Linear,
    Radial,
}

/// One gradient stop: where, and its straight color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stop {
    pub offset: f32,
    pub color: [f32; 4],
}

/// A gradient in its own space; `to_gradient` maps an output pixel there.
///
/// Linear runs from `start` to `end`. Radial grows from `focal` to the
/// circle of `radius` round `start`. Past either end the end color holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub kind: GradientKind,
    /// `x' = a x + b y + c`, `y' = d x + e y + f`.
    pub to_gradient: [f32; 6],
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub focal: [f32; 2],
    pub radius: f32,
    pub stops: Vec<Stop>,
    /// Multiplies every stop's alpha.
    pub opacity: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Paint {
    /// A straight color, its opacity already in its alpha.
    Solid([f32; 4]),
    Gradient(Gradient),
}

/// A pixel rectangle, `x0..x1` by `y0..y1`, inside the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Bounds {
    /// The pixels `edges` may touch, inside a `width` by `height` output.
    /// `None` when that is nothing.
    #[must_use]
    pub fn of(edges: &[Edge], width: u32, height: u32) -> Option<Self> {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for edge in edges {
            x0 = x0.min(edge.x0).min(edge.x1);
            y0 = y0.min(edge.y0).min(edge.y1);
            x1 = x1.max(edge.x0).max(edge.x1);
            y1 = y1.max(edge.y0).max(edge.y1);
        }
        if !(x0 <= x1 && y0 <= y1) {
            return None;
        }
        let clamp = |value: f32, limit: u32| value.clamp(0.0, limit as f32) as u32;
        let bounds = Self {
            x0: clamp(x0.floor(), width),
            y0: clamp(y0.floor(), height),
            x1: clamp(x1.ceil(), width),
            y1: clamp(y1.ceil(), height),
        };
        (bounds.x0 < bounds.x1 && bounds.y0 < bounds.y1).then_some(bounds)
    }

    #[must_use]
    pub fn area(self) -> u64 {
        u64::from(self.x1 - self.x0) * u64::from(self.y1 - self.y0)
    }
}

/// A filled outline.
#[derive(Clone, Debug, PartialEq)]
pub struct Fill {
    pub edges: Vec<Edge>,
    pub rule: FillRule,
    pub paint: Paint,
    pub bounds: Bounds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskMode {
    Add,
    Subtract,
    Intersect,
    Lighten,
    Darken,
    Difference,
}

impl MaskMode {
    /// The number the GPU reads.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Subtract => 1,
            Self::Intersect => 2,
            Self::Lighten => 3,
            Self::Darken => 4,
            Self::Difference => 5,
        }
    }

    /// The mask so far after `coverage` joins it by this mode.
    #[must_use]
    pub fn combine(self, mask: f32, coverage: f32) -> f32 {
        match self {
            Self::Add => mask + coverage - mask * coverage,
            Self::Subtract => mask * (1.0 - coverage),
            Self::Intersect => mask * coverage,
            Self::Lighten => mask.max(coverage),
            Self::Darken => mask.min(coverage),
            Self::Difference => (mask - coverage).abs(),
        }
    }
}

/// Mask expansion: the mask's outline stroked twice the expansion wide,
/// added to the mask when it grows and taken away when it shrinks.
#[derive(Clone, Debug, PartialEq)]
pub struct Expansion {
    pub grow: bool,
    pub edges: Vec<Edge>,
}

/// A layer mask, always a non-zero fill.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub mode: MaskMode,
    pub inverted: bool,
    pub opacity: f32,
    pub edges: Vec<Edge>,
    pub expansion: Option<Expansion>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatteMode {
    Alpha,
    AlphaInverted,
    Luma,
    LumaInverted,
}

impl MatteMode {
    /// The number the GPU reads; 0 is no matte.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Alpha => 1,
            Self::AlphaInverted => 2,
            Self::Luma => 3,
            Self::LumaInverted => 4,
        }
    }

    /// How much of the layer shows over a premultiplied matte pixel.
    #[must_use]
    pub fn factor(self, matte: [f32; 4]) -> f32 {
        let luma = 0.2125 * matte[0] + 0.7154 * matte[1] + 0.0721 * matte[2];
        match self {
            Self::Alpha => matte[3],
            Self::AlphaInverted => 1.0 - matte[3],
            Self::Luma => luma,
            Self::LumaInverted => 1.0 - luma,
        }
    }
}

/// The layer a matte reads, drawn on its own clear canvas.
#[derive(Clone, Debug, PartialEq)]
pub struct Matte {
    pub mode: MatteMode,
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub opacity: f32,
    pub masks: Vec<Mask>,
    pub matte: Option<Matte>,
    pub items: Vec<Item>,
}

impl Group {
    /// The mask before the first one joins: full when the first one takes
    /// away or intersects, empty otherwise.
    #[must_use]
    pub fn mask_start(&self) -> f32 {
        match self.masks.first().map(|mask| mask.mode) {
            Some(MaskMode::Subtract | MaskMode::Intersect) => 1.0,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Fill(Fill),
    Group(Group),
}

/// One frame, bottom item first.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub items: Vec<Item>,
}

impl Scene {
    /// Pixels times edges the GPU tests for this frame: what each fill's
    /// box costs, and what each group's masks cost over the whole output.
    #[must_use]
    pub fn gpu_cost(&self) -> u64 {
        fn cost(items: &[Item], area: u64) -> u64 {
            items.iter().fold(0_u64, |sum, item| {
                let own = match item {
                    Item::Fill(fill) => fill.bounds.area().saturating_mul(fill.edges.len() as u64),
                    Item::Group(group) => {
                        let masks: u64 = group
                            .masks
                            .iter()
                            .map(|mask| {
                                let grown = mask.expansion.as_ref().map_or(0, |e| e.edges.len());
                                (mask.edges.len() + grown) as u64
                            })
                            .sum();
                        let matte = group
                            .matte
                            .as_ref()
                            .map_or(0, |matte| cost(&matte.items, area));
                        area.saturating_mul(masks.max(1))
                            .saturating_add(cost(&group.items, area))
                            .saturating_add(matte)
                    }
                };
                sum.saturating_add(own)
            })
        }
        let area = u64::from(self.width) * u64::from(self.height);
        cost(&self.items, area)
    }

    /// Every edge in the frame, fills, masks and mattes counted.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        fn count(items: &[Item]) -> usize {
            items
                .iter()
                .map(|item| match item {
                    Item::Fill(fill) => fill.edges.len(),
                    Item::Group(group) => {
                        let masks: usize = group
                            .masks
                            .iter()
                            .map(|mask| {
                                mask.edges.len()
                                    + mask.expansion.as_ref().map_or(0, |e| e.edges.len())
                            })
                            .sum();
                        masks
                            + count(&group.items)
                            + group.matte.as_ref().map_or(0, |matte| count(&matte.items))
                    }
                })
                .sum()
        }
        count(&self.items)
    }
}
