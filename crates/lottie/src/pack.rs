//! A scene packed for [`crate::WGSL`]: one list of RGBA32F texels and the
//! draws that read it.
//!
//! Texel 0 is the output width and height. A fill's record is seven texels:
//! its pixel bounds; edge start, edge count, even-odd, paint kind (0 solid,
//! 1 linear, 2 radial); the solid color, or stop start, stop count, opacity
//! and radius; two rows of the map to gradient space; start and end; focal
//! point. A group's record is three: the whole output; mask start, mask
//! count, matte mode and opacity; the mask's starting value. Each mask is
//! two texels: edge start, edge count, mode, inverted; opacity, expansion
//! edge start, expansion edge count, grow. An edge is one texel, a stop
//! two. Indices stay below 2^24, so they are exact as `f32`.

use crate::scene::{Edge, Fill, FillRule, GradientKind, Group, Item, Paint, Scene};

/// Texels in one row of the data texture.
pub const DATA_WIDTH: u32 = 2_048;

/// One draw, in the order its list draws them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedItem {
    /// Instance `slot` of the fill pipeline.
    Fill { slot: u32 },
    /// Draw `items` on a canvas of their own, `matte` on another, then
    /// instance `slot` of the group pipeline lands them.
    Group {
        slot: u32,
        items: Vec<PackedItem>,
        matte: Option<Vec<PackedItem>>,
    },
}

/// What the GPU reads for one scene.
#[derive(Clone, Debug, PartialEq)]
pub struct Packed {
    /// The data texture, [`DATA_WIDTH`] texels a row, the last row padded.
    pub texels: Vec<[f32; 4]>,
    /// The first texel of each slot's record: the instance attribute.
    pub records: Vec<u32>,
    pub items: Vec<PackedItem>,
}

impl Packed {
    /// Rows of the data texture.
    #[must_use]
    pub fn rows(&self) -> u32 {
        (self.texels.len() as u32).div_ceil(DATA_WIDTH)
    }
}

/// Pack `scene`. Slots within one list are consecutive, so a run of fills
/// is one instanced draw.
#[must_use]
pub fn pack(scene: &Scene) -> Packed {
    let mut packed = Packed {
        texels: vec![[scene.width as f32, scene.height as f32, 0.0, 0.0]],
        records: Vec::new(),
        items: Vec::new(),
    };
    let whole = [0.0, 0.0, scene.width as f32, scene.height as f32];
    packed.items = pack_list(&mut packed, &scene.items, whole);
    let padded = packed.texels.len().div_ceil(DATA_WIDTH as usize) * DATA_WIDTH as usize;
    packed.texels.resize(padded, [0.0; 4]);
    packed
}

fn pack_list(packed: &mut Packed, items: &[Item], whole: [f32; 4]) -> Vec<PackedItem> {
    let first = packed.records.len() as u32;
    packed.records.resize(packed.records.len() + items.len(), 0);
    let mut out = Vec::with_capacity(items.len());
    for (offset, item) in items.iter().enumerate() {
        let slot = first + offset as u32;
        match item {
            Item::Fill(fill) => {
                packed.records[slot as usize] = pack_fill(packed, fill);
                out.push(PackedItem::Fill { slot });
            }
            Item::Group(group) => {
                packed.records[slot as usize] = pack_group(packed, group, whole);
                let items = pack_list(packed, &group.items, whole);
                let matte = group
                    .matte
                    .as_ref()
                    .map(|matte| pack_list(packed, &matte.items, whole));
                out.push(PackedItem::Group { slot, items, matte });
            }
        }
    }
    out
}

fn push_edges(packed: &mut Packed, edges: &[Edge]) -> (f32, f32) {
    let start = packed.texels.len() as f32;
    packed.texels.extend(
        edges
            .iter()
            .map(|edge| [edge.x0, edge.y0, edge.x1, edge.y1]),
    );
    (start, edges.len() as f32)
}

fn pack_fill(packed: &mut Packed, fill: &Fill) -> u32 {
    let (edge_start, edge_count) = push_edges(packed, &fill.edges);
    let even_odd = if fill.rule == FillRule::EvenOdd {
        1.0
    } else {
        0.0
    };
    let bounds = [
        fill.bounds.x0 as f32,
        fill.bounds.y0 as f32,
        fill.bounds.x1 as f32,
        fill.bounds.y1 as f32,
    ];
    let record = match &fill.paint {
        Paint::Solid(color) => [
            bounds,
            [edge_start, edge_count, even_odd, 0.0],
            *color,
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
        ],
        Paint::Gradient(gradient) => {
            let stop_start = packed.texels.len() as f32;
            for stop in &gradient.stops {
                let [r, g, b, a] = stop.color;
                packed.texels.push([stop.offset, r, g, b]);
                packed.texels.push([a, 0.0, 0.0, 0.0]);
            }
            let kind = match gradient.kind {
                GradientKind::Linear => 1.0,
                GradientKind::Radial => 2.0,
            };
            let [a, b, c, d, e, f] = gradient.to_gradient;
            [
                bounds,
                [edge_start, edge_count, even_odd, kind],
                [
                    stop_start,
                    gradient.stops.len() as f32,
                    gradient.opacity,
                    gradient.radius,
                ],
                [a, b, c, 0.0],
                [d, e, f, 0.0],
                [
                    gradient.start[0],
                    gradient.start[1],
                    gradient.end[0],
                    gradient.end[1],
                ],
                [gradient.focal[0], gradient.focal[1], 0.0, 0.0],
            ]
        }
    };
    let at = packed.texels.len() as u32;
    packed.texels.extend(record);
    at
}

fn pack_group(packed: &mut Packed, group: &Group, whole: [f32; 4]) -> u32 {
    let mut masks = Vec::with_capacity(group.masks.len() * 2);
    for mask in &group.masks {
        let (start, count) = push_edges(packed, &mask.edges);
        let (grown_start, grown_count, grow) = match &mask.expansion {
            Some(expansion) => {
                let (start, count) = push_edges(packed, &expansion.edges);
                (start, count, if expansion.grow { 1.0 } else { 0.0 })
            }
            None => (0.0, 0.0, 0.0),
        };
        masks.push([
            start,
            count,
            mask.mode.code() as f32,
            if mask.inverted { 1.0 } else { 0.0 },
        ]);
        masks.push([mask.opacity, grown_start, grown_count, grow]);
    }
    let mask_start = packed.texels.len() as f32;
    packed.texels.extend(masks);
    let matte = group
        .matte
        .as_ref()
        .map_or(0.0, |matte| matte.mode.code() as f32);
    let at = packed.texels.len() as u32;
    packed.texels.extend([
        whole,
        [mask_start, group.masks.len() as f32, matte, group.opacity],
        [group.mask_start(), 0.0, 0.0, 0.0],
    ]);
    at
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bounds, Mask, MaskMode};

    fn fill() -> Item {
        let edges = vec![
            Edge {
                x0: 3.0,
                y0: 0.0,
                x1: 3.0,
                y1: 4.0,
            },
            Edge {
                x0: 0.0,
                y0: 4.0,
                x1: 0.0,
                y1: 0.0,
            },
        ];
        Item::Fill(Fill {
            bounds: Bounds::of(&edges, 4, 4).unwrap(),
            edges,
            rule: FillRule::EvenOdd,
            paint: Paint::Solid([0.5, 0.25, 1.0, 0.75]),
        })
    }

    #[test]
    fn a_fill_record_names_its_edges_and_color() {
        let packed = pack(&Scene {
            width: 4,
            height: 4,
            items: vec![fill(), fill()],
        });
        assert_eq!(packed.texels[0], [4.0, 4.0, 0.0, 0.0]);
        assert_eq!(
            packed.items,
            vec![PackedItem::Fill { slot: 0 }, PackedItem::Fill { slot: 1 }]
        );
        let record = packed.records[0] as usize;
        assert_eq!(packed.texels[record], [0.0, 0.0, 3.0, 4.0]);
        assert_eq!(packed.texels[record + 1], [1.0, 2.0, 1.0, 0.0]);
        assert_eq!(packed.texels[record + 2], [0.5, 0.25, 1.0, 0.75]);
        assert_eq!(packed.texels[1], [3.0, 0.0, 3.0, 4.0]);
        assert_eq!(packed.texels.len() % DATA_WIDTH as usize, 0);
        assert_eq!(packed.rows(), 1);
    }

    #[test]
    fn a_group_packs_its_masks_then_its_children_after_it() {
        let packed = pack(&Scene {
            width: 4,
            height: 4,
            items: vec![
                fill(),
                Item::Group(Group {
                    opacity: 0.5,
                    masks: vec![Mask {
                        mode: MaskMode::Subtract,
                        inverted: true,
                        opacity: 0.25,
                        edges: Vec::new(),
                        expansion: None,
                    }],
                    matte: None,
                    items: vec![fill()],
                }),
            ],
        });
        let PackedItem::Group { slot, items, matte } = &packed.items[1] else {
            panic!("{:?}", packed.items);
        };
        assert_eq!(*slot, 1);
        assert_eq!(items, &vec![PackedItem::Fill { slot: 2 }]);
        assert!(matte.is_none());
        let record = packed.records[1] as usize;
        let head = packed.texels[record + 1];
        assert_eq!(head[1..], [1.0, 0.0, 0.5]);
        assert_eq!(packed.texels[record + 2][0], 1.0);
        let mask = head[0] as usize;
        assert_eq!(packed.texels[mask][2..], [1.0, 1.0]);
        assert_eq!(packed.texels[mask + 1][0], 0.25);
    }
}
