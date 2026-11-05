//! Eval's frame layers as render layers, so every shell draws a frame the
//! same way.

use cutlass_eval::{Composed, FrameLayers, GpuSource, YuvSamples};

use crate::{Canvas, Layer, Planes, RenderError, Renderer, Rgba};

impl Renderer {
    /// Clear `canvas` and draw `frame`, which must be the canvas's size.
    pub fn draw_frame(&self, canvas: &Canvas, frame: &FrameLayers) -> Result<(), RenderError> {
        if canvas.size() != (frame.width, frame.height) {
            return Err(RenderError::Size {
                width: frame.width,
                height: frame.height,
            });
        }
        self.draw(canvas, &layers_of(frame)?)
    }
}

/// `composed` as layers: a frame eval already composed on the CPU becomes
/// one layer the size of the canvas.
pub fn frame_layers(composed: Composed) -> Result<FrameLayers, RenderError> {
    match composed {
        Composed::Layers(layers) => Ok(layers),
        Composed::Frame(frame) => {
            let (width, height) = (frame.extent.width, frame.extent.height);
            FrameLayers::pixels(frame).ok_or(RenderError::Size { width, height })
        }
    }
}

/// The render layers for `frame`, a compound's child frame included.
fn layers_of(frame: &FrameLayers) -> Result<Vec<Layer<'_>>, RenderError> {
    let mut layers = Vec::with_capacity(frame.layers.len());
    for layer in &frame.layers {
        layers.push(match &layer.source {
            GpuSource::Yuv {
                width,
                height,
                samples,
                tone,
            } => {
                let planes = match samples {
                    YuvSamples::Bytes(bytes) => Planes::packed(*width, *height, bytes)?,
                    YuvSamples::Surface(surface) => Planes::surface(surface, *width, *height),
                }
                .with_tone(*tone);
                match layer.backdrop {
                    Some(backdrop) => Layer::YuvOverBlur {
                        planes,
                        placement: layer.placement,
                        cover: backdrop.cover,
                        radius: backdrop.radius,
                    },
                    None => Layer::Yuv(planes, layer.placement),
                }
            }
            GpuSource::Rgba(picture) => Layer::Rgba(
                Rgba {
                    width: picture.width(),
                    height: picture.height(),
                    bytes: picture.rgba(),
                },
                layer.placement,
            ),
            GpuSource::Nested(child) => Layer::Nested {
                layers: layers_of(child)?,
                size: (child.width, child.height),
                placement: layer.placement,
            },
            GpuSource::Effect(steps) => Layer::Effect(steps),
            GpuSource::Vector(scene) => Layer::Vector {
                scene,
                placement: layer.placement,
            },
            GpuSource::Blend { mode, layers } => {
                if (layers.width, layers.height) != (frame.width, frame.height) {
                    return Err(RenderError::Size {
                        width: layers.width,
                        height: layers.height,
                    });
                }
                Layer::Blend {
                    mode: *mode,
                    layers: layers_of(layers)?,
                }
            }
            GpuSource::Transition {
                kind,
                t,
                outgoing,
                incoming,
            } => Layer::Transition {
                kind: *kind,
                t: *t,
                outgoing: Rgba {
                    width: outgoing.width(),
                    height: outgoing.height(),
                    bytes: outgoing.rgba(),
                },
                incoming: Rgba {
                    width: incoming.width(),
                    height: incoming.height(),
                    bytes: incoming.rgba(),
                },
            },
        });
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use cutlass_compositor::Frame;
    use cutlass_eval::{Composed, FrameLayers, GpuLayer, GpuSource};
    use cutlass_gpu::{Extent, Placement};
    use cutlass_picture::CpuBytes;

    use super::frame_layers;
    use crate::{RenderError, Renderer};

    fn renderer() -> Option<Renderer> {
        let renderer = Renderer::for_test().expect("a GPU is required");
        if renderer.is_none() {
            eprintln!("skipped: no GPU");
        }
        renderer
    }

    /// Four opaque pixels a row, in two rows: red, green, blue, white, then
    /// the same backwards.
    fn pixels() -> Vec<u8> {
        let row = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 255, 255],
        ];
        let mut bytes: Vec<u8> = row.iter().flatten().copied().collect();
        bytes.extend(row.iter().rev().flatten());
        bytes
    }

    fn picture_layer(width: u32, height: u32, rgba: Vec<u8>) -> GpuLayer {
        GpuLayer {
            source: GpuSource::Rgba(CpuBytes::new(width, height, rgba).expect("bytes")),
            placement: Placement::canvas(width, height),
            backdrop: None,
        }
    }

    #[test]
    fn a_frame_of_layers_draws_pixel_for_pixel() {
        let Some(renderer) = renderer() else {
            return;
        };
        let frame = FrameLayers {
            width: 4,
            height: 2,
            layers: vec![picture_layer(4, 2, pixels())],
        };
        let canvas = renderer.canvas(4, 2).expect("canvas");
        renderer.draw_frame(&canvas, &frame).expect("draw");
        assert_eq!(renderer.read(&canvas).expect("read"), pixels());
    }

    #[test]
    fn a_compound_child_draws_on_its_own_canvas_first() {
        let Some(renderer) = renderer() else {
            return;
        };
        let child = FrameLayers {
            width: 4,
            height: 2,
            layers: vec![picture_layer(4, 2, pixels())],
        };
        let frame = FrameLayers {
            width: 4,
            height: 2,
            layers: vec![GpuLayer {
                source: GpuSource::Nested(Box::new(child)),
                placement: Placement::canvas(4, 2),
                backdrop: None,
            }],
        };
        let canvas = renderer.canvas(4, 2).expect("canvas");
        renderer.draw_frame(&canvas, &frame).expect("draw");
        assert_eq!(renderer.read(&canvas).expect("read"), pixels());
    }

    #[test]
    fn an_effect_layer_runs_over_the_layers_before_it() {
        let Some(renderer) = renderer() else {
            return;
        };
        let invert = cutlass_effects::Step {
            block: cutlass_effects::Block::Invert,
            values: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let frame = FrameLayers {
            width: 4,
            height: 2,
            layers: vec![
                picture_layer(4, 2, pixels()),
                GpuLayer {
                    source: GpuSource::Effect(vec![invert]),
                    placement: Placement::canvas(4, 2),
                    backdrop: None,
                },
            ],
        };
        let canvas = renderer.canvas(4, 2).expect("canvas");
        renderer.draw_frame(&canvas, &frame).expect("draw");
        let expected: Vec<u8> = pixels()
            .chunks(4)
            .flat_map(|pixel| [255 - pixel[0], 255 - pixel[1], 255 - pixel[2], pixel[3]])
            .collect();
        assert_eq!(renderer.read(&canvas).expect("read"), expected);
    }

    #[test]
    fn a_cpu_frame_becomes_one_layer_that_draws_unchanged() {
        let composed = Composed::Frame(Frame {
            extent: Extent {
                width: 4,
                height: 2,
            },
            rgba: pixels(),
        });
        let frame = frame_layers(composed).expect("layers");
        assert_eq!((frame.width, frame.height, frame.layers.len()), (4, 2, 1));
        let Some(renderer) = renderer() else {
            return;
        };
        let canvas = renderer.canvas(4, 2).expect("canvas");
        renderer.draw_frame(&canvas, &frame).expect("draw");
        assert_eq!(renderer.read(&canvas).expect("read"), pixels());
    }

    #[test]
    fn a_cpu_frame_with_the_wrong_byte_count_is_refused() {
        let composed = Composed::Frame(Frame {
            extent: Extent {
                width: 4,
                height: 2,
            },
            rgba: vec![0; 5],
        });
        assert!(matches!(
            frame_layers(composed),
            Err(RenderError::Size {
                width: 4,
                height: 2
            })
        ));
    }

    #[test]
    fn a_frame_and_a_canvas_of_different_sizes_draw_nothing() {
        let Some(renderer) = renderer() else {
            return;
        };
        let canvas = renderer.canvas(4, 2).expect("canvas");
        let blank = renderer.read(&canvas).expect("read");
        let frame = FrameLayers {
            width: 2,
            height: 4,
            layers: vec![picture_layer(2, 4, pixels())],
        };
        assert!(matches!(
            renderer.draw_frame(&canvas, &frame),
            Err(RenderError::Size {
                width: 2,
                height: 4
            })
        ));
        assert_eq!(renderer.read(&canvas).expect("read"), blank);
    }
}
