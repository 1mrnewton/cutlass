//! Draw a contact sheet of effect recipes: each recipe is a pair, its
//! preview scene as it is and then through the recipe at its poster
//! moment, so a change to a block can be judged by eye before and after.
//!
//! With no ids, the sheet holds every bundled recipe that uses one of the
//! blocks named in [`DEFAULT_BLOCKS`].
//!
//! ```sh
//! cargo run -p cutlass-render --release --example effect_sheet -- sheet.png
//! cargo run -p cutlass-render --release --example effect_sheet -- sheet.png glow soft-glow
//! ```

#![allow(clippy::print_stdout)]

use std::error::Error;
use std::path::PathBuf;

use cutlass_effects::{Block, Recipe, bundled, preview_scene};
use cutlass_encode::{CodecId, CodecParams, CpuFrame, Encoder, FrameInput, FrameTime, PixelFormat};
use cutlass_render::{Layer, Placement, Renderer, Rgba};

type Failure = Box<dyn Error>;

/// The blocks whose recipes the sheet holds when none are named.
const DEFAULT_BLOCKS: &[Block] = &[
    Block::Blur,
    Block::Glow,
    Block::Focus,
    Block::ZoomBlur,
    Block::MotionBlur,
];

/// Pixels between pictures.
const GAP: u32 = 4;
/// Recipes to a row of the sheet, each as a pair of pictures.
const PAIRS: usize = 3;

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: effect_sheet <out.png> [recipe id ...]")?;
    let ids: Vec<String> = args.collect();
    let recipes: Vec<&Recipe> = if ids.is_empty() {
        bundled()
            .iter()
            .filter(|recipe| {
                recipe
                    .passes
                    .iter()
                    .any(|pass| DEFAULT_BLOCKS.contains(&pass.block))
            })
            .collect()
    } else {
        ids.iter()
            .map(|id| {
                bundled()
                    .iter()
                    .find(|recipe| recipe.id == *id)
                    .ok_or_else(|| format!("no bundled recipe {id}"))
            })
            .collect::<Result<_, _>>()?
    };
    let renderer = Renderer::headless()?;
    let mut rows = Vec::with_capacity(recipes.len());
    for recipe in &recipes {
        let scene = preview_scene(recipe.preview_scene()).ok_or("a recipe names no scene")?;
        let (width, height) = (scene.width, scene.height);
        let canvas = renderer.canvas(width, height)?;
        let steps = recipe.steps_at(recipe.poster_time());
        let picture = Rgba {
            width,
            height,
            bytes: &scene.rgba,
        };
        renderer.draw(
            &canvas,
            &[
                Layer::Rgba(picture, Placement::canvas(width, height)),
                Layer::Effect(&steps, None),
            ],
        )?;
        rows.push((width, height, scene.rgba.clone(), renderer.read(&canvas)?));
        println!("{}", recipe.id);
    }
    // Every pair takes the largest scene's cell, PAIRS to a row.
    let cell_w = rows.iter().map(|(w, ..)| w * 2 + GAP).max().unwrap_or(1) + GAP * 2;
    let cell_h = rows.iter().map(|(_, h, ..)| *h).max().unwrap_or(1) + GAP;
    let lines = u32::try_from(rows.len().div_ceil(PAIRS))?;
    let width = cell_w * u32::try_from(PAIRS.min(rows.len().max(1)))?;
    let height = (cell_h * lines).max(1);
    let mut sheet = vec![32u8; (width * height * 4) as usize];
    for (index, (w, h, before, after)) in rows.iter().enumerate() {
        let column = u32::try_from(index % PAIRS)?;
        let line = u32::try_from(index / PAIRS)?;
        let (left, top) = (column * cell_w, line * cell_h);
        for (x, picture) in [(left, before), (left + w + GAP, after)] {
            for y in 0..*h {
                let from = (y * w * 4) as usize;
                let to = (((top + y) * width + x) * 4) as usize;
                sheet[to..to + (w * 4) as usize]
                    .copy_from_slice(&picture[from..from + (w * 4) as usize]);
            }
        }
    }
    std::fs::write(&out, png(width, height, sheet)?)?;
    println!(
        "{} recipes, {width}x{height}: {}",
        rows.len(),
        out.display()
    );
    Ok(())
}

fn png(width: u32, height: u32, rgba: Vec<u8>) -> Result<Vec<u8>, Failure> {
    let mut encoder =
        cutlass_png::PngEncoder::open(&CodecParams::video(CodecId::Png, width, height))?;
    let time = FrameTime::new(0, 1, 1)?;
    let frame = CpuFrame::new(width, height, PixelFormat::Rgba8, rgba, time)?;
    let packets = encoder.push_frame(&FrameInput::Cpu(frame))?;
    Ok(packets.into_iter().flat_map(|packet| packet.data).collect())
}
