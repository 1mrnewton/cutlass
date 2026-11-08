//! Write every filter's poster as a binary PPM named by its id, to look
//! the filters over side by side:
//!
//! ```sh
//! cargo run -p cutlass-effects --release --example filter_posters -- <folder>
//! ```

use std::io::Write;

fn main() -> std::io::Result<()> {
    let out = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| std::io::Error::other("usage: filter_posters <output folder>"))?;
    std::fs::create_dir_all(&out)?;
    let (width, height) = (144, 96);
    for look in cutlass_effects::looks() {
        let rgba = cutlass_effects::look_poster(look, width, height);
        let mut file = std::fs::File::create(out.join(format!("{}.ppm", look.id)))?;
        write!(file, "P6\n{width} {height}\n255\n")?;
        let rgb: Vec<u8> = rgba
            .chunks(4)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        file.write_all(&rgb)?;
    }
    Ok(())
}
