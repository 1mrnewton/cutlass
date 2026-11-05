//! Write every sticker's file into the folder named on the command line.

fn main() {
    let Some(folder) = std::env::args_os().nth(1).map(std::path::PathBuf::from) else {
        return;
    };
    for sticker in cutlass_stickers::STICKERS {
        if let Ok(bytes) = sticker.bytes() {
            let _ = std::fs::write(folder.join(sticker.file_name()), bytes);
        }
    }
}
