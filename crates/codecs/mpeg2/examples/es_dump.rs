//! Lists the access units of an elementary stream.

use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(input) = args.get(1) else {
        return Err("usage: es_dump in.m2v".into());
    };
    let stream = std::fs::read(input)?;
    let mut out = std::io::stdout().lock();
    for unit in cutlass_mpeg2::es::split(&stream)? {
        writeln!(
            out,
            "{} type {} display {} bytes {}",
            unit.decode_index,
            unit.coding_type,
            unit.display_index,
            unit.data.len()
        )?;
    }
    Ok(())
}
