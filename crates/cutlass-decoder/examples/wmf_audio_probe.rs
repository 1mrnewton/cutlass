//! Smoke test for the Windows audio reader: open, stream, seek, re-stream.
//!
//! Usage: `cargo run --release -p cutlass-decoder --example wmf_audio_probe -- <file> [rate] [channels]`

#[cfg(target_os = "windows")]
fn main() {
    use std::path::Path;
    use std::time::Instant;

    use cutlass_core::AudioReader;

    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: wmf_audio_probe <file> [rate] [channels]");
    let rate: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(48_000);
    let channels: u16 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let path = Path::new(&path);

    println!("file: {}", path.display());
    match cutlass_decoder::probe(path) {
        Ok(p) => println!(
            "probe: {}x{} {} frames @ {:.3} fps, has_audio={}",
            p.width,
            p.height,
            p.frame_count,
            p.frame_rate.as_f64(),
            p.has_audio
        ),
        Err(e) => println!("probe failed: {e}"),
    }

    let start = Instant::now();
    let mut reader = match cutlass_decoder::WmfAudioReader::open(path, rate, channels) {
        Ok(r) => r,
        Err(e) => {
            println!("audio open failed: {e}");
            return;
        }
    };
    println!(
        "opened at {rate} Hz x{channels} in {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    let describe = |label: &str, reader: &mut cutlass_decoder::WmfAudioReader| {
        let mut buf = vec![0f32; rate as usize * channels as usize]; // 1 second
        let pos_before = reader.position();
        let start = Instant::now();
        match reader.read(&mut buf) {
            Ok(frames) => {
                let ms = start.elapsed().as_secs_f64() * 1e3;
                let n = frames * channels as usize;
                let rms = if n > 0 {
                    (buf[..n]
                        .iter()
                        .map(|s| f64::from(*s) * f64::from(*s))
                        .sum::<f64>()
                        / n as f64)
                        .sqrt()
                } else {
                    0.0
                };
                let peak = buf[..n].iter().fold(0f32, |m, s| m.max(s.abs()));
                println!(
                    "{label}: {frames} frames in {ms:.1} ms (pos {pos_before:?} -> {:?}), rms {rms:.4}, peak {peak:.4}",
                    reader.position()
                );
            }
            Err(e) => println!("{label}: read failed: {e}"),
        }
    };

    describe("read #1 (t=0s)", &mut reader);
    describe("read #2 (t=1s)", &mut reader);

    // Long backward-ish seek then a short forward hop.
    match reader.seek_to_frame(rate as i64 / 2) {
        Ok(()) => describe("read after seek to 0.5s", &mut reader),
        Err(e) => println!("seek failed: {e}"),
    }
    match reader.seek_to_frame(rate as i64 * 3) {
        Ok(()) => describe("read after seek to 3.0s", &mut reader),
        Err(e) => println!("seek failed: {e}"),
    }

    // Waveform peaks end to end (the strips worker path).
    let start = Instant::now();
    match cutlass_decoder::audio_peaks(path, 200) {
        Ok(peaks) => println!(
            "audio_peaks: {} buckets in {:.1} ms, max {:.4}",
            peaks.len(),
            start.elapsed().as_secs_f64() * 1e3,
            peaks.iter().fold(0f32, |m, p| m.max(*p)),
        ),
        Err(e) => println!("audio_peaks failed: {e}"),
    }
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("wmf_audio_probe is Windows-only");
}
