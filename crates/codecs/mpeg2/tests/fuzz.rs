//! Bounded random campaigns over the decoder and the encoder.
//!
//! Each runs a few hundred seeded iterations under `cargo test`. Set
//! `MPEG2_FUZZ_SECS` or `MPEG2_ENCODE_FUZZ_SECS` for a timed campaign, and
//! `MPEG2_FUZZ_SEED` for another seed; the summary goes to stdout.
//!
//! Decoder: mutated packets injected into real streams (a field stream
//! among them, also cut one packet per field), mutated streams
//! split at random boundaries, mutated configuration, and random reset and
//! drain. A panic fails the run. A refused packet leaves the decoder as it
//! was, so the stream around it decodes to the clean pictures. No picture
//! is larger than the limit, no push returns more than a few, and
//! splitting clean bytes in any chunks gives the units splitting them
//! whole does.
//!
//! Encoder: random sizes, rates, bit rates, pictures, timestamps, forced
//! keyframes, drain, reset, and cancel. `open` accepts exactly the
//! parameters Main profile holds. Every drained run decodes to the
//! accepted pictures, in order. At most four pictures are inside the
//! encoder. A replay without the refused pushes writes the same bytes.

#[cfg(test)]
mod fuzz {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoOutput, Yuv420p};
    use cutlass_encode::{VideoEncoder, VideoInput};
    use cutlass_mpeg2::es::{AccessUnit, Splitter, split};
    use cutlass_mpeg2::{Mpeg2Decoder, Mpeg2Encoder};

    /// Luma samples a fuzzed picture may have: well above every corpus
    /// stream, well below what a hostile header asks for.
    const MAX_PIXELS: u64 = 1 << 20;
    const SMOKE_DECODE: u64 = 400;
    const SMOKE_ENCODE: u64 = 60;
    /// Slowest single iteration allowed: a hang, not a slow machine.
    const SLOWEST: Duration = Duration::from_secs(20);

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.next() % 100 < percent
        }
    }

    /// How long to run, and the seed: a timed campaign from the
    /// environment, else the smoke count.
    fn budget(secs: &str, smoke: u64, salt: u64) -> (Option<Instant>, u64, u64) {
        let seconds = std::env::var(secs)
            .ok()
            .and_then(|value| value.parse::<u64>().ok());
        let seed = std::env::var("MPEG2_FUZZ_SEED")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0x6D70_6567_3200_0000)
            ^ salt;
        match seconds {
            Some(seconds) if seconds > 0 => (
                Some(Instant::now() + Duration::from_secs(seconds)),
                u64::MAX,
                seed | 1,
            ),
            _ => (None, smoke, seed | 1),
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    fn packet(data: Vec<u8>, pts: i64) -> Packet {
        Packet {
            stream: 0,
            data,
            pts,
            duration: 1,
            timescale: 25,
            keyframe: false,
            side: Vec::new(),
        }
    }

    fn digest(frame: &VideoOutput) -> u64 {
        let mut hasher = DefaultHasher::new();
        frame.pts.hash(&mut hasher);
        frame.picture.width().hash(&mut hasher);
        frame.picture.height().hash(&mut hasher);
        frame.picture.y().hash(&mut hasher);
        frame.picture.u().hash(&mut hasher);
        frame.picture.v().hash(&mut hasher);
        hasher.finish()
    }

    /// Checks the bounds on one push and keeps its pictures.
    fn take(result: Result<Vec<VideoOutput>, CodecError>, out: &mut Vec<u64>) -> bool {
        match result {
            Ok(frames) => {
                assert!(frames.len() <= 4, "{} pictures from one push", frames.len());
                for frame in &frames {
                    let (w, h) = (frame.picture.width(), frame.picture.height());
                    assert!(u64::from(w) * u64::from(h) <= MAX_PIXELS, "{w}x{h}");
                    out.push(digest(frame));
                }
                true
            }
            Err(_) => false,
        }
    }

    fn picture(width: u32, height: u32, kind: usize, t: u32, rng: &mut Rng) -> Yuv420p {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut plane = |pw: usize, ph: usize, base: u32| -> Vec<u8> {
            (0..pw * ph)
                .map(|i| {
                    let (x, y) = ((i % pw) as u32, (i / pw) as u32);
                    match kind {
                        0 => rng.next() as u8,
                        1 => base as u8,
                        2 => ((x * 7 + y * 3 + t * 5 + base) % 256) as u8,
                        _ => {
                            if (x / 2 + y / 2 + t).is_multiple_of(2) {
                                0
                            } else {
                                255
                            }
                        }
                    }
                })
                .collect()
        };
        let y = plane(w, h, 16 + t);
        let u = plane(cw, ch, 128);
        let v = plane(cw, ch, 200);
        Yuv420p::try_new(width, height, y, u, v).unwrap()
    }

    /// Our own streams beside the FFmpeg fixtures: VBR, CBR, and a padded size.
    fn encoded(width: u32, height: u32, bitrate: u32, constant: bool, rng: &mut Rng) -> Vec<u8> {
        let mut params = CodecParams::video(CodecId::Mpeg2, width, height);
        params.bitrate = bitrate;
        params.constant_rate = constant;
        let mut encoder = Mpeg2Encoder::open(&params).unwrap();
        let mut stream = Vec::new();
        for t in 0..14 {
            let input = VideoInput {
                picture: picture(width, height, 2 + (t as usize % 2), t, rng),
                pts: i64::from(t),
                duration: 1,
                timescale: 25,
                force_keyframe: false,
            };
            for packet in encoder.push(&input).unwrap() {
                stream.extend(packet.data);
            }
        }
        for packet in encoder.drain().unwrap() {
            stream.extend(packet.data);
        }
        stream
    }

    fn mutate(data: &mut Vec<u8>, rng: &mut Rng) {
        const CODES: [u8; 9] = [0x00, 0x01, 0x2C, 0xAF, 0xB2, 0xB3, 0xB5, 0xB7, 0xB8];
        for _ in 0..1 + rng.below(4) {
            if data.is_empty() {
                data.push(0);
            }
            let at = rng.below(data.len());
            match rng.below(7) {
                0 => data[at] ^= 1 << rng.below(8),
                1 => data[at] = [0x00, 0xFF, rng.next() as u8][rng.below(3)],
                2 => data.truncate(at.max(1)),
                3 => {
                    let code = CODES[rng.below(CODES.len())];
                    data.splice(at..at, [0, 0, 1, code]);
                }
                4 => {
                    let len = rng.below(64).min(data.len() - at);
                    let piece = data[at..at + len].to_vec();
                    let to = rng.below(data.len());
                    data.splice(to..to, piece);
                }
                5 => {
                    let end = (at + 1 + rng.below(16)).min(data.len());
                    data[at..end].fill(0);
                }
                _ => {
                    let end = (at + 1 + rng.below(8)).min(data.len());
                    for byte in &mut data[at..end] {
                        *byte = rng.next() as u8;
                    }
                }
            }
        }
        if data.is_empty() {
            data.push(0xB3);
        }
    }

    /// Bytes before the first GOP or picture start code.
    fn sequence_bytes(unit: &[u8]) -> usize {
        (0..unit.len().saturating_sub(3))
            .find(|&at| unit[at..at + 3] == [0, 0, 1] && matches!(unit[at + 3], 0x00 | 0xB8))
            .unwrap_or(unit.len())
    }

    struct Stream {
        bytes: Vec<u8>,
        units: Vec<AccessUnit>,
        clean: Vec<u64>,
        /// The units are single fields, not the splitter's pairs.
        fields_apart: bool,
    }

    /// `units` with each field pair cut into one packet per field, as a
    /// container that stores fields apart delivers them.
    fn fields_apart(units: &[AccessUnit]) -> Vec<AccessUnit> {
        let mut out = Vec::new();
        for unit in units {
            let second = unit
                .data
                .windows(4)
                .enumerate()
                .filter(|(_, w)| *w == [0, 0, 1, 0])
                .map(|(at, _)| at)
                .nth(1);
            match second {
                Some(at) => {
                    for data in [&unit.data[..at], &unit.data[at..]] {
                        out.push(AccessUnit {
                            data: data.to_vec(),
                            ..unit.clone()
                        });
                    }
                }
                None => out.push(unit.clone()),
            }
        }
        out
    }

    fn clean(units: &[AccessUnit]) -> Vec<u64> {
        let mut decoder = Mpeg2Decoder::new(MAX_PIXELS);
        let mut out = Vec::new();
        for unit in units {
            let pts = unit.display_index as i64;
            assert!(take(
                decoder.push(&packet(unit.data.clone(), pts)),
                &mut out
            ));
        }
        assert!(take(decoder.drain(), &mut out));
        out
    }

    #[test]
    fn the_decoder_survives_hostile_streams() {
        let (deadline, count, seed) = budget("MPEG2_FUZZ_SECS", SMOKE_DECODE, 0xDEC);
        let mut rng = Rng(seed);
        let mut sources: Vec<Vec<u8>> = [
            "progressive.m2v",
            "interlaced.m2v",
            "chroma422.m2v",
            "mpeg1.m1v",
            "inter_fields.m2v",
        ]
        .into_iter()
        .map(fixture)
        .collect();
        sources.push(encoded(64, 48, 0, false, &mut rng));
        sources.push(encoded(96, 64, 300_000, true, &mut rng));
        sources.push(encoded(100, 70, 0, false, &mut rng));
        let mut corpus: Vec<Stream> = sources
            .into_iter()
            .map(|bytes| {
                let units = split(&bytes).unwrap();
                let clean = clean(&units);
                Stream {
                    bytes,
                    units,
                    clean,
                    fields_apart: false,
                }
            })
            .collect();
        // The field stream again, one packet per field: a refused packet
        // between two fields must leave the first one open.
        let bytes = fixture("inter_fields.m2v");
        let units = fields_apart(&split(&bytes).unwrap());
        let clean_fields = clean(&units);
        assert_eq!(clean_fields, corpus[4].clean, "fields apart decode alike");
        corpus.push(Stream {
            bytes,
            units,
            clean: clean_fields,
            fields_apart: true,
        });

        let started = Instant::now();
        let (mut iterations, mut refused, mut pictures, mut slowest) =
            (0u64, 0u64, 0u64, Duration::ZERO);
        let mut modes = [0u64; 4];
        while iterations < count && deadline.is_none_or(|end| Instant::now() < end) {
            iterations += 1;
            let begun = Instant::now();
            let stream = &corpus[rng.below(corpus.len())];
            let mode = rng.below(4);
            modes[mode] += 1;
            let mut out = Vec::new();
            match mode {
                // One mutated packet injected: refused, it changes nothing.
                0 => {
                    let at = rng.below(stream.units.len() + 1);
                    let mut bad = stream.units[rng.below(stream.units.len())].data.clone();
                    mutate(&mut bad, &mut rng);
                    let mut decoder = Mpeg2Decoder::new(MAX_PIXELS);
                    let mut injected_refused = false;
                    let mut later_refused = false;
                    for (index, unit) in stream.units.iter().enumerate() {
                        if index == at {
                            let pts = rng.next() as i64;
                            injected_refused =
                                !take(decoder.push(&packet(bad.clone(), pts)), &mut out);
                        }
                        let pts = unit.display_index as i64;
                        let ok = take(decoder.push(&packet(unit.data.clone(), pts)), &mut out);
                        later_refused |= !ok;
                    }
                    if at == stream.units.len() {
                        let pts = rng.next() as i64;
                        injected_refused = !take(decoder.push(&packet(bad.clone(), pts)), &mut out);
                    }
                    take(decoder.drain(), &mut out);
                    if injected_refused {
                        refused += 1;
                        assert!(!later_refused, "a refused packet changed the decoder");
                        assert_eq!(out, stream.clean, "a refused packet changed the pictures");
                    }
                }
                // A mutated stream split in random chunks.
                1 => {
                    let mut bytes = stream.bytes.clone();
                    if rng.chance(80) {
                        mutate(&mut bytes, &mut rng);
                    }
                    let mut splitter = Splitter::new();
                    let mut units = Vec::new();
                    let mut at = 0;
                    let mut failed = false;
                    while at < bytes.len() {
                        let end = (at + 1 + rng.below(4096)).min(bytes.len());
                        match splitter.push(&bytes[at..end]) {
                            Ok(found) => units.extend(found),
                            Err(_) => {
                                failed = true;
                                break;
                            }
                        }
                        at = end;
                    }
                    if !failed && let Ok(last) = splitter.finish() {
                        units.extend(last);
                    }
                    if bytes == stream.bytes && !stream.fields_apart {
                        assert!(!failed);
                        assert_eq!(units, stream.units, "chunked split differs");
                    }
                    let mut decoder = Mpeg2Decoder::new(MAX_PIXELS);
                    for unit in units {
                        if unit.data.is_empty() {
                            continue;
                        }
                        let pts = unit.display_index as i64;
                        if !take(decoder.push(&packet(unit.data, pts)), &mut out) && rng.chance(50)
                        {
                            decoder.reset();
                        }
                    }
                    take(decoder.drain(), &mut out);
                }
                // Mutated configuration, then the clean stream.
                2 => {
                    let first = &stream.units[0].data;
                    let mut extra = first[..sequence_bytes(first)].to_vec();
                    mutate(&mut extra, &mut rng);
                    let mut decoder = Mpeg2Decoder::new(MAX_PIXELS);
                    if decoder.configure(&extra).is_err() {
                        refused += 1;
                    }
                    for unit in &stream.units {
                        take(decoder.push(&packet(unit.data.clone(), 0)), &mut out);
                    }
                    take(decoder.drain(), &mut out);
                }
                // Reset and drain anywhere, odd timestamps.
                _ => {
                    let mut decoder = Mpeg2Decoder::new(MAX_PIXELS);
                    for unit in &stream.units {
                        match rng.below(10) {
                            0 => decoder.reset(),
                            1 => {
                                take(decoder.drain(), &mut out);
                            }
                            _ => {}
                        }
                        let mut access = packet(unit.data.clone(), rng.next() as i64);
                        access.duration = rng.next() as i64;
                        access.timescale = rng.next() as u32;
                        take(decoder.push(&access), &mut out);
                    }
                    take(decoder.drain(), &mut out);
                    take(decoder.drain(), &mut out);
                }
            }
            pictures += out.len() as u64;
            slowest = slowest.max(begun.elapsed());
            assert!(slowest < SLOWEST, "iteration {iterations} took {slowest:?}");
        }
        println!(
            "decoder fuzz: seed {seed:#x}, {iterations} iterations in {:.0} s, modes {modes:?}, \
             {refused} refusals checked, {pictures} pictures, slowest {slowest:?}",
            started.elapsed().as_secs_f64()
        );
    }

    /// Rates from Table 6-4 as (duration, timescale), then rates and
    /// timestamps the encoder must refuse.
    const RATES: [(i64, u32); 12] = [
        (1001, 24_000),
        (1, 24),
        (1, 25),
        (1001, 30_000),
        (1, 30),
        (1, 50),
        (1001, 60_000),
        (1, 60),
        (1, 15),
        (1, 120),
        (-1, 25),
        (1, 0),
    ];

    struct Op {
        input: Option<VideoInput>,
        drain: bool,
        reset: bool,
        cancel: bool,
    }

    fn valid_open(width: u32, height: u32, bitrate: u32) -> bool {
        width > 0
            && height > 0
            && width.is_multiple_of(2)
            && height.is_multiple_of(2)
            && width <= 1920
            && height <= 1152
            && bitrate <= 80_000_000
    }

    /// Plays `ops` and returns every packet's bytes, and which pushes were
    /// accepted. Drained runs are decoded and checked here.
    fn play(encoder: &mut Mpeg2Encoder, ops: &[Op]) -> (Vec<Vec<u8>>, Vec<bool>) {
        let mut bytes = Vec::new();
        let mut accepted = Vec::new();
        let mut run: Vec<Packet> = Vec::new();
        let mut run_pts: Vec<i64> = Vec::new();
        let mut cancelled = false;
        for op in ops {
            if op.cancel {
                encoder.cancel_flag().store(true, Ordering::SeqCst);
                cancelled = true;
            }
            if op.reset {
                encoder.reset();
                run.clear();
                run_pts.clear();
                cancelled = false;
            }
            if let Some(input) = &op.input {
                match encoder.push(input) {
                    Ok(packets) => {
                        accepted.push(true);
                        run_pts.push(input.pts);
                        bytes.extend(packets.iter().map(|p| p.data.clone()));
                        run.extend(packets);
                        assert!(run_pts.len() - run.len() <= 4, "too many pictures held");
                    }
                    Err(_) => accepted.push(false),
                }
            }
            if op.drain && cancelled {
                assert_eq!(encoder.drain().unwrap_err(), CodecError::Cancelled);
            } else if op.drain {
                let packets = encoder.drain().unwrap();
                bytes.extend(packets.iter().map(|p| p.data.clone()));
                run.extend(packets);
                assert_eq!(run.len(), run_pts.len(), "a drained run lost a picture");
                if !run.is_empty() {
                    let mut decoder = Mpeg2Decoder::new(MAX_PIXELS);
                    decoder.configure(encoder.extradata()).unwrap();
                    let mut times = Vec::new();
                    for packet in &run {
                        for frame in decoder.push(packet).unwrap() {
                            times.push(frame.pts);
                        }
                    }
                    for frame in decoder.drain().unwrap() {
                        times.push(frame.pts);
                    }
                    assert_eq!(times, run_pts, "the run decodes to its pictures in order");
                }
                run.clear();
                run_pts.clear();
            }
        }
        (bytes, accepted)
    }

    #[test]
    fn the_encoder_survives_hostile_use() {
        let (deadline, count, seed) = budget("MPEG2_ENCODE_FUZZ_SECS", SMOKE_ENCODE, 0xE4C);
        let mut rng = Rng(seed);
        let started = Instant::now();
        let (mut iterations, mut opened, mut pushed, mut refused, mut slowest) =
            (0u64, 0u64, 0u64, 0u64, Duration::ZERO);
        while iterations < count && deadline.is_none_or(|end| Instant::now() < end) {
            iterations += 1;
            let begun = Instant::now();
            let (width, height) = match rng.below(20) {
                0 => (720, 576),
                1 => (1 + 2 * rng.below(40) as u32, 2 + 2 * rng.below(40) as u32),
                2 => (1922, 1080),
                3 => (0, 16),
                _ => (2 + 2 * rng.below(100) as u32, 2 + 2 * rng.below(80) as u32),
            };
            let bitrate = match rng.below(6) {
                0 => 0,
                1 => 1,
                2 => 90_000_000,
                _ => rng.below(20_000_000) as u32,
            };
            let mut params = CodecParams::video(CodecId::Mpeg2, width, height);
            params.bitrate = bitrate;
            params.constant_rate = rng.chance(40);
            let first = Mpeg2Encoder::open(&params);
            assert_eq!(
                first.is_ok(),
                valid_open(width, height, bitrate),
                "{width}x{height} {bitrate}"
            );
            let (Ok(mut first), Ok(mut second)) = (first, Mpeg2Encoder::open(&params)) else {
                continue;
            };
            opened += 1;
            let rate = if rng.chance(90) {
                RATES[rng.below(8)]
            } else {
                RATES[rng.below(RATES.len())]
            };
            let mut pts = 0i64;
            let mut ops = Vec::new();
            for t in 0..1 + rng.below(40) as u32 {
                let roll = rng.below(100);
                let mut op = Op {
                    input: None,
                    drain: (70..78).contains(&roll),
                    reset: (78..83).contains(&roll),
                    cancel: roll == 83,
                };
                if !(70..=83).contains(&roll) {
                    let (w, h) = if rng.chance(4) {
                        (width + 2, height)
                    } else {
                        (width, height)
                    };
                    let (duration, timescale) = if rng.chance(5) {
                        RATES[rng.below(RATES.len())]
                    } else {
                        rate
                    };
                    pts += if rng.chance(5) {
                        -(rng.below(5) as i64)
                    } else {
                        duration.max(1)
                    };
                    op.input = Some(VideoInput {
                        picture: picture(w, h, rng.below(4), t, &mut rng),
                        pts,
                        duration,
                        timescale,
                        force_keyframe: rng.chance(10),
                    });
                }
                ops.push(op);
            }
            ops.push(Op {
                input: None,
                drain: true,
                reset: false,
                cancel: false,
            });
            let (bytes, accepted) = play(&mut first, &ops);
            pushed += accepted.iter().filter(|&&a| a).count() as u64;
            refused += accepted.iter().filter(|&&a| !a).count() as u64;

            // The same ops without the refused pushes.
            let mut index = 0;
            let replay: Vec<Op> = ops
                .into_iter()
                .map(|mut op| {
                    if op.input.is_some() {
                        let keep = accepted[index];
                        index += 1;
                        if !keep {
                            op.input = None;
                        }
                    }
                    op
                })
                .collect();
            let (again, _) = play(&mut second, &replay);
            assert!(again == bytes, "a refused push changed the stream");

            slowest = slowest.max(begun.elapsed());
            assert!(slowest < SLOWEST, "iteration {iterations} took {slowest:?}");
        }
        println!(
            "encoder fuzz: seed {seed:#x}, {iterations} iterations in {:.0} s, {opened} opened, \
             {pushed} pictures coded, {refused} pushes refused, slowest {slowest:?}",
            started.elapsed().as_secs_f64()
        );
    }
}
