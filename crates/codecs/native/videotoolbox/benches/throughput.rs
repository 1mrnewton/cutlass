//! Decode and encode throughput for VideoToolbox.
//!
//! Decode reads the H.264 and HEVC clips in `local-files/` at the repo root.
//! Frames are dropped without a CPU readback, which is the preview path.
//! Encode fills NV12 or P010 buffers on the CPU and times the encoder
//! separately from that fill. Run with `cargo bench -p cutlass-videotoolbox`.

#![cfg_attr(
    target_vendor = "apple",
    expect(
        clippy::print_stdout,
        reason = "benchmark timings are the program's output"
    )
)]
#![cfg_attr(
    target_vendor = "apple",
    expect(
        clippy::print_stderr,
        reason = "benchmark skips and failures are reported"
    )
)]

#[cfg(target_vendor = "apple")]
mod bench {
    use std::collections::HashMap;
    use std::num::NonZeroU32;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use cutlass_core::{Packet, PixelFormat, Timestamp, VideoCodec, VideoStreamConfig};
    use cutlass_videotoolbox::{Decoder, DecoderOptions, Encoder, EncoderConfig};

    const ENCODE_WARMUP: usize = 24;
    const ENCODE_FRAMES: usize = 90;

    pub fn main() {
        println!("VideoToolbox throughput, release build");
        println!("decode drops frames, no CPU readback");
        println!("encode uses constant buffers; the fill time is not included in the fps");
        println!();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
        let media = root.join("local-files");
        match std::fs::read_dir(&media) {
            Ok(entries) => {
                let mut files: Vec<PathBuf> = entries
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| {
                        path.extension()
                            .and_then(|ext| ext.to_str())
                            .is_some_and(|ext| {
                                ext.eq_ignore_ascii_case("mp4") || ext.eq_ignore_ascii_case("mov")
                            })
                    })
                    .collect();
                files.sort();
                if files.is_empty() {
                    println!("no clips in {}", media.display());
                }
                for path in files {
                    match decode_file(&path) {
                        Ok(()) => {}
                        Err(error) => eprintln!("{}: {error}", path.display()),
                    }
                }
            }
            Err(error) => println!("skipping decode, {}: {error}", media.display()),
        }
        println!();
        for (codec, format, width, height, bitrate) in [
            (VideoCodec::H264, PixelFormat::Nv12, 1280, 720, 6_000_000),
            (VideoCodec::H264, PixelFormat::Nv12, 1920, 1080, 12_000_000),
            (VideoCodec::H264, PixelFormat::Nv12, 3840, 2160, 30_000_000),
            (VideoCodec::Hevc, PixelFormat::Nv12, 1920, 1080, 8_000_000),
            (VideoCodec::Hevc, PixelFormat::Nv12, 3840, 2160, 20_000_000),
            (VideoCodec::Hevc, PixelFormat::P010, 1920, 1080, 8_000_000),
            (VideoCodec::Hevc, PixelFormat::P010, 3840, 2160, 20_000_000),
        ] {
            match encode(codec, format, width, height, bitrate) {
                Ok(()) => {}
                Err(error) => eprintln!("{codec:?} {width}x{height} {format:?}: {error}"),
            }
        }
    }

    fn decode_file(path: &Path) -> Result<(), String> {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("clip");
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        let movie = parse_mp4(&bytes)?;
        let config = VideoStreamConfig {
            codec: movie.codec,
            width: movie.width,
            height: movie.height,
            config_record: movie.config,
        };
        let packets = movie.packets.len();
        // The first pass brings the hardware decoder up. The second is timed,
        // and it does not include creating the session.
        let mut warmup =
            Decoder::new(&config, DecoderOptions::default()).map_err(|error| error.to_string())?;
        let warmed = decode_pass(&mut warmup, &movie.packets)?;
        let hardware = warmup.is_hardware().unwrap_or(false);
        if warmed != packets {
            return Err(format!("decoded {warmed} frames from {packets} packets"));
        }
        let mut decoder =
            Decoder::new(&config, DecoderOptions::default()).map_err(|error| error.to_string())?;
        let started = Instant::now();
        let frames = decode_pass(&mut decoder, &movie.packets)?;
        let elapsed = started.elapsed();
        if frames != packets {
            return Err(format!("decoded {frames} frames from {packets} packets"));
        }
        let kind = if hardware { "hardware" } else { "software" };
        print_rate(
            &format!(
                "decode {name} {}x{} {:?} {kind}",
                movie.width, movie.height, movie.codec
            ),
            frames,
            elapsed,
        );
        Ok(())
    }

    fn decode_pass(decoder: &mut Decoder, packets: &[Packet]) -> Result<usize, String> {
        let mut frames = 0;
        for packet in packets {
            decoder
                .send_packet(packet)
                .map_err(|error| error.to_string())?;
            frames += take_frames(decoder)?;
        }
        decoder.flush().map_err(|error| error.to_string())?;
        frames += take_frames(decoder)?;
        Ok(frames)
    }

    fn take_frames(decoder: &mut Decoder) -> Result<usize, String> {
        let mut frames = 0;
        while let Some(frame) = decoder.receive_frame() {
            std::hint::black_box(frame.map_err(|error| error.to_string())?);
            frames += 1;
        }
        Ok(frames)
    }

    fn encode(
        codec: VideoCodec,
        format: PixelFormat,
        width: u32,
        height: u32,
        bitrate: u32,
    ) -> Result<(), String> {
        let config = EncoderConfig {
            pixel_format: format,
            bitrate: Some(bitrate),
            keyframe_interval: Some(60),
            frame_rate: Some(30.0),
            require_hardware: true,
            ..EncoderConfig::new(codec, width, height)
        };
        let mut encoder = match Encoder::new(&config) {
            Ok(encoder) => encoder,
            Err(error) if error.is_unsupported() => {
                println!("encode {codec:?} {width}x{height} {format:?}: skipped, {error}");
                return Ok(());
            }
            Err(error) => return Err(error.to_string()),
        };
        // Warmup locks the decode-time shift (it needs 17 packets) before the
        // timer, and leaves the encoder session open.
        let mut ignored = Duration::ZERO;
        let warmed = send_frames(&mut encoder, format, ENCODE_WARMUP, 0, false, &mut ignored)?;
        let hardware = encoder.is_hardware().unwrap_or(false);
        let started = Instant::now();
        let mut fill = Duration::ZERO;
        let produced = send_frames(
            &mut encoder,
            format,
            ENCODE_FRAMES,
            i64::try_from(ENCODE_WARMUP).map_err(|_| "warmup")?,
            true,
            &mut fill,
        )?;
        let encode = started.elapsed().saturating_sub(fill);
        if warmed + produced != ENCODE_WARMUP + ENCODE_FRAMES {
            return Err(format!(
                "encoder emitted {} packets for {} frames",
                warmed + produced,
                ENCODE_WARMUP + ENCODE_FRAMES
            ));
        }
        let kind = if hardware { "hardware" } else { "software" };
        print_rate(
            &format!("encode {codec:?} {width}x{height} {format:?} {bitrate} bps {kind}"),
            ENCODE_FRAMES,
            encode,
        );
        Ok(())
    }

    fn send_frames(
        encoder: &mut Encoder,
        format: PixelFormat,
        count: usize,
        first_pts: i64,
        finish: bool,
        fill_time: &mut Duration,
    ) -> Result<usize, String> {
        let timescale = NonZeroU32::new(30).ok_or("timescale")?;
        let mut packets = 0;
        for index in 0..count {
            let filling = Instant::now();
            let mut image = encoder
                .create_pixel_buffer()
                .map_err(|error| error.to_string())?;
            image
                .write_planes(|planes| {
                    for plane in planes.iter_mut() {
                        plane
                            .data
                            .fill(if format == PixelFormat::Nv12 { 128 } else { 0 });
                    }
                })
                .map_err(|error| error.to_string())?;
            *fill_time += filling.elapsed();
            let step = i64::try_from(index).map_err(|_| "frame index")?;
            let pts = Timestamp::new(first_pts + step, timescale);
            encoder
                .send_frame(image, pts, Some(Timestamp::new(1, timescale)), false)
                .map_err(|error| error.to_string())?;
            packets += take_packets(encoder)?;
        }
        if finish {
            encoder.flush().map_err(|error| error.to_string())?;
            packets += take_packets(encoder)?;
        }
        Ok(packets)
    }

    fn take_packets(encoder: &mut Encoder) -> Result<usize, String> {
        let mut packets = 0;
        while let Some(packet) = encoder.receive_packet() {
            std::hint::black_box(packet.map_err(|error| error.to_string())?);
            packets += 1;
        }
        Ok(packets)
    }

    fn print_rate(label: &str, frames: usize, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        let fps = frames as f64 / seconds.max(1e-9);
        let ms = seconds * 1000.0 / (frames as f64).max(1.0);
        println!("{label}: {frames} frames in {seconds:.3} s, {fps:.1} fps, {ms:.2} ms/frame");
    }

    struct Movie {
        codec: VideoCodec,
        width: u32,
        height: u32,
        config: Vec<u8>,
        packets: Vec<Packet>,
    }

    struct Track {
        id: u32,
        codec: VideoCodec,
        width: u32,
        height: u32,
        timescale: u32,
        config: Vec<u8>,
        packets: Vec<Packet>,
    }

    #[derive(Clone, Copy)]
    struct Trex {
        duration: u32,
        size: u32,
        flags: u32,
    }

    fn parse_mp4(data: &[u8]) -> Result<Movie, String> {
        let mut track = None;
        let mut trex = HashMap::new();
        let mut fragments = Vec::new();
        let mut pos = 0;
        while pos + 8 <= data.len() {
            let (kind, payload, next) = box_at(data, pos)?;
            match &kind {
                b"moov" => {
                    let parsed = parse_moov(data, payload, next)?;
                    track = Some(parsed.0);
                    trex = parsed.1;
                }
                b"moof" => fragments.push(pos),
                _ => {}
            }
            pos = next;
        }
        let mut track = track.ok_or("no video track")?;
        if track.packets.is_empty() {
            for moof in fragments {
                parse_moof(data, moof, track.id, &trex, &mut track)?;
            }
        } else if !fragments.is_empty() {
            return Err("video track has both a sample table and fragments".to_string());
        }
        if track.packets.is_empty() {
            return Err("video track has no samples".to_string());
        }
        let timescale = NonZeroU32::new(track.timescale).ok_or("track timescale is 0")?;
        for packet in &mut track.packets {
            packet.pts.timescale = timescale;
            if let Some(dts) = &mut packet.dts {
                dts.timescale = timescale;
            }
            if let Some(duration) = &mut packet.duration {
                duration.timescale = timescale;
            }
        }
        Ok(Movie {
            codec: track.codec,
            width: track.width,
            height: track.height,
            config: track.config,
            packets: track.packets,
        })
    }

    fn parse_moov(
        data: &[u8],
        start: usize,
        end: usize,
    ) -> Result<(Track, HashMap<u32, Trex>), String> {
        let mut tracks = Vec::new();
        let mut trex = HashMap::new();
        for (kind, payload, next) in children(data, start, end)? {
            match &kind {
                b"trak" => {
                    if let Some(track) = parse_trak(data, payload, next)? {
                        tracks.push(track);
                    }
                }
                b"mvex" => {
                    for (kind, payload, _) in children(data, payload, next)? {
                        if kind == *b"trex" {
                            let (id, defaults) = parse_trex(data, payload)?;
                            trex.insert(id, defaults);
                        }
                    }
                }
                _ => {}
            }
        }
        let track = tracks
            .into_iter()
            .next()
            .ok_or("moov has no avc1 or hvc1 track")?;
        Ok((track, trex))
    }

    fn parse_trak(data: &[u8], start: usize, end: usize) -> Result<Option<Track>, String> {
        let mdia = child(data, start, end, b"mdia")?;
        let minf = child(data, mdia.0, mdia.1, b"minf")?;
        let stbl = child(data, minf.0, minf.1, b"stbl")?;
        let stsd = child(data, stbl.0, stbl.1, b"stsd")?;
        let Some(entry) = parse_stsd(data, stsd.0, stsd.1)? else {
            return Ok(None);
        };
        let tkhd = child(data, start, end, b"tkhd")?;
        let mdhd = child(data, mdia.0, mdia.1, b"mdhd")?;
        let packets = parse_sample_table(data, stbl.0, stbl.1)?;
        Ok(Some(Track {
            id: track_id(data, tkhd.0)?,
            codec: entry.codec,
            width: entry.width,
            height: entry.height,
            timescale: timescale_of(data, mdhd.0)?,
            config: entry.config,
            packets,
        }))
    }

    struct SampleEntry {
        codec: VideoCodec,
        width: u32,
        height: u32,
        config: Vec<u8>,
    }

    fn parse_stsd(data: &[u8], start: usize, end: usize) -> Result<Option<SampleEntry>, String> {
        if start + 8 > end {
            return Err("truncated stsd".to_string());
        }
        let entry = start + 8;
        if entry + 8 > end {
            return Ok(None);
        }
        let (kind, payload, entry_end) = box_at(data, entry)?;
        let codec = match &kind {
            b"avc1" | b"avc3" => VideoCodec::H264,
            b"hvc1" | b"hev1" => VideoCodec::Hevc,
            _ => return Ok(None),
        };
        let width = u32::from(be_u16(data, payload + 24)?);
        let height = u32::from(be_u16(data, payload + 26)?);
        let atom = if codec == VideoCodec::H264 {
            b"avcC"
        } else {
            b"hvcC"
        };
        let mut config = None;
        let mut child_at = payload + 78;
        while child_at + 8 <= entry_end {
            let (kind, payload, next) = box_at(data, child_at)?;
            if kind == *atom {
                config = Some(data[payload..next].to_vec());
            }
            child_at = next;
        }
        Ok(Some(SampleEntry {
            codec,
            width,
            height,
            config: config.ok_or("sample entry has no config record")?,
        }))
    }

    fn parse_sample_table(data: &[u8], start: usize, end: usize) -> Result<Vec<Packet>, String> {
        let stsz = child(data, start, end, b"stsz")?;
        let (fixed, sizes) = sample_sizes(data, stsz.0, stsz.1)?;
        if sizes.is_empty() {
            return Ok(Vec::new());
        }
        let count = sizes.len();
        let stco = children(data, start, end)?
            .into_iter()
            .find(|child| child.0 == *b"stco" || child.0 == *b"co64");
        let Some((kind, payload, next)) = stco else {
            return Err("video track has samples but no chunk offsets".to_string());
        };
        let chunks = chunk_offsets(data, payload, next, kind == *b"co64")?;
        let stsc = child(data, start, end, b"stsc")?;
        let groups = sample_groups(data, stsc.0, chunks.len())?;
        let stts = child(data, start, end, b"stts")?;
        let (dts, duration) = decode_times(data, stts.0, count)?;
        let offsets = match child(data, start, end, b"ctts") {
            Ok((payload, _)) => composition_offsets(data, payload, count)?,
            Err(_) => vec![0; count],
        };
        let sync = match child(data, start, end, b"stss") {
            Ok((payload, _)) => sync_samples(data, payload)?,
            Err(_) => (0..count).collect(),
        };
        let mut packets = Vec::with_capacity(count);
        let mut sample = 0;
        for (chunk_index, samples_in_chunk) in groups.iter().enumerate() {
            let mut offset = chunks
                .get(chunk_index)
                .copied()
                .ok_or("chunk offset missing")?;
            for _ in 0..*samples_in_chunk {
                if sample >= count {
                    return Err(format!("chunks hold more than the {count} samples in stsz"));
                }
                let size = fixed.unwrap_or(sizes[sample]);
                let end = offset + size;
                let bytes = data
                    .get(offset..end)
                    .ok_or("sample runs past the end of the file")?
                    .to_vec();
                let pts = dts[sample] + offsets[sample];
                packets.push(bare_packet(
                    bytes,
                    pts,
                    dts[sample],
                    duration[sample],
                    sync.contains(&sample),
                ));
                offset = end;
                sample += 1;
            }
        }
        if sample != count {
            return Err(format!(
                "sample table describes {count} samples but chunks hold {sample}"
            ));
        }
        Ok(packets)
    }

    fn parse_moof(
        data: &[u8],
        moof: usize,
        track_id: u32,
        trex: &HashMap<u32, Trex>,
        track: &mut Track,
    ) -> Result<(), String> {
        let (_, moof_payload, moof_end) = box_at(data, moof)?;
        for (kind, payload, next) in children(data, moof_payload, moof_end)? {
            if kind != *b"traf" {
                continue;
            }
            let boxes = children(data, payload, next)?;
            let Some((_, tfhd_at, _)) = boxes.iter().find(|(kind, ..)| *kind == *b"tfhd") else {
                continue;
            };
            let header = parse_tfhd(data, *tfhd_at)?;
            if header.track_id != track_id {
                continue;
            }
            let mut defaults = trex.get(&track_id).copied().unwrap_or(Trex {
                duration: 0,
                size: 0,
                flags: 0,
            });
            if let Some(duration) = header.duration {
                defaults.duration = duration;
            }
            if let Some(size) = header.size {
                defaults.size = size;
            }
            if let Some(flags) = header.flags {
                defaults.flags = flags;
            }
            // An explicit base wins. Otherwise offsets are from this moof,
            // which is what `default-base-is-moof` means.
            let base = if let Some(offset) = header.base_offset {
                offset
            } else if header.base_is_moof {
                u64::try_from(moof).map_err(|_| "moof")?
            } else {
                return Err("fragment has no data base".to_string());
            };
            let mut dts = match boxes.iter().find(|(kind, ..)| *kind == *b"tfdt") {
                Some((_, tfdt_at, _)) => parse_tfdt(data, *tfdt_at)?,
                None => track
                    .packets
                    .last()
                    .and_then(|packet| packet.dts)
                    .map(|dts| dts.value)
                    .unwrap_or(0),
            };
            for (kind, payload, _) in boxes {
                if kind == *b"trun" {
                    dts = append_trun(data, payload, base, dts, defaults, track)?;
                }
            }
        }
        Ok(())
    }

    struct Tfhd {
        track_id: u32,
        base_offset: Option<u64>,
        base_is_moof: bool,
        duration: Option<u32>,
        size: Option<u32>,
        flags: Option<u32>,
    }

    fn parse_tfhd(data: &[u8], start: usize) -> Result<Tfhd, String> {
        let flags = be_u32(data, start)? & 0x00ff_ffff;
        let mut pos = start + 8;
        let track_id = be_u32(data, start + 4)?;
        let base_offset = if flags & 0x1 != 0 {
            let offset = be_u64(data, pos)?;
            pos += 8;
            Some(offset)
        } else {
            None
        };
        if flags & 0x2 != 0 {
            pos += 4;
        }
        let duration = if flags & 0x8 != 0 {
            let duration = be_u32(data, pos)?;
            pos += 4;
            Some(duration)
        } else {
            None
        };
        let size = if flags & 0x10 != 0 {
            let size = be_u32(data, pos)?;
            pos += 4;
            Some(size)
        } else {
            None
        };
        let sample_flags = if flags & 0x20 != 0 {
            Some(be_u32(data, pos)?)
        } else {
            None
        };
        Ok(Tfhd {
            track_id,
            base_offset,
            base_is_moof: flags & 0x2_0000 != 0,
            duration,
            size,
            flags: sample_flags,
        })
    }

    fn parse_tfdt(data: &[u8], start: usize) -> Result<i64, String> {
        let version = data.get(start).copied().ok_or("truncated tfdt")?;
        if version == 1 {
            i64::try_from(be_u64(data, start + 4)?)
                .map_err(|_| "tfdt does not fit in i64".to_string())
        } else {
            Ok(i64::from(be_u32(data, start + 4)?))
        }
    }

    fn append_trun(
        data: &[u8],
        start: usize,
        base: u64,
        mut dts: i64,
        defaults: Trex,
        track: &mut Track,
    ) -> Result<i64, String> {
        let version = data.get(start).copied().ok_or("truncated trun")?;
        let flags = be_u32(data, start)? & 0x00ff_ffff;
        let count = be_u32(data, start + 4)? as usize;
        let mut pos = start + 8;
        let data_offset = if flags & 0x1 != 0 {
            let offset = be_i32(data, pos)?;
            pos += 4;
            offset
        } else {
            0
        };
        let first_flags = if flags & 0x4 != 0 {
            let flags = be_u32(data, pos)?;
            pos += 4;
            Some(flags)
        } else {
            None
        };
        let mut file_pos =
            i64::try_from(base).map_err(|_| "sample base does not fit")? + i64::from(data_offset);
        for index in 0..count {
            let duration = if flags & 0x100 != 0 {
                let duration = be_u32(data, pos)?;
                pos += 4;
                duration
            } else {
                defaults.duration
            };
            let size = if flags & 0x200 != 0 {
                let size = be_u32(data, pos)?;
                pos += 4;
                size
            } else {
                defaults.size
            };
            let sample_flags = if flags & 0x400 != 0 {
                let flags = be_u32(data, pos)?;
                pos += 4;
                flags
            } else if index == 0 {
                first_flags.unwrap_or(defaults.flags)
            } else {
                defaults.flags
            };
            let offset = if flags & 0x800 != 0 {
                let offset = if version == 1 {
                    i64::from(be_i32(data, pos)?)
                } else {
                    i64::from(be_u32(data, pos)?)
                };
                pos += 4;
                offset
            } else {
                0
            };
            let size = usize::try_from(size).map_err(|_| "sample size")?;
            let begin = usize::try_from(file_pos).map_err(|_| "negative sample offset")?;
            let bytes = data
                .get(begin..begin + size)
                .ok_or("fragment sample runs past the end of the file")?
                .to_vec();
            let keyframe = sample_flags & 0x0001_0000 == 0;
            track.packets.push(bare_packet(
                bytes,
                dts + offset,
                dts,
                i64::from(duration),
                keyframe,
            ));
            file_pos += i64::try_from(size).map_err(|_| "sample size")?;
            dts += i64::from(duration);
        }
        Ok(dts)
    }

    fn bare_packet(data: Vec<u8>, pts: i64, dts: i64, duration: i64, keyframe: bool) -> Packet {
        let timescale = NonZeroU32::new(1).expect("1 is a timescale");
        Packet {
            data,
            pts: Timestamp::new(pts, timescale),
            dts: Some(Timestamp::new(dts, timescale)),
            duration: Some(Timestamp::new(duration, timescale)),
            keyframe,
        }
    }

    fn parse_trex(data: &[u8], start: usize) -> Result<(u32, Trex), String> {
        Ok((
            be_u32(data, start + 4)?,
            Trex {
                duration: be_u32(data, start + 12)?,
                size: be_u32(data, start + 16)?,
                flags: be_u32(data, start + 20)?,
            },
        ))
    }

    fn track_id(data: &[u8], start: usize) -> Result<u32, String> {
        let version = data.get(start).copied().ok_or("truncated tkhd")?;
        let id_at = if version == 1 { start + 20 } else { start + 12 };
        be_u32(data, id_at)
    }

    fn timescale_of(data: &[u8], start: usize) -> Result<u32, String> {
        let version = data.get(start).copied().ok_or("truncated mdhd")?;
        let at = if version == 1 { start + 20 } else { start + 12 };
        be_u32(data, at)
    }

    fn sample_sizes(
        data: &[u8],
        start: usize,
        end: usize,
    ) -> Result<(Option<usize>, Vec<usize>), String> {
        let fixed = be_u32(data, start + 4)? as usize;
        let count = be_u32(data, start + 8)? as usize;
        if fixed != 0 {
            return Ok((Some(fixed), vec![fixed; count]));
        }
        let mut sizes = Vec::with_capacity(count);
        let mut pos = start + 12;
        for _ in 0..count {
            if pos + 4 > end {
                return Err("truncated stsz".to_string());
            }
            sizes.push(be_u32(data, pos)? as usize);
            pos += 4;
        }
        Ok((None, sizes))
    }

    fn chunk_offsets(
        data: &[u8],
        start: usize,
        end: usize,
        wide: bool,
    ) -> Result<Vec<usize>, String> {
        let count = be_u32(data, start + 4)? as usize;
        let mut offsets = Vec::with_capacity(count);
        let mut pos = start + 8;
        let width = if wide { 8 } else { 4 };
        for _ in 0..count {
            if pos + width > end {
                return Err("truncated chunk offsets".to_string());
            }
            let offset = if wide {
                usize::try_from(be_u64(data, pos)?).map_err(|_| "chunk offset does not fit")?
            } else {
                usize::try_from(be_u32(data, pos)?).map_err(|_| "chunk offset")?
            };
            offsets.push(offset);
            pos += width;
        }
        Ok(offsets)
    }

    /// `stsc` entries cover chunks from `first_chunk` up to the next entry.
    /// The last entry runs through `chunks`, which comes from `stco`/`co64`.
    fn sample_groups(data: &[u8], start: usize, chunks: usize) -> Result<Vec<u32>, String> {
        let count = usize::try_from(be_u32(data, start + 4)?).map_err(|_| "stsc entry count")?;
        let mut entries = Vec::with_capacity(count);
        let mut pos = start + 8;
        for _ in 0..count {
            entries.push((be_u32(data, pos)?, be_u32(data, pos + 4)?));
            pos += 12;
        }
        if chunks == 0 {
            return Ok(Vec::new());
        }
        if entries.is_empty() {
            return Err("stsc is empty".to_string());
        }
        let mut groups = vec![0; chunks];
        for index in 0..entries.len() {
            let first = usize::try_from(entries[index].0).map_err(|_| "stsc first_chunk")?;
            if first == 0 || first > chunks {
                return Err(format!("stsc first_chunk {first} is outside 1..={chunks}"));
            }
            let next = match entries.get(index + 1) {
                Some((next, _)) => usize::try_from(*next).map_err(|_| "stsc first_chunk")?,
                None => chunks + 1,
            };
            if next <= first || next > chunks + 1 {
                return Err("stsc entries are not in order".to_string());
            }
            for group in &mut groups[first - 1..next - 1] {
                *group = entries[index].1;
            }
        }
        if groups.contains(&0) {
            return Err("stsc does not cover every chunk".to_string());
        }
        Ok(groups)
    }

    fn decode_times(
        data: &[u8],
        start: usize,
        samples: usize,
    ) -> Result<(Vec<i64>, Vec<i64>), String> {
        let entries = be_u32(data, start + 4)? as usize;
        let mut dts = Vec::with_capacity(samples);
        let mut duration = Vec::with_capacity(samples);
        let mut time = 0_i64;
        let mut pos = start + 8;
        for _ in 0..entries {
            let count = be_u32(data, pos)? as usize;
            let delta = i64::from(be_u32(data, pos + 4)?);
            pos += 8;
            for _ in 0..count {
                dts.push(time);
                duration.push(delta);
                time += delta;
            }
        }
        if dts.len() != samples {
            return Err(format!(
                "stts has {} samples, stsz has {samples}",
                dts.len()
            ));
        }
        Ok((dts, duration))
    }

    fn composition_offsets(data: &[u8], start: usize, samples: usize) -> Result<Vec<i64>, String> {
        let version = data.get(start).copied().ok_or("truncated ctts")?;
        let entries = be_u32(data, start + 4)? as usize;
        let mut offsets = Vec::with_capacity(samples);
        let mut pos = start + 8;
        for _ in 0..entries {
            let count = be_u32(data, pos)? as usize;
            let offset = if version == 1 {
                i64::from(be_i32(data, pos + 4)?)
            } else {
                i64::from(be_u32(data, pos + 4)?)
            };
            pos += 8;
            offsets.extend(std::iter::repeat_n(offset, count));
        }
        if offsets.len() != samples {
            return Err("ctts sample count does not match stsz".to_string());
        }
        Ok(offsets)
    }

    fn sync_samples(data: &[u8], start: usize) -> Result<std::collections::HashSet<usize>, String> {
        let count = be_u32(data, start + 4)? as usize;
        let mut sync = std::collections::HashSet::with_capacity(count);
        let mut pos = start + 8;
        for _ in 0..count {
            let number = be_u32(data, pos)? as usize;
            if number > 0 {
                sync.insert(number - 1);
            }
            pos += 4;
        }
        Ok(sync)
    }

    fn child(
        data: &[u8],
        start: usize,
        end: usize,
        name: &[u8; 4],
    ) -> Result<(usize, usize), String> {
        children(data, start, end)?
            .into_iter()
            .find(|child| child.0 == *name)
            .map(|child| (child.1, child.2))
            .ok_or_else(|| format!("missing {}", std::str::from_utf8(name).unwrap_or("box")))
    }

    fn children(
        data: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Vec<([u8; 4], usize, usize)>, String> {
        let mut found = Vec::new();
        let mut pos = start;
        while pos + 8 <= end {
            let (kind, payload, next) = box_at(data, pos)?;
            if next > end {
                return Err("child box extends past its parent".to_string());
            }
            found.push((kind, payload, next));
            pos = next;
        }
        Ok(found)
    }

    fn box_at(data: &[u8], pos: usize) -> Result<([u8; 4], usize, usize), String> {
        let size32 = be_u32(data, pos)?;
        let kind = data
            .get(pos + 4..pos + 8)
            .ok_or("truncated box type")?
            .try_into()
            .map_err(|_| "truncated box type")?;
        let (header, size) = if size32 == 1 {
            let wide =
                usize::try_from(be_u64(data, pos + 8)?).map_err(|_| "box size does not fit")?;
            (16, wide)
        } else if size32 == 0 {
            (8, data.len() - pos)
        } else {
            (8, size32 as usize)
        };
        if size < header || pos + size > data.len() {
            return Err("box size is out of range".to_string());
        }
        Ok((kind, pos + header, pos + size))
    }

    fn be_u16(data: &[u8], pos: usize) -> Result<u16, String> {
        let bytes: [u8; 2] = data
            .get(pos..pos + 2)
            .ok_or("truncated")?
            .try_into()
            .map_err(|_| "truncated")?;
        Ok(u16::from_be_bytes(bytes))
    }

    fn be_u32(data: &[u8], pos: usize) -> Result<u32, String> {
        let bytes: [u8; 4] = data
            .get(pos..pos + 4)
            .ok_or("truncated")?
            .try_into()
            .map_err(|_| "truncated")?;
        Ok(u32::from_be_bytes(bytes))
    }

    fn be_u64(data: &[u8], pos: usize) -> Result<u64, String> {
        let bytes: [u8; 8] = data
            .get(pos..pos + 8)
            .ok_or("truncated")?
            .try_into()
            .map_err(|_| "truncated")?;
        Ok(u64::from_be_bytes(bytes))
    }

    fn be_i32(data: &[u8], pos: usize) -> Result<i32, String> {
        let bytes: [u8; 4] = data
            .get(pos..pos + 4)
            .ok_or("truncated")?
            .try_into()
            .map_err(|_| "truncated")?;
        Ok(i32::from_be_bytes(bytes))
    }
}

fn main() {
    #[cfg(target_vendor = "apple")]
    bench::main();
}
