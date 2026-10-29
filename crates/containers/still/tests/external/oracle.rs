//! Packet bytes against ffprobe, and animated WebP against webpinfo.
//!
//! ffprobe 8.1.1 reports a still picture as one packet the size of the file
//! and invents a 1/25 clock. This crate does not copy that clock. GIF packet
//! positions and delays are the comparison. FFmpeg 8.1.1 does not split an
//! animated WebP, so webpinfo is that oracle.

use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use cutlass_codec::{Packet, Timescale};
use cutlass_container::{Demux, Input};
use cutlass_still::{StillDemux, StillKind, still_limits};

use crate::support::{ffprobe, tool_on_path, webpinfo};

#[test]
fn independent_files_match_ffprobe_packet_bytes_and_gif_delays() {
    if !tool_on_path("ffprobe") || !tool_on_path("magick") {
        eprintln!("skipped: ffprobe and magick are not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutlass-still-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp");

    let jpeg = dir.join("one.jpg");
    let png = dir.join("one.png");
    let webp = dir.join("still.webp");
    let gif = dir.join("unequal.gif");
    assert!(run(
        "magick",
        &["-size", "8x8", "xc:red", jpeg.to_str().expect("jpeg")]
    ));
    assert!(run(
        "magick",
        &["-size", "8x8", "xc:blue", png.to_str().expect("png")]
    ));
    assert!(run(
        "magick",
        &["-size", "8x8", "xc:green", webp.to_str().expect("webp")]
    ));
    assert!(run(
        "magick",
        &[
            "-delay",
            "5",
            "-size",
            "1x1",
            "xc:red",
            "-delay",
            "20",
            "-size",
            "1x1",
            "xc:blue",
            gif.to_str().expect("gif"),
        ]
    ));

    compare_still(&jpeg, StillKind::Jpeg);
    compare_still(&png, StillKind::Png);
    compare_still(&webp, StillKind::Webp);
    compare_gif(&gif);

    if tool_on_path("python3") && tool_on_path("webpinfo") {
        let animated = dir.join("anim.webp");
        let script = r#"
from PIL import Image
red = Image.new("RGB", (8, 8), "red")
blue = Image.new("RGB", (8, 8), "blue")
red.save(r.anim, save_all=True, append_images=[blue], duration=[100, 400], loop=0, format="WEBP")
"#
        .replace("r.anim", &format!("{:?}", animated));
        let made = Command::new("python3")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("python");
        assert!(
            made.status.success(),
            "{}",
            String::from_utf8_lossy(&made.stderr)
        );
        compare_animated_webp(&animated);
    } else {
        eprintln!("skipped: python3 or webpinfo is not on PATH");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

fn compare_still(path: &Path, kind: StillKind) {
    let bytes = std::fs::read(path).expect("read");
    let mut demux = open(&bytes);
    assert_eq!(demux.kind(), kind);
    assert!(!demux.animated());
    assert_eq!(demux.media_end(), None);
    let size = demux.streams()[0].params().size().expect("size");
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), bytes.as_slice());
    assert!(packet.timestamp().is_missing());
    assert_eq!(packet.timescale(), Timescale::new(0));
    assert!(demux.read_packet().expect("end").is_none());

    let probed = ffprobe(path);
    assert_eq!(probed.packets.len(), 1, "{}", path.display());
    assert_eq!(
        probed.packets[0].size,
        u64::try_from(bytes.len()).expect("len")
    );
    assert_eq!(probed.width, size.width());
    assert_eq!(probed.height, size.height());
    let pos = probed.packets[0].pos.expect("pos");
    let end = usize::try_from(pos + probed.packets[0].size).expect("end");
    let start = usize::try_from(pos).expect("pos");
    assert_eq!(packet.bytes(), &bytes[start..end]);
}

fn compare_gif(path: &Path) {
    let bytes = std::fs::read(path).expect("read");
    let mut demux = open(&bytes);
    assert_eq!(demux.kind(), StillKind::Gif);
    assert_eq!(
        demux.streams()[0].timescale().ticks_per_second(),
        100,
        "GIF delays are centiseconds"
    );
    let packets = read_all(&mut demux);
    let probed = ffprobe(path);
    assert_eq!(probed.time_base.as_deref(), Some("1/100"));
    assert_eq!(packets.len(), probed.packets.len());
    assert!(packets.len() >= 2, "unequal gif should have two frames");
    let mut previous = None;
    for (index, (packet, probe)) in packets.iter().zip(&probed.packets).enumerate() {
        let pos = probe.pos.expect("pos");
        let start = usize::try_from(pos).expect("pos");
        let end = usize::try_from(pos + probe.size).expect("end");
        assert_eq!(packet.bytes(), &bytes[start..end], "gif packet {index}");
        assert_eq!(
            packet.timestamp().presentation(),
            probe.pts,
            "gif pts {index}"
        );
        assert_eq!(
            packet.timestamp().duration(),
            probe.duration,
            "gif delay {index}"
        );
        assert_eq!(packet.timescale().ticks_per_second(), 100);
        if let Some(previous) = previous {
            assert_ne!(packet.timestamp().duration(), Some(previous));
        }
        previous = packet.timestamp().duration();
    }
    let mut joined = Vec::new();
    for packet in &packets {
        joined.extend_from_slice(packet.bytes());
    }
    assert_eq!(joined, bytes);
}

fn compare_animated_webp(path: &Path) {
    let bytes = std::fs::read(path).expect("read");
    let (frames, loop_count) = webpinfo(path);
    assert!(frames.len() >= 2, "animated webp needs two frames");
    let mut demux = open(&bytes);
    assert!(demux.animated());
    assert_eq!(demux.frame_count(), frames.len());
    assert_eq!(demux.loop_count(), loop_count);
    assert_eq!(demux.streams()[0].timescale().ticks_per_second(), 1000);
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), frames.len());
    let mut clock = 0_i64;
    for (index, (packet, frame)) in packets.iter().zip(&frames).enumerate() {
        let start = if index == 0 { 0 } else { frame.offset };
        let end = frames
            .get(index + 1)
            .map(|next| next.offset)
            .unwrap_or(bytes.len());
        assert_eq!(packet.bytes(), &bytes[start..end], "webp packet {index}");
        assert_eq!(packet.timestamp().presentation(), Some(clock));
        assert_eq!(packet.timestamp().duration(), Some(frame.duration));
        clock += frame.duration;
    }
    assert_ne!(frames[0].duration, frames[1].duration);
    let mut joined = Vec::new();
    for packet in &packets {
        joined.extend_from_slice(packet.bytes());
    }
    assert_eq!(joined, bytes);
}

fn open(bytes: &[u8]) -> StillDemux<Cursor<Vec<u8>>> {
    let len = u64::try_from(bytes.len()).expect("len");
    StillDemux::open(Input::random(
        Cursor::new(bytes.to_vec()),
        len,
        still_limits(),
    ))
    .expect("open")
}

fn read_all(demux: &mut impl Demux) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

fn run(program: &str, args: &[&str]) -> bool {
    let made = Command::new(program).args(args).output().expect(program);
    assert!(
        made.status.success(),
        "{program} {}",
        String::from_utf8_lossy(&made.stderr)
    );
    true
}
