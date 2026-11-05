//! Files `flac` and `ffmpeg` write, demuxed here and compared with
//! `flac --analyze` (frame bytes), `ffprobe` (packet times and sizes), and
//! the codec's MD5 check; and files muxed here, checked by `flac -t`,
//! `metaflac`, and `ffprobe`.

use std::fs::File;
use std::path::{Path, PathBuf};

use cutlass_codec::{CodecId, StreamCodec, Timescale};
use cutlass_container::{Demux, Landing, Mux, NewStream, SeekTo, TrackDuration};
use cutlass_flac::{FlacDecoder, FrameHeader, Md5Check, crc};
use cutlass_flac_container::{FlacDemux, FlacMux};

use crate::support::{
    Kind, Rng, analyze, have, path_str, run, scratch, signal, succeeds, write_wav,
};

pub struct Made {
    pub name: String,
    pub path: PathBuf,
    pub seektable: bool,
}

/// One packet as ffprobe reports it.
#[derive(Debug, PartialEq, Eq)]
pub struct Probed {
    pub pts: i64,
    pub duration: i64,
    pub size: usize,
}

pub fn ffprobe_packets(path: &Path) -> Vec<Probed> {
    let out = run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=pts,duration,size",
            "-of",
            "csv=p=0",
            path_str(path),
        ],
    );
    String::from_utf8(out)
        .expect("utf-8")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            Probed {
                pts: fields[0].parse().expect("pts"),
                duration: fields[1].parse().expect("duration"),
                size: fields[2].parse().expect("size"),
            }
        })
        .collect()
}

pub fn open(path: &Path) -> FlacDemux<File> {
    let file = File::open(path).expect("open");
    let length = file.metadata().expect("metadata").len();
    FlacDemux::open(file, length).expect("demux")
}

pub fn read_all(demux: &mut FlacDemux<File>) -> Vec<cutlass_codec::Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

/// Files written by libFLAC and ffmpeg with different metadata and framing.
pub fn made_files(dir: &Path) -> Vec<Made> {
    let mut made = Vec::new();
    let stereo = signal(Kind::Music, 2, 16, 44_100 * 12, 1);
    let wav = dir.join("stereo.wav");
    write_wav(&wav, &stereo, 2, 44_100, 16);
    let deep = signal(Kind::Music, 6, 24, 96_000 * 3, 2);
    let deep_wav = dir.join("deep.wav");
    write_wav(&deep_wav, &deep, 6, 96_000, 24);
    let mono = signal(Kind::Music, 1, 8, 8000 * 20, 3);
    let mono_wav = dir.join("mono.wav");
    write_wav(&mono_wav, &mono, 1, 8000, 8);
    let picture = dir.join("cover.png");
    run(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64",
            "-frames:v",
            "1",
            path_str(&picture),
        ],
    );
    let picture_arg = format!("--picture={}", path_str(&picture));
    let cases: Vec<(&str, &Path, Vec<&str>, bool)> = vec![
        ("default-5", &wav, vec!["-5"], true),
        (
            "picture-tags-8",
            &wav,
            vec![
                "-8",
                &picture_arg,
                "-T",
                "ARTIST=cutlass",
                "-T",
                "TITLE=test",
            ],
            true,
        ),
        (
            "no-seektable-0",
            &wav,
            vec!["-0", "--no-seektable", "--no-padding"],
            false,
        ),
        ("dense-seektable", &wav, vec!["-5", "-S", "1s"], true),
        (
            "tiny-blocks",
            &wav,
            vec!["--lax", "-b", "16", "--no-seektable"],
            false,
        ),
        ("big-blocks", &wav, vec!["--lax", "-b", "65535"], true),
        ("s24-6ch-96k", &deep_wav, vec!["-5"], true),
        ("s8-mono", &mono_wav, vec!["-5", "--no-seektable"], false),
    ];
    for (name, source, args, seektable) in cases {
        let out = dir.join(format!("{name}.flac"));
        let mut all = vec!["-s", "-f"];
        all.extend(args);
        all.extend([path_str(source), "-o", path_str(&out)]);
        run("flac", &all);
        made.push(Made {
            name: name.to_owned(),
            path: out,
            seektable,
        });
    }
    let ffmpeg_out = dir.join("ffmpeg-made.flac");
    run(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-y",
            "-i",
            path_str(&wav),
            "-c:a",
            "flac",
            path_str(&ffmpeg_out),
        ],
    );
    made.push(Made {
        name: "ffmpeg-made".to_owned(),
        path: ffmpeg_out,
        // ffmpeg writes STREAMINFO, VORBIS_COMMENT, and PADDING only.
        seektable: false,
    });
    made
}

fn decode_md5(packets: &[cutlass_codec::Packet], extradata: &[u8]) -> Md5Check {
    let info = cutlass_flac::StreamInfo::parse(extradata).expect("streaminfo");
    let mut decoder = FlacDecoder::new(Some(info));
    for packet in packets {
        decoder.decode(packet.bytes()).expect("decode");
    }
    decoder.md5_check()
}

#[test]
fn independently_written_files_demux_frame_for_frame() {
    if !have("flac") || !have("ffmpeg") || !have("ffprobe") {
        return;
    }
    let dir = scratch("container-demux");
    for made in made_files(&dir) {
        let bytes = std::fs::read(&made.path).expect("read");
        let mut demux = open(&made.path);
        assert_eq!(
            demux.seek_points().is_empty(),
            !made.seektable,
            "{}",
            made.name
        );
        let stream = &demux.streams()[0];
        assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Flac));
        assert_eq!(stream.params().extradata(), &bytes[8..42], "{}", made.name);
        let extradata = stream.params().extradata().to_vec();
        let packets = read_all(&mut demux);
        let frames = analyze(&made.path);
        assert_eq!(packets.len(), frames.len(), "{} frame count", made.name);
        for (packet, frame) in packets.iter().zip(&frames) {
            assert_eq!(
                packet.bytes(),
                &bytes[frame.offset..frame.offset + frame.bits / 8],
                "{} frame at {}",
                made.name,
                frame.offset
            );
        }
        let probed = ffprobe_packets(&made.path);
        let ours: Vec<Probed> = packets
            .iter()
            .map(|packet| Probed {
                pts: packet.timestamp().presentation().expect("pts"),
                duration: packet.timestamp().duration().expect("duration"),
                size: packet.bytes().len(),
            })
            .collect();
        assert_eq!(ours, probed, "{} against ffprobe", made.name);
        assert_eq!(
            decode_md5(&packets, &extradata),
            Md5Check::Match,
            "{}",
            made.name
        );
        eprintln!(
            "{}: {} frames match flac --analyze and ffprobe, md5 matches",
            made.name,
            packets.len()
        );
    }
}

#[test]
fn seeks_land_on_the_frame_holding_the_target() {
    if !have("flac") || !have("ffmpeg") {
        return;
    }
    let dir = scratch("container-seek");
    let mut rng = Rng(77);
    for made in made_files(&dir) {
        let mut demux = open(&made.path);
        let packets = read_all(&mut demux);
        let total = demux.stream_info().total_samples;
        let mut scans = 0u64;
        for _ in 0..25 {
            let target = (rng.next() % total) as i64;
            let landing = demux
                .seek(SeekTo {
                    track: 0,
                    time: target,
                    end: None,
                })
                .expect("seek");
            let Landing::Packet {
                time,
                byte_position,
                ..
            } = landing.landing
            else {
                panic!("{}: no landing for {target}", made.name);
            };
            scans += u64::from(landing.payload_scans);
            assert_eq!(landing.index_used, made.seektable, "{}", made.name);
            let packet = demux.read_packet().expect("read").expect("packet");
            let start = packet.timestamp().presentation().expect("pts");
            let span = packet.timestamp().duration().expect("duration");
            assert_eq!(start, time);
            assert!(
                start <= target && target < start + span,
                "{} {target}",
                made.name
            );
            let expected = packets
                .iter()
                .find(|p| p.timestamp().presentation() == Some(start))
                .expect("frame");
            assert_eq!(&packet, expected);
            assert!(byte_position > demux.first_frame() || start == 0);
        }
        let end = demux
            .seek(SeekTo {
                track: 0,
                time: total as i64,
                end: None,
            })
            .expect("seek");
        assert_eq!(end.landing, Landing::End);
        eprintln!("{}: 25 seeks exact, {scans} frames walked", made.name);
    }
}

fn mux_packets(
    path: &Path,
    extradata: &[u8],
    rate: u32,
    channels: u16,
    packets: &[cutlass_codec::Packet],
) {
    let out = File::create(path).expect("create");
    let mut mux = FlacMux::new(out);
    let params = cutlass_codec::StreamParams::try_audio(
        StreamCodec::Known(CodecId::Flac),
        rate,
        channels,
        extradata.to_vec(),
    )
    .expect("params");
    let track = mux
        .add_stream(NewStream {
            params,
            timescale: Timescale::new(rate),
            language: None,
        })
        .expect("stream");
    let mut ticks = 0;
    for packet in packets {
        mux.write_packet(packet).expect("write");
        ticks += packet.timestamp().duration().expect("duration");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(rate),
        ticks,
    }])
    .expect("finish");
}

fn metaflac(path: &Path, arg: &str) -> String {
    String::from_utf8(run("metaflac", &[arg, path_str(path)]))
        .expect("utf-8")
        .trim()
        .to_owned()
}

#[test]
fn files_muxed_here_pass_flac_metaflac_and_ffprobe() {
    if !have("flac") || !have("ffmpeg") || !have("ffprobe") || !have("metaflac") {
        return;
    }
    let dir = scratch("container-mux");
    for made in made_files(&dir) {
        let mut demux = open(&made.path);
        let stream = demux.streams()[0].clone();
        let packets = read_all(&mut demux);
        let out = dir.join(format!("{}-remuxed.flac", made.name));
        mux_packets(
            &out,
            stream.params().extradata(),
            stream.params().sample_rate().expect("rate"),
            stream.params().channels().expect("channels"),
            &packets,
        );
        let (ok, stderr) = succeeds("flac", &["-s", "-t", path_str(&out)]);
        assert!(ok, "{}: flac -t refused: {stderr}", made.name);
        for arg in [
            "--show-total-samples",
            "--show-md5sum",
            "--show-sample-rate",
            "--show-channels",
            "--show-bps",
        ] {
            assert_eq!(
                metaflac(&out, arg),
                metaflac(&made.path, arg),
                "{} {arg}",
                made.name
            );
        }
        let listing = String::from_utf8(run(
            "metaflac",
            &["--list", "--block-type=SEEKTABLE", path_str(&out)],
        ))
        .expect("utf-8");
        let points = listing
            .lines()
            .filter(|line| line.contains("point ") && !line.contains("PLACEHOLDER"))
            .count();
        assert!(points > 0, "{}: no seek points:\n{listing}", made.name);
        assert_eq!(
            ffprobe_packets(&out),
            ffprobe_packets(&made.path),
            "{}",
            made.name
        );
        let again = read_all(&mut open(&out));
        assert_eq!(again, packets, "{}", made.name);
        eprintln!(
            "{}: remuxed file passes flac -t, metaflac ({points} seek points), ffprobe",
            made.name
        );
    }
}

/// Rewrites a libFLAC frame header to the variable-block strategy.
fn as_variable(frame: &[u8], sample: u64) -> Vec<u8> {
    let mut header = FrameHeader::parse(frame).expect("header");
    let len = header.len;
    header.variable_block = true;
    header.number = sample;
    let mut out = header.to_bytes().expect("bytes");
    out.extend_from_slice(&frame[len..frame.len() - 2]);
    let crc = crc::crc16(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

#[test]
fn a_variable_block_stream_muxes_and_demuxes() {
    if !have("flac") || !have("ffprobe") {
        return;
    }
    let dir = scratch("container-variable");
    let source = signal(Kind::Music, 2, 16, 4096 * 8, 9);
    let wav = dir.join("source.wav");
    write_wav(&wav, &source, 2, 44_100, 16);
    let mut encodes = Vec::new();
    for block in ["4096", "1024"] {
        let flac = dir.join(format!("b{block}.flac"));
        run(
            "flac",
            &[
                "-s",
                "-f",
                "-b",
                block,
                path_str(&wav),
                "-o",
                path_str(&flac),
            ],
        );
        let mut demux = open(&flac);
        let extradata = demux.streams()[0].params().extradata().to_vec();
        encodes.push((extradata, read_all(&mut demux)));
    }
    let mut packets = Vec::new();
    let mut sample = 0u64;
    let mut turn = 0usize;
    while sample < 4096 * 8 {
        let pick = if sample.is_multiple_of(4096) {
            turn % 2
        } else {
            1
        };
        let block = [4096u64, 1024][pick];
        let frame = &encodes[pick].1[usize::try_from(sample / block).expect("index")];
        let data = as_variable(frame.bytes(), sample);
        let timestamp = cutlass_codec::Timestamp::try_new(
            Some(sample as i64),
            Some(sample as i64),
            Some(block as i64),
        )
        .expect("timestamp");
        packets.push(
            cutlass_codec::Packet::try_new(0, data, timestamp, Timescale::new(44_100), true)
                .expect("packet"),
        );
        sample += block;
        turn += 1;
    }
    let mut extradata = encodes[0].0.clone();
    // min_block 1024, max_block 4096.
    extradata[..4].copy_from_slice(&[0x04, 0x00, 0x10, 0x00]);
    let out = dir.join("variable.flac");
    mux_packets(&out, &extradata, 44_100, 2, &packets);
    let (ok, stderr) = succeeds("flac", &["-s", "-t", path_str(&out)]);
    assert!(ok, "flac -t refused the variable stream: {stderr}");
    let probed = ffprobe_packets(&out);
    let times: Vec<i64> = packets
        .iter()
        .map(|p| p.timestamp().presentation().expect("pts"))
        .collect();
    assert_eq!(probed.iter().map(|p| p.pts).collect::<Vec<_>>(), times);
    let mut demux = open(&out);
    assert_eq!(read_all(&mut demux), packets);
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 4096 + 1500,
            end: None,
        })
        .expect("seek");
    let Landing::Packet { time, .. } = landing.landing else {
        panic!("landing");
    };
    assert!(time <= 5596 && 5596 < time + 4096);
    assert!(read_all(&mut demux).len() < packets.len());
}
