//! Apple H.264 and AAC export. The live preview readers stay with the window.
//! This pass opens its own decoder so a play can keep running.

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};

use cutlass_audiotoolbox::Decoder as AacDecoder;
use cutlass_audiotoolbox::Encoder as AacEncoder;
use cutlass_color::{Matrix, YuvPlanes, to_rgba};
use cutlass_core::{
    AudioCodec, AudioSamples, AudioStreamConfig, ColorRange, ColorTags, Packet, PixelFormat,
    Timestamp, VideoCodec, VideoFrame, VideoStreamConfig,
};
use cutlass_engine::{MixClip, cues, file_path, mix, pieces, ticks_to_samples};
use cutlass_media::{Content, Probe, VideoSession};
use cutlass_mp4::MovieFile;
use cutlass_playback::{Ahead, Decode};
use cutlass_project::Project;
use cutlass_videotoolbox::{Encoder as H264Encoder, EncoderConfig, PixelBuffer};

use crate::Error;
use crate::picture::{self, Plate, Sprite};

const KEYFRAME_EVERY: usize = 8;
const SOUND_CHANNELS: u16 = 2;

struct Session(VideoSession);

impl Decode for Session {
    type Image = PixelBuffer;
    type Error = cutlass_media::Error;

    fn send(&mut self, packet: &Packet) -> Result<(), Self::Error> {
        self.0.send_packet(packet)
    }

    fn receive(&mut self) -> Result<Option<VideoFrame<PixelBuffer>>, Self::Error> {
        self.0.receive_frame()
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.0.flush()
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.0.reset()
    }
}

struct Film {
    packets: Vec<Packet>,
    session: Session,
    ahead: Ahead<PixelBuffer>,
}

struct Tape {
    rate: u32,
    channels: u16,
    planar: Vec<f32>,
}

struct PictureTrack {
    scale: NonZeroU32,
    width: u32,
    height: u32,
    config: Vec<u8>,
    packets: Vec<Packet>,
}

struct SoundTrack {
    scale: NonZeroU32,
    config: Vec<u8>,
    packets: Vec<Packet>,
}

pub(crate) fn file(project: &Project, path: &Path) -> Result<(), Error> {
    let duration = project.duration().value;
    let rate = project.settings().rate().rate();
    let times = picture::frame_times(duration, rate);
    let picture = if times.is_empty() {
        None
    } else {
        let (width, height) =
            picture::encode_size(project.settings().width(), project.settings().height())
                .ok_or(Error::Canvas)?;
        Some(encode_picture(project, width, height, &times)?)
    };
    let sound = encode_sound(project, duration)?;
    if picture.is_none() && sound.is_none() {
        return Err(Error::Empty);
    }
    let bytes = mux(picture.as_ref(), sound.as_ref())?;
    write_bytes(path, &bytes)
}

fn encode_picture(
    project: &Project,
    width: u32,
    height: u32,
    times: &[i64],
) -> Result<PictureTrack, Error> {
    let rate = project.settings().rate().rate();
    let scale = match NonZeroU32::new(rate.numerator()) {
        Some(scale) => scale,
        None => return Err(Error::Picture("the frame rate is zero".into())),
    };
    let ticks = i64::from(rate.denominator());
    let frames_per_second = f64::from(rate.numerator()) / f64::from(rate.denominator());
    let mut encoder = H264Encoder::new(&EncoderConfig {
        codec: VideoCodec::H264,
        width,
        height,
        pixel_format: PixelFormat::Nv12,
        color_range: ColorRange::Limited,
        bitrate: Some(bitrate(width, height, frames_per_second)),
        keyframe_interval: Some(u32::try_from(KEYFRAME_EVERY).unwrap_or(8)),
        b_frames: false,
        frame_rate: Some(frames_per_second),
        real_time: false,
        require_hardware: false,
    })
    .map_err(|error| Error::Picture(error.to_string()))?;
    let background = project.settings().background().channels();
    let mut films = HashMap::new();
    let mut probes = HashMap::new();
    for (index, time) in times.iter().copied().enumerate() {
        let mut plate = picture::plate(width, height, background);
        for shot in picture::shots(project, time) {
            let film = film(&mut films, &mut probes, shot.source)?;
            let Some((pixels, picture_width, picture_height)) = pull(film, shot.time)? else {
                continue;
            };
            picture::paint(
                &mut plate,
                &Sprite {
                    pixels: &pixels,
                    width: picture_width,
                    height: picture_height,
                    media: shot.media,
                    position: shot.position,
                    scale: shot.scale,
                    rotation: shot.rotation,
                    opacity: shot.opacity,
                    flip_horizontal: shot.flip_horizontal,
                    flip_vertical: shot.flip_vertical,
                },
            );
        }
        let pts = i64::try_from(index).unwrap_or(0).saturating_mul(ticks);
        send_plate(
            &mut encoder,
            &plate,
            Timestamp::new(pts, scale),
            Timestamp::new(ticks, scale),
            index.is_multiple_of(KEYFRAME_EVERY),
        )?;
    }
    encoder
        .flush()
        .map_err(|error| Error::Picture(error.to_string()))?;
    let mut packets = Vec::new();
    while let Some(packet) = encoder.receive_packet() {
        packets.push(packet.map_err(|error| Error::Picture(error.to_string()))?);
    }
    if packets.len() != times.len() {
        return Err(Error::Picture(format!(
            "the encoder returned {} frames for {} pictures",
            packets.len(),
            times.len()
        )));
    }
    stamp(&mut packets, scale, ticks);
    let config = encoder
        .config_record()
        .ok_or_else(|| Error::Picture("the encoder gave no config record".into()))?
        .to_vec();
    Ok(PictureTrack {
        scale,
        width,
        height,
        config,
        packets,
    })
}

fn film<'a>(
    films: &'a mut HashMap<String, Film>,
    probes: &mut HashMap<String, Probe>,
    source: &str,
) -> Result<&'a mut Film, Error> {
    if !films.contains_key(source) {
        let probe = load(probes, source).map_err(Error::Picture)?;
        films.insert(source.to_owned(), open_film(probe)?);
    }
    match films.get_mut(source) {
        Some(film) => Ok(film),
        None => Err(Error::Picture(format!("{source} was not opened"))),
    }
}

fn open_film(probe: &Probe) -> Result<Film, Error> {
    let index = probe
        .tracks()
        .iter()
        .position(|track| track.enabled && matches!(track.content, Content::Video { .. }))
        .ok_or_else(|| Error::Picture("the file has no picture".into()))?;
    let packets = probe
        .samples(index)
        .ok_or_else(|| Error::Picture("the file has no picture".into()))?
        .to_vec();
    if packets.is_empty() {
        return Err(Error::Picture("the file has no picture".into()));
    }
    let session = probe
        .open_video(index)
        .map_err(|error| Error::Picture(error.to_string()))?;
    let depth = match NonZeroUsize::new(8) {
        Some(depth) => depth,
        None => return Err(Error::Picture("the queue has no depth".into())),
    };
    Ok(Film {
        packets,
        session: Session(session),
        ahead: Ahead::new(depth),
    })
}

fn pull(film: &mut Film, time: Timestamp) -> Result<Option<(Vec<u8>, u32, u32)>, Error> {
    film.ahead
        .seek_to(&mut film.session, &film.packets, time)
        .map_err(|error| Error::Picture(error.to_string()))?;
    let Some(frame) = film.ahead.at(time) else {
        return Ok(None);
    };
    let color = frame.color;
    let image = frame.image.clone();
    let width = u32::try_from(image.width())
        .map_err(|_| Error::Picture("the picture is too wide".into()))?;
    let height = u32::try_from(image.height())
        .map_err(|_| Error::Picture("the picture is too tall".into()))?;
    if width == 0 || height == 0 {
        return Err(Error::Picture("the picture has no size".into()));
    }
    let pixels = rgba_of(&image, color, width, height)?;
    Ok(Some((pixels, width, height)))
}

fn rgba_of(
    image: &PixelBuffer,
    color: Option<ColorTags>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Error> {
    let (format, range) = image
        .format()
        .ok_or_else(|| Error::Picture("the picture has no pixel format".into()))?;
    let matrix = Matrix::resolve(color.map(|tags| tags.matrix), width, height)
        .unwrap_or_else(|_| Matrix::for_size(width, height));
    image
        .read_planes(|planes| {
            let Some(y) = planes.first() else {
                return Err(Error::Picture("the picture has no luma".into()));
            };
            let Some(chroma) = planes.get(1) else {
                return Err(Error::Picture("the picture has no chroma".into()));
            };
            to_rgba(&YuvPlanes {
                format,
                range,
                matrix,
                width,
                height,
                y: y.data,
                y_stride: y.bytes_per_row,
                chroma: chroma.data,
                chroma_stride: chroma.bytes_per_row,
            })
            .map_err(|error| Error::Picture(error.to_string()))
        })
        .map_err(|error| Error::Picture(error.to_string()))?
}

fn send_plate(
    encoder: &mut H264Encoder,
    plate: &Plate,
    pts: Timestamp,
    duration: Timestamp,
    keyframe: bool,
) -> Result<(), Error> {
    let mut buffer = encoder
        .create_pixel_buffer()
        .map_err(|error| Error::Picture(error.to_string()))?;
    buffer
        .write_planes(|planes| write_nv12(plate, planes))
        .map_err(|error| Error::Picture(error.to_string()))??;
    encoder
        .send_frame(buffer, pts, Some(duration), keyframe)
        .map_err(|error| Error::Picture(error.to_string()))
}

fn write_nv12(
    plate: &Plate,
    planes: &mut [cutlass_videotoolbox::PlaneMut<'_>],
) -> Result<(), Error> {
    let Some(y_plane) = planes.first_mut() else {
        return Err(Error::Picture("the encoder gave no luma plane".into()));
    };
    for row in 0..plate.height {
        for col in 0..plate.width {
            let (red, green, blue) = rgb(plate, col, row)?;
            let (y, _, _) = ycbcr(red, green, blue);
            put(y_plane.data, y_plane.bytes_per_row, row, col as usize, y)?;
        }
    }
    let Some(chroma) = planes.get_mut(1) else {
        return Err(Error::Picture("the encoder gave no chroma plane".into()));
    };
    let rows = plate.height / 2;
    let cols = plate.width / 2;
    for row in 0..rows {
        for col in 0..cols {
            let mut red = 0u32;
            let mut green = 0u32;
            let mut blue = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (r, g, b) = rgb(plate, col * 2 + dx, row * 2 + dy)?;
                    red += u32::from(r);
                    green += u32::from(g);
                    blue += u32::from(b);
                }
            }
            let (_, cb, cr) = ycbcr(
                u8::try_from(red / 4).unwrap_or(0),
                u8::try_from(green / 4).unwrap_or(0),
                u8::try_from(blue / 4).unwrap_or(0),
            );
            let at = col as usize * 2;
            put(chroma.data, chroma.bytes_per_row, row, at, cb)?;
            put(chroma.data, chroma.bytes_per_row, row, at + 1, cr)?;
        }
    }
    Ok(())
}

fn rgb(plate: &Plate, col: u32, row: u32) -> Result<(u8, u8, u8), Error> {
    let index = (row as usize * plate.width as usize + col as usize) * 4;
    let Some(pixel) = plate.pixels.get(index..index + 3) else {
        return Err(Error::Picture("a canvas pixel is missing".into()));
    };
    Ok((pixel[0], pixel[1], pixel[2]))
}

fn put(plane: &mut [u8], stride: usize, row: u32, col: usize, value: u8) -> Result<(), Error> {
    let index = (row as usize).saturating_mul(stride).saturating_add(col);
    let Some(slot) = plane.get_mut(index) else {
        return Err(Error::Picture("an encoder plane is short".into()));
    };
    *slot = value;
    Ok(())
}

/// Rec. 709 limited. The canvas RGB is encoded as that and the file is tagged
/// the same way. The source transfer is not tone-mapped.
fn ycbcr(red: u8, green: u8, blue: u8) -> (u8, u8, u8) {
    let red = f64::from(red);
    let green = f64::from(green);
    let blue = f64::from(blue);
    let luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    let y = 16.0 + luma * 219.0 / 255.0;
    let cb = 128.0 + (blue - luma) / 1.8556 * 224.0 / 255.0;
    let cr = 128.0 + (red - luma) / 1.5748 * 224.0 / 255.0;
    (byte(y), byte(cb), byte(cr))
}

fn byte(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn bitrate(width: u32, height: u32, frames_per_second: f64) -> u32 {
    let bits = f64::from(width) * f64::from(height) * frames_per_second * 0.08;
    let bits = bits.clamp(2_000_000.0, 20_000_000.0).round();
    if bits.is_finite() && bits > 0.0 {
        bits as u32
    } else {
        2_000_000
    }
}

fn encode_sound(project: &Project, duration_us: i64) -> Result<Option<SoundTrack>, Error> {
    let found = cues(project);
    if found.is_empty() || duration_us <= 0 {
        return Ok(None);
    }
    let mut probes = HashMap::new();
    let mut tapes = HashMap::new();
    let mut rate = 0u32;
    for cue in &found {
        if tapes.contains_key(cue.source()) {
            continue;
        }
        let probe = load(&mut probes, cue.source()).map_err(Error::Sound)?;
        let tape = open_tape(probe)?;
        if tape.rate == 0 {
            return Err(Error::Sound("the sound has no sample rate".into()));
        }
        if rate == 0 {
            rate = tape.rate;
        } else if tape.rate != rate {
            return Err(Error::Sound(
                "the sound rates differ, and resampling is not in yet".into(),
            ));
        }
        tapes.insert(cue.source().to_owned(), tape);
    }
    let scale = match NonZeroU32::new(rate) {
        Some(scale) => scale,
        None => return Err(Error::Sound("the sound has no sample rate".into())),
    };
    let mut encoder =
        AacEncoder::new(rate, SOUND_CHANNELS).map_err(|error| Error::Sound(error.to_string()))?;
    let mut stored = Vec::new();
    let mut cursor = 0i64;
    while cursor < duration_us {
        let until = duration_us.min(cursor.saturating_add(1_000_000));
        let asks = pieces(project, &found, cursor, until);
        let mut clips = Vec::new();
        for ask in &asks {
            let Some(tape) = tapes.get(&ask.source) else {
                return Err(Error::Sound(format!("{} was not opened", ask.source)));
            };
            clips.push(window(tape, ask)?);
        }
        let samples = mix(&clips, cursor, until, rate, SOUND_CHANNELS);
        if samples.frames() > 0 {
            let packets = encoder
                .encode(&samples)
                .map_err(|error| Error::Sound(error.to_string()))?;
            stored.extend(packets);
        }
        cursor = until;
    }
    stored.extend(
        encoder
            .finish()
            .map_err(|error| Error::Sound(error.to_string()))?,
    );
    if stored.is_empty() {
        return Err(Error::Sound("the encoder gave no sound".into()));
    }
    let mut packets = Vec::with_capacity(stored.len());
    let mut next = 0i64;
    for packet in stored {
        packets.push(Packet {
            data: packet.data,
            pts: Timestamp::new(next, scale),
            dts: Some(Timestamp::new(next, scale)),
            duration: Some(Timestamp::new(1024, scale)),
            keyframe: true,
        });
        next += 1024;
    }
    let config = encoder.config_record().to_vec();
    Ok(Some(SoundTrack {
        scale,
        config,
        packets,
    }))
}

fn open_tape(probe: &Probe) -> Result<Tape, Error> {
    let index = probe
        .tracks()
        .iter()
        .position(|track| {
            track.enabled
                && matches!(&track.content, Content::Audio(config) if cutlass_audiotoolbox::decodes(config.codec))
        })
        .ok_or_else(|| Error::Sound("the file has no AAC sound".into()))?;
    let Content::Audio(config) = &probe.tracks()[index].content else {
        return Err(Error::Sound("the file has no AAC sound".into()));
    };
    let packets = probe
        .samples(index)
        .ok_or_else(|| Error::Sound("the file has no AAC sound".into()))?;
    let mut decoder = AacDecoder::new(config).map_err(|error| Error::Sound(error.to_string()))?;
    let channels = usize::from(config.channels);
    let mut planes = vec![Vec::new(); channels];
    for packet in packets {
        extend_planes(
            &mut planes,
            &decoder
                .decode(&packet.data)
                .map_err(|error| Error::Sound(error.to_string()))?,
        )?;
    }
    extend_planes(
        &mut planes,
        &decoder
            .flush()
            .map_err(|error| Error::Sound(error.to_string()))?,
    )?;
    let frames = planes.first().map(Vec::len).unwrap_or(0);
    let mut planar = Vec::with_capacity(frames.saturating_mul(channels));
    for plane in &planes {
        if plane.len() != frames {
            return Err(Error::Sound("a sound channel is short".into()));
        }
        planar.extend_from_slice(plane);
    }
    Ok(Tape {
        rate: config.sample_rate,
        channels: config.channels,
        planar,
    })
}

fn extend_planes(planes: &mut [Vec<f32>], samples: &AudioSamples) -> Result<(), Error> {
    for (index, plane) in planes.iter_mut().enumerate() {
        let Some(channel) = u16::try_from(index)
            .ok()
            .and_then(|channel| samples.plane(channel))
        else {
            return Err(Error::Sound("a sound channel is missing".into()));
        };
        plane.extend_from_slice(channel);
    }
    Ok(())
}

fn window(tape: &Tape, piece: &cutlass_engine::Piece) -> Result<MixClip, Error> {
    let Some(first) = ticks_to_samples(piece.first_tick, piece.tick_scale, tape.rate) else {
        return Err(Error::Sound("the sound window does not fit".into()));
    };
    let Some(last) = ticks_to_samples(piece.last_tick, piece.tick_scale, tape.rate) else {
        return Err(Error::Sound("the sound window does not fit".into()));
    };
    let count = usize::try_from(last.saturating_sub(first))
        .map_err(|_| Error::Sound("the sound window does not fit".into()))?;
    let channels = usize::from(tape.channels);
    let frames = tape.planar.len().checked_div(channels).unwrap_or(0);
    let start = usize::try_from(first).unwrap_or(usize::MAX);
    let mut planar = vec![0.0; count.saturating_mul(channels)];
    if start < frames && channels > 0 {
        let take = count.min(frames - start);
        for channel in 0..channels {
            let from = channel * frames + start;
            let into = channel * count;
            let Some(src) = tape.planar.get(from..from + take) else {
                continue;
            };
            let Some(dst) = planar.get_mut(into..into + take) else {
                continue;
            };
            dst.copy_from_slice(src);
        }
    }
    Ok(MixClip::new(
        piece.origin_us,
        piece.span_us,
        piece.volume as f32,
        tape.channels,
        planar,
    ))
}

fn load<'a>(cache: &'a mut HashMap<String, Probe>, source: &str) -> Result<&'a Probe, String> {
    if !cache.contains_key(source) {
        let Some(path) = file_path(source) else {
            return Err(format!("{source} is not a file"));
        };
        let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
        let probe = Probe::parse(&bytes).map_err(|error| format!("{path}: {error}"))?;
        cache.insert(source.to_owned(), probe);
    }
    match cache.get(source) {
        Some(probe) => Ok(probe),
        None => Err(format!("{source} was not opened")),
    }
}

fn stamp(packets: &mut [Packet], scale: NonZeroU32, ticks: i64) {
    let mut next = 0i64;
    for packet in packets {
        packet.pts = Timestamp::new(next, scale);
        packet.dts = Some(Timestamp::new(next, scale));
        packet.duration = Some(Timestamp::new(ticks, scale));
        next += ticks;
    }
}

fn mux(picture: Option<&PictureTrack>, sound: Option<&SoundTrack>) -> Result<Vec<u8>, Error> {
    let movie_scale = match picture {
        Some(picture) => picture.scale,
        None => match sound {
            Some(sound) => sound.scale,
            None => return Err(Error::Empty),
        },
    };
    let mut movie = MovieFile::new(movie_scale);
    if let Some(picture) = picture {
        let config = VideoStreamConfig {
            codec: VideoCodec::H264,
            width: picture.width,
            height: picture.height,
            config_record: picture.config.clone(),
        };
        let color = ColorTags {
            primaries: 1,
            transfer: 1,
            matrix: 1,
            full_range: Some(false),
        };
        movie
            .add_video_color(picture.scale, config, color, picture.packets.clone())
            .map_err(|error| Error::Picture(error.to_string()))?;
    }
    if let Some(sound) = sound {
        let mut packets = sound.packets.clone();
        if let Some(picture) = picture {
            pad_sound(&mut packets, sound.scale, picture.scale);
        }
        let config = AudioStreamConfig {
            codec: AudioCodec::Aac,
            sample_rate: sound.scale.get(),
            channels: SOUND_CHANNELS,
            config_record: sound.config.clone(),
        };
        movie
            .add_audio(sound.scale, config, packets)
            .map_err(|error| Error::Sound(error.to_string()))?;
    }
    movie
        .to_bytes()
        .map_err(|error| Error::File(error.to_string()))
}

fn pad_sound(packets: &mut [Packet], audio_scale: NonZeroU32, movie_scale: NonZeroU32) {
    if packets.is_empty() || audio_scale == movie_scale {
        return;
    }
    let samples = packets
        .iter()
        .filter_map(|packet| packet.duration)
        .map(|time| u64::try_from(time.value).unwrap_or(0))
        .sum::<u64>();
    let audio = u64::from(audio_scale.get());
    let movie = u64::from(movie_scale.get());
    let mut extra = 0u64;
    while extra < audio {
        if samples.saturating_add(extra).saturating_mul(movie) % audio == 0 {
            break;
        }
        extra += 1;
    }
    let Some(last) = packets.last_mut() else {
        return;
    };
    let Some(duration) = &mut last.duration else {
        return;
    };
    duration.value = duration
        .value
        .saturating_add(i64::try_from(extra).unwrap_or(0));
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut partial = path.as_os_str().to_os_string();
    partial.push(".part");
    let partial = PathBuf::from(partial);
    std::fs::write(&partial, bytes).map_err(|error| Error::File(error.to_string()))?;
    if let Err(error) = std::fs::rename(&partial, path) {
        let _ = std::fs::remove_file(&partial);
        return Err(Error::File(error.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::num::NonZeroUsize;

    use cutlass_core::{Packet, Timestamp};
    use cutlass_media::{Content, Probe};
    use cutlass_mp4::MovieFile;
    use cutlass_playback::Ahead;
    use cutlass_project::{
        AspectRatio, Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment,
        Speed, TimeRange,
    };

    use super::Session;
    use crate::picture::frame_times;
    use crate::write;

    #[test]
    fn a_fixture_cut_becomes_an_h264_file() {
        let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
        let mut packets = fixture.packets;
        let shift = packets
            .iter()
            .filter_map(|packet| packet.dts)
            .map(|time| time.value)
            .min()
            .expect("dts");
        for packet in &mut packets {
            packet.pts.value -= shift;
            if let Some(time) = &mut packet.dts {
                time.value -= shift;
            }
        }
        let scale = packets[0].pts.timescale;
        let end = packets
            .iter()
            .map(|packet| {
                packet.pts.value + packet.duration.map(|duration| duration.value).unwrap_or(0)
            })
            .max()
            .expect("end");
        let start = packets
            .iter()
            .map(|packet| packet.pts.value)
            .min()
            .expect("pts");
        let dir = std::env::temp_dir().join(format!("cutlass-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp");
        let source = dir.join("source.mp4");
        let output = dir.join("out.mp4");
        std::fs::write(
            &source,
            source_file(&fixture.config, scale, &packets).expect("source"),
        )
        .expect("write source");
        let mut project = Project::new(ProjectId::from_bytes([4; 16]), "Cut");
        let mut history = History::new();
        let Some(source_text) = source.to_str() else {
            panic!("the temp path is not text");
        };
        let asset =
            Asset::video(source_text, Timestamp::new(end, scale), 128, 72, false).expect("asset");
        let added = history
            .apply(&mut project, &[Command::AddAsset(asset)])
            .expect("add");
        let Outcome::Asset(id) = added[0] else {
            panic!("the asset was not stored");
        };
        let stored = project.asset(id).expect("stored").clone();
        let range = TimeRange::new(
            Timestamp::new(start, scale),
            Timestamp::new(end - start, scale),
        )
        .expect("range");
        let segment =
            Segment::media(&stored, range, 0, Channels::Picture, Speed::ONE).expect("segment");
        let main = project.timeline().main_track().id();
        history
            .apply(
                &mut project,
                &[
                    Command::SetCanvas {
                        width: 128,
                        height: 72,
                        ratio: AspectRatio::Original,
                    },
                    Command::Place {
                        track: Some(main),
                        segment,
                    },
                ],
            )
            .expect("place");
        write::file(&project, &output).expect("export");
        let bytes = std::fs::read(&output).expect("read export");
        let probe = Probe::parse(&bytes).expect("probe");
        let video = probe
            .tracks()
            .iter()
            .position(|track| matches!(track.content, Content::Video { .. }))
            .expect("video track");
        let samples = probe.samples(video).expect("samples");
        let expected = frame_times(project.duration().value, project.settings().rate().rate());
        assert_eq!(samples.len(), expected.len());
        assert!(picture_varies(&probe, video, samples));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn source_file(
        config: &cutlass_core::VideoStreamConfig,
        scale: NonZeroU32,
        packets: &[Packet],
    ) -> Result<Vec<u8>, cutlass_mp4::Error> {
        let mut movie = MovieFile::new(scale);
        movie.add_video(scale, config.clone(), packets.to_vec())?;
        movie.to_bytes()
    }

    fn picture_varies(probe: &Probe, index: usize, packets: &[Packet]) -> bool {
        let Ok(session) = probe.open_video(index) else {
            return false;
        };
        let mut session = Session(session);
        let depth = match NonZeroUsize::new(4) {
            Some(depth) => depth,
            None => return false,
        };
        let mut ahead = Ahead::new(depth);
        let time = packets[0].pts;
        if ahead.seek_to(&mut session, packets, time).is_err() {
            return false;
        }
        let Some(frame) = ahead.at(time) else {
            return false;
        };
        let Ok(varied) = frame.image.read_planes(|planes| {
            let Some(y) = planes.first() else {
                return false;
            };
            let mut low = 255u8;
            let mut high = 0u8;
            for row in 0..y.height {
                let start = row * y.bytes_per_row;
                let Some(end) = start.checked_add(y.width) else {
                    continue;
                };
                let Some(line) = y.data.get(start..end) else {
                    continue;
                };
                for sample in line {
                    low = low.min(*sample);
                    high = high.max(*sample);
                }
            }
            high > low
        }) else {
            return false;
        };
        varied
    }
}
