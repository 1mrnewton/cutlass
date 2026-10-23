//! The sound under the playhead.
//!
//! On Apple, play decodes the audible clips into the output device at their
//! volume and the picture follows that clock. A failure is written to the
//! app log, and the cut stays on the wall clock. A fade, a mute, and a scrub
//! are not heard. Other systems keep the wall clock.

use cutlass_engine::Lead;
use cutlass_project::Project;

#[cfg(target_vendor = "apple")]
use std::collections::{HashMap, HashSet};
#[cfg(target_vendor = "apple")]
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
#[cfg(target_vendor = "apple")]
use std::thread::JoinHandle;
#[cfg(target_vendor = "apple")]
use std::time::Duration;

#[cfg(target_vendor = "apple")]
use cutlass_audiotoolbox::{Decoder, decodes};
#[cfg(target_vendor = "apple")]
use cutlass_core::{AudioStreamConfig, Packet};
#[cfg(target_vendor = "apple")]
use cutlass_engine::{Cue, MixClip, Piece, clock_us, cues, mix, pieces, ticks_to_samples};
#[cfg(target_vendor = "apple")]
use cutlass_media::{Content, Probe};
#[cfg(target_vendor = "apple")]
use cutlass_output::Output;
#[cfg(target_vendor = "apple")]
use cutlass_project::FrameRate;

/// How much timeline time one fill covers.
#[cfg(target_vendor = "apple")]
const CHUNK_US: i64 = 250_000;
/// Keep at least this much audio queued ahead of the clock.
#[cfg(target_vendor = "apple")]
const AHEAD_US: i64 = 400_000;
/// The device is opened stereo. A mono clip is copied to both sides.
#[cfg(target_vendor = "apple")]
const OUT_CHANNELS: u16 = 2;

/// Plays the audible clips, or stays quiet when this system has no output.
pub(crate) struct Hear {
    #[cfg(target_vendor = "apple")]
    live: Live,
}

impl Hear {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(target_vendor = "apple")]
            live: Live::new(),
        }
    }

    /// Starts hearing at `origin_us`. A cut with no audible clip stays on the wall clock.
    pub(crate) fn start(&mut self, project: &Project, origin_us: i64) {
        #[cfg(target_vendor = "apple")]
        self.live.start(project, origin_us);
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (project, origin_us);
        }
    }

    /// Stops the device. A second call does nothing.
    pub(crate) fn stop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.live.stop();
    }

    /// Feeds the device and reports where the playhead belongs.
    pub(crate) fn follow(
        &mut self,
        project: &Project,
        duration_us: i64,
        rate: cutlass_project::FrameRate,
    ) -> Lead {
        #[cfg(target_vendor = "apple")]
        {
            self.live.follow(project, duration_us, rate)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (project, duration_us, rate);
            Lead::Silent
        }
    }
}

#[cfg(target_vendor = "apple")]
struct Live {
    bridge: Option<Bridge>,
    output: Option<Output>,
    generation: u64,
    origin_us: i64,
    end_us: i64,
    filled_us: i64,
    rate: u32,
    waiting: bool,
    cues: Vec<Cue>,
    mode: Mode,
}

#[cfg(target_vendor = "apple")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Idle,
    Silent,
    Live,
}

#[cfg(target_vendor = "apple")]
impl Live {
    fn new() -> Self {
        Self {
            bridge: None,
            output: None,
            generation: 0,
            origin_us: 0,
            end_us: 0,
            filled_us: 0,
            rate: 0,
            waiting: false,
            cues: Vec::new(),
            mode: Mode::Idle,
        }
    }

    fn start(&mut self, project: &Project, origin_us: i64) {
        self.stop();
        let found = cues(project);
        if found.is_empty() {
            self.mode = Mode::Silent;
            return;
        }
        self.ensure();
        if self.bridge.is_none() {
            log::warn!("The audio thread did not start. The cut stays on the wall clock.");
            self.mode = Mode::Silent;
            return;
        }
        self.generation = self.generation.wrapping_add(1);
        self.origin_us = origin_us;
        self.end_us = project.duration().value;
        self.filled_us = origin_us;
        self.rate = 0;
        self.cues = found;
        self.mode = Mode::Live;
        self.waiting = false;
        if !self.post(project) {
            self.mode = Mode::Silent;
        }
    }

    fn stop(&mut self) {
        if self.output.is_none() && !self.waiting && self.mode != Mode::Live {
            self.mode = Mode::Idle;
            return;
        }
        self.generation = self.generation.wrapping_add(1);
        self.mode = Mode::Idle;
        self.waiting = false;
        self.cues.clear();
        self.rate = 0;
        self.filled_us = 0;
        if let Some(output) = &mut self.output {
            let _ = output.stop();
        }
        self.output = None;
        if let Some(bridge) = &self.bridge {
            bridge.post(Message::Stop);
        }
    }

    fn follow(&mut self, project: &Project, duration_us: i64, rate: FrameRate) -> Lead {
        if self.mode == Mode::Silent {
            return Lead::Silent;
        }
        if self.mode != Mode::Live {
            return Lead::Silent;
        }
        self.end_us = duration_us;
        self.pump(project);
        if self.mode == Mode::Silent {
            return Lead::Silent;
        }
        self.lead(duration_us, rate)
    }

    fn ensure(&mut self) {
        if self.bridge.is_none() {
            self.bridge = Bridge::start();
        }
    }

    fn pump(&mut self, project: &Project) {
        let answer = self.bridge.as_ref().and_then(Bridge::take);
        if let Some(answer) = answer {
            self.waiting = false;
            self.take(answer);
        }
        if self.mode != Mode::Live || self.waiting || self.filled_us >= self.end_us {
            return;
        }
        let ahead = self.filled_us.saturating_sub(self.heard_us());
        if self.output.is_none() || ahead < AHEAD_US {
            self.post(project);
        }
    }

    fn post(&mut self, project: &Project) -> bool {
        let until = self.end_us.min(self.filled_us.saturating_add(CHUNK_US));
        if until <= self.filled_us {
            return false;
        }
        let Some(bridge) = &self.bridge else {
            return false;
        };
        let fill = Fill {
            generation: self.generation,
            from_us: self.filled_us,
            until_us: until,
            sources: self
                .cues
                .iter()
                .map(|cue| cue.source().to_owned())
                .collect(),
            pieces: pieces(project, &self.cues, self.filled_us, until),
        };
        let posted = bridge.post(Message::Fill(fill));
        self.waiting = posted;
        posted
    }

    fn take(&mut self, answer: Answer) {
        let (generation, from_us, until_us, rate, clips) = match answer {
            Answer::Missing { generation } => {
                if generation == self.generation && self.output.is_none() {
                    self.mode = Mode::Silent;
                }
                return;
            }
            Answer::Ready {
                generation,
                from_us,
                until_us,
                rate,
                clips,
            } => (generation, from_us, until_us, rate, clips),
        };
        if generation != self.generation || self.mode != Mode::Live {
            return;
        }
        if self.rate == 0 {
            self.rate = rate;
        }
        if self.rate == 0 {
            log::warn!("The audio file reported no sample rate. The cut stays on the wall clock.");
            self.mode = Mode::Silent;
            return;
        }
        let samples = mix(&clips, from_us, until_us, self.rate, OUT_CHANNELS);
        if samples.frames() == 0 {
            self.filled_us = until_us;
            return;
        }
        if self.output.is_none() {
            match Output::new(self.rate, OUT_CHANNELS) {
                Ok(mut output) => match output.play(&samples) {
                    Ok(()) => self.output = Some(output),
                    Err(error) => {
                        log::warn!(
                            "The audio device did not start: {error}. The cut stays on the wall clock."
                        );
                        self.mode = Mode::Silent;
                        return;
                    }
                },
                Err(error) => {
                    log::warn!(
                        "The audio device did not open: {error}. The cut stays on the wall clock."
                    );
                    self.mode = Mode::Silent;
                    return;
                }
            }
        } else if let Some(output) = &mut self.output
            && let Err(error) = output.extend(&samples)
        {
            log::warn!(
                "The audio device stopped taking samples: {error}. The cut stays on the wall clock."
            );
            let _ = output.stop();
            self.output = None;
            self.mode = Mode::Silent;
            return;
        }
        self.filled_us = until_us;
    }

    fn heard_us(&self) -> i64 {
        let Some(output) = &self.output else {
            return self.origin_us;
        };
        match output.clock() {
            Ok(Some(time)) => clock_us(self.origin_us, time),
            _ => self.origin_us,
        }
    }

    fn lead(&self, duration_us: i64, rate: FrameRate) -> Lead {
        let Some(output) = &self.output else {
            return Lead::Waiting;
        };
        let Ok(Some(time)) = output.clock() else {
            return Lead::Waiting;
        };
        let at = clock_us(self.origin_us, time);
        let elapsed = u64::try_from(at.saturating_sub(self.origin_us).max(0)).unwrap_or(u64::MAX);
        Lead::from(cutlass_engine::advance(
            self.origin_us,
            Duration::from_micros(elapsed),
            duration_us,
            rate,
        ))
    }
}

#[cfg(target_vendor = "apple")]
enum Message {
    Stop,
    Fill(Fill),
}

#[cfg(target_vendor = "apple")]
struct Fill {
    generation: u64,
    from_us: i64,
    until_us: i64,
    sources: Vec<String>,
    pieces: Vec<Piece>,
}

#[cfg(target_vendor = "apple")]
enum Answer {
    Missing {
        generation: u64,
    },
    Ready {
        generation: u64,
        from_us: i64,
        until_us: i64,
        rate: u32,
        clips: Vec<MixClip>,
    },
}

#[cfg(target_vendor = "apple")]
struct Bridge {
    ask: Arc<Note<Message>>,
    sound: Arc<Slot<Answer>>,
    join: Option<JoinHandle<()>>,
}

#[cfg(target_vendor = "apple")]
impl Bridge {
    fn start() -> Option<Self> {
        let ask = Arc::new(Note::new());
        let sound = Arc::new(Slot::new());
        let ask_thread = Arc::clone(&ask);
        let sound_thread = Arc::clone(&sound);
        let join = std::thread::Builder::new()
            .name("cutlass-hear".to_owned())
            .spawn(move || hear_loop(ask_thread, sound_thread))
            .ok();
        join.is_some().then_some(Self { ask, sound, join })
    }

    fn post(&self, message: Message) -> bool {
        self.ask.post(message);
        true
    }

    fn take(&self) -> Option<Answer> {
        self.sound.take()
    }
}

#[cfg(target_vendor = "apple")]
impl Drop for Bridge {
    fn drop(&mut self) {
        self.ask.close();
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(target_vendor = "apple")]
fn hear_loop(ask: Arc<Note<Message>>, sound: Arc<Slot<Answer>>) {
    let mut worker = Worker::default();
    while let Some(message) = ask.wait() {
        match message {
            Message::Stop => {
                worker.tapes.clear();
                worker.layout = None;
            }
            Message::Fill(fill) => {
                let answer = worker.fill(fill);
                sound.publish(answer);
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Default)]
struct Worker {
    generation: u64,
    layout: Option<u32>,
    files: HashMap<String, Opened>,
    closed: HashSet<String>,
    tapes: HashMap<(String, i64), Tape>,
}

#[cfg(target_vendor = "apple")]
impl Worker {
    fn fill(&mut self, fill: Fill) -> Answer {
        if self.generation != fill.generation {
            self.tapes.clear();
            self.layout = None;
            self.closed.clear();
            self.generation = fill.generation;
        }
        let Some(rate) = self.rate_of(&fill.sources) else {
            log::warn!("No clip opened for playback. The cut stays on the wall clock.");
            return Answer::Missing {
                generation: fill.generation,
            };
        };
        let mut clips = Vec::new();
        for ask in &fill.pieces {
            if let Some(clip) = self.clip(ask) {
                clips.push(clip);
            }
        }
        Answer::Ready {
            generation: fill.generation,
            from_us: fill.from_us,
            until_us: fill.until_us,
            rate,
            clips,
        }
    }

    fn rate_of(&mut self, sources: &[String]) -> Option<u32> {
        if let Some(rate) = self.layout {
            return Some(rate);
        }
        for source in sources {
            if self.open(source).is_some() {
                return self.layout;
            }
        }
        None
    }

    fn open(&mut self, source: &str) -> Option<()> {
        if self.files.contains_key(source) {
            return Some(());
        }
        if self.closed.contains(source) {
            return None;
        }
        let opened = match read_audio(source) {
            Ok(opened) => opened,
            Err(reason) => {
                log::warn!("{reason}");
                self.closed.insert(source.to_owned());
                return None;
            }
        };
        if self.layout.is_none() {
            self.layout = Some(opened.rate);
        }
        self.files.insert(source.to_owned(), opened);
        Some(())
    }

    fn clip(&mut self, ask: &Piece) -> Option<MixClip> {
        self.open(&ask.source)?;
        let key = (ask.source.clone(), ask.clip_start_us);
        if !self.tapes.contains_key(&key) {
            let opened = self.files.get(&ask.source)?;
            self.tapes.insert(key.clone(), Tape::new(opened));
        }
        let rate = self.files.get(&ask.source)?.rate;
        let first = ticks_to_samples(ask.first_tick, ask.tick_scale, rate)?;
        let last = ticks_to_samples(ask.last_tick, ask.tick_scale, rate)?;
        let count = usize::try_from(last.saturating_sub(first)).ok()?;
        if count == 0 {
            return None;
        }
        let tape = self.tapes.get_mut(&key)?;
        let planar = tape.pull(first, count)?;
        let channels = tape.channels;
        Some(MixClip::new(
            ask.origin_us,
            ask.span_us,
            ask.volume as f32,
            channels,
            planar,
        ))
    }
}

#[cfg(target_vendor = "apple")]
struct Opened {
    path: String,
    config: AudioStreamConfig,
    packets: Vec<Packet>,
    rate: u32,
}

#[cfg(target_vendor = "apple")]
fn read_audio(source: &str) -> Result<Opened, String> {
    let Some(path) = crate::frames::file_path(source) else {
        return Err(format!("Audio is not a file ({source})."));
    };
    let bytes = std::fs::read(path).map_err(|error| format!("Could not read {path}: {error}."))?;
    let probe = Probe::parse(&bytes)
        .map_err(|error| format!("Could not open {path} for audio: {error}."))?;
    let Some(index) = probe.tracks().iter().position(|track| {
        track.enabled && matches!(&track.content, Content::Audio(config) if decodes(config.codec))
    }) else {
        return Err(format!("No AAC track in {path}."));
    };
    let Content::Audio(config) = &probe.tracks()[index].content else {
        return Err(format!("No AAC track in {path}."));
    };
    if config.sample_rate == 0 || config.channels == 0 {
        return Err(format!("The AAC track in {path} has no rate or channels."));
    }
    let Some(packets) = probe.samples(index).map(<[Packet]>::to_vec) else {
        return Err(format!("The AAC track in {path} has no samples."));
    };
    Ok(Opened {
        path: path.to_owned(),
        rate: config.sample_rate,
        config: config.clone(),
        packets,
    })
}

#[cfg(target_vendor = "apple")]
struct Tape {
    path: String,
    config: AudioStreamConfig,
    packets: Vec<Packet>,
    rate: u32,
    channels: u16,
    decoder: Option<Decoder>,
    next_packet: usize,
    pending_at: u64,
    pending: Vec<Vec<f32>>,
    failed: bool,
}

#[cfg(target_vendor = "apple")]
impl Tape {
    fn new(opened: &Opened) -> Self {
        Self {
            path: opened.path.clone(),
            config: opened.config.clone(),
            packets: opened.packets.clone(),
            rate: opened.rate,
            channels: opened.config.channels,
            decoder: None,
            next_packet: 0,
            pending_at: 0,
            pending: Vec::new(),
            failed: false,
        }
    }

    fn pull(&mut self, first: u64, count: usize) -> Option<Vec<f32>> {
        if self.failed || count == 0 || self.channels == 0 {
            return None;
        }
        if !self.can_continue(first) && !self.seek(first) {
            return None;
        }
        let need = first.saturating_add(count as u64);
        let mut steps = 0;
        let limit = self.packets.len().saturating_add(2);
        while self.pending_end() < need && steps < limit {
            steps += 1;
            if !self.decode_next() {
                break;
            }
        }
        let skip = usize::try_from(first.saturating_sub(self.pending_at)).ok()?;
        let channels = usize::from(self.channels);
        let have = self.pending.first().map(Vec::len).unwrap_or(0);
        let mut planar = vec![0f32; count.saturating_mul(channels)];
        let take = count.min(have.saturating_sub(skip));
        for channel in 0..channels {
            let plane = self.pending.get(channel)?;
            if skip > plane.len() {
                continue;
            }
            let end = skip.saturating_add(take).min(plane.len());
            let dest = channel * count;
            planar[dest..dest + (end - skip)].copy_from_slice(&plane[skip..end]);
        }
        let consumed = skip.saturating_add(take);
        for plane in &mut self.pending {
            if consumed < plane.len() {
                plane.drain(..consumed);
            } else {
                plane.clear();
            }
        }
        self.pending_at = first.saturating_add(take as u64);
        Some(planar)
    }

    fn can_continue(&self, first: u64) -> bool {
        self.decoder.is_some() && first >= self.pending_at
    }

    fn pending_end(&self) -> u64 {
        let frames = self.pending.first().map(Vec::len).unwrap_or(0) as u64;
        self.pending_at.saturating_add(frames)
    }

    fn fail(&mut self, reason: String) {
        if self.failed {
            return;
        }
        log::warn!("{reason}");
        self.failed = true;
    }

    fn seek(&mut self, first: u64) -> bool {
        let decoder = match Decoder::new(&self.config) {
            Ok(decoder) => decoder,
            Err(error) => {
                let path = self.path.clone();
                self.fail(format!(
                    "Could not open the AAC decoder for {path}: {error}."
                ));
                return false;
            }
        };
        self.decoder = Some(decoder);
        self.pending = vec![Vec::new(); usize::from(self.channels)];
        let index = self
            .packets
            .iter()
            .rposition(|packet| packet_sample(packet, self.rate) <= first)
            .unwrap_or(0);
        self.next_packet = index;
        self.pending_at = self
            .packets
            .get(index)
            .map(|packet| packet_sample(packet, self.rate))
            .unwrap_or(0);
        true
    }

    fn decode_next(&mut self) -> bool {
        let Some(decoder) = &mut self.decoder else {
            return false;
        };
        let decoded = if self.next_packet < self.packets.len() {
            let packet = &self.packets[self.next_packet];
            self.next_packet += 1;
            decoder.decode(&packet.data)
        } else {
            decoder.flush()
        };
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                let path = self.path.clone();
                self.fail(format!("Could not decode audio in {path}: {error}."));
                return false;
            }
        };
        if decoded.frames() == 0 {
            return self.next_packet < self.packets.len();
        }
        for channel in 0..self.channels {
            let Some(plane) = decoded.plane(channel) else {
                let path = self.path.clone();
                self.fail(format!("The audio in {path} is missing a channel."));
                return false;
            };
            if let Some(pending) = self.pending.get_mut(usize::from(channel)) {
                pending.extend_from_slice(plane);
            }
        }
        true
    }
}

#[cfg(target_vendor = "apple")]
fn packet_sample(packet: &Packet, rate: u32) -> u64 {
    if packet.pts.value <= 0 || rate == 0 {
        return 0;
    }
    let ticks = u128::from(packet.pts.value as u64);
    let scale = u128::from(packet.pts.timescale.get());
    if scale == 0 {
        return 0;
    }
    u64::try_from(ticks.saturating_mul(u128::from(rate)) / scale).unwrap_or(u64::MAX)
}

#[cfg(target_vendor = "apple")]
struct Note<T> {
    state: Mutex<NoteState<T>>,
    wake: Condvar,
}

#[cfg(target_vendor = "apple")]
struct NoteState<T> {
    value: Option<T>,
    closed: bool,
}

#[cfg(target_vendor = "apple")]
impl<T> Note<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new(NoteState {
                value: None,
                closed: false,
            }),
            wake: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, NoteState<T>> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn post(&self, value: T) {
        let mut state = self.lock();
        if state.closed {
            return;
        }
        state.value = Some(value);
        self.wake.notify_one();
    }

    fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        state.value = None;
        self.wake.notify_one();
    }

    fn wait(&self) -> Option<T> {
        let mut state = self.lock();
        loop {
            if let Some(value) = state.value.take() {
                return Some(value);
            }
            if state.closed {
                return None;
            }
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }
}

#[cfg(target_vendor = "apple")]
struct Slot<T> {
    inner: Mutex<Option<T>>,
}

#[cfg(target_vendor = "apple")]
impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn publish(&self, value: T) {
        *self.lock() = Some(value);
    }

    fn take(&self) -> Option<T> {
        self.lock().take()
    }

    fn lock(&self) -> MutexGuard<'_, Option<T>> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use std::num::NonZeroU32;

    fn scale(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("scale")
    }

    #[test]
    fn a_constant_survives_a_pull_from_the_start_and_from_later() {
        use cutlass_audiotoolbox::Encoder;
        use cutlass_core::{AudioSamples, Packet, Timestamp};

        let rate = 48_000u32;
        let frames = 16_384usize;
        let source = AudioSamples::new(rate, 1, vec![0.25; frames]).expect("constant");
        let mut encoder = Encoder::new(rate, 1).expect("encoder");
        let mut encoded = encoder.encode(&source).expect("encode");
        encoded.extend(encoder.finish().expect("finish"));
        let mut packets = Vec::new();
        let mut pts = 0i64;
        for packet in encoded {
            packets.push(Packet {
                data: packet.data,
                pts: Timestamp::new(pts, scale(rate)),
                dts: None,
                duration: Some(Timestamp::new(1_024, scale(rate))),
                keyframe: true,
            });
            pts += 1_024;
        }
        let opened = super::Opened {
            path: "tone.m4a".to_owned(),
            config: cutlass_core::AudioStreamConfig {
                codec: cutlass_core::AudioCodec::Aac,
                sample_rate: rate,
                channels: 1,
                config_record: encoder.config_record().to_vec(),
            },
            packets,
            rate,
        };
        let mut tape = super::Tape::new(&opened);
        let early = tape.pull(0, 4_096).expect("start");
        assert_eq!(early.len(), 4_096);
        assert!(early.iter().any(|sample| sample.abs() > 0.1));
        let later = tape.pull(8_192, 1_024).expect("later");
        let mean = later.iter().sum::<f32>() / later.len() as f32;
        assert!((mean - 0.25).abs() < 0.05, "mean {mean} from a 0.25 tone");
    }
}
