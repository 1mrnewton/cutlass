//! One Media Foundation transform, synchronous or asynchronous, behind the
//! same three calls: feed a sample, drain, flush.
//!
//! The in-box software codecs are synchronous: input goes in until the
//! transform says it is not accepting, and output comes out until it asks
//! for more input. A hardware encoder is asynchronous. It says when it wants
//! input and when it has output through its event queue, and calling it at
//! any other time is an error. Here the queue is polled, so a caller still
//! sees one blocking call that returns the samples that were ready.
//!
//! A transform that changes its output type mid-stream calls the codec's
//! `choose` back to pick the new type before more output is read.

use std::thread;
use std::time::{Duration, Instant};

use super::com::{self, Com, DeviceManager, Hresult};

/// Output samples one call may read before the loop gives up on a
/// transform that never asks for more input.
const MAX_OUTPUTS: usize = 256;

/// How long an asynchronous transform may go without an event before a
/// call gives up on it.
const QUIET: Duration = Duration::from_secs(5);

/// Picks the output type again after a stream change.
pub type Choose<'a> = &'a mut dyn FnMut(&Com) -> Result<(), Hresult>;

pub struct Transform {
    mft: Com,
    events: Option<Com>,
    /// `METransformNeedInput` events not yet answered with a sample.
    credit: u32,
    provides: bool,
    size: usize,
    hardware: bool,
    /// Dropped after the transform that holds it.
    _device: Option<DeviceManager>,
}

impl Transform {
    /// Wrap a transform, unlocking it first when it is asynchronous.
    pub fn new(mft: Com, hardware: bool, device: Option<DeviceManager>) -> Result<Self, Hresult> {
        let events = if com::unlock_async(&mft)? {
            Some(com::event_generator(&mft)?)
        } else {
            None
        };
        Ok(Self {
            mft,
            events,
            credit: 0,
            provides: false,
            size: 0,
            hardware,
            _device: device,
        })
    }

    pub fn mft(&self) -> &Com {
        &self.mft
    }

    /// Whether this runs on the GPU: a hardware encoder, or a decoder that
    /// was handed a Direct3D device.
    pub fn hardware(&self) -> bool {
        self.hardware
    }

    /// Read the output stream's sample rules after its type is set.
    pub fn refresh(&mut self) -> Result<(), Hresult> {
        let info = com::output_stream_info(&self.mft)?;
        self.provides = info.flags & com::MFT_OUTPUT_STREAM_PROVIDES_SAMPLES != 0;
        self.size = info.size as usize;
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), Hresult> {
        self.refresh()?;
        com::message(&self.mft, com::MFT_MESSAGE_NOTIFY_BEGIN_STREAMING)?;
        com::message(&self.mft, com::MFT_MESSAGE_NOTIFY_START_OF_STREAM)
    }

    /// Hand in one sample and return the output that is ready.
    pub fn feed(&mut self, sample: &Com, choose: Choose<'_>) -> Result<Vec<Com>, Hresult> {
        if self.events.is_some() {
            return self.feed_async(sample, choose);
        }
        let mut out = Vec::new();
        let mut code = com::process_input(&self.mft, sample);
        if code == com::MF_E_NOTACCEPTING {
            out = self.pull_all(choose)?;
            code = com::process_input(&self.mft, sample);
        }
        if code < 0 {
            return Err(code);
        }
        out.extend(self.pull_all(choose)?);
        Ok(out)
    }

    /// Every sample the transform still holds.
    pub fn drain(&mut self, choose: Choose<'_>) -> Result<Vec<Com>, Hresult> {
        com::message(&self.mft, com::MFT_MESSAGE_NOTIFY_END_OF_STREAM)?;
        com::message(&self.mft, com::MFT_MESSAGE_COMMAND_DRAIN)?;
        let out = if self.events.is_some() {
            self.drain_async(choose)?
        } else {
            self.pull_all(choose)?
        };
        let _ = com::message(&self.mft, com::MFT_MESSAGE_NOTIFY_START_OF_STREAM);
        Ok(out)
    }

    /// Drop what is in flight. The configuration stays.
    pub fn flush(&mut self) {
        let _ = com::message(&self.mft, com::MFT_MESSAGE_COMMAND_FLUSH);
        if let Some(events) = &self.events {
            while let Ok(Some(kind)) = com::next_event(events, false) {
                if kind == com::EVENT_NEED_INPUT {
                    self.credit += 1;
                }
            }
        }
    }

    fn pull_all(&mut self, choose: Choose<'_>) -> Result<Vec<Com>, Hresult> {
        let mut out = Vec::new();
        for _ in 0..MAX_OUTPUTS {
            match self.pull(choose)? {
                Some(sample) => out.push(sample),
                None => break,
            }
        }
        Ok(out)
    }

    /// One output sample, or `None` when the transform wants input.
    fn pull(&mut self, choose: Choose<'_>) -> Result<Option<Com>, Hresult> {
        for _ in 0..2 {
            let provided = if self.provides {
                None
            } else {
                Some(com::empty_sample(self.size)?)
            };
            match com::process_output(&self.mft, provided) {
                Ok(sample) => return Ok(Some(sample)),
                Err(com::MF_E_TRANSFORM_NEED_MORE_INPUT) => return Ok(None),
                Err(com::MF_E_TRANSFORM_STREAM_CHANGE) => {
                    choose(&self.mft)?;
                    self.refresh()?;
                    // An asynchronous transform sends a fresh
                    // `METransformHaveOutput` once the new type is set, and
                    // reading before it is `E_UNEXPECTED`.
                    if self.events.is_some() {
                        return Ok(None);
                    }
                }
                Err(code) => return Err(code),
            }
        }
        Ok(None)
    }

    fn feed_async(&mut self, sample: &Com, choose: Choose<'_>) -> Result<Vec<Com>, Hresult> {
        let mut out = Vec::new();
        let mut quiet = Instant::now();
        while self.credit == 0 {
            match self.event()? {
                Some(com::EVENT_NEED_INPUT) => self.credit += 1,
                Some(com::EVENT_HAVE_OUTPUT) => {
                    out.extend(self.pull(choose)?);
                    quiet = Instant::now();
                }
                Some(_) => {}
                None if quiet.elapsed() > QUIET => return Err(com::MF_E_NOTACCEPTING),
                None => thread::sleep(Duration::from_millis(1)),
            }
        }
        let code = com::process_input(&self.mft, sample);
        if code < 0 {
            return Err(code);
        }
        self.credit -= 1;
        while let Some(kind) = self.event()? {
            match kind {
                com::EVENT_NEED_INPUT => self.credit += 1,
                com::EVENT_HAVE_OUTPUT => out.extend(self.pull(choose)?),
                _ => {}
            }
        }
        Ok(out)
    }

    fn drain_async(&mut self, choose: Choose<'_>) -> Result<Vec<Com>, Hresult> {
        let mut out = Vec::new();
        let mut quiet = Instant::now();
        loop {
            match self.event()? {
                Some(com::EVENT_DRAIN_COMPLETE) => return Ok(out),
                Some(com::EVENT_HAVE_OUTPUT) => {
                    out.extend(self.pull(choose)?);
                    quiet = Instant::now();
                }
                Some(com::EVENT_NEED_INPUT) => self.credit += 1,
                Some(_) => {}
                None if quiet.elapsed() > QUIET => return Ok(out),
                None => thread::sleep(Duration::from_millis(1)),
            }
        }
    }

    fn event(&self) -> Result<Option<u32>, Hresult> {
        match &self.events {
            Some(events) => com::next_event(events, false),
            None => Ok(None),
        }
    }
}
