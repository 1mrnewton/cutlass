//! Cross-thread "the user is interacting with the preview" gate.
//!
//! The preview worker, the filmstrip/waveform strip worker, the thumbnail
//! worker, and (soon) the proxy transcoder all decode media and submit GPU
//! work — on one shared iGPU plus one hardware decode engine. While the user
//! scrubs or plays back, background tile work directly steals decode and GPU
//! time from the frames the user is watching, which field logs showed as
//! multi-hundred-millisecond `composite_ms` for 228×128 tiles.
//!
//! [`InteractionGate`] is the coordination point: UI callbacks that already
//! see transport state mark it, and background workers poll [`busy`] between
//! work items, deferring (never dropping) their queues while it reports
//! interaction. Everything is atomics — no locks, safe to touch from the UI
//! thread at pointer-move rate.
//!
//! [`busy`]: InteractionGate::busy

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How long after the last playhead move the gate stays closed: long enough
/// to bridge the gaps inside a scrub drag (pointer-move bursts), short
/// enough that tiles resume promptly once the user settles.
const SCRUB_HOLDOFF: Duration = Duration::from_millis(500);

/// Shared interaction state; see the module docs. Clone the [`Arc`] into
/// every worker that should yield to the preview.
pub struct InteractionGate {
    /// Milliseconds since `epoch` of the last playhead change (scrub tick or
    /// playback step). `u64::MAX` sentinel = never touched.
    last_touch_ms: AtomicU64,
    /// Transport is running: gate stays closed for the whole playback, not
    /// just [`SCRUB_HOLDOFF`] past each tick.
    playing: AtomicBool,
    /// Monotonic zero point for `last_touch_ms`.
    epoch: Instant,
}

impl InteractionGate {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            last_touch_ms: AtomicU64::new(u64::MAX),
            playing: AtomicBool::new(false),
            epoch: Instant::now(),
        })
    }

    /// Record a playhead interaction (scrub tick, playback step).
    pub fn touch(&self) {
        let ms = self.epoch.elapsed().as_millis() as u64;
        self.last_touch_ms.store(ms, Ordering::Relaxed);
    }

    /// Record transport play/pause.
    pub fn set_playing(&self, playing: bool) {
        self.playing.store(playing, Ordering::Relaxed);
    }

    /// True while background decode/GPU work should defer: transport is
    /// playing, or a playhead interaction happened within [`SCRUB_HOLDOFF`].
    pub fn busy(&self) -> bool {
        if self.playing.load(Ordering::Relaxed) {
            return true;
        }
        let last = self.last_touch_ms.load(Ordering::Relaxed);
        if last == u64::MAX {
            return false;
        }
        let now = self.epoch.elapsed().as_millis() as u64;
        now.saturating_sub(last) < SCRUB_HOLDOFF.as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_opens_when_idle_and_closes_on_interaction() {
        let gate = InteractionGate::new();
        assert!(!gate.busy(), "fresh gate is open");

        gate.touch();
        assert!(gate.busy(), "a scrub tick closes the gate");

        gate.set_playing(true);
        assert!(gate.busy(), "playback holds the gate closed");
        gate.set_playing(false);
        // Still within the scrub holdoff of the earlier touch.
        assert!(gate.busy());
    }
}
