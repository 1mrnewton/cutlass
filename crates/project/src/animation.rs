//! One named animation on a segment.
//!
//! The project stores the name and the duration. It does not open the name
//! and it does not play the animation.

use crate::Error;

/// Where an animation sits on a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationKind {
    In,
    Out,
    Combo,
    Loop,
}

impl AnimationKind {
    pub const ALL: [Self; 4] = [Self::In, Self::Out, Self::Combo, Self::Loop];
}

/// A name and a duration. The duration is in microseconds and is greater than zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Animation {
    name: String,
    duration_us: i64,
}

/// The four slots. Absent when every slot is empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnimationSlots {
    intro: Option<Animation>,
    outro: Option<Animation>,
    combo: Option<Animation>,
    repeat: Option<Animation>,
}

impl Animation {
    /// An animation. The name must be non-empty and the duration greater than zero.
    pub fn new(name: impl Into<String>, duration_us: i64) -> Result<Self, Error> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::Animation);
        }
        if duration_us < 0 {
            return Err(Error::NegativeTime);
        }
        if duration_us == 0 {
            return Err(Error::ZeroDuration);
        }
        Ok(Self { name, duration_us })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn duration_us(&self) -> i64 {
        self.duration_us
    }

    pub(crate) const fn set_duration(&mut self, duration_us: i64) {
        self.duration_us = duration_us;
    }
}

impl AnimationSlots {
    pub(crate) const fn empty() -> Self {
        Self {
            intro: None,
            outro: None,
            combo: None,
            repeat: None,
        }
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.intro.is_none()
            && self.outro.is_none()
            && self.combo.is_none()
            && self.repeat.is_none()
    }

    pub(crate) const fn get(&self, kind: AnimationKind) -> Option<&Animation> {
        match kind {
            AnimationKind::In => self.intro.as_ref(),
            AnimationKind::Out => self.outro.as_ref(),
            AnimationKind::Combo => self.combo.as_ref(),
            AnimationKind::Loop => self.repeat.as_ref(),
        }
    }

    pub(crate) fn set(&mut self, kind: AnimationKind, animation: Option<Animation>) {
        match kind {
            AnimationKind::In => self.intro = animation,
            AnimationKind::Out => self.outro = animation,
            AnimationKind::Combo => self.combo = animation,
            AnimationKind::Loop => self.repeat = animation,
        }
    }

    /// Clamps every slot to `span`. In and out then shrink so they still fit
    /// together. A slot that would last zero is removed.
    pub(crate) fn refit(&mut self, span: i64) {
        clamp(&mut self.combo, span);
        clamp(&mut self.repeat, span);
        clamp(&mut self.intro, span);
        clamp(&mut self.outro, span);
        let intro = duration_of(self.intro.as_ref());
        let outro = duration_of(self.outro.as_ref());
        let Some(total) = intro.checked_add(outro) else {
            self.intro = None;
            self.outro = None;
            return;
        };
        if total > span {
            let share = i128::from(intro) * i128::from(span) / i128::from(total);
            let intro_next = i64::try_from(share).unwrap_or_default();
            let outro_next = span.saturating_sub(intro_next);
            write(&mut self.intro, intro_next);
            write(&mut self.outro, outro_next);
        }
    }
}

const fn duration_of(animation: Option<&Animation>) -> i64 {
    match animation {
        Some(animation) => animation.duration_us(),
        None => 0,
    }
}

fn clamp(slot: &mut Option<Animation>, span: i64) {
    let Some(animation) = slot else {
        return;
    };
    if animation.duration_us() > span {
        write(slot, span);
    }
}

fn write(slot: &mut Option<Animation>, duration: i64) {
    if duration <= 0 {
        *slot = None;
        return;
    }
    if let Some(animation) = slot {
        animation.set_duration(duration);
    }
}
