//! CPU framebuffers for the portable device path.
//!
//! Ids start at 1. One pass may be open. Readback waits until submit.

use cutlass_gpu::{Blend, ClearColor, Extent, Framebuffer, Pass};

use super::error::AndroidError;

#[derive(Debug)]
struct Slot {
    id: u64,
    extent: Extent,
    rgba: Vec<u8>,
    submitted: bool,
}

#[derive(Clone, Copy, Debug)]
struct OpenPass {
    framebuffer: Framebuffer,
    ended: bool,
    blend: Blend,
}

/// CPU framebuffers and the single open pass.
#[derive(Debug)]
pub(crate) struct Frames {
    next_id: u64,
    slots: Vec<Slot>,
    open: Option<OpenPass>,
}

impl Frames {
    pub(crate) fn new() -> Self {
        Self {
            next_id: 1,
            slots: Vec::new(),
            open: None,
        }
    }

    pub(crate) fn create(&mut self, extent: Extent) -> Result<Framebuffer, AndroidError> {
        let len = extent.rgba_len().ok_or(AndroidError::Extent)?;
        let id = self.next_id;
        let next = id.checked_add(1).ok_or(AndroidError::Extent)?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(len)
            .map_err(|_| AndroidError::Extent)?;
        self.slots
            .try_reserve(1)
            .map_err(|_| AndroidError::Extent)?;
        rgba.resize(len, 0);
        self.next_id = next;
        self.slots.push(Slot {
            id,
            extent,
            rgba,
            submitted: false,
        });
        Ok(Framebuffer::new(id))
    }

    pub(crate) fn destroy(&mut self, framebuffer: Framebuffer) -> Result<(), AndroidError> {
        if self
            .open
            .is_some_and(|open| open.framebuffer == framebuffer)
        {
            self.open = None;
        }
        let index = self.index(framebuffer)?;
        self.slots.swap_remove(index);
        Ok(())
    }

    pub(crate) fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, AndroidError> {
        if self.open.is_some() {
            return Err(AndroidError::PassOpen);
        }
        let slot = self.slot_mut(framebuffer)?;
        slot.submitted = false;
        self.open = Some(OpenPass {
            framebuffer,
            ended: false,
            blend: Blend::Over,
        });
        Ok(Pass::new(framebuffer))
    }

    pub(crate) fn set_blend(&mut self, pass: Pass, blend: Blend) -> Result<(), AndroidError> {
        let Some(open) = self.open.as_mut() else {
            return Err(AndroidError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(AndroidError::NoPass);
        }
        if open.ended {
            return Err(AndroidError::Ended);
        }
        open.blend = blend;
        Ok(())
    }

    pub(crate) fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), AndroidError> {
        self.require_drawing(pass)?;
        let slot = self.slot_mut(pass.framebuffer())?;
        fill_rgba(&mut slot.rgba, color.rgba);
        Ok(())
    }

    pub(crate) fn end(&mut self, pass: Pass) -> Result<(), AndroidError> {
        let Some(open) = self.open.as_mut() else {
            return Err(AndroidError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(AndroidError::NoPass);
        }
        if open.ended {
            return Err(AndroidError::Ended);
        }
        open.ended = true;
        Ok(())
    }

    pub(crate) fn submit(&mut self, pass: Pass) -> Result<(), AndroidError> {
        let Some(open) = self.open else {
            return Err(AndroidError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(AndroidError::NoPass);
        }
        if !open.ended {
            return Err(AndroidError::NotEnded);
        }
        let slot = self.slot_mut(pass.framebuffer())?;
        slot.submitted = true;
        self.open = None;
        Ok(())
    }

    pub(crate) fn read(&self, framebuffer: Framebuffer) -> Result<&[u8], AndroidError> {
        let slot = self.slot(framebuffer)?;
        if !slot.submitted {
            return Err(AndroidError::NotSubmitted);
        }
        Ok(slot.rgba.as_slice())
    }

    pub(crate) fn require_drawing(&self, pass: Pass) -> Result<(), AndroidError> {
        match self.open {
            Some(open) if open.framebuffer == pass.framebuffer() && !open.ended => Ok(()),
            Some(open) if open.framebuffer == pass.framebuffer() => Err(AndroidError::Ended),
            _ => Err(AndroidError::NoPass),
        }
    }

    pub(crate) fn validate_source(
        &self,
        pass: Pass,
        extent: Extent,
        byte_len: usize,
    ) -> Result<(), AndroidError> {
        let expected = self.accepted_len(pass, extent)?;
        if byte_len != expected {
            return Err(AndroidError::Bytes);
        }
        Ok(())
    }

    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub(crate) fn check_extent(&self, pass: Pass, extent: Extent) -> Result<usize, AndroidError> {
        self.accepted_len(pass, extent)
    }

    pub(crate) fn blend_of(&self, pass: Pass) -> Result<Blend, AndroidError> {
        self.require_drawing(pass)?;
        match self.open {
            Some(open) if open.framebuffer == pass.framebuffer() && !open.ended => Ok(open.blend),
            _ => Err(AndroidError::NoPass),
        }
    }

    pub(crate) fn drawing_rgba(&mut self, pass: Pass) -> Result<&mut [u8], AndroidError> {
        self.require_drawing(pass)?;
        let slot = self.slot_mut(pass.framebuffer())?;
        Ok(slot.rgba.as_mut_slice())
    }

    fn index(&self, framebuffer: Framebuffer) -> Result<usize, AndroidError> {
        self.slots
            .iter()
            .position(|slot| slot.id == framebuffer.id())
            .ok_or(AndroidError::MissingFramebuffer)
    }

    fn slot(&self, framebuffer: Framebuffer) -> Result<&Slot, AndroidError> {
        let index = self.index(framebuffer)?;
        self.slots
            .get(index)
            .ok_or(AndroidError::MissingFramebuffer)
    }

    fn slot_mut(&mut self, framebuffer: Framebuffer) -> Result<&mut Slot, AndroidError> {
        let index = self.index(framebuffer)?;
        self.slots
            .get_mut(index)
            .ok_or(AndroidError::MissingFramebuffer)
    }

    fn accepted_len(&self, pass: Pass, extent: Extent) -> Result<usize, AndroidError> {
        self.require_drawing(pass)?;
        let stored = self.slot(pass.framebuffer())?.extent;
        if stored != extent {
            return Err(AndroidError::Size);
        }
        extent.rgba_len().ok_or(AndroidError::Extent)
    }
}

fn fill_rgba(destination: &mut [u8], pixel: [u8; 4]) {
    for chunk in destination.as_chunks_mut::<4>().0 {
        *chunk = pixel;
    }
}
