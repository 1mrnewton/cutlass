//! Compound clips in the window: make one from the selection, open it to
//! edit what is inside, go back, and ungroup it.
//!
//! An open compound clip swaps the window's session for one on its inside
//! draft; the project's session waits underneath. Back stores the inside
//! on the compound clip as one undo step. A save while inside writes the
//! whole project with the inside as it is now.

use std::time::Instant;

use cutlass_model::{Draft, MaterialId, MaterialRef, SegmentId};
use cutlass_session::{Request, Session};

use super::{Shell, ShellError, locate};

/// A level the window went into: the session it left, the compound clip
/// it opened, and where the playhead was.
pub(crate) struct Outer {
    session: Session,
    segment_id: SegmentId,
    material_id: MaterialId,
    /// The inside as it was opened, so Back stores it only when it moved.
    opened: Draft,
    playhead: cutlass_model::Micros,
    saved: Option<(u64, Instant)>,
}

impl Shell {
    /// The selected clip plays a compound draft: its material and inside.
    fn selected_compound(&self) -> Option<(SegmentId, MaterialId, Draft)> {
        let draft = self.draft()?;
        let id = self.selected()?;
        let (segment, _) = locate(draft, id)?;
        match draft.materials.get(segment.material_id.as_str())? {
            MaterialRef::Draft(material) => Some((
                segment.id.clone(),
                material.id.clone(),
                material.draft.as_deref()?.clone(),
            )),
            _ => None,
        }
    }

    /// The selected clip is a compound clip.
    #[must_use]
    pub fn compound_selected(&self) -> bool {
        self.selected_compound().is_some()
    }

    /// How many compound clips deep the window is: 0 on the project.
    #[must_use]
    pub fn compound_depth(&self) -> usize {
        self.outer.len()
    }

    /// Make the selected clips one compound clip, and select it.
    pub fn create_compound(&mut self) -> Result<(), ShellError> {
        if self.selection.is_empty() {
            return Err(ShellError::Unavailable("nothing is selected".to_owned()));
        }
        for id in self.selection.clone() {
            self.refuse_locked(&id)?;
        }
        let segment_id = SegmentId::new(self.fresh("compound"));
        let material_id = MaterialId::new(self.fresh("inside"));
        self.apply(Request::CreateCompound {
            segment_ids: self.selection.clone(),
            segment_id: segment_id.clone(),
            material_id,
        })?;
        self.select(Some(segment_id));
        Ok(())
    }

    /// Put the selected compound clip's clips back on the timeline.
    pub fn ungroup_compound(&mut self) -> Result<(), ShellError> {
        let (segment_id, _, _) = self
            .selected_compound()
            .ok_or_else(|| ShellError::Unavailable("pick a compound clip".to_owned()))?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::Ungroup { segment_id })?;
        self.select(None);
        Ok(())
    }

    /// Edit the inside of the selected compound clip in the window.
    pub fn open_compound(&mut self) -> Result<(), ShellError> {
        let (segment_id, material_id, inside) = self
            .selected_compound()
            .ok_or_else(|| ShellError::Unavailable("pick a compound clip".to_owned()))?;
        if self.unsaved() {
            self.save()?;
        }
        let session = self.session.take().ok_or(ShellError::NoProject)?;
        let mut opened = Session::open(inside.clone());
        opened.set_linkage(session.linkage());
        self.outer.push(Outer {
            session,
            segment_id,
            material_id,
            opened: inside,
            playhead: self.playhead,
            saved: self.saved,
        });
        self.session = Some(opened);
        self.saved = Some((0, Instant::now()));
        self.swapped(cutlass_model::Micros(0));
        Ok(())
    }

    /// Leave the open compound clip. What changed inside is stored on it as
    /// one undo step, and the compound clip is selected again.
    pub fn close_compound(&mut self) -> Result<(), ShellError> {
        let outer = self
            .outer
            .pop()
            .ok_or_else(|| ShellError::Unavailable("no compound clip is open".to_owned()))?;
        let inner = self.session.take().ok_or(ShellError::NoProject)?;
        let inside = inner.draft().clone();
        self.session = Some(outer.session);
        self.saved = outer.saved;
        self.swapped(outer.playhead);
        if inside != outer.opened {
            self.apply(Request::SetInside {
                material_id: outer.material_id,
                inside: Box::new(inside),
            })?;
        }
        self.select(Some(outer.segment_id));
        Ok(())
    }

    /// The project's draft with every open inside stored in place: what a
    /// save writes.
    pub(super) fn root_draft(&self) -> Option<Draft> {
        let mut draft = self.session.as_ref()?.draft().clone();
        for outer in self.outer.iter().rev() {
            let mut parent = Session::open(outer.session.draft().clone());
            parent
                .apply(
                    0,
                    &[Request::SetInside {
                        material_id: outer.material_id.clone(),
                        inside: Box::new(draft),
                    }],
                )
                .ok()?;
            draft = parent.draft().clone();
        }
        Some(draft)
    }

    /// The window shows another session now: what it drew and played for
    /// the last one goes.
    fn swapped(&mut self, playhead: cutlass_model::Micros) {
        self.pause();
        self.lanes_cache.replace(None);
        self.preview_cache.borrow_mut().clear();
        self.transport.forget_mixer();
        self.transport.forget_files();
        self.selection.clear();
        self.clip_menu = None;
        self.cancel_clip_drag();
        self.selected_marker = None;
        self.selected_beat = None;
        self.marks.clear();
        self.playhead = playhead;
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialRef, MaterialVideo, Micros, Segment, SegmentId, TimeRange};
    use cutlass_store::Folder;

    use crate::shell::tests::{Temp, shell};

    fn project(temp: &Temp) -> crate::shell::Shell {
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("v");
        video.kind = "video".to_owned();
        video.duration = Micros(20_000_000);
        draft.materials.videos.push(video);
        for (id, start) in [("a", 0), ("b", 2_000_000), ("c", 4_000_000)] {
            let mut clip = Segment::new(id, "v", TimeRange::new(start, 2_000_000));
            clip.source_range = Some(TimeRange::new(0, 2_000_000));
            draft.tracks[0].segments.push(clip);
        }
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    fn ids(desktop: &crate::shell::Shell) -> Vec<String> {
        desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.as_str().to_owned())
            .collect()
    }

    #[test]
    fn a_compound_clip_is_made_opened_edited_saved_and_ungrouped() {
        let temp = Temp::new();
        let mut desktop = project(&temp);
        desktop.select(Some(SegmentId::new("a")));
        desktop.toggle_selected(SegmentId::new("b"));
        desktop.create_compound().unwrap();
        assert_eq!(ids(&desktop).len(), 2);
        assert!(desktop.compound_selected());
        let compound = desktop.selected().cloned().unwrap();

        desktop.open_compound().unwrap();
        assert_eq!(desktop.compound_depth(), 1);
        assert_eq!(ids(&desktop), ["a", "b"]);
        assert_eq!(desktop.playhead(), Micros(0));
        // Remove b inside; the window undoes inside edits on their own.
        desktop.select(Some(SegmentId::new("b")));
        desktop.remove_selected().unwrap();
        assert_eq!(ids(&desktop), ["a"]);
        // A save while inside writes the whole project with the edit.
        desktop.save().unwrap();
        let saved = cutlass_store::open(&temp.path).unwrap().folder.draft;
        assert_eq!(saved.tracks[0].segments.len(), 2);
        let Some(MaterialRef::Draft(material)) = saved
            .materials
            .get(saved.tracks[0].segments[0].material_id.as_str())
        else {
            panic!("the saved project lost its compound clip");
        };
        assert_eq!(material.draft.as_ref().unwrap().tracks[0].segments.len(), 1);

        desktop.close_compound().unwrap();
        assert_eq!(desktop.compound_depth(), 0);
        assert_eq!(desktop.selected(), Some(&compound));
        let draft = desktop.draft().unwrap();
        assert_eq!(
            draft.tracks[0].segments[0].target.duration,
            Micros(2_000_000)
        );
        // Back is one undo step on the project.
        desktop.undo().unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(4_000_000)
        );
        desktop.select(Some(compound));
        desktop.ungroup_compound().unwrap();
        assert_eq!(ids(&desktop), ["a", "b", "c"]);
        assert!(desktop.close_compound().is_err());
        desktop.select(Some(SegmentId::new("a")));
        assert!(!desktop.compound_selected());
        assert!(desktop.open_compound().is_err());
    }
}
