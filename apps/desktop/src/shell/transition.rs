//! Transitions on the timeline: the bar over each cut that has one, which
//! is picked, deleted, and made longer or shorter by its edges, and the
//! shelf card dropped on a cut or on a clip.

use cutlass_editor::{
    BarEdge, Pointer, TransitionBar, TransitionMiss, bar_length, blend_of, transition_bars,
    transition_everywhere, transition_length_request, transition_limits, transition_of_card,
    transition_offers, transition_request,
};
use std::sync::Arc;

use cutlass_model::{MaterialId, Micros, SegmentId};
use cutlass_session::Request;

use cutlass_editor::FreshIds;

use super::{Shell, ShellError, snap_window};

/// What a transition dropped on the lanes did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionDrop {
    /// It is on the cut.
    Placed,
    /// The cut has a transition already. The new one waits for
    /// [`Shell::answer_transition`].
    Asks,
}

fn miss(miss: TransitionMiss) -> ShellError {
    match miss {
        TransitionMiss::Locked => ShellError::Locked,
        TransitionMiss::BadScale => {
            ShellError::Unavailable("the drag could not be read".to_owned())
        }
        TransitionMiss::OffLanes => ShellError::Unavailable("the drag missed the lanes".to_owned()),
        TransitionMiss::NotMain => {
            ShellError::Unavailable("a transition sits on the main track".to_owned())
        }
        TransitionMiss::NoCut => {
            ShellError::Unavailable("a transition goes between two clips that touch".to_owned())
        }
        TransitionMiss::TooShort => ShellError::Unavailable("the clips are too short".to_owned()),
    }
}

impl Shell {
    /// The transitions on cuts, as the lanes draw them: built again only
    /// when the revision moves.
    #[must_use]
    pub fn transitions(&self) -> Arc<[TransitionBar]> {
        let Some(session) = self.session.as_ref() else {
            return Arc::from([]);
        };
        let revision = session.revision();
        if let Some((built, bars)) = self.bars_cache.borrow().as_ref()
            && *built == revision
        {
            return Arc::clone(bars);
        }
        let built: Arc<[TransitionBar]> = transition_bars(session.draft()).into();
        self.bars_cache
            .replace(Some((revision, Arc::clone(&built))));
        built
    }

    /// The picked transition, while it is still on its cut.
    #[must_use]
    pub fn selected_transition(&self) -> Option<TransitionBar> {
        let picked = self.selected_transition.as_ref()?;
        self.transitions()
            .iter()
            .find(|bar| &bar.segment_id == picked)
            .cloned()
    }

    /// Pick the transition that leads out of clip `segment_id`, and let go
    /// of the clips, marker or beat that were picked. `None`, or a clip
    /// with no transition on a cut, picks nothing.
    pub fn select_transition(&mut self, segment_id: Option<SegmentId>) {
        let found =
            segment_id.filter(|id| self.transitions().iter().any(|bar| &bar.segment_id == id));
        self.select(None);
        self.selected_transition = found;
    }

    /// Take the picked transition off its cut. One undo step.
    pub fn remove_selected_transition(&mut self) -> Result<(), ShellError> {
        let bar = self
            .selected_transition()
            .ok_or_else(|| ShellError::Unavailable("no transition is selected".to_owned()))?;
        self.apply(Request::RemoveTransition {
            segment_id: bar.segment_id,
        })?;
        self.selected_transition = None;
        Ok(())
    }

    /// The shortest and the longest the transition out of `segment_id` can
    /// be, for its two clips.
    #[must_use]
    pub fn transition_limits(&self, segment_id: &SegmentId) -> Option<(Micros, Micros)> {
        transition_limits(self.draft()?, segment_id)
    }

    /// Make the transition out of `segment_id` last `duration`, held to its
    /// limits. While the pointer is down the changes are one undo step.
    pub fn set_transition_length(
        &mut self,
        segment_id: &SegmentId,
        duration: Micros,
    ) -> Result<(), ShellError> {
        let draft = self.session_draft()?;
        if transition_limits(draft, segment_id).is_none() {
            return Err(ShellError::Unavailable(
                "the clip has no transition".to_owned(),
            ));
        }
        match transition_length_request(draft, segment_id, duration) {
            Some(request) => self.apply(request),
            None => Ok(()),
        }
    }

    /// Drag an edge of the transition out of `segment_id` to `x`, in pixels
    /// from the left of the lanes. A transition that crosses its cut grows
    /// both ways from it.
    pub fn drag_transition_edge(
        &mut self,
        segment_id: &SegmentId,
        edge: BarEdge,
        x: f64,
    ) -> Result<(), ShellError> {
        let bar = self
            .transitions()
            .iter()
            .find(|bar| &bar.segment_id == segment_id)
            .cloned()
            .ok_or_else(|| ShellError::Unavailable("the clip has no transition".to_owned()))?;
        let time = self
            .lane_time_at(x)
            .ok_or_else(|| ShellError::Unavailable("the drag could not be read".to_owned()))?;
        self.set_transition_length(segment_id, bar_length(&bar, edge, time))
    }

    /// The length the shelf gives the transition out of `segment_id`,
    /// fitted to its clips: what Reset puts back.
    #[must_use]
    pub fn transition_preset(&self, segment_id: &SegmentId) -> Option<Micros> {
        let draft = self.draft()?;
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| &segment.id == segment_id)?;
        let blend = blend_of(draft, segment)?;
        let preset = transition_offers()
            .iter()
            .find(|offer| offer.kind.effect_id() == blend.effect_id)?
            .duration;
        let (_, longest) = transition_limits(draft, segment_id)?;
        Some(Micros(preset.min(longest.0)))
    }

    /// Put the transition out of `segment_id` back to the shelf's length.
    pub fn reset_transition_length(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        let preset = self
            .transition_preset(segment_id)
            .ok_or_else(|| ShellError::Unavailable("the clip has no transition".to_owned()))?;
        self.set_transition_length(segment_id, preset)
    }

    /// Put the transition out of `segment_id` on every other cut of its
    /// track, each fitted to its clips. One undo step. Says how many cuts
    /// took it.
    pub fn apply_transition_to_all(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        let mut ids = FreshIds::new(self.session_draft()?, self.next_id);
        let mut fresh = |prefix: &str| ids.make(prefix);
        let requests = transition_everywhere(self.session_draft()?, segment_id, &mut fresh);
        let next_id = ids.next;
        self.next_id = next_id;
        if requests.is_empty() {
            self.set_notice("There is no other cut to put it on".to_owned());
            return Ok(());
        }
        self.apply_requests(&requests)?;
        self.set_notice(match requests.len() {
            1 => "Applied to 1 other cut".to_owned(),
            n => format!("Applied to {n} other cuts"),
        });
        Ok(())
    }

    /// Store a transition where it was dropped on the main track: on the
    /// cut beside the pointer, or on the cut the clip under it begins at
    /// (the one it ends at for a clip nothing touches before).
    ///
    /// The two clips must touch. Neither moves, and a preset longer than
    /// they have room for is shortened. A cut that has a transition asks
    /// first, unless the user said not to be asked.
    pub fn drop_transition(
        &mut self,
        card_id: &str,
        x: f64,
        y: f64,
    ) -> Result<TransitionDrop, ShellError> {
        let offer = transition_of_card(card_id).ok_or_else(|| {
            ShellError::Unavailable("the transition is not in the library".to_owned())
        })?;
        let pointer = Pointer {
            x: self.time_zero_x(x),
            y,
        };
        let material_id = MaterialId::new(self.fresh("transition"));
        let draft = self.session_draft()?;
        let request = transition_request(
            draft,
            offer,
            pointer,
            self.scale(),
            snap_window(self.pixel_size),
            material_id,
        )
        .map_err(miss)?;
        let Request::SetTransition { segment_id, .. } = &request else {
            return self.apply(request).map(|()| TransitionDrop::Placed);
        };
        let segment_id = segment_id.clone();
        let taken = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == segment_id)
            .is_some_and(|segment| blend_of(draft, segment).is_some());
        if taken && !self.settings.confirm.replace_transition_unasked {
            self.asked_transition = Some(request);
            return Ok(TransitionDrop::Asks);
        }
        self.apply(request)?;
        self.select_transition(Some(segment_id));
        Ok(TransitionDrop::Placed)
    }

    /// The names of the transition on the cut and of the one waiting to
    /// take its place, while the question is open.
    #[must_use]
    pub fn asked_transition(&self) -> Option<(String, String)> {
        let Some(Request::SetTransition {
            segment_id,
            transition,
        }) = self.asked_transition.as_ref()
        else {
            return None;
        };
        let old = self
            .transitions()
            .iter()
            .find(|bar| &bar.segment_id == segment_id)?
            .name
            .clone();
        Some((old, transition.name.clone()))
    }

    /// Answer the question a transition dropped on a taken cut asked:
    /// `replace` puts it there as one undo step, and `never_ask` stops the
    /// question from here on. Not replacing leaves the cut as it was.
    pub fn answer_transition(&mut self, replace: bool, never_ask: bool) -> Result<(), ShellError> {
        let Some(request) = self.asked_transition.take() else {
            return Ok(());
        };
        if !replace {
            return Ok(());
        }
        if never_ask {
            self.update_settings(|settings| settings.confirm.replace_transition_unasked = true);
        }
        let picked = match &request {
            Request::SetTransition { segment_id, .. } => Some(segment_id.clone()),
            _ => None,
        };
        self.apply(request)?;
        self.select_transition(picked);
        Ok(())
    }
}
