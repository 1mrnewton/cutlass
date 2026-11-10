//! Groups of clips, and attributes copied from one clip onto others.
//!
//! Grouped clips are picked together and move together. Copy attributes
//! takes a clip's placement and volume; Paste attributes puts them on the
//! selected clips that can take them.

use cutlass_model::SegmentId;
use cutlass_session::{ClipAttributes, EditError, Request};

use super::{Shell, ShellError};

impl Shell {
    /// The clips grouped with clip `id`, itself left out.
    pub(super) fn group_mates(&self, id: &SegmentId) -> Vec<SegmentId> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        let segments = || draft.tracks.iter().flat_map(|track| &track.segments);
        let Some(group) = segments()
            .find(|segment| &segment.id == id)
            .map(|segment| segment.group_id.as_str())
            .filter(|group| !group.is_empty())
        else {
            return Vec::new();
        };
        segments()
            .filter(|segment| segment.group_id == group && &segment.id != id)
            .map(|segment| segment.id.clone())
            .collect()
    }

    /// The groups the selected clips are in, each once.
    fn selected_groups(&self) -> Vec<String> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        let mut groups: Vec<String> = Vec::new();
        for segment in draft.tracks.iter().flat_map(|track| &track.segments) {
            if self.selection.contains(&segment.id)
                && !segment.group_id.is_empty()
                && !groups.contains(&segment.group_id)
            {
                groups.push(segment.group_id.clone());
            }
        }
        groups
    }

    /// Two or more clips are picked and are not one group already, so
    /// Group can make them one.
    #[must_use]
    pub fn group_enabled(&self) -> bool {
        if self.selection.len() < 2 {
            return false;
        }
        let Some(draft) = self.draft() else {
            return false;
        };
        let mut groups = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| self.selection.contains(&segment.id))
            .map(|segment| segment.group_id.as_str());
        let Some(first) = groups.next() else {
            return false;
        };
        first.is_empty() || groups.any(|group| group != first)
    }

    /// Make the selected clips one group, taking them out of any group
    /// they were in. One undo step.
    pub fn group_selected(&mut self) -> Result<(), ShellError> {
        if self.selection.len() < 2 {
            return Err(ShellError::Unavailable(
                "pick two or more clips to group".to_owned(),
            ));
        }
        let segment_ids = self.selection.clone();
        let group_id = self.fresh("group");
        let count = segment_ids.len();
        self.apply(Request::SetGroup {
            segment_ids,
            group_id,
        })?;
        self.set_notice(format!("Grouped {count} clips"));
        Ok(())
    }

    /// A picked clip is in a group, or is a compound clip, so Ungroup has
    /// something to take apart.
    #[must_use]
    pub fn ungroup_enabled(&self) -> bool {
        !self.selected_groups().is_empty() || self.compound_selected()
    }

    /// Take apart every group a selected clip is in: its clips stay where
    /// they are and move alone again. With no group picked, a compound
    /// clip is taken apart instead. One undo step.
    pub fn ungroup_selected(&mut self) -> Result<(), ShellError> {
        let groups = self.selected_groups();
        if groups.is_empty() {
            return self.ungroup_compound();
        }
        let segment_ids: Vec<SegmentId> = self
            .session_draft()?
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| groups.contains(&segment.group_id))
            .map(|segment| segment.id.clone())
            .collect();
        self.apply(Request::SetGroup {
            segment_ids,
            group_id: String::new(),
        })?;
        self.set_notice(match groups.len() {
            1 => "Ungrouped".to_owned(),
            n => format!("Ungrouped {n} groups"),
        });
        Ok(())
    }

    /// One clip is picked, so its attributes can be copied.
    #[must_use]
    pub fn copy_attributes_enabled(&self) -> bool {
        self.selection.len() == 1
    }

    /// Keep the selected clip's placement and volume for Paste attributes.
    pub fn copy_attributes(&mut self) -> Result<(), ShellError> {
        if self.selection.len() != 1 {
            return Err(ShellError::Unavailable(
                "pick one clip to copy attributes from".to_owned(),
            ));
        }
        let segment = self
            .selected_segment()
            .ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
        self.attributes = Some(ClipAttributes {
            clip: Some(segment.clip.clone()),
            uniform_scale: Some(segment.uniform_scale.clone()),
            volume: Some(segment.volume),
        });
        self.set_notice("Copied attributes".to_owned());
        Ok(())
    }

    /// Attributes were copied and a clip is picked to paste them onto.
    #[must_use]
    pub fn paste_attributes_enabled(&self) -> bool {
        self.attributes.is_some() && !self.selection.is_empty()
    }

    /// Put the copied placement and volume on the selected clips: the
    /// placement on those that draw, the volume on those that sound. One
    /// undo step.
    pub fn paste_attributes(&mut self) -> Result<(), ShellError> {
        let attributes = self
            .attributes
            .clone()
            .ok_or_else(|| ShellError::Unavailable("no attributes are copied".to_owned()))?;
        if self.selection.is_empty() {
            return Err(ShellError::Unavailable("nothing is selected".to_owned()));
        }
        let segment_ids = self.selection.clone();
        let count = segment_ids.len();
        self.apply(Request::PasteAttributes {
            segment_ids,
            attributes: Box::new(attributes),
        })
        .map_err(|error| match error {
            ShellError::Edit(EditError::TrackKind) => ShellError::Unavailable(
                "these clips have nothing the copied attributes fit".to_owned(),
            ),
            other => other,
        })?;
        self.set_notice(match count {
            1 => "Pasted attributes".to_owned(),
            n => format!("Pasted attributes on {n} clips"),
        });
        Ok(())
    }
}
