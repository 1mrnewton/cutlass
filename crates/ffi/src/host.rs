//! Session requests, undo, redo, and rollback.
//!
//! One apply is one combo. A failed request leaves the draft unchanged.

use cutlass_session::{Draft, LinkageSettings, Request, Session};

use crate::error::FfiError;

/// The open cut and its history.
pub struct Commands {
    session: Session,
}

impl Commands {
    /// `Session::open`, then `set_linkage`. Does not write a folder.
    #[must_use]
    pub fn open(draft: Draft, linkage: LinkageSettings) -> Self {
        let mut session = Session::open(draft);
        session.set_linkage(linkage);
        Self { session }
    }

    /// The open draft.
    #[must_use]
    pub fn draft(&self) -> &Draft {
        self.session.draft()
    }

    /// The revision the next apply must name.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.session.revision()
    }

    /// The linkage switches stored by [`Self::open`].
    #[must_use]
    pub fn linkage(&self) -> LinkageSettings {
        self.session.linkage()
    }

    /// The session these calls forward to.
    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// One combo. Pass the slice to a single `Session::apply`. Do not loop.
    pub fn apply(&mut self, expected: u64, requests: &[Request]) -> Result<u64, FfiError> {
        let revision = self.session.apply(expected, requests)?;
        Ok(revision)
    }

    /// Restore the draft from before the last apply. The revision still advances.
    pub fn undo(&mut self) -> Result<u64, FfiError> {
        let revision = self.session.undo()?;
        Ok(revision)
    }

    /// Reapply the draft that undo removed. The revision advances again.
    pub fn redo(&mut self) -> Result<u64, FfiError> {
        let revision = self.session.redo()?;
        Ok(revision)
    }

    /// Snapshot the open draft. `apply` still pushes undo on its own.
    pub fn begin(&mut self) -> Result<(), FfiError> {
        self.session.begin()?;
        Ok(())
    }

    /// Open a recording, or mark the transaction `begin` already opened.
    pub fn record(&mut self) -> Result<(), FfiError> {
        self.session.record()?;
        Ok(())
    }

    /// Close the transaction. A changed draft is one recorded step, not a second undo push.
    pub fn commit(&mut self) -> Result<(), FfiError> {
        self.session.commit()?;
        Ok(())
    }

    /// Fold the open draft into the last recorded step.
    pub fn amend(&mut self) -> Result<(), FfiError> {
        self.session.amend()?;
        Ok(())
    }

    /// Put the open draft back where it was at [`Self::begin`] or [`Self::record`].
    pub fn rollback(&mut self) -> Result<(), FfiError> {
        self.session.rollback()?;
        Ok(())
    }

    /// Whether `record` has marked the open transaction.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.session.is_recording()
    }
}

#[cfg(test)]
mod tests {
    use cutlass_session::{
        Draft, EditError, LinkageSettings, Micros, Request, TrackAttribute, TrackId,
    };

    use super::Commands;
    use crate::error::FfiError;

    fn opened() -> Commands {
        Commands::open(Draft::empty("D"), LinkageSettings::default())
    }

    fn main_id(commands: &Commands) -> TrackId {
        commands.draft().tracks[0].id.clone()
    }

    fn attribute(track_id: TrackId, bit: i32, on: bool) -> Request {
        Request::SetTrackAttribute { track_id, bit, on }
    }

    #[test]
    fn open_stores_the_linkage() {
        let draft = Draft::empty("D");
        let linkage = LinkageSettings::default();
        let commands = Commands::open(draft.clone(), linkage);
        assert!(commands.linkage().enabled);
        assert_eq!(commands.linkage(), linkage);
        assert_eq!(commands.session().linkage(), linkage);
        assert_eq!(commands.draft(), &draft);
        assert_eq!(commands.revision(), 0);
        assert!(!commands.is_recording());

        let linkage = LinkageSettings {
            enabled: false,
            ..LinkageSettings::default()
        };
        let commands = Commands::open(Draft::empty("D"), linkage);
        assert!(!commands.linkage().enabled);
        assert_eq!(commands.linkage(), linkage);
        assert_eq!(commands.session().linkage(), linkage);
    }

    #[test]
    fn apply_mute_sets_the_attribute_and_keeps_duration_zero() {
        let mut commands = opened();
        let track = main_id(&commands);
        let revision = commands
            .apply(0, &[attribute(track, TrackAttribute::MUTE, true)])
            .unwrap();
        assert_eq!(revision, 1);
        assert_eq!(commands.revision(), 1);
        assert!(
            commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
        assert_eq!(commands.draft().duration, Micros(0));
    }

    #[test]
    fn one_apply_of_mute_and_hide_undoes_in_one_step() {
        let mut commands = opened();
        let track = main_id(&commands);
        let revision = commands
            .apply(
                0,
                &[
                    attribute(track.clone(), TrackAttribute::MUTE, true),
                    attribute(track, TrackAttribute::HIDE, true),
                ],
            )
            .unwrap();
        assert_eq!(revision, 1);
        let flags = commands.draft().tracks[0].attribute;
        assert!(flags.contains(TrackAttribute::MUTE));
        assert!(flags.contains(TrackAttribute::HIDE));

        let revision = commands.undo().unwrap();
        assert_eq!(revision, 2);
        assert_eq!(commands.revision(), 2);
        assert_eq!(commands.draft().tracks[0].attribute, TrackAttribute(0));

        let error = commands.undo().unwrap_err();
        assert!(matches!(error, FfiError::Edit(EditError::EmptyUndo)));
        assert!(error.to_string().contains("nothing to undo"));
        assert_eq!(commands.revision(), 2);
        assert_eq!(commands.draft().tracks[0].attribute, TrackAttribute(0));
    }

    #[test]
    fn a_failed_request_keeps_the_earlier_requests_off_the_draft() {
        let mut commands = opened();
        let before = commands.draft().clone();
        let track = main_id(&commands);
        let error = commands
            .apply(
                0,
                &[
                    attribute(track, TrackAttribute::MUTE, true),
                    attribute(TrackId::new("missing"), TrackAttribute::HIDE, true),
                ],
            )
            .unwrap_err();
        assert!(matches!(error, FfiError::Edit(EditError::MissingTrack)));
        assert!(error.to_string().contains("track"));
        assert_eq!(commands.draft(), &before);
        assert_eq!(commands.revision(), 0);
    }

    #[test]
    fn a_stale_revision_writes_nothing() {
        let mut commands = opened();
        let track = main_id(&commands);
        commands
            .apply(0, &[attribute(track.clone(), TrackAttribute::MUTE, true)])
            .unwrap();
        let muted = commands.draft().clone();
        let error = commands
            .apply(0, &[attribute(track, TrackAttribute::MUTE, false)])
            .unwrap_err();
        assert!(matches!(
            error,
            FfiError::Edit(EditError::RevisionMismatch {
                expected: 0,
                actual: 1
            })
        ));
        assert_eq!(commands.draft(), &muted);
        assert_eq!(commands.revision(), 1);
        assert!(
            commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
    }

    #[test]
    fn undo_then_redo_restores_the_mute() {
        let mut commands = opened();
        let track = main_id(&commands);
        commands
            .apply(0, &[attribute(track, TrackAttribute::MUTE, true)])
            .unwrap();
        commands.undo().unwrap();
        assert_eq!(commands.draft().tracks[0].attribute, TrackAttribute(0));
        commands.redo().unwrap();
        assert!(
            commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
        let error = commands.redo().unwrap_err();
        assert!(matches!(error, FfiError::Edit(EditError::EmptyRedo)));
        assert!(
            commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
        assert_eq!(commands.revision(), 3);
    }

    #[test]
    fn an_empty_apply_does_not_move_the_revision() {
        let mut commands = opened();
        assert_eq!(commands.apply(0, &[]).unwrap(), 0);
        assert_eq!(commands.revision(), 0);
        assert!(matches!(
            commands.undo().unwrap_err(),
            FfiError::Edit(EditError::EmptyUndo)
        ));
        assert_eq!(commands.revision(), 0);
    }

    #[test]
    fn begin_twice_is_history_open_and_commit_with_no_transaction_is_history_closed() {
        let mut commands = opened();
        let before = commands.draft().clone();
        let closed = commands.commit().unwrap_err();
        assert!(matches!(closed, FfiError::Edit(EditError::HistoryClosed)));
        assert_eq!(commands.draft(), &before);
        assert_eq!(commands.revision(), 0);

        commands.begin().unwrap();
        let open = commands.begin().unwrap_err();
        assert!(matches!(open, FfiError::Edit(EditError::HistoryOpen)));
        assert_eq!(commands.draft(), &before);
        assert_eq!(commands.revision(), 0);
    }

    #[test]
    fn rollback_restores_the_draft_from_begin_and_drops_the_undo_apply_pushed() {
        let mut commands = opened();
        let track = main_id(&commands);
        commands
            .apply(0, &[attribute(track.clone(), TrackAttribute::MUTE, true)])
            .unwrap();
        let at_begin = commands.draft().clone();
        commands.begin().unwrap();
        commands
            .apply(1, &[attribute(track, TrackAttribute::HIDE, true)])
            .unwrap();
        assert!(
            commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::HIDE)
        );
        commands.rollback().unwrap();
        assert_eq!(commands.draft(), &at_begin);
        assert!(
            commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
        assert!(
            !commands.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::HIDE)
        );
        assert_eq!(commands.revision(), 2);
        assert!(!commands.is_recording());
        commands.undo().unwrap();
        assert_eq!(commands.draft().tracks[0].attribute, TrackAttribute(0));
        assert!(matches!(
            commands.undo().unwrap_err(),
            FfiError::Edit(EditError::EmptyUndo)
        ));
    }

    #[test]
    fn record_and_commit_forward_and_amend_needs_a_recorded_step() {
        let mut commands = opened();
        let before = commands.draft().clone();
        let error = commands.amend().unwrap_err();
        assert!(matches!(error, FfiError::Edit(EditError::NothingToAmend)));
        assert_eq!(commands.draft(), &before);

        commands.record().unwrap();
        assert!(commands.is_recording());
        commands.commit().unwrap();
        assert!(!commands.is_recording());
        assert_eq!(commands.draft(), &before);
        assert_eq!(commands.revision(), 0);
    }
}
