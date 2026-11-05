//! The project socket: while the window holds a project it listens on a
//! loopback port named in `.locked`, so `cutlass mcp` on the same project
//! sends its tool calls here and the window applies them, one undo step
//! each, as it does the assistant's.

use cutlass_agent::{ProjectSocket, ToolOutcome, Wake};

use super::Shell;

impl Shell {
    /// Listen for other hosts on the project held now and on each one
    /// opened later. `wake` asks the window for a frame when a call comes.
    pub fn listen(&mut self, wake: Wake) {
        self.wake = Some(wake);
        self.open_socket();
    }

    /// Open the socket for the held project, when the window listens.
    pub(super) fn open_socket(&mut self) {
        self.socket = None;
        let (Some(wake), Some(hold)) = (self.wake.clone(), self.hold.as_mut()) else {
            return;
        };
        let socket = match ProjectSocket::open(wake) {
            Ok(socket) => socket,
            Err(error) => {
                cutlass_logger::warn!("project socket: {error}");
                return;
            }
        };
        if let Err(error) = hold.set_address(&socket.address(), socket.token()) {
            cutlass_logger::warn!("project socket: {error}");
            return;
        }
        self.socket = Some(socket);
    }

    /// Run the calls other hosts sent. A call that needs the user's allow
    /// is refused, as no one is asked for it here. True when one ran.
    pub fn serve_forwarded(&mut self) -> bool {
        let Some(waiting) = self.socket.as_ref().map(ProjectSocket::take) else {
            return false;
        };
        let ran = !waiting.is_empty();
        for forwarded in waiting {
            let run = self.run_agent_tool(&forwarded.call, &[]);
            let outcome = match run.wants {
                Some(action) => ToolOutcome {
                    content: format!(
                        "refused: {} needs the user's allow, which is given in the Cutlass window's assistant",
                        action.prompt()
                    ),
                    is_error: true,
                },
                None => run.outcome,
            };
            forwarded.answer(outcome);
        }
        ran
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutlass_agent::{Forward, ToolCall, ToolSink};
    use cutlass_model::{Draft, Segment, TimeRange};
    use cutlass_store::Folder;

    use crate::shell::tests::{Temp, shell};

    #[test]
    fn a_second_host_edits_the_project_through_the_window() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "color".into(),
                is_color_clip: true,
                solid_color: "0xFF336699".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0]
            .segments
            .push(Segment::new("one", "color", TimeRange::new(0, 2_000_000)));
        draft.duration = cutlass_model::Micros(2_000_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert_eq!(cutlass_store::held_address(&temp.path), None);
        desktop.listen(Arc::new(|| {}));
        let (address, token) = cutlass_store::held_address(&temp.path).unwrap();
        let revision = desktop.session().unwrap().revision();

        let caller = std::thread::spawn(move || {
            let mut forward = Forward::connect(&address, &token).unwrap();
            let read = forward.run_tool(&ToolCall {
                id: "1".to_owned(),
                name: "draft.query".to_owned(),
                arguments: "project".to_owned(),
            });
            let split = serde_json::json!({
                "expected_revision": revision,
                "requests": [{"op": "split", "segment_id": "one", "at": 1_000_000, "right_id": "two"}],
            });
            let edit = forward.run_tool(&ToolCall {
                id: "2".to_owned(),
                name: "draft.apply".to_owned(),
                arguments: split.to_string(),
            });
            (read, edit)
        });
        let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !caller.is_finished() && std::time::Instant::now() < until {
            desktop.serve_forwarded();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let (read, edit) = caller.join().unwrap();
        assert!(!read.is_error, "{}", read.content);
        assert!(!edit.is_error, "{}", edit.content);
        let segments = &desktop.draft().unwrap().tracks[0].segments;
        assert_eq!(segments.len(), 2, "the window holds the split");
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 1);

        // Closing the project stops the socket and clears the address.
        desktop.close_project().unwrap();
        assert_eq!(cutlass_store::held_address(&temp.path), None);
    }
}
