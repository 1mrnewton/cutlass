//! Host approval for changes the draft undo stack does not cover.
//!
//! The model does not accept an action by passing a flag. A refusal does not
//! run the change.

use std::fmt;

/// A change the host must allow before it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalAction {
    /// Replace a file outside the undo stack.
    ExportReplace {
        /// The file that would be replaced.
        path: String,
    },
    /// Send the conversation to a cloud model.
    EnableCloudModel,
    /// Send sandbox source and input off the machine.
    EnableOnlineSandbox,
    /// Send tool arguments to a remote server.
    EnableRemoteMcp {
        /// The server the user would enable.
        server: String,
    },
    /// Delete media bytes the draft undo cannot restore.
    DeleteMedia {
        /// The media file that would be deleted.
        path: String,
    },
    /// Add a skill folder from outside the app.
    AddExternalSkill {
        /// The skill folder that would be added.
        path: String,
    },
}

impl ApprovalAction {
    fn name(&self) -> &'static str {
        match self {
            Self::ExportReplace { .. } => "export",
            Self::EnableCloudModel => "cloud model",
            Self::EnableOnlineSandbox => "online sandbox",
            Self::EnableRemoteMcp { .. } => "remote mcp",
            Self::DeleteMedia { .. } => "delete media",
            Self::AddExternalSkill { .. } => "external skill",
        }
    }
}

/// The host's decision for one approval.
pub trait Approver {
    /// Returns whether the host allows `action`.
    fn decide(&mut self, action: &ApprovalAction) -> bool;
}

/// The host refused an approval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalError {
    /// The action the host refused.
    pub action: ApprovalAction,
}

impl fmt::Display for ApprovalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.action.name();
        write!(formatter, "refused {name}")
    }
}

impl std::error::Error for ApprovalError {}

/// `model_says_approved` is ignored. Only the host's Approver decides.
pub fn authorize(
    approver: &mut dyn Approver,
    action: ApprovalAction,
    model_says_approved: bool,
) -> Result<(), ApprovalError> {
    let _ = model_says_approved;
    if approver.decide(&action) {
        Ok(())
    } else {
        Err(ApprovalError { action })
    }
}

/// Runs `change` only after the host allows it.
pub fn authorize_then<T>(
    approver: &mut dyn Approver,
    action: ApprovalAction,
    model_says_approved: bool,
    change: impl FnOnce() -> T,
) -> Result<T, ApprovalError> {
    authorize(approver, action, model_says_approved)?;
    Ok(change())
}

#[cfg(test)]
mod tests {
    use super::{ApprovalAction, ApprovalError, Approver, authorize_then};

    struct ScriptedApprover {
        allow: bool,
        asked: Vec<ApprovalAction>,
    }

    impl Approver for ScriptedApprover {
        fn decide(&mut self, action: &ApprovalAction) -> bool {
            self.asked.push(action.clone());
            self.allow
        }
    }

    fn refuse(action: ApprovalAction, name: &str) -> ApprovalError {
        let mut approver = ScriptedApprover {
            allow: false,
            asked: Vec::new(),
        };
        let mut ran = 0;
        let error = authorize_then(&mut approver, action.clone(), false, || ran += 1).unwrap_err();
        assert_eq!(ran, 0);
        assert_eq!(approver.asked, vec![action.clone()]);
        let text = error.to_string();
        assert!(text.contains("refused"), "{text}");
        assert!(text.contains(name), "{text}");
        assert_eq!(error.action, action);
        error
    }

    #[test]
    fn export_replace_asks_and_a_refusal_does_not_run() {
        refuse(
            ApprovalAction::ExportReplace {
                path: "out.mp4".to_owned(),
            },
            "export",
        );
    }

    #[test]
    fn cloud_model_refusal_does_not_apply() {
        refuse(ApprovalAction::EnableCloudModel, "cloud model");
    }

    #[test]
    fn online_sandbox_refusal_does_not_apply() {
        refuse(ApprovalAction::EnableOnlineSandbox, "online sandbox");
    }

    #[test]
    fn remote_mcp_refusal_does_not_apply() {
        refuse(
            ApprovalAction::EnableRemoteMcp {
                server: "studio".to_owned(),
            },
            "remote mcp",
        );
    }

    #[test]
    fn delete_media_refusal_does_not_apply() {
        refuse(
            ApprovalAction::DeleteMedia {
                path: "clip.mov".to_owned(),
            },
            "delete media",
        );
    }

    #[test]
    fn external_skill_refusal_does_not_apply() {
        refuse(
            ApprovalAction::AddExternalSkill {
                path: "skills/extra".to_owned(),
            },
            "external skill",
        );
    }

    #[test]
    fn model_flag_does_not_approve() {
        let mut approver = ScriptedApprover {
            allow: false,
            asked: Vec::new(),
        };
        let mut ran = 0;
        let action = ApprovalAction::EnableCloudModel;
        let error = authorize_then(&mut approver, action.clone(), true, || ran += 1).unwrap_err();
        assert_eq!(ran, 0);
        assert_eq!(error.action, action);
        assert_eq!(approver.asked, vec![action]);
        assert!(error.to_string().contains("refused"));
    }

    #[test]
    fn host_allow_runs_the_change() {
        let mut approver = ScriptedApprover {
            allow: true,
            asked: Vec::new(),
        };
        let mut ran = 0;
        let action = ApprovalAction::ExportReplace {
            path: "out.mp4".to_owned(),
        };
        let value = authorize_then(&mut approver, action.clone(), false, || {
            ran += 1;
            "saved"
        })
        .unwrap();
        assert_eq!(value, "saved");
        assert_eq!(ran, 1);
        assert_eq!(approver.asked, vec![action]);
    }
}
