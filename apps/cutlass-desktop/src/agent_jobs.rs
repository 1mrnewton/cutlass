//! Strict, bounded background-job tools for the desktop agent.
//!
//! The app owns one [`JobManager`] for its lifetime. This module only exposes
//! sanitized snapshots from that registry; monotonic timestamps and other
//! registry internals never cross the tool boundary.
//! `cancel_requested` records an accepted request; `cancellable` says whether
//! the registry still accepts cancellation for the job.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use cutlass_ai::{HostToolSpec, ToolOutput, ToolTier};
use cutlass_jobs::{JobId, JobManager, JobSnapshot, JobState};
use serde_json::{Map, Value, json};

const JOB_LIST: &str = "job_list";
const JOB_STATUS: &str = "job_status";
const JOB_CANCEL: &str = "job_cancel";
const TOOL_NAMES: [&str; 3] = [JOB_LIST, JOB_STATUS, JOB_CANCEL];
const MAX_LIST_JOBS: usize = 128;

pub fn specs() -> Vec<HostToolSpec> {
    vec![
        spec(
            JOB_LIST,
            "List known background jobs newest first, including state, progress, whether cancellation was requested, whether cancellation is still accepted, and elapsed time.",
            empty_object_schema(),
            ToolTier::ReadOnly,
        ),
        spec(
            JOB_STATUS,
            "Read one background job by job_id, including distinct cancel_requested and cancellable fields.",
            job_id_schema(),
            ToolTier::ReadOnly,
        ),
        spec(
            JOB_CANCEL,
            "Request cooperative cancellation of a queued or running background job that has not begun committing.",
            job_id_schema(),
            ToolTier::Workspace,
        ),
    ]
}

pub fn call(
    manager: &JobManager,
    name: &str,
    arguments: &Value,
    cancel: &AtomicBool,
) -> Result<ToolOutput, String> {
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled before the job tool could run".into());
    }
    let request = parse_request(name, arguments)?;
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled before the job tool could run".into());
    }

    let value = match request {
        Request::List => list_jobs(manager),
        Request::Status(id) => {
            let snapshot = manager.get(id).ok_or_else(|| unknown_job(id))?;
            json!({
                "status": "ok",
                "job": snapshot_value(snapshot, Instant::now())
            })
        }
        Request::Cancel(id) => cancel_job(manager, id)?,
    };

    serde_json::to_string(&value)
        .map(ToolOutput::text)
        .map_err(|error| format!("could not encode job-tool response: {error}"))
}

fn list_jobs(manager: &JobManager) -> Value {
    let snapshots = manager.snapshot();
    let total = snapshots.len();
    let now = Instant::now();
    let jobs = snapshots
        .into_iter()
        .take(MAX_LIST_JOBS)
        .map(|snapshot| snapshot_value(snapshot, now))
        .collect::<Vec<_>>();
    json!({
        "status": "ok",
        "jobs": jobs,
        "total": total,
        "truncated": total > MAX_LIST_JOBS
    })
}

fn cancel_job(manager: &JobManager, id: JobId) -> Result<Value, String> {
    let before = manager.get(id).ok_or_else(|| unknown_job(id))?;
    if before.state.is_terminal() {
        return Err(terminal_cancel_error(id, before.state));
    }
    if !before.cancellable {
        return Err(commit_phase_cancel_error(id));
    }

    if manager.cancel(id) {
        return Ok(json!({
            "status": "ok",
            "job_id": id.raw(),
            "cancel_requested": true
        }));
    }

    // The job may have crossed its commit boundary or terminal transition
    // between `get` and `cancel`. Never claim cancellation when the registry
    // refused it, and distinguish a live commit-phase job from a terminal one.
    match manager.get(id) {
        None => Err(unknown_job(id)),
        Some(after) if after.state.is_terminal() => Err(terminal_cancel_error(id, after.state)),
        Some(after) if !after.cancellable => Err(commit_phase_cancel_error(id)),
        Some(_) => Err(format!(
            "job {} did not accept the cancellation request",
            id.raw()
        )),
    }
}

fn snapshot_value(snapshot: JobSnapshot, now: Instant) -> Value {
    let JobSnapshot {
        id,
        label,
        state,
        cancellable,
        progress,
        detail,
        cancel_requested,
        started,
        finished,
    } = snapshot;
    let elapsed_ms = finished
        .unwrap_or(now)
        .saturating_duration_since(started)
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;

    json!({
        "job_id": id.raw(),
        "label": label,
        "state": state_name(state),
        "cancellable": cancellable,
        "progress": progress,
        "detail": detail,
        "cancel_requested": cancel_requested,
        "elapsed_ms": elapsed_ms
    })
}

fn state_name(state: JobState) -> &'static str {
    match state {
        JobState::Queued => "queued",
        JobState::Running => "running",
        JobState::Done => "done",
        JobState::Failed => "failed",
        JobState::Cancelled => "cancelled",
    }
}

fn unknown_job(id: JobId) -> String {
    format!("job {} is unknown or has been pruned", id.raw())
}

fn terminal_cancel_error(id: JobId, state: JobState) -> String {
    format!(
        "job {} is already {} and cannot be cancelled",
        id.raw(),
        state_name(state)
    )
}

fn commit_phase_cancel_error(id: JobId) -> String {
    format!(
        "job {} has begun committing and no longer accepts cancellation",
        id.raw()
    )
}

fn spec(name: &str, description: &str, parameters: Value, tier: ToolTier) -> HostToolSpec {
    HostToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
        tier,
    }
}

fn empty_object_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {},
        "required": []
    })
}

fn job_id_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "job_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Positive background-job identifier."
            }
        },
        "required": ["job_id"]
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Request {
    List,
    Status(JobId),
    Cancel(JobId),
}

fn parse_request(name: &str, arguments: &Value) -> Result<Request, String> {
    if !TOOL_NAMES.contains(&name) {
        return Err(format!("unknown job tool '{name}'"));
    }
    let object = arguments
        .as_object()
        .ok_or_else(|| format!("{name} arguments must be an object"))?;

    match name {
        JOB_LIST => {
            validate_no_fields(name, object)?;
            Ok(Request::List)
        }
        JOB_STATUS => parse_job_id(name, object).map(Request::Status),
        JOB_CANCEL => parse_job_id(name, object).map(Request::Cancel),
        _ => Err(format!("unknown job tool '{name}'")),
    }
}

fn validate_no_fields(tool: &str, object: &Map<String, Value>) -> Result<(), String> {
    if !object.is_empty() {
        return Err(format!("{tool} does not accept arguments"));
    }
    Ok(())
}

fn parse_job_id(tool: &str, object: &Map<String, Value>) -> Result<JobId, String> {
    if object.keys().any(|key| key != "job_id") {
        return Err(format!("{tool} has an unknown argument"));
    }
    let value = object
        .get("job_id")
        .ok_or_else(|| format!("{tool} is missing required argument 'job_id'"))?;
    let raw = value.as_u64().filter(|raw| *raw != 0).ok_or_else(|| {
        format!("{tool} argument 'job_id' must be a positive integer that fits in u64")
    })?;
    Ok(JobId::from_raw(raw))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::{Arc, Condvar, Mutex, mpsc};
    use std::time::Duration;

    use super::*;

    const WAIT: Duration = Duration::from_secs(10);

    fn manager_with_terminals() -> (JobManager, mpsc::Receiver<JobSnapshot>) {
        let manager = JobManager::new();
        let (tx, rx) = mpsc::channel();
        manager.subscribe(move |snapshot| {
            if snapshot.state.is_terminal() {
                let _ = tx.send(snapshot);
            }
        });
        (manager, rx)
    }

    fn terminal_for(rx: &mpsc::Receiver<JobSnapshot>, id: JobId) -> JobSnapshot {
        loop {
            let snapshot = rx.recv_timeout(WAIT).expect("timed out waiting for job");
            if snapshot.id == id {
                assert!(
                    !snapshot.cancellable,
                    "terminal job {id} still accepts cancellation"
                );
                return snapshot;
            }
        }
    }

    fn invoke(manager: &JobManager, name: &str, arguments: Value) -> ToolOutput {
        call(manager, name, &arguments, &AtomicBool::new(false)).expect("job tool call")
    }

    fn output_json(output: &ToolOutput) -> Value {
        assert!(output.images.is_empty());
        serde_json::from_str(&output.text).expect("compact JSON tool output")
    }

    fn job_row(value: &Value) -> &Map<String, Value> {
        let row = value.as_object().expect("job row object");
        assert_eq!(
            row.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "cancel_requested",
                "cancellable",
                "detail",
                "elapsed_ms",
                "job_id",
                "label",
                "progress",
                "state",
            ])
        );
        assert!(row["job_id"].as_u64().is_some());
        assert!(row["label"].is_string());
        assert!(row["state"].is_string());
        assert!(row["progress"].is_null() || row["progress"].is_number());
        assert!(row["detail"].is_string());
        assert!(row["cancel_requested"].is_boolean());
        assert!(row["cancellable"].is_boolean());
        assert!(row["elapsed_ms"].as_u64().is_some());
        row
    }

    #[test]
    fn specs_have_exact_names_tiers_and_strict_schemas() {
        let registry = specs();
        assert_eq!(
            registry
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            TOOL_NAMES
        );
        assert_eq!(
            registry.iter().map(|entry| entry.tier).collect::<Vec<_>>(),
            [ToolTier::ReadOnly, ToolTier::ReadOnly, ToolTier::Workspace]
        );
        assert_eq!(registry[0].parameters, empty_object_schema());
        assert_eq!(registry[1].parameters, job_id_schema());
        assert_eq!(registry[2].parameters, job_id_schema());
        for entry in registry {
            assert_eq!(entry.parameters["type"], "object");
            assert_eq!(entry.parameters["additionalProperties"], false);
            assert!(!entry.description.is_empty());
        }
    }

    #[test]
    fn list_is_newest_first_capped_truncated_and_shape_stable() {
        let (manager, terminal_rx) = manager_with_terminals();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let count = MAX_LIST_JOBS + 1;
        let mut ids = Vec::with_capacity(count);
        for index in 0..count {
            let gate = Arc::clone(&gate);
            ids.push(manager.spawn(format!("job-{index}"), move |_| {
                let (lock, wake) = &*gate;
                let mut released = lock.lock().expect("gate lock");
                while !*released {
                    released = wake.wait(released).expect("gate wait");
                }
                Ok("released".into())
            }));
        }

        let output = invoke(&manager, JOB_LIST, json!({}));
        {
            let (lock, wake) = &*gate;
            *lock.lock().expect("gate lock") = true;
            wake.notify_all();
        }
        for _ in 0..count {
            terminal_rx
                .recv_timeout(WAIT)
                .expect("blocked list fixture did not finish");
        }

        assert!(!output.text.contains('\n'), "response must be compact JSON");
        let value = output_json(&output);
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["jobs", "status", "total", "truncated"])
        );
        assert_eq!(value["status"], "ok");
        assert_eq!(value["total"], count);
        assert_eq!(value["truncated"], true);
        let jobs = value["jobs"].as_array().expect("jobs array");
        assert_eq!(jobs.len(), MAX_LIST_JOBS);
        let actual_ids = jobs
            .iter()
            .map(|job| job_row(job)["job_id"].as_u64().unwrap())
            .collect::<Vec<_>>();
        let expected_ids = ids
            .iter()
            .rev()
            .take(MAX_LIST_JOBS)
            .map(|id| id.raw())
            .collect::<Vec<_>>();
        assert_eq!(actual_ids, expected_ids);
        for job in jobs {
            let row = job_row(job);
            assert!(matches!(row["state"].as_str(), Some("queued" | "running")));
            assert!(row["progress"].is_null());
            assert_eq!(row["detail"], "");
            assert_eq!(row["cancel_requested"], false);
            assert_eq!(row["cancellable"], true);
            assert!(row.get("started").is_none());
            assert!(row.get("finished").is_none());
        }
    }

    #[test]
    fn status_reports_unknown_and_every_state_with_progress() {
        let (manager, terminal_rx) = manager_with_terminals();

        let queued_manager = manager.clone();
        let (queued_tx, queued_rx) = mpsc::channel();
        manager.subscribe(move |snapshot| {
            if snapshot.label == "queued-status" && snapshot.state == JobState::Queued {
                let output = call(
                    &queued_manager,
                    JOB_STATUS,
                    &json!({ "job_id": snapshot.id.raw() }),
                    &AtomicBool::new(false),
                );
                let _ = queued_tx.send(output);
            }
        });
        let queued_id = manager.spawn("queued-status", |_| Ok("done".into()));
        let queued = output_json(
            &queued_rx
                .recv_timeout(WAIT)
                .expect("queued status callback")
                .expect("queued status call"),
        );
        let queued_row = job_row(&queued["job"]);
        assert_eq!(queued_row["state"], "queued");
        assert_eq!(queued_row["cancellable"], true);
        assert_eq!(queued_row["cancel_requested"], false);
        assert!(queued_row["progress"].is_null());
        terminal_for(&terminal_rx, queued_id);

        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let running_id = manager.spawn("progress-status", move |context| {
            context.set_progress(0.25, "one quarter");
            ready_tx.send(()).expect("ready receiver");
            release_rx.recv().expect("release running job");
            Ok("complete".into())
        });
        ready_rx.recv_timeout(WAIT).expect("running job ready");
        let running = output_json(&invoke(
            &manager,
            JOB_STATUS,
            json!({ "job_id": running_id.raw() }),
        ));
        let running_row = job_row(&running["job"]);
        assert_eq!(running["status"], "ok");
        assert_eq!(running_row["state"], "running");
        assert_eq!(running_row["cancellable"], true);
        assert_eq!(running_row["cancel_requested"], false);
        assert_eq!(running_row["progress"], 0.25);
        assert_eq!(running_row["detail"], "one quarter");

        release_tx.send(()).expect("release running job");
        terminal_for(&terminal_rx, running_id);
        let done = output_json(&invoke(
            &manager,
            JOB_STATUS,
            json!({ "job_id": running_id.raw() }),
        ));
        let done_row = job_row(&done["job"]);
        assert_eq!(done_row["state"], "done");
        assert_eq!(done_row["cancellable"], false);
        assert_eq!(done_row["cancel_requested"], false);
        assert_eq!(done_row["progress"], 1.0);
        assert_eq!(done_row["detail"], "complete");

        let failed_id = manager.spawn("failed-status", |context| {
            context.set_progress(0.375, "before failure");
            Err("fixture failed".into())
        });
        terminal_for(&terminal_rx, failed_id);
        let failed = output_json(&invoke(
            &manager,
            JOB_STATUS,
            json!({ "job_id": failed_id.raw() }),
        ));
        let failed_row = job_row(&failed["job"]);
        assert_eq!(failed_row["state"], "failed");
        assert_eq!(failed_row["cancellable"], false);
        assert_eq!(failed_row["cancel_requested"], false);
        assert_eq!(failed_row["progress"], 0.375);
        assert_eq!(failed_row["detail"], "fixture failed");

        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let cancelled_id = manager.spawn("cancelled-status", move |_| {
            started_tx.send(()).expect("started receiver");
            finish_rx.recv().expect("finish cancelled job");
            Ok("stopped".into())
        });
        started_rx.recv_timeout(WAIT).expect("cancel fixture ready");
        assert!(manager.cancel(cancelled_id));
        finish_tx.send(()).expect("finish cancelled job");
        terminal_for(&terminal_rx, cancelled_id);
        let cancelled = output_json(&invoke(
            &manager,
            JOB_STATUS,
            json!({ "job_id": cancelled_id.raw() }),
        ));
        let cancelled_row = job_row(&cancelled["job"]);
        assert_eq!(cancelled_row["state"], "cancelled");
        assert_eq!(cancelled_row["cancellable"], false);
        assert!(cancelled_row["progress"].is_null());
        assert_eq!(cancelled_row["cancel_requested"], true);

        let error = call(
            &manager,
            JOB_STATUS,
            &json!({ "job_id": u64::MAX }),
            &AtomicBool::new(false),
        )
        .expect_err("unknown status must fail");
        assert!(error.contains("unknown or has been pruned"));
    }

    #[test]
    fn status_and_cancel_distinguish_live_commit_phase() {
        let (manager, terminal_rx) = manager_with_terminals();
        let (committing_tx, committing_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (observed_tx, observed_rx) = mpsc::channel();
        let id = manager.spawn("commit-phase-status", move |context| {
            let token = context.cancellation_token();
            assert!(context.try_begin_commit());
            committing_tx
                .send(token.clone())
                .expect("commit token receiver");
            release_rx.recv().expect("release commit-phase job");
            observed_tx
                .send((context.cancelled(), token.cancelled()))
                .expect("cancellation observer");
            Ok("committed".into())
        });

        let token = committing_rx
            .recv_timeout(WAIT)
            .expect("job did not enter commit phase");
        let status = output_json(&invoke(&manager, JOB_STATUS, json!({ "job_id": id.raw() })));
        let row = job_row(&status["job"]);
        assert_eq!(row["state"], "running");
        assert_eq!(row["cancellable"], false);
        assert_eq!(row["cancel_requested"], false);

        let error = call(
            &manager,
            JOB_CANCEL,
            &json!({ "job_id": id.raw() }),
            &AtomicBool::new(false),
        )
        .expect_err("commit-phase cancellation must fail");
        assert_eq!(
            error,
            format!(
                "job {} has begun committing and no longer accepts cancellation",
                id.raw()
            )
        );
        assert!(error.len() <= 96);
        assert!(!token.cancelled());
        let committing = manager.get(id).expect("commit-phase job");
        assert_eq!(committing.state, JobState::Running);
        assert!(!committing.cancellable);
        assert!(!committing.cancel_requested);

        release_tx.send(()).expect("release commit-phase job");
        assert_eq!(
            observed_rx
                .recv_timeout(WAIT)
                .expect("job did not report cancellation state"),
            (false, false)
        );
        let terminal = terminal_for(&terminal_rx, id);
        assert_eq!(terminal.state, JobState::Done);
        assert!(!terminal.cancel_requested);
        assert!(!token.cancelled());

        let done = output_json(&invoke(&manager, JOB_STATUS, json!({ "job_id": id.raw() })));
        let done_row = job_row(&done["job"]);
        assert_eq!(done_row["state"], "done");
        assert_eq!(done_row["cancellable"], false);
        assert_eq!(done_row["cancel_requested"], false);
    }

    #[test]
    fn cancel_handles_queued_and_running_but_rejects_terminal_and_unknown() {
        let (manager, terminal_rx) = manager_with_terminals();

        let queued_manager = manager.clone();
        let (response_tx, response_rx) = mpsc::channel();
        manager.subscribe(move |snapshot| {
            if snapshot.label == "queued-cancel" && snapshot.state == JobState::Queued {
                let response = call(
                    &queued_manager,
                    JOB_CANCEL,
                    &json!({ "job_id": snapshot.id.raw() }),
                    &AtomicBool::new(false),
                );
                let _ = response_tx.send(response);
            }
        });
        let (saw_cancel_tx, saw_cancel_rx) = mpsc::channel();
        let queued_id = manager.spawn("queued-cancel", move |context| {
            saw_cancel_tx
                .send(context.cancelled())
                .expect("queued cancel observer");
            Ok("stopped".into())
        });
        let queued_cancel = output_json(
            &response_rx
                .recv_timeout(WAIT)
                .expect("queued cancel callback")
                .expect("queued cancellation"),
        );
        assert_eq!(
            queued_cancel,
            json!({
                "status": "ok",
                "job_id": queued_id.raw(),
                "cancel_requested": true
            })
        );
        assert!(saw_cancel_rx.recv_timeout(WAIT).expect("queued job ran"));
        assert_eq!(
            terminal_for(&terminal_rx, queued_id).state,
            JobState::Cancelled
        );

        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let running_id = manager.spawn("running-cancel", move |_| {
            started_tx.send(()).expect("running cancel observer");
            release_rx.recv().expect("release running cancel");
            Ok("stopped".into())
        });
        started_rx.recv_timeout(WAIT).expect("running job ready");
        let running_cancel = output_json(&invoke(
            &manager,
            JOB_CANCEL,
            json!({ "job_id": running_id.raw() }),
        ));
        assert_eq!(running_cancel["status"], "ok");
        assert_eq!(running_cancel["job_id"], running_id.raw());
        assert_eq!(running_cancel["cancel_requested"], true);
        assert!(manager.get(running_id).unwrap().cancel_requested);
        release_tx.send(()).expect("release running cancel");
        assert_eq!(
            terminal_for(&terminal_rx, running_id).state,
            JobState::Cancelled
        );

        let terminal_error = call(
            &manager,
            JOB_CANCEL,
            &json!({ "job_id": running_id.raw() }),
            &AtomicBool::new(false),
        )
        .expect_err("terminal cancellation must fail");
        assert!(terminal_error.contains("already cancelled"));
        assert!(terminal_error.contains("cannot be cancelled"));

        let unknown_error = call(
            &manager,
            JOB_CANCEL,
            &json!({ "job_id": u64::MAX }),
            &AtomicBool::new(false),
        )
        .expect_err("unknown cancellation must fail");
        assert!(unknown_error.contains("unknown or has been pruned"));
    }

    #[test]
    fn parser_rejects_malformed_ids_shapes_and_extra_fields_without_effects() {
        let (manager, terminal_rx) = manager_with_terminals();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let id = manager.spawn("argument-fixture", move |_| {
            started_tx.send(()).expect("argument fixture observer");
            release_rx.recv().expect("release argument fixture");
            Ok("done".into())
        });
        started_rx
            .recv_timeout(WAIT)
            .expect("argument fixture ready");

        let overflow: Value =
            serde_json::from_str(r#"{"job_id":18446744073709551616}"#).expect("overflow JSON");
        let malformed = [
            Value::Null,
            json!([]),
            json!({}),
            json!({ "job_id": 0 }),
            json!({ "job_id": -1 }),
            json!({ "job_id": 1.5 }),
            json!({ "job_id": "1" }),
            json!({ "job_id": true }),
            overflow,
            json!({ "job_id": id.raw(), "extra": true }),
        ];
        for name in [JOB_STATUS, JOB_CANCEL] {
            for arguments in &malformed {
                assert!(
                    call(&manager, name, arguments, &AtomicBool::new(false)).is_err(),
                    "{name} accepted malformed arguments: {arguments}"
                );
            }
        }
        assert!(!manager.get(id).unwrap().cancel_requested);

        for arguments in [Value::Null, json!([]), json!({ "extra": true })] {
            assert!(call(&manager, JOB_LIST, &arguments, &AtomicBool::new(false)).is_err());
        }
        assert!(call(&manager, "job_future", &json!({}), &AtomicBool::new(false)).is_err());

        release_tx.send(()).expect("release argument fixture");
        terminal_for(&terminal_rx, id);
    }
}
