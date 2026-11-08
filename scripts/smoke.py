"""Smoke test the built desktop binary with no window.

Usage: python3 scripts/smoke.py path/to/cutlass

It points the Cutlass home at a temporary folder, makes a project with
`cutlass new`, then drives it through `cutlass mcp` on stdio: it lists the
tools, reads the draft, places a clip, runs a sandbox script, and checks
the edit was saved and the lock let go. It runs the same on macOS, Windows and Linux.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        home = scratch / "home"
        home.mkdir()
        env = dict(os.environ)
        # Each system reads one of these for the Cutlass home.
        env["HOME"] = str(home)
        env["APPDATA"] = str(home / "AppData")
        env["XDG_DATA_HOME"] = str(home / "share")

        made = subprocess.run(
            [binary, "new", scratch / "projects"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert made.returncode == 0, made.stderr
        project = Path(made.stdout.strip())
        assert (project / "draft_info.json").is_file(), project

        segment = {
            "id": "a",
            "material_id": "media",
            "target_timerange": {"start": 0, "duration": 1_000_000},
        }
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "draft_query", "arguments": {"input": "project"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "draft_apply", "arguments": {
                 "expected_revision": 0,
                 # A place names a material the draft holds, so the
                 # picture is imported in the same step.
                 "requests": [
                     {"op": "import", "medium": {"video": {
                         "id": "media", "type": "video", "duration": 1_000_000}}},
                     {"op": "place", "track_id": "main", "segment": segment},
                 ],
             }}},
            # The local sandbox: a Rhai script in a child process.
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "sandbox_run", "arguments": {
                 "source": "let t = 0; for c in input.clips { t += c } #{ total: t }",
                 "input": {"clips": [1, 2, 3]},
             }}},
            # A script naming the project folder is refused before it runs.
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
             "params": {"name": "sandbox_run", "arguments": {
                 "source": "1", "input": {"path": str(project)},
             }}},
        ]
        served = subprocess.run(
            [binary, "mcp", project],
            input="".join(json.dumps(message) + "\n" for message in messages),
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert served.returncode == 0, served.stderr
        answers = [json.loads(line) for line in served.stdout.splitlines() if line.strip()]
        assert len(answers) == 6, served.stdout
        assert answers[0]["result"]["serverInfo"]["name"] == "cutlass", answers[0]
        assert len(answers[1]["result"]["tools"]) > 10, answers[1]
        read = answers[2]["result"]["content"][0]["text"]
        assert "revision=0" in read, read
        applied = answers[3]["result"]["content"][0]["text"]
        assert applied == "revision=1 committed=true", answers[3]
        ran = answers[4]["result"]
        assert not ran["isError"] and json.loads(ran["content"][0]["text"]) == {"total": 6}, ran
        refused = answers[5]["result"]
        assert refused["isError"] and "project" in refused["content"][0]["text"], refused

        draft = json.loads((project / "draft_info.json").read_text())
        placed = [segment["id"] for track in draft["tracks"] for segment in track["segments"]]
        assert placed == ["a"], placed
        locked = project / ".locked"
        assert not locked.exists() or '"open":true' not in locked.read_text(), locked.read_text()
    print("smoke: ok")


if __name__ == "__main__":
    main()
