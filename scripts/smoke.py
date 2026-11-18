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
        served = subprocess.Popen(
            [binary, "mcp", project],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True,
        )

        def send(message: dict) -> None:
            served.stdin.write(json.dumps(message) + "\n")
            served.stdin.flush()

        def ask(message: dict) -> dict:
            send(message)
            line = served.stdout.readline()
            assert line, served.stderr.read()
            return json.loads(line)

        def call(id: int, name: str, arguments: dict) -> dict:
            return ask({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})["result"]

        try:
            hello = ask({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            assert hello["result"]["serverInfo"]["name"] == "cutlass", hello
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools = ask({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            assert len(tools["result"]["tools"]) > 10, tools

            # A project opens at a revision of its own, so read it first.
            read = call(3, "draft_query", {"input": "project"})["content"][0]["text"]
            assert read.startswith("revision="), read
            revision = int(read.split()[0].removeprefix("revision="))
            applied = call(4, "draft_apply", {
                "expected_revision": revision,
                # A place names a material the draft holds, so the
                # picture is imported in the same step.
                "requests": [
                    {"op": "import", "medium": {"video": {
                        "id": "media", "type": "video", "duration": 1_000_000}}},
                    {"op": "place", "track_id": "main", "segment": segment},
                ],
            })["content"][0]["text"]
            assert applied == f"revision={revision + 1} committed=true", applied

            # The local sandbox: a Rhai script in a child process.
            ran = call(5, "sandbox_run", {
                "source": "let t = 0; for c in input.clips { t += c } #{ total: t }",
                "input": {"clips": [1, 2, 3]},
            })
            assert not ran["isError"] and json.loads(ran["content"][0]["text"]) == {"total": 6}, ran
            # A script naming the project folder is refused before it runs.
            refused = call(6, "sandbox_run", {"source": "1", "input": {"path": str(project)}})
            assert refused["isError"] and "project" in refused["content"][0]["text"], refused

            served.stdin.close()
            assert served.wait(timeout=120) == 0, served.stderr.read()
        finally:
            if served.poll() is None:
                served.kill()
                served.wait()

        draft = json.loads((project / "draft_info.json").read_text())
        placed = [segment["id"] for track in draft["tracks"] for segment in track["segments"]]
        assert placed == ["a"], placed
        locked = project / ".locked"
        assert not locked.exists() or '"open":true' not in locked.read_text(), locked.read_text()
    print("smoke: ok")


if __name__ == "__main__":
    main()
