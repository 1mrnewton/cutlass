# Release reference (gotchas from alpha-0.6.1)

## Versioning conventions

- **Cargo** uses semver prerelease: `0.6.1-alpha.0`.
- **Git tag** drops the `-alpha.N` suffix: `alpha-0.6.1`.
- **Artifact filenames** use the Cargo version, not the tag:
  `Cutlass-0.6.1-alpha.0-macos-arm64.zip`.
- Bumping only the tag without Cargo (or the reverse) breaks packaging and
  confuses downloaders — bump both together.
- `CFBundleShortVersionString` is `0.6.1-alpha` (no `.0`);
  `CFBundleVersion` matches Cargo (`0.6.1-alpha.0`).
- `Info.plist` has shipped stale (still `0.5.3` while Cargo was `0.6.0`) —
  always diff it on prep.

## Workflows

| Trigger | Workflow | Role |
|---------|----------|------|
| PR / push `main` | `.github/workflows/ci.yml` | fmt, clippy **`-D warnings`**, tests; macOS + Windows release **compile** smoke |
| Tag `alpha-*` | `.github/workflows/release.yml` | Full package + GitHub Release |
| Tag `py-v*` | `.github/workflows/pywheels.yml` | PyPI wheels only |

Release matrix: macOS arm64 (`macos-14`), Linux x86_64 (UI preview; media dormant),
Windows x86_64 + arm64 (zip + Inno Setup). No Intel macOS runner.

## Branch layout

- Day-to-day work: `macos-dev`.
- Public releases: merge into `main`, then tag `main`.
- `main` is often checked out in a **separate git worktree** (e.g.
  `cutlass-main`). `git checkout main` from `macos-dev` then fails — tag
  `origin/main` after `git fetch` instead of fighting worktrees.
- Prefer merge commits for release PRs (matches prior `Release alpha-*` PRs).

## CI timing

- `build / test / clippy` (Linux): often ~10–15 min after deps cached.
- macOS release compile smoke: ~5–10 min.
- Windows release compile smoke: commonly **15–25+ min** — do not abort early.
- Full `release.yml` (all platforms + publish): plan for ~20–40 min.

## Clippy / fmt lessons

- Local `cargo clippy --workspace --all-targets` **without** `-D warnings` can
  look clean while CI fails. Always match CI flags before opening the PR.
- After a Slint / toolchain bump, expect a burst of `collapsible_if`,
  `manual_is_multiple_of`, `redundant_closure`, and
  `items_after_test_module` (move helpers **above** `#[cfg(test)] mod tests`).
- `cargo clippy ... --fix --allow-dirty` then `cargo fmt --all` is the fastest
  cleanup; re-run with `-D warnings` until exit 0.
- Commit clippy cleanup separately from the version/changelog prep commit.

## CHANGELOG shape

The Release action does **not** generate notes. Whatever is in `CHANGELOG.md`
at the tagged commit becomes the Release body. Keep only the latest section
plus the “previous releases on GitHub” pointer (same pattern as 0.6.0 / 0.6.1).

## Packaging scripts

| Script | Output |
|--------|--------|
| `scripts/package-macos.sh` | `dist/Cutlass-*-macos-*.zip` (adhoc codesign `.app`) |
| `scripts/package-linux.sh` | `dist/Cutlass-*-linux-*.tar.gz` |
| `scripts/package-windows-installer.ps1` | zip + `Setup.exe` |

`dist/` is gitignored. Optional local smoke after bump:

```bash
cargo build --release -p cutlass-desktop
./scripts/package-macos.sh
# confirm zip name matches new Cargo version
```

## Gatekeeper / signing (current alpha)

- macOS zips are **adhoc-signed**, not notarized — users may need
  Right-click → Open once (`packaging/macos/INSTALL.txt`).
- Windows installers are unsigned — SmartScreen warning expected.
- Do not promise notarization / Authenticode in release notes unless shipped.

## What not to mix in

- **Python**: `crates/cutlass-py` is workspace-excluded; ship with `py-v*` only.
- **Mobile**: Xcode / TestFlight / Android JNI — not `release.yml`.
- Do not force-push tags or rewrite release history unless the user explicitly
  orders a re-cut.

## Useful commands

```bash
# Last desktop release + commits since
gh release list --limit 5
git log alpha-<prev>..HEAD --oneline

# PR checks
gh pr checks <n> --watch
gh run view <id> --log-failed

# After tag
gh run list --workflow=release.yml --limit 3
gh release view alpha-<ver>
```

## Example: alpha-0.6.1

- Prep commits on `macos-dev`: version/changelog, then clippy cleanup.
- PR #29 → all CI green (Windows ~20 min) → merge to `main`.
- Tagged `origin/main` as `alpha-0.6.1` (local `main` was in another worktree).
- `release.yml` published six assets; GitHub showed Latest = `alpha-0.6.1`.
