---
name: cutlass-release
description: >-
  Cut a public Cutlass desktop alpha GitHub Release (version bump, changelog,
  PR into main, green CI, tag from main, watch release.yml). Use when the user
  asks to release, ship, publish, cut an alpha, bump for release, or prepare
  alpha-X.Y.Z desktop binaries.
---

# Cutlass desktop alpha release

Only run this when the user **explicitly** asks to release / publish / cut an
alpha. Tagging and pushing are allowed in that case (see `git-workflow`).

Canonical packaging notes: [`packaging/README.md`](../../../packaging/README.md).
Pitfalls and version math: [`reference.md`](reference.md).
Cursor rule (hard gates): [`.cursor/rules/release.mdc`](../../../.cursor/rules/release.mdc).

## Ship path (do not skip)

```
macos-dev prep → push → PR → main → green CI → merge → tag main → release.yml
```

Never tag `macos-dev`. Never publish binaries from a dirty or unmerged branch.

## Checklist

Copy and track:

```
Release progress:
- [ ] 1. Decide next Cargo + tag versions
- [ ] 2. Bump manifests (Cargo, lock, Info.plist, packaging README)
- [ ] 3. Rewrite CHANGELOG.md for this release only
- [ ] 4. Local fmt + clippy -D warnings clean
- [ ] 5. Commit prep (+ any CI fixes) on macos-dev; push
- [ ] 6. Open PR macos-dev → main
- [ ] 7. Wait for ALL PR checks green; fix/push until green
- [ ] 8. Merge PR into main
- [ ] 9. Tag alpha-X.Y.Z on origin/main; push tag
- [ ] 10. Watch Release workflow; verify GitHub Latest + assets
```

## Step 1 — Version

| Artifact | Pattern | Example |
|----------|---------|---------|
| Cargo workspace | `X.Y.Z-alpha.N` | `0.6.1-alpha.0` |
| Git tag | `alpha-X.Y.Z` | `alpha-0.6.1` |
| Info.plist short | `X.Y.Z-alpha` | `0.6.1-alpha` |
| Info.plist build | same as Cargo | `0.6.1-alpha.0` |
| Zip names | from Cargo | `Cutlass-0.6.1-alpha.0-macos-arm64.zip` |

1. `gh release list --limit 5` and read root `Cargo.toml` `[workspace.package].version`.
2. Pick the next version **past** the last shipped Cargo version.
3. Default: patch bump within the current minor (`0.6.0` → `0.6.1`) for polish /
   fixes; minor bump for a large feature drop. Confirm with the user if unclear.

`cutlass-py` versions/tags are independent — leave them alone unless asked.

## Step 2 — Manifest bump (on `macos-dev`)

1. Root [`Cargo.toml`](../../../Cargo.toml): `[workspace.package].version`.
2. Refresh lock: `cargo update -p cutlass-desktop --precise <new-version>` (or
   equivalent) so every workspace crate in `Cargo.lock` matches.
3. [`packaging/macos/Info.plist`](../../../packaging/macos/Info.plist): both
   version keys (this file often lags — fix it every release).
4. [`packaging/README.md`](../../../packaging/README.md): version examples.

## Step 3 — Changelog

Rewrite [`CHANGELOG.md`](../../../CHANGELOG.md) in the existing shape:

- Header + link to prior releases on GitHub
- One `## [alpha-X.Y.Z] — YYYY-MM-DD` section (Added / Changed / Removed as needed)
- Bottom link: `[alpha-X.Y.Z]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-X.Y.Z`

Derive bullets from `git log <last-tag>..HEAD`. User-facing only; omit internal
gallery/dev-only noise unless it matters to downloaders.

`release.yml` sets `body_path: CHANGELOG.md` — the **entire file** is the
Release notes. Do not leave old release sections in the file.

## Step 4 — Local CI parity (before PR)

CI runs (see [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

If clippy fails with many `collapsible_if` / similar:

```bash
cargo clippy --workspace --all-targets --fix --allow-dirty -- -D warnings
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

Commit fixes separately (`fix: clear clippy -D warnings for CI`). Do not merge red.

## Step 5 — PR → main

1. Commit prep on `macos-dev`, push (`git push origin macos-dev`).
2. Open PR targeting `main` (title like prior: `Release alpha-X.Y.Z: …`).
3. Watch checks: `gh pr checks <n> --watch`.
   Expected: `build / test / clippy`, `release build / macos arm64`,
   `release build / windows x86_64`, plus GitGuardian.
4. Windows release compile often takes **15–25+ minutes** — keep waiting.
5. Merge with a merge commit when green (`gh pr merge <n> --merge`). Keep
   `macos-dev` unless the user wants the branch deleted.

## Step 6 — Tag from `main` only

```bash
git fetch origin main
# Prefer tagging the remote tip (local `main` may live in another worktree):
git tag alpha-X.Y.Z origin/main
git push origin alpha-X.Y.Z
```

If `git checkout main` fails with “already used by worktree”, **do not** force
checkout — tag `origin/main` as above.

Confirm: `git rev-parse origin/main` equals `git rev-list -n1 alpha-X.Y.Z`.

## Step 7 — Publish + verify

Tag push triggers [`.github/workflows/release.yml`](../../../.github/workflows/release.yml).

```bash
gh run list --workflow=release.yml --limit 3
gh run watch <id> --exit-status
gh release view alpha-X.Y.Z --json tagName,publishedAt,url,assets \
  --jq '{tagName, publishedAt, url, assets: [.assets[].name]}'
gh release list --limit 3   # should show Latest = this tag
```

Expect assets:

- `Cutlass-<cargo>-macos-arm64.zip`
- `Cutlass-<cargo>-linux-x86_64.tar.gz`
- `Cutlass-<cargo>-windows-x86_64.zip` + `-Setup.exe`
- `Cutlass-<cargo>-windows-arm64.zip` + `-Setup.exe`

Release settings (intentional): `prerelease: false`, `make_latest: true` so
alpha builds appear as Latest (still named `alpha-*`).

## Out of scope unless asked

- Apple notarization / Developer ID (adhoc Gatekeeper dance remains)
- DMG, Intel macOS
- Windows Authenticode
- `cutlass-py` / `py-v*` / PyPI (`pywheels.yml`)
- Mobile / TestFlight
