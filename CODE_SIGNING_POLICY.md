# Code signing policy

Free code signing provided by [SignPath.io](https://signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

This policy covers the Windows installers Cutlass publishes on its [GitHub releases](https://github.com/1mrnewton/cutlass/releases): `Cutlass-x86_64-setup.exe` and `Cutlass-aarch64-setup.exe`, and the `cutlass.exe` inside each. They are built from this public repository only, and nothing else is signed with the SignPath Foundation certificate. The macOS, iOS and Android packages are signed with the project's own Apple and Android keys, outside this policy.

## How a release is built and signed

- A release starts from a version tag `v<version>` on `main`. The release workflow, [`.github/workflows/release.yml`](.github/workflows/release.yml), refuses a tag whose commit is not on `main`, or on which the required `rust` check did not pass.
- Every change reaches `main` through a pull request, with the checks green and the branch up to date with `main`.
- The installers are built on GitHub-hosted runners from the tagged commit, smoke tested by installing them, and sent to SignPath from that same run. Nothing built on a personal machine is signed.
- Each signing request is approved by hand in SignPath by an approver below.
- The signed installers go up as a draft release with a SHA-256 checksum beside each. A check job downloads the draft and verifies every checksum before the draft is published by hand.

## Team roles

| Role | Who |
| --- | --- |
| Committers and reviewers | [1mrnewton](https://github.com/1mrnewton) |
| Approvers | [1mrnewton](https://github.com/1mrnewton) |

Changes from other contributors arrive as pull requests and are reviewed by a committer before they are merged. Everyone in these roles uses multi-factor authentication on GitHub and on SignPath.

## Privacy

Cutlass has no account, no telemetry and no analytics. Footage and projects stay on your disk. The program connects to another computer only in these cases:

- **Update check:** once a day it reads the latest release from `api.github.com`, and downloads the installer when you choose to update. Turn it off with Check for updates in Settings.
- **Captions:** the first time you make captions, it downloads the speech model from `huggingface.co`, after you agree. The speech itself is heard on your computer.
- **Assistant:** the assistant sends your request and the project details it needs to the model provider you set in Settings. With a local runner, such as Ollama, the llama.cpp server or LM Studio, that stays on your computer. With a hosted provider, such as OpenAI or Anthropic, it goes to that provider under its own terms, with the key you give.
- **MCP servers:** the assistant calls the MCP servers you add, and only those.

This program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it.

## Reporting a problem

If you find a signed file that was not built from this repository, or think a signing key or account is compromised, email [founder@1mrnewton.com](mailto:founder@1mrnewton.com).
