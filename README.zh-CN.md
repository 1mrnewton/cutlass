# Cutlass

[English](README.md) · **简体中文** · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass 是一款时间线视频编辑器。桌面、iOS 和 Android 共用同一份草稿和同一个会话。任一外壳上的手势都会变成同一条请求。同一份草稿、同一个时刻，播放和导出看到的是同一帧。

文档中的时间是带符号的微秒计数。片段以这个单位保存两个范围。源范围是它使用的媒体部分。目标范围是它在时间线上的位置。

译文与这份代码树不一致时，以[英文 README](README.md) 为准。

## 现状

工作区还处于早期。

| 部分 | 状态 |
| --- | --- |
| `cutlass-model` | 草稿：轨道、片段、素材池，以及微秒时间 |
| `cutlass-session` | 放置、移动和修剪作为一步撤销。失败的请求让草稿保持原样。撤销和重做都会推进修订号 |
| 其余 crate | 空成员。它们为引擎、容器、编解码器、播放、导出和 agent 保留位置 |
| `apps/desktop` | 打印 `cutlass` 后返回。窗口稍后才有 |
| `apps/ios`、`apps/android` | 稍后进入这个目录树 |

容器和编解码器是原生 Rust，每个系列或每种编解码器一个 crate。在 Apple、Windows 和 Android 上，平台媒体框架是快速路径。Linux 使用 Rust 编解码器和软件预览。

## 布局

```text
apps/desktop/          egui 与 eframe
crates/                每个组件一个 crate
crates/containers/     每个容器系列一个 crate
crates/codecs/         每种编解码器一个 crate
```

crate 名称使用 `cutlass-` 前缀。桌面二进制直接链接这些 crate。手机外壳只链接 `cutlass-ffi`。

## 构建

安装 stable Rust。`rust-toolchain.toml` 选定通道，以及 Clippy 和 rustfmt。

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` 在整个工作区运行 Clippy，并把警告视为错误。Ubuntu 上的 CI 会运行这些检查以及 `cargo deny`。

## 许可证

Cutlass 可依你的选择，使用下列任一许可证：

* Apache License 2.0（[LICENSE-APACHE](LICENSE-APACHE) 或 <https://www.apache.org/licenses/LICENSE-2.0>）
* MIT 许可证（[LICENSE-MIT](LICENSE-MIT) 或 <https://opensource.org/licenses/MIT>）

Copyright (c) 2026 Emmanuel Aidoo
