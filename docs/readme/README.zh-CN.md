<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>免费、开源的视频编辑器，还有一个可以对话的助手。</b><br>
  支持 macOS、Windows 和 Linux。
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">下载</a> ·
  <a href="#what-you-can-do">功能</a> ·
  <a href="#the-assistant">助手</a> ·
  <a href="#formats">格式</a> ·
  <a href="#build-from-source">构建</a> ·
  <a href="#not-here-yet">尚未实现</a> ·
  <a href="https://cutlass.sh">网站</a>
</p>

[English](../../README.md) · **简体中文** · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass 是一款时间线视频编辑器。导入视频、声音和图片，在轨道上剪辑，加上文字、特效和调色，然后导出成片。每一步都可以自己动手，也可以用平常的话描述想要的剪辑，让助手在时间线上完成，之后你可以修改或撤销。

它是免费的。不需要账号，没有水印，也不上传任何内容。素材留在你的硬盘上，用到机器学习的工具都在你自己的电脑上运行。

在编辑器之下，编解码器是我们自己写的。H.264、HEVC、AV1、ProRes、AAC、Opus 以及其余的编解码器都在这个仓库里用 Rust 写成。Cutlass 不捆绑 FFmpeg。

<a name="download"></a>
## 下载

| 系统 | 文件 |
| --- | --- |
| macOS，Apple 芯片 | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows，64 位 | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux，x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

在[发布](https://github.com/1mrnewton/cutlass/releases)页面上，每个文件旁边都有一个 `.sha256` 校验和。安装之后，Cutlass 会自己检查新版本：菜单里的 **Check for updates…**。

这些构建还没有用开发者证书签名，所以第一次启动要多一步：

- **macOS:** 先打开一次应用，然后在 System Settings › Privacy & Security 里点 **Open Anyway** 允许它。
- **Windows:** 如果 SmartScreen 拦住了安装程序，选择 **More info** › **Run anyway**。
- **Linux:** 用 `chmod +x Cutlass-x86_64.AppImage` 让文件可以执行，然后运行它。

<a name="what-you-can-do"></a>
## 你可以做什么

**在时间线上剪辑。** 视频、音频和文字轨道。分割、修剪、合拢空隙、定格、倒放、镜像、旋转和裁剪。把片段编组，或者折叠成复合片段，以后再打开。替换一个片段并保留它的编辑，或者把一个片段的属性复制到另一个上。大素材可以通过代理来剪。

**速度和动画。** 设置恒定速度，或者选一条速度曲线，拖动上面的点来调整形状。用关键帧让片段动起来，可以一次移动或删除多个关键帧。入场、出场和组合动画可以直接套用。

**调色。** 基础调节、HSL、曲线和色轮。载入 `.cube` LUT，并让肤色在它之下保持自然。200 个滤镜，以及时间线上的调节图层。

**特效和转场。** 200 个特效、180 个转场和 13 种混合模式。用线性、镜像、圆形、矩形、心形或星形蒙版遮住片段。抠掉绿幕，或者点一下物体把它抠出来。

**文字、字幕和贴纸。** 文字可以设置字体、描边、阴影、背景、气泡和特效，并带入场、出场或循环动画。字幕是在你的电脑上从片段里听出来的（目前只支持英语），之后以列表形式编辑：重打一行、查找和替换、选一个模板、给单个词设置样式，再把结果导出为 SRT 文件。贴纸和形状像其他片段一样放在时间线上。

**在你电脑上运行的画面工具。** 防抖、增强画质、画面降噪、用于流畅慢动作的光流、重新打光、自动重构图和运动模糊。什么都不会发到别处。

**声音。** 音量和淡入淡出，从视频中提取音频，按声音同步视频和音频，以及标记节拍。

**导出。** MP4 或 MOV，可以导出整条时间线、一个范围或选中的片段。也可以只导出声音、GIF 或动态 WebP。在 macOS 和 Windows 上，系统的硬件编解码器能用时就由它来做，设置里有一个开关可以关掉。

<a name="the-assistant"></a>
## 助手

说出你想要的，助手就用时间线发送的同一种请求来编辑打开的项目。它做的每一次修改都是一步撤销。

- **它在你的电脑上运行。** 助手和你自己电脑上的模型运行器对话，比如 [Ollama](https://ollama.com)、llama.cpp 服务器或 LM Studio。在 Settings › Assistant 里设置地址和模型。走这条路时，提示词不会离开你的电脑。
- **它会先问你。** 任何会把数据发出这台电脑的事，都要等你允许。
- **它有技能。** 字幕、去除停顿、动态图形、产品演示和项目设置随应用一起提供，它还能调用你添加的 MCP 服务器里的工具。
- **其他智能体也能操作 Cutlass。** `cutlass mcp /path/to/project` 不开窗口，通过 stdio 提供同样的工具，所以 Claude 或任何 MCP 客户端都能读取和编辑项目。

<a name="formats"></a>
## 格式

下面的解码器和编码器是 [`crates/codecs`](../../crates/codecs) 和 [`crates/containers`](../../crates/containers) 里的 Rust 代码，以 FFmpeg 作为外部参照来测试。每个 crate 的 README 都写明它支持什么，以及它的速度和 FFmpeg 相比如何。

| | 打开 | 导出 |
| --- | :---: | :---: |
| **文件** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4、MOV，以及单独的声音 |
| **视频** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG。HEVC 在 macOS 和 Windows 上通过系统编码器导出 |
| **声音** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **图片** | JPEG, PNG, WebP, GIF，动态或静态 | GIF、动态 WebP |

<a name="build-from-source"></a>
## 从源码构建

你需要稳定版 [Rust](https://rustup.rs) 工具链和一个 C 编译器。在 Linux 上还需要 GTK 和 ALSA 的头文件：

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

然后：

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

CI 运行的检查（最后一项需要 [cargo-nextest](https://nexte.st)）：

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## 尚未实现

Cutlass 还很年轻。在应用里，画出来但还没做好的功能会变暗并标上 **Later**，所以你总能知道哪些能用。目前比较大的空缺：

- **更多容器。** MKV、WebM、AVI 和 MPEG-TS 文件还打不开。
- **人声工具。** 降噪、人声增强、响度标准化、人声分离和文字转语音。
- **生成式工具。** AI 移除、扩展和风格化。
- **跟踪。** 跟随主体的文字、贴纸和蒙版。
- **更多语言的字幕。**
- **给助手用的托管模型。** 目前它使用本地模型。
- **手机。** iOS 和 Android 应用在 [`apps`](../../apps) 里，共用同一个 Rust 核心，但桌面版优先。

附上出问题的文件的错误报告，是你能发给我们的最有用的东西。[提交 issue](https://github.com/1mrnewton/cutlass/issues)。

## 代码结构

| 文件夹 | 里面有什么 |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | 桌面编辑器，一个 egui 窗口。它构建出 `cutlass` 可执行文件 |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | 手机应用，用 SwiftUI 和 Kotlin Compose 编写。它们通过 `crates/ffi` 使用 Rust 核心 |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | 项目、带撤销的编辑请求，以及所有应用共用的编辑器行为 |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | 把一个项目和一个时间点变成一帧画面和它的声音，用于预览和导出 |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | 每种编解码器和每种文件格式各一个 crate |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | 系统的硬件编解码器 |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | 滤镜、特效、转场、蒙版、调色、文字和字幕 |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | 在设备上运行的画面模型和语音模型 |
| [`crates/agent`](../../crates/agent) | 助手：它的工具、技能和 MCP |

## 参与贡献

用真实素材来用 Cutlass，告诉我们哪里出了问题。[issue 表单](https://github.com/1mrnewton/cutlass/issues/new/choose)涵盖错误、功能、性能，以及打不开的文件。

要提交代码？请先运行上面的三项检查。共享的 crate 不依赖任何单一平台，所以同一个编辑在桌面和手机上都能用。

<a name="license"></a>
## 许可证

Cutlass 可以在两种许可证中任选一种免费使用。请选一种。

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

